import os

from compiler.initial_partitioner import validate_partition
from compiler.workload import qubit_to_qpu_map


def _check(name, passed, detail=""):
    return {"check": name, "passed": bool(passed), "detail": detail}


def validate_simulation(simulator, topology, num_logical_qubits, qpu_capacities, label):

    checks = []

    checks.append(_check(
        f"{label}: physical topology connected", topology.is_connected(),
        f"{len(topology.links)} links over {len(topology.qpu_capacities)} QPUs",
    ))

    nonexistent = []
    endpoints_bad = []

    for path in sorted(simulator.admission.admitted_paths):

        for a, b in zip(path[:-1], path[1:]):
            if not topology.has_link(a, b):
                nonexistent.append(f"{a}-{b} in {path}")

        if len(path) < 2 or path[0] == path[-1]:
            endpoints_bad.append(str(path))

    checks.append(_check(
        f"{label}: admitted routes use only existing physical links",
        not nonexistent, "; ".join(nonexistent[:5]) or
        f"{len(simulator.admission.admitted_paths)} distinct admitted paths checked",
    ))
    checks.append(_check(
        f"{label}: remote communications use valid multi-QPU paths and locals use none",
        not endpoints_bad, "; ".join(endpoints_bad[:5]),
    ))

    for step, partition in simulator.partition_history:
        problems = validate_partition(partition, num_logical_qubits, qpu_capacities)
        if problems:
            checks.append(_check(
                f"{label}: partition after step {step} valid", False, "; ".join(problems)
            ))
            break
    else:
        checks.append(_check(
            f"{label}: every partition respects capacity and assigns each qubit once", True,
            f"{len(simulator.partition_history)} partitions checked",
        ))

    records = simulator.step_records

    checks.append(_check(
        f"{label}: no admitted load exceeded link capacity",
        all(r.capacity_invariant_ok for r in records),
        f"{sum(1 for r in records if not r.capacity_invariant_ok)} violating steps",
    ))
    checks.append(_check(
        f"{label}: link load equals the sum of live reservations (no leaked load)",
        all(r.load_ledger_ok for r in records),
    ))
    checks.append(_check(
        f"{label}: link state history never shows load above capacity",
        all(row["active_load"] <= row["capacity"] + 1e-9
            for row in simulator.link_state_history),
    ))
    checks.append(_check(
        f"{label}: selective rerouting never modified unaffected communications",
        sum(r.unaffected_route_violations for r in records) == 0,
        f"{sum(r.unaffected_communications for r in records if r.affected_communications)} "
        "unaffected communications observed on congestion windows",
    ))

    bad_moves = []

    for outcome in simulator.repartition_outcomes:

        if not outcome.accepted:
            continue

        old_map = qubit_to_qpu_map(outcome.old_partition)
        new_map = qubit_to_qpu_map(outcome.new_partition)
        moved = {q for q in new_map if new_map[q] != old_map[q]}
        outside = moved - set(outcome.affected_qubits)

        if outside:
            bad_moves.append(f"step {outcome.step}: {sorted(outside)}")

    checks.append(_check(
        f"{label}: selective repartitioning moved only affected qubits",
        not bad_moves, "; ".join(bad_moves) or
        f"{sum(1 for o in simulator.repartition_outcomes if o.accepted)} accepted events checked",
    ))

    inconsistent = [
        f"step {b.step} block {b.block_index}"
        for b in simulator.block_decisions
        if b.executed_method == "qaoa" and abs(b.executed_energy - b.qaoa_energy) > 1e-9
    ]

    checks.append(_check(
        f"{label}: reported QAOA energy equals the energy of the executed selection",
        not inconsistent, "; ".join(inconsistent[:5]),
    ))

    return checks


def validate_output_location(directory):

    outside = [
        path for path in directory.written
        if os.path.commonpath([os.path.abspath(path), directory.root]) != directory.root
    ]

    return [_check(
        "all artifacts written inside the seed directory", not outside,
        f"{len(directory.written)} files under {directory.root}",
    )]
