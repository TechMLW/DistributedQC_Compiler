import copy
import shutil
import tempfile

from experiment.output import RunDirectory
from experiment.runner import run_experiment
from experiment.seeds import SeedPlan


ALWAYS_STOCHASTIC = ("link_properties", "network_evolution")


def _run_in_temporary_directory(settings, master_seed):

    root = tempfile.mkdtemp(prefix=f"repro_{master_seed}_")

    try:
        directory = RunDirectory(root)
        directory.prepare()
        result = run_experiment(
            copy.deepcopy(settings), SeedPlan.from_master(master_seed), directory,
            console=lambda message: None, make_graphs=False,
        )
        return result.fingerprint, result.passed_validation, result.summary
    finally:
        shutil.rmtree(root, ignore_errors=True)


def verify_reproducibility(settings, master_seed, comparison_seed=None, console=print):

    comparison_seed = master_seed + 1 if comparison_seed is None else comparison_seed

    console(f"reproducibility: running seed {master_seed} twice and seed {comparison_seed} once "
            "in temporary directories")

    first, first_valid, first_summary = _run_in_temporary_directory(settings, master_seed)
    second, second_valid, _ = _run_in_temporary_directory(settings, master_seed)
    other, other_valid, _ = _run_in_temporary_directory(settings, comparison_seed)

    same_seed = {
        component: first[component] == second[component] for component in first
    }

    stochastic = list(ALWAYS_STOCHASTIC)

    if settings.circuit_file is None and settings.benchmark in ("layered_random", "random"):
        stochastic.append("circuit")

    if settings.topology == "random_connected":
        stochastic.append("topology_links")

    different_seed = {component: first[component] != other[component] for component in stochastic}

    report = {
        "master_seed": master_seed,
        "comparison_seed": comparison_seed,
        "same_seed_components_identical": same_seed,
        "different_seed_stochastic_components_differ": different_seed,
        "validation_passed": {
            "first": first_valid, "second": second_valid, "comparison": other_valid,
        },
        "qaoa_blocks_attempted": first_summary["qaoa_blocks_attempted"],
        "reroute_blocks": first_summary["reroute_blocks"],
        "reproducible": all(same_seed.values()),
        "seed_sensitive": all(different_seed.values()),
    }

    for component, identical in same_seed.items():
        console(f"  same seed      {component:<20} {'identical' if identical else 'DIFFERENT'}")

    for component, differs in different_seed.items():
        console(f"  different seed {component:<20} {'differs' if differs else 'IDENTICAL'}")

    console(f"reproducible: {report['reproducible']}; seed-sensitive: {report['seed_sensitive']}")

    return report
