# Implementation Log: Selective Congestion-Triggered Qubit Routing

The simulator was refactored from a fixed 12-qubit / 6-QPU demonstration into a
variable-network research simulator. The rule it now follows:

> The experimenter sets the boundaries (qubits, QPUs, capacities, benchmark, topology
> model, steps, threshold, link-property ranges). The system generates or calculates
> everything inside those boundaries from one master seed.

The line numbers in section 1 refer to the pre-refactor tree. That tree is identical
byte-for-byte to `../src.zip`, so every reference can be checked there.

---

## 1. Original hardcoded assumptions and where they were

| # | Assumption | Location (pre-refactor) | Effect |
|---|---|---|---|
| 1 | `RANDOM_SEED = 42` | `config.py:1` | Every run was the same experiment |
| 2 | Global `random.seed(config.RANDOM_SEED)` | `main.py:80` | The circuit depended on the global RNG state. NetworkX KL also used the global RNG, so it was coupled implicitly |
| 3 | `NetworkMonitor(seed=config.RANDOM_SEED)` | `main.py:95` | Network evolution was tied to the same fixed seed |
| 4 | `QAOA_SEED = 7` | `config.py:59`, `compiler/quantum_optimizer.py:113` | Every QAOA call reused one seed, independent of the run |
| 5 | `NUM_LOGICAL_QUBITS = 12`, `NUM_QPUS = 6`, `QPU_CAPACITY = 3` | `config.py:3-5`, `main.py:69-70` | No CLI control over the experiment size |
| 6 | `BENCHMARK_LAYERS = 16` | `config.py:76`, `main.py:83` | Fixed at 16 layers. 12 qubits × 16 layers ÷ 2 always gave 96 CX gates |
| 7 | `TOPOLOGY_EDGES` (8 hand-written links with fixed latency/fidelity/capacity) | `config.py:9-18`, `main.py:72` | One fixed physical network |
| 8 | `bell_pairs: 24` on every link and `INITIAL_BELL_PAIRS_PER_LINK = 24` | `config.py:10-17, 28` | One uniform Bell-pair pool |
| 9 | Global constant used as the Bell-pair scarcity denominator | `compiler/router.py:85`, `compiler/communication_cost.py:28` | Scarcity was not measured against each link's own pool |
| 10 | Global link bounds `LATENCY_MAX = 30`, `CAPACITY_MIN = 2`, `CAPACITY_MAX = 4` | `config.py:35-39`, `network/monitor.py:27,43` | Every link was clamped to the same range whatever its initial value |
| 11 | Absolute `MAX_ROUTE_HOPS = 3` | `config.py:42`, `compiler/router.py:141,168` | QPU pairs more than 3 hops apart had no route at all (for example, a line of 8 QPUs) |
| 12 | Plain bisection with `len(nodes)//2` and no capacity input | `compiler/kl_partitioner.py:23-26`, `compiler/dqc_compiler.py:37-40` | Capacity was never checked. Uneven splits (5 or 7 QPUs) could overfill a QPU |
| 13 | A second, unused greedy `Partitioner` with its own `max_size` rule | `compiler/partitioner.py:6-17` | Two competing initial-partition implementations |
| 14 | `Benchmarks.qft()` returned a single n-qubit `QFT` instruction | `compiler/benchmarks.py:23` | `CommunicationGraph` counted only `len(qubits) == 2` (`communication_graph.py:19`), so QFT produced **zero** interaction edges |
| 15 | `q._index` private attribute | `compiler/analyzer.py:17` | Relied on a private Qiskit attribute |
| 16 | `release_all_load()` at the start and end of every window | `simulation/window_executor.py:109, 251`, `network/link.py:114` | Congestion was cleared by wiping every link, not by releasing what completed |
| 17 | `queued_load` rebuilt from a shortest path per deferred item | `simulation/window_executor.py:233-247` | Queued demand was shown as physical link load it never held |
| 18 | Bell pairs checked per communication at admission but not reserved | `compiler/admission.py:54`, `simulation/window_executor.py:217` | Several communications could be admitted against the same pairs and then fail at execution |
| 19 | QAOA selection followed by hidden fallbacks (`_first_admissible`, `_restore`) | `compiler/selective_router.py:124-137, 189, 231` | The run was logged as `"qaoa"` even when a fallback route was executed. The reported QAOA energy did not describe what ran |
| 20 | QUBO had no capacity model, only `load_i·load_j/capacity` contention | `compiler/route_optimization.py:151` | A QAOA or classical optimum could be jointly infeasible |
| 21 | Repartitioning was a global swap/move search over **all** qubits | `compiler/optimizer.py:78-81, 104`, `simulation/adaptive_simulator.py:166` | Not selective. Any qubit could move |
| 22 | Repartition trigger checked latency first, then fidelity, then Bell pairs, then persistent congestion, over every link | `compiler/repartition_trigger.py:28, 47-53` | With `LATENCY_MAX = 30` and `TRIGGER_LATENCY_LIMIT = 28`, ordinary latency drift, not congestion, was the main repartition trigger |
| 23 | Outputs written to shared `graphs/` and `Logs/`, deleted before each run | `main.py:22-23, 58, 62` | Every run overwrote the previous one |
| 24 | The classical baseline replayed the same seed but shared output folders | `main.py:374` | Baseline artifacts were not tied to a seed |
| 25 | Hardcoded `window_for_step` with `(step - 1) % len(windows)` and no documented model | `compiler/workload.py:76` | Kept, but now documented as cyclic replay (section 20) |

Fixed QPU counts, per-QPU qubit counts and gate counts were never written as literals in
the algorithms. They followed from the constants above. After the refactor, no algorithm
reads a QPU count, qubit count, gate count or edge list from a constant.

The `tests/` directory was deleted, together with `.pytest_cache/`.

Modules removed because they were superseded:

| Removed | Replaced by |
|---|---|
| `compiler/benchmarks.py`, `compiler/analyzer.py` | `compiler/circuits.py` |
| `compiler/kl_partitioner.py`, `compiler/partitioner.py` | `compiler/initial_partitioner.py` (one implementation) |
| `compiler/dqc_compiler.py` | explicit pipeline in `experiment/runner.py` |
| `compiler/optimizer.py`, `compiler/objective.py` | `compiler/selective_repartitioner.py`, `compiler/partition_objective.py` |
| `compiler/repartition_trigger.py` | `compiler/repartition_policy.py` |
| `network/monitor.py` | `network/evolution.py` |
| `network/qpu.py` | `Topology.qpu_capacities` |
| `simulation/window_executor.py` | `simulation/adaptive_simulator.py`, `simulation/records.py` |

---

## 2. Circuit generation (`compiler/circuits.py`)

`generate_benchmark(name, num_qubits, layers, rng)` takes the qubit count and layer count
from the experiment. The only randomness comes from `rng`, a `random.Random` derived from
`circuit_seed`. No global RNG is used.

| Benchmark | Structure | Randomized by seed |
|---|---|---|
| `layered_random` | Each layer shuffles the qubits and pairs neighbours: `⌊n/2⌋` CX per layer | yes |
| `random` | Each qubit is active in a layer with p = 0.5. Active qubits are paired, idle ones get a random RZ. The gate count itself is random | yes |
| `ghz` | H + CX chain | no |
| `qft` | `synth_qft_full(n)`, decomposed | no |
| `hardware_efficient` | `efficient_su2(n, reps=layers, entanglement="linear")` with seeded parameter values | parameters only |

External input: `--circuit-file path.qasm` (OpenQASM 2, with a QASM 3 fallback). The qubit
count is read from the file. If `--qubits` is also given, the two must match.

`normalize_circuit` transpiles any circuit containing operations on more than two qubits,
or non-standard composite gates, to the basis `{cx, u}` using `seed_transpiler` from the
circuit seed. This fixes the QFT zero-edge bug. The whole pipeline then works from this
normalized circuit.

The gate count is never assumed. With 24 qubits and 16 layers, `layered_random` produces
192 two-qubit gates. With 32 qubits it produces 256. `circuit_summary.json` records the
counts that were actually generated.

## 3. Communication-weight calculation (`compiler/communication_graph.py`)

For every non-directive instruction acting on exactly two qubits (u, v), with barriers,
measurements, resets and delays excluded:

```
weight(u,v) += 1      (the edge is created with weight 1 if it does not exist)
```

All logical qubits are nodes, including qubits that never interact. No physical quantity
(latency, fidelity, capacity, Bell pairs, congestion) enters this graph. The self-check
verifies the example from the specification: `CX(0,4)×3 + CX(2,4)` gives
`w(0,4)=3, w(2,4)=1`.

## 4. Initial partition (`compiler/initial_partitioner.py`)

`CapacityAwareKLPartitioner(rng, restarts, refinement_passes).partition(graph, qpu_capacities)`
works in three stages.

1. **Recursive, capacity-proportional bisection.** The QPU id list is split in half
   (left, right). The left side receives
   `round(n · C_left / (C_left + C_right))` qubits, clipped to
   `[n − C_right, C_left]`. Neither half can therefore receive more qubits than its
   QPUs can hold. Heterogeneous capacities are supported.
2. **Kernighan–Lin with seeded restarts.** Each bisection runs `PARTITION_KL_RESTARTS`
   times. Each restart builds a random initial split of the target sizes from the
   partition RNG, which is derived from `partition_seed`. NetworkX KL then refines it
   using the actual edge weights. KL swaps pairs, so side sizes are preserved; a runtime
   guard asserts this. The split with the lowest weighted cut wins, with ties broken by
   sorted membership. NetworkX never uses its own global randomness here.
3. **Capacity-respecting refinement.** Up to `PARTITION_REFINEMENT_PASSES` passes apply
   single moves (only into QPUs with free capacity) and swaps, whenever they reduce the
   weighted cut.

The result goes through `validate_partition`: each qubit is assigned exactly once, no QPU
is over capacity, and every QPU id exists.

**Order of stages.** The initial partition is computed from the logical graph *before*
the physical topology is generated, following the specification. QPU ids are therefore
assigned without topology awareness (section 23).

Found and fixed during this work: NetworkX returns the two KL sides in swapped order.
The side seeded from `A` comes back second. The partitioner now matches sides by size,
not by position.

## 5. Enforcing QPU capacity

- **Hard input check.** `ExperimentSettings.validate` rejects runs where
  `qubits > Σ capacities`, for example `Invalid experiment: 30 logical qubits exceed total
  QPU capacity 24`.
- **Initial partition.** Guaranteed by the proportional split sizes in section 4 and
  re-validated afterwards.
- **Selective repartition.** Neighbour generation creates a *move* only into a QPU with
  free capacity. A full target QPU is reachable only through a *swap* with another
  affected qubit. `_capacity_respected` is re-checked before acceptance.
- **Every partition in `partition_history`** is validated after the run
  (`validation/invariants.py`).

## 6. Physical topology (`network/topology_generator.py`)

`generate_physical_edges(model, N, rng, extra_link_probability)` uses only N and the
topology RNG. It never sees the circuit, the communication graph or the partition.

| Model | Edges |
|---|---|
| `line` | `(i, i+1)` |
| `ring` | line plus `(0, N−1)` when N > 2 |
| `grid` | near-square grid with `⌈√N⌉` columns and a partial last row. Every node has an edge from above or to its left, so the grid is connected |
| `mesh` | all pairs |
| `star` | `(0, i)` |
| `random_connected` | (1) a random spanning tree: shuffle the QPUs, then attach each one to a uniformly chosen earlier QPU. This is connected by construction. (2) each remaining pair is added with probability `--extra-link-probability` (default 0.25) |

`generate_topology` raises an error if the result is disconnected. The self-check tests
all models for N ∈ {1, 2, 3, 5, 7, 8, 12} × 25 seeds. `Topology.connect` rejects self-loops,
duplicate links and unknown QPUs.

Repartitioning never touches the topology. `topology_initial.json` and
`topology_final.json` list the same links; only link properties evolve.

## 7. Independent per-link properties (`generate_link_properties`)

Each link (a, b) gets its own RNG, `derived_rng(topology_seed, "link-properties-a-b")`,
and draws:

| Property | Default range (`config.py`) | CLI override |
|---|---|---|
| latency | integer in `LINK_LATENCY_RANGE = (5, 30)` | `--link-latency-range` |
| fidelity | uniform in `LINK_FIDELITY_RANGE = (0.90, 0.995)` | `--link-fidelity-range` |
| capacity (concurrent load units) | integer in `LINK_CAPACITY_RANGE = (4, 12)` | `--link-capacity-range` |
| Bell-pair pool | integer in `LINK_BELL_PAIR_POOL_RANGE = (24, 64)` | `--link-bell-pairs-range` |
| Bell-pair regeneration per window | uniform in `(8, 20)` | `--link-regeneration-range` |

The initial values are frozen in `Link.baseline` (a `LinkBaseline`). **Bell-pair
scarcity** is now `1 − available / link.bell_pair_pool`, measured against each link's own
pool; the global `INITIAL_BELL_PAIRS_PER_LINK` is gone.

**Capacity calibration.** A first run with capacities 3–8 was congested in 50 of 50
windows. On a capacity-3 link, 2 communications already exceed the 0.60 threshold, so
utilization moved in steps of 33%. With 4–12, Run A had 96 of 400 link-windows congested
before rerouting. That leaves room for both normal and congested states. Both results
are in this log so the choice can be revisited.

## 8. Seeds (`utils/seeding.py`, `experiment/seeds.py`)

- **Master seed.** `--seed N`, or when omitted
  `secrets.randbelow(...)` in `MASTER_SEED_RANGE = (100000, 999999)`, redrawn if
  `Results/simulation_<seed>` already exists. There is no fixed default seed anywhere.
- **Derived seeds.**
  `derive_seed(master, label) = first 4 bytes of SHA-256("master/label")`, taken as an
  unsigned 32-bit integer. The labels are `circuit`, `topology`, `partition`, `network`
  and `qaoa`. SHA-256 is independent of Python's hash randomization and of call order.
- **Sub-streams** are derived the same way from the component seeds:
  - `topology_seed/"edges"`
  - `topology_seed/"link-properties-a-b"`
  - `network_seed/"link-evolution-a-b"`
  - `partition_seed/"kernighan-lin"`
  - `circuit_seed/"benchmark"`
  - `qaoa_seed/"step-S-block-B"`: each QAOA invocation gets its own seed for both its
    initial parameters and its sampler
- `manifest.json` stores the master seed, all five derived seeds, every control and a
  snapshot of every model parameter in `config.py`.

## 9. How a seed reproduces an experiment

With the same master seed, controls and code, every random draw comes from the same
derived stream in the same order. Per-link streams also make each link's draws
independent of how many other links exist. Dictionary iteration is over sorted keys or
integer keys, and ties are broken deterministically: by path tuples in routing, by sorted
membership in KL, by `(energy, bitstring)` in QAOA decoding, and by move order in
repartitioning.

`python main.py --seed S --verify-reproducibility` runs seed S twice and seed S+1 once in
temporary directories, which are deleted afterwards so no permanent results are left. It
compares SHA-256 fingerprints of the circuit, communication graph, topology links, link
properties, network evolution, initial partition, final partition, summary metrics,
reroutes and QAOA/classical route decisions. For Run A with QAOA and baseline, all 10
components matched between the two S runs.

## 10. How different seeds create different experiments

S+1 changed every stochastic component: the circuit (for randomized benchmarks), the
topology links (for `random_connected`), the link properties and the network evolution.
Non-random parts stay fixed when the benchmark or topology model is deterministic (GHZ,
QFT, ring and so on). The link properties and the network evolution still change.
Runs A and B (seeds 814372 and 814373, identical controls) produced different physical
networks:

- A: `0-2, 0-3, 0-5, 1-2, 1-4, 2-3, 3-4, 3-5`
- B: `0-2, 0-3, 0-4, 1-3, 1-4, 2-3, 2-5`

They also produced different circuits and different initial partitions.

## 11. Logical interaction → QPU pair

Each execution window's gate pairs become `Communication(qubit_u, qubit_v, load)`. At
admission time, **not** at creation, the current partition is applied:

```
QPU_u = partition[q_u];  QPU_v = partition[q_v]
QPU_u == QPU_v  → local: counted in local_interactions, never sent to admission
QPU_u != QPU_v  → remote: routed on the physical topology
```

Because QPUs are resolved at admission, a queued communication whose qubits were
co-located by a repartition becomes local (`became_local_after_repartition`).
`AdmissionController.admit` raises an error if asked to reserve a link for a local pair,
or for a route whose endpoints are not `(QPU_u, QPU_v)`.

## 12. QPU pair → physical path (`compiler/router.py`)

`Router.candidate_routes(topology, src, dst)` returns up to `CANDIDATE_ROUTES_PER_PAIR = 4`
loop-free paths from `nx.shortest_simple_paths` on the physical graph, weighted by current
latency. Paths are limited to `hop_distance(src, dst) + MAX_EXTRA_HOPS (2)` hops. The
limit is relative, so distant pairs always have a route.

`Topology.links_on_path` raises `NonexistentPhysicalLinkError` for any hop that is not a
physical link. It is called for every candidate route and every reservation, so an
invented edge such as (0, 4) cannot be used. The self-check confirms that on a 0-1-2 line
the only route from 0 to 2 is `0-1-2`.

## 13. Link load

- **Runtime communication load** = `LOAD_PER_REMOTE_INTERACTION` (1.0) per remote
  interaction.
- A reservation adds that load to **every link on the route** and takes
  `load × BELL_PAIRS_PER_UNIT_LOAD` Bell pairs from each link's available pool
  (`Link.reserve`).
- `link.active_load` equals the sum of live reservations. `load_ledger_consistent()`
  recomputes this from the allocation table after admission, after rerouting and after
  releases. A mismatch is logged and counted as a validation failure.
- **Static logical weight** (section 3) is used only by the initial partitioner and the
  static partition metric. It is never used as link load.

## 14. Congestion detection (`network/link.py`, `compiler/congestion.py`)

For each link, `utilization = active_load / capacity`.

| Condition | State |
|---|---|
| `utilization < threshold` | normal |
| `threshold ≤ utilization < 1` | congested |
| `utilization ≥ 1` | saturated |

An admission is valid only if `active_load + load ≤ capacity` and
`available_bell_pairs ≥ required` on every link of the route. Otherwise it is **refused**
and counted in `prevented_capacity_violations`, `rejected_for_capacity` or
`rejected_for_bell_pairs`. Nothing is clamped; `Link.reserve` raises an error if the rule
would be broken. Capacity evolution cannot shrink a link below its live reservations.
Instead it raises the floor to `ceil(active_load)` and counts
`capacity_reductions_blocked_by_reservations`.

`CongestionDetector.analyse` returns the congested and saturated link keys. Among the
communications admitted **this window**, it classifies as **affected** those whose route
uses a congested link and as **unaffected** the rest. It also counts in-flight
communications (admitted earlier, already transmitting) that are on congested links.
These are reported but cannot be rerouted.

## 15. Selective rerouting (`compiler/selective_router.py`)

1. Only the affected allocations are handled. Unaffected allocations are never released.
2. Affected communications are grouped into **blocks** that share congested links, with
   at most `QAOA_MAX_QUBITS = 12` decision variables per block.
3. For each block in turn:
   - release only that block's reservations (`consumed=False`, so the Bell pairs are
     returned);
   - build candidates: option 0 is the current route, followed by alternative routes that
     individually pass admission against the current state (each rejection is a
     prevented violation);
   - build the QUBO (section 18), solve it classically, and solve it with QAOA if enabled;
   - admit the selected routes.

   Blocks not yet processed keep their reservations. This makes "keep current routes"
   always feasible, so no hidden fallback is ever needed.
4. `unaffected_route_violations` compares every unaffected route before and after
   rerouting. It was 0 in every run.

Terminal and log output: `rerouting.log` records, for each block, the variable count, the
pairwise capacity conflicts, the classical and QAOA energies, the executed method and
energy, and each changed route with its cost before and after. `affected_link_utilization.csv`
holds each congested link's utilization before and after.

## 16. Selective repartitioning

### Trigger (`compiler/repartition_policy.py`)

The trigger reasons are configurable through
`REPARTITION_TRIGGER_REASONS = ("congestion", "resource")`.

- **congestion.** A link is still congested *after* selective rerouting, has been
  congested for at least `CONGESTION_PERSISTENCE_WINDOWS = 2` consecutive windows, and
  carries this window's traffic. The endpoints of that traffic are the affected qubits.
- **resource.** Communications were deferred for capacity or Bell pairs in at least
  `RESOURCE_DEFERRAL_PERSISTENCE_WINDOWS = 2` consecutive windows. The endpoints of the
  deferred communications are the affected qubits.
- **latency / fidelity.** Available but **disabled by default**, so generic drift cannot
  silently drive repartitioning.
- Several reasons at once are logged as `combined` with detail, for example
  `congestion+resource`.
- A cooldown of `REPARTITION_COOLDOWN_WINDOWS = 3` applies after an accepted
  repartition. Suppressed triggers are logged.

### Candidate search (`compiler/selective_repartitioner.py`)

1. Unaffected qubits are fixed. The neighbourhood contains only moves of affected qubits
   into QPUs with free capacity, and swaps between **two affected qubits**.
2. Up to `REPARTITION_MAX_MOVES_PER_EVENT = 2` greedy best-improvement steps.
3. Each candidate is evaluated on the **unchanged** physical topology. The evaluation
   covers the current queue plus the next circuit period of windows
   (`REPARTITION_LOOKAHEAD_WINDOWS = None` means one full period). For every window:
   - start from the load that stays in flight into the next window;
   - rebuild the remote communications under the candidate partition;
   - route each one over the physical candidate routes, keeping only those feasible
     against the projected loads and Bell pairs, choosing the lowest congestion-aware cost;
   - add its load to the projected link loads.

   ```
   objective = Σ load × route_cost + REPARTITION_DEFERRAL_PENALTY × deferred_load + migration_cost
   migration_cost = Σ_moved REPARTITION_MIGRATION_WEIGHT × base_cost(old QPU → new QPU)
   ```
4. A candidate is accepted only if moves exist, no unaffected qubit moved, capacity is
   respected, and `objective_after ≤ (1 − 0.05) × objective_before`. Otherwise the event
   is logged with its rejection reason: `no_improving_selective_move` or
   `improvement_below_margin`.
5. The new partition applies from the next window. In-flight communications finish on
   their reserved routes.

**Horizon fix made during calibration.** With a 4-window horizon, Run A accepted 14
repartitions, made 52 qubit moves, and the static communication cost rose from 5963 to
6952. The repartitioner was chasing the next few random layers. With a full-period
horizon, Run A accepted 1 (2 qubits) and the cost fell to 5876.

## 17. Communication cost (`compiler/communication_cost.py`)

`RouteCostModel` is the only route-cost formula; the two competing formulas that existed
before were merged. All terms are explicit, configurable, and computed from each link's
**current** state along the actual route:

```
base  = W_lat · Σ latency_l + W_hop · hops + W_infid · (1 − Π fidelity_l)
total = base
      + W_util  · max_l projected_util_l
      + W_cong  · Σ_l [util_l ≥ θ] · (1 + (util_l − θ)/(1 − θ))
      + W_bell  · max_l bell_scarcity_l
projected_util_l = (background_load_l + added_load) / capacity_l
```

The weights (`ROUTE_COST_*`) are 1, 5, 100, 20, 60 and 20.

| Quantity | Definition | Used by |
|---|---|---|
| Static logical weight | Gate count per logical pair | Initial partition, static partition metric |
| Runtime load | `LOAD_PER_REMOTE_INTERACTION` per communication | Admission, link load |
| Link utilization | `active_load / capacity` | Congestion state, cost terms |
| Communication cost | `load × total route cost` | Per-window cost (`window_communication_cost`), reroute costs |
| Primary route | Lowest `base` cost | Initial admission (congestion-unaware, like a routing table) |
| Static partition cost | `Σ_remote edges weight × base cost of the best route` | Reported for initial and final partitions |
| Repartition objective | Section 16 | Selective repartition acceptance |

Initial admission deliberately uses the congestion-unaware `base` cost. Congestion is
handled by the selective stage, which is the behaviour under study.

## 18. What QAOA optimizes (`compiler/route_optimization.py`, `compiler/quantum_optimizer.py`)

**Decision variables:** `x[c,r] = 1` if affected communication c in the current block uses
candidate route r. QAOA is used only for this selection among physically valid candidate
routes. It is not used for congestion detection, capacity checks, topology or
shortest-path search.

```
E(x) = Σ a[c,r]·x[c,r]
     + Σ_{c≠c'} b[(c,r),(c',r')]·x[c,r]·x[c',r']
     + P · Σ_c (Σ_r x[c,r] − 1)²

a      = load_c · total_route_cost(r | background) / scale
b      = W_share · Σ_shared l (load_c·load_c'/cap_l)
       + Σ_shared l max(0, G_l(B+l_c+l_c') − G_l(B+l_c) − G_l(B+l_c') + G_l(B)) / scale
       + V                  if load_c + load_c' > residual capacity (or residual Bell pairs) on a shared link
G_l(x) = W_cong · x · excess_l(x)    (the link's congestion cost as a function of total load)
P, V   = penalty scale × (1 + max a + max row coupling)
```

The G term is the exact pairwise interaction of a link's congestion cost. It penalises two
reroutes that together push a previously-normal link over the threshold. It was added
after the first runs showed rerouting sometimes *increasing* the number of congested
links.

**Hard constraints.** One-hot and **pairwise** capacity violations are encoded in the
QUBO. Violations involving three or more communications cannot be expressed pairwise, so
they are enforced *exactly* when decoding. QAOA samples are sorted by `(energy,
bitstring)`, and the first one that is one-hot **and** passes
`is_capacity_feasible` (exact per-link load and Bell pairs) becomes QAOA's solution. An
infeasible QAOA answer can therefore never be executed.

**Classical baseline.** The same QUBO is searched exhaustively over the capacity-feasible
selections when the search space is at most 20000. Otherwise it uses feasible local
search starting from "keep current".

**Selection policy.** `QAOA_SELECTION_POLICY = "qaoa_first"`: when QAOA is feasible, its
selection is executed, even when the classical optimum is lower. This keeps the
QAOA-hybrid and classical-only runs genuinely different. `best_of` is available.

**Reported energy = executed energy.** `executed_energy` in `qaoa_results.csv` is computed
from the admitted selection. The runtime validation check "reported QAOA energy equals the
energy of the executed selection" passed in every run. If an admission ever failed
(counted as `reroute_admission_failures`, 0 in all runs), the block would be restored and
logged as `keep_current_after_admission_failure`, never as `qaoa`.

**Implementation.** Statevector `qaoa_ansatz`, reps 2, COBYLA with 60 iterations, 1024
shots. Initial parameters and sampler use the per-invocation derived seed. This is a
noiseless simulation; no quantum advantage is claimed. In the final Run A, QAOA was
feasible in 79 of 79 blocks and matched the exact classical optimum in 54 of them. Before
the joint threshold term was added the QUBO was easier: 96 of 108 matched. Over the full
50 windows, the classical-only baseline had a *lower* mean window cost (788 vs 829).

## 19. Transient resource release

The execution model has one step per execution window:

1. The network evolves.
2. The next circuit layer's remote interactions plus queued communications are admitted.
3. Congestion detection, selective rerouting and the trigger run.
4. The window's admissions start (`in_flight = True`).
5. `complete_window()` decrements `remaining_windows` for every in-flight allocation.

Allocations that reach 0 are released **individually** with `consumed=True`: their load
leaves every link on their route, and their reserved Bell pairs become consumed.

A communication lasts `max(1, ceil(route_latency / WINDOW_DURATION))` windows, with
`WINDOW_DURATION = 30`. Multi-hop or slow routes therefore hold their load across windows,
and congestion disappears only when that load is released. No function resets loads to
zero.

**Queue.** Deferred communications are kept, not deleted. Their age increases each window
and they are retried first (oldest first). They are dropped only when
`age > MAX_QUEUE_AGE_WINDOWS (8)`; drops are logged and counted. Queued communications
hold no link load. The previous `queued_load` link attribute was removed because it
counted phantom load.

**Separate counters:**

| Counter | Meaning |
|---|---|
| `active_load` | Live reservations |
| `queue_length_after` | Communications waiting in the queue |
| `congested_windows` | Historical count of congested windows per link |
| `consecutive_congested_windows` | Persistence, used by the trigger |

Bell pairs regenerate each window per link:
`bell_pair_regeneration × U(1 ± 0.25)`, capped at `pool − reserved`.

## 20. Network evolution (`network/evolution.py`)

Each link has its own RNG, `network_seed/"link-evolution-a-b"`, and evolves around its own
baseline with mean reversion `EVOLUTION_MEAN_REVERSION = 0.15`:

- **Latency:** Gaussian step of 8% of the baseline value, bounded to
  `[0.5×, 2×] baseline`.
- **Fidelity:** Gaussian step with σ = 0.004, bounded to
  `[baseline − 0.08, baseline + 0.005]`.
- **Capacity:** with probability 0.10, changes by ±1; otherwise reverts toward the
  baseline. Bounded to `baseline ± 2`, never below 1, and never below live reservations.
- **Bell-pair regeneration:** jittered ±25%.

Every link draws the same number of values each step, whichever branch it takes. The
random streams therefore stay aligned between the QAOA run and the classical baseline.
Every step's per-link state is recorded in `metrics/link_state_history.csv`: step, link,
latency, fidelity, capacity, active load, utilization, state, available and reserved
Bell pairs, and persistence.

The workload replays the circuit's DAG layers cyclically. With 16 windows and 50 steps,
the circuit runs about 3.1 times. Each step injects one layer as new traffic.

## 21. Files stored under each seed

```
Results/simulation_<seed>/
├── manifest.json                 seeds, controls, model parameters, circuit + topology summary,
│                                 initial link properties, initial/final partition, fingerprints
├── circuit/  circuit.qasm, circuit_summary.json, communication_graph.json (weighted edges)
├── topology/ topology_initial.json, topology_final.json
├── logs/     simulation.log, rerouting.log, repartition.log
├── metrics/  step_records.csv, link_state_history.csv, link_summary.csv, reroute_log.csv,
│             affected_link_utilization.csv, repartition_events.csv, route_decision_blocks.csv,
│             baseline_comparison.csv, summary.json, validation_report.json
├── graphs/   communication_graph, initial_partition, final_partition, topology_initial,
│             topology_final, utilization_timeline (link × window heatmap), congestion_timeline,
│             rerouting_timeline, affected_link_utilization, repartition_NN_step_SSS
├── quantum/  qaoa_results.csv, qaoa_summary.json, qaoa_vs_classical.png
└── baseline_classical/  logs/ and metrics/ of the classical-only replay (same seeds)
```

`RunDirectory.path()` refuses any path outside the seed directory. Every written path is
recorded, and "all artifacts written inside the seed directory" is part of
`validation_report.json`. Re-running an existing seed with the **same** controls refreshes
the directory. Re-running with **different** controls is refused unless `--overwrite` is
given. A seed sweep writes each run to its own seed directory, plus a sweep index at
`Results/sweep_<first>_<n>_seeds/`.

## 22. Example commands

```bash
python main.py                                            # fresh seed, default boundaries (24q, 6 QPUs x 4, random_connected, 50 windows)
python main.py --seed 814372                              # reproduce a run
python main.py --qubits 24 --qpus 6 --qpu-capacity 4 --topology random_connected --seed 814372   # Run A
python main.py --qubits 24 --qpus 6 --qpu-capacity 4 --topology random_connected --seed 814373   # Run B
python main.py --qubits 32 --qpus 8 --qpu-capacity 4 --topology random_connected --seed 814374   # Run C
python main.py --qpus 4 --qpu-capacities 5,3,4,2 --qubits 13 --topology ring
python main.py --benchmark qft --qubits 16 --qpus 4 --qpu-capacity 4 --topology grid
python main.py --circuit-file my_circuit.qasm --qpus 4 --qpu-capacity 8
python main.py --link-capacity-range 6,16 --congestion-threshold 0.7 --steps 100
python main.py --seeds 814372,814373,814374 --qubits 24 --qpus 6      # seed sweep, same boundaries
python main.py --num-seeds 10 --no-quantum                            # 10 fresh seeds
python main.py --seed 814372 --verify-reproducibility                 # same seed x2 + seed+1, nothing kept
python main.py --no-quantum            # classical only, no baseline replay
python -m validation.self_check        # structural self-check (replaces tests/)
```

Results of the three reference runs with the final code (all runtime validation checks
passed, 21/21 each):

| Run | Links | Congested link-windows (before → after reroute) | Reroutes | Repartitions accepted | Static cost (initial → final) |
|---|---|---|---|---|---|
| A 814372 | 8 | 99 → 76 | 98 | 0 of 23 triggers | 5963 → 5963 |
| B 814373 | 7 | 165 → 157 | 86 | 10 of 28 (15 distinct qubits relocated) | 5988 → 5557 |
| C 814374 (32q/8 QPUs) | 12 | 118 → 74 | 129 | 0 of 16 | 8286 → 8286 |

`python -m validation.self_check` passes 26 of 26 checks in about 4 seconds. The checks
cover:

- weight calculation and QFT decomposition;
- connectivity of every topology model;
- seed-dependence of `random_connected`;
- the "no invented edge" rule;
- heterogeneous partition capacity;
- end-to-end invariants on all 6 topologies and all 5 benchmarks;
- output location;
- same-seed reproducibility;
- different-seed variation;
- QAOA reproducibility on runs that actually execute QAOA blocks.

## 23. Remaining assumptions and limitations

- **Traffic abstraction.** Circuit layers are injected one per window and replayed
  cyclically. Gate dependencies between layers are not enforced: a layer can start while
  earlier remote gates are still in flight. Duration is `ceil(route latency / 30)` windows.
- **One load unit per remote gate.** Every link on the path consumes one Bell pair per
  load unit (a simple entanglement-swapping model). Purification and heralding failures
  are not modeled, and fidelity affects cost but not success.
- **Topology-agnostic QPU labelling.** The initial partition decides which qubit groups
  go together, but not which physical QPU hosts each group. This follows the specified
  order (partition before topology). A topology-aware mapping would lower the initial
  cost and is a natural extension.
- **Rerouting objective ≠ congested-link count.** Rerouting minimises congestion-aware
  cost, so moving load off a saturated link can push a neighbour over the threshold.
  This happened in 1–10 windows per reference run.
- **Pairwise capacity encoding.** Violations involving three or more communications are
  not in the QUBO. They are handled exactly by feasible-only decoding. A QAOA run with no
  feasible sample falls back to the classical solution, logged as such.
- **Block decomposition.** Blocks of at most 12 variables are optimized one after
  another. This is sequential, not globally joint, and is the same for QAOA and classical.
- **QAOA** is simulated on a noiseless statevector. Its runtime and its quality relative
  to the classical baseline are reported, not assumed.
- **Migration** of a qubit is costed in the repartition objective (base route cost
  between the old and new QPU). It does not reserve link capacity or consume Bell pairs.
- **Capacity evolution** is floored at live reservations. Evolution can therefore differ
  slightly between the QAOA and classical replays once their loads diverge; the random
  draws stay aligned.
- **Heuristic weights.** Cost weights, penalty scales and the capacity range are
  engineering choices recorded in the manifest, not calibrated against hardware.
- **Other directories.** `src3_seed_outputs/` (an earlier partial attempt), the old
  top-level `graphs/` and `Logs/*.csv` outputs, and `Results/simulation_463752843/`
  (empty apart from a console log from that attempt) were left untouched. Nothing in the
  current code writes or reads them.
