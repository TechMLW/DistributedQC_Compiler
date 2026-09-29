from dataclasses import dataclass, field
from itertools import product


TOLERANCE = 1e-9


@dataclass
class RouteChoiceEntry:
    communication_id: str
    load: float
    bell_pairs_per_link: float


@dataclass
class RouteChoiceProblem:
    entries: list
    candidates: dict
    residual_capacity: dict
    residual_bell_pairs: dict
    variables: list
    linear: list
    quadratic: dict
    onehot_penalty: float
    capacity_penalty: float
    raw_costs: dict
    cost_scale: float
    capacity_conflict_pairs: int = 0

    @property
    def num_variables(self):
        return len(self.variables)

    @property
    def communication_ids(self):
        return [entry.communication_id for entry in self.entries]

    def variable_groups(self):

        groups = {cid: [] for cid in self.communication_ids}

        for index, (cid, _) in enumerate(self.variables):
            groups[cid].append(index)

        return groups

    def selection_to_bits(self, selection):

        bits = [0] * self.num_variables
        positions = {variable: index for index, variable in enumerate(self.variables)}

        for cid, route_index in selection.items():
            bits[positions[(cid, route_index)]] = 1

        return bits

    def energy(self, bits):

        total = sum(self.linear[index] for index, bit in enumerate(bits) if bit)

        for (i, j), coefficient in self.quadratic.items():
            if bits[i] and bits[j]:
                total += coefficient

        for indices in self.variable_groups().values():
            chosen = sum(bits[index] for index in indices)
            total += self.onehot_penalty * (chosen - 1) ** 2

        return total

    def selection_energy(self, selection):
        return self.energy(self.selection_to_bits(selection))

    def decode(self, bits):

        selection = {}

        for cid, indices in self.variable_groups().items():

            chosen = [index for index in indices if bits[index]]

            if len(chosen) != 1:
                return None

            selection[cid] = self.variables[chosen[0]][1]

        return selection

    def is_capacity_feasible(self, selection):

        load = {}
        bell_pairs = {}

        for entry in self.entries:

            route = self.candidates[entry.communication_id][selection[entry.communication_id]]

            for key in route.link_keys():
                load[key] = load.get(key, 0.0) + entry.load
                bell_pairs[key] = bell_pairs.get(key, 0.0) + entry.bell_pairs_per_link

        return all(
            load[key] <= self.residual_capacity[key] + TOLERANCE
            and bell_pairs[key] <= self.residual_bell_pairs[key] + TOLERANCE
            for key in load
        )

    def raw_cost(self, selection):
        return sum(
            self.raw_costs[(cid, route_index)] for cid, route_index in selection.items()
        )


def build_route_choice_problem(entries, candidates, topology, cost_model,
                               background_load, contention_weight,
                               onehot_penalty_scale, capacity_penalty_scale):

    residual_capacity = {
        key: link.capacity - background_load.get(key, 0.0)
        for key, link in topology.links.items()
    }
    residual_bell_pairs = {
        key: link.available_bell_pairs for key, link in topology.links.items()
    }

    variables = []
    raw_costs = {}

    for entry in entries:
        for route_index, route in enumerate(candidates[entry.communication_id]):
            variables.append((entry.communication_id, route_index))
            raw_costs[(entry.communication_id, route_index)] = entry.load * (
                cost_model.congestion_aware_cost(
                    route, topology, added_load=entry.load, background_load=background_load
                )
            )

    cost_scale = max(raw_costs.values(), default=0.0) or 1.0

    linear = [raw_costs[variable] / cost_scale for variable in variables]

    load_by_id = {entry.communication_id: entry.load for entry in entries}
    bell_by_id = {entry.communication_id: entry.bell_pairs_per_link for entry in entries}

    contention = {}
    conflicts = set()

    for i in range(len(variables)):
        for j in range(i + 1, len(variables)):

            cid_i, route_i = variables[i]
            cid_j, route_j = variables[j]

            if cid_i == cid_j:
                continue

            shared = set(candidates[cid_i][route_i].link_keys()).intersection(
                candidates[cid_j][route_j].link_keys()
            )

            if not shared:
                continue

            sharing = 0.0
            threshold_crossing = 0.0

            for key in sorted(shared):

                link = topology.links[key]

                if (
                    load_by_id[cid_i] + load_by_id[cid_j] > residual_capacity[key] + TOLERANCE
                    or bell_by_id[cid_i] + bell_by_id[cid_j]
                    > residual_bell_pairs[key] + TOLERANCE
                ):
                    conflicts.add((i, j))

                if link.capacity > 0:
                    sharing += load_by_id[cid_i] * load_by_id[cid_j] / link.capacity

                threshold_crossing += cost_model.joint_congestion_interaction(
                    link, background_load.get(key, 0.0), load_by_id[cid_i], load_by_id[cid_j]
                )

            value = contention_weight * sharing + threshold_crossing / cost_scale

            if value > 0:
                contention[(i, j)] = value

    row_totals = [0.0] * len(variables)

    for (i, j), value in contention.items():
        row_totals[i] += value
        row_totals[j] += value

    penalty_base = 1.0 + max(linear, default=0.0) + max(row_totals, default=0.0)

    onehot_penalty = onehot_penalty_scale * penalty_base
    capacity_penalty = capacity_penalty_scale * penalty_base

    quadratic = dict(contention)

    for pair in conflicts:
        quadratic[pair] = quadratic.get(pair, 0.0) + capacity_penalty

    return RouteChoiceProblem(
        entries=list(entries),
        candidates={entry.communication_id: candidates[entry.communication_id]
                    for entry in entries},
        residual_capacity=residual_capacity,
        residual_bell_pairs=residual_bell_pairs,
        variables=variables,
        linear=linear,
        quadratic=quadratic,
        onehot_penalty=onehot_penalty,
        capacity_penalty=capacity_penalty,
        raw_costs=raw_costs,
        cost_scale=cost_scale,
        capacity_conflict_pairs=len(conflicts),
    )


@dataclass
class RouteSolution:
    selection: dict
    energy: float
    method: str
    feasible: bool
    evaluations: int = 0
    details: dict = field(default_factory=dict)


class ClassicalRouteSolver:

    def __init__(self, exhaustive_limit):
        self.exhaustive_limit = exhaustive_limit

    def solve(self, problem):

        ids = problem.communication_ids
        option_counts = [len(problem.candidates[cid]) for cid in ids]

        search_space = 1

        for count in option_counts:
            search_space *= count

        if search_space <= self.exhaustive_limit:
            return self._exhaustive(problem, ids, option_counts, search_space)

        return self._local_search(problem, ids)

    def _exhaustive(self, problem, ids, option_counts, search_space):

        best_selection = None
        best_energy = float("inf")
        evaluations = 0

        for combination in product(*[range(count) for count in option_counts]):

            selection = dict(zip(ids, combination))
            evaluations += 1

            if not problem.is_capacity_feasible(selection):
                continue

            energy = problem.selection_energy(selection)

            if energy < best_energy - TOLERANCE:
                best_energy = energy
                best_selection = selection

        return RouteSolution(
            selection=best_selection or {},
            energy=best_energy,
            method="classical_exhaustive",
            feasible=best_selection is not None,
            evaluations=evaluations,
            details={"search_space": search_space},
        )

    def _local_search(self, problem, ids):

        selection = {cid: 0 for cid in ids}
        evaluations = 1

        if not problem.is_capacity_feasible(selection):
            return RouteSolution({}, float("inf"), "classical_local_search", False, 1)

        energy = problem.selection_energy(selection)
        improved = True

        while improved:

            improved = False

            for cid in ids:
                for option in range(len(problem.candidates[cid])):

                    if option == selection[cid]:
                        continue

                    trial = dict(selection)
                    trial[cid] = option
                    evaluations += 1

                    if not problem.is_capacity_feasible(trial):
                        continue

                    trial_energy = problem.selection_energy(trial)

                    if trial_energy < energy - TOLERANCE:
                        selection, energy = trial, trial_energy
                        improved = True

        return RouteSolution(
            selection=selection,
            energy=energy,
            method="classical_local_search",
            feasible=True,
            evaluations=evaluations,
        )
