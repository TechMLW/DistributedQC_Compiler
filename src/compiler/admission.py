import math
from dataclasses import dataclass


REJECT_NO_ROUTE = "no_route"
REJECT_CAPACITY = "hard_capacity"
REJECT_BELL_PAIRS = "bell_pairs"


@dataclass
class Allocation:
    communication: object
    source_qpu: int
    destination_qpu: int
    route: object
    load: float
    bell_pairs_per_link: float
    admitted_step: int
    remaining_windows: int
    in_flight: bool = False

    @property
    def id(self):
        return self.communication.id


class AdmissionController:

    def __init__(self, topology, bell_pairs_per_unit_load, window_duration):

        self.topology = topology
        self.bell_pairs_per_unit_load = bell_pairs_per_unit_load
        self.window_duration = window_duration

        self.allocations = {}
        self.admitted_paths = set()

        self.reset_window_counters()

    def reset_window_counters(self):

        self.prevented_capacity_violations = 0
        self.rejected_for_capacity = 0
        self.rejected_for_bell_pairs = 0
        self.rejected_for_no_route = 0

    def bell_pairs_for(self, load):
        return load * self.bell_pairs_per_unit_load

    def check_route(self, route, load, count_rejection=True):

        if route is None:
            reason = REJECT_NO_ROUTE
        else:
            reason = None
            bell_pairs = self.bell_pairs_for(load)

            for link in self.topology.links_on_path(route.path):

                if link.capacity_would_overflow(load):
                    reason = REJECT_CAPACITY
                    break

                if link.bell_pairs_insufficient(bell_pairs):
                    reason = REJECT_BELL_PAIRS
                    break

        if reason is not None and count_rejection:
            self._count_rejection(reason)

        return reason is None, reason

    def _count_rejection(self, reason):

        self.prevented_capacity_violations += 1

        if reason == REJECT_CAPACITY:
            self.rejected_for_capacity += 1
        elif reason == REJECT_BELL_PAIRS:
            self.rejected_for_bell_pairs += 1
        else:
            self.rejected_for_no_route += 1

    def feasible_routes(self, routes, load, count_rejection=True):
        return [
            route for route in routes
            if self.check_route(route, load, count_rejection)[0]
        ]

    def duration_windows(self, route):

        if route.is_local:
            return 1

        latency = sum(link.latency for link in self.topology.links_on_path(route.path))

        return max(1, math.ceil(latency / self.window_duration))

    def admit(self, communication, source_qpu, destination_qpu, route, step):

        if source_qpu == destination_qpu or route.is_local:
            raise ValueError(f"{communication.id} is local and must not reserve a physical link")

        if route.path[0] != source_qpu or route.path[-1] != destination_qpu:
            raise ValueError(
                f"Route {route.label()} does not connect QPU {source_qpu} to QPU {destination_qpu}"
            )

        admissible, reason = self.check_route(route, communication.load)

        if not admissible:
            return False, reason

        bell_pairs = self.bell_pairs_for(communication.load)

        for link in self.topology.links_on_path(route.path):
            link.reserve(communication.load, bell_pairs)

        self.admitted_paths.add(route.path)

        self.allocations[communication.id] = Allocation(
            communication=communication,
            source_qpu=source_qpu,
            destination_qpu=destination_qpu,
            route=route,
            load=communication.load,
            bell_pairs_per_link=bell_pairs,
            admitted_step=step,
            remaining_windows=self.duration_windows(route),
        )

        return True, None

    def release(self, communication_id, consumed):

        allocation = self.allocations.pop(communication_id)

        for link in self.topology.links_on_path(allocation.route.path):
            link.release(allocation.load, allocation.bell_pairs_per_link, consumed)

        return allocation

    def route_of(self, communication_id):

        allocation = self.allocations.get(communication_id)

        return allocation.route if allocation else None

    def reroutable(self, step):
        return [
            allocation
            for allocation in self.allocations.values()
            if allocation.admitted_step == step and not allocation.in_flight
        ]

    def in_flight(self):
        return [a for a in self.allocations.values() if a.in_flight]

    def start_admitted(self, step):

        for allocation in self.allocations.values():
            if allocation.admitted_step == step:
                allocation.in_flight = True

    def complete_window(self):

        completed = []

        for communication_id in sorted(self.allocations):

            allocation = self.allocations[communication_id]

            if not allocation.in_flight:
                continue

            allocation.remaining_windows -= 1

            if allocation.remaining_windows <= 0:
                completed.append(self.release(communication_id, consumed=True))

        return completed

    def load_by_link(self):

        loads = {key: 0.0 for key in self.topology.links}

        for allocation in self.allocations.values():
            for key in allocation.route.link_keys():
                loads[key] += allocation.load

        return loads

    def load_ledger_consistent(self):

        expected = self.load_by_link()

        return all(
            abs(link.active_load - expected[key]) < 1e-6
            for key, link in self.topology.links.items()
        )

    def persisting_load(self):

        loads = {key: 0.0 for key in self.topology.links}

        for allocation in self.allocations.values():
            if allocation.remaining_windows > 1:
                for key in allocation.route.link_keys():
                    loads[key] += allocation.load

        return loads
