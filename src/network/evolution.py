import math
from dataclasses import dataclass

from utils.seeding import derived_rng


@dataclass(frozen=True)
class EvolutionParameters:
    latency_step_fraction: float
    latency_bounds_fraction: tuple
    fidelity_step: float
    fidelity_max_degradation: float
    fidelity_max_improvement: float
    capacity_change_probability: float
    capacity_max_deviation: int
    mean_reversion: float
    regeneration_jitter: float


class NetworkEvolution:

    def __init__(self, network_seed, parameters):

        self.network_seed = network_seed
        self.parameters = parameters
        self._link_rngs = {}

        self.capacity_reductions_blocked = 0

    def _rng(self, key):

        if key not in self._link_rngs:
            self._link_rngs[key] = derived_rng(
                self.network_seed, f"link-evolution-{key[0]}-{key[1]}"
            )

        return self._link_rngs[key]

    def evolve(self, topology):

        parameters = self.parameters
        changes = []

        for link in topology.all_links():

            rng = self._rng(link.key)
            baseline = link.baseline

            latency_noise = rng.gauss(0.0, 1.0)
            fidelity_noise = rng.gauss(0.0, 1.0)
            capacity_change_draw = rng.random()
            capacity_direction_draw = rng.random()
            capacity_reversion_draw = rng.random()
            regeneration_draw = rng.uniform(
                1.0 - parameters.regeneration_jitter, 1.0 + parameters.regeneration_jitter
            )

            low, high = parameters.latency_bounds_fraction
            latency = (
                link.latency
                + latency_noise * parameters.latency_step_fraction * baseline.latency
                + parameters.mean_reversion * (baseline.latency - link.latency)
            )
            link.latency = round(
                min(high * baseline.latency, max(low * baseline.latency, latency)), 3
            )

            fidelity = (
                link.fidelity
                + fidelity_noise * parameters.fidelity_step
                + parameters.mean_reversion * (baseline.fidelity - link.fidelity)
            )
            link.fidelity = round(
                min(
                    min(0.9999, baseline.fidelity + parameters.fidelity_max_improvement),
                    max(baseline.fidelity - parameters.fidelity_max_degradation, fidelity),
                ),
                6,
            )

            proposed_capacity = link.capacity

            if capacity_change_draw < parameters.capacity_change_probability:
                proposed_capacity += 1 if capacity_direction_draw < 0.5 else -1
            elif (
                link.capacity != baseline.capacity
                and capacity_reversion_draw < parameters.mean_reversion
            ):
                proposed_capacity += 1 if baseline.capacity > link.capacity else -1

            proposed_capacity = min(
                baseline.capacity + parameters.capacity_max_deviation,
                max(max(1, baseline.capacity - parameters.capacity_max_deviation),
                    proposed_capacity),
            )

            committed_floor = math.ceil(link.active_load - 1e-9)

            if proposed_capacity < committed_floor:
                self.capacity_reductions_blocked += 1
                proposed_capacity = committed_floor

            link.capacity = proposed_capacity

            regenerated = link.regenerate_bell_pairs(
                link.bell_pair_regeneration * regeneration_draw
            )

            changes.append({
                "link": link.key,
                "latency": link.latency,
                "fidelity": link.fidelity,
                "capacity": link.capacity,
                "bell_pairs_regenerated": regenerated,
            })

        topology.sync_routing_weights()

        return changes
