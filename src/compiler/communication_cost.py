from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RouteCostWeights:
    latency: float
    hop: float
    infidelity: float
    utilization: float
    congestion: float
    bell_scarcity: float

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class RouteCostBreakdown:
    latency_term: float
    hop_term: float
    infidelity_term: float
    utilization_term: float
    congestion_term: float
    bell_scarcity_term: float

    @property
    def base(self):
        return self.latency_term + self.hop_term + self.infidelity_term

    @property
    def total(self):
        return (
            self.base
            + self.utilization_term
            + self.congestion_term
            + self.bell_scarcity_term
        )


ZERO_BREAKDOWN = RouteCostBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


class RouteCostModel:

    def __init__(self, weights):
        self.weights = weights

    def breakdown(self, route, topology, added_load=0.0, background_load=None):

        if route is None or route.is_local:
            return ZERO_BREAKDOWN

        links = topology.links_on_path(route.path)
        weights = self.weights

        fidelity = 1.0
        bottleneck = 0.0
        congestion_excess = 0.0
        scarcity = 0.0

        for link in links:

            fidelity *= link.fidelity

            current = (
                background_load.get(link.key, 0.0)
                if background_load is not None else link.active_load
            )

            projected = (
                (current + added_load) / link.capacity if link.capacity > 0 else 1.0
            )

            bottleneck = max(bottleneck, projected)

            if projected >= link.congestion_threshold:
                headroom = max(1e-9, 1.0 - link.congestion_threshold)
                congestion_excess += (
                    1.0 + (projected - link.congestion_threshold) / headroom
                )

            scarcity = max(scarcity, link.bell_pair_scarcity)

        return RouteCostBreakdown(
            latency_term=weights.latency * sum(link.latency for link in links),
            hop_term=weights.hop * route.hop_count,
            infidelity_term=weights.infidelity * (1.0 - fidelity),
            utilization_term=weights.utilization * bottleneck,
            congestion_term=weights.congestion * congestion_excess,
            bell_scarcity_term=weights.bell_scarcity * scarcity,
        )

    def link_congestion_cost(self, link, total_load):

        if link.capacity <= 0:
            return 0.0

        utilization = total_load / link.capacity

        if utilization < link.congestion_threshold:
            return 0.0

        headroom = max(1e-9, 1.0 - link.congestion_threshold)
        excess = 1.0 + (utilization - link.congestion_threshold) / headroom

        return self.weights.congestion * total_load * excess

    def joint_congestion_interaction(self, link, background, first_load, second_load):

        interaction = (
            self.link_congestion_cost(link, background + first_load + second_load)
            - self.link_congestion_cost(link, background + first_load)
            - self.link_congestion_cost(link, background + second_load)
            + self.link_congestion_cost(link, background)
        )

        return max(0.0, interaction)

    def base_cost(self, route, topology):
        return self.breakdown(route, topology).base

    def congestion_aware_cost(self, route, topology, added_load=0.0, background_load=None):
        return self.breakdown(route, topology, added_load, background_load).total

    def communication_cost(self, load, route, topology):
        return load * self.congestion_aware_cost(route, topology)
