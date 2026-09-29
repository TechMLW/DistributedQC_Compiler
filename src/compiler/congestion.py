from dataclasses import dataclass, field


@dataclass
class CongestionReport:
    congested_link_keys: list = field(default_factory=list)
    saturated_link_keys: list = field(default_factory=list)
    affected_communication_ids: list = field(default_factory=list)
    unaffected_communication_ids: list = field(default_factory=list)
    in_flight_on_congested: int = 0
    link_utilization: dict = field(default_factory=dict)

    @property
    def has_congestion(self):
        return bool(self.congested_link_keys)

    @property
    def affected_count(self):
        return len(self.affected_communication_ids)

    @property
    def unaffected_count(self):
        return len(self.unaffected_communication_ids)


class CongestionDetector:

    def analyse(self, topology, admission, step):

        congested = [link.key for link in topology.congested_links()]
        saturated = [link.key for link in topology.saturated_links()]
        congested_set = set(congested)

        affected = []
        unaffected = []

        for allocation in sorted(admission.reroutable(step), key=lambda a: a.id):

            if congested_set.intersection(allocation.route.link_keys()):
                affected.append(allocation.id)
            else:
                unaffected.append(allocation.id)

        in_flight_on_congested = sum(
            1
            for allocation in admission.in_flight()
            if congested_set.intersection(allocation.route.link_keys())
        )

        return CongestionReport(
            congested_link_keys=congested,
            saturated_link_keys=saturated,
            affected_communication_ids=affected,
            unaffected_communication_ids=unaffected,
            in_flight_on_congested=in_flight_on_congested,
            link_utilization={
                key: link.utilization for key, link in sorted(topology.links.items())
            },
        )
