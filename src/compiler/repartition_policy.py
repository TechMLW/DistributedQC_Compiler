from dataclasses import dataclass, field


REASON_CONGESTION = "congestion"
REASON_RESOURCE = "resource"
REASON_LATENCY = "latency"
REASON_FIDELITY = "fidelity"
REASON_COMBINED = "combined"

SUPPORTED_REASONS = (REASON_CONGESTION, REASON_RESOURCE, REASON_LATENCY, REASON_FIDELITY)


@dataclass
class TriggerDecision:
    step: int
    triggered: bool = False
    suppressed_by_cooldown: bool = False
    reasons: list = field(default_factory=list)
    trigger_link_keys: list = field(default_factory=list)
    affected_qubits: list = field(default_factory=list)
    details: dict = field(default_factory=dict)

    @property
    def reason(self):

        if not self.reasons:
            return "none"

        if len(self.reasons) > 1:
            return REASON_COMBINED

        return self.reasons[0]

    @property
    def reason_detail(self):
        return "+".join(self.reasons) if self.reasons else "none"


class RepartitionTriggerPolicy:

    def __init__(
        self,
        enabled_reasons,
        congestion_persistence_windows,
        resource_persistence_windows,
        latency_limit,
        fidelity_limit,
        cooldown_windows,
    ):

        unknown = set(enabled_reasons) - set(SUPPORTED_REASONS)

        if unknown:
            raise ValueError(f"Unknown repartition trigger reasons: {sorted(unknown)}")

        self.enabled_reasons = tuple(enabled_reasons)
        self.congestion_persistence_windows = congestion_persistence_windows
        self.resource_persistence_windows = resource_persistence_windows
        self.latency_limit = latency_limit
        self.fidelity_limit = fidelity_limit
        self.cooldown_windows = cooldown_windows

        self.consecutive_deferral_windows = 0
        self.last_repartition_step = None

    def evaluate(self, step, topology, admission, resource_deferred):

        if resource_deferred:
            self.consecutive_deferral_windows += 1
        else:
            self.consecutive_deferral_windows = 0

        decision = TriggerDecision(step=step)

        current_traffic = [
            allocation for allocation in admission.allocations.values()
            if allocation.admitted_step == step
        ]

        affected_qubits = set()
        trigger_links = set()

        def traffic_on(keys):
            return [
                allocation for allocation in current_traffic
                if keys.intersection(allocation.route.link_keys())
            ]

        if REASON_CONGESTION in self.enabled_reasons:

            persistent = {
                link.key for link in topology.all_links()
                if link.is_congested
                and link.consecutive_congested_windows >= self.congestion_persistence_windows
            }
            carrying = traffic_on(persistent)

            if carrying:
                decision.reasons.append(REASON_CONGESTION)
                trigger_links |= {
                    key for allocation in carrying
                    for key in allocation.route.link_keys() if key in persistent
                }
                affected_qubits |= {q for a in carrying for q in a.communication.qubits}
                decision.details["persistent_congested_links"] = sorted(persistent)

        if (
            REASON_RESOURCE in self.enabled_reasons
            and resource_deferred
            and self.consecutive_deferral_windows >= self.resource_persistence_windows
        ):
            decision.reasons.append(REASON_RESOURCE)
            affected_qubits |= {
                q for communication in resource_deferred for q in communication.qubits
            }
            decision.details["deferred_communications"] = len(resource_deferred)

        for reason, predicate in (
            (REASON_LATENCY, lambda link: link.latency > self.latency_limit),
            (REASON_FIDELITY, lambda link: link.fidelity < self.fidelity_limit),
        ):

            if reason not in self.enabled_reasons:
                continue

            degraded = {link.key for link in topology.all_links() if predicate(link)}
            carrying = traffic_on(degraded)

            if carrying:
                decision.reasons.append(reason)
                trigger_links |= degraded
                affected_qubits |= {q for a in carrying for q in a.communication.qubits}

        decision.trigger_link_keys = sorted(trigger_links)
        decision.affected_qubits = sorted(affected_qubits)

        if not decision.reasons:
            return decision

        if (
            self.last_repartition_step is not None
            and step - self.last_repartition_step < self.cooldown_windows
        ):
            decision.suppressed_by_cooldown = True
            return decision

        decision.triggered = True

        return decision

    def record_repartition(self, step):
        self.last_repartition_step = step
