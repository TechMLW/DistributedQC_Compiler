from dataclasses import asdict, dataclass


NORMAL = "normal"
CONGESTED = "congested"
SATURATED = "saturated"

TOLERANCE = 1e-9


@dataclass(frozen=True)
class LinkBaseline:
    latency: float
    fidelity: float
    capacity: int
    bell_pair_pool: int
    bell_pair_regeneration: float


class Link:

    def __init__(
        self,
        source,
        destination,
        latency,
        fidelity,
        capacity,
        bell_pair_pool,
        bell_pair_regeneration,
        congestion_threshold,
    ):

        self.source = source
        self.destination = destination

        self.baseline = LinkBaseline(
            latency=latency,
            fidelity=fidelity,
            capacity=capacity,
            bell_pair_pool=bell_pair_pool,
            bell_pair_regeneration=bell_pair_regeneration,
        )

        self.latency = latency
        self.fidelity = fidelity
        self.capacity = capacity
        self.bell_pair_pool = bell_pair_pool
        self.bell_pair_regeneration = bell_pair_regeneration
        self.congestion_threshold = congestion_threshold

        self.available_bell_pairs = float(bell_pair_pool)
        self.reserved_bell_pairs = 0.0
        self.consumed_bell_pairs = 0.0

        self.active_load = 0.0

        self.rejected_allocations = 0
        self.congested_windows = 0
        self.saturated_windows = 0
        self.consecutive_congested_windows = 0
        self.peak_utilization = 0.0

    @property
    def key(self):
        a, b = self.source, self.destination
        return (a, b) if a <= b else (b, a)

    @property
    def available_capacity(self):
        return self.capacity - self.active_load

    @property
    def utilization(self):

        if self.capacity <= 0:
            return 1.0 if self.active_load > TOLERANCE else 0.0

        return self.active_load / self.capacity

    @property
    def is_congested(self):
        return self.utilization + TOLERANCE >= self.congestion_threshold

    @property
    def is_saturated(self):
        return self.utilization + TOLERANCE >= 1.0

    @property
    def state(self):

        if self.is_saturated:
            return SATURATED

        if self.is_congested:
            return CONGESTED

        return NORMAL

    @property
    def bell_pair_scarcity(self):

        if self.bell_pair_pool <= 0:
            return 1.0

        return max(0.0, 1.0 - self.available_bell_pairs / self.bell_pair_pool)

    def capacity_would_overflow(self, load):
        return self.active_load + load > self.capacity + TOLERANCE

    def bell_pairs_insufficient(self, bell_pairs):
        return self.available_bell_pairs + TOLERANCE < bell_pairs

    def reserve(self, load, bell_pairs):

        if self.capacity_would_overflow(load) or self.bell_pairs_insufficient(bell_pairs):
            self.rejected_allocations += 1
            raise ValueError(f"Reservation on link {self.key} would violate its limits")

        self.active_load += load
        self.available_bell_pairs -= bell_pairs
        self.reserved_bell_pairs += bell_pairs

        self.peak_utilization = max(self.peak_utilization, self.utilization)

    def release(self, load, bell_pairs, consumed):

        self.active_load -= load
        self.reserved_bell_pairs -= bell_pairs

        if abs(self.active_load) < TOLERANCE:
            self.active_load = 0.0

        if abs(self.reserved_bell_pairs) < TOLERANCE:
            self.reserved_bell_pairs = 0.0

        if self.active_load < 0 or self.reserved_bell_pairs < 0:
            raise RuntimeError(f"Link {self.key} released more than it reserved")

        if consumed:
            self.consumed_bell_pairs += bell_pairs
        else:
            self.available_bell_pairs += bell_pairs

    def regenerate_bell_pairs(self, amount):

        ceiling = self.bell_pair_pool - self.reserved_bell_pairs
        before = self.available_bell_pairs

        self.available_bell_pairs = max(before, min(ceiling, before + amount))

        return self.available_bell_pairs - before

    def observe_window(self):

        if self.is_congested:
            self.congested_windows += 1
            self.consecutive_congested_windows += 1
        else:
            self.consecutive_congested_windows = 0

        if self.is_saturated:
            self.saturated_windows += 1

    def to_dict(self):

        return {
            "source": self.key[0],
            "destination": self.key[1],
            "latency": round(self.latency, 4),
            "fidelity": round(self.fidelity, 6),
            "capacity": self.capacity,
            "bell_pair_pool": self.bell_pair_pool,
            "bell_pair_regeneration": round(self.bell_pair_regeneration, 4),
            "available_bell_pairs": round(self.available_bell_pairs, 4),
            "active_load": round(self.active_load, 4),
            "congestion_threshold": self.congestion_threshold,
            "baseline": asdict(self.baseline),
        }

    def __repr__(self):
        return (
            f"Link({self.key[0]}<->{self.key[1]}, latency={self.latency:.1f}, "
            f"fidelity={self.fidelity:.4f}, capacity={self.capacity}, "
            f"load={self.active_load:.1f}, state={self.state}, "
            f"bell_pairs={self.available_bell_pairs:.1f}/{self.bell_pair_pool})"
        )
