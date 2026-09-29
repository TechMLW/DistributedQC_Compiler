import os
import secrets
from dataclasses import dataclass

from utils.seeding import derive_seed


SEED_COMPONENTS = ("circuit", "topology", "partition", "network", "qaoa")


@dataclass(frozen=True)
class SeedPlan:
    master_seed: int
    derived: dict

    @classmethod
    def from_master(cls, master_seed):

        if master_seed < 0:
            raise ValueError("The master seed must be a non-negative integer")

        return cls(
            master_seed=master_seed,
            derived={
                f"{component}_seed": derive_seed(master_seed, component)
                for component in SEED_COMPONENTS
            },
        )

    def seed_for(self, component):
        return self.derived[f"{component}_seed"]

    def to_dict(self):
        return {"master_seed": self.master_seed, "derived_seeds": dict(self.derived)}


def generate_master_seed(results_directory, seed_range):

    low, high = seed_range

    while True:

        candidate = low + secrets.randbelow(high - low + 1)

        if not os.path.exists(os.path.join(results_directory, f"simulation_{candidate}")):
            return candidate
