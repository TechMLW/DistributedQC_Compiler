import hashlib
import random


SEED_MODULUS = 2 ** 32


def derive_seed(parent_seed, label):

    digest = hashlib.sha256(f"{parent_seed}/{label}".encode("utf-8")).digest()

    return int.from_bytes(digest[:4], "big") % SEED_MODULUS


def derived_rng(parent_seed, label):
    return random.Random(derive_seed(parent_seed, label))
