"""Synthetic drive-cycle-style current profiles, with a fixed train/test split.

A profile is one episode of cell current I(t) in amps, sampled every
``dt_phys`` seconds. Positive current is discharge, negative is charge; the
Joule heating I^2 R does not depend on the sign.

Each profile is a random sequence of driving segments (city cruise, highway,
acceleration bursts, regenerative braking, rests), and with some probability a
fast-charge block with a tapering tail. The parameters below are illustrative,
chosen to give a realistic mix of light and heavy thermal load.

Profiles are generated from ``default_rng([seed, split_code, index])``, so the
train and test sets are disjoint random streams and fully reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

SPLITS = {"train": 0, "test": 1}

# (probability, duration range in s, C-rate range). Negative C-rate = regen.
DRIVE_SEGMENTS = {
    "cruise": (0.40, (60, 300), (0.2, 0.6)),
    "highway": (0.20, (120, 600), (1.0, 1.8)),
    "burst": (0.20, (5, 30), (2.0, 4.0)),
    "regen": (0.12, (5, 20), (-1.5, -0.5)),
    "rest": (0.08, (30, 120), (0.0, 0.0)),
}
FAST_CHARGE_DURATION_S = (600, 1200)
FAST_CHARGE_C_RATE = (1.5, 3.0)
FAST_CHARGE_TAPER_FRACTION = 0.3  # last 30% of the block tapers to 40% of the peak
NOISE_C_RATE = 0.05


@dataclass(frozen=True)
class Profile:
    current: np.ndarray  # A, one sample per dt_phys
    split: str
    index: int
    has_fast_charge: bool


def generate_profile(
    split: str,
    index: int,
    *,
    seed: int,
    n_samples: int,
    dt: float,
    capacity_Ah: float,
    fast_charge_prob: float,
) -> Profile:
    rng = np.random.default_rng([seed, SPLITS[split], index])
    one_c = capacity_Ah  # 1C in amps
    names = list(DRIVE_SEGMENTS)
    probs = np.array([DRIVE_SEGMENTS[n][0] for n in names])
    probs /= probs.sum()

    def drive(n: int) -> np.ndarray:
        if n <= 0:
            return np.empty(0)
        out = []
        total = 0
        while total < n:
            name = names[rng.choice(len(names), p=probs)]
            _, (d_lo, d_hi), (c_lo, c_hi) = DRIVE_SEGMENTS[name]
            length = max(1, int(rng.uniform(d_lo, d_hi) / dt))
            level = rng.uniform(c_lo, c_hi) * one_c
            noise = 0.0 if name == "rest" else NOISE_C_RATE * one_c
            out.append(level + noise * rng.standard_normal(length))
            total += length
        return np.concatenate(out)[:n]

    has_fc = bool(rng.random() < fast_charge_prob)
    if has_fc:
        fc_len = min(n_samples, int(rng.uniform(*FAST_CHARGE_DURATION_S) / dt))
        peak = rng.uniform(*FAST_CHARGE_C_RATE) * one_c
        taper_len = int(FAST_CHARGE_TAPER_FRACTION * fc_len)
        fc = np.full(fc_len, peak)
        fc[fc_len - taper_len :] = np.linspace(peak, 0.4 * peak, taper_len)
        fc = -fc  # charging current is negative
        start = int(rng.integers(0, n_samples - fc_len + 1))
        current = np.concatenate(
            [drive(start), fc, drive(n_samples - start - fc_len)]
        )
    else:
        current = drive(n_samples)

    return Profile(current=current.astype(float), split=split, index=index, has_fast_charge=has_fc)


class LoadProfiles:
    """The fixed set of train and held-out test profiles defined by a config."""

    def __init__(self, cfg: dict):
        t, p = cfg["time"], cfg["profiles"]
        self.dt = float(t["dt_phys"])
        self.n_samples = int(round(float(t["episode_s"]) / self.dt))
        self.seed = int(p["seed"])
        self.sizes = {"train": int(p["n_train"]), "test": int(p["n_test"])}
        self.capacity_Ah = float(cfg["pack"]["capacity_Ah"])
        self.fast_charge_prob = float(p["fast_charge_prob"])
        self._get = lru_cache(maxsize=None)(self._generate)

    def _generate(self, split: str, index: int) -> Profile:
        if not 0 <= index < self.sizes[split]:
            raise IndexError(f"{split} profile {index} out of range (0..{self.sizes[split] - 1})")
        return generate_profile(
            split,
            index,
            seed=self.seed,
            n_samples=self.n_samples,
            dt=self.dt,
            capacity_Ah=self.capacity_Ah,
            fast_charge_prob=self.fast_charge_prob,
        )

    def get(self, split: str, index: int) -> Profile:
        return self._get(split, index)

    def count(self, split: str) -> int:
        return self.sizes[split]
