"""Synthetic electrical characterisation campaign.

Emulates what a parametric test campaign on PID test structures returns:
for every structure, on every die of every wafer, a measured threshold voltage
shift relative to an antenna-free reference transistor, and a gate leakage
pass/fail. Noise sources included:
  * wafer-to-wafer plasma variation (lognormal),
  * radial plasma non-uniformity (die position),
  * device-to-device dispersion of the damage (lognormal),
  * intrinsic Vt mismatch of the device and its reference (Pelgrom law),
  * measurement noise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .physics import (OXIDES, WAFER_RADIUS_MM, antenna_current, damage,
                      solve_oxide_stress)

A_VT_MV_UM = 3.0       # Pelgrom coefficient, mV.um
MEAS_NOISE_MV = 0.8
DVT_SPEC_MV = 20.0     # failure criterion on |dVt|


def die_map(n_dies: int, seed: int = 1) -> np.ndarray:
    """Return die radii (mm), roughly uniform in area over the wafer."""
    rng = np.random.default_rng(seed)
    return WAFER_RADIUS_MM * 0.97 * np.sqrt(rng.random(n_dies))


def simulate_campaign(doe: pd.DataFrame, n_wafers: int = 3, n_dies: int = 25,
                      seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    radii = die_map(n_dies, seed + 1)
    frames = []
    for w in range(n_wafers):
        wafer_factor = rng.lognormal(0.0, 0.15)
        for die, r in enumerate(radii):
            df = doe.copy()
            df["wafer"] = f"W{w + 1:02d}"
            df["die"] = die
            df["radius_mm"] = r
            frames.append((df, wafer_factor))
    out = []
    for df, wf in frames:
        dvt = np.zeros(len(df))
        bd = np.zeros(len(df), dtype=bool)
        for ox_name, idx in df.groupby("oxide").indices.items():
            ox = OXIDES[ox_name]
            sub = df.iloc[idx]
            i_ant = antenna_current(sub.antenna_um2.values, sub.perimeter_um.values,
                                    sub.layer.values, sub.radius_mm.values, wf)
            _, j_fn = solve_oxide_stress(i_ant, sub.gate_um2.values, sub.diode_um2.values, ox)
            _, d, b = damage(j_fn, sub.gate_um2.values, ox, rng)
            dvt[idx], bd[idx] = d, b
        sigma = np.sqrt(2) * A_VT_MV_UM / np.sqrt(df.gate_um2.values)
        df = df.assign(
            dvt_mv=dvt + rng.normal(0, sigma) + rng.normal(0, MEAS_NOISE_MV, len(df)),
            gate_leak_fail=bd,
        )
        out.append(df)
    data = pd.concat(out, ignore_index=True)
    data["fail"] = (data.dvt_mv.abs() > DVT_SPEC_MV) | data.gate_leak_fail
    return data
