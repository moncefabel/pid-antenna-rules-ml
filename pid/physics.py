"""Simplified physical model of plasma induced charging damage (PID).

The model is deliberately compact and illustrative. It reproduces the qualitative
mechanisms described in the PID literature, not the behaviour of any real process:

1. During a plasma step, a floating conductor (the "antenna") connected to a gate
   collects a charging current proportional to its exposed area and perimeter.
2. In steady state that current must flow to the substrate through the gate oxide
   (Fowler-Nordheim and benign direct tunnelling) or through a protection diode.
3. The charge fluence injected by Fowler-Nordheim tunnelling creates traps, which
   shift the threshold voltage (power law) and can trigger oxide breakdown (Weibull).

Units: lengths in um for geometry, cm for current densities, C/cm2 for fluence.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

UM2_TO_CM2 = 1e-8

# Fowler-Nordheim constants for SiO2 (textbook order of magnitude)
FN_A = 1.25e-6  # A / V^2
FN_B = 2.4e8    # V / cm


@dataclass(frozen=True)
class OxideSpec:
    """Gate oxide class. Values are illustrative, not taken from any PDK."""
    name: str
    tox_nm: float
    g_leak: float      # benign (direct tunnelling) leakage, A / (cm2 . V)
    dvt_coef: float    # mV per (C/cm2)^dvt_exp
    dvt_exp: float
    qbd_scale: float   # Weibull scale of charge to breakdown, C/cm2 (for 1 um2)
    qbd_shape: float   # Weibull slope


OXIDES: dict[str, OxideSpec] = {
    "thin":   OxideSpec("thin",   1.8, 2.0e-1, 6.0,  0.55, 40.0, 1.4),
    "medium": OxideSpec("medium", 3.5, 1.0e-4, 12.0, 0.55, 20.0, 1.9),
    "thick":  OxideSpec("thick",  7.0, 1.0e-8, 22.0, 0.55, 12.0, 2.6),
}

# Relative charging intensity of the plasma step that patterns each layer.
LAYER_FACTOR: dict[str, float] = {
    "POLY": 0.6, "M1": 1.0, "M2": 1.0, "M3": 1.2, "M4": 1.4, "VIA1": 0.8,
}
# Fraction of the charging current that is perimeter (edge) driven per layer.
LAYER_PERIM_WEIGHT: dict[str, float] = {
    "POLY": 0.5, "M1": 0.35, "M2": 0.35, "M3": 0.35, "M4": 0.35, "VIA1": 0.1,
}

J_PLASMA_AREA = 2.0e-4   # A/cm2, nominal area charging current density
STEP_TIME_S = 20.0       # effective exposure time of the charging step
WAFER_RADIUS_MM = 150.0
DIODE_VON = 0.7          # V
DIODE_G = 1.0            # A / (cm2 . V) above turn-on (finite diode capability)


def radial_uniformity(r_mm, edge_boost: float = 0.8):
    """Plasma non-uniformity: charging is stronger at the wafer edge."""
    return 1.0 + edge_boost * (np.asarray(r_mm) / WAFER_RADIUS_MM) ** 2


def antenna_current(area_um2, perim_um, layer, r_mm, wafer_factor=1.0):
    """Charging current (A) collected by the antenna during the plasma step."""
    layer = np.asarray(layer)
    lf = np.vectorize(LAYER_FACTOR.get)(layer)
    pw = np.vectorize(LAYER_PERIM_WEIGHT.get)(layer)
    # Perimeter term normalised so a 2 um wide line weighs area and edge equally.
    i_area = (1 - pw) * np.asarray(area_um2)
    i_perim = pw * np.asarray(perim_um) * 1.0
    return (J_PLASMA_AREA * UM2_TO_CM2 * lf * radial_uniformity(r_mm)
            * wafer_factor * (i_area + i_perim))


def fn_current_density(v, tox_nm):
    """Fowler-Nordheim current density (A/cm2) for an oxide voltage v (V)."""
    e = np.maximum(np.asarray(v, dtype=float), 1e-6) / (tox_nm * 1e-7)  # V/cm
    return FN_A * e**2 * np.exp(-FN_B / e)


def solve_oxide_stress(i_ant, gate_um2, diode_um2, oxide: OxideSpec, n_iter: int = 80):
    """Solve the steady-state current balance for the oxide voltage.

    i_ant = A_gate * (J_FN(V) + g_leak * V) + A_diode * G_d * max(V - V_on, 0)

    Returns (V, J_FN) where J_FN is the damaging current density through the gate.
    """
    i_ant = np.asarray(i_ant, dtype=float)
    a_g = np.asarray(gate_um2, dtype=float) * UM2_TO_CM2
    a_d = np.asarray(diode_um2, dtype=float) * UM2_TO_CM2
    lo = np.zeros_like(i_ant)
    hi = np.full_like(i_ant, 40.0)

    def balance(v):
        return (a_g * (fn_current_density(v, oxide.tox_nm) + oxide.g_leak * v)
                + a_d * DIODE_G * np.maximum(v - DIODE_VON, 0.0) - i_ant)

    for _ in range(n_iter):  # vectorised bisection, balance is monotonic in v
        mid = 0.5 * (lo + hi)
        pos = balance(mid) > 0
        hi = np.where(pos, mid, hi)
        lo = np.where(pos, lo, mid)
    v = 0.5 * (lo + hi)
    return v, fn_current_density(v, oxide.tox_nm)


def damage(j_fn, gate_um2, oxide: OxideSpec, rng: np.random.Generator | None = None):
    """Map the injected fluence to a Vt shift (mV) and a breakdown outcome.

    With rng=None the median behaviour is returned and the breakdown output is a
    probability. With an rng, the Vt shift gets lognormal device-to-device
    dispersion and breakdown is sampled.
    """
    q = np.asarray(j_fn) * STEP_TIME_S  # C/cm2
    dvt = oxide.dvt_coef * q ** oxide.dvt_exp
    area_scale = np.asarray(gate_um2, dtype=float)
    p_bd = 1.0 - np.exp(-area_scale * (q / oxide.qbd_scale) ** oxide.qbd_shape)
    if rng is None:
        return q, dvt, p_bd
    dvt = dvt * rng.lognormal(0.0, 0.25, size=np.shape(dvt))
    bd = rng.random(np.shape(p_bd)) < p_bd
    return q, dvt, bd
