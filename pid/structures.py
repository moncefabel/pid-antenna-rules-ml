"""Design of experiments (DOE) and GDS layout of PID antenna test structures.

Each test structure is a MOS transistor whose gate is connected to an antenna
drawn on a given layer, optionally protected by a reverse-biased N+/P diode.
Two antenna shapes are used to separate area-driven from perimeter-driven charging:
  * "plate": a solid rectangle (low perimeter / area)
  * "comb":  narrow fingers (high perimeter / area)

The layouts are demonstrative. They use a generic layer map and are NOT checked
against the design rules of any real technology.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

LAYERS = ["POLY", "M1", "M2", "M3", "M4", "VIA1"]
OXIDE_CLASSES = ["thin", "medium", "thick"]
GATE_AREAS_UM2 = [0.2, 1.0, 4.0]
DIODE_AREAS_UM2 = [0.0, 0.2, 1.0, 4.0]
SHAPES = ["plate", "comb"]
COMB_FINGER_W_UM = 0.4
COMB_SPACE_UM = 0.4

# Generic GDS layer map (layer, datatype)
GDS_LAYERS = {
    "ACTIVE": (1, 0), "POLY": (2, 0), "CONTACT": (3, 0), "M1": (4, 0),
    "VIA1": (5, 0), "M2": (6, 0), "VIA2": (7, 0), "M3": (8, 0), "VIA3": (9, 0),
    "M4": (10, 0), "NPLUS": (11, 0), "PAD": (12, 0), "TEXT": (63, 0),
}


@dataclass
class TestStructure:
    sid: str
    oxide: str
    layer: str
    shape: str
    gate_um2: float
    antenna_ratio: float     # antenna area / gate area
    diode_um2: float

    @property
    def antenna_um2(self) -> float:
        return self.antenna_ratio * self.gate_um2

    @property
    def perimeter_um(self) -> float:
        a = self.antenna_um2
        if self.shape == "plate":
            return 4.0 * np.sqrt(a)  # square plate
        # comb: fingers of width w, total area a -> perimeter ~ 2 * a / w
        return 2.0 * a / COMB_FINGER_W_UM

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(antenna_um2=self.antenna_um2, perimeter_um=self.perimeter_um)
        return d


def build_doe(n_random: int = 420, seed: int = 0) -> pd.DataFrame:
    """Build the DOE: a structured backbone plus space-filling random samples.

    Backbone: for every (oxide, layer, diode) a sweep of antenna ratios on the
    1 um2 plate, which is the classical way antenna limits are characterised.
    Random part: log-uniform sampling of the whole space to let the model learn
    interactions (shape, gate area, layer) that the backbone does not cover.
    """
    rng = np.random.default_rng(seed)
    rows: list[TestStructure] = []
    ar_sweep = np.geomspace(30, 30000, 10)
    k = 0
    for ox in OXIDE_CLASSES:
        for layer in LAYERS:
            for d in DIODE_AREAS_UM2:
                for ar in ar_sweep:
                    rows.append(TestStructure(f"S{k:04d}", ox, layer, "plate", 1.0,
                                              float(np.round(ar, 1)), d))
                    k += 1
    for _ in range(n_random):
        rows.append(TestStructure(
            f"S{k:04d}",
            rng.choice(OXIDE_CLASSES), rng.choice(LAYERS), rng.choice(SHAPES),
            float(rng.choice(GATE_AREAS_UM2)),
            float(np.round(10 ** rng.uniform(np.log10(20), np.log10(50000)), 1)),
            float(rng.choice(DIODE_AREAS_UM2)),
        ))
        k += 1
    return pd.DataFrame([r.to_dict() for r in rows])


# ---------------------------------------------------------------------------
# GDS generation
# ---------------------------------------------------------------------------
def _layer_kw(name: str) -> dict:
    layer, dt = GDS_LAYERS[name]
    return {"layer": layer, "datatype": dt}


def _text_kw() -> dict:
    layer, dt = GDS_LAYERS["TEXT"]
    return {"layer": layer, "texttype": dt}


def _via_stack_to(layer: str) -> list[str]:
    order = ["POLY", "M1", "M2", "M3", "M4"]
    vias = {"M1": "CONTACT", "M2": "VIA1", "M3": "VIA2", "M4": "VIA3"}
    target = "M1" if layer == "VIA1" else layer
    return [vias[l] for l in order[1:order.index(target) + 1]]


def draw_structure(ts: TestStructure):
    """Return a gdstk.Cell drawing one antenna test structure."""
    import gdstk

    cell = gdstk.Cell(ts.sid)
    pad = 60.0
    # Gate, source and drain pads
    for i, name in enumerate(["G", "S", "D", "B"]):
        x0 = i * (pad + 20)
        cell.add(gdstk.rectangle((x0, -pad - 40), (x0 + pad, -40), **_layer_kw("PAD")))
        cell.add(gdstk.Label(name, (x0 + pad / 2, -40 - pad / 2), **_text_kw()))

    # Transistor: active area crossed by a poly gate of the requested area
    l_gate = 0.5
    w_gate = ts.gate_um2 / l_gate
    cell.add(gdstk.rectangle((-1.0, 0), (l_gate + 1.0, w_gate), **_layer_kw("ACTIVE")))
    cell.add(gdstk.rectangle((0, -0.5), (l_gate, w_gate + 0.5), **_layer_kw("POLY")))

    # Connection from gate to the antenna layer (via stack)
    x_c, y_c = 0.0, w_gate + 0.5
    cell.add(gdstk.rectangle((x_c, y_c), (x_c + l_gate, y_c + 2.0), **_layer_kw("POLY")))
    for v in _via_stack_to(ts.layer):
        cell.add(gdstk.rectangle((x_c + 0.1, y_c + 0.6), (x_c + 0.4, y_c + 0.9), **_layer_kw(v)))

    # Antenna
    ant_layer = "M1" if ts.layer == "VIA1" else ts.layer
    x0, y0 = 5.0, y_c
    a = ts.antenna_um2
    if ts.shape == "plate":
        side = float(np.sqrt(a))
        cell.add(gdstk.rectangle((x0, y0), (x0 + side, y0 + side), **_layer_kw(ant_layer)))
        width = side
    else:
        pitch = COMB_FINGER_W_UM + COMB_SPACE_UM
        side = float(np.sqrt(a * pitch / COMB_FINGER_W_UM))  # square envelope
        n_fingers = max(2, int(round(side / pitch)))
        f_len = a / (n_fingers * COMB_FINGER_W_UM)
        for i in range(n_fingers):
            xs = x0 + i * pitch
            cell.add(gdstk.rectangle((xs, y0 + 1.0), (xs + COMB_FINGER_W_UM, y0 + 1.0 + f_len),
                                     **_layer_kw(ant_layer)))
        cell.add(gdstk.rectangle((x0, y0), (x0 + n_fingers * pitch, y0 + 1.0), **_layer_kw(ant_layer)))
        width = n_fingers * pitch
    if ts.layer == "VIA1":  # via-array antenna on top of the M1 plate/comb
        cell.add(gdstk.rectangle((x0, y0), (x0 + min(width, 20.0), y0 + 0.4), **_layer_kw("VIA1")))
    # strap from via stack to antenna
    cell.add(gdstk.rectangle((x_c, y_c), (x0 + 0.5, y_c + 1.0), **_layer_kw(ant_layer)))

    # Protection diode (N+ in P-well) tied to the gate node at the lowest metal
    if ts.diode_um2 > 0:
        s = float(np.sqrt(ts.diode_um2))
        dx, dy = -6.0 - s, y_c
        cell.add(gdstk.rectangle((dx, dy), (dx + s, dy + s), **_layer_kw("ACTIVE")))
        cell.add(gdstk.rectangle((dx - 0.2, dy - 0.2), (dx + s + 0.2, dy + s + 0.2), **_layer_kw("NPLUS")))
        cell.add(gdstk.rectangle((dx, dy), (x_c, dy + 0.5), **_layer_kw("M1")))

    label = f"{ts.sid} {ts.oxide} {ts.layer} {ts.shape} AR={ts.antenna_ratio:g} D={ts.diode_um2:g}"
    cell.add(gdstk.Label(label, (0, -pad - 60), **_text_kw()))
    return cell


def build_test_chip(doe: pd.DataFrame, path: str, max_structures: int = 48) -> list[str]:
    """Write a GDS test chip placing a representative subset of the DOE on a grid."""
    import gdstk

    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    top = gdstk.Cell("PID_TESTCHIP")
    # pick a representative subset: one per (shape, diode, AR decade), thick oxide on M2
    sub = doe.assign(ar_dec=np.floor(np.log10(doe.antenna_ratio)))
    sub = sub.groupby(["shape", "diode_um2", "ar_dec"], as_index=False).first()
    sub = sub.head(max_structures)
    names = []
    pitch_x, pitch_y = 420.0, 520.0
    ncol = 8
    for i, row in enumerate(sub.itertuples(index=False)):
        ts = TestStructure(row.sid, row.oxide, row.layer, row.shape, row.gate_um2,
                           row.antenna_ratio, row.diode_um2)
        c = draw_structure(ts)
        lib.add(c)
        top.add(gdstk.Reference(c, (i % ncol * pitch_x, -(i // ncol) * pitch_y)))
        names.append(ts.sid)
    lib.add(top)
    lib.write_gds(path)
    return names
