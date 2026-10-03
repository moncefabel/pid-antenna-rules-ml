"""Render a few cells of the GDS test chip to PNG for quick visual review."""
import gdstk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from pid.structures import GDS_LAYERS

COLORS = {"ACTIVE": "#7fbf7f", "POLY": "#d9534f", "M1": "#5b9bd5", "M2": "#9b59b6", "M3": "#f0ad4e",
          "M4": "#16a085", "NPLUS": "#c8e6c9", "CONTACT": "k", "VIA1": "k", "VIA2": "k", "VIA3": "k", "PAD": "#bbbbbb"}
inv = {v[0]: k for k, v in GDS_LAYERS.items()}

lib = gdstk.read_gds("layout/pid_testchip.gds")
cells = {c.name: c for c in lib.cells}
def label(c):
    t = [l.text for l in c.labels if " AR=" in l.text]
    return t[0] if t else c.name


def ar(c):
    return float(label(c).split("AR=")[1].split()[0])


cand = [c for c in lib.cells if c.name != "PID_TESTCHIP"]
comb = max((c for c in cand if " comb " in label(c) and ar(c) < 3000), key=ar)
plate = max((c for c in cand if " plate " in label(c) and "D=1" in label(c) and ar(c) < 3000), key=ar)
picks = [comb, plate, plate]
fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
for k, (ax, cell) in enumerate(zip(axes, picks)):
    for poly in sorted(cell.polygons, key=lambda p: p.layer == GDS_LAYERS["NPLUS"][0], reverse=True):
        name = inv.get(poly.layer, "?")
        ax.add_patch(Polygon(poly.points, closed=True, fc=COLORS.get(name, "grey"),
                             ec="k", lw=0.2, alpha=0.75 if name != "PAD" else 0.4))
    (x0, y0), (x1, y1) = cell.bounding_box()
    if k == 2:  # zoom on transistor, via stack and protection diode
        x0, y0, x1, y1 = -9, -2, 9, 8
    ax.set_xlim(x0 - 2, x1 + 2); ax.set_ylim(y0 - 2, y1 + 2); ax.set_aspect("equal")
    ax.set_title(label(cell) + (" (zoom: gate, vias, diode)" if k == 2 else ""), fontsize=8)
    ax.set_xlabel("um")
fig.suptitle("PID antenna test structures (generic layers, not DRC checked)")
fig.tight_layout(); fig.savefig("results/fig5_layout_preview.png", dpi=130)
print("results/fig5_layout_preview.png")
