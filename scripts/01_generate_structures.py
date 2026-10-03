"""Step 1: design the test structure DOE and draw a GDS test chip."""
from pathlib import Path

from pid.structures import build_doe, build_test_chip

out = Path("results"); out.mkdir(exist_ok=True)
Path("layout").mkdir(exist_ok=True)
doe = build_doe()
doe.to_csv(out / "doe.csv", index=False)
names = build_test_chip(doe, "layout/pid_testchip.gds")
print(f"DOE: {len(doe)} structures, GDS test chip with {len(names)} cells -> layout/pid_testchip.gds")
