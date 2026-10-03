"""Step 2: emulate the electrical characterisation campaign (dVt, gate leakage)."""
import pandas as pd

from pid.simulate import simulate_campaign

doe = pd.read_csv("results/doe.csv")
data = simulate_campaign(doe, n_wafers=3, n_dies=25)
data.to_csv("results/measurements.csv.gz", index=False)
print(f"{len(data)} measurements, global fail rate {data.fail.mean():.1%}")
print(data.groupby(["oxide", "diode_um2"]).fail.mean().unstack().round(3))
