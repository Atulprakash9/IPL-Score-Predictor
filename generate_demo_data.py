"""Create SYNTHETIC examples to test the pipeline. NOT historical IPL data."""
from pathlib import Path
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
rows = []
for _ in range(1500):
    over = int(rng.integers(5, 20))
    balls = int(rng.integers(0, 6))
    overs = over + balls / 6
    wickets = int(rng.integers(0, min(10, over // 2 + 2)))
    rate = float(rng.uniform(5.5, 11.5))
    runs = max(0, round(overs * rate + rng.normal(0, 7)))
    projected = runs + (20 - overs) * max(4.5, rate - 0.28 * wickets)
    final_score = max(runs, round(projected + rng.normal(0, 13)))
    rows.append((over, balls, runs, wickets, final_score))
path = Path(__file__).resolve().parent / 'data' / 'demo_synthetic.csv'
pd.DataFrame(rows, columns=['overs','balls','runs','wickets','final_score']).to_csv(path, index=False)
print(f'Created {path} (synthetic demonstration data, NOT real IPL data)')
