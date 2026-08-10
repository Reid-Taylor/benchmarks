from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
NUM_SAMPLES = 100_000
NUM_CLUSTERS = 5
EPSILON = 0.05
X_LOW, X_HIGH = 0.0, 10.0

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(SEED)

# Each cluster is a distinct y = f_k(x) manifold, offset so curves do
# not intersect in (x, y) space -- K is unambiguous from the geometry.
relations = [
    lambda x: np.sin(x) + 10.0,
    lambda x: np.cos(x) + 4.0,
    lambda x: np.zeros_like(x) - 2.0,
    lambda x: 0.5 * x - 8.0,
    lambda x: -0.3 * x - 14.0,
]

cluster = rng.integers(0, NUM_CLUSTERS, size=NUM_SAMPLES)
x = rng.uniform(X_LOW, X_HIGH, size=NUM_SAMPLES)

y = np.empty(NUM_SAMPLES)
for k, f in enumerate(relations):
    mask = cluster == k
    y[mask] = f(x[mask])

x = x + rng.normal(0.0, EPSILON, NUM_SAMPLES)
y = y + rng.normal(0.0, EPSILON, NUM_SAMPLES)

df = pd.DataFrame(
    {
        "id": np.arange(NUM_SAMPLES),
        "x": x,
        "y": y,
        # Ground-truth label; drop before fitting, use for scoring.
        "cluster": cluster,
    }
)

# Shuffle so train/val/test are cluster-balanced.
df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

n_train = int(NUM_SAMPLES * 0.8)
n_val = int(NUM_SAMPLES * 0.9)

df.iloc[:n_train, :].to_parquet(DATA_DIR / "clusters_train.parquet")
df.iloc[n_train:n_val, :].to_parquet(DATA_DIR / "clusters_val.parquet")
df.iloc[n_val:, :].to_parquet(DATA_DIR / "clusters_test.parquet")
