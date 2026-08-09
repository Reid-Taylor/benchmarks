import numpy as np
import pandas as pd

NUM_SAMPLES = 100_000
NUM_CLUSTERS = 4
EPSILON = 25

id = np.tile(np.arange(20_000), NUM_SAMPLES // 20_000)

relation_y_1 = lambda x: x**2
relation_y_2 = lambda x: np.log2(x)
relation_y_3 = lambda x: np.negative(x)
relation_y_4 = lambda x: 1/x

x = np.random.random(NUM_SAMPLES) * 1000
y = np.where(id % NUM_CLUSTERS == 0, relation_y_1(x),
             np.where(id % NUM_CLUSTERS == 1, relation_y_2(x),
                      np.where(id % NUM_CLUSTERS == 2, relation_y_3(x),
                               relation_y_4(x))))

x = x + np.random.normal(0, EPSILON, NUM_SAMPLES)
y = y + np.random.normal(0, EPSILON, NUM_SAMPLES)

df = pd.DataFrame(
    {
        "id": id,
        "x": x,
        "y": y,
    }
)

df.iloc[:int(NUM_SAMPLES*0.8),:].to_parquet("./src/benchmarks/synthetic/cluster_test/data/clusters_train.parquet")
df.iloc[int(NUM_SAMPLES*0.8):int(NUM_SAMPLES*0.9),:].to_parquet("./src/benchmarks/synthetic/cluster_test/data/clusters_val.parquet")
df.iloc[int(NUM_SAMPLES*0.9):,:].to_parquet("./src/benchmarks/synthetic/cluster_test/data/clusters_test.parquet")
