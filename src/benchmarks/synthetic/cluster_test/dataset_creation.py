import relflow as rf
import numpy as np
import torch
import pandas as pd

NUM_SAMPLES = 1_000_000
NUM_CLUSTERS = 4

id = np.arange(NUM_SAMPLES)

relation_y_1 = lambda x: np.sin(x)
relation_y_2 = lambda x: np.log2(x)
relation_y_3 = lambda x: np.negative(x)
relation_y_4 = lambda x: x

relation_z_1 = lambda x: np.cos(x)
relation_z_2 = lambda x: np.log10(x)
relation_z_3 = lambda x: np.negative(x)
relation_z_4 = lambda x: 0


x = np.random.random(NUM_SAMPLES) * 1000
y = np.where(id % NUM_CLUSTERS == 0, relation_y_1(x),
             np.where(id % NUM_CLUSTERS == 1, relation_y_2(x),
                      np.where(id % NUM_CLUSTERS == 2, relation_y_3(x),
                               relation_y_4(x))))
z = np.where(id % NUM_CLUSTERS == 0, relation_z_1(x),
             np.where(id % NUM_CLUSTERS == 1, relation_z_2(x),
                      np.where(id % NUM_CLUSTERS == 2, relation_z_3(x),
                               relation_z_4(x)))) 

df = pd.DataFrame(
    {
        "id": id,
        "x": x,
        "y": y,
        "z": z
    }
)

import os 
print(os.getcwd())

df.iloc[:int(NUM_SAMPLES*0.8),:].to_parquet("./src/benchmarks/synthetic/cluster_test/data/clusters_train.parquet")
df.iloc[int(NUM_SAMPLES*0.8):int(NUM_SAMPLES*0.9),:].to_parquet("./src/benchmarks/synthetic/cluster_test/data/clusters_val.parquet")
df.iloc[int(NUM_SAMPLES*0.9):,:].to_parquet("./src/benchmarks/synthetic/cluster_test/data/clusters_test.parquet")
