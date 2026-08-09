import relflow as rf
import numpy as np
import torch
import pandas as pd

NUM_SAMPLES = 100000
NUM_CLUSTERS = 4

id = np.arange(NUM_SAMPLES)

"""
In order to test out our clustering approach, I create a synthetic dataset with IDs which will be clustered. The relationships between x and y, z, and v must be non-sensical when taking in pairwise relation to each other, but must be defined by one of four clustering rules--perhaps a id % k approach will suffice. 
"""

relation_y_1 = lambda x: np.sin(x)
relation_y_2 = lambda x: np.exp(x)
relation_y_3 = lambda x: np.negative(x)
relation_y_4 = lambda x: x

relation_z_1 = lambda x: np.cos(x)
relation_z_2 = lambda x: np.log(x)
relation_z_3 = lambda x: np.negative(x)
relation_z_4 = lambda x: 0


x = np.random.random(NUM_SAMPLES) * 100
y = np.where(id % NUM_CLUSTERS == 0, relation_y_1(x),
             np.where(id % NUM_CLUSTERS == 1, relation_y_2(x),
                      np.where(id % NUM_CLUSTERS == 2, relation_y_3(x),
                               relation_y_4(x))))
z = np.where(id % NUM_CLUSTERS == 0, relation_z_1(x),
             np.where(id % NUM_CLUSTERS == 1, relation_z_2(x),
                      np.where(id % NUM_CLUSTERS == 2, relation_z_3(x),
                               relation_z_4(x)))) 

pd.DataFrame(
    {
        "id": id,
        "x": x,
        "y": y,
        "z": z
    }
).to_parquet("./src/benchmarks/synthetic/cluster_test/data/clusters.parquet")
