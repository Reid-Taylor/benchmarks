import os
from datetime import datetime
from pathlib import Path

import lightning.pytorch as lit
import relflow as rf
import wandb
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from relflow.data.datasets.streaming import StreamingDataModule
from relflow.structs.enums import Suffix
from torch.optim import AdamW

# shared Unix account: force wandb to use MY key, never ~/.netrc
assert os.environ.get("WANDB_API_KEY"), "source ~/reid/wandb.env before running"
wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)


def adamw(lr: float, **kwargs):
    # json2vec calls model.optimizer(model); AdamW wants an iterable of params
    return lambda model: AdamW(model.parameters(), lr=lr, **kwargs)

model = rf.Model.from_tree(
    d_model=64,
    n_layers=4,
    n_heads=4,
    batch_size=256,

    x_field = rf.Number(),
    y_field = rf.Number(),
    # z_field = rf.Number(),
    id = rf.Cluster(capacity = 115_000, n_clusters=(3,15), ema_decay=0.99, gumbel_tau=0.1, revive_temperature=0.9)
)


datamodule = StreamingDataModule(
    model=model,
    root=Path(__file__).parent,
    suffix=Suffix.parquet,
    train=r"./src/benchmarks/synthetic/cluster_test/data/clusters_train.parquet$",
    validate=r"./src/benchmarks/synthetic/cluster_test/data/clusters_val.parquet$",
    test=r"./src/benchmarks/synthetic/cluster_test/data/clusters_test.parquet$",
    file_buffer_size=200,
    observation_buffer_size=10_000,
    num_workers=12,
    replacement=True,
)


def trainer(logger):
    return lit.Trainer(
        callbacks=[
            rf.RollbackCheckpoint(monitor="loss/validate", mode="min"),
            EarlyStopping(monitor="loss/validate", mode="min", patience=10),
        ],
        min_epochs=150,
        precision="bf16",
        logger=logger,
        limit_train_batches=100,
        limit_val_batches=250,
        limit_test_batches=1000
    )

logger = WandbLogger(
    entity="rebridgers-independent", 
    project="synthetic", 
    name=datetime.now().strftime("%Y-%m-%d %H:%M"),
    config={
        "learning_rate": 2e-5,
        "architecture": "RelFlow",
        "dataset": "cluster",
        "min_epochs": 150,
    },
)

# phase: pretrain
model.update(dropout=0.05)
model.update(rf.where("type") == "cluster", p_mask=0.25, p_prune=0.05)
model.update(rf.where("type") == "number", p_mask=0.25, p_prune=0.05)
model.optimizer = adamw(2e-5, weight_decay=0.01)

trainer(logger).fit(model=model, datamodule=datamodule)

# phase: finetune
model.update(rf.where("type") == "number", p_mask=0.0, p_prune=0.0)
model.update(rf.where("type") == "cluster", p_mask=0.5, p_prune=0.3)
model.optimizer = adamw(2e-5, weight_decay=0.01)
trainer(logger).fit(model=model, datamodule=datamodule)

# phase: polish
model.optimizer = adamw(2e-6, weight_decay=0.01)
trainer(logger).fit(model=model, datamodule=datamodule)

model.save("checkpoint.ckpt")
trainer(logger).test(model=model, datamodule=datamodule)

