"""Train a RelFlow model on the stitched rel-f1 parquet.

Target: predict the finishing ``position`` of each result within a race
(mirrors RelBench's ``results-position`` regression task, framed inside the
race's local context so the model can also read the driver, constructor,
qualifying, and standings context of every other entry in the same race).

The schema follows the rel-f1 shape produced by ``f1_stitching.py``:

* the race is the singleton root and carries the joined circuit descriptors;
* ``results``, ``qualifying``, and ``driver_standings`` are repeated branches;
* driver / constructor / nationality identifiers are ``rf.Entity`` inside
  branches because Entity needs at least two configured slots per observation
  and encodes observation-local repeated equality (a driver appearing in both
  the qualifying and results branches, for example);
* circuit country and race name live once per race, so they are ``rf.Category``
  with a persistent bounded vocabulary.

Run::

    python f1_training.py --parquet rel_f1_stitched.parquet --max-epochs 10
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

import lightning.pytorch as lit
import polars as pl
import relflow as rf
import torch
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.optim import AdamW

assert os.environ.get("WANDB_API_KEY"), "source ~/reid/wandb.env before running"
wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)

def adamw(lr: float, **kwargs):
    # json2vec calls model.optimizer(model); AdamW wants an iterable of params
    return lambda model: AdamW(model.parameters(), lr=lr, **kwargs)


# RelBench rel-f1 official temporal cutoffs.
VALID_TS = datetime(2005, 1, 1)
TEST_TS = datetime(2010, 1, 1)

# Bounded vocabularies. Sizes are soft ceilings sized above the observed
# rel-f1 cardinalities so the Category vocabularies never saturate.
N_RACE_NAMES = 64          # ~52 unique race names in DB
N_COUNTRIES = 96           # ~52 circuit countries
N_STATUS = 256             # ~140 statusIds

# Branch lengths. F1 grids peak around 26 entries; standings hold the full
# active field for a season.
N_DRIVERS_PER_RACE = 32
N_STANDINGS_PER_RACE = 48


def build_model() -> rf.Model:
    return rf.Model(
        d_model=128,
        n_layers=8,
        n_heads=4,
        batch_size=32,
        embed=True,
        optimizer=lambda module: torch.optim.AdamW(
            module.parameters(), lr=1e-3, weight_decay=1e-4
        ),
        year=rf.Number(),
        round=rf.Number(),
        race_name=rf.Category(size=N_RACE_NAMES, query="[*].name"),
        date=rf.DateParts(dateparts=["month_of_year", "day_of_week", "week_of_year"]),
        circuit_country=rf.Category(size=N_COUNTRIES),
        circuit_lat=rf.Number(),
        circuit_lng=rf.Number(),
        circuit_alt=rf.Number(),
        # One record per driver-entry in the race.
        results=rf.Branch(
            length=N_DRIVERS_PER_RACE,
            driverId=rf.Entity(),
            constructorId=rf.Entity(),
            driver_nationality=rf.Entity(),
            constructor_nationality=rf.Entity(),
            grid=rf.Number(),
            laps=rf.Number(),
            points=rf.Number(),
            statusId=rf.Category(size=N_STATUS),
            position=rf.Number(target=True),
        ),
        # Qualifying context: same driver/constructor axes as results.
        qualifying=rf.Branch(
            length=N_DRIVERS_PER_RACE,
            driverId=rf.Entity(),
            constructorId=rf.Entity(),
            driver_nationality=rf.Entity(),
            constructor_nationality=rf.Entity(),
            position=rf.Number(),
        ),
        # Season-to-date driver standings entering the race.
        driver_standings=rf.Branch(
            length=N_STANDINGS_PER_RACE,
            driverId=rf.Entity(),
            driver_nationality=rf.Entity(),
            points=rf.Number(),
            position=rf.Number(),
            wins=rf.Number(),
        ),
    )


def split_by_date(
    frame: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    train = frame.filter(pl.col("date") < VALID_TS)
    validate = frame.filter(
        (pl.col("date") >= VALID_TS) & (pl.col("date") < TEST_TS)
    )
    test = frame.filter(pl.col("date") >= TEST_TS)
    return train, validate, test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parquet",
        type=Path,
        default=Path("rel_f1_stitched.parquet"),
        help="Path to the nested parquet produced by f1_stitching.py.",
    )
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override model.batch_size for larger hardware.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path("checkpoints"),
    )
    parser.add_argument(
        "--artifact", type=Path, default=Path("rel_f1_model.pt"),
        help="Where to write the compact RelFlow artifact after training.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    frame = pl.read_parquet(args.parquet)
    train, validate, test = split_by_date(frame)
    print( 
        f"races: train={len(train):,}  validate={len(validate):,}  "
        f"test={len(test):,}"
    )

    model = build_model()
    if args.batch_size is not None:
        model.batch_size = args.batch_size

    datamodule = rf.PolarsDataModule(
        model=model,
        train=train,
        validate=validate,
        test=test,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )

    checkpoint = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        monitor="loss/validate",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="rel-f1-{epoch:02d}-{loss/validate:.4f}",
    )

    trainer = lit.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint],
        log_every_n_steps=10,
    )

    logger = WandbLogger(
        entity="rebridgers-independent", 
        project="rel-bench/formula1", 
        name=datetime.now().strftime("%Y-%m-%d %H:%M"),
        config={
            "learning_rate": 2e-5,
            "architecture": "Json2Vec",
            "dataset": "Zinc15",
            "min_epochs": 150,
        },
    )


    # phase: pretrain
    model.update(dropout=0.05)
    model.update(rf.where("type") == "entity", p_mask=0.15, p_prune=0.05)
    model.update(rf.where("type") == "category", p_mask=0.15, p_prune=0.05)
    model.update(rf.where("type") == "number", p_mask=0.15, p_prune=0.05)
    model.update(rf.where("type") == "dateparts", p_mask=0.15, p_prune=0.05)
    model.optimizer = adamw(2e-5, weight_decay=0.01)

    trainer(logger).fit(model=model, datamodule=datamodule)

    # phase: finetune
    model.update(rf.where("type") == "entity", p_mask=0.0, p_prune=0.0)
    model.update(rf.where("type") == "category", p_mask=0.0, p_prune=0.0)
    model.update(rf.where("type") == "number", p_mask=0.0, p_prune=0.0)
    model.update(rf.where("type") == "dateparts", p_mask=0.0, p_prune=0.0)
    model.update(rf.where("name") == "mwt", p_mask=0.5, p_prune=0.3)
    model.update(rf.where("name") == "logp", p_mask=0.5, p_prune=0.3)
    model.update(rf.where("name") == "reactive", p_mask=0.5, p_prune=0.3)

    model.optimizer = adamw(2e-5, weight_decay=0.01)
    trainer(logger).test(model=model, datamodule=datamodule, verbose=True)

    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.artifact)
    print(f"Saved model artifact to {args.artifact}")


if __name__ == "__main__":
    main()
