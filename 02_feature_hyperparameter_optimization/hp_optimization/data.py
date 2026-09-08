from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from .config import AIRPORTS, BATCH_SIZE, LABEL_COL, RANDOM_STATE, VAL_SIZE
from .features import build_feature_matrix
from .runtime import DEVICE, NUM_WORKERS

@dataclass
class SequenceRecord:
    airport: str
    flight_id: str
    timestamps: np.ndarray
    X: np.ndarray
    y: np.ndarray

    @property
    def length(self) -> int:
        return int(len(self.y))

    @property
    def has_pushback(self) -> int:
        return int(np.any(self.y > 0.5))

def build_sequences_for_airport(
    airport: str,
    data_path: Path,
    feature_variant: str,
    track_windows: list[int],
) -> tuple[list[SequenceRecord], list[str]]:
    print(f"Loading {airport}: {data_path}", flush=True)
    df = pd.read_parquet(data_path)
    df = df.sort_values(["flight_id", "timestamp"]).reset_index(drop=True)

    required_columns = [
        "flight_id",
        "timestamp",
        LABEL_COL,
        "compute_gs",
        "compute_track",
    ]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{airport}: missing columns: {missing}")

    sequences: list[SequenceRecord] = []
    feature_names: list[str] | None = None

    for flight_id, flight_df in df.groupby("flight_id", sort=False):
        flight_df = (
            flight_df[required_columns]
            .dropna(subset=[LABEL_COL, "compute_gs", "compute_track"])
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        if flight_df.empty:
            continue

        timestamps = pd.to_datetime(flight_df["timestamp"]).to_numpy()
        gs = flight_df["compute_gs"].to_numpy(dtype=np.float32)
        track = flight_df["compute_track"].to_numpy(dtype=np.float32)
        y = flight_df[LABEL_COL].astype(np.float32).to_numpy()
        y = (y >= 0.5).astype(np.float32)

        X, current_feature_names = build_feature_matrix(
            gs=gs,
            track=track,
            feature_variant=feature_variant,
            track_windows=track_windows,
        )

        if feature_names is None:
            feature_names = current_feature_names
        elif feature_names != current_feature_names:
            raise RuntimeError("Inconsistent feature names within airport")

        sequences.append(
            SequenceRecord(
                airport=airport,
                flight_id=str(flight_id),
                timestamps=timestamps,
                X=X,
                y=y,
            )
        )

    if not sequences or feature_names is None:
        raise ValueError(f"{airport}: no valid sequences")

    positive_flights = sum(sequence.has_pushback for sequence in sequences)
    positive_points = int(sum(sequence.y.sum() for sequence in sequences))
    all_points = int(sum(sequence.length for sequence in sequences))

    print(
        f"{airport}: flights={len(sequences)}, "
        f"positive_flights={positive_flights}, "
        f"positive_points={positive_points}/{all_points}",
        flush=True,
    )
    return sequences, feature_names

def create_airport_splits(
    sequences_by_airport: dict[str, list[SequenceRecord]],
) -> dict[str, dict[str, list[int]]]:
    splits: dict[str, dict[str, list[int]]] = {}

    for airport, sequences in sequences_by_airport.items():
        indices = np.arange(len(sequences))
        labels = np.asarray([sequence.has_pushback for sequence in sequences])

        if len(np.unique(labels)) < 2:
            raise ValueError(f"{airport}: train/validation split needs both classes")

        train_idx, val_idx = train_test_split(
            indices,
            test_size=VAL_SIZE,
            stratify=labels,
            random_state=RANDOM_STATE,
        )
        splits[airport] = {
            "train": sorted(int(index) for index in train_idx),
            "validation": sorted(int(index) for index in val_idx),
        }

    return splits

def collect_fold_sequences(
    sequences_by_airport: dict[str, list[SequenceRecord]],
    airport_splits: dict[str, dict[str, list[int]]],
    test_airport: str,
) -> tuple[list[SequenceRecord], list[SequenceRecord], list[SequenceRecord]]:
    train_sequences: list[SequenceRecord] = []
    validation_sequences: list[SequenceRecord] = []

    for airport in AIRPORTS:
        if airport == test_airport:
            continue
        sequences = sequences_by_airport[airport]
        train_sequences.extend(
            sequences[index] for index in airport_splits[airport]["train"]
        )
        validation_sequences.extend(
            sequences[index] for index in airport_splits[airport]["validation"]
        )

    test_sequences = list(sequences_by_airport[test_airport])
    return train_sequences, validation_sequences, test_sequences

def fit_scaler(sequences: list[SequenceRecord]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(np.concatenate([sequence.X for sequence in sequences], axis=0))
    return scaler

def transform_sequences(
    sequences: list[SequenceRecord],
    scaler: StandardScaler,
) -> list[SequenceRecord]:
    return [
        SequenceRecord(
            airport=sequence.airport,
            flight_id=sequence.flight_id,
            timestamps=sequence.timestamps,
            X=scaler.transform(sequence.X).astype(np.float32),
            y=sequence.y,
        )
        for sequence in sequences
    ]

def build_split_dataframe(
    train_sequences: list[SequenceRecord],
    validation_sequences: list[SequenceRecord],
    test_sequences: list[SequenceRecord],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for split_name, sequences in [
        ("train", train_sequences),
        ("validation", validation_sequences),
        ("oof_test", test_sequences),
    ]:
        for sequence in sequences:
            records.append(
                {
                    "split": split_name,
                    "airport": sequence.airport,
                    "flight_id": sequence.flight_id,
                    "has_pushback": sequence.has_pushback,
                    "length": sequence.length,
                    "n_pushback_points": int(sequence.y.sum()),
                }
            )
    return pd.DataFrame(records)

class PointwiseDataset(Dataset):
    def __init__(self, sequences: list[SequenceRecord]):
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int):
        sequence = self.sequences[index]
        return (
            torch.from_numpy(sequence.X),
            torch.from_numpy(sequence.y),
            sequence.length,
            index,
        )

def collate_pointwise_batch(batch):
    X_list, y_list, lengths, indices = zip(*batch)
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    max_length = int(lengths_tensor.max().item())
    batch_size = len(X_list)
    n_features = int(X_list[0].shape[-1])

    X_padded = torch.zeros(
        batch_size,
        max_length,
        n_features,
        dtype=torch.float32,
    )
    y_padded = torch.zeros(batch_size, max_length, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_length, dtype=torch.float32)

    for batch_index, (X, y, length) in enumerate(zip(X_list, y_list, lengths)):
        X_padded[batch_index, :length, :] = X[:length]
        y_padded[batch_index, :length] = y[:length]
        mask[batch_index, :length] = 1.0

    return (
        X_padded,
        y_padded,
        mask,
        lengths_tensor,
        torch.tensor(indices, dtype=torch.long),
    )

def seed_dataloader_worker(worker_id: int) -> None:
    """Seed Python and NumPy in each DataLoader worker from PyTorch's seed."""
    del worker_id
    worker_seed = int(torch.initial_seed() % (2**32))
    random.seed(worker_seed)
    np.random.seed(worker_seed)

def make_loader(
    sequences: list[SequenceRecord],
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))

    kwargs: dict[str, Any] = {
        "batch_size": BATCH_SIZE,
        "shuffle": shuffle,
        "num_workers": NUM_WORKERS,
        "pin_memory": DEVICE.type == "cuda",
        "collate_fn": collate_pointwise_batch,
        "generator": generator,
        "worker_init_fn": seed_dataloader_worker,
    }
    if NUM_WORKERS > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(PointwiseDataset(sequences), **kwargs)
