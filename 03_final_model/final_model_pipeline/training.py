from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

from .config import (
    DENSE_UNITS, EARLY_STOPPING_PATIENCE, EPOCHS, LR_REDUCE_FACTOR,
    LR_REDUCE_PATIENCE, MIN_LR, TCN_DILATIONS, TCN_FILTERS, TCN_KERNEL_SIZE,
)
from .data import SequenceRecord, make_loader
from .model import PointwiseTCN
from .runtime import DEVICE, USE_AMP

@dataclass
class FoldTrainingResult:
    model: nn.Module
    history: pd.DataFrame
    best_epoch: int
    best_val_aucpr: float
    final_val_metrics: dict[str, float]
    class_weight_0: float
    class_weight_1: float

def calculate_class_weights(
    train_sequences: list[SequenceRecord],
    positive_weight_scale: float,
) -> tuple[float, float]:
    y_train = np.concatenate([sequence.y for sequence in train_sequences])
    n_positive = float(np.sum(y_train == 1.0))
    n_negative = float(np.sum(y_train == 0.0))
    n_total = n_positive + n_negative

    if n_positive == 0 or n_negative == 0:
        raise ValueError("Class balancing requires both classes")

    weight_for_0 = n_total / (2.0 * n_negative)
    weight_for_1 = n_total / (2.0 * n_positive)
    weight_for_1 *= float(positive_weight_scale)
    return float(weight_for_0), float(weight_for_1)

def weighted_masked_loss(
    criterion,
    logits,
    y,
    mask,
    class_weights,
):
    loss_per_point = criterion(logits, y)
    weights = torch.where(y == 1.0, class_weights[1], class_weights[0])
    weighted_mask = weights * mask
    denominator = weighted_mask.sum().clamp_min(1.0)
    return (loss_per_point * weighted_mask).sum() / denominator

def evaluate_pointwise(
    model: nn.Module,
    data_loader: DataLoader,
    criterion,
    class_weights,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_points = 0.0
    all_y: list[np.ndarray] = []
    all_probability: list[np.ndarray] = []

    with torch.inference_mode():
        for X, y, mask, _, _ in data_loader:
            X = X.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            mask = mask.to(DEVICE, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=USE_AMP):
                logits = model(X)
                loss = weighted_masked_loss(
                    criterion,
                    logits,
                    y,
                    mask,
                    class_weights,
                )

            batch_points = float(mask.sum().item())
            total_loss += float(loss.item()) * batch_points
            total_points += batch_points

            probability = torch.sigmoid(logits)
            valid = mask.bool()
            all_y.append(y[valid].detach().cpu().numpy())
            all_probability.append(
                probability[valid].detach().float().cpu().numpy()
            )

    y_true = np.concatenate(all_y)
    probabilities = np.concatenate(all_probability)
    predictions = (probabilities >= 0.5).astype(np.float32)

    metrics = {
        "loss": total_loss / max(total_points, 1.0),
        "accuracy": float(accuracy_score(y_true, predictions)),
    }

    if len(np.unique(y_true)) < 2:
        metrics["aucpr"] = float("nan")
        metrics["aucroc"] = float("nan")
    else:
        metrics["aucpr"] = float(average_precision_score(y_true, probabilities))
        metrics["aucroc"] = float(roc_auc_score(y_true, probabilities))
    return metrics

def train_fold(
    test_airport: str,
    train_sequences: list[SequenceRecord],
    validation_sequences: list[SequenceRecord],
    n_features: int,
    learning_rate: float,
    dropout: float,
    positive_weight_scale: float,
    loader_seed: int,
    run: wandb.sdk.wandb_run.Run,
) -> FoldTrainingResult:
    train_loader = make_loader(
        train_sequences,
        shuffle=True,
        seed=loader_seed,
    )
    validation_loader = make_loader(
        validation_sequences,
        shuffle=False,
        seed=loader_seed + 1,
    )

    model = PointwiseTCN(
        n_features=n_features,
        filters=TCN_FILTERS,
        kernel_size=TCN_KERNEL_SIZE,
        dilations=TCN_DILATIONS,
        dense_units=DENSE_UNITS,
        dropout=dropout,
    ).to(DEVICE)

    weight_0, weight_1 = calculate_class_weights(
        train_sequences,
        positive_weight_scale,
    )
    class_weights = torch.tensor(
        [weight_0, weight_1],
        dtype=torch.float32,
        device=DEVICE,
    )

    criterion = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    amp_scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    best_val_aucpr = -np.inf
    best_epoch = 0
    best_state_dict = None
    early_stop_counter = 0
    lr_reduce_counter = 0
    history: list[dict[str, float]] = []

    epoch_key = f"fold/{test_airport}/epoch"
    run.define_metric(epoch_key)
    for metric_name in [
        "train_loss",
        "val_loss",
        "val_aucpr",
        "val_aucroc",
        "val_accuracy",
        "learning_rate",
    ]:
        run.define_metric(
            f"fold/{test_airport}/{metric_name}",
            step_metric=epoch_key,
        )

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_sum = torch.zeros((), device=DEVICE)
        train_weight_sum = torch.zeros((), device=DEVICE)

        for X, y, mask, _, _ in train_loader:
            X = X.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            mask = mask.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                logits = model(X)
                loss = weighted_masked_loss(
                    criterion,
                    logits,
                    y,
                    mask,
                    class_weights,
                )

            amp_scaler.scale(loss).backward()
            amp_scaler.step(optimizer)
            amp_scaler.update()

            batch_weight = mask.sum()
            train_loss_sum += loss.detach() * batch_weight
            train_weight_sum += batch_weight

        train_loss = float(
            (train_loss_sum / train_weight_sum.clamp_min(1.0)).item()
        )
        validation_metrics = evaluate_pointwise(
            model,
            validation_loader,
            criterion,
            class_weights,
        )
        current_lr = float(optimizer.param_groups[0]["lr"])

        record = {
            "epoch": epoch,
            "learning_rate": current_lr,
            "train_loss": train_loss,
            "val_loss": validation_metrics["loss"],
            "val_aucpr": validation_metrics["aucpr"],
            "val_aucroc": validation_metrics["aucroc"],
            "val_accuracy": validation_metrics["accuracy"],
        }
        history.append(record)

        run.log(
            {
                epoch_key: epoch,
                f"fold/{test_airport}/train_loss": train_loss,
                f"fold/{test_airport}/val_loss": validation_metrics["loss"],
                f"fold/{test_airport}/val_aucpr": validation_metrics["aucpr"],
                f"fold/{test_airport}/val_aucroc": validation_metrics["aucroc"],
                f"fold/{test_airport}/val_accuracy": validation_metrics[
                    "accuracy"
                ],
                f"fold/{test_airport}/learning_rate": current_lr,
            }
        )

        val_aucpr = validation_metrics["aucpr"]
        improved = not np.isnan(val_aucpr) and val_aucpr > best_val_aucpr

        if improved:
            best_val_aucpr = float(val_aucpr)
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            early_stop_counter = 0
            lr_reduce_counter = 0
        else:
            early_stop_counter += 1
            lr_reduce_counter += 1

        if lr_reduce_counter >= LR_REDUCE_PATIENCE:
            old_lr = float(optimizer.param_groups[0]["lr"])
            new_lr = max(old_lr * LR_REDUCE_FACTOR, MIN_LR)
            if new_lr < old_lr:
                for parameter_group in optimizer.param_groups:
                    parameter_group["lr"] = new_lr
            lr_reduce_counter = 0

        if early_stop_counter >= EARLY_STOPPING_PATIENCE:
            break

    if best_state_dict is None:
        raise RuntimeError(f"{test_airport}: no valid best checkpoint")

    model.load_state_dict(best_state_dict)
    final_metrics = evaluate_pointwise(
        model,
        validation_loader,
        criterion,
        class_weights,
    )

    return FoldTrainingResult(
        model=model,
        history=pd.DataFrame(history),
        best_epoch=best_epoch,
        best_val_aucpr=best_val_aucpr,
        final_val_metrics=final_metrics,
        class_weight_0=weight_0,
        class_weight_1=weight_1,
    )

def predict_probabilities(
    model: nn.Module,
    sequences: list[SequenceRecord],
    loader_seed: int,
) -> list[np.ndarray]:
    loader = make_loader(
        sequences,
        shuffle=False,
        seed=loader_seed,
    )
    predictions: list[np.ndarray | None] = [None] * len(sequences)
    model.eval()

    with torch.inference_mode():
        for X, _, _, lengths, indices in loader:
            X = X.to(DEVICE, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                logits = model(X)
            probability = torch.sigmoid(logits).detach().float().cpu().numpy()

            for row, (length, sequence_index) in enumerate(zip(lengths, indices)):
                predictions[int(sequence_index)] = probability[
                    row,
                    : int(length),
                ].astype(np.float32)

    if any(prediction is None for prediction in predictions):
        raise RuntimeError("Missing predictions for one or more sequences")
    return [prediction for prediction in predictions if prediction is not None]
