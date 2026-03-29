from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import LSTM_MODEL_PATH, RANDOM_STATE, SEQUENCE_LENGTH
from .evaluation import metrics_row


torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    # This can only be set once per process, so ignore repeated initialization.
    pass


SEQUENCE_NUMERIC_COLS = [
    "capacity",
    "latitude",
    "longitude",
    "occupancy_clean",
    "available_spaces",
    "utilization",
    "queue_length",
    "is_special_day",
    "is_weekend",
    "hour",
    "minute",
    "day_of_week",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "traffic_level",
    "vehicle_priority",
    "capacity_pressure",
    "parking_stress_index",
]

SEQUENCE_CATEGORICAL_COLS = ["system_code", "vehicle_type", "traffic_condition_nearby", "day_name"]


class SequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]


class LSTMRegressor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )
        self.head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.lstm(inputs)
        last_hidden = outputs[:, -1, :]
        return self.head(last_hidden).squeeze(-1)


@dataclass
class SequenceBundle:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    metadata_test: pd.DataFrame
    feature_columns: list[str]
    scaler_mean: np.ndarray
    scaler_std: np.ndarray


def build_sequence_bundle(clean_df: pd.DataFrame, split_time: pd.Timestamp) -> SequenceBundle:
    encoded = pd.get_dummies(
        clean_df[SEQUENCE_CATEGORICAL_COLS],
        prefix=SEQUENCE_CATEGORICAL_COLS,
        dtype=float,
    )
    design = pd.concat([clean_df[SEQUENCE_NUMERIC_COLS].reset_index(drop=True), encoded.reset_index(drop=True)], axis=1)
    feature_columns = design.columns.tolist()

    sequences: list[np.ndarray] = []
    targets: list[float] = []
    metadata_rows: list[dict[str, object]] = []

    for (_, _), group in clean_df.groupby(["system_code", "session_id"], sort=True):
        group = group.sort_values("time_slot")
        if len(group) < SEQUENCE_LENGTH + 3:
            continue
        indices = group.index.to_list()
        for end_position in range(SEQUENCE_LENGTH - 1, len(indices) - 2):
            target_position = end_position + 2
            current_indices = indices[end_position - SEQUENCE_LENGTH + 1 : end_position + 1]
            x_window = design.loc[current_indices, feature_columns].to_numpy(dtype=np.float32)
            target_value = float(clean_df.loc[indices[target_position], "utilization"])
            target_time = pd.Timestamp(clean_df.loc[indices[target_position], "time_slot"])
            current_time = pd.Timestamp(clean_df.loc[indices[end_position], "time_slot"])
            sequences.append(x_window)
            targets.append(target_value)
            metadata_rows.append(
                {
                    "system_code": clean_df.loc[indices[end_position], "system_code"],
                    "time_slot": current_time,
                    "target_time_slot": target_time,
                    "capacity": float(clean_df.loc[indices[end_position], "capacity"]),
                    "target_utilization_1h": target_value,
                }
            )

    x = np.stack(sequences)
    y = np.asarray(targets, dtype=np.float32)
    metadata = pd.DataFrame(metadata_rows)

    train_mask = metadata["target_time_slot"] <= split_time
    x_train = x[train_mask.to_numpy()]
    y_train = y[train_mask.to_numpy()]
    x_test = x[~train_mask.to_numpy()]
    y_test = y[~train_mask.to_numpy()]
    metadata_test = metadata.loc[~train_mask].reset_index(drop=True)

    mean = x_train.reshape(-1, x_train.shape[-1]).mean(axis=0)
    std = x_train.reshape(-1, x_train.shape[-1]).std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    return SequenceBundle(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        metadata_test=metadata_test,
        feature_columns=feature_columns,
        scaler_mean=mean,
        scaler_std=std,
    )


def train_lstm_forecaster(bundle: SequenceBundle) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(bundle.x_train) == 0 or len(bundle.x_test) == 0:
        raise ValueError("LSTM sequence bundle is empty. Check the sequence preparation logic.")

    validation_size = max(1, int(len(bundle.x_train) * 0.1))
    x_val = bundle.x_train[-validation_size:]
    y_val = bundle.y_train[-validation_size:]
    x_train = bundle.x_train[:-validation_size]
    y_train = bundle.y_train[:-validation_size]

    train_loader = DataLoader(SequenceDataset(x_train, y_train), batch_size=128, shuffle=True)
    val_loader = DataLoader(SequenceDataset(x_val, y_val), batch_size=256, shuffle=False)
    test_loader = DataLoader(SequenceDataset(bundle.x_test, bundle.y_test), batch_size=256, shuffle=False)

    model = LSTMRegressor(input_dim=bundle.x_train.shape[-1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    best_state = None
    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0

    for _epoch in range(25):
        model.train()
        for features, targets in train_loader:
            optimizer.zero_grad()
            predictions = model(features)
            loss = loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for features, targets in val_loader:
                predictions = model(features)
                val_losses.append(loss_fn(predictions, targets).item())
        mean_val_loss = float(np.mean(val_losses))
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_state = {key: value.clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    predictions = []
    with torch.no_grad():
        for features, _targets in test_loader:
            predictions.append(model(features).numpy())
    y_pred = np.concatenate(predictions)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_columns": bundle.feature_columns,
            "sequence_length": SEQUENCE_LENGTH,
            "scaler_mean": bundle.scaler_mean,
            "scaler_std": bundle.scaler_std,
        },
        LSTM_MODEL_PATH,
    )

    metrics = metrics_row("LSTM", bundle.y_test, y_pred)
    prediction_frame = bundle.metadata_test.copy()
    prediction_frame["model"] = "LSTM"
    prediction_frame["predicted_utilization_1h"] = y_pred
    prediction_frame["predicted_available_spaces_1h"] = (
        prediction_frame["capacity"] * (1 - prediction_frame["predicted_utilization_1h"].clip(0, 1))
    ).round(0)
    return metrics, prediction_frame
