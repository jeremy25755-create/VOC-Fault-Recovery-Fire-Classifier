"""Leakage-free preparation for multi-location indoor-fire classification."""

from __future__ import annotations

import csv
from bisect import bisect_right
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


FEATURE_COLUMNS = (
    "CO2_Room",
    "CO_Room",
    "H2_Room",
    "Humidity_Room",
    "PM05_Room",
    "PM100_Room",
    "PM10_Room",
    "PM25_Room",
    "PM40_Room",
    "PM_Room_Typical_Size",
    "PM_Total_Room",
    "Temperature_Room",
    "UV_Room",
    "VOC_Room_RAW",
)
LABEL_TO_INDEX = {"Background": 0, "Fire": 1, "Nuisance": 2}
INDEX_TO_LABEL = {value: key for key, value in LABEL_TO_INDEX.items()}


@dataclass(frozen=True)
class MinMaxState:
    minimum: np.ndarray
    maximum: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        span = np.maximum(self.maximum - self.minimum, 1e-8)
        return (values - self.minimum) / span


@dataclass(frozen=True)
class _WindowSegment:
    sensor_id: str
    features: np.ndarray
    labels: np.ndarray
    starts: np.ndarray


class MultiSensorWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Lazy windows that never cross Sensor ID, split, or invalid time gaps."""

    def __init__(self, segments: list[_WindowSegment], window_length: int) -> None:
        if not segments:
            raise ValueError("split does not contain any valid windows")
        self.segments = tuple(segments)
        self.window_length = int(window_length)
        counts = np.asarray([len(segment.starts) for segment in segments], dtype=np.int64)
        self._cumulative = np.cumsum(counts)
        self.targets = np.concatenate(
            [
                segment.labels[segment.starts + self.window_length - 1]
                for segment in segments
            ]
        ).astype(np.int64, copy=False)

    def __len__(self) -> int:
        return int(self._cumulative[-1])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        segment_index = bisect_right(self._cumulative, index)
        previous = 0 if segment_index == 0 else int(self._cumulative[segment_index - 1])
        segment = self.segments[segment_index]
        start = int(segment.starts[index - previous])
        window = np.ascontiguousarray(
            segment.features[start : start + self.window_length].T,
            dtype=np.float32,
        )
        target = int(segment.labels[start + self.window_length - 1])
        return torch.from_numpy(window), torch.tensor(target, dtype=torch.long)


@dataclass(frozen=True)
class ClassificationDatasetBundle:
    train: MultiSensorWindowDataset
    validation: MultiSensorWindowDataset
    test: MultiSensorWindowDataset
    static_observations: np.ndarray
    class_weights: np.ndarray
    scaler: MinMaxState
    metadata: dict[str, object]


def _load_csv(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features: list[list[float]] = []
    labels: list[int] = []
    timestamps: list[float] = []
    dates: list[str] = []
    sensor_ids: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"Date", "Sensor_ID", "ternary_label", *FEATURE_COLUMNS}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"CSV is missing columns: {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            sensor_id = row["Sensor_ID"].strip()
            if not sensor_id:
                raise ValueError(f"row {row_number} has an empty Sensor_ID")
            label_text = row["ternary_label"].strip()
            if label_text not in LABEL_TO_INDEX:
                raise ValueError(f"row {row_number} has unknown label: {label_text}")
            try:
                values = [float(row[name]) for name in FEATURE_COLUMNS]
                parsed_time = datetime.fromisoformat(row["Date"].strip())
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid numeric/date value at row {row_number}") from error
            features.append(values)
            labels.append(LABEL_TO_INDEX[label_text])
            timestamps.append(parsed_time.timestamp())
            dates.append(parsed_time.date().isoformat())
            sensor_ids.append(sensor_id)

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    t = np.asarray(timestamps, dtype=np.float64)
    d = np.asarray(dates, dtype="U10")
    s = np.asarray(sensor_ids, dtype=str)
    if x.ndim != 2 or x.shape[1] != len(FEATURE_COLUMNS):
        raise ValueError("unexpected feature matrix shape")
    if len(x) < 3 or not np.isfinite(x).all() or not np.isfinite(t).all():
        raise ValueError("CSV contains insufficient, NaN, or infinite data")
    return x, y, t, d, s


def _valid_window_starts(
    timestamps: np.ndarray,
    window_length: int,
    stride: int,
    max_gap_seconds: float,
) -> np.ndarray:
    if len(timestamps) < window_length:
        return np.empty(0, dtype=np.int64)
    gaps = np.diff(timestamps)
    bad_gap = (gaps <= 0.0) | (gaps > max_gap_seconds)
    prefix = np.concatenate(([0], np.cumsum(bad_gap, dtype=np.int64)))
    starts = np.arange(0, len(timestamps) - window_length + 1, stride, dtype=np.int64)
    ends = starts + window_length - 1
    return starts[(prefix[ends] - prefix[starts]) == 0]


def _parse_iso_date(value: str, name: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{name} must use YYYY-MM-DD format: {value}") from error


def prepare_dataset(
    csv_path: str | Path,
    window_length: int = 60,
    stride: int = 1,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    max_gap_seconds: float = 30.0,
    split_mode: str = "fraction",
    train_start_date: str = "2022-07-04",
    train_end_date: str = "2022-07-06",
    validation_date: str = "2022-07-07",
    test_date: str = "2022-07-08",
) -> ClassificationDatasetBundle:
    """Create leakage-free windows grouped by Sensor ID and data split."""
    path = Path(csv_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if window_length < 2:
        raise ValueError("window_length must be at least 2")
    if stride < 1:
        raise ValueError("stride must be positive")
    if max_gap_seconds <= 0.0:
        raise ValueError("max_gap_seconds must be positive")
    if split_mode not in {"fraction", "date"}:
        raise ValueError("split_mode must be 'fraction' or 'date'")
    if split_mode == "fraction":
        if not 0.0 < train_fraction < 1.0 or not 0.0 < validation_fraction < 1.0:
            raise ValueError("split fractions must be between 0 and 1")
        if train_fraction + validation_fraction >= 1.0:
            raise ValueError("train + validation fractions must be below 1")

    train_start = _parse_iso_date(train_start_date, "train_start_date")
    train_end = _parse_iso_date(train_end_date, "train_end_date")
    valid_day = _parse_iso_date(validation_date, "validation_date")
    test_day = _parse_iso_date(test_date, "test_date")
    if split_mode == "date" and not (train_start <= train_end < valid_day < test_day):
        raise ValueError("date split must satisfy train_start <= train_end < validation < test")

    x, y, timestamps, dates, sensor_ids = _load_csv(path)
    unique_sensor_ids = sorted(set(sensor_ids.tolist()))
    indices_by_sensor: dict[str, np.ndarray] = {}
    split_indices: dict[str, dict[str, np.ndarray]] = {
        "train": {},
        "validation": {},
        "test": {},
    }
    assigned = np.zeros(len(x), dtype=bool)

    for sensor_id in unique_sensor_ids:
        sensor_index = np.flatnonzero(sensor_ids == sensor_id)
        sensor_index = sensor_index[np.argsort(timestamps[sensor_index], kind="stable")]
        indices_by_sensor[sensor_id] = sensor_index
        if split_mode == "date":
            sensor_dates = dates[sensor_index]
            masks = {
                "train": (sensor_dates >= train_start.isoformat())
                & (sensor_dates <= train_end.isoformat()),
                "validation": sensor_dates == valid_day.isoformat(),
                "test": sensor_dates == test_day.isoformat(),
            }
            for name, mask in masks.items():
                selected = sensor_index[mask]
                split_indices[name][sensor_id] = selected
                assigned[selected] = True
        else:
            n_sensor_rows = len(sensor_index)
            train_end_row = int(np.floor(n_sensor_rows * train_fraction))
            validation_end_row = int(
                np.floor(n_sensor_rows * (train_fraction + validation_fraction))
            )
            blocks = {
                "train": sensor_index[:train_end_row],
                "validation": sensor_index[train_end_row:validation_end_row],
                "test": sensor_index[validation_end_row:],
            }
            for name, selected in blocks.items():
                split_indices[name][sensor_id] = selected
                assigned[selected] = True

    train_indices = np.concatenate(list(split_indices["train"].values()))
    if len(train_indices) < 3:
        raise ValueError("training split is empty or too small")
    scaler = MinMaxState(
        minimum=x[train_indices].min(axis=0),
        maximum=x[train_indices].max(axis=0),
    )
    x_scaled = scaler.transform(x).astype(np.float32)

    datasets: dict[str, MultiSensorWindowDataset] = {}
    window_counts: dict[str, int] = {}
    window_counts_by_sensor: dict[str, dict[str, int]] = {}
    row_counts: dict[str, int] = {}
    row_counts_by_sensor: dict[str, dict[str, int]] = {}
    class_counts: dict[str, list[int]] = {}
    end_timestamp_ranges: dict[str, list[str]] = {}

    for name in ("train", "validation", "test"):
        segments: list[_WindowSegment] = []
        window_counts_by_sensor[name] = {}
        row_counts_by_sensor[name] = {}
        end_times: list[float] = []
        for sensor_id in unique_sensor_ids:
            selected = split_indices[name][sensor_id]
            row_counts_by_sensor[name][sensor_id] = int(len(selected))
            starts = _valid_window_starts(
                timestamps[selected],
                window_length,
                stride,
                max_gap_seconds,
            )
            window_counts_by_sensor[name][sensor_id] = int(len(starts))
            if len(starts) == 0:
                continue
            segments.append(
                _WindowSegment(
                    sensor_id=sensor_id,
                    features=np.ascontiguousarray(x_scaled[selected]),
                    labels=np.ascontiguousarray(y[selected]),
                    starts=starts,
                )
            )
            end_times.extend(timestamps[selected][starts + window_length - 1].tolist())
        dataset = MultiSensorWindowDataset(segments, window_length)
        datasets[name] = dataset
        window_counts[name] = len(dataset)
        row_counts[name] = int(sum(row_counts_by_sensor[name].values()))
        class_counts[name] = np.bincount(dataset.targets, minlength=3).astype(int).tolist()
        end_timestamp_ranges[name] = [
            datetime.fromtimestamp(min(end_times)).isoformat(),
            datetime.fromtimestamp(max(end_times)).isoformat(),
        ]

    train_targets = datasets["train"].targets
    counts = np.bincount(train_targets, minlength=3).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError("every class must appear in the training split")
    class_weights = (len(train_targets) / (3.0 * counts)).astype(np.float32)

    within_sensor_deltas = np.concatenate(
        [
            np.diff(timestamps[indices_by_sensor[sensor_id]])
            for sensor_id in unique_sensor_ids
            if len(indices_by_sensor[sensor_id]) > 1
        ]
    )
    split_description = (
        "calendar dates grouped independently by Sensor_ID; windows never cross "
        "Sensor IDs or train/validation/test boundaries"
        if split_mode == "date"
        else "chronological fractions within each Sensor_ID; windows never cross Sensor IDs or splits"
    )
    return ClassificationDatasetBundle(
        train=datasets["train"],
        validation=datasets["validation"],
        test=datasets["test"],
        static_observations=np.ascontiguousarray(x_scaled[train_indices], dtype=np.float64),
        class_weights=class_weights,
        scaler=scaler,
        metadata={
            "csv_path": str(path),
            "sensor_ids": unique_sensor_ids,
            "feature_columns": list(FEATURE_COLUMNS),
            "label_column": "ternary_label",
            "label_to_index": LABEL_TO_INDEX,
            "rows": len(x),
            "rows_used": int(assigned.sum()),
            "rows_excluded_outside_requested_dates": int((~assigned).sum()),
            "window_length": window_length,
            "nominal_window_minutes": window_length * 10.0 / 60.0,
            "stride": stride,
            "max_gap_seconds": max_gap_seconds,
            "split_mode": split_description,
            "split_dates": {
                "train": [train_start.isoformat(), train_end.isoformat()],
                "validation": valid_day.isoformat(),
                "test": test_day.isoformat(),
            }
            if split_mode == "date"
            else None,
            "row_counts_by_split": row_counts,
            "row_counts_by_sensor_and_split": row_counts_by_sensor,
            "window_counts": window_counts,
            "window_counts_by_sensor_and_split": window_counts_by_sensor,
            "class_counts_by_split": class_counts,
            "window_target": "ternary_label at the final row of each Sensor-ID-specific window",
            "window_end_timestamp_ranges": end_timestamp_ranges,
            "timestamp_first": datetime.fromtimestamp(timestamps.min()).isoformat(),
            "timestamp_last": datetime.fromtimestamp(timestamps.max()).isoformat(),
            "within_sensor_timestamp_delta_median_seconds": float(
                np.median(within_sensor_deltas)
            ),
            "within_sensor_timestamp_delta_max_seconds": float(within_sensor_deltas.max()),
            "scaler_fit_scope": "all training rows across included Sensor IDs",
            "scaler_fit_row_count": int(len(train_indices)),
            "scaler_min": scaler.minimum.tolist(),
            "scaler_max": scaler.maximum.tolist(),
            "class_weights": class_weights.tolist(),
        },
    )


def make_loaders(
    bundle: ClassificationDatasetBundle,
    batch_size: int = 20,
    seed: int = 2026,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    generator = torch.Generator().manual_seed(seed)
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    return (
        DataLoader(bundle.train, shuffle=True, generator=generator, **common),
        DataLoader(bundle.validation, shuffle=False, **common),
        DataLoader(bundle.test, shuffle=False, **common),
    )
