"""Evaluate trained no-graph checkpoints with test-only VOC sensor faults.

The clean test data and the corrupted test data are evaluated with the exact
same trained weights.  Noise is injected into the test split only, after the
training-fitted min-max transform.  Therefore ``--noise-level 0.20`` means a
Gaussian standard deviation equal to 20% of the training VOC range.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


from voc_fire.data import (
    FEATURE_COLUMNS,
    INDEX_TO_LABEL,
    MultiSensorWindowDataset,
    _WindowSegment,
    prepare_dataset,
)
from voc_fire.metrics import classification_metrics
from voc_fire.model import FireRIFTGNoGraphClassifier


SCALAR_METRICS = (
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
)
CLASS_METRICS = ("f1", "recall", "precision")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoints",
        type=Path,
        required=True,
        help="A model.pt file or a result directory containing run_*/model.pt.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="CSV override. By default, each checkpoint's original CSV is used.",
    )
    parser.add_argument(
        "--fault",
        choices=("gaussian", "bias", "drift", "stuck", "dropout"),
        default="gaussian",
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        default=0.20,
        help="Fault magnitude as a fraction of the training VOC range.",
    )
    parser.add_argument(
        "--stuck-value",
        type=float,
        default=0.0,
        help="Raw VOC value used only when --fault stuck is selected.",
    )
    parser.add_argument("--noise-seed", type=int, default=9100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--clip-to-train-range",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Clip corrupted normalized VOC values to [0, 1].",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory; defaults to <checkpoint-dir>/voc_noise_evaluation.",
    )
    return parser.parse_args()


def find_checkpoints(path: Path) -> list[Path]:
    resolved = path.resolve()
    if resolved.is_file():
        if resolved.name != "model.pt":
            raise ValueError(f"checkpoint file must be named model.pt: {resolved}")
        return [resolved]
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    checkpoints = sorted(resolved.glob("run_*/model.pt"))
    if not checkpoints:
        checkpoints = sorted(resolved.rglob("model.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"no model.pt files found under {resolved}")
    return checkpoints


def saved_argument(arguments: dict[str, object], name: str, default: object) -> object:
    value = arguments.get(name, default)
    return default if value is None else value


def rebuild_test_bundle(checkpoint: dict[str, object], csv_override: Path | None):
    arguments = dict(checkpoint.get("arguments", {}))
    checkpoint_data = arguments.get("data")
    if csv_override is None and checkpoint_data is None:
        raise ValueError("checkpoint has no saved data path; provide --data")
    csv_path = csv_override.resolve() if csv_override is not None else Path(checkpoint_data).resolve()
    return prepare_dataset(
        csv_path,
        window_length=int(saved_argument(arguments, "window_length", 60)),
        stride=int(saved_argument(arguments, "stride", 1)),
        train_fraction=float(saved_argument(arguments, "train_fraction", 0.70)),
        validation_fraction=float(saved_argument(arguments, "validation_fraction", 0.15)),
        max_gap_seconds=float(saved_argument(arguments, "max_gap_seconds", 30.0)),
        split_mode=str(saved_argument(arguments, "split_mode", "fraction")),
        train_start_date=str(saved_argument(arguments, "train_start_date", "2022-07-04")),
        train_end_date=str(saved_argument(arguments, "train_end_date", "2022-07-06")),
        validation_date=str(saved_argument(arguments, "validation_date", "2022-07-07")),
        test_date=str(saved_argument(arguments, "test_date", "2022-07-08")),
    )


def build_model(checkpoint: dict[str, object], device: torch.device) -> FireRIFTGNoGraphClassifier:
    arguments = dict(checkpoint.get("arguments", {}))
    feature_columns = tuple(checkpoint.get("feature_columns", FEATURE_COLUMNS))
    if feature_columns != FEATURE_COLUMNS:
        raise ValueError(
            "checkpoint feature order differs from the current no-graph data module"
        )
    model = FireRIFTGNoGraphClassifier(
        n_nodes=len(feature_columns),
        window_length=int(saved_argument(arguments, "window_length", 60)),
        hidden_dim=int(saved_argument(arguments, "hidden_size", 120)),
        n_classes=3,
        encoder_mode=str(saved_argument(arguments, "encoder_mode", "current-mixer")),
    )
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval()


def corrupt_voc_dataset(
    clean: MultiSensorWindowDataset,
    *,
    minimum: float,
    maximum: float,
    fault: str,
    level: float,
    stuck_value: float,
    seed: int,
    clip: bool,
) -> MultiSensorWindowDataset:
    """Clone a test dataset and corrupt each underlying VOC row exactly once."""
    if level < 0.0:
        raise ValueError("noise level must be non-negative")
    voc_index = FEATURE_COLUMNS.index("VOC_Room_RAW")
    span = max(maximum - minimum, 1e-8)
    generator = np.random.default_rng(seed)
    noisy_segments: list[_WindowSegment] = []

    for segment in clean.segments:
        features = segment.features.copy()
        values = features[:, voc_index]
        if fault == "gaussian":
            values += generator.normal(0.0, level, size=len(values)).astype(np.float32)
        elif fault == "bias":
            values += np.float32(level)
        elif fault == "drift":
            values += np.linspace(0.0, level, len(values), dtype=np.float32)
        elif fault == "stuck":
            values.fill(np.float32((stuck_value - minimum) / span))
        elif fault == "dropout":
            values.fill(np.float32((0.0 - minimum) / span))
        else:
            raise ValueError(f"unsupported fault: {fault}")
        if clip:
            np.clip(values, 0.0, 1.0, out=values)
        noisy_segments.append(
            _WindowSegment(
                sensor_id=segment.sensor_id,
                features=np.ascontiguousarray(features),
                labels=segment.labels,
                starts=segment.starts,
            )
        )
    return MultiSensorWindowDataset(noisy_segments, clean.window_length)


@torch.inference_mode()
def predict(
    model: FireRIFTGNoGraphClassifier,
    dataset: MultiSensorWindowDataset,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    actual_parts: list[np.ndarray] = []
    predicted_parts: list[np.ndarray] = []
    probability_parts: list[np.ndarray] = []
    for windows, labels in loader:
        logits = model(windows.to(device, non_blocking=True))
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
        actual_parts.append(labels.numpy())
        predicted_parts.append(probabilities.argmax(axis=1))
        probability_parts.append(probabilities)
    return (
        np.concatenate(actual_parts),
        np.concatenate(predicted_parts),
        np.concatenate(probability_parts),
    )


def metric_delta(clean: dict[str, object], noisy: dict[str, object]) -> dict[str, object]:
    return {
        **{name: float(noisy[name]) - float(clean[name]) for name in SCALAR_METRICS},
        "per_class": {
            class_name: {
                name: float(noisy["per_class"][class_name][name])
                - float(clean["per_class"][class_name][name])
                for name in CLASS_METRICS
            }
            for class_name in (INDEX_TO_LABEL[index] for index in range(3))
        },
    }


def summarize_runs(runs: list[dict[str, object]], condition: str) -> dict[str, object]:
    result: dict[str, object] = {"mean": {}, "std": {}, "per_class": {}}
    for name in SCALAR_METRICS:
        values = np.asarray([run[condition][name] for run in runs], dtype=np.float64)
        result["mean"][name] = float(values.mean())
        result["std"][name] = float(values.std(ddof=0))
    for class_index in range(3):
        class_name = INDEX_TO_LABEL[class_index]
        result["per_class"][class_name] = {"mean": {}, "std": {}}
        for name in CLASS_METRICS:
            values = np.asarray(
                [run[condition]["per_class"][class_name][name] for run in runs],
                dtype=np.float64,
            )
            result["per_class"][class_name]["mean"][name] = float(values.mean())
            result["per_class"][class_name]["std"][name] = float(values.std(ddof=0))
    return result


def print_condition(name: str, summary: dict[str, object]) -> None:
    print(f"\n{name} TEST RESULTS (mean +/- SD)", flush=True)
    for metric in SCALAR_METRICS:
        print(
            f"{metric:<16}{summary['mean'][metric]:.4f} +/- "
            f"{summary['std'][metric]:.4f}",
            flush=True,
        )
    print(f"\n{name} PER-CLASS RESULTS (mean +/- SD, %)", flush=True)
    print(f"{'Class':<12}{'F1':>20}{'Recall':>20}{'Precision':>20}", flush=True)
    for class_index in range(3):
        class_name = INDEX_TO_LABEL[class_index]
        values = summary["per_class"][class_name]
        print(
            f"{class_name:<12}"
            f"{100 * values['mean']['f1']:8.2f} +/- {100 * values['std']['f1']:<7.2f}"
            f"{100 * values['mean']['recall']:8.2f} +/- {100 * values['std']['recall']:<7.2f}"
            f"{100 * values['mean']['precision']:8.2f} +/- {100 * values['std']['precision']:<7.2f}",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    checkpoints = find_checkpoints(args.checkpoints)
    root = args.checkpoints.resolve()
    default_output_root = root if root.is_dir() else root.parent
    output = (
        args.output.resolve()
        if args.output is not None
        else default_output_root / "voc_noise_evaluation"
    )
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    runs: list[dict[str, object]] = []

    print(
        f"checkpoints={len(checkpoints)} fault={args.fault} "
        f"noise_level={args.noise_level:.4f} device={device}",
        flush=True,
    )
    for run_index, checkpoint_path in enumerate(checkpoints, start=1):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        bundle = rebuild_test_bundle(checkpoint, args.data)
        voc_index = FEATURE_COLUMNS.index("VOC_Room_RAW")
        noisy_test = corrupt_voc_dataset(
            bundle.test,
            minimum=float(bundle.scaler.minimum[voc_index]),
            maximum=float(bundle.scaler.maximum[voc_index]),
            fault=args.fault,
            level=args.noise_level,
            stuck_value=args.stuck_value,
            seed=args.noise_seed + run_index - 1,
            clip=args.clip_to_train_range,
        )
        model = build_model(checkpoint, device)
        actual, clean_predicted, clean_probabilities = predict(
            model,
            bundle.test,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        noisy_actual, noisy_predicted, noisy_probabilities = predict(
            model,
            noisy_test,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        if not np.array_equal(actual, noisy_actual):
            raise RuntimeError("clean and corrupted test targets differ")
        clean_metrics = classification_metrics(actual, clean_predicted)
        noisy_metrics = classification_metrics(actual, noisy_predicted)
        delta = metric_delta(clean_metrics, noisy_metrics)
        run_name = checkpoint_path.parent.name
        np.savez_compressed(
            output / f"{run_name}_predictions.npz",
            actual=actual,
            clean_predicted=clean_predicted,
            clean_probabilities=clean_probabilities,
            noisy_predicted=noisy_predicted,
            noisy_probabilities=noisy_probabilities,
        )
        runs.append(
            {
                "run": run_index,
                "checkpoint": str(checkpoint_path),
                "checkpoint_seed": checkpoint.get("run_seed"),
                "noise_seed": args.noise_seed + run_index - 1,
                "clean": clean_metrics,
                "noisy": noisy_metrics,
                "delta_noisy_minus_clean": delta,
            }
        )
        print(
            f"[{run_index:02d}/{len(checkpoints):02d}] {run_name} "
            f"clean_macro_F1={clean_metrics['macro_f1']:.4f} "
            f"noisy_macro_F1={noisy_metrics['macro_f1']:.4f} "
            f"delta={delta['macro_f1']:+.4f}",
            flush=True,
        )

    clean_summary = summarize_runs(runs, "clean")
    noisy_summary = summarize_runs(runs, "noisy")
    report = {
        "experiment": "test-only VOC_Room_RAW fault on trained no-graph checkpoints",
        "fault": args.fault,
        "noise_level_fraction_of_training_voc_range": args.noise_level,
        "noise_seed": args.noise_seed,
        "stuck_value_raw": args.stuck_value if args.fault == "stuck" else None,
        "clip_to_training_range": args.clip_to_train_range,
        "checkpoint_count": len(checkpoints),
        "clean_summary": clean_summary,
        "noisy_summary": noisy_summary,
        "runs": runs,
    }
    with (output / "voc_noise_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, default=str)
    print_condition("CLEAN", clean_summary)
    print_condition("VOC-NOISY", noisy_summary)
    print(f"\nFull results: {output / 'voc_noise_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
