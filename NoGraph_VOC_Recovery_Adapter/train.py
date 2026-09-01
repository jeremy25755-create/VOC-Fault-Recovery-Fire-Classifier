"""Train clean-only VOC recovery graphs for frozen no-graph checkpoints."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset


MODEL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_ROOT.parent
BASE_ROOT = PROJECT_ROOT / "RIFTG_Without_Graphs"
sys.path.insert(0, str(BASE_ROOT))

from riftg_fire_no_graph.data import (  # noqa: E402
    FEATURE_COLUMNS,
    INDEX_TO_LABEL,
    MultiSensorWindowDataset,
    _WindowSegment,
    prepare_dataset,
)
from riftg_fire_no_graph.metrics import classification_metrics  # noqa: E402
from riftg_fire_no_graph.model import FireRIFTGNoGraphClassifier  # noqa: E402

sys.path.insert(0, str(MODEL_ROOT))
from model import NoGraphVOCRecoveryClassifier  # noqa: E402


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
    parser.add_argument("--base-checkpoints", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--recovery-epochs", type=int, default=30)
    parser.add_argument("--adapter-epochs", type=int, default=5)
    parser.add_argument("--recovery-learning-rate", type=float, default=0.001)
    parser.add_argument("--adapter-learning-rate", type=float, default=0.0003)
    parser.add_argument("--adapter-distill-weight", type=float, default=1.0)
    parser.add_argument("--train-sample-step", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--recovery-hidden-dim", type=int, default=64)
    parser.add_argument("--recovery-heads", type=int, default=4)
    parser.add_argument("--adapter-hidden-dim", type=int, default=32)
    parser.add_argument("--short-kernel", type=int, default=9)
    parser.add_argument("--long-kernel", type=int, default=21)
    parser.add_argument("--trend-loss-weight", type=float, default=0.2)
    parser.add_argument("--threshold-quantile", type=float, default=0.999)
    parser.add_argument("--noise-level", type=float, default=0.20)
    parser.add_argument("--noise-seed", type=int, default=9100)
    parser.add_argument("--seed", type=int, default=52026)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--clip-to-train-range",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def saved(arguments: dict[str, object], name: str, default: object) -> object:
    value = arguments.get(name, default)
    return default if value is None else value


def find_checkpoints(path: Path) -> list[Path]:
    path = path.resolve()
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    checkpoints = sorted(path.glob("run_*/model.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"no run_*/model.pt files found under {path}")
    return checkpoints


def rebuild_bundle(checkpoint: dict[str, object], data_override: Path | None):
    arguments = dict(checkpoint.get("arguments", {}))
    checkpoint_data = arguments.get("data")
    if data_override is None and checkpoint_data is None:
        raise ValueError("checkpoint has no data path; provide --data")
    data_path = data_override.resolve() if data_override else Path(checkpoint_data).resolve()
    return prepare_dataset(
        data_path,
        window_length=int(saved(arguments, "window_length", 60)),
        stride=int(saved(arguments, "stride", 1)),
        train_fraction=float(saved(arguments, "train_fraction", 0.70)),
        validation_fraction=float(saved(arguments, "validation_fraction", 0.15)),
        max_gap_seconds=float(saved(arguments, "max_gap_seconds", 20.0)),
        split_mode=str(saved(arguments, "split_mode", "date")),
        train_start_date=str(saved(arguments, "train_start_date", "2022-07-04")),
        train_end_date=str(saved(arguments, "train_end_date", "2022-07-06")),
        validation_date=str(saved(arguments, "validation_date", "2022-07-07")),
        test_date=str(saved(arguments, "test_date", "2022-07-08")),
    )


def build_base(checkpoint: dict[str, object], device: torch.device):
    arguments = dict(checkpoint.get("arguments", {}))
    checkpoint_features = tuple(checkpoint.get("feature_columns", FEATURE_COLUMNS))
    if checkpoint_features != FEATURE_COLUMNS:
        raise ValueError("checkpoint feature order differs from current data module")
    model = FireRIFTGNoGraphClassifier(
        n_nodes=len(FEATURE_COLUMNS),
        window_length=int(saved(arguments, "window_length", 60)),
        hidden_dim=int(saved(arguments, "hidden_size", 120)),
        n_classes=3,
        encoder_mode=str(saved(arguments, "encoder_mode", "current-mixer")),
    )
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval()


def make_loader(
    dataset,
    batch_size: int,
    device: torch.device,
    num_workers: int,
    shuffle: bool = False,
    seed: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def add_voc_noise(
    clean: MultiSensorWindowDataset,
    level: float,
    seed: int,
    clip: bool,
) -> MultiSensorWindowDataset:
    if level < 0:
        raise ValueError("noise level must be non-negative")
    voc_index = FEATURE_COLUMNS.index("VOC_Room_RAW")
    generator = np.random.default_rng(seed)
    segments: list[_WindowSegment] = []
    for segment in clean.segments:
        features = segment.features.copy()
        values = features[:, voc_index]
        values += generator.normal(0.0, level, size=len(values)).astype(np.float32)
        if clip:
            np.clip(values, 0.0, 1.0, out=values)
        segments.append(
            _WindowSegment(
                sensor_id=segment.sensor_id,
                features=np.ascontiguousarray(features),
                labels=segment.labels,
                starts=segment.starts,
            )
        )
    return MultiSensorWindowDataset(segments, clean.window_length)


def r2_score(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
    residual = np.square(actual - predicted).sum()
    total = np.square(actual - actual.mean()).sum()
    return float(1.0 - residual / max(total, 1e-12))


@torch.inference_mode()
def recovery_r2(model, loader: DataLoader, device: torch.device) -> float:
    actual_parts: list[np.ndarray] = []
    predicted_parts: list[np.ndarray] = []
    model.eval()
    for windows, _ in loader:
        windows = windows.to(device, non_blocking=True)
        mean, _, _ = model.recover(windows)
        actual = model.recovery.target_trend(windows[:, model.voc_index, :])
        actual_parts.append(actual.cpu().numpy())
        predicted_parts.append(mean.cpu().numpy())
    return r2_score(np.concatenate(actual_parts), np.concatenate(predicted_parts))


def recovery_loss(
    model: NoGraphVOCRecoveryClassifier,
    windows: Tensor,
    trend_weight: float,
) -> Tensor:
    mean, scale, _ = model.recover(windows)
    target = model.recovery.target_trend(windows[:, model.voc_index, :])
    normalized = (target - mean) / scale
    nll = (0.5 * normalized.square() + torch.log(scale)).mean()
    level = nn.functional.smooth_l1_loss(mean, target)
    slope = nn.functional.smooth_l1_loss(
        mean[:, 1:] - mean[:, :-1],
        target[:, 1:] - target[:, :-1],
    )
    return nll + level + trend_weight * slope


def train_recovery(
    model,
    train_loader,
    validation_loader,
    epochs: int,
    learning_rate: float,
    trend_weight: float,
    device: torch.device,
    run_number: int,
) -> list[dict[str, float]]:
    optimizer = torch.optim.AdamW(
        model.recovery_parameters(), lr=learning_rate, weight_decay=1e-4
    )
    best_state: dict[str, Tensor] | None = None
    best_r2 = -np.inf
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.recovery.train()
        total_loss = 0.0
        count = 0
        for windows, _ in train_loader:
            windows = windows.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = recovery_loss(model, windows, trend_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.recovery.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(windows)
            count += len(windows)
        validation_r2 = recovery_r2(model, validation_loader, device)
        mean_loss = total_loss / max(count, 1)
        history.append(
            {"epoch": epoch, "train_loss": mean_loss, "validation_r2": validation_r2}
        )
        if validation_r2 > best_r2:
            best_r2 = validation_r2
            best_state = copy.deepcopy(model.recovery.state_dict())
        print(
            f"[run {run_number:02d}] recovery {epoch:02d}/{epochs} "
            f"loss={mean_loss:.6f} val_R2={validation_r2:.4f}",
            flush=True,
        )
    if best_state is None:
        raise RuntimeError("recovery training produced no checkpoint")
    model.recovery.load_state_dict(best_state)
    model.freeze_recovery()
    return history


@torch.inference_mode()
def calibrate_thresholds(model, loader, quantile: float, device: torch.device):
    relation_parts: list[np.ndarray] = []
    point_parts: list[np.ndarray] = []
    mode_parts: list[np.ndarray] = []
    model.eval()
    for windows, _ in loader:
        windows = windows.to(device, non_blocking=True)
        mean, scale, _ = model.recover(windows)
        relation, point = model.raw_fault_scores(windows, mean, scale)
        repaired = model.repaired_windows(windows, mean)
        repaired_logits, _ = model.fault_logits(repaired)
        relation_parts.append(relation.cpu().numpy())
        point_parts.append(point.cpu().numpy())
        mode_parts.append(repaired_logits.argmax(dim=-1).cpu().numpy())
    relation = np.concatenate(relation_parts)
    point = np.concatenate(point_parts)
    mode = np.concatenate(mode_parts)
    global_relation = float(np.quantile(relation, quantile))
    global_point = float(np.quantile(point, quantile))
    relation_threshold = np.empty(3, dtype=np.float32)
    point_threshold = np.empty(3, dtype=np.float32)
    for class_index in range(3):
        selected = mode == class_index
        if int(selected.sum()) < 100:
            relation_threshold[class_index] = global_relation
            point_threshold[class_index] = global_point
        else:
            relation_threshold[class_index] = np.quantile(
                relation[selected], quantile
            )
            point_threshold[class_index] = np.quantile(point[selected], quantile)
    model.set_thresholds(relation_threshold, point_threshold)
    return relation_threshold.tolist(), point_threshold.tolist()


@torch.inference_mode()
def forced_validation_metrics(model, loader, device: torch.device):
    actual_parts: list[np.ndarray] = []
    predicted_parts: list[np.ndarray] = []
    model.eval()
    for windows, labels in loader:
        logits = model(
            windows.to(device, non_blocking=True), force_fault=True
        )
        actual_parts.append(labels.numpy())
        predicted_parts.append(logits.argmax(dim=-1).cpu().numpy())
    return classification_metrics(
        np.concatenate(actual_parts), np.concatenate(predicted_parts)
    )


def train_adapter(
    model,
    train_loader,
    validation_loader,
    epochs: int,
    learning_rate: float,
    distill_weight: float,
    class_weights: np.ndarray,
    device: torch.device,
    run_number: int,
) -> list[dict[str, float]]:
    optimizer = torch.optim.AdamW(
        model.adapter_parameters(), lr=learning_rate, weight_decay=1e-4
    )
    weight = torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    best_state: dict[str, Tensor] | None = None
    best_f1 = -np.inf
    history: list[dict[str, float]] = []
    temperature = 2.0
    for epoch in range(1, epochs + 1):
        model.adapter.train()
        model.base.eval()
        model.recovery.eval()
        total_loss = 0.0
        count = 0
        for windows, labels in train_loader:
            windows = windows.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.no_grad():
                teacher_logits = model.base(windows)
                mean, _, _ = model.recover(windows)
                repaired = model.repaired_windows(windows, mean)
                encoded = model.base.encoder(repaired)
                repaired_base_logits = model.base.classifier(encoded)
            logits = repaired_base_logits + model.adapter(encoded.detach())
            classification = criterion(logits, labels)
            distillation = nn.functional.kl_div(
                nn.functional.log_softmax(logits / temperature, dim=-1),
                nn.functional.softmax(teacher_logits / temperature, dim=-1),
                reduction="batchmean",
            ) * (temperature**2)
            loss = classification + distill_weight * distillation
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.adapter.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(windows)
            count += len(windows)
        metrics = forced_validation_metrics(model, validation_loader, device)
        mean_loss = total_loss / max(count, 1)
        validation_f1 = float(metrics["macro_f1"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": mean_loss,
                "validation_accuracy": float(metrics["accuracy"]),
                "validation_macro_f1": validation_f1,
            }
        )
        if validation_f1 > best_f1:
            best_f1 = validation_f1
            best_state = copy.deepcopy(model.adapter.state_dict())
        print(
            f"[run {run_number:02d}] adapter {epoch:02d}/{epochs} "
            f"loss={mean_loss:.6f} val_accuracy={metrics['accuracy']:.4f} "
            f"val_macro_F1={validation_f1:.4f}",
            flush=True,
        )
    if best_state is None:
        raise RuntimeError("adapter training produced no checkpoint")
    model.adapter.load_state_dict(best_state)
    model.adapter.eval()
    return history


@torch.inference_mode()
def predict_base(model, loader, device: torch.device):
    actual_parts: list[np.ndarray] = []
    prediction_parts: list[np.ndarray] = []
    probability_parts: list[np.ndarray] = []
    model.eval()
    for windows, labels in loader:
        probabilities = torch.softmax(
            model(windows.to(device, non_blocking=True)), dim=-1
        ).cpu().numpy()
        actual_parts.append(labels.numpy())
        prediction_parts.append(probabilities.argmax(axis=1))
        probability_parts.append(probabilities)
    return (
        np.concatenate(actual_parts),
        np.concatenate(prediction_parts),
        np.concatenate(probability_parts),
    )


@torch.inference_mode()
def predict_robust(model, loader, device: torch.device, force_fault: bool | None):
    actual_parts: list[np.ndarray] = []
    prediction_parts: list[np.ndarray] = []
    probability_parts: list[np.ndarray] = []
    fault_parts: list[np.ndarray] = []
    ratio_parts: list[np.ndarray] = []
    attention_sum = np.zeros(
        (model.recovery.heads, len(model.source_indices)), dtype=np.float64
    )
    count = 0
    model.eval()
    for windows, labels in loader:
        logits, info = model(
            windows.to(device, non_blocking=True),
            return_diagnostics=True,
            force_fault=force_fault,
        )
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
        actual_parts.append(labels.numpy())
        prediction_parts.append(probabilities.argmax(axis=1))
        probability_parts.append(probabilities)
        fault_parts.append(info["detected_fault"].cpu().numpy())
        ratio_parts.append(info["fault_ratio"].cpu().numpy())
        attention_sum += info["recovery_attention"].sum(dim=0).cpu().numpy()
        count += len(windows)
    return (
        np.concatenate(actual_parts),
        np.concatenate(prediction_parts),
        np.concatenate(probability_parts),
        np.concatenate(fault_parts),
        np.concatenate(ratio_parts),
        attention_sum / max(count, 1),
    )


def summarize(runs: list[dict[str, object]], condition: str) -> dict[str, object]:
    result: dict[str, object] = {"mean": {}, "std": {}, "per_class": {}}
    for metric in SCALAR_METRICS:
        values = np.asarray([run[condition][metric] for run in runs], dtype=float)
        result["mean"][metric] = float(values.mean())
        result["std"][metric] = float(values.std(ddof=0))
    for class_index in range(3):
        name = INDEX_TO_LABEL[class_index]
        result["per_class"][name] = {"mean": {}, "std": {}}
        for metric in CLASS_METRICS:
            values = np.asarray(
                [run[condition]["per_class"][name][metric] for run in runs],
                dtype=float,
            )
            result["per_class"][name]["mean"][metric] = float(values.mean())
            result["per_class"][name]["std"][metric] = float(values.std(ddof=0))
    return result


def print_summary(title: str, summary: dict[str, object]) -> None:
    print(f"\n{title} TEST RESULTS (mean +/- SD)", flush=True)
    for metric in SCALAR_METRICS:
        print(
            f"{metric:<16}{summary['mean'][metric]:.4f} +/- "
            f"{summary['std'][metric]:.4f}",
            flush=True,
        )
    print(f"\n{title} PER-CLASS RESULTS (mean +/- SD, %)", flush=True)
    print(f"{'Class':<12}{'F1':>20}{'Recall':>20}{'Precision':>20}", flush=True)
    for class_index in range(3):
        name = INDEX_TO_LABEL[class_index]
        values = summary["per_class"][name]
        print(
            f"{name:<12}"
            f"{100*values['mean']['f1']:8.2f} +/- {100*values['std']['f1']:<7.2f}"
            f"{100*values['mean']['recall']:8.2f} +/- {100*values['std']['recall']:<7.2f}"
            f"{100*values['mean']['precision']:8.2f} +/- {100*values['std']['precision']:<7.2f}",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    if args.recovery_epochs < 1 or args.adapter_epochs < 1:
        raise ValueError("recovery and adapter epochs must be positive")
    if not 0.5 < args.threshold_quantile < 1.0:
        raise ValueError("threshold quantile must be between 0.5 and 1")
    checkpoints = find_checkpoints(args.base_checkpoints)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    runs: list[dict[str, object]] = []
    print(
        f"checkpoints={len(checkpoints)} clean_only_training=True "
        f"test_noise=VOC_Gaussian_{100*args.noise_level:.1f}% device={device}",
        flush=True,
    )
    for run_index, checkpoint_path in enumerate(checkpoints):
        run_number = run_index + 1
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        run_seed = int(checkpoint.get("run_seed", args.seed + run_index))
        set_seed(run_seed)
        bundle = rebuild_bundle(checkpoint, args.data)
        base = build_base(checkpoint, device)
        voc_index = FEATURE_COLUMNS.index("VOC_Room_RAW")
        source_indices = [i for i in range(len(FEATURE_COLUMNS)) if i != voc_index]
        model = NoGraphVOCRecoveryClassifier(
            base_model=base,
            voc_index=voc_index,
            source_indices=torch.as_tensor(source_indices),
            window_length=base.window_length,
            recovery_hidden_dim=args.recovery_hidden_dim,
            recovery_heads=args.recovery_heads,
            short_kernel=args.short_kernel,
            long_kernel=args.long_kernel,
            adapter_hidden_dim=args.adapter_hidden_dim,
        ).to(device)
        train_subset = Subset(
            bundle.train, range(0, len(bundle.train), args.train_sample_step)
        )
        train_loader = make_loader(
            train_subset,
            args.batch_size,
            device,
            args.num_workers,
            True,
            run_seed,
        )
        validation_loader = make_loader(
            bundle.validation, args.batch_size, device, args.num_workers
        )
        clean_loader = make_loader(
            bundle.test, args.batch_size, device, args.num_workers
        )
        noisy_test = add_voc_noise(
            bundle.test,
            args.noise_level,
            args.noise_seed + run_index,
            args.clip_to_train_range,
        )
        noisy_loader = make_loader(
            noisy_test, args.batch_size, device, args.num_workers
        )
        recovery_history = train_recovery(
            model,
            train_loader,
            validation_loader,
            args.recovery_epochs,
            args.recovery_learning_rate,
            args.trend_loss_weight,
            device,
            run_number,
        )
        adapter_history = train_adapter(
            model,
            train_loader,
            validation_loader,
            args.adapter_epochs,
            args.adapter_learning_rate,
            args.adapter_distill_weight,
            bundle.class_weights,
            device,
            run_number,
        )
        relation_threshold, point_threshold = calibrate_thresholds(
            model, validation_loader, args.threshold_quantile, device
        )
        actual, base_clean_pred, base_clean_prob = predict_base(
            base, clean_loader, device
        )
        noisy_actual, base_noisy_pred, base_noisy_prob = predict_base(
            base, noisy_loader, device
        )
        (
            robust_clean_actual,
            robust_clean_pred,
            robust_clean_prob,
            clean_fault,
            clean_ratio,
            clean_attention,
        ) = predict_robust(model, clean_loader, device, None)
        (
            robust_noisy_actual,
            robust_noisy_pred,
            robust_noisy_prob,
            noisy_fault,
            noisy_ratio,
            noisy_attention,
        ) = predict_robust(model, noisy_loader, device, None)
        (
            oracle_actual,
            oracle_pred,
            oracle_prob,
            _,
            _,
            _,
        ) = predict_robust(model, noisy_loader, device, True)
        if not all(
            np.array_equal(actual, other)
            for other in (
                noisy_actual,
                robust_clean_actual,
                robust_noisy_actual,
                oracle_actual,
            )
        ):
            raise RuntimeError("test targets differ between conditions")
        conditions = {
            "base_clean": classification_metrics(actual, base_clean_pred),
            "base_noisy": classification_metrics(actual, base_noisy_pred),
            "robust_clean": classification_metrics(actual, robust_clean_pred),
            "robust_noisy": classification_metrics(actual, robust_noisy_pred),
            "oracle_repaired_noisy": classification_metrics(actual, oracle_pred),
        }
        run_dir = args.output / f"run_{run_number:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        recovery_parameters = sum(p.numel() for p in model.recovery.parameters())
        adapter_parameters = sum(p.numel() for p in model.adapter.parameters())
        source_names = [FEATURE_COLUMNS[i] for i in source_indices]
        attention_mean = noisy_attention.mean(axis=0)
        attention_ranking = [
            {"sensor": source_names[i], "weight": float(attention_mean[i])}
            for i in np.argsort(-attention_mean)
        ]
        clean_fault_rate_by_class = {
            INDEX_TO_LABEL[i]: float(clean_fault[actual == i].mean())
            for i in range(3)
        }
        noisy_fault_rate_by_class = {
            INDEX_TO_LABEL[i]: float(noisy_fault[actual == i].mean())
            for i in range(3)
        }
        run_result = {
            "run": run_number,
            "base_checkpoint": str(checkpoint_path),
            "base_checkpoint_seed": checkpoint.get("run_seed"),
            "module_seed": run_seed,
            "noise_seed": args.noise_seed + run_index,
            "relation_threshold": relation_threshold,
            "point_threshold": point_threshold,
            "clean_fault_rate": float(clean_fault.mean()),
            "noisy_fault_rate": float(noisy_fault.mean()),
            "clean_fault_rate_by_class": clean_fault_rate_by_class,
            "noisy_fault_rate_by_class": noisy_fault_rate_by_class,
            "clean_fault_ratio_mean": float(clean_ratio.mean()),
            "noisy_fault_ratio_mean": float(noisy_ratio.mean()),
            "recovery_parameter_count": recovery_parameters,
            "adapter_parameter_count": adapter_parameters,
            "recovery_history": recovery_history,
            "adapter_history": adapter_history,
            "attention_ranking": attention_ranking,
            **conditions,
        }
        runs.append(run_result)
        torch.save(
            {
                "model_state": {
                    key: value.detach().cpu() for key, value in model.state_dict().items()
                },
                "base_checkpoint": str(checkpoint_path),
                "arguments": vars(args),
                "feature_columns": FEATURE_COLUMNS,
                "source_indices": source_indices,
                "voc_index": voc_index,
                "run_result": run_result,
            },
            run_dir / "model.pt",
        )
        np.savez_compressed(
            run_dir / "predictions.npz",
            actual=actual,
            base_clean_predicted=base_clean_pred,
            base_clean_probabilities=base_clean_prob,
            base_noisy_predicted=base_noisy_pred,
            base_noisy_probabilities=base_noisy_prob,
            robust_clean_predicted=robust_clean_pred,
            robust_clean_probabilities=robust_clean_prob,
            robust_noisy_predicted=robust_noisy_pred,
            robust_noisy_probabilities=robust_noisy_prob,
            oracle_repaired_noisy_predicted=oracle_pred,
            oracle_repaired_noisy_probabilities=oracle_prob,
            clean_fault=clean_fault,
            noisy_fault=noisy_fault,
            clean_fault_ratio=clean_ratio,
            noisy_fault_ratio=noisy_ratio,
            clean_attention=clean_attention,
            noisy_attention=noisy_attention,
        )
        with (run_dir / "metrics.json").open("w", encoding="utf-8") as stream:
            json.dump(run_result, stream, ensure_ascii=False, indent=2, default=str)
        print(
            f"[run {run_number:02d}] base clean/noisy macro_F1="
            f"{conditions['base_clean']['macro_f1']:.4f}/"
            f"{conditions['base_noisy']['macro_f1']:.4f} robust="
            f"{conditions['robust_clean']['macro_f1']:.4f}/"
            f"{conditions['robust_noisy']['macro_f1']:.4f} oracle="
            f"{conditions['oracle_repaired_noisy']['macro_f1']:.4f}",
            flush=True,
        )
        print(
            f"  fault rate clean/noisy={100*clean_fault.mean():.2f}%/"
            f"{100*noisy_fault.mean():.2f}% thresholds are mode-conditional",
            flush=True,
        )
        print(
            "  clean fault rate by class "
            + ", ".join(
                f"{name}={100*clean_fault_rate_by_class[name]:.2f}%"
                for name in ("Background", "Fire", "Nuisance")
            ),
            flush=True,
        )
    condition_names = (
        "base_clean",
        "base_noisy",
        "robust_clean",
        "robust_noisy",
        "oracle_repaired_noisy",
    )
    summaries = {name: summarize(runs, name) for name in condition_names}
    report = {
        "experiment": "frozen no-graph classifier plus clean-only VOC recovery graph and adapter",
        "noise_used_during_training": False,
        "test_fault": {
            "feature": "VOC_Room_RAW",
            "type": "Gaussian",
            "level_fraction_of_training_range": args.noise_level,
        },
        "arguments": vars(args),
        "summaries": summaries,
        "runs": runs,
    }
    result_path = args.output / "summary.json"
    with result_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, default=str)
    for name in condition_names:
        print_summary(name.upper(), summaries[name])
    print(f"\nFull results: {result_path}", flush=True)


if __name__ == "__main__":
    main()
