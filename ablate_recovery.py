"""Ablation study for the VOC recovery pipeline (detection signal / sensor
embedding) across several VOC fault types.

The model is run once per split to collect per-window diagnostics (clean
logits, repaired logits, relation/point fault scores); the detection-signal
ablation is then recomputed from those arrays.  The global threshold is
re-calibrated on the validation split exactly as in
train_recovery.calibrate_thresholds.

The optional --retrain-no-embedding stage retrains the recovery graph with
sensor_embedding frozen at zero (same seed and recipe as the saved run) to
measure whether the learned sensor identity vector matters.

Checkpoints must come from the current train_recovery.py (no adapter, global
thresholds).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

from evaluate_noise import corrupt_voc_dataset
from train_recovery import (
    build_base,
    find_checkpoints,
    make_loader,
    r2_score,
    rebuild_bundle,
    saved,
    set_seed,
    train_recovery as fit_recovery,
)
from voc_fire.data import FEATURE_COLUMNS, INDEX_TO_LABEL
from voc_fire.metrics import classification_metrics
from voc_fire.recovery import NoGraphVOCRecoveryClassifier

LABELS = [INDEX_TO_LABEL[i] for i in range(3)]
FAULTS = ("gaussian", "bias", "drift", "stuck", "dropout")
SIGNALS = ("both", "relation", "point")
VOC_INDEX = FEATURE_COLUMNS.index("VOC_Room_RAW")
SOURCE_INDICES = [i for i in range(len(FEATURE_COLUMNS)) if i != VOC_INDEX]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-checkpoints", type=Path, required=True,
                        help="train_recovery.py output directory (run_*/model.pt) or one model.pt")
    parser.add_argument("--base-checkpoints", type=Path, default=None,
                        help="override base checkpoint location (run_*/model.pt, same order)")
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--faults", nargs="+", choices=FAULTS, default=list(FAULTS))
    parser.add_argument("--noise-level", type=float, default=None,
                        help="fault magnitude; defaults to the level used in training")
    parser.add_argument("--stuck-value", type=float, default=0.0)
    parser.add_argument("--retrain-no-embedding", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def new_model(base_checkpoint: dict, arguments: dict, device: torch.device):
    base = build_base(base_checkpoint, device)
    return NoGraphVOCRecoveryClassifier(
        base_model=base,
        voc_index=VOC_INDEX,
        source_indices=torch.as_tensor(SOURCE_INDICES),
        window_length=base.window_length,
        recovery_hidden_dim=int(saved(arguments, "recovery_hidden_dim", 64)),
        recovery_heads=int(saved(arguments, "recovery_heads", 4)),
        short_kernel=int(saved(arguments, "short_kernel", 9)),
        long_kernel=int(saved(arguments, "long_kernel", 21)),
    ).to(device)


@torch.inference_mode()
def collect(model, loader, device: torch.device, with_trend: bool = False) -> dict:
    """Per-window diagnostics that do not depend on the threshold."""
    model.eval()
    parts: dict[str, list[np.ndarray]] = defaultdict(list)
    for windows, labels in loader:
        windows = windows.to(device, non_blocking=True)
        _, info = model(windows, return_diagnostics=True)
        parts["actual"].append(labels.numpy())
        parts["clean_logits"].append(info["clean_logits"].cpu().numpy())
        parts["repaired_logits"].append(info["repaired_logits"].cpu().numpy())
        parts["relation"].append(info["relation_score"].cpu().numpy())
        parts["point"].append(info["point_score"].cpu().numpy())
        if with_trend:
            target = model.recovery.target_trend(windows[:, model.voc_index, :])
            parts["trend_actual"].append(target.cpu().numpy())
            parts["trend_pred"].append(info["recovered_voc_trend"].cpu().numpy())
    return {key: np.concatenate(value) for key, value in parts.items()}


def calibrate(split: dict, quantile: float) -> tuple[float, float]:
    return (
        float(np.quantile(split["relation"], quantile)),
        float(np.quantile(split["point"], quantile)),
    )


def detect(split: dict, thresholds: tuple[float, float], signals: str) -> np.ndarray:
    relation_ratio = split["relation"] / max(thresholds[0], 1e-8)
    point_ratio = split["point"] / max(thresholds[1], 1e-8)
    ratio = {
        "both": np.maximum(relation_ratio, point_ratio),
        "relation": relation_ratio,
        "point": point_ratio,
    }[signals]
    return ratio > 1.0


def summarize(actual: np.ndarray, predicted: np.ndarray, fault_rate: float | None = None) -> dict:
    metrics = classification_metrics(actual, predicted)
    result = {"macro_f1": metrics["macro_f1"], "accuracy": metrics["accuracy"]}
    for name in LABELS:
        result[f"f1_{name}"] = metrics["per_class"][name]["f1"]
    if fault_rate is not None:
        result["fault_rate"] = fault_rate
    return result


def condition_metrics(split: dict, detected: np.ndarray) -> dict:
    logits = np.where(detected[:, None], split["repaired_logits"], split["clean_logits"])
    return summarize(split["actual"], logits.argmax(-1), float(detected.mean()))


def analyze(model, loaders: dict, quantile: float, device) -> dict:
    splits = {"val": collect(model, loaders["val"], device)}
    splits["clean"] = collect(model, loaders["clean"], device, with_trend=True)
    for fault, loader in loaders["faults"].items():
        splits[fault] = collect(model, loader, device)
    conditions = ["clean", *loaders["faults"]]
    thresholds = calibrate(splits["val"], quantile)

    grid: dict[str, dict] = {
        "base": {c: summarize(splits[c]["actual"], splits[c]["clean_logits"].argmax(-1)) for c in conditions},
        "oracle": {
            c: condition_metrics(splits[c], np.ones(len(splits[c]["actual"]), dtype=bool))
            for c in conditions
        },
    }
    for signals in SIGNALS:
        grid[signals] = {c: condition_metrics(splits[c], detect(splits[c], thresholds, signals)) for c in conditions}

    clean = splits["clean"]
    r2 = {"overall": r2_score(clean["trend_actual"], clean["trend_pred"])}
    for index, name in enumerate(LABELS):
        mask = clean["actual"] == index
        r2[name] = r2_score(clean["trend_actual"][mask], clean["trend_pred"][mask])
    return {"grid": grid, "recovery_r2": r2}


def reproduction_check(model, analysis: dict, run_result: dict, faults, level, saved_level, saved_clip) -> str:
    grid = analysis["grid"]
    pairs = [(grid["both"]["clean"]["macro_f1"], run_result["robust_clean"]["macro_f1"]),
             (grid["base"]["clean"]["macro_f1"], run_result["base_clean"]["macro_f1"])]
    if "gaussian" in faults and level == saved_level and not saved_clip:
        pairs += [(grid["both"]["gaussian"]["macro_f1"], run_result["robust_noisy"]["macro_f1"]),
                  (grid["oracle"]["gaussian"]["macro_f1"], run_result["oracle_repaired_noisy"]["macro_f1"]),
                  (grid["base"]["gaussian"]["macro_f1"], run_result["base_noisy"]["macro_f1"])]
    difference = max(abs(a - b) for a, b in pairs)
    return (f"max |macro_F1 diff| vs saved metrics = {difference:.2e} over {len(pairs)} checks; "
            f"saved thresholds = relation {float(model.relation_threshold):.4f}, "
            f"point {float(model.point_threshold):.4f}")


def retrain_no_embedding(base_checkpoint, bundle, arguments, run_seed, run_number, device, args):
    set_seed(run_seed)
    model = new_model(base_checkpoint, arguments, device)
    model.recovery.sensor_embedding.data.zero_()
    model.recovery.sensor_embedding.requires_grad_(False)
    batch_size = int(saved(arguments, "batch_size", 1024))
    step = int(saved(arguments, "train_sample_step", 5))
    train_loader = make_loader(
        Subset(bundle.train, range(0, len(bundle.train), step)),
        batch_size, device, args.num_workers, True, run_seed,
    )
    validation_loader = make_loader(bundle.validation, batch_size, device, args.num_workers)
    fit_recovery(model, train_loader, validation_loader,
                 int(saved(arguments, "recovery_epochs", 30)),
                 float(saved(arguments, "recovery_learning_rate", 0.001)),
                 float(saved(arguments, "trend_loss_weight", 0.2)), device, run_number)
    return model


def stats(runs: list[dict], getter) -> tuple[float, float]:
    values = np.asarray([getter(run) for run in runs], dtype=np.float64)
    return float(values.mean()), float(values.std())


def print_table(title: str, rows: list[tuple[str, str]], columns: list[str], runs: list[dict], metric: str) -> None:
    print(f"\n{title}")
    print(f"{'':<34}" + "".join(f"{c:>15}" for c in columns))
    for label, key in rows:
        cells = []
        for column in columns:
            if metric not in runs[0]["grid"][key][column]:
                cells.append("-")
                continue
            mean, std = stats(runs, lambda run: run["grid"][key][column][metric])
            cells.append(f"{100 * mean:6.2f}+/-{100 * std:<4.2f}")
        print(f"{label:<34}" + "".join(f"{cell:>15}" for cell in cells))


def report(name: str, runs: list[dict], faults: list[str], default_only: bool) -> None:
    columns = ["clean", *faults]
    print(f"\n{'=' * 20} {name}  (mean +/- SD over {len(runs)} runs, %) {'=' * 20}")
    mean_r2 = {k: stats(runs, lambda run, k=k: run["recovery_r2"][k]) for k in runs[0]["recovery_r2"]}
    print("recovery R2 (clean test): " + "  ".join(f"{k}={v[0]:+.3f}" for k, v in mean_r2.items()))
    print_table("[A] macro_F1: no recovery / robust (both signals) / oracle repair", [
        ("base, no recovery", "base"),
        ("recovery (robust, both signals)", "both"),
        ("oracle repair (always replace)", "oracle"),
    ], columns, runs, "macro_f1")
    if default_only:
        return
    rows = [(f"{s} signal", s) for s in SIGNALS]
    print_table("[B] Detection-signal ablation: macro_F1", rows, columns, runs, "macro_f1")
    print_table("[B] Detection-signal ablation: fraction flagged as VOC fault "
                "(clean column = false-alarm rate)", rows, columns, runs, "fault_rate")
    print_table("[C] Nuisance F1", [("base, no recovery", "base"), ("recovery (both signals)", "both")],
                columns, runs, "f1_Nuisance")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    recovery_paths = find_checkpoints(args.recovery_checkpoints)
    base_override = find_checkpoints(args.base_checkpoints) if args.base_checkpoints else None
    output = args.output or (
        (args.recovery_checkpoints if args.recovery_checkpoints.is_dir() else args.recovery_checkpoints.parent)
        / "ablation" / "ablation.json"
    )
    all_runs: dict[str, list[dict]] = defaultdict(list)
    raw = []
    print(f"runs={len(recovery_paths)} faults={args.faults} device={device}", flush=True)
    for index, path in enumerate(recovery_paths):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        arguments = dict(checkpoint["arguments"])
        run_result = checkpoint["run_result"]
        base_path = base_override[index] if base_override else Path(checkpoint["base_checkpoint"])
        base_checkpoint = torch.load(base_path, map_location="cpu", weights_only=False)
        bundle = rebuild_bundle(base_checkpoint, args.data)
        quantile = float(saved(arguments, "threshold_quantile", 0.999))
        saved_level = float(saved(arguments, "noise_level", 0.20))
        level = saved_level if args.noise_level is None else args.noise_level
        saved_clip = bool(saved(arguments, "clip_to_train_range", False))
        loaders = {
            "val": make_loader(bundle.validation, args.batch_size, device, args.num_workers),
            "clean": make_loader(bundle.test, args.batch_size, device, args.num_workers),
            "faults": {
                fault: make_loader(
                    corrupt_voc_dataset(
                        bundle.test,
                        minimum=float(bundle.scaler.minimum[VOC_INDEX]),
                        maximum=float(bundle.scaler.maximum[VOC_INDEX]),
                        fault=fault, level=level, stuck_value=args.stuck_value,
                        seed=int(run_result["noise_seed"]), clip=saved_clip,
                    ),
                    args.batch_size, device, args.num_workers,
                )
                for fault in args.faults
            },
        }
        model = new_model(base_checkpoint, arguments, device)
        try:
            model.load_state_dict(checkpoint["model_state"])
        except RuntimeError as error:
            raise RuntimeError(
                f"{path} does not match the current model (it likely predates the removal of the "
                "adapter and per-class thresholds). Re-run train_recovery.py to regenerate it."
            ) from error
        model.eval()
        analysis = analyze(model, loaders, quantile, device)
        print(f"[run {index + 1:02d}] reproduction check: "
              + reproduction_check(model, analysis, run_result, args.faults, level, saved_level, saved_clip),
              flush=True)
        all_runs["trained recovery (saved checkpoints)"].append(analysis)
        entry = {"checkpoint": str(path), "full": analysis}
        if args.retrain_no_embedding:
            variant = retrain_no_embedding(
                base_checkpoint, bundle, arguments, int(run_result["module_seed"]), index + 1, device, args
            )
            variant_analysis = analyze(variant, loaders, quantile, device)
            all_runs["retrained with sensor_embedding = 0"].append(variant_analysis)
            entry["no_embedding"] = variant_analysis
        raw.append(entry)

    for name, runs in all_runs.items():
        report(name, runs, args.faults, default_only=name.startswith("retrained"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"arguments": {k: str(v) for k, v in vars(args).items()}, "runs": raw}, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved: {output}", flush=True)


if __name__ == "__main__":
    main()
