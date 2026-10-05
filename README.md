# VOC Fault-Recovery Fire Classifier

Clean-data-only indoor-fire classification with selective VOC sensor recovery.
The base classifier is frozen before a directed recovery graph learns to
reconstruct `VOC_Room_RAW` from the other 13 sensors. At inference time, clean
windows bypass recovery, while detected VOC faults use the reconstructed trend.

## Project layout

```text
data/                       Processed Indoor Fire Dataset
voc_fire/                   Shared data, model, recovery, and metric code
train_base.py               Train the three-class base classifier
diagnose_recovery.py        Measure clean-only VOC reconstruction quality
train_recovery.py           Train and evaluate the VOC recovery graph
ablate_recovery.py          Ablate detection signals and the sensor embedding
evaluate_noise.py           Stress-test base checkpoints with VOC faults
requirements.txt            Runtime dependencies
```

Generated checkpoints and reports go under `results/` and are ignored by Git.

## Dataset

`data/Indoor_Fire_Except_Sensor0011.csv` is the public Indoor Fire Dataset with
sensor node 0011 excluded.

- Pascal Vorwerk, *Indoor Fire Dataset with Distributed Multi-Sensor Nodes*,
  Mendeley Data, Version 1, 2023
- DOI: <https://doi.org/10.17632/npk2zcm85h.1>
- License: CC BY 4.0

## Setup

Run the following commands from this directory:

```powershell
python -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
```

## Reproduce the pipeline

1. Train the frozen base classifier.

```powershell
& ".\.venv\Scripts\python.exe" ".\train_base.py" `
  --epochs 10 --runs 3 --batch-size 20 --device cuda `
  --output ".\results\base_mlp_3seeds"
```

2. Optionally verify that VOC is recoverable from the remaining sensors.

```powershell
& ".\.venv\Scripts\python.exe" ".\diagnose_recovery.py" `
  --device cuda --output ".\results\voc_recovery_diagnostic"
```

3. Train and evaluate the clean-only VOC recovery graph.

```powershell
& ".\.venv\Scripts\python.exe" ".\train_recovery.py" `
  --base-checkpoints ".\results\base_mlp_3seeds" `
  --noise-level 0.20 --device cuda `
  --output ".\results\voc_recovery_3seeds"
```

4. To evaluate a frozen base model without recovery, inject a test-only fault.

```powershell
& ".\.venv\Scripts\python.exe" ".\evaluate_noise.py" `
  --checkpoints ".\results\base_mlp_3seeds" `
  --fault gaussian --noise-level 0.20 --device cuda
```

The base model and recovery module train only on clean data. Synthetic faults
are applied to a copy of the test split and never enter training.
