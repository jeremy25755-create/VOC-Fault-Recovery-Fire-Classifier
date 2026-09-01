# VOC Fault-Recovery Fire Classifier

A fault-tolerant indoor-fire classification pipeline that trains only on clean
data. A graph recovery module estimates the VOC signal from the remaining
sensors, detects inconsistent VOC behavior, and selectively combines the
frozen MLP classifier with a compact fault-mode adapter.

## Repository layout

- `RIFTG_Without_Graphs/`: MLP-based three-class base classifier
- `VOC_Recovery_R2_Diagnostic/`: clean-only VOC recoverability analysis
- `NoGraph_VOC_Recovery_Adapter/`: recovery graph and fault-mode adapter
- `Dataset/`: processed dataset used by the experiments

## Dataset

`Dataset/Indoor_Fire_Except_Sensor0011.csv` is derived from the public Indoor
Fire Dataset with sensor node 0011 excluded.

- Pascal Vorwerk, *Indoor Fire Dataset with Distributed Multi-Sensor Nodes*,
  Mendeley Data, Version 1, 2023.
- DOI: https://doi.org/10.17632/npk2zcm85h.1
- License: CC BY 4.0

## Setup

```powershell
python -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
```

## 1. Train the frozen MLP base classifier

```powershell
& ".\.venv\Scripts\python.exe" ".\RIFTG_Without_Graphs\train.py" `
  --data ".\Dataset\Indoor_Fire_Except_Sensor0011.csv" `
  --epochs 10 --runs 3 --batch-size 20 --device cuda `
  --output ".\results\base_mlp_3seeds"
```

## 2. Check VOC recoverability

```powershell
& ".\.venv\Scripts\python.exe" ".\VOC_Recovery_R2_Diagnostic\train.py" `
  --data ".\Dataset\Indoor_Fire_Except_Sensor0011.csv" `
  --device cuda --output ".\results\voc_recovery_diagnostic"
```

## 3. Train and evaluate the recovery adapter

```powershell
& ".\.venv\Scripts\python.exe" ".\NoGraph_VOC_Recovery_Adapter\train.py" `
  --base-checkpoints ".\results\base_mlp_3seeds" `
  --data ".\Dataset\Indoor_Fire_Except_Sensor0011.csv" `
  --noise-level 0.20 --device cuda `
  --output ".\results\voc_recovery_adapter_3seeds"
```

Noise is injected only into the test copy; the base classifier and recovery
module are trained on clean data. Generated checkpoints and results are not
versioned.

