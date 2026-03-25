# Bus Travel Time Prediction

Physics-informed hybrid models for urban bus travel time prediction on 100m road sections.

## Models

| Model | Type | Description |
|-------|------|-------------|
| **ANN** | Tabular NN | Feedforward network with BatchNorm + Dropout |
| **PINN** | Tabular NN | Physics-Informed NN with LWR PDE residual loss |
| **Phy-LSTM** | Sequential | LSTM + context features + physics loss |
| **XGBoost** | Gradient boosting | Tabular features, fast inference |
| **Hybrid** | Adaptive routing | Routes to XGBoost (congested) or Phy-LSTM (normal) based on previous trip travel time |

## Repository Structure

```
bus-tt-prediction/
├── src/bus_tt/
│   ├── constants.py          # Shared constants (section length, test dates, etc.)
│   ├── data/                 # Data loading, splitting, feature engineering, datasets
│   ├── models/               # ANN, PINN, LSTM/PhyLSTM, XGBoost, Hybrid router
│   ├── losses/               # PhysicsLoss (LWR PDE), FocalLoss
│   ├── train/                # Training loops (PyTorch + XGBoost) + model/loss registry
│   ├── tune/                 # Optuna search spaces + tuning drivers
│   ├── eval/                 # Metrics, multi-model comparison, latency benchmarking
│   └── utils/                # Seed, paths, logging
├── scripts/                  # CLI entry points: train, tune, evaluate, latency_check
├── configs/                  # YAML configs for training and tuning
│   ├── train/                # One YAML per model (ann, pinn, phylstm, xgb)
│   └── tune/                 # One YAML per model
├── data/sample/              # Small sample dataset for testing the pipeline
└── outputs/                  # Generated artifacts (models, predictions, reports)
```

## Quick Start

```bash
# Install
pip install -e .

# Train a model
python scripts/train.py --config configs/train/phylstm.yaml

# Tune hyperparameters
python scripts/tune.py --config configs/tune/phylstm.yaml

# Evaluate
python scripts/evaluate.py --config configs/train/phylstm.yaml --checkpoint outputs/models/phylstm.pth

# Latency benchmark
python scripts/latency_check.py --data data/sample/sample_bus_travel_times.csv
```

## Data

Place your dataset CSV at `data/sample/sample_bus_travel_times.csv`. The expected format has columns:
- `Date` — trip date
- `Start time of the trip` — departure time (HH:MM or HH:MM:SS)
- `Section 1` through `Section N` — travel time (seconds) per 100m section

A small synthetic sample is provided in `data/sample/` to verify the pipeline runs end-to-end without the full dataset.

## Configuration

All training and tuning is driven by YAML configs in `configs/`. Key fields:

```yaml
data_path: data/sample/sample_bus_travel_times.csv
model:
  type: phylstm          # ann | pinn | phylstm | xgb
  params:
    hidden_dim: 64
    dropout: 0.2
loss:
  type: physics
  phy_lambda: 0.14
training:
  lr: 0.005
  batch_size: 256
  max_epochs: 150
```

## Physics Loss

The physics-informed loss adds a PDE residual term from the LWR traffic flow model:

```
L = L_data + λ · mean(R²)
```

where `R = dv/dt + dF(v)/dx` is the Greenshields flux conservation residual.
