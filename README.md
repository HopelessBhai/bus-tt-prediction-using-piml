# Bus Travel Time Prediction

Physics-informed hybrid models for urban bus travel time prediction on 100m road sections.

## Models

| Model | Type | Description |
|-------|------|-------------|
| **ANN** | Tabular NN | Feedforward network with BatchNorm + Dropout |
| **PINN** | Tabular NN | Physics-Informed NN with Aw-Rascle PDE residual loss |
| **LSTM** | Sequential | LSTM + context features, trained with MSE only |
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
│   ├── losses/               # PhysicsLoss (Aw-Rascle PDE), FocalLoss
│   ├── train/                # Training loops (PyTorch + XGBoost) + model/loss registry
│   ├── tune/                 # Optuna search spaces + tuning drivers
│   ├── eval/                 # Metrics, multi-model comparison, latency benchmarking
│   └── utils/                # Seed, paths, logging
├── scripts/                  # CLI entry points: train, tune, evaluate, latency_check
├── configs/                  # YAML configs for training and tuning
│   ├── train/                # One YAML per model (ann, pinn, lstm, phylstm, xgb)
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
  type: phylstm          # ann | pinn | lstm | phylstm | xgb
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

The physics-informed loss is based on the Aw-Rascle second-order traffic flow formulation:

```
∂tρ + ∂x(ρv) = 0
∂t(ρw) + ∂x(ρvw) = 0,  where w = v + p(ρ)
```

In this implementation, the regularization uses a speed-based residual surrogate:

```
L = L_data + λ · mean(R²)
```

where `R = dv/dt + dF(v)/dx`.

### Reference

- A. Aw and M. Rascle, *Resurrection of "Second Order" Models of Traffic Flow*, SIAM Journal on Applied Mathematics, 60(3), 916-938 (2000). DOI: [10.1137/S0036139997332099](https://doi.org/10.1137/S0036139997332099)
