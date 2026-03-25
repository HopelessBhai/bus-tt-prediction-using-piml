"""Default Optuna search spaces for each model."""

import optuna


def phylstm_space(trial: optuna.Trial) -> dict:
    return {
        "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 128]),
        "dropout": trial.suggest_float("dropout", 0.1, 0.4, step=0.05),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "phy_lambda": trial.suggest_float("phy_lambda", 0.01, 0.5, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
    }


def lstm_space(trial: optuna.Trial) -> dict:
    return {
        "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 128]),
        "dropout": trial.suggest_float("dropout", 0.1, 0.4, step=0.05),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
    }


def ann_space(trial: optuna.Trial) -> dict:
    n_layers = trial.suggest_int("n_layers", 1, 3)
    dims = [trial.suggest_categorical(f"dim_{i}", [64, 128, 256]) for i in range(n_layers)]
    return {
        "hidden_dims": dims,
        "dropout": trial.suggest_float("dropout", 0.05, 0.4, step=0.05),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
    }


def pinn_space(trial: optuna.Trial) -> dict:
    n_layers = trial.suggest_int("n_layers", 2, 4)
    dims = [trial.suggest_categorical(f"dim_{i}", [64, 128, 256]) for i in range(n_layers)]
    return {
        "layer_dims": dims,
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "phy_lambda": trial.suggest_float("phy_lambda", 0.01, 1.0, log=True),
    }


def xgb_space(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=100),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0, step=0.05),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0, step=0.05),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


SPACE_REGISTRY = {
    "phylstm": phylstm_space,
    "lstm": lstm_space,
    "ann": ann_space,
    "pinn": pinn_space,
    "xgb": xgb_space,
}
