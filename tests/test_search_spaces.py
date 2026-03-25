"""Tests for bus_tt.tune.search_spaces."""

import optuna

from bus_tt.tune.search_spaces import (
    phylstm_space,
    lstm_space,
    ann_space,
    pinn_space,
    xgb_space,
    SPACE_REGISTRY,
)


def _trial():
    study = optuna.create_study()
    return study.ask()


class TestSpaceRegistry:
    def test_all_keys(self):
        assert set(SPACE_REGISTRY.keys()) == {"phylstm", "lstm", "ann", "pinn", "xgb"}


class TestPhylstmSpace:
    def test_returns_dict(self):
        hp = phylstm_space(_trial())
        assert isinstance(hp, dict)

    def test_required_keys(self):
        hp = phylstm_space(_trial())
        for k in ("hidden_dim", "dropout", "lr", "phy_lambda", "batch_size", "weight_decay"):
            assert k in hp

    def test_hidden_dim_values(self):
        hp = phylstm_space(_trial())
        assert hp["hidden_dim"] in [32, 64, 128]


class TestLstmSpace:
    def test_returns_dict(self):
        hp = lstm_space(_trial())
        assert isinstance(hp, dict)

    def test_required_keys(self):
        hp = lstm_space(_trial())
        for k in ("hidden_dim", "dropout", "lr", "batch_size", "weight_decay"):
            assert k in hp


class TestAnnSpace:
    def test_returns_dict(self):
        hp = ann_space(_trial())
        assert isinstance(hp, dict)

    def test_has_hidden_dims(self):
        hp = ann_space(_trial())
        assert "hidden_dims" in hp
        assert isinstance(hp["hidden_dims"], list)
        assert len(hp["hidden_dims"]) >= 1

    def test_has_lr(self):
        hp = ann_space(_trial())
        assert "lr" in hp
        assert hp["lr"] > 0


class TestPinnSpace:
    def test_returns_dict(self):
        hp = pinn_space(_trial())
        assert isinstance(hp, dict)

    def test_has_layer_dims(self):
        hp = pinn_space(_trial())
        assert "layer_dims" in hp
        assert len(hp["layer_dims"]) >= 2


class TestXgbSpace:
    def test_returns_dict(self):
        hp = xgb_space(_trial())
        assert isinstance(hp, dict)

    def test_required_keys(self):
        hp = xgb_space(_trial())
        for k in ("n_estimators", "max_depth", "learning_rate", "subsample",
                   "colsample_bytree", "reg_alpha", "reg_lambda"):
            assert k in hp
