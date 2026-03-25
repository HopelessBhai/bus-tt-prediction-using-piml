"""End-to-end integration test: load sample CSV -> features -> train -> evaluate."""

import numpy as np
import torch
from torch.utils.data import DataLoader

from bus_tt.data.io import load_raw
from bus_tt.data.features import add_time_features, get_section_cols, build_speed_df, build_samples
from bus_tt.data.split import train_test_mask
from bus_tt.data.datasets import TabularDataset, SeqCtxDataset
from bus_tt.models.ann import ANNModel
from bus_tt.models.lstm import PhyLSTMModel
from bus_tt.losses.physics import PhysicsLoss
from bus_tt.train.train_torch import train_tabular, train_seq
from bus_tt.train.train_xgb import train_xgb
from bus_tt.eval.metrics import compute_all
from bus_tt.constants import DROP_SECTIONS, SECTION_LENGTH_M


class TestE2EPipeline:
    def _load_and_prep(self):
        df = load_raw("data/sample/sample_bus_travel_times.csv")
        df = add_time_features(df)
        section_cols = get_section_cols(df, drop_prefix=DROP_SECTIONS)
        speed_df = build_speed_df(df, section_cols)
        samples = build_samples(df, section_cols, speed_df[section_cols].to_numpy())
        return df, section_cols, samples

    def test_data_pipeline(self):
        df, section_cols, samples = self._load_and_prep()
        assert len(df) == 10
        assert len(section_cols) == 5
        assert samples["X_xgb"].shape[0] > 0

    def test_ann_e2e(self):
        _, _, samples = self._load_and_prep()
        n = samples["X_xgb"].shape[0]
        split = max(1, n // 2)
        X_tr, X_val = samples["X_xgb"][:split], samples["X_xgb"][split:]
        y_tr, y_val = samples["y_speed"][:split], samples["y_speed"][split:]

        tl = DataLoader(TabularDataset(X_tr, y_tr), batch_size=8)
        vl = DataLoader(TabularDataset(X_val, y_val), batch_size=8)

        model = ANNModel(input_dim=6, hidden_dims=[16])
        result = train_tabular(
            model, tl, vl, torch.nn.MSELoss(),
            lr=0.01, max_epochs=5, patience=3, device="cpu",
        )
        assert result["best_val"] < float("inf")

        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(X_val, dtype=torch.float32)).numpy().flatten()
        y_true_tt = SECTION_LENGTH_M / np.clip(y_val, 1e-3, None)
        y_pred_tt = SECTION_LENGTH_M / np.clip(preds, 1e-3, None)
        metrics = compute_all(y_true_tt, y_pred_tt)
        assert "MAE" in metrics

    def test_xgb_e2e(self):
        _, _, samples = self._load_and_prep()
        n = samples["X_xgb"].shape[0]
        split = max(1, n // 2)
        model = train_xgb(
            samples["X_xgb"][:split], samples["y_speed"][:split],
            samples["X_xgb"][split:], samples["y_speed"][split:],
            params={"n_estimators": 10, "max_depth": 3},
        )
        preds = model.predict(samples["X_xgb"][split:])
        assert preds.shape[0] > 0

    def test_phylstm_e2e(self):
        _, _, samples = self._load_and_prep()
        n = samples["X_seq"].shape[0]
        split = max(1, n // 2)

        tl = DataLoader(
            SeqCtxDataset(samples["X_seq"][:split], samples["X_ctx"][:split], samples["y_speed"][:split]),
            batch_size=8,
        )
        vl = DataLoader(
            SeqCtxDataset(samples["X_seq"][split:], samples["X_ctx"][split:], samples["y_speed"][split:]),
            batch_size=8,
        )

        model = PhyLSTMModel(hidden_dim=16, dropout=0.1)
        criterion = PhysicsLoss(phy_lambda=0.1)
        result = train_seq(
            model, tl, vl, criterion,
            lr=0.01, max_epochs=3, patience=3, device="cpu",
        )
        assert result["best_val"] < float("inf")
