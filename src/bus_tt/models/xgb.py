import xgboost as xgb


def build_xgb_model(params: dict | None = None, seed: int = 42) -> xgb.XGBRegressor:
    defaults = dict(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.01,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
    )
    if params:
        defaults.update(params)
    return xgb.XGBRegressor(**defaults)
