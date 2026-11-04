from __future__ import annotations

import argparse
from pathlib import Path
import json
from typing import Dict

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from .generator import Config, Generator


N_SEEDS = 10
CATBOOST_PARAMS = {
    "iterations": 500,
    "depth": 6,
    "learning_rate": 0.1,
}


# Map class-like names to registry keys used by Generator
DYN_NAME_MAP: Dict[str, str] = {
    "RecencyGapDep": "recency_gap",
    "RollingVolatilityDep": "rolling_volatility",
    "RollingCorrDep": "rolling_corr",
    "RollingTrendDep": "rolling_trend",
    "RollingMeanDep": "rolling_mean",
    "RollingMinMaxDep": "rolling_minmax",
    "StreakDep": "streak",
}


def _write_dataset(out_dir: Path, cfg: Config, ds_obs, ds_opt, meta: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    ds_obs.X_train.to_csv(out_dir / "observables_train.csv", index=False)
    ds_obs.X_test.to_csv(out_dir / "observables_test.csv", index=False)
    ds_opt.X_train.to_csv(out_dir / "optimal_train.csv", index=False)
    ds_opt.X_test.to_csv(out_dir / "optimal_test.csv", index=False)
    pd.Series(ds_obs.y_train, name="y").to_csv(out_dir / "y_train.csv", index=False)
    pd.Series(ds_obs.y_test, name="y").to_csv(out_dir / "y_test.csv", index=False)
    with (out_dir / "config_used.json").open("w") as f:
        json.dump(meta, f, indent=2)


def _cat_features_indices(df: pd.DataFrame) -> list[int]:
    cats: list[int] = []
    for i, c in enumerate(df.columns):
        if not is_numeric_dtype(df[c]):
            cats.append(i)
    return cats


def _time_cv_splits(n: int, n_splits: int = 3) -> list[tuple[np.ndarray, np.ndarray]]:
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    cut = [int(n * (i + 1) / (n_splits + 1)) for i in range(n_splits)] + [n]
    for i in range(n_splits):
        tr_end = cut[i]
        va_end = cut[i + 1]
        if tr_end < 100 or (va_end - tr_end) < 100:
            continue
        tr = np.arange(0, tr_end)
        va = np.arange(tr_end, va_end)
        splits.append((tr, va))
    return splits


def _train_catboost_classification(Xtr: pd.DataFrame, ytr, Xte: pd.DataFrame, yte, seed: int) -> float:
    from catboost import CatBoostClassifier, Pool
    for df in (Xtr, Xte):
        for c in df.columns:
            if not is_numeric_dtype(df[c]):
                df[c] = df[c].astype("category")
    cats = _cat_features_indices(Xtr)
    model = CatBoostClassifier(
        iterations=CATBOOST_PARAMS["iterations"],
        depth=CATBOOST_PARAMS["depth"],
        learning_rate=CATBOOST_PARAMS["learning_rate"],
        loss_function="Logloss",
        verbose=False,
        random_state=seed,
    )
    model.fit(Pool(Xtr, ytr, cat_features=cats))
    pred = model.predict(Xte)
    return float((pred == yte).mean())


def _train_catboost_regression(Xtr: pd.DataFrame, ytr, Xte: pd.DataFrame, yte, seed: int) -> tuple[float, float]:
    from catboost import CatBoostRegressor, Pool
    from sklearn.metrics import r2_score
    for df in (Xtr, Xte):
        for c in df.columns:
            if not is_numeric_dtype(df[c]):
                df[c] = df[c].astype("category")
    cats = _cat_features_indices(Xtr)
    final = CatBoostRegressor(
        iterations=CATBOOST_PARAMS["iterations"],
        depth=CATBOOST_PARAMS["depth"],
        learning_rate=CATBOOST_PARAMS["learning_rate"],
        loss_function="RMSE",
        verbose=False,
        random_state=seed
    )
    final.fit(Pool(Xtr, ytr, cat_features=cats))
    pred = final.predict(Xte)
    rmse = float(((pred - yte) ** 2).mean() ** 0.5)
    r2 = float(r2_score(yte, pred))
    return rmse, r2


def main():
    ap = argparse.ArgumentParser("tabgen-generate-datasets-drifts")
    ap.add_argument(
        "--output-root", type=Path, default=Path("./_datasets_tabgen_drifts"), help="Output root directory"
    )
    ap.add_argument("--task", choices=["classification", "regression"], default="classification")
    args = ap.parse_args()

    dyn_list = [
        "RecencyGapDep",
        "RollingVolatilityDep",
        "RollingCorrDep",
        "RollingTrendDep",
        "RollingMeanDep",
        "RollingMinMaxDep",
        "StreakDep",
    ]

    # # Drift configurations to explore
    # drift_sets = [
    #     [],
    #     ["seasonality"],
    #     ["piecewise_drift"],
    #     ["seasonality", "piecewise_drift"],
    # ]

    for n_dyn in range(1, 4):
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(seed)
            # sample c_dyn, c_noise
            c_dyn = float(rng.uniform(0.3, 0.7))
            c_noise = float(rng.uniform(0.0, max(0.0, 1.0 - c_dyn)))
            # sample n_num, n_cat in [0,5]
            n_num = int(rng.integers(0, 6))
            n_cat = int(rng.integers(0, 6))
            # sample N in [20000, 100000]
            N = int(rng.integers(20000, 100001))
            
            # Sample a valid (entities_dim, dependency) pair.
            entities_dim = int(rng.integers(0, 3))
            dyns = [
                dyn_list[rng.integers(0, len(dyn_list))] for _ in range(n_dyn)
            ]
            dyn_keys = [DYN_NAME_MAP[dyn] for dyn in dyns]

            # Choose dependencies based on seed
            dependencies = dyn_keys

            # Entity cardinality policy
            if entities_dim == 1:
                entity_card = 25
            elif entities_dim == 2:
                entity_card = 5
            else:
                entity_card = 0

            seq_dyn = dependencies
            cfg = Config(
                n_samples=N,
                task=args.task,
                n_num=n_num,
                n_cat=n_cat,
                cat_cardinality=10,
                sequence_dynamics=seq_dyn,
                n_entities=max(1, entity_card) if entities_dim > 0 else 1,
                entity_tuple_dims=entities_dim,
                c_dyn=c_dyn,
                c_noise=c_noise,
                temperature=1.0,
                test_size=0.2,
                random_state=seed,
            )
            gen = Generator(cfg)
            ds_obs, ds_opt, rev_map = gen.generate()

            drift_name = "none"
            dep_dir = ("none" if not dependencies else '+'.join(dyn_keys))
            run_dir = (
                args.output_root
                / f"seed_{seed}"
                / f"N_{N}"
                / f"dep_{dep_dir}"
                / f"dim_{entities_dim}"
                / f"drift_{drift_name}"
            )
            meta = {
                "seed": seed,
                "dynamic": dyns,
                "dynamic_key": dyn_keys,
                "dependencies_used": dependencies,
                "drift": [],
                "entities_dim": entities_dim,
                "entity_cardinality": entity_card,
                "n_num": n_num,
                "n_cat": n_cat,
                "n_samples": N,
                "c_dyn": c_dyn,
                "c_noise": c_noise,
                "name_map": rev_map,
                "task": args.task,
            }
            _write_dataset(run_dir, cfg, ds_obs, ds_opt, meta)

            # Train CatBoost metrics
            metrics: dict = {"task": args.task}
            try:
                if args.task == "classification":
                    acc_obs = _train_catboost_classification(
                        ds_obs.X_train.copy(), ds_obs.y_train.values,
                        ds_obs.X_test.copy(), ds_obs.y_test.values,
                        seed,
                    )
                    acc_opt = _train_catboost_classification(
                        ds_opt.X_train.copy(), ds_opt.y_train.values,
                        ds_opt.X_test.copy(), ds_opt.y_test.values,
                        seed,
                    )
                    metrics.update({"acc_observables": acc_obs, "acc_optimal": acc_opt})
                else:
                    rmse_obs, r2_obs = _train_catboost_regression(
                        ds_obs.X_train.copy(), ds_obs.y_train.values,
                        ds_obs.X_test.copy(), ds_obs.y_test.values,
                        seed,
                    )
                    rmse_opt, r2_opt = _train_catboost_regression(
                        ds_opt.X_train.copy(), ds_opt.y_train.values,
                        ds_opt.X_test.copy(), ds_opt.y_test.values,
                        seed,
                    )
                    metrics.update({
                        "rmse_observables": rmse_obs,
                        "rmse_optimal": rmse_opt,
                        "r2_observables": r2_obs,
                        "r2_optimal": r2_opt,
                    })
            except Exception as e:
                metrics["error"] = f"{type(e).__name__}: {e}"
            with (run_dir / "metrics.json").open("w") as f:
                json.dump(metrics, f, indent=2)

            print(
                f"seed={seed} dep={dep_dir} dim={entities_dim} drift={drift_name} N={N} n_num={n_num} n_cat={n_cat} → {run_dir}"
            )


if __name__ == "__main__":
    main()
