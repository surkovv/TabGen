from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from .generator import Config, Generator


CATBOOST_PARAMS = {
    "iterations": 500,
    "depth": 6,
    "learning_rate": 0.1,
}

DYN_NAME_MAP: Dict[str, str] = {
    "RecencyGapDep": "recency_gap",
    "RollingVolatilityDep": "rolling_volatility",
    "RollingCorrDep": "rolling_corr",
    "RollingTrendDep": "rolling_trend",
    "RollingMeanDep": "rolling_mean",
    "RollingMinMaxDep": "rolling_minmax",
    "StreakDep": "streak",
}

NO_DRIFT_DYNS = list(DYN_NAME_MAP.values())
DRIFT_SETS = [
    [],
    ["seasonality"],
    ["piecewise_drift"],
    ["seasonality", "piecewise_drift"],
]


def _write_dataset(out_dir: Path, ds_obs, ds_opt, meta: dict) -> None:
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


def _entity_cardinality_for_dim(entities_dim: int) -> int:
    if entities_dim == 1:
        return 25
    if entities_dim == 2:
        return 5
    return 0


def _sample_common_params(rng: np.random.Generator, sample_seed: int, task: str) -> dict:
    c_dyn = float(rng.uniform(0.3, 0.7))
    c_noise = float(rng.uniform(0.0, max(0.0, 1.0 - c_dyn)))
    n_num = int(rng.integers(0, 6))
    n_cat = int(rng.integers(0, 6))
    while n_num + n_cat == 0:
        n_num = int(rng.integers(0, 6))
        n_cat = int(rng.integers(0, 6))
    return {
        "n_samples": int(rng.integers(20_000, 100_001)),
        "task": task,
        "n_num": n_num,
        "n_cat": n_cat,
        "cat_cardinality": 10,
        "c_dyn": c_dyn,
        "c_noise": c_noise,
        "temperature": 1.0,
        "test_size": 0.2,
        "random_state": sample_seed,
    }


def _build_no_drift_config(rng: np.random.Generator, sample_seed: int, task: str) -> tuple[Config, dict]:
    base = _sample_common_params(rng, sample_seed, task)
    entities_dim = int(rng.integers(0, 3))
    entity_cardinality = _entity_cardinality_for_dim(entities_dim)
    dyn_key = str(rng.choice(NO_DRIFT_DYNS))
    cfg = Config(
        **base,
        sequence_dynamics=[dyn_key],
        n_entities=max(1, entity_cardinality) if entities_dim > 0 else 1,
        entity_tuple_dims=entities_dim,
    )
    meta = {
        "suite": "no_drift",
        "dynamic": dyn_key,
        "dynamic_key": dyn_key,
        "dependencies_used": [dyn_key],
        "drift": [],
        "entities_dim": entities_dim,
        "entity_cardinality": entity_cardinality,
        "n_num": base["n_num"],
        "n_cat": base["n_cat"],
        "n_samples": base["n_samples"],
        "c_dyn": base["c_dyn"],
        "c_noise": base["c_noise"],
        "task": task,
        "sample_seed": sample_seed,
        "generator_config": asdict(cfg),
    }
    return cfg, meta


def _build_drift_config(rng: np.random.Generator, sample_seed: int, task: str) -> tuple[Config, dict]:
    base = _sample_common_params(rng, sample_seed, task)
    entities_dim = int(rng.integers(0, 3))
    entity_cardinality = _entity_cardinality_for_dim(entities_dim)
    dependency_options = ["no"] + NO_DRIFT_DYNS
    dep_choice = str(rng.choice(dependency_options))
    dependencies = [] if dep_choice == "no" else [dep_choice]
    drift = list(DRIFT_SETS[int(rng.integers(0, len(DRIFT_SETS)))])
    cfg = Config(
        **base,
        sequence_dynamics=dependencies + drift,
        n_entities=max(1, entity_cardinality) if entities_dim > 0 else 1,
        entity_tuple_dims=entities_dim,
    )
    meta = {
        "suite": "drift",
        "dynamic": dep_choice,
        "dynamic_key": dep_choice,
        "dependencies_used": dependencies,
        "drift": drift,
        "entities_dim": entities_dim,
        "entity_cardinality": entity_cardinality,
        "n_num": base["n_num"],
        "n_cat": base["n_cat"],
        "n_samples": base["n_samples"],
        "c_dyn": base["c_dyn"],
        "c_noise": base["c_noise"],
        "task": task,
        "sample_seed": sample_seed,
        "generator_config": asdict(cfg),
    }
    return cfg, meta


def _run_sample(out_dir: Path, cfg: Config, meta: dict) -> dict:
    gen = Generator(cfg)
    ds_obs, ds_opt, rev_map = gen.generate()
    meta = dict(meta)
    meta["name_map"] = rev_map
    _write_dataset(out_dir, ds_obs, ds_opt, meta)

    metrics: dict = {"task": meta["task"]}
    try:
        if meta["task"] == "classification":
            metrics["acc_observables"] = _train_catboost_classification(
                ds_obs.X_train.copy(),
                ds_obs.y_train.values,
                ds_obs.X_test.copy(),
                ds_obs.y_test.values,
                cfg.random_state,
            )
            metrics["acc_optimal"] = _train_catboost_classification(
                ds_opt.X_train.copy(),
                ds_opt.y_train.values,
                ds_opt.X_test.copy(),
                ds_opt.y_test.values,
                cfg.random_state,
            )
        else:
            metrics["note"] = "Regression metrics are not implemented in this sampler."
    except Exception as exc:
        metrics["error"] = f"{type(exc).__name__}: {exc}"

    with (out_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    return meta


def _sample_suite(suite: str, output_root: Path, count: int, task: str, random_seed: int) -> list[dict]:
    suite_root = output_root / suite
    suite_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(random_seed)
    manifest: list[dict] = []

    for sample_idx in range(count):
        sample_seed = int(rng.integers(0, 1_000_000_000))
        sample_rng = np.random.default_rng(sample_seed)
        if suite == "no_drift":
            cfg, meta = _build_no_drift_config(sample_rng, sample_seed, task)
            dyn_slug = meta["dynamic_key"]
            drift_slug = "none"
        else:
            cfg, meta = _build_drift_config(sample_rng, sample_seed, task)
            dyn_slug = meta["dynamic_key"]
            drift_slug = "none" if not meta["drift"] else "+".join(meta["drift"])

        sample_dir = suite_root / f"sample_{sample_idx:02d}__dep_{dyn_slug}__dim_{meta['entities_dim']}__drift_{drift_slug}"
        saved_meta = _run_sample(sample_dir, cfg, meta)
        manifest.append({
            "sample_index": sample_idx,
            "path": str(sample_dir),
            "config": saved_meta,
        })
        print(f"{suite} sample={sample_idx} seed={sample_seed} -> {sample_dir}")

    with (suite_root / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main():
    ap = argparse.ArgumentParser("tabgen-generate-random-samples")
    ap.add_argument(
        "--output-root",
        type=Path,
        default=Path("./out/random_samples"),
        help="Output root inside the repo",
    )
    ap.add_argument("--count", type=int, default=3, help="Number of random datasets to generate per suite")
    ap.add_argument("--task", choices=["classification", "regression"], default="classification")
    ap.add_argument(
        "--suite",
        choices=["no_drift", "drift", "both"],
        default="both",
        help="Which suite to sample from",
    )
    ap.add_argument("--random-seed", type=int, default=0, help="Seed controlling the random sample selection")
    args = ap.parse_args()

    suites = ["no_drift", "drift"] if args.suite == "both" else [args.suite]
    summary = {}
    for offset, suite in enumerate(suites):
        summary[suite] = _sample_suite(
            suite=suite,
            output_root=args.output_root,
            count=args.count,
            task=args.task,
            random_seed=args.random_seed + offset,
        )

    with (args.output_root / "manifest.json").open("w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
