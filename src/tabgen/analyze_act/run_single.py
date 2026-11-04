from __future__ import annotations

import argparse
import json
import logging
import random
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score

from .main import (
    AnalyzeActPredictor,
    CatBoostClassifierPipeline,
    CatBoostRegressorPipeline,
    OpenAILLMClient,
)


LOGGER = logging.getLogger(__name__)
OUTPUT_FILENAMES = (
    "X_train_fe.csv",
    "X_test_fe.csv",
    "predictions.csv",
    "run_metadata.json",
)


def _load_dataset(dataset_dir: Path) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    paths = {
        "X_train": dataset_dir / "observables_train.csv",
        "y_train": dataset_dir / "y_train.csv",
        "X_test": dataset_dir / "observables_test.csv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required dataset file(s): " + ", ".join(missing))

    X_train = pd.read_csv(paths["X_train"])
    X_test = pd.read_csv(paths["X_test"])
    y_frame = pd.read_csv(paths["y_train"])
    if y_frame.shape[1] != 1:
        raise ValueError(f"Expected one target column in {paths['y_train']}, found {y_frame.shape[1]}.")
    if len(X_train) != len(y_frame):
        raise ValueError(f"Training features and targets have different row counts: {len(X_train)} != {len(y_frame)}.")
    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("observables_train.csv and observables_test.csv must have identical columns in the same order.")
    return X_train, y_frame.iloc[:, 0], X_test


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [output_dir / name for name in OUTPUT_FILENAMES if (output_dir / name).exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Output file(s) already exist: {joined}. Pass --overwrite to replace them.")


def run_dataset(
    dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    model: str = "gpt-5-mini",
    task: str = "classification",
    description: str = "Synthetic supervised-learning task with chronologically ordered rows.",
    num_iterations: int = 5,
    max_code_attempts: int = 5,
    max_output_tokens: int = 2048,
    reasoning_effort: str | None = "low",
    random_seed: int = 0,
    overwrite: bool = False,
    llm=None,
    pipeline=None,
) -> dict:
    dataset_path = Path(dataset_dir).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    _prepare_output_dir(output_path, overwrite=overwrite)
    X_train, y_train, X_test = _load_dataset(dataset_path)

    random.seed(random_seed)
    np.random.seed(random_seed)

    response_kwargs = {"max_output_tokens": int(max_output_tokens)}
    if reasoning_effort is not None:
        response_kwargs["reasoning"] = {"effort": reasoning_effort}
    active_llm = llm or OpenAILLMClient(model=model, **response_kwargs)

    if task == "classification":
        active_pipeline = pipeline or CatBoostClassifierPipeline(random_seed=random_seed)
        score_func_name = "accuracy"
        score_func = accuracy_score
    elif task == "regression":
        active_pipeline = pipeline or CatBoostRegressorPipeline(random_seed=random_seed)
        score_func_name = "r2"
        score_func = r2_score
    else:
        raise ValueError(f"Unsupported task: {task!r}")

    started_at = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="analyze_act_single_") as working_dir:
        predictor = AnalyzeActPredictor(
            llm=active_llm,
            pipeline=active_pipeline,
            num_iterations=num_iterations,
            working_dir=working_dir,
            score_func_name=score_func_name,
            score_func=score_func,
            max_code_attempts=max_code_attempts,
        )
        predictions = predictor.fit_predict(X_train, y_train, X_test, description=description)

    elapsed_seconds = time.monotonic() - started_at
    predictor.X_train_fe_.to_csv(output_path / "X_train_fe.csv", index=False)
    predictor.X_test_fe_.to_csv(output_path / "X_test_fe.csv", index=False)
    pd.Series(np.asarray(predictions).reshape(-1), name="prediction").to_csv(
        output_path / "predictions.csv", index=False
    )

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(dataset_path),
        "output_dir": str(output_path),
        "backend": "openai_responses" if isinstance(active_llm, OpenAILLMClient) else type(active_llm).__name__,
        "model": getattr(active_llm, "model", model),
        "task": task,
        "description": description,
        "num_iterations": int(num_iterations),
        "max_code_attempts": int(max_code_attempts),
        "max_output_tokens_per_request": int(max_output_tokens),
        "reasoning_effort": reasoning_effort,
        "random_seed": int(random_seed),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "input_feature_count": X_train.shape[1],
        "selected_feature_count": predictor.X_train_fe_.shape[1],
        "best_validation_score": float(predictor.best_score_),
        "best_fold_scores": [float(score) for score in predictor.best_fold_scores_],
        "llm_request_count": getattr(active_llm, "request_count", None),
        "llm_token_usage": getattr(active_llm, "usage_totals", None),
        "outputs": {name: str(output_path / name) for name in OUTPUT_FILENAMES[:-1]},
    }
    (output_path / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    LOGGER.info("Analyze&Act outputs written to %s in %.2f seconds", output_path, elapsed_seconds)
    return metadata


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tabgen-analyze-act",
        description="Run Analyze&Act on one generated TabGen dataset and write selected features and predictions.",
    )
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Directory containing observables_train.csv, observables_test.csv, and y_train.csv.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (default: <dataset-dir>/analyze_act_output).")
    parser.add_argument("--model", default="gpt-5-mini", help="OpenAI model identifier.")
    parser.add_argument("--task", choices=("classification", "regression"), default="classification")
    parser.add_argument("--description", default="Synthetic supervised-learning task with chronologically ordered rows.")
    parser.add_argument("--num-iterations", type=int, default=5)
    parser.add_argument("--max-code-attempts", type=int, default=5)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--reasoning-effort", choices=("none", "minimal", "low", "medium", "high"), default="low")
    parser.add_argument("--random-seed", type=int, default=0, help="Seed for NumPy and the downstream CatBoost model; hosted LLM calls may remain nondeterministic.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing runner outputs in the output directory.")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    output_dir = args.output_dir or (args.dataset_dir / "analyze_act_output")
    try:
        metadata = run_dataset(
            args.dataset_dir,
            output_dir,
            model=args.model,
            task=args.task,
            description=args.description,
            num_iterations=args.num_iterations,
            max_code_attempts=args.max_code_attempts,
            max_output_tokens=args.max_output_tokens,
            reasoning_effort=args.reasoning_effort,
            random_seed=args.random_seed,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
