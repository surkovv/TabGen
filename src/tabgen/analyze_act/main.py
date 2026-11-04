from __future__ import annotations

import logging
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from pandas.api.types import is_bool_dtype, is_integer_dtype, is_numeric_dtype
from sklearn.model_selection import TimeSeriesSplit


LOGGER = logging.getLogger(__name__)


INIT_MESSAGE = (
    "You are a data scientist helping with feature engineering for a supervised learning problem.\n"
    "Task description:\n{description}\n\n"
    "Here is X_train.head():\n{X_train_head}\n\n"
    "Here is y_train.head():\n{y_train_head}\n\n"
    "Our evaluation metric is: {score_func_name}.\n"
    "As a rule of thumb, test data chronologically follows the training data, but you should analyze the actual patterns by yourself."
    "Propose 5 concrete, safe, and deterministic feature engineering hypotheses that could improve the model.\n"
    "Be specific and avoid vague suggestions."
)

NEW_HYPOTHESES_MESSAGE = (
    "Previous iteration results:\n"
    "Old score: {old_score}\n"
    "New score: {new_score}\n"
    "{is_best_score_updated}\n\n"
    "Based on this, refine or propose new feature engineering hypotheses. Explain why they might help."
)

ANALYSIS_MESSAGE = (
    "Please write Python analysis code (enclosed in a single triple-backticked block) that:\n"
    "1) Reads ""X_train.pkl"" and ""y_train.pkl"" from the current directory using pickle\n"
    "2) Performs the analysis to verify the hypotheses."
    "The code will be executed, and I will return its stdout to you."
    "Note: keep the code < 150 lines."
)

IMPLEMENTATION_MESSAGE = (
    "Based on this analysis:\n{analysis}\n\n"
    "Say with hypotheses hold and which don't.\n"
    "THen, write Python code (in a single triple-backticked block) that:\n"
    "- Reads ""X_train.pkl"", ""y_train.pkl"", ""X_test.pkl"" from the current directory using pickle\n"
    "- Creates transformed pandas DataFrames named X_train_fe and X_test_fe with identical columns.\n"
    "- Saves them as 'X_train_fe.pkl' and 'X_test_fe.pkl' via pickle using the current working directory.\n"
    "- Code must be deterministic, side-effect free (except for writing the 2 pickle files), and must not rely on internet or external data.\n"
    "- Avoid using y in transformations in order to reduce chances of overfitting."
    "If needed, you may import common Python libraries (numpy, pandas, sklearn)."
    "Note: keep the code < 150 lines."
)

DIRECT_FE_INIT_MESSAGE = (
    "You are a data scientist helping with feature engineering for a supervised learning problem.\n"
    "Task description:\n{description}\n\n"
    "Here is X_train.head():\n{X_train_head}\n\n"
    "Here is y_train.head():\n{y_train_head}\n\n"
    "Our evaluation metric is: {score_func_name}.\n"
    "As a rule of thumb, test data chronologically follows the training data, but you should analyze the actual patterns by yourself.\n"
    "Write Python code (in a single triple-backticked block) that:\n"
    "- Reads \"X_train.pkl\", \"y_train.pkl\", \"X_test.pkl\" from the current directory using pickle\n"
    "- Creates transformed pandas DataFrames named X_train_fe and X_test_fe with identical columns.\n"
    "- Saves them as 'X_train_fe.pkl' and 'X_test_fe.pkl' via pickle using the current working directory.\n"
    "- Code must be deterministic, side-effect free (except for writing the 2 pickle files), and must not rely on internet or external data.\n"
    "- Avoid using y in transformations to reduce leakage risk.\n"
    "Note: keep the code < 150 lines."
)

DIRECT_FE_REVISE_MESSAGE = (
    "Previous iteration results:\n"
    "Old score: {old_score}\n"
    "New score: {new_score}\n"
    "Fold scores: {fold_scores}\n\n"
    "Based on this, refine the feature engineering. Return ONLY a single Python code block (triple-backticked) that again reads the pickles and writes X_train_fe.pkl and X_test_fe.pkl.\n"
    "Do not use y in transformations. Ensure determinism and identical columns for train/test. Keep code < 150 lines."
)

CODE_EXTRACTION_FAILED = (
    "I could not find a valid Python code block in your last response. "
    "Please return a single Python code block enclosed in triple backticks."
)

CODE_EXECUTION_FAILED = (
    "Your code failed to execute:\n{msg}\n\n"
    "Please correct the code and return a single Python code block only."
)


class LLMClient(Protocol):
    def generate(self, messages: list[dict[str, str]]) -> str:
        ...


class OpenAILLMClient:
    def __init__(
        self,
        model: str = "gpt-5-mini",
        *,
        client: Any | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        project: str | None = None,
        **response_kwargs: Any,
    ) -> None:
        self.model = model
        self.response_kwargs = dict(response_kwargs)
        self.request_count = 0
        self.usage_totals = {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_output_tokens": 0,
            "total_tokens": 0,
        }
        self.client = client or self._build_client(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            project=project,
        )

    def generate(self, messages: list[dict[str, str]]) -> str:
        started_at = time.monotonic()
        LOGGER.info(
            "OpenAI request started: model=%s messages=%d input_chars=%d kwargs=%s",
            self.model,
            len(messages),
            sum(len(message.get("content", "")) for message in messages),
            sorted(self.response_kwargs),
        )
        response = self.client.responses.create(
            model=self.model,
            input=self._to_responses_input(messages),
            **self.response_kwargs,
        )
        self.request_count += 1
        self._record_usage(self._response_get(response, "usage"))
        text = self._extract_output_text(response)
        elapsed = time.monotonic() - started_at
        LOGGER.info(
            "OpenAI request finished: model=%s elapsed=%.2fs output_chars=%d status=%r usage=%r",
            self.model,
            elapsed,
            len(text),
            self._response_get(response, "status"),
            self._response_get(response, "usage"),
        )
        return text

    def _record_usage(self, usage) -> None:
        if usage is None:
            return
        input_details = self._response_get(usage, "input_tokens_details")
        output_details = self._response_get(usage, "output_tokens_details")
        values = {
            "input_tokens": self._response_get(usage, "input_tokens"),
            "cached_input_tokens": self._response_get(input_details, "cached_tokens"),
            "output_tokens": self._response_get(usage, "output_tokens"),
            "reasoning_output_tokens": self._response_get(output_details, "reasoning_tokens"),
            "total_tokens": self._response_get(usage, "total_tokens"),
        }
        for key, value in values.items():
            if value is not None:
                self.usage_totals[key] += int(value)

    @staticmethod
    def _build_client(
        *,
        api_key: str | None,
        base_url: str | None,
        organization: str | None,
        project: str | None,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAILLMClient requires the OpenAI Python package. "
                "Install it with `pip install -e .[openai]` or `pip install openai`."
            ) from exc

        kwargs = {}
        if api_key is not None:
            kwargs["api_key"] = api_key
        if base_url is not None:
            kwargs["base_url"] = base_url
        if organization is not None:
            kwargs["organization"] = organization
        if project is not None:
            kwargs["project"] = project
        return OpenAI(**kwargs)

    @staticmethod
    def _to_responses_input(messages: list[dict[str, str]]) -> list[dict[str, str]]:
        return [{"role": message["role"], "content": message["content"]} for message in messages]

    @staticmethod
    def _extract_output_text(response) -> str:
        output_text = getattr(response, "output_text", None)
        if output_text:
            return str(output_text)
        if isinstance(response, dict):
            output_text = response.get("output_text")
            if output_text:
                return str(output_text)
            output = response.get("output", [])
        else:
            output = getattr(response, "output", [])
        text_parts = []
        for item in output or []:
            content = item.get("content", []) if isinstance(item, dict) else getattr(item, "content", [])
            for part in content or []:
                if isinstance(part, dict):
                    text = part.get("text")
                else:
                    text = getattr(part, "text", None)
                if text:
                    text_parts.append(str(text))
        if text_parts:
            return "".join(text_parts)
        status = OpenAILLMClient._response_get(response, "status")
        incomplete_details = OpenAILLMClient._response_get(response, "incomplete_details")
        incomplete_reason = OpenAILLMClient._response_get(incomplete_details, "reason")
        error = OpenAILLMClient._response_get(response, "error")
        usage = OpenAILLMClient._response_get(response, "usage")
        raise ValueError(
            "OpenAI response did not contain output text. "
            f"status={status!r}, incomplete_reason={incomplete_reason!r}, "
            f"error={error!r}, usage={usage!r}. "
            "If status is 'incomplete' with reason 'max_output_tokens', increase max_output_tokens."
        )

    @staticmethod
    def _response_get(value, key: str):
        if value is None:
            return None
        if isinstance(value, dict):
            return value.get(key)
        return getattr(value, key, None)


@dataclass
class _Message:
    role: str
    content: str


class MessageList:
    def __init__(self) -> None:
        self._messages: list[_Message] = []

    def append_user_message(self, content: str) -> None:
        self._messages.append(_Message(role="user", content=content))

    def append_agent_message(self, content: str) -> None:
        self._messages.append(_Message(role="assistant", content=content))

    def get_messages(self) -> list[dict[str, str]]:
        return [{"role": message.role, "content": message.content} for message in self._messages]


class RunnableEnv:
    def __init__(self, working_dir: str | Path):
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)

    def run_code(self, code: str, timeout: int = 180) -> tuple[int, str]:
        timeout = min(180, int(timeout))
        try:
            result = subprocess.run(
                [sys.executable, "-I", "-u", "-c", code],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                cwd=str(self.working_dir),
            )
        except subprocess.TimeoutExpired as exc:
            out = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout.decode() if exc.stdout else "")
            err = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr.decode() if exc.stderr else "")
            return 1, f"Execution timed out after {timeout}s.\nSTDOUT:\n{out}\nSTDERR:\n{err}"
        except Exception as exc:
            return 1, f"Failed to execute code: {exc}"

        if result.returncode == 0:
            return 0, result.stdout
        return (
            result.returncode,
            f"Execution failed (exit code {result.returncode}).\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
        )

    def open(self, filename: str, mode: str = "r"):
        full_path = self.working_dir / filename
        return open(full_path, mode)

    def clear_working_dir(self) -> None:
        for entry in os.scandir(self.working_dir):
            path = entry.path
            if entry.is_file() or entry.is_symlink():
                os.unlink(path)
            elif entry.is_dir():
                shutil.rmtree(path)


class _CatBoostBase:
    is_classifier = True
    default_params = {
        "iterations": 500,
        "depth": 6,
        "verbose": 0,
    }

    def __init__(self, **params):
        base = dict(self.default_params)
        base.update(params)
        self.params = base
        self.model = None
        self.cat_feature_indices: list[int] | None = None

    def _detect_cat_features(self, X: pd.DataFrame) -> list[int]:
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        return [index for index, column in enumerate(X_df.columns) if not is_numeric_dtype(X_df[column])]

    def fit(self, X, y) -> None:
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        y_ser = y if isinstance(y, pd.Series) else pd.Series(y)
        self.cat_feature_indices = self._detect_cat_features(X_df)
        train_pool = Pool(X_df, y_ser, cat_features=self.cat_feature_indices or None)
        model_cls = CatBoostClassifier if self.is_classifier else CatBoostRegressor
        self.model = model_cls(**self.params)
        self.model.fit(train_pool)

    def predict(self, X):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        test_pool = Pool(X_df, cat_features=self.cat_feature_indices or None)
        predictions = self.model.predict(test_pool)
        predictions = np.asarray(predictions)
        if predictions.ndim > 1 and predictions.shape[1] == 1:
            predictions = predictions.ravel()
        return predictions


class CatBoostClassifierPipeline(_CatBoostBase):
    is_classifier = True


class CatBoostRegressorPipeline(_CatBoostBase):
    is_classifier = False


class AnalyzeActFlow:
    def __init__(
        self,
        llm: LLMClient,
        num_iterations: int,
        pipeline,
        X_train,
        y_train,
        X_test,
        description: str,
        score_func_name: str,
        score_func: Callable,
        working_dir: str | Path,
        max_code_attempts: int = 5,
    ):
        self.llm = llm
        self.num_iterations = int(num_iterations)
        self.pipeline = pipeline
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.description = description
        self.score_func_name = score_func_name
        self.score_func = score_func
        self.message_list = MessageList()
        self.env = RunnableEnv(working_dir)
        self.max_code_attempts = int(max_code_attempts)
        LOGGER.info(
            "%s initialized: train_shape=%s test_shape=%s metric=%s iterations=%d max_code_attempts=%d working_dir=%s",
            self.__class__.__name__,
            getattr(self.X_train, "shape", None),
            getattr(self.X_test, "shape", None),
            self.score_func_name,
            self.num_iterations,
            self.max_code_attempts,
            self.env.working_dir,
        )

        with self.env.open("X_train.pkl", "wb") as handle:
            pickle.dump(self.X_train, handle)
        with self.env.open("X_test.pkl", "wb") as handle:
            pickle.dump(self.X_test, handle)
        with self.env.open("y_train.pkl", "wb") as handle:
            pickle.dump(self.y_train, handle)

    def loop(self):
        LOGGER.info("%s loop started", self.__class__.__name__)
        current_score, scores = self._get_validation_score()
        best_fold_scores = list(scores)
        LOGGER.info("Initial validation score: %.6f folds=%s", current_score, scores)
        new_score = None
        old_score = current_score
        best_transformations = (self.X_train, self.X_test)
        for iteration in range(self.num_iterations):
            LOGGER.info("Full flow iteration %d/%d started", iteration + 1, self.num_iterations)
            if iteration == 0:
                self._formulate_init_hypotheses()
            else:
                self._formulate_new_hypotheses(old_score, new_score, scores)
            analysis = self._submit_hypotheses_verification_code()
            new_transformations = self._implement_transformations(analysis)
            new_score, scores = self._get_validation_score(new_transformations)
            LOGGER.info("Iteration %d validation score: %.6f folds=%s", iteration + 1, new_score, scores)
            old_score = current_score
            if new_score > current_score:
                current_score = new_score
                best_transformations = new_transformations
                best_fold_scores = list(scores)
                LOGGER.info("Best transformation updated: new_score=%.6f old_score=%.6f", current_score, old_score)
            else:
                LOGGER.info("Best transformation kept: best_score=%.6f candidate_score=%.6f", current_score, new_score)
        self.best_X_train_, self.best_X_test_ = best_transformations
        self.best_score_ = current_score
        self.best_fold_scores_ = best_fold_scores
        predictions = self._generate_predictions(self.best_X_train_, self.best_X_test_)
        LOGGER.info("%s loop finished: predictions=%d best_score=%.6f", self.__class__.__name__, len(predictions), current_score)
        return predictions

    def _get_validation_score(self, transformations=None):
        if transformations is None:
            transformations = (self.X_train, self.X_test)
        X_train, _ = transformations
        splitter = TimeSeriesSplit(n_splits=4)
        scores = []
        LOGGER.info("Validation started: rows=%d columns=%d", len(X_train), len(getattr(X_train, "columns", [])))
        for fold_index, (train_ids, val_ids) in enumerate(splitter.split(X_train), start=1):
            started_at = time.monotonic()
            self.pipeline.fit(X_train.iloc[train_ids], self.y_train.iloc[train_ids])
            y_pred = self.pipeline.predict(X_train.iloc[val_ids])
            score = self.score_func(self.y_train.iloc[val_ids].reset_index(drop=True).to_numpy(), y_pred)
            LOGGER.info(
                "Validation fold %d finished: train_rows=%d val_rows=%d score=%.6f elapsed=%.2fs",
                fold_index,
                len(train_ids),
                len(val_ids),
                score,
                time.monotonic() - started_at,
            )
            scores.append(score)
        return float(np.mean(scores)), scores

    def _formulate_init_hypotheses(self) -> None:
        self.message_list.append_user_message(
            INIT_MESSAGE.format(
                description=self.description,
                X_train_head=self.X_train.head().to_string(max_cols=100),
                y_train_head=self.y_train.head().to_string(),
                score_func_name=self.score_func_name,
            )
        )
        LOGGER.info("Requesting initial hypotheses: messages=%d", len(self.message_list.get_messages()))
        response = self.llm.generate(self.message_list.get_messages())
        self.message_list.append_agent_message(response)
        LOGGER.info("Initial hypotheses received: chars=%d", len(response))

    def _formulate_new_hypotheses(self, old_score, new_score, scores) -> None:
        status = "Score improved. Best score updated." if old_score < new_score else "Score did not improve. Best score was not updated."
        fold_scores = "[" + " ".join(str(round(score, 3)) for score in scores) + "]"
        self.message_list.append_user_message(
            NEW_HYPOTHESES_MESSAGE.format(
                old_score=old_score,
                new_score=new_score,
                is_best_score_updated=fold_scores + "\n" + status,
            )
        )
        LOGGER.info("Requesting revised hypotheses: old_score=%.6f new_score=%.6f", old_score, new_score)
        response = self.llm.generate(self.message_list.get_messages())
        self.message_list.append_agent_message(response)
        LOGGER.info("Revised hypotheses received: chars=%d", len(response))

    def _submit_hypotheses_verification_code(self):
        self.message_list.append_user_message(ANALYSIS_MESSAGE)
        LOGGER.info("Requesting analysis code")
        return self._code_generation_loop()

    def _implement_transformations(self, analysis):
        LOGGER.info("Requesting implementation code: analysis_chars=%d", len(analysis))
        self.message_list.append_user_message(IMPLEMENTATION_MESSAGE.format(analysis=analysis))

        def has_generated():
            try:
                with self.env.open("X_train_fe.pkl", "rb") as handle:
                    pickle.load(handle)
                with self.env.open("X_test_fe.pkl", "rb") as handle:
                    pickle.load(handle)
            except FileNotFoundError as exc:
                return False, str(exc)
            return True, ""

        self._code_generation_loop(additional_checks=[has_generated])
        with self.env.open("X_train_fe.pkl", "rb") as handle:
            X_train = pickle.load(handle)
        with self.env.open("X_test_fe.pkl", "rb") as handle:
            X_test = pickle.load(handle)
        LOGGER.info("Loaded generated transformations: train_shape=%s test_shape=%s", getattr(X_train, "shape", None), getattr(X_test, "shape", None))
        return X_train, X_test

    def _generate_predictions(self, X_train, X_test):
        started_at = time.monotonic()
        LOGGER.info("Final pipeline fit started: train_shape=%s test_shape=%s", getattr(X_train, "shape", None), getattr(X_test, "shape", None))
        self.pipeline.fit(X_train, self.y_train)
        predictions = self.pipeline.predict(X_test)
        LOGGER.info("Final prediction finished: predictions=%d elapsed=%.2fs", len(predictions), time.monotonic() - started_at)
        return predictions

    def _code_generation_loop(self, additional_checks=None):
        additional_checks = additional_checks or []
        last_message = ""
        for attempt in range(1, self.max_code_attempts + 1):
            LOGGER.info("Code generation attempt %d/%d started: messages=%d", attempt, self.max_code_attempts, len(self.message_list.get_messages()))
            response = self.llm.generate(self.message_list.get_messages())
            self.message_list.append_agent_message(response)
            LOGGER.info("Code generation attempt %d response received: chars=%d", attempt, len(response))
            status, code = self._try_extract_code(response)
            if not status:
                last_message = CODE_EXTRACTION_FAILED
                LOGGER.warning("Code generation attempt %d failed: no code block", attempt)
                self.message_list.append_user_message(last_message)
                continue
            LOGGER.info("Code extracted on attempt %d: code_chars=%d", attempt, len(code))
            started_at = time.monotonic()
            status, msg = self._execute_code(code)
            if status != 0:
                last_message = CODE_EXECUTION_FAILED.format(msg=msg)
                LOGGER.warning("Generated code execution failed on attempt %d: status=%s output_chars=%d elapsed=%.2fs", attempt, status, len(msg), time.monotonic() - started_at)
                self.message_list.append_user_message(last_message)
                continue
            LOGGER.info("Generated code execution succeeded on attempt %d: output_chars=%d elapsed=%.2fs", attempt, len(msg), time.monotonic() - started_at)
            ok = True
            for check in additional_checks:
                result, message = check()
                if not result:
                    last_message = message
                    LOGGER.warning("Generated-code check failed on attempt %d: %s", attempt, message)
                    self.message_list.append_user_message(last_message)
                    ok = False
                    break
            if ok:
                LOGGER.info("Code generation attempt %d accepted", attempt)
                return msg
        raise RuntimeError(f"Generated code did not pass after {self.max_code_attempts} attempts. Last issue: {last_message}")

    def _try_extract_code(self, text: str):
        if not isinstance(text, str):
            return False, "No text content to extract code from."
        pattern_lang = re.compile(r"```(?:python|py)\s*(.*?)```", re.DOTALL | re.IGNORECASE)
        match = pattern_lang.search(text)
        if not match:
            match = re.search(r"```\s*(.*?)```", text, re.DOTALL)
        if match:
            code = match.group(1).strip()
            if code:
                return True, code
        return False, "No code block found."

    def _execute_code(self, code: str):
        return self.env.run_code(code)


class AnalyzeActFeatureOnlyFlow(AnalyzeActFlow):
    def loop(self):
        LOGGER.info("%s loop started", self.__class__.__name__)
        current_score, scores = self._get_validation_score()
        best_fold_scores = list(scores)
        LOGGER.info("Initial validation score: %.6f folds=%s", current_score, scores)
        best_transformations = (self.X_train, self.X_test)
        old_score = current_score
        new_score = None
        for iteration in range(self.num_iterations):
            LOGGER.info("Feature-only iteration %d/%d started", iteration + 1, self.num_iterations)
            if iteration == 0:
                self._request_direct_features_init()
            else:
                self._request_direct_features_refine(old_score, new_score, scores)
            new_transformations = self._implement_transformations_direct()
            new_score, scores = self._get_validation_score(new_transformations)
            LOGGER.info("Iteration %d validation score: %.6f folds=%s", iteration + 1, new_score, scores)
            old_score = current_score
            if new_score > current_score:
                current_score = new_score
                best_transformations = new_transformations
                best_fold_scores = list(scores)
                LOGGER.info("Best transformation updated: new_score=%.6f old_score=%.6f", current_score, old_score)
            else:
                LOGGER.info("Best transformation kept: best_score=%.6f candidate_score=%.6f", current_score, new_score)
        self.best_X_train_, self.best_X_test_ = best_transformations
        self.best_score_ = current_score
        self.best_fold_scores_ = best_fold_scores
        predictions = self._generate_predictions(self.best_X_train_, self.best_X_test_)
        LOGGER.info("%s loop finished: predictions=%d best_score=%.6f", self.__class__.__name__, len(predictions), current_score)
        return predictions

    def _request_direct_features_init(self) -> None:
        self.message_list.append_user_message(
            DIRECT_FE_INIT_MESSAGE.format(
                description=self.description,
                X_train_head=self.X_train.head().to_string(max_cols=100),
                y_train_head=self.y_train.head().to_string(),
                score_func_name=self.score_func_name,
            )
        )
        LOGGER.info("Requesting direct feature code: messages=%d", len(self.message_list.get_messages()))

    def _request_direct_features_refine(self, old_score, new_score, scores) -> None:
        fold_scores = "[" + " ".join(str(round(score, 3)) for score in scores) + "]"
        self.message_list.append_user_message(
            DIRECT_FE_REVISE_MESSAGE.format(
                old_score=old_score,
                new_score=new_score,
                fold_scores=fold_scores,
            )
        )
        LOGGER.info("Requesting refined direct feature code: old_score=%.6f new_score=%.6f", old_score, new_score)

    def _implement_transformations_direct(self):
        def has_generated():
            try:
                with self.env.open("X_train_fe.pkl", "rb") as handle:
                    pickle.load(handle)
                with self.env.open("X_test_fe.pkl", "rb") as handle:
                    pickle.load(handle)
            except FileNotFoundError as exc:
                return False, str(exc)
            return True, ""

        self._code_generation_loop(additional_checks=[has_generated])
        with self.env.open("X_train_fe.pkl", "rb") as handle:
            X_train = pickle.load(handle)
        with self.env.open("X_test_fe.pkl", "rb") as handle:
            X_test = pickle.load(handle)
        LOGGER.info("Loaded generated transformations: train_shape=%s test_shape=%s", getattr(X_train, "shape", None), getattr(X_test, "shape", None))
        return X_train, X_test


def _infer_default_metric(y: pd.Series) -> tuple[str, Callable]:
    y_ser = y if isinstance(y, pd.Series) else pd.Series(y)
    non_null = y_ser.dropna()
    unique = non_null.unique()
    if is_bool_dtype(y_ser) or is_integer_dtype(y_ser) or (len(unique) <= 20 and np.all(np.mod(unique, 1) == 0)):
        from sklearn.metrics import accuracy_score
        return "accuracy", accuracy_score
    from sklearn.metrics import r2_score
    return "r2", r2_score


class AnalyzeActPredictor:
    def __init__(
        self,
        llm: LLMClient,
        pipeline=None,
        num_iterations: int = 5,
        working_dir: str | Path | None = None,
        score_func_name: str | None = None,
        score_func: Callable | None = None,
        max_code_attempts: int = 5,
    ):
        self.llm = llm
        self.pipeline = pipeline or CatBoostClassifierPipeline()
        self.num_iterations = int(num_iterations)
        self.working_dir = Path(working_dir) if working_dir is not None else Path(tempfile.mkdtemp(prefix="analyze_act_"))
        self.score_func_name = score_func_name
        self.score_func = score_func
        self.max_code_attempts = int(max_code_attempts)
        self._fit_data: dict[str, object] = {}

    def fit(self, X_train, y_train, description: str = "Supervised learning task"):
        X_df = X_train if isinstance(X_train, pd.DataFrame) else pd.DataFrame(X_train)
        y_ser = y_train if isinstance(y_train, pd.Series) else pd.Series(y_train)
        if self.score_func is None or self.score_func_name is None:
            self.score_func_name, self.score_func = _infer_default_metric(y_ser)
        LOGGER.info(
            "%s fit: train_shape=%s metric=%s description_chars=%d",
            self.__class__.__name__,
            X_df.shape,
            self.score_func_name,
            len(description),
        )
        self._fit_data = {
            "description": description,
            "X_train": X_df.reset_index(drop=True),
            "y_train": y_ser.reset_index(drop=True),
        }
        return self

    def predict(self, X_test):
        if not self._fit_data:
            raise RuntimeError("Call fit() before predict().")
        X_df = X_test if isinstance(X_test, pd.DataFrame) else pd.DataFrame(X_test)
        LOGGER.info("%s predict: test_shape=%s", self.__class__.__name__, X_df.shape)
        flow = AnalyzeActFlow(
            llm=self.llm,
            num_iterations=self.num_iterations,
            pipeline=self.pipeline,
            X_train=self._fit_data["X_train"],
            y_train=self._fit_data["y_train"],
            X_test=X_df.reset_index(drop=True),
            description=self._fit_data["description"],
            score_func_name=self.score_func_name,
            score_func=self.score_func,
            working_dir=self.working_dir,
            max_code_attempts=self.max_code_attempts,
        )
        predictions = flow.loop()
        self.flow_ = flow
        self.X_train_fe_ = flow.best_X_train_
        self.X_test_fe_ = flow.best_X_test_
        self.best_score_ = flow.best_score_
        self.best_fold_scores_ = flow.best_fold_scores_
        self.predictions_ = predictions
        return predictions

    def fit_predict(self, X_train, y_train, X_test, description: str = "Supervised learning task"):
        return self.fit(X_train, y_train, description=description).predict(X_test)


class AnalyzeActFeatureOnlyPredictor(AnalyzeActPredictor):
    def predict(self, X_test):
        if not self._fit_data:
            raise RuntimeError("Call fit() before predict().")
        X_df = X_test if isinstance(X_test, pd.DataFrame) else pd.DataFrame(X_test)
        LOGGER.info("%s predict: test_shape=%s", self.__class__.__name__, X_df.shape)
        flow = AnalyzeActFeatureOnlyFlow(
            llm=self.llm,
            num_iterations=self.num_iterations,
            pipeline=self.pipeline,
            X_train=self._fit_data["X_train"],
            y_train=self._fit_data["y_train"],
            X_test=X_df.reset_index(drop=True),
            description=self._fit_data["description"],
            score_func_name=self.score_func_name,
            score_func=self.score_func,
            working_dir=self.working_dir,
            max_code_attempts=self.max_code_attempts,
        )
        predictions = flow.loop()
        self.flow_ = flow
        self.X_train_fe_ = flow.best_X_train_
        self.X_test_fe_ = flow.best_X_test_
        self.best_score_ = flow.best_score_
        self.best_fold_scores_ = flow.best_fold_scores_
        self.predictions_ = predictions
        return predictions
