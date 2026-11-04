import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from tabgen.analyze_act import (
    CatBoostClassifierPipeline,
    CatBoostRegressorPipeline,
    AnalyzeActFlow,
    OpenAILLMClient,
)
from tabgen.analyze_act.run_single import run_dataset


class MockLLM:
    def __init__(self):
        self.stages = []

    def generate(self, messages):
        last_message = messages[-1]["content"]
        if "Propose 5 concrete" in last_message:
            self.stages.append("hypotheses")
            return "Use deterministic numeric transformations and preserve categorical columns."
        if "Please write Python analysis code" in last_message:
            self.stages.append("analysis")
            return """```python
import pickle

with open("X_train.pkl", "rb") as f:
    X_train = pickle.load(f)
with open("y_train.pkl", "rb") as f:
    y_train = pickle.load(f)

print("rows", len(X_train), "labels", len(y_train))
```"""
        if "Based on this analysis" in last_message:
            self.stages.append("implementation")
            return """```python
import pickle

with open("X_train.pkl", "rb") as f:
    X_train = pickle.load(f)
with open("X_test.pkl", "rb") as f:
    X_test = pickle.load(f)

def transform(df):
    out = df.copy()
    out["num_1_sq"] = out["num_1"] * out["num_1"]
    return out

X_train_fe = transform(X_train)
X_test_fe = transform(X_test)

with open("X_train_fe.pkl", "wb") as f:
    pickle.dump(X_train_fe, f)
with open("X_test_fe.pkl", "wb") as f:
    pickle.dump(X_test_fe, f)
```"""
        raise AssertionError(f"Unexpected prompt: {last_message[:120]}")


class MajorityByNumSignPipeline:
    def fit(self, X_train, y_train):
        self.positive_label = int(pd.Series(y_train).mode().iloc[0])
        return self

    def predict(self, X_test):
        return (X_test["num_1"].to_numpy() > 0).astype(int)


class SquareThresholdPipeline:
    def fit(self, X_train, y_train):
        self.has_square = "num_1_sq" in X_train.columns
        return self

    def predict(self, X_test):
        if self.has_square:
            return (X_test["num_1_sq"].to_numpy() > 1.0).astype(int)
        return np.zeros(len(X_test), dtype=int)


class FakeResponses:
    def __init__(self):
        self.last_request = None

    def create(self, **kwargs):
        self.last_request = kwargs
        return type(
            "Response",
            (),
            {
                "output_text": "```python\nprint('ok')\n```",
                "usage": {
                    "input_tokens": 100,
                    "input_tokens_details": {"cached_tokens": 25},
                    "output_tokens": 20,
                    "output_tokens_details": {"reasoning_tokens": 5},
                    "total_tokens": 120,
                },
            },
        )()


class FakeOpenAIClient:
    def __init__(self):
        self.responses = FakeResponses()


class AnalyzeActTest(unittest.TestCase):
    def test_catboost_defaults_match_benchmark_configs(self):
        for pipeline in (CatBoostClassifierPipeline(), CatBoostRegressorPipeline()):
            self.assertEqual(pipeline.params["iterations"], 500)
            self.assertEqual(pipeline.params["depth"], 6)
            self.assertEqual(pipeline.params["verbose"], 0)

        custom = CatBoostClassifierPipeline(iterations=100, learning_rate=0.1)
        self.assertEqual(custom.params["iterations"], 100)
        self.assertEqual(custom.params["depth"], 6)
        self.assertEqual(custom.params["learning_rate"], 0.1)

    def test_openai_llm_client_uses_responses_api(self):
        fake_client = FakeOpenAIClient()
        llm = OpenAILLMClient(
            model="gpt-test",
            client=fake_client,
            temperature=0.2,
            max_output_tokens=512,
        )
        text = llm.generate([{"role": "user", "content": "write code"}])

        self.assertEqual(text, "```python\nprint('ok')\n```")
        self.assertEqual(fake_client.responses.last_request["model"], "gpt-test")
        self.assertEqual(fake_client.responses.last_request["input"], [{"role": "user", "content": "write code"}])
        self.assertEqual(fake_client.responses.last_request["temperature"], 0.2)
        self.assertEqual(fake_client.responses.last_request["max_output_tokens"], 512)
        self.assertEqual(llm.request_count, 1)
        self.assertEqual(llm.usage_totals["input_tokens"], 100)
        self.assertEqual(llm.usage_totals["cached_input_tokens"], 25)
        self.assertEqual(llm.usage_totals["output_tokens"], 20)
        self.assertEqual(llm.usage_totals["reasoning_output_tokens"], 5)
        self.assertEqual(llm.usage_totals["total_tokens"], 120)

    def test_mock_llm_exercises_analyze_and_act_stages(self):
        X_train = pd.DataFrame(
            {
                "t": np.linspace(0.0, 1.0, 24),
                "num_1": np.linspace(-2.0, 2.0, 24),
                "cat_1": ["a", "b"] * 12,
            }
        )
        y_train = pd.Series((X_train["num_1"] > 0).astype(int))
        X_test = pd.DataFrame(
            {
                "t": [1.1, 1.2, 1.3],
                "num_1": [-0.5, 0.2, 1.5],
                "cat_1": ["a", "b", "a"],
            }
        )

        llm = MockLLM()
        with tempfile.TemporaryDirectory() as tmpdir:
            flow = AnalyzeActFlow(
                llm=llm,
                num_iterations=1,
                pipeline=MajorityByNumSignPipeline(),
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                description="Synthetic binary classification task",
                score_func_name="accuracy",
                score_func=accuracy_score,
                working_dir=tmpdir,
            )
            y_pred = flow.loop()

        self.assertEqual(llm.stages, ["hypotheses", "analysis", "implementation"])
        self.assertEqual(len(y_pred), len(X_test))
        np.testing.assert_array_equal(y_pred, np.array([0, 1, 1]))

    def test_single_dataset_runner_writes_selected_features_and_predictions(self):
        X_train = pd.DataFrame(
            {
                "t": np.linspace(0.0, 1.0, 24),
                "num_1": np.linspace(-2.0, 2.0, 24),
                "cat_1": ["a", "b"] * 12,
            }
        )
        y_train = pd.Series((X_train["num_1"].abs() > 1.0).astype(int), name="y")
        X_test = pd.DataFrame(
            {
                "t": [1.1, 1.2, 1.3],
                "num_1": [-0.5, 0.2, 1.5],
                "cat_1": ["a", "b", "a"],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "dataset"
            output_dir = Path(tmpdir) / "outputs"
            dataset_dir.mkdir()
            X_train.to_csv(dataset_dir / "observables_train.csv", index=False)
            X_test.to_csv(dataset_dir / "observables_test.csv", index=False)
            y_train.to_csv(dataset_dir / "y_train.csv", index=False)

            metadata = run_dataset(
                dataset_dir,
                output_dir,
                num_iterations=1,
                llm=MockLLM(),
                pipeline=SquareThresholdPipeline(),
            )

            X_train_fe = pd.read_csv(output_dir / "X_train_fe.csv")
            X_test_fe = pd.read_csv(output_dir / "X_test_fe.csv")
            predictions = pd.read_csv(output_dir / "predictions.csv")

            self.assertIn("num_1_sq", X_train_fe.columns)
            self.assertEqual(list(X_train_fe.columns), list(X_test_fe.columns))
            self.assertEqual(len(predictions), len(X_test))
            self.assertEqual(metadata["selected_feature_count"], X_train_fe.shape[1])
            self.assertTrue((output_dir / "run_metadata.json").is_file())


if __name__ == "__main__":
    unittest.main()
