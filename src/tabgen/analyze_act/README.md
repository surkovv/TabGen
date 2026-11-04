Analyze&Act Predictor
=========================

`tabgen.analyze_act` contains the system-only extraction of the Analyze&Act benchmark agent. It accepts:

- `X_train`: training features as a pandas `DataFrame`
- `y_train`: training labels as a pandas `Series`
- `X_test`: test features as a pandas `DataFrame`

and returns `y_pred` for `X_test`.

The package includes a single-dataset runner in `run_single.py` and an OpenAI Responses API adapter. Benchmark-wide task loaders are intentionally not included. Library callers may provide any LLM adapter with one method:

```python
class LLMClient:
    def generate(self, messages: list[dict[str, str]]) -> str:
        ...
```

The return value must be assistant text. When the system asks for code, that text must contain one fenced Python code block.

OpenAI Client
-------------

Install the optional OpenAI dependency:

```bash
pip install -e ".[openai]"
export OPENAI_API_KEY=...
```

Use `OpenAILLMClient` with the Responses API:

```python
from tabgen.analyze_act import OpenAILLMClient

llm = OpenAILLMClient(
    model="gpt-5-mini",
    max_output_tokens=2048,
)
```

Additional keyword arguments are passed to `client.responses.create(...)`. You can inject a preconfigured OpenAI-compatible client with `OpenAILLMClient(client=...)`.

Main Classes
------------

- `AnalyzeActFlow`: full Analyze&Act loop with hypothesis generation, analysis-code execution, implementation-code execution, validation scoring, and final prediction.
- `AnalyzeActFeatureOnlyFlow`: ablated loop that skips hypothesis and analysis stages and directly asks for feature engineering code.
- `AnalyzeActPredictor`: sklearn-like wrapper around `AnalyzeActFlow`.
- `AnalyzeActFeatureOnlyPredictor`: sklearn-like wrapper around `AnalyzeActFeatureOnlyFlow`.
- `OpenAILLMClient`: OpenAI Responses API adapter implementing `generate(messages) -> str`.
- `CatBoostClassifierPipeline`: default classification pipeline with benchmark CatBoost defaults.
- `CatBoostRegressorPipeline`: default regression pipeline with benchmark CatBoost defaults.

Full Analyze&Act Flow
-------------------------

`AnalyzeActFlow` performs these stages:

1. Compute an initial time-series cross-validation score using the raw features.
2. Ask the LLM for feature-engineering hypotheses.
3. Ask the LLM for analysis code that reads `X_train.pkl` and `y_train.pkl`.
4. Execute the analysis code in an isolated temporary working directory.
5. Ask the LLM for implementation code that reads `X_train.pkl`, `y_train.pkl`, and `X_test.pkl`.
6. Execute the implementation code and require it to write `X_train_fe.pkl` and `X_test_fe.pkl`.
7. Score the transformed training features with `TimeSeriesSplit`.
8. Keep the best feature transformation.
9. Fit the downstream pipeline on the selected training features and return predictions for transformed `X_test`.

The prompts are copied verbatim from the source Analyze&Act implementation, including original wording and typos. This keeps the extracted system behavior aligned with the benchmark agent.

Predictor Usage
---------------

```python
from tabgen.analyze_act import AnalyzeActPredictor, CatBoostClassifierPipeline


class MyLLM:
    def generate(self, messages):
        # Return assistant text. Code-generation requests must include a Python block.
        return """```python
import pickle

with open("X_train.pkl", "rb") as f:
    X_train = pickle.load(f)
with open("X_test.pkl", "rb") as f:
    X_test = pickle.load(f)

X_train_fe = X_train.copy()
X_test_fe = X_test.copy()

with open("X_train_fe.pkl", "wb") as f:
    pickle.dump(X_train_fe, f)
with open("X_test_fe.pkl", "wb") as f:
    pickle.dump(X_test_fe, f)
```"""


predictor = AnalyzeActPredictor(
    llm=MyLLM(),
    pipeline=CatBoostClassifierPipeline(),
    num_iterations=1,
)

y_pred = predictor.fit_predict(
    X_train,
    y_train,
    X_test,
    description="Synthetic binary classification task",
)
```

For the feature-only ablation:

```python
from tabgen.analyze_act import AnalyzeActFeatureOnlyPredictor

predictor = AnalyzeActFeatureOnlyPredictor(llm=MyLLM(), num_iterations=3)
y_pred = predictor.fit_predict(X_train, y_train, X_test)
```

CatBoost Defaults
-----------------

The default CatBoost pipelines preserve the benchmark configuration:

- `iterations=500`
- `depth=6`
- `verbose=0`

Any CatBoost parameter can still be overridden by passing it to `CatBoostClassifierPipeline(...)` or `CatBoostRegressorPipeline(...)`.

Generated Code Contract
-----------------------

The generated implementation code must:

- read `X_train.pkl`, `y_train.pkl`, and `X_test.pkl` from the current working directory
- create pandas `DataFrame`s named `X_train_fe` and `X_test_fe`
- ensure both frames have identical columns
- write `X_train_fe.pkl` and `X_test_fe.pkl`
- avoid external data and network access

The generated analysis code must:

- read `X_train.pkl` and `y_train.pkl`
- print useful analysis to stdout
- avoid writing required output files

Execution Model
---------------

Generated code is executed with:

```bash
python -I -u -c "<generated code>"
```

The `-I` flag puts Python in isolated mode. The process runs in a temporary or caller-provided working directory containing the pickled data files. Execution timeout is capped at 180 seconds.

Validation
----------

Run the mock-LLM full-stage test:

```bash
PYTHONPATH=src python3 -m unittest tests.test_analyze_act
```

Run the live OpenAI smoke test only when API access is configured:

```bash
pip install -e ".[openai]"
export OPENAI_API_KEY=...
RUN_OPENAI_LIVE_TEST=1 PYTHONPATH=src python3 -m unittest tests.test_analyze_act_openai_live
```

The live test defaults to `OPENAI_MODEL=gpt-5-mini`, `OPENAI_MAX_OUTPUT_TOKENS=2048`, and `OPENAI_REASONING_EFFORT=low` for reasoning models. Increase `OPENAI_MAX_OUTPUT_TOKENS` if the API returns an incomplete response before visible output.

Enable Analyze&Act progress logs from your own runner:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
```

The logs include request sizes, code-generation attempts, generated-code execution status, validation folds, and final prediction counts. Full prompts and generated code are not logged by default.

The test verifies that a mock LLM drives the system through:

- hypotheses
- analysis
- implementation
- generated feature validation
- time-series CV scoring
- final prediction

Limitations
-----------

- The system executes LLM-generated Python. Use it only in a sandboxed environment appropriate for untrusted code.
- The default validation splitter is `TimeSeriesSplit(n_splits=4)`.
- The wrapper infers a default metric from `y_train` when one is not provided.
- `OpenAILLMClient` requires `OPENAI_API_KEY` unless the caller injects a configured client.
