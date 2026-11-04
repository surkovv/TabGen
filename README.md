TabGen
======

TabGen is a synthetic tabular benchmark generator for temporal and entity-conditioned learning problems. Each generated run contains two aligned views:

- `observables_*`: realistic observable features for downstream models
- `optimal_*`: row-local features plus latent sufficient statistics used to construct the target

The repo also includes a small Analyze&Act predictor extraction under `tabgen.analyze_act`.

The generated datasets and Croissant metadata are available on [Hugging Face](https://huggingface.co/datasets/surokpro2/TabGen/).

Install
-------

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

Without editable install, run commands with `PYTHONPATH=src`.

Generate Datasets
-----------------

```bash
PYTHONPATH=src python3 -m tabgen.generator.generate_datasets --output-root out/all_datasets/no_drift
PYTHONPATH=src python3 -m tabgen.generator.generate_datasets_drifts --output-root out/all_datasets/drift
PYTHONPATH=src python3 -m tabgen.generator.generate_datasets_n_dyn --output-root out/all_datasets/n_dyn
PYTHONPATH=src python3 -m tabgen.generator.generate_random_samples --output-root out/all_datasets/random_samples --count 3 --suite both
```

Current family sizes:

- `no_drift`: 63 runs
- `drift`: 36 runs
- `n_dyn`: 30 runs
- `random_samples`: configurable with `--count`

Each run writes:

- `observables_train.csv`, `observables_test.csv`
- `optimal_train.csv`, `optimal_test.csv`
- `y_train.csv`, `y_test.csv`
- `config_used.json`, `metrics.json`

Croissant Metadata
------------------

Generate one Croissant file per benchmark family:

```bash
PYTHONPATH=src python3 -m tabgen.generator.generate_croissant --suite-dir out/all_datasets
```

Or generate metadata for one family:

```bash
PYTHONPATH=src python3 -m tabgen.generator.generate_croissant --family-dir out/all_datasets/no_drift
```

Family-level `croissant.json` files include file distributions, SHA-256 checksums, run manifests, baseline metrics, and required Responsible AI metadata.

Analyze&Act Predictor
-------------------------

`tabgen.analyze_act` provides the full Analyze&Act ML pipeline by default. It accepts `X_train`, `y_train`, `X_test`, and an LLM client implementing `generate(messages) -> str`, then returns `y_pred`.

```python
from tabgen.analyze_act import AnalyzeActPredictor, CatBoostClassifierPipeline

predictor = AnalyzeActPredictor(
    llm=my_llm,
    pipeline=CatBoostClassifierPipeline(),
    num_iterations=5,
)
y_pred = predictor.fit_predict(X_train, y_train, X_test, description="Classification task")
```

Use `AnalyzeActFeatureOnlyPredictor` only for the feature-only ablation. See `src/tabgen/analyze_act/README.md` for the full execution contract and mock-LLM test.

For OpenAI-backed runs:

```bash
pip install -e ".[openai]"
export OPENAI_API_KEY=...
```

```python
from tabgen.analyze_act import OpenAILLMClient

llm = OpenAILLMClient(model="gpt-5-mini", max_output_tokens=2048)
```

Enable progress logs with Python `logging.basicConfig(level=logging.INFO)`.

Single-Dataset Analyze&Act Runner
---------------------------------

After generating a suite, run Analyze&Act on one dataset directory with:

```bash
export OPENAI_API_KEY=...
tabgen-analyze-act \
  --dataset-dir out/all_datasets/no_drift/seed_0/dyn_recency_gap/dim_0 \
  --output-dir out/analyze_act_recency_seed_0 \
  --model gpt-5-mini \
  --num-iterations 5 \
  --random-seed 0
```

The input directory must contain `observables_train.csv`, `observables_test.csv`, and `y_train.csv`. The runner writes:

- `X_train_fe.csv`: the selected engineered training features
- `X_test_fe.csv`: the matching engineered test features
- `predictions.csv`: downstream-model predictions for the test rows
- `run_metadata.json`: inputs, model/backend, seed, validation scores, elapsed time, request count, and API-reported token usage

`--random-seed` controls NumPy and CatBoost; hosted LLM responses may remain nondeterministic. With five iterations, the benchmark configuration uses 21 downstream fits and normally makes 15 LLM requests plus any repair calls. In the reported experiments, one dataset used about 30k output, 200k cached-input, and 40k uncached-input tokens on average. Actual runtime and token usage are recorded in `run_metadata.json`; monetary cost depends on the provider's current model pricing. Existing outputs are protected unless `--overwrite` is passed.

Project Layout
--------------

- `src/tabgen/generator/`: synthetic data generator and suite scripts
- `src/tabgen/generator/README.md`: generator API and schema details
- `src/tabgen/analyze_act/`: Analyze&Act predictor
- `src/tabgen/analyze_act/README.md`: predictor API and generated-code contract
- `tests/test_analyze_act.py`: mock-LLM full-stage test

Sanity Checks
-------------

```bash
PYTHONPATH=src python3 -m tabgen.generator.generate_datasets --help
PYTHONPATH=src python3 -m tabgen.generator.generate_datasets_drifts --help
PYTHONPATH=src python3 -m tabgen.generator.generate_datasets_n_dyn --help
PYTHONPATH=src python3 -m tabgen.generator.generate_random_samples --help
PYTHONPATH=src python3 -m unittest tests.test_analyze_act
```
