Synthetic Dataset Generator
===========================

Overview
--------

`tabgen.generator` generates time-stamped synthetic tabular datasets with two aligned feature views:

- observable view: `t`, row-local features, entity labels, and exposed dependency observables
- optimal view: row-local features plus latent sufficient statistics used to define the target

The package is intended for evaluation settings where one wants both a realistic predictive task and a corresponding upper-bound view.

Core API
--------

```python
from tabgen.generator import Config, Generator

cfg = Config(
    n_samples=2000,
    task="classification",
    n_num=5,
    n_cat=2,
    cat_cardinality=4,
    sequence_dynamics=["seasonality", "recency_gap"],
    n_entities=12,
    entity_tuple_dims=1,
    c_dyn=0.5,
    c_noise=0.2,
    temperature=1.0,
    test_size=0.2,
    random_state=42,
)

gen = Generator(cfg)
ds_obs, ds_opt, reverse_name_map = gen.generate()
```

Returned objects:

- `ds_obs`: observable `Dataset`
- `ds_opt`: optimal `Dataset`
- `reverse_name_map`: mapping from anonymized observable columns back to their original semantic names

Configuration Reference
-----------------------

`Config` exposes the following main controls:

- `n_samples`: total number of rows
- `task`: `classification` or `regression`
- `n_num`: number of local numerical features
- `n_cat`: number of local categorical features
- `cat_cardinality`: maximum cardinality for local categorical features
- `sequence_dynamics`: list of dependency names from the registry
- `n_entities`: entity cardinality used by dependencies that require grouping
- `entity_tuple_dims`: number of entity dimensions
- `c_dyn`: strength of dynamic signal in the target
- `c_noise`: strength of noise in the target
- `temperature`: logistic temperature for classification
- `test_size`: chronological holdout fraction
- `random_state`: seed controlling generation

Train/Test Semantics
--------------------

The generator samples timestamps, sorts rows chronologically, and then splits by time:

- the earlier segment becomes training data
- the later segment becomes test data

This prevents future leakage and makes the suites better aligned with temporal evaluation.

Observable and Optimal Schemas
------------------------------

Observable schema:

- `t`
- anonymized local features such as `num_1`, `cat_1`
- generated entity columns when a dependency needs entities
- observable dynamic signals `D_O`

Optimal schema:

- original row-local features `z_num_*`, `z_cat_*`
- latent sufficient statistics `ds_*`

Files written by the batch scripts:

- `observables_train.csv`
- `observables_test.csv`
- `optimal_train.csv`
- `optimal_test.csv`
- `y_train.csv`
- `y_test.csv`
- `config_used.json`
- `metrics.json`

Dependency Registry
-------------------

The registry is implemented in `dependencies.py`. Active entries include:

- `seasonality`
- `piecewise_drift`
- `recency_gap`
- `rolling_volatility`
- `rolling_corr`
- `rolling_trend`
- `rolling_mean`
- `rolling_minmax`
- `streak`

Dependencies may expose:

- observable engineered signals
- latent sufficient statistics
- entity-conditioned behavior

Batch Entrypoints
-----------------

- `python -m tabgen.generator.generate_datasets --output-root ./out/no_drift`
- `python -m tabgen.generator.generate_datasets_drifts --output-root ./out/drift`
- `python -m tabgen.generator.generate_datasets_n_dyn --output-root ./out/n_dyn`
- `python -m tabgen.generator.generate_random_samples --output-root ./out/random_samples --count 3 --suite both`

Baseline Metrics
----------------

The batch scripts train CatBoost baselines and save the results in `metrics.json`.

Metric behavior:

- non-numeric columns are treated as categorical features
- classification runs report accuracy for both observable and optimal views
- regression runs report RMSE and `R^2` for both views

This is meant as a lightweight baseline, not as a full benchmark leaderboard.

Croissant Export
----------------

To export one generated run as Croissant JSON-LD:

```bash
PYTHONPATH=src python3 -m tabgen.generator.generate_croissant \
  --dataset-dir out/all_datasets/no_drift/seed_0/dyn_recency_gap/dim_0
```

To export one benchmark family:

```bash
PYTHONPATH=src python3 -m tabgen.generator.generate_croissant \
  --family-dir out/all_datasets/no_drift
```

To export one Croissant file for each family under a benchmark root:

```bash
PYTHONPATH=src python3 -m tabgen.generator.generate_croissant \
  --suite-dir out/all_datasets
```

By default, the exporter writes `croissant.json` into the selected run or family directory. Family-level export includes:

- file-level `distribution` entries for all run artifacts
- a manifest `recordSet` covering all runs in the family
- a baseline-metrics `recordSet` covering all runs in the family
