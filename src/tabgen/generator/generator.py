from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional

import numpy as np
from scipy.stats import zscore
from .dependencies import get_dependency_registry
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class Config:
    # Dataset size and task
    n_samples: int = 2000
    task: str = "classification"  # or "regression"

    # Row-local features Z
    n_num: int = 3
    n_cat: int = 2
    cat_cardinality: int = 4

    # Sequence Dynamics to enable (names from dependency registry)
    sequence_dynamics: Optional[List[str]] = None

    # Entities for grouped dynamics
    n_entities: int = 10
    entity_tuple_dims: int = 1

    # Mixing and classification temperature
    c_dyn: float = 0.5
    c_noise: float = 0.2
    temperature: float = 1.0

    # Split and RNG
    test_size: float = 0.2
    random_state: int = 42


@dataclass
class Dataset:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


class Generator:
    def __init__(self, config: Config):
        self.cfg = config
        self.rng = np.random.default_rng(self.cfg.random_state)

    def _make_numeric(self, n: int) -> np.ndarray:
        return self.rng.standard_normal((n, self.cfg.n_num))

    def _make_categorical(self, n: int) -> np.ndarray:
        k = int(self.cfg.n_cat)
        if k <= 0:
            return np.empty((n, 0), dtype=object)
        Kmax = max(int(self.cfg.cat_cardinality), 2)
        cats = np.empty((n, k), dtype=object)
        # Per-feature random cardinality in [2, Kmax]; sample from feature-specific Dirichlet
        for j in range(k):
            Kj = int(self.rng.integers(2, Kmax + 1))
            p = self.rng.dirichlet(alpha=np.ones(Kj))
            idx = self.rng.choice(Kj, size=n, p=p)
            cats[:, j] = np.array([f"c{ii}" for ii in idx], dtype=object)
        return cats

    def _unique_name(self, df: pd.DataFrame, base: str) -> str:
        name = base
        k = 1
        while name in df.columns:
            name = f"{base}__{k}"
            k += 1
        return name

    def _anonymize_observables(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """Anonymize observable feature column names (keep 't' unchanged).
        - Numeric-like columns become num_1, num_2, ...
        - Categorical-like columns become cat_1, cat_2, ...
        Type heuristic:
          * dtype object/category → categorical-like
          * names starting with 'entity_' → categorical-like
          * otherwise use dtype to decide
        Order within each type is randomized (seeded by cfg.random_state).
        """
        cols = list(df.columns)
        rename_map: Dict[str, str] = {}

        keep_time = 't' in df.columns

        cat_like: List[str] = []
        num_like: List[str] = []
        for c in cols:
            if c == 't':
                continue
            dtype = str(df[c].dtype)
            if dtype in ("object", "category") or c.startswith("entity_"):
                cat_like.append(c)
            else:
                num_like.append(c)

        if len(num_like) > 1:
            num_like = list(np.array(num_like)[self.rng.permutation(len(num_like))])
        if len(cat_like) > 1:
            cat_like = list(np.array(cat_like)[self.rng.permutation(len(cat_like))])

        for i, c in enumerate(num_like, start=1):
            rename_map[c] = f"num_{i}"
        for i, c in enumerate(cat_like, start=1):
            rename_map[c] = f"cat_{i}"

        adjusted_map: Dict[str, str] = {}
        taken = set(cols)
        for old, new in rename_map.items():
            name = new
            k = 1
            while name in taken and name != old:
                name = f"{new}__{k}"
                k += 1
            adjusted_map[old] = name
            taken.discard(old)
            taken.add(name)

        renamed = df.rename(columns=adjusted_map)
        new_order: List[str] = []
        if keep_time:
            new_order.append('t')
        new_order.extend([adjusted_map.get(c, c) for c in num_like])
        new_order.extend([adjusted_map.get(c, c) for c in cat_like])

        reversed_adj_map = {v: k for k, v in adjusted_map.items()}
        return renamed[new_order], reversed_adj_map

    def _build_s_loc(self, Xn: np.ndarray, C: np.ndarray) -> np.ndarray:
        """s_loc = sum over numeric and categorical features.
        - Numeric: each standardized and scaled by its own N(0,1) weight.
        - Categorical: treat each feature as a whole by sampling per-category
          coefficients, centering them, mapping to rows, standardizing the
          resulting vector, and scaling by a single feature-level N(0,1) weight.
        This keeps a categorical feature's aggregate influence comparable to a
        single numeric feature, regardless of cardinality.
        """
        n = Xn.shape[0]
        s = np.zeros(n, dtype=float)
        # numeric features
        for i in range(Xn.shape[1]):
            x = Xn[:, i].astype(float)
            if np.std(x) > 1e-12:
                xz = zscore(x, ddof=0)
                wi = float(self.rng.normal(0.0, 1.0))
                s += wi * xz
        # categorical features (composite treatment)
        if C is not None and C.size > 0 and C.shape[1] > 0:
            for j in range(C.shape[1]):
                ser = pd.Series(C[:, j].astype(str)).astype('category')
                codes = ser.cat.codes.values
                Kj = len(ser.cat.categories)
                beta = self.rng.normal(0.0, 1.0, size=Kj)
                beta = beta - beta.mean()
                v = beta[codes]
                if np.std(v) > 1e-12:
                    vz = zscore(v.astype(float), ddof=0)
                    aj = float(self.rng.normal(0.0, 1.0))
                    s += aj * vz
        return s


    def _compute_sequence_dynamics(self, t: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, list]]:
        """Return (D_S, entity_cols, D_O, groups).
        - D_S: dict of dynamic sufficient statistics per-output
        - entity_cols: generated entity columns
        - D_O: optional observables
        - groups: mapping group_id (dependency instance) -> list of D_S keys
        """
        D_S: Dict[str, np.ndarray] = {}
        entities: Dict[str, np.ndarray] = {}
        D_O: Dict[str, np.ndarray] = {}
        groups: Dict[str, list] = {}

        selected = list(self.cfg.sequence_dynamics or [])
        registry_classes = get_dependency_registry()
        counts: Dict[str, int] = {}
        for name in selected:
            if name not in registry_classes:
                continue
            dep = registry_classes[name]()
            k = counts.get(name, 0)
            key_ent = f"entity_{name}_{k}"

            # Let dependency generate its own entities when required
            ent_vec = None
            if getattr(dep, "requires_entity", False):
                dims = max(int(self.cfg.entity_tuple_dims), 0)
                n_entities = max(int(self.cfg.n_entities), 1)
                # Pass context and ask dependency to synthesize entities
                if hasattr(dep, "_set_context"):
                    dep._set_context(n=len(t), rng=self.rng)
                ent_vec = dep._gen_entities(entities_dim=dims, entities_cardinality=n_entities)
                if ent_vec is not None and ent_vec.size > 0:
                    for j in range(ent_vec.shape[1]):
                        entities[f"{key_ent}__d{j}"] = ent_vec[:, j]

            obs_map, suff_map = dep.compute(t, ent_vec, self.rng)
            for base, vec in obs_map.items():
                D_O[f"{base}_{k}"] = vec
            for base, vec in suff_map.items():
                key = f"ds_{base}_{k}"
                D_S[key] = vec
                gid = f"{name}#{k}"
                groups.setdefault(gid, []).append(key)

            counts[name] = k + 1

        return D_S, entities, D_O, groups

    def generate(self) -> Tuple[Dataset, Dataset, dict]:
        n = self.cfg.n_samples
        # Build time and enforce chronological ordering
        t = self.rng.uniform(0.0, 1.0, size=n)
        order = np.argsort(t)
        t = t[order]

        # Row-local features Z (numeric + categorical), aligned to time
        Xn_raw = self._make_numeric(n)
        Xn = Xn_raw[order]
        C = self._make_categorical(n)
        if C.shape[1] > 0:
            C = C[order]

        # Compute Sequence Dynamics (per-instance entities generated internally)
        D_S, ent_cols, D_O, groups = self._compute_sequence_dynamics(t)

        # Build s_loc from Z and s_dyn from D_S; combine with mixing and noise
        s_loc = self._build_s_loc(Xn, C)
        if np.std(s_loc) > 1e-12:
            s_loc = zscore(s_loc, ddof=0)
        else:
            s_loc = np.zeros_like(s_loc)
            
        # Build s_dyn with group-normalized weights to avoid multiplicity dominance
        s_dyn = np.zeros(n, dtype=float)
        for gid, keys in groups.items():
            # Collect standardized vectors for this dependency instance
            vecs = []
            for key in keys:
                arr = np.asarray(D_S[key])
                if arr.dtype == object:
                    labels = pd.Series(arr.astype(str))
                    cats = labels.astype('category').cat.categories
                    codes = labels.astype('category').cat.codes.values
                    K = len(cats)
                    beta = self.rng.normal(0.0, 1.0, size=K)
                    beta = beta - beta.mean()
                    v = beta[codes]
                    if np.std(v) > 1e-12:
                        vecs.append(zscore(v.astype(float), ddof=0))
                    else:
                        vecs.append(np.zeros_like(arr, dtype=float))
                else:
                    x = arr.astype(float)
                    if x.std() > 1e-12:
                        vecs.append(zscore(x, ddof=0))
                    else:
                        vecs.append(np.zeros_like(x))
            if not vecs:
                continue
            m = len(vecs)
            w = self.rng.normal(0.0, 1.0, size=m)
            norm = float(np.linalg.norm(w))
            if not np.isfinite(norm) or norm < 1e-12:
                w = np.ones(m, dtype=float) / np.sqrt(m)
            else:
                w = w / norm
            a_g = float(self.rng.normal(0.0, 1.0))
            gsum = np.zeros(n, dtype=float)
            for wi, vi in zip(w, vecs):
                gsum += wi * vi
            s_dyn += a_g * gsum

        # Standardize the combined dynamic score to stabilize scale across seeds
        if np.std(s_dyn) > 1e-12:
            s_dyn = zscore(s_dyn, ddof=0)
        else:
            s_dyn = np.zeros_like(s_dyn)

        c_dyn = float(self.cfg.c_dyn)
        c_noise = float(self.cfg.c_noise)
        c_loc = max(0.0, 1.0 - c_dyn - c_noise)
        eps = self.rng.standard_normal(n)
        score = c_dyn ** 0.5 * s_dyn + c_noise ** 0.5 * eps + c_loc ** 0.5 * s_loc

        if self.cfg.task == "regression":
            y = score
        else:
            temp = max(1e-6, float(self.cfg.temperature))
            prob = 1.0 / (1.0 + np.exp(-(score / temp)))
            y = (self.rng.random(size=prob.shape) < prob).astype(int)

        # Observable view: {t, Z, D_O} (+ entity_* only here)
        X_obs = pd.DataFrame({"t": t})
        for i in range(Xn.shape[1]):
            X_obs[f"z_num_{i}"] = Xn[:, i]
        if C.shape[1] > 0:
            for j in range(C.shape[1]):
                X_obs[f"z_cat_{j}"] = C[:, j]
        for name, vec in ent_cols.items():
            X_obs[name] = vec
        for name, vec in D_O.items():
            X_obs[name] = vec
        # Anonymize observables (keep 't' only)
        X_obs, rev_map = self._anonymize_observables(X_obs)

        # Optimal view: {Z, D_S}
        X_opt = pd.DataFrame()
        for i in range(Xn.shape[1]):
            X_opt[f"z_num_{i}"] = Xn[:, i]
        if C.shape[1] > 0:
            for j in range(C.shape[1]):
                X_opt[f"z_cat_{j}"] = C[:, j]
        for key, vec in D_S.items():
            X_opt[key] = vec

        # Chronological split by time (first chunk is train, last chunk is test)
        split_idx = int(np.floor(n * (1.0 - self.cfg.test_size)))
        split_idx = min(max(split_idx, 1), n - 1)  # ensure non-empty splits

        y_series = pd.Series(y, name="y")

        ds_obs = Dataset(
            X_train=X_obs.iloc[:split_idx].reset_index(drop=True),
            y_train=y_series.iloc[:split_idx].reset_index(drop=True),
            X_test=X_obs.iloc[split_idx:].reset_index(drop=True),
            y_test=y_series.iloc[split_idx:].reset_index(drop=True),
        )
        ds_opt = Dataset(
            X_train=X_opt.iloc[:split_idx].reset_index(drop=True),
            y_train=y_series.iloc[:split_idx].reset_index(drop=True),
            X_test=X_opt.iloc[split_idx:].reset_index(drop=True),
            y_test=y_series.iloc[split_idx:].reset_index(drop=True),
        )
        return ds_obs, ds_opt, rev_map
