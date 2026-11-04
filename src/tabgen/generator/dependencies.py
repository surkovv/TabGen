from __future__ import annotations

from typing import Optional, Dict, Type

import numpy as np
import pandas as pd
from scipy.stats import zscore


class Dep:
    requires_entity: bool = False

    def compute(
        self, t: np.ndarray, entity: Optional[np.ndarray], rng: np.random.Generator
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Return (observables, latents) as dicts mapping base feature names to arrays.
        The generator will apply unique naming and instance suffixes.
        """
        raise NotImplementedError

    # Context is injected by the caller (generator) before calling _gen_entities/compute
    _context_n: Optional[int] = None
    _context_rng: Optional[np.random.Generator] = None

    def _set_context(self, n: int, rng: np.random.Generator) -> None:
        self._context_n = int(n)
        self._context_rng = rng

    def _gen_entities(self, entities_dim: int, entities_cardinality: int) -> Optional[np.ndarray]:
        """
        Default entity generator: independent uniform categories per dimension.
        - Returns an (n, entities_dim) ndarray of dtype object with labels 'c{idx}'.
        - If entities_dim <= 0, returns None.
        """
        dims = int(max(0, entities_dim))
        if dims <= 0:
            return None
        n = int(self._context_n or 0)
        if n <= 0:
            raise ValueError("_gen_entities requires context n to be set before calling")
        rng = self._context_rng or np.random.default_rng()
        K = int(max(2, entities_cardinality))
        mat = np.empty((n, dims), dtype=object)
        for j in range(dims):
            ids = rng.integers(0, K, size=n)
            mat[:, j] = np.array([f"c{ii}" for ii in ids], dtype=object)
        return mat


class SeasonalityDep(Dep):
    requires_entity = False
    provide_observable = False  # seasonality is latent-only

    def compute(self, t: np.ndarray, entity: Optional[np.ndarray], rng: np.random.Generator) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        # Higher-frequency seasonality: 8..15 cycles over the time span
        t_span = (t.max() - t.min()) + 1e-9
        cycles = rng.uniform(5, 15)
        P = max(t_span / float(cycles), 1e-6)
        phase = float(rng.uniform(0, 2 * np.pi))
        s = np.sin(2 * np.pi * t / P + phase)
        return ({}, {"seasonality": zscore(s.astype(float), ddof=0)})


class PiecewiseDriftDep(Dep):
    requires_entity = False
    provide_observable = False  # latent-only like seasonality

    def compute(
        self, t: np.ndarray, entity: Optional[np.ndarray], rng: np.random.Generator
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Single-cycle piecewise (V-shaped) drift over the observed time span.
        - Normalize u = (t - t_min) / (t_max - t_min)
        - Triangular wave with elbow in the middle: v = 2*|u - 1/2|
        - Standardize to zero-mean/unit-variance
        """
        t_min = float(np.min(t))
        span = float((np.max(t) - t_min) + 1e-12)
        u = (t - t_min) / span
        tri = 2.0 * np.abs(u - 0.5)
        return ({}, {"piecewise_drift": zscore(tri.astype(float), ddof=0)})


# class MultiHarmonicSeasonalityDep(Dep):
#     requires_entity = False
#     provide_observable = False

#     def __init__(
#         self,
#         min_harmonics: int = 2,
#         max_harmonics: int = 3,
#         cycles_min: int = 5,
#         cycles_max: int = 15,
#         include_cos: bool = False,
#     ):
#         # Number of harmonics will be sampled uniformly in [min_harmonics, max_harmonics]
#         self.min_h = int(max(1, min_harmonics))
#         self.max_h = int(max(self.min_h, max_harmonics))
#         # Per-harmonic cycles (like SeasonalityDep: cycles in [8,15])
#         self.cycles_min = int(cycles_min)
#         self.cycles_max = int(max(cycles_min, cycles_max))
#         self.include_cos = bool(include_cos)

#     def compute(self, t: np.ndarray, entity: Optional[np.ndarray], rng: np.random.Generator) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
#         # Normalize time to [0,1] span (aligns with SeasonalityDep's use of span)
#         t_min = float(np.min(t))
#         span = float((np.max(t) - t_min) + 1e-9)
#         u = (t - t_min) / span

#         # Sample number of harmonics and integer cycles per harmonic
#         n_h = int(rng.integers(self.min_h, self.max_h + 1))
#         cycles = [rng.uniform(self.min_h, self.max_h)]
#         while len(cycles) < n_h:
#             cycles.append(
#                 cycles[-1] * rng.integers(2, 4)
#             )

#         lat: dict[str, np.ndarray] = {}
#         for i, ccy in enumerate(cycles):
#             phase_s = float(rng.uniform(0, 2 * np.pi))
#             s = np.sin(2 * np.pi * ccy * u + phase_s)
#             lat[f"mharm_sin_c{int(i)}"] = zscore(s.astype(float), ddof=0)
#             if self.include_cos:
#                 phase_c = float(rng.uniform(0, 2 * np.pi))
#                 c = np.cos(2 * np.pi * ccy * u + phase_c)
#                 lat[f"mharm_cos_c{int(i)}"] = zscore(c.astype(float), ddof=0)
#         return ({}, lat)




class RecencyGapDep(Dep):
    requires_entity = True
    provide_observable = False  # transformation should be engineered; keep latent-only

    def compute(self, t: np.ndarray, entity: Optional[np.ndarray], rng: np.random.Generator) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        if entity is None:
            # No entities provided: fall back to global gaps
            df = pd.DataFrame({"row": np.arange(len(t)), "t": t})
            df_sorted = df.sort_values(["t"]).reset_index(drop=True)
            gap = df_sorted["t"].diff().values
        else:
            ent_cols = entity if entity.ndim == 2 else entity.reshape(-1, 1)
            df = pd.DataFrame({"row": np.arange(len(t)), "t": t})
            for j in range(ent_cols.shape[1]):
                df[f"entity{j}"] = ent_cols[:, j]
            sort_keys = [f"entity{j}" for j in range(ent_cols.shape[1])] + ["t"]
            df_sorted = df.sort_values(sort_keys).reset_index(drop=True)
            gap = df_sorted.groupby(sort_keys[:-1])["t"].diff().values
        df_sorted["gap"] = gap
        if entity is None:
            df_sorted["rank_in_entity"] = np.arange(len(df_sorted))
            # Avoid carrying 'gap' to the left side of merge to prevent suffixes
            df0 = df_sorted[["row", "rank_in_entity"]].copy()
        else:
            df_sorted["rank_in_entity"] = df_sorted.groupby(sort_keys[:-1]).cumcount()
            df0 = df.sort_values(sort_keys).reset_index(drop=True)
            df0["rank_in_entity"] = df0.groupby(sort_keys[:-1]).cumcount()
        merge_keys = ([f"entity{j}" for j in range(ent_cols.shape[1])] if entity is not None else []) + ["rank_in_entity"]
        df_final = (
            df0.merge(
                df_sorted[merge_keys + ["gap"]],
                on=merge_keys,
                how="left",
            )
            .sort_values("row")
            .reset_index(drop=True)
        )
        gap_raw = pd.Series(df_final["gap"]).fillna(0.0).values.astype(float)
        g = np.log1p(gap_raw)
        return ({}, {"recency_gap": zscore(g.astype(float), ddof=0)})


class RollingVolatilityDep(Dep):
    requires_entity = True
    provide_observable = True   # expose raw per-entity signal as observable

    def __init__(self, window: Optional[int] = None):
        self.window = int(window) if window is not None else None

    def compute(self, t: np.ndarray, entity: Optional[np.ndarray], rng: np.random.Generator) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        ent_cols = entity if entity is not None and entity.ndim == 2 else (entity.reshape(-1, 1) if entity is not None else None)
        df0 = pd.DataFrame({"row": np.arange(len(t)), "t": t})
        if ent_cols is not None:
            for j in range(ent_cols.shape[1]):
                df0[f"entity{j}"] = ent_cols[:, j]
            sort_keys = [f"entity{j}" for j in range(ent_cols.shape[1])] + ["t"]
        else:
            sort_keys = ["t"]
        df0 = df0.sort_values(sort_keys).reset_index(drop=True)

        # Structured volatility: raw_signal = a_t * z_t, where a_t follows AR(1)
        raw = np.zeros(len(df0))
        group_cols = [f"entity{j}" for j in range(ent_cols.shape[1])] if ent_cols is not None else []
        for e, grp in (df0.groupby(group_cols, sort=False) if group_cols else [(None, df0)]):
            idx = grp.index.to_numpy()
            m = len(idx)
            if m == 0:
                continue
            # AR(1) amplitude process a_t and standard normal z_t
            phi = float(rng.uniform(0.9, 0.99))
            a = np.zeros(m)
            a[0] = rng.standard_normal()
            for k in range(1, m):
                a[k] = phi * a[k - 1] + ((1 - phi**2) ** 0.5) * rng.standard_normal()
            z = rng.standard_normal(m)
            raw[idx] = a * z

        df0["raw_signal"] = raw
        # rolling std of lagged raw_signal per entity estimates the time-varying volatility
        w = self.window if self.window is not None else int(rng.integers(5, 26))
        if ent_cols is not None:
            roll = (
                df0.groupby([f"entity{j}" for j in range(ent_cols.shape[1])])["raw_signal"]
                .apply(lambda col: col.shift(1).rolling(w, min_periods=5).std())
                .reset_index(level=0, drop=True)
            )
        else:
            roll = df0["raw_signal"].shift(1).rolling(w, min_periods=5).std()
        vol = np.nan_to_num(roll.values, nan=0.0)
        return (
            {"raw_rolling_volatility": raw.astype(float)},
            {"rolling_volatility": zscore(vol.astype(float), ddof=0)},
        )


class NeighborAverageDep(Dep):
    # Temporarily disabled; reserved for future extension (entity tuples, topology)
    requires_entity = True
    provide_observable = True

    def __init__(self, *args, **kwargs):
        pass

    def compute(self, t: np.ndarray, entity: Optional[np.ndarray], rng: np.random.Generator):
        raise NotImplementedError("NeighborAverageDep is disabled in this build.")


def _ensure_group_cols(entity: Optional[np.ndarray]):
    if entity is None:
        return None, []
    ent_cols = entity if entity.ndim == 2 else entity.reshape(-1, 1)
    group_cols = [f"entity{j}" for j in range(ent_cols.shape[1])]
    return ent_cols, group_cols


class RollingCorrDep(Dep):
    requires_entity = True
    provide_observable = True

    def __init__(self, window: Optional[int] = None):
        self.window = int(window) if window is not None else None

    def compute(self, t, entity, rng):
        ent_cols, group_cols = _ensure_group_cols(entity)
        df = pd.DataFrame({"t": t})
        if group_cols:
            for idx, name in enumerate(group_cols):
                df[name] = ent_cols[:, idx]
        df = df.sort_values(group_cols + ["t"]).reset_index(drop=True)

        # Generate correlated raws with time-varying common intensity
        
        def gen_pair(m):
            phi = rng.uniform(0.9, 0.99)
            sig = rng.standard_normal(size=m)
            for i in range(1, m):
                sig[i] = (phi) * sig[i-1] + ((1-phi**2) ** 0.5) * rng.standard_normal()
            bias = rng.uniform(-3, 3)
            rho = np.arctan((sig + bias)) / (np.pi / 2)
            x1 = rng.standard_normal(m)
            x2 = rng.standard_normal(m)
            x = x1
            y = x1 * rho + x2 * np.sqrt(np.clip(1 - rho**2, 0, 1))
            
            return x, y

        if group_cols:
            x = np.zeros(len(df))
            y = np.zeros(len(df))
            for _, grp in df.groupby(group_cols, sort=False):
                idx = grp.index.to_numpy()
                xi, yi = gen_pair(len(idx))
                x[idx] = xi
                y[idx] = yi
        else:
            x, y = gen_pair(len(df))
        df["x_i"], df["x_j"] = x, y

        w = self.window if self.window is not None else int(rng.integers(5, 26))
        if group_cols:
            corr = (
                df.groupby(group_cols)
                .apply(lambda g: g[["x_i", "x_j"]].shift(1).rolling(w, min_periods=5).corr().unstack().iloc[:, 1])
                .reset_index(level=0, drop=True)
            )
        else:
            corr = df[["x_i", "x_j"]].shift(1).rolling(w, min_periods=5).corr().unstack().iloc[:, 1]
        corr = np.nan_to_num(corr.values, nan=0.0)
        return (
            {"raw_rolling_corr_x": df["x_i"].values.astype(float), "raw_rolling_corr_y": df["x_j"].values.astype(float)},
            {"rolling_corr": zscore(corr, ddof=0)}
        )


class RollingTrendDep(Dep):
    requires_entity = True
    provide_observable = True

    def __init__(self, window: Optional[int] = None):
        self.window = int(window) if window is not None else None

    def compute(self, t, entity, rng):
        ent_cols, group_cols = _ensure_group_cols(entity)
        df = pd.DataFrame({"t": t})
        if group_cols:
            for idx, name in enumerate(group_cols):
                df[name] = ent_cols[:, idx]
        # Simulate a raw process per entity for trend estimation
        # Structured raw per entity: cumulative sum of AR(1) (local level)
        def gen_raw(m: int) -> np.ndarray:
            phi = float(rng.uniform(0.9, 0.99))
            ar = np.zeros(m)
            ar[0] = rng.normal(0.0, 1)
            for i in range(1, m):
                ar[i] = phi * ar[i - 1] + (1 - phi**2) ** 0.5 * rng.normal(0.0, 1)
            return np.cumsum(ar)
        if group_cols:
            raw = np.zeros(len(df))
            for _, grp in df.groupby(group_cols, sort=False):
                idx = grp.index.to_numpy()
                raw[idx] = gen_raw(len(idx))
            df["raw"] = raw
        else:
            df["raw"] = gen_raw(len(df))
        df = df.sort_values(group_cols + ["t"]).reset_index(drop=True)
        w = self.window if self.window is not None else int(rng.integers(5, 26))
        def slope(series):
            y = series.values
            x = np.arange(len(y))
            y = pd.Series(y).shift(1)
            x = pd.Series(x).shift(1)
            return pd.Series(y).rolling(w, min_periods=5).apply(
                lambda s: np.polyfit(np.arange(len(s)), s, 1)[0], raw=False
            )
        if group_cols:
            sl = df.groupby(group_cols)["raw"].apply(slope).reset_index(level=0, drop=True)
        else:
            sl = slope(df["raw"]).reset_index(drop=True)
        sl = np.nan_to_num(sl.values, nan=0.0)
        # Expose raw process to enable exact FE reconstruction
        return ({"raw_rolling_trend": df["raw"].values.astype(float)}, {"rolling_trend": zscore(sl, ddof=0)})


class StreakDep(Dep):
    requires_entity = True
    provide_observable = True

    def __init__(self, use_log: bool = True, cap: int = 100, vocab_size: int = 50, p_stick: float = 0.9):
        self.use_log = bool(use_log)
        self.cap = int(cap)
        self.vocab_size = int(max(2, vocab_size))
        self.p_stick = float(min(max(p_stick, 0.0), 0.9999))

    def compute(self, t, entity, rng):
        ent_cols, group_cols = _ensure_group_cols(entity)
        n = len(t)
        df = pd.DataFrame({"t": t})
        if group_cols:
            for idx, name in enumerate(group_cols):
                df[name] = ent_cols[:, idx]
            sort_keys = group_cols + ["t"]
        else:
            sort_keys = ["t"]
        df = df.sort_values(sort_keys).reset_index(drop=True)

        # Generate a sticky categorical sequence per group to induce streaks
        def gen_ids(m: int) -> np.ndarray:
            out = np.empty(m, dtype=int)
            out[0] = int(rng.integers(0, self.vocab_size))
            for i in range(1, m):
                if rng.random() < self.p_stick:
                    out[i] = out[i-1]
                else:
                    # pick a different id
                    nxt = int(rng.integers(0, self.vocab_size - 1))
                    out[i] = nxt if nxt < out[i-1] else nxt + 1
            return out

        if group_cols:
            raw_ids = np.empty(len(df), dtype=int)
            for _, grp in df.groupby(group_cols, sort=False):
                idx = grp.index.to_numpy()
                raw_ids[idx] = gen_ids(len(idx))
        else:
            raw_ids = gen_ids(len(df))
        df["raw_id"] = raw_ids

        # Compute consecutive streak length per group
        def streak_len(series: pd.Series) -> pd.Series:
            s = series.to_numpy()
            out = np.ones(len(s), dtype=int)
            for i in range(1, len(s)):
                out[i] = out[i-1] + 1 if s[i] == s[i-1] else 1
            return pd.Series(out)

        if group_cols:
            streak = df.groupby(group_cols)["raw_id"].apply(streak_len).reset_index(level=0, drop=True)
        else:
            streak = streak_len(df["raw_id"])  # global

        streak = streak.clip(upper=self.cap).astype(float)
        val = np.log1p(streak.values) if self.use_log else streak.values
        ds = zscore(val, ddof=0)
        obs = {"raw_streak_id": np.array([f"id_{i}" for i in df["raw_id"].to_numpy()], dtype=object)}
        return (obs, {"streak_len": ds})


class RollingMeanDep(Dep):
    requires_entity = True
    provide_observable = True

    def __init__(self, window: Optional[int] = None):
        self.window = int(window) if window is not None else None

    def compute(self, t, entity, rng):
        ent_cols, group_cols = _ensure_group_cols(entity)
        df = pd.DataFrame({"t": t})
        if group_cols:
            for idx, name in enumerate(group_cols):
                df[name] = ent_cols[:, idx]
        # Simulate structured raw per entity: AR(1) * N(0,1)
        def gen_raw(m: int) -> np.ndarray:
            phi = float(rng.uniform(0.9, 0.99))
            ar = np.zeros(m)
            ar[0] = rng.normal(0.0, 1)
            for i in range(1, m):
                ar[i] = phi * ar[i - 1] + (1-phi**2) ** 0.5 * rng.normal(0.0, 1)
            return ar
        if group_cols:
            raw = np.zeros(len(df))
            for _, grp in df.groupby(group_cols, sort=False):
                idx = grp.index.to_numpy()
                raw[idx] = gen_raw(len(idx))
            df["raw"] = raw
        else:
            df["raw"] = gen_raw(len(df))
        df = df.sort_values(group_cols + ["t"]).reset_index(drop=True)
        w = self.window if self.window is not None else int(rng.integers(5, 26))
        if group_cols:
            mean = df.groupby(group_cols)["raw"].apply(lambda s: s.shift(1).rolling(w, min_periods=5).mean()).reset_index(level=0, drop=True)
        else:
            mean = df["raw"].shift(1).rolling(w, min_periods=5).mean()
        mean = mean.fillna(0.0).values
        return ({"raw_rolling_mean": df["raw"].values.astype(float)}, {"rolling_mean": zscore(mean, ddof=0)})


class RollingMinMaxDep(Dep):
    requires_entity = True
    provide_observable = True

    def __init__(self, window: Optional[int] = None):
        self.window = int(window) if window is not None else None

    def compute(self, t, entity, rng):
        ent_cols, group_cols = _ensure_group_cols(entity)
        df = pd.DataFrame({"t": t})
        if group_cols:
            for idx, name in enumerate(group_cols):
                df[name] = ent_cols[:, idx]
        df["raw"] = rng.standard_normal(len(df))
        df = df.sort_values(group_cols + ["t"]).reset_index(drop=True)
        w = self.window if self.window is not None else int(rng.integers(5, 26))
        if group_cols:
            rmin = df.groupby(group_cols)["raw"].apply(lambda s: s.shift(1).rolling(w, min_periods=5).min()).reset_index(level=0, drop=True)
            rmax = df.groupby(group_cols)["raw"].apply(lambda s: s.shift(1).rolling(w, min_periods=5).max()).reset_index(level=0, drop=True)
        else:
            rmin = df["raw"].shift(1).rolling(w, min_periods=5).min()
            rmax = df["raw"].shift(1).rolling(w, min_periods=5).max()
        rmin = rmin.fillna(0.0).values
        rmax = rmax.fillna(0.0).values
        return (
            {"raw_rolling_minmax": df["raw"].values.astype(float)},
            {"rolling_min": zscore(rmin, ddof=0), "rolling_max": zscore(rmax, ddof=0)}
        )


def get_dependency_registry() -> Dict[str, Type[Dep]]:
    return {
        "seasonality": SeasonalityDep,
        "piecewise_drift": PiecewiseDriftDep,
        "recency_gap": RecencyGapDep,
        "rolling_volatility": RollingVolatilityDep,
        "rolling_corr": RollingCorrDep,
        "rolling_trend": RollingTrendDep,
        "rolling_mean": RollingMeanDep,
        "rolling_minmax": RollingMinMaxDep,
        "streak": StreakDep,
    }


__all__ = [
    "Dep",
    "SeasonalityDep",
    "PiecewiseDriftDep",
    "RecencyGapDep",
    "RollingVolatilityDep",
    "RollingCorrDep",
    "RollingTrendDep",
    "RollingMeanDep",
    "RollingMinMaxDep",
    "StreakDep",
    "get_dependency_registry",
]
