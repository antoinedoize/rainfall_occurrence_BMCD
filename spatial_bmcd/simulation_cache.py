"""On-disk cache for the replicate ensembles of the spatial diagnostics.

An ensemble of ``N_REPLICATES`` seasonal Cholesky trajectories costs several
minutes (~7.5 min for 30 replicates at 26 stations, 1980-2020), which is paid
again at every kernel restart even when nothing but a plotting detail has
changed. :func:`simulate_ensemble_cached` runs the simulation **once** per
parameter set and reloads it afterwards.

The cache key is a SHA-256 fingerprint of everything the trajectories depend on:

- the seasonal ``theta`` (normalised, so ``(lam, ...)`` and ``{"lam": ...}``
  hash alike), the seeds, ``n_burn`` and the remaining simulator arguments;
- the station list and, per station and season, the parameters the simulator
  actually reads: ``lat``/``lon`` (the covariance distances) and
  ``params_dry``/``params_wet`` (the ``q`` closures -- the closures themselves
  are unhashable but are entirely determined by those tuples);
- the simulated calendar, and the NaN pattern of the observation mask;
- the **source of the simulation code path** (:func:`_sim_code_functions`),
  normalised through :mod:`ast`, so editing the model invalidates the cache
  while reformatting, comments or docstrings do not.

Any of those changing yields a different key, hence a fresh run; nothing else
does. If you add a helper to the simulation call chain, add it to
:func:`_sim_code_functions` -- otherwise a change inside it goes unnoticed and
a stale ensemble is reloaded. ``refresh=True`` forces a re-run either way.

One entry is two files in :data:`CACHE_DIR`: ``<label>_<key>.npz`` (the
replicates as one ``float32`` ``(n_reps, n_days, n_stations)`` array, deflated
-- 0/1/NaN compresses to a few MB out of ~50) and ``<label>_<key>.json`` (the
readable manifest, so an entry can be identified without loading the array).
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import textwrap
import time
import types
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent / "experiment_outputs" / "sim_ensemble_cache"

# Bump when the on-disk layout or the fingerprint content changes, to retire
# every entry written by an older version rather than misread it.
_CACHE_FORMAT_VERSION = 1


def _sim_code_functions() -> tuple:
    """The simulation call chain whose source the fingerprint covers.

    Imported lazily: :mod:`spatial_diagnostics` pulls matplotlib, which this
    module has no other reason to need.
    """
    from spatial_bmcd import spatial_model as sm
    from spatial_bmcd.spatial_diagnostics import history_to_Rbin

    return (
        sm.simulate_cholesky_seasonal,
        sm.normalize_theta,
        sm.build_field_cov_matrices,
        sm.build_C_from_sigma,
        sm.station_distance_matrix,
        sm._distance_matrix_from_coords.__wrapped__,  # lru_cache wrapper
        sm._assemble_selected_field,
        sm.season_of_dates,
        history_to_Rbin,
    )


def _code_digest(code, drop_docstring: bool = False) -> tuple:
    """Deterministic summary of a code object, recursing into nested functions.

    ``repr`` of a code object carries its memory address, so the constants are
    walked rather than repr-ed wholesale.
    """
    consts = code.co_consts[1:] if drop_docstring else code.co_consts
    return (
        code.co_name,
        code.co_code,
        tuple(_code_digest(c) if isinstance(c, types.CodeType) else repr(c) for c in consts),
        code.co_names,
        code.co_varnames,
    )


def _normalised_source(fn) -> str:
    """`fn`'s source as a dumped AST, without its docstring.

    Going through :mod:`ast` drops comments, blank lines and formatting; the
    docstring is popped explicitly. What is left is the behaviour of the
    function, which is what the cache must react to.
    """
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):
        # No source file -- a function defined in a notebook cell or exec'd.
        # Its bytecode still changes with its behaviour, which is what matters.
        code = getattr(fn, "__code__", None)
        if code is None:
            raise TypeError(f"cannot fingerprint {fn!r}: no source and no bytecode")
        return repr(_code_digest(code, drop_docstring=fn.__doc__ is not None))
    tree = ast.parse(textwrap.dedent(source))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        first = node.body[0] if node.body else None
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            node.body.pop(0)
            if not node.body:
                node.body.append(ast.Pass())
    return ast.dump(tree)


def code_fingerprint(functions=None) -> str:
    """SHA-256 (16 hex chars) of the source of the simulation call chain."""
    functions = _sim_code_functions() if functions is None else functions
    blob = "\n".join(f"{fn.__module__}.{fn.__qualname__}\n{_normalised_source(fn)}" for fn in functions)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _float(x) -> str:
    """Exact, round-trippable text form of a float, so 0.1 never collides."""
    return repr(float(x))


def _params_digest(params_by_season: Dict[str, Dict[str, dict]], station_names: Sequence[str]) -> dict:
    """The part of the per-station parameters the simulator actually reads.

    ``q_d_dry_function`` / ``q_d_wet_function`` are closures, hence unhashable,
    but :func:`spatial_model.build_dict_model_params` builds them from
    ``params_dry`` / ``params_wet`` alone -- those tuples stand in for them.
    """
    return {
        season: {
            name: {
                "lat": _float(p[name]["lat"]),
                "lon": _float(p[name]["lon"]),
                "params_dry": [_float(v) for v in p[name]["params_dry"]],
                "params_wet": [_float(v) for v in p[name]["params_wet"]],
            }
            for name in station_names
            if name in p
        }
        for season, p in sorted(params_by_season.items())
    }


def _array_digest(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def ensemble_manifest(
    *,
    dates,
    theta_by_season: Dict[str, object],
    params_by_season: Dict[str, Dict[str, dict]],
    station_names: Sequence[str],
    seeds: Sequence[int],
    mask: Optional[pd.DataFrame] = None,
    simulate_kwargs: Optional[dict] = None,
    extra: Optional[dict] = None,
) -> dict:
    """The JSON-serialisable description of an ensemble, hashed into its key."""
    from spatial_bmcd.spatial_model import normalize_theta

    dates = pd.DatetimeIndex(pd.to_datetime(dates))
    station_names = list(station_names)
    manifest = {
        "format_version": _CACHE_FORMAT_VERSION,
        "seeds": [int(s) for s in seeds],
        "simulate_kwargs": {k: _float(v) if isinstance(v, float) else v
                            for k, v in sorted((simulate_kwargs or {}).items())},
        "stations": station_names,
        "theta": {
            season: {k: _float(v) for k, v in sorted(normalize_theta(th).items())}
            for season, th in sorted(theta_by_season.items())
        },
        "params": _params_digest(params_by_season, station_names),
        "calendar": {
            "start": str(dates[0].date()),
            "end": str(dates[-1].date()),
            "n_days": int(len(dates)),
            "hash": _array_digest(dates.asi8),
        },
        "code": code_fingerprint(),
        "extra": extra or {},
    }
    if mask is not None:
        manifest["mask"] = {
            "columns": list(map(str, mask.columns)),
            "hash": _array_digest(np.packbits(mask.notna().to_numpy().ravel())),
        }
    return manifest


def ensemble_key(manifest: dict) -> str:
    """SHA-256 (16 hex chars) of a manifest -- the cache key."""
    blob = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _paths(cache_dir: Path, label: str, key: str) -> tuple:
    stem = f"{label}_{key}" if label else key
    return cache_dir / f"{stem}.npz", cache_dir / f"{stem}.json"


def save_ensemble(reps: Sequence[pd.DataFrame], manifest: dict, label: str,
                  cache_dir: Path = CACHE_DIR) -> Path:
    """Write an ensemble and its manifest under the manifest's key."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz_path, json_path = _paths(cache_dir, label, ensemble_key(manifest))
    values = np.stack([r.to_numpy(dtype="float32") for r in reps])
    index = pd.DatetimeIndex(reps[0].index)
    np.savez_compressed(
        npz_path,
        values=values,
        # As datetime64 rather than int64, so the index resolution (ns/us/...)
        # survives the round trip and the reloaded frames compare equal.
        index=index.to_numpy(),
        index_name=np.asarray(index.name if index.name is not None else ""),
        index_freq=np.asarray(index.freqstr if index.freq is not None else ""),
        columns=np.asarray(list(map(str, reps[0].columns))),
        manifest=json.dumps(manifest, sort_keys=True),
    )
    json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return npz_path


def load_ensemble(manifest: dict, label: str, cache_dir: Path = CACHE_DIR):
    """The cached ensemble for `manifest`, or ``None`` if there is no usable one."""
    npz_path, _ = _paths(Path(cache_dir), label, ensemble_key(manifest))
    if not npz_path.exists():
        return None
    try:
        with np.load(npz_path, allow_pickle=False) as z:
            values, index, columns = z["values"], z["index"], z["columns"]
            name = str(z["index_name"]) if "index_name" in z else ""
            freq = str(z["index_freq"]) if "index_freq" in z else ""
    except Exception as exc:  # truncated by an interrupted write, unreadable, ...
        print(f"[cache] ignoring unreadable {npz_path.name} ({exc})")
        return None
    idx = pd.DatetimeIndex(index, name=name or None, freq=freq or None)
    cols = list(columns)
    return [pd.DataFrame(v, index=idx, columns=cols) for v in values]


def simulate_ensemble_cached(
    *,
    dates,
    theta_by_season: Dict[str, object],
    params_by_season: Dict[str, Dict[str, dict]],
    station_names: Sequence[str],
    seeds: Sequence[int],
    mask: Optional[pd.DataFrame] = None,
    label: str = "ensemble",
    cache_dir: Path = CACHE_DIR,
    refresh: bool = False,
    verbose: bool = True,
    simulate: Optional[Callable] = None,
    **simulate_kwargs,
):
    """`len(seeds)` seasonal Cholesky replicates, simulated once and cached.

    Each replicate is ``simulate_cholesky_seasonal(dates, theta_by_season,
    params_by_season, station_names=station_names, seed=seed, **simulate_kwargs)``
    turned into a days x stations 0/1 frame. With `mask` given (typically the
    observed occurrence matrix) every replicate is reindexed on the mask's
    calendar and columns and NaN-ed wherever the mask is -- the frames are then
    directly comparable with the observations.

    Returns the list of replicates, from the cache when one matches the
    fingerprint of the arguments (see the module docstring), otherwise
    simulating and storing them. `refresh` re-runs and overwrites.
    """
    from spatial_bmcd.spatial_diagnostics import history_to_Rbin
    from spatial_bmcd.spatial_model import simulate_cholesky_seasonal

    simulate = simulate_cholesky_seasonal if simulate is None else simulate
    seeds = [int(s) for s in seeds]
    manifest = ensemble_manifest(
        dates=dates, theta_by_season=theta_by_season, params_by_season=params_by_season,
        station_names=station_names, seeds=seeds, mask=mask,
        simulate_kwargs=simulate_kwargs, extra={"label": label},
    )
    key = ensemble_key(manifest)

    if not refresh:
        reps = load_ensemble(manifest, label, cache_dir)
        if reps is not None:
            if verbose:
                print(f"[{label}] {len(reps)} replicates loaded from cache {key} "
                      f"(delete {_paths(Path(cache_dir), label, key)[0].name} or pass "
                      f"refresh=True to re-simulate)", flush=True)
            return reps

    n = len(seeds)
    reps, t0 = [], time.perf_counter()
    for k, seed in enumerate(seeds):
        rep = history_to_Rbin(simulate(
            dates, theta_by_season, params_by_season,
            station_names=list(station_names), seed=seed, **simulate_kwargs,
        ))
        if mask is not None:
            rep = (rep.reindex(index=mask.index, columns=mask.columns)
                      .where(mask.notna()).astype("float32"))
        else:
            rep = rep.astype("float32")
        reps.append(rep)
        if not verbose:
            continue
        if k == 0:
            dt = time.perf_counter() - t0
            print(f"[{label}] 1 replicate in {dt:.1f} s -> ~{dt * n / 60:.1f} min for {n}",
                  flush=True)
        elif (k + 1) % 10 == 0:
            print(f"[{label}]   {k + 1}/{n} ({time.perf_counter() - t0:.0f} s)", flush=True)

    npz_path = save_ensemble(reps, manifest, label, cache_dir)
    if verbose:
        print(f"[{label}] {len(reps)} replicates of {reps[0].shape} (days x stations), "
              f"{sum(r.memory_usage(deep=True).sum() for r in reps) / 1e6:.0f} MB in RAM, "
              f"{time.perf_counter() - t0:.0f} s -> cached as {npz_path.name} "
              f"({npz_path.stat().st_size / 1e6:.1f} MB)", flush=True)
    return reps


def list_cache(cache_dir: Path = CACHE_DIR) -> pd.DataFrame:
    """One row per cached ensemble: label, key, size, calendar, seeds, code fingerprint."""
    cache_dir = Path(cache_dir)
    rows = []
    for json_path in sorted(cache_dir.glob("*.json")):
        npz_path = json_path.with_suffix(".npz")
        m = json.loads(json_path.read_text())
        seeds = m.get("seeds", [])
        rows.append({
            "label": m.get("extra", {}).get("label", ""),
            "key": json_path.stem.rsplit("_", 1)[-1],
            "n_replicates": len(seeds),
            "seeds": f"{seeds[0]}-{seeds[-1]}" if seeds else "",
            "n_stations": len(m.get("stations", [])),
            "calendar": f"{m['calendar']['start']}..{m['calendar']['end']}",
            "code": m.get("code", ""),
            "MB": round(npz_path.stat().st_size / 1e6, 1) if npz_path.exists() else np.nan,
            "modified": (pd.Timestamp(npz_path.stat().st_mtime, unit="s").round("s")
                         if npz_path.exists() else pd.NaT),
        })
    columns = ["label", "key", "n_replicates", "seeds", "n_stations",
               "calendar", "code", "MB", "modified"]
    return pd.DataFrame(rows, columns=columns)
