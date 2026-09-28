"""
generate_samples.py
====================================================================
Draw a single-seed sample set from the input distributions defined in
``data/settings.json`` for the anchored canal-wall analytical model.

Adapted from Deltares-research/ms5-asset-performance
(case_studies/ark_main/mock/generate_mock_samples.py): same
"load settings.json -> array of (n_samples, n_variables)" shape and the
same truncated-distribution-via-clipping spirit, generalised from
truncated-normal-only to every ``distribution_type`` that appears in
this project's settings.json (normal, lognormal, gumbel, uniform,
deterministic), matching the marginal definitions used across the
source repo's own ``_build_marginal`` (run_mc.py, export_wall_params.py).

Surrogate-training sampling, correlation-in-u-space and the mock
polynomial moment model from the source repo are intentionally NOT
reproduced here -- this script only produces MC-style samples to feed
the real analytical model (see run_analytical_model.py).
====================================================================
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy import stats as st

_EULER = 0.5772156649  # Euler-Mascheroni constant, for Gumbel mu<->mean


def _is_deterministic(v: dict) -> bool:
    return v.get("distribution_type", "normal").lower() in ("deterministic", "constant", "fixed")


def _build_marginal(v: dict):
    """Frozen scipy distribution for one variable (None if deterministic).
    Mirrors the source repo's run_mc._build_marginal / export_wall_params
    so samples here are drawn from exactly the same marginals."""
    dist_type = v.get("distribution_type", "normal").lower()
    mean = float(v["mean"])
    std = float(v["standard_deviation"])
    lo = float(v.get("lower_bound", -np.inf))
    hi = float(v.get("upper_bound", np.inf))

    if _is_deterministic(v) or std == 0:
        return None

    if dist_type in ("normal", "norm", "n", "gaussian"):
        a = (lo - mean) / std if np.isfinite(lo) else -10.0
        b = (hi - mean) / std if np.isfinite(hi) else 10.0
        return st.truncnorm(a, b, loc=mean, scale=std)

    if dist_type in ("lognormal", "lognorm"):
        if mean <= 0:
            raise ValueError(f"Variable '{v['name']}': lognormal requires mean > 0.")
        sigma = float(np.sqrt(np.log(1.0 + (std / mean) ** 2)))
        mu = float(np.log(mean) - 0.5 * sigma ** 2)
        return st.lognorm(sigma, scale=np.exp(mu))

    if dist_type in ("gumbel", "gumbel_max", "gumbel_r"):
        beta = std * np.sqrt(6) / np.pi
        return st.gumbel_r(loc=mean - _EULER * beta, scale=beta)

    if dist_type in ("gumbel_min", "gumbel_l"):
        beta = std * np.sqrt(6) / np.pi
        return st.gumbel_l(loc=mean + _EULER * beta, scale=beta)

    if dist_type in ("uniform", "unif"):
        return st.uniform(loc=lo, scale=hi - lo)

    return st.norm(loc=mean, scale=std)


def _truncated_ppf(dist, u: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """PPF of `dist` truncated to [lo, hi] (idempotent for dists already
    supported there, e.g. truncnorm/uniform)."""
    F_lo = float(dist.cdf(lo)) if np.isfinite(lo) else 0.0
    F_hi = float(dist.cdf(hi)) if np.isfinite(hi) else 1.0
    return dist.ppf(F_lo + u * (F_hi - F_lo))


def sample_variables(variables: List[dict], n_samples: int, seed: int = 42) -> np.ndarray:
    """Draw independent MC samples for every variable in `variables`.

    Deterministic variables are held at their mean for every sample.
    Stochastic variables are sampled independently (no u-space
    correlation here -- add a Nataf/copula step if you later need
    correlated soil parameters, as the source repo does for FORM/MC).

    Returns:
        Array of shape (n_samples, n_variables), columns in the same
        order as `variables`.
    """
    n_vars = len(variables)
    samples = np.empty((n_samples, n_vars))
    rng = np.random.default_rng(seed)

    for j, v in enumerate(variables):
        if _is_deterministic(v):
            samples[:, j] = float(v["mean"])
            continue
        dist = _build_marginal(v)
        lo = float(v.get("lower_bound", -np.inf))
        hi = float(v.get("upper_bound", np.inf))
        u = rng.uniform(size=n_samples)
        samples[:, j] = _truncated_ppf(dist, u, lo, hi)

    return samples


def main(settings_path: Path, out_dir: Path, n_samples: int = 1_000, seed: int = 42) -> Path:
    with open(settings_path, "r") as f:
        settings = json.load(f)
    variables = settings["variables"]
    var_names = [v["name"] for v in variables]

    print(f"Sampling {n_samples} draws (seed={seed}) for {len(variables)} variables...")
    samples = sample_variables(variables, n_samples=n_samples, seed=seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"samples_seed{seed}.npy", samples)

    df = pd.DataFrame(samples, columns=var_names)
    csv_path = out_dir / f"samples_seed{seed}.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {out_dir / f'samples_seed{seed}.npy'}")
    print(f"  Saved: {csv_path}  shape={df.shape}")
    return csv_path
# ---------------------------------------------------------------- #
# Example usage:
# If this routine is ran a .csv file containing 1000 samples with 
# seed 42 in the path: data/generate_samples_example/samples_seed42.csv
# ---------------------------------------------------------------- #

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--settings", type=str, default="/Users/lorenzobrocchi_tum/Desktop/PhD - 01:12:25 - 30:11:2028/DELTARES/AAA_deltares_code/engineering_parameters.json")
    parser.add_argument("--out-dir", type=str, default="/Users/lorenzobrocchi_tum/Desktop/PhD - 01:12:25 - 30:11:2028/DELTARES/AAA_deltares_code/data/generate_samples_example")
    parser.add_argument("--n-samples", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(
        settings_path=Path(args.settings),
        out_dir=Path(args.out_dir),
        n_samples=args.n_samples,
        seed=args.seed,
    )
