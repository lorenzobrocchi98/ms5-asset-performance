"""
run_mc_fast.py
====================================================================
Streams MC chunks straight into a DuckDB file: one row per sample,
with M and V as native n_points-element arrays (indexable, e.g.
M[0] = value at the roller/anchor) and anchor_force / hinge_force
as scalars. z_grid (constant across samples) is stored once in a
metadata table. Scales to large sample counts without holding
everything in memory -- DuckDB spills to disk, and samples are
drawn and solved one chunk at a time.

Adapted to call the closed-form 'analytical_model.py' (Coulomb
earth-pressure / statically-determinate roller-hinge beam) instead
of the old fast_canal_wall_model.run_mc surrogate. Because
solve_batch() is a single vectorized call over a whole samples
DataFrame rather than a generator, chunking is done here: each
chunk is drawn with generate_samples.sample_variables() (same
marginals as generate_samples.py) and solved with
analytical_model.solve_batch() in one shot.

====================================================================
"""

from __future__ import annotations

import json
import time
from pathlib import Path


import duckdb  

import numpy as np
import pandas as pd

from analytical_model import SoilLayer, WallGeometry, solve_batch
from generate_samples import sample_variables


def init_db(path: Path, n_z: int, z_grid: np.ndarray) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(path))
    con.execute(f"CREATE OR REPLACE TABLE results "
                f"(sample_id BIGINT, anchor_force DOUBLE, hinge_force DOUBLE, "
                f"M DOUBLE[{n_z}], V DOUBLE[{n_z}])")
    con.execute("CREATE OR REPLACE TABLE metadata (z_grid DOUBLE[])")
    con.execute("INSERT INTO metadata VALUES (?)", [z_grid.tolist()])
    return con


def write_chunk(con: duckdb.DuckDBPyConnection, start_id: int, out: dict):
    """out: dict from analytical_model.solve_batch (keys z, V, M, R_anchor, R_hinge)."""
    n = out["M"].shape[0]
    df = pd.DataFrame({
        "sample_id": np.arange(start_id, start_id + n),
        "anchor_force": out["R_anchor"],
        "hinge_force": out["R_hinge"],
        "M": list(out["M"]),
        "V": list(out["V"]),
    })
    con.register("df", df)
    con.execute("INSERT INTO results SELECT * FROM df")

def init_samples_db(path: Path, var_names: list) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(path))
    cols = ", ".join(f'"{name}" DOUBLE' for name in var_names)
    con.execute(f"CREATE OR REPLACE TABLE samples (sample_id BIGINT, {cols})")
    return con

def write_samples_chunk(con: duckdb.DuckDBPyConnection, start_id: int, samples: pd.DataFrame):
    """samples: the DataFrame of drawn input variables for this chunk, same
    row order/sample_id as the matching chunk written to the results DB."""
    df = samples.copy()
    df.insert(0, "sample_id", np.arange(start_id, start_id + len(df)))
    con.register("df", df)
    con.execute("INSERT INTO samples SELECT * FROM df")

def run_mc(n_samples: int, variables: list, geom: WallGeometry, layers: list,
           delta_ratio: float, seed: int, chunk_size: int, on_chunk) -> dict:
    """Draw + solve `n_samples` MC samples in chunks of `chunk_size`.

    Each chunk gets its own sub-seed spawned from a single root RNG, so
    the whole run is reproducible from `seed` regardless of chunk_size,
    and chunks are statistically independent draws.
    """
    var_names = [v["name"] for v in variables]
    root_rng = np.random.default_rng(seed)
    m_ed_max_chunks = []

    n_done, i = 0, 0
    while n_done < n_samples:
        n_chunk = min(chunk_size, n_samples - n_done)
        chunk_seed = int(root_rng.integers(0, 2**32 - 1))

        arr = sample_variables(variables, n_samples=n_chunk, seed=chunk_seed)
        samples = pd.DataFrame(arr, columns=var_names)

        out = solve_batch(samples, geom, layers, None, delta_ratio=delta_ratio)
        m_ed_max_chunks.append(np.max(np.abs(out["M"]), axis=1))

        on_chunk(i, out, samples)
        n_done += n_chunk
        i += 1

    m_ed_max = np.concatenate(m_ed_max_chunks)
    return {"mean_M_Ed_max": float(m_ed_max.mean()), "std_M_Ed_max": float(m_ed_max.std())}


def main(settings_path: Path, db_path: Path, samples_db_path: Path,
          n_samples: int, chunk_size: int, seed: int):
    settings = json.loads(settings_path.read_text())
    variables = settings["variables"]
    v = {x["name"]: x["mean"] for x in variables}
    p = settings.get("parameters", {})

    # analytical_model.SoilLayer only uses .name/.thickness for stratigraphy
    # geometry here -- solve_batch re-reads sampled phi/c/gamma per row from
    # the samples DataFrame, same as the previous fast-model pipeline.
    layers = [
        SoilLayer(name="Sand", thickness=v["Sand_thickness"],
                  gamma=v["Sand_soilgamdry"], gamma_sat=v["Sand_soilgamwet"],
                  phi_k=v["Sand_soilphi"], c_k=v["Sand_soilcohesion"]),
        SoilLayer(name="Clay", thickness=v["Clay_thickness"],
                  gamma=v["Clay_soilgamdry"], gamma_sat=v["Clay_soilgamwet"],
                  phi_k=v["Clay_soilphi"], c_k=v["Clay_soilcohesion"]),
    ]

    embedment_depth = p.get("embedment_depth")
    if embedment_depth is None:
        embedment_depth = max(0.0, v["anchor_level"] - v["dredge_level"])

    geom = WallGeometry(top_level=v["top_level"], dredge_level=v["dredge_level"],
                         anchor_level=v["anchor_level"],
                         embedment_depth=float(embedment_depth),
                         hinge_fraction=float(p.get("hinge_fraction", 0.9)),
                         n_points=20)
    delta_ratio = float(p.get("delta_ratio", 2.0 / 3.0))

    var_names = [x["name"] for x in variables]
    z_grid = np.linspace(geom.anchor_level, geom.hinge_level, geom.n_points)

    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.unlink(missing_ok=True)
    con = init_db(db_path, len(z_grid), z_grid)

    samples_db_path.parent.mkdir(parents=True, exist_ok=True)
    samples_db_path.unlink(missing_ok=True)
    samples_con = init_samples_db(samples_db_path, var_names)

    t0 = time.time()

    def on_chunk(i, out, samples):
        write_chunk(con, i * chunk_size, out)
        write_samples_chunk(samples_con, i * chunk_size, samples)
        print(f"  chunk {i}: {len(out['M']):,} rows written, {time.time()-t0:6.1f}s elapsed")

    summary = run_mc(n_samples, variables, geom, layers, delta_ratio,
                      seed=seed, chunk_size=chunk_size, on_chunk=on_chunk)

    dt = time.time() - t0
    n_rows = con.execute("SELECT count(*) FROM results").fetchone()[0]
    print(f"\n{n_rows:,} rows in {dt:.1f}s ({n_rows/dt:,.0f} samples/s)  "
          f"db size: {db_path.stat().st_size/1e6:.1f} MB")
    print(f"M_Ed_max: mean={summary['mean_M_Ed_max']:.1f}  std={summary['std_M_Ed_max']:.1f}")
    con.close()
    samples_con.close()
# ------------------------------------------------------------------------------ #
# Montecarlo simulation of the analytical model: expected time ~ 5 mins
# ca 250 000 samples/sec - Default setting: seed 42, total smaples generated 
# 1e8, a single chunk of 1e6 samples.
#
# NOTE: the dataset is fully reproducible for a fixed tuple (chunk_size, seed)
#       Changing either the size or the seed will lead to a different result 
# ------------------------------------------------------------------------------ #

if __name__ == "__main__":
    from argparse import ArgumentParser

    ap = ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--settings", default="/Users/lorenzobrocchi_tum/Desktop/PhD - 01:12:25 - 30:11:2028/DELTARES/AAA_deltares_code/engineering_parameters.json")
    ap.add_argument("--db", default="/Users/lorenzobrocchi_tum/Desktop/PhD - 01:12:25 - 30:11:2028/DELTARES/AAA_deltares_code/results/mc_fast.duckdb")
    ap.add_argument("--samples-db", default="/Users/lorenzobrocchi_tum/Desktop/PhD - 01:12:25 - 30:11:2028/DELTARES/AAA_deltares_code/results/mc_samples.duckdb")
    ap.add_argument("--n-samples", type=int, default=100_000_000)
    ap.add_argument("--chunk-size", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

main(Path(args.settings), Path(args.db), Path(args.samples_db),
         args.n_samples, args.chunk_size, args.seed)