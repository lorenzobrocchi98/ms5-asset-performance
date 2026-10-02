"""
Fragility curve evaluation for the sheet-pile wall moment limit state:

    LSF_wall = M_capacity * (1 - Cr) / (|M| * theta_M) - 1

Fragility curve: P(LSF_wall <= 0 | Cr) as a function of the corrosion ratio Cr,
evaluated on a fixed Cr grid.

Failure condition, rearranged (avoids dividing by |M|*theta_M, which can be
~0 for degenerate MC draws):

    LSF_wall <= 0   <=>   |M| * theta_M  >=  M_capacity * (1 - Cr)

Inputs
------
- cross-section_properties.json  -> M_capacity (fixed, per 1 m run of wall)
- corrosion_ratio_grid.json      -> Cr grid (100 points in [0, 1])
- mc_fast.duckdb    (table: results)  -> sample_id, M
    M is a LIST per row (moment profile for that MC sample). The acting
    moment |M| for the LSF is taken as the max ABSOLUTE value in that list.
- mc_samples.duckdb (table: samples)  -> model_factor_M (= theta_M)

Output
------
- fragility_curve.json  -> {"cr": [...], "pf": [...], "n_samples": N}
- fragility_curve.png   -> plot (log-scale y-axis)

Requires: duckdb, numpy, matplotlib  (pip install duckdb numpy matplotlib)
"""

import json
from pathlib import Path

import duckdb
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- paths: adjust if your files live elsewhere ----------------------------
HERE = Path(__file__).parent
CROSS_SECTION_JSON = HERE / "cross-section_properties.json"
CR_GRID_JSON = HERE / "corrosion_ratio_grid.json"
MC_FAST_DB = HERE /'results'/'mc_fast.duckdb'
MC_SAMPLES_DB = HERE /'results' / "mc_samples.duckdb"
OUTPUT_JSON = HERE / 'results'/ "fragility_curve.json"
OUTPUT_PNG = HERE / "fragility_curve.png"


def load_inputs():
    with open(CROSS_SECTION_JSON) as f:
        cross_section = json.load(f)
    m_capacity = cross_section["M_capacity"]

    with open(CR_GRID_JSON) as f:
        cr_grid = json.load(f)
    cr_values = np.array(cr_grid["cr_values"], dtype=float)

    return m_capacity, cr_values


def attach_and_check_schema(con):
    """Attach both databases (read-only) and report the join key available."""
    con.execute(f"ATTACH '{MC_FAST_DB}' AS mc_fast (READ_ONLY)")
    con.execute(f"ATTACH '{MC_SAMPLES_DB}' AS mc_samples (READ_ONLY)")

    results_cols = {r[0] for r in con.execute("DESCRIBE mc_fast.results").fetchall()}
    samples_cols = {r[0] for r in con.execute("DESCRIBE mc_samples.samples").fetchall()}

    print("results columns:", sorted(results_cols))
    print("samples columns:", sorted(samples_cols))

    has_sample_id = "sample_id" in results_cols and "sample_id" in samples_cols
    return has_sample_id


def build_demand_cte(has_sample_id):
    """
    Per-sample demand D = |M|_max * theta_M.
    |M|_max = max(abs(x)) over the M list for that sample.
    """
    if has_sample_id:
        join_clause = """
            FROM mc_fast.results r
            JOIN mc_samples.samples s
              ON r.sample_id = s.sample_id
        """
    else:
        # Fallback: align by row order. Only safe if both tables were
        # generated with the same sample ordering (verified via row-count
        # check in main()).
        join_clause = """
            FROM (SELECT row_number() OVER () AS rn, M FROM mc_fast.results) r
            JOIN (SELECT row_number() OVER () AS rn, model_factor_M
                  FROM mc_samples.samples) s
              ON r.rn = s.rn
        """

    return f"""
        SELECT
            list_max(list_transform(r.M, x -> abs(x))) AS m_abs_max,
            s.model_factor_M AS theta_m
        {join_clause}
    """


def compute_fragility(con, demand_cte, m_capacity, cr_values):
    """
    Single pass over the joined data: one COUNT(*) FILTER per Cr grid point,
    so the whole 100M-row join is scanned once regardless of grid size.
    """
    filters = []
    for i, cr in enumerate(cr_values):
        threshold = m_capacity * (1.0 - cr)
        filters.append(
            f"COUNT(*) FILTER (WHERE m_abs_max * theta_m >= {threshold:.10g}) AS n_{i}"
        )
    filters_sql = ",\n            ".join(filters)

    full_query = f"""
        WITH demand AS (
            {demand_cte}
        )
        SELECT
            {filters_sql},
            COUNT(*) AS n_total
        FROM demand
    """
    row = con.execute(full_query).fetchone()
    n_total = row[-1]
    n_fail = np.array(row[:-1], dtype=float)
    pf = n_fail / n_total
    return pf, n_total


def main():
    m_capacity, cr_values = load_inputs()
    print(f"M_capacity = {m_capacity:.3e} N*m/m")
    print(f"Cr grid: {len(cr_values)} points in [{cr_values.min()}, {cr_values.max()}]")

    con = duckdb.connect()
    has_sample_id = attach_and_check_schema(con)

    if not has_sample_id:
        n_results = con.execute("SELECT COUNT(*) FROM mc_fast.results").fetchone()[0]
        n_samples = con.execute("SELECT COUNT(*) FROM mc_samples.samples").fetchone()[0]
        if n_results != n_samples:
            raise RuntimeError(
                f"No shared sample_id and row counts differ "
                f"({n_results} vs {n_samples}) -- cannot safely align rows."
            )
        print(
            "WARNING: no shared 'sample_id' column found in both tables; "
            "falling back to positional row alignment. Verify this matches "
            "how the two Monte Carlo runs were generated (same seed/order)."
        )

    demand_cte = build_demand_cte(has_sample_id)
    pf, n_total = compute_fragility(con, demand_cte, m_capacity, cr_values)
    print(f"Computed fragility curve from {n_total:,} samples.")

    # --- save outputs -------------------------------------------------
    out = {
        "description": "Fragility curve P(LSF_wall <= 0 | Cr) for the sheet-pile wall moment limit state",
        "n_samples": int(n_total),
        "cr": cr_values.tolist(),
        "pf": pf.tolist(),
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {OUTPUT_JSON}")

    plt.figure(figsize=(7, 5))
    plt.plot(cr_values, pf, marker="o", markersize=3, linewidth=1.5)
    plt.xlabel("Corrosion ratio, Cr [-]")
    plt.ylabel(r"$P(LSF_{wall} \leq 0)$")
    plt.title("Fragility curve — wall moment limit state")
    plt.yscale("log")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150)
    print(f"Wrote {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
