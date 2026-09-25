'''
analytical_model.py
====================================================================
This routine represents a simple anlytical model for a canal-wall 
cross-section.
Input: a realization created by the script 'generating_samples.py' 
       of the input random variables in 'engineering_parameters.json' 
Output: a matrix with Shear and Bending moment diagram for each 
        discretized point along the beam
--------------------------------------------------------------------
Main modeling hypotheses:
1) The anchor level is a ROLLER support. Everything above it is
   neglected. The anchor is considered perfectly perpendicular to 
   the beam.
2)The second constraint is an HINGE and is located at the 90% of the 
   embedded length. Everything below is neglected.
3) Between the roller (anchor) and the hinge, the wall is therefore
   a classic STATICALLY DETERMINATE simply-supported beam.
4) The fluid is in Hydrostatic condition.
5) The beam is discretized in 'n_points' and the internal actions
   are evaluated only at that level.
--------------------------------------------------------------------
WORKFLOW:
 1. Data organised in classes:      SoilLayer,Geometry...
 2. Coulomb coefficients:           coulomb_ka / coulomb_kp
 3. Earth + water pressure, each side (uniform overload on the LEFT
    / retained side + the two water levels are all included):
                                    solve_batch(), steps "pressures"
 4. Equilibrium -> the 2 external reactions (roller/anchor + hinge):
                                    solve_batch(), steps "statics"
 5. 20 equally-spaced points, roller -> hinge, shear & moment:
                                    solve_batch(), steps "V, M"
====================================================================
'''
# ---------------------------------------------------------------- #
# Import
# ---------------------------------------------------------------- #

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd 

GAMMA_WATER = 9.81  # kN/m3

# ---------------------------------------------------------------- #
# 1. DATA, ORGANISED IN CLASSES
# ---------------------------------------------------------------- #

@dataclass
class SoilLayer:
    """One homogeneous soil layer, characterised top-to-bottom."""
    name: str
    thickness: float                # m
    gamma: float                    # kN/m3, moist/bulk unit weight
    gamma_sat: float                # kN/m3, saturated unit weight
    phi_k: float                    # deg, characteristic effective friction angle
    c_k: float = 0.0                # kPa, characteristic effective cohesion
    delta_ratio: float = 2.0 / 3.0  # wall friction delta = delta_ratio * phi


@dataclass
class WaterLevels:
    level_back: float    # m, water level on retained (canal/high) side, from a common datum
    level_front: float   # m, water level on excavation/land (front) side, from same datum

@dataclass
class WallGeometry:
    top_level: float
    dredge_level: float
    anchor_level: float
    embedment_depth: float
    hinge_fraction: float = 0.9
    n_points: int = 20

    @property
    def toe_level(self) -> float:
        return self.dredge_level - self.embedment_depth

    @property
    def hinge_level(self) -> float:
        return self.dredge_level - self.hinge_fraction * self.embedment_depth
    
    @property
    def span(self) -> float:
            """Length of the statically-determinate beam, anchor -> hinge."""
            return self.anchor_level - self.hinge_level
    
@dataclass
class SectionProperties:
    """For the EN 1993-5 structural check."""
    f_yk: float            # MPa, characteristic yield strength
    W_el: float            # cm3/m, elastic section modulus per metre run



def split_stratigraphy_below(top_level: float, layers: List[SoilLayer],
                              cut_level: float) -> List[SoilLayer]:
    """
    Dividing the Wet soil to the dry soil, fundamental passage for the evaluation
    of active and passive earth pressure.
    """
    out: List[SoilLayer] = []
    z = top_level
    for layer in layers:
        layer_bot = z - layer.thickness
        if layer_bot >= cut_level:
            z = layer_bot
            continue  # entirely above the cut: not part of the front stack
        if z <= cut_level:
            out.append(layer)  # entirely below the cut: keep as-is
        else:
            thickness_below = cut_level - layer_bot
            out.append(SoilLayer(
                name=layer.name, thickness=thickness_below,
                gamma=layer.gamma, gamma_sat=layer.gamma_sat,
                phi_k=layer.phi_k, c_k=layer.c_k, delta_ratio=layer.delta_ratio,
            ))
        z = layer_bot
    if not out:
        raise ValueError(
            f"No stratigraphy remains below cut_level={cut_level} -- the "
            f"soil profile (down to {z}) does not reach that depth.")
    return out

# ---------------------------------------------------------------- #
# 2. COULOMB EARTH-PRESSURE COEFFICIENTS
# ---------------------------------------------------------------- #

def coulomb_ka(phi_deg: float, delta_deg: float, beta_deg: float = 0.0,
                theta_deg: float = 0.0) -> float:
    """Coulomb active earth pressure coefficient.
    phi   : soil friction angle [deg]
    delta : wall friction angle [deg]
    beta  : backfill slope angle from horizontal [deg]
    theta : wall inclination from vertical [deg] (0 = vertical wall)
    """
    phi, delta, beta, theta = map(np.radians, (phi_deg, delta_deg, beta_deg, theta_deg))
    num = np.cos(phi - theta) ** 2
    denom = (np.cos(theta) ** 2 * np.cos(delta + theta) *
              (1 + np.sqrt((np.sin(phi + delta) * np.sin(phi - beta)) /
                            (np.cos(delta + theta) * np.cos(theta - beta)))) ** 2)
    return num / denom


def coulomb_kp(phi_deg: float, delta_deg: float, beta_deg: float = 0.0,
                theta_deg: float = 0.0) -> float:
    """Coulomb passive earth pressure coefficient (sign convention: delta
    resists wall movement into the soil, so use -delta of the active case
    when the wall is being pushed toward the passive side).
    NB: over-predicts Kp for delta > ~ (2/3)*phi -- see module docstring.
    """
    phi, delta, beta, theta = map(np.radians, (phi_deg, delta_deg, beta_deg, theta_deg))
    num = np.cos(phi + theta) ** 2
    denom = (np.cos(theta) ** 2 * np.cos(delta - theta) *
              (1 - np.sqrt((np.sin(phi + delta) * np.sin(phi + beta)) /
                            (np.cos(delta - theta) * np.cos(theta - beta)))) ** 2)
    return num / denom

# ================================================================ #
# vectorization helpers (layers / z / samples all handled at once)
# ================================================================ #

def _layer_geometry(top_level: float, layers: List[SoilLayer]):
    t = np.array([l.thickness for l in layers])
    top = top_level - np.concatenate([[0.0], np.cumsum(t)[:-1]])
    return top, top - t


def _layer_idx(z: np.ndarray, top: np.ndarray, bot: np.ndarray) -> np.ndarray:
    """Which layer each z-point falls in, vectorized over z."""
    within = (z[None, :] <= top[:, None] + 1e-9) & (z[None, :] >= bot[:, None] - 1e-9)
    idx = np.argmax(within, axis=0)
    idx[~within.any(axis=0)] = len(top) - 1
    return idx


def _sigma_v(z, top, bot, gamma, gamma_sat, water_level):
    """(n_samples, n_z) effective vertical stress -- splits each layer into
    a dry slice (above the water table) and a saturated slice (below it)."""
    seg_bot = np.maximum(bot[:, None], z[None, :])                 # (L, Z)
    overlap = np.clip(top[:, None] - seg_bot, 0, None)              # (L, Z)
    wt = water_level[:, None, None]                                  # (N, 1, 1)
    dry = np.clip(wt, seg_bot, top[:, None]) - seg_bot                # (N, L, Z)
    sat = overlap - dry
    return (gamma[:, :, None] * dry + gamma_sat[:, :, None] * sat).sum(1)


def _gather(samples: pd.DataFrame, layers: List[SoilLayer], suffix: str) -> np.ndarray:
    return np.stack([samples[f"{l.name}_{suffix}"].to_numpy() for l in layers], axis=1)


# ---------------------------------------------------------------- #
# 3 + 4 + 5.  pressures  ->  equilibrium  ->  shear & moment
# ---------------------------------------------------------------- #

def solve_batch(samples: pd.DataFrame, geom: WallGeometry, layers: List[SoilLayer],
                 factors: None,
                 delta_ratio: float = 2.0 / 3.0) -> dict:
    """
    Vectorized, closed-form solve of the statically-determinate span.

    samples : one row per Monte Carlo sample, columns named
              "<layer.name>_soilphi/soilcohesion/soilgamdry/soilgamwet",
              plus "canal_level" (back/left water level), "phreatic_level"
              (front/right water level), "uniform_load_left" (surcharge on
              the retained/left side), and optionally "model_factor_M".
    geom    : Geometry (defines the roller-to-hinge span, see hypotheses).
    layers  : full back-side (retained/left) soil stratigraphy, top to bottom.

    Returns a dict:
      z          (n_points,)          elevations, anchor -> hinge
      V, M       (n_samples, n_points) shear force / bending moment
      R_anchor   (n_samples,)          roller reaction == the anchor force
      R_hinge    (n_samples,)          hinge reaction
    """
    factors = factors 
    n = len(samples)

    # ---- step 5: the statically-determinate span, 20 equal points ----
    z = np.linspace(geom.anchor_level, geom.hinge_level, geom.n_points)

    # ---- soil stratigraphy geometry (back = full profile from top_level,
    #      front = same profile re-based to start at dredge_level) ----
    top, bot = _layer_geometry(geom.top_level, layers)
    front_layers = split_stratigraphy_below(geom.top_level, layers, geom.dredge_level)
    ftop, fbot = _layer_geometry(geom.dredge_level, front_layers)

    phi_k, c_k, gam, gams = (_gather(samples, layers, s) for s in
                              ("soilphi", "soilcohesion", "soilgamdry", "soilgamwet"))
    phi_kf, c_kf, gamf, gamsf = (_gather(samples, front_layers, s) for s in
                                  ("soilphi", "soilcohesion", "soilgamdry", "soilgamwet"))

    wb = samples["canal_level"].to_numpy()               # water level, back/LEFT (retained) side
    wf = samples["phreatic_level"].to_numpy()              # water level, front/right (dredge) side
    q = samples["uniform_load_left"].to_numpy()              # uniform overload, LEFT-hand side
    mf = samples["model_factor_M"].to_numpy() if "model_factor_M" in samples else np.ones(n)

    idx_b = _layer_idx(z, top, bot)
    idx_f = _layer_idx(np.minimum(z, geom.dredge_level), ftop, fbot)

    # ---- step 3: earth + water pressure on EACH side -------------- #
    # back / left (retained) side: active pressure + surcharge + water
    phi_d_b = np.degrees(np.arctan(np.tan(np.radians(phi_k[:, idx_b]))))
    c_d_b = c_k[:, idx_b] 
    Ka = coulomb_ka(phi_d_b, delta_ratio * phi_d_b)
    sv_b = _sigma_v(z, top, bot, gam, gams, wb) + q[:, None]   # + left-hand overload
    p_back = np.clip(Ka * sv_b - 2 * c_d_b * np.sqrt(Ka), 0, None) + \
        GAMMA_WATER * np.clip(wb[:, None] - z[None, :], 0, None)                  # + back-side water

    # front / right (dredge) side: passive resistance + water, only below dredge level
    phi_d_f = np.degrees(np.arctan(np.tan(np.radians(phi_kf[:, idx_f])) ))
    c_d_f = c_kf[:, idx_f] 
    Kp = coulomb_kp(phi_d_f, delta_ratio * phi_d_f)
    sv_f = _sigma_v(z, ftop, fbot, gamf, gamsf, wf)
    p_front = (Kp * sv_f + 2 * c_d_f * np.sqrt(Kp)) + \
        GAMMA_WATER * np.clip(wf[:, None] - z[None, :], 0, None)                  # front-side water

    below_dredge = (z < geom.dredge_level)[None, :]
    w = p_back - np.where(below_dredge, p_front, 0.0)   # net distributed load, kN/m per m depth

    # ---- step 4: EQUILIBRIUM -> the two external reactions --------- #
    # Free-body integration of the load ALONE (no reactions yet), starting
    # from V=0, M=0 at the anchor -- this is exactly hypothesis (1),
    # "neglect what is above": the beam has no load and no internal
    # actions before the roller.
    dz = -np.diff(z)                                     # positive, z decreases anchor -> hinge
    V_free = np.concatenate([np.zeros((n, 1)),
                              np.cumsum(0.5 * (w[:, :-1] + w[:, 1:]) * dz, axis=1)], axis=1)
    M_free = np.concatenate([np.zeros((n, 1)),
                              np.cumsum(0.5 * (V_free[:, :-1] + V_free[:, 1:]) * dz, axis=1)], axis=1)

    # Sum of moments = 0 about the hinge solves the roller (anchor) reaction
    # directly -- closed form, no iteration:
    R_anchor = M_free[:, -1] / geom.span
    # Sum of forces = 0 then closes the loop for the hinge reaction:
    R_hinge = R_anchor - V_free[:, -1]

    # ---- step 5: shear & moment at the 20 points ------------------- #
    V = V_free - R_anchor[:, None]
    M = (M_free - R_anchor[:, None] * (geom.anchor_level - z)[None, :]) * mf[:, None]
    # By construction M[:, 0] == M[:, -1] == 0 -- zero moment at both
    # ends, i.e. the statically-determinate boundary condition holds
    # exactly (see the self-check in __main__ below).

    return {"z": z, "V": V, "M": M, "R_anchor": R_anchor, "R_hinge": R_hinge}
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# ---------------------------------------------------------------- #
# Example usage
# ---------------------------------------------------------------- #
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon

# --- Plotting function
def plot_beam_diagrams(z, V, M, geom, R_anchor, R_hinge, savepath=None):
    """
    Draw the statically-determinate beam (roller at the anchor, hinge at the
    bottom) together with its shear force and bending moment diagrams.

    z, V, M : (n_points,) arrays from solve_batch(...)["z"/"V"/"M"][sample]
    geom    : the Geometry used to solve that sample
    R_anchor, R_hinge : reactions for that sample (scalars)
    """
    x = geom.anchor_level - z          # distance from the roller (anchor), 0 -> L
    L = geom.span

    fig, (ax_beam, ax_v, ax_m) = plt.subplots(
        3, 1, figsize=(9, 8), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.4, 1.4], "hspace": 0.12},
    )

    # ---------------------------------------------------------------- #
    # 1) beam schematic: roller at x=0 (anchor), hinge/pin at x=L
    # ---------------------------------------------------------------- #
    ax_beam.plot([0, L], [0, 0], color="black", linewidth=3, solid_capstyle="butt")

    # roller support (triangle + circles) at the anchor, x = 0
    tri_h = 0.06 * L if L > 0 else 1
    roller_tri = Polygon([[0, 0], [-tri_h, -tri_h], [tri_h, -tri_h]],
                          closed=True, facecolor="white", edgecolor="black", zorder=5)
    ax_beam.add_patch(roller_tri)
    for dx in (-tri_h * 0.5, tri_h * 0.5):
        ax_beam.add_patch(Circle((dx, -tri_h * 1.35), tri_h * 0.28,
                                  facecolor="white", edgecolor="black", zorder=5))
    ax_beam.plot([-tri_h * 1.3, tri_h * 1.3], [-tri_h * 1.65, -tri_h * 1.65],
                 color="black", linewidth=1.2)
    ax_beam.text(0, tri_h * 1.6, "roller\n(anchor)", ha="center", va="bottom", fontsize=9)

    # pin/hinge support at the bottom, x = L
    hinge_tri = Polygon([[L, 0], [L - tri_h, -tri_h], [L + tri_h, -tri_h]],
                         closed=True, facecolor="white", edgecolor="black", zorder=5)
    ax_beam.add_patch(hinge_tri)
    hatch_y = -tri_h * 1.15
    ax_beam.plot([L - tri_h * 1.2, L + tri_h * 1.2], [hatch_y, hatch_y],
                 color="black", linewidth=1.2)
    for hx in np.linspace(L - tri_h * 1.1, L + tri_h * 1.1, 7):
        ax_beam.plot([hx, hx - tri_h * 0.35], [hatch_y, hatch_y - tri_h * 0.35],
                     color="black", linewidth=0.8)
    ax_beam.plot(L, 0, marker="o", markersize=5, markerfacecolor="white",
                 markeredgecolor="black", zorder=6)
    ax_beam.text(L, tri_h * 1.6, "hinge\n(toe cut-off)", ha="center", va="bottom", fontsize=9)

    # reaction arrows
    ax_beam.annotate("", xy=(0, tri_h * 0.9), xytext=(0, -tri_h * 2.4),
                      arrowprops=dict(arrowstyle="-|>", color="tab:red", lw=1.6))
    ax_beam.text(0, -tri_h * 2.6, f"$R_{{anchor}}$={R_anchor:.0f} kN/m",
                 ha="center", va="top", fontsize=9, color="tab:red")
    ax_beam.annotate("", xy=(L, tri_h * 0.9), xytext=(L, -tri_h * 2.4),
                      arrowprops=dict(arrowstyle="-|>", color="tab:red", lw=1.6))
    ax_beam.text(L, -tri_h * 2.6, f"$R_{{hinge}}$={R_hinge:.0f} kN/m",
                 ha="center", va="top", fontsize=9, color="tab:red")

    ax_beam.set_ylim(-tri_h * 4, tri_h * 3)
    ax_beam.axis("off")
    ax_beam.set_title("Statically-determinate span: roller (anchor) \u2192 hinge", fontsize=11)

    # ---------------------------------------------------------------- #
    # 2) shear force diagram
    # ---------------------------------------------------------------- #
    ax_v.axhline(0, color="black", linewidth=0.8)
    ax_v.plot(x, V, color="tab:blue", linewidth=1.6)
    ax_v.fill_between(x, V, 0, color="tab:blue", alpha=0.25)
    i_vmax = np.argmax(np.abs(V))
    ax_v.annotate(f"{V[i_vmax]:.0f} kN/m", xy=(x[i_vmax], V[i_vmax]),
                  xytext=(0, 10 if V[i_vmax] >= 0 else -14),
                  textcoords="offset points", ha="center", fontsize=8)
    ax_v.set_ylabel("V [kN/m]")
    ax_v.grid(alpha=0.3)

    # ---------------------------------------------------------------- #
    # 3) bending moment diagram
    # ---------------------------------------------------------------- #
    ax_m.axhline(0, color="black", linewidth=0.8)
    ax_m.plot(x, M, color="tab:orange", linewidth=1.6)
    ax_m.fill_between(x, M, 0, color="tab:orange", alpha=0.25)
    i_mmax = np.argmax(np.abs(M))
    ax_m.annotate(f"{M[i_mmax]:.0f} kNm/m", xy=(x[i_mmax], M[i_mmax]),
                  xytext=(0, 10 if M[i_mmax] >= 0 else -14),
                  textcoords="offset points", ha="center", fontsize=8)
    ax_m.set_ylabel("M [kNm/m]")
    ax_m.set_xlabel("distance from roller/anchor, x [m]")
    ax_m.grid(alpha=0.3)

    fig.suptitle("Shear force and bending moment \u2013 roller/hinge free body", y=0.995)
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig

# --- Calling the functions
if __name__ == "__main__":
    layers = [
        SoilLayer(name="Sand", thickness=6.0, gamma=17.0, gamma_sat=19.5, phi_k=32.5, c_k=0.0),
        SoilLayer(name="Clay", thickness=20.0, gamma=15.0, gamma_sat=17.5, phi_k=22.0, c_k=10.0),
    ]
    geom = WallGeometry(top_level=5.0, dredge_level=-2.0, anchor_level=4.0,
                     embedment_depth=5.287, hinge_fraction=0.9, n_points=20)

    sample = pd.DataFrame([{
        "Sand_soilphi": 32.5, "Sand_soilcohesion": 0.0, "Sand_soilgamdry": 17.0, "Sand_soilgamwet": 19.5,
        "Clay_soilphi": 22.0, "Clay_soilcohesion": 10.0, "Clay_soilgamdry": 15.0, "Clay_soilgamwet": 17.5,
        "canal_level": 4.5, "phreatic_level": 0.5, "uniform_load_left": 10.0, "model_factor_M": 1.0,
    }])

    out = solve_batch(sample, geom, layers, None, delta_ratio=2.0 / 3.0)
    z, V, M = out["z"], out["V"][0], out["M"][0]

    plot_beam_diagrams(z,V, M, geom,
                    out["R_anchor"][0], out["R_hinge"][0],
                    savepath="beam_diagrams.png")
    print(f"span (anchor -> hinge)   : {geom.span:.3f} m  "
          f"[{geom.anchor_level:.2f} m -> {geom.hinge_level:.3f} m]")
    print(f"R_anchor (roller/anchor) : {out['R_anchor'][0]:8.2f} kN/m")
    print(f"R_hinge                  : {out['R_hinge'][0]:8.2f} kN/m")
    print(f"M at anchor / hinge ends : {M[0]:.6f} / {M[-1]:.6f} kNm/m  (both must be ~0)")
    i = np.argmax(np.abs(M))
    print(f"Max |M|                  : {M[i]:.1f} kNm/m at z = {z[i]:.2f} m")
    print(f"Max |V|                  : {np.max(np.abs(V)):.1f} kN/m")
    print()
    print(pd.DataFrame({"z": z, "V": V, "M": M}).round(2).to_string(index=False))