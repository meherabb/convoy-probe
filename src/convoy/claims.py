"""The numbers stated in the text of the paper, recomputed from the run outputs.

Each :class:`Claim` holds the place in the paper, the LaTeX fragment that states the number
(``quote``), the value as the paper states it (``value``, plain text) and a function that
recomputes that value from ``results/v6.6``. ``tests/test_paper_claims.py`` checks that every
recomputed value equals the stated one; ``scripts/verify_paper_numbers.py`` prints the report and,
given the LaTeX source of the paper (``--paper main.tex``), also checks that every quote is in it.
The numbers of the tables are checked separately (:mod:`convoy.paper`, ``tests/test_paper_tables.py``).

Values are written with ASCII minus signs and the rounding of the paper.
"""
from __future__ import annotations

import csv
import math
import re
from collections.abc import Callable
from dataclasses import dataclass

from .paper import MODELS, Results, fmt
from .registered import N_BOOT_EXPLORATORY

__all__ = ["Claim", "CLAIMS", "check", "quotes_in"]


@dataclass(frozen=True)
class Claim:
    where: str
    quote: str
    value: str
    compute: Callable[[Results], str]


# ---- helpers ---------------------------------------------------------------------------------------------------------

def f(v, nd=2):
    return fmt(v, nd).replace("−", "-")


def rng(vals, nd=2, sep=" to "):
    vals = [float(v) for v in vals]
    return f"{f(min(vals), nd)}{sep}{f(max(vals), nd)}"


def ci(est, lo, hi, nd=2):
    return f"{f(est, nd)} [{f(lo, nd)}, {f(hi, nd)}]"


def pct(v):
    return f"{100 * float(v):.0f}%"


def est(R, prefix, model):
    return float(R.prediction(prefix, model)["estimate"])


def preds(R, prefix, models=MODELS):
    return {m: R.prediction(prefix, m) for m in models}


def supported(R, prefix, models=MODELS):
    return [m for m, r in preds(R, prefix, models).items() if r["verdict"].startswith(("SUPPORTED", "HOLDS"))]


def row_ci(r, e="estimate", lo="ci_lo", hi="ci_hi", nd=2):
    return ci(r[e], r[lo], r[hi], nd)


def parse_ci(s):
    """``'1.00 [0.99, 1.00]'`` -> ``(1.0, 0.99, 1.0)`` (formatted tables of the run)."""
    a, b, c = (float(x) for x in re.findall(r"-?\d+\.\d+", s))
    return a, b, c


def stage_hours(R):
    """Hours of every checkpointed stage when it was first computed (each stage counted once)."""
    seen = {}
    for r in R.rows("manifest/stage_log.csv"):
        seen.setdefault(r["stage"], float(r["original_seconds"] or 0))
    return seen


def e1(R, model):
    return R.one("results/E1_beta_s.csv", model=model)


def t3(R, model):
    return R.one("tables/T3_E1_geometry_share.csv", model=model)


def support(R, testbed, condition, model):
    return R.one("results/support_levels.csv", testbed=testbed, condition=condition, model=model)


def bsw(R, test, model):
    return R.one("results/BSW_tests.csv", test=test, model=model)


def wdr(R, test, model):
    return R.one("results/WDR_tests.csv", test=test, model=model)


def e5(R, test, model):
    return R.one("results/E5_contrasts.csv", test=test, model=model)


def e5beta(R, condition, model):
    return R.one("results/E5real_beta_by_condition.csv", condition=condition, model=model)


def by_parallax(R, model, lo):
    return R.one("results/E1_beta_by_parallax.csv", model=model, par_lo=str(lo))


def cr_bin(R, model_name, b):
    return R.one("tables/T21e_convoy_real_by_parallax.csv", model=model_name, testbed="CONVOY-REAL", bin=b)


def t1b(R, dataset):
    return R.one("tables/T1b_data.csv", dataset=dataset)


def twin_row(R, flow, regime):
    rel = "results/twin_flow_recall_by_regime.csv" if flow else "results/twin_recall_by_regime.csv"
    return R.one(rel, regime=regime)


def cfg(R, *keys):
    d = R.json("manifest/run_config.json")
    for k in keys:
        d = d[k]
    return d


def names(models):
    return ", ".join(models)


FIVE = MODELS[:5]
NAME = {"DUSt3R": "DUSt3R", "MASt3R": "MASt3R", "MonST3R": "MonST3R", "VGGT-1B": "VGGT", "pi3": "π³",
        "MapAnything": "MapAnything"}


def _argext(R, prefix, fn):
    p = preds(R, prefix)
    m = fn(p, key=lambda k: float(p[k]["estimate"]))
    return m, float(p[m]["estimate"])


# ---- the claims ------------------------------------------------------------------------------------------------------

def _claims():
    C = []
    add = lambda where, quote, value, compute: C.append(Claim(where, quote, value, compute))

    # Abstract
    add("Abstract", r"geometry shares between $-0.04$ and $0.19$", "-0.04 and 0.19",
        lambda R: rng([est(R, "P1 degeneracy (CONVOY-3D)", m) for m in MODELS], sep=" and "))
    add("Abstract", r"whereas classical triangulation gives $1.00$", "1.00",
        lambda R: f(parse_ci(t3(R, "Classical triangulation, exact correspondences")["beta_s_ci"])[0]))
    add("Abstract", r"raises the share for all six models (by $0.14$ to $0.39$)", "6 models, 0.14 to 0.39",
        lambda R: f"{len(supported(R, 'H6'))} models, " + rng([est(R, 'H6', m) for m in MODELS]))
    add("Abstract", r"a wall only 15\,cm high already raises it", "6 models",
        lambda R: f"{len(supported(R, 'H13'))} models")
    add("Abstract", r"lifting the object raises it for five of six models, also on real backgrounds", "5 and 5",
        lambda R: f"{len(supported(R, 'H5'))} and {len(supported(R, 'H11a'))}")
    add("Abstract", r"the shares stay at or below $0.27$", "0.27",
        lambda R: f(max(float(r["beta_geo"]) for r in R.rows("results/real_driving_beta.csv"))))

    # Section 3: the probe
    add("Sec. 3", r"through $0.62$ ($w=0.5$)", "0.62", lambda R: f(_design_beta(R, 0.5)))
    add("Sec. 3", r"reproduce $\beta(w)$ computed over the same movers to three decimals", "max |diff| < 5e-4",
        lambda R: _oracle_agreement(R))
    add("Sec. 3", r"($0.61$ at $w=0.5$)", "0.61",
        lambda R: f(R.one("results/oracle_validation.csv", w_true="0.5")["beta_theory_identity"]))
    add("Sec. 3", r"recovers $w$ to within $3\times10^{-4}$", "3e-4",
        lambda R: f"{math.ceil(1e4 * max(abs(float(r['median_w_recovered']) - float(r['w_true'])) for r in R.rows('results/oracle_validation.csv')))}e-4")
    add("Sec. 3", r"if $d\ge5$\,px, if it has at least 40 visible pixels after erosion, and if at least 85\% of its silhouette",
        "5 px, 40 px, 2 px erosion", lambda R: f"{cfg(R, 'PREREG', 'parallax_min_px'):.0f} px, "
        f"{cfg(R, 'PREREG', 'min_object_pixels')} px, {cfg(R, 'PREREG', 'erosion_px')} px erosion")
    add("Sec. 3", r"for objects with $|x_{\mathrm{geo}}-\rhom|\ge0.3$", "0.3",
        lambda R: f"{cfg(R, 'PREREG', 'x6_min_lever')}")
    add("Sec. 3", r"\emph{parallel} if $|\varepsilon|<1$\,px and \emph{perpendicular} if $|\varepsilon|\ge3$\,px", "1 px, 3 px",
        lambda R: f"{cfg(R, 'PREREG', 'eps_parallel_max'):.0f} px, {cfg(R, 'PREREG', 'eps_perp_min'):.0f} px")

    # Section 4: experimental design
    add("Sec. 4 Models", r"MonST3R~\cite{zhang2025monst3r} at $512\times384$ in fp32", "512x384 fp32 (3 models)",
        lambda R: "; ".join(sorted({f"{r['input']} {r['precision']}" for r in R.rows("tables/T1a_models.csv")
                                    if r["family"] == "croco"})) + f" ({sum(r['family'] == 'croco' for r in R.rows('tables/T1a_models.csv'))} models)")
    add("Sec. 4 Models", r"VGGT-1B~\cite{wang2025vggt} ($504\times378$, fp16), $\pi^3$~\cite{wang2026pi3} ($560\times420$, fp16) and MapAnything~\cite{keetha2025mapanything} ($504\times378$, fp32)",
        "504x378 fp16, 560x420 fp16, 504x378 fp32",
        lambda R: ", ".join(f"{R.one('tables/T1a_models.csv', model=m)['input']} {R.one('tables/T1a_models.csv', model=m)['precision']}"
                            for m in ("VGGT", "π³", "MapAnything")))
    add("Sec. 4 Models", r"MASt3R and VGGT list Virtual KITTI among their training data", "MASt3R, VGGT",
        lambda R: names(r["model"] for r in R.rows("tables/T1a_models.csv") if r["virtual_kitti_in_training"] == "yes"))
    add("Sec. 4 Testbeds", r"Its main set has 240 scenes with 715 movers", "240 scenes, 715 movers",
        lambda R: f"{t1b(R, 'CONVOY-3D main (+ knockout)')['items']} scenes, {t1b(R, 'CONVOY-3D main (+ knockout)')['movers']} movers")
    add("Sec. 4 Testbeds", r"one to four per scene", "1 to 4",
        lambda R: (lambda c: f"{min(c.values())} to {max(c.values())}")(_movers_per_scene(R)))
    add("Sec. 4 Testbeds", r"(focal length 520--820\,px, height 1.3--1.8\,m, baseline 0.5--3\,m)", "520-820 px, 1.3-1.8 m, 0.5-3 m",
        lambda R: _camera_ranges(R))
    add("Sec. 4 Testbeds", r"with movers at 5--45\,m and $s\in[-0.6,0.85]$", "5-45 m, [-0.6, 0.85]",
        lambda R: "{:.0f}-{:.0f} m, [{}, {}]".format(*cfg(R, "MOVER_DESIGN", "depth_range_m"), *cfg(R, "MOVER_DESIGN", "s_range_main")))
    add("Sec. 4 Testbeds", r"(8\% at $s\in[1.1,1.4]$, without a static explanation)", "8%, [1.1, 1.4]", lambda R: _beyond(R))
    add("Sec. 4 Testbeds", r"\emph{E5} renders 72 base scenes in five conditions", "72 scenes, 5 conditions",
        lambda R: f"{cfg(R, 'RUN_CONFIG', 'n_scenes_ablation')} scenes, {len(cfg(R, 'ABLATION_DESIGN', 'conditions'))} conditions")
    add("Sec. 4 Testbeds", r"lifted by 0.8--1.6\,m (F)", "0.8-1.6 m", lambda R: "{}-{} m".format(*cfg(R, "ABLATION_DESIGN", "lift_m")))
    add("Sec. 4 Testbeds", r"renders 64 scenes with the second camera at 0.5, 1, 2 and 3 times the baseline", "64 scenes; 0.5, 1, 2, 3",
        lambda R: f"{cfg(R, 'RUN_CONFIG', 'n_scenes_bsw')} scenes; " + ", ".join(f"{v:g}" for v in cfg(R, "SWEEP_DESIGN", "levels").values()))
    add("Sec. 4 Testbeds", r"set renders 64 scenes without a wall and behind walls that hide at least the lowest 15, 35 and 70\,cm",
        "64 scenes; 15, 35, 70 cm", lambda R: f"{cfg(R, 'RUN_CONFIG', 'n_scenes_wdr')} scenes; "
        + ", ".join(f"{100 * v:.0f}" for v in cfg(R, "WALL_DOSE_DESIGN", "levels").values() if v > 0) + " cm")
    add("Sec. 4 Testbeds", r"or is lifted by 0.25--0.50\,m", "0.25-0.50 m", lambda R: "{:.2f}-{:.2f} m".format(*cfg(R, "RABL_DESIGN", "lift_m")))
    add("Sec. 4 Testbeds", r"vehicle poses (five scenes)", "5 scenes", lambda R: f"{len(cfg(R, 'VK_DESIGN', 'scenes'))} scenes")
    add("Sec. 4 Testbeds", r"front-camera pairs from 64 validation logs with a 1--3\,m baseline", "64 val logs, 1-3 m, ring_front_center",
        lambda R: f"{R.one('results/real_driving_prevalence.csv')['n_logs']} {cfg(R, 'AV2_DESIGN', 'split')} logs, "
        + "{:.0f}-{:.0f} m, ".format(*cfg(R, "AV2_DESIGN", "pair_baseline_m")) + cfg(R, "AV2_DESIGN", "camera"))
    add("Sec. 4 Testbeds", r"agrees with the measured depth to 0.6--1.4\% (TUM) and to a median of 0.3\% (Argoverse~2)", "0.6-1.4%, 0.3%",
        lambda R: _tum_check(R) + ", " + re.search(r"median (\d\.\d+)", t1b(R, "Argoverse 2 (64 validation logs)")["check"]).group(1).replace("0.003", "0.3%"))
    add("Sec. 4 Gate", r"a log-depth slope of the static pixels in $[0.7,1.3]$ and an AbsRel of at most $0.25$", "[0.7, 1.3], 0.25",
        lambda R: "[{}, {}], {}".format(*cfg(R, "PREREG", "gate_slope_range"), cfg(R, "PREREG", "gate_absrel_max")))
    add("Sec. 4 Gate", r"All six models pass everywhere (static AbsRel 0.01--0.13)", "all PASS, 0.01-0.13",
        lambda R: _gate(R, flat=False))
    add("Sec. 4 Gate", r"fails for all six (slope near $0$, AbsRel 0.38--0.43)", "all FAIL, |slope| <= 0.01, 0.38-0.43",
        lambda R: _gate(R, flat=True))
    add("Sec. 4 Statistics", r"with 10{,}000 resamples", "10000", lambda R: f"{cfg(R, 'RUN_CONFIG', 'n_boot')}")
    add("Sec. 4 Statistics", r"Exploratory medians use 2{,}000 resamples", "2000", lambda R: f"{N_BOOT_EXPLORATORY}")
    add("Sec. 4 Statistics", r"bootstrap $p$-values across models are below $0.02$ in every such case", "max 0.013 < 0.02",
        lambda R: (lambda p: f"max {f(p, 3)} < 0.02" if p < 0.02 else f"max {p}")(_max_p_supported(R)))
    add("Sec. 4 Implementation", r"A single NVIDIA T4 GPU, pinned commits of the official code (MonST3R runs on the DUSt3R code base)",
        "Tesla T4; MonST3R croco family", lambda R: f"{R.json('logs/env_log.json')['gpu']}; MonST3R "
        f"{R.json('manifest/models.json')['MonST3R']['family']} family")
    add("Sec. 4 Implementation", r"a repeated forward pass gives bit-identical outputs", "0.0 and 0.0",
        lambda R: (lambda d: f"{d['fp32_max_abs_diff']} and {d['amp_max_abs_diff']}")(R.json("manifest/env_log.json")["determinism_check"]))
    add("Sec. 4 Implementation", r"and all stages took about 10\,h (supplement)", "10 h",
        lambda R: f"{sum(stage_hours(R).values()) / 3600:.0f} h")
    add("Sec. 4 Implementation", r"(median end-point error 0.26\,px on static CONVOY-3D pixels)", "0.26",
        lambda R: f(R.json("manifest/flow.json")["epe_static_convoy3d_median"]))

    # Section 5.1
    add("Sec. 5.1", r"for 220 purely parallel, identifiable movers in 159 CONVOY-3D scenes", "220, 159",
        lambda R: f"{e1(R, 'DUSt3R')['n']}, {e1(R, 'DUSt3R')['n_clips']}")
    add("Sec. 5.1", r"gives $1.00$ with exact correspondences and $1.00$ $[0.99,1.00]$ with RAFT flow", "1.00; 1.00 [0.99, 1.00]",
        lambda R: f(parse_ci(t3(R, "Classical triangulation, exact correspondences")["beta_s_ci"])[0]) + "; "
        + ci(*parse_ci(t3(R, "Classical triangulation, RAFT-large flow")["beta_s_ci"])))
    add("Sec. 5.1", r"between $0.09$ (VGGT, $[0.00,0.19]$) and $0.19$ (MASt3R, $[0.09,0.30]$)",
        "VGGT 0.09 [0.00, 0.19]; MASt3R 0.19 [0.09, 0.30]",
        lambda R: (lambda lo, hi: f"{NAME[lo]} {row_ci(e1(R, lo), 'beta_s')}; {NAME[hi]} {row_ci(e1(R, hi), 'beta_s')}")(
            min(FIVE, key=lambda m: float(e1(R, m)["beta_s"])), max(FIVE, key=lambda m: float(e1(R, m)["beta_s"]))))
    add("Sec. 5.1", r"MapAnything gives $-0.04$ $[-0.15,0.08]$", "-0.04 [-0.15, 0.08]",
        lambda R: row_ci(e1(R, "MapAnything"), "beta_s"))
    add("Sec. 5.1", r"correspond to prior weights between $0.89$ and $1$", "0.89 to 1",
        lambda R: (lambda w: f"{f(min(w))} to {max(w):g}")([float(e1(R, m)["w_eff"]) for m in MODELS]))
    add("Sec. 5.1", r"The locked prediction that the models inherit the degeneracy ($\bs\in[0.8,1.2]$) fails for all six",
        "6 FALSIFIED", lambda R: f"{sum(r['verdict'] == 'FALSIFIED' for r in preds(R, 'P1 degeneracy (CONVOY-3D)').values())} FALSIFIED")
    add("Sec. 5.1", r"Their single-image predictions have slopes between $-0.06$ and $0.02$", "-0.06 to 0.02",
        lambda R: rng([e1(R, m)["beta_mono"] for m in MODELS]))
    add("Sec. 5.1", r"all with intervals that include zero", "6 of 6 include zero",
        lambda R: f"{sum(float(e1(R, m)['beta_mono_lo']) <= 0 <= float(e1(R, m)['beta_mono_hi']) for m in MODELS)} of 6 include zero")
    add("Sec. 5.1", r"The 89 A1 movers have higher full-pair slopes ($0.05$ to $0.33$) and higher single-image slopes ($0.01$ to $0.15$) than E1",
        "89; 0.05 to 0.33; 0.01 to 0.15; higher for 6 of 6",
        lambda R: f"{a1(R, 'DUSt3R')['n']}; " + rng([a1(R, m)["beta_full"] for m in MODELS]) + "; "
        + rng([a1(R, m)["beta_mono"] for m in MODELS]) + "; higher for "
        + f"{sum(float(a1(R, m)['beta_full']) > float(e1(R, m)['beta_s']) and float(a1(R, m)['beta_mono']) > float(e1(R, m)['beta_mono']) for m in MODELS)} of 6")
    add("Sec. 5.1", r"With the mover deleted, no model stays more than $0.04$ above its single-image slope", "max 0.03 <= 0.04",
        lambda R: _at_most(max(float(a1(R, m)["beta_knockout"]) - float(a1(R, m)["beta_mono"]) for m in MODELS), 0.04))
    add("Sec. 5.1", r"agrees between the two samples to within $0.04$ for every model", "max 0.03 <= 0.04",
        lambda R: _at_most(max(abs(_excess_e1(R, m) - _excess_a1(R, m)) for m in MODELS), 0.04))
    add("Sec. 5.1", r"lowers $\bs$ by $0.07$ to $0.26$ for five models", "5 models, 0.07 to 0.26",
        lambda R: (lambda s: f"{len(s)} models, " + rng([est(R, 'A1 correspondence', m) for m in s]))(supported(R, "A1 correspondence")))
    add("Sec. 5.1", r"only MapAnything shows no dependence ($0.04$ $[-0.01,0.10]$)", "MapAnything 0.04 [-0.01, 0.10]",
        lambda R: (lambda m: f"{names(m)} " + row_ci(R.prediction("A1 correspondence", m[0])))(
            [m for m in MODELS if m not in supported(R, "A1 correspondence")]))
    add("Sec. 5.1", r"reach $0.55$ $[0.37,0.75]$ and $0.61$ $[0.37,0.89]$ for the 40 movers with more than 40\,px of parallax",
        "0.55 [0.37, 0.75] and 0.61 [0.37, 0.89], n = 40",
        lambda R: f"{row_ci(by_parallax(R, 'DUSt3R', 40), 'beta')} and {row_ci(by_parallax(R, 'MASt3R', 40), 'beta')}, "
        f"n = {by_parallax(R, 'DUSt3R', 40)['n']}")
    add("Sec. 5.1", r"the four other models stay at or below $0.15$ in every identifiable parallax bin", "max 0.15",
        lambda R: "max " + f(max(float(r["beta"]) for r in R.rows("results/E1_beta_by_parallax.csv")
                                 if r["model"] not in ("DUSt3R", "MASt3R") and float(r["par_lo"]) >= 5)))
    add("Sec. 5.1", r"Cars and crates standing on the road receive shares between $-0.01$ and $0.08$", "-0.01 to 0.08",
        lambda R: rng([support(R, "CONVOY-3D", "cars and crates", m)["estimate"] for m in MODELS]))
    add("Sec. 5.1", r"floating crates receive between $0.27$ and $0.53$ from five models ($-0.09$ for MapAnything)",
        "0.27 to 0.53; MapAnything -0.09",
        lambda R: rng([support(R, "CONVOY-3D", "floating crates", m)["estimate"] for m in FIVE]) + "; MapAnything "
        + f(support(R, "CONVOY-3D", "floating crates", "MapAnything")["estimate"]))

    # Section 5.2
    add("Sec. 5.2", r"raises $\bs$ for all six models, by $0.14$ (MapAnything) to $0.39$ (MASt3R)", "MapAnything 0.14 to MASt3R 0.39",
        lambda R: (lambda a, b: f"{NAME[a[0]]} {f(a[1])} to {NAME[b[0]]} {f(b[1])}")(_argext(R, "H6", min), _argext(R, "H6", max)))
    add("Sec. 5.2", r"Lifting the object off the ground (H5) raises it for five models, by $0.11$ to $0.31$", "5 models, 0.11 to 0.31",
        lambda R: (lambda s: f"{len(s)} models, " + rng([est(R, 'H5', m) for m in s]))(supported(R, "H5")))
    add("Sec. 5.2", r"for MapAnything the interval includes zero ($0.08$ $[-0.04,0.21]$)", "0.08 [-0.04, 0.21]",
        lambda R: row_ci(R.prediction("H5", "MapAnything")))
    add("Sec. 5.2", r"(H7) for four models; for VGGT and $\pi^3$ the interaction is positive but its interval includes zero",
        "4 models; VGGT, π³ positive, not supported",
        lambda R: (lambda s, o: f"{len(s)} models; {', '.join(NAME[m] for m in o)} "
                   + ("positive, not supported" if all(est(R, 'H7', m) > 0 and float(R.prediction('H7', m)['ci_lo']) < 0 for m in o) else "check"))(
            supported(R, "H7"), [m for m in MODELS if m not in supported(R, "H7")]))
    add("Sec. 5.2", r"Removing only the contact shadow has a small effect ($0.003$ to $0.07$, exploratory)", "0.003 to 0.07",
        lambda R: (lambda v: f"{f(min(v), 3)} to {f(max(v))}")([float(e5(R, "SH", m)["estimate"]) for m in MODELS]))
    add("Sec. 5.2", r"places a lifted object $1.8$ to $2.1$ times farther", "1.8 to 2.1",
        lambda R: (lambda v: f"{math.exp(min(v)):.1f} to {math.exp(max(v)):.1f}")([est(R, "H8", m) for m in MODELS]))
    add("Sec. 5.2", r"(H8, $0.57$ to $0.76$ in log depth)", "0.57 to 0.76", lambda R: rng([est(R, "H8", m) for m in MODELS]))
    add("Sec. 5.2", r"inserted into 80 real image pairs, once on the real support", "80 pairs, 141 crates",
        lambda R: (lambda r: f"{r['n_pairs']} pairs, {r['n']} crates")(R.one("results/E5real_contrasts.csv", model="DUSt3R", test="H11a")))
    add("Sec. 5.2", r"Lifting raises $\bs$ for five models, by $0.20$ to $0.30$ (H11a)", "5 models, 0.20 to 0.30",
        lambda R: (lambda s: f"{len(s)} models, " + rng([est(R, 'H11a', m) for m in s]))(supported(R, "H11a")))
    add("Sec. 5.2", r"from shares of $0.03$--$0.21$ on the support to $0.27$--$0.51$ in the air", "0.03-0.21 to 0.27-0.51",
        lambda R: (lambda s: rng([e5beta(R, "G", m)["beta"] for m in s], sep="-") + " to "
                   + rng([e5beta(R, "F", m)["beta"] for m in s], sep="-"))(supported(R, "H11a")))
    add("Sec. 5.2", r"The exception is $\pi^3$ ($-0.06$ $[-0.31,0.16]$)", "pi3 -0.06 [-0.31, 0.16]",
        lambda R: (lambda o: f"{names(o)} " + row_ci(R.prediction("H11a", o[0])))([m for m in MODELS if m not in supported(R, "H11a")]))
    add("Sec. 5.2", r"moves the wrong way ($-0.46$, contradicting H11b)", "-0.46 CONTRADICTED",
        lambda R: f"{f(est(R, 'H11b', 'pi3'))} {R.prediction('H11b', 'pi3')['verdict']}")
    add("Sec. 5.2", r"VGGT's single-image depth does not shift ($0.00$ $[-0.05,0.05]$)", "0.00 [-0.05, 0.05]",
        lambda R: row_ci(R.prediction("H11b", "VGGT-1B")))
    add("Sec. 5.2", r"but only a median 21\% of the mover", "21%",
        lambda R: pct(R.one("tables/T20a_dose_response.csv", wall="15 cm")["share hidden"]))
    add("Sec. 5.2", r"(H13, 145 movers; $0.15$ to $0.43$", "145 movers, 6 models, 0.15 to 0.43",
        lambda R: f"{wdr(R, 'H13', 'DUSt3R')['n']} movers, {len(supported(R, 'H13'))} models, " + rng([est(R, 'H13', m) for m in MODELS]))
    add("Sec. 5.2", r"(ratios $0.90$, $1.03$ and $1.19$; exploratory)", "0.90, 1.03, 1.19",
        lambda R: ", ".join(f(wdr(R, "dose share", m)["estimate"]) for m in ("DUSt3R", "MASt3R", "MapAnything")))
    add("Sec. 5.2", r"(ratios $0.57$, $0.63$ and $0.41$)", "0.57, 0.63, 0.41",
        lambda R: ", ".join(f(wdr(R, "dose share", m)["estimate"]) for m in ("MonST3R", "VGGT-1B", "pi3")))

    # Section 5.3
    add("Sec. 5.3", r"it holds for both ($0.10$ $[0.01,0.19]$ and $0.09$ $[0.04,0.14]$, H12)", "0.10 [0.01, 0.19] and 0.09 [0.04, 0.14]",
        lambda R: f"{row_ci(bsw(R, 'H12', 'DUSt3R'))} and {row_ci(bsw(R, 'H12', 'MASt3R'))}")
    add("Sec. 5.3", r"the other four models rise by $0.04$ to $0.11$", "0.04 to 0.11",
        lambda R: rng([bsw(R, "H12", m)["estimate"] for m in MODELS[2:]]))
    add("Sec. 5.3", r"their share grows by $0.09$ (VGGT) to $0.24$ (MASt3R)", "VGGT 0.09 to MASt3R 0.24",
        lambda R: (lambda v: (lambda a, b: f"{NAME[a]} {f(v[a])} to {NAME[b]} {f(v[b])}")(min(v, key=v.get), max(v, key=v.get)))(
            {m: float(bsw(R, "H12-floating", m)["estimate"]) for m in MODELS}))
    add("Sec. 5.3", r"while grounded movers change by $-0.01$ to $0.03$", "-0.01 to 0.03",
        lambda R: rng([bsw(R, "H12-grounded", m)["estimate"] for m in MODELS]))
    add("Sec. 5.3", r"the difference between the two rises excludes zero for every model except DUSt3R", "all but DUSt3R",
        lambda R: "all but " + names(NAME[m] for m in MODELS if float(bsw(R, "H12-interaction", m)["ci_lo"]) <= 0))

    # Section 5.4
    add("Sec. 5.4", r"(CONVOY-REAL; 108 movers, 80 pairs) receive shares of $0.21$ ($\pi^3$) to $0.74$ (MASt3R)",
        "108, 80; pi3 0.21 to MASt3R 0.74",
        lambda R: (lambda rs: f"{rs[0]['n']}, {rs[0]['n_pairs']}; " + (lambda lo, hi: f"{lo['model']} {f(lo['beta_s'])} to {hi['model']} {f(hi['beta_s'])}")(
            min(rs, key=lambda r: float(r["beta_s"])), max(rs, key=lambda r: float(r["beta_s"]))))(
            [r for r in R.rows("results/real_E1.csv") if r["model"] in MODELS]))
    add("Sec. 5.4", r"(median 74 against 16\,px)", "74 vs 16",
        lambda R: f"{float(cr_bin(R, 'DUSt3R', 'all')['median_parallax']):.0f} vs "
        f"{float(R.one('tables/T21e_convoy_real_by_parallax.csv', model='DUSt3R', testbed='CONVOY-3D floating crates', bin='all')['median_parallax']):.0f}")
    add("Sec. 5.4", r"above 80\,px all six models reach $0.64$ to $0.85$", "0.64 to 0.85",
        lambda R: rng([parse_ci(cr_bin(R, NAME[m], ">80 px")["beta_ci"])[0] for m in MODELS]))
    add("Sec. 5.4", r"over 72 identifiable parallel vehicles (54 tracks)", "72, 54",
        lambda R: (lambda r: f"{r['n']}, {r['n_tracks']}")(R.one("results/real_driving_beta.csv", model="DUSt3R")))
    add("Sec. 5.4", r"is $0.17$ $[0.11,0.28]$ for DUSt3R, $0.27$ $[0.15,0.46]$ for MASt3R and between $0.00$ and $0.06$ for the other four models",
        "0.17 [0.11, 0.28]; 0.27 [0.15, 0.46]; 0.00 to 0.06",
        lambda R: "; ".join(row_ci(R.one("results/real_driving_beta.csv", model=m), "beta_geo") for m in ("DUSt3R", "MASt3R"))
        + "; " + rng([R.one("results/real_driving_beta.csv", model=m)["beta_geo"] for m in MODELS[2:]]))
    add("Sec. 5.4", r"but only 16 to 18 vehicles with hidden contact enter that comparison", "16 to 18",
        lambda R: (lambda v: f"{min(v)} to {max(v)}")([int(r["n_hidden"]) for r in R.select("results/X6_contact_test.csv", dataset="Argoverse 2")]))

    # Section 6.1
    add("Sec. 6.1", r"(340 parallel and 253 perpendicular movers)", "340, 253",
        lambda R: (lambda r: f"{r['n_par']}, {r['n_perp']}")(R.rows("results/E2_masks.csv")[0]))
    add("Sec. 6.1", r"the changes in $\bs$ and in M2 recall have a Spearman correlation of $-0.71$", "60 changes, -0.71",
        lambda R: _spearman_tradeoff(R))
    add("Sec. 6.1", r"hiding the contact lowers M2 recall for five models (by $0.04$ to $0.26$)", "5 models, 0.04 to 0.26",
        lambda R: (lambda v: f"{len(v)} models, " + rng(v))(
            [-float(r["delta_recall"]) for r in R.select("results/mask_tradeoff.csv", source="E5", level="CH") if float(r["delta_recall"]) < 0]))

    # Section 6.2
    add("Sec. 6.2", r"77\% of the moving-vehicle pixels lie within 1\,px", "77%",
        lambda R: pct(_pooled(R.select("results/prevalence_two_view.csv", dataset="Virtual KITTI 2 (vehicles)"))))
    add("Sec. 6.2", r"of 83\%, 69\% and 50\% of the moving-vehicle tracks", "83%, 69%, 50%",
        lambda R: ", ".join(pct(r["share_static_fit_below_1px"]) for r in R.rows("results/prevalence_kview_vkitti.csv")))
    add("Sec. 6.2", r"57\% of 1{,}304 measurable moving vehicles are in the parallel regime and 16\% have $0.85\le s\le1.15$",
        "57% of 1304, 16%, [0.85, 1.15]",
        lambda R: (lambda r: f"{pct(r['frac_parallel'])} of {r['n_vehicles']}, {pct(r['frac_keeping_pace'])}, ")(
            R.one("results/real_driving_prevalence.csv")) + "[{}, {}]".format(*cfg(R, "PREREG", "x6_pace_s")))
    add("Sec. 6.2", r"keeps pace with the camera ($0.85\le s<1$) and moves exactly in parallel", "0.85",
        lambda R: f"{cfg(R, 'PREREG', 'x6_pace_s')[0]}")
    add("Sec. 6.2", r"for walking people in TUM the share is 14\% of moving pixels", "14%",
        lambda R: pct(_pooled(R.rows("results/prevalence_tum.csv"))))
    add("Sec. 6.2", r"the geometric answer lies at least $6.7$ times too far", "6.7",
        lambda R: f"{1 / (1 - cfg(R, 'PREREG', 'x6_pace_s')[0]):.1f}")
    add("Sec. 6.2", r"is between $-0.004$ and $0.009$ for all six models", "-0.004 to 0.009",
        lambda R: rng([r["median_g"] for r in R.select("results/X6_pace.csv", set="AV2")], nd=3))
    add("Sec. 6.2", r"$97$--$100\%$ of them are placed closer to the single-image depth", "97-100%",
        lambda R: (lambda v: f"{100 * min(v):.0f}-{100 * max(v):.0f}%")(
            [float(r["share_closer_to_single_image"]) for r in R.select("results/X6_pace.csv", set="AV2")]))
    add("Sec. 6.2", r"(27--30 vehicles in 13--15 tracks; exploratory)", "27-30, 13-15",
        lambda R: (lambda rs: "{}-{}, {}-{}".format(min(int(r["n_s_below_1"]) for r in rs), max(int(r["n_s_below_1"]) for r in rs),
                                                    min(int(r["n_tracks"]) for r in rs), max(int(r["n_tracks"]) for r in rs)))(
            R.select("results/X6_pace.csv", set="AV2")))

    # Section 6.3
    TW = lambda R: R.json("results/twin_thresholds.json")
    add("Sec. 6.3", r"On Argoverse~2, 625 of the 1{,}304 moving vehicles triangulate behind the camera and have no static twin",
        "625 of 1304", lambda R: f"{TW(R)['n_no_static_solution']} of {TW(R)['n_moving']}")
    add("Sec. 6.3", r"For the other 679 we compute the twin's contact height", "679 = all regimes",
        lambda R: (lambda n: f"{n} = all regimes" if str(n) == twin_row(R, False, "all")["n"] else f"{n}")(
            TW(R)["n_moving"] - TW(R)["n_no_static_solution"]))
    add("Tab. 3 caption", r"With labels, epipolar recall outside the parallel regime is 1 by construction", "1.00, 1.00; 0.54 px < 1 px",
        lambda R: f"{f(twin_row(R, False, 'intermediate')['det_epipolar'])}, {f(twin_row(R, False, 'perpendicular')['det_epipolar'])}; "
        f"{f(TW(R)['eps_px'])} px < {cfg(R, 'PREREG', 'eps_parallel_max'):.0f} px")
    add("Sec. 6.3", r"95th percentile over 1{,}375 parked vehicles ($0.30$\,m)", "1375, 0.30 m",
        lambda R: (lambda t: f"{t['n_parked']}, {f(t['height_m'])} m")(R.json("results/twin_thresholds.json")))
    add("Sec. 6.3", r"95th percentile over the parked vehicles ($0.54$\,px)", "0.54 px",
        lambda R: f"{f(R.json('results/twin_thresholds.json')['eps_px'])} px")
    add("Sec. 6.3", r"the twin test flags $0.91$ of the parallel vehicles against $0.25$ for the epipolar test", "0.91 vs 0.25",
        lambda R: (lambda r: f"{f(r['det_twin'])} vs {f(r['det_epipolar'])}")(twin_row(R, False, "parallel")))
    add("Sec. 6.3", r"it recovers $0.95$ of the parallel vehicles that the epipolar test misses", "0.95",
        lambda R: f(twin_row(R, False, "parallel")["twin_given_epi_miss"]))
    add("Sec. 6.3", r"a plane was found in all 669 pairs", "100% of 669",
        lambda R: (lambda s: f"{pct(s['roads']['found_share'])} of {s['n_pairs']}")(R.json("manifest/static_twin_flow.json")["status"]))
    add("Sec. 6.3", r"5\% false positives on 1{,}372 parked vehicles", "1372",
        lambda R: f"{R.json('results/twin_flow_thresholds.json')['thresholds']['n_parked']}")
    add("Sec. 6.3", r"the twin test flags $0.29$ $[0.22,0.36]$ of the moving vehicles and the epipolar test $0.08$ $[0.05,0.10]$",
        "0.29 [0.22, 0.36]; 0.08 [0.05, 0.10]",
        lambda R: (lambda r: f"{row_ci(r, 'det_twin', 'det_twin_lo', 'det_twin_hi')}; "
                   f"{row_ci(r, 'det_epipolar', 'det_epipolar_lo', 'det_epipolar_hi')}")(twin_row(R, True, "parallel")))
    add("Sec. 6.3", r"the registered difference, $0.21$ $[0.13,0.29]$ over 543 vehicles in 214 tracks, is supported",
        "0.21 [0.13, 0.29], 543, 214, SUPPORTED",
        lambda R: (lambda h: f"{ci(h['estimate'], h['ci_lo'], h['ci_hi'])}, {h['n']}, {h['n_tracks']}, {h['verdict']}")(
            R.json("results/twin_flow_thresholds.json")["H14"]))
    add("Sec. 6.3", r"(perpendicular regime: $0.09$ against $0.93$)", "0.09 vs 0.93",
        lambda R: (lambda r: f"{f(r['det_twin'])} vs {f(r['det_epipolar'])}")(twin_row(R, True, "perpendicular")))

    # Section 7
    add("Sec. 7", r"(difference in H6, $0.04$ $[-0.01,0.10]$)", "0.04 [-0.01, 0.10]", lambda R: ci(*parse_ci(_t21d(R, "H6")["difference_ci"])))
    add("Sec. 7", r"($0.10$ against $0.55$, difference $-0.45$ $[-0.71,-0.26]$)", "0.10 vs 0.55, -0.45 [-0.71, -0.26], n = 40",
        lambda R: (lambda r: f"{f(r['MonST3R'])} vs {f(r['DUSt3R'])}, {ci(*parse_ci(r['difference_ci']))}, n = {r['n']}")(
            _t21d(R, "beta_s, CONVOY-3D movers with parallax > 40")))
    add("Sec. 7", r"($R^2\le0.10$ on CONVOY-3D)", "max 0.10", lambda R: "max " + f(max(float(e1(R, m)["r2"]) for m in MODELS)))
    add("Sec. 7", r"leave the share of the multi-view models between $0.09$ and $0.27$", "0.09 to 0.27",
        lambda R: rng([v for r in R.rows("results/E4_summary.csv") for k, v in r.items() if k.startswith(("straight_beta", "curved_beta"))]))
    add("Sec. 7", r"Of the registered predictions, P1, P3a and P4 failed", "P1 6/6, P3a 5/6 (1 partial), P4 3/3 falsified",
        lambda R: _failed_summary(R))
    add("Sec. 7", r"up to eight views on straight and curved paths", "K = 2, 4, 8",
        lambda R: "K = " + ", ".join(str(k) for k in cfg(R, "RUN_CONFIG", "e4_window_sizes")))
    add("Sec. 7", r"on 10-frame synthetic clips detected movers in neither regime at 5\% false positives",
        "10-frame clips; perpendicular recall 0.17 and 0.07 at 5% FPR",
        lambda R: ("10-frame clips; perpendicular recall 0.17 and 0.07 at 5% FPR"
                   if all(t in R.json("manifest/amendment_v6_1.json")["withdrawn"]["H10 (Easi3R on 10-frame clips)"]
                          for t in ("recall 0.17 and 0.07 at a 5% static false-positive rate", "detects movers in neither regime"))
                   else "check"))
    add("Sec. 7", r"P2 held only in part", "8 of 20 supported",
        lambda R: (lambda rs: f"{sum(r['verdict'] == 'SUPPORTED' for r in rs)} of {len(rs)} supported")(
            [r for r in R.rows("results/prediction_summary.csv") if r["prediction"].startswith("P2")]))
    add("Sec. 7", r"H11b is contradicted for $\pi^3$", "pi3 CONTRADICTED",
        lambda R: "; ".join(f"{m} {r['verdict']}" for m, r in preds(R, "H11b").items() if r["verdict"] == "CONTRADICTED"))
    add("Sec. 7", r"Too few objects met the sample rules to test H9 on Virtual KITTI~2 vehicles", "6 < 20",
        lambda R: f"{_h9_steps(R)[-1][0]} < {cfg(R, 'PREREG', 'h9_min_movers')}")
    add("Sec. 7", r"Bonn RGB-D Dynamic~\cite{palazzolo2019refusion} (two to nine per model)", "2 to 9",
        lambda R: (lambda c: f"{min(c.values())} to {max(c.values())}")(_bonn_counts(R)))

    # Supplement A: numerical checks
    P = lambda R: R.json("logs/proposition_checks.json")
    add("Supp. A", r"parallel movers have $|\varepsilon|<5\times10^{-10}$\,px", "< 5e-10 px",
        lambda R: "< 5e-10 px" if R.json("logs/testbed_checks.json")["parallel_movers_max_pixel_eps_px"] < 5e-10 else "check")
    add("Supp. A", r"agree with \cref{prop:parallel} to $3\times10^{-11}$ in log depth", "< 3e-11",
        lambda R: "< 3e-11" if R.json("logs/testbed_checks.json")["prop1_rendered_max_abs_dev_log_depth"] < 3e-11 else "check")
    add("Supp. A", r"Over 9{,}915 random configurations", "9915", lambda R: f"{P(R)['P1_n_trials']}")
    add("Supp. A", r"differ by at most $9.6\times10^{-9}$\,px", "9.6e-09", lambda R: f"{P(R)['P1_max_reprojection_diff_px']:.1e}")
    add("Supp. A", r"to a relative error of $4.9\times10^{-8}$", "4.9e-08",
        lambda R: f"{P(R)['P1_max_rel_error_triangulated_vs_Z_over_1_minus_s']:.1e}")
    add("Supp. A", r"for $s>1$ every configuration triangulates behind the camera", "1.0",
        lambda R: f"{P(R)['P1_fraction_s_gt_1_triangulating_behind_camera']}")
    add("Supp. A", r"Over 20{,}000 configurations, \cref{eq:twin} holds to $8.9\times10^{-14}$\,m", "20000, 8.9e-14",
        lambda R: f"{P(R)['P4_n_trials']}, {P(R)['P4_max_twin_height_error_m']:.1e}")
    add("Supp. A", r"every twin with $0<s<1$ lies below the road", "1.0", lambda R: f"{P(R)['P4_fraction_twin_below_road_for_0_lt_s_lt_1']}")
    add("Supp. A", r"with a root-mean-square residual of $4\times10^{-13}$\,px, the same mover seen from a curved path leaves $2.7$\,px, and a mover that copies the curved path fits again to $4\times10^{-13}$\,px",
        "4e-13, 2.7, 4e-13",
        lambda R: f"{P(R)['P2_straight_K8_median_static_fit_rms_px']:.0e}, {P(R)['P2_curved_K8_median_static_fit_rms_px']:.1f}, "
        f"{P(R)['P2_curved_convoy_K8_median_static_fit_rms_px']:.0e}")
    add("Supp. A", r"gives slopes $1.000$, $0.819$, $0.613$, $0.359$ and $0.000$", "1.000, 0.819, 0.613, 0.359, 0.000",
        lambda R: ", ".join(f(r["beta_s"], 3) for r in R.rows("results/oracle_validation.csv")))
    add("Supp. A", r"equal to $\beta(w)$ over the same movers to three decimals", "max |diff| < 5e-4",
        lambda R: _oracle_agreement(R))
    add("Supp. A", r"(over the design grid of $s$, $\beta(0.5)=0.62$)", "0.62", lambda R: f(_design_beta(R, 0.5)))
    add("Supp. A", r"median recovered weights $-0.0002$, $0.2497$, $0.4998$, $0.7498$ and $1.0000$", "-0.0002, 0.2497, 0.4998, 0.7498, 1.0000",
        lambda R: ", ".join(f(r["median_w_recovered"], 4) for r in R.rows("results/oracle_validation.csv")))

    # Supplement B: designed sets, sample rules
    D = lambda R: R.json("logs/design_checks.json")
    add("Supp. B", r"($640\times480$; $500\times375$ for Virtual KITTI~2)", "640x480; 500x375",
        lambda R: f"{cfg(R, 'RUN_CONFIG', 'W')}x{cfg(R, 'RUN_CONFIG', 'H')}; " + "{}x{}".format(*cfg(R, "VK_DESIGN", "crop_wh")))
    add("Supp. B", r"The run used one NVIDIA T4 GPU (16\,GB) with 4 CPU cores, PyTorch 2.11 and CUDA 12.8",
        "Tesla T4, 4 cores, torch 2.11, CUDA 12.8",
        lambda R: (lambda e: f"{e['gpu']}, {e['cpu_count']} cores, torch {'.'.join(e['torch'].split('.')[:2])}, CUDA {e['cuda']}")(R.json("logs/env_log.json")))
    add("Supp. B", r"model inference took 7.6\,h of the roughly 10\,h of all stages", "7.6 h, 10 h",
        lambda R: (lambda h: f"{sum(v for k, v in h.items() if k.startswith('eval:') and not k.endswith(':BONN')) / 3600:.1f} h, "
                   f"{sum(h.values()) / 3600:.0f} h")(stage_hours(R)))
    add("Supp. B", r"is $0.49$, $2.12$ and $3.39$ times that at $\times1$", "0.49, 2.12, 3.39",
        lambda R: ", ".join(f(D(R)[f"bsw_parallax_ratio_{k}_to_B1_median"]) for k in ("B05", "B2", "B3")))
    add("Supp. B", r"all 192 base movers are measurable at every level", "192 of 192",
        lambda R: f"{D(R)['bsw_measurable_at_every_level']} of {D(R)['bsw_n_base_movers']}")
    add("Supp. B", r"every model's median single-image error of the 148 H12 movers agrees across the four levels to within $10^{-4}$",
        "148, 6 models < 1e-4", lambda R: _bsw_mono_spread(R))
    add("Supp. B", r"the median share of the mover they hide is $0.21$, $0.34$ and $0.54$", "0.21, 0.34, 0.54",
        lambda R: ", ".join(f(R.one("tables/T20a_dose_response.csv", wall=w)["share hidden"]) for w in ("15 cm", "35 cm", "70 cm")))
    add("Supp. B", r"with centred $500\times375$ crops", "clone, camera 0, 500x375",
        lambda R: (lambda v: f"{v['variation']}, camera {v['camera']}, {v['crop_wh'][0]}x{v['crop_wh'][1]}")(cfg(R, "VK_DESIGN")))
    add("Supp. B", r"for 97.6\% of the E5 movers and for 188, 186 and 187 movers at 15, 35 and 70\,cm", "97.6%; 188, 186, 187",
        lambda R: f"{100 * R.json('logs/testbed_checks.json')['abl_walls_designed_frac']:.1f}%; "
        + ", ".join(str(D(R)[f"wdr_W{k}_contact_hidden_both_views"]) for k in (15, 35, 70)))
    add("Supp. B", r"(at least 5 contact pixels in both views)", "5", lambda R: f"{cfg(R, 'PREREG', 'h13_contact_px_min')}")
    add("Supp. B", r"(at least 5 pixels within 3\,cm of the fitted plane)", "3 cm", lambda R: f"{100 * cfg(R, 'PREREG', 'rabl_contact_band_m'):.0f} cm")
    add("Supp. B", r"if their box moves by at least 0.5\,m between the images and are parked if it moves by at most 0.15\,m", "0.5 m, 0.15 m",
        lambda R: f"{cfg(R, 'PREREG', 'kt_moving_disp_m')} m, {cfg(R, 'PREREG', 'kt_parked_disp_m')} m")
    add("Supp. B", r"at most 30\% of the box lies outside the image, at most half of their LiDAR returns are hidden, and at least 40 LiDAR pixels remain",
        "30%, 50%, 40 px", lambda R: f"{pct(cfg(R, 'AV2_DESIGN', 'max_truncation'))}, {pct(cfg(R, 'AV2_DESIGN', 'max_occlusion'))}, "
        f"{cfg(R, 'PREREG', 'kt_min_lidar_px')} px")

    # Supplement C: the locked predictions P2-P4
    p2 = lambda R, key: [r for r in R.rows("results/prediction_summary.csv") if r["prediction"].startswith("P2") and r["model"].endswith(key)]
    add("Supp. C", r"It holds for M1 with exact correspondences ($0.83$ $[0.80,0.87]$) and with flow ($0.68$ $[0.62,0.74]$)",
        "0.83 [0.80, 0.87]; 0.68 [0.62, 0.74]",
        lambda R: "; ".join(row_ci(p2(R, k)[0]) for k in ("M1 epipolar residual (exact corr.)", "M1 epipolar residual (flow)")))
    add("Supp. C", r"for M2 with exact correspondences in every model ($0.30$ to $0.48$)", "6 SUPPORTED, 0.30 to 0.48",
        lambda R: (lambda rs: f"{sum(r['verdict'] == 'SUPPORTED' for r in rs)} SUPPORTED, " + rng([r["estimate"] for r in rs]))(
            p2(R, "M2 rigid-flow residual (exact corr.)")))
    add("Supp. C", r"with flow, M2 differences are positive but below $0.3$ ($0.18$ to $0.25$)", "6 PARTIAL, 0.18 to 0.25",
        lambda R: (lambda rs: f"{sum(r['verdict'] == 'PARTIAL' for r in rs)} PARTIAL, " + rng([r["estimate"] for r in rs]))(
            p2(R, "M2 rigid-flow residual (flow)")))
    add("Supp. C", r"model confidence (M4) shows no difference for DUSt3R, MonST3R and MapAnything", "DUSt3R, MonST3R, MapAnything",
        lambda R: names(r["model"].split(" / ")[0] for r in p2(R, "M4 model confidence") if r["verdict"] == "FALSIFIED"))
    add("Supp. C", r"the correlations are at most $0.12$, so it fails", "max 0.12",
        lambda R: "max " + f(max(est(R, "P3a", m) for m in MODELS)))
    add("Supp. C", r"it holds for all six models ($0.03$ to $0.34$", "6 SUPPORTED, 0.03 to 0.34",
        lambda R: f"{len(supported(R, 'P3b'))} SUPPORTED, " + rng([est(R, "P3b", m) for m in MODELS]))
    add("Supp. C", r"it fails for VGGT, $\pi^3$ and MapAnything", "3 FALSIFIED",
        lambda R: f"{sum(r['verdict'] == 'FALSIFIED' for r in R.rows('results/prediction_summary.csv') if r['prediction'].startswith('P4'))} FALSIFIED")
    add("Supp. C", r"Holm-adjusted bootstrap $p$-values of the supported H5--H13 contrasts are below $0.001$ except H5 for DUSt3R ($0.004$) and VGGT ($0.010$), H7 for MapAnything ($0.007$) and H12 for DUSt3R ($0.013$)",
        "H5 DUSt3R 0.004, H5 VGGT-1B 0.010, H7 MapAnything 0.007, H12 DUSt3R 0.013", lambda R: _p_exceptions(R))
    add("Supp. C", r"in the CONVOY-3D design range $[-0.47,1.90]$", "[-0.47, 1.90]",
        lambda R: (lambda s, x: f"[{f(-math.log(1 - s[0]))}, {f(-math.log(1 - s[1]))}]"
                   if [round(-math.log(1 - v), 2) for v in s] == [round(v, 2) for v in x] else "check")(
            cfg(R, "MOVER_DESIGN", "s_range_main"), cfg(R, "PREREG", "h9_x_range")))
    add("Supp. C", r"have a slope on $x_{\mathrm{geo}}$ below $0.3$ (supported if the estimate is below $0.3$ and its interval ends below $0.5$)",
        "0.3, 0.5", lambda R: f"{cfg(R, 'PREREG', 'h9_beta_max')}, {cfg(R, 'PREREG', 'h9_ci_hi_max')}")
    add("Supp. C", r"to be tested only with at least 20 vehicles from at least 5 pairs", "20, 5",
        lambda R: f"{cfg(R, 'PREREG', 'h9_min_movers')}, {cfg(R, 'PREREG', 'h9_min_pairs')}")
    add("Supp. C", r"of 1{,}247 moving vehicles in 289 pairs, 889 are in the parallel regime, 860 have at least 40 visible pixels after erosion, 30 have at least 85\% of their silhouette unoccluded, 15 have at least 5\,px of parallax and 6 have $x_{\mathrm{geo}}$ in the design range",
        "1247 in 289; 889; 860; 30; 15; 6",
        lambda R: (lambda st: f"{st[0][0]} in {st[0][1]}; " + "; ".join(str(v) for v, _p in st[1:]))(_h9_steps(R)))
    add("Supp. C", r"so H9 remains untested", "NOT TESTABLE, n = 6",
        lambda R: (lambda rs: f"{rs[0]['verdict']}, n = {rs[0]['n']}" if len({(r['verdict'], r['n']) for r in rs}) == 1 else "check")(
            [r for r in R.rows("results/E6_vkitti_h9.csv") if r["model"] in MODELS]))

    # Supplement D: additional results
    add("Supp. D", r"It is $0.17$, $0.25$, $0.11$, $0.08$, $0.14$ and $0.02$ on the 220 E1 movers", "0.17, 0.25, 0.11, 0.08, 0.14, 0.02; 220",
        lambda R: ", ".join(f(_excess_e1(R, m)) for m in MODELS) + f"; {e1(R, 'DUSt3R')['n']}")
    add("Supp. D", r"and $0.17$, $0.22$, $0.12$, $0.08$, $0.16$ and $0.04$ on the 89 A1 movers", "0.17, 0.22, 0.12, 0.08, 0.16, 0.04; 89",
        lambda R: ", ".join(f(_excess_a1(R, m)) for m in MODELS) + f"; {a1(R, 'DUSt3R')['n']}")
    add("Supp. D", r"although the raw slopes of the A1 movers are $0.10$ to $0.14$ higher", "0.10 to 0.14",
        lambda R: rng([float(a1(R, m)["beta_full"]) - float(e1(R, m)["beta_s"]) for m in MODELS]))
    add("Supp. D", r"the A1 slopes ($0.05$, $0.10$, $0.02$, $0.16$, $0.12$ and $0.01$)", "0.05, 0.10, 0.02, 0.16, 0.12, 0.01",
        lambda R: ", ".join(f(a1(R, m)["beta_knockout"]) for m in MODELS))
    add("Supp. D", r"lie at most $0.04$ above the single-image slopes of the same movers ($0.14$, $0.11$, $0.12$, $0.15$, $0.09$ and $0.01$)",
        "max 0.03 <= 0.04; 0.14, 0.11, 0.12, 0.15, 0.09, 0.01",
        lambda R: _at_most(max(float(a1(R, m)["beta_knockout"]) - float(a1(R, m)["beta_mono"]) for m in MODELS), 0.04) + "; "
        + ", ".join(f(a1(R, m)["beta_mono"]) for m in MODELS))
    add("Supp. D", r"the median $\rhom$ of the same 135 movers rises by $0.16$ (DUSt3R), $0.11$ (MASt3R), $0.36$ (MonST3R), $0.05$ (VGGT), $-0.01$ ($\pi^3$) and $0.13$ (MapAnything)",
        "135; 0.16, 0.11, 0.36, 0.05, -0.01, 0.13", lambda R: _wdr_mono_rise(R))
    add("Supp. D", r"369 of the 1{,}263 moving vehicles with enough flow have none, which leaves the 894", "369 of 1263, 894",
        lambda R: (lambda t: f"{t['n_no_static_solution']} of {t['n_moving']}, "
                   + (f"{t['n_moving'] - t['n_no_static_solution']}" if str(t["n_moving"] - t["n_no_static_solution"]) == twin_row(R, True, "all")["n"] else "check"))(
            R.json("results/twin_flow_thresholds.json")["thresholds"]))
    add("Supp. D", r"raises the share from $0.18$ to $0.21$ (VGGT), from $0.20$ to $0.27$ ($\pi^3$) and from $0.09$ to $0.12$ (MapAnything)",
        "0.18 to 0.21, 0.20 to 0.27, 0.09 to 0.12",
        lambda R: ", ".join(f"{f(r['straight_beta_K2'])} to {f(r['straight_beta_K8'])}" for r in R.rows("results/E4_summary.csv")))
    x6 = lambda R, ds, m: R.one("results/X6_contact_test.csv", dataset=ds, model=m)
    add("Supp. D", r"hidden-minus-visible contact differences of the per-object share are $0.10$, $0.16$, $0.07$, $-0.01$, $-0.06$ and $0.06$",
        "0.10, 0.16, 0.07, -0.01, -0.06, 0.06; all include zero",
        lambda R: ", ".join(f(x6(R, "Argoverse 2", m)["diff_hidden_minus_visible"]) for m in MODELS) + "; "
        + ("all include zero" if all(float(x6(R, "Argoverse 2", m)["ci_lo"]) <= 0 <= float(x6(R, "Argoverse 2", m)["ci_hi"]) for m in MODELS) else "check"))
    add("Supp. D", r"only MASt3R shows a positive difference ($0.15$ $[0.07,0.26]$)", "MASt3R 0.15 [0.07, 0.26]",
        lambda R: "; ".join(f"{r['model']} {row_ci(r, 'median_paired_diff')}" for r in R.select("results/X6_contact_matched.csv", dataset="Argoverse 2")
                            if r["direction"] != "no clear change"))
    add("Supp. D", r"intervals above zero for MASt3R, MonST3R, $\pi^3$ and MapAnything and below zero for VGGT", "higher: MASt3R, MonST3R, pi3, MapAnything; lower: VGGT-1B",
        lambda R: "higher: " + names(r["model"] for r in R.select("results/X6_contact_test.csv", dataset="Virtual KITTI 2") if r["direction"] == "higher")
        + "; lower: " + names(r["model"] for r in R.select("results/X6_contact_test.csv", dataset="Virtual KITTI 2") if r["direction"] == "lower"))
    add("Supp. D", r"after matching, only MASt3R remains above zero and VGGT below", "higher: MASt3R; lower: VGGT-1B",
        lambda R: "higher: " + names(r["model"] for r in R.select("results/X6_contact_matched.csv", dataset="Virtual KITTI 2") if r["direction"] == "higher")
        + "; lower: " + names(r["model"] for r in R.select("results/X6_contact_matched.csv", dataset="Virtual KITTI 2") if r["direction"] == "lower"))
    add("Supp. D", r"between 40 and 80\,px it is $0.83$, $0.98$, $0.72$, $0.52$, $0.30$ and $0.53$", "0.83, 0.98, 0.72, 0.52, 0.30, 0.53",
        lambda R: ", ".join(f(parse_ci(cr_bin(R, NAME[m], "40-80 px")["beta_ci"])[0]) for m in MODELS))
    add("Supp. D", r"and above 80\,px $0.81$, $0.64$, $0.85$, $0.82$, $0.83$ and $0.68$", "0.81, 0.64, 0.85, 0.82, 0.83, 0.68",
        lambda R: ", ".join(f(parse_ci(cr_bin(R, NAME[m], ">80 px")["beta_ci"])[0]) for m in MODELS))
    add("Supp. D", r"(DUSt3R, MASt3R, MonST3R, VGGT, $\pi^3$, MapAnything; 37 and 46 movers)", "37, 46",
        lambda R: f"{cr_bin(R, 'DUSt3R', '40-80 px')['n']}, {cr_bin(R, 'DUSt3R', '>80 px')['n']}")
    add("Supp. D", r"against $0.16$ to $0.85$ between 20 and 40\,px (20 movers; $-0.08$ for $\pi^3$)", "0.16 to 0.85, 20 movers; pi3 -0.08",
        lambda R: rng([parse_ci(cr_bin(R, NAME[m], "20-40 px")["beta_ci"])[0] for m in MODELS if m != "pi3"])
        + f", {cr_bin(R, 'DUSt3R', '20-40 px')['n']} movers; pi3 " + f(parse_ci(cr_bin(R, 'π³', '20-40 px')['beta_ci'])[0]))
    rob = lambda R, level: [float(r["beta_s"]) for r in R.select("results/robustness.csv", factor="cam_kind", level=level)]
    add("Supp. D", r"higher for forward-looking cameras ($-0.01$ to $0.35$ across models) than for oblique ($-0.04$ to $0.17$) and side-looking ones ($0.02$ to $0.18$)",
        "-0.01 to 0.35; -0.04 to 0.17; 0.02 to 0.18", lambda R: "; ".join(rng(rob(R, k)) for k in ("forward", "oblique", "side")))
    add("Supp. D", r"It is lowest in the shortest third of focal lengths for all six models", "6 models",
        lambda R: f"{sum(min(R.select('results/robustness.csv', model=m, factor='focal_tercile'), key=lambda r: float(r['beta_s']))['level'] == 'short f' for m in MODELS)} models")

    # Figure captions
    add("Fig. 1 caption", r"($s=0.84$; true depth 6.2\,m, geometric answer 39.5\,m)", "s 0.84, 6.2 m, 39.5 m", lambda R: _teaser(R))
    add("Fig. 10 caption", r"(4{,}711, 4{,}531 and 4{,}014 tracks)", "4711, 4531, 4014",
        lambda R: ", ".join(r["n_tracks"] for r in R.rows("results/prevalence_kview_vkitti.csv")))
    add("Fig. 8 caption", r"whose geometric answer is 6.7 times too far", "6.7", lambda R: f"{1 / (1 - cfg(R, 'PREREG', 'implied_factor_s')):.1f}")
    add("Fig. 3 caption", r"$\bs$ of the same 158 movers in each condition", "158",
        lambda R: "; ".join(sorted({r["n"] for r in R.rows("results/E5_beta_by_condition.csv")})))
    add("Fig. 7 caption", r"$\bs$ of the same 141 crates in both conditions", "141",
        lambda R: "; ".join(sorted({r["n"] for r in R.rows("results/E5real_beta_by_condition.csv")})))
    return C


# ---- helpers used by several claims -------------------------------------------------------------------------------

def a1(R, model):
    return R.one("results/A1_knockout.csv", model=model)


def _excess_e1(R, model):
    """Slope of rho - rho_mono on x over the E1 movers: beta_s - beta_mono (same movers, same x)."""
    return float(e1(R, model)["beta_s"]) - float(e1(R, model)["beta_mono"])


def _excess_a1(R, model):
    return float(a1(R, model)["beta_full"]) - float(a1(R, model)["beta_mono"])


def _at_most(v, bound):
    return f"max {f(v)} <= {f(bound)}" if v <= bound else f"max {f(v)}"


def _design_beta(R, w):
    """beta(w) over a uniform grid of the design range of s, as the run tabulated it (notebook cell 2.4)."""
    return next(r["beta_ols_over_design"] for r in R.json("logs/a2_and_identity.json")["slope_weight"] if abs(r["w"] - w) < 1e-9)


def _oracle_agreement(R):
    d = max(abs(float(r["beta_s"]) - float(r["beta_theory_identity"])) for r in R.rows("results/oracle_validation.csv"))
    return "max |diff| < 5e-4" if d < 5e-4 else f"max |diff| {d:.1e}"


def _t21d(R, prefix):
    return next(r for r in R.rows("tables/T21d_monst3r_vs_dust3r.csv") if r["contrast"].startswith(prefix))


def _h9_steps(R):
    """(vehicles, pairs) of the H9 attrition table, from all moving vehicles to the registered sample."""
    rows = [r for r in R.rows("tables/T10d_vkitti_sample.csv") if "v6.0 rule" not in r["step"]]
    return [(int(r["vehicles"]), int(r["pairs"])) for r in rows]


def _bsw_mono_spread(R):
    rows = [r for r in R.rows("results/BSW_beta.csv") if r["median_rho_mono"]]
    n = {r["n"] for r in rows}
    spread = {m: max(float(r["median_rho_mono"]) for r in rows if r["model"] == m)
              - min(float(r["median_rho_mono"]) for r in rows if r["model"] == m) for m in MODELS}
    ok = sum(v < 1e-4 for v in spread.values())
    return f"{'/'.join(sorted(n))}, {ok} models < 1e-4"


def _wdr_mono_rise(R):
    rows = R.rows("results/WDR_beta.csv")
    get = lambda m, lv: next(r for r in rows if r["model"] == m and r["level"] == lv)
    n = {get(m, lv)["n"] for m in MODELS for lv in ("G", "W70")}
    rise = [float(get(m, "W70")["median_rho_mono"]) - float(get(m, "G")["median_rho_mono"]) for m in MODELS]
    return f"{'/'.join(sorted(n))}; " + ", ".join(f(v) for v in rise)


def _movers_per_scene(R):
    counts = {}
    with open(R.path("manifest/convoy3d_main_movers.csv"), newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            counts[r["scene"]] = counts.get(r["scene"], 0) + 1
    return counts


def _beyond(R):
    with open(R.path("manifest/convoy3d_main_movers.csv"), newline="", encoding="utf-8") as fh:
        s = [float(r["s"]) for r in csv.DictReader(fh)]
    b = [v for v in s if v > 1]
    return f"{100 * len(b) / len(s):.0f}%, [{min(b):.1f}, {max(b):.1f}]"


def _camera_ranges(R):
    cam = cfg(R, "CAMERA_DESIGN")
    b = [v for k in ("forward", "oblique", "side") for v in cam[k]["baseline_m"]]
    return "{:.0f}-{:.0f} px, {}-{} m, {}-{:g} m".format(*cam["focal_px"], *cam["height_m"], min(b), max(b))


def _tum_check(R):
    vals = [float(re.search(r"\((\d\.\d+)\)", r["check"]).group(1)) for r in R.rows("tables/T1b_data.csv")
            if r["dataset"].startswith("TUM fr3")]
    return f"{100 * min(vals):.1f}-{100 * max(vals):.1f}%"


def _gate(R, flat):
    rows = [r for r in R.rows("results/gate.csv") if (r["set"] == "FLAT") == flat]
    if flat:
        ok = all(r["gate"] == "FAIL" for r in rows)
        sl = max(abs(float(r["slope_median"])) for r in rows)
        return f"{'all FAIL' if ok else 'check'}, |slope| <= {f(sl) if sl > 0.005 else '0.01'}, " + rng([r["absrel_median"] for r in rows], sep="-")
    ok = all(r["gate"] == "PASS" for r in rows)
    return f"{'all PASS' if ok else 'check'}, " + rng([r["absrel_median"] for r in rows], sep="-")


def _max_p_supported(R):
    ps = [float(r["p_holm"]) for r in R.rows("results/prediction_summary.csv")
          if r["prediction"][:3] in ("H5 ", "H6 ", "H7 ", "H8 ", "H11", "H12", "H13") and r["verdict"] == "SUPPORTED"]
    return max(ps)


def _p_exceptions(R):
    out = []
    for r in R.rows("results/prediction_summary.csv"):
        key = r["prediction"].split()[0]
        if key in ("H5", "H6", "H7", "H8", "H11a", "H11b", "H12", "H13") and r["verdict"] == "SUPPORTED" and float(r["p_holm"]) >= 0.001:
            out.append(f"{key} {r['model']} {f(r['p_holm'], 3)}")
    return ", ".join(out)


def _pooled(rows):
    n = sum(float(r["n_moving_pixels"]) for r in rows)
    return sum(float(r["frac_parallel"]) * float(r["n_moving_pixels"]) for r in rows) / n


def _spearman_tradeoff(R):
    from scipy.stats import spearmanr
    rows = R.rows("results/mask_tradeoff.csv")
    rho = spearmanr([float(r["delta_beta"]) for r in rows], [float(r["delta_recall"]) for r in rows]).statistic
    return f"{len(rows)} changes, {f(rho)}"


def _failed_summary(R):
    rows = R.rows("results/prediction_summary.csv")

    def count(prefix, verdict):
        return sum(r["verdict"] == verdict for r in rows if r["prediction"].startswith(prefix))

    n = lambda prefix: sum(r["prediction"].startswith(prefix) for r in rows)
    return (f"P1 {count('P1 degeneracy (CONVOY-3D)', 'FALSIFIED')}/{n('P1 degeneracy (CONVOY-3D)')}, "
            f"P3a {count('P3a', 'FALSIFIED')}/{n('P3a')} ({count('P3a', 'PARTIAL')} partial), "
            f"P4 {count('P4', 'FALSIFIED')}/{n('P4')} falsified")


def _bonn_counts(R):
    counts = {}
    for r in R.select("results/bonn_geometry_share.csv", grouping="kind"):
        counts[r["model"]] = counts.get(r["model"], 0) + int(r["n"])
    return counts


def _teaser(R):
    with open(R.path("manifest/convoy3d_main_movers.csv"), newline="", encoding="utf-8") as fh:
        r = next(r for r in csv.DictReader(fh) if r["uid"] == "MAIN-0005-0")
    return f"s {f(r['s'])}, {float(r['gt_Z_true']):.1f} m, {float(r['gt_Zgeo_med']):.1f} m"


CLAIMS = _claims()


def check(R=None):
    """``[(claim, recomputed value, ok)]`` for every claim."""
    R = R or Results()
    out = []
    for c in CLAIMS:
        try:
            v = c.compute(R)
        except Exception as e:                      # report, do not hide, a claim that cannot be recomputed
            v = f"ERROR {type(e).__name__}: {e}"
        out.append((c, v, v == c.value))
    return out


def quotes_in(tex):
    """Claims whose LaTeX fragment is not found in ``tex`` (whitespace-insensitive)."""
    norm = lambda s: re.sub(r"\s+", " ", s)
    t = norm(tex)
    return [c for c in CLAIMS if norm(c.quote) not in t]
