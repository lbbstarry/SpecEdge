#!/usr/bin/env python
"""Check every number printed in the paper against the artifact it came from.

Each entry below pairs a value as it appears in the manuscript with the field
of the experiment output that produced it. The comparison rounds the artifact
value to the precision the paper prints, so a claim passes only if the two
agree at the stated number of digits.

    python scripts/verify_claims.py            # check all
    python scripts/verify_claims.py --section VIII

Exits nonzero if any claim fails, so it can gate a submission.
"""
from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "output" / "revision_v4"

# (paper value, section, artifact file, dotted JSON path, what it is)
# Paper value is written exactly as the manuscript prints it, including
# trailing zeros, because the printed precision sets the comparison tolerance.
CLAIMS: list[tuple[str, str, str, str, str]] = [
    # --- Section V, reference noise floor -------------------------------
    ("2.42", "V-B", "e1b_noise_floor.json",
     "standard.systematic_1px.cd_mean.mean", "CD sigma_ref, in-distribution"),
    ("2.65", "V-B", "e1b_noise_floor.json",
     "extreme.systematic_1px.cd_mean.mean", "CD sigma_ref, Extreme (= tau_sigma)"),
    ("0.17", "V-B", "e1b_noise_floor.json",
     "standard.systematic_1px.lwr_3sigma.mean", "LWR sigma_ref, in-distribution"),
    ("0.27", "V-B", "e1b_noise_floor.json",
     "extreme.systematic_1px.lwr_3sigma.mean", "LWR sigma_ref, Extreme"),
    ("0.08", "V-B", "e1b_noise_floor.json",
     "standard.systematic_1px.ler_mean_3sigma.mean", "LER sigma_ref, in-distribution"),
    ("0.13", "V-B", "e1b_noise_floor.json",
     "extreme.systematic_1px.ler_mean_3sigma.mean", "LER sigma_ref, Extreme"),

    # --- Section VIII-B, failure onset ----------------------------------
    ("0.454", "VIII-B", "e3_onset.json", "breakpoint", "breakpoint estimate"),
    ("0.304", "VIII-B", "e3_onset.json", "breakpoint_ci95.0",
     "n-out-of-n CI low, quoted only to reject as optimistic"),
    ("0.528", "VIII-B", "e3_onset.json", "breakpoint_ci95.1",
     "n-out-of-n CI high, quoted only to reject as optimistic"),
    ("0.362", "VIII-B", "e24_breakpoint_ci/summary.json", "m_of_n.ci95.0",
     "m-out-of-n CI low, the reported interval"),
    ("0.682", "VIII-B", "e24_breakpoint_ci/summary.json", "m_of_n.ci95.1",
     "m-out-of-n CI high, the reported interval"),
    ("22", "VIII-B", "e24_breakpoint_ci/summary.json", "m_of_n.size",
     "subsample size m"),
    ("0.29", "VIII-B", "e24_breakpoint_ci/summary.json",
     "n_of_n.frac_at_estimate",
     "share of n-out-of-n replicates returning the estimate exactly"),
    ("0.865", "VIII-B", "e24_breakpoint_ci/summary.json",
     "n_of_n.frac_within_0.05",
     "share of n-out-of-n replicates within 0.05 of the estimate"),
    ("59", "VIII-B", "e24_breakpoint_ci/summary.json", "n_of_n.n_distinct",
     "distinct breakpoint values over 2000 n-out-of-n replicates"),
    ("15", "VIII-B", "e3_onset.json", "n_below", "samples below breakpoint"),
    ("47", "VIII-B", "e3_onset.json", "n_above", "samples above breakpoint"),
    ("11.09", "VIII-B", "e3_onset.json", "cd_mae_mean_below_bp",
     "mean CD MAE below breakpoint"),
    ("0.43", "VIII-B", "e3_onset.json", "cd_mae_mean_above_bp",
     "mean CD MAE above breakpoint"),
    ("39", "VIII-B", "e3_onset.json", "standard_overlap_n",
     "in-distribution samples in the overlapping fg region"),
    ("4.55", "VIII-B", "e3_onset.json", "standard_overlap_cd_mae_max",
     "max CD MAE in that region"),
    ("0.052", "VIII-B", "e3_onset.json", "standard_overlap_cd_mae_median",
     "median CD MAE in that region"),
    ("-3.07", "VIII-B", "e3_onset.json", "logistic.coef_fg_z",
     "logistic fg coefficient"),
    ("0.069", "VIII-B", "e3_onset.json", "logistic.lr_p_fg",
     "logistic fg likelihood-ratio p"),
    ("0.28", "VIII-B", "e3_onset.json", "logistic.coef_rbbox_z",
     "logistic r_bbox coefficient"),
    ("0.93", "VIII-B", "e3_onset.json", "logistic.lr_p_rbbox",
     "logistic r_bbox likelihood-ratio p"),

    # --- Section VIII-B, sup-F treatment of the estimated breakpoint ----
    ("13.50", "VIII-B", "e16_stats/summary.json", "breakpoint.sup_F_observed",
     "observed sup-F"),
    ("5.0e-4", "VIII-B", "e16_stats/summary.json", "breakpoint.p_bootstrap_null",
     "bootstrap null p (the defensible one)"),
    ("1.99", "VIII-B", "e16_stats/summary.json", "breakpoint.null_F_median",
     "simulated null sup-F median"),
    ("4.53", "VIII-B", "e16_stats/summary.json", "breakpoint.null_F_p95",
     "simulated null sup-F 95th percentile"),
    ("9.1e-7", "VIII-B", "e16_stats/summary.json", "breakpoint.p_naive_F_3_nm5",
     "naive F(3,n-5) p, quoted to show it is three orders too small"),

    # --- Section VIII-C, CNN-consensus control --------------------------
    ("0.975", "VIII-C", "e1a_summary.json", "extreme.consensus_vs_lithoref_iou_mean",
     "agreement between the two references, Extreme"),
    ("5.483", "VIII-C", "e1a_summary.json", "extreme.segformer_cd_mae_vs_consensus",
     "SegFormer CD MAE against the consensus reference, Extreme"),
    ("0.610", "VIII-C", "e1a_summary.json", "standard.segformer_cd_mae_vs_consensus",
     "same, in-distribution"),

    # --- Section VIII-D, layout-bbox control ----------------------------
    ("0.73", "VIII-D", "summary_revision_v4.json", "e8.spearman_rbbox_vs_fg",
     "Spearman(r_bbox, fg): the collinearity"),
    ("-0.398", "VIII-D", "summary_revision_v4.json", "e8.spearman_rbbox_vs_err",
     "Spearman(r_bbox, CD MAE)"),
    ("-0.174", "VIII-D", "summary_revision_v4.json",
     "e8.partial_rbbox_vs_err_given_fg", "partial correlation given fg"),
    ("0.18", "VIII-D", "summary_revision_v4.json", "e8.p_partial",
     "p of that partial correlation"),

    # --- Section VIII-G, the runtime guard ------------------------------
    ("0.910", "VIII-G", "e4_summary.json", "extreme_segformer.auroc_disagreement",
     "guard AUROC, SegFormer x Extreme"),
    ("62", "VIII-G", "e4_summary.json", "extreme_segformer.n", "n for that AUROC"),
    ("8", "VIII-G", "e4_summary.json", "extreme_segformer.n_fail",
     "failures among them"),
    ("0.729", "VIII-G", "e4_summary.json", "extreme_all_models.auroc_disagreement",
     "guard AUROC, four frontends x Extreme"),
    ("0.774", "VIII-G", "e4_summary.json", "pooled_all.auroc_disagreement",
     "guard AUROC, pooled"),
    ("488", "VIII-G", "e4_summary.json", "pooled_all.n", "n pooled"),
    ("0.764", "VIII-G", "e4_summary.json", "pooled_all.auroc_fg_dev",
     "foreground-deviation baseline AUROC, pooled"),
    ("0.758", "VIII-G", "e4_summary.json", "pooled_all.auroc_cc_dev",
     "component-count-deviation baseline AUROC, pooled"),

    # --- Section VII-C, guard applied to the design verdicts (E15) ------
    # Everything here is read at the pre-committed d* = in-distribution P95;
    # the P80 and P90 rows exist in the artifact and give the same wrong-call
    # counts but different flag rates, which is what the text now says.
    ("0.629", "VII-C", "e15_guard_dfm/summary.json", "routed.p95_hrnet.flag_rate",
     "flag rate at the pre-committed threshold"),
    ("39", "VII-C", "e15_guard_dfm/summary.json", "routed.p95_hrnet.n_flagged",
     "samples flagged at that threshold"),
    ("12", "VII-C", "e15_guard_dfm/summary.json", "baseline_monitored_only.wrong",
     "wrong design calls deploying SegFormer alone"),
    ("6", "VII-C", "e15_guard_dfm/summary.json", "routed.p95_hrnet.wrong",
     "wrong calls after routing to HRNet"),
    ("4", "VII-C", "e15_guard_dfm/summary.json", "routed.p95_unet.wrong",
     "wrong calls after routing to U-Net, the best retrospective choice"),

    # --- Section VIII-G, policy comparison (E22) ------------------------
    ("0.470", "VIII-G", "e22_policy/summary.json",
     "cd_error.deeplabv3plus.extreme.always_fallback.mean",
     "always-DeepLabV3+ Extreme CD MAE, the baseline the guard must beat"),
    ("0.252", "VIII-G", "e22_policy/summary.json",
     "cd_error.deeplabv3plus.standard.always_fallback.mean",
     "same policy in distribution"),
    ("0.394", "VIII-G", "e22_policy/summary.json",
     "cd_error.hrnet.extreme.always_fallback.mean", "always-HRNet Extreme"),
    ("0.393", "VIII-G", "e22_policy/summary.json",
     "cd_error.hrnet.extreme.routed.mean", "guard->HRNet Extreme: the tie"),
    ("3.37", "VIII-G", "e22_policy/summary.json",
     "cd_error.hrnet.extreme.routed.max",
     "guarded Extreme max, the tail bound every pairing reaches"),
    ("3.48", "VIII-G", "e22_policy/summary.json",
     "cd_error.hrnet.extreme.always_fallback.max",
     "always-HRNet Extreme max, which the guard beats"),
    ("5", "VIII-G", "e22_policy/summary.json",
     "design_verdicts.deeplabv3plus.always_fallback",
     "wrong design calls deploying DeepLabV3+ alone"),
    ("6", "VIII-G", "e22_policy/summary.json",
     "design_verdicts.hrnet.always_fallback", "same for HRNet alone"),

    # --- Section VIII-H, training support -------------------------------
    ("0.167", "VIII-H", "e5b_train_support.json", "train_fg_min",
     "minimum foreground ratio seen in training"),
    ("4", "VIII-H", "e5b_train_support.json", "extreme_below_train_min",
     "Extreme samples below it"),

    # --- Section VII-D, controlled test of the mechanism (E23) ----------
    ("586", "VII-D", "e23_mechanism/summary.json", "n_masks",
     "orientable reference masks perturbed"),
    ("0.0010", "VII-D", "e23_mechanism/summary.json",
     "by_c.1.0.iou_spread_across_arms",
     "IoU range across the three matched fields at c=1"),
    ("0.0047", "VII-D", "e23_mechanism/summary.json",
     "by_c.3.0.iou_spread_across_arms", "same at c=3"),
    ("1.881", "VII-D", "e23_mechanism/summary.json",
     "by_c.1.0.arms.constant.cd_err_mean",
     "constant offset moves CD by 0.94*2c at c=1"),
    ("0.072", "VII-D", "e23_mechanism/summary.json",
     "by_c.1.0.arms.rademacher.cd_err_mean",
     "two-point field leaves CD essentially exact"),
    ("5.636", "VII-D", "e23_mechanism/summary.json",
     "by_c.3.0.arms.constant.cd_err_mean", "constant offset at c=3"),
    ("0.209", "VII-D", "e23_mechanism/summary.json",
     "by_c.3.0.arms.rademacher.cd_err_mean", "two-point field at c=3"),
    ("7.34", "VII-D", "e23_mechanism/summary.json",
     "by_c.1.0.arms.constant.ler_ref_mean", "reference LER 3-sigma level"),
    ("12.20", "VII-D", "e23_mechanism/summary.json",
     "by_c.3.0.arms.rademacher.ler_pert_mean", "two-point LER at c=3"),
    ("14.07", "VII-D", "e23_mechanism/summary.json",
     "by_c.3.0.arms.gaussian.ler_pert_mean",
     "Gaussian LER at c=3: same L1 norm, different shape"),
    ("0.879", "VII-D", "e23_mechanism/summary.json", "attenuation.slope",
     "slope of 1-IoU on (L/A)*mean|delta|, predicted 1.0"),
    ("0.989", "VII-D", "e23_mechanism/summary.json", "attenuation.pearson_r",
     "its correlation"),
    ("1758", "VII-D", "e23_mechanism/summary.json", "attenuation.n",
     "cases in that regression"),

    # --- Section VII-B, decoupling under cross-validation ---------------
    ("2352", "VII-B", "e21_decoupling_cv/summary.json", "n_records",
     "frontend-image records under cross-validation"),
    ("588", "VII-B", "e21_decoupling_cv/summary.json", "n_images",
     "images, each held out exactly once"),

    # --- Section VIII-G, conditional risk model -------------------------
    ("0.690", "VIII-G", "e19_risk_model/calibration.json",
     "cross_fit_repeats.auroc_mean",
     "risk-model AUROC in distribution, mean over 40 fold assignments"),
    ("0.025", "VIII-G", "e19_risk_model/calibration.json",
     "cross_fit_repeats.auroc_sd", "its sd over those 40 assignments"),
    ("0.731", "VIII-G", "e19_risk_model/calibration.json", "extreme.auroc",
     "risk-model AUROC out of window"),
    ("0.023", "VIII-G", "e19_risk_model/calibration.json", "in_dist.ece",
     "expected calibration error in distribution"),
    ("0.120", "VIII-G", "e19_risk_model/calibration.json", "extreme.ece",
     "expected calibration error out of window (the level does not transfer)"),
    ("0.263", "VIII-G", "e19_risk_model/calibration.json",
     "drift.mean_predicted_extreme", "mean predicted risk out of window"),
    ("0.185", "VIII-G", "e19_risk_model/calibration.json", "extreme.base_rate",
     "observed failure rate out of window"),
]


def resolve(doc, dotted: str):
    """Walk a dotted path, accepting integer keys as list indices.

    Some artifacts key by a float rendered as a string, so "by_c.1.0.slope"
    has to reach the literal key "1.0". When a segment does not resolve, it is
    rejoined with the next one before giving up.
    """
    node = doc
    parts = dotted.split(".")
    i = 0
    while i < len(parts):
        key = parts[i]
        if isinstance(node, list):
            node = node[int(key)]
            i += 1
            continue
        if key in node:
            node = node[key]
            i += 1
            continue
        merged = f"{key}.{parts[i + 1]}" if i + 1 < len(parts) else None
        if merged is not None and merged in node:
            node = node[merged]
            i += 2
            continue
        raise KeyError(key)
    return node


def matches(paper: str, actual: float) -> bool:
    """True if the artifact value rounds to what the paper prints.

    Scientific notation is compared at 2% relative tolerance, because the paper
    prints one significant digit for those and rounding is not the right test.
    """
    if "e" in paper.lower():
        return abs(actual - float(paper)) <= 0.02 * abs(float(paper))
    decimals = -Decimal(paper).as_tuple().exponent
    return round(float(actual), decimals) == float(paper)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--section", help="only check claims from this section")
    args = ap.parse_args()

    claims = CLAIMS
    if args.section:
        claims = [c for c in CLAIMS if c[1].startswith(args.section)]
        if not claims:
            print(f"no claims for section {args.section!r}", file=sys.stderr)
            return 2

    cache: dict[str, object] = {}
    failed = missing = 0

    print(f"{'':<6}{'sec':<9}{'paper':>10}  {'artifact':>12}  claim")
    print("-" * 96)
    for paper, section, filename, path, what in claims:
        if filename not in cache:
            f = ARTIFACTS / filename
            cache[filename] = json.loads(f.read_text()) if f.exists() else None
        doc = cache[filename]

        if doc is None:
            print(f"{'MISS':<6}{section:<9}{paper:>10}  {'-':>12}  {what}")
            print(f"{'':<6}  {filename} not found; run the experiment first")
            missing += 1
            continue
        try:
            actual = resolve(doc, path)
        except (KeyError, IndexError, ValueError):
            print(f"{'MISS':<6}{section:<9}{paper:>10}  {'-':>12}  {what}")
            print(f"{'':<6}  no field {path!r} in {filename}")
            missing += 1
            continue

        ok = matches(paper, actual)
        shown = f"{actual:.6g}" if isinstance(actual, float) else str(actual)
        print(f"{'ok' if ok else 'FAIL':<6}{section:<9}{paper:>10}  {shown:>12}  {what}")
        if not ok:
            print(f"{'':<6}  {filename}:{path}")
            failed += 1

    print("-" * 96)
    checked = len(claims)
    print(f"{checked - failed - missing}/{checked} claims verified"
          + (f", {failed} disagree" if failed else "")
          + (f", {missing} unavailable" if missing else ""))
    return 1 if failed or missing else 0


if __name__ == "__main__":
    sys.exit(main())
