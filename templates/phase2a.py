"""
ieeg-ib phase2a — Graph-theoretic network fragmentation (H2).

Hypothesis H2: The MI-weighted network graph G(t) undergoes measurable
fragmentation in the 5-minute preictal window: density ρ(t), mean degree d̄(t),
and clustering coefficient C(t) all decrease significantly before seizure onset.

Dataset structure (from diagnostic + phase1b findings):
  - task-ictal EDFs: ~120s preictal + ictal, seizure onset at t=120.0s
  - task-interictal EDFs: ~300s separate interictal recordings
  - Sampling rate: 500 Hz

This notebook:
  1. Loads task-interictal EDF → interictal G(t) baseline (density, degree, clustering)
  2. Loads task-ictal EDF → preictal G(t) over 120s window
  3. Computes Wilcoxon signed-rank test: interictal > preictal for each metric
  4. Computes convergent validity: preictal Phi(t) and ρ(t) should correlate
  5. Saves results JSON, CSV, and visualizations

Config:
  MAX_PATIENTS  = 57    (all available)
  MAX_CHANNELS  = 16    (top by variance)
  WINDOW_SEC    = 10    (longer windows for stable graph estimation)
  STEP_SEC      = 10    (non-overlapping for independence)
  K_NN          = 5     (Kraskov k)
  EDGE_PCT      = 80    (keep top 20% of edges)
"""

import os, sys, csv, glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MAX_PATIENTS  = 57
MAX_CHANNELS  = 16
WINDOW_SEC    = 10
STEP_SEC      = 10
K_NN          = 5
EDGE_PCT      = 80    # keep edges above this percentile weight

# ---------------------------------------------------------------------------
# Locate core library
# ---------------------------------------------------------------------------
def _find_core(name):
    for dp, _, fns in os.walk("/kaggle/input"):
        if f"{name}.py" in fns:
            return dp
    return None

sys.path.insert(0, _find_core("ieeg_core"))
import ieeg_core as core

import mne
mne.set_log_level("WARNING")

# ---------------------------------------------------------------------------
# Dataset root
# ---------------------------------------------------------------------------
DATASET_ROOT = None
for dp, dirs, _ in os.walk("/kaggle/input"):
    if any(d.startswith("sub-") for d in dirs):
        DATASET_ROOT = dp
        break

print("=" * 60)
print("PHASE 2A: Graph-theoretic network fragmentation")
print("=" * 60)
print(f"Dataset root: {DATASET_ROOT}")

subject_dirs = sorted(glob.glob(os.path.join(DATASET_ROOT, "sub-*")))
print(f"Subjects found: {len(subject_dirs)}")


# ---------------------------------------------------------------------------
# Helpers (same as phase1b/1c)
# ---------------------------------------------------------------------------
def get_seizure_onset(events_path):
    if not events_path or not os.path.exists(events_path):
        return None
    with open(events_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            tt = row.get("trial_type", "").lower()
            if "sz onset" in tt or "seizure onset" in tt:
                try:
                    return float(row["onset"])
                except (KeyError, ValueError):
                    continue
            if any(kw in tt for kw in ("sz", "seizure", "ictal")):
                try:
                    return float(row["onset"])
                except (KeyError, ValueError):
                    continue
    return None


def load_edf(path, t_start=None, t_end=None):
    raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
    if t_start is not None or t_end is not None:
        raw.crop(tmin=t_start or 0.0, tmax=t_end)
    raw.load_data(verbose=False)
    raw.pick_types(eeg=True, ecog=True, seeg=True, misc=False,
                   stim=False, exclude="bads")
    raw.filter(0.5, 200.0, method="iir", verbose=False)
    raw.notch_filter(np.arange(60, 201, 60), verbose=False)
    raw.set_eeg_reference("average", verbose=False)
    return raw.get_data(), raw.ch_names, raw.info["sfreq"]


def top_channels(data, n):
    return np.argsort(np.var(data, axis=1))[-n:]


def infer_recording_type(ch_names):
    """Return 'ECoG', 'SEEG', or 'Unknown' based on channel naming."""
    names_upper = [c.upper() for c in ch_names]
    n_ecog = sum(1 for c in names_upper if any(p in c for p in ("G", "GRID", "STRIP")))
    n_seeg = sum(1 for c in names_upper
                 if any(c.startswith(p) for p in ("A", "B", "C", "D", "H", "OF", "AM", "HC")))
    if n_ecog > n_seeg:
        return "ECoG"
    elif n_seeg > 0:
        return "SEEG"
    return "Unknown"


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
from scipy import stats

all_results = []
patients_done = 0

for subj_dir in subject_dirs:
    if patients_done >= MAX_PATIENTS:
        break
    subj_id = os.path.basename(subj_dir)

    ieeg_dir = os.path.join(subj_dir, "ses-presurgery", "ieeg")
    if not os.path.isdir(ieeg_dir):
        continue

    ictal_edfs = sorted(f for f in os.listdir(ieeg_dir)
                        if "task-ictal" in f and f.endswith("_ieeg.edf"))
    inter_edfs = sorted(f for f in os.listdir(ieeg_dir)
                        if "task-interictal" in f and f.endswith("_ieeg.edf"))

    if not ictal_edfs or not inter_edfs:
        print(f"\n{subj_id}: missing EDF(s), skipping")
        continue

    ictal_edf  = os.path.join(ieeg_dir, ictal_edfs[0])
    inter_edf  = os.path.join(ieeg_dir, inter_edfs[0])
    events_tsv = ictal_edf.replace("_ieeg.edf", "_events.tsv")

    t_seizure = get_seizure_onset(events_tsv)
    if t_seizure is None:
        print(f"\n{subj_id}: no seizure onset, skipping")
        continue

    print(f"\n{'='*50}")
    print(f"Patient : {subj_id}  (t_seizure={t_seizure:.1f}s)")

    try:
        inter_data, ch_names, fs = load_edf(inter_edf)
        prei_data, _, _          = load_edf(ictal_edf, t_start=0.0, t_end=t_seizure)
    except Exception as e:
        print(f"  Load error: {e}")
        continue

    rec_type = infer_recording_type(ch_names)

    # Channel selection
    n_common = min(inter_data.shape[0], prei_data.shape[0])
    top_idx  = top_channels(inter_data[:n_common], MAX_CHANNELS)
    inter_sel = inter_data[top_idx]
    prei_sel  = prei_data[top_idx]
    n_ch      = len(top_idx)
    print(f"  {rec_type} | {n_ch} channels")

    win  = int(WINDOW_SEC * fs)
    step = int(STEP_SEC   * fs)

    # ── Compute network metrics ──────────────────────────────────────────────
    print("  Computing interictal network metrics...")
    metrics_inter, _ = core.mi_network_metrics(
        inter_sel, win, step, k=K_NN, edge_percentile=EDGE_PCT,
        max_channels=MAX_CHANNELS)

    print("  Computing preictal network metrics...")
    metrics_prei, prei_centers = core.mi_network_metrics(
        prei_sel, win, step, k=K_NN, edge_percentile=EDGE_PCT,
        max_channels=MAX_CHANNELS)

    rho_inter = metrics_inter["density"]
    rho_prei  = metrics_prei["density"]
    deg_inter = metrics_inter["mean_degree"]
    deg_prei  = metrics_prei["mean_degree"]
    clust_inter = metrics_inter["clustering"]
    clust_prei  = metrics_prei["clustering"]

    if len(rho_inter) < 3 or len(rho_prei) < 3:
        print("  Too few windows, skipping")
        continue

    # Skip saturated interictal networks (density==1 → near-zero MI baseline,
    # same artifact as HUP080; std→0 makes Cohen's d blow up to 10^12).
    if np.mean(rho_inter) > 0.98:
        print(f"  SKIP — saturated interictal network (mean_rho={np.mean(rho_inter):.3f})")
        continue

    # ── Per-patient Wilcoxon: interictal > preictal ─────────────────────────
    # Use Mann-Whitney U (one-sided) since samples come from different epochs
    _, p_rho   = stats.mannwhitneyu(rho_inter,   rho_prei,   alternative="greater")
    _, p_deg   = stats.mannwhitneyu(deg_inter,   deg_prei,   alternative="greater")
    _, p_clust = stats.mannwhitneyu(clust_inter, clust_prei, alternative="greater")

    # Effect sizes — clamp std to avoid blowup when interictal is near-constant
    def cohens_d(a, b):
        return (np.mean(a) - np.mean(b)) / max(float(np.std(a)), 1e-4)

    d_rho   = cohens_d(rho_inter,   rho_prei)
    d_deg   = cohens_d(deg_inter,   deg_prei)
    d_clust = cohens_d(clust_inter, clust_prei)

    pct_drop_rho   = 100 * (np.mean(rho_inter)   - np.mean(rho_prei))   / (np.mean(rho_inter)   + 1e-12)
    pct_drop_deg   = 100 * (np.mean(deg_inter)   - np.mean(deg_prei))   / (np.mean(deg_inter)   + 1e-12)
    pct_drop_clust = 100 * (np.mean(clust_inter) - np.mean(clust_prei)) / (np.mean(clust_inter) + 1e-12)

    print(f"  ρ:  inter={np.mean(rho_inter):.3f}  prei={np.mean(rho_prei):.3f}  "
          f"drop={pct_drop_rho:.1f}%  d={d_rho:.2f}  p={p_rho:.4f}")
    print(f"  d̄:  inter={np.mean(deg_inter):.3f}  prei={np.mean(deg_prei):.3f}  "
          f"drop={pct_drop_deg:.1f}%  d={d_deg:.2f}  p={p_deg:.4f}")
    print(f"  C:  inter={np.mean(clust_inter):.3f}  prei={np.mean(clust_prei):.3f}  "
          f"drop={pct_drop_clust:.1f}%  d={d_clust:.2f}  p={p_clust:.4f}")

    # ── Convergent validity: Phi(t) correlation with ρ(t) in preictal ───────
    phi_prei, _ = core.rolling_global_mi(prei_sel, win, step, k=K_NN,
                                         max_channels=MAX_CHANNELS)
    min_len = min(len(phi_prei), len(rho_prei))
    if min_len >= 3:
        try:
            r_phi_rho, p_phi_rho = stats.pearsonr(phi_prei[:min_len], rho_prei[:min_len])
            if not np.isfinite(r_phi_rho):
                r_phi_rho, p_phi_rho = 0.0, 1.0
        except Exception:
            r_phi_rho, p_phi_rho = 0.0, 1.0
    else:
        r_phi_rho, p_phi_rho = 0.0, 1.0

    print(f"  Convergent validity Phi(t)~ρ(t): r={r_phi_rho:.3f}  p={p_phi_rho:.4f}")

    # ── Save numpy arrays ────────────────────────────────────────────────────
    np.save(f"/kaggle/working/rho_inter_{subj_id}.npy",   rho_inter)
    np.save(f"/kaggle/working/rho_prei_{subj_id}.npy",    rho_prei)
    np.save(f"/kaggle/working/deg_inter_{subj_id}.npy",   deg_inter)
    np.save(f"/kaggle/working/deg_prei_{subj_id}.npy",    deg_prei)
    np.save(f"/kaggle/working/clust_inter_{subj_id}.npy", clust_inter)
    np.save(f"/kaggle/working/clust_prei_{subj_id}.npy",  clust_prei)

    # ── Per-patient plot ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)

    t_i = np.arange(len(rho_inter))   * STEP_SEC
    t_p = np.arange(len(rho_prei))    * STEP_SEC - t_seizure

    for ax, metric_i, metric_p, label, color in zip(
        axes,
        [rho_inter, deg_inter, clust_inter],
        [rho_prei,  deg_prei,  clust_prei],
        ["Density ρ(t)", "Mean degree d̄(t)", "Clustering C(t)"],
        ["steelblue", "darkorange", "mediumseagreen"],
    ):
        ax.plot(t_i, metric_i, color=color, alpha=0.5, lw=1, label="interictal")
        ax.axhline(np.mean(metric_i), color=color, ls="--", alpha=0.7,
                   label=f"inter mean={np.mean(metric_i):.3f}")

    # Second pass: overlay preictal on same axes with different x-axis
    fig2, axes2 = plt.subplots(3, 1, figsize=(12, 10))
    for ax2, mi, mp, label, color in zip(
        axes2,
        [rho_inter, deg_inter, clust_inter],
        [rho_prei,  deg_prei,  clust_prei],
        ["Density ρ(t)", "Mean degree d̄(t)", "Clustering C(t)"],
        ["steelblue", "darkorange", "mediumseagreen"],
    ):
        t_ip = np.arange(len(mi)) * STEP_SEC
        t_pp = np.arange(len(mp)) * STEP_SEC - t_seizure
        ax2.plot(t_ip, mi, color=color, alpha=0.4, lw=1.2, label="interictal")
        ax2.plot(t_pp, mp, color="black", lw=1.5, label="preictal")
        ax2.axhline(np.mean(mi), color=color, ls="--", alpha=0.6)
        ax2.axvline(0, color="red", lw=1.5, ls=":", label="seizure onset")
        ax2.set(ylabel=label)
        ax2.legend(fontsize=7)

    axes2[-1].set_xlabel("Time rel. to seizure (s)")
    fig2.suptitle(f"{subj_id} — Network metrics ({rec_type}, {n_ch} ch)", y=1.01)
    plt.tight_layout()
    fig2.savefig(f"/kaggle/working/graph_plot_{subj_id}.png", dpi=120)
    plt.close("all")

    all_results.append({
        "patient_id":      subj_id,
        "rec_type":        rec_type,
        "n_channels":      int(n_ch),
        "fs":              float(fs),
        # density
        "mu_rho_inter":    round(float(np.mean(rho_inter)),   5),
        "mu_rho_prei":     round(float(np.mean(rho_prei)),    5),
        "pct_drop_rho":    round(float(pct_drop_rho),         3),
        "cohens_d_rho":    round(float(d_rho),                4),
        "p_rho":           round(float(p_rho),                5),
        # mean degree
        "mu_deg_inter":    round(float(np.mean(deg_inter)),   5),
        "mu_deg_prei":     round(float(np.mean(deg_prei)),    5),
        "pct_drop_deg":    round(float(pct_drop_deg),         3),
        "cohens_d_deg":    round(float(d_deg),                4),
        "p_deg":           round(float(p_deg),                5),
        # clustering
        "mu_clust_inter":  round(float(np.mean(clust_inter)), 5),
        "mu_clust_prei":   round(float(np.mean(clust_prei)),  5),
        "pct_drop_clust":  round(float(pct_drop_clust),       3),
        "cohens_d_clust":  round(float(d_clust),              4),
        "p_clust":         round(float(p_clust),              5),
        # convergent validity
        "r_phi_rho":       round(float(r_phi_rho),            4),
        "p_phi_rho":       round(float(p_phi_rho),            4),
    })
    patients_done += 1

# ---------------------------------------------------------------------------
# Aggregate cross-patient analysis
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("AGGREGATE: Cross-patient Wilcoxon signed-rank test (H2)")
print("=" * 60)

if not all_results:
    print("WARNING: No patients processed.")
    summary = {"network_fragmentation_detected": False, "n_patients_processed": 0}
    core.format_result("network_fragmentation_detected", False)
    core.save_results(summary, "/kaggle/working/phase2a_results.json")
else:
    # Pair-wise: for each patient, compare mean_inter > mean_prei
    # This is the signed-rank test across patients (between-patient level)
    rho_diffs   = [r["mu_rho_inter"]   - r["mu_rho_prei"]   for r in all_results]
    deg_diffs   = [r["mu_deg_inter"]   - r["mu_deg_prei"]   for r in all_results]
    clust_diffs = [r["mu_clust_inter"] - r["mu_clust_prei"] for r in all_results]

    def safe_wilcoxon(diffs, alternative="greater"):
        try:
            _, p = stats.wilcoxon(diffs, alternative=alternative)
            return float(p) if np.isfinite(p) else 1.0
        except Exception:
            return 1.0

    p_wil_rho   = safe_wilcoxon(rho_diffs)
    p_wil_deg   = safe_wilcoxon(deg_diffs)
    p_wil_clust = safe_wilcoxon(clust_diffs)

    n_total = len(all_results)
    n_rho_sig   = sum(1 for r in all_results if r["p_rho"]   < 0.05)
    n_deg_sig   = sum(1 for r in all_results if r["p_deg"]   < 0.05)
    n_clust_sig = sum(1 for r in all_results if r["p_clust"] < 0.05)
    n_all_sig   = sum(1 for r in all_results
                      if r["p_rho"] < 0.05 and r["p_deg"] < 0.05 and r["p_clust"] < 0.05)

    # Use nanmean + clip Cohen's d at ±100 to prevent blowup from near-zero-std patients
    def safe_mean(vals, clip=100.0):
        arr = np.clip(np.array(vals, dtype=float), -clip, clip)
        return float(np.nanmean(arr))

    mean_d_rho   = safe_mean([r["cohens_d_rho"]   for r in all_results])
    mean_d_deg   = safe_mean([r["cohens_d_deg"]   for r in all_results])
    mean_d_clust = safe_mean([r["cohens_d_clust"] for r in all_results])

    mean_drop_rho   = float(np.nanmean([r["pct_drop_rho"]   for r in all_results]))
    mean_drop_deg   = float(np.nanmean([r["pct_drop_deg"]   for r in all_results]))
    mean_drop_clust = float(np.nanmean([r["pct_drop_clust"] for r in all_results]))

    mean_r_phi_rho = float(np.nanmean([r["r_phi_rho"] for r in all_results]))

    # Fragmentation detected if all three metrics show significant group-level drop
    frag_detected = (p_wil_rho < 0.05 and p_wil_deg < 0.05 and p_wil_clust < 0.05)

    # Subgroup means
    ecog_rho_drop = [r["pct_drop_rho"] for r in all_results if r["rec_type"] == "ECoG"]
    seeg_rho_drop = [r["pct_drop_rho"] for r in all_results if r["rec_type"] == "SEEG"]

    print(f"\nN patients: {n_total}")
    print(f"Density ρ:    Wilcoxon p={p_wil_rho:.4f}  mean_d={mean_d_rho:.3f}  "
          f"mean_drop={mean_drop_rho:.1f}%  n_sig={n_rho_sig}/{n_total}")
    print(f"Mean degree:  Wilcoxon p={p_wil_deg:.4f}  mean_d={mean_d_deg:.3f}  "
          f"mean_drop={mean_drop_deg:.1f}%  n_sig={n_deg_sig}/{n_total}")
    print(f"Clustering:   Wilcoxon p={p_wil_clust:.4f}  mean_d={mean_d_clust:.3f}  "
          f"mean_drop={mean_drop_clust:.1f}%  n_sig={n_clust_sig}/{n_total}")
    print(f"All 3 sig:    {n_all_sig}/{n_total} patients")
    print(f"Phi~rho convergent validity: mean r={mean_r_phi_rho:.3f}")
    print(f"Network fragmentation detected: {frag_detected}")

    # ── Aggregate violin/box plot ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, key_i, key_p, label, color in zip(
        axes,
        ["mu_rho_inter",   "mu_deg_inter",   "mu_clust_inter"],
        ["mu_rho_prei",    "mu_deg_prei",    "mu_clust_prei"],
        ["Density ρ",      "Mean degree d̄",  "Clustering C"],
        ["steelblue",      "darkorange",      "mediumseagreen"],
    ):
        vals_i = [r[key_i] for r in all_results]
        vals_p = [r[key_p] for r in all_results]
        parts = ax.violinplot([vals_i, vals_p], positions=[0, 1], showmedians=True)
        for body in parts["bodies"]:
            body.set_facecolor(color)
            body.set_alpha(0.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Interictal", "Preictal"])
        ax.set_ylabel(label)
        ax.set_title(f"{label}\np_wilcoxon={p_wil_rho:.4f}" if "ρ" in label
                     else (f"{label}\np_wilcoxon={p_wil_deg:.4f}" if "d̄" in label
                           else f"{label}\np_wilcoxon={p_wil_clust:.4f}"))
    fig.suptitle(f"Phase 2A — Network Fragmentation (N={n_total} patients)")
    plt.tight_layout()
    fig.savefig("/kaggle/working/phase2a_violin.png", dpi=150)
    plt.close()

    # ── Save per-patient CSV ─────────────────────────────────────────────────
    import csv as csv_mod
    fieldnames = list(all_results[0].keys())
    with open("/kaggle/working/phase2a_patients.csv", "w", newline="") as f:
        w = csv_mod.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_results)
    print(f"  CSV saved: /kaggle/working/phase2a_patients.csv")

    summary = {
        "n_patients_processed":         n_total,
        "network_fragmentation_detected": frag_detected,
        "wilcoxon_p_density":           round(p_wil_rho,   5),
        "wilcoxon_p_mean_degree":       round(p_wil_deg,   5),
        "wilcoxon_p_clustering":        round(p_wil_clust, 5),
        "mean_cohens_d_density":        round(mean_d_rho,   4),
        "mean_cohens_d_mean_degree":    round(mean_d_deg,   4),
        "mean_cohens_d_clustering":     round(mean_d_clust, 4),
        "mean_pct_drop_density":        round(mean_drop_rho,   3),
        "mean_pct_drop_mean_degree":    round(mean_drop_deg,   3),
        "mean_pct_drop_clustering":     round(mean_drop_clust, 3),
        "n_patients_all_3_sig":         n_all_sig,
        "pct_patients_all_3_sig":       round(100 * n_all_sig / n_total, 1),
        "mean_r_phi_rho":               round(mean_r_phi_rho, 4),
        "ecog_mean_pct_drop_density":   round(float(np.mean(ecog_rho_drop)), 3) if ecog_rho_drop else None,
        "seeg_mean_pct_drop_density":   round(float(np.mean(seeg_rho_drop)), 3) if seeg_rho_drop else None,
        "per_patient":                  all_results,
    }

    core.format_result("n_patients_processed",         n_total)
    core.format_result("network_fragmentation_detected", frag_detected)
    core.format_result("wilcoxon_p_density",           summary["wilcoxon_p_density"])
    core.format_result("wilcoxon_p_mean_degree",       summary["wilcoxon_p_mean_degree"])
    core.format_result("wilcoxon_p_clustering",        summary["wilcoxon_p_clustering"])
    core.format_result("mean_cohens_d_density",        summary["mean_cohens_d_density"])
    core.format_result("mean_pct_drop_density",        summary["mean_pct_drop_density"])
    core.format_result("pct_patients_all_3_sig",       summary["pct_patients_all_3_sig"])
    core.format_result("mean_r_phi_rho",               summary["mean_r_phi_rho"])

    core.save_results(summary, "/kaggle/working/phase2a_results.json")
    print(f"\nProcessed {n_total} patients.")
    print("Done.")
