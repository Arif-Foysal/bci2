"""
ieeg-ib phase1c — Full 57-patient MI collapse analysis (H1 validation).

Pilot (phase1b) confirmed:
  - ECoG patients: 30-37% MI drop preictal vs interictal (Cohen's d up to 1.40)
  - MI is already collapsed at t=0 of the 2-min preictal window → collapse
    occurred >2 min before seizure onset; threshold-crossing τ_lead is
    not estimable from this dataset design.
  - Primary test: Wilcoxon signed-rank across all patients (μ_inter > μ_prei)

This notebook computes Phi(t) for all 57 patients, runs the statistical
analysis, and separates SEEG vs ECoG subgroups.

Outputs:
  phase1c_results.json       — aggregate stats + per-patient table
  phase1c_patients.csv       — tidy CSV for downstream analysis
  phi_plot_summary.png       — violin plot: μ_inter vs μ_prei by electrode type
  phi_plot_<pid>.png         — per-patient Phi(t) traces (ECoG only, max 10)
"""

import os, sys, csv, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config (do not increase MAX_CHANNELS without re-benchmarking timing)
# ---------------------------------------------------------------------------
MAX_CHANNELS   = 16
WINDOW_SEC     = 5
STEP_SEC       = 5
K_NN           = 5
MAX_PATIENTS   = 57   # set lower for a faster debug run

# ---------------------------------------------------------------------------
# Core library
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
from scipy import stats as scipy_stats

DATASET_ROOT = None
for dp, dirs, _ in os.walk("/kaggle/input"):
    if any(d.startswith("sub-") for d in dirs):
        DATASET_ROOT = dp
        break

print("=" * 60)
print("PHASE 1C: Full 57-patient MI analysis")
print("=" * 60)
print(f"Dataset: {DATASET_ROOT}")

subject_dirs = sorted(glob.glob(os.path.join(DATASET_ROOT, "sub-*")))
print(f"Subjects: {len(subject_dirs)}")

# ---------------------------------------------------------------------------
# Helpers (same as phase1b)
# ---------------------------------------------------------------------------
def get_seizure_onset(events_tsv):
    if not events_tsv or not os.path.exists(events_tsv):
        return None
    with open(events_tsv, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            tt = row.get("trial_type", "").lower()
            if "sz onset" in tt or "seizure onset" in tt or (
                    "sz" in tt and "offset" not in tt):
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


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
all_rows = []       # one dict per patient for CSV + JSON
n_done   = 0
ecog_plots_saved = 0

for subj_dir in subject_dirs[:MAX_PATIENTS]:
    subj_id = os.path.basename(subj_dir)
    ieeg_dir = os.path.join(subj_dir, "ses-presurgery", "ieeg")
    if not os.path.isdir(ieeg_dir):
        print(f"{subj_id}: no ieeg dir, skip")
        continue

    ictal_edfs = sorted(f for f in os.listdir(ieeg_dir)
                        if "task-ictal" in f and f.endswith("_ieeg.edf"))
    inter_edfs = sorted(f for f in os.listdir(ieeg_dir)
                        if "task-interictal" in f and f.endswith("_ieeg.edf"))
    if not ictal_edfs or not inter_edfs:
        print(f"{subj_id}: missing ictal or interictal EDF, skip")
        continue

    ictal_edf  = os.path.join(ieeg_dir, ictal_edfs[0])
    inter_edf  = os.path.join(ieeg_dir, inter_edfs[0])
    events_tsv = ictal_edf.replace("_ieeg.edf", "_events.tsv")

    t_seizure = get_seizure_onset(events_tsv)
    if t_seizure is None:
        print(f"{subj_id}: no seizure onset, skip")
        continue

    # Electrode type from filename: acq-seeg vs acq-ecog
    elec_type = "SEEG" if "acq-seeg" in ictal_edfs[0] else "ECoG"

    print(f"\n{subj_id} [{elec_type}] t_sz={t_seizure:.0f}s", flush=True)

    try:
        inter_data, _, fs = load_edf(inter_edf)
        prei_data, _, _   = load_edf(ictal_edf, t_start=0.0, t_end=t_seizure)
    except Exception as e:
        print(f"  load error: {e}")
        continue

    n_common = min(inter_data.shape[0], prei_data.shape[0])
    top_idx  = np.argsort(np.var(inter_data, axis=1))[-MAX_CHANNELS:]
    top_idx  = top_idx[top_idx < n_common][:MAX_CHANNELS]
    inter_sel = inter_data[top_idx]
    prei_sel  = prei_data[top_idx]
    n_ch = len(top_idx)

    win  = int(WINDOW_SEC * fs)
    step = int(STEP_SEC   * fs)

    phi_inter, _ = core.rolling_global_mi(inter_sel, win, step, k=K_NN,
                                           max_channels=MAX_CHANNELS)
    phi_prei, prei_centers = core.rolling_global_mi(prei_sel, win, step,
                                                     k=K_NN, max_channels=MAX_CHANNELS)

    if len(phi_inter) < 3 or len(phi_prei) < 3:
        print(f"  too few windows, skip")
        continue

    # Quality check: flag unreliable interictal baselines before analysis
    baseline_ok, baseline_reason = core.baseline_quality_check(phi_inter)
    if not baseline_ok:
        print(f"  SKIP — bad baseline: {baseline_reason}", flush=True)
        continue

    # Statistical analysis
    trend = core.preictal_trend_analysis(phi_prei, phi_inter, step_sec=STEP_SEC)

    row = {
        "patient_id":           subj_id,
        "electrode_type":       elec_type,
        "n_channels":           int(n_ch),
        "fs":                   float(fs),
        "n_inter_windows":      int(len(phi_inter)),
        "n_prei_windows":       int(len(phi_prei)),
        "mu_inter":             trend["mu_inter"],
        "mu_prei":              trend["mu_prei"],
        "pct_drop":             trend["pct_drop"],
        "cohens_d":             trend["cohens_d"],
        "slope_nats_per_s":     trend["slope_nats_per_s"],
        "mk_tau":               trend["mk_tau"],
        "mk_pvalue":            trend["mk_pvalue"],
        "early_mean":           trend["early_mean"],
        "late_mean":            trend["late_mean"],
        "early_late_pvalue":    trend["early_late_pvalue"],
        "r_squared":            trend["r_squared"],
        "p_slope":              trend["p_slope"],
        "baseline_ok":          True,
    }
    all_rows.append(row)
    n_done += 1

    print(f"  μ_inter={trend['mu_inter']:.4f}  μ_prei={trend['mu_prei']:.4f}  "
          f"drop={trend['pct_drop']:+.1f}%  d={trend['cohens_d']:.2f}  "
          f"MK_τ={trend['mk_tau']:.3f}", flush=True)

    # Per-patient plot (ECoG only, max 10 to save disk)
    if elec_type == "ECoG" and ecog_plots_saved < 10:
        fig, axes = plt.subplots(2, 1, figsize=(11, 6))
        t_i = np.arange(len(phi_inter)) * STEP_SEC
        axes[0].plot(t_i, phi_inter, "steelblue", lw=1.2)
        axes[0].axhline(trend["mu_inter"], color="gray", ls="--", lw=1,
                        label=f"μ_inter={trend['mu_inter']:.4f}")
        axes[0].set(xlabel="Time (s)", ylabel="Φ(t) [nats]",
                    title=f"{subj_id} ECoG — Interictal (N={n_ch}ch)")
        axes[0].legend(fontsize=8)

        t_p = prei_centers / fs - t_seizure
        axes[1].plot(t_p, phi_prei, "darkorange", lw=1.2)
        axes[1].axhline(trend["mu_prei"], color="sienna", ls="--", lw=1,
                        label=f"μ_prei={trend['mu_prei']:.4f}  drop={trend['pct_drop']:.1f}%")
        axes[1].axhline(trend["mu_inter"], color="gray", ls=":", lw=1,
                        label="μ_inter (reference)")
        axes[1].axvline(0, color="black", lw=2, label="seizure onset")
        axes[1].set(xlabel="Time rel. to seizure (s)", ylabel="Φ(t) [nats]",
                    title=f"Preictal  d={trend['cohens_d']:.2f}  MKτ={trend['mk_tau']:.3f}")
        axes[1].legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(f"/kaggle/working/phi_plot_{subj_id}.png", dpi=120)
        plt.close()
        ecog_plots_saved += 1

# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("AGGREGATE STATISTICS")
print("=" * 60)

if not all_rows:
    print("No patients processed.")
    core.format_result("n_patients_processed", 0)
    core.save_results({"n_patients_processed": 0}, "/kaggle/working/phase1c_results.json")
    sys.exit(0)

mu_inter_arr = np.array([r["mu_inter"]   for r in all_rows])
mu_prei_arr  = np.array([r["mu_prei"]    for r in all_rows])
pct_drops    = np.array([r["pct_drop"]   for r in all_rows])
cohens_ds    = np.array([r["cohens_d"]   for r in all_rows])
mk_taus      = np.array([r["mk_tau"]     for r in all_rows])
types        = np.array([r["electrode_type"] for r in all_rows])

# Wilcoxon signed-rank: H0: μ_inter == μ_prei
w_stat, w_p = scipy_stats.wilcoxon(mu_inter_arr, mu_prei_arr, alternative="greater")

# Subgroup stats
for etype in ("ECoG", "SEEG"):
    mask = types == etype
    if mask.sum() == 0:
        continue
    d_sub     = pct_drops[mask]
    cohens_sub = cohens_ds[mask]
    if mask.sum() >= 2:
        _, wp_sub = scipy_stats.wilcoxon(mu_inter_arr[mask], mu_prei_arr[mask],
                                          alternative="greater")
    else:
        wp_sub = float("nan")
    print(f"\n{etype} (n={mask.sum()}):")
    print(f"  mean pct_drop = {d_sub.mean():.1f}% ± {d_sub.std():.1f}%")
    print(f"  mean Cohen's d = {cohens_sub.mean():.3f} ± {cohens_sub.std():.3f}")
    print(f"  Wilcoxon p = {wp_sub:.4f}")

print(f"\nAll patients (n={len(all_rows)}):")
print(f"  Wilcoxon signed-rank: W={w_stat:.1f}, p={w_p:.4f}")
print(f"  mean pct_drop = {pct_drops.mean():.1f}% ± {pct_drops.std():.1f}%")
print(f"  mean Cohen's d = {cohens_ds.mean():.3f} ± {cohens_ds.std():.3f}")
print(f"  median MK_tau = {np.median(mk_taus):.4f}")
print(f"  pct patients with pct_drop > 10%: "
      f"{100*(pct_drops > 10).mean():.1f}%")

# ---------------------------------------------------------------------------
# Summary violin plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

for ax, metric, ylabel, title in [
    (axes[0], "pct_drop",  "MI drop (%)",  "% MI drop  (interictal → preictal)"),
    (axes[1], "cohens_d",  "Cohen's d",    "Effect size (Cohen's d)"),
]:
    ecog_vals = [r[metric] for r in all_rows if r["electrode_type"] == "ECoG"]
    seeg_vals = [r[metric] for r in all_rows if r["electrode_type"] == "SEEG"]
    data_to_plot = [v for v in [ecog_vals, seeg_vals] if v]
    labels = [l for l, v in [("ECoG", ecog_vals), ("SEEG", seeg_vals)] if v]
    parts = ax.violinplot(data_to_plot, showmedians=True)
    for i, c in enumerate(["steelblue", "coral"]):
        if i < len(parts["bodies"]):
            parts["bodies"][i].set_facecolor(c)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set(ylabel=ylabel, title=title)

plt.suptitle(f"Phase 1C: MI collapse — {len(all_rows)} patients\n"
             f"Wilcoxon p={w_p:.4f},  mean d={cohens_ds.mean():.2f}",
             fontsize=11)
plt.tight_layout()
plt.savefig("/kaggle/working/phi_plot_summary.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# Save CSV and JSON
# ---------------------------------------------------------------------------
if all_rows:
    keys = list(all_rows[0].keys())
    with open("/kaggle/working/phase1c_patients.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_rows)
    print("\nSaved phase1c_patients.csv")

ecog_mask = types == "ECoG"
seeg_mask = types == "SEEG"

summary = {
    "n_patients_processed": len(all_rows),
    "n_ecog":  int(ecog_mask.sum()),
    "n_seeg":  int(seeg_mask.sum()),
    "wilcoxon_W":         float(w_stat),
    "wilcoxon_p":         float(w_p),
    "mean_pct_drop_all":  round(float(pct_drops.mean()), 2),
    "std_pct_drop_all":   round(float(pct_drops.std()),  2),
    "mean_cohens_d_all":  round(float(cohens_ds.mean()), 3),
    "median_mk_tau":      round(float(np.median(mk_taus)), 4),
    "pct_patients_drop_gt10": round(float(100*(pct_drops > 10).mean()), 1),
    "ecog_mean_pct_drop": round(float(pct_drops[ecog_mask].mean()), 2) if ecog_mask.any() else None,
    "ecog_mean_cohens_d": round(float(cohens_ds[ecog_mask].mean()), 3) if ecog_mask.any() else None,
    "seeg_mean_pct_drop": round(float(pct_drops[seeg_mask].mean()), 2) if seeg_mask.any() else None,
    "seeg_mean_cohens_d": round(float(cohens_ds[seeg_mask].mean()), 3) if seeg_mask.any() else None,
    "per_patient": all_rows,
}

core.format_result("n_patients_processed", summary["n_patients_processed"])
core.format_result("n_ecog",              summary["n_ecog"])
core.format_result("n_seeg",              summary["n_seeg"])
core.format_result("wilcoxon_p",          f"{summary['wilcoxon_p']:.4f}")
core.format_result("mean_pct_drop_all",   summary["mean_pct_drop_all"])
core.format_result("mean_cohens_d_all",   summary["mean_cohens_d_all"])
core.format_result("pct_patients_drop_gt10", summary["pct_patients_drop_gt10"])
if ecog_mask.any():
    core.format_result("ecog_mean_pct_drop", summary["ecog_mean_pct_drop"])
    core.format_result("ecog_mean_cohens_d", summary["ecog_mean_cohens_d"])
if seeg_mask.any():
    core.format_result("seeg_mean_pct_drop", summary["seeg_mean_pct_drop"])
    core.format_result("seeg_mean_cohens_d", summary["seeg_mean_cohens_d"])

core.save_results(summary, "/kaggle/working/phase1c_results.json")
print("\nDone.")
