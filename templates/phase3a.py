"""
ieeg-ib phase3a — MI advance over signal variance (H3).

Hypothesis H3: The MI collapse precedes the variance collapse by at least 15
seconds in >80% of seizures, and MI and variance are uncorrelated (|r| < 0.4)
during interictal periods — establishing MI as an independent early biomarker.

This notebook:
  1. For each patient, loads interictal and preictal EDF
  2. Computes Phi(t) (MI-based signal) and V(t) (channel-mean variance)
  3. Detects collapse time for each: first window 2σ below interictal baseline
  4. Computes Δτ = t_collapse_V − t_collapse_Phi (positive = MI collapses first)
  5. Tests H3: Wilcoxon one-sided test that Phi collapses earlier than V
  6. Computes interictal Pearson correlation between Phi(t) and V(t)

Config:
  MAX_PATIENTS  = 57
  MAX_CHANNELS  = 16
  WINDOW_SEC    = 5
  STEP_SEC      = 5
  K_NN          = 5
  SIGMA_THRESH  = 2.0   (collapse threshold = mean - SIGMA * std)
  MIN_PERSIST   = 3     (relaxed; only 24 windows max in 120s at 5s step)
"""

import os, sys, csv, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MAX_PATIENTS  = 57
MAX_CHANNELS  = 16
WINDOW_SEC    = 5
STEP_SEC      = 5
K_NN          = 5
SIGMA_THRESH  = 2.0
MIN_PERSIST   = 3

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
print("PHASE 3A: MI advance over signal variance")
print("=" * 60)
print(f"Dataset root: {DATASET_ROOT}")

subject_dirs = sorted(glob.glob(os.path.join(DATASET_ROOT, "sub-*")))
print(f"Subjects found: {len(subject_dirs)}")


# ---------------------------------------------------------------------------
# Helpers
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
    names_upper = [c.upper() for c in ch_names]
    n_ecog = sum(1 for c in names_upper if any(p in c for p in ("G", "GRID", "STRIP")))
    n_seeg = sum(1 for c in names_upper
                 if any(c.startswith(p) for p in ("A", "B", "C", "D", "H", "OF", "AM", "HC")))
    if n_ecog > n_seeg:
        return "ECoG"
    elif n_seeg > 0:
        return "SEEG"
    return "Unknown"


def rolling_variance(data, window_samples, step_samples):
    """Mean channel variance in each rolling window. Returns V(t), centers."""
    n_channels, n_samples = data.shape
    starts = np.arange(0, n_samples - window_samples + 1, step_samples)
    V = np.empty(len(starts))
    centers = starts + window_samples // 2
    for w, s in enumerate(starts):
        V[w] = np.mean(np.var(data[:, s:s + window_samples], axis=1))
    return V, centers


def find_collapse(signal, baseline_signal, sigma=2.0, min_persist=3):
    """Return index of first persistent drop below (mu-sigma*std) of baseline.
    Returns None if no collapse detected."""
    mu  = np.mean(baseline_signal)
    std = np.std(baseline_signal)
    thr = mu - sigma * std
    below = signal < thr
    count = 0
    for i, b in enumerate(below):
        count = count + 1 if b else 0
        if count >= min_persist:
            return i - min_persist + 1
    return None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
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

    n_common = min(inter_data.shape[0], prei_data.shape[0])
    top_idx   = top_channels(inter_data[:n_common], MAX_CHANNELS)
    inter_sel = inter_data[top_idx]
    prei_sel  = prei_data[top_idx]
    n_ch      = len(top_idx)
    print(f"  {rec_type} | {n_ch} channels")

    win  = int(WINDOW_SEC * fs)
    step = int(STEP_SEC   * fs)

    # ── Compute Phi(t) — MI signal ───────────────────────────────────────────
    print("  Computing Phi(t) MI signal...")
    phi_inter, _       = core.rolling_global_mi(inter_sel, win, step, k=K_NN,
                                                 max_channels=MAX_CHANNELS)
    phi_prei,  prei_c  = core.rolling_global_mi(prei_sel,  win, step, k=K_NN,
                                                 max_channels=MAX_CHANNELS)

    # ── Compute V(t) — rolling variance signal ───────────────────────────────
    print("  Computing V(t) variance signal...")
    V_inter, _   = rolling_variance(inter_sel, win, step)
    V_prei,  v_c = rolling_variance(prei_sel,  win, step)

    if len(phi_inter) < 3 or len(phi_prei) < 3:
        print("  Too few windows, skipping")
        continue

    # ── Collapse detection ───────────────────────────────────────────────────
    phi_col_idx = find_collapse(phi_prei, phi_inter, sigma=SIGMA_THRESH,
                                min_persist=MIN_PERSIST)
    V_col_idx   = find_collapse(V_prei,   V_inter,   sigma=SIGMA_THRESH,
                                min_persist=MIN_PERSIST)

    phi_col_s = (float(phi_col_idx) * STEP_SEC) if phi_col_idx is not None else None
    V_col_s   = (float(V_col_idx)   * STEP_SEC) if V_col_idx   is not None else None

    # Δτ = t_collapse_V - t_collapse_Phi (positive = MI collapses first)
    delta_tau = None
    if phi_col_s is not None and V_col_s is not None:
        delta_tau = V_col_s - phi_col_s

    # ── Interictal correlation between Phi and V ─────────────────────────────
    min_len = min(len(phi_inter), len(V_inter))
    r_inter, p_r_inter = stats.pearsonr(phi_inter[:min_len], V_inter[:min_len])

    # ── Preictal trend comparison ────────────────────────────────────────────
    phi_trend = core.preictal_trend_analysis(phi_prei, phi_inter, step_sec=STEP_SEC)
    pct_drop_phi   = phi_trend["pct_drop"]
    cohens_d_phi   = phi_trend["cohens_d"]

    # Variance trend (manual calculation mirroring preictal_trend_analysis)
    mu_V_inter  = float(np.mean(V_inter))
    mu_V_prei   = float(np.mean(V_prei))
    pct_drop_V  = 100 * (mu_V_inter - mu_V_prei) / (mu_V_inter + 1e-12)
    cohens_d_V  = (mu_V_inter - mu_V_prei) / (float(np.std(V_inter)) + 1e-12)

    print(f"  Phi collapse: idx={phi_col_idx}  t={phi_col_s}s  "
          f"drop={pct_drop_phi:.1f}%  d={cohens_d_phi:.2f}")
    print(f"  V   collapse: idx={V_col_idx}    t={V_col_s}s  "
          f"drop={pct_drop_V:.1f}%  d={cohens_d_V:.2f}")
    print(f"  Δτ (V - Phi): {delta_tau}s")
    print(f"  Interictal r(Phi,V): {r_inter:.3f}  p={p_r_inter:.4f}")

    # ── Save arrays ──────────────────────────────────────────────────────────
    np.save(f"/kaggle/working/phi_inter_p3_{subj_id}.npy", phi_inter)
    np.save(f"/kaggle/working/phi_prei_p3_{subj_id}.npy",  phi_prei)
    np.save(f"/kaggle/working/V_inter_p3_{subj_id}.npy",   V_inter)
    np.save(f"/kaggle/working/V_prei_p3_{subj_id}.npy",    V_prei)

    # ── Per-patient overlay plot ─────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    t_p = np.arange(len(phi_prei)) * STEP_SEC - t_seizure
    t_v = np.arange(len(V_prei))   * STEP_SEC - t_seizure

    # Normalise both to [0,1] for overlay comparison
    phi_n = (phi_prei - phi_prei.min()) / (np.ptp(phi_prei) + 1e-12)
    V_n   = (V_prei   - V_prei.min())   / (np.ptp(V_prei)   + 1e-12)

    axes[0].plot(t_p, phi_n, color="steelblue", lw=1.5, label="Phi(t) [norm]")
    axes[0].plot(t_v, V_n,   color="darkorange", lw=1.5, label="V(t) [norm]")
    axes[0].axvline(0, color="red", lw=1.5, ls=":", label="seizure")
    if phi_col_s is not None:
        axes[0].axvline(phi_col_s - t_seizure, color="steelblue", ls="--",
                        label=f"Phi collapse (t={phi_col_s:.0f}s)")
    if V_col_s is not None:
        axes[0].axvline(V_col_s - t_seizure, color="darkorange", ls="--",
                        label=f"V collapse (t={V_col_s:.0f}s)")
    axes[0].set(xlabel="Time rel. seizure (s)", ylabel="Normalised signal",
                title=f"{subj_id} — Preictal MI vs Variance overlay")
    axes[0].legend(fontsize=8)

    axes[1].scatter([r_inter], [pct_drop_phi - pct_drop_V],
                    color="steelblue", s=60)
    axes[1].axhline(0, color="gray", ls="--")
    axes[1].set(xlabel="Interictal r(Phi, V)", ylabel="Δdrop: Phi% - V%",
                title=f"Single patient summary")

    plt.tight_layout()
    fig.savefig(f"/kaggle/working/p3_plot_{subj_id}.png", dpi=120)
    plt.close()

    all_results.append({
        "patient_id":       subj_id,
        "rec_type":         rec_type,
        "n_channels":       int(n_ch),
        "phi_col_s":        round(phi_col_s, 2) if phi_col_s is not None else None,
        "V_col_s":          round(V_col_s,   2) if V_col_s   is not None else None,
        "delta_tau_s":      round(delta_tau, 2) if delta_tau is not None else None,
        "phi_col_detected": phi_col_idx is not None,
        "V_col_detected":   V_col_idx   is not None,
        "pct_drop_phi":     round(pct_drop_phi, 3),
        "pct_drop_V":       round(pct_drop_V,   3),
        "cohens_d_phi":     round(cohens_d_phi, 4),
        "cohens_d_V":       round(cohens_d_V,   4),
        "r_inter_phi_V":    round(float(r_inter),    4),
        "p_r_inter":        round(float(p_r_inter),  4),
    })
    patients_done += 1

# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("AGGREGATE RESULTS (H3)")
print("=" * 60)

if not all_results:
    print("WARNING: No patients processed.")
    summary = {"pct_seizures_mi_leads_variance": 0.0, "n_patients_processed": 0}
    core.format_result("pct_seizures_mi_leads_variance", 0.0)
    core.save_results(summary, "/kaggle/working/phase3a_results.json")
else:
    import csv as csv_mod

    n_total = len(all_results)

    # H3a: MI collapses before V (Δτ > 0)
    both_detected = [r for r in all_results
                     if r["phi_col_detected"] and r["V_col_detected"]]
    n_both = len(both_detected)
    n_mi_leads = sum(1 for r in both_detected if r["delta_tau_s"] > 0)
    n_mi_leads_15 = sum(1 for r in both_detected
                        if r["delta_tau_s"] is not None and r["delta_tau_s"] > 15)

    pct_mi_leads    = 100 * n_mi_leads    / n_both if n_both > 0 else 0.0
    pct_mi_leads_15 = 100 * n_mi_leads_15 / n_both if n_both > 0 else 0.0

    delta_taus = [r["delta_tau_s"] for r in both_detected if r["delta_tau_s"] is not None]
    mean_delta_tau = float(np.mean(delta_taus)) if delta_taus else 0.0
    std_delta_tau  = float(np.std(delta_taus))  if delta_taus else 0.0

    # Wilcoxon test: Δτ > 0
    if len(delta_taus) >= 2:
        _, p_wil_delta = stats.wilcoxon(delta_taus, alternative="greater")
    else:
        p_wil_delta = 1.0

    # H3b: Interictal |r(Phi, V)| < 0.4
    r_vals = [abs(r["r_inter_phi_V"]) for r in all_results]
    mean_abs_r = float(np.mean(r_vals))
    pct_uncorr = 100 * sum(1 for r in r_vals if r < 0.4) / n_total

    # ECoG vs SEEG subgroups
    ecog_delta = [r["delta_tau_s"] for r in both_detected
                  if r["rec_type"] == "ECoG" and r["delta_tau_s"] is not None]
    seeg_delta = [r["delta_tau_s"] for r in both_detected
                  if r["rec_type"] == "SEEG" and r["delta_tau_s"] is not None]

    mean_pct_drop_phi = float(np.mean([r["pct_drop_phi"] for r in all_results]))
    mean_pct_drop_V   = float(np.mean([r["pct_drop_V"]   for r in all_results]))

    print(f"N patients processed:        {n_total}")
    print(f"Both collapses detected:     {n_both}/{n_total}")
    print(f"MI leads V (Δτ > 0):         {n_mi_leads}/{n_both}  ({pct_mi_leads:.1f}%)")
    print(f"MI leads V by >15s:          {n_mi_leads_15}/{n_both}  ({pct_mi_leads_15:.1f}%)")
    print(f"Mean Δτ:                     {mean_delta_tau:.1f}s ± {std_delta_tau:.1f}s")
    print(f"Wilcoxon p (Δτ > 0):         {p_wil_delta:.4f}")
    print(f"Mean |r(Phi,V)| interictal:  {mean_abs_r:.3f}")
    print(f"% patients |r| < 0.4:        {pct_uncorr:.1f}%")
    print(f"Mean pct drop Phi:           {mean_pct_drop_phi:.1f}%")
    print(f"Mean pct drop V:             {mean_pct_drop_V:.1f}%")

    # ── Aggregate scatter: Δτ vs rec_type ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for rec, color in [("ECoG", "steelblue"), ("SEEG", "darkorange")]:
        pts = [(r["delta_tau_s"], r["cohens_d_phi"]) for r in both_detected
               if r["rec_type"] == rec and r["delta_tau_s"] is not None]
        if pts:
            xs, ys = zip(*pts)
            ax.scatter(xs, ys, label=rec, color=color, alpha=0.7, s=50)
    ax.axvline(0, color="gray", ls="--")
    ax.axvline(15, color="red", ls=":", label="Δτ=15s target")
    ax.set(xlabel="Δτ (s)  [positive = MI collapses first]",
           ylabel="Cohen's d (MI drop)",
           title="Δτ by recording type")
    ax.legend()

    ax2 = axes[1]
    ax2.scatter([r["r_inter_phi_V"] for r in all_results],
                [r["pct_drop_phi"]  for r in all_results],
                alpha=0.6, s=40, color="steelblue")
    ax2.axvline(-0.4, color="red", ls=":")
    ax2.axvline(0.4,  color="red", ls=":", label="|r|=0.4 boundary")
    ax2.set(xlabel="Interictal r(Phi, V)",
            ylabel="Preictal Phi drop (%)",
            title="Independence vs. MI signal strength")
    ax2.legend()

    fig.suptitle(f"Phase 3A — MI vs Variance (N={n_total})")
    plt.tight_layout()
    fig.savefig("/kaggle/working/phase3a_summary.png", dpi=150)
    plt.close()

    # ── CSV ──────────────────────────────────────────────────────────────────
    fieldnames = list(all_results[0].keys())
    with open("/kaggle/working/phase3a_patients.csv", "w", newline="") as f:
        w = csv_mod.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_results)

    summary = {
        "n_patients_processed":          n_total,
        "n_both_collapses_detected":     n_both,
        "n_mi_leads_variance":           n_mi_leads,
        "pct_seizures_mi_leads_variance": round(pct_mi_leads,    2),
        "n_mi_leads_by_15s":             n_mi_leads_15,
        "pct_mi_leads_by_15s":           round(pct_mi_leads_15,  2),
        "mean_delta_tau_s":              round(mean_delta_tau,    2),
        "std_delta_tau_s":               round(std_delta_tau,     2),
        "wilcoxon_p_delta_tau":          round(p_wil_delta,       5),
        "mean_abs_r_phi_V_interictal":   round(mean_abs_r,        4),
        "pct_patients_r_below_0pt4":     round(pct_uncorr,        2),
        "mean_pct_drop_phi":             round(mean_pct_drop_phi, 3),
        "mean_pct_drop_V":               round(mean_pct_drop_V,   3),
        "ecog_mean_delta_tau_s":         round(float(np.mean(ecog_delta)), 2) if ecog_delta else None,
        "seeg_mean_delta_tau_s":         round(float(np.mean(seeg_delta)), 2) if seeg_delta else None,
        "per_patient":                   all_results,
    }

    core.format_result("n_patients_processed",          n_total)
    core.format_result("pct_seizures_mi_leads_variance", summary["pct_seizures_mi_leads_variance"])
    core.format_result("pct_mi_leads_by_15s",           summary["pct_mi_leads_by_15s"])
    core.format_result("mean_delta_tau_s",              summary["mean_delta_tau_s"])
    core.format_result("wilcoxon_p_delta_tau",          summary["wilcoxon_p_delta_tau"])
    core.format_result("mean_abs_r_phi_V_interictal",   summary["mean_abs_r_phi_V_interictal"])
    core.format_result("pct_patients_r_below_0pt4",     summary["pct_patients_r_below_0pt4"])

    core.save_results(summary, "/kaggle/working/phase3a_results.json")
    print(f"\nDone. Processed {n_total} patients.")
