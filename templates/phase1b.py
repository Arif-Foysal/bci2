"""
ieeg-ib phase1b — Rolling MI collapse on pilot patients (HUP iEEG dataset).

Dataset structure (from diagnostic):
  - task-ictal EDFs: ~120s preictal + ictal, seizure onset at t=120.0s
  - task-interictal EDFs: ~300s separate interictal recordings
  - Events: *_events.tsv with BOM, trial_type="sz onset" / "sz offset"
  - Sampling rate: 500 Hz (no resampling needed)

This notebook:
  1. Loads task-interictal EDF → interictal Phi(t) baseline
  2. Loads task-ictal EDF → preictal Phi(t) (t=0 to t_seizure)
  3. Detects MI collapse, computes tau_lead = t_seizure - t_collapse
  4. Saves per-patient Phi arrays, plots, and aggregate JSON

Config:
  N_PILOT_PATIENTS = 3
  MAX_CHANNELS     = 16   (top by variance — controls speed)
  WINDOW_SEC       = 5    (rolling window)
  STEP_SEC         = 5    (step between windows)
  K_NN             = 5    (Kraskov k)
"""

import os, sys, csv, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_PILOT_PATIENTS = 3
MAX_CHANNELS     = 16
WINDOW_SEC       = 5
STEP_SEC         = 5
K_NN             = 5

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
# Helpers
# ---------------------------------------------------------------------------
DATASET_ROOT = None
for dp, dirs, _ in os.walk("/kaggle/input"):
    if any(d.startswith("sub-") for d in dirs):
        DATASET_ROOT = dp
        break

print("=" * 60)
print("PHASE 1B: Rolling MI collapse — pilot patients")
print("=" * 60)
print(f"Dataset root: {DATASET_ROOT}")

subject_dirs = sorted(glob.glob(os.path.join(DATASET_ROOT, "sub-*")))
print(f"Subjects: {len(subject_dirs)}")


def get_seizure_onset(events_path):
    """Read events.tsv (UTF-8-BOM safe) and return onset of 'sz onset' in seconds."""
    if not events_path or not os.path.exists(events_path):
        return None
    with open(events_path, encoding="utf-8-sig") as f:   # utf-8-sig strips BOM
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            trial_type = row.get("trial_type", "").lower()
            if "sz onset" in trial_type or "seizure onset" in trial_type:
                try:
                    return float(row["onset"])
                except (KeyError, ValueError):
                    continue
            # Fallback: any ictal-keyword row
            if any(kw in trial_type for kw in ("sz", "seizure", "ictal")):
                try:
                    return float(row["onset"])
                except (KeyError, ValueError):
                    continue
    return None


def load_edf(path, t_start=None, t_end=None):
    """Load EDF, crop, apply bandpass + notch + CAR, return (data, ch_names, fs)."""
    raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
    if t_start is not None or t_end is not None:
        raw.crop(tmin=t_start or 0.0, tmax=t_end)
    raw.load_data(verbose=False)

    # Keep only SEEG/ECoG channels
    raw.pick_types(eeg=True, ecog=True, seeg=True, misc=False,
                   stim=False, exclude="bads")

    raw.filter(0.5, 200.0, method="iir", verbose=False)
    raw.notch_filter(np.arange(60, 201, 60), verbose=False)
    raw.set_eeg_reference("average", verbose=False)

    return raw.get_data(), raw.ch_names, raw.info["sfreq"]


def top_channels(data, n):
    """Return row indices of n highest-variance channels."""
    return np.argsort(np.var(data, axis=1))[-n:]


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
all_results = []
patients_done = 0

for subj_dir in subject_dirs:
    if patients_done >= N_PILOT_PATIENTS:
        break
    subj_id = os.path.basename(subj_dir)

    # ── Find ictal and interictal EDFs ──────────────────────────────────────
    ieeg_dir = os.path.join(subj_dir, "ses-presurgery", "ieeg")
    if not os.path.isdir(ieeg_dir):
        continue

    ictal_edfs = sorted(f for f in os.listdir(ieeg_dir)
                        if "task-ictal" in f and f.endswith("_ieeg.edf"))
    inter_edfs = sorted(f for f in os.listdir(ieeg_dir)
                        if "task-interictal" in f and f.endswith("_ieeg.edf"))

    if not ictal_edfs or not inter_edfs:
        print(f"\n{subj_id}: missing ictal or interictal EDF, skipping")
        continue

    ictal_edf  = os.path.join(ieeg_dir, ictal_edfs[0])   # first ictal run
    inter_edf  = os.path.join(ieeg_dir, inter_edfs[0])   # first interictal run
    events_tsv = ictal_edf.replace("_ieeg.edf", "_events.tsv")

    # ── Parse seizure onset ─────────────────────────────────────────────────
    t_seizure = get_seizure_onset(events_tsv)
    if t_seizure is None:
        print(f"\n{subj_id}: no seizure onset in events file, skipping")
        continue

    print(f"\n{'='*50}")
    print(f"Patient : {subj_id}")
    print(f"Ictal   : {ictal_edfs[0]}  (t_seizure={t_seizure:.1f}s)")
    print(f"Interict: {inter_edfs[0]}")

    # ── Load data ────────────────────────────────────────────────────────────
    try:
        print("  Loading interictal epoch...")
        inter_data, ch_names, fs = load_edf(inter_edf)
        print(f"  Interictal: {inter_data.shape}  fs={fs:.0f}Hz")

        print("  Loading preictal epoch (t=0 to t_seizure)...")
        prei_data, _, _ = load_edf(ictal_edf, t_start=0.0, t_end=t_seizure)
        print(f"  Preictal:   {prei_data.shape}")
    except Exception as e:
        print(f"  Load error: {e}")
        continue

    # ── Channel selection (top MAX_CHANNELS by variance on interictal) ──────
    top_idx = top_channels(inter_data, MAX_CHANNELS)
    # Align: only keep channels that exist in both recordings
    n_common = min(inter_data.shape[0], prei_data.shape[0])
    top_idx  = top_idx[top_idx < n_common][:MAX_CHANNELS]

    inter_sel = inter_data[top_idx]
    prei_sel  = prei_data[top_idx]
    n_ch = len(top_idx)
    print(f"  Using {n_ch} channels (top variance)")

    # ── Rolling Phi(t) ───────────────────────────────────────────────────────
    win  = int(WINDOW_SEC * fs)
    step = int(STEP_SEC   * fs)

    print("  Computing interictal Phi(t)...")
    phi_inter, _ = core.rolling_global_mi(inter_sel, win, step, k=K_NN,
                                           max_channels=MAX_CHANNELS)
    print(f"  → {len(phi_inter)} windows, mean={np.mean(phi_inter):.4f}")

    print("  Computing preictal Phi(t)...")
    phi_prei, prei_centers = core.rolling_global_mi(prei_sel, win, step, k=K_NN,
                                                     max_channels=MAX_CHANNELS)
    print(f"  → {len(phi_prei)} windows, mean={np.mean(phi_prei):.4f}")

    if len(phi_inter) < 3 or len(phi_prei) < 3:
        print("  Too few windows, skipping patient")
        continue

    # ── Collapse detection ───────────────────────────────────────────────────
    mu_inter  = float(np.mean(phi_inter))
    sig_inter = float(np.std(phi_inter))
    threshold = mu_inter - 2.0 * sig_inter

    # Walk preictal Phi(t) from start; declare collapse when 10 consecutive
    # windows stay below threshold
    below = phi_prei < threshold
    count, collapse_idx = 0, None
    for i, b in enumerate(below):
        count = count + 1 if b else 0
        if count >= min(10, len(phi_prei) // 3):   # relax persistence for short clips
            collapse_idx = i - count + 1
            break

    tau_lead = None
    if collapse_idx is not None:
        t_collapse_s = float(prei_centers[collapse_idx]) / fs
        tau_lead = float(t_seizure - t_collapse_s)
        print(f"  Collapse at t={t_collapse_s:.1f}s  tau_lead={tau_lead:.1f}s")
    else:
        print(f"  No persistent collapse detected  "
              f"(threshold={threshold:.4f}, min_prei={phi_prei.min():.4f})")

    # ── Persist arrays ───────────────────────────────────────────────────────
    np.save(f"/kaggle/working/phi_inter_{subj_id}.npy", phi_inter)
    np.save(f"/kaggle/working/phi_prei_{subj_id}.npy",  phi_prei)

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))

    t_i = np.arange(len(phi_inter)) * STEP_SEC
    axes[0].plot(t_i, phi_inter, "steelblue", lw=1.5)
    axes[0].axhline(mu_inter,  color="gray", ls="--", alpha=0.7, label=f"mean={mu_inter:.3f}")
    axes[0].axhline(threshold, color="red",  ls=":",  alpha=0.8, label=f"threshold={threshold:.3f}")
    axes[0].set(xlabel="Time (s)", ylabel="Φ(t) [nats]",
                title=f"{subj_id} — Interictal baseline  (N={n_ch} ch)")
    axes[0].legend(fontsize=8)

    t_p = prei_centers / fs - t_seizure   # negative = before seizure
    axes[1].plot(t_p, phi_prei, "darkorange", lw=1.5)
    axes[1].axhline(threshold,  color="red",   ls=":",  alpha=0.8, label="collapse threshold")
    axes[1].axvline(0,          color="black", ls="-",  lw=2,      label="seizure onset")
    if collapse_idx is not None:
        axes[1].axvline(t_p[collapse_idx], color="green", ls="--", lw=1.5,
                        label=f"MI collapse (τ={tau_lead:.0f}s)")
    axes[1].set(xlabel="Time rel. to seizure (s)", ylabel="Φ(t) [nats]",
                title=f"{subj_id} — Preictal window")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(f"/kaggle/working/phi_plot_{subj_id}.png", dpi=150)
    plt.close()

    all_results.append({
        "patient_id":           subj_id,
        "n_channels":           int(n_ch),
        "fs":                   float(fs),
        "t_seizure_s":          float(t_seizure),
        "n_preictal_windows":   int(len(phi_prei)),
        "n_interictal_windows": int(len(phi_inter)),
        "mu_inter":             round(mu_inter, 5),
        "sig_inter":            round(sig_inter, 5),
        "threshold":            round(threshold, 5),
        "collapse_detected":    collapse_idx is not None,
        "tau_lead_seconds":     round(tau_lead, 2) if tau_lead is not None else None,
        "phi_prei_min":         round(float(phi_prei.min()), 5),
    })
    patients_done += 1

# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("AGGREGATE RESULTS")
print("=" * 60)

n_total    = len(all_results)
n_collapse = sum(1 for r in all_results if r["collapse_detected"])
tau_leads  = [r["tau_lead_seconds"] for r in all_results
              if r["tau_lead_seconds"] is not None]

mean_tau = round(float(np.mean(tau_leads)),  2) if tau_leads else 0.0
std_tau  = round(float(np.std(tau_leads)),   2) if tau_leads else 0.0

summary = {
    "n_patients_processed":   n_total,
    "n_seizures_processed":   n_total,
    "n_collapse_detected":    n_collapse,
    "pct_collapse_detected":  round(100 * n_collapse / n_total, 1) if n_total else 0.0,
    "mean_tau_lead_seconds":  mean_tau,
    "std_tau_lead_seconds":   std_tau,
    "min_tau_lead_seconds":   round(float(min(tau_leads)), 2) if tau_leads else None,
    "max_tau_lead_seconds":   round(float(max(tau_leads)), 2) if tau_leads else None,
    "per_patient":            all_results,
}

core.format_result("n_patients_processed",  summary["n_patients_processed"])
core.format_result("n_seizures_processed",  summary["n_seizures_processed"])
core.format_result("n_collapse_detected",   summary["n_collapse_detected"])
core.format_result("pct_collapse_detected", summary["pct_collapse_detected"])
core.format_result("mean_tau_lead_seconds", summary["mean_tau_lead_seconds"])
core.format_result("std_tau_lead_seconds",  summary["std_tau_lead_seconds"])

core.save_results(summary, "/kaggle/working/phase1b_results.json")

if n_total == 0:
    print("WARNING: No patients processed.")
else:
    print(f"Processed {n_total} patients; collapse in {n_collapse}/{n_total}.")
    if tau_leads:
        print(f"Mean τ_lead = {mean_tau:.1f}s ± {std_tau:.1f}s")
print("\nDone.")
