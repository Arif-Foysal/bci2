"""
ieeg-ib phase4a — Transfer Entropy SOZ identification (H4).

Hypothesis H4: The TE asymmetry index A_i = TE_{i→rest} − TE_{rest→i}
identifies the seizure onset zone (SOZ) with AUC > 0.75. SOZ channels act
as "information black holes" — receiving more TE than they emit — so A_i
is strongly negative for SOZ channels.

This notebook:
  1. Loads ictal EDF (uses full recording including ictal period for TE estimation)
  2. Computes TE_{i→j} for all channel pairs using ieeg_core.te_asymmetry_index
  3. Computes A_i = mean(TE_{i→j}) - mean(TE_{j→i}) for each channel
  4. Compares A_i against binary SOZ labels (from electrodes.tsv or BIDS sidecar)
  5. Computes AUC, top-1 and top-3 precision, and ROC curves
  6. Reports aggregate AUC across all patients

Notes:
  - SOZ labels are read from *_electrodes.tsv (column: seizure_zone or group)
  - If no per-electrode labels available, uses whole-recording TE asymmetry
    as a within-patient relative ranking metric
  - MAX_CHANNELS = 16 (speed); uses full ictal EDF for TE (need ictal activity)

Config:
  MAX_PATIENTS  = 57
  MAX_CHANNELS  = 16
  TE_LAG        = 10    (samples = 20ms at 500 Hz)
  EMBED_DIM     = 3
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
TE_LAG        = 10
EMBED_DIM     = 3

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
print("PHASE 4A: Transfer Entropy SOZ Identification")
print("=" * 60)
print(f"Dataset root: {DATASET_ROOT}")

subject_dirs = sorted(glob.glob(os.path.join(DATASET_ROOT, "sub-*")))
print(f"Subjects found: {len(subject_dirs)}")

participants_meta = {}  # loaded below after helpers are defined


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_edf_ictal(path, t_ictal_start=None, duration_s=60.0):
    """Load ictal portion of EDF. Uses t_ictal_start seconds into the file."""
    raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
    total_dur = raw.times[-1]
    # Use ictal period: from t_ictal_start to t_ictal_start + duration_s
    if t_ictal_start is not None:
        t0 = min(t_ictal_start, total_dur - 5.0)
        t1 = min(t0 + duration_s, total_dur)
    else:
        # Fallback: last 60s of recording
        t1 = total_dur
        t0 = max(0.0, t1 - duration_s)
    raw.crop(tmin=t0, tmax=t1)
    raw.load_data(verbose=False)
    raw.pick_types(eeg=True, ecog=True, seeg=True, misc=False,
                   stim=False, exclude="bads")
    raw.filter(0.5, 200.0, method="iir", verbose=False)
    raw.notch_filter(np.arange(60, 201, 60), verbose=False)
    raw.set_eeg_reference("average", verbose=False)
    return raw.get_data(), raw.ch_names, raw.info["sfreq"]


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


def mni_to_lobe(x, y, z):
    """Assign coarse lobe label from MNI coordinates (mm).
    Rules are simplified but reliable for gross lobe assignment.
    """
    try:
        ax = abs(float(x))
        ay = float(y)
        az = float(z)
    except (ValueError, TypeError):
        return "UNKNOWN"
    # Mesial temporal: hippocampus / amygdala / parahippocampal
    if ax < 32 and -40 < ay < 12 and az < 0:
        return "MTL"
    # Lateral temporal
    if ax > 28 and -65 < ay < 12 and az < 20:
        return "TEMPORAL"
    # Insular
    if 25 < ax < 48 and -20 < ay < 25 and -10 < az < 22:
        return "INSULAR"
    # Parietal
    if ay < -45 and az > 18:
        return "PARIETAL"
    # Occipital
    if ay < -70:
        return "OCCIPITAL"
    # Frontal (default for anterior / superior)
    if ay > 10 or az > 25:
        return "FRONTAL"
    return "UNKNOWN"


# Which inferred lobes count as positive for each participants.tsv target value
_TARGET_LOBES = {
    "MTL":      {"MTL"},
    "TEMPORAL": {"TEMPORAL"},
    "FRONTAL":  {"FRONTAL"},
    "MFL":      {"FRONTAL"},
    "FP":       {"FRONTAL", "PARIETAL"},
    "INSULAR":  {"INSULAR"},
    "PARIETAL": {"PARIETAL"},
    "OCCIPITAL":{"OCCIPITAL"},
}


def load_participants_meta(dataset_root):
    """Load participants.tsv → dict keyed by participant_id (e.g. 'sub-HUP060')."""
    ptcp_path = os.path.join(dataset_root, "participants.tsv")
    meta = {}
    if not os.path.exists(ptcp_path):
        return meta
    with open(ptcp_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            pid = row.get("participant_id", "").strip()
            if pid:
                meta[pid] = row
    return meta


def load_soz_labels(ieeg_dir, ch_names, subj_id, participants_meta):
    """Derive per-electrode SOZ labels using MNI coordinates + patients.tsv target.

    Strategy:
      1. Check if electrodes.tsv has an explicit SOZ column — use it if present.
      2. Otherwise use MNI x/y/z in electrodes.tsv + the patient's 'target' lobe
         from participants.tsv to label electrodes in that lobe as SOZ (1).
    Returns binary array (1=SOZ, 0=non-SOZ), or None if not determinable.
    """
    # --- Explicit SOZ column search (keeps backward compat with annotated sets) ---
    # BIDS electrodes.tsv lives at sub-<id>/ level, two directories above ieeg/
    subj_dir = os.path.normpath(os.path.join(ieeg_dir, "..", ".."))
    elec_files = glob.glob(os.path.join(subj_dir, "*electrodes.tsv"))
    elec_files += glob.glob(os.path.join(ieeg_dir, "..", "*electrodes.tsv"))
    elec_files += glob.glob(os.path.join(ieeg_dir, "*electrodes.tsv"))
    elec_files += glob.glob(os.path.join(ieeg_dir, "../../", "**", "*electrodes.tsv"),
                            recursive=True)

    soz_by_name = {}
    coords_by_name = {}
    for ef in elec_files:
        try:
            with open(ef, encoding="utf-8-sig") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    name = row.get("name", row.get("channel", "")).strip().upper()
                    if not name:
                        continue
                    sz_zone = row.get("seizure_zone", row.get("group", "")).strip().lower()
                    soz_col = row.get("soz", row.get("SOZ", "")).strip().lower()
                    is_soz = (
                        sz_zone in ("soz", "sz_onset", "seizure_onset", "1", "yes") or
                        soz_col in ("soz", "1", "yes", "true") or
                        "soz" in sz_zone or "onset" in sz_zone
                    )
                    soz_by_name[name] = int(is_soz)
                    # Also cache MNI coords for fallback
                    try:
                        coords_by_name[name] = (
                            float(row.get("x", "nan")),
                            float(row.get("y", "nan")),
                            float(row.get("z", "nan")),
                        )
                    except (ValueError, TypeError):
                        pass
        except Exception:
            continue

    if soz_by_name:
        labels = np.array([soz_by_name.get(c.strip().upper(), 0) for c in ch_names])
        if labels.sum() > 0:
            return labels, "explicit_column"

    # --- MNI coordinate + participants.tsv target fallback ---
    pmeta = participants_meta.get(subj_id, {})
    target = pmeta.get("target", "").strip().upper()
    if not target or target == "N/A":
        return None, "no_target"

    target_lobes = _TARGET_LOBES.get(target, set())
    if not target_lobes:
        return None, f"unknown_target_{target}"

    if not coords_by_name:
        return None, "no_coords"

    labels_list = []
    for c in ch_names:
        key = c.strip().upper()
        coords = coords_by_name.get(key)
        if coords is None or any(np.isnan(v) for v in coords):
            labels_list.append(0)
        else:
            inferred_lobe = mni_to_lobe(*coords)
            labels_list.append(1 if inferred_lobe in target_lobes else 0)

    labels = np.array(labels_list)
    if labels.sum() == 0:
        return None, f"no_matches_for_{target}"
    return labels, f"mni_{target}"


def roc_auc(scores, labels):
    """Compute AUC given continuous scores (higher = more SOZ) and binary labels."""
    from sklearn.metrics import roc_auc_score
    if labels.sum() == 0 or labels.sum() == len(labels):
        return 0.5
    try:
        return float(roc_auc_score(labels, scores))
    except Exception:
        return 0.5


def top_k_precision(scores, labels, k):
    """Precision of top-k channels ranked by score (descending)."""
    top_idx = np.argsort(scores)[-k:]
    return float(labels[top_idx].sum() / k)


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


# ---------------------------------------------------------------------------
# Load participants metadata (used for MNI-based SOZ labeling)
# ---------------------------------------------------------------------------
participants_meta = load_participants_meta(DATASET_ROOT)
print(f"Loaded metadata for {len(participants_meta)} patients")

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
from sklearn.metrics import roc_curve
all_results = []
all_aucs = []
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
    if not ictal_edfs:
        print(f"\n{subj_id}: no ictal EDF, skipping")
        continue

    ictal_edf  = os.path.join(ieeg_dir, ictal_edfs[0])
    events_tsv = ictal_edf.replace("_ieeg.edf", "_events.tsv")
    t_seizure  = get_seizure_onset(events_tsv)

    print(f"\n{'='*50}")
    print(f"Patient : {subj_id}  (t_seizure={t_seizure}s)")

    try:
        # Load 60s of ictal data starting at seizure onset
        data, ch_names, fs = load_edf_ictal(ictal_edf,
                                             t_ictal_start=t_seizure,
                                             duration_s=60.0)
    except Exception as e:
        print(f"  Load error: {e}")
        continue

    rec_type = infer_recording_type(ch_names)

    # Channel selection
    n_ch_all = data.shape[0]
    if n_ch_all > MAX_CHANNELS:
        top_idx = np.argsort(np.var(data, axis=1))[-MAX_CHANNELS:]
        data    = data[top_idx]
        sel_names = [ch_names[i] for i in top_idx]
    else:
        sel_names = ch_names
    n_ch = data.shape[0]
    print(f"  {rec_type} | {n_ch} channels  fs={fs:.0f}Hz")

    # ── Compute TE asymmetry index ───────────────────────────────────────────
    try:
        A, te_matrix = core.te_asymmetry_index(data, lag=int(TE_LAG),
                                                embed_dim=EMBED_DIM)
    except Exception as e:
        print(f"  TE error: {e}")
        continue

    print(f"  A_i range: [{A.min():.4f}, {A.max():.4f}]")
    print(f"  Channels with A_i < 0 (sinks): {(A < 0).sum()}/{n_ch}")

    # SOZ candidates are channels with most negative A_i (largest net receivers)
    # Score = -A_i (higher score = more likely SOZ)
    soz_score = -A

    # ── Load SOZ labels if available ─────────────────────────────────────────
    result = load_soz_labels(ieeg_dir, sel_names, subj_id, participants_meta)
    if result is None:
        soz_labels, label_source = None, "none"
    else:
        soz_labels, label_source = result
    has_labels = soz_labels is not None

    auc = 0.5
    prec1 = None
    prec3 = None
    n_soz_labeled = 0
    print(f"  SOZ label source: {label_source}")

    if has_labels:
        n_soz_labeled = int(soz_labels.sum())
        auc    = roc_auc(soz_score, soz_labels)
        prec1  = top_k_precision(soz_score, soz_labels, k=1)
        prec3  = top_k_precision(soz_score, soz_labels, k=min(3, n_ch))
        print(f"  SOZ labels: {n_soz_labeled}/{n_ch} channels")
        print(f"  AUC: {auc:.3f}  prec@1: {prec1:.2f}  prec@3: {prec3:.2f}")

        # Per-patient ROC plot
        fpr, tpr, _ = roc_curve(soz_labels, soz_score)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC={auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set(xlabel="FPR", ylabel="TPR",
               title=f"{subj_id} — TE SOZ ROC ({rec_type})")
        ax.legend()
        fig.savefig(f"/kaggle/working/p4_roc_{subj_id}.png", dpi=100)
        plt.close()
    else:
        print(f"  No SOZ labels found — reporting A_i distribution only")

    # ── Save ─────────────────────────────────────────────────────────────────
    np.save(f"/kaggle/working/A_i_{subj_id}.npy",        A)
    np.save(f"/kaggle/working/te_matrix_{subj_id}.npy",  te_matrix)

    all_results.append({
        "patient_id":       subj_id,
        "rec_type":         rec_type,
        "n_channels":       int(n_ch),
        "has_soz_labels":   has_labels,
        "label_source":     label_source,
        "n_soz_labeled":    n_soz_labeled,
        "auc":              round(auc, 4),
        "prec_at_1":        round(prec1, 4) if prec1 is not None else None,
        "prec_at_3":        round(prec3, 4) if prec3 is not None else None,
        "A_i_min":          round(float(A.min()), 5),
        "A_i_max":          round(float(A.max()), 5),
        "A_i_range":        round(float(A.max() - A.min()), 5),
        "n_sinks":          int((A < 0).sum()),
        "pct_sinks":        round(100 * (A < 0).mean(), 1),
    })
    if has_labels:
        all_aucs.append(auc)
    patients_done += 1

# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("AGGREGATE RESULTS (H4)")
print("=" * 60)

if not all_results:
    print("WARNING: No patients processed.")
    summary = {"soz_auc": 0.5, "n_patients_processed": 0}
    core.format_result("soz_auc", 0.5)
    core.save_results(summary, "/kaggle/working/phase4a_results.json")
else:
    import csv as csv_mod
    n_total        = len(all_results)
    n_with_labels  = sum(1 for r in all_results if r["has_soz_labels"])
    mean_auc       = float(np.mean(all_aucs))  if all_aucs else 0.5
    std_auc        = float(np.std(all_aucs))   if all_aucs else 0.0
    n_above_chance = sum(1 for a in all_aucs if a > 0.5)

    prec1_vals = [r["prec_at_1"] for r in all_results
                  if r["prec_at_1"] is not None]
    prec3_vals = [r["prec_at_3"] for r in all_results
                  if r["prec_at_3"] is not None]
    mean_prec1 = float(np.mean(prec1_vals)) if prec1_vals else 0.0
    mean_prec3 = float(np.mean(prec3_vals)) if prec3_vals else 0.0

    # Mean TE asymmetry range (measure of signal quality even without labels)
    mean_A_range = float(np.mean([r["A_i_range"] for r in all_results]))
    mean_pct_sinks = float(np.mean([r["pct_sinks"] for r in all_results]))

    print(f"N patients processed:        {n_total}")
    print(f"N patients with SOZ labels:  {n_with_labels}")
    print(f"Mean AUC:                    {mean_auc:.3f} ± {std_auc:.3f}")
    print(f"N patients AUC > 0.5:        {n_above_chance}/{n_with_labels}")
    print(f"Mean precision@1:            {mean_prec1:.3f}")
    print(f"Mean precision@3:            {mean_prec3:.3f}")
    print(f"Mean A_i range:              {mean_A_range:.4f}")
    print(f"Mean % sink channels:        {mean_pct_sinks:.1f}%")

    # Aggregate AUC boxplot + comparison to HFO benchmark (0.71)
    if all_aucs:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].boxplot(all_aucs, positions=[0], widths=0.4)
        axes[0].scatter(np.zeros(len(all_aucs)) + np.random.randn(len(all_aucs)) * 0.05,
                        all_aucs, alpha=0.5, s=30, color="steelblue")
        axes[0].axhline(0.5,  color="gray",   ls="--", label="chance")
        axes[0].axhline(0.71, color="orange", ls=":",  label="HFO benchmark (0.71)")
        axes[0].axhline(0.75, color="red",    ls=":",  label="H4 target (0.75)")
        axes[0].set(ylabel="AUC", xticks=[0], xticklabels=["TE Asym."],
                    title="TE Asymmetry SOZ AUC")
        axes[0].legend(fontsize=8)

        # Per-patient A_i range (signal quality)
        recs  = [r["rec_type"]  for r in all_results]
        aranges = [r["A_i_range"] for r in all_results]
        colors = ["steelblue" if r == "ECoG" else "darkorange" for r in recs]
        axes[1].bar(range(len(aranges)), aranges, color=colors, alpha=0.6)
        axes[1].set(xlabel="Patient index", ylabel="A_i range (nats)",
                    title="TE Asymmetry Signal Range per Patient")
        from matplotlib.patches import Patch
        axes[1].legend(handles=[Patch(color="steelblue", label="ECoG"),
                                  Patch(color="darkorange", label="SEEG")])

        fig.suptitle(f"Phase 4A — TE SOZ Identification (N={n_total})")
        plt.tight_layout()
        fig.savefig("/kaggle/working/phase4a_summary.png", dpi=150)
        plt.close()

    # CSV
    fieldnames = list(all_results[0].keys())
    with open("/kaggle/working/phase4a_patients.csv", "w", newline="") as f:
        w = csv_mod.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_results)

    summary = {
        "n_patients_processed":      n_total,
        "n_patients_with_soz_labels": n_with_labels,
        "soz_auc":                   round(mean_auc,  4),
        "std_auc":                   round(std_auc,   4),
        "n_auc_above_chance":        n_above_chance,
        "pct_auc_above_chance":      round(100 * n_above_chance / n_with_labels, 1)
                                     if n_with_labels > 0 else 0.0,
        "mean_precision_at_1":       round(mean_prec1, 4),
        "mean_precision_at_3":       round(mean_prec3, 4),
        "mean_A_i_range":            round(mean_A_range,    5),
        "mean_pct_sink_channels":    round(mean_pct_sinks,  2),
        "per_patient":               all_results,
    }

    core.format_result("n_patients_processed",  n_total)
    core.format_result("soz_auc",               summary["soz_auc"])
    core.format_result("std_auc",               summary["std_auc"])
    core.format_result("n_auc_above_chance",    summary["n_auc_above_chance"])
    core.format_result("mean_precision_at_1",   summary["mean_precision_at_1"])
    core.format_result("mean_precision_at_3",   summary["mean_precision_at_3"])
    core.format_result("mean_A_i_range",        summary["mean_A_i_range"])

    core.save_results(summary, "/kaggle/working/phase4a_results.json")
    print(f"\nDone. Processed {n_total} patients.")
