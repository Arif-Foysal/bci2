"""
ieeg-ib phase1_diag — Dataset structure diagnostic.

Prints full file inventory for first 3 subjects, reads all TSV files,
and reads MNE annotations from the first EDF file to determine how
seizure onset times are stored in this dataset.
"""

import os, sys, json

def _find_core(name):
    for dp, _, fns in os.walk("/kaggle/input"):
        if f"{name}.py" in fns:
            return dp
    return None

sys.path.insert(0, _find_core("ieeg_core"))
import ieeg_core as core

import glob
import mne
mne.set_log_level("WARNING")

print("=" * 60)
print("PHASE 1 DIAGNOSTIC: Dataset structure")
print("=" * 60)

# Locate dataset root
DATASET_ROOT = None
for dp, dirs, fns in os.walk("/kaggle/input"):
    if any(d.startswith("sub-") for d in dirs):
        DATASET_ROOT = dp
        break

print(f"\nDataset root: {DATASET_ROOT}")
subject_dirs = sorted(glob.glob(os.path.join(DATASET_ROOT, "sub-*")))
print(f"Total subjects: {len(subject_dirs)}")

# ---------------------------------------------------------------------------
# 1. Full inventory of first 3 subjects
# ---------------------------------------------------------------------------
for subj_dir in subject_dirs[:3]:
    subj_id = os.path.basename(subj_dir)
    print(f"\n{'─'*50}")
    print(f"SUBJECT: {subj_id}")
    for dp, dirs, fns in os.walk(subj_dir):
        rel = os.path.relpath(dp, subj_dir)
        for f in sorted(fns):
            full = os.path.join(dp, f)
            size = os.path.getsize(full)
            print(f"  {rel}/{f}  ({size:,} bytes)")

# ---------------------------------------------------------------------------
# 2. Read all TSV files from first subject and print their headers + 3 rows
# ---------------------------------------------------------------------------
first_sub = subject_dirs[0]
print(f"\n{'='*60}")
print(f"TSV CONTENTS — {os.path.basename(first_sub)}")
print(f"{'='*60}")

for dp, _, fns in os.walk(first_sub):
    for f in sorted(fns):
        if not f.endswith(".tsv"):
            continue
        path = os.path.join(dp, f)
        print(f"\n  File: {f}")
        try:
            with open(path) as fh:
                lines = fh.readlines()
            print(f"  Header: {lines[0].rstrip()}")
            for row in lines[1:4]:
                print(f"    {row.rstrip()}")
            if len(lines) > 4:
                print(f"    ... ({len(lines)-1} data rows total)")
        except Exception as e:
            print(f"  ERROR: {e}")

# ---------------------------------------------------------------------------
# 3. Read MNE annotations from first ictal EDF file
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("MNE ANNOTATIONS — first ictal EDF")
print(f"{'='*60}")

edf_files = sorted(glob.glob(os.path.join(DATASET_ROOT, "sub-*", "**", "*ictal*ieeg.edf"),
                              recursive=True))
if not edf_files:
    edf_files = sorted(glob.glob(os.path.join(DATASET_ROOT, "sub-*", "**", "*.edf"),
                                 recursive=True))

print(f"EDF files found: {len(edf_files)}")
for edf in edf_files[:5]:
    print(f"  {os.path.relpath(edf, DATASET_ROOT)}")

if edf_files:
    edf = edf_files[0]
    print(f"\nReading: {os.path.basename(edf)}")
    try:
        raw = mne.io.read_raw_edf(edf, preload=False, verbose=False)
        print(f"  Duration: {raw.times[-1]:.1f}s  Channels: {len(raw.ch_names)}  FS: {raw.info['sfreq']:.0f}Hz")
        print(f"  Channel names (first 10): {raw.ch_names[:10]}")
        print(f"  Annotations ({len(raw.annotations)} total):")
        for ann in raw.annotations[:10]:
            print(f"    onset={ann['onset']:.2f}s  duration={ann['duration']:.2f}s  description='{ann['description']}'")
        if len(raw.annotations) == 0:
            print("    (no annotations in EDF header)")
    except Exception as e:
        print(f"  ERROR: {e}")

# ---------------------------------------------------------------------------
# 4. Check corresponding events.tsv for first EDF
# ---------------------------------------------------------------------------
if edf_files:
    base = os.path.basename(edf_files[0]).replace("_ieeg.edf", "")
    dp   = os.path.dirname(edf_files[0])
    events_tsv = os.path.join(dp, base + "_events.tsv")
    print(f"\nExpected events.tsv: {events_tsv}")
    print(f"Exists: {os.path.exists(events_tsv)}")
    if os.path.exists(events_tsv):
        with open(events_tsv) as fh:
            for i, line in enumerate(fh):
                print(f"  {line.rstrip()}")
                if i > 5: break
    else:
        # Try without acq/run suffixes
        import re
        variants = [
            re.sub(r'_run-\d+', '', base),
            re.sub(r'_acq-[a-z]+', '', base),
            re.sub(r'_acq-[a-z]+_run-\d+', '', base),
        ]
        for v in variants:
            p = os.path.join(dp, v + "_events.tsv")
            print(f"  Trying: {os.path.basename(p)}  exists={os.path.exists(p)}")

# ---------------------------------------------------------------------------
# 5. Check dataset-level files (participants.tsv, dataset_description.json)
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("DATASET-LEVEL FILES")
print(f"{'='*60}")
for f in sorted(os.listdir(DATASET_ROOT)):
    full = os.path.join(DATASET_ROOT, f)
    if os.path.isfile(full):
        print(f"  {f}  ({os.path.getsize(full):,} bytes)")
        if f.endswith(".json") or (f.endswith(".tsv") and os.path.getsize(full) < 5000):
            with open(full, errors="replace") as fh:
                content = fh.read(2000)
            print("    " + content[:500].replace("\n", "\n    "))

results = {"diagnostic_complete": True}
core.format_result("diagnostic_complete", True)
core.save_results(results, "/kaggle/working/phase1_diag_results.json")
print("\nDiagnostic complete.")
