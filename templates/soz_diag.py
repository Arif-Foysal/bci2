"""SOZ label diagnostic — find where HUP electrode annotations live."""
import os, sys, csv, glob

def _find_core(name):
    for dp, _, fns in os.walk("/kaggle/input"):
        if f"{name}.py" in fns:
            return dp
    return None

sys.path.insert(0, _find_core("ieeg_core"))
import ieeg_core as core

DATASET_ROOT = None
for dp, dirs, _ in os.walk("/kaggle/input"):
    if any(d.startswith("sub-") for d in dirs):
        DATASET_ROOT = dp
        break

print(f"Dataset root: {DATASET_ROOT}")

# 1. Find all electrodes.tsv files
elec_files = sorted(glob.glob(os.path.join(DATASET_ROOT, "**", "*electrodes.tsv"),
                               recursive=True))
print(f"\nTotal *electrodes.tsv files: {len(elec_files)}")

for ef in elec_files[:8]:
    rel = os.path.relpath(ef, DATASET_ROOT)
    with open(ef, encoding="utf-8-sig") as f:
        lines = f.readlines()
    cols = lines[0].strip() if lines else "(empty)"
    n_rows = len(lines) - 1
    print(f"\n  {rel}  ({n_rows} electrodes)")
    print(f"  Columns: {cols}")
    for row in lines[1:3]:
        print(f"    {row.strip()}")

# 2. Check dataset-level participants.tsv
ptcp = os.path.join(DATASET_ROOT, "participants.tsv")
print(f"\n{'='*60}")
print("participants.tsv")
if os.path.exists(ptcp):
    with open(ptcp, encoding="utf-8-sig") as f:
        lines = f.readlines()
    print(f"  Columns: {lines[0].strip()}")
    for row in lines[1:5]:
        print(f"  {row.strip()}")
else:
    print("  NOT FOUND")

# 3. Check dataset_description.json for any mention of SOZ / resection
desc = os.path.join(DATASET_ROOT, "dataset_description.json")
if os.path.exists(desc):
    with open(desc) as f:
        import json
        d = json.load(f)
    print(f"\ndataset_description keys: {list(d.keys())}")

# 4. Walk first subject for any non-standard files
import glob as _g
first_sub = sorted(_g.glob(os.path.join(DATASET_ROOT, "sub-*")))[0]
print(f"\n{'='*60}")
print(f"All files under {os.path.basename(first_sub)}:")
for dp, _, fns in os.walk(first_sub):
    for fn in sorted(fns):
        full = os.path.join(dp, fn)
        print(f"  {os.path.relpath(full, DATASET_ROOT)}  ({os.path.getsize(full):,}b)")

core.format_result("soz_diag_complete", True)
import json
with open("/kaggle/working/soz_diag_results.json", "w") as f:
    json.dump({"n_elec_files": len(elec_files)}, f)
print("\nDone.")
