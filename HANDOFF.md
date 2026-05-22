# Handoff — ieeg-ib Pipeline (2026-05-22)

## What just happened
We re-ran Phases 2a, 3a, and 4a on the quality-gated 41-patient cohort
(phase1c_patients.csv whitelist). All three notebooks load the whitelist via
a **substring** filename match (the actual file on Kaggle is named
`ieeg-ib-phase1c_phase1c_patients.csv` due to upload renaming).

## Immediately pending: Phase 2a result

Phase 2a is **currently running on Kaggle**:
```
python pipeline.py status phase2a   # check if done
python pipeline.py wait phase2a     # block until done (may timeout at 180 min; use
                                    # python -c trick below for longer wait)
```

Long-wait one-liner:
```bash
source .venv/bin/activate && python -c "
import sys; sys.argv=['pipeline.py']
from pipeline import load_config, cmd_wait
config = load_config()
sys.exit(cmd_wait(config, 'phase2a', max_minutes=600))
"
```

### What was fixed in this run
- `src/ieeg_core.py`: added `fixed_threshold` param to `mi_network_metrics()` and
  new helper `mi_network_threshold()`. Previously the per-window 80th-percentile
  threshold made density always ~20% — insensitive to real MI changes. Now the
  interictal MI distribution sets a fixed threshold applied to both periods.
- `notebooks/phase2a.py`: calls `core.mi_network_threshold()` on interictal data
  then passes result as `fixed_threshold=` to both interictal and preictal calls.
  Also fixed integration classification (was requiring p_rho < 0.05 with a
  one-sided "greater" test which never fires for integration patients; now uses
  d_rho < -0.3 and d_deg < -0.3 without p-value gate).
- Core was uploaded via `python pipeline.py upload-src` then 5-min propagation
  wait before push.

### When phase2a completes
1. `python pipeline.py fetch phase2a`
2. `python pipeline.py tail phase2a` — verify:
   - "Phase-1c whitelist loaded: 41 patients"
   - n_patients_processed ≈ 38 (41 minus saturated patients)
   - n_fragmentation, n_integration, n_unclassified printed
   - mean_r_phi_rho (Spearman) among fragmentation patients
3. Update `paper/main.tex` Phase 2 section (lines ~700-737) with new counts.
   Old numbers: "Twelve of 54... leaving 42... 16 fragmentation / 15 integration /
   11 unclassified... mean r = -0.60"

## Paper update status (paper/main.tex)

All numbers updated **except Phase 2** which awaits the run above:

| Phase | Key metric | Paper value | Status |
|-------|-----------|-------------|--------|
| 1c cohort | n | 41 | ✓ done |
| 1c Type A/B/Ind | counts | 20/15/6 | ✓ done |
| 1c group stat | p=0.41, d=−2.65 | verified | ✓ done |
| 2a fragmentation | 16/15/11 split | **PENDING** | ← update after run |
| 2a convergent validity | r=−0.60 | **PENDING** | ← update after run |
| 3a invisible precursor | 6/15 (40%) | ✓ done |
| 3a r(Phi,V) Type A/B | 0.40 / 0.58 | ✓ done |
| 4a AUC | 0.509 (MTL: 0.635) | ✓ done |

After updating Phase 2, also update `paper/response_to_reviewers.md` with the
final verified numbers for all phases.

## Key file locations
- Manuscript: `paper/main.tex`
- Response letter: `paper/response_to_reviewers.md`  
- Core library: `src/ieeg_core.py` (uploaded to Kaggle as `ieeg-ib-core`)
- Phase 2 notebook: `notebooks/phase2a.py`
- Kaggle token: `.env` (KAGGLE_API_TOKEN=...)

## Phase 2 paper section location
In `paper/main.tex`, search for "Phase 2" results around line 700:
- "Twelve of 54 patients" → update exclusion count and total
- "16 patients show simultaneous significant decreases" → update to n_fragmentation
- "15 patients show simultaneous significant increases" → update to n_integration
- "11 patients show mixed" → update to n_unclassified
- "mean r = -0.60" → update to mean_r_phi_rho_frag from results
- Abstract line ~73: "16 fragmentation vs. 15 integration among 31 non-saturated patients;
  Spearman r = -0.60" → update all three numbers

## If phase 2a results still look wrong
The graph analysis is sensitive to data quality. If n_fragmentation + n_integration
is still far from the paper's 31, check the log for per-patient density values:
`python pipeline.py tail phase2a` should show lines like:
  "ρ: inter=0.XXX  prei=0.XXX  drop=XX%  d=X.XX  p=0.XXXX"
After the fix, inter density should be ~0.20 (= 20th percentile baseline) and
prei should vary above or below 0.20 meaningfully.
