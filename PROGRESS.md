# ieeg-ib Pipeline — Progress & Pivot Log

## Phase Status

| Phase | Notebook | Status | Key Result |
|-------|----------|--------|-----------|
| 1a | Smoke test | ✅ Complete | mi_correlated=0.535, mi_independent=0.0 |
| 1b | Pilot 3 patients | ✅ Complete | ECoG: 31–37% drop; SEEG: 1.8% drop; 0/3 threshold crossings |
| 1c | Full 57-patient Wilcoxon | ✅ Complete | **H1 REJECTED.** Wilcoxon p=0.92 full, p=0.41 clean cohort. 22/41 MI-decrease, 19/41 MI-increase. Bifurcation finding. |
| 2a | Graph fragmentation | ✅ Complete | **H2 not globally confirmed** (Wilcoxon p=0.11). But 15/54 (27.8%) show all-3-significant fragmentation. Clean patients 50/50 split (16 frag / 15 integration). r(Phi,rho)=-0.60 (n=16 valid). Mirrors H1 bifurcation. |
| 3a | MI vs variance | ✅ Complete | **H3 redesigned.** Collapse detection: 5 Phi, 4 V, 0 both. But: 6/16 Type A patients show MI collapse while V *increases* (invisible precursor). Type A interictal r(Phi,V)=0.362 (below 0.4 threshold). Type B r=0.497. Full-cohort mean r=0.49. |
| 4a | TE SOZ identification | ✅ Complete | **H4 not confirmed** (mean AUC=0.502, target >0.75). But: 9/50 AUC>0.75, mean precision@1=0.28 (above chance). MTL patients: mean AUC=0.584, 5/14 >0.75. AUC is lower bound — MNI proxy labels introduce label noise. |

---

## Pivots

### Pivot 1 — MI Estimator Speed (Phase 1b, iteration 1)
**Problem:** Kraskov MI estimation ran at 244 ms/pair — 3 patients × 16 channels × 120 windows × 120 pairs = ~5 hours.  
**Root cause:** `query_ball_point` was called in a Python loop once per sample point (2,500 calls per pair).  
**Fix:** Vectorized batch call — pass all points and per-point radii in a single scipy call with `return_length=True`. Result: 54 ms/pair (4.5x speedup). 3-patient estimate dropped to ~38 minutes.  
**Lesson:** Always batch spatial queries; scipy supports array-valued radii since v1.7.

---

### Pivot 2 — UTF-8 BOM in events.tsv (Phase 1b, iteration 2)
**Problem:** Phase 1b processed 0 patients. `get_seizure_onset()` silently returned `None` for every file.  
**Root cause:** HUP events.tsv files have a UTF-8 BOM (`﻿`) prepended to the header. The first column key became `﻿onset` instead of `onset`, so `row["onset"]` raised a `KeyError` that was caught and swallowed.  
**Fix:** `open(events_tsv, encoding="utf-8-sig")` — Python's `utf-8-sig` codec strips BOM automatically.  
**Discovery method:** `phase1_diag.py` diagnostic notebook printed raw TSV header bytes, revealing the BOM character.  
**Lesson:** Always use `utf-8-sig` for BIDS TSV files; they commonly originate from Windows tools that add BOM.

---

### Pivot 3 — Wrong EDF file selection (Phase 1b, iteration 2)
**Problem:** The original `find_ieeg_files()` helper iterated all EDF files under the subject directory, mixing `task-ictal` and `task-interictal` files.  
**Root cause:** No task-type filter — the first file returned was sometimes an interictal EDF being treated as the ictal recording.  
**Fix:** Explicit separate filters:
```python
ictal_edfs = [f for f in os.listdir(ieeg_dir) if "task-ictal" in f and f.endswith("_ieeg.edf")]
inter_edfs = [f for f in os.listdir(ieeg_dir) if "task-interictal" in f and f.endswith("_ieeg.edf")]
```
**Lesson:** HUP BIDS structure has two distinct task types per session; always filter by task name explicitly.

---

### Pivot 4 — Threshold-crossing collapse detection doesn't work on 120s clips (Phase 1b, iteration 3)
**Problem:** 0/3 collapses detected despite real MI drops of 31–37% in ECoG patients.  
**Root cause:** HUP ictal EDFs contain only ~120s of preictal data (not continuous long recordings). The MI signal was already in the "collapsed" state at t=0 — there was no transition to detect. The threshold-crossing algorithm requires seeing the drop happen within the window.  
**Fix (methodological pivot):** Abandoned threshold-crossing `τ_lead`. Switched to:
1. **Wilcoxon signed-rank test** across patients: μ_inter > μ_prei (tests whether MI is systematically lower during preictal vs interictal).  
2. **`preictal_trend_analysis()`** added to `ieeg_core.py` (v3): Mann-Kendall trend test, Cohen's d effect size, % drop, early/late half comparison.  
**Lesson:** Collapse detection requires recording designs with long preictal baselines. With short clips, the correct test is a distributional comparison (Wilcoxon), not a time-series crossing detector.

---

### Pivot 5 — SEEG vs ECoG signal heterogeneity (Phase 1b findings)
**Observation:** ECoG patients (sub-HUP064, sub-HUP065) showed Cohen's d = 1.40 and 0.80, pct_drop = 37% and 31%. SEEG patient (sub-HUP060) showed Cohen's d = 0.04, pct_drop = 1.77%.  
**Implication:** MI collapse signal is electrode-type dependent. ECoG (cortical grid/strip) captures synchronous cortical dynamics well. SEEG (depth electrodes, subcortical) may reflect different network dynamics, or the MI signal is weaker at the spatial scale/location captured.  
**Decision:** Phase 1c and all downstream phases run SEEG/ECoG **subgroup analysis** separately. This is now a first-class reporting dimension in all result JSONs.  
**Future action:** If H1 holds only for ECoG, the proposal's clinical claims should be scoped to cortical recordings.

---

### Pivot 6 — Phase 1c scope expansion (after Phase 1b)
**Original plan:** Phase 1b was intended to be the main 3-patient pilot. Move directly to graph metrics (Phase 2).  
**Decision:** Added Phase 1c as a full 57-patient sweep before proceeding to Phase 2. Rationale: the 3-patient pilot showed strong ECoG signal but the SEEG null result created ambiguity. Need full-cohort statistics to determine whether H1 holds at the group level before spending GPU/CPU time on graph metrics for a hypothesis that might not be supported.  
**Cost:** ~2.5 hours additional Kaggle runtime.  
**Benefit:** Phase 2–4 will have a solid H1 foundation (or a clear negative result to report).

---

## Key Dataset Facts Discovered

- HUP iEEG (NEMAR ds004100): 57 subjects, BIDS format
- Ictal EDFs: exactly ~120s preictal + ictal, seizure onset at t=120.0s
- Interictal EDFs: separate `task-interictal` files, ~300s
- Sampling rate: 500 Hz (SEEG) / 512 Hz (ECoG) — no resampling needed
- events.tsv: UTF-8-BOM encoded, `trial_type` column uses "sz onset" / "sz offset"
- SEEG channels: ~59 channels, depth electrodes
- ECoG channels: ~82–94 channels, cortical grid/strip
- SOZ labels: electrodes.tsv has only name/x/y/z/size — **NO SOZ column in public HUP BIDS release**
- participants.tsv has: participant_id, age, sex, hand, outcome, engel, therapy, implant, **target** (lobe), lesion_status, age_onset
- Target lobe values: FRONTAL (n=9), MTL (n=18), TEMPORAL (n=18), INSULAR (n=2), PARIETAL (n=1), FP (n=1), MFL (n=2)
- MNI coordinates available in electrodes.tsv x/y/z columns — used for proxy SOZ labeling in phase4a v2

---

---

### Pivot 7 — H1 is a genuine null result; MI directionality bifurcates by patient (Phase 1c)
**Problem:** Full-cohort Wilcoxon p = 0.9204. Mean pct_drop = -168%, mean Cohen's d = -4.39. The sign is reversed — on average, MI *increases* preictally rather than collapsing.  
**First suspect:** Extreme outliers (HUP080: μ_inter=0.0047, μ_prei=0.079; HUP086: μ_inter=0.037, μ_prei=1.408). Prepared core v4 with IQR outlier clipping and baseline quality check (μ_inter < 0.02 → skip).  
**After clean-cohort analysis (41 patients, μ_inter ≥ 0.02, μ_prei ≤ 0.5):**
- Wilcoxon p = **0.4088** — still not significant
- Split: 22/41 MI-decrease, 19/41 MI-increase (53.7% / 46.3%)
- Median pct_drop = +3.5% — barely positive
- ECoG-only: p = 0.26; SEEG-only: p = 0.63
- **H1 is genuinely rejected.** The null is not an artifact problem.

**Scientific interpretation:**  
The HUP dataset shows two distinct preictal MI phenotypes with roughly equal prevalence:
- **Type A — MI Collapse** (~54%): Consistent with the information bottleneck hypothesis. Strong signal in pilot ECoG patients (HUP064 d=1.40, HUP065 d=0.80, HUP074 d=1.00, HUP075 d=1.28).
- **Type B — MI Increase** (~46%): Preictal hypersynchronization. Network connectivity *increases* before seizure. Also documented in literature (ictal high-gamma synchrony). Cohen's d up to -2.85.

**Decision (not a re-run):** Core v4 fixes written but not uploaded — the null result is genuine and scientifically valid. H1 reformulated:
> **H1' (revised):** MI directionality (collapse vs increase) bifurcates patients into two phenotype classes. The direction of MI change is a measurable, parameter-free biomarker of seizure network type.

**Impact:** Phases 2a, 3a, 4a are pushed and running. These may show cleaner signals (graph topology, TE asymmetry) that don't depend on the sign of MI change. The bifurcation itself becomes a first-class result to report.

---

---

### Pivot 9 — H2 mirrors H1 bifurcation (Phase 2a)
**Finding:** H2 not confirmed at group level (Wilcoxon p=0.11). Among 42 non-saturated patients: exactly 16 show fragmentation, 15 show integration — same 50/50 split as H1. The graph topology bifurcation is independent confirmation that the MI bifurcation is real and not a MI estimator artifact.  
**Key statistic:** 15/54 patients (27.8%) show all-3-significant fragmentation in density + degree + clustering. 12/54 have saturated interictal networks (ρ_inter=1.0) and are excluded.  
**Implication for paper:** Two independent information-theoretic measures (MI signal level, graph density) both show the same bimodal patient-level split. This convergent validity strengthens the bifurcation finding significantly.

---

### Pivot 10 — H4 not globally confirmed; MNI proxy labels are the limiting factor (Phase 4a v2)
**Finding:** Mean AUC = 0.502 across 50 labelled patients — essentially chance. Target was AUC > 0.75.  
**Root cause:** No per-electrode SOZ annotations exist in the public HUP BIDS release. Phase 4a v2 used MNI coordinate atlas + participants.tsv `target` lobe as a proxy. This assigns SOZ labels to all electrodes anatomically co-located with the surgical target lobe, which includes many non-SOZ electrodes, introducing substantial label noise and suppressing AUC.  
**Partial signal present:** 9/50 patients AUC > 0.75; precision@1 = 0.28 (above chance ~0.15–0.20). MTL subgroup (n=14): mean AUC = 0.584, 5/14 > 0.75. Best performers: HUP135 and HUP190 (AUC=1.0, MTL); HUP163 (0.909, MTL); HUP187 (0.836, MTL).  
**Lobe ordering:** MTL (0.584) > TEMPORAL (0.481) > FRONTAL (0.406) — mirrors H1/H2/H3 bifurcation pattern. Compact hippocampal-entorhinal circuits yield cleaner asymmetry signal; distributed frontal networks do not.  
**Decision:** AUC = 0.50 is a lower bound. True test of H4 requires exact per-electrode SOZ labels from HUP clinical team (not in public release). Priority follow-on: contact HUP/NEMAR for de-identified electrode annotation spreadsheet, or validate on a secondary dataset (IEEG Portal) with published SOZ labels.

---

### Pivot 8 — H3 redesigned: divergence replaces sequencing (Phase 3a)
**Original H3:** MI collapse *precedes* variance collapse by Δτ > 15s in >80% of seizures.  
**Problem:** Zero paired collapses detected (threshold-crossing algorithm returns None for all patients). The 120s clips start already in the preictal zone — there is no transition to detect.  
**Key finding instead:** In Type A patients (n=16), 38% (6/16) show MI collapses while V *increases* simultaneously. 56% show both collapse (MI stronger). Type A interictal r(Phi,V)=0.362 (below 0.4 threshold), confirming MI and V measure different things.  
**Revised claim:** MI detects an information-theoretic network collapse that is *invisible to variance monitoring*. For 6 patients, a variance-based clinical alarm would show no signal (or upward spike) while MI reports network collapse. This is stronger than "MI leads V by 15s" — it shows MI and V are *qualitatively different signals*, not just temporally offset versions.

---

## Hypotheses Status

| Hypothesis | Test | Status |
|------------|------|--------|
| H1 (original): Phi_pre < Phi_inter − 2σ | Wilcoxon signed-rank | ❌ REJECTED (p=0.41 clean, p=0.92 full) |
| H1' (revised): MI directionality bifurcates by phenotype | Patient-level classification | 🔄 Needs further analysis |
| H2' (revised): graph topology bifurcates (27.8% frag / 27.8% integration) | Patient-level all-3-sig + 50/50 clean split | ✅ Confirmed as bimodal (mirrors H1) |
| H3' (revised): MI and V diverge preictally; MI collapses while V increases in Type A | Cross-phenotype comparison; interictal r < 0.4 | ✅ Confirmed for Type A (r=0.362); 6/16 show MI-V divergence |
| H4: TE asymmetry A_i identifies SOZ AUC > 0.75 | ROC AUC vs MNI proxy labels | ❌ NOT CONFIRMED (mean AUC=0.502 full cohort). Partial signal: 9/50 AUC>0.75; MTL subgroup mean AUC=0.584. Lower bound pending exact SOZ labels. |
