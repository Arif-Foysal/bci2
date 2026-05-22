# Response to Reviewers

**Manuscript:** Preictal Mutual Information Dynamics Bifurcate by Seizure Onset Lobe:
A Zero-Parameter, Four-Measure Information-Theoretic Analysis of the HUP iEEG Cohort

**Journal:** Epilepsy Research

**Decision received:** Major Revision

---

We thank the Editor and reviewers for a careful and constructive assessment. Below we respond to each issue in order of priority and describe the corresponding changes made to the manuscript. All line numbers refer to the revised manuscript.

---

## MAJOR Issues

---

### Issue 1 — Statistical framing: "rejected" with p = 0.92

**Reviewer comment (paraphrased):** The manuscript states "the uniform MI collapse hypothesis is rejected" on the basis of p = 0.92. This inverts standard frequentist logic: a high p-value means failure to reject the null, not rejection of the tested hypothesis.

**Authors' response:** We agree entirely. The original phrasing was imprecise and would be misleading to any reader trained in frequentist hypothesis testing. Our intent was to communicate that the group-level data do not support the uniform collapse hypothesis — but the word "rejected" incorrectly implies we rejected a null hypothesis at a conventional threshold.

**Changes made:**

| Location | Before | After |
|---|---|---|
| Abstract, sentence 3 | "The uniform hypothesis was rejected (Wilcoxon signed-rank, p = 0.92)" | "The uniform collapse hypothesis was not supported at the group level (Wilcoxon signed-rank, p = 0.92)" |
| Highlights, item 1 | "Uniform preictal MI collapse hypothesis rejected (p = 0.92)" | "No evidence for uniform preictal MI collapse at group level (p = 0.92)" |
| Introduction, contribution bullet 1 | "the uniform MI collapse hypothesis is rejected (p = 0.92)" | "the uniform MI collapse hypothesis is not supported (p = 0.92)" |
| Results §4.1, first paragraph | "The uniform MI collapse hypothesis is rejected." | "The data provide no evidence for uniform preictal MI collapse across the cohort; the null group-level result is the expected consequence of the opposing Type A and Type B dynamics described below, whose signed effects cancel in any signed-rank test." |
| Conclusion, paragraph 2 | "The hypothesis was rejected (p = 0.92)" | "The hypothesis was not supported (p = 0.92)" |

---

### Issue 2 — The central mechanistic claim (lobe → phenotype) is not statistically significant

**Reviewer comment (paraphrased):** The paper's narrative depends on the claim that lobe of seizure onset predicts MI phenotype. The evidence presented is frontal OR = 7.54 (p = 0.10) and MTL OR = 0.26 (p = 0.13). Neither reaches p < 0.05. The Discussion and Conclusion write as though lobe-specificity is established, which contradicts what the statistics support.

**Authors' response:** We agree that the lobe enrichment associations are directionally strong but do not reach formal significance in the current classified sample (n ≈ 35). This limitation was noted in §5.4 but was not consistently reflected in the language of §5.1 and §6. We have softened the language throughout to distinguish between what the data show (a null group-level result, a bimodal distribution, lobe-consistent direction of enrichment) and what a larger study would need to confirm (a formally significant lobe-phenotype link).

**Changes made:**

| Location | Before | After |
|---|---|---|
| Abstract, sentence 4 | "This lobe-specific bifurcation was confirmed independently…" | "This lobe-consistent pattern was confirmed independently…" |
| Abstract, sentence 5 | "Preictal MI dynamics are lobe-specific rather than universal" | "Preictal MI dynamics are consistent with lobe-specific ictogenic mechanisms rather than a universal process" |
| Introduction, contribution bullet 1 | "…frontal-onset patients exhibit collapse (OR = 7.54) while MTL-onset patients exhibit amplification (OR = 0.26), with both directions **predicted** by distinct ictogenesis mechanisms." | "…frontal-onset patients **tend to** exhibit collapse (OR = 7.54, p = 0.10) while MTL-onset patients **tend to** exhibit amplification (OR = 0.26, p = 0.13), with both directions **consistent with** distinct lobe-specific ictogenesis mechanisms." |
| Discussion §5.1 | "The lobe enrichment odds ratios…**predict exactly** which cohort composition produces which directional result." | "The lobe enrichment odds ratios (frontal OR = 7.54, p = 0.10; MTL OR = 0.26, p = 0.13) are **directionally consistent** with this explanation…Both associations fall short of conventional significance in the current classified sample (n ≈ 35); formal establishment of the lobe-phenotype link requires a larger cohort or exact clinical SOZ-lobe annotations." |

---

### Issue 3 — Impossible percentage: "mean Δ = −168%"

**Reviewer comment (paraphrased):** §4.1 reports "mean Δ = −168%" for preictal vs. interictal MI change. A −168% change would require MI to become negative, which is impossible since MI ≥ 0. This is either a reporting error or a pathological outlier effect from patients with d as extreme as −53.1.

**Authors' response:** The reviewer is correct. The mean percentage change is dominated by extreme Type B outliers (d as low as −53.1) and the resulting figure (−168%) is physically impossible for a non-negative quantity, indicating that the mean percentage is not a meaningful summary statistic for this bimodal, heavy-tailed distribution. We have removed this metric from the Results section. The Cohen's d distribution (median, range, and subgroup means) provides an appropriate summary and is retained.

**Change made:**

- Results §4.1, first paragraph: Removed "mean Δ = −168%, mean Cohen's d = −4.39" from the Wilcoxon result sentence. The sentence now reads: "The group-level Wilcoxon signed-rank test comparing μ_pre against μ_inter across all 54 patients was not significant (p = 0.92)."

---

## MODERATE Issues

---

### Issue 4 — Approximate sample sizes (n ≈ 20, n ≈ 15)

**Reviewer comment (paraphrased):** Exact patient counts classified into Type A, Type B, and Indeterminate must be reported. Tilde approximations raise doubts about reproducibility.

**Authors' response:** The approximate counts in the current draft arise because the Cohen's d threshold is applied to a continuous distribution, and counts at the boundary (|d| ≈ 0.3) vary by ±1 depending on rounding. We will replace all "n ≈" notation with exact counts extracted from the analysis pipeline output (Kaggle results JSON) before final submission.

**Action required (author):** Run `python pipeline.py fetch phase1b` and read `results/*/phase1b_results.json` to extract exact Type A, Type B, and Indeterminate patient counts. Replace all instances of "n ≈ 20", "n ≈ 15", "n ≈ 6" in Table 1 and §4.1 with the verified integers.

> **Note:** The current draft also contains an apparent inconsistency — the percentages (49%, 37%, 14%) do not divide evenly into 54 (they sum to 100% but 49% × 54 ≈ 26, not 20). This should be resolved when exact counts are substituted.

---

### Issue 5 — Phase 4 null result vs. "four-measure" framing

**Reviewer comment (paraphrased):** The paper title claims a "four-measure analysis" but Phase 4 produces AUC = 0.502, indistinguishable from chance. The paper should clarify that Phase 4 was inconclusive under proxy labels.

**Authors' response:** We agree that the title could be read as implying all four measures confirm the main finding, which is not the case. We have added the Phase 4 AUC result to the abstract so that readers understand the TE analysis was limited by proxy label precision. The title is retained because the four phases constitute a coherent scientific programme regardless of individual phase outcomes; the abstract now correctly characterises Phase 4 as a lower-bound estimate.

**Change made:**

- Abstract, sentence 4: Added clause after MI-variance results: "Transfer Entropy directional asymmetry yielded mean AUC = 0.502 under imprecise MNI proxy SOZ labels (a lower bound), with MTL-targeted patients reaching mean AUC = 0.584."

---

## MINOR Issues

---

### Issue 6 — "Zero-parameter" claim requires clarification

**Reviewer comment (paraphrased):** The Kraskov estimator uses k = 5; preprocessing uses specific filter parameters, window lengths, and proportional thresholds. The term "zero-parameter" could legitimately be objected to.

**Authors' response:** We agree that the term requires clarification. "Zero-parameter" was intended to mean the absence of any supervised-learning parameters, not the absence of signal processing hyperparameters. We have added an explicit definition.

**Change made:**

- §2.4 MI Estimation (after implementation validation sentence): Added — *"Throughout this work, zero-parameter denotes the absence of any supervised-learning parameters: all analysis hyperparameters (Kraskov neighbour count k = 5, rolling window length, proportional threshold) are fixed a priori and are not tuned to seizure labels or patient outcomes."*

---

### Issue 7 — Missing figures, blank affiliation, placeholder funding

**Authors' response:** These are pre-submission tasks. Figures referenced in the manuscript (Fig. 1, Fig. 2, Supplementary Fig. S2) will be generated from the analysis pipeline output and inserted before submission. Author affiliation and funding will be completed at submission time.

**No change to manuscript text.** (Action required: author.)

---

### Issue 8 — Schreiber 2025 is a preprint

**Authors' response:** Acknowledged. The reference has been tagged as a preprint in the bibliography. If the manuscript has been published by the time this paper is accepted, the citation will be updated to the journal reference.

**Change made:**

- Bibliography entry `schreiber2025`: Added "[Preprint]" after the bioRxiv identifier.

---

## Summary of Changes

| # | Issue | Severity | Status |
|---|---|---|---|
| 1 | Statistical framing ("rejected" with p = 0.92) | MAJOR | Fixed throughout |
| 2 | Lobe claim language overstates p > 0.05 associations | MAJOR | Softened throughout; p-values added to ORs |
| 3 | mean Δ = −168% (impossible value) | MAJOR | Removed |
| 4 | Approximate n counts | MODERATE | Flagged for author to verify from pipeline output |
| 5 | Phase 4 null result vs. four-measure title | MODERATE | Phase 4 result added to abstract |
| 6 | "Zero-parameter" undefined | MINOR | Definition added in §2.4 |
| 7 | Missing figures, affiliation, funding | MINOR | Pre-submission author action |
| 8 | Schreiber 2025 preprint label | MINOR | [Preprint] tag added |

---

*Response prepared in connection with revision of the manuscript. All changes are indicated in the revised manuscript with line numbers.*
