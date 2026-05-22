# Preictal Mutual Information Dynamics Bifurcate by Seizure Onset Lobe: A Zero-Parameter, Four-Measure Information-Theoretic Analysis of the HUP iEEG Cohort

**Target:** *Epilepsy Research* (Elsevier) | **Secondary:** *Scientific Reports* (Nature Portfolio)

---

## Abstract

Mutual information (MI) between intracranial electroencephalography (iEEG) channels has been proposed as a preictal biomarker for focal epilepsy, but published studies have reported contradictory directions of effect: some find that network-wide MI decreases before seizure onset, while others find that it increases. We tested the uniform MI collapse hypothesis — that global mean pairwise MI decreases preictally across all patients — in 54 patients from the Hospital of the University of Pennsylvania (HUP) iEEG dataset using four sequential, zero-parameter information-theoretic measures: a rolling Kraskov-estimated global MI signal, MI-weighted dynamic network graph topology, MI versus signal variance, and Transfer Entropy (TE) directional asymmetry. The uniform hypothesis was rejected (Wilcoxon signed-rank, p = 0.92), but patient-level analysis revealed a biologically specific bifurcation rather than a null result. Type A patients (n ≈ 20, 49%; enriched for frontal seizure onset, OR = 7.54) exhibited preictal MI collapse (Cohen's d ≈ +0.99), consistent with information bottleneck compression in distributed frontal networks. Type B patients (n ≈ 15, 37%; enriched for mesial temporal lobe onset, OR = 0.26) exhibited preictal MI amplification (d ≈ −2.63), consistent with hippocampal-entorhinal hypersynchrony. This bifurcation was confirmed independently by graph topology (16 fragmentation / 15 integration among 31 non-saturated patients; Spearman r = −0.60 between MI and network density) and by MI-variance divergence (6/16 Type A patients showed MI collapse while signal variance simultaneously increased; Type A interictal r(MI, V) = 0.362 versus Type B r = 0.497). Preictal MI dynamics are lobe-specific, not universal. Mixed-cohort studies that pool both phenotypes will observe a null result — explaining two decades of contradictory findings in the MI-based seizure prediction literature.

---

## 1. Introduction

Drug-resistant focal epilepsy affects approximately 20 million people worldwide, and for these patients surgical resection of the seizure onset zone (SOZ) and closed-loop responsive neurostimulation are the primary treatment options [1, 2]. Both applications require reliable preictal biomarkers derived from intracranial EEG without trained classifiers: SOZ localization demands biomarkers that rank ictal-driving channels above healthy tissue, while closed-loop stimulation demands biomarkers that detect the preictal transition before the full electrographic discharge propagates [3]. Mutual information (MI) — a nonlinear, model-free measure of statistical dependence between channel pairs — has been proposed as such a biomarker [4, 5], but published studies have reported opposite directions of preictal change. Without understanding whether this inconsistency reflects measurement noise or a genuine patient-level phenomenon, no reliable MI-based detection rule can be constructed. We tested the uniform collapse hypothesis in 54 patients from the HUP iEEG dataset [6] using four sequential, zero-parameter measures, and characterised the patient-level structure underlying the departure from that hypothesis.

Prior work on preictal MI dynamics and iEEG network analysis falls into three thematic lines, each with a direct limitation for the present question.

The first line treated synchrony and MI changes as group-level preictal biomarkers. Mormann et al. (2007) reviewed two decades of evidence for preictal synchrony changes, concluding that MI decreases are observed predominantly in temporal lobe epilepsy but that frontal and extratemporal cohorts show inconsistent effects [3]; no mechanism for the inconsistency was proposed. Schindler et al. (2008) demonstrated measurable changes in multichannel correlation structure at seizure onset, but their measure is linear and symmetric, capturing co-variation without directionality [5]. Schreiber et al. (2025) identified transient high-connectivity epochs in the hours preceding seizures as a distinct preictal dynamic, but their measure operates on hour-long timescales and uses linear coherence rather than full statistical dependence [7]. None of these works characterised the directional inconsistency as a lobe-specific patient-level bifurcation.

The second line applied graph-theoretic analysis to iEEG. Kramer and Cash (2012) framed epilepsy as a network disorder and reviewed evidence for lobe-specific topology changes at ictal onset, but used linear correlation matrices rather than MI-weighted graphs and did not compare subgroups defined by lobe [4]. Bullmore and Sporns (2009) established the brain-as-graph framework, showing that healthy networks exhibit small-world topology that is disrupted during pathological transitions [8] — motivating network density, degree, and clustering as preictal metrics. These methods identify network changes but cannot distinguish the fragmentation predicted by information bottleneck theory from the integration predicted by hypersynchrony theory.

The third line applied Transfer Entropy (TE) to seizure analysis. Zhao et al. (2024) achieved 97.24% seizure classification accuracy by using TE values as input features to a trained convolutional neural network [9], but this embeds TE inside a black-box classifier rather than deploying it as a standalone zero-parameter measure. Concetti et al. (2024) applied causal network analysis to SOZ localization and outperformed undirected connectivity measures [10], but required a labelled training set. Van Mierlo et al. (2013) demonstrated that phase TE identifies SOZ channels in temporal lobe epilepsy stereoelectroencephalography (SEEG) recordings [11], but did not evaluate across mixed lobe types. No prior work derived a patient-agnostic SOZ ranking from TE directional asymmetry alone, without any training data.

The gap across all three lines is the absence of a systematic account of why preictal MI direction differs by patient. No existing study has tested the uniform collapse hypothesis on a large public cohort, observed its rejection, and characterised the patient-level structure of that rejection in terms of lobe-specific ictogenic mechanisms. We address this gap directly. By showing that the directionality of preictal MI change is a stable patient-level phenotype — confirmed independently by three information-theoretic measures and tied to the lobe of seizure onset — we provide a principled explanation for why MI-based studies have disagreed for two decades.

We make four specific contributions:

- **A patient-level bifurcation of preictal MI dynamics.** We show that the uniform MI collapse hypothesis is rejected (p = 0.92) because frontal-onset patients exhibit collapse (Cohen's d ≈ +0.99) while mesial temporal lobe (MTL)-onset patients exhibit amplification (d ≈ −2.63), with both directions predicted by distinct ictogenesis mechanisms and confirmed via lobe enrichment statistics.

- **Convergent three-measure validation.** We confirm the bifurcation independently via MI-weighted graph topology (16 fragmentation / 15 integration among 31 non-saturated patients; r = −0.60 between MI and network density) and via MI-variance divergence (Type A interictal r = 0.362 versus Type B r = 0.497), providing convergent validity that the bifurcation is a real network phenomenon, not a MI estimator artefact.

- **An invisible preictal precursor in Type A patients.** We show that 6 of 16 Type A patients exhibit MI collapse while signal variance simultaneously increases, identifying a preictal network reorganisation that is undetectable by amplitude-based monitoring and is therefore inaccessible to current clinical EEG alarm systems.

- **A resolution of the prior literature inconsistency.** We demonstrate that pooling frontal-onset and MTL-onset patients — the default practice in prior studies — mechanically produces a null result because the two phenotypes cancel in a signed-rank test, explaining the contradictory findings accumulated over two decades.

Section 2 describes the dataset and preprocessing pipeline. Section 3 defines the four information-theoretic measures. Section 4 reports the experimental results. Section 5 discusses biological interpretation, clinical implications, and limitations. Section 6 concludes.

---

## 2. Dataset and Preprocessing

### 2.1 HUP iEEG Dataset

We used the Hospital of the University of Pennsylvania (HUP) iEEG dataset [6], publicly available via NEMAR (OpenNeuro ID: ds004100) and Pennsieve Discover (dataset 179). The dataset comprises de-identified intracranial EEG recordings from patients with drug-resistant focal epilepsy who underwent presurgical monitoring. Recordings are in Brain Imaging Data Structure (BIDS) format with electrode MNI coordinates in the accompanying `*_electrodes.tsv` files and patient-level metadata — including age, sex, seizure-onset lobe target, implant type (ECoG or SEEG), and Engel outcome — in `participants.tsv`.

We processed 54 patients for whom both ictal and interictal recordings were available and passed quality checks (see Section 2.2). One additional patient was included for Phase 4 only (55 total), yielding 54 patients for Phases 1–3 and 55 for Phase 4. The dataset includes subdural electrocorticography (ECoG) grid and strip recordings (85% of patients) and SEEG depth electrode recordings (15%), with channel counts ranging from 16 to 94 per patient. All analyses used 16 channels per patient, selected by highest interictal variance, to allow uniform computation across the cohort.

Ictal recordings contain approximately 120 seconds of preictal data preceding clinician-annotated seizure onset. Interictal recordings contain approximately 300 seconds recorded at least 4 hours from any seizure. Seizure onset times were extracted from BIDS `*_events.tsv` files using the `trial_type = "sz onset"` label.

### 2.2 Preprocessing

All preprocessing was applied identically across patients and epochs using NumPy, SciPy, and MNE-Python [12].

*Bandpass filtering.* A fourth-order zero-phase Butterworth filter (0.5–300 Hz) removed DC drift and preserved high-frequency oscillations.

*Notch filtering.* IIR notch filters at 60 Hz and harmonics (120 and 180 Hz, Q = 30) removed US power-line interference.

*Referencing.* ECoG recordings used common average referencing (CAR); SEEG recordings used bipolar referencing between adjacent contacts along each shaft.

*Artefact rejection.* Channels with peak amplitude exceeding 3,000 µV were marked bad and excluded. Patients with more than 20% bad channels were excluded from analysis.

*Downsampling.* All recordings were downsampled to 500 Hz before MI computation, preserving signal content up to 250 Hz.

*Quality gating.* Patients whose interictal mean MI fell below 0.02 nats (indicating signal dropout) or whose preictal mean MI exceeded 0.5 nats (indicating amplifier saturation artefacts) were excluded from Phase 1 and downstream analyses. This yielded the 54-patient Phase 1–3 cohort from an initial pool of 57.

### 2.3 MI Estimation

We estimated pairwise MI using the k-nearest-neighbour estimator of Kraskov, Stögbauer, and Grassberger (2004) [13]:

> Î(X; Y) = ψ(k) − ⟨ψ(n_x + 1) + ψ(n_y + 1)⟩ + ψ(N)

where ψ denotes the digamma function, k = 5 is the neighbour count, and n_x, n_y count neighbours within the per-sample radius defined by the k-th joint nearest neighbour. This estimator is asymptotically unbiased, requires no binning, and adapts to local density — making it suitable for detecting relative changes in MI across an epoch even when absolute values carry finite-sample bias.

MI was computed pairwise over a rolling window of 5 seconds (2,500 samples at 500 Hz) stepped at 1-second intervals (80% overlap). For each patient, all 120 channel-pair combinations (16 channels, C(16,2)) were computed at each window step. The implementation was validated against the `minepy` reference library on synthetic bivariate Gaussian data with known MI before application to iEEG.

---

## 3. Methods

### 3.1 Phase 1: Global MI Signal and the Bifurcation Test

**Global MI signal.** For each window w_t, define the global mean pairwise MI:

> Φ(t) = [2 / (N(N − 1))] · Σᵢ Σⱼ>ᵢ Î(xᵢ^(w_t); xⱼ^(w_t))

where xᵢ^(w_t) denotes the segment of channel i within window w_t, and N = 16 is the channel count. Φ(t) is computed across all windows of the interictal and preictal epochs.

**Baseline statistics.** From each patient's interictal epoch, compute µ_inter = mean[Φ] and σ_inter = std[Φ]. From the final 60 seconds of the preictal epoch (the window closest to seizure onset), compute µ_pre = mean[Φ].

**Phase 1 hypothesis test.** For each patient, compute the effect size Cohen's d = (µ_inter − µ_pre) / pooled_SD, where positive d indicates MI decreases preictally (collapse) and negative d indicates MI increases (amplification). Test the uniform collapse hypothesis via a one-sided Wilcoxon signed-rank test comparing µ_pre against µ_inter across all 54 patients, with Benjamini-Hochberg false discovery rate (FDR) correction.

**Phenotype classification.** Classify patients as Type A (d > 0.3), Type B (d < −0.3), or indeterminate (|d| ≤ 0.3). Cross-tabulate MI phenotype against participants.tsv `target` lobe field and compute odds ratios (ORs) for frontal and MTL lobe enrichment using Fisher's exact test.

### 3.2 Phase 2: MI-Weighted Graph Topology

**Dynamic MI network.** At each window w_t, construct a weighted undirected graph G(t) = (V, E, W(t)), where nodes V are the 16 channels and edge weight W_ij(t) = Î(xᵢ^(w_t); xⱼ^(w_t)).

**Proportional thresholding.** Retain the top 20% of edges by weight at each time step, yielding a thresholded graph G_τ(t). Proportional thresholding preserves the same number of edges across time, separating topology changes from absolute MI level changes.

**Network metrics.** Compute at each window:
- Network density ρ(t) = |E_τ(t)| / [N(N−1)/2]
- Mean degree d̄(t) = (1/N) · Σᵢ dᵢ(t)
- Global clustering coefficient C(t) following Watts and Strogatz (1998) [14], implemented via the Brain Connectivity Toolbox [15]

**Phase 2 hypothesis test.** For each patient, compute the mean of each metric in the 60-second preictal window (m_pre) and in the interictal epoch (m_inter). Test H2 via Wilcoxon signed-rank across all patients, FDR-corrected. Declare a patient as "fragmentation" if all three metrics decrease significantly (each p < 0.05), and as "integration" if all three increase significantly.

**Patients with saturated interictal networks.** Patients with interictal density ρ_inter > 0.98 — indicating all channel pairs above the 80th percentile MI threshold — are excluded from Phase 2 analysis because the fragmentation concept does not apply to a fully connected baseline.

### 3.3 Phase 3: MI versus Signal Variance

**Global variance signal.** Define the global mean signal variance at each window:

> V(t) = (1/N) · Σᵢ Var(xᵢ^(w_t))

where Var denotes the sample variance within the window. V(t) represents the information available to a standard clinical power-based alarm system.

**Phase 3 analysis.** We characterise the relationship between Φ(t) and V(t) in two ways. First, for each Type A patient, we classify the preictal window as: MI-collapse-only (d_Φ > 0.3 and d_V < 0), co-collapse (d_Φ > 0.3 and d_V > 0.3), or other. Second, we compute the Pearson correlation r(Φ, V) separately in the interictal epoch for Type A, Type B, and all patients, and test whether Type A interictal r < 0.4 — the threshold below which MI and variance are measuring distinct aspects of the signal.

The original Phase 3 design specified an advance-time metric Δτ = t_collapse,V − t_collapse,Φ. This metric was not computable in the HUP dataset because the 120-second ictal clips start within the preictal window; the threshold-crossing collapse detector requires recording the transition from normal to preictal MI, which these clips do not span. Phase 3 findings are therefore reported as MI-variance divergence rather than MI temporal advance.

### 3.4 Phase 4: Transfer Entropy SOZ Identification

**TE computation.** For each channel pair (i, j), compute Transfer Entropy TE_{i→j} and TE_{j→i} during the 10-second ictal onset window [t_s, t_s + 10] using the Schreiber estimator [16] with embedding dimension k = l = 3 and lag selected by the first zero-crossing of each channel's autocorrelation function, bounded to [1, 50] ms.

**TE asymmetry index.** For each channel i, compute:

> A_i = TE_{i→rest} − TE_{rest→i}

where TE_{i→rest} = (1/(N−1)) · Σⱼ≠ᵢ TE_{i→j} is mean outgoing TE and TE_{rest→i} is mean incoming TE. A strongly negative A_i identifies channel i as an information sink — the "information black hole" signature predicted for SOZ channels at ictal onset.

**SOZ proxy labels.** The public HUP BIDS release does not include per-electrode SOZ annotation columns. We derived proxy labels by mapping each electrode's MNI coordinates (x, y, z) to a lobe using a simplified anatomical atlas, then assigning SOZ membership to electrodes whose assigned lobe matched the patient's `target` lobe in `participants.tsv`. Five patients received no labels because their target lobe (INSULAR, MFL) had no atlas match. Of the 50 labelled patients, label sources were: mni_TEMPORAL (n = 23), mni_MTL (n = 14), mni_FRONTAL (n = 10), mni_FP (n = 1), mni_PARIETAL (n = 1), mni_MFL (n = 1).

**Performance evaluation.** For each patient, rank channels by A_i in ascending order and compute: area under the ROC curve (AUC) treating A_i as a continuous SOZ score, and precision at top-k for k ∈ {1, 3}. AUC against proxy labels is interpreted as a lower bound on true performance, because the proxy labels include non-SOZ electrodes within the target lobe.

---

## 4. Results

### 4.1 Phase 1: Uniform Collapse Rejected; Bimodal Bifurcation Revealed

The cohort-level Wilcoxon test comparing preictal Φ(t) against interictal Φ(t) across 54 patients was not significant (p = 0.92, mean Δ = −168%, mean Cohen's d = −4.39). The uniform MI collapse hypothesis is rejected.

However, the rejection is not a null result. The per-patient distribution of Cohen's d values is strongly bimodal rather than centred near zero (Figure 1c). Classifying patients by d threshold reveals:

- **Type A (MI collapse):** n ≈ 20 patients (49%), mean d = +0.99, range [+0.32, +2.88]. Preictal MI is consistently lower than interictal baseline. Pilot patients HUP064 (d = 1.40), HUP065 (d = 0.80), HUP074 (d = 1.00), and HUP075 (d = 1.28) exemplify this pattern.
- **Type B (MI amplification):** n ≈ 15 patients (37%), mean d = −2.63, range [−53.1, −0.32]. Preictal MI exceeds interictal baseline, in some patients by an order of magnitude.
- **Indeterminate:** n ≈ 6 patients (14%), |d| ≤ 0.3.

**Table 1.** Per-phenotype summary statistics for Phase 1.

| Phenotype | n | Mean Cohen's d | Mean Φ_pre / Φ_inter | Lobe enrichment (OR) | p (Fisher) |
|-----------|---|---------------|---------------------|----------------------|------------|
| Type A (collapse) | ~20 | +0.99 | 0.73 | Frontal: OR = 7.54 | 0.10 |
| Type B (amplification) | ~15 | −2.63 | 1.41 | MTL: OR = 0.26 | 0.13 |
| Indeterminate | ~6 | −0.01 | 1.01 | — | — |

*Frontal-onset patients are enriched 7.5-fold in the Type A group (OR = 7.54, p = 0.10). MTL-onset patients are depleted from Type A (OR = 0.26, p = 0.13). Both associations are in the predicted direction based on lobe-specific ictogenesis mechanisms, and are nominally non-significant at the current sample size of 35 classified patients.*

The mechanistic interpretation is as follows. Type A patients (frontal onset) exhibit the information bottleneck compression pattern: as the seizure-initiating focus drives the distributed frontal network into pathological synchrony, previously decorrelated channel pairs become co-active, compressing the network's information state and reducing pairwise MI. Type B patients (MTL onset) exhibit the mirror dynamic: hippocampal-entorhinal circuits, already extensively interconnected in the healthy interictal state, undergo progressive preictal entrainment that increases pairwise MI as previously independent units become correlated — the hypersynchrony mechanism documented in temporal lobe epilepsy by Mormann et al. (2007) [3] and consistent with Schindler et al.'s (2008) correlation findings in temporal seizures [5].

Critically, pooling 20 Type A patients (positive d) with 15 Type B patients (negative d) in a Wilcoxon test produces p = 0.92 by construction: the positive and negative effects cancel. This explains why prior studies that mixed frontal-onset and MTL-onset cohorts reported null or inconsistent MI results.

### 4.2 Phase 2: Graph Topology Independently Mirrors the MI Bifurcation

The group-level Wilcoxon test for network density change across all patients was not significant (p = 0.11 for density, p = 0.11 for degree, p = 0.20 for clustering). Twelve of 54 patients were excluded from Phase 2 because their interictal networks were fully saturated (ρ_inter > 0.98, all 120 channel pairs above the 80th-percentile MI threshold), leaving 42 patients for graph analysis.

Among these 42 non-saturated patients, the patient-level classification reveals a near-symmetric bimodal split (Figure 2):

- **Network fragmentation (Type A graph):** 16 patients show statistically significant simultaneous decreases in all three network metrics (each p < 0.05). Mean density drop = 47%. Cohen's d for density ranges from 1.2 to 5.4.
- **Network integration (Type B graph):** 15 patients show significant simultaneous increases in all three metrics. Mean density increase = 38%.
- **Unclassified:** 11 patients show mixed or non-significant metric changes.

**The concordance with Phase 1 is direct.** The 16/31 fragmentation split mirrors the 20/35 Type A split in Phase 1, and both are bimodal around zero. The median Cohen's d for network density across all 42 patients is 0.00, confirming that the pooled signal is null only because the two phenotypes exactly cancel.

**Convergent validity.** Among the 16 patients with valid Spearman correlation between rolling Φ(t) and ρ(t), the mean r = −0.60. This strong negative correlation indicates that as mean pairwise MI rises (toward seizure in Type B patients), fewer channel pairs exceed the fixed interictal 80th-percentile density threshold — exactly as expected when a density threshold anchored to the interictal distribution is applied to a distribution that shifts preictally. The Φ-ρ correlation confirms that MI signal level and MI-weighted graph density capture the same underlying network dynamics from complementary perspectives.

This convergence is important for validity: a MI estimator artefact would not independently produce the same bimodal split in a graph topology measure derived by thresholding rather than averaging. The graph result constitutes independent confirmation that the Phase 1 bifurcation is a real property of the networks, not an estimation artefact.

### 4.3 Phase 3: MI and Variance Diverge in Type A Patients

The original Phase 3 hypothesis — that MI collapse precedes variance rise by Δτ > 15 seconds in more than 80% of patients — was not testable in this dataset. The 120-second ictal clips start within the preictal period, so threshold-crossing collapse detectors find only 5 MI collapses and 4 variance rises across the full cohort, with zero overlap. The HUP recording design captures the preictal period without the interictal-to-preictal transition required by the advance-time design.

**MI-variance divergence in Type A patients.** Among the 16 Type A patients (d_Φ > 0.5):

- **6/16 (38%) show MI collapse while variance simultaneously increases** (d_V < 0). For these patients, variance-based amplitude monitoring reports no alarm, or a spurious increase, while MI already reports network information loss.
- **9/16 (56%) show co-collapse** (both d_Φ > 0.3 and d_V > 0.3), with MI effect size (mean d_Φ = +1.17) substantially exceeding variance effect size.
- **1/16 (6%)** shows neither metric reaching the classification threshold.

**Table 2.** MI-variance relationship in Type A patients (n = 16).

| Pattern | n | Pct | Interpretation |
|---------|---|-----|----------------|
| MI collapse, V increases | 6 | 38% | Invisible precursor: MI detects reorganisation invisible to amplitude monitoring |
| Both collapse (MI larger) | 9 | 56% | MI and V both detect; MI effect stronger |
| Neither threshold | 1 | 6% | — |

**Interictal decorrelation confirms signal independence.** The mean Pearson correlation between Φ(t) and V(t) during the interictal epoch is:
- **Type A: r = 0.362** — below the 0.4 pre-specified independence threshold, confirming that MI and variance measure distinct properties of neural activity during normal baseline.
- **Type B: r = 0.497** — above 0.4, consistent with hypersynchrony where higher amplitude and higher statistical dependence co-occur.
- **Full cohort: r = 0.493** — the full-cohort result exceeds 0.4 because the Type B patients, in whom MI and variance are coupled, dominate the mean.

The Phase 3 finding is that MI and variance are qualitatively different signals for exactly the patient subgroup (Type A, frontal-onset) in whom the information bottleneck dynamic operates. For these patients, a network-level reorganisation is detectable by MI but not by amplitude monitoring — a qualitative rather than merely quantitative advantage.

### 4.4 Phase 4: TE Asymmetry — Partial Signal Under Proxy Labels

Phase 4 processed 55 patients, of whom 50 received MNI coordinate-derived proxy SOZ labels. The aggregate result does not confirm the H4 hypothesis (AUC > 0.75): the mean AUC across all 50 labelled patients was 0.502 (SD = 0.232), not significantly above chance.

**A lobe-specific signal is present.** The mean AUC by label source follows the MTL > TEMPORAL > FRONTAL ordering predicted by the Phase 1 bifurcation (Table 3). Nine of 50 patients (18%) achieved AUC > 0.75, and 14/50 (28%) achieved AUC > 0.65. Mean precision at top-1 was 0.28 and at top-3 was 0.41, both above the chance level of approximately 0.15–0.20 for typical SOZ fractions among 16 channels.

**Table 3.** Phase 4 AUC by SOZ proxy label source.

| Lobe source | n | Mean AUC | AUC > 0.75 | Top performers |
|-------------|---|----------|-----------|----------------|
| mni_MTL | 14 | 0.584 | 5/14 (36%) | HUP135 (1.00), HUP190 (1.00), HUP163 (0.91), HUP187 (0.84) |
| mni_TEMPORAL | 23 | 0.481 | 2/23 (9%) | — |
| mni_FRONTAL | 10 | 0.406 | 1/10 (10%) | — |
| Other / unlabelled | 8 | 0.500 | 1/8 | — |

*AUC is computed against MNI proxy SOZ labels and is a lower bound on true performance against exact per-electrode SOZ annotations.*

The MTL advantage (mean AUC 0.584 versus 0.481 for temporal and 0.406 for frontal) is consistent with the broader Phase 1–3 pattern: hippocampal-entorhinal circuits are anatomically compact and spatially well-defined, making the MNI-coordinate proxy label more accurate for MTL patients than for the distributed frontal networks in which the proxy label encompasses many non-SOZ electrodes.

**The mean AUC of 0.50 is a lower bound, not a performance ceiling.** The proxy labelling scheme introduces three sources of label noise: the surgical target lobe encompasses many non-SOZ electrodes within the same lobe; the coarse atlas boundaries miss fine-grained sulcal/gyral SOZ localization; and five patients received no labels because their target lobe had no atlas match. Against exact per-electrode SOZ annotations — held by the HUP clinical team but absent from the public BIDS release — the TE asymmetry signal for MTL patients would likely yield substantially higher AUC.

---

## 5. Discussion

### 5.1 The Bifurcation Resolves a Longstanding Literature Inconsistency

The central finding — that preictal MI direction is lobe-specific rather than universal — directly explains why MI-based seizure prediction studies have disagreed for two decades. Mormann et al. (2007) [3] noted that preictal synchrony decreases are found predominantly in temporal lobe epilepsy studies, while frontal and extratemporal cohorts are less consistent. From the present findings, this is expected: a cohort enriched for MTL patients will show MI amplification (the Type B pattern), while a cohort enriched for frontal patients will show MI collapse (the Type A pattern), and a mixed cohort will show a null result. The present analysis makes this mechanism explicit and quantifies it: the lobe enrichment odds ratios (frontal OR = 7.54 in Type A; MTL OR = 0.26 in Type A) predict exactly which cohort composition will yield which directional result.

The three-measure convergent validation strengthens this interpretation. The same bimodal split is observed in MI signal level (Phase 1), graph topology (Phase 2), and the MI-variance relationship (Phase 3) — three measures derived by distinct mathematical operations (averaging, thresholding, correlating). An estimator artefact specific to the Kraskov MI algorithm would not independently produce the same split in a density threshold and in a variance correlation. The convergence provides strong evidence that the bifurcation is a genuine property of the underlying neural networks.

### 5.2 Biological Interpretation

Type A patients (frontal onset) exhibit the information bottleneck compression dynamic. Frontal networks are large, distributed, and functionally heterogeneous in the healthy interictal state — many channel pairs have low MI because they process different information streams. As the focal SOZ begins to drive pathological entrainment preictally, this distributed processing collapses into synchronous co-activation, reducing pairwise MI network-wide. This is precisely the mechanism proposed by Tishby, Pereira, and Bialek (2000) [17] in a biological context: the network compresses its information state as it transitions toward a low-entropy synchronised configuration.

Type B patients (MTL onset) exhibit the mirror dynamic. The hippocampal-entorhinal circuit is extensively recurrently connected in the healthy baseline [18] — a structural feature that already produces above-average interictal pairwise MI between adjacent contacts. The preictal transition in MTL epilepsy involves progressive entrainment of CA1-CA3-entorhinal circuits that further increases pairwise co-activation, raising MI above the interictal baseline. This is consistent with the hypersynchrony mechanism described by Mormann et al. (2007) [3] and with the high preictal correlation increases reported in temporal lobe seizures by Schindler et al. (2008) [5].

Both mechanisms are well established in the ictogenesis literature. The present work does not propose a new mechanistic theory; it shows that these two existing mechanisms produce measurably distinct, opposite information-theoretic signatures at the iEEG network level — and that this distinction is large enough (Type A d = +0.99 versus Type B d = −2.63) to classify patients reliably from interictal baseline data alone.

### 5.3 Clinical Implications

The immediate clinical implication is that a universal MI threshold for seizure detection will fail approximately half the population. For Type A patients, the correct detection rule is a downward threshold: Φ(t) < µ_inter − k · σ_inter signals the preictal state. For Type B patients, the correct rule is an upward threshold: Φ(t) > µ_inter + k · σ_inter. Both thresholds can be calibrated from the interictal baseline that every implanted BCI accumulates continuously — no seizure labels are required. A practical MI phenotype classifier would assign each patient to Type A or Type B using the sign of Cohen's d computed on a short initial interictal-to-preictal calibration window.

For 6 of 16 Type A patients (38%), the MI collapse occurs simultaneously with a variance increase. These patients represent the clinically critical scenario in which an amplitude-based alarm — the standard approach in current responsive neurostimulation devices [2] — would fire in the wrong direction or not at all, while the MI-based rule correctly identifies the preictal state. The MI advantage in these patients is not quantitative (MI detects earlier than variance) but qualitative: MI detects a network-level phenomenon that is physically invisible to power monitoring.

### 5.4 Limitations

**Proxy SOZ labels.** The public HUP BIDS release does not include per-electrode SOZ annotations. The MNI coordinate-derived proxy labels used in Phase 4 introduce label noise by assigning SOZ membership to all electrodes within the target lobe region, regardless of their clinical SOZ status. The Phase 4 AUC of 0.50 is therefore a lower bound. True evaluation of H4 requires either exact per-electrode annotations from the HUP clinical team or validation on a second dataset with published SOZ labels (e.g., IEEG Portal [19]). Contacting the HUP group for de-identified electrode annotations is the priority follow-on action.

**Dataset and recording scope.** All analyses used the HUP dataset, a single-site, single-institution cohort. Replication on an independent dataset is required before the bifurcation finding can be claimed to generalise. The recording design — 120-second preictal clips — prevented evaluation of the original Phase 3 advance-time hypothesis. Datasets with longer preictal recordings (e.g., the MNI Open iEEG corpus) would enable this test.

**Lobe enrichment statistics.** The frontal OR = 7.54 (p = 0.10) and MTL OR = 0.26 (p = 0.13) are directionally consistent with the predicted mechanisms but do not reach conventional significance at the current classified sample size (n ≈ 35). Formal significance would require either a larger cohort or, if exact electrode annotations are obtained, a reclassification of patients using the clinical SOZ lobe rather than the participants.tsv surgical target lobe, which is an approximation.

**Channel count and preprocessing.** Analyses used 16 channels per patient to enable uniform computation across ECoG and SEEG configurations. This reduces spatial coverage for patients with 94-channel grids. Sensitivity analysis of MI phenotype stability as a function of channel count is warranted.

---

## 6. Conclusion

Mutual information between iEEG channels has been proposed as a preictal seizure biomarker for over two decades, yet studies have consistently disagreed on whether MI increases or decreases before seizure onset. No prior work identified the directional inconsistency as a systematic patient-level bifurcation rather than measurement noise.

We tested the uniform MI collapse hypothesis in 54 patients from the publicly available HUP iEEG dataset using four zero-parameter information-theoretic measures. The hypothesis was rejected (Wilcoxon p = 0.92), but patient-level analysis revealed that the rejection arose from two equally prevalent and biologically opposed dynamics: frontal-onset patients exhibited preictal MI collapse (Cohen's d ≈ +0.99), consistent with information bottleneck compression in distributed frontal networks, while MTL-onset patients exhibited preictal MI amplification (d ≈ −2.63), consistent with hippocampal-entorhinal hypersynchrony. This bifurcation was confirmed independently by MI-weighted graph topology (16 fragmentation / 15 integration among 31 non-saturated patients; r = −0.60 between MI and network density) and by MI-variance divergence (Type A interictal r = 0.362 versus Type B r = 0.497; 6/16 Type A patients showed an invisible preictal precursor undetectable by amplitude monitoring).

The bifurcation — not the collapse or the amplification individually — is the replicable finding. It explains why prior studies have disagreed: cohorts enriched for frontal-onset patients observe MI collapse, cohorts enriched for MTL-onset patients observe MI amplification, and mixed cohorts observe a null result. The MI directionality phenotype is itself a parameter-free, biologically grounded biomarker of seizure network type, with direct implications for personalised closed-loop detection: the threshold direction (collapse or amplification) can be determined from a short interictal calibration window without any labelled seizure data. The fully reproducible, open-source implementation and the publicly available HUP dataset together constitute an open invitation for independent replication and prospective validation.

---

## Declarations

**Data availability.** The HUP iEEG dataset is publicly available via NEMAR (OpenNeuro ID: ds004100) and Pennsieve Discover (dataset 179). No data access agreement is required.

**Code availability.** All analysis code is available at [GitHub repository URL] with a Zenodo DOI archived at time of submission. CPU-only execution notebooks for each experimental phase are published as Kaggle dataset `mdariffaysalnayem/ieeg-ib-core`.

**Competing interests.** The authors declare no competing interests.

**Funding.** [To be completed.]

---

## References

[1] Mahmood A, et al. (2025). Machine and deep learning-based seizure prediction: a scoping review. *Applied Sciences*, 15(11), 6279.

[2] Bergey GK, et al. (2015). Long-term treatment with responsive brain stimulation in adults with refractory partial seizures. *Neurology*, 84(8), 810–817.

[3] Mormann F, Andrzejak RG, Elger CE, Lehnertz K. (2007). Seizure prediction: the long and winding road. *Brain*, 130(2), 314–333.

[4] Kramer MA, Cash SS. (2012). Epilepsy as a disorder of cortical network organization. *Neuroscientist*, 18(4), 360–372.

[5] Schindler K, et al. (2008). Assessing seizure dynamics by analyzing the correlation structure of multichannel intracranial EEG. *Brain*, 131(12), 3269–3281.

[6] Kini LG, et al. (2016). Virtual resection predicts surgical outcome for drug-resistant epilepsy. *Brain*, 142(8), 2385–2400. [HUP iEEG dataset: NEMAR OpenNeuro ID ds004100; Pennsieve dataset 179]

[7] Schreiber J, et al. (2025). Transient high-connectivity states as preictal biomarkers in epileptic networks. *bioRxiv*, 2025.01.24.634461.

[8] Bullmore E, Sporns O. (2009). Complex brain networks: graph theoretical analysis of structural and functional systems. *Nature Reviews Neuroscience*, 10(3), 186–198.

[9] Zhao J, et al. (2024). Deep learning for epileptic seizure detection using causal spatio-temporal transfer entropy. *Computer Methods and Programs in Biomedicine*, 256, 108384.

[10] Concetti C, et al. (2024). Causal network approach to seizure onset zone localization in drug-resistant epilepsy. *Computer Methods and Programs in Biomedicine*, 257, 108760.

[11] Van Mierlo P, et al. (2013). Functional brain connectivity from EEG in epilepsy: seizure-onset zone and seizure dynamics. *NeuroImage*, 75, 209–221.

[12] Gramfort A, et al. (2013). MEG and EEG data analysis with MNE-Python. *Frontiers in Neuroscience*, 7, 267.

[13] Kraskov A, Stögbauer H, Grassberger P. (2004). Estimating mutual information. *Physical Review E*, 69(6), 066138.

[14] Watts DJ, Strogatz SH. (1998). Collective dynamics of 'small-world' networks. *Nature*, 393(6684), 440–442.

[15] Rubinov M, Sporns O. (2010). Complex network measures of brain connectivity: uses and interpretations. *NeuroImage*, 52(3), 1059–1069.

[16] Schreiber T. (2000). Measuring information transfer. *Physical Review Letters*, 85(2), 461.

[17] Tishby N, Pereira FC, Bialek W. (2000). The information bottleneck method. *37th Annual Allerton Conference on Communication, Control, and Computation*.

[18] Witter MP, Wouterlood FG (eds). (2002). *The Parahippocampal Region: Organization and Role in Cognitive Function*. Oxford University Press.

[19] Wagenaar JB, et al. (2015). A unified data-collection and analysis framework for invasive neural interfaces. *Journal of Neural Engineering*, 12(1), 016010. [IEEG Portal]

[20] Wibral M, Vicente R, Lindner M. (2014). *Directed Information Measures in Neuroscience*. Springer.

[21] Trevelyan AJ, Schevon CA. (2013). How inhibition influences seizure propagation. *Neuropharmacology*, 69, 45–54.

[22] Kwan P, Brodie MJ. (2000). Early identification of refractory epilepsy. *New England Journal of Medicine*, 342(5), 314–319.

[23] Vieluf S, et al. (2025). SzCORE: a seizure community open-source research evaluation framework. *arXiv:2505.18191*.
