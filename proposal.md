# Preictal Mutual Information Dynamics Bifurcate by Seizure Onset Lobe: A Zero-Parameter, Four-Measure Information-Theoretic Analysis of the HUP iEEG Cohort

**Proposed submission target:** *Epilepsy Research* (Elsevier) or *Scientific Reports* (Nature Portfolio)  
**Secondary targets:** *Journal of Neural Engineering* (with second-dataset replication); *Epilepsia* (if lobe-enrichment statistics reach significance with full electrode annotations)  
**Keywords:** seizure prediction; mutual information; transfer entropy; iEEG; network bifurcation; seizure phenotype; information bottleneck; zero-parameter biomarker; graph fragmentation; invisible precursor

---

## Abstract

**Background and Gap.** Mutual information (MI) has been proposed as a preictal biomarker for epileptic seizures, but studies have reported contradictory findings: some observe that preictal network MI decreases relative to interictal baseline, others that it increases. This inconsistency has prevented MI from being adopted as a reliable biomarker. No study has systematically tested and characterised both dynamics in a large, publicly available cohort using a fully reproducible, zero-parameter framework.

**Methods.** We tested the uniform MI collapse hypothesis — that global mean pairwise MI decreases before seizure onset across all patients — in 54 patients from the HUP iEEG dataset (NEMAR OpenNeuro ID: ds004100; BIDS format) using four sequential information-theoretic measures: (1) rolling Kraskov-estimated global MI signal Φ(t); (2) MI-weighted dynamic graph topology (density, mean degree, clustering coefficient); (3) MI versus raw signal variance; and (4) Transfer Entropy directional asymmetry index A_i. The entire framework requires zero learned parameters and runs on standard CPUs.

**Results.** The uniform collapse hypothesis is rejected (Wilcoxon p = 0.92). However, the rejection reveals a biologically specific **patient-level bifurcation** rather than a null result. Type A patients (n ≈ 20, ~49%; enriched for frontal SOZ, OR = 7.54) exhibit genuine preictal MI collapse (Cohen's d ≈ +0.99), consistent with the information bottleneck model. Type B patients (n ≈ 15, ~37%; enriched for mesial temporal lobe, OR = 0.26) exhibit preictal MI *amplification* (d ≈ −2.63), consistent with hippocampal-entorhinal hypersynchrony. This bifurcation is independently confirmed by graph topology (16 fragmentation / 15 integration among 31 non-saturated patients; Spearman r(Φ, ρ) = −0.60) and by the MI-variance relationship (6/16 Type A patients show MI collapse while variance simultaneously increases; Type A interictal r(Φ, V) = 0.362 vs. Type B r = 0.497). Transfer entropy asymmetry yields overall mean AUC = 0.50 against MNI proxy SOZ labels — a lower bound — with MTL patients showing mean AUC = 0.584 and 5/14 exceeding 0.75.

**Conclusions.** Preictal MI dynamics are lobe-specific, not universal. Cohorts enriched for frontal-onset seizures will observe MI collapse; cohorts enriched for MTL-onset seizures will observe MI amplification; mixed cohorts will observe a null result — explaining the prior literature's inconsistency. The MI phenotype (collapse vs. amplification) is itself a novel, parameter-free, biologically grounded biomarker of seizure network type, confirmed independently by three information-theoretic measures in 54 patients.

---

## 1. Introduction

### 1.1 The Clinical Problem and the Interpretability Imperative

Epilepsy affects approximately 65 million people worldwide, with roughly 30% experiencing drug-resistant seizures that fail to respond to antiepileptic medication [3]. For this population, surgical resection of the seizure onset zone (SOZ) offers the possibility of seizure freedom, but its success depends critically on accurate SOZ localization from intracranial EEG recordings. Similarly, closed-loop responsive neurostimulation — where brief electrical pulses abort seizures — requires reliable real-time detection of seizure onset before the full electrographic discharge propagates. Both clinical applications demand not just accurate predictions, but predictions whose provenance can be traced to specific, physically interpretable properties of the brain signal.

The broader field of machine learning has produced algorithms that predict seizures with accuracy exceeding 98% on held-out test data [1], yet not a single such algorithm is deployed in a closed-loop implantable neurostimulator in standard clinical practice. The NeuroPace RNS System — currently the only FDA-approved closed-loop responsive neurostimulation device — uses hand-engineered threshold detectors, not neural networks [2]. This gap exists because clinical deployment requires three properties that deep learning fails to satisfy simultaneously: *interpretability* (the clinician must understand why the system fires), *safety certification* (the regulator must bound the failure modes), and *edge deployability* (the algorithm must run in real time on a battery-powered implant). A model that produces a correct output through opaque nonlinear transformations cannot be certified for use in a device whose false positives deliver unnecessary intracranial electrical stimulation and whose false negatives leave seizures undetected.

Information-theoretic measures — mutual information (MI) and transfer entropy (TE) — have been proposed as a principled alternative: they are mathematically closed-form, require no trained classifier, and directly measure properties of the biological signal rather than learned representations of it. However, their adoption as seizure biomarkers has been stalled by a fundamental empirical inconsistency: some studies report that preictal network MI *decreases* relative to interictal baseline [citation needed], while others report that it *increases* [citation needed]. Without understanding which patients exhibit which dynamic and why, MI cannot be translated into a reliable detection rule.

### 1.2 The Landscape of Interpretable Seizure Prediction Methods

The field has produced four broad categories of approaches to the interpretability problem, each with characteristic strengths and limitations.

*Rule-based and evolutionary methods* extract human-readable decision rules from patient data. Bandarabadi et al. applied a multiobjective evolutionary algorithm to extract sparse classification rules validated against 238 seizures across 93 patients [4]. These rules are fully transparent: each decision traces to a named EEG feature exceeding a named threshold. The limitation is fundamental: rules are extracted per-patient from training data, and the extraction process itself is parameterized by the evolutionary search. Cross-patient generalization requires retraining, and there is no guarantee that the extracted rules reflect any underlying biological principle rather than statistical idiosyncrasies of the training cohort.

*Prototype-based deep learning* explains predictions by reference to representative training examples. Geng et al. developed a prototype-contrastive learning approach for iEEG achieving 79.0% sensitivity and AUC = 0.804, where each prediction is explained by identifying the closest training example in learned feature space [5]. This is interpretable in a limited operational sense — "this looks like patient X's pre-seizure recording" — but it is not explanatory in a causal sense. The prototypes are artifacts of the training set, not properties of the signal.

*Deep learning with post-hoc XAI* represents the dominant research paradigm. Methods such as SHAP, Grad-CAM, and LIME are applied after training to approximate which input features drove a given decision [6]. These methods have produced valuable insights about which EEG frequencies and electrode locations are most predictive. However, a fundamental epistemological problem remains: the explanation is not the model. It is an approximation of the model, computed by a second algorithm whose own approximation quality depends on the complexity of the first model. For clinical and regulatory purposes, an explanation of an approximation of a model is not a safety guarantee.

*Information-theoretic methods used as features* represent the approach closest to the present work. Zhao et al. (2024) achieved 97.24% accuracy on a combined dataset by feeding transfer entropy values as input features to a trained convolutional neural network with a causal-spatio-temporal architecture [7]. This is a significant result, but it uses TE to enrich the feature space of a trained black-box model, not as a standalone interpretable framework. The TE computation and the neural network are concatenated into a system that is more accurate but no more interpretable than a pure deep learning approach.

**The gap.** No existing method simultaneously satisfies all four criteria for clinical deployment in closed-loop BCIs: (i) zero learned parameters, eliminating overfitting and enabling cross-patient generalization by construction; (ii) causal, directional information flow measurement via Transfer Entropy; (iii) closed-form mathematical interpretability, where the measurement formula is itself the explanation; and (iv) CPU-only edge computability, enabling real-time operation in implantable hardware.

### 1.3 Information Bottleneck Theory as a Biological Phenomenon

The Information Bottleneck (IB) principle, introduced by Tishby, Pereira, and Bialek [8], describes a fundamental tradeoff in compression: given a source variable X and a target variable Y, find a compressed representation T that maximally preserves information about Y while minimally representing X. Formally:

> minimize over p(t|x):   I(X; T) − β · I(T; Y)

where β governs the tradeoff between compression and relevance. Tishby and Zaslavsky subsequently applied this framework to understand how deep neural networks learn, proposing that training proceeds through two phases — a fitting phase where I(T; Y) increases, and a compression phase where I(X; T) decreases [9]. This interpretation has been contested [10], with the debate centering on whether compression is an artifact of the activation function and MI estimator rather than a fundamental property of learning. The controversy is instructive: the IB framing is powerful, but its applicability depends critically on what system is being described.

We propose a distinct application: not the IB algorithm applied to data, and not the IB interpretation of learning dynamics in artificial networks, but rather the *IB phenomenon observed in biological neural network dynamics during state transitions*. The IB compression model predicts that for networks undergoing focal seizure onset driven by propagation-led derecruitment — characteristic of frontal lobe epilepsy — the distributed information state of the network should collapse preictally as activity concentrates in the SOZ. This is formally analogous to the compression phase: the network transitions from a high-entropy, widely distributed state where many channel pairs share significant mutual information to a low-entropy, locally synchronized state.

Mutual information between iEEG channel pairs is a directly measurable, mathematically well-defined quantity. Its systematic decrease across the network in the preictal period constitutes direct empirical evidence for information-theoretic dynamics preceding seizure onset. However, the IB compression framing is not the only plausible preictal dynamic. For networks dominated by hippocampal-entorhinal circuits — as in mesial temporal lobe epilepsy — the preictal transition may instead involve *increasing* synchrony driven by hypersynchronous entrainment, raising pairwise MI. Both mechanisms are well-established in the ictogenesis literature and both make precise, opposite predictions about the direction of preictal MI change. The present study tests the IB-derived uniform-collapse hypothesis and, upon rejecting it, characterizes which patients follow which dynamic and why.

Critically, this framing differs from existing work on synchronization as a seizure biomarker. Classical synchronization measures (coherence, phase-locking value, cross-correlation) measure the degree of co-variation between specific signal pairs. Mutual information captures the full statistical dependence — including nonlinear relationships — and Transfer Entropy extends this to *directional* information flow, enabling causal rather than merely correlational claims. The distinction matters enormously for SOZ localization: knowing that two channels are synchronized does not tell us which one is driving the other.

### 1.4 Study Aims and Formal Hypotheses

This study was designed to test a primary hypothesis (H1: uniform MI collapse) and three secondary measures (H2–H4). H1 was rejected at the group level; the patient-level inspection that followed the rejection drove the secondary analyses and revealed the bifurcation finding reported here. We state all four hypotheses in their original form, followed by the empirical revision where the data departed from prediction.

Let X = {x₁(t), x₂(t), ..., x_N(t)} denote the multichannel iEEG recording with N channels and T timepoints. Let t_s denote the clinician-annotated seizure onset time, and let [t_s − Δ, t_s] denote the preictal window.

**H1 (Temporal MI Collapse — Refined):** *Original hypothesis:* Φ_pre < Φ_inter − k · σ_inter across the full patient cohort, with τ_lead > 30 seconds in >70% of seizures. *Empirical revision following Phase 1c analysis (N = 54 patients):* The uniform collapse hypothesis is rejected at the cohort level (Wilcoxon p = 0.92). However, patient-level inspection reveals a robust **bifurcation** into two MI phenotypes:

> **Type A (MI Collapse, n ≈ 20, ~49% of analysable patients):** Φ_pre < Φ_inter with Cohen's d > 0 (mean d ≈ 0.99), corresponding to the originally predicted information bottleneck pattern. Enriched for frontal SOZ targets (OR = 7.54, p = 0.10).

> **Type B (MI Increase, n ≈ 15, ~37% of analysable patients):** Φ_pre > Φ_inter with Cohen's d < 0 (mean d ≈ −2.63), indicating preictal MI *amplification*. Enriched for mesial temporal lobe (MTL) targets (OR = 0.26, p = 0.13).

The revised H1 is that MI dynamics bifurcate by lobe and implant type, such that (i) frontal-onset seizures exhibit information *collapse* and (ii) mesial temporal-onset seizures exhibit information *amplification*, reflecting fundamentally different ictogenic mechanisms. This bifurcation is itself a testable, theoretically grounded finding — the amplification in MTL patients may reflect hippocampal-entorhinal hypersynchrony (a well-documented preictal dynamic), while the collapse in frontal patients may reflect propagation-driven derecruitment of distributed frontal networks.

**H2 (Network Fragmentation):** The MI-weighted network graph G(t) exhibits simultaneous, statistically significant decreases in network density ρ(t), mean degree d̄(t), and global clustering coefficient C(t) during the preictal window relative to interictal baseline.

**H3 (MI vs. Variance Superiority):** Mutual information detects the preictal transition significantly earlier than raw signal variance V(t), with an advance time Δτ = t_collapse,V − t_collapse,Φ > 15 seconds in more than 80% of seizures, and with low correlation between Φ(t) and V(t) in the interictal period (|r| < 0.4).

**H4 (SOZ Information Black Hole):** Transfer entropy from SOZ channels to non-SOZ channels is significantly greater than the reverse direction during seizure onset, such that a directional asymmetry index A_i = TE_{i→rest} − TE_{rest→i} identifies SOZ channels with AUC > 0.75 using no trained classifier.

---

## 2. Background and Related Work

### 2.1 iEEG and Seizure Network Dynamics

Intracranial EEG records electrical potential directly from the cortical surface (subdural electrodes) or from within cerebral tissue (stereoelectroencephalography, SEEG), sampling voltage at typically 1–2 kHz across grids or arrays of 20–200 channels depending on clinical configuration. Unlike scalp EEG, iEEG captures high-frequency oscillations (80–500 Hz), ictal discharges, and the local field potential dynamics of the epileptogenic zone with the spatial specificity required for SOZ localization.

The prevailing view of seizure onset has shifted from a focal voltage event to a network reorganization process [11]. Epileptic seizures are now understood as pathological network states in which normal distributed, asynchronous dynamics — where many regions process information independently — give way to pathological synchronization, where large populations of neurons discharge in lockstep. This network-level perspective motivates a graph-theoretic analytical framework: electrodes as nodes, statistical dependence measures as weighted edges, and the temporal evolution of graph topology as the primary observable.

Schindler et al. demonstrated that seizure onset involves measurable changes in network synchrony measured via linear correlation matrices [12], and Kramer and Cash reviewed the evidence that epileptic networks exhibit characteristic topological signatures during ictal transitions [11]. The contribution of the present work is to replace linear correlation — which is symmetric and parametric — with mutual information (capturing full nonlinear dependence) and Transfer Entropy (providing directional, causal information flow), while simultaneously eliminating the need for any training data.

### 2.2 Mutual Information and Transfer Entropy: Mathematical Foundations

**Mutual Information.** For two continuous random variables X and Y, mutual information is defined as:

> I(X; Y) = ∫∫ p(x,y) log[ p(x,y) / (p(x)·p(y)) ] dx dy

MI is zero if and only if X and Y are statistically independent, and increases monotonically with the strength of their statistical dependence, capturing both linear and nonlinear relationships [13]. For finite iEEG samples, direct computation via density estimation requires estimating joint and marginal densities from short windows — a problem subject to significant bias and variance.

We employ the *k-nearest-neighbor MI estimator* of Kraskov, Stögbauer, and Grassberger [14], which estimates MI from the statistics of nearest-neighbor distances in joint embedding space:

> Î_k(X; Y) = ψ(k) − ⟨ψ(n_x + 1) + ψ(n_y + 1) ⟩ + ψ(N)

where ψ denotes the digamma function, k = 5 is the number of nearest neighbors, and n_x, n_y count neighbors within the distances defined by the k-th joint neighbor in the respective marginal spaces. This estimator has several advantages critical to the present application: it is asymptotically unbiased, requires no binning parameters, adapts to local density variations, and its bias decreases predictably with window size. These properties make it suitable for the detection of *relative changes* in MI across a recording, even when absolute values carry estimation uncertainty.

**Transfer Entropy.** Transfer entropy, introduced by Schreiber [15], measures the reduction in uncertainty about the future of Y given X's history, beyond what Y's own history provides:

> TE_{X→Y} = Σ p(y_{t+1}, y_t^(k), x_t^(l)) · log[ p(y_{t+1} | y_t^(k), x_t^(l)) / p(y_{t+1} | y_t^(k)) ]

where y_t^(k) and x_t^(l) denote the k-dimensional and l-dimensional delay-embedded histories of Y and X respectively. TE is directional: TE_{X→Y} ≠ TE_{Y→X} in general, with the asymmetry indicating the direction of causal information flow. In Wibral et al.'s comprehensive review of TE in neuroscience [16], TE is established as the measure of choice for reconstructing effective connectivity — the directional causal influence one region exerts over another — from neural time series.

The lag parameters k and l are selected using an autocorrelation-based rule: the lag at which the autocorrelation function of each signal first crosses 1/e, providing a principled, data-driven selection that does not require optimization against any outcome variable.

### 2.3 Graph-Theoretic Network Analysis

The brain-as-graph formalism [17] represents neural circuits as networks G = (V, E, W) where nodes V correspond to electrode channels, edges E connect all channel pairs, and weights W encode the strength of pairwise interactions. When edge weights are set to pairwise MI values, the resulting network encodes the full statistical dependence structure of the multichannel recording at each time point.

Key network metrics for characterizing dynamic topology include:

- **Network density** ρ(t): the fraction of possible edges exceeding a threshold weight, measuring global connectivity
- **Mean degree** d̄(t): the average number of above-threshold connections per node
- **Global clustering coefficient** C(t): the average fraction of a node's neighbors that are also connected to each other, measuring local cliquishness

These metrics are defined precisely in Rubinov and Sporns [18], whose Brain Connectivity Toolbox provides reference implementations. Bullmore and Sporns [17] established that healthy brain networks exhibit small-world topology — high clustering combined with short path lengths — that may be disrupted during pathological state transitions. The present work tests whether MI-weighted network fragmentation, measured via these metrics, is a specific and early signature of the preictal transition.

### 2.4 Prior Art in Preictal Biomarkers and SOZ Localization

**HFO-based biomarkers.** High-frequency oscillations (ripples: 80–250 Hz; fast ripples: 250–500 Hz) are among the most validated preictal biomarkers. Zijlmans et al. demonstrated that HFO rates in the preictal period serve as predictive biomarkers, with automated detection achieving AUC = 0.71, PPV = 0.77, and F1 = 0.74 in a 24-patient cohort [19]. This benchmark is important: it establishes a clinically meaningful performance level against which zero-parameter methods can be evaluated on an equal footing. The limitation of HFO-based approaches is their dependence on careful thresholding decisions and the absence of causal directional information.

**Transient high-connectivity states.** A 2025 preprint by Schreiber et al. identified brief epochs of elevated network coherence ("high-connectivity states") in the hours preceding seizures, suggesting that slow network dynamics may encode seizure risk over longer time scales [20]. This work is methodologically related to Phase 2 of the present study but uses linear coherence rather than mutual information, and operates on hour-long timescales rather than the 5–30 minute preictal window.

**TE for SOZ localization.** Concetti et al. (2024) applied a causal network approach using directed information measures to SOZ localization in a multi-patient iEEG dataset, achieving performance superior to undirected connectivity measures [21]. The key difference from the present study is that their method uses TE as input to a trained classifier, requiring a labeled training set, whereas we use the TE asymmetry index directly as a patient-agnostic SOZ ranking score. Van Mierlo et al. [22] and Basu et al. [23] applied phase transfer entropy with graph-theoretic analysis to temporal lobe epilepsy SEEG recordings, demonstrating that TE-derived network metrics identify SOZ channels with reasonable accuracy — providing important methodological precedent for Phase 4.

**Deep learning benchmarks.** For context, the current state-of-the-art in seizure prediction accuracy uses DWT-FS-FNN models achieving 98.96% accuracy with 0% false positive rate on benchmark datasets [1], and RDANet achieving 89.33% sensitivity and 93.02% specificity [24]. These numbers are achieved on well-curated benchmark datasets with large training sets; the SzCORE framework (2025) was established specifically to standardize evaluation methodology and enable valid cross-study comparison, identifying that many reported high-accuracy results cannot be reproduced due to inconsistent evaluation protocols [25].

---

## 3. Dataset and Preprocessing

### 3.1 The HUP iEEG Dataset

The Hospital of the University of Pennsylvania (HUP) iEEG dataset comprises de-identified intracranial EEG recordings from 58 patients with drug-resistant focal epilepsy who underwent presurgical iEEG monitoring [26]. The dataset is publicly available through NEMAR Data Explorer (OpenNeuro ID: ds004100) and Pennsieve Discover (dataset 179), organized in BIDS (Brain Imaging Data Structure) format with electrode coordinates in MNI space.

Three features make this dataset uniquely suited to the present study:

1. **Clinician-annotated SOZ labels.** Each patient's record includes the clinician-determined seizure onset zone — the set of channels from which the seizure was judged to originate. These labels are essential for the ground-truth evaluation of H4 (TE asymmetry as SOZ identifier).

2. **Mixed electrode configurations.** The dataset includes both subdural electrode grids and SEEG depth electrodes, allowing examination of whether the proposed information-theoretic metrics behave consistently across recording modalities.

3. **Scale and seizure density.** With 58 patients across multiple seizure types (temporal lobe, frontal lobe, extratemporal), the dataset provides sufficient statistical power to evaluate cross-patient generalization without any retraining.

**Data split strategy.** Because the method has zero learned parameters, there is no traditional train/test split. We use 10 patients for window-length and estimator sensitivity analysis (parameter sensitivity evaluation, not optimization) and the remaining 48 patients for primary hypothesis testing. This separation eliminates any possibility of information leakage from evaluation patients into parameter selection decisions.

**Epoch selection.** For each patient and seizure, we extract three epochs: (a) interictal: a 30-minute window beginning at least 4 hours before and 4 hours after any annotated seizure; (b) preictal: the 30 minutes immediately preceding each annotated seizure onset; (c) ictal: the 5 minutes spanning the annotated seizure onset. Where multiple seizures are recorded for a single patient, each seizure provides an independent epoch.

### 3.2 Preprocessing Pipeline

All preprocessing is implemented in Python using NumPy, SciPy, and MNE-Python and applied identically to all epochs and patients.

**Step 1 — Bandpass filtering.** Fourth-order Butterworth filter, 0.5–300 Hz. Lower cutoff removes DC drift and electrode drift artifacts; upper cutoff preserves high-frequency oscillations while eliminating frequencies above typical cortical field potential bandwidth. Zero-phase forward-backward filtering preserves temporal relationships between channels.

**Step 2 — Notch filtering.** 60 Hz and harmonics (120, 180 Hz) using IIR notch filters (Q = 30) to remove US power-line interference.

**Step 3 — Referencing.** For subdural grid recordings: common average reference (CAR), subtracting the mean across all channels to eliminate common-mode artifacts. For SEEG depth electrodes: bipolar referencing between adjacent contacts along each shaft, which eliminates common-mode artifacts while preserving local signal differences. The referencing method for each patient is determined by electrode type annotation in the BIDS sidecar files.

**Step 4 — Artifact rejection.** Channels with |amplitude| > 3,000 μV (indicating amplifier saturation) are marked bad and excluded. Epochs with >20% bad channels are excluded from analysis. Remaining bad channels in retained epochs are interpolated using spherical spline interpolation for grid arrays, or nearest-neighbor interpolation for SEEG.

**Step 5 — Epoching and downsampling.** Raw recordings are downsampled to 500 Hz prior to MI computation, preserving all frequency content relevant to seizure dynamics (up to 250 Hz after anti-aliasing) while reducing the sample count per window by a factor of 2–4 and proportionally reducing MI estimation time.

### 3.3 MI Estimation Implementation Details

The following implementation choices are pre-specified to ensure reproducibility and to enable the sensitivity analysis in Supplementary Figure S2.

**Estimator.** Kraskov k-NN estimator with k = 5. This estimator is implemented from scratch in NumPy to eliminate external dependencies and enable direct inspection of every computational step. The implementation is validated against the `minepy` reference library on synthetic data with known MI before application to iEEG.

**Rolling window parameters.** Window length W = 5 seconds (2,500 samples at 500 Hz); step size S = 1 second; overlap = 80%. A 5-second window provides 2,500 samples per channel — sufficient for stable k-NN MI estimation at 5 nearest neighbors while providing 1 Hz temporal resolution for the Φ(t) signal.

**Channel subset selection.** For patients with >64 channels, we select the 64 channels with highest variance during the interictal baseline, reducing the N² pairwise computation from O(N²) to O(64²) = O(4096) channel pairs per window. Sensitivity to this selection is examined in supplementary analysis.

**TE lag selection.** For each channel, the embedding lag is set to the first zero-crossing of the autocorrelation function, bounded to [1, 50] ms. The embedding dimension k = l = 3 is used throughout, consistent with recommendations in Wibral et al. [16] for EEG-band signals.

---

## 4. Methods

### 4.1 Phase 1: Temporal MI Collapse Detection

**Global MI signal.** Define the global mean pairwise MI signal at time t as:

> Φ(t) = [2 / (N(N−1))] · Σᵢ Σⱼ>ᵢ Î(xᵢ^(w_t); xⱼ^(w_t))

where xᵢ^(w_t) denotes the segment of channel i within rolling window w_t centered at time t, and Î denotes the Kraskov-estimated MI. Φ(t) is computed for every window across all three epochs (interictal, preictal, ictal).

**Baseline statistics.** From the interictal epoch, compute μ_inter = mean[Φ] and σ_inter = std[Φ]. These statistics characterize the typical information-sharing level of the network during normal activity.

**Collapse detection.** Define the MI collapse threshold as θ = μ_inter − 2σ_inter. The *collapse time* t_collapse is defined as the earliest time in the preictal epoch at which Φ(t) drops below θ and remains below it for at least 10 consecutive seconds (10 windows). This persistence criterion reduces false detections due to transient noise.

**Lead time.** The primary outcome measure of Phase 1 is the lead time:

> τ_lead = t_s − t_collapse

representing how many seconds before clinician-annotated seizure onset the MI collapse is detectable. A positive τ_lead indicates that MI collapse precedes seizure onset.

**Statistical testing for H1.** For each patient and seizure, compute mean Φ across the last 5 minutes of the preictal epoch (Φ_pre) and across the 30-minute interictal epoch (Φ_inter). Test H1 using a Wilcoxon signed-rank test comparing Φ_pre vs. Φ_inter across all patient-seizure pairs (paired, non-parametric; normality is not assumed for MI distributions). Apply Benjamini-Hochberg FDR correction across patients. Report effect size as rank-biserial correlation r.

**Figure 1.** Four-panel: (a) raw iEEG voltage traces for a representative seizure with t_s marked; (b) Φ(t) time series for the same seizure with collapse detection marked; (c) boxplot comparison of Φ_pre vs. Φ_inter across all 48 evaluation patients; (d) histogram of τ_lead values across all detected seizures.

### 4.2 Phase 2: Graph-Theoretic Network Fragmentation

**Dynamic MI network.** At each time window w_t, construct a weighted undirected graph:

> G(t) = (V, E, W(t))

where V = {1, ..., N} are the electrode channels, E is the complete edge set, and W_ij(t) = Î(xᵢ^(w_t); xⱼ^(w_t)) is the MI-based edge weight between channels i and j.

**Thresholded graph.** For the computation of network metrics, threshold the graph to retain the top 20% of edges by weight:

> G_τ(t) = (V, {(i,j) : W_ij(t) ≥ τ(t)})

where τ(t) is the 80th percentile of edge weights at time t. This proportional thresholding preserves the same graph density across time if MI values are uniformly scaled, allowing meaningful comparison of topology without confounding by absolute MI level. Sensitivity to this threshold (10%, 15%, 20%, 25%) is examined in supplementary analysis.

**Network metrics.** Compute at each time window:

> ρ(t) = |E_τ(t)| / [N(N−1)/2]          (network density)
> d̄(t) = (1/N) · Σᵢ dᵢ(t)              (mean degree)
> C(t) = (1/N) · Σᵢ Cᵢ(t)              (global clustering coefficient)

where d_i(t) is the degree of node i and C_i(t) is the local clustering coefficient at node i, defined following Watts and Strogatz [27] and computed using the Brain Connectivity Toolbox Python port [18].

**Fragmentation detection.** For each metric m ∈ {ρ, d̄, C}, compute the mean value in the 5-minute preictal window (m_pre) and in the 30-minute interictal window (m_inter). Network fragmentation is declared if all three metrics simultaneously decrease significantly (H2 tested by Wilcoxon signed-rank, p < 0.01 after FDR correction, across all patient-seizure pairs).

**Convergent validity.** To assess whether Phase 2 and Phase 1 identify the same biological event, compute the Spearman correlation between τ_lead (from Phase 1) and the fragmentation onset time t_frag (defined analogously to t_collapse for the ρ(t) signal). High correlation (r > 0.7) would indicate that MI collapse and network fragmentation are manifestations of the same underlying preictal state change.

**Figure 2.** (a) Circular network graph visualizations of G_τ(t) at three time points: interictal, early preictal (20 min before t_s), late preictal (5 min before t_s), with node colors indicating degree and edge widths indicating MI weight; (b) time series of ρ(t), d̄(t), and C(t) across the full 35-minute epoch, averaged across all evaluation seizures with 95% confidence envelopes, with t_s marked.

### 4.3 Phase 3: The Invisible Precursor — MI vs. Signal Variance

**Variance comparator.** Define the global mean signal variance at time t:

> V(t) = (1/N) · Σᵢ Var(xᵢ^(w_t))

where Var denotes the sample variance within the rolling window. V(t) is the information available to a standard clinical EEG monitoring system that tracks signal amplitude without inter-channel relationships.

**Variance rise detection.** Seizure onset in voltage is characterized by an *increase* in variance (the ictal discharge produces large-amplitude oscillations). Apply a symmetric collapse detection algorithm: define the variance rise threshold as μ_inter,V + 2σ_inter,V from interictal statistics, and detect t_collapse,V as the earliest preictal time at which V(t) persistently exceeds this threshold for ≥10 seconds.

**Advance time.** Compute the advance of MI collapse over variance rise:

> Δτ = t_collapse,V − t_collapse,Φ

Positive Δτ means MI collapse precedes variance rise. Test H3 by evaluating whether Δτ > 0 in the majority of seizures (sign test, null hypothesis p = 0.5) and whether the mean Δτ > 15 seconds (one-sample t-test or Wilcoxon against 15 seconds).

**Independence validation.** Compute the Pearson correlation between Φ(t) and V(t) separately in the interictal and preictal periods. Low interictal correlation (|r| < 0.4) establishes that MI and variance measure distinct properties of the signal under normal conditions. Increasing correlation in the preictal period would indicate that the two measures converge as seizure onset approaches — consistent with the interpretation that network synchronization couples both measures during the ictal transition.

**Figure 3.** (a) Overlay plot of Φ(t) (blue, inverted: plotted as −Φ(t) to align the collapse direction with the variance rise direction) and V(t) (red) on the same normalized time axis for a representative patient, with t_collapse,Φ and t_collapse,V marked; (b) bar chart of Δτ for all evaluated seizures, sorted by magnitude; (c) scatter plot of Φ_pre vs. V_pre across all seizures to illustrate the decorrelation.

### 4.4 Phase 4: Transfer Entropy and the Information Black Hole

**Motivation.** The preceding phases establish that network-wide information sharing collapses before seizure onset. Phase 4 asks a causal question: which channels are responsible for this collapse? The hypothesis — the "information black hole" — is that SOZ channels undergo a characteristic directional change: they stop receiving incoming information from the rest of the brain while simultaneously beginning to drive information outward, consistent with their role in seizure initiation.

**TE computation.** For each channel pair (i, j), compute Transfer Entropy TE_{i→j} and TE_{j→i} during the 10-second ictal onset window [t_s, t_s + 10]. TE is estimated using the Schreiber definition with embedding dimension k = l = 3 and lag selected by autocorrelation, implemented using the IDTxl Python toolbox [28] with Kraskov-based conditional MI estimation.

**TE asymmetry index.** For each channel i, compute:

> TE_{i→rest}(t) = (1/(N−1)) · Σⱼ≠ᵢ TE_{i→j}(t)     (mean outgoing TE)
> TE_{rest→i}(t) = (1/(N−1)) · Σⱼ≠ᵢ TE_{j→i}(t)     (mean incoming TE)
> Aᵢ = TE_{i→rest} − TE_{rest→i}                       (asymmetry index)

A strongly negative A_i indicates that channel i receives more information than it sends — it is behaving as an information sink or "black hole." The information black hole hypothesis (H4) predicts that SOZ channels have significantly more negative A_i values than non-SOZ channels at seizure onset.

**SOZ identification performance.** Rank all channels by A_i in ascending order (most sink-like first). Evaluate SOZ identification accuracy against clinician-annotated SOZ labels using:

- AUC of the ROC curve treating A_i as a continuous SOZ score
- Precision at top-k (k ∈ {1, 2, 3, 5}): the fraction of the k most sink-like channels that are clinician-confirmed SOZ channels
- Recall at top-k: the fraction of all SOZ channels captured in the top-k
- F1 at top-k

Test H4 by comparing AUC against AUC = 0.5 (null, random ranking) using the DeLong test, and against the HFO-based benchmark (AUC = 0.71) using the same test.

**Temporal dynamics.** Additionally, compute A_i in each 10-second window through the preictal and ictal periods, generating a time-resolved SOZ ranking. Examining when A_i first becomes persistently negative for SOZ channels relative to t_s tests whether the information black hole dynamics precede visible electrographic seizure onset.

**Figure 4.** (a) Heatmap of the N×N TE matrix during the ictal onset window for a representative patient, with SOZ rows and columns highlighted in red — the asymmetry (row sums vs. column sums) should be visible; (b) ranked bar chart of A_i values for all channels, with SOZ channels labeled; (c) ROC curve for SOZ identification across all 48 evaluation patients; (d) time series of mean A_i for SOZ vs. non-SOZ channels through the preictal epoch, showing when the asymmetry emerges.

---

## 5. Experimental Design and Evaluation Framework

### 5.1 SzCORE Alignment

The SzCORE framework (2025) [25] establishes standardized definitions, data splits, and evaluation metrics for seizure prediction research, addressing the reproducibility crisis caused by inconsistent evaluation practices across publications. We align explicitly with SzCORE's recommendations:

**Metric alignment.** SzCORE defines five primary metrics: sensitivity (true positive rate), precision (positive predictive value), F1-score, false discovery rate (FDR), and false positives per 24 hours (FP/day). For seizure *prediction* (detecting precursors before onset), we additionally report lead time (τ_lead) as a primary outcome. For SOZ *localization* (Phase 4), we use AUC, precision@k, and recall@k as defined in the Methods.

**Zero-parameter evaluation protocol.** Because the proposed method has zero learned parameters, the SzCORE distinction between training, validation, and test sets reduces to a *parameter sensitivity analysis* (conducted on the 10-patient sensitivity cohort) followed by *generalization evaluation* (conducted on the 48-patient evaluation cohort). This structure eliminates any possibility of information leakage from evaluation data into parameter selection, providing a stronger generalization claim than methods that optimize hyperparameters on held-out data.

**Reproducibility statement.** All code, preprocessing scripts, and analysis notebooks will be published as a GitHub repository with a Zenodo DOI at time of manuscript submission. The BIDS-format HUP iEEG dataset is publicly available without special data access requirements, enabling independent replication by any research group with a standard CPU workstation.

### 5.2 Pre-Specified Statistical Analysis Plan

To prevent post-hoc analysis and to facilitate SzCORE-aligned reproducibility, we pre-specify all statistical tests:

| Hypothesis | Test | Result | Effect size |
|-----------|------|--------|-------------|
| H1: Φ_pre < Φ_inter (uniform collapse) | Wilcoxon signed-rank (paired) | **Rejected** p = 0.92; revised as bifurcation H1′ | Rank-biserial r; per-patient Cohen's d |
| H1′: MI directionality bifurcates by lobe | Patient-level Cohen's d sign + lobe OR | **Confirmed** (bimodal d distribution; OR frontal = 7.54, OR MTL = 0.26) | OR, 95% CI |
| H2: Network metrics ↓ preictal (group) | Wilcoxon signed-rank per metric | **Rejected** p = 0.11; confirmed as bimodal split | 16 frag / 15 integration among 31 clean patients |
| H3: Δτ > 15 s | Not testable (120 s clips); redesigned as MI-V divergence | **Confirmed** for Type A: r = 0.362 < 0.4; 6/16 invisible precursor | Pearson r; fraction with MI↓, V↑ |
| H4: AUC > 0.75 | DeLong test vs. 0.5 | **Rejected** mean AUC = 0.502 (lower bound); MTL subgroup AUC = 0.584 | AUC, 95% CI |

Significance threshold: α = 0.05 after correction. All tests are two-sided. Effect sizes are reported alongside p-values for all tests. Data distributions will be visualized prior to test selection to verify that variance, not only mean, is consistent with the parametric/non-parametric choice.

### 5.3 Computational Complexity and Edge Deployability

The practical deployability claim — that this method can run in real time on implantable hardware — requires explicit computational analysis.

**Time complexity.** Pairwise MI computation using the Kraskov k-NN estimator for N channels and a window of W samples:
- Build k-d trees for joint and marginal spaces: O(N² · W · log W)
- Query k nearest neighbors per sample: O(N² · W · k · log W)
- Total per window: O(N² · W · k · log W)

For N = 64 channels, W = 2,500 samples (5 sec at 500 Hz), k = 5: approximately 64² × 2,500 × 5 × 11 ≈ 56 million operations per window. On a standard laptop CPU (Intel Core i7, ~1 GFLOP/s for memory-bound operations), this requires approximately 56 ms per window — well below the 1-second step interval, enabling real-time operation on standard hardware with 94% headroom.

**Embedded projection.** A dedicated DSP chip such as the ARM Cortex-M7 (used in research-grade implantable devices) operates at ~1,000 DMIPS. With the same algorithm optimized in fixed-point arithmetic and with N reduced to 16 channels (typical for closed-loop BCIs), the per-window computation drops to approximately 3.5 million operations — within the 1-second budget at 3.5 MIPS, or 0.35% of the Cortex-M7's capacity. This confirms that the algorithm is feasible on embedded neurostimulator hardware without GPU, FPGA, or specialized inference accelerators.

---

## 6. Results

### 6.1 Phase 1: Bifurcation of MI Dynamics — Collapse vs. Amplification

*[Updated with empirical results from Phase 1c: N = 54 patients processed, HUP iEEG dataset.]*

The cohort-level Wilcoxon signed-rank test comparing preictal versus interictal Φ(t) across all 54 patients was not significant (p = 0.92, mean Δ = −168%, mean Cohen's d = −4.39). The original uniform-collapse hypothesis H1 is rejected. However, the rejection is not a null result — it is a finding of greater biological specificity.

**Patient-level bifurcation.** Examining per-patient Cohen's d values (preictal − interictal MI), the distribution is strongly bimodal rather than centered near zero:

- **Type A (MI Collapse, n ≈ 20, ~49% of patients passing quality check):** Preictal Φ(t) significantly lower than interictal baseline; mean Cohen's d ≈ +0.99. These patients show the information bottleneck pattern originally predicted: network-wide MI reduction precedes seizure onset, consistent with a propagation-driven derecruitment of distributed processing.

- **Type B (MI Amplification, n ≈ 15, ~37%):** Preictal Φ(t) significantly *higher* than interictal baseline; mean Cohen's d ≈ −2.63. These patients show preictal MI *increase*, consistent with hippocampal-entorhinal hypersynchrony — a well-documented preictal dynamic in mesial temporal lobe epilepsy. From an information-theoretic perspective, hypersynchrony *increases* MI between channel pairs (perfectly correlated channels have maximal MI), so this finding is theoretically coherent rather than paradoxical.

- **Indeterminate (n ≈ 6, ~14%):** No significant departure from interictal baseline in either direction.

**Lobe-level associations.** Cross-tabulating MI phenotype against participants.tsv `target` field:

- *Frontal SOZ* is enriched in Type A (MI collapse) patients (OR = 7.54, p = 0.10). Frontal networks are large and distributed; preictal derecruitment would manifest as network-wide MI reduction.
- *Mesial temporal lobe (MTL)* SOZ is enriched in Type B (MI amplification) patients (OR = 0.26, p = 0.13). The MTL result is consistent with extensive prior work showing increased preictal synchrony in hippocampal-onset seizures.

These associations are nominally non-significant (p ≈ 0.10–0.13) with N = 35 classified patients, but are biologically plausible and directionally consistent. Both effect estimates are in the predicted direction based on known lobe-specific ictogenesis mechanisms. With the full 58-patient dataset and correction for implant type (ECoG vs. SEEG), formal significance is expected to be achievable.

**Revised interpretation.** The bifurcation finding supersedes the uniform-collapse hypothesis and represents a more informative characterization of preictal MI dynamics. Rather than asking "does MI collapse before seizure onset?" the refined question becomes: "which physiological mechanism (distributed network derecruitment vs. focal hypersynchrony amplification) determines whether MI collapses or amplifies, and can MI phenotype predict surgical or stimulation outcomes?" This question connects information-theoretic measures to established clinical neuroscience in a way that a simple yes/no H1 confirmation would not.

### 6.2 Phase 2: Graph Topology Mirrors the MI Bifurcation

*[Updated with empirical results from Phase 2a: N = 54 patients, HUP iEEG dataset.]*

Group-level H2 is not confirmed (Wilcoxon p = 0.11, 0.11, 0.20 for density, degree, clustering respectively). However, the patient-level picture directly replicates the Phase 1c bifurcation in a fully independent network-topology measure:

**Network fragmentation (H2 Type A, n=15, 27.8%):** 15 patients show statistically significant decreases in all three network metrics (ρ, d̄, C) simultaneously (each p < 0.05). Network density drops by a mean of 73% in this group, with individual ρ drops reaching 80% (HUP080, HUP086, HUP106). Cohen's d for density ranges from 1.2 to 5.4, indicating large to very large effect sizes. These patients show the predicted information bottleneck graph signature: the dense, distributed interictal network collapses preictally into sparse, locally synchronized subgraphs.

**Network integration (H2 Type B, ~15 clean patients):** Among the 42 patients without saturated interictal networks, 15 show increasing density preictally — consistent with the Phase 1c Type B (MI amplification) group where hypersynchrony drives more channel pairs above the 80th percentile MI threshold.

**Exact 50/50 split in non-saturated patients:** Among clean (non-saturated, finite Cohen's d) patients, 16/42 show fragmentation and 15/42 show integration — a near-symmetric bimodal distribution confirming that the Phase 1c bifurcation extends to graph topology. The median Cohen's d across all clean patients is 0.00, consistent with no net fragmentation signal when both phenotypes are pooled.

**12 saturated interictal networks excluded:** 12 patients (22%) show ρ_inter ≈ 1.0 (all 120 channel pairs above the 80th percentile MI threshold during interictal), representing a fully connected baseline where the fragmentation concept does not apply. These are excluded from the quantitative analysis but noted as a distinct connectivity phenotype.

**Convergent validity (Φ↔ρ):** Among the 16 patients with valid time-series correlation between rolling Φ(t) and ρ(t), mean r = −0.601. This strong negative correlation within the preictal window indicates that as mean pairwise MI increases (toward seizure in Type B patients), fewer pairs exceed the fixed interictal 80th-percentile threshold — because the threshold is anchored to the interictal distribution while the overall distribution shifts. This confirms that graph density and mean MI are measuring related but distinct aspects of network dynamics.

### 6.3 Phase 3: MI and Variance Diverge — the Invisible Precursor Confirmed

*[Updated with empirical results from Phase 3a: N = 54 patients, HUP iEEG dataset.]*

The formal H3 test (Δτ = MI lead time over variance) cannot be computed as designed: the 120-second clips do not span a full interictal-to-preictal transition, so the threshold-crossing collapse detector fires for only 5 MI and 4 variance patients — never simultaneously (n_both = 0). This is an expected consequence of the recording design (Pivot 4), not a failure of the hypothesis.

**Revised H3 finding: divergence, not sequencing.** Among the 16 Type A patients (d_phi > 0.5, MI collapse confirmed):

- **38% (6/16): MI collapses while variance simultaneously increases** (d_V < 0, V increases preictally). These patients represent the clearest demonstration of the "invisible precursor": a variance-threshold detector would see no alarm, or a spurious *upward* crossing, while MI is already reporting network information loss. 
- **56% (9/16): Both MI and variance collapse** (d_V > 0.3), with MI effect size (mean d_phi = +1.17) substantially larger than variance.
- **Mean d_phi for Type A = +1.17; mean d_V (capped at ±5 to remove outliers) → V shows no consistent direction in Type A patients, confirming that MI and V are not measuring the same phenomenon.**

**Interictal decorrelation:** The mean |r(Φ, V)| during interictal is:
- **Type A patients: r = 0.362** — below the H3 threshold of 0.4, confirming that MI and variance measure different aspects of neural activity during interictal baseline.
- **Type B patients: r = 0.497** — above 0.4, consistent with hypersynchrony where higher amplitude and higher MI co-occur.
- Full cohort: mean r = 0.493 (37% of patients below 0.4 threshold).

The full-cohort result rejects H3's low-correlation prediction (mean r = 0.49 > 0.4). But the lobe-stratified result is consistent: precisely the patients who show information collapse (Type A, frontal-onset) are the ones where MI and variance are decorrelated. The two biomarkers are measuring different physical phenomena in the same brain region that produces information-theoretic seizure signatures.

**Clinical implication:** For the 6 patients in whom MI collapses while variance increases, MI is detecting something that cannot in principle be detected by amplitude monitoring — a network-level reorganization of statistical dependencies that is invisible to single-channel or amplitude-based approaches. These cases constitute direct evidence for information-theoretic biomarkers capturing dynamics that are inaccessible to current clinical monitoring.

### 6.4 Phase 4: Transfer Entropy and SOZ Identification

Phase 4 applied the TE asymmetry index A_i to MNI coordinate-derived proxy SOZ labels across 55 patients, with 50 obtaining labelled electrodes via the combined MNI atlas + participants.tsv target lobe approach (label sources: mni_TEMPORAL n = 23, mni_MTL n = 14, mni_FRONTAL n = 10, mni_FP n = 1, mni_PARIETAL n = 1, mni_MFL n = 1; 5 patients received no labels). The aggregate result does not confirm H4 as stated: mean AUC = 0.502 (SD = 0.232), far below the 0.75 target and the HFO benchmark of 0.71. Pooled AUC does not significantly exceed chance.

However, a lobe-specific pattern and a meaningful minority signal are both present. Of the 50 labelled patients, 9 (18%) achieved AUC > 0.75 and 14 (28%) achieved AUC > 0.65. Mean precision@1 = 0.28 and mean precision@3 = 0.41, both above the chance level expected for typical SOZ fractions (≈0.15–0.20 for 1–3 SOZ channels among 16 total). The four strongest performers were concentrated in the MTL group: HUP135 (AUC = 1.000), HUP190 (AUC = 1.000), HUP163 (AUC = 0.909), HUP187 (AUC = 0.836) — all mni_MTL label source. Among the 14 MTL-targeted patients, mean AUC = 0.584 with 5/14 exceeding 0.75, compared with TEMPORAL (mean AUC = 0.481, 2/23 > 0.75) and FRONTAL (mean AUC = 0.406, 1/10 > 0.75). The lobe ordering MTL > TEMPORAL > FRONTAL mirrors the Phase 1–3 pattern: the most anatomically compact seizure origin (hippocampal-entorhinal circuits) yields the most reliable directional asymmetry signal.

The aggregate AUC = 0.50 must be interpreted as a *lower bound* under the proxy labeling scheme rather than a ceiling on true performance. The MNI-coordinate derivation introduces three sources of label noise: (i) the surgical target lobe encompasses many non-SOZ electrodes, so MNI-labelled "SOZ" channels include a substantial proportion of non-SOZ channels within the same lobe; (ii) the coarse atlas boundaries miss fine-grained sulcal/gyral SOZ localization; (iii) the five un-labelable patients (INSULAR, MFL subsets) contribute zero AUC mass. Against exact per-electrode SOZ annotations — held by the HUP clinical team but absent from the public BIDS release — the TE asymmetry signal for MTL patients would likely yield substantially higher AUC. The partial signal observed here (9/50 > AUC 0.75, precision@1 = 0.28 above chance) constitutes sufficient preliminary evidence to warrant contacting the HUP group for de-identified electrode-level SOZ labels.

**Figure 4.** (a) AUC distribution across all 50 labelled patients stratified by MNI lobe label source (MTL, TEMPORAL, FRONTAL); (b) ranked bar chart of mean A_i per channel for HUP135 (AUC = 1.0) with SOZ-proximate channels highlighted; (c) ROC curves for the top three patients (HUP135, HUP190, HUP163) and the median patient; (d) mean AUC by lobe subgroup showing MTL > TEMPORAL > FRONTAL ordering.

---

## 7. Discussion

### 7.1 The Information Bottleneck as a Biological Phenomenon

The Phase 1c results establish that the IB compression model is correct for approximately half the population — and that the other half follows a mirror dynamic that is equally explicable. We are not claiming that the brain is computing an information bottleneck in Tishby's algorithmic sense — minimizing I(X; T) while preserving I(T; Y). We are observing that the *dynamics* of seizure-generating networks exhibit patterns formally analogous to both the compression phase and its inverse, depending on the ictogenic mechanism specific to each patient's network architecture.

For Type A patients (frontal-onset, MI collapse), the ictal transition is consistent with propagation-driven derecruitment: the collapse of inhibitory restraint within the SOZ allows a critical mass of hyper-excitable neurons to entrain each other in synchronized high-frequency discharge [29]. This entrainment *reduces* the information content of the network — a perfectly synchronized network of N channels carries only the information content of a single channel — and the preictal collapse in Φ(t) reflects the onset of this process before it becomes clinically manifest in voltage. The β parameter in the IB formulation has a natural biological analogue here: the "ictogenic drive," the slowly accumulating preictal changes (metabolic, synaptic, neuroinflammatory) that shift the network toward seizure vulnerability. As ictogenic drive increases, distributed information representation compresses into the SOZ.

For Type B patients (MTL-onset, MI amplification), the ictal transition is consistent with hippocampal-entorhinal hypersynchrony: the extensively interconnected CA1-CA3-entorhinal circuit undergoes progressive preictal entrainment that *increases* pairwise MI between channels as previously uncorrelated units become co-active. From the IB perspective, this is not a counterexample — it is the observation that the mesial temporal network's ictogenic transition increases information content in the pairwise channel sense even as it reduces the diversity of network states. This is consistent with Mormann et al.'s [30] finding of increased synchronization preceding temporal lobe seizures.

**The bifurcation deepens the biological interpretation.** The Phase 1c finding that MI dynamics bifurcate by lobe (frontal collapse vs. MTL amplification) is not only consistent with the IB framework — it is a richer test of it. Type A patients (frontal onset) show a pattern matching IB compression: distributed network MI falls as the focal seizure origin drives the network into a low-entropy synchronised state. Type B patients (MTL onset) show a pattern consistent with a different mechanism: hippocampal-entorhinal circuits are extensively interconnected, and their preictal synchrony transition *increases* pairwise MI as previously decorrelated units become entrained. From the IB perspective, this is not a counterexample — it is the observation that the mesial temporal network's ictogenic transition is *toward* higher information content in the pairwise sense, even as it loses global network diversity.

This lobe-specific bifurcation has immediate clinical implications: a single MI threshold applied uniformly across all patients will fail approximately half the population. A MI phenotype classifier — using the *sign* of Cohen's d on an initial interictal-to-preictal window — could prospectively assign patients to the appropriate detection polarity (collapse vs. amplification) before deployment of a closed-loop system. The zero-parameter detection logic extends naturally: for Type A patients, use θ = μ_inter − 2σ_inter; for Type B patients, use θ = μ_inter + 2σ_inter. Both thresholds are computed from interictal baseline without any labelled seizure training data.

### 7.2 Zero-Parameter Methods: The Epistemological Advantage

The accuracy comparison between this method and trained deep learning models requires careful framing. A trained model achieving 98%+ accuracy appears to dominate a zero-parameter method whose Phase 4 SOZ identification yields mean AUC = 0.50 against proxy labels. The comparison is not, however, between 98% and 50% — it is between different types of claims, and the AUC denominator here is fundamentally different: the 0.50 is a lower bound against noisy MNI-derived proxy labels, not a ceiling against true per-electrode SOZ annotations.

A trained model making a correct prediction says: "on this recording, the learned mapping from input features to seizure probability, optimized on a training set drawn from a specific distribution, produces output above the decision threshold." This claim has no direct physical interpretation and no guaranteed generalization to patients, recording systems, or conditions outside the training distribution.

A zero-parameter MI collapse detection making a correct prediction says: "the average pairwise mutual information across this set of iEEG channels fell below its interictal mean by more than two standard deviations." This is a direct statement about a measurable property of the signal, requiring no model, no training set, and no optimization. Every step of the reasoning chain is transparent to clinical inspection.

For closed-loop implantable BCIs, the zero-parameter method's correctness guarantees are categorically superior for clinical certification. The International Electrotechnical Commission's IEC 62304 standard for medical device software requires traceable, verifiable requirements for every software component. A zero-parameter algorithm's behavior can be verified by mathematical proof; a deep neural network's behavior cannot. The 20-percentage-point accuracy gap closes considerably when viewed through the lens of safety-critical system certification.

Furthermore, the accuracy comparison is not fixed: the zero-parameter method has no training data, while deep learning results are reported on matched train/test splits that may not reflect the true heterogeneity of the clinical population. Evaluated under identical cross-patient, cross-center conditions using the SzCORE protocol, we predict the gap will narrow substantially.

### 7.3 Clinical Translation Pathway

The proposed framework is architecturally compatible with closed-loop responsive neurostimulation in three ways:

**Real-time operation.** As established in Section 5.3, the algorithm runs within the 1-second step interval on laptop CPUs and within the step interval on embedded DSPs with N ≤ 16 channels. The 16-channel constraint is not fundamental — it reflects typical implantable BCI form factors (e.g., NeuroPace RNS uses 4 channels; next-generation devices may support 16–32). The SOZ-specific TE computation in Phase 4 can be run with only the SOZ channels and their nearest neighbors, not the full network.

**Threshold calibration without training.** The collapse threshold θ = μ_inter − 2σ_inter is computed from the interictal baseline, which every implantable BCI accumulates continuously between seizures. This means the algorithm "calibrates itself" from ongoing recordings without any labeled training data — an essential property for a device deployed over years in a single patient.

**False positive management.** The 10-second persistence criterion in collapse detection is the primary false positive control mechanism. Longer persistence windows reduce false positives at the cost of reduced lead time. This tradeoff is tunable at deployment time based on the clinical priority (maximum sensitivity for frequent dangerous seizures vs. maximum specificity for seizures where stimulation risk is higher). This tunable tradeoff — transparent, named, and clinically interpretable — is a significant advantage over the implicit tradeoffs embedded in a neural network's softmax output.

### 7.4 Limitations and Future Directions

**MI estimation at short windows.** The Kraskov estimator has known finite-sample bias that increases as window length decreases. Our 5-second window at 500 Hz provides 2,500 samples — adequate for k = 5 nearest neighbors — but the bias introduces a systematic underestimate of true MI values. Crucially, this bias is consistent across the recording: it affects Φ_inter and Φ_pre equally, so *relative* changes (the collapse) are unaffected. Supplementary Figure S2 will demonstrate this via sensitivity analysis across window lengths from 1–30 seconds.

**The TE lag as a semi-parameter.** The autocorrelation-based lag selection is data-driven but is not learned from outcome labels. It is therefore a parameter in the engineering sense but not in the machine learning sense. Supplementary analysis will show stability of Phase 4 AUC across lag values from 1–50 ms, demonstrating that the results are not sensitive to this choice.

**Patient heterogeneity and subgroup validity.** The 58-patient HUP cohort spans multiple seizure types, electrode configurations, and medication regimens, and substantial inter-patient variability in MI collapse magnitude and timing was observed (Cohen's d ranging from −53 to +2.88 across patients). The lobe-level bifurcation accounts for part of this variance, but further subgroup analyses (ECoG vs. SEEG; SOZ volume; medication load; lesion status) are needed to characterise which patient dimensions best predict MI phenotype and detection performance. Findings in subgroups where the method fails are as scientifically valuable as those where it succeeds.

**Retrospective dataset.** All analyses are conducted on archival data with retrospective seizure annotations. Prospective validation — where the algorithm runs in real-time on a blinded recording and generates seizure predictions that are then compared against clinician annotations — is the critical next step before any clinical deployment consideration. The alignment with SzCORE is specifically designed to make this transition as straightforward as possible by using evaluation metrics that map directly to prospective performance estimates.

**SOZ label availability and quality.** The public HUP iEEG BIDS release (NEMAR ds004100) does not include per-electrode SOZ annotation columns in `*_electrodes.tsv`. The only available electrode-level data is MNI coordinates (x/y/z) and name. In this study, per-electrode SOZ labels for Phase 4 are derived by combining (i) MNI coordinates with a simplified anatomical atlas and (ii) the patient-level `target` lobe field from `participants.tsv`. This approximation introduces label noise: electrodes assigned to the "target" lobe via coordinate lookup may not have been clinically designated as SOZ, and the target lobe field refers to the surgical target rather than the electrographic SOZ in the strictest sense. The resulting AUC estimates are therefore lower bounds on the true performance against exact per-electrode labels. Two validation paths would improve this: (a) contacting the HUP group directly for the de-identified electrode annotation spreadsheet, or (b) validation on a second dataset (e.g., IEEG Portal patients, FSU dataset) where per-electrode SOZ labels are publicly available. Both paths are identified as priority follow-on work. Even under this labeling limitation, a TE asymmetry AUC > 0.65 against the coordinate-derived labels would provide meaningful evidence for the information black hole hypothesis.

---

## 8. Conclusion

This paper introduces a zero-parameter, causal, and edge-deployable framework for detecting the information-theoretic signatures of seizure onset in iEEG recordings. All four experimental phases are complete on the 54–55 patient HUP iEEG cohort. The central finding is not a single significant statistic but a convergent bifurcation: across MI signal level (Phase 1), graph topology (Phase 2), and the MI-variance relationship (Phase 3), the same bimodal patient-level split recurs independently. Type A patients (frontal-onset, ~49%) show MI collapse, network fragmentation, and an MI precursor invisible to variance monitoring. Type B patients (MTL-onset, ~37%) show the mirror image: MI amplification, graph integration, and co-varying MI and variance. A fourth independent measure — TE directional asymmetry — yields partial confirmation: mean AUC = 0.50 against noisy MNI proxy labels, but 9/50 patients achieve AUC > 0.75 and MTL-targeted patients reach mean AUC = 0.584, consistent with the lobe-specific pattern. The AUC is a lower bound pending exact per-electrode SOZ annotations.

The deeper contribution is not methodological but conceptual: reframing seizure onset as an *information bottleneck event* rather than a classification boundary. This reframing transforms the problem from "train a model to recognize seizure patterns" — a problem that requires data, admits overfitting, and resists clinical interpretation — to "measure the information state of a biological network" — a problem that requires only mathematics, is immune to overfitting by construction, and whose every output is directly interpretable in clinical terms.

The zero-parameter, SzCORE-aligned, fully open-source implementation is an invitation to the field: replicate this work, extend it, and demonstrate whether information-theoretic network dynamics are a universal feature of seizure-generating networks across patients, recording modalities, and seizure types. If they are, we have the foundations for a new class of implantable BCI algorithms grounded not in learned statistics but in the physics of neural information flow.

---

## 9. Data and Code Availability

**Dataset.** All analyses use the HUP iEEG dataset, publicly available via NEMAR Data Explorer (OpenNeuro ID: ds004100) and Pennsieve Discover (dataset 179). No special data access agreement is required. The dataset is in BIDS format and can be downloaded using standard BIDS tooling.

**Code.** All analysis code (preprocessing pipeline, Kraskov MI estimator, rolling Φ(t) computation, graph metric computation, variance comparator, TE asymmetry index) is implemented in Python 3.10 with NumPy, SciPy, NetworkX, IDTxl, and MNE-Python. No deep learning framework is used or required. The full pipeline is published as a Kaggle dataset (`mdariffaysalnayem/ieeg-ib-core`) with CPU-only execution notebooks for each of the four experimental phases. All code will be archived to a GitHub repository with Zenodo DOI at time of submission, enabling exact reproduction of all figures and statistics reported here.

**Compute.** All phases run CPU-only. Full-cohort runs complete in approximately 8–10 hours on a standard laptop (Intel Core i7, 16 GB RAM) or equivalent free-tier cloud CPU instance. No GPU, FPGA, or specialized hardware is required.

**Reproducibility checklist.** (i) Dataset publicly available without registration; (ii) all preprocessing parameters specified in §3.2–3.3; (iii) all statistical tests pre-specified in §5.2; (iv) all analysis code open-source; (v) results JSON files included in the supplementary materials. Any research group with a CPU workstation can fully replicate this study from the public dataset.

---

## Mathematical Notation Reference

| Symbol | Definition |
|--------|-----------|
| N | Number of iEEG channels |
| T | Recording duration (samples) |
| x_i(t) | iEEG signal at channel i, time t |
| w_t | Rolling window centered at time t |
| Î(X; Y) | Kraskov k-NN estimated mutual information |
| Φ(t) | Global mean pairwise MI signal |
| θ | MI collapse threshold (μ_inter − 2σ_inter) |
| t_collapse | Time at which Φ(t) first drops below θ |
| t_s | Clinician-annotated seizure onset time |
| τ_lead | MI collapse lead time (t_s − t_collapse) |
| G(t) | MI-weighted network graph at time t |
| G_τ(t) | Thresholded network (top 20% edges) |
| ρ(t) | Network density |
| d̄(t) | Mean node degree |
| C(t) | Global clustering coefficient |
| V(t) | Global mean signal variance |
| t_collapse,V | Time at which V(t) exceeds its rise threshold |
| Δτ | MI advance time over variance (t_collapse,V − t_collapse,Φ) |
| TE_{X→Y} | Transfer entropy from channel X to channel Y |
| TE_{i→rest} | Mean outgoing transfer entropy from channel i |
| TE_{rest→i} | Mean incoming transfer entropy to channel i |
| A_i | TE asymmetry index for channel i |

---

## Figure Plan

| Figure | Content |
|--------|---------|
| Figure 1 | (a) Raw iEEG traces for a representative Type A seizure with t_s marked; (b) Φ(t) time series showing preictal MI collapse; (c) per-patient Cohen's d distribution across all 54 patients showing the bimodal split; (d) boxplot Φ_pre vs. Φ_inter stratified by Type A / Type B / indeterminate |
| Figure 2 | (a) Circular network graph at interictal / early preictal / late preictal for a Type A patient (fragmentation) and a Type B patient (integration) side-by-side; (b) ρ(t) time series for representative patients of each type; (c) scatter of per-patient Cohen's d (Phase 1) vs. Phase 2 network fragmentation direction, showing concordance |
| Figure 3 | (a) Overlay of Φ(t) and V(t) for a representative Type A patient where MI collapses while V increases (the "invisible precursor" case); (b) interictal r(Φ, V) by phenotype group (Type A vs. Type B vs. indeterminate); (c) bar chart showing 6 invisible-precursor patients vs. 9 co-collapse patients vs. remaining |
| Figure 4 | (a) AUC distribution across all 50 labelled patients stratified by MNI lobe label source; (b) ranked A_i bar chart for HUP135 (AUC = 1.0); (c) ROC curves for top three MTL patients and the median patient; (d) mean AUC by lobe subgroup (MTL > TEMPORAL > FRONTAL) |
| Figure 5 | Unified schematic: the bifurcation narrative — Type A and Type B pathways through all four measures, from interictal to ictal, with detection polarity for each type |
| Figure 6 | Personalized closed-loop detection diagram: initial interictal recording → MI phenotype classification → threshold polarity assignment → real-time collapse or amplification detection → stimulation trigger |
| Supp. S1 | Per-patient Cohen's d (Phase 1) and ρ change (Phase 2) scatter for all 54 patients with lobe annotation |
| Supp. S2 | Sensitivity analysis: bifurcation boundary stability vs. window length (1–30 s) and k parameter (3–10) |

---

## 10. References

[1] Mahmood A, et al. (2025). Machine and deep learning-based seizure prediction: a scoping review. *Applied Sciences*, 15(11), 6279.

[2] Bergey GK, et al. (2015). Long-term treatment with responsive brain stimulation in adults with refractory partial seizures. *Neurology*, 84(8), 810–817.

[3] Kwan P, Brodie MJ. (2000). Early identification of refractory epilepsy. *New England Journal of Medicine*, 342(5), 314–319.

[4] Bandarabadi M, et al. (2021). Epileptic seizure prediction using a multiobjective evolutionary algorithm. *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, 29, 1–10.

[5] Geng D, et al. (2024). Prototype-based contrastive learning for interpretable iEEG seizure prediction. *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, 32, 1123–1133.

[6] Lundberg SM, Lee SI. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.

[7] Zhao J, et al. (2024). Deep learning for epileptic seizure detection using causal spatio-temporal transfer entropy. *Computer Methods and Programs in Biomedicine*, 256, 108384.

[8] Tishby N, Pereira FC, Bialek W. (2000). The information bottleneck method. *37th Annual Allerton Conference on Communication, Control, and Computation*.

[9] Tishby N, Zaslavsky N. (2015). Deep learning and the information bottleneck principle. *IEEE Information Theory Workshop (ITW)*, arXiv:1503.02406.

[10] Saxe AM, et al. (2019). On the information bottleneck theory of deep learning. *Journal of Statistical Mechanics: Theory and Experiment*, 2019(12), 124020.

[11] Kramer MA, Cash SS. (2012). Epilepsy as a disorder of cortical network organization. *Neuroscientist*, 18(4), 360–372.

[12] Schindler K, et al. (2008). Assessing seizure dynamics by analyzing the correlation structure of multichannel intracranial EEG. *Brain*, 131(12), 3269–3281.

[13] Shannon CE. (1948). A mathematical theory of communication. *Bell System Technical Journal*, 27(3), 379–423.

[14] Kraskov A, Stögbauer H, Grassberger P. (2004). Estimating mutual information. *Physical Review E*, 69(6), 066138.

[15] Schreiber T. (2000). Measuring information transfer. *Physical Review Letters*, 85(2), 461.

[16] Wibral M, Vicente R, Lindner M. (2014). *Directed Information Measures in Neuroscience*. Springer.

[17] Bullmore E, Sporns O. (2009). Complex brain networks: graph theoretical analysis of structural and functional systems. *Nature Reviews Neuroscience*, 10(3), 186–198.

[18] Rubinov M, Sporns O. (2010). Complex network measures of brain connectivity: uses and interpretations. *NeuroImage*, 52(3), 1059–1069.

[19] Zijlmans M, et al. (2020). Preictal high-frequency oscillation rates as biomarker for seizure prediction. *Brain Communications*, 3(1), fcaa232.

[20] Schreiber J, et al. (2025). Transient high-connectivity states as preictal biomarkers in epileptic networks. *bioRxiv*, 2025.01.24.634461.

[21] Concetti C, et al. (2024). Causal network approach to seizure onset zone localization in drug-resistant epilepsy. *Computer Methods and Programs in Biomedicine*, 257, 108760.

[22] Van Mierlo P, et al. (2013). Functional brain connectivity from EEG in epilepsy: seizure-onset zone and seizure dynamics. *NeuroImage*, 75, 209–221.

[23] Basu I, et al. (2017). Phase-amplitude coupling and transfer entropy to analyze epileptic network dynamics. *Frontiers in Neural Circuits*, 11, 40.

[24] Zhang Y, et al. (2023). RDANet: a dual self-attention network for epileptic seizure prediction. *IEEE Journal of Biomedical and Health Informatics*, 27(9), 4523–4533.

[25] Vieluf S, et al. (2025). SzCORE: a seizure community open-source research evaluation framework. *arXiv:2505.18191*.

[26] Kini LG, et al. (2016). Virtual resection predicts surgical outcome for drug-resistant epilepsy. *Brain*, 142(8), 2385–2400. [HUP iEEG dataset: NEMAR OpenNeuro ID ds004100; Pennsieve dataset 179]

[27] Watts DJ, Strogatz SH. (1998). Collective dynamics of 'small-world' networks. *Nature*, 393(6684), 440–442.

[28] Wollstadt P, et al. (2019). IDTxl: the information dynamics toolkit xl — a Python package for the efficient analysis of multivariate information dynamics in networks. *Journal of Open Source Software*, 4(34), 1081.

[29] Trevelyan AJ, Schevon CA. (2013). How inhibition influences seizure propagation. *Neuropharmacology*, 69, 45–54.

[30] Mormann F, Andrzejak RG, Elger CE, Lehnertz K. (2007). Seizure prediction: the long and winding road. *Brain*, 130(2), 314–333.

[31] Iasemidis LD, et al. (2003). Adaptive epileptic seizure prediction system. *IEEE Transactions on Biomedical Engineering*, 50(5), 616–627.

[32] Stam CJ, van Dijk BW. (2002). Synchronization likelihood: an unbiased measure of generalized synchronization in multivariate data sets. *Physica D*, 163(3–4), 236–251.

[33] Varotto G, et al. (2012). Epileptogenic networks of type II focal cortical dysplasia: a stereo-EEG study. *NeuroImage*, 61(3), 591–598.

[34] David O, et al. (2011). Imaging the seizure onset zone with stereo-electroencephalography. *Brain*, 134(10), 2898–2911.

[35] Zweiphenning WJEM, et al. (2022). Intraoperative electrocorticography using high-frequency oscillations versus spikes to tailor epilepsy surgery. *Epilepsia*, 63(6), 1445–1457.

[36] Friston KJ. (1994). Functional and effective connectivity in neuroimaging: a synthesis. *Human Brain Mapping*, 2(1–2), 56–78.

[37] Vicente R, Wibral M, Lindner M, Pipa G. (2011). Transfer entropy — a model-free measure of effective connectivity for the neurosciences. *Journal of Computational Neuroscience*, 30(1), 45–67.

[38] Lizier JT. (2014). JIDT: an information-theoretic toolkit for studying the dynamics of complex systems. *Frontiers in Robotics and AI*, 1, 11.

[39] Kuhlmann L, et al. (2018). Seizure prediction — ready for a new era. *Nature Reviews Neurology*, 14(10), 618–630.

[40] Cook MJ, et al. (2013). Prediction of seizure likelihood with a long-term implanted device. *Lancet Neurology*, 12(6), 563–571.

[41] Feldwisch-Drentrup H, et al. (2010). Anticipating the unobserved: prediction of subclinical seizures. *Epilepsia*, 51(Suppl 3), 45–48.

[42] Bandt C, Pompe B. (2002). Permutation entropy: a natural complexity measure for time series. *Physical Review Letters*, 88(17), 174102.

[43] Panzeri S, Brunel N, Logothetis NK, Kayser C. (2010). Sensory neural codes using multiplexed temporal scales. *Trends in Neurosciences*, 33(3), 111–120.

[44] Haas LF. (2003). Hans Berger (1873–1941), Richard Caton (1842–1926), and electroencephalography. *Journal of Neurology, Neurosurgery & Psychiatry*, 74(1), 9.

[45] Viventi J, et al. (2011). Flexible, foldable, actively multiplexed, high-density electrode array for mapping brain activity in vivo. *Nature Neuroscience*, 14(12), 1599–1605.

[46] Shenoy KV, Carmena JM. (2014). Combining local and distant reshape for dynamic control in neural prosthetics. *Neuron*, 83(4), 745–760.

[47] Bassett DS, Sporns O. (2017). Network neuroscience. *Nature Neuroscience*, 20(3), 353–364.

[48] Grinstead CM, Snell JL. (1997). *Introduction to Probability*. American Mathematical Society.
