"""
Publication figure generation for the HUP iEEG MI bifurcation paper.
Saves 4 figures at 300 DPI into els-cas-templates/figs/.
"""
import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT    = "/mnt/files/Developer/bci2"
RES1    = f"{ROOT}/results/ieeg-ib-phase1c"
RES2    = f"{ROOT}/results/ieeg-ib-phase2a"
RES3    = f"{ROOT}/results/ieeg-ib-phase3a"
RES4    = f"{ROOT}/results/ieeg-ib-phase4a"
OUTDIR  = f"{ROOT}/els-cas-templates/figs"
os.makedirs(OUTDIR, exist_ok=True)

# ── palette ───────────────────────────────────────────────────────────────────
C_A     = "#1A6FBF"   # Type A  – strong blue
C_B     = "#C0392B"   # Type B  – strong red
C_INDET = "#7F8C8D"   # indeterminate – grey
C_INTER = "#27AE60"   # interictal – green
C_PREI  = "#E67E22"   # preictal – orange
C_SZ    = "#8E44AD"   # seizure onset – purple
C_V     = "#E67E22"   # variance – orange

FONT    = {"fontsize": 9, "fontfamily": "sans-serif"}
TITLE   = {"fontsize": 10, "fontweight": "bold", "fontfamily": "sans-serif"}
LABEL   = {"fontsize": 8.5, "fontfamily": "sans-serif"}
TICK    = 8
PANEL   = {"fontsize": 13, "fontweight": "bold", "fontfamily": "sans-serif"}
DPI     = 300

# ── helper ────────────────────────────────────────────────────────────────────
def panel_label(ax, txt, x=-0.14, y=1.04):
    ax.text(x, y, txt, transform=ax.transAxes, **PANEL)

def clean_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=TICK)

def load_phi(pid, kind):
    """Load rolling Φ(t) array for a patient from phase3 results."""
    key = "prei" if kind == "prei" else "inter"
    p = f"{RES3}/phi_{key}_p3_{pid}.npy"
    return np.load(p) if os.path.exists(p) else None

def load_V(pid, kind):
    key = "prei" if kind == "prei" else "inter"
    p = f"{RES3}/V_{key}_p3_{pid}.npy"
    return np.load(p) if os.path.exists(p) else None

# ── load CSVs ─────────────────────────────────────────────────────────────────
df1 = pd.read_csv(f"{RES1}/phase1c_patients.csv")
df2 = pd.read_csv(f"{RES2}/phase2a_patients.csv")
df3 = pd.read_csv(f"{RES3}/phase3a_patients.csv")
df4 = pd.read_csv(f"{RES4}/phase4a_patients.csv")

# ── phenotype classification ──────────────────────────────────────────────────
def phenotype(d):
    if d > 0.3:  return "A"
    if d < -0.3: return "B"
    return "I"

df1["pheno"] = df1["cohens_d"].apply(phenotype)

# Phase 2 classification
THRESH_SAT = 0.98
def classify_p2(row):
    if row["mu_rho_inter"] > THRESH_SAT:
        return "saturated"
    # p_rho is one-sided (decrease direction)
    frag = (row["p_rho"] < 0.05 and row["pct_drop_rho"] > 0 and
            row["p_deg"] < 0.05 and row["pct_drop_deg"] > 0 and
            row["p_clust"] < 0.05 and row["pct_drop_clust"] > 0)
    intg = (row["p_rho"] > 0.95 and row["pct_drop_rho"] < 0 and
            row["p_deg"] > 0.95 and row["pct_drop_deg"] < 0 and
            row["p_clust"] > 0.95 and row["pct_drop_clust"] < 0)
    if frag: return "fragmentation"
    if intg: return "integration"
    return "unclassified"

df2["class_p2"] = df2.apply(classify_p2, axis=1)

# ── merge phenotype into df2, df3, df4 ────────────────────────────────────────
pheno_map = df1.set_index("patient_id")["pheno"].to_dict()
df2["pheno"] = df2["patient_id"].map(pheno_map)
df3["pheno"] = df3["patient_id"].map(pheno_map)
df4["pheno"] = df4["patient_id"].map(pheno_map)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Phase 1: Bimodal Cohen's d + representative time series
# ─────────────────────────────────────────────────────────────────────────────
fig1 = plt.figure(figsize=(6.85, 3.2), dpi=DPI)
gs   = GridSpec(1, 3, figure=fig1, wspace=0.42, left=0.08, right=0.97,
                top=0.88, bottom=0.26)
ax_hist = fig1.add_subplot(gs[0, 0])
ax_A    = fig1.add_subplot(gs[0, 1])
ax_B    = fig1.add_subplot(gs[0, 2])

# Panel A – Cohen's d distribution
d_vals = df1["cohens_d"].values
d_clip = np.clip(d_vals, -10, 4)           # clip extreme outliers for display
colors = [C_A if p == "A" else (C_B if p == "B" else C_INDET)
          for p in df1["pheno"]]

ax_hist.scatter(d_clip, np.random.default_rng(42).uniform(0.05, 0.95, len(d_clip)),
                c=colors, s=28, alpha=0.85, zorder=3, linewidths=0)

# KDE overlays
for pheno, col, lw in [("A", C_A, 1.8), ("B", C_B, 1.8)]:
    vals = np.clip(df1.loc[df1.pheno == pheno, "cohens_d"].values, -10, 4)
    if len(vals) > 3:
        kde = gaussian_kde(vals, bw_method=0.5)
        xs = np.linspace(-10, 4, 300)
        ys = kde(xs)
        ys = ys / ys.max() * 0.88
        ax_hist.plot(xs, ys, color=col, lw=lw, zorder=4)

ax_hist.axvline(0.3,  color=C_A, lw=1.2, ls="--", alpha=0.7)
ax_hist.axvline(-0.3, color=C_B, lw=1.2, ls="--", alpha=0.7)
ax_hist.axvline(0,    color="black", lw=0.8, alpha=0.4)
ax_hist.set_xlabel("Cohen's $d$  (interictal $-$ preictal MI)", **LABEL)
ax_hist.set_ylabel("Patients (jittered)", **LABEL)
ax_hist.set_xlim(-10.5, 4.2)
ax_hist.set_ylim(-0.05, 1.05)
ax_hist.set_yticks([])
ax_hist.set_xticks([-10, -8, -6, -4, -2, 0, 2, 4])
ax_hist.set_xticklabels(["-10", "-8", "-6", "-4", "-2", "0", "+2", "+4"], fontsize=7)
clean_ax(ax_hist)
# legend
patches = [mpatches.Patch(color=C_A,     label=f"Type A (collapse, $n=21$)"),
           mpatches.Patch(color=C_B,     label=f"Type B (amplif., $n=25$)"),
           mpatches.Patch(color=C_INDET, label=f"Indet. ($n=8$)")]
ax_hist.legend(handles=patches, fontsize=6.5, loc="upper right",
               framealpha=0.88, facecolor="white", edgecolor="lightgrey",
               handlelength=0.9)
panel_label(ax_hist, "A")

# Panels B & C – representative Φ(t) time series
def plot_phi_series(ax, pid, col_prei, title):
    phi_i = load_phi(pid, "inter")
    phi_p = load_phi(pid, "prei")
    if phi_i is None or phi_p is None:
        ax.text(0.5, 0.5, "data\nnot\ncached", ha="center", va="center",
                transform=ax.transAxes, fontsize=8)
        return

    n_i, n_p = len(phi_i), len(phi_p)
    t_i = np.arange(n_i)         # interictal windows (1s step)
    t_p = np.arange(n_p) + n_i   # preictal windows

    # normalise both to interictal mean so plots are on same scale
    mu_i = phi_i.mean()
    phi_i_n = phi_i / mu_i
    phi_p_n = phi_p / mu_i

    l_inter, = ax.plot(t_i, phi_i_n, color=C_INTER, lw=1.3, label="Interictal")
    ax.fill_between(t_i, phi_i_n, alpha=0.18, color=C_INTER)

    l_prei, = ax.plot(t_p, phi_p_n, color=col_prei, lw=1.5, label="Preictal")
    ax.fill_between(t_p, phi_p_n, alpha=0.18, color=col_prei)

    # interictal mean ± 1 SD band
    mu, sd = 1.0, phi_i_n.std()
    ax.axhspan(mu - sd, mu + sd, color=C_INTER, alpha=0.08)
    ax.axhline(mu, color=C_INTER, lw=0.9, ls="--", alpha=0.6)

    l_sz = ax.axvline(n_i, color=C_SZ, lw=1.6, ls="-", zorder=5, label="Seizure onset")
    ax.set_xlim(0, n_i + n_p + 1)
    ax.set_xticks([n_i // 2, n_i + n_p // 2])
    ax.set_xticklabels(["Inter.", "Pre."], fontsize=7)
    # Mark seizure onset with a small label inside the axes, just below the top
    ax.text(n_i + 1, 0.96, "Onset", ha="left", va="top", fontsize=6.5,
            color=C_SZ, fontweight="bold", transform=ax.get_xaxis_transform())
    ax.set_ylabel("$\\Phi(t)$ / $\\mu_{\\mathrm{inter}}$", **LABEL)
    ax.set_title(f"{pid.replace('sub-', '')}  ($d={df1.loc[df1.patient_id==pid, 'cohens_d'].values[0]:.2f}$)",
                 fontsize=8.5, pad=3)
    clean_ax(ax)
    return l_inter, l_prei, l_sz

# Pick exemplars: strongest Type A with cached data, and a moderate Type B
typeA_pts = df1[df1.pheno == "A"].sort_values("cohens_d", ascending=False)["patient_id"].tolist()
typeB_pts = df1[(df1.pheno == "B") & (df1.cohens_d > -10)].sort_values("cohens_d")["patient_id"].tolist()

pid_A = next((p for p in typeA_pts if load_phi(p, "prei") is not None), None)
pid_B = next((p for p in typeB_pts if load_phi(p, "prei") is not None), None)

leg_handles = []
if pid_A:
    handles = plot_phi_series(ax_A, pid_A, C_A, "Type A")
    if handles:
        leg_handles = list(handles)   # [l_inter, l_prei, l_sz]
    panel_label(ax_A, "B")
if pid_B:
    plot_phi_series(ax_B, pid_B, C_B, "Type B")
    panel_label(ax_B, "C")

# shared legend below panels B and C
if leg_handles:
    fig1.legend(handles=leg_handles,
                labels=["Interictal", "Preictal", "Seizure onset"],
                loc="lower center",
                bbox_to_anchor=(0.67, 0.01),
                ncol=3, fontsize=7.5, framealpha=0.0,
                handlelength=1.2, columnspacing=1.0)

fig1.suptitle("Preictal MI dynamics bifurcate by seizure onset lobe  "
              "($N=54$,  $p_{\\mathrm{Wilcoxon}}=0.92$)",
              fontsize=9.5, y=0.99)

fig1.savefig(f"{OUTDIR}/fig1_phase1_bifurcation.pdf", dpi=DPI, bbox_inches="tight")
fig1.savefig(f"{OUTDIR}/fig1_phase1_bifurcation.png", dpi=DPI, bbox_inches="tight")
print("Figure 1 saved.")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — Phase 2: Graph topology
# ─────────────────────────────────────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(6.85, 2.9), dpi=DPI)
fig2.subplots_adjust(wspace=0.38, left=0.09, right=0.97, top=0.88, bottom=0.17)
ax_scat, ax_bar = axes2

# Panel A – interictal vs preictal density scatter
col_map = {"fragmentation": C_A, "integration": C_B,
           "unclassified": C_INDET, "saturated": "#BDBDBD"}
mksz    = {"fragmentation": 38, "integration": 38,
           "unclassified": 28, "saturated": 22}
mkstyle = {"fragmentation": "o", "integration": "s",
           "unclassified": "^", "saturated": "x"}

for cl, col in col_map.items():
    sub = df2[df2.class_p2 == cl]
    ax_scat.scatter(sub["mu_rho_inter"], sub["mu_rho_prei"],
                    c=col, s=mksz[cl], marker=mkstyle[cl],
                    label=cl.capitalize(), alpha=0.85, linewidths=0.4,
                    edgecolors="white", zorder=3)

# identity line
lim = [0, 1.05]
ax_scat.plot(lim, lim, "k--", lw=0.9, alpha=0.5, zorder=2)
ax_scat.fill_between(lim, lim, [1.05, 1.05], color=C_B, alpha=0.05)
ax_scat.fill_between(lim, [0, 0], lim, color=C_A, alpha=0.05)

ax_scat.set_xlabel("Mean network density — interictal $\\rho_{\\mathrm{inter}}$", **LABEL)
ax_scat.set_ylabel("Mean network density — preictal $\\rho_{\\mathrm{pre}}$", **LABEL)
ax_scat.set_xlim(0, 1.05); ax_scat.set_ylim(0, 1.05)
ax_scat.legend(fontsize=6.5, loc="lower right", framealpha=0.88,
               facecolor="white", edgecolor="lightgrey",
               handlelength=0.9, markerscale=0.9)
clean_ax(ax_scat)
panel_label(ax_scat, "A")

# Panel B – stacked bar: Phase 2 classification breakdown
counts = df2["class_p2"].value_counts()
cats   = ["fragmentation", "integration", "unclassified", "saturated"]
vals   = [counts.get(c, 0) for c in cats]
labels = ["Fragmentation", "Integration", "Unclassified", "Saturated\n(excluded)"]
cols_b = [C_A, C_B, C_INDET, "#BDBDBD"]

bars = ax_bar.bar(labels, vals, color=cols_b, width=0.55, edgecolor="white", lw=0.8)
for bar, v in zip(bars, vals):
    ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                str(v), ha="center", va="bottom", fontsize=9, fontweight="bold")

ax_bar.set_ylabel("Number of patients", **LABEL)
ax_bar.set_ylim(0, max(vals) * 1.22)
ax_bar.set_xticklabels(labels, fontsize=7.5)
clean_ax(ax_bar)
panel_label(ax_bar, "B", x=-0.18)

fig2.suptitle("MI-weighted network topology independently mirrors the MI bifurcation  "
              "($N_{{\\mathrm{non-sat}}}=42$)",
              fontsize=9.5, y=0.99)

fig2.savefig(f"{OUTDIR}/fig2_phase2_graph.pdf", dpi=DPI, bbox_inches="tight")
fig2.savefig(f"{OUTDIR}/fig2_phase2_graph.png", dpi=DPI, bbox_inches="tight")
print("Figure 2 saved.")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — Phase 3: MI vs Variance
# ─────────────────────────────────────────────────────────────────────────────
fig3, axes3 = plt.subplots(1, 2, figsize=(6.85, 2.9), dpi=DPI)
fig3.subplots_adjust(wspace=0.40, left=0.10, right=0.97, top=0.88, bottom=0.17)
ax_scat3, ax_bar3 = axes3

# Clip extreme d_V outliers for display
d_phi = df3["cohens_d_phi"].values
d_V   = np.clip(df3["cohens_d_V"].values, -8, 5)
phenos = df3["pheno"].values

# Classify MI-variance divergence (for Type A)
# Invisible precursor: d_phi > 0.3 AND d_V < 0
# Co-collapse: d_phi > 0.3 AND d_V > 0.3
inv_mask = (df3["cohens_d_phi"] > 0.3) & (df3["cohens_d_V"] < 0)
coc_mask = (df3["cohens_d_phi"] > 0.3) & (df3["cohens_d_V"] > 0.3)

col_pts = []
mk_pts  = []
sz_pts  = []
for i, p in enumerate(phenos):
    if inv_mask.iloc[i]:
        col_pts.append("#FF6B35"); mk_pts.append("*"); sz_pts.append(90)
    elif p == "A":
        col_pts.append(C_A);       mk_pts.append("o"); sz_pts.append(32)
    elif p == "B":
        col_pts.append(C_B);       mk_pts.append("s"); sz_pts.append(32)
    else:
        col_pts.append(C_INDET);   mk_pts.append("^"); sz_pts.append(24)

# plot by group for legend
for pts, col, mk, sz, lbl in [
    (df3.index[inv_mask],               "#FF6B35", "*", 90, "Invisible precursor"),
    (df3.index[(phenos == "A") & ~inv_mask], C_A, "o", 32, "Type A (other)"),
    (df3.index[phenos == "B"],           C_B, "s", 32, "Type B"),
    (df3.index[phenos == "I"],           C_INDET, "^", 24, "Indeterminate"),
]:
    ax_scat3.scatter(d_phi[pts], d_V[pts], c=col, s=sz, marker=mk,
                     label=lbl, alpha=0.85, linewidths=0.3,
                     edgecolors="white", zorder=3)

# quadrant shading
ax_scat3.axhline(0, color="grey", lw=0.8, ls="--", alpha=0.5)
ax_scat3.axvline(0, color="grey", lw=0.8, ls="--", alpha=0.5)
ax_scat3.fill_between([0.3, 10], [0, 0], [-8, -8], color="#FF6B35", alpha=0.07)
ax_scat3.text(1.5, -6.5, "Invisible\nprecursor\nzone", color="#FF6B35",
              fontsize=7, ha="center", style="italic")

ax_scat3.set_xlabel("Cohen's $d_{\\Phi}$  (MI effect size)", **LABEL)
ax_scat3.set_ylabel("Cohen's $d_V$  (variance effect size, clipped at $\\pm$8)", **LABEL)
ax_scat3.set_xlim(-10.5, 3.5); ax_scat3.set_ylim(-8.5, 5.5)
ax_scat3.legend(fontsize=6.5, loc="upper left", framealpha=0.88,
                facecolor="white", edgecolor="lightgrey",
                handlelength=0.9, markerscale=0.85)
clean_ax(ax_scat3)
panel_label(ax_scat3, "A")

# Panel B – interictal r(Φ, V) by phenotype
r_A = df3.loc[df3["pheno"] == "A", "r_inter_phi_V"].dropna()
r_B = df3.loc[df3["pheno"] == "B", "r_inter_phi_V"].dropna()
r_all = df3["r_inter_phi_V"].dropna()

groups = ["Type A\n(frontal)", "Type B\n(MTL)", "Full cohort"]
means  = [r_A.mean(), r_B.mean(), r_all.mean()]
sems   = [r_A.sem(), r_B.sem(), r_all.sem()]
cols_r = [C_A, C_B, "#555555"]

bars3 = ax_bar3.bar(groups, means, yerr=sems, color=cols_r, width=0.45,
                    capsize=5, error_kw={"elinewidth": 1.4, "ecolor": "black"},
                    edgecolor="white", lw=0.8, alpha=0.9)
for bar, m in zip(bars3, means):
    ax_bar3.text(bar.get_x() + bar.get_width() / 2, m + 0.02,
                 f"{m:.3f}", ha="center", va="bottom", fontsize=8.5,
                 fontweight="bold")

# independence threshold
ax_bar3.axhline(0.4, color="black", lw=1.4, ls=":", zorder=5)
ax_bar3.text(0.03, 0.41, "$r = 0.4$ threshold", fontsize=7, va="bottom", ha="left",
             transform=ax_bar3.get_yaxis_transform())

ax_bar3.set_ylabel("Interictal Pearson $r(\\Phi, V)$", **LABEL)
ax_bar3.set_ylim(0, 0.75)
ax_bar3.set_xticklabels(groups, fontsize=8)
clean_ax(ax_bar3)
panel_label(ax_bar3, "B", x=-0.20)

n_inv = inv_mask.sum()
fig3.suptitle(f"MI and signal variance diverge in frontal-onset patients  "
              f"($n_{{\\mathrm{{inv}}}}={n_inv}/16$ Type A)",
              fontsize=9.5, y=0.99)

fig3.savefig(f"{OUTDIR}/fig3_phase3_mi_variance.pdf", dpi=DPI, bbox_inches="tight")
fig3.savefig(f"{OUTDIR}/fig3_phase3_mi_variance.png", dpi=DPI, bbox_inches="tight")
print("Figure 3 saved.")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — Phase 4: TE asymmetry
# ─────────────────────────────────────────────────────────────────────────────
fig4, axes4 = plt.subplots(1, 2, figsize=(6.85, 2.9), dpi=DPI)
fig4.subplots_adjust(wspace=0.42, left=0.10, right=0.97, top=0.88, bottom=0.17)
ax_auc, ax_te = axes4

# Panel A – AUC strip + box by lobe
lobe_order = ["mni_MTL", "mni_TEMPORAL", "mni_FRONTAL"]
lobe_labels = ["MTL", "Temporal", "Frontal"]
lobe_cols   = [C_B, C_INDET, C_A]   # MTL=red (Type B), Frontal=blue (Type A)

rng = np.random.default_rng(7)
for i, (lobe, col) in enumerate(zip(lobe_order, lobe_cols)):
    sub = df4[df4["label_source"] == lobe]["auc"].values
    jit = rng.uniform(-0.18, 0.18, len(sub))
    ax_auc.scatter(np.full(len(sub), i) + jit, sub,
                   c=col, s=28, alpha=0.75, linewidths=0, zorder=3)
    # box: median, IQR
    q25, med, q75 = np.percentile(sub, [25, 50, 75])
    ax_auc.plot([i - 0.22, i + 0.22], [med, med], color=col, lw=2.2, zorder=4)
    ax_auc.plot([i - 0.22, i - 0.22, i + 0.22, i + 0.22, i - 0.22],
                [q25, q75, q75, q25, q25], color=col, lw=1.2, zorder=4)
    # mean dot
    ax_auc.scatter(i, sub.mean(), c=col, s=55, marker="D",
                   edgecolors="white", lw=0.8, zorder=5)
    # annotate mean above the highest data point to avoid overlapping the strip
    ax_auc.text(i, max(sub) + 0.06, f"$\\mu\\!=\\!{sub.mean():.2f}$",
                ha="center", va="bottom", fontsize=7.5, color=col,
                fontweight="bold")

ax_auc.axhline(0.5, color="grey", lw=1.2, ls="--", alpha=0.7, zorder=2)
ax_auc.axhline(0.75, color="black", lw=1.0, ls=":", alpha=0.7, zorder=2)
# Place reference-line labels at the RIGHT edge of the axes (axes coords x,
# data coords y) so they never overlap with strips/boxes
ax_auc.text(0.98, 0.51, "Chance", fontsize=7, color="grey", ha="right", va="bottom",
            transform=ax_auc.get_yaxis_transform())
ax_auc.text(0.98, 0.76, "Target AUC", fontsize=7, ha="right", va="bottom",
            transform=ax_auc.get_yaxis_transform())
ax_auc.set_xticks([0, 1, 2]); ax_auc.set_xticklabels(lobe_labels, fontsize=8.5)
ax_auc.set_ylabel("ROC AUC (TE asymmetry vs proxy SOZ label)", **LABEL)
ax_auc.set_ylim(-0.05, 1.15)
clean_ax(ax_auc)
panel_label(ax_auc, "A")

# Panel B – TE asymmetry index for best MTL patient (HUP135, AUC=1.0)
pid_te = "sub-HUP135"
ai_path = f"{RES4}/A_i_{pid_te}.npy"
te_path = f"{RES4}/te_matrix_{pid_te}.npy"

if os.path.exists(ai_path) and os.path.exists(te_path):
    A_i = np.load(ai_path)
    te  = np.load(te_path)
    n_ch = len(A_i)

    # sort channels by A_i ascending (most sink-like first = predicted SOZ)
    order  = np.argsort(A_i)
    A_sort = A_i[order]
    ch_lbl = [f"Ch{o+1:02d}" for o in order]
    cols_ai = [C_B if v < 0 else C_A for v in A_sort]

    ax_te.barh(range(n_ch), A_sort, color=cols_ai, height=0.72,
               edgecolor="white", lw=0.4, alpha=0.9)
    ax_te.axvline(0, color="black", lw=0.9, alpha=0.7)
    ax_te.set_yticks(range(n_ch))
    ax_te.set_yticklabels(ch_lbl, fontsize=6)
    ax_te.set_xlabel("TE asymmetry index $A_i = \\mathrm{TE}_{i→\\mathrm{rest}} "
                     "- \\mathrm{TE}_{\\mathrm{rest}→i}$", **LABEL)
    ax_te.set_title(f"HUP135 (MTL, AUC = 1.00)  — "
                    "information sinks (red) = predicted SOZ",
                    fontsize=7.5, pad=3)
    # annotate most negative (SOZ candidate) — placed right of zero line,
    # pointing left so the text never overlaps the bar itself
    ax_te.text(0.0002, 0, "← SOZ candidate",
               va="center", ha="left", fontsize=6.5, color=C_B, fontweight="bold")
    clean_ax(ax_te)
    ax_te.spines["left"].set_visible(False)
    ax_te.tick_params(left=False)
    panel_label(ax_te, "B")
else:
    ax_te.text(0.5, 0.5, "TE data\nnot found", ha="center", va="center",
               transform=ax_te.transAxes, fontsize=9)

fig4.suptitle("TE asymmetry identifies SOZ in MTL patients  "
              "(proxy-label lower bound;  MTL mean AUC $= 0.584$)",
              fontsize=9.5, y=0.99)

fig4.savefig(f"{OUTDIR}/fig4_phase4_te_soz.pdf", dpi=DPI, bbox_inches="tight")
fig4.savefig(f"{OUTDIR}/fig4_phase4_te_soz.png", dpi=DPI, bbox_inches="tight")
print("Figure 4 saved.")

print(f"\nAll figures written to {OUTDIR}/")
