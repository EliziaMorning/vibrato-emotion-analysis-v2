"""
Visualization & Statistical Analysis
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(__file__), "results", "data")
FIG_DIR  = os.path.join(os.path.dirname(__file__), "results", "figures")
STAT_DIR = os.path.join(os.path.dirname(__file__), "results", "stats")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(STAT_DIR, exist_ok=True)

# ── color palette ──────────────────────────────────────────────────────────────
C_NV = "#5B9BD5"   # blue  - No Vibrato
C_V  = "#ED7D31"   # orange - Vibrato
ALPHA_STRIP = 0.7

# ── load data ──────────────────────────────────────────────────────────────────
df1 = pd.read_csv(os.path.join(DATA_DIR, "model1_avd.csv"))
df2 = pd.read_csv(os.path.join(DATA_DIR, "model2_categorical.csv"))
merged = pd.read_csv(os.path.join(DATA_DIR, "merged.csv"))

nv1 = df1[df1["condition"] == "No Vibrato"]
v1  = df1[df1["condition"] == "Vibrato"]
nv2 = df2[df2["condition"] == "No Vibrato"]
v2  = df2[df2["condition"] == "Vibrato"]


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1  Model 1 – Violin + Strip  (Arousal highlighted)
# ══════════════════════════════════════════════════════════════════════════════
def fig1_violin_model1():
    dims = ["arousal", "valence", "dominance"]
    labels_pretty = ["Arousal", "Valence", "Dominance"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=False)
    fig.suptitle("Model 1 (wav2vec2-MSP-dim)  ·  Dimensional Emotion Scores",
                 fontsize=13, fontweight="bold", y=1.01)

    for ax, dim, lbl in zip(axes, dims, labels_pretty):
        data_nv = nv1[dim].values
        data_v  = v1[dim].values

        # violin
        vp = ax.violinplot([data_nv, data_v], positions=[0, 1],
                           showmedians=True, showextrema=False, widths=0.6)
        vp["bodies"][0].set_facecolor(C_NV); vp["bodies"][0].set_alpha(0.45)
        vp["bodies"][1].set_facecolor(C_V);  vp["bodies"][1].set_alpha(0.45)
        vp["cmedians"].set_color("black"); vp["cmedians"].set_linewidth(2)

        # strip (jitter)
        rng = np.random.default_rng(42)
        jit_nv = rng.uniform(-0.08, 0.08, len(data_nv))
        jit_v  = rng.uniform(-0.08, 0.08, len(data_v))
        ax.scatter(jit_nv,       data_nv, color=C_NV, s=30,
                   alpha=ALPHA_STRIP, zorder=3, edgecolors="white", linewidths=0.4)
        ax.scatter(1 + jit_v,    data_v,  color=C_V,  s=30,
                   alpha=ALPHA_STRIP, zorder=3, edgecolors="white", linewidths=0.4)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["No Vibrato", "Vibrato"], fontsize=10)
        ax.set_title(lbl, fontsize=12)
        ax.set_ylabel("Score" if ax == axes[0] else "", fontsize=9)
        ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig1_model1_violin.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2  Model 1 – Bar + CI  (all three dims)
# ══════════════════════════════════════════════════════════════════════════════
def cohen_d(a, b):
    pooled = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
    return (np.mean(a) - np.mean(b)) / pooled if pooled else 0.0

def bootstrap_ci(a, b, n=5000, seed=42):
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        sa = rng.choice(a, len(a), replace=True)
        sb = rng.choice(b, len(b), replace=True)
        diffs.append(np.mean(sa) - np.mean(sb))
    return np.percentile(diffs, [2.5, 97.5])

def fig2_bar_model1():
    """
    Dual-panel bar chart.
    Left:  full 0–1 axis  → honest view, shows scores cluster near 0.333
    Right: zoomed axis    → makes small differences readable, with explicit
                            Δ annotations and 'axis break' warning
    """
    dims = ["arousal", "valence", "dominance"]
    labels_pretty = ["Arousal", "Valence", "Dominance"]
    x = np.arange(len(dims))
    width = 0.32

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Model 1 (wav2vec2-MSP-dim)  ·  Group Means ± SEM",
                 fontsize=13, fontweight="bold")

    means_nv, means_v, sems_nv, sems_v = [], [], [], []
    for dim in dims:
        nv_vals = nv1[dim].values
        v_vals  = v1[dim].values
        means_nv.append(np.mean(nv_vals))
        means_v.append(np.mean(v_vals))
        sems_nv.append(np.std(nv_vals, ddof=1) / np.sqrt(len(nv_vals)) * 1.96)
        sems_v.append(np.std(v_vals,  ddof=1) / np.sqrt(len(v_vals))  * 1.96)

    for ax, ylim, title_suffix in [
        (ax_full, (0.0, 0.5),   "Full scale (0–0.5)"),
        (ax_zoom, (0.328, 0.340), "Zoomed — ⚠ axis does not start at 0"),
    ]:
        for i, lbl in enumerate(labels_pretty):
            ax.bar(x[i] - width/2, means_nv[i], width, color=C_NV, alpha=0.85,
                   yerr=sems_nv[i], capsize=4, label="No Vibrato" if i == 0 else "")
            ax.bar(x[i] + width/2, means_v[i],  width, color=C_V,  alpha=0.85,
                   yerr=sems_v[i],  capsize=4, label="Vibrato"    if i == 0 else "")

        ax.set_xticks(x); ax.set_xticklabels(labels_pretty, fontsize=11)
        ax.set_ylabel("Mean Score", fontsize=10)
        ax.set_ylim(*ylim)
        ax.set_title(title_suffix, fontsize=10, style="italic")
        ax.legend(fontsize=9)
        ax.spines[["top","right"]].set_visible(False)

    # annotate absolute Δ on zoomed panel
    for i in range(len(dims)):
        delta = means_nv[i] - means_v[i]
        top   = max(means_nv[i] + sems_nv[i], means_v[i] + sems_v[i]) + 0.0003
        ax_zoom.text(x[i], top, f"Δ={delta*1000:.2f}×10⁻³",
                     ha="center", va="bottom", fontsize=8, color="dimgray")

    # warn about axis break
    ax_zoom.text(0.5, 0.02,
                 "Note: all scores span < 0.006 on a 0–1 scale.\n"
                 "Differences are statistically detectable but practically tiny.",
                 transform=ax_zoom.transAxes, ha="center", va="bottom",
                 fontsize=7.5, color="firebrick",
                 bbox=dict(boxstyle="round,pad=0.3", fc="#fff3f3", alpha=0.85))

    path = os.path.join(FIG_DIR, "fig2_model1_bar.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3  Model 2 – Violin + Strip  (happy highlighted)
# ══════════════════════════════════════════════════════════════════════════════
def fig3_violin_model2():
    cats = ["happy", "neutral", "sad", "angry"]
    labels_pretty = ["Happy", "Neutral", "Sad", "Angry"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 5), sharey=False)
    fig.suptitle("Model 2 (HuBERT-IEMOCAP)  ·  Categorical Emotion Probabilities",
                 fontsize=13, fontweight="bold", y=1.01)

    for ax, cat, lbl in zip(axes, cats, labels_pretty):
        data_nv = nv2[cat].values
        data_v  = v2[cat].values

        vp = ax.violinplot([data_nv, data_v], positions=[0, 1],
                           showmedians=True, showextrema=False, widths=0.6)
        vp["bodies"][0].set_facecolor(C_NV); vp["bodies"][0].set_alpha(0.45)
        vp["bodies"][1].set_facecolor(C_V);  vp["bodies"][1].set_alpha(0.45)
        vp["cmedians"].set_color("black"); vp["cmedians"].set_linewidth(2)

        rng = np.random.default_rng(42)
        jit_nv = rng.uniform(-0.08, 0.08, len(data_nv))
        jit_v  = rng.uniform(-0.08, 0.08, len(data_v))
        ax.scatter(jit_nv,    data_nv, color=C_NV, s=30,
                   alpha=ALPHA_STRIP, zorder=3, edgecolors="white", linewidths=0.4)
        ax.scatter(1+jit_v,   data_v,  color=C_V,  s=30,
                   alpha=ALPHA_STRIP, zorder=3, edgecolors="white", linewidths=0.4)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["No Vib", "Vib"], fontsize=9)
        ax.set_title(lbl, fontsize=12)
        ax.set_ylabel("Probability" if ax == axes[0] else "", fontsize=9)
        ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig3_model2_violin.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4  Model 2 – Bar + CI
# ══════════════════════════════════════════════════════════════════════════════
def fig4_bar_model2():
    cats = ["happy", "neutral", "sad", "angry"]
    labels_pretty = ["Happy", "Neutral", "Sad", "Angry"]
    x = np.arange(len(cats))
    width = 0.32

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Model 2  ·  Group Means ± 95% CI",
                 fontsize=13, fontweight="bold")

    for i, (cat, lbl) in enumerate(zip(cats, labels_pretty)):
        nv_vals = nv2[cat].values
        v_vals  = v2[cat].values
        m_nv, m_v = np.mean(nv_vals), np.mean(v_vals)
        sem_nv = np.std(nv_vals, ddof=1) / np.sqrt(len(nv_vals)) * 1.96
        sem_v  = np.std(v_vals,  ddof=1) / np.sqrt(len(v_vals))  * 1.96

        ax.bar(x[i] - width/2, m_nv, width, color=C_NV, alpha=0.85,
               yerr=sem_nv, capsize=4, label="No Vibrato" if i == 0 else "")
        ax.bar(x[i] + width/2, m_v,  width, color=C_V,  alpha=0.85,
               yerr=sem_v,  capsize=4, label="Vibrato" if i == 0 else "")

    ax.set_xticks(x); ax.set_xticklabels(labels_pretty, fontsize=11)
    ax.set_ylabel("Mean Probability", fontsize=10)
    ax.legend(fontsize=10)
    ax.spines[["top","right"]].set_visible(False)

    # annotate absolute Δ above each pair
    for i, (cat, lbl) in enumerate(zip(cats, labels_pretty)):
        nv_vals = nv2[cat].values
        v_vals  = v2[cat].values
        m_nv, m_v = np.mean(nv_vals), np.mean(v_vals)
        sem_nv = np.std(nv_vals, ddof=1) / np.sqrt(len(nv_vals)) * 1.96
        sem_v  = np.std(v_vals,  ddof=1) / np.sqrt(len(v_vals))  * 1.96
        delta = m_nv - m_v
        top   = max(m_nv + sem_nv, m_v + sem_v) + 0.01
        sign  = "+" if delta >= 0 else ""
        ax.text(x[i], top, f"Δ={sign}{delta:.3f}", ha="center", va="bottom",
                fontsize=8, color="dimgray")

    path = os.path.join(FIG_DIR, "fig4_model2_bar.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5  Fusion – Arousal × Happy Scatter
# ══════════════════════════════════════════════════════════════════════════════
def confidence_ellipse(x, y, ax, n_std=1.5, **kwargs):
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    rx, ry = np.sqrt(1 + pearson), np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=rx*2, height=ry*2, **kwargs)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_x, mean_y = np.mean(x), np.mean(y)
    t = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)
    ellipse.set_transform(t + ax.transData)
    return ax.add_patch(ellipse)

def fig5_fusion_scatter():
    nv = merged[merged["condition"] == "No Vibrato"]
    v  = merged[merged["condition"] == "Vibrato"]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(nv["arousal"], nv["happy"], color=C_NV, s=60, alpha=0.8,
               label="No Vibrato", edgecolors="white", linewidths=0.5)
    ax.scatter(v["arousal"],  v["happy"],  color=C_V,  s=60, alpha=0.8,
               label="Vibrato",    edgecolors="white", linewidths=0.5)

    # centroids
    ax.scatter(nv["arousal"].mean(), nv["happy"].mean(),
               color=C_NV, s=150, marker="D", zorder=5, edgecolors="white")
    ax.scatter(v["arousal"].mean(),  v["happy"].mean(),
               color=C_V,  s=150, marker="D", zorder=5, edgecolors="white")

    # confidence ellipses
    confidence_ellipse(nv["arousal"].values, nv["happy"].values, ax,
                       n_std=1.5, facecolor=C_NV, alpha=0.12, edgecolor=C_NV, lw=1.5)
    confidence_ellipse(v["arousal"].values,  v["happy"].values,  ax,
                       n_std=1.5, facecolor=C_V,  alpha=0.12, edgecolor=C_V,  lw=1.5)

    # Spearman corr annotation
    r_all, p_all = stats.spearmanr(merged["arousal"], merged["happy"])
    ax.annotate(f"Spearman r = {r_all:.3f}  (p={p_all:.3f})",
                xy=(0.04, 0.96), xycoords="axes fraction",
                fontsize=9, va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    ax.set_xlabel("Arousal  [Model 1]", fontsize=11)
    ax.set_ylabel("Happy probability  [Model 2]", fontsize=11)
    ax.set_title("Fusion  ·  Arousal × Happy Probability", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines[["top","right"]].set_visible(False)

    path = os.path.join(FIG_DIR, "fig5_fusion_scatter.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6  Fusion – Effect Size (Cohen's d) comparison
# ══════════════════════════════════════════════════════════════════════════════
def fig6_effect_size():
    variables = {
        "Arousal\n[M1]":   (nv1["arousal"],   v1["arousal"]),
        "Valence\n[M1]":   (nv1["valence"],   v1["valence"]),
        "Dominance\n[M1]": (nv1["dominance"], v1["dominance"]),
        "Happy\n[M2]":     (nv2["happy"],     v2["happy"]),
        "Neutral\n[M2]":   (nv2["neutral"],   v2["neutral"]),
        "Sad\n[M2]":       (nv2["sad"],       v2["sad"]),
        "Angry\n[M2]":     (nv2["angry"],     v2["angry"]),
    }
    names = list(variables.keys())
    ds = [cohen_d(nv.values, v.values) for nv, v in variables.values()]

    colors = [C_NV if "[M1]" in n else C_V for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(names, ds, color=colors, alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.8)

    # reference lines
    for x_ref, lbl in [(0.2, "small"), (0.5, "medium"), (0.8, "large")]:
        ax.axvline( x_ref, color="grey", lw=0.7, ls="--")
        ax.axvline(-x_ref, color="grey", lw=0.7, ls="--")
        ax.text(x_ref + 0.02, len(names)-0.5, lbl, color="grey", fontsize=7)

    # value labels
    for bar, d in zip(bars, ds):
        xpos = bar.get_width() + (0.02 if d >= 0 else -0.02)
        ha   = "left" if d >= 0 else "right"
        ax.text(xpos, bar.get_y() + bar.get_height()/2,
                f"{d:.3f}", va="center", ha=ha, fontsize=8)

    ax.set_xlabel("Cohen's d  (No Vibrato − Vibrato)", fontsize=10)
    ax.set_title("Fusion  ·  Effect Size Across Both Models\n"
                 "Positive = No Vibrato higher  |  Negative = Vibrato higher",
                 fontsize=11, fontweight="bold")

    patch_m1 = mpatches.Patch(color=C_NV, alpha=0.85, label="Model 1 variables")
    patch_m2 = mpatches.Patch(color=C_V,  alpha=0.85, label="Model 2 variables")
    ax.legend(handles=[patch_m1, patch_m2], fontsize=9, loc="lower right")
    ax.spines[["top","right"]].set_visible(False)

    path = os.path.join(FIG_DIR, "fig6_effect_size.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# ══════════════════════════════════════════════════════════════════════════════
# STATISTICS – Raw data tables + Mann-Whitney + Cohen's d + Bootstrap CI
# ══════════════════════════════════════════════════════════════════════════════
def run_stats():
    rows = []

    def add_rows(nv_vals, v_vals, var_name, model):
        for cond, vals in [("No Vibrato", nv_vals), ("Vibrato", v_vals)]:
            rows.append({
                "model":     model,
                "variable":  var_name,
                "condition": cond,
                "mean":  round(np.mean(vals), 5),
                "sd":    round(np.std(vals, ddof=1), 5),
                "median":round(np.median(vals), 5),
                "iqr":   round(np.percentile(vals,75)-np.percentile(vals,25), 5),
                "min":   round(np.min(vals), 5),
                "max":   round(np.max(vals), 5),
            })

    # Model 1
    for dim in ["arousal", "valence", "dominance"]:
        add_rows(nv1[dim].values, v1[dim].values, dim.capitalize(), "wav2vec2-MSP-dim")

    # Model 2
    for cat in ["happy", "neutral", "sad", "angry"]:
        add_rows(nv2[cat].values, v2[cat].values, cat.capitalize(), "HuBERT-IEMOCAP")

    desc_df = pd.DataFrame(rows)
    desc_df.to_csv(os.path.join(STAT_DIR, "descriptive_stats.csv"), index=False)
    print("Saved descriptive_stats.csv")

    # inferential stats
    inf_rows = []
    specs = [
        ("wav2vec2-MSP-dim", "Arousal",   nv1["arousal"],   v1["arousal"]),
        ("wav2vec2-MSP-dim", "Valence",   nv1["valence"],   v1["valence"]),
        ("wav2vec2-MSP-dim", "Dominance", nv1["dominance"], v1["dominance"]),
        ("HuBERT-IEMOCAP",  "Happy",     nv2["happy"],     v2["happy"]),
        ("HuBERT-IEMOCAP",  "Neutral",   nv2["neutral"],   v2["neutral"]),
        ("HuBERT-IEMOCAP",  "Sad",       nv2["sad"],       v2["sad"]),
        ("HuBERT-IEMOCAP",  "Angry",     nv2["angry"],     v2["angry"]),
    ]
    for model, var, nv_vals, v_vals in specs:
        nv_a, v_a = nv_vals.values, v_vals.values
        u, p = stats.mannwhitneyu(nv_a, v_a, alternative="two-sided")
        d = cohen_d(nv_a, v_a)
        ci = bootstrap_ci(nv_a, v_a)
        r_sp, p_sp = stats.spearmanr(
            merged["arousal"], merged["happy"]
        ) if var in ["Arousal","Happy"] else (None, None)
        abs_diff = abs(np.mean(nv_a) - np.mean(v_a))
        # practical significance: differences < 0.005 on a 0-1 scale flagged as trivial
        practical = ("trivial (delta < 0.005 on 0-1 scale)"
                     if abs_diff < 0.005 else "meaningful")
        inf_rows.append({
            "model": model,
            "variable": var,
            "mean_NV": round(np.mean(nv_a), 5),
            "mean_V":  round(np.mean(v_a), 5),
            "abs_diff": round(abs_diff, 5),
            "MW_U": round(u, 1),
            "p_value": round(p, 4),
            "significant_p05": p < 0.05,
            "cohens_d": round(d, 4),
            "effect_size": ("large" if abs(d)>0.8 else "medium" if abs(d)>0.5
                            else "small" if abs(d)>0.2 else "negligible"),
            "bootstrap_ci_95_low":  round(ci[0], 5),
            "bootstrap_ci_95_high": round(ci[1], 5),
            "practical_significance": practical,
        })

    inf_df = pd.DataFrame(inf_rows)
    inf_df.to_csv(os.path.join(STAT_DIR, "inferential_stats.csv"), index=False)
    print("Saved inferential_stats.csv")

    # Spearman: arousal vs happy (cross-model)
    r_sp, p_sp = stats.spearmanr(merged["arousal"], merged["happy"])
    sp_df = pd.DataFrame([{
        "analysis": "Spearman correlation  Arousal (M1) × Happy (M2)",
        "rho": round(r_sp, 4),
        "p_value": round(p_sp, 4),
        "N": len(merged),
    }])
    sp_df.to_csv(os.path.join(STAT_DIR, "spearman_corr.csv"), index=False)
    print("Saved spearman_corr.csv")

    # pretty print
    print("\n── Inferential Statistics ─────────────────────────────────────────")
    cols = ["model","variable","mean_NV","mean_V","abs_diff",
            "p_value","significant_p05","cohens_d","effect_size","practical_significance"]
    print(inf_df[cols].to_string(index=False))
    print(f"\nSpearman ρ(Arousal, Happy) = {r_sp:.4f}, p = {p_sp:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=== Generating figures ===")
    fig1_violin_model1()
    fig2_bar_model1()
    fig3_violin_model2()
    fig4_bar_model2()
    fig5_fusion_scatter()
    fig6_effect_size()

    print("\n=== Running statistics ===")
    run_stats()
    print("\nDone.")
