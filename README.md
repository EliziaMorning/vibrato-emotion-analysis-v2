# Vibrato & Emotion Analysis v2

Cross-model analysis of how vocal vibrato affects AI-perceived emotional dimensions, using two complementary emotion recognition frameworks.

---

## Research Question

Does the presence of vibrato in sustained vocal notes shift AI-predicted emotional scores, and do two theoretically distinct models converge on the same answer?

---

## Dataset

| Group | Files | Description |
|-------|-------|-------------|
| No Vibrato (NV) | NV_1 – NV_15 | Sustained notes without vibrato |
| Vibrato (V) | V_1 – V_15 | Same phrases with vibrato |

- 30 recordings total · single vocalist · `.m4a` format  
- Analysis segment: last **0.60 s** (sustain region where vibrato is applied)

---

## Models

### Model 1 — Dimensional (wav2vec2-MSP-dim)
`audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim`  
Outputs continuous scores for **Arousal / Valence / Dominance** (0–1).  
Based on the *dimensional* view of emotion (Russell's circumplex model).

### Model 2 — Categorical (HuBERT-IEMOCAP)
`superb/hubert-large-superb-er`  
Outputs probability distributions over **Happy / Neutral / Sad / Angry**.  
Based on the *categorical* view of discrete emotional states (trained on IEMOCAP).

**Rationale for using two models:** Emotion theory is divided between dimensional and categorical frameworks. If vibrato genuinely shifts perceived emotion, the effect should appear regardless of which framework is used. Agreement between models strengthens confidence; disagreement reveals which dimension or category captures the effect.

---

## Visualizations

### Model 1 — Dimensional Scores
| | |
|---|---|
| ![fig1](results/figures/fig1_model1_violin.png) | ![fig2](results/figures/fig2_model1_bar.png) |
| **Fig 1.** Violin + strip plots. Each dot = one recording (n=15 per group). Median shown as horizontal line inside violin. | **Fig 2.** Group means ± 95% CI. Arousal shows a Vibrato-higher trend; Dominance is significantly lower with Vibrato. |

### Model 2 — Categorical Probabilities
| | |
|---|---|
| ![fig3](results/figures/fig3_model2_violin.png) | ![fig4](results/figures/fig4_model2_bar.png) |
| **Fig 3.** Probability distributions by condition. Sad probability visibly higher in Vibrato group. | **Fig 4.** Group means ± 95% CI. Happy/Neutral/Angry all higher in No Vibrato; Sad clearly reversed. |

### Cross-Model Fusion
| | |
|---|---|
| ![fig5](results/figures/fig5_fusion_scatter.png) | ![fig6](results/figures/fig6_effect_size.png) |
| **Fig 5.** Arousal (Model 1) × Happy probability (Model 2). Each point = one recording. Diamonds = group centroids; shaded ellipses = 1.5σ confidence region. | **Fig 6.** Cohen's d for all variables across both models. Negative d = Vibrato group higher. Reference lines at small/medium/large thresholds. |

---

## Raw Data

- [`results/data/model1_avd.csv`](results/data/model1_avd.csv) — per-file Arousal/Valence/Dominance scores  
- [`results/data/model2_categorical.csv`](results/data/model2_categorical.csv) — per-file Happy/Neutral/Sad/Angry probabilities  
- [`results/data/merged.csv`](results/data/merged.csv) — both models joined by file  
- [`results/stats/descriptive_stats.csv`](results/stats/descriptive_stats.csv) — mean, SD, median, IQR, min, max per variable per condition  

---

## Statistical Analysis

### Methods

| Test | Purpose |
|------|---------|
| **Mann-Whitney U** | Non-parametric group comparison (no normality assumption; n=15) |
| **Cohen's d** | Effect size — magnitude of difference independent of sample size |
| **Bootstrap 95% CI** | Confidence interval without distributional assumptions |
| **Spearman ρ** | Cross-model correlation (Arousal × Happy probability) |

### Results

| Model | Variable | p-value | Significant | Cohen's d | Effect Size | Mean NV | Mean V |
|-------|----------|---------|-------------|-----------|-------------|---------|--------|
| wav2vec2-MSP-dim | Arousal | 0.106 | — | −0.607 | medium | 0.331 | 0.334 |
| wav2vec2-MSP-dim | Valence | 0.320 | — | −0.232 | small | 0.336 | 0.334 |
| wav2vec2-MSP-dim | Dominance | **0.016** | ✓ | 0.929 | large | 0.332 | 0.332 |
| HuBERT-IEMOCAP | Happy | **0.007** | ✓ | 1.137 | large | 0.422 | 0.348 |
| HuBERT-IEMOCAP | Neutral | **0.001** | ✓ | 1.321 | large | 0.237 | 0.155 |
| HuBERT-IEMOCAP | Sad | **0.001** | ✓ | −1.391 | large | 0.322 | 0.487 |
| HuBERT-IEMOCAP | Angry | **< 0.001** | ✓ | 1.464 | large | 0.014 | 0.005 |

**Cross-model correlation:** Spearman ρ(Arousal, Happy) = −0.168, p = 0.374 (not significant)

Full tables → [`results/stats/`](results/stats/)

---

## Key Findings

1. **Vibrato increases Arousal** (Model 1, medium effect, trend) — consistent with the intuition that vibrato adds acoustic energy and perceived intensity.

2. **Vibrato shifts categorical emotion toward Sad** (Model 2, large effect, p=0.001) while simultaneously reducing Happy, Neutral, and Angry probabilities.

3. **The two models diverge on the valence axis.** The dimensional model (Model 1) does not detect a significant Valence change, while the categorical model (Model 2) shows a clear shift from positive/neutral to sad. This suggests vibrato primarily encodes *intensity* rather than *positivity*.

4. **Dominance decreases with vibrato** (large effect, p=0.016). This may reflect that vibrato, by introducing pitch modulation, reduces the percept of vocal control or assertiveness.

5. **Cross-model correlation is near zero** (ρ = −0.17), confirming that Arousal and Happy probability are not equivalent measures — the two frameworks capture distinct emotional dimensions.

---

## Interpretation

The pattern *Arousal ↑ + Sad ↑* is consistent with emotional states such as longing, yearning, or pathos — sensations often associated with expressive singing that uses vibrato. This aligns with aesthetic theories of vibrato as a technique that adds expressive "color" by introducing tension rather than brightness.

The divergence between Model 1 (dimensional) and Model 2 (categorical) is itself a finding: vibrato-induced acoustic changes do not map cleanly onto a single emotional framework, underscoring the importance of cross-model validation in AI emotion analysis.

---

## Limitations

- Single vocalist; results may not generalize across voice types or recording conditions.
- Both models were trained on speech data (not singing), which may affect calibration.
- n=15 per group limits statistical power; effects should be interpreted with caution.
- Analysis targets the last 0.60 s of each clip; phrase-level context is excluded.

---

## Code

```
analyze.py      — audio loading, Model 1 + Model 2 inference, CSV export
visualize.py    — 6 figures + statistical tests
```

**Dependencies:** `torch`, `transformers`, `librosa`, `scipy`, `matplotlib`, `numpy`, `pandas`
