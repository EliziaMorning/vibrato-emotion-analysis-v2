# Vibrato & Emotion Analysis v2

Cross-model analysis of how vocal vibrato affects AI-perceived emotion, using two complementary frameworks on the same 30 recordings.

---

## Research Question

Does the presence of vibrato in a sustained vocal note shift the emotional scores predicted by AI emotion models — and do two theoretically distinct models agree?

---

## Dataset

| Group | Files | Description |
|-------|-------|-------------|
| No Vibrato (NV) | NV_1 – NV_15 | Sustained notes without vibrato |
| Vibrato (V) | V_1 – V_15 | Same phrases sung with vibrato |

- 30 recordings · single vocalist · `.m4a` source files
- Analysis segment: last **0.60 s** of the sustain, extracted via energy-based endpoint detection

### Preprocessing Pipeline

```
m4a  →  [ffmpeg] 44.1 kHz mono WAV
     →  [librosa] silence trim (top_db=30) + RMS normalise to 0.08
     →  energy endpoint → last 0.60 s + 0.10 s pad + 15 ms fade
     →  [model] resample to 16 kHz, amplitude normalise, inference
```

---

## Models & Rationale

| | Model | Framework | Output |
|-|-------|-----------|--------|
| **M1** | `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` | Dimensional (Russell's circumplex) | Continuous: Arousal / Valence / Dominance |
| **M2** | `superb/hubert-large-superb-er` (IEMOCAP) | Categorical (discrete states) | Probability: Happy / Neutral / Sad / Angry |

**Why two models?** Emotion science is divided between dimensional and categorical theories. Using both lets us check whether any effect appears regardless of which framework is applied — agreement strengthens confidence; divergence is itself a finding.

---

## Visualizations

### Model 1 — Dimensional Scores

| | |
|---|---|
| ![fig1](results/figures/fig1_model1_violin.png) | ![fig2](results/figures/fig2_model1_bar.png) |
| **Fig 1.** Violin + strip plots for Arousal / Valence / Dominance. Each dot = one recording (n=15 per group). | **Fig 2.** Group means ± SEM. Left panel: full 0–0.5 scale. Right panel: zoomed with explicit Δ labels and axis-break warning. |

### Model 2 — Categorical Probabilities

| | |
|---|---|
| ![fig3](results/figures/fig3_model2_violin.png) | ![fig4](results/figures/fig4_model2_bar.png) |
| **Fig 3.** Probability distributions by condition. Happy visibly higher in No Vibrato; Neutral higher in Vibrato. | **Fig 4.** Group means ± SEM with absolute Δ annotations. |

### Cross-Model Fusion

| | |
|---|---|
| ![fig5](results/figures/fig5_fusion_scatter.png) | ![fig6](results/figures/fig6_effect_size.png) |
| **Fig 5.** Arousal (M1) × Happy probability (M2) scatter. Diamonds = group centroids; ellipses = 1.5σ confidence regions. Spearman ρ annotated. | **Fig 6.** Cohen's d for all 7 variables across both models on a single axis. Dashed lines at small / medium / large thresholds. |

---

## Raw Data

| File | Contents |
|------|---------|
| [`results/data/model1_avd.csv`](results/data/model1_avd.csv) | Per-file Arousal / Valence / Dominance (raw regression logits) |
| [`results/data/model2_categorical.csv`](results/data/model2_categorical.csv) | Per-file Happy / Neutral / Sad / Angry probabilities |
| [`results/data/merged.csv`](results/data/merged.csv) | Both models joined by file |
| [`results/data/segment_metadata.csv`](results/data/segment_metadata.csv) | Segment start / end times per file |

---

## Statistical Analysis

### Methods

| Method | Purpose |
|--------|---------|
| **Mann-Whitney U** | Non-parametric two-group comparison (no normality assumption; n=15) |
| **Cohen's d** | Effect size — magnitude of difference independent of p-value |
| **Bootstrap 95% CI** | Confidence interval on group mean difference (5,000 resamples) |
| **Spearman ρ** | Cross-model correlation: Arousal (M1) × Happy probability (M2) |

### Results

| Model | Variable | Mean NV | Mean V | Abs. Δ | p-value | Cohen's d | Effect | Note |
|-------|----------|---------|--------|--------|---------|-----------|--------|------|
| wav2vec2-MSP-dim | Arousal | 0.0205 | 0.0239 | 0.0034 | 0.184 | −0.49 | small | V↑ trend, non-significant |
| wav2vec2-MSP-dim | Valence | −0.0036 | −0.0039 | 0.0004 | 0.561 | 0.07 | negligible | no difference |
| wav2vec2-MSP-dim | Dominance | 0.0137 | 0.0188 | 0.0051 | 0.062 | −0.58 | medium | V↑ borderline |
| HuBERT-IEMOCAP | **Happy** | **0.518** | **0.379** | **0.139** | **0.009** | **0.97** | **large** | **NV significantly happier** |
| HuBERT-IEMOCAP | **Neutral** | **0.248** | **0.305** | **0.057** | **0.028** | **−0.87** | **large** | **V significantly more neutral** |
| HuBERT-IEMOCAP | Sad | 0.217 | 0.296 | 0.079 | 0.056 | −0.71 | medium | V↑ borderline |
| HuBERT-IEMOCAP | Angry | 0.016 | 0.020 | 0.004 | 0.455 | −0.22 | small | no difference |

**Cross-model:** Spearman ρ(Arousal, Happy) = 0.172, p = 0.364 — not significant

> **Note on Model 1 scale:** All dimensional scores are raw regression logits (range ~0.007–0.037), not a 0–1 probability scale. Both groups score near zero on all three dimensions, reflecting that a speech-trained model assigns low emotional salience to sustained singing tones. Absolute differences are small; results should be interpreted as trends rather than definitive scores.

Full tables → [`results/stats/`](results/stats/)

---

## Key Findings

1. **Vibrato increases Arousal (trend, M1).** Vibrato group Arousal is consistently higher (V=0.024 vs NV=0.021), though not statistically significant (p=0.184) at n=15. The direction is consistent with the original v1 analysis.

2. **Vibrato decreases Happy and increases Neutral (M2, significant).** With vibrato, the AI assigns significantly lower Happy probability (−13.9 pp, p=0.009, d=0.97) and significantly higher Neutral probability (+5.7 pp, p=0.028, d=−0.87).

3. **Vibrato trends toward Sad (M2, borderline).** Sad probability is higher in the vibrato condition (+7.9 pp, p=0.056, d=−0.71) — not significant at α=0.05 but consistent in direction.

4. **The two models do not correlate** (Spearman ρ=0.17, p=0.364). Arousal (intensity dimension, M1) and Happy probability (valence category, M2) capture different aspects — a high-Arousal sample is not necessarily rated as Happy.

### Interpretation

The pattern **Arousal↑ + Happy↓ + Neutral↑** suggests that vibrato increases perceived emotional *intensity* while simultaneously shifting the valence away from positive affect. The AI perceives vibrato-singing as more emotionally charged but less bright or cheerful — consistent with vibrato conveying expressiveness, longing, or pathos rather than simple positivity.

---

## Limitations

- Single vocalist; results may not generalise across voice types or styles.
- Both models trained on speech data, not singing — absolute scores should be treated as relative comparisons only.
- n=15 per group; effects should be replicated with larger samples.
- Segment boundaries detected automatically; slight variation across files.

---

## Code

| File | Description |
|------|-------------|
| `pipeline.py` | Full preprocessing + segmentation + inference pipeline |
| `analyze.py` | Standalone inference script (reference) |
| `visualize.py` | All 6 figures + statistical tests |

**Dependencies:** `torch`, `transformers`, `librosa`, `soundfile`, `scipy`, `matplotlib`, `numpy`, `pandas`
