# Vibrato Emotion Analysis v2

비브라토 유무에 따른 AI 감정 분석 점수 변화 연구 — 두 모델 교차 검증 확장판

![fig1](results/figures/fig1_model1_violin.png)
![fig3](results/figures/fig3_model2_violin.png)

---

## 프로젝트 개요

노래할 때 비브라토를 넣으면 AI 감정 분석 점수가 어떻게 달라지는지에 대해, 직접 녹음한 데이터를 두 가지 AI 모델로 정량 비교했습니다.

기존 v1에서 단일 모델(wav2vec2 차원 감정)로 Arousal 상승 경향을 확인했고, 이번 v2에서는 **범주 감정 모델(HuBERT-IEMOCAP)** 을 추가하여 교차 검증하고, 통계 기법도 적용했습니다.

---

## 실험 방법

- 같은 프레이즈를 No Vibrato / Vibrato로 각각 직접 녹음
- 오디오 전처리 후 sustain 마지막 구간 추출
- 두 AI 모델에 각각 입력하여 감정 점수 산출
- Mann-Whitney U, Cohen's d, Bootstrap CI, Spearman ρ 적용

### 전처리 파이프라인

```
[원본 .m4a]
    │
    ▼  ffmpeg 변환
[44.1 kHz mono WAV]
    │
    ▼  librosa silence trim (top_db=30) + RMS 정규화 (target=0.08)
[전처리 완료 WAV]
    │
    ▼  에너지 기반 끝점 탐지 → 마지막 0.60s + 0.10s pad + 15ms fade
[Sustain 구간 WAV]
    │
    ▼  16 kHz 리샘플링 → 모델 입력
```

전처리 코드: [`pipeline.py`](pipeline.py)

---

## 데이터셋

| 구분 | 개수 | 설명 |
|------|------|------|
| No Vibrato (NV) | 15 | 비브라토 없이 녹음 |
| Vibrato (V) | 15 | 비브라토 포함 녹음 |
| Total | 30 | 개인 보컬 녹음 데이터 |

분석 구간: 각 녹음의 sustain 마지막 **0.60초**

---

## 사용 모델

### Model 1 — 차원 감정 (Dimensional)

- `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim`
- 출력: **Arousal / Valence / Dominance** (연속값 회귀)
- Russell의 Circumplex Model 기반 — 감정을 연속적 차원으로 표현
- v1과의 연속성 확보 및 Arousal(강도) 변화 측정 목적

### Model 2 — 범주 감정 (Categorical)

- `superb/hubert-large-superb-er` (IEMOCAP 기반)
- 출력: **Happy / Neutral / Sad / Angry** (확률값)
- 이산적 감정 범주 이론 기반
- Model 1과 독립적 프레임워크로 교차 검증

두 모델이 같은 방향을 가리키면 신뢰도가 높아지고, 다르게 나오면 그 불일치 자체가 분석 대상이 됩니다.

---

## 시행착오

### phrase-level vs sustain-level

처음에는 녹음 전체(phrase-level)를 분석했습니다. 그런데 조건 차이가 크게 보이지 않았습니다.

원인을 다시 보니, 실제 비브라토 조작 포인트는 마지막 sustain 구간이었습니다. 전체 구간 대비 비브라토를 준 길이가 짧기 때문입니다.
분석 단위를 마지막 0.60s 구간으로 바꾼 뒤 차이가 더 잘 보였습니다.

### pipeline() vs 직접 모델 로딩

HuggingFace의 `pipeline("audio-classification")`을 처음 사용했을 때 Model 1 점수가 모두 ~0.333으로 압축됐습니다.
원인은 pipeline이 회귀 출력에 softmax를 잘못 적용해서 세 값의 합이 1.0이 되게 만들었기 때문입니다.
`AutoModelForAudioClassification`으로 직접 로딩하여 raw logits를 꺼내는 방식으로 수정했습니다.

---

## 결과

### 1) Model 1 — Arousal

비브라토 조건에서 Arousal이 더 높아지는 경향이 보였습니다.

![fig2](results/figures/fig2_model1_bar.png)

| 조건 | Mean | SD | Median |
|------|------|-----|--------|
| No Vibrato | 0.02047 | 0.00842 | 0.01843 |
| Vibrato | 0.02392 | 0.00536 | 0.02329 |

**Mann-Whitney U**: U=80, p=0.184 / **Cohen's d**: −0.49 (small)

> Model 1 점수는 0.007–0.037 범위로 매우 좁습니다. speech 학습 모델이 성악 sustain 구간에 낮은 감정 현저성을 할당하는 특성 때문입니다. 절대값보다 **방향성** 중심으로 해석하는 것이 적절합니다.

---

### 2) Model 1 — Valence

Valence는 두 조건 차이가 매우 작았습니다.

| 조건 | Mean | SD | Median |
|------|------|-----|--------|
| No Vibrato | −0.00356 | 0.00313 | −0.00329 |
| Vibrato | −0.00392 | 0.00655 | −0.00333 |

**Mann-Whitney U**: U=127, p=0.561 / **Cohen's d**: 0.07 (negligible)

---

### 3) Model 1 — Dominance

Dominance도 비브라토 조건에서 다소 높은 경향이 있었으나 유의하지 않습니다.

| 조건 | Mean | SD | Median |
|------|------|-----|--------|
| No Vibrato | 0.01370 | 0.00744 | 0.01307 |
| Vibrato | 0.01877 | 0.00984 | 0.01979 |

**Mann-Whitney U**: U=67, p=0.062 / **Cohen's d**: −0.58 (medium)

---

### 4) Model 2 — Happy / Neutral / Sad / Angry

![fig4](results/figures/fig4_model2_bar.png)

비브라토 조건에서 **Happy 확률이 유의하게 감소**하고 **Neutral / Sad가 증가**했습니다.

| 변수 | NV 평균 | V 평균 | 절대 Δ | p값 | 유의 | Cohen's d | 효과 크기 |
|------|---------|--------|--------|-----|------|-----------|-----------|
| **Happy** | **0.518** | **0.379** | **0.139** | **0.009** | **✓** | **0.97** | **large** |
| **Neutral** | **0.248** | **0.305** | **0.057** | **0.028** | **✓** | **−0.87** | **large** |
| Sad | 0.217 | 0.296 | 0.079 | 0.056 | — | −0.71 | medium |
| Angry | 0.016 | 0.020 | 0.004 | 0.455 | — | −0.22 | small |

---

### 5) 전체 분포 비교

![fig1](results/figures/fig1_model1_violin.png)
![fig3](results/figures/fig3_model2_violin.png)

---

### 6) 교차 모델 Fusion 분석

![fig5](results/figures/fig5_fusion_scatter.png)

Arousal(M1)과 Happy(M2)의 Spearman ρ = 0.17 (p=0.364) — 유의한 상관 없음.
강도(intensity) 차원과 긍정성(positivity) 차원은 독립적으로 작동하며, 두 모델은 서로 다른 감정 측면을 측정합니다.

---

### 7) 효과 크기 통합 비교

![fig6](results/figures/fig6_effect_size.png)

---

## Raw Data

### Model 1 — Arousal / Valence / Dominance (샘플별 전체)

| 파일 | 조건 | Arousal | Valence | Dominance |
|------|------|---------|---------|-----------|
| NV_1.wav | No Vibrato | 0.01651 | −0.00643 | 0.01435 |
| NV_2.wav | No Vibrato | 0.01843 | −0.00322 | 0.01640 |
| NV_3.wav | No Vibrato | 0.01510 | −0.00329 | 0.01108 |
| NV_4.wav | No Vibrato | 0.00773 | −0.00524 | 0.01993 |
| NV_5.wav | No Vibrato | 0.03744 | −0.00187 | 0.00111 |
| NV_6.wav | No Vibrato | 0.02390 | −0.00494 | 0.02259 |
| NV_7.wav | No Vibrato | 0.02422 | −0.00798 | 0.02545 |
| NV_8.wav | No Vibrato | 0.03122 | −0.00251 | 0.01012 |
| NV_9.wav | No Vibrato | 0.01363 | −0.00698 | 0.01307 |
| NV_10.wav | No Vibrato | 0.01864 | −0.00540 | 0.00836 |
| NV_11.wav | No Vibrato | 0.02641 | −0.00061 | 0.01048 |
| NV_12.wav | No Vibrato | 0.01249 | 0.00003 | 0.02520 |
| NV_13.wav | No Vibrato | 0.01370 | −0.00750 | 0.01344 |
| NV_14.wav | No Vibrato | 0.01565 | 0.00233 | 0.01299 |
| NV_15.wav | No Vibrato | 0.03203 | 0.00016 | 0.00100 |
| V_1.wav | Vibrato | 0.02057 | −0.00266 | 0.01358 |
| V_2.wav | Vibrato | 0.02955 | −0.00218 | 0.01349 |
| V_3.wav | Vibrato | 0.02642 | −0.00502 | 0.01359 |
| V_4.wav | Vibrato | 0.02494 | −0.00333 | 0.01979 |
| V_5.wav | Vibrato | 0.03149 | −0.00252 | 0.02992 |
| V_6.wav | Vibrato | 0.02089 | −0.00555 | 0.02195 |
| V_7.wav | Vibrato | 0.02559 | −0.00276 | 0.03043 |
| V_8.wav | Vibrato | 0.01964 | −0.01132 | 0.02273 |
| V_9.wav | Vibrato | 0.02228 | −0.00893 | 0.01794 |
| V_10.wav | Vibrato | 0.02258 | 0.01072 | 0.01112 |
| V_11.wav | Vibrato | 0.02329 | −0.01580 | 0.02789 |
| V_12.wav | Vibrato | 0.01490 | −0.00636 | 0.02187 |
| V_13.wav | Vibrato | 0.03063 | 0.00741 | −0.00602 |
| V_14.wav | Vibrato | 0.01492 | −0.00809 | 0.01158 |
| V_15.wav | Vibrato | 0.03110 | −0.00245 | 0.03171 |

### Model 2 — Happy / Neutral / Sad / Angry (샘플별 전체)

| 파일 | 조건 | Happy | Neutral | Sad | Angry |
|------|------|-------|---------|-----|-------|
| NV_1.wav | No Vibrato | 0.494 | 0.274 | 0.200 | 0.032 |
| NV_2.wav | No Vibrato | 0.210 | 0.309 | 0.474 | 0.007 |
| NV_3.wav | No Vibrato | 0.389 | 0.221 | 0.366 | 0.024 |
| NV_4.wav | No Vibrato | 0.414 | 0.288 | 0.290 | 0.008 |
| NV_5.wav | No Vibrato | 0.564 | 0.249 | 0.176 | 0.010 |
| NV_6.wav | No Vibrato | 0.574 | 0.141 | 0.261 | 0.024 |
| NV_7.wav | No Vibrato | 0.301 | 0.376 | 0.308 | 0.015 |
| NV_8.wav | No Vibrato | 0.705 | 0.186 | 0.094 | 0.015 |
| NV_9.wav | No Vibrato | 0.646 | 0.231 | 0.078 | 0.045 |
| NV_10.wav | No Vibrato | 0.647 | 0.185 | 0.158 | 0.009 |
| NV_11.wav | No Vibrato | 0.660 | 0.220 | 0.110 | 0.010 |
| NV_12.wav | No Vibrato | 0.455 | 0.290 | 0.246 | 0.009 |
| NV_13.wav | No Vibrato | 0.656 | 0.198 | 0.129 | 0.016 |
| NV_14.wav | No Vibrato | 0.373 | 0.361 | 0.256 | 0.010 |
| NV_15.wav | No Vibrato | 0.689 | 0.194 | 0.107 | 0.009 |
| V_1.wav | Vibrato | 0.380 | 0.328 | 0.283 | 0.009 |
| V_2.wav | Vibrato | 0.526 | 0.361 | 0.096 | 0.017 |
| V_3.wav | Vibrato | 0.630 | 0.224 | 0.136 | 0.011 |
| V_4.wav | Vibrato | 0.335 | 0.351 | 0.296 | 0.018 |
| V_5.wav | Vibrato | 0.365 | 0.318 | 0.305 | 0.012 |
| V_6.wav | Vibrato | 0.282 | 0.292 | 0.413 | 0.013 |
| V_7.wav | Vibrato | 0.289 | 0.407 | 0.289 | 0.016 |
| V_8.wav | Vibrato | 0.345 | 0.270 | 0.363 | 0.022 |
| V_9.wav | Vibrato | 0.282 | 0.263 | 0.438 | 0.016 |
| V_10.wav | Vibrato | 0.551 | 0.234 | 0.209 | 0.006 |
| V_11.wav | Vibrato | 0.243 | 0.348 | 0.392 | 0.018 |
| V_12.wav | Vibrato | 0.354 | 0.347 | 0.274 | 0.026 |
| V_13.wav | Vibrato | 0.597 | 0.252 | 0.148 | 0.003 |
| V_14.wav | Vibrato | 0.236 | 0.386 | 0.360 | 0.018 |
| V_15.wav | Vibrato | 0.273 | 0.195 | 0.434 | 0.099 |

### 기술통계 (Descriptive Statistics)

**Model 1 — Arousal**

| 조건 | Mean | SD | Median | IQR | Min | Max |
|------|------|-----|--------|-----|-----|-----|
| No Vibrato | 0.02047 | 0.00842 | 0.01843 | 0.01092 | 0.00773 | 0.03744 |
| Vibrato | 0.02392 | 0.00536 | 0.02329 | 0.00725 | 0.01490 | 0.03149 |

**Model 1 — Valence**

| 조건 | Mean | SD | Median | IQR | Min | Max |
|------|------|-----|--------|-----|-----|-----|
| No Vibrato | −0.00356 | 0.00313 | −0.00329 | 0.00468 | −0.00798 | 0.00233 |
| Vibrato | −0.00392 | 0.00655 | −0.00333 | 0.00474 | −0.01580 | 0.01072 |

**Model 1 — Dominance**

| 조건 | Mean | SD | Median | IQR | Min | Max |
|------|------|-----|--------|-----|-----|-----|
| No Vibrato | 0.01370 | 0.00744 | 0.01307 | 0.00787 | 0.00100 | 0.02545 |
| Vibrato | 0.01877 | 0.00984 | 0.01979 | 0.01178 | −0.00602 | 0.03171 |

**Model 2 — Happy**

| 조건 | Mean | SD | Median | IQR | Min | Max |
|------|------|-----|--------|-----|-----|-----|
| No Vibrato | 0.518 | 0.155 | 0.564 | 0.251 | 0.210 | 0.705 |
| Vibrato | 0.379 | 0.132 | 0.345 | 0.171 | 0.236 | 0.630 |

**Model 2 — Neutral**

| 조건 | Mean | SD | Median | IQR | Min | Max |
|------|------|-----|--------|-----|-----|-----|
| No Vibrato | 0.248 | 0.067 | 0.231 | 0.093 | 0.141 | 0.376 |
| Vibrato | 0.305 | 0.063 | 0.318 | 0.092 | 0.195 | 0.407 |

**Model 2 — Sad**

| 조건 | Mean | SD | Median | IQR | Min | Max |
|------|------|-----|--------|-----|-----|-----|
| No Vibrato | 0.217 | 0.113 | 0.200 | 0.156 | 0.078 | 0.474 |
| Vibrato | 0.296 | 0.109 | 0.296 | 0.136 | 0.096 | 0.438 |

**Model 2 — Angry**

| 조건 | Mean | SD | Median | IQR | Min | Max |
|------|------|-----|--------|-----|-----|-----|
| No Vibrato | 0.016 | 0.011 | 0.010 | 0.011 | 0.007 | 0.045 |
| Vibrato | 0.020 | 0.022 | 0.016 | 0.007 | 0.003 | 0.099 |

---

## 통계 분석

| 기법 | 목적 | 선택 이유 |
|------|------|-----------|
| Mann-Whitney U test | 두 그룹 차이 유의성 검증 | 정규성 가정 없음, n=15 소표본에 적합 |
| Cohen's d | 효과 크기 정량화 | p값만으로는 실질적 차이 크기를 알 수 없음 |
| Bootstrap 95% CI | 평균 차이 신뢰구간 추정 | 정규분포 가정 없는 신뢰구간 (5,000회 리샘플) |
| Spearman ρ | 교차 모델 상관 분석 | 비모수 상관, Arousal × Happy 관계 검증 |

### 추론통계 전체 결과

| 모델 | 변수 | NV 평균 | V 평균 | 절대 Δ | p값 | 유의 | Cohen's d | 효과 크기 | Bootstrap 95% CI |
|------|------|---------|--------|--------|-----|------|-----------|-----------|-----------------|
| wav2vec2-MSP-dim | Arousal | 0.02047 | 0.02392 | 0.00345 | 0.184 | — | −0.49 | small | [−0.008, 0.002] |
| wav2vec2-MSP-dim | Valence | −0.00356 | −0.00392 | 0.00036 | 0.561 | — | 0.07 | negligible | [−0.003, 0.004] |
| wav2vec2-MSP-dim | Dominance | 0.01370 | 0.01877 | 0.00507 | 0.062 | — | −0.58 | medium | [−0.011, 0.001] |
| **HuBERT-IEMOCAP** | **Happy** | **0.518** | **0.379** | **0.139** | **0.009** | **✓** | **0.97** | **large** | **[0.040, 0.237]** |
| **HuBERT-IEMOCAP** | **Neutral** | **0.248** | **0.305** | **0.057** | **0.028** | **✓** | **−0.87** | **large** | **[−0.102, −0.010]** |
| HuBERT-IEMOCAP | Sad | 0.217 | 0.296 | 0.079 | 0.056 | — | −0.71 | medium | [−0.154, −0.003] |
| HuBERT-IEMOCAP | Angry | 0.016 | 0.020 | 0.004 | 0.455 | — | −0.22 | small | [−0.018, 0.006] |

### 교차 모델 상관 분석

| 분석 | Spearman ρ | p값 | N | 해석 |
|------|-----------|-----|---|------|
| Arousal (M1) × Happy (M2) | 0.172 | 0.364 | 30 | 비유의 — 두 모델은 독립적 차원 포착 |

> Arousal(강도)이 높다고 해서 Happy(긍정 정서) 확률이 높아지는 것은 아닙니다. 두 모델이 서로 다른 감정 차원을 측정하고 있음을 확인.

---

## 핵심 결론

**1. Arousal 상승 경향 (Model 1, 비유의)**  
비브라토 조건에서 Arousal이 일관되게 높게 나타났습니다 (NV=0.021, V=0.024, p=0.184, d=−0.49). 통계적으로 유의하지는 않으나 v1 연구와 방향이 일치하며, 비브라토가 피치 진동을 통해 음성 에너지 인식을 높이는 경향을 시사합니다.

**2. Happy 감소 + Neutral 증가 (Model 2, 유의)**  
비브라토 조건에서 Happy 확률이 유의하게 감소하고 (−13.9%p, p=0.009, d=0.97) Neutral이 유의하게 증가했습니다 (+5.7%p, p=0.028, d=−0.87). 비브라토가 긍정적 감정보다 중립적·표현적 감정으로 분류되는 방향으로 이동시킵니다.

**3. Arousal↑ + Happy↓ 패턴**  
비브라토는 **감정적 강도는 높이되, 밝음보다는 표현적 깊이** 쪽으로 감정 인식을 이동시키는 경향이 있습니다. 이는 비브라토를 단순한 기교가 아닌 정서적 색채를 부여하는 표현 수단으로 보는 성악 이론과 부합합니다.

**4. 두 모델 간 비상관**  
Arousal(M1)과 Happy(M2)의 Spearman ρ=0.17 (p=0.364)로 유의한 상관이 없습니다. 강도 차원과 긍정성 차원은 독립적으로 작동하며, 두 모델은 서로 다른 감정 측면을 포착합니다.

---

## 한계점

이번 실험은 동일한 프레이즈를 반복 녹음하여 비브라토 유무만 비교하려고 했지만, 실제 보컬 녹음에서는 완전히 동일한 컨디션을 유지하기 어려워 분석 결과에 일부 영향을 주었을 가능성이 있습니다.

단일 화자(single vocalist), 단일 프레이즈만을 사용한 실험이기 때문에 결과를 일반화하기에는 한계가 있습니다.

사용한 AI 모델은 singing voice 전용 모델이 아니라 speech emotion recognition 기반 모델이므로, 노래의 감정 표현을 완전히 음악적으로 이해한다고 보기 어렵습니다.

비브라토는 주로 마지막 sustain 구간에만 존재했기 때문에, 분석 구간 설정에 따라 결과가 달라질 가능성이 있습니다. 실제로 초기 phrase-level 분석에서는 차이가 거의 나타나지 않았으며, sustain-level 분석으로 수정한 이후에야 감정 점수 변화 경향이 관찰되었습니다.

---

## 폴더 구조

```
vibrato-emotion-analysis-v2/
├── audio/                  # 원본 .m4a 녹음 파일
├── data/
│   ├── processed/          # 전처리 완료 WAV (44.1 kHz)
│   └── segments/           # Sustain 구간 추출 WAV (22 kHz, 0.6s)
├── results/
│   ├── figures/            # 시각화 그래프 6종
│   │   ├── fig1_model1_violin.png
│   │   ├── fig2_model1_bar.png
│   │   ├── fig3_model2_violin.png
│   │   ├── fig4_model2_bar.png
│   │   ├── fig5_fusion_scatter.png
│   │   └── fig6_effect_size.png
│   ├── data/               # 샘플별 감정 점수 CSV
│   └── stats/              # 기술통계 + 추론통계 CSV
├── pipeline.py             # 전처리 → 구간 추출 → 추론 파이프라인
├── visualize.py            # 시각화 + 통계 분석
└── analyze.py              # 단독 추론 스크립트 (참고용)
```
