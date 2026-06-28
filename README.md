# 비브라토 유무에 따른 AI 감정 분석 점수 변화 연구 v2

> **Vibrato & Emotion Analysis v2** — 비브라토 적용 여부가 AI 감정 인식 모델의 출력값에 미치는 영향을 두 개의 이론적 프레임워크로 교차 검증한 연구

---

## 1. 개요

본 연구는 동일한 성악 프레이즈를 **비브라토 없이(No Vibrato)** / **비브라토 있이(Vibrato)** 각각 15회 녹음하고, 두 가지 AI 감정 분석 모델을 적용하여 조건 간 감정 점수 차이를 정량적으로 분석한다.

기존 v1 연구에서 단일 모델(wav2vec2 dimensional)로 Arousal 상승 경향을 확인하였으나, 본 연구에서는 이를 확장하여:

- **교차 검증**: 차원 이론(dimensional) 모델 + 범주 이론(categorical) 모델 동시 적용
- **다각적 시각화**: 분포, 평균 비교, 감정 공간, 효과 크기를 각각 시각화
- **정량적 통계**: Mann-Whitney U, Cohen's d, Bootstrap 95% CI, Spearman ρ 적용
- **Raw Data 공개**: 샘플별 전체 점수 및 기술통계 제공

### 연구 질문

> 비브라토 적용이 AI 감정 분석 모델의 출력값을 유의미하게 변화시키는가?  
> 두 이론 체계에서 일관된 방향의 변화가 관찰되는가?

---

## 2. 데이터셋

| 조건 | 파일명 | 수량 |
|------|--------|------|
| No Vibrato | NV_1 ~ NV_15 | 15개 |
| Vibrato | V_1 ~ V_15 | 15개 |

- 단일 성악 발성자 · 동일 프레이즈 반복 녹음 · `.m4a` 원본
- 분석 구간: 각 녹음의 sustain 마지막 **0.60초**
- 원본 파일: [`audio/`](audio/)

---

## 3. 실험 방법

### 3-1. 전처리 파이프라인

```
[원본 .m4a]
    │
    ▼  ffmpeg 변환
[44.1 kHz mono WAV]
    │
    ▼  librosa silence trim (top_db=30) + RMS 정규화 (target=0.08)
[전처리 완료 WAV]  →  data/processed/
    │
    ▼  에너지 기반 끝점 탐지 → 마지막 0.60s + 0.10s pad + 15ms fade
[Sustain 구간 WAV]  →  data/segments/
    │
    ▼  16 kHz 리샘플링 + amplitude 정규화 → 모델 입력
```

전처리 코드: [`pipeline.py`](pipeline.py)

### 3-2. 사용 모델

#### Model 1 — Dimensional (차원 이론)
- **모델**: `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim`
- **출력**: Arousal / Valence / Dominance (연속값 회귀 출력)
- **이론 배경**: Russell의 Circumplex Model — 감정은 연속적 차원으로 존재
- **선택 근거**: 기존 v1 연구와의 연속성 확보 및 intensity(Arousal) 변화 측정

#### Model 2 — Categorical (범주 이론)
- **모델**: `superb/hubert-large-superb-er` (IEMOCAP 기반)
- **출력**: Happy / Neutral / Sad / Angry (확률값)
- **이론 배경**: 이산적 감정 범주 — 감정은 구별되는 상태로 존재
- **선택 근거**: 차원 모델과 독립적 프레임워크로 교차 검증

#### 두 모델을 사용한 이유
감정 이론은 차원(dimensional)과 범주(categorical) 두 체계로 나뉜다. 비브라토가 실제로 감정 인식에 영향을 미친다면, 이론 체계와 무관하게 감지되어야 한다. 두 모델이 같은 방향을 가리키면 결과 신뢰도가 높아지고, 다르게 나오면 그 불일치 자체가 분석 대상이 된다.

### 3-3. 분석 구조

```
Model 1 ── Arousal / Valence / Dominance ──┐
                                            ├── Fusion 분석 (교차 검증)
Model 2 ── Happy / Neutral / Sad / Angry ──┘
```

---

## 4. 시각화

### Model 1 — 차원 감정 점수

**Fig 1. Violin + Strip Plot** — 분포 형태 + 개별 샘플(n=15) 동시 표시

![fig1](results/figures/fig1_model1_violin.png)

> x축: 조건(No Vibrato / Vibrato), y축: 감정 점수(회귀 출력값)  
> 바이올린: 분포 형태 / 점: 개별 샘플 15개 / 가로선: 중앙값

---

**Fig 2. Bar + CI (이중 패널)** — 전체 스케일 vs 확대 비교

![fig2](results/figures/fig2_model1_bar.png)

> **좌**: 0–0.5 전체 스케일 (실제 점수 범위를 정직하게 표시)  
> **우**: 확대 스케일 + 절대 Δ 표기 (⚠ 축이 0에서 시작하지 않음)  
> 모든 점수가 0.007–0.037 범위에 집중됨 — speech 모델의 sustain 음성 처리 특성

---

### Model 2 — 범주 감정 확률

**Fig 3. Violin + Strip Plot** — Happy / Neutral / Sad / Angry 분포

![fig3](results/figures/fig3_model2_violin.png)

> x축: 조건, y축: 감정 확률(0–1)  
> No Vibrato에서 Happy 분포가 명확히 높고, Vibrato에서 Neutral / Sad가 높음

---

**Fig 4. Bar + CI** — 4개 감정 범주 평균 비교

![fig4](results/figures/fig4_model2_bar.png)

> 각 막대 위 Δ값은 절대 차이(No Vibrato − Vibrato)

---

### Fusion — 교차 모델 분석

**Fig 5. Arousal × Happy 산점도** — 두 모델의 핵심 변수 상관 관계

![fig5](results/figures/fig5_fusion_scatter.png)

> x축: Arousal(Model 1) / y축: Happy 확률(Model 2)  
> 색상: 조건 / 마름모: 그룹 중심 / 타원: 1.5σ 신뢰 구간  
> 두 변수가 상관되지 않음(ρ=0.17, p=0.364) → 두 모델은 독립적 차원 포착

---

**Fig 6. Effect Size 통합 비교** — 두 모델 전체 변수 Cohen's d

![fig6](results/figures/fig6_effect_size.png)

> 양수: No Vibrato 더 높음 / 음수: Vibrato 더 높음  
> 파란 막대: Model 1 변수 / 주황 막대: Model 2 변수  
> 점선: small(0.2) / medium(0.5) / large(0.8) 기준선

---

## 5. Raw Data

### 5-1. 샘플별 전체 점수

- [Model 1 전체 점수 (Arousal/Valence/Dominance)](results/data/model1_avd.csv)
- [Model 2 전체 점수 (Happy/Neutral/Sad/Angry)](results/data/model2_categorical.csv)
- [두 모델 통합 데이터](results/data/merged.csv)
- [Sustain 구간 메타데이터 (시작/끝 시간)](results/data/segment_metadata.csv)

### 5-2. 기술통계 (Descriptive Statistics)

전체 기술통계 테이블: [descriptive_stats.csv](results/stats/descriptive_stats.csv)

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

## 6. 통계 분석

### 6-1. 사용 기법

| 기법 | 목적 | 선택 이유 |
|------|------|-----------|
| **Mann-Whitney U test** | 두 그룹 차이 유의성 검증 | 정규성 가정 없음, n=15 소표본에 적합 |
| **Cohen's d** | 효과 크기 정량화 | p값만으로는 실질적 차이 크기를 알 수 없음 |
| **Bootstrap 95% CI** | 평균 차이 신뢰구간 추정 | 정규분포 가정 없는 신뢰구간 (5,000회 리샘플) |
| **Spearman ρ** | 교차 모델 상관 분석 | 비모수 상관, Arousal × Happy 관계 검증 |

### 6-2. 추론 통계 결과

전체 결과: [inferential_stats.csv](results/stats/inferential_stats.csv)

| 모델 | 변수 | NV 평균 | V 평균 | 절대 Δ | p값 | 유의 | Cohen's d | 효과 크기 | 95% CI |
|------|------|---------|--------|--------|-----|------|-----------|-----------|--------|
| wav2vec2-MSP-dim | Arousal | 0.02047 | 0.02392 | 0.00345 | 0.184 | — | −0.49 | small | [−0.008, 0.002] |
| wav2vec2-MSP-dim | Valence | −0.00356 | −0.00392 | 0.00036 | 0.561 | — | 0.07 | negligible | [−0.003, 0.004] |
| wav2vec2-MSP-dim | Dominance | 0.01370 | 0.01877 | 0.00507 | 0.062 | — | −0.58 | medium | [−0.011, 0.001] |
| **HuBERT-IEMOCAP** | **Happy** | **0.518** | **0.379** | **0.139** | **0.009** | **✓** | **0.97** | **large** | **[0.040, 0.237]** |
| **HuBERT-IEMOCAP** | **Neutral** | **0.248** | **0.305** | **0.057** | **0.028** | **✓** | **−0.87** | **large** | **[−0.102, −0.010]** |
| HuBERT-IEMOCAP | Sad | 0.217 | 0.296 | 0.079 | 0.056 | — | −0.71 | medium | [−0.154, −0.003] |
| HuBERT-IEMOCAP | Angry | 0.016 | 0.020 | 0.004 | 0.455 | — | −0.22 | small | [−0.018, 0.006] |

### 6-3. 교차 모델 상관 분석

전체 결과: [spearman_corr.csv](results/stats/spearman_corr.csv)

| 분석 | Spearman ρ | p값 | N | 해석 |
|------|-----------|-----|---|------|
| Arousal (M1) × Happy (M2) | 0.172 | 0.364 | 30 | 비유의 — 두 모델은 독립적 차원 포착 |

> Arousal(강도)이 높다고 해서 Happy(긍정 정서) 확률이 높아지는 것은 아니다.  
> 두 모델이 서로 다른 감정 차원을 측정하고 있음을 확인.

### 6-4. Model 1 해석 시 주의사항

Model 1의 모든 점수는 0.007–0.037 범위에 집중되어 있어 0–1 스케일 기준으로 절대 차이가 매우 작다. 이는 MSP-Improv 음성 데이터로 학습된 모델이 성악 sustain 구간에 대해 낮은 감정 현저성(emotional salience)을 할당하는 특성에서 기인한다. 통계적 유의성보다 **방향성(direction)** 중심으로 해석하는 것이 적절하다.

---

## 7. 결론

### 핵심 발견

**1. Arousal 상승 경향 (Model 1, 비유의)**  
비브라토 조건에서 Arousal이 일관되게 높게 나타났다 (NV=0.021, V=0.024, p=0.184, d=−0.49). 통계적으로 유의하지는 않으나 v1 연구와 방향이 일치하며, 비브라토가 피치 진동을 통해 음성 에너지 인식을 높이는 경향을 시사한다.

**2. Happy 감소 + Neutral 증가 (Model 2, 유의)**  
비브라토 조건에서 Happy 확률이 유의하게 감소하고 (−13.9%p, p=0.009, d=0.97) Neutral이 유의하게 증가했다 (+5.7%p, p=0.028, d=−0.87). 비브라토가 긍정적 감정보다 중립적·표현적 감정으로 분류되는 방향으로 이동시킨다.

**3. Sad 증가 경향 (Model 2, 경계)**  
Sad 확률이 비브라토 조건에서 높은 경향이 있으나 (p=0.056) α=0.05 기준으로는 유의하지 않다. Happy 감소와 함께 해석하면 비브라토가 긍정 감정에서 벗어나는 방향을 일관되게 지지한다.

**4. 두 모델 간 비상관 (Fusion)**  
Arousal(M1)과 Happy(M2)의 Spearman ρ=0.17 (p=0.364)로 유의한 상관이 없다. 강도(intensity) 차원과 긍정성(positivity) 차원은 독립적으로 작동하며, 두 모델은 서로 다른 감정 측면을 포착한다.

### 종합 해석

`Arousal↑ + Happy↓ + Neutral↑` 패턴은 비브라토가 **감정적 강도는 높이되, 긍정적 밝음보다는 표현적 깊이** 쪽으로 감정 인식을 이동시킴을 시사한다. 이는 비브라토를 단순한 기교가 아닌 **정서적 색채(emotional coloring)**를 부여하는 표현 수단으로 보는 성악 이론과 부합한다.

### 한계

- 단일 발성자 · 단일 프레이즈로 일반화에 한계
- speech 학습 모델을 성악 데이터에 적용 → 절대 점수 해석 주의
- n=15로 통계 검증력(statistical power)이 낮음
- 에너지 기반 자동 구간 탐지의 미세한 편차 존재

---

## 8. 파일 구조

```
vibrato-emotion-analysis-v2/
├── audio/                  # 원본 .m4a 녹음 파일
├── data/
│   ├── processed/          # 전처리 완료 WAV (44.1 kHz)
│   └── segments/           # Sustain 구간 추출 WAV (22 kHz, 0.6s)
├── results/
│   ├── figures/            # 시각화 그래프 6종
│   ├── data/               # 샘플별 감정 점수 CSV
│   └── stats/              # 기술통계 + 추론통계 CSV
├── pipeline.py             # 전처리 → 구간 추출 → 추론 파이프라인
├── visualize.py            # 시각화 + 통계 분석
└── analyze.py              # 단독 추론 스크립트 (참고용)
```

**의존성**: `torch` · `transformers` · `librosa` · `soundfile` · `scipy` · `matplotlib` · `numpy` · `pandas`
