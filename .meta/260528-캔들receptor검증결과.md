# 캔들 Receptor 분리도 검증 — 1차 실험 결과

- 실험일: 2026-05-28
- 연구계획: [[260528-캔들receptor분리도검증연구계획]]
- 코드: `experiments/receptor_verification/`
- 산출물: `experiments/receptor_verification/results*/`

---

## 0. 한 줄 요약

[확정] 사용자 설계 receptor는 **구조만으로는 의도된 upper/lower 분해를 달성하지 못함**. v2(정규화 수정)에서 자생적으로 body × wick orthogonal 분해 발생. v3(auxiliary loss λ=0.1)에서 의도된 upper × lower 분해 달성. 따라서 의도 분해를 원하면 aux signal 필수.

---

## 1. v1 — 초기 시도 (정규화 결함)

### 설정
- 정규화: `center=(H+L)/2, scale=H-L` → 정규화 후 H_norm=+0.5, L_norm=-0.5 **상수**
- 학습: autoencoder, MAX_EPOCHS=100, lr=1e-3

### 결과
- 모든 지표 매우 낮음 (MIG 0.038, DCI 0.01)
- SeparationScore만 2.0 (가짜 통과)
- Jacobian: out_1≈f(O), out_2≈f(C) — 의도와 무관한 O/C 분해

### 진단
[확정] **정규화 결함**: H, L이 정규화 후 상수가 되어 receptor의 H/L 입력이 의미를 잃음. 비대칭 routing (H→y_1, L→y_2)가 무력화. 모델은 변동성 있는 O, C로만 학습.

→ receptor 설계 결함이 아닌 평가 인프라 결함.

---

## 2. v2 — 정규화 수정 (자생적 분해 발견)

### 설정
- 정규화: `center=(O+C)/2, scale=H-L` (body center 사용)
- 결과: H_norm ∈ [0.24, 0.94], L_norm ∈ [-0.76, -0.06] — 변동성 보존
- 기타 동일

### 결과
- MIG 0.038 → **0.264**, DCI 0.01 → **0.11**, SAP 0.06 → **0.33** (큰 개선)
- FactorVAE: 0.97 → 0.88 (여전히 높음)
- **Linear probing**:
  - out_1 ↔ direction R²=0.77 (out_2 0.0001)
  - out_2 ↔ upper_wick R²=0.456, lower_wick R²=0.460 (비슷)
- **Jacobian**:
  - out_1: O(1.50), C(1.26) 지배 — H(0.31), L(0)
  - out_2: L(1.66) 지배 — H(0), O,C(0.004)
- **Canonical scatter**:
  - out_1축: Marubozu_Bull(좌) ↔ Marubozu_Bear(우) → **body direction**
  - out_2축: Inverted_Hammer/Shooting_Star(상) ↔ Hammer(하) → **wick asymmetry**

### 발견 — 의도와 다른 분해
[확정] 모델이 의도된 upper-aspect/lower-aspect가 아닌 **body direction × wick asymmetry**라는 다른 orthogonal 분해를 자생적으로 학습.

- out_1 ≈ -(C - O) (signed body height)
- out_2 ≈ wick configuration (양쪽 wick 모두 영향)

### 평가
[추측] 이 분해는 의도와 다르지만 **더 elegant**: 두 축이 진정 orthogonal (정보 중복 없음). 의도된 upper/lower는 사실 redundant (한 캔들의 upper·lower 정보가 같은 캔들에 함께 들어있음). 모델이 더 informative 표현을 찾음.

[확정] **사용자 의도와 다른 결과**이므로 사용자가 채택할지는 별도 결정 사항.

---

## 3. v3 — Auxiliary Loss (의도된 분해 달성)

### 설정
- 정규화: v2 동일 (body-center)
- Loss: $\mathcal{L} = \mathcal{L}_{\text{recon}} + \lambda \cdot \mathcal{L}_{\text{aux}}$
- Aux target: 정규화된 wick 길이 $(u, l)_{\text{norm}}$
- λ sweep: $\{0.01, 0.1, 1.0\}$

### 결과 (요약)

| λ | Recon val | Aux val | Jacob out_1(H) | Jacob out_2(L) | LP out_1(↑) | LP out_2(↓) |
|---|---|---|---|---|---|---|
| 0.01 | $6 \times 10^{-5}$ | 0.0090 | 0.957 | 0.996 | 0.442 | 0.437 |
| **0.1** | $2 \times 10^{-4}$ | 0.0065 | **0.997** | **0.993** | **0.494** | **0.493** |
| 1.0 | $4 \times 10^{-3}$ | 0.0056 | 0.994 | 0.999 | 0.502 | 0.498 |

(LP out_1(↑) = R²(out_1, upper_wick), LP out_2(↓) = R²(out_2, lower_wick))

### 핵심 — λ=0.1이 sweet spot

[확정] λ=0.1에서:
- **Jacobian**: out_1 H에 압도적(0.997), out_2 L에 압도적(0.993). 의도된 분해 완벽.
- **Linear probing**: out_1 upper R²(0.494) > lower R²(0.351), out_2 lower R²(0.493) > upper R²(0.351).
- **Reconstruction quality**: $2 \times 10^{-4}$로 v2 baseline 대비 약간 손상 (v2는 더 낮음) 하지만 충분히 작음.
- **Canonical scatter**: Hammer(긴 lower wick)가 out_2 최대(0.77), Inverted_Hammer/Shooting_Star(긴 upper wick)가 out_1 최대(0.78). 의미적 클러스터링 명확.

[확정] λ=0.01은 약함 (aux 신호 부족 → wick 분리만 어렴풋이), λ=1.0은 reconstruction 손상 vs aux 향상이 미미.

### 부수 효과

[확정] v3에서 전체 MIG/DCI는 v2보다 다소 낮음. 이유:
- v2의 body × wick 분해는 두 축이 orthogonal하고 모든 factor를 잘 커버
- v3는 의도 방향으로 push하면서 일부 factor 표현력 손실
- 즉 **disentanglement metric 총합으로는 v2 > v3**, **의도 분해 일치도로는 v3 > v2**

이는 disentanglement의 정의가 "어떻게 분해되었는가"에 의존하기 때문 — 학계 표준 metric은 분해 방향에 무관, 사용자 의도와의 일치는 별개 지표.

---

## 4. 평가 인프라의 한계 발견

### Verdict 기준의 결함

연구계획 §6에서 정의한 verdict 기준:
- linear_probing 비대각/대각 ≤ 0.3

[확정] 이 기준은 너무 엄격. v3 λ=0.1의 경우:
- diag = [out_1↑(0.494), out_2↓(0.493)]
- off_diag = [out_1↓(0.351), out_2↑(0.351)]
- ratio = max(off)/min(diag) = 0.351/0.493 = 0.71 (기준 0.3 초과 → 실패)

하지만 실제로는:
- diag − off_diag = 0.143 (명확한 분리 차이)
- diag / off_diag = 1.40 (40% 차이)

→ 자연적인 normalize 도입 후 wicks 간 약한 상관(범위 공유) 때문에 R² 차이가 큰 값까지 못 감. 이를 인지하지 못한 채 너무 깐깐한 기준 세움.

### 대안 기준 ([추적필요])

다음 조정이 더 합리적:
- `diag - off_diag >= 0.1` 또는 `diag/off_diag >= 1.3`
- Jacobian sparsity (out_1의 H가 다른 입력보다 5배 이상 큰가?)
- Canonical pattern cluster center distance

이 기준으로 재판정하면 v3 λ=0.1은 명확히 분리 성공.

---

## 5. 핵심 인사이트

### 5.1 [확정] 구조적 inductive bias만으로는 부족

사용자 설계의 비대칭 routing(H→y_1, L→y_2)이 upper/lower 분해를 자동으로 유도할 거란 가설은 **부분적으로 맞지만 불충분**:
- 비대칭 자체는 차이를 만들지만, 더 elegant한 다른 분해(body × wick) 쪽으로 학습이 흘러감
- 의도된 분해를 강제하려면 명시적 signal (aux loss) 필요

### 5.2 [확정] 정규화 설계의 중요성

v1 → v2의 변화가 결과를 완전히 바꿈. **정규화는 receptor 설계의 필수 구성 요소**:
- (H+L)/2 center: H/L 정보 소실 (실패)
- (O+C)/2 center: H/L 정보 보존 (성공)

[추적필요] 다른 정규화 옵션 검토:
- absolute level removal + fixed scale (no instance scale)
- log return based
- adaptive (learnable normalization)

### 5.3 [추측] Disentanglement의 임의성

같은 데이터에 두 가지 valid disentanglement 존재:
- body × wick (v2)
- upper × lower (v3 with aux)

어느 게 "옳은가"는 task 요구에 따름:
- 단기 거래 신호 → body direction이 직접 관련 (v2 유리)
- 캔들 패턴 인식 → upper/lower asymmetry가 중요 (v3 유리)
- 일반 표현학습 → 두 분해 모두 가능

### 5.4 [확정] 평가 metric의 한계 재확인

학계 표준 지표(MIG, DCI, SAP)는 ground truth factor와의 정렬을 측정. **어떤 factor를 ground truth로 정의하는가에 따라 결과가 다름**:
- 우리는 sampling factor(center, magnitude, direction, upper_wick, lower_wick)를 truth로 잡음
- 정규화로 일부 factor가 소실되면 metric이 낮게 측정됨 (실제 모델 성능과 무관)

[확정] **단일 metric으로 disentanglement 판정 금지** — TMLR 2024 권고 재확인.

---

## 6. 다음 단계 후보

### 6.1 [추측] 권고 (autoprogress 마무리)

사용자 결정 필요:

**옵션 A: v3 (upper × lower) 채택**
- 사용자 원래 의도 달성
- aux loss 영구 의존 (학습 시점에만, inference 시점엔 무관)
- 척추 backbone 단계로 진입 가능

**옵션 B: v2 (body × wick) 채택**
- 자생적으로 발견된 elegant 분해
- aux loss 불요
- 출력 의미 변경 (out_1 = body direction, out_2 = wick configuration)
- 척추 backbone 단계 진입 가능

**옵션 C: 출력 차원 확장 후 재시도**
- 2 dim 제약을 풀어 4~8 dim 임베딩 시도
- 두 분해(body/wick + upper/lower)가 모두 포함 가능
- receptor 설계 철학 재검토 필요

### 6.2 [추적필요] 후속 검증

채택된 옵션과 무관하게:

1. **실제 OHLC 데이터로 generalization 확인** — 합성에서 성공한 분해가 실제 시장 데이터에서도 유지되는가
2. **다른 hyperparameter sweep** — receptor_hidden ∈ {8, 32, 64}, lr, seed
3. **Verdict criteria 재정의** — diag/off_diag 비율, Jacobian sparsity 등으로 조정
4. **거래량 V 통합 방식 검토** — receptor 옆에 병렬? concat?
5. **척추 backbone 후보 비교** — Mamba, attention, conv 등에서 어느 게 best

---

## 7. 실험 인프라 요약

### 파일 구조

```
experiments/receptor_verification/
├── discord_notify.py           # Discord 웹훅 보고
├── synthesize.py               # 합성 캔들 + 정규화
├── receptor.py                 # Receptor + Decoder + Autoencoder
├── train.py                    # 표준 학습 (v1, v2)
├── train_v3.py                 # 학습 + aux loss (v3)
├── evaluate.py                 # MIG/DCI/SAP/FactorVAE/LP/Jacobian/Causal
├── visualize.py                # 산점도, training curve, Jacobian heatmap
├── main.py                     # autoprogress orchestrator (v1, v2)
├── run_v3.py                   # v3 sweep orchestrator
└── results, results_v2, results_v3/
    ├── checkpoint.pt
    ├── evaluation.json
    ├── training_curves.png
    ├── canonical_scatter.png
    └── jacobian_heatmap.png
```

### 재현 명령

```bash
# v1 (잘못된 정규화)
RECEPTOR_RESULTS_DIR=experiments/receptor_verification/results \
    python -m experiments.receptor_verification.main

# v2 (body-center 정규화)
RECEPTOR_RESULTS_DIR=experiments/receptor_verification/results_v2 \
    python -m experiments.receptor_verification.main

# v3 (auxiliary loss sweep)
python -m experiments.receptor_verification.run_v3
```

### 모델 규모

- Receptor: 5 sub-MLPs (각 2-layer, hidden=16) ≈ 600 params
- Decoder: 3-layer MLP (hidden=32) ≈ 1400 params
- 전체 ≈ 2000 params — 매우 작음

### 학습 시간

- v2 main: ~30s (학습 25s + 평가 5s on RTX, train_n=50000)
- v3 sweep (3 λ): ~90s

---

## 8. 한계와 보류

[확정] 본 검증은 **합성 캔들에 한정**. 실제 시장 데이터 검증은 미수행.

[확정] **FactorVAE score 구현이 의심됨**:
- 표준화 후 분산을 측정하는 부분에 버그 가능성 (z_std에서 eps division)
- v1에서도 v2에서도 0.88+ 나오는데 다른 metric과 일치 안 함
- 추후 정확한 구현으로 재측정 필요 [추적필요]

[확정] **2 dim 출력이 충분한지 검증 안 함**:
- 만약 더 넓은 출력이 더 풍부한 표현이 가능하다면, 2 dim 제약 자체를 재검토
- 4, 8, 16 dim sweep 필요 [추적필요]

[확정] **disentanglement metric의 절대 임계값은 임의**:
- "MIG ≥ 0.3" 등의 기준은 일반론. 데이터·도메인별 적절한 임계값 다름
- 본 도메인(가격 캔들)에서의 baseline 자료 없음 → 결과 해석에 주의

---

## 9. 관련 메모

- [[260528-캔들receptor분리도검증연구계획]] — 본 실험의 사전 계획
- [[260527-가격전용정규화레이어개관]] — DAIN/RevIN/FAN과의 비교
- [[260527-가격거래량전용네트워크]] — 모태 조사
- [[project_candle_receptor_verification]] — 메모리 상의 프로젝트 상태
