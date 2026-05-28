# 캔들 Receptor 분리도 검증 연구 계획

- 작성일: 2026-05-28
- 모태 설계 논의: 본 노트 작성 시점 대화 (사용자–에이전트 receptor 구조 협의)
- 관련 메모: [[260527-가격전용정규화레이어개관]], [[260527-FAN]], [[260527-RevIN]], [[260527-가격거래량전용네트워크]]
- 목적: 사용자가 설계한 candle receptor의 2-dim 출력이 의도된 upper/lower 분리를 달성하는지를 **학술적으로 인정되는 방법**으로 검증하는 절차 정의

---

## 0. 배경

[확정] 본 프로젝트는 가격(OHLCV) 전용 네트워크의 **receptor**(입력층 + 초기 처리 layer)를 새로 설계하는 것을 목표로 함. 모태 조사([[260527-가격거래량전용네트워크]], [[260527-가격전용정규화레이어개관]])에서 확인된 바, 기존 SOTA(DAIN, RevIN, FAN)는 모두 일반 시계열 정규화이며 캔들의 구조적 제약을 architecture에 명시적으로 박지 않음. 사용자는 OHLCV의 V는 단순 log-표준화로 충분하다고 보고, HLOC에 집중한 candle-aware receptor를 제안.

[확정] 제안된 receptor 구조의 핵심 가설: **각 캔들의 정보를 (upper-aspect, lower-aspect) 2-dim 임베딩으로 분해 가능하다**. 이 분해가 학습으로 자생적으로 달성되는지가 본 검증의 대상.

[확정] 분리가 의도대로 일어나지 않으면 이 receptor 설계의 의미가 사라지므로, 검증은 다음 단계(척추 backbone 결합)로 가기 전 필수.

---

## 1. 연구 가설

**H1 (주가설)**: 제안된 구조적 receptor를 autoencoder loss로 학습시키면, 출력 두 차원 $out_1, out_2$이 각각 캔들의 upper-aspect와 lower-aspect 정보를 분리해서 표현한다.

- $out_1$은 $H$, $\max(O,C)$, upper wick 길이 같은 upper 관련 factor와 강한 상관을 가짐
- $out_2$는 $L$, $\min(O,C)$, lower wick 길이 같은 lower 관련 factor와 강한 상관을 가짐
- 두 차원 간 cross-leak (예: $out_1$이 lower wick과 상관)이 작음

**H0 (귀무가설)**: 학습 후에도 두 차원이 upper/lower로 분리되지 않거나, 두 차원이 모두 비슷한 entangled 정보를 담거나, 의도와 다른 분해가 일어남.

[추적필요] 가설 검증을 통해 분리 성공 여부 결정 → 다음 단계 진행 여부 결정.

---

## 2. Receptor 아키텍처 정의

### 2.1 구조

```
입력층 (4):   H,  O,  C,  L     ∈ ℝ
                  └─┬──┘
                    ▼
              MLP_OC: ℝ² → ℝ³
                    │
                    ▼
           hoc_1, hoc_2, hoc_3   ∈ ℝ

y_1 = MLP_upper(H, hoc_1) ∈ ℝ
y_2 = MLP_lower(L, hoc_3) ∈ ℝ

out_1 = MLP_combU(y_1, hoc_2) ∈ ℝ      ← upper-aspect
out_2 = MLP_combL(hoc_2, y_2) ∈ ℝ      ← lower-aspect

출력 (2): (out_1, out_2)
```

### 2.2 학습 가능 모듈

| 모듈 | 입력 차원 | 출력 차원 | 비고 |
|---|---|---|---|
| MLP_OC | 2 | 3 | hidden: $[2, h, 3]$ 또는 $[2, h, h, 3]$ |
| MLP_upper | 2 | 1 | $H$와 $hoc_1$ 결합 |
| MLP_lower | 2 | 1 | $L$과 $hoc_3$ 결합 |
| MLP_combU | 2 | 1 | $y_1$과 $hoc_2$ 결합 |
| MLP_combL | 2 | 1 | $hoc_2$과 $y_2$ 결합 |

### 2.3 비대칭 routing의 정당성

[확정] $y_1$이 $H$와만 직접 만나고, $y_2$가 $L$과만 직접 만남. $H \geq \max(O,C)$, $L \leq \min(O,C)$이라는 데이터 통계 구조 때문에, 학습이 자연스럽게 $hoc_1, hoc_3$의 역할 분화를 유도할 가능성 있음.

[확정] 최종 출력 $out_1, out_2$의 분리 routing ($out_1$은 $y_1, hoc_2$만, $out_2$는 $hoc_2, y_2$만) 때문에 $hoc_2$는 양쪽 모두에 유용해야 함 → **body 정보로 수렴 압력**.

### 2.4 하이퍼파라미터 (초기값)

- 모든 MLP의 hidden size $h$: 16 (작게 시작, 필요 시 증가)
- Activation: ReLU (단순함 우선, GELU/SiLU도 검토 대상)
- Dropout: 없음 (검증 단계라 단순화)
- 학습률: Adam $10^{-3}$
- batch size: 256
- epoch: 100 (early stopping 없음, 메모리 feedback 따라)

---

## 3. 학습 신호: Autoencoder

### 3.1 왜 Autoencoder인가

[확정] 학술 표준 (Self-Supervised Learning for Time Series, arXiv 2306.10125; Target-Embedding Autoencoders, arXiv 2001.08345)에 따라, **representation 평가가 목적이면 task-agnostic 학습이 적합**. forecasting 같은 task-specific 학습은 task에 유리한 편향을 representation에 박음 → "이 representation은 forecasting에 좋다" 외엔 결론 어려움.

[확정] Autoencoder는:
- task-agnostic → receptor 본연의 표현력만 평가
- "원본을 복원 가능한 정보가 잠재 공간에 들어있는가"라는 question과 직접 매칭
- decoder가 단순해서 receptor 자체 평가 가능
- 합성 데이터의 ground truth factor와 직접 비교 용이

### 3.2 Decoder 구조

receptor 출력 $(out_1, out_2)$로부터 원본 HOCL 복원:

```
Decoder: ℝ² → ℝ⁴
hidden: [2, h_dec, h_dec, 4]
```

$h_{\text{dec}}$: 16~32 (receptor와 비슷한 규모)

### 3.3 Loss

**기본 MSE reconstruction loss**:

$$\mathcal{L}_{\text{recon}} = \frac{1}{N}\sum_{i=1}^{N} \bigl\| (H,O,C,L)_i - (\hat{H},\hat{O},\hat{C},\hat{L})_i \bigr\|_2^2$$

[추측] **선택적 추가 항** (옵션, 필요 시):
- OHLC 제약 위반 penalty: $\max(0, \max(O,C) - H) + \max(0, L - \min(O,C))$
- 단 합성 데이터에서는 입력이 항상 제약 만족이므로 출력만 자유. decoder가 위반 출력을 내는지 모니터링.

[추적필요] 제약 항이 분리도에 영향을 주는지 ablation 필요.

### 3.4 학습 관련 [확정 권고]

- early stopping **OFF** ([[feedback_no_aggressive_early_stop]])
- MAX_EPOCHS=100까지 강제 진행
- validation loss는 monitoring만, 학습 종료 트리거 아님

---

## 4. 합성 캔들 데이터

### 4.1 Factors of variation 정의

[확정] Disentanglement 평가는 ground truth factor가 정의되어야 가능. 합성 데이터에서 다음 factor를 **독립적으로 sampling**:

| Factor | 기호 | 분포 | 범위 |
|---|---|---|---|
| Body center | $c$ | $\mathcal{U}(0, 100)$ | 가격 레벨, 절대값은 큰 의미 없음 (정규화 가정) |
| Body magnitude | $m$ | $\mathcal{U}(0, 5)$ | $|C - O|$ |
| Body direction | $d$ | $\text{Bernoulli}(0.5)$ | $+1$이면 bull ($C > O$), $-1$이면 bear |
| Upper wick length | $u$ | $\mathcal{U}(0, 5)$ | $H - \max(O, C)$ |
| Lower wick length | $l$ | $\mathcal{U}(0, 5)$ | $\min(O, C) - L$ |

### 4.2 HOCL 합성 공식

위 factor로부터 결정론적으로:

$$O = c - d \cdot m/2, \quad C = c + d \cdot m/2$$
$$H = \max(O, C) + u, \quad L = \min(O, C) - l$$

[확정] 5개 factor가 4개 출력 HOCL에 매핑. 자유도 일치 (한 factor는 잉여처럼 보이지만 body direction이 discrete라 실제로는 5 factor가 4 continuous + 1 discrete = 4 + 1 = 5 dof). 사실 $c, m, d, u, l$이 정확히 HOCL을 결정 → 5 factor가 정확히 4-dim HOCL을 1대1 매핑하는 reparameterization.

### 4.3 Train/Test split

- **Train**: 50,000 캔들, 위 분포에서 독립 sampling
- **Validation**: 10,000 캔들, 동일 분포
- **Test (random)**: 10,000 캔들, 동일 분포
- **Test (canonical)**: 별도 — 캐노니컬 캔들 패턴 8종 × 100 = 800 캔들

### 4.4 Canonical patterns (정성 검증용)

| 종류 | 정의 | 의미 |
|---|---|---|
| Marubozu Bull | $u=0, l=0, d=+1, m=4$ | 완전 bullish, wick 없음 |
| Marubozu Bear | $u=0, l=0, d=-1, m=4$ | 완전 bearish, wick 없음 |
| Hammer | $u=0, l=4, d \in \{\pm 1\}, m=1$ | 긴 lower wick (지지) |
| Inverted Hammer | $u=4, l=0, d \in \{\pm 1\}, m=1$ | 긴 upper wick (저항) |
| Shooting Star | $u=4, l=0, d=-1, m=1$ | bearish reversal |
| Doji | $m \approx 0, u \approx l \approx 1$ | 결정 못함 |
| Long-Legged Doji | $m \approx 0, u=3, l=3$ | 양쪽 압력 모두 강함 |
| Spinning Top | $m=1, u=2, l=2$ | 양쪽 wick 균형 |

[확정] 각 패턴 100개 × body center $c \in \mathcal{U}(0, 100)$ 무작위 → canonical은 형태가 고정, 위치만 변동.

### 4.5 입력 정규화

[확정] HOCL을 receptor에 넣기 전 instance-wise 정규화. body center $c$가 receptor 학습에 들어가지 않게:

$$(\tilde{H}, \tilde{O}, \tilde{C}, \tilde{L}) = \frac{(H, O, C, L) - c}{s}$$

여기서 $s$는 scale (예: instance별 $\max - \min$ 또는 fixed). 절대 가격 레벨 정보는 제거. **이 정규화 정의도 receptor 설계의 일부**.

[추적필요] 정규화 방식 선택이 분리도에 영향. 후속 ablation 필요.

---

## 5. 평가 방법론

학습된 receptor에 대해 다음 다중 지표를 모두 적용. 단일 지표 신뢰 금지 (TMLR 2024 "Correcting Flaws in Common Disentanglement Metrics" 권고).

### 5.1 표준 disentanglement 지표

#### MIG (Mutual Information Gap)

각 factor $z_k$에 대해, 잠재 차원과의 mutual information을 모두 계산. 가장 높은 MI를 가진 차원의 값에서 두 번째 높은 차원의 값을 뺀 정규화 차이:

$$\text{MIG}_k = \frac{1}{H(z_k)}\bigl[I(z_k; out_{(1)}) - I(z_k; out_{(2)})\bigr]$$

전체 MIG는 $\frac{1}{K}\sum_k \text{MIG}_k$. 1에 가까우면 완벽 분리.

[확정] 출처: Chen et al. 2018, β-TCVAE (NeurIPS 2018).

#### DCI (Disentanglement, Completeness, Informativeness)

잠재 → factor regression의 importance matrix $R \in \mathbb{R}^{D_{\text{latent}} \times D_{\text{factor}}}$ 산출 후 3개 지표:

- **Disentanglement**: 각 잠재 차원이 한 factor에 얼마나 집중하는가
- **Completeness**: 각 factor가 한 잠재 차원에 얼마나 집중하는가
- **Informativeness**: regression 자체의 정확도 ($R^2$)

[확정] 출처: Eastwood & Williams 2018, ICLR.

#### Linear probing

각 factor를 잠재 차원으로부터 linear regression. $R^2$ 기록:

| | $\to c$ | $\to m$ | $\to d$ | $\to u$ | $\to l$ |
|---|---|---|---|---|---|
| $out_1$ | $R^2_{...}$ | $R^2_{...}$ | $R^2_{...}$ | **$R^2_{...}$** | $R^2_{...}$ |
| $out_2$ | $R^2_{...}$ | $R^2_{...}$ | $R^2_{...}$ | $R^2_{...}$ | **$R^2_{...}$** |

분리 성공 시 표의 대각 영역에서 $R^2$가 크고 그 외에서 작음. 특히 $\text{corr}(out_1, l) \approx 0$, $\text{corr}(out_2, u) \approx 0$이어야 함.

#### FactorVAE score

각 factor에 대해 다음 절차:
1. 해당 factor를 한 값으로 고정한 미니배치 sampling
2. 각 잠재 차원의 분산 측정
3. 분산이 가장 작은 차원이 "이 factor를 인코딩한 차원"으로 추정
4. 분류기를 학습해서 (factor, 추정 차원) 쌍의 정확도 측정

[확정] 출처: Kim & Mnih 2018, ICML.

#### SAP (Separated Attribute Predictability)

각 factor에 대해 모든 잠재 차원의 예측 정확도(분류) 또는 R²(회귀) 계산. Top-2 차원의 차이가 클수록 분리.

[확정] 출처: Kumar et al. 2018, ICLR.

### 5.2 도메인 특화 지표

#### Jacobian analysis

학습 후 임의 캔들에서 partial derivative:

$$J_{ij} = \frac{\partial \, out_i}{\partial \, x_j}, \quad i \in \{1, 2\}, j \in \{H, O, C, L\}$$

전체 test set 평균. 분리 성공이면:

| | $\partial / \partial H$ | $\partial / \partial O$ | $\partial / \partial C$ | $\partial / \partial L$ |
|---|---|---|---|---|
| $out_1$ | **클 것** | 중 | 중 | **작을 것** |
| $out_2$ | **작을 것** | 중 | 중 | **클 것** |

PyTorch `torch.autograd.grad`로 직접 계산.

#### Causal intervention

테스트 캔들 baseline에서 $H$만 $\Delta$ 증가 (제약 유지). $\Delta out_1, \Delta out_2$ 측정. $|\Delta out_1| \gg |\Delta out_2|$이면 $out_1$이 upper 정보 보유.

마찬가지로 $L$ 조작.

#### 분리도 정량 지표

$$\text{SeparationScore} = \frac{|J_{1,H}| - |J_{1,L}|}{|J_{1,H}| + |J_{1,L}| + \epsilon} + \frac{|J_{2,L}| - |J_{2,H}|}{|J_{2,L}| + |J_{2,H}| + \epsilon}$$

값 범위 $[-2, 2]$, 2에 가까우면 완전 분리.

### 5.3 정성적 검증

#### Canonical pattern 산점도

학습된 receptor에 canonical 800 캔들 통과 → $(out_1, out_2)$ 2D plot. 각 점을 패턴 종류로 색칠.

기대 패턴:
- Marubozu Bull, Inverted Hammer 등 upper-dominant: $out_1$ 큰 쪽
- Marubozu Bear, Hammer 등 lower-dominant: $out_2$ 큰 쪽
- Doji 등 balanced: 원점 근처
- Long-Legged Doji: 양쪽 모두 큰 quadrant

산점도가 **의미 있는 클러스터링**을 보이면 분리 정성 검증 통과.

#### Reconstruction quality

$out_1$만 사용 vs $out_2$만 사용해서 decoder 통과 → 어떤 정보가 복원되는지 확인:
- $out_1$ 단독 → $H$, $\max(O,C)$ 잘 복원, $L$ 못 복원이면 분리 성공
- 양쪽 다 비슷하게 복원하면 entangled

---

## 6. 결정 기준

### 6.1 분리 성공 판정

다음 4개 조건 중 3개 이상 만족 시 **분리 성공**:

1. **MIG**: $\geq 0.3$ (factor당 평균)
2. **DCI Disentanglement**: $\geq 0.5$
3. **Linear probing 비대각/대각 비율**: $\frac{\max(\text{off-diag})}{\min(\text{diag})} \leq 0.3$
4. **SeparationScore**: $\geq 1.0$

[추측] 임계값들은 본 연구의 초기 가설값. 실제 결과 보고 조정 가능. 다만 사전에 정의해두지 않으면 결과에 맞춰 임계값 조정하는 cherry-picking 위험.

### 6.2 분리 실패 판정

다음 중 어느 하나라도 발생:
1. MIG $< 0.1$
2. Linear probing 대각 $R^2$가 비대각 $R^2$과 비슷 (차이 $< 0.2$)
3. Canonical pattern 산점도에서 클러스터 분리 안 됨
4. Jacobian이 비대칭이 명확하지 않음

### 6.3 모호한 경우

위 두 기준 사이 → 추가 분석 필요:
- 학습률, hidden size 등 하이퍼파라미터 sweep
- Auxiliary loss 추가 시도 (다음 절)
- 학습 시간 연장
- 다른 random seed로 재현성 확인

---

## 7. 실패 시 대응 (fallback)

[확정] 분리 실패 시 단계적 대응:

### 7.1 약한 보조 손실 추가 (auxiliary loss)

$$\mathcal{L}_{\text{aux}} = \lambda \cdot \bigl[(out_1 - u_{\text{target}})^2 + (out_2 - l_{\text{target}})^2\bigr]$$

여기서 $u_{\text{target}}, l_{\text{target}}$은 합성 데이터의 ground truth upper/lower factor (또는 derived feature). $\lambda$는 작게 시작 ($10^{-3}$~$10^{-2}$).

### 7.2 구조적 inductive bias 강화

```
hoc_1 ← MLP_1(O, C) + α · max(O, C)
hoc_2 ← MLP_2(O, C) + β · (O+C)/2
hoc_3 ← MLP_3(O, C) + α · min(O, C)
```

$\alpha, \beta$는 학습 가능 scalar (초기값 1).

### 7.3 MLP 분리 (옵션 D)

OC → 단일 MLP → 3 출력 대신, OC → 3개 독립 MLP:
- $hoc_1 = \text{MLP}_{\text{upper}}(O, C)$
- $hoc_2 = \text{MLP}_{\text{body}}(O, C)$
- $hoc_3 = \text{MLP}_{\text{lower}}(O, C)$

파라미터 3배.

### 7.4 출력 차원 확장

2 dim이 캔들 정보를 담기에 너무 좁다면, 사용자 의도(2-dim 의미 분리)를 포기하고 더 넓은 임베딩으로 전환:
- 4 dim, 8 dim, 16 dim, 32 dim 비교
- 단 이 경우 receptor의 design philosophy가 변경됨 → 별도 설계 결정

### 7.5 설계 폐기

위 모든 대응에도 분리 실패 → receptor 구조 자체가 데이터에 맞지 않음. 다른 아키텍처로 전환 필요. [[260527-가격전용정규화레이어개관]]의 다른 후보들 (Pseudo-PCA reparameterization, Residual Factor Network 등) 재검토.

---

## 8. 구현 계획

### 8.1 Phase 1: 인프라

| 항목 | 산출물 |
|---|---|
| 합성 캔들 generator | `nnpipeline/synth/candle_factory.py` (또는 `experiments/`) |
| Receptor architecture | `nnpipeline/prototype/candle_receptor.py` |
| Decoder | 동일 파일 |
| Autoencoder wrapper | 동일 파일 |
| 학습 loop | `experiments/receptor_verification/train.py` |
| 평가 모듈 (MIG/DCI/probing/Jacobian) | `experiments/receptor_verification/eval.py` |
| 시각화 | `experiments/receptor_verification/visualize.py` 또는 notebook |

### 8.2 Phase 2: 학습

- MAX_EPOCHS=100, early stopping OFF
- 학습률 $10^{-3}$ baseline, $10^{-4}$, $10^{-2}$ sweep
- Hidden size $h \in \{8, 16, 32\}$ sweep
- Random seed $\{1, 2, 3\}$로 재현성 확인

### 8.3 Phase 3: 평가

- Test set과 canonical set 모두에서 5개 표준 지표 + 도메인 특화 지표 계산
- 결과를 표·시각화로 정리
- 결정 기준에 따라 성공/실패/모호 판정

### 8.4 Phase 4: 보고

- 분리 성공 시: 다음 단계 (척추 + backbone) 설계
- 실패 시: fallback 옵션 중 선택 후 재실험
- 모호 시: 추가 분석

---

## 9. 한계와 주의

### 9.1 평가 지표의 알려진 한계

[확정] TMLR 2024 "Correcting Flaws in Common Disentanglement Metrics":
- MIG, SAP 등은 regularization 강도에 따라 prematurely peak하고 degrade
- 단일 지표 결과 단정 금지

[확정] arXiv 1911.11791 "A Preliminary Study of Disentanglement With Insights on the Inadequacy of Metrics":
- 동일 표현에 대해 지표 간 결과 자주 불일치
- "분리"의 통일된 정의가 학계에 없음

→ 본 연구도 다중 지표 cross-check 채택.

### 9.2 합성 vs 실제

[확정] 본 검증은 **합성 캔들에 한정**. 실제 시장 데이터에서 동일 분리도가 유지되는지는 별도 검증 필요 [추적필요].

합성 데이터의 한계:
- factor가 독립 sampling — 실제 캔들은 factor 간 상관 (예: body 큰 캔들이 wick도 길다 등)
- 노이즈 분포 — 실제는 fat-tailed, 합성은 uniform
- 캔들 패턴의 빈도 — 실제는 도지가 매우 빈번, 합성은 균일

### 9.3 학습 신호의 task-specificity

[확정] Autoencoder로 학습한 receptor가 forecasting에서도 동일 분리도를 유지하는 보장 없음. 학습 신호를 forecasting으로 바꾸면 다른 분리가 일어날 가능성.

[추적필요] Phase 4 이후 forecasting task로도 재학습해서 비교 ablation.

### 9.4 분리의 정의 모호성

"upper-aspect"의 정확한 정의가 학술적으로 합의된 것 아님. 본 연구에서는:
- upper wick 길이 ($H - \max(O,C)$)
- $H$ 절대값
- bullish strength (body direction이 + 일 때 $C - O$)

이런 derived feature와의 상관을 "upper-aspect 인코딩"의 정의로 사용.

→ 사용자가 다른 의미를 의도하면 평가 대상 factor 재정의 필요.

---

## 10. 참고문헌

### 표준 disentanglement 지표
- Chen, R. T. Q., et al. (2018). Isolating Sources of Disentanglement in Variational Autoencoders. NeurIPS. (β-TCVAE, MIG)
- Eastwood, C., & Williams, C. K. I. (2018). A Framework for the Quantitative Evaluation of Disentangled Representations. ICLR. (DCI)
- Kim, H., & Mnih, A. (2018). Disentangling by Factorising. ICML. (FactorVAE)
- Kumar, A., et al. (2018). Variational Inference of Disentangled Latent Concepts from Unlabeled Observations. ICLR. (SAP)

### 한계와 비판
- "Correcting Flaws in Common Disentanglement Metrics" (TMLR 2024): https://openreview.net/pdf?id=c8WJ4Vozb2
- "A Preliminary Study of Disentanglement With Insights on the Inadequacy of Metrics": https://arxiv.org/pdf/1911.11791

### 시계열 representation learning
- "Self-Supervised Learning for Time Series Analysis" (2023 survey): https://arxiv.org/pdf/2306.10125
- "Target-Embedding Autoencoders for Supervised Representation Learning": https://arxiv.org/pdf/2001.08345

### 모태 조사
- [[260527-가격거래량전용네트워크]]
- [[260527-가격전용정규화레이어개관]]
- [[260527-FAN]]
- [[260527-RevIN]]
- [[260527-DAIN]]

---

## 11. Open questions ([추적필요])

1. 입력 정규화 방식 (instance-wise $(x-c)/s$의 $s$ 정의)이 분리도에 미치는 영향
2. OHLC 제약 위반 penalty 항이 분리에 도움 되는지
3. 학습 신호를 autoencoder에서 forecasting으로 바꾸면 분리도가 어떻게 변하는지
4. 합성 데이터에서 분리 성공 시 실제 시장 데이터로 generalization 여부
5. 출력 차원 2 vs 더 큰 차원의 trade-off (사용자 의도 vs 표현력)
6. 학습 후 분리가 일어났다 해도, $hoc_1$과 $hoc_3$의 역할이 사용자 의도와 일치하는지 (내부 노드 검증)
