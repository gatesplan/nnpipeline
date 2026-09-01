# Architecture - Decay Bank

> 본 문서는 `Architecture.md` 의 보조 설계 문서.
> `DecayBank` (임베딩 시퀀스를 다중 시간스케일 지수 감쇠 상태로 누적하는 척추 모듈) 의 설계를 다룬다.
> 본 문서 내 구현 코드 작성 금지. 구조와 책임 경계까지만.
> 구현체: `nnpipeline.prototype.decay_bank.DecayBank`.

---

## 1. 모듈 목적

[확정] **DecayBank** 는 receptor 가 만든 임베딩 시퀀스 $e_1..e_n$ 을 받아,
학습 가능한 감쇠율 $\lambda_k$ 별 지수 누적 상태를 **시점 $n$ 기준**으로 요약하는 척추(backbone) 모듈.

$$h_k[t] = \lambda_k \, h_k[t-1] + (1-\lambda_k)\, e_t, \quad k = 1..K$$

[확정] 설계 동기 — 금융 시계열의 세 성질을 학습이 아닌 **구조**로 보장:
1. **recency bias**: 모든 과거 정보는 감쇠를 통과해야만 $n$ 에 도달. 최근일수록 강하게 남는다.
2. **모멘텀 감쇠 신호**: 빠른 스케일과 느린 스케일 상태의 차이 $h_{fast} - h_{slow}$ 가
   "이전 움직임이 약해지고 있는가" 를 직접 표현 (MACD 의 일반화형 — EMA 대상이 가격이 아닌 학습된 임베딩).
3. **다중 스케일 schema**: 반감기 스펙트럼이 큰 구조(느린 스케일)와 최근 디테일(빠른 스케일)을 분리 보존.

[확정] LSTM/일반 Transformer 를 쓰지 않는 이유: LSTM 은 망각이 불투명하고 스케일 분해가 없으며,
Transformer 는 위치를 학습된 임베딩으로 처리해 recency 가 구조적 보장이 아님.
본 모듈은 선형 재귀 (SSM 계열의 최소 형태) 로 위 성질들을 그래프에 새긴다.

---

## 2. 내부 구조

[확정] **감쇠율 파라미터화**: $\lambda_k = \sigma(\ell_k)$ (sigmoid logit $\ell_k$ 저장).
학습 중에도 $\lambda_k \in (0,1)$ 이 구조적으로 보장됨.
초기화는 반감기 지정: $\lambda_k = 2^{-1/h_k}$, 기본 $h = (2, 8, 32)$ 캔들.

[확정] **bias correction** (기본 on): 상태를 가중치 합 $1-\lambda_k^n$ 으로 나눠
창 길이·스케일 무관한 가중평균으로 정규화 (Adam bias correction 과 동일 원리).
이것이 없으면 짧은 창에서 느린 스케일 상태가 체계적으로 작아져,
fast-slow 차이가 모멘텀이 아닌 크기 왜곡을 반영하게 됨.

[확정] **인접 스케일 차이 출력** (`include_diffs`, 기본 on): $h_k - h_{k+1}$ ($k=1..K-1$) 를
스케일 축에 이어붙여 출력. downstream 선형층이 학습으로 발견할 수도 있는 값이지만,
구조에 직접 노출하여 모멘텀 감쇠 신호의 존재를 보장 (receptor 의 hoc_2 addition 과 같은 철학).

[확정] **계산 형태**: 시점 $n$ 상태만 필요하면 닫힌 형태
$h_k[n] = (1-\lambda_k)\sum_t \lambda_k^{n-1-t} e_t$ 를 einsum 으로 병렬 계산 (지수 ≥ 0, 수치 안정).
전체 궤적 (`return_sequence=True`) 은 시간 루프.

[확정] 파라미터 수 = $K$ ($\lambda$ logit 뿐). 표현력은 입력 임베딩과 downstream 이 담당,
본 모듈은 시간 누적 구조만 제공.

---

## 3. 외부 인터페이스

[확정] **입력**: `e` shape `(..., n, d)`. 시간축은 끝-1, 과거 → 최근 순. $d$ 임의
(OHLCVReceptor 출력 3, ReceptorBundle 출력 등 무관하게 동작).

[확정] **출력**: `(..., out_scales, d)`, `out_scales = 2K-1` (diffs 포함 시).
스케일 축 순서: $[h_1..h_K, h_1{-}h_2, .., h_{K-1}{-}h_K]$ (빠른 → 느린).
`return_sequence=True` 시 `(..., n, out_scales, d)`.

[확정] **책임 경계**:
- 임베딩 생성은 책임 외 (receptor / bundle 영역)
- 출력 해석·결합은 downstream 책임
- 정규화 가정 없음 — 선형 연산이라 입력 스케일 그대로 보존

---

## 4. 검증 결과 (260901, 합성 데이터)

[확정] 잠재 반감기 (4, 16, 64) AR 성분을 심은 합성 수익률로 120봉 → 1~5봉 누적수익률 예측
(`experiments/decay_bank_forecast/`):
- receptor → bank(K=4) → head 가 **관측 기반 feasible bound (Kalman) 의 99~100%** 도달.
  oracle (진짜 잠재 상태 인지) 대비 ~71% 인데, Kalman 도 71.1% — 남은 gap 전부가
  아키텍처 손실이 아닌 원리적 상태 불확실성. 필터링·receptor·head 손실 모두 ≈ 0.
- **λ 는 학습으로 확실히 이동** (초기값 무관하게 수렴), gradient 살아있음 → learnable 유지 가치 확인.
- 학습된 반감기는 진짜 τ 보다 짧게 앉음 — Kalman 정상해의 유효 감쇠 φ(1−K) < φ 와 정합.
- **K=8 은 K=4 대비 이득 없음** (둘 다 bound 포화). 여분 스케일은 기존 스케일 근처로
  군집(cluster)함 — collapse 는 병리가 아니라 "필요 스케일 수로의 적응적 축소"로 나타남.
  간격 정칙화 불필요 (이 세팅 기준). 군집 수 = 유효 K 를 λ 계기판 진단 항목으로 사용 가능.
- 과제 horizon 이 짧으면 (≤5봉) 가장 긴 학습 반감기가 ~15봉에서 멈춤 — 장기 구조 활용에는
  더 먼 horizon 또는 telescoping 필요.

[확정] 지저분함 사다리 검증 (260902, `experiments/decay_bank_ladder/`): 변동성 군집 (stochvol),
레짐 전환, fat tail (Student-t) + 점프를 누적해도 (초과첨도 최대 ~58) receptor → bank(K=4) → head 가
particle filter feasible bound 의 **96~100% 유지**. 선형 가우시안 밖에서도 포화가 거의 깨지지 않음 —
고정 λ 의 한계 (시변 최적 gain) 를 비선형 receptor + head 의 암묵적 재정규화가 보상하는 것으로 해석.
λ 계기판도 반응: 레짐 층 추가 시 느린 스케일 반감기가 ~30 → ~57 로 상승 (레짐 지속 시간 추론용 장기 기억).
남은 소폭 손실 (L3 fat tail 에서 -4~6%p) 은 이상치 대응 (robust 누적) 검토 근거.
receptor 를 LayerNorm 으로 교체 (260902, Price Receptor 문서 §7) 후 재실행: 학습이 완전 안정화된
상태에서 capture 94~100% 재확인 — BatchNorm 시절 결과가 checkpoint 선택의 산물이 아님을 확인.
L3 에서 느린 스케일 반감기가 ~210 (창 길이 초과) 으로 상승 — 이상치 억제용 근사-균등 평균으로 분화한 것으로 해석.

## 5. 미결

- **채널별 λ 분리** — 현재 λ 는 스케일당 스칼라 (전 채널 공유). 채널별 분리 시 표현력 vs 해석성 트레이드오프.
- **downstream 결합 방식** — (out_scales, d) 를 flatten 후 MLP 로 갈지, 스케일 축을 구조적으로
  다루는 비교층을 둘지. now-anchored 읽기 (제안 3) 와의 결합 지점.
- **망원 구간화 (제안 2) 와의 결합** — telescoping bundle 출력을 본 모듈 입력으로 쓸 때의 시간축 의미.
- **과잉 K + 사후 병합/prune** — 여분 스케일이 군집하는 성질을 이용해, 큰 K 로 학습 후 군집 병합 +
  downstream 기여 없는 스케일 제거로 유효 K 를 자동 결정하는 절차. group sparsity 병용 검토.
- **실데이터 검증** — 합성 (선형 가우시안) 에서는 bound 포화. 실데이터의 비선형·비정상 구조에서
  같은 결론이 유지되는지 미확인.

---

## 6. 참조

- 구조 검증 (미학습 분리도): `experiments/decay_bank_verification/`
- 학습 검증 (λ 거동·손실 분해): `experiments/decay_bank_forecast/`
- 모듈 문서: `nnpipeline/prototype/decay_bank/for-agent-moduleinfo.md`
- receptor 설계: `Architecture - Price Receptor.md`
