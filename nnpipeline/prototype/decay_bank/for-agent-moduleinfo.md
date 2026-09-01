# DecayBank

임베딩 시퀀스를 받아 다중 시간스케일 지수 감쇠 상태 (learnable EMA bank) 를 시점 n 기준으로 누적하는 척추 모듈.
recency bias 와 모멘텀 감쇠 신호 (fast - slow 차이) 를 학습이 아닌 구조로 보장한다.

상세 설계 근거는 `Architecture - Decay Bank.md` 참조.

## DecayBank

`nn.Module` 상속. 입력 시퀀스 (..., n, d) → 시점 n 상태 (..., out_scales, d).
각 스케일 k 마다 h_k[t] = λ_k·h_k[t-1] + (1-λ_k)·e_t 재귀. λ_k 는 sigmoid logit 으로
저장되어 학습 중에도 (0, 1) 구조적 보장. include_diffs=True 면 인접 스케일 차이
(fast - slow) 를 출력 스케일 축에 이어붙임 — 모멘텀 감쇠의 직접 표현.

### Properties
n_scales: int                   # 감쇠 스케일 개수 K
out_scales: int                 # 출력 스케일 축 크기. K (+ K-1 if include_diffs)
lambdas: torch.Tensor           # 현재 유효 감쇠율 λ_k ∈ (0,1). shape (K,)
half_lives: torch.Tensor        # 현재 유효 반감기 (학습으로 변동 가능). shape (K,)
include_diffs: bool
bias_correction: bool

### __init__
__init__(half_lives: tuple = (2.0, 8.0, 32.0), learnable: bool = True,
         include_diffs: bool = True, bias_correction: bool = True)
    raise TypeError, ValueError
    half_lives: 양수 순증가 시퀀스 (빠른 스케일 → 느린 스케일). λ_k = 2^(-1/h_k) 로 초기화.
    learnable=False 면 λ logit 을 buffer 로 등록 (고정).
    include_diffs=True 는 스케일 2 개 이상 필요.
    bias_correction=True 면 가중치 합 (1-λ^n) 으로 나눠 창 길이·스케일 무관 가중평균화
    (Adam bias correction 과 동일 원리 — 느린 스케일이 짧은 창에서 체계적으로 작아지는 왜곡 제거).
    파라미터 수 = K (λ logit 뿐).

### Methods

forward(e: torch.Tensor, return_sequence: bool = False) -> torch.Tensor
    raise ValueError
    e: (..., n, d) — 임베딩 시퀀스. 시간축 끝-1, 과거 → 최근 순. d 임의 (receptor 출력 3 등).
    반환: (..., out_scales, d) — 시점 n 상태. 스케일 축 순서: [h_1..h_K, h_1-h_2, ..., h_{K-1}-h_K].
    return_sequence=True: (..., n, out_scales, d) — 전체 시점 궤적 (검증·시각화·중간 supervision 용).
    final 모드는 닫힌 형태 einsum (병렬), sequence 모드는 시간 루프.

## 의도된 inductive bias

- **구조적 recency bias**: 모든 과거 정보는 감쇠를 통과해서만 시점 n 에 도달. 최근일수록 강하게
  남는 것이 학습 대상이 아니라 구조적 보장.
- **모멘텀 감쇠의 직접 표현**: h_fast - h_slow 가 "이전 움직임이 약해지고 있는가" 신호.
  MACD 의 일반화형 (EMA 대상이 가격이 아닌 학습된 임베딩).
- **다중 스케일 schema**: 반감기 스펙트럼이 큰 구조 (느린 스케일) 와 최근 디테일 (빠른 스케일) 을
  분리 보존.

## 책임 경계

- 입력 임베딩 생성은 책임 외 (receptor / bundle 의 영역). d 에 무관하게 동작.
- 출력 (..., out_scales, d) 의 해석·결합은 downstream 책임.
- 정규화된 입력 가정 없음 — 선형 연산이므로 입력 스케일 그대로 보존.
