# OHLCVReceptor

캔들 (HOCL) 과 거래량 (V) 을 받아 3 차원 임베딩 (upper-aspect, lower-aspect, volume-aspect) 을 출력하는
per-candle tokenizer. 시퀀스 안에서 파라미터 공유.

상세 설계 근거는 `Architecture - Price Receptor.md` 참조.

## OHLCVReceptor

`nn.Module` 상속. 5 개 입력 (H, O, C, L, V) → 3 개 출력 (out_1, out_2, out_v).
비대칭 routing 으로 H → upper 경로, L → lower 경로 분리. hoc_2 가 양쪽 출력에 element-wise add 되어
공통 baseline 으로 작동. 거래량은 별도 결합 경로 (out_v) 로 처리되며 y_1[0], y_2[0] 만 volume gradient 수신.

### 구조 상수 (클래스 상수, hyperparameter 아님)
SIDE_DIM = 2                    # hoc_2, y_1, y_2 의 차원
HIDDEN = 2                      # Linear_combU/combL 의 hidden 차원
HIDDEN_V = 4                    # Linear_combPV 의 hidden 차원
COMBV_INPUT_DIM = 5             # y_1[0], hoc_2, y_2[0], V

### __init__
__init__()
    인자 없음. 구조는 클래스 상수로 고정.
    내부에 6 개 명명 layer (linear_oc, linear_upper, linear_lower, linear_comb_u, linear_comb_l,
    linear_comb_pv) + LayerNorm (norm_pv) + 3 개 최종 projection (proj_u, proj_l, proj_v) 생성.
    총 파라미터 약 71 + LayerNorm affine 10.

### Methods

forward(hocl: torch.Tensor, v: torch.Tensor) -> torch.Tensor
    raise ValueError
    hocl: (..., 4) — 정규화된 HOCL. 마지막 차원 순서 [H, O, C, L].
    v: (..., 1) — log + 표준화된 거래량 scalar.
    반환: (..., 3) — [out_1, out_2, out_v]. 각각 upper-aspect, lower-aspect, volume-aspect.
    마지막 차원 검증 후 처리. 정규화는 외부 책임 (본 모듈 범위 외).

## 내부 흐름

1. `Linear_OC(O, C)` → (hoc_1, hoc_2, hoc_3). 선형층, 활성함수 없음.
2. `Linear_upper(H, hoc_1) + tanh` → y_1 ∈ ℝ^SIDE_DIM
3. `Linear_lower(L, hoc_3) + tanh` → y_2 ∈ ℝ^SIDE_DIM
4. `Linear_combU(y_1 + hoc_2) + LeakyReLU` → h_u → `proj_u` → out_1
5. `Linear_combL(y_2 + hoc_2) + LeakyReLU` → h_l → `proj_l` → out_2
6. `LayerNorm(y_1[0], hoc_2, y_2[0], V)` → `Linear_combPV + LeakyReLU` → h_v → `proj_v` → out_v

## 의도된 inductive bias

- **OHLC 부등식 비대칭 routing**: H 는 upper 경로, L 은 lower 경로에만 입력. 캔들 내부 구조 ($H \geq \max(O,C) \geq \min(O,C) \geq L$) 를 그래프 구조에 반영.
- **공통 baseline (hoc_2 addition)**: $hoc_2$ 가 element-wise add 로 양쪽에 동일 신호 전달. 학습 신호 없이도 구조 자체가 공통 의미를 강제.
- **거래량 분리·선택적 공유**: out_v 는 별도 결합 경로. y_1[0], y_2[0] 만 volume 신호 받음 → 자동 specialization (첫 노드는 volume-interactive, 둘째 노드는 pure price).
- **LayerNorm (norm_pv)**: V (큰 스케일) 와 receptor 처리값 (작은 스케일) 의 분포 차이 흡수.
  per-sample 정규화라 train/eval 동작 동일.

## 정규화 가정

receptor 는 정규화된 입력을 가정. 정규화는 외부에서 수행:
- HOCL 정규화 방향: `log` 변환 후 윈도우 마지막 close 차감 + 스케일 정규화 (구체 방식은 외부 결정)
- V 정규화: log + 표준화

상세는 `Architecture - Price Receptor.md` §5 참조.
