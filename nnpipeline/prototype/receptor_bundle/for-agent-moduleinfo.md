# ReceptorBundle

receptor 또는 receptor bundle 들을 자식 노드로 묶어 시간 위계 표현을 만드는 composite 컨테이너.
multi-resolution candle hierarchy 의 한 단계를 표현한다.

## ReceptorBundle

`nn.Module` 상속. 자식 리스트 + aggregator (MLP) 로 구성. 자식이 또 `ReceptorBundle` 일 수 있어
재귀 트리 구성 가능 (Composite pattern).

자식별로 시간 순서대로 입력을 분할 전달하고, 각 자식 출력 `(..., 3)` 을 시간순으로 stack 한 뒤
flatten 하여 aggregator 에 통과시킨다. 출력 차원은 입력 receptor 와 동일한 3 으로 고정.

### Properties
components: nn.ModuleList         # 자식 모듈 리스트 (시간 순서). nn.Module.children() 충돌 회피로 components 명명
aggregator: nn.Module             # (N*3,) → (3,) 매핑. 매 layer activation 은 aggregator 내부 책임
n_leaves: int                     # 이 bundle 이 포괄하는 leaf receptor 의 총 개수

### __init__
__init__(children: list, aggregator: nn.Module)
    raise TypeError
    raise ValueError
    children: nn.Module 인스턴스 리스트 (1 개 이상). 시간 순서대로 정렬되어 있다고 가정.
    원소가 자식 ReceptorBundle 이면 그 bundle 의 n_leaves 만큼 leaf 점유.
    원소가 일반 nn.Module (예: OHLCVReceptor) 이면 leaf 1 개로 간주.
    children 내 인스턴스는 모두 서로 다른 객체여야 함 (positional 분리 학습 보장).
    aggregator: nn.Module 인스턴스. 입력 `(..., len(children) * 3)` → 출력 `(..., 3)`. 사용자가 dim 정합성 보장.
    n_leaves 는 `sum(getattr(c, 'n_leaves', 1) for c in children)` 으로 계산되어 instance attribute 로 고정.

### Methods

forward(hocl: torch.Tensor, v: torch.Tensor) -> torch.Tensor
    raise ValueError
    hocl: (..., n_leaves, 4) — 정규화된 HOCL 시퀀스.
    v: (..., n_leaves, 1) — 정규화된 거래량 시퀀스.
    반환: (..., 3) — aggregator 출력. leading dims 보존.

    검증:
    - hocl 마지막 차원 4
    - v 마지막 차원 1
    - hocl.dim() >= 2
    - hocl.shape[-2] == n_leaves
    - v.shape[-2] == n_leaves

    분배 규칙:
    - 자식이 `n_leaves` 속성을 가지면 (= subtree) `(..., child.n_leaves, *)` 슬라이스 전달
    - 자식이 그렇지 않으면 (= leaf) 시퀀스 축에서 단일 indexing 으로 그 차원 제거 후 `(..., *)` 전달
    - 자식 호출 결과 `(..., 3)` 을 시간 순서대로 모아 `torch.stack(dim=-2)` → `flatten(start_dim=-2)` →
      aggregator

## 의도된 설계

- **Composite pattern**: leaf (receptor) 와 internal node (bundle) 를 동일 인터페이스 `(hocl, v) -> (..., 3)`
  로 다룬다. 재귀 트리로 multi-resolution hierarchy 구성.
- **Positional (non-shared) weight**: 자식 인스턴스 중복 금지로 강제. 시간 위치별 분포 차이
  (예: 일중 거래량 U 자 패턴) 를 모델이 표현할 수 있도록.
- **시간 분할 = 자식 구성**: bundle 자체는 균등 분할을 강제하지 않는다. 자식의 `n_leaves` 합이 부모의
  `n_leaves` 가 되므로 비균등 묶음 (예: `(5, 10, 30, 10, 5)`) 자연 표현.
- **Channel mix (= full mix)**: aggregator 가 `N*3 → 3` 매핑으로 자식 출력의 모든 feature 를 섞는다.
  depthwise 분리는 도입하지 않음. 출력 dim 은 receptor 와 동일한 3 으로 고정.
- **Aggregator DI**: aggregator 를 외부 주입으로 받음. 단순 `nn.Linear` 또는 `Pyramid` (interlayer 에
  activation 삽입) 모두 가능. 매 layer activation 은 aggregator 내부 책임.
- **표현력 조절 = 압축 깊이**: 출력 dim 을 키우는 대신 hierarchy 의 어디서 멈출지로 표현력을 조절한다.
  많은 출력 노드가 필요하면 4h 까지 압축하지 말고 15m 등에서 멈춰 `48 * 3` 노드 형태로 사용.

## 사용 예

```python
# 1m → 5m → 15m → 1h, 총 60 leaves
five_m_bundles_per_15m = []
for _ in range(4):
    five_m_bundles = []
    for _ in range(3):
        receptors = [OHLCVReceptor() for _ in range(5)]
        agg = Pyramid(15, 3, depth=2, interlayer=[nn.LeakyReLU()])
        five_m_bundles.append(ReceptorBundle(children=receptors, aggregator=agg))
    fifteen = ReceptorBundle(
        children=five_m_bundles,
        aggregator=Pyramid(9, 3, depth=2, interlayer=[nn.LeakyReLU()]),
    )
    five_m_bundles_per_15m.append(fifteen)

one_h = ReceptorBundle(
    children=five_m_bundles_per_15m,
    aggregator=Pyramid(12, 3, depth=2, interlayer=[nn.LeakyReLU()]),
)
# one_h.n_leaves == 60
# one_h(hocl=(B,60,4), v=(B,60,1)) → (B, 3)
```
