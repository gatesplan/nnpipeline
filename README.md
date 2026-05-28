# Neural Network Pipeline

자주 쓰는 1D MLP 형상 프리셋 + 가격 시계열 도메인 receptor / receptor bundle 컨테이너 PyTorch 모음.

각 MLP 프리셋은 `nn.Sequential` 의 서브클래스라 만들자마자 torch 네트워크 객체로 쓸 수 있다.
활성함수/정규화/드롭아웃 같이 층 사이에 끼우는 모듈은 사용자가 직접 인스턴스로 넘기고, 라이브러리는 그것들을 deepcopy 해서 사이에 끼워 넣는다. 라이브러리가 활성함수 종류나 정규화 종류를 알 필요가 없다.

## 설치

```bash
pip install fish-nnpipeline
```

## MLP 형상

### Cylinder

동일 폭의 Linear 층을 `depth`개 쌓는다.

```python
from torch import nn
from nnpipeline import Cylinder

net = Cylinder(
    in_features=64,
    depth=3,
    interlayer=[nn.BatchNorm1d(64), nn.ReLU()],
)
```

```
Cylinder(
  (0): Linear(in_features=64, out_features=64)
  (1): BatchNorm1d(64, ...)
  (2): ReLU()
  (3): Linear(in_features=64, out_features=64)
  (4): BatchNorm1d(64, ...)
  (5): ReLU()
  (6): Linear(in_features=64, out_features=64)
)
```

### Pyramid

폭이 `in_features`에서 `out_features`로 선형 보간되며 Linear 층 `depth`개를 쌓는다.

```python
from torch import nn
from nnpipeline import Pyramid

net = Pyramid(
    in_features=100,
    out_features=10,
    depth=4,
    interlayer=[nn.ReLU()],
    pipe_head=[nn.LayerNorm(100)],
    pipe_end=[nn.Tanh()],
)
```

```
Pyramid(
  (0): LayerNorm((100,))
  (1): Linear(in_features=100, out_features=78)
  (2): ReLU()
  (3): Linear(in_features=78, out_features=55)
  (4): ReLU()
  (5): Linear(in_features=55, out_features=32)
  (6): ReLU()
  (7): Linear(in_features=32, out_features=10)
  (8): Tanh()
)
```

## MLP 파라미터 규칙

`Cylinder`, `Pyramid` 가 공통으로 받는 모듈 리스트 파라미터:

| 파라미터 | 위치 | 설명 |
|---|---|---|
| `pipe_head` | 맨 앞 | 첫 Linear 전에 한 번 |
| `interlayer` | Linear 사이 | 마지막 Linear 뒤에는 붙지 않음 |
| `pipe_end` | 맨 뒤 | 마지막 Linear 뒤에 한 번 |

- 기본값은 모두 빈 리스트.
- 안에 들어가는 모듈은 `deepcopy` 되어 들어가므로, 동일 인스턴스를 넘겨도 사이사이에 독립 모듈이 생성된다.
- `depth >= 1` 이어야 한다.

## Receptor

### OHLCVReceptor

한 캔들 (HOCL) + 거래량 (V) 을 받아 3-dim 임베딩 `[out_1, out_2, out_v]` 로 변환하는 per-candle tokenizer.
시퀀스 안에서 파라미터 공유. 비대칭 routing 으로 H → upper 경로, L → lower 경로 분리, V 는 별도 결합 경로로 통합.

```python
import torch
from nnpipeline import OHLCVReceptor

receptor = OHLCVReceptor()
hocl = torch.randn(32, 4)  # 정규화된 [H, O, C, L]
v = torch.randn(32, 1)     # 정규화된 거래량 (log + z-norm)
out = receptor(hocl, v)    # (32, 3)
```

입력 정규화는 외부 책임. 자세한 설계는 `Architecture - Price Receptor.md` 참조.

## Receptor Bundle

### ReceptorBundle

receptor 또는 다른 bundle 을 자식 노드로 묶어 multi-resolution 시간 위계 (예: 1m → 5m → 15m → 1h) 를
표현하는 composite 컨테이너. 자식 인스턴스는 모두 서로 다른 객체여야 한다 (positional weight 강제).
출력 차원은 receptor 와 동일한 3 으로 고정. aggregator 는 외부 주입 (`Pyramid` 등 활용 가능).

```python
import torch
from torch import nn
from nnpipeline import OHLCVReceptor, ReceptorBundle, Pyramid

# 1m 캔들 5 개를 묶어 5m representation 생성
receptors = [OHLCVReceptor() for _ in range(5)]
bundle = ReceptorBundle(
    children=receptors,
    aggregator=Pyramid(15, 3, depth=2, interlayer=[nn.LeakyReLU()]),
)
hocl = torch.randn(4, 5, 4)  # batch 4, n_leaves 5
v = torch.randn(4, 5, 1)
out = bundle(hocl, v)        # (4, 3)
```

자식이 또 `ReceptorBundle` 일 수 있으므로 재귀적으로 깊은 hierarchy 구성 가능. 비균등 묶음 (예: `(5, 10, 30, 10, 5)`)
도 자식 구성으로 자연 표현된다.
