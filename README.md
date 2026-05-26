# Neural Network Pipeline

자주 쓰는 1D MLP 형상을 미리 정의해 둔 PyTorch 프리셋 모음.

각 프리셋은 `nn.Sequential`의 서브클래스라 만들자마자 torch 네트워크 객체로 쓸 수 있다.
활성함수/정규화/드롭아웃 같이 층 사이에 끼우는 모듈은 사용자가 직접 인스턴스로 넘기고, 라이브러리는 그것들을 deepcopy해서 사이에 끼워 넣는다. 라이브러리가 활성함수 종류나 정규화 종류를 알 필요가 없다.

## 설치

```bash
pip install fish-nnpipeline
```

## 형상

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

## 파라미터 규칙

세 프리셋이 공통으로 받는 모듈 리스트 파라미터:

| 파라미터 | 위치 | 설명 |
|---|---|---|
| `pipe_head` | 맨 앞 | 첫 Linear 전에 한 번 |
| `interlayer` | Linear 사이 | 마지막 Linear 뒤에는 붙지 않음 |
| `pipe_end` | 맨 뒤 | 마지막 Linear 뒤에 한 번 |

- 기본값은 모두 빈 리스트.
- 안에 들어가는 모듈은 `deepcopy`되어 들어가므로, 동일 인스턴스를 넘겨도 사이사이에 독립 모듈이 생성된다.
- `depth >= 1`이어야 한다.
