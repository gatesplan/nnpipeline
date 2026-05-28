# Cylinder

동일 폭의 Linear 층을 depth개 쌓는 1D MLP 프리셋. nn.Sequential 상속.

## Cylinder

`nn.Sequential` 을 상속하는 동일 폭 MLP 빌더. depth 개의 `Linear(in_features, in_features)` 를 쌓고,
Linear 사이에는 `interlayer` 모듈 리스트가 `deepcopy` 되어 끼워진다. 시퀀스 앞뒤로는 `pipe_head`,
`pipe_end` 가 마찬가지로 deepcopy 되어 붙는다.

### Properties
in_features: int                # 입력·출력 폭 (모든 Linear 가 동일 폭)
depth: int                      # Linear 층 개수

### __init__
__init__(in_features: int, depth: int, interlayer: list = None, pipe_head: list = None, pipe_end: list = None)
    raise ValueError
    raise TypeError
    in_features 는 1 이상의 정수, depth 는 1 이상의 정수.
    interlayer, pipe_head, pipe_end 는 nn.Module 리스트 (None 이면 빈 리스트로 처리).
    리스트 내 모든 원소는 nn.Module 인스턴스여야 함.
    None 이 아닌 모듈은 모두 deepcopy 되어 시퀀스에 삽입됨 (원본 공유 방지).

### Methods

상속받은 `nn.Sequential` 메서드 그대로 사용. 별도 메서드 없음.
