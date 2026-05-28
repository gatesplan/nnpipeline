# Pyramid

in_features 에서 out_features 로 폭이 선형 보간되는 Linear 층 depth 개를 쌓는 1D MLP 프리셋.

## Pyramid

`nn.Sequential` 을 상속. in_features → out_features 폭으로 선형 보간된 시퀀스로 Linear 층 depth 개를 쌓는다.
Linear 사이에는 `interlayer` 모듈들이 `deepcopy` 되어 삽입, 앞뒤로 `pipe_head`, `pipe_end` 가 deepcopy 되어 붙는다.

폭 보간 규칙:
- depth == 1: [in_features, out_features]
- depth >= 2: in_features 와 out_features 양 끝값은 정확히 유지, 중간 폭은 정수 반올림.

### Properties
in_features: int                # 첫 Linear 입력 폭
out_features: int               # 마지막 Linear 출력 폭
depth: int                      # Linear 층 개수

### __init__
__init__(in_features: int, out_features: int, depth: int, interlayer: list = None, pipe_head: list = None, pipe_end: list = None)
    raise ValueError
    raise TypeError
    in_features, out_features, depth 모두 1 이상의 정수.
    interlayer, pipe_head, pipe_end 는 nn.Module 리스트 (None 이면 빈 리스트).
    None 이 아닌 모듈은 모두 deepcopy 되어 시퀀스에 삽입됨.

### Methods

상속받은 `nn.Sequential` 메서드 그대로 사용. 별도 메서드 없음.
