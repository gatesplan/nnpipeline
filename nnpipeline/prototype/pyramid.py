from copy import deepcopy
from torch import nn


def _validate_module_list(value, name):
    if not isinstance(value, list):
        raise TypeError(f"{name}는 list여야 합니다. 받은 타입: {type(value).__name__}")
    for i, m in enumerate(value):
        if not isinstance(m, nn.Module):
            raise TypeError(
                f"{name}[{i}]가 nn.Module 인스턴스가 아닙니다. 받은 타입: {type(m).__name__}"
            )


def _linear_width_sequence(in_features: int, out_features: int, depth: int) -> list:
    """
    in_features에서 out_features까지 depth+1개 노드로 선형 보간한 폭 시퀀스.
    양 끝값은 정확히 in_features, out_features를 유지하고 중간은 정수 반올림.
    """
    if depth == 1:
        return [in_features, out_features]

    step = (out_features - in_features) / depth
    seq = [in_features]
    for i in range(1, depth):
        seq.append(int(round(in_features + step * i)))
    seq.append(out_features)
    return seq


class Pyramid(nn.Sequential):
    """
    in_features에서 out_features로 폭이 선형 보간되는 Linear 층 depth개를 쌓는 1D MLP 프리셋.
    Linear 사이에 interlayer 모듈들이 deepcopy되어 끼워진다.
    앞뒤로 pipe_head, pipe_end가 deepcopy되어 붙는다.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        depth: int,
        interlayer: list = None,
        pipe_head: list = None,
        pipe_end: list = None,
    ):
        if not isinstance(in_features, int) or in_features < 1:
            raise ValueError(f"in_features는 1 이상의 정수여야 합니다. 받은 값: {in_features}")
        if not isinstance(out_features, int) or out_features < 1:
            raise ValueError(f"out_features는 1 이상의 정수여야 합니다. 받은 값: {out_features}")
        if not isinstance(depth, int) or depth < 1:
            raise ValueError(f"depth는 1 이상의 정수여야 합니다. 받은 값: {depth}")

        interlayer = interlayer if interlayer is not None else []
        pipe_head = pipe_head if pipe_head is not None else []
        pipe_end = pipe_end if pipe_end is not None else []

        _validate_module_list(interlayer, "interlayer")
        _validate_module_list(pipe_head, "pipe_head")
        _validate_module_list(pipe_end, "pipe_end")

        widths = _linear_width_sequence(in_features, out_features, depth)

        modules = []
        modules.extend(deepcopy(m) for m in pipe_head)

        for i in range(depth):
            modules.append(nn.Linear(widths[i], widths[i + 1]))
            if i < depth - 1:
                modules.extend(deepcopy(m) for m in interlayer)

        modules.extend(deepcopy(m) for m in pipe_end)

        super().__init__(*modules)
