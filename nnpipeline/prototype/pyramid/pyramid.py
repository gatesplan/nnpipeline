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
    # in_features ~ out_features 사이 depth+1 개 노드 폭을 선형 보간 (양 끝은 정확히 유지)
    if depth == 1:
        return [in_features, out_features]

    step = (out_features - in_features) / depth
    seq = [in_features]
    for i in range(1, depth):
        seq.append(int(round(in_features + step * i)))
    seq.append(out_features)
    return seq


class Pyramid(nn.Sequential):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        depth: int,
        interlayer: list = None,
        pipe_head: list = None,
        pipe_end: list = None,
    ):
        # 인자 검증
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

        # 모듈 시퀀스 구성
        modules = []
        modules.extend(deepcopy(m) for m in pipe_head)

        for i in range(depth):
            modules.append(nn.Linear(widths[i], widths[i + 1]))
            if i < depth - 1:
                modules.extend(deepcopy(m) for m in interlayer)

        modules.extend(deepcopy(m) for m in pipe_end)

        super().__init__(*modules)
