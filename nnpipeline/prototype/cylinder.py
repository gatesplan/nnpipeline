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


class Cylinder(nn.Sequential):
    """
    동일 폭의 Linear 층을 depth개 쌓는 1D MLP 프리셋.
    Linear 사이에 interlayer 모듈들이 deepcopy되어 끼워진다.
    앞뒤로 pipe_head, pipe_end가 deepcopy되어 붙는다.
    """

    def __init__(
        self,
        in_features: int,
        depth: int,
        interlayer: list = None,
        pipe_head: list = None,
        pipe_end: list = None,
    ):
        if not isinstance(in_features, int) or in_features < 1:
            raise ValueError(f"in_features는 1 이상의 정수여야 합니다. 받은 값: {in_features}")
        if not isinstance(depth, int) or depth < 1:
            raise ValueError(f"depth는 1 이상의 정수여야 합니다. 받은 값: {depth}")

        interlayer = interlayer if interlayer is not None else []
        pipe_head = pipe_head if pipe_head is not None else []
        pipe_end = pipe_end if pipe_end is not None else []

        _validate_module_list(interlayer, "interlayer")
        _validate_module_list(pipe_head, "pipe_head")
        _validate_module_list(pipe_end, "pipe_end")

        modules = []
        modules.extend(deepcopy(m) for m in pipe_head)

        for i in range(depth):
            modules.append(nn.Linear(in_features, in_features))
            if i < depth - 1:
                modules.extend(deepcopy(m) for m in interlayer)

        modules.extend(deepcopy(m) for m in pipe_end)

        super().__init__(*modules)
