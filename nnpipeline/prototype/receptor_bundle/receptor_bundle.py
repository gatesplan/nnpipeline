import torch
from torch import nn


class ReceptorBundle(nn.Module):

    def __init__(self, children: list, aggregator: nn.Module):
        super().__init__()

        # children 검증
        if not isinstance(children, list):
            raise TypeError(
                f"children는 list여야 합니다. 받은 타입: {type(children).__name__}"
            )
        if len(children) == 0:
            raise ValueError("children는 비어있을 수 없습니다.")
        for i, c in enumerate(children):
            if not isinstance(c, nn.Module):
                raise TypeError(
                    f"children[{i}]가 nn.Module 인스턴스가 아닙니다. 받은 타입: {type(c).__name__}"
                )
        ids = [id(c) for c in children]
        if len(set(ids)) != len(ids):
            raise ValueError(
                "children 내 인스턴스는 모두 서로 다른 객체여야 합니다 (시간 위치별 분리 학습)."
            )

        # aggregator 검증
        if not isinstance(aggregator, nn.Module):
            raise TypeError(
                f"aggregator는 nn.Module 인스턴스여야 합니다. 받은 타입: {type(aggregator).__name__}"
            )

        # nn.Module이 children() 메서드를 가지므로 attribute 이름 충돌 회피
        self.components = nn.ModuleList(children)
        self.aggregator = aggregator

        # subtree 자식은 n_leaves 속성으로 자기 leaf 수 알림. 없으면 leaf로 간주
        self.n_leaves = sum(getattr(c, "n_leaves", 1) for c in children)

    def forward(self, hocl: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # hocl: (..., n_leaves, 4), v: (..., n_leaves, 1)
        if hocl.shape[-1] != 4:
            raise ValueError(
                f"hocl 마지막 차원은 4여야 합니다. 받은 shape: {tuple(hocl.shape)}"
            )
        if v.shape[-1] != 1:
            raise ValueError(
                f"v 마지막 차원은 1이어야 합니다. 받은 shape: {tuple(v.shape)}"
            )
        if hocl.dim() < 2:
            raise ValueError(
                f"hocl은 최소 2개 차원이 필요합니다 (n_leaves 축 포함). 받은 shape: {tuple(hocl.shape)}"
            )
        if hocl.shape[-2] != self.n_leaves:
            raise ValueError(
                f"hocl 끝-1 차원은 n_leaves({self.n_leaves})여야 합니다. 받은 shape: {tuple(hocl.shape)}"
            )
        if v.shape[-2] != self.n_leaves:
            raise ValueError(
                f"v 끝-1 차원은 n_leaves({self.n_leaves})여야 합니다. 받은 shape: {tuple(v.shape)}"
            )

        outputs = []
        offset = 0
        for child in self.components:
            if hasattr(child, "n_leaves"):
                # subtree: 시퀀스 차원 유지하여 자식에게 위임
                child_leaves = child.n_leaves
                ch = hocl[..., offset:offset + child_leaves, :]
                cv = v[..., offset:offset + child_leaves, :]
            else:
                # leaf: indexing 으로 시퀀스 차원 제거
                child_leaves = 1
                ch = hocl[..., offset, :]
                cv = v[..., offset, :]
            outputs.append(child(ch, cv))
            offset += child_leaves

        # 자식별 출력 (..., 3) 을 시간 순서대로 stack → (..., N, 3) → flatten (..., N*3)
        stacked = torch.stack(outputs, dim=-2)
        flat = stacked.flatten(start_dim=-2)
        return self.aggregator(flat)
