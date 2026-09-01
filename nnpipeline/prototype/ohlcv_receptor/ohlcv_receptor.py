import torch
from torch import nn


class OHLCVReceptor(nn.Module):

    # 구조 상수 — 설계 단계에서 확정. hyperparameter 아님.
    SIDE_DIM = 2          # hoc_2, y_1, y_2 차원
    HIDDEN = 2            # Linear_combU/combL hidden
    HIDDEN_V = 4          # Linear_combPV hidden
    COMBV_INPUT_DIM = 1 + SIDE_DIM + 1 + 1  # y_1[0], hoc_2, y_2[0], V

    def __init__(self):
        super().__init__()

        # 명명 layer 6 개
        self.linear_oc = nn.Linear(2, 1 + self.SIDE_DIM + 1)
        self.linear_upper = nn.Linear(2, self.SIDE_DIM)
        self.linear_lower = nn.Linear(2, self.SIDE_DIM)
        self.linear_comb_u = nn.Linear(self.SIDE_DIM, self.HIDDEN)
        self.linear_comb_l = nn.Linear(self.SIDE_DIM, self.HIDDEN)
        # LayerNorm: V (큰 스케일)와 receptor 처리값 (작은 스케일) 균형.
        # BatchNorm 은 end-to-end 학습 시 running stats 지연으로 eval 이 불안정했음 (260902 교체)
        self.norm_pv = nn.LayerNorm(self.COMBV_INPUT_DIM)
        self.linear_comb_pv = nn.Linear(self.COMBV_INPUT_DIM, self.HIDDEN_V)

        # Unnamed 최종 projection — scalar 압축
        self.proj_u = nn.Linear(self.HIDDEN, 1)
        self.proj_l = nn.Linear(self.HIDDEN, 1)
        self.proj_v = nn.Linear(self.HIDDEN_V, 1)

    def forward(self, hocl: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # hocl: (..., 4), v: (..., 1)
        if hocl.shape[-1] != 4:
            raise ValueError(f"hocl 마지막 차원은 4여야 합니다. 받은 shape: {tuple(hocl.shape)}")
        if v.shape[-1] != 1:
            raise ValueError(f"v 마지막 차원은 1이어야 합니다. 받은 shape: {tuple(v.shape)}")

        H = hocl[..., 0:1]
        O = hocl[..., 1:2]
        C = hocl[..., 2:3]
        L = hocl[..., 3:4]

        # Linear_OC: 선형층, ReLU 없음. hoc_1, hoc_2, hoc_3 분할
        hoc = self.linear_oc(torch.cat([O, C], dim=-1))
        hoc_1 = hoc[..., 0:1]
        hoc_2 = hoc[..., 1:1 + self.SIDE_DIM]
        hoc_3 = hoc[..., 1 + self.SIDE_DIM:]

        # 비대칭 routing: H → upper, L → lower (tanh 활성 - signed input 대칭 처리)
        y_1 = torch.tanh(self.linear_upper(torch.cat([H, hoc_1], dim=-1)))
        y_2 = torch.tanh(self.linear_lower(torch.cat([L, hoc_3], dim=-1)))

        # out_1, out_2: element-wise add + LeakyReLU hidden + projection
        h_u = torch.nn.functional.leaky_relu(self.linear_comb_u(y_1 + hoc_2))
        h_l = torch.nn.functional.leaky_relu(self.linear_comb_l(y_2 + hoc_2))
        out_1 = self.proj_u(h_u)
        out_2 = self.proj_l(h_l)

        # out_v: LayerNorm 통과 후 LeakyReLU hidden + projection
        # 첫 노드만 사용하여 그 노드들만 volume 학습 신호 받음
        combv_input = torch.cat([y_1[..., 0:1], hoc_2, y_2[..., 0:1], v], dim=-1)
        combv_input_norm = self.norm_pv(combv_input)
        h_v = torch.nn.functional.leaky_relu(self.linear_comb_pv(combv_input_norm))
        out_v = self.proj_v(h_v)

        return torch.cat([out_1, out_2, out_v], dim=-1)
