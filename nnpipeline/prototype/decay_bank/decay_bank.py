import math

import torch
from torch import nn


class DecayBank(nn.Module):
    """다중 시간스케일 지수 감쇠 상태 은행.

    임베딩 시퀀스 (..., n, d) 를 받아 학습 가능한 감쇠율 λ_k 별 지수이동평균 상태를
    시점 n 기준으로 누적한다. 시간 정보가 구조적으로 최근 시점에 강하게 남는
    recency bias 를 그래프 자체에 새기는 것이 목적.

        h_k[t] = λ_k · h_k[t-1] + (1 - λ_k) · e_t

    인접 스케일 상태 차이 (fast - slow) 는 "움직임이 약해지고 있는가" 를 직접 표현하는
    모멘텀 감쇠 신호 — include_diffs=True 면 출력에 구조적으로 포함된다.
    """

    def __init__(
        self,
        half_lives: tuple = (2.0, 8.0, 32.0),
        learnable: bool = True,
        include_diffs: bool = True,
        bias_correction: bool = True,
    ):
        super().__init__()

        # half_lives 검증
        try:
            hl = [float(h) for h in half_lives]
        except (TypeError, ValueError):
            raise TypeError(
                f"half_lives 는 숫자 시퀀스여야 합니다. 받은 값: {half_lives!r}"
            )
        if len(hl) == 0:
            raise ValueError("half_lives 는 비어있을 수 없습니다.")
        if any(h <= 0 for h in hl):
            raise ValueError(f"half_lives 는 모두 양수여야 합니다. 받은 값: {hl}")
        if any(b <= a for a, b in zip(hl, hl[1:])):
            raise ValueError(
                f"half_lives 는 순증가(빠른 스케일 → 느린 스케일)여야 합니다. 받은 값: {hl}"
            )
        if include_diffs and len(hl) < 2:
            raise ValueError(
                "include_diffs=True 는 스케일 2 개 이상이 필요합니다 (인접 차이 계산)."
            )

        # λ_k = 2^(-1/half_life) — half_life 캔들 후 가중치 절반.
        # sigmoid logit 으로 저장하여 학습 시에도 λ ∈ (0, 1) 구조적 보장.
        lam = torch.tensor([2.0 ** (-1.0 / h) for h in hl], dtype=torch.float32)
        logit = torch.log(lam / (1.0 - lam))
        if learnable:
            self.lambda_logit = nn.Parameter(logit)
        else:
            self.register_buffer("lambda_logit", logit)

        self.n_scales = len(hl)
        self.include_diffs = include_diffs
        self.bias_correction = bias_correction

    @property
    def lambdas(self) -> torch.Tensor:
        # 현재 유효 감쇠율 λ_k ∈ (0, 1). shape (n_scales,)
        return torch.sigmoid(self.lambda_logit)

    @property
    def half_lives(self) -> torch.Tensor:
        # 현재 유효 반감기 (학습으로 변할 수 있음). shape (n_scales,)
        return -math.log(2.0) / torch.log(self.lambdas)

    @property
    def out_scales(self) -> int:
        # 출력 스케일 축 크기: K 개 상태 (+ K-1 개 인접 차이)
        if self.include_diffs:
            return 2 * self.n_scales - 1
        return self.n_scales

    def forward(self, e: torch.Tensor, return_sequence: bool = False) -> torch.Tensor:
        # e: (..., n, d) — 임베딩 시퀀스. 시간축은 끝-1, 과거 → 최근 순.
        # 반환: (..., out_scales, d) 시점 n 상태.
        #       return_sequence=True 면 (..., n, out_scales, d) 전체 궤적.
        if e.dim() < 2:
            raise ValueError(
                f"e 는 최소 2 개 차원 (시간, 채널) 이 필요합니다. 받은 shape: {tuple(e.shape)}"
            )
        n = e.shape[-2]
        if n < 1:
            raise ValueError(f"시간축 길이는 1 이상이어야 합니다. 받은 shape: {tuple(e.shape)}")

        lam = self.lambdas

        if return_sequence:
            states = self._forward_sequence(e, lam)   # (..., n, K, d)
        else:
            states = self._forward_final(e, lam)      # (..., K, d)

        if self.include_diffs:
            # 인접 스케일 차이 (fast - slow): 스케일 축에 그대로 이어붙임
            diffs = states[..., :-1, :] - states[..., 1:, :]
            states = torch.cat([states, diffs], dim=-2)

        return states

    def _forward_final(self, e: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        # 선형 재귀의 닫힌 형태: h_k[n] = (1-λ_k) Σ_t λ_k^(n-1-t) e_t.
        # 지수가 전부 0 이상이라 수치적으로 안정.
        n = e.shape[-2]
        exps = torch.arange(n - 1, -1, -1, dtype=e.dtype, device=e.device)  # (n,)
        w = (1.0 - lam).unsqueeze(-1) * lam.unsqueeze(-1).pow(exps)         # (K, n)
        states = torch.einsum("kn,...nd->...kd", w, e)                      # (..., K, d)
        if self.bias_correction:
            # 가중치 합 = 1 - λ^n. 나눠서 창 길이·스케일 무관 가중평균으로 정규화
            states = states / (1.0 - lam.pow(n)).unsqueeze(-1)
        return states

    def _forward_sequence(self, e: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        n = e.shape[-2]
        lam_col = lam.unsqueeze(-1)                                         # (K, 1)
        h = e.new_zeros(*e.shape[:-2], self.n_scales, e.shape[-1])          # (..., K, d)
        outs = []
        for t in range(n):
            et = e[..., t, :].unsqueeze(-2)                                 # (..., 1, d)
            h = lam_col * h + (1.0 - lam_col) * et
            if self.bias_correction:
                outs.append(h / (1.0 - lam.pow(t + 1)).unsqueeze(-1))
            else:
                outs.append(h)
        return torch.stack(outs, dim=-3)                                    # (..., n, K, d)
