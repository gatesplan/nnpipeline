import math

import pytest
import torch
from torch import nn

from nnpipeline.prototype.decay_bank import DecayBank


class TestDecayBankInit:

    def test_rejects_non_numeric_half_lives(self):
        with pytest.raises(TypeError):
            DecayBank(half_lives=("a", "b"))

    def test_rejects_empty_half_lives(self):
        with pytest.raises(ValueError):
            DecayBank(half_lives=())

    def test_rejects_nonpositive_half_life(self):
        with pytest.raises(ValueError):
            DecayBank(half_lives=(0.0, 8.0))
        with pytest.raises(ValueError):
            DecayBank(half_lives=(-2.0, 8.0))

    def test_rejects_non_increasing_half_lives(self):
        with pytest.raises(ValueError):
            DecayBank(half_lives=(8.0, 2.0))
        with pytest.raises(ValueError):
            DecayBank(half_lives=(8.0, 8.0))

    def test_rejects_diffs_with_single_scale(self):
        with pytest.raises(ValueError):
            DecayBank(half_lives=(8.0,), include_diffs=True)

    def test_single_scale_without_diffs_ok(self):
        bank = DecayBank(half_lives=(8.0,), include_diffs=False)
        assert bank.n_scales == 1
        assert bank.out_scales == 1

    def test_initial_half_lives_match_spec(self):
        hl = (2.0, 8.0, 32.0)
        bank = DecayBank(half_lives=hl)
        assert torch.allclose(
            bank.half_lives, torch.tensor(hl), rtol=1e-4
        )

    def test_lambdas_in_open_unit_interval(self):
        bank = DecayBank()
        lam = bank.lambdas
        assert torch.all(lam > 0.0) and torch.all(lam < 1.0)

    def test_learnable_registers_parameter(self):
        bank = DecayBank(learnable=True)
        params = dict(bank.named_parameters())
        assert "lambda_logit" in params

    def test_non_learnable_registers_buffer(self):
        bank = DecayBank(learnable=False)
        assert len(list(bank.parameters())) == 0
        assert "lambda_logit" in dict(bank.named_buffers())

    def test_parameter_count_equals_n_scales(self):
        bank = DecayBank(half_lives=(2.0, 8.0, 32.0, 128.0))
        assert sum(p.numel() for p in bank.parameters()) == 4


class TestDecayBankShapes:

    def test_final_state_shape_with_diffs(self):
        bank = DecayBank(half_lives=(2.0, 8.0, 32.0))  # K=3 → out_scales=5
        e = torch.randn(7, 16, 3)
        out = bank(e)
        assert out.shape == (7, 5, 3)

    def test_final_state_shape_without_diffs(self):
        bank = DecayBank(half_lives=(2.0, 8.0), include_diffs=False)
        e = torch.randn(16, 4)
        out = bank(e)
        assert out.shape == (2, 4)

    def test_sequence_shape(self):
        bank = DecayBank(half_lives=(2.0, 8.0))  # K=2 → out_scales=3
        e = torch.randn(5, 12, 3)
        out = bank(e, return_sequence=True)
        assert out.shape == (5, 12, 3, 3)

    def test_arbitrary_leading_dims(self):
        bank = DecayBank(half_lives=(2.0, 8.0))
        e = torch.randn(2, 3, 10, 6)
        out = bank(e)
        assert out.shape == (2, 3, 3, 6)

    def test_rejects_one_dim_input(self):
        bank = DecayBank()
        with pytest.raises(ValueError):
            bank(torch.randn(10))


class TestDecayBankSemantics:

    def test_constant_input_recovers_constant(self):
        # bias correction 덕에 상수 시퀀스의 상태는 창 길이·스케일 무관하게 그 상수
        bank = DecayBank(half_lives=(2.0, 8.0, 32.0), include_diffs=False)
        e = torch.full((1, 5, 2), 3.0)  # 짧은 창 (n=5 << 반감기 32)
        out = bank(e)
        assert torch.allclose(out, torch.full((1, 3, 2), 3.0), atol=1e-5)

    def test_constant_input_diffs_are_zero(self):
        bank = DecayBank(half_lives=(2.0, 8.0, 32.0))
        e = torch.full((1, 20, 1), 1.5)
        out = bank(e)
        diffs = out[:, 3:, :]  # 스케일 축 뒤쪽 K-1 개가 인접 차이
        assert torch.allclose(diffs, torch.zeros_like(diffs), atol=1e-5)

    def test_final_matches_sequence_last_step(self):
        bank = DecayBank(half_lives=(2.0, 8.0, 32.0))
        e = torch.randn(4, 30, 3)
        final = bank(e)
        seq = bank(e, return_sequence=True)
        assert torch.allclose(final, seq[:, -1], atol=1e-5)

    def test_old_impulse_decays_faster_in_fast_scale(self):
        # t=0 에만 impulse: 빠른 스케일 상태가 느린 스케일보다 작아야 함 (더 많이 잊음)
        bank = DecayBank(half_lives=(2.0, 32.0), include_diffs=False)
        e = torch.zeros(1, 20, 1)
        e[0, 0, 0] = 1.0
        out = bank(e)
        h_fast, h_slow = out[0, 0, 0].item(), out[0, 1, 0].item()
        assert h_fast < h_slow

    def test_recent_impulse_stronger_in_fast_scale(self):
        # 마지막 시점에만 impulse: 빠른 스케일이 더 강하게 반응 (recency bias)
        bank = DecayBank(half_lives=(2.0, 32.0), include_diffs=False)
        e = torch.zeros(1, 20, 1)
        e[0, -1, 0] = 1.0
        out = bank(e)
        h_fast, h_slow = out[0, 0, 0].item(), out[0, 1, 0].item()
        assert h_fast > h_slow

    def test_weakening_sequence_negative_diff(self):
        # 증분이 점점 약해지는 상승: 최근 평균 (fast) < 장기 평균 (slow) → diff 음수
        bank = DecayBank(half_lives=(2.0, 32.0))
        e = (0.9 ** torch.arange(30, dtype=torch.float32)).view(1, 30, 1)
        out = bank(e)
        diff = out[0, 2, 0].item()  # [h_f, h_s, h_f - h_s]
        assert diff < 0.0

    def test_accelerating_sequence_positive_diff(self):
        bank = DecayBank(half_lives=(2.0, 32.0))
        e = (1.1 ** torch.arange(30, dtype=torch.float32)).view(1, 30, 1)
        out = bank(e)
        diff = out[0, 2, 0].item()
        assert diff > 0.0

    def test_gradient_flows_to_lambda(self):
        bank = DecayBank(half_lives=(2.0, 8.0))
        e = torch.randn(3, 15, 2)
        bank(e).sum().backward()
        assert bank.lambda_logit.grad is not None
        assert torch.all(torch.isfinite(bank.lambda_logit.grad))

    def test_gradient_flows_through_sequence_mode(self):
        bank = DecayBank(half_lives=(2.0, 8.0))
        e = torch.randn(3, 15, 2, requires_grad=True)
        bank(e, return_sequence=True).sum().backward()
        assert e.grad is not None

    def test_robust_rejects_nonpositive_clip(self):
        with pytest.raises(ValueError):
            DecayBank(robust_clip=0.0)
        with pytest.raises(ValueError):
            DecayBank(robust_clip=-1.0)

    def test_robust_rejects_non_numeric_clip(self):
        with pytest.raises(TypeError):
            DecayBank(robust_clip="big")

    def test_robust_shapes_match_default(self):
        bank = DecayBank(half_lives=(2.0, 8.0, 32.0), robust_clip=3.0)
        e = torch.randn(7, 16, 3)
        assert bank(e).shape == (7, 5, 3)
        assert bank(e, return_sequence=True).shape == (7, 16, 5, 3)

    def test_robust_final_matches_sequence_last_step(self):
        bank = DecayBank(half_lives=(2.0, 8.0), robust_clip=3.0)
        e = torch.randn(4, 30, 3)
        assert torch.allclose(bank(e), bank(e, return_sequence=True)[:, -1], atol=1e-5)

    def test_robust_constant_input_unchanged(self):
        # 상수 시퀀스는 innovation 이 0 → clipping 무영향, 상태는 그 상수
        bank = DecayBank(half_lives=(2.0, 8.0), include_diffs=False, robust_clip=3.0)
        e = torch.full((1, 20, 2), 2.5)
        assert torch.allclose(bank(e), torch.full((1, 2, 2), 2.5), atol=1e-5)

    def test_robust_equals_default_without_outliers(self):
        # 정상 범위 innovation 만 있으면 (완만한 시퀀스) robust 와 기본이 거의 일치
        base = DecayBank(half_lives=(4.0, 16.0), robust_clip=None)
        rob = DecayBank(half_lives=(4.0, 16.0), robust_clip=10.0)
        t = torch.arange(40, dtype=torch.float32)
        e = torch.sin(t / 5.0).view(1, 40, 1)
        assert torch.allclose(base(e), rob(e), atol=1e-4)

    def test_robust_suppresses_jump(self):
        # 점프 하나가 낀 시퀀스: robust 상태가 기본 상태보다 무점프 상태에 가까워야 함
        base = DecayBank(half_lives=(2.0, 8.0), include_diffs=False)
        rob = DecayBank(half_lives=(2.0, 8.0), include_diffs=False, robust_clip=3.0)
        torch.manual_seed(0)
        clean = torch.randn(1, 40, 1) * 0.1
        jumped = clean.clone()
        jumped[0, 35, 0] += 5.0
        ref = base(clean)
        err_base = (base(jumped) - ref).abs().sum()
        err_rob = (rob(jumped) - ref).abs().sum()
        assert err_rob < err_base * 0.5

    def test_dual_requires_robust_clip(self):
        with pytest.raises(ValueError):
            DecayBank(robust_dual=True)

    def test_dual_shapes(self):
        bank = DecayBank(half_lives=(2.0, 8.0, 32.0), robust_clip=3.0, robust_dual=True)
        assert bank.out_scales == 10
        e = torch.randn(7, 16, 3)
        assert bank(e).shape == (7, 10, 3)
        assert bank(e, return_sequence=True).shape == (7, 16, 10, 3)

    def test_dual_first_block_matches_plain_bank(self):
        # 미적용 블록은 robust 없는 기본 bank 출력과 동일해야 함
        plain = DecayBank(half_lives=(2.0, 8.0))
        dual = DecayBank(half_lives=(2.0, 8.0), robust_clip=3.0, robust_dual=True)
        e = torch.randn(4, 30, 3)
        assert torch.allclose(dual(e)[..., :3, :], plain(e), atol=1e-5)

    def test_dual_second_block_matches_robust_bank(self):
        rob = DecayBank(half_lives=(2.0, 8.0), robust_clip=3.0)
        dual = DecayBank(half_lives=(2.0, 8.0), robust_clip=3.0, robust_dual=True)
        e = torch.randn(4, 30, 3)
        assert torch.allclose(dual(e)[..., 3:, :], rob(e), atol=1e-5)

    def test_dual_gradient_flows(self):
        bank = DecayBank(half_lives=(2.0, 8.0), robust_clip=3.0, robust_dual=True)
        e = torch.randn(3, 15, 2, requires_grad=True)
        bank(e).sum().backward()
        assert e.grad is not None
        assert torch.all(torch.isfinite(bank.lambda_logit.grad))

    def test_robust_gradient_flows(self):
        bank = DecayBank(half_lives=(2.0, 8.0), robust_clip=3.0)
        e = torch.randn(3, 15, 2, requires_grad=True)
        bank(e).sum().backward()
        assert e.grad is not None
        assert bank.lambda_logit.grad is not None
        assert torch.all(torch.isfinite(bank.lambda_logit.grad))

    def test_no_bias_correction_shrinks_slow_scale(self):
        # 보정 없으면 짧은 창에서 느린 스케일 상태가 체계적으로 작아짐 (설계 근거 확인)
        bank = DecayBank(half_lives=(2.0, 32.0), include_diffs=False, bias_correction=False)
        e = torch.full((1, 5, 1), 1.0)
        out = bank(e)
        h_fast, h_slow = out[0, 0, 0].item(), out[0, 1, 0].item()
        assert h_slow < h_fast < 1.0
