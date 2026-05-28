import pytest
import torch
from torch import nn

from nnpipeline.prototype.receptor_bundle import ReceptorBundle


# 테스트용 더미 receptor: candle 1개 받아 3 차원 출력
class _DummyReceptor(nn.Module):

    def __init__(self, marker: float = 0.0):
        super().__init__()
        self.marker = marker

    def forward(self, hocl: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # hocl: (..., 4), v: (..., 1) → (..., 3)
        leading = hocl.shape[:-1]
        out = torch.zeros(*leading, 3)
        out[..., 0] = self.marker
        out[..., 1] = hocl[..., 0]
        out[..., 2] = v[..., 0]
        return out


# aggregator 단순화용: 입력 (..., N*3) → (..., 3) Linear
def _make_linear_aggregator(n_units: int) -> nn.Module:
    return nn.Linear(n_units * 3, 3)


class TestReceptorBundleInit:

    def test_rejects_non_list_children(self):
        with pytest.raises(TypeError):
            ReceptorBundle(children="not a list", aggregator=_make_linear_aggregator(1))

    def test_rejects_empty_children(self):
        with pytest.raises(ValueError):
            ReceptorBundle(children=[], aggregator=_make_linear_aggregator(1))

    def test_rejects_non_module_child(self):
        with pytest.raises(TypeError):
            ReceptorBundle(
                children=[_DummyReceptor(), "not a module"],
                aggregator=_make_linear_aggregator(2),
            )

    def test_rejects_duplicate_instances(self):
        shared = _DummyReceptor()
        with pytest.raises(ValueError):
            ReceptorBundle(
                children=[shared, shared],
                aggregator=_make_linear_aggregator(2),
            )

    def test_rejects_non_module_aggregator(self):
        with pytest.raises(TypeError):
            ReceptorBundle(
                children=[_DummyReceptor()],
                aggregator="not a module",
            )


class TestReceptorBundleNLeaves:

    def test_all_leaf_children(self):
        bundle = ReceptorBundle(
            children=[_DummyReceptor() for _ in range(5)],
            aggregator=_make_linear_aggregator(5),
        )
        assert bundle.n_leaves == 5

    def test_nested_bundles(self):
        inner_bundles = [
            ReceptorBundle(
                children=[_DummyReceptor() for _ in range(3)],
                aggregator=_make_linear_aggregator(3),
            )
            for _ in range(2)
        ]
        outer = ReceptorBundle(children=inner_bundles, aggregator=_make_linear_aggregator(2))
        assert outer.n_leaves == 6

    def test_mixed_children(self):
        # leaf 와 subtree 혼재
        sub = ReceptorBundle(
            children=[_DummyReceptor() for _ in range(2)],
            aggregator=_make_linear_aggregator(2),
        )
        leaf = _DummyReceptor()
        bundle = ReceptorBundle(children=[sub, leaf], aggregator=_make_linear_aggregator(2))
        assert bundle.n_leaves == 3

    def test_uneven_split(self):
        # 비균등 분할: (5, 10, 30, 10, 5) 같은 구성
        sub_sizes = [5, 10, 30, 10, 5]
        subs = [
            ReceptorBundle(
                children=[_DummyReceptor() for _ in range(s)],
                aggregator=_make_linear_aggregator(s),
            )
            for s in sub_sizes
        ]
        outer = ReceptorBundle(children=subs, aggregator=_make_linear_aggregator(len(sub_sizes)))
        assert outer.n_leaves == sum(sub_sizes)


class TestReceptorBundleForward:

    def test_single_level_shape(self):
        bundle = ReceptorBundle(
            children=[_DummyReceptor() for _ in range(5)],
            aggregator=_make_linear_aggregator(5),
        )
        hocl = torch.randn(2, 5, 4)
        v = torch.randn(2, 5, 1)
        out = bundle(hocl, v)
        assert out.shape == (2, 3)

    def test_nested_two_level_shape(self):
        inner_bundles = [
            ReceptorBundle(
                children=[_DummyReceptor() for _ in range(3)],
                aggregator=_make_linear_aggregator(3),
            )
            for _ in range(2)
        ]
        outer = ReceptorBundle(children=inner_bundles, aggregator=_make_linear_aggregator(2))
        hocl = torch.randn(2, 6, 4)
        v = torch.randn(2, 6, 1)
        out = outer(hocl, v)
        assert out.shape == (2, 3)

    def test_three_level_nesting_60_leaves(self):
        # 5 * 3 * 4 = 60 leaves. 1m → 5m → 15m → 1h 패턴
        fifteen_m_bundles = []
        for _ in range(4):
            five_m_bundles = []
            for _ in range(3):
                leaves = [_DummyReceptor() for _ in range(5)]
                five_m_bundles.append(
                    ReceptorBundle(children=leaves, aggregator=_make_linear_aggregator(5))
                )
            fifteen_m_bundles.append(
                ReceptorBundle(children=five_m_bundles, aggregator=_make_linear_aggregator(3))
            )
        one_h = ReceptorBundle(
            children=fifteen_m_bundles, aggregator=_make_linear_aggregator(4)
        )
        assert one_h.n_leaves == 60
        hocl = torch.randn(2, 60, 4)
        v = torch.randn(2, 60, 1)
        out = one_h(hocl, v)
        assert out.shape == (2, 3)

    def test_no_batch_dim(self):
        bundle = ReceptorBundle(
            children=[_DummyReceptor() for _ in range(3)],
            aggregator=_make_linear_aggregator(3),
        )
        hocl = torch.randn(3, 4)
        v = torch.randn(3, 1)
        out = bundle(hocl, v)
        assert out.shape == (3,)

    def test_multi_leading_dims(self):
        bundle = ReceptorBundle(
            children=[_DummyReceptor() for _ in range(3)],
            aggregator=_make_linear_aggregator(3),
        )
        hocl = torch.randn(2, 4, 3, 4)
        v = torch.randn(2, 4, 3, 1)
        out = bundle(hocl, v)
        assert out.shape == (2, 4, 3)

    def test_rejects_wrong_hocl_last_dim(self):
        bundle = ReceptorBundle(
            children=[_DummyReceptor() for _ in range(3)],
            aggregator=_make_linear_aggregator(3),
        )
        hocl = torch.randn(2, 3, 5)
        v = torch.randn(2, 3, 1)
        with pytest.raises(ValueError):
            bundle(hocl, v)

    def test_rejects_wrong_v_last_dim(self):
        bundle = ReceptorBundle(
            children=[_DummyReceptor() for _ in range(3)],
            aggregator=_make_linear_aggregator(3),
        )
        hocl = torch.randn(2, 3, 4)
        v = torch.randn(2, 3, 2)
        with pytest.raises(ValueError):
            bundle(hocl, v)

    def test_rejects_wrong_leaf_count(self):
        bundle = ReceptorBundle(
            children=[_DummyReceptor() for _ in range(3)],
            aggregator=_make_linear_aggregator(3),
        )
        hocl = torch.randn(2, 5, 4)
        v = torch.randn(2, 5, 1)
        with pytest.raises(ValueError):
            bundle(hocl, v)

    def test_order_preservation_via_marker(self):
        # marker가 다른 더미 receptor 3개를 자식으로. aggregator는 passthrough
        # 출력의 첫 3 성분 = 첫 자식 출력 (marker=1, H=hocl[0,0], v=v[0,0])
        class _PassthroughHead(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x[..., :3]

        children = [_DummyReceptor(marker=m) for m in (1.0, 2.0, 3.0)]
        bundle = ReceptorBundle(children=children, aggregator=_PassthroughHead())
        hocl = torch.zeros(3, 4)
        hocl[0, 0] = 7.0
        v = torch.zeros(3, 1)
        v[0, 0] = 11.0
        out = bundle(hocl, v)
        # 첫 자식 output: (marker=1.0, H=7.0, v=11.0)
        assert torch.allclose(out, torch.tensor([1.0, 7.0, 11.0]))

    def test_order_preservation_last_child(self):
        # 마지막 자식이 마지막 시점 입력을 받는지 — passthrough를 마지막 3개로
        class _PassthroughTail(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x[..., -3:]

        children = [_DummyReceptor(marker=m) for m in (1.0, 2.0, 3.0)]
        bundle = ReceptorBundle(children=children, aggregator=_PassthroughTail())
        hocl = torch.zeros(3, 4)
        hocl[2, 0] = 13.0
        v = torch.zeros(3, 1)
        v[2, 0] = 17.0
        out = bundle(hocl, v)
        # 마지막 자식 output: (marker=3.0, H=13.0, v=17.0)
        assert torch.allclose(out, torch.tensor([3.0, 13.0, 17.0]))

    def test_uneven_split_forward(self):
        # 비균등 분할 forward 정상 동작
        sub_sizes = [2, 5, 3]
        subs = [
            ReceptorBundle(
                children=[_DummyReceptor() for _ in range(s)],
                aggregator=_make_linear_aggregator(s),
            )
            for s in sub_sizes
        ]
        outer = ReceptorBundle(children=subs, aggregator=_make_linear_aggregator(len(sub_sizes)))
        hocl = torch.randn(4, 10, 4)
        v = torch.randn(4, 10, 1)
        out = outer(hocl, v)
        assert out.shape == (4, 3)


class TestReceptorBundleIntegration:

    def test_real_receptor_with_pyramid_aggregator(self):
        # 실제 OHLCVReceptor + Pyramid aggregator 통합 동작 확인
        from nnpipeline.prototype.ohlcv_receptor import OHLCVReceptor
        from nnpipeline.prototype.pyramid import Pyramid

        receptors = [OHLCVReceptor() for _ in range(5)]
        aggregator = Pyramid(
            in_features=15, out_features=3, depth=2, interlayer=[nn.LeakyReLU()]
        )
        bundle = ReceptorBundle(children=receptors, aggregator=aggregator)

        hocl = torch.randn(4, 5, 4)
        v = torch.randn(4, 5, 1)
        out = bundle(hocl, v)
        assert out.shape == (4, 3)

    def test_nested_real_receptor_three_level(self):
        # 5 * 3 * 4 = 60 leaves, 실제 receptor + Pyramid 로 3-level 구성
        from nnpipeline.prototype.ohlcv_receptor import OHLCVReceptor
        from nnpipeline.prototype.pyramid import Pyramid

        fifteen_m_bundles = []
        for _ in range(4):
            five_m_bundles = []
            for _ in range(3):
                receptors = [OHLCVReceptor() for _ in range(5)]
                agg = Pyramid(in_features=15, out_features=3, depth=2, interlayer=[nn.LeakyReLU()])
                five_m_bundles.append(ReceptorBundle(children=receptors, aggregator=agg))
            fifteen_m_bundles.append(
                ReceptorBundle(
                    children=five_m_bundles,
                    aggregator=Pyramid(9, 3, depth=2, interlayer=[nn.LeakyReLU()]),
                )
            )
        one_h = ReceptorBundle(
            children=fifteen_m_bundles,
            aggregator=Pyramid(12, 3, depth=2, interlayer=[nn.LeakyReLU()]),
        )
        assert one_h.n_leaves == 60

        hocl = torch.randn(4, 60, 4)
        v = torch.randn(4, 60, 1)
        out = one_h(hocl, v)
        assert out.shape == (4, 3)

    def test_gradient_flows_to_all_receptors(self):
        # 모든 자식 receptor 파라미터에 gradient 가 흐르는지 (시간 위치별 학습 보장)
        from nnpipeline.prototype.ohlcv_receptor import OHLCVReceptor
        from nnpipeline.prototype.pyramid import Pyramid

        receptors = [OHLCVReceptor() for _ in range(3)]
        bundle = ReceptorBundle(
            children=receptors,
            aggregator=Pyramid(9, 3, depth=1),
        )
        hocl = torch.randn(4, 3, 4)
        v = torch.randn(4, 3, 1)
        out = bundle(hocl, v)
        loss = out.sum()
        loss.backward()

        for i, r in enumerate(receptors):
            for name, p in r.named_parameters():
                assert p.grad is not None, f"receptor[{i}].{name} 에 gradient 없음"
                assert p.grad.abs().sum().item() > 0, f"receptor[{i}].{name} gradient 가 모두 0"
