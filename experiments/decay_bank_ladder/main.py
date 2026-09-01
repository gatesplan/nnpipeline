"""지저분함 사다리 실험 — 어떤 지저분함이 receptor→DecayBank 의 bound 포화를 깨는가.

층별 (L0 → L3) 로:
  1. 정형화 사실 점검 (초과첨도, |δ| 자기상관) — 생성기가 실제로 지저분한지 확인
  2. oracle R² (진짜 상태, 닫힌 형태) — MC 교차검증 포함
  3. particle filter R² (관측 기반 feasible bound)
  4. receptor → bank(K=4) → Pyramid 학습 → model R²
  5. capture = model / PF — 이 비율이 무너지는 층이 아키텍처의 약점

실행: python -m experiments.decay_bank_ladder.main
"""

from pathlib import Path

import torch

from experiments.decay_bank_forecast.main import r2_per_horizon
from experiments.decay_bank_forecast.main_decompose import (
    HL_K4,
    ReceptorBankForecaster,
    train_and_eval,
)
from experiments.decay_bank_verification.synthesize import candles_from_increments
from experiments.decay_bank_ladder.synthesize import (
    LEVELS,
    make_series,
    mc_oracle,
    oracle_ladder,
    particle_filter_predict,
)

RESULTS_DIR = Path(__file__).parent / "results"

WINDOW = 120
HORIZONS = (1, 2, 3, 4, 5)
N_TRAIN = 8192
N_VAL = 2048
N_PARTICLES = 512
SEED = 7


def abs_autocorr(delta: torch.Tensor, lag: int) -> float:
    """|δ| 의 시계열 자기상관 (변동성 군집 지표). 배치 평균."""
    a = delta.abs()
    x, y = a[:, :-lag], a[:, lag:]
    xc = x - x.mean(dim=1, keepdim=True)
    yc = y - y.mean(dim=1, keepdim=True)
    corr = (xc * yc).mean(dim=1) / (xc.std(dim=1) * yc.std(dim=1)).clamp_min(1e-12)
    return corr.mean().item()


def excess_kurtosis(delta: torch.Tensor) -> float:
    x = delta.flatten()
    xc = x - x.mean()
    return (xc.pow(4).mean() / xc.pow(2).mean().pow(2)).item() - 3.0


def build_level_data(cfg, n_samples: int, seed: int):
    n_total = WINDOW + max(HORIZONS)
    delta, s, g, r = make_series(cfg, n_samples, n_total, seed=seed)
    dw = delta[:, :WINDOW]
    hocl, v = candles_from_increments(dw, mean_step=dw.abs().mean().item(), seed=seed)
    cum = torch.cumsum(delta[:, WINDOW:], dim=-1)
    targets = torch.stack([cum[:, k - 1] for k in HORIZONS], dim=-1)
    state_n = (s[:, WINDOW - 1], g[:, WINDOW - 1], r[:, WINDOW - 1])
    return dw, hocl, v, targets, state_n


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    lines = [
        f"지저분함 사다리: window={WINDOW}, horizons={HORIZONS}, "
        f"train/val={N_TRAIN}/{N_VAL}, particles={N_PARTICLES}",
        "",
        f"{'level':>12} | {'kurt':>6} | {'ac|d|1':>6} | {'ac|d|10':>7} | "
        f"{'oracle@5':>8} | {'PF@5':>6} | {'model@5':>7} | {'capture':>7} | learned hl",
        "-" * 110,
    ]

    for name, cfg in LEVELS:
        dw_tr, hocl_tr, v_tr, y_tr, _ = build_level_data(cfg, N_TRAIN, seed=SEED)
        dw_va, hocl_va, v_va, y_va, (s_va, g_va, r_va) = build_level_data(
            cfg, N_VAL, seed=SEED + 999
        )

        mu, sd = y_tr.mean(dim=0, keepdim=True), y_tr.std(dim=0, keepdim=True)
        y_tr_n, y_va_n = (y_tr - mu) / sd, (y_va - mu) / sd

        # oracle (닫힌 형태) + MC 교차검증
        oracle_pred = oracle_ladder(s_va, g_va, r_va, cfg, HORIZONS)
        mc_pred = mc_oracle(
            s_va[:256], g_va[:256], r_va[:256], cfg, HORIZONS, n_paths=2000, seed=SEED
        )
        cc = torch.corrcoef(torch.stack([oracle_pred[:256, -1], mc_pred[:, -1]]))[0, 1].item()
        r2_oracle = r2_per_horizon((oracle_pred - mu) / sd, y_va_n)

        # feasible bound (particle filter)
        pf_pred = particle_filter_predict(dw_va, cfg, HORIZONS, N_PARTICLES, seed=SEED)
        r2_pf = r2_per_horizon((pf_pred - mu) / sd, y_va_n)

        # 모델 학습
        torch.manual_seed(SEED)
        model = ReceptorBankForecaster(HL_K4, "pyramid")
        r2_model, hls = train_and_eval(
            model, (hocl_tr, v_tr), (hocl_va, v_va), y_tr_n, y_va_n
        )

        cap = r2_model[-1].item() / r2_pf[-1].item()
        hl_str = "[" + ", ".join(f"{h:.1f}" for h in hls) + "]"
        lines.append(
            f"{name:>12} | {excess_kurtosis(dw_tr):>6.2f} | {abs_autocorr(dw_tr, 1):>6.3f} | "
            f"{abs_autocorr(dw_tr, 10):>7.3f} | {r2_oracle[-1].item():>8.3f} | "
            f"{r2_pf[-1].item():>6.3f} | {r2_model[-1].item():>7.3f} | {cap:>7.1%} | {hl_str}"
        )
        lines.append(f"{'':>12} | oracle-vs-MC corr @5: {cc:.4f}")
        print(lines[-2])
        print(lines[-1])

    report = "\n".join(lines)
    (RESULTS_DIR / "report_ladder.txt").write_text(report, encoding="utf-8")
    print(f"\nsaved -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()
