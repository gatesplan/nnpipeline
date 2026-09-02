"""innovation clipping (robust_clip=3.0) 을 켠 모델로 사다리 4 층 재실행.

main.py 와 동일한 데이터·프로토콜. 판정 기준:
  1. L3 (fat tail + 점프) 의 capture 가 기본 모델 (93.9%) 보다 회복되는가
  2. L3 에서 학습된 반감기 스펙트럼에 창 길이를 넘는 스케일 (기본 모델에서 ~210) 이 사라지는가
  3. L0~L2 (이상치 없는 층) 에서 성능 저하가 없는가

실행: python -m experiments.decay_bank_ladder.main_robust
"""

import torch

from experiments.decay_bank_forecast.main import r2_per_horizon
from experiments.decay_bank_forecast.main_decompose import (
    HL_K4,
    ReceptorBankForecaster,
    train_and_eval,
)
from experiments.decay_bank_ladder.main import (
    HORIZONS,
    N_PARTICLES,
    N_TRAIN,
    N_VAL,
    RESULTS_DIR,
    SEED,
    build_level_data,
)
from experiments.decay_bank_ladder.synthesize import LEVELS, particle_filter_predict

ROBUST_CLIP = 3.0


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    lines = [
        f"robust ladder: robust_clip={ROBUST_CLIP}, horizons={HORIZONS}, "
        f"train/val={N_TRAIN}/{N_VAL}",
        "",
        f"{'level':>12} | {'PF@5':>6} | {'model@5':>7} | {'capture':>7} | learned hl",
        "-" * 80,
    ]

    for name, cfg in LEVELS:
        _, hocl_tr, v_tr, y_tr, _ = build_level_data(cfg, N_TRAIN, seed=SEED)
        dw_va, hocl_va, v_va, y_va, _ = build_level_data(cfg, N_VAL, seed=SEED + 999)

        mu, sd = y_tr.mean(dim=0, keepdim=True), y_tr.std(dim=0, keepdim=True)
        y_tr_n, y_va_n = (y_tr - mu) / sd, (y_va - mu) / sd

        pf_pred = particle_filter_predict(dw_va, cfg, HORIZONS, N_PARTICLES, seed=SEED)
        r2_pf = r2_per_horizon((pf_pred - mu) / sd, y_va_n)

        torch.manual_seed(SEED)
        model = ReceptorBankForecaster(HL_K4, "pyramid", robust_clip=ROBUST_CLIP)
        r2_model, hls = train_and_eval(
            model, (hocl_tr, v_tr), (hocl_va, v_va), y_tr_n, y_va_n
        )

        cap = r2_model[-1].item() / r2_pf[-1].item()
        hl_str = "[" + ", ".join(f"{h:.1f}" for h in hls) + "]"
        lines.append(
            f"{name:>12} | {r2_pf[-1].item():>6.3f} | {r2_model[-1].item():>7.3f} | "
            f"{cap:>7.1%} | {hl_str}"
        )
        print(lines[-1])

    report = "\n".join(lines)
    (RESULTS_DIR / "report_robust.txt").write_text(report, encoding="utf-8")
    print(f"\nsaved -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()
