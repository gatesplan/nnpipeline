"""일봉 횡단면 상대수익률 예측: 120봉 → 다음 1~5봉 수익률의 동일 날짜 내 상대 위치.

타깃: 같은 날짜에 윈도우가 끝나는 종목들 집합 안에서 미래 k 봉 누적수익률을
z-score 로 변환한 값. 시장 공통 성분이 제거되어 "어느 종목이 상대적으로 오를 것인가"
만 남는다. 참여 종목 20 개 미만인 날짜는 제외.

평가: out-of-sample R² + 일별 IC (날짜별 예측-실현 상관의 평균, t 통계량 병기).

비교 모델은 main.py 와 동일 (A zero / B linear / C flatten / D bank / E dual).
실행: python -m experiments.daily_forecast.main_cross
"""

from pathlib import Path

from loguru import logger

logger.remove()

import math

import torch

from candle_data_manager import CandleDataAPI

from experiments.decay_bank_forecast.main_decompose import HL_K4, ReceptorBankForecaster
from experiments.receptor_real_data.cdm_data import LARGE_CAP, SMALL_CAP
from experiments.daily_forecast.data import build_arrays, load_stocks, make_inputs, make_targets
from experiments.daily_forecast.main import (
    DEVICE,
    EPOCHS,
    HORIZONS,
    SEED,
    WINDOW,
    YEARS,
    FlattenForecaster,
    linear_features,
    r2_per_horizon,
    train_model,
)
from experiments.daily_forecast.main_vol import ridge

RESULTS_DIR = Path(__file__).parent / "results"

MIN_GROUP = 20


def build_groups(ts_all, starts, window: int):
    """분할별로 윈도우 종료 날짜 → 그룹 id. 반환: (date_ids dict, n_dates dict)."""
    date_ids, n_dates = {}, {}
    for k, st in starts.items():
        end_ts = ts_all[st + window - 1]
        uniq, inv = torch.unique(end_ts, return_inverse=True)
        date_ids[k] = inv
        n_dates[k] = len(uniq)
    return date_ids, n_dates


def filter_small_groups(starts, date_ids, min_group: int = MIN_GROUP):
    """참여 종목이 min_group 미만인 날짜의 시작 인덱스 제거. 그룹 id 재부여."""
    out_starts, out_gids = {}, {}
    for k in starts:
        cnt = torch.bincount(date_ids[k])
        keep = cnt[date_ids[k]] >= min_group
        st = starts[k][keep]
        _, inv = torch.unique(date_ids[k][keep], return_inverse=True)
        out_starts[k], out_gids[k] = st, inv
    return out_starts, out_gids


def cs_normalize(y, gid):
    """날짜 그룹 내 z-score. y (B,H), gid (B,) → (B,H)."""
    n = int(gid.max().item()) + 1
    cnt = torch.bincount(gid, minlength=n).clamp(min=1).float().unsqueeze(-1)
    mean = torch.zeros(n, y.shape[1], device=y.device).index_add_(0, gid, y) / cnt
    yc = y - mean[gid]
    var = torch.zeros(n, y.shape[1], device=y.device).index_add_(0, gid, yc ** 2) / cnt
    return yc / (var.sqrt()[gid] + 1e-8)


def daily_ic(pred, target, gid):
    """날짜별 Pearson 상관 → (평균, t 통계량). horizon 별. 반환 두 (H,) 텐서."""
    n = int(gid.max().item()) + 1
    cnt = torch.bincount(gid, minlength=n).clamp(min=1).float().unsqueeze(-1)

    def gmean(x):
        return torch.zeros(n, x.shape[1], device=x.device).index_add_(0, gid, x) / cnt

    pc = pred - gmean(pred)[gid]
    tc = target - gmean(target)[gid]
    cov = gmean(pc * tc)
    vp, vt = gmean(pc ** 2), gmean(tc ** 2)
    ic = cov / (vp.sqrt() * vt.sqrt() + 1e-12)               # (n_dates, H)
    mean = ic.mean(dim=0)
    tstat = mean / (ic.std(dim=0) + 1e-12) * math.sqrt(ic.shape[0])
    return mean, tstat


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    torch.manual_seed(SEED)

    api = CandleDataAPI()
    stocks = load_stocks(LARGE_CAP + SMALL_CAP, api, years=YEARS)
    hocl_all, v_all, ts_all, starts, _ = build_arrays(stocks, WINDOW, max(HORIZONS))
    hocl_all, v_all = hocl_all.to(DEVICE), v_all.to(DEVICE)
    ts_all = ts_all.to(DEVICE)
    starts = {k: v.to(DEVICE) for k, v in starts.items()}

    date_ids, _ = build_groups(ts_all, starts, WINDOW)
    starts, gids = filter_small_groups(starts, date_ids)

    y_n, ys_raw = {}, {}
    for k in starts:
        y_raw = make_targets(hocl_all, starts[k], WINDOW, HORIZONS)
        ys_raw[k] = y_raw
        y_n[k] = cs_normalize(y_raw, gids[k])

    n_dates = {k: int(gids[k].max().item()) + 1 for k in gids}
    lines = [
        f"일봉 횡단면 상대수익률 예측: {len(stocks)} 종목, window={WINDOW}, horizons={HORIZONS}, "
        f"epochs={EPOCHS}, device={DEVICE}, 최소 그룹 {MIN_GROUP} 종목",
        f"windows: " + ", ".join(f"{k}={len(starts[k])} ({n_dates[k]}일)" for k in starts),
        "",
        f"{'model':>9} | " + " | ".join(f"R2@{k}" for k in HORIZONS)
        + " | IC@5 (t) | learned hl",
        "-" * 105,
    ]

    def fmt(name, r2, ic5=None, t5=None, hls=None):
        row = f"{name:>9} | " + " | ".join(f"{v:+.4f}" for v in r2.tolist())
        row += f" | {ic5:+.4f} ({t5:+.1f})" if ic5 is not None else " |     -"
        if hls:
            row += " | [" + ", ".join(f"{h:.1f}" for h in hls) + "]"
        return row

    r2_zero = r2_per_horizon(torch.zeros_like(y_n["test"]), y_n["test"])
    lines.append(fmt("A_zero", r2_zero))
    print(lines[-1])

    x_tr = linear_features(hocl_all, v_all, starts["train"])
    x_te = linear_features(hocl_all, v_all, starts["test"])
    pred_lin = ridge(x_tr, y_n["train"], x_te)
    ic, t = daily_ic(pred_lin, y_n["test"], gids["test"])
    lines.append(fmt("B_linear", r2_per_horizon(pred_lin, y_n["test"]), ic[-1].item(), t[-1].item()))
    print(lines[-1])

    configs = [
        ("C_flatten", lambda: FlattenForecaster()),
        ("D_bank", lambda: ReceptorBankForecaster(HL_K4, "pyramid")),
        ("E_dual", lambda: ReceptorBankForecaster(HL_K4, "pyramid", robust_clip=3.0, robust_dual=True)),
    ]
    for name, factory in configs:
        torch.manual_seed(SEED)
        r2, hls, model = train_model(
            factory(), hocl_all, v_all, starts, y_n["train"], y_n["val"], y_n["test"]
        )
        model.eval()
        with torch.no_grad():
            preds = []
            for i in range(0, len(starts["test"]), 8192):
                h, v = make_inputs(hocl_all, v_all, starts["test"][i:i + 8192], WINDOW)
                preds.append(model(h, v))
        ic, t = daily_ic(torch.cat(preds), y_n["test"], gids["test"])
        lines.append(fmt(name, r2, ic[-1].item(), t[-1].item(), hls))
        print(lines[-1])

    report = "\n".join(lines)
    (RESULTS_DIR / "report_cross.txt").write_text(report, encoding="utf-8")
    print(f"\nsaved -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()
