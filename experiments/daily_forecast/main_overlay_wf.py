"""vol-targeting overlay 의 walk-forward 기간 확장 검증.

연도별 fold: test = 해당 연도 (2020~2025+), val = 직전 1년, train = 그 이전 전부.
각 fold 에서 변동성 모델을 새로 학습하고, val 보정 상수를 산출해 test 에 적용.
2020 (급락 포함), 2022 (하락장) 등 국면별로 PRED 우위가 유지되는지 확인.

실행: python -m experiments.daily_forecast.main_overlay_wf
"""

from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

logger.remove()

import math

import torch

from candle_data_manager import CandleDataAPI

from experiments.decay_bank_forecast.main_decompose import HL_K4, ReceptorBankForecaster
from experiments.receptor_real_data.cdm_data import LARGE_CAP, SMALL_CAP
from experiments.daily_forecast.data import build_arrays, load_stocks
from experiments.daily_forecast.main import DEVICE, SEED, WINDOW, YEARS, train_model
from experiments.daily_forecast.main import HORIZONS
from experiments.daily_forecast.main_vol import make_vol_targets
from experiments.daily_forecast.main_overlay import LEV_MAX, LEV_MIN, TARGET_ANN_VOL, WARMUP_DAYS
from experiments.daily_forecast.main_voltarget import (
    MIN_STOCKS,
    next_day_returns,
    predict_test_sigma,
    trailing_sigma,
)

RESULTS_DIR = Path(__file__).parent / "results"

TEST_YEARS = (2020, 2021, 2022, 2023, 2024, 2025)
COST_PER_TURNOVER = 0.0010


def _ts(year: int) -> float:
    return datetime(year, 1, 1, tzinfo=timezone.utc).timestamp()


def day_series(model, hocl_all, v_all, ts_all, st, mu, sd):
    """분할의 날짜 시계열: EW 수익률, 종목평균 σ̂/σ_trail 비율, trailing·미래 포트폴리오 변동성."""
    sigma_pred = predict_test_sigma(model, hocl_all, v_all, st, mu, sd)
    sigma_trail = trailing_sigma(hocl_all, st, WINDOW)
    ret_next = next_day_returns(hocl_all, st, WINDOW)
    end_ts = ts_all[st + WINDOW - 1]

    uniq_ts, inv = torch.unique(end_ts, sorted=True, return_inverse=True)
    r_list, ratio_list = [], []
    for d in range(len(uniq_ts)):
        m = inv == d
        if int(m.sum().item()) < MIN_STOCKS:
            continue
        r_list.append(ret_next[m].mean().item())
        ratio_list.append((sigma_pred[m].mean() / (sigma_trail[m].mean() + 1e-12)).item())

    n = len(r_list)
    r_ew = torch.tensor(r_list)
    ratio = torch.tensor(ratio_list)
    trail_p = torch.tensor([
        r_ew[max(0, t - WARMUP_DAYS):t].std().item() * math.sqrt(252) if t >= 5 else float("nan")
        for t in range(n)
    ])
    fwd = torch.tensor([
        r_ew[t:t + 5].std().item() * math.sqrt(252) if t + 5 <= n else float("nan")
        for t in range(n)
    ])
    return r_ew, ratio, trail_p, fwd


def overlay_metrics(r_ew, levs):
    rets = torch.tensor([levs[i] * r_ew[WARMUP_DAYS + i].item() for i in range(len(levs))])
    levs_t = torch.tensor(levs)
    n = len(rets)
    roll = torch.stack([rets[i:i + 20].std() for i in range(max(1, n - 20))]) * math.sqrt(252)
    mae = (roll - TARGET_ANN_VOL).abs().mean().item()
    rstd = roll.std().item()
    ann_ret = rets.mean().item() * 252
    ann_vol = rets.std().item() * math.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")
    cum = torch.cumprod(1.0 + rets, dim=0)
    maxdd = ((cum - torch.cummax(cum, dim=0).values) / torch.cummax(cum, dim=0).values).min().item()
    return mae, rstd, sharpe, maxdd


def run_fold(api_stocks, year: int, last_year: bool):
    cutoff = (_ts(year - 1), _ts(year), None if last_year else _ts(year + 1))
    hocl_all, v_all, ts_all, starts, _ = build_arrays(
        api_stocks, WINDOW, max(HORIZONS), cutoff_ts=cutoff
    )
    if len(starts["test"]) == 0 or len(starts["val"]) == 0 or len(starts["train"]) < 5000:
        return None
    hocl_all, v_all = hocl_all.to(DEVICE), v_all.to(DEVICE)
    ts_all = ts_all.to(DEVICE)
    starts = {k: v.to(DEVICE) for k, v in starts.items()}

    ys = {k: make_vol_targets(hocl_all, starts[k], WINDOW, HORIZONS) for k in starts}
    mu, sd = ys["train"].mean(dim=0, keepdim=True), ys["train"].std(dim=0, keepdim=True)
    y_n = {k: (v - mu) / sd for k, v in ys.items()}
    torch.manual_seed(SEED)
    _, _, model = train_model(
        ReceptorBankForecaster(HL_K4, "pyramid"),
        hocl_all, v_all, starts, y_n["train"], y_n["val"], y_n["test"],
    )
    mu, sd = mu.to(DEVICE), sd.to(DEVICE)

    r_v, ratio_v, trail_v, fwd_v = day_series(model, hocl_all, v_all, ts_all, starts["val"], mu, sd)
    r_t, ratio_t, trail_t, fwd_t = day_series(model, hocl_all, v_all, ts_all, starts["test"], mu, sd)
    if len(r_t) < WARMUP_DAYS + 30 or len(r_v) < WARMUP_DAYS + 10:
        return None

    vv = torch.arange(WARMUP_DAYS, len(r_v) - 5)
    cal_t = math.exp((torch.log(fwd_v[vv]) - torch.log(trail_v[vv])).mean().item())
    cal_p = math.exp((torch.log(fwd_v[vv]) - torch.log(trail_v[vv] * ratio_v[vv])).mean().item())

    tv = torch.arange(WARMUP_DAYS, len(r_t) - 5)
    corr_t = torch.corrcoef(torch.stack([torch.log(trail_t[tv]), torch.log(fwd_t[tv])]))[0, 1].item()
    corr_p = torch.corrcoef(
        torch.stack([torch.log(trail_t[tv] * ratio_t[tv]), torch.log(fwd_t[tv])])
    )[0, 1].item()

    def lev(x):
        return min(max(TARGET_ANN_VOL / (x + 1e-12), LEV_MIN), LEV_MAX)

    n = len(r_t)
    lev_tr = [lev(trail_t[t].item() * cal_t) for t in range(WARMUP_DAYS, n)]
    lev_pr = [lev(trail_t[t].item() * ratio_t[t].item() * cal_p) for t in range(WARMUP_DAYS, n)]
    m_tr = overlay_metrics(r_t, lev_tr)
    m_pr = overlay_metrics(r_t, lev_pr)

    base_ann = r_t[WARMUP_DAYS:].mean().item() * 252
    base_vol = r_t[WARMUP_DAYS:].std().item() * math.sqrt(252)
    return {
        "year": year, "days": n - WARMUP_DAYS,
        "base_ann": base_ann, "base_vol": base_vol,
        "corr_t": corr_t, "corr_p": corr_p, "tr": m_tr, "pr": m_pr,
    }


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    api = CandleDataAPI()
    stocks = load_stocks(LARGE_CAP + SMALL_CAP, api, years=YEARS)

    lines = [
        f"walk-forward overlay: {len(stocks)} 종목, 연도별 fold (val=직전 1년, train=그 이전 전부), "
        f"목표 {TARGET_ANN_VOL:.0%}, 비용 10bp×|ΔL|",
        "",
        f"{'year':>5} | {'일수':>4} | {'시장수익':>8} | {'시장변동':>8} | {'corr T/P':>13} | "
        f"{'MAE T/P':>15} | {'rollstd T/P':>15} | {'Sharpe T/P':>12} | {'MDD T/P':>17}",
        "-" * 130,
    ]

    folds = []
    for i, year in enumerate(TEST_YEARS):
        res = run_fold(stocks, year, last_year=(i == len(TEST_YEARS) - 1))
        if res is None:
            lines.append(f"{year:>5} | 데이터 부족으로 생략")
            print(lines[-1])
            continue
        folds.append(res)
        lines.append(
            f"{res['year']:>5} | {res['days']:>4} | {res['base_ann']:>+8.1%} | {res['base_vol']:>8.1%} | "
            f"{res['corr_t']:>+.3f}/{res['corr_p']:>+.3f} | "
            f"{res['tr'][0]:>6.2%}/{res['pr'][0]:>6.2%} | "
            f"{res['tr'][1]:>6.2%}/{res['pr'][1]:>6.2%} | "
            f"{res['tr'][2]:>5.2f}/{res['pr'][2]:>5.2f} | "
            f"{res['tr'][3]:>+7.1%}/{res['pr'][3]:>+7.1%}"
        )
        print(lines[-1])

    if folds:
        import statistics as stats
        def avg(f):
            return stats.mean(f)
        lines += [
            "",
            f"fold 평균: corr TRAIL {avg([f['corr_t'] for f in folds]):+.3f} vs PRED {avg([f['corr_p'] for f in folds]):+.3f}",
            f"           MAE  TRAIL {avg([f['tr'][0] for f in folds]):.2%} vs PRED {avg([f['pr'][0] for f in folds]):.2%}",
            f"           rstd TRAIL {avg([f['tr'][1] for f in folds]):.2%} vs PRED {avg([f['pr'][1] for f in folds]):.2%}",
            f"PRED 승수 (fold 단위): corr {sum(f['corr_p'] > f['corr_t'] for f in folds)}/{len(folds)}, "
            f"MAE {sum(f['pr'][0] < f['tr'][0] for f in folds)}/{len(folds)}, "
            f"rollstd {sum(f['pr'][1] < f['tr'][1] for f in folds)}/{len(folds)}, "
            f"Sharpe {sum(f['pr'][2] > f['tr'][2] for f in folds)}/{len(folds)}",
        ]
        for ln in lines[-4:]:
            print(ln)

    report = "\n".join(lines)
    (RESULTS_DIR / "report_overlay_wf.txt").write_text(report, encoding="utf-8")
    print(f"\nsaved -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()
