"""Binance USDT 선물 (일봉, 40종목) 에서 주식 실험 일체 재검증.

과제: ① 방향 수익률 R² ② 실현변동성 R² ③ 횡단면 IC + 롱숏 백테스트
      ④ vol-targeting overlay (TRAIL vs PRED, val 보정)

주식 대비 조건 차이:
- 선물이므로 공매도가 차입 제약 없이 대칭 (funding rate 는 미모델, 양다리 간 대체로 상쇄)
- 수수료: Binance 선물 기본 (VIP0) taker 0.05% = 5bp × turnover
- 연환산 365일 (연중무휴 거래)
- 데이터 종료 시점: 2026-02 (분석 시점 대비 약 6개월 이전)

실행: python -m experiments.daily_forecast.main_crypto
"""

from pathlib import Path

from loguru import logger

logger.remove()

import math

import numpy as np
import torch

from candle_data_manager import CandleDataAPI

from experiments.decay_bank_forecast.main_decompose import HL_K4, ReceptorBankForecaster
from experiments.daily_forecast.data import build_arrays, make_inputs, make_targets
from experiments.daily_forecast.main import (
    DEVICE,
    HORIZONS,
    SEED,
    WINDOW,
    FlattenForecaster,
    linear_features,
    r2_per_horizon,
    train_model,
)
from experiments.daily_forecast.main_vol import make_vol_targets, ridge, vol_features
from experiments.daily_forecast.main_cross import cs_normalize, daily_ic
from experiments.daily_forecast.main_longshort import QUANTILE, backtest_ls
from experiments.daily_forecast.main_voltarget import (
    next_day_returns,
    predict_test_sigma,
    trailing_sigma,
)
from experiments.daily_forecast.report_notify import notify

RESULTS_DIR = Path(__file__).parent / "results"

ANN = 365
FEE = 0.0005          # Binance USDT 선물 VIP0 taker 5bp
MIN_GROUP_C = 15
TARGET_ANN_VOL = 0.30  # 크립토 변동성 수준에 맞춘 목표 (주식 15% 상당의 상대 위치)
LEV_MIN, LEV_MAX = 0.2, 2.0
WARMUP_DAYS = 20


def load_cryptos(api, min_candles: int = 600):
    markets = api.load(
        archetype="CRYPTO", exchange="BINANCE", tradetype="FUTURES",
        quote="USDT", timeframe="1d",
    )
    results = []
    for m in markets:
        df = m.candles
        df = df[["timestamp", "high", "open", "close", "low", "volume"]].copy()
        df = df[df["volume"] > 0]
        if len(df) < min_candles:
            continue
        ts = df["timestamp"].values.astype(np.int64)
        log_hocl = np.log(
            np.stack([df["high"].values, df["open"].values, df["close"].values, df["low"].values], axis=1)
        ).astype(np.float32)
        log_v = np.log(df["volume"].values).astype(np.float32)
        name = m.symbol.base if hasattr(m, "symbol") else "?"
        results.append((name, ts, log_hocl, log_v))
    return results


def fmt_r2(name, r2, hls=None):
    row = f"{name:>9} | " + " | ".join(f"{v:+.4f}" for v in r2.tolist())
    if hls:
        row += " | [" + ", ".join(f"{h:.1f}" for h in hls) + "]"
    return row


def ls_stats_crypto(name, series):
    long_r = torch.tensor(series["long"])
    short_r = torch.tensor(series["short"])
    ew_r = torch.tensor(series["ew"])
    to = torch.tensor(series["turnover"])
    ls = long_r - short_r
    ann_ret = ls.mean().item() * ANN
    ann_vol = ls.std().item() * math.sqrt(ANN)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")
    net = ls - to * FEE
    net_sharpe = (net.mean().item() * ANN) / (net.std().item() * math.sqrt(ANN) + 1e-12)
    cum = torch.cumprod(1.0 + ls, dim=0)
    maxdd = ((cum - torch.cummax(cum, dim=0).values) / torch.cummax(cum, dim=0).values).min().item()
    return (
        f"{name:>9} | {ann_ret:>+8.2%} | {ann_vol:>6.2%} | {sharpe:>6.2f} | {net_sharpe:>6.2f} | "
        f"{maxdd:>+7.2%} | {to.mean().item():>5.3f} | {(long_r - ew_r).mean().item() * ANN:>+8.2%} | "
        f"{(ew_r - short_r).mean().item() * ANN:>+8.2%} | {(-short_r).mean().item() * ANN:>+8.2%}"
    )


def overlay_section(model, hocl_all, v_all, ts_all, starts, mu, sd):
    def day_series(st):
        sigma_pred = predict_test_sigma(model, hocl_all, v_all, st, mu, sd)
        sigma_trail = trailing_sigma(hocl_all, st, WINDOW)
        ret_next = next_day_returns(hocl_all, st, WINDOW)
        end_ts = ts_all[st + WINDOW - 1]
        uniq_ts, inv = torch.unique(end_ts, sorted=True, return_inverse=True)
        r_list, ratio_list = [], []
        for d in range(len(uniq_ts)):
            m = inv == d
            if int(m.sum().item()) < MIN_GROUP_C:
                continue
            r_list.append(ret_next[m].mean().item())
            ratio_list.append((sigma_pred[m].mean() / (sigma_trail[m].mean() + 1e-12)).item())
        r_ew = torch.tensor(r_list)
        ratio = torch.tensor(ratio_list)
        n = len(r_ew)
        trail_p = torch.tensor([
            r_ew[max(0, t - WARMUP_DAYS):t].std().item() * math.sqrt(ANN) if t >= 5 else float("nan")
            for t in range(n)
        ])
        fwd = torch.tensor([
            r_ew[t:t + 5].std().item() * math.sqrt(ANN) if t + 5 <= n else float("nan")
            for t in range(n)
        ])
        return r_ew, ratio, trail_p, fwd

    r_v, ratio_v, trail_v, fwd_v = day_series(starts["val"])
    r_t, ratio_t, trail_t, fwd_t = day_series(starts["test"])
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
    rows = []
    for name, sig_fn in [
        ("BASE", None),
        ("TR_CAL", lambda t: trail_t[t].item() * cal_t),
        ("PR_CAL", lambda t: trail_t[t].item() * ratio_t[t].item() * cal_p),
    ]:
        levs = [1.0 if sig_fn is None else lev(sig_fn(t)) for t in range(WARMUP_DAYS, n)]
        rets = torch.tensor([levs[i] * r_t[WARMUP_DAYS + i].item() for i in range(len(levs))])
        levs_t = torch.tensor(levs)
        roll = torch.stack([rets[i:i + 20].std() for i in range(max(1, len(rets) - 20))]) * math.sqrt(ANN)
        mae = (roll - TARGET_ANN_VOL).abs().mean().item()
        rstd = roll.std().item()
        ann_ret = rets.mean().item() * ANN
        ann_vol = rets.std().item() * math.sqrt(ANN)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")
        dl = torch.cat([levs_t[:1] * 0, (levs_t[1:] - levs_t[:-1]).abs()])
        net = rets - dl * FEE
        net_sharpe = (net.mean().item() * ANN) / (net.std().item() * math.sqrt(ANN) + 1e-12)
        cum = torch.cumprod(1.0 + rets, dim=0)
        maxdd = ((cum - torch.cummax(cum, dim=0).values) / torch.cummax(cum, dim=0).values).min().item()
        rows.append(
            f"{name:>6} | {ann_vol:>7.2%} | {mae:>7.3%} | {rstd:>7.3%} | {ann_ret:>+8.2%} | "
            f"{sharpe:>6.2f} | {net_sharpe:>6.2f} | {maxdd:>+7.2%}"
        )
    header = (
        f"overlay (목표 {TARGET_ANN_VOL:.0%}): corr TRAIL {corr_t:+.3f} vs PRED {corr_p:+.3f}\n"
        f"{'scheme':>6} | {'연변동':>7} | {'목표MAE':>7} | {'롤링std':>7} | {'연수익':>8} | "
        f"{'Sharpe':>6} | {'netShp':>6} | {'MDD':>7}"
    )
    return header + "\n" + "\n".join(rows)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    torch.manual_seed(SEED)

    api = CandleDataAPI()
    cryptos = load_cryptos(api)
    hocl_all, v_all, ts_all, starts, tick_ids = build_arrays(cryptos, WINDOW, max(HORIZONS))
    hocl_all, v_all = hocl_all.to(DEVICE), v_all.to(DEVICE)
    ts_all = ts_all.to(DEVICE)
    starts = {k: v.to(DEVICE) for k, v in starts.items()}
    tick_ids = {k: v.to(DEVICE) for k, v in tick_ids.items()}

    lines = [
        f"Binance USDT 선물 일봉 검증: {len(cryptos)} 종목, window={WINDOW}, "
        f"수수료 {FEE*1e4:.0f}bp (선물 VIP0 taker), 연환산 {ANN}일",
        f"windows: " + ", ".join(f"{k}={len(starts[k])}" for k in starts),
        "",
        "== ① 방향 수익률 R² (1~5봉 누적) ==",
    ]

    # ① 방향
    ys = {k: make_targets(hocl_all, starts[k], WINDOW, HORIZONS) for k in starts}
    mu_d, sd_d = ys["train"].mean(dim=0, keepdim=True), ys["train"].std(dim=0, keepdim=True)
    y_n = {k: (v - mu_d) / sd_d for k, v in ys.items()}
    lines.append(fmt_r2("A_zero", r2_per_horizon(torch.zeros_like(y_n["test"]), y_n["test"])))
    x_tr = linear_features(hocl_all, v_all, starts["train"])
    x_te = linear_features(hocl_all, v_all, starts["test"])
    lines.append(fmt_r2("B_linear", r2_per_horizon(ridge(x_tr, y_n["train"], x_te), y_n["test"])))
    torch.manual_seed(SEED)
    r2, hls, _ = train_model(
        ReceptorBankForecaster(HL_K4, "pyramid"),
        hocl_all, v_all, starts, y_n["train"], y_n["val"], y_n["test"],
    )
    lines.append(fmt_r2("D_bank", r2, hls))
    print("\n".join(lines[-3:]))

    # ② 변동성
    lines += ["", "== ② 실현변동성 R² (1~5봉 log RV) =="]
    ys = {k: make_vol_targets(hocl_all, starts[k], WINDOW, HORIZONS) for k in starts}
    mu_v, sd_v = ys["train"].mean(dim=0, keepdim=True), ys["train"].std(dim=0, keepdim=True)
    y_n = {k: (v - mu_v) / sd_v for k, v in ys.items()}
    lines.append(fmt_r2("A_zero", r2_per_horizon(torch.zeros_like(y_n["test"]), y_n["test"])))
    xv_tr = vol_features(hocl_all, v_all, starts["train"], WINDOW)
    xv_te = vol_features(hocl_all, v_all, starts["test"], WINDOW)
    lines.append(fmt_r2("B_linear", r2_per_horizon(ridge(xv_tr, y_n["train"], xv_te), y_n["test"])))
    torch.manual_seed(SEED)
    r2c, _, _ = train_model(
        FlattenForecaster(), hocl_all, v_all, starts, y_n["train"], y_n["val"], y_n["test"]
    )
    lines.append(fmt_r2("C_flatten", r2c))
    torch.manual_seed(SEED)
    r2v, hlsv, vol_model = train_model(
        ReceptorBankForecaster(HL_K4, "pyramid"),
        hocl_all, v_all, starts, y_n["train"], y_n["val"], y_n["test"],
    )
    lines.append(fmt_r2("D_bank", r2v, hlsv))
    print("\n".join(lines[-4:]))

    # ③ 횡단면 + 롱숏
    lines += ["", f"== ③ 횡단면 IC + 롱숏 (상/하위 {QUANTILE:.0%}, 달러중립, {FEE*1e4:.0f}bp) =="]
    date_ids = {}
    for k in starts:
        end_ts = ts_all[starts[k] + WINDOW - 1]
        _, date_ids[k] = torch.unique(end_ts, return_inverse=True)
    keep = {}
    for k in starts:
        cnt = torch.bincount(date_ids[k])
        keep[k] = cnt[date_ids[k]] >= MIN_GROUP_C
    st_c = {k: starts[k][keep[k]] for k in starts}
    tid_c = {k: tick_ids[k][keep[k]] for k in tick_ids}
    gids = {}
    for k in st_c:
        end_ts = ts_all[st_c[k] + WINDOW - 1]
        _, gids[k] = torch.unique(end_ts, sorted=True, return_inverse=True)
    y_cs = {}
    for k in st_c:
        y_raw = make_targets(hocl_all, st_c[k], WINDOW, HORIZONS)
        y_cs[k] = cs_normalize(y_raw, gids[k])
    ret_next = next_day_returns(hocl_all, st_c["test"], WINDOW)
    n_dates = int(gids["test"].max().item()) + 1

    xc_tr = linear_features(hocl_all, v_all, st_c["train"])
    xc_te = linear_features(hocl_all, v_all, st_c["test"])
    pred_lin = ridge(xc_tr, y_cs["train"], xc_te)
    ic_l, t_l = daily_ic(pred_lin, y_cs["test"], gids["test"])
    series = backtest_ls(pred_lin[:, 0], ret_next, tid_c["test"], gids["test"], n_dates)
    lines.append(
        f"{'model':>9} | {'연수익':>8} | {'연변동':>6} | {'Sharpe':>6} | {'netShp':>6} | "
        f"{'MDD':>7} | {'턴오버':>5} | {'롱알파':>8} | {'숏알파':>8} | {'숏단독PL':>8}"
    )
    lines.append(ls_stats_crypto("LS_linear", series))
    torch.manual_seed(SEED)
    _, hls_cs, cs_model = train_model(
        ReceptorBankForecaster(HL_K4, "pyramid"),
        hocl_all, v_all, st_c, y_cs["train"], y_cs["val"], y_cs["test"],
    )
    cs_model.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, len(st_c["test"]), 8192):
            h, v = make_inputs(hocl_all, v_all, st_c["test"][i:i + 8192], WINDOW)
            preds.append(cs_model(h, v))
    pred_bank = torch.cat(preds)
    ic_b, t_b = daily_ic(pred_bank, y_cs["test"], gids["test"])
    series = backtest_ls(pred_bank[:, 0], ret_next, tid_c["test"], gids["test"], n_dates)
    lines.append(ls_stats_crypto("LS_bank", series))
    lines.append(
        f"IC@1 (t): linear {ic_l[0].item():+.4f} ({t_l[0].item():+.1f}), "
        f"bank {ic_b[0].item():+.4f} ({t_b[0].item():+.1f}) / "
        f"IC@5 (t): linear {ic_l[-1].item():+.4f} ({t_l[-1].item():+.1f}), "
        f"bank {ic_b[-1].item():+.4f} ({t_b[-1].item():+.1f})"
    )
    print("\n".join(lines[-4:]))

    # ④ overlay
    lines += ["", "== ④ vol-targeting overlay (EW 크립토 바스켓) =="]
    lines.append(overlay_section(
        vol_model, hocl_all, v_all, ts_all, starts, mu_v.to(DEVICE), sd_v.to(DEVICE)
    ))
    print(lines[-1])

    report = "\n".join(lines)
    (RESULTS_DIR / "report_crypto.txt").write_text(report, encoding="utf-8")
    print(f"\nsaved -> {RESULTS_DIR}")
    notify("크립토 검증 완료 — 상세는 다음 보고에서.", prefix="[nnpipeline]")


if __name__ == "__main__":
    main()
