"""Multi-stock FC 학습 orchestrator. Discord 보고 + TensorBoard."""
from __future__ import annotations

import time
from pathlib import Path

import torch

from candle_data_manager import CandleDataAPI

from experiments.receptor_verification.discord_notify import notify

from .cdm_data import LARGE_CAP, SMALL_CAP, load_multi_stock
from .evaluate import collect_receptor_outputs, derive_factors, FACTOR_NAMES
from .train_multi import MultiTrainConfig, train_multi_stock


def main():
    run_name = f"multi_stock_fc_{time.strftime('%y%m%d_%H%M%S')}"
    notify(f"Multi-stock FC 학습 시작 ({run_name}).\n"
           f"대형주 ~100 + 잡주 ~100, 5년 데이터, window=60.\n"
           f"TensorBoard: runs/{run_name}",
           prefix="[Multi-Stock | Start]")

    # 데이터 로드
    api = CandleDataAPI()
    tickers = list(set(LARGE_CAP + SMALL_CAP))
    notify(f"종목 fetch 시작 ({len(tickers)}개).", prefix="[Multi-Stock | Data]")
    t_data = time.time()
    stocks = load_multi_stock(tickers, api, years=5, min_candles=200)
    api.close()
    dt_data = time.time() - t_data

    success_tickers = [s[0] for s in stocks]
    total_windows = sum(max(0, len(s[1]) - 60) for s in stocks)
    notify(f"데이터 로드 완료: {len(stocks)}/{len(tickers)} 종목, "
           f"총 {total_windows} 윈도우 (학습+val+test). {dt_data:.1f}s.\n"
           f"성공 종목 sample: {success_tickers[:20]}",
           prefix="[Multi-Stock | Data]")

    # 학습
    cfg = MultiTrainConfig(run_name=run_name, max_epochs=100, batch_size=256)
    model, log_dir, dt_train = train_multi_stock(cfg, stocks)

    # Discord 학습 완료 + 평가
    device = next(model.parameters()).device

    # 첫 종목 (예: TSLA가 있으면 거기서) 으로 sample 평가
    if any(s[0] == "TSLA" for s in stocks):
        sample_stock = [s for s in stocks if s[0] == "TSLA"][0]
    else:
        sample_stock = stocks[0]
    ticker, log_hocl, log_v = sample_stock
    z, f = collect_receptor_outputs(model, log_hocl, log_v, cfg.window, device, subset="test")

    # Linear probing R²
    import numpy as np
    lines = [f"**Multi-stock FC 학습 완료** ({dt_train:.1f}s)", "",
             f"TensorBoard 결과: `tensorboard --logdir {log_dir.parent}` 후 `{log_dir.name}` 선택", "",
             f"**Sample ticker [{ticker}] disentanglement**:"]
    lines.append("| output | top factor | R² | second | gap |")
    lines.append("|---|---|---|---|---|")
    for i, oname in enumerate(["out_1", "out_2", "out_v"]):
        r2_list = []
        for k, fname in enumerate(FACTOR_NAMES):
            x = z[:, i:i+1]
            y = f[:, k]
            if x.std() < 1e-9:
                r2 = 0.0
            else:
                corr = np.corrcoef(x[:, 0], y)[0, 1]
                r2 = float(corr ** 2) if not np.isnan(corr) else 0.0
            r2_list.append((fname, r2))
        r2_list.sort(key=lambda t: -t[1])
        top = r2_list[0]
        second = r2_list[1]
        lines.append(f"| {oname} | {top[0]} | {top[1]:.3f} | {second[0]} ({second[1]:.3f}) | {top[1]-second[1]:.3f} |")

    notify("\n".join(lines), prefix="[Multi-Stock | 학습 완료]")

    # 체크포인트 저장
    ckpt_path = Path("experiments/receptor_real_data/results_multi") / f"{run_name}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "config": cfg.__dict__,
        "tickers_used": success_tickers,
        "duration_sec": dt_train,
    }, ckpt_path)
    notify(f"체크포인트 저장: {ckpt_path}", prefix="[Multi-Stock | Done]")


if __name__ == "__main__":
    main()
