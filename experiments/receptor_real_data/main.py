"""자동 orchestrator. TSLA, MVIS × AE, FC 4 조합 실행. Discord 보고."""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from experiments.receptor_verification.discord_notify import notify

from .data import fetch_ohlcv, to_log_arrays
from .evaluate import evaluate_all, verdict
from .train import TrainConfig, train_autoencoder, train_forecast


RESULTS_DIR = Path("experiments/receptor_real_data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _format_eval(eval_res, vd, paradigm: str, ticker: str) -> str:
    lines = []
    lines.append(f"**{ticker} | {paradigm}** | n_test={eval_res['n_test_samples']}")
    lines.append("")
    lines.append("**Linear probing R² (out × factor)**:")
    for out_name, row in eval_res["linear_probing_r2"].items():
        line = f"  {out_name}: "
        line += " | ".join(f"{k}={v:.3f}" for k, v in row.items())
        lines.append(line)
    lines.append("")
    lines.append("**Verdict (max R² factor per output)**:")
    for out_name, v in vd.items():
        lines.append(
            f"  {out_name}: top={v['top_factor']} ({v['top_r2']:.3f}), "
            f"2nd={v['second_factor']} ({v['second_r2']:.3f}), gap={v['gap']:.3f}"
        )
    lines.append("")
    lines.append("**Jacobian |∂out/∂input| (last candle)**:")
    for out_name, row in eval_res["jacobian_abs_mean"].items():
        line = f"  {out_name}: "
        line += " | ".join(f"{k}={v:.3f}" for k, v in row.items())
        lines.append(line)
    return "\n".join(lines)


def run_one(ticker: str, paradigm: str, comb_activation: str = "leaky_relu") -> dict:
    tag = f"{ticker} {paradigm} [{comb_activation}]"
    notify(f"{tag}: 시작.", prefix=f"[Real Data | {tag}]")

    df = fetch_ohlcv(ticker, period="5y")
    log_hocl, log_v = to_log_arrays(df)

    cfg = TrainConfig(ticker=ticker, paradigm=paradigm, comb_activation=comb_activation)
    if paradigm == "autoencoder":
        result = train_autoencoder(cfg, log_hocl, log_v)
    elif paradigm == "forecast":
        result = train_forecast(cfg, log_hocl, log_v)
    else:
        raise ValueError(f"unknown paradigm: {paradigm}")

    notify(
        f"{tag}: 학습 완료 ({result.duration_sec:.1f}s).\n"
        f"final val_loss={result.final_val_loss:.6f}\n"
        f"per_channel val: {result.per_channel_val_losses[-1]}",
        prefix=f"[Real Data | {tag}]",
    )

    # 평가
    device = next(result.model.parameters()).device
    eval_res = evaluate_all(result.model, log_hocl, log_v, cfg.window, device)
    vd = verdict(eval_res)

    # 저장
    out_dir = RESULTS_DIR / f"{ticker}_{paradigm}_{comb_activation}"
    out_dir.mkdir(exist_ok=True)
    torch.save(
        {
            "config": asdict(cfg),
            "model_state": result.model.state_dict(),
            "train_losses": result.train_losses,
            "val_losses": result.val_losses,
            "per_channel_train_losses": result.per_channel_train_losses,
            "per_channel_val_losses": result.per_channel_val_losses,
        },
        out_dir / "checkpoint.pt",
    )
    (out_dir / "evaluation.json").write_text(
        json.dumps({"eval": eval_res, "verdict": vd}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    notify(_format_eval(eval_res, vd, paradigm, ticker),
           prefix=f"[Real Data | {tag} | Eval]")

    return {"ticker": ticker, "paradigm": paradigm, "comb_activation": comb_activation,
            "eval": eval_res, "verdict": vd}


def main():
    notify("Real Data Receptor 검증 시작. TSLA, MVIS × (AE, FC) × (leaky_relu, relu) — 총 8 runs. "
           "Linear_upper/lower는 tanh, 결합 layer 활성함수만 비교.",
           prefix="[Real Data Receptor | Start]")
    all_results = []
    for ticker in ["TSLA", "MVIS"]:
        for paradigm in ["autoencoder", "forecast"]:
            for activation in ["leaky_relu", "relu"]:
                all_results.append(run_one(ticker, paradigm, activation))

    # 종합 요약
    cmp_lines = ["**전체 verdict 비교**", ""]
    cmp_lines.append("| ticker | paradigm | act | out_1 top (R²) | out_2 top (R²) | out_v top (R²) |")
    cmp_lines.append("|---|---|---|---|---|---|")
    for r in all_results:
        vd = r["verdict"]
        cmp_lines.append(
            f"| {r['ticker']} | {r['paradigm']} | {r['comb_activation']} | "
            f"{vd['out_1']['top_factor']} ({vd['out_1']['top_r2']:.3f}) | "
            f"{vd['out_2']['top_factor']} ({vd['out_2']['top_r2']:.3f}) | "
            f"{vd['out_3']['top_factor']} ({vd['out_3']['top_r2']:.3f}) |"
        )
    notify("\n".join(cmp_lines), prefix="[Real Data Receptor | 종합]")

    # 종합 결과 저장
    (RESULTS_DIR / "summary.json").write_text(
        json.dumps([{"ticker": r["ticker"], "paradigm": r["paradigm"],
                     "comb_activation": r["comb_activation"],
                     "verdict": r["verdict"]} for r in all_results],
                   ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    notify("전체 종료. summary.json 저장.", prefix="[Real Data Receptor | Done]")


if __name__ == "__main__":
    main()
