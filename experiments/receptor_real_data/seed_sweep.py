"""MVIS FC × 3 활성함수 × 3 seed 재현성 확인."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from experiments.receptor_verification.discord_notify import notify

from .data import fetch_ohlcv, to_log_arrays
from .evaluate import evaluate_all, verdict
from .train import TrainConfig, train_forecast


RESULTS_DIR = Path("experiments/receptor_real_data/results_seed_sweep")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    notify("MVIS FC seed sweep 시작 (3 activations × 3 seeds = 9 runs).",
           prefix="[Seed Sweep | Start]")

    df = fetch_ohlcv("MVIS", period="5y")
    log_hocl, log_v = to_log_arrays(df)

    rows = []
    for activation in ["tanh", "leaky_relu", "relu"]:
        for seed in [1, 2, 3]:
            cfg = TrainConfig(
                ticker="MVIS", paradigm="forecast",
                comb_activation=activation, seed=seed,
            )
            result = train_forecast(cfg, log_hocl, log_v)
            device = next(result.model.parameters()).device
            eval_res = evaluate_all(result.model, log_hocl, log_v, cfg.window, device)
            vd = verdict(eval_res)
            row = {
                "activation": activation, "seed": seed,
                "out_1_top": vd["out_1"]["top_factor"],
                "out_1_r2": vd["out_1"]["top_r2"],
                "out_2_top": vd["out_2"]["top_factor"],
                "out_2_r2": vd["out_2"]["top_r2"],
                "out_v_top": vd["out_3"]["top_factor"],
                "out_v_r2": vd["out_3"]["top_r2"],
                "final_val_loss": result.final_val_loss,
            }
            rows.append(row)
            notify(
                f"{activation} seed={seed}: out_1 {row['out_1_top']} R²={row['out_1_r2']:.3f}, "
                f"out_v R²={row['out_v_r2']:.3f}, val_loss={row['final_val_loss']:.6f}",
                prefix="[Seed Sweep]",
            )

    # 활성함수별 평균·표준편차
    from statistics import mean, stdev
    summary_lines = ["**MVIS FC | activation × seed 재현성**", ""]
    summary_lines.append("| act | seed | out_1 top | out_1 R² | out_v R² | val_loss |")
    summary_lines.append("|---|---|---|---|---|---|")
    for r in rows:
        summary_lines.append(
            f"| {r['activation']} | {r['seed']} | {r['out_1_top']} | "
            f"{r['out_1_r2']:.3f} | {r['out_v_r2']:.3f} | {r['final_val_loss']:.6f} |"
        )

    summary_lines.append("")
    summary_lines.append("**활성함수별 out_1 R² 통계**")
    summary_lines.append("| act | mean | stdev | min | max |")
    summary_lines.append("|---|---|---|---|---|")
    for activation in ["tanh", "leaky_relu", "relu"]:
        vals = [r["out_1_r2"] for r in rows if r["activation"] == activation]
        summary_lines.append(
            f"| {activation} | {mean(vals):.3f} | {stdev(vals):.3f} | "
            f"{min(vals):.3f} | {max(vals):.3f} |"
        )

    summary_text = "\n".join(summary_lines)
    notify(summary_text, prefix="[Seed Sweep | 종합]")

    # 저장
    (RESULTS_DIR / "results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
