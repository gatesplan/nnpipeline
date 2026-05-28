"""v3 실행 orchestrator. λ sweep + 평가 + 보고."""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .discord_notify import notify
from .evaluate import evaluate_all, verdict
from .main import _format_evaluation_summary
from .train_v3 import TrainConfigV3, train_v3
from .visualize import canonical_scatter, jacobian_heatmap, training_curves


RESULTS_ROOT = Path("experiments/receptor_verification/results_v3")
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)


def run_one(aux_lambda: float, seed: int = 1) -> dict:
    cfg = TrainConfigV3(aux_lambda=aux_lambda, seed=seed)
    notify(f"v3 학습 시작: λ={aux_lambda}, seed={seed}.",
           prefix="[Phase 5 | v3 sweep]")
    t0 = time.time()
    result = train_v3(cfg)
    dt_train = time.time() - t0

    out_dir = RESULTS_ROOT / f"lambda_{aux_lambda:.4f}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": asdict(cfg),
            "model_state": result.model.state_dict(),
            "train_losses": result.train_losses,
            "val_losses": result.val_losses,
            "train_aux_losses": result.train_aux_losses,
            "val_aux_losses": result.val_aux_losses,
        },
        out_dir / "checkpoint.pt",
    )
    training_curves(result.train_losses, result.val_losses,
                    out_dir / "training_curves.png")

    device = next(result.model.parameters()).device
    t1 = time.time()
    eval_res = evaluate_all(result.model, device)
    vd = verdict(eval_res)
    dt_eval = time.time() - t1

    payload = {
        "mig": eval_res.mig,
        "dci": eval_res.dci,
        "sap": eval_res.sap,
        "linear_probing": eval_res.linear_probing,
        "factorvae": eval_res.factorvae,
        "jacobian": eval_res.jacobian,
        "causal": eval_res.causal,
        "reconstruction": eval_res.reconstruction,
        "verdict": vd,
        "train_duration_sec": dt_train,
        "eval_duration_sec": dt_eval,
        "final_recon_loss_train": result.final_train_loss,
        "final_recon_loss_val": result.final_val_loss,
        "final_aux_loss_train": result.train_aux_losses[-1],
        "final_aux_loss_val": result.val_aux_losses[-1],
    }
    (out_dir / "evaluation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    canonical_scatter(result.model, device, out_dir / "canonical_scatter.png")
    jacobian_heatmap(eval_res.jacobian["jacobian_abs_mean"],
                     out_dir / "jacobian_heatmap.png")

    summary = _format_evaluation_summary(eval_res, vd)
    notify(
        f"**λ={aux_lambda}** ({dt_train:.1f}s 학습, {dt_eval:.1f}s 평가)\n"
        f"recon val={result.final_val_loss:.6f}, aux val={result.val_aux_losses[-1]:.6f}\n\n"
        + summary,
        prefix="[Phase 5 | v3 결과]",
    )
    return {"lambda": aux_lambda, "verdict": vd, "eval": eval_res, "payload": payload}


def main() -> None:
    notify("v3 sweep 시작: λ ∈ {0.01, 0.1, 1.0}.", prefix="[Phase 5 | v3 시작]")
    results = []
    for lam in [0.01, 0.1, 1.0]:
        results.append(run_one(lam, seed=1))

    # 최종 비교 보고
    cmp_lines = ["**v3 sweep 종합 비교**"]
    cmp_lines.append("")
    cmp_lines.append("| λ | outcome | MIG | DCI disent | SAP | LP up_diag | LP low_diag | LP up_off | LP low_off |")
    cmp_lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        ev = r["eval"]
        lp = ev.linear_probing
        cmp_lines.append(
            f"| {r['lambda']} | {r['verdict']['outcome']} | "
            f"{ev.mig['mig_mean']:.3f} | {ev.dci['disentanglement']:.3f} | {ev.sap['sap_mean']:.3f} | "
            f"{lp['out_1']['upper_wick']:.3f} | {lp['out_2']['lower_wick']:.3f} | "
            f"{lp['out_1']['lower_wick']:.3f} | {lp['out_2']['upper_wick']:.3f} |"
        )
    notify("\n".join(cmp_lines), prefix="[Phase 5 | v3 sweep 종합]")


if __name__ == "__main__":
    main()
