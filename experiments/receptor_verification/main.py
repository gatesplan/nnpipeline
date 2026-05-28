"""Autoprogress orchestrator.

Phase 1 (인프라) → 2 (학습) → 3 (평가) → 4 (보고/판정).
각 phase 시작·종료를 Discord로 보고한다.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .discord_notify import notify
from .evaluate import evaluate_all, verdict
from .train import TrainConfig, train
from .visualize import canonical_scatter, jacobian_heatmap, training_curves


import os

RESULTS_DIR = Path(os.environ.get(
    "RECEPTOR_RESULTS_DIR",
    "experiments/receptor_verification/results",
))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _format_evaluation_summary(eval_res, vd) -> str:
    lines = []
    lines.append(f"**판정: {vd['outcome']}** ({vd['n_pass']}/4 통과)")
    lines.append("")
    lines.append("**조건 충족:**")
    for k, v in vd["conditions"].items():
        lines.append(f"- {k}: {'✓' if v else '✗'}")
    lines.append("")
    lines.append(f"**MIG**: mean={eval_res.mig['mig_mean']:.4f}")
    for k, v in eval_res.mig["per_factor"].items():
        lines.append(f"  - {k}: {v:.4f}")
    lines.append("")
    lines.append(f"**DCI**: disent={eval_res.dci['disentanglement']:.4f}, "
                 f"complete={eval_res.dci['completeness']:.4f}, "
                 f"info_R²={eval_res.dci['informativeness_mean_r2']:.4f}")
    lines.append("")
    lines.append(f"**SAP**: mean={eval_res.sap['sap_mean']:.4f}")
    lines.append("")
    lines.append(f"**FactorVAE**: acc={eval_res.factorvae['factorvae_mean_accuracy']:.4f}")
    lines.append("")
    lines.append(f"**Linear probing R²** (out_i × factor):")
    for out_name, row in eval_res.linear_probing.items():
        line = f"  {out_name}: "
        line += ", ".join(f"{k}={v:.3f}" for k, v in row.items())
        lines.append(line)
    lines.append("")
    lines.append(f"**Jacobian |∂out_i/∂x_j| 평균**:")
    for out_name, row in eval_res.jacobian["jacobian_abs_mean"].items():
        line = f"  {out_name}: "
        line += ", ".join(f"{k}={v:.4f}" for k, v in row.items())
        lines.append(line)
    lines.append(f"  SeparationScore = {eval_res.jacobian['separation_score']:.4f}")
    lines.append("")
    lines.append("**Reconstruction MSE per channel** (정규화 후 단위):")
    for k, row in eval_res.reconstruction.items():
        line = f"  {k}: "
        line += ", ".join(f"{ch}={v:.4f}" for ch, v in row.items())
        lines.append(line)
    return "\n".join(lines)


def run_phase1() -> None:
    notify(
        "Phase 1: 인프라 구현 완료. discord_notify, synthesize, receptor, "
        "train, evaluate, visualize, main 모듈 생성. PyTorch + CUDA 사용 가능.",
        prefix="[Phase 1 | 완료]",
    )


def run_phase2() -> dict:
    notify("Phase 2: 학습 시작. cfg: hidden=16, lr=1e-3, max_epochs=100, "
           "train_n=50000, val_n=10000, seed=1.", prefix="[Phase 2 | 시작]")
    cfg = TrainConfig()
    t0 = time.time()
    result = train(cfg)
    dt = time.time() - t0

    # 저장
    torch.save(
        {
            "config": asdict(cfg),
            "model_state": result.model.state_dict(),
            "train_losses": result.train_losses,
            "val_losses": result.val_losses,
        },
        RESULTS_DIR / "checkpoint.pt",
    )

    training_curves(result.train_losses, result.val_losses,
                    RESULTS_DIR / "training_curves.png")

    notify(
        f"Phase 2: 학습 완료. {dt:.1f}s 소요. "
        f"final train_loss={result.final_train_loss:.6f}, "
        f"val_loss={result.final_val_loss:.6f}. "
        f"체크포인트와 학습곡선 저장됨.",
        prefix="[Phase 2 | 완료]",
    )
    return {
        "config": cfg,
        "model": result.model,
        "train_losses": result.train_losses,
        "val_losses": result.val_losses,
        "duration_sec": dt,
    }


def run_phase3(model: torch.nn.Module) -> tuple:
    notify("Phase 3: 다중지표 평가 시작. MIG, DCI, SAP, FactorVAE, "
           "Linear probing, Jacobian, Causal intervention, Reconstruction.",
           prefix="[Phase 3 | 시작]")
    device = next(model.parameters()).device
    t0 = time.time()
    eval_res = evaluate_all(model, device)
    vd = verdict(eval_res)
    dt = time.time() - t0

    # 저장
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
        "duration_sec": dt,
    }
    (RESULTS_DIR / "evaluation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 시각화
    canonical_scatter(model, device, RESULTS_DIR / "canonical_scatter.png")
    jacobian_heatmap(eval_res.jacobian["jacobian_abs_mean"],
                     RESULTS_DIR / "jacobian_heatmap.png")

    notify(f"Phase 3: 평가 완료 ({dt:.1f}s). 결과 JSON·시각화 저장.",
           prefix="[Phase 3 | 완료]")

    return eval_res, vd


def run_phase4(eval_res, vd) -> None:
    summary = _format_evaluation_summary(eval_res, vd)

    # 추가 권고
    if vd["outcome"] == "분리 성공":
        rec = "다음 단계: 척추 backbone 설계 및 V 채널 통합 검토."
    elif vd["outcome"] == "분리 실패":
        rec = ("실패. fallback (연구계획 §7): "
               "(1) auxiliary loss → (2) 구조적 inductive bias 강화 → "
               "(3) MLP 분리 → (4) 출력 차원 확장 → (5) 설계 폐기")
    else:
        rec = ("모호. 추가 분석: hyperparameter sweep (hidden ∈ {8,32}, "
               "lr ∈ {1e-2, 1e-4}), seed {2,3} 재현성 확인.")

    full_report = (
        f"{summary}\n\n"
        f"**권고**: {rec}\n"
        f"**산출물**: experiments/receptor_verification/results/ 참고"
    )
    notify(full_report, prefix="[Phase 4 | 보고 및 판정]")


def main() -> None:
    notify("자동진행 시작. Phase 1~4 순차 진행.", prefix="[Receptor Verification | Run]")
    run_phase1()
    phase2 = run_phase2()
    eval_res, vd = run_phase3(phase2["model"])
    run_phase4(eval_res, vd)
    notify("전체 종료.", prefix="[Receptor Verification | Done]")


if __name__ == "__main__":
    main()
