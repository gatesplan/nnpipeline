"""학습 완료 후 multi-stock 모델의 확장 disentanglement 분석.

확장 candidate factor (12+개):
- 단일 candle 기반: upper_wick, lower_wick, body_signed, body_abs, range, volume_z
- 비율 기반: upper_wick_ratio, lower_wick_ratio, body_position, close_position
- 시계열 기반 (윈도우 마지막 캔들 기준):
  - return_from_open (당일 수익률)
  - rolling_volatility_5d, rolling_volatility_20d
  - cumulative_return_5d, cumulative_return_20d
  - direction (binary)

각 출력 × 각 factor의 R² 측정 후 Discord에 상세 해석 보고.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset

from candle_data_manager import CandleDataAPI

from experiments.receptor_verification.discord_notify import notify

from .cdm_data import LARGE_CAP, SMALL_CAP, load_multi_stock, build_multi_stock_dataset
from .data import WindowDataset, split_train_val_test
from .models import CandleForecaster


def derive_extended_factors(
    hocl: torch.Tensor, v: torch.Tensor
) -> Tuple[torch.Tensor, List[str]]:
    """확장 factor 산출. (B, N, 4), (B, N, 1) → (B, N, num_factors)."""
    H = hocl[..., 0]
    O = hocl[..., 1]
    C = hocl[..., 2]
    L = hocl[..., 3]
    V = v[..., 0]

    eps = 1e-8
    range_ = H - L

    upper_wick = H - torch.maximum(O, C)
    lower_wick = torch.minimum(O, C) - L
    body_signed = C - O
    body_abs = torch.abs(C - O)
    upper_wick_ratio = upper_wick / (range_ + eps)
    lower_wick_ratio = lower_wick / (range_ + eps)
    body_position = ((O + C) / 2 - L) / (range_ + eps)   # body 중심의 range 내 위치
    close_position = (C - L) / (range_ + eps)            # close의 range 내 위치
    direction = (C > O).float()                          # 1 if bull else 0
    return_intraday = C - O                              # 정규화된 수익률 (= body_signed)
    # 시계열 기반 — 윈도우 내 cumulative
    cumret_5d = C - torch.roll(C, shifts=5, dims=-1)
    cumret_5d[..., :5] = 0
    cumret_20d = C - torch.roll(C, shifts=20, dims=-1)
    cumret_20d[..., :20] = 0
    # 변동성 추정 (rolling range mean)
    range_5d = range_.unfold(-1, 5, 1).mean(dim=-1) if range_.shape[-1] >= 5 else range_
    # 단순화: 그냥 H-L 사용

    factors = torch.stack([
        upper_wick,           # 0
        lower_wick,           # 1
        body_signed,          # 2
        body_abs,             # 3
        range_,               # 4
        V,                    # 5 volume_z
        upper_wick_ratio,     # 6
        lower_wick_ratio,     # 7
        body_position,        # 8
        close_position,       # 9
        direction,            # 10
        cumret_5d,            # 11
        cumret_20d,           # 12
    ], dim=-1)

    names = [
        "upper_wick", "lower_wick", "body_signed", "body_abs", "range",
        "volume_z", "upper_wick_ratio", "lower_wick_ratio",
        "body_position", "close_position", "direction",
        "cumret_5d", "cumret_20d",
    ]
    return factors, names


def collect_outputs_and_factors(
    model, stocks, window: int, device: torch.device,
    batch_size: int = 256, subset: str = "test"
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    test_ds = build_multi_stock_dataset(stocks, window=window, forecast_step=1, split=subset)
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    z_all, f_all = [], []
    factor_names = None
    model.eval()
    with torch.no_grad():
        for hocl, v, _tgt, _ref in loader:
            hocl, v = hocl.to(device), v.to(device)
            z = model.receptor(hocl, v)
            f, names = derive_extended_factors(hocl, v)
            z_all.append(z.cpu().numpy())
            f_all.append(f.cpu().numpy())
            factor_names = names

    z_arr = np.concatenate(z_all, axis=0).reshape(-1, 3)
    f_arr = np.concatenate(f_all, axis=0).reshape(-1, len(factor_names))
    return z_arr, f_arr, factor_names


def r2_matrix(z: np.ndarray, f: np.ndarray, factor_names: List[str]) -> Dict:
    """3 outputs × N factors R² 매트릭스."""
    n_out = z.shape[1]
    result = {}
    for i in range(n_out):
        oname = ["out_1", "out_2", "out_v"][i]
        row = {}
        for k, fname in enumerate(factor_names):
            x = z[:, i]
            y = f[:, k]
            if x.std() < 1e-9 or np.isnan(y).any():
                r2 = 0.0
            else:
                corr = np.corrcoef(x, y)[0, 1]
                r2 = float(corr ** 2) if not np.isnan(corr) else 0.0
            row[fname] = r2
        result[oname] = row
    return result


def find_run_dir() -> Path:
    """가장 최근 multi_stock_fc run 찾기."""
    runs = sorted(Path("runs").glob("multi_stock_fc_*"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise FileNotFoundError("No multi_stock_fc run found in runs/")
    return runs[-1]


def find_ckpt() -> Path:
    """가장 최근 체크포인트 찾기."""
    ckpts = sorted(Path("experiments/receptor_real_data/results_multi").glob("multi_stock_fc_*.pt"),
                   key=lambda p: p.stat().st_mtime)
    if not ckpts:
        raise FileNotFoundError("No multi_stock_fc checkpoint found")
    return ckpts[-1]


def main():
    ckpt_path = find_ckpt()
    notify(f"학습 완료 모델 분석 시작.\n체크포인트: `{ckpt_path.name}`",
           prefix="[Multi-Stock | Analysis]")

    # 데이터 재로드 (동일 종목)
    api = CandleDataAPI()
    tickers = list(set(LARGE_CAP + SMALL_CAP))
    stocks = load_multi_stock(tickers, api, years=5, min_candles=200)
    api.close()

    # 모델 로드
    ckpt = torch.load(ckpt_path, weights_only=False)
    model = CandleForecaster(window=ckpt["config"]["window"])
    model.load_state_dict(ckpt["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Test set 분석
    notify("Test set에서 receptor 출력과 13개 factor 수집 중...",
           prefix="[Multi-Stock | Analysis]")
    z, f, factor_names = collect_outputs_and_factors(
        model, stocks, ckpt["config"]["window"], device, subset="test"
    )
    notify(f"Test 샘플: {z.shape[0]}개 (모든 종목 합산), "
           f"수집 완료. R² 매트릭스 계산.",
           prefix="[Multi-Stock | Analysis]")

    # R² 매트릭스
    r2 = r2_matrix(z, f, factor_names)

    # 각 출력의 top 5 factor 추출
    summary_lines = ["**Multi-stock 학습 종료 후 disentanglement 분석**", ""]
    summary_lines.append(
        f"Test set 합산 {z.shape[0]}개 샘플, {len(factor_names)}개 candidate factor 측정."
    )
    summary_lines.append("")

    interpretations = []

    for oname in ["out_1", "out_2", "out_v"]:
        row = r2[oname]
        sorted_factors = sorted(row.items(), key=lambda x: -x[1])
        top5 = sorted_factors[:5]

        summary_lines.append(f"### {oname} — 상위 5개 factor")
        summary_lines.append(f"| 순위 | factor | R² | 설명 |")
        summary_lines.append(f"|---|---|---|---|")
        for rank, (fname, r2_val) in enumerate(top5, 1):
            desc = _describe_factor(fname)
            summary_lines.append(f"| {rank} | {fname} | {r2_val:.4f} | {desc} |")
        summary_lines.append("")

        # 해석
        top1, top1_r2 = sorted_factors[0]
        top2_r2 = sorted_factors[1][1]
        gap = top1_r2 - top2_r2
        if top1_r2 < 0.05:
            interp = (f"{oname}는 13개 factor 중 어느 것과도 강한 상관이 없음 (최대 R²={top1_r2:.3f}). "
                      f"의미 있는 분해 미달성. 학습이 우리 정의 factor와 다른 정보를 인코딩 중.")
        elif top1_r2 < 0.3:
            interp = (f"{oname}는 '{top1}'와 약한 상관 (R²={top1_r2:.3f}). "
                      f"부분적으로만 그 factor를 인코딩. 다른 정보도 혼합.")
        elif top1_r2 < 0.7:
            interp = (f"{oname}는 '{top1}'와 중간 상관 (R²={top1_r2:.3f}). "
                      f"주로 그 factor를 인코딩하지만 완벽 분리는 아님. "
                      f"두 번째 factor와 격차 {gap:.3f}.")
        else:
            interp = (f"{oname}는 '{top1}'와 강한 상관 (R²={top1_r2:.3f}). "
                      f"명확하게 그 factor를 인코딩. 두 번째 factor와 격차 {gap:.3f}.")
        interpretations.append(f"**{oname} 해석**: {interp}")
        summary_lines.append(interpretations[-1])
        summary_lines.append("")

    # 종합 해석
    summary_lines.append("---")
    summary_lines.append("### 전체 종합 해석")
    summary_lines.append("")

    out1_top_r2 = sorted(r2["out_1"].values(), reverse=True)[0]
    out2_top_r2 = sorted(r2["out_2"].values(), reverse=True)[0]
    outv_top_r2 = sorted(r2["out_v"].values(), reverse=True)[0]

    summary_lines.append(f"- **out_v** (volume aspect): 최대 R² = {outv_top_r2:.3f}. "
                         f"{'volume과 강한 분리 유지' if outv_top_r2 > 0.7 else '약한 분리'}.")
    summary_lines.append(f"- **out_1** (의도: upper aspect): 최대 R² = {out1_top_r2:.3f}. "
                         f"{'강한 단일 factor 인코딩' if out1_top_r2 > 0.5 else '명확한 단일 factor 미발견 — 추상적 representation 학습 중일 가능성'}.")
    summary_lines.append(f"- **out_2** (의도: lower aspect): 최대 R² = {out2_top_r2:.3f}. "
                         f"{'단일 종목 dead 문제 해결됨' if out2_top_r2 > 0.05 else '여전히 약함'}.")
    summary_lines.append("")

    # 다음 단계 제안
    summary_lines.append("### 다음 단계 후보")
    if out1_top_r2 < 0.1 and out2_top_r2 < 0.1:
        summary_lines.append(
            "- out_1, out_2가 정의된 factor와 안 맞음. "
            "더 추상적 factor (cross-asset return, beta, momentum) 측정 필요."
        )
        summary_lines.append(
            "- 또는 receptor가 다음 캔들 예측에 유용한 representation을 학습 중이지만 "
            "그게 단순 candle shape factor가 아닐 수 있음."
        )
    else:
        summary_lines.append(
            f"- 최고 R² factor가 의미 있게 등장 — receptor 활용 가능."
        )

    notify("\n".join(summary_lines), prefix="[Multi-Stock | 분석 결과]")

    # JSON 저장
    out_path = Path("experiments/receptor_real_data/results_multi") / f"{ckpt_path.stem}_analysis.json"
    out_path.write_text(json.dumps({
        "r2_matrix": r2,
        "factor_names": factor_names,
        "n_test_samples": int(z.shape[0]),
        "ckpt_path": str(ckpt_path),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    notify(f"분석 결과 저장: `{out_path.name}`", prefix="[Multi-Stock | Done]")


def _describe_factor(fname: str) -> str:
    """factor 의미 한 줄 설명."""
    return {
        "upper_wick": "위꼬리 절대 길이 (H - max(O,C))",
        "lower_wick": "아래꼬리 절대 길이 (min(O,C) - L)",
        "body_signed": "방향 있는 body 크기 (C - O), 양수=양봉",
        "body_abs": "body 크기 절대값 |C - O|",
        "range": "캔들 전체 폭 (H - L)",
        "volume_z": "거래량 (rolling z-score)",
        "upper_wick_ratio": "위꼬리 / range 비율 (0~1)",
        "lower_wick_ratio": "아래꼬리 / range 비율 (0~1)",
        "body_position": "body 중심의 range 내 위치 (0=아래, 1=위)",
        "close_position": "close의 range 내 위치",
        "direction": "양봉 여부 (1 if C > O else 0)",
        "cumret_5d": "최근 5 캔들 누적 수익률",
        "cumret_20d": "최근 20 캔들 누적 수익률",
    }.get(fname, "")


if __name__ == "__main__":
    main()
