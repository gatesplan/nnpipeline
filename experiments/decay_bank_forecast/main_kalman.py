"""관측 기반 feasible bound (Kalman) 계산 + recep_K4_lin 발산 재확인.

oracle R² 는 **진짜 잠재 상태**를 아는 상한이라, 관측 (δ 시퀀스) 만으로 도달 가능한
상한이 아니다. 생성 모델을 아는 Kalman filter 로 관측 기반 최적 예측을 계산하면:

    capture(bank) ≈ capture(kalman)  →  남은 gap 은 아키텍처 손실이 아니라
                                        원리적으로 복원 불가능한 상태 불확실성

실행: python -m experiments.decay_bank_forecast.main_kalman
"""

import torch

from experiments.decay_bank_forecast.main import r2_per_horizon
from experiments.decay_bank_forecast.main_decompose import (
    HL_K4,
    HORIZONS,
    N_TRAIN,
    N_VAL,
    RESULTS_DIR,
    SEED,
    WINDOW,
    ReceptorBankForecaster,
    train_and_eval,
)
from experiments.decay_bank_forecast.synthesize import (
    COMP_STD,
    NOISE_STD,
    TRUE_HALF_LIVES,
    make_forecast_dataset,
    oracle_predictions,
)


def kalman_posterior_means(delta: torch.Tensor) -> torch.Tensor:
    """생성 모델을 아는 Kalman filter 로 창 마지막 시점 잠재 상태의 사후 평균 추정.

    delta: (B, n) 관측 증분. 반환: (B, J) 사후 평균 E[s[n] | δ_1..δ_n].
    공분산 갱신은 데이터 무관 (선형 가우시안) 이라 배치 공유.
    """
    phis = torch.tensor([2.0 ** (-1.0 / t) for t in TRUE_HALF_LIVES], dtype=torch.float64)
    J = len(phis)
    A = torch.diag(phis)
    Q = torch.diag((COMP_STD ** 2) * (1.0 - phis ** 2))
    H = torch.ones(1, J, dtype=torch.float64)
    R = torch.tensor([[NOISE_STD ** 2]], dtype=torch.float64)

    B, n = delta.shape
    y = delta.to(torch.float64)
    m = torch.zeros(B, J, dtype=torch.float64)
    P = torch.eye(J, dtype=torch.float64) * (COMP_STD ** 2)  # 정상 사전 분포

    eye = torch.eye(J, dtype=torch.float64)
    for t in range(n):
        # predict (t=0 은 정상 사전 그대로 관측 갱신)
        if t > 0:
            m = m @ A.T
            P = A @ P @ A.T + Q
        # update
        S = (H @ P @ H.T + R).item()
        K = (P @ H.T / S).squeeze(-1)                        # (J,)
        innov = y[:, t] - m.sum(dim=-1)                      # H m = Σ_j m_j
        m = m + innov.unsqueeze(-1) * K
        P = (eye - K.unsqueeze(-1) @ H) @ P
    return m.to(torch.float32)


def main():
    _, _, y_tr, _, _, _ = make_forecast_dataset(N_TRAIN, WINDOW, HORIZONS, seed=SEED, extras=True)
    hocl_va, v_va, y_va, oracle_va, delta_va, _ = make_forecast_dataset(
        N_VAL, WINDOW, HORIZONS, seed=SEED + 999, extras=True
    )
    mu, sd = y_tr.mean(dim=0, keepdim=True), y_tr.std(dim=0, keepdim=True)
    y_va_n = (y_va - mu) / sd
    r2_oracle = r2_per_horizon((oracle_va - mu) / sd, y_va_n)

    m_hat = kalman_posterior_means(delta_va)
    kalman_pred = oracle_predictions(m_hat, HORIZONS)
    r2_kalman = r2_per_horizon((kalman_pred - mu) / sd, y_va_n)

    lines = [
        "관측 기반 feasible bound (Kalman, 생성 모델 완전 인지):",
        f"{'horizon':>8} | {'oracle R2':>9} | {'kalman R2':>9} | {'kalman/oracle':>13}",
        "-" * 48,
    ]
    for j, k in enumerate(HORIZONS):
        lines.append(
            f"{k:>8} | {r2_oracle[j].item():>9.4f} | {r2_kalman[j].item():>9.4f} | "
            f"{r2_kalman[j].item() / r2_oracle[j].item():>13.1%}"
        )

    # recep_K4_lin 발산 재확인 — seed 만 변경
    hocl_tr, v_tr, y_tr2, _, _, _ = make_forecast_dataset(N_TRAIN, WINDOW, HORIZONS, seed=SEED, extras=True)
    y_tr_n = (y_tr2 - mu) / sd
    torch.manual_seed(SEED + 1)
    model = ReceptorBankForecaster(HL_K4, "linear")
    r2_rerun, hls = train_and_eval(model, (hocl_tr, v_tr), (hocl_va, v_va), y_tr_n, y_va_n)
    lines += [
        "",
        f"recep_K4_lin rerun (seed+1): R2@1={r2_rerun[0].item():.3f}, "
        f"R2@5={r2_rerun[-1].item():.3f}, hl={[f'{h:.1f}' for h in hls]}",
    ]

    report = "\n".join(lines)
    print(report)
    (RESULTS_DIR / "report_kalman.txt").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
