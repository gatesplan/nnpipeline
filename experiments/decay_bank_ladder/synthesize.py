"""지저분함 사다리 (messiness ladder) 합성 생성기.

실데이터의 정형화 사실들을 알려진 구성 요소로 층층이 쌓되, oracle 을 유지한다:

  L0 base:     다중 시간스케일 AR(1) 드리프트 + 가우시안 잡음 (기존 forecast 실험과 동일 구조)
  L1 stochvol: log 변동성 g_t 가 자체 AR(1) — 변동성 군집 + (혼합효과로) fat tail
  L2 regime:   2-상태 마르코프 레짐 — turbulent 에서 변동성 ×2, 드리프트 혁신 ×1.5
  L3 fat:      잡음을 Student-t(df=4) 로 + 점프 (확률 p 로 큰 정규 잡음 대체)

관측: δ_t = M_t · (Σ_j s_j[t] + ε_t),  M_t = exp(g_t) · f(r_t)

oracle 이 전 층에서 닫힌 형태인 이유 — 조건부 기대가 인수분해된다:
  E[δ_{t+i} | state_t] = (Σ_j φ_j^i s_j[t]) · E[exp(g_{t+i})|g_t] · E[f(r_{t+i})|r_t]
  (드리프트 혁신·ε·점프 모두 평균 0, g 는 로그정규 닫힌 형태, r 은 전이행렬 거듭제곱,
   g 와 r 과 드리프트 혁신은 상호 독립)
"""

import math

import torch
from torch.distributions import Normal, StudentT

# 드리프트 (기존 forecast 실험과 동일)
TRUE_HALF_LIVES = (4.0, 16.0, 64.0)
COMP_STD = 0.002
NOISE_STD = 0.004

# L1: stochastic volatility
VOL_HALF_LIFE = 24.0
VOL_LOGSTD = 0.5          # log 변동성의 정상 표준편차 (승수 범위 대략 e^±0.5)

# L2: regime switching
REG_STAY_CALM = 1.0 - 1.0 / 200.0   # calm 평균 지속 200 봉
REG_STAY_TURB = 1.0 - 1.0 / 60.0    # turbulent 평균 지속 60 봉
REG_VOL_BOOST = 2.0
REG_DRIFT_BOOST = 1.5

# L3: fat tails + jumps
T_DF = 4.0
JUMP_P = 0.02
JUMP_STD_MULT = 6.0


class LadderConfig:

    def __init__(self, stochvol: bool = False, regime: bool = False, fat: bool = False):
        self.stochvol = stochvol
        self.regime = regime
        self.fat = fat


LEVELS = [
    ("L0_base", LadderConfig()),
    ("L1_stochvol", LadderConfig(stochvol=True)),
    ("L2_regime", LadderConfig(stochvol=True, regime=True)),
    ("L3_fat", LadderConfig(stochvol=True, regime=True, fat=True)),
]

_PHIS = torch.tensor([2.0 ** (-1.0 / t) for t in TRUE_HALF_LIVES])
_PHI_G = 2.0 ** (-1.0 / VOL_HALF_LIFE)
_TRANS = torch.tensor([
    [REG_STAY_CALM, 1.0 - REG_STAY_CALM],
    [1.0 - REG_STAY_TURB, REG_STAY_TURB],
])
_TURB_STATIONARY = (1.0 - REG_STAY_CALM) / ((1.0 - REG_STAY_CALM) + (1.0 - REG_STAY_TURB))


def _drift_innov_std(r: torch.Tensor, cfg: LadderConfig) -> torch.Tensor:
    # r: (...,) 0/1 → (..., J) 혁신 표준편차
    base = COMP_STD * torch.sqrt(1.0 - _PHIS ** 2)            # (J,)
    if not cfg.regime:
        return base.expand(*r.shape, len(_PHIS))
    boost = torch.where(r.bool(), REG_DRIFT_BOOST, 1.0)
    return base * boost.unsqueeze(-1)


def _vol_multiplier(g: torch.Tensor, r: torch.Tensor, cfg: LadderConfig) -> torch.Tensor:
    m = torch.exp(g) if cfg.stochvol else torch.ones_like(g)
    if cfg.regime:
        m = m * torch.where(r.bool(), REG_VOL_BOOST, 1.0)
    return m


def _sample_eps(shape, cfg: LadderConfig) -> torch.Tensor:
    if not cfg.fat:
        return torch.randn(shape) * NOISE_STD
    t_scale = NOISE_STD * math.sqrt((T_DF - 2.0) / T_DF)      # 분산을 NOISE_STD² 로 맞춤
    eps_t = StudentT(T_DF).sample(shape) * t_scale
    eps_j = torch.randn(shape) * (NOISE_STD * JUMP_STD_MULT)
    is_jump = torch.rand(shape) < JUMP_P
    return torch.where(is_jump, eps_j, eps_t)


def eps_log_prob(x: torch.Tensor, cfg: LadderConfig) -> torch.Tensor:
    """ε 의 로그 밀도 (particle filter 가중치용)."""
    if not cfg.fat:
        return Normal(0.0, NOISE_STD).log_prob(x)
    t_scale = NOISE_STD * math.sqrt((T_DF - 2.0) / T_DF)
    lt = StudentT(T_DF, 0.0, t_scale).log_prob(x) + math.log(1.0 - JUMP_P)
    lj = Normal(0.0, NOISE_STD * JUMP_STD_MULT).log_prob(x) + math.log(JUMP_P)
    return torch.logaddexp(lt, lj)


def _step(s, g, r, cfg: LadderConfig):
    """상태 1 스텝 전이. s (..., J), g (...), r (...) → 갱신된 (s, g, r)."""
    if cfg.regime:
        p_turb = torch.where(r.bool(), REG_STAY_TURB, 1.0 - REG_STAY_CALM)
        r = (torch.rand(r.shape) < p_turb).to(r.dtype)
    if cfg.stochvol:
        g = _PHI_G * g + torch.randn(g.shape) * (VOL_LOGSTD * math.sqrt(1.0 - _PHI_G ** 2))
    s = _PHIS * s + torch.randn(s.shape) * _drift_innov_std(r, cfg)
    return s, g, r


def _init_state(shape, cfg: LadderConfig):
    s = torch.randn(*shape, len(_PHIS)) * COMP_STD
    g = torch.randn(shape) * VOL_LOGSTD if cfg.stochvol else torch.zeros(shape)
    r = (torch.rand(shape) < _TURB_STATIONARY).float() if cfg.regime else torch.zeros(shape)
    return s, g, r


def make_series(cfg: LadderConfig, n_samples: int, n_total: int, seed: int = 0):
    """반환: (delta (B, n), s_traj (B, n, J), g_traj (B, n), r_traj (B, n))."""
    torch.manual_seed(seed)
    s, g, r = _init_state((n_samples,), cfg)
    deltas, ss, gs, rs = [], [], [], []
    for t in range(n_total):
        if t > 0:
            s, g, r = _step(s, g, r, cfg)
        m = _vol_multiplier(g, r, cfg)
        eps = _sample_eps((n_samples,), cfg)
        deltas.append(m * (s.sum(dim=-1) + eps))
        ss.append(s)
        gs.append(g)
        rs.append(r)
    return (
        torch.stack(deltas, dim=1),
        torch.stack(ss, dim=1),
        torch.stack(gs, dim=1),
        torch.stack(rs, dim=1),
    )


def oracle_ladder(
    s: torch.Tensor, g: torch.Tensor, r: torch.Tensor, cfg: LadderConfig, horizons: tuple
) -> torch.Tensor:
    """닫힌 형태 oracle: E[Σ_{i=1..k} δ_{t+i} | s, g, r]. 반환 (B, len(horizons))."""
    max_h = max(horizons)
    reg_f = torch.tensor([1.0, REG_VOL_BOOST])
    trans_pow = _TRANS.clone()

    step_means = []
    for i in range(1, max_h + 1):
        drift = (s * _PHIS ** i).sum(dim=-1)                          # (B,)
        vol = torch.ones_like(g)
        if cfg.stochvol:
            var_i = (VOL_LOGSTD ** 2) * (1.0 - _PHI_G ** (2 * i))
            vol = torch.exp((_PHI_G ** i) * g + 0.5 * var_i)
        reg = torch.ones_like(g)
        if cfg.regime:
            reg = (trans_pow @ reg_f)[r.long()]
            trans_pow = trans_pow @ _TRANS
        step_means.append(drift * vol * reg)
    cum = torch.cumsum(torch.stack(step_means, dim=-1), dim=-1)       # (B, max_h)
    return torch.stack([cum[:, k - 1] for k in horizons], dim=-1)


def mc_oracle(
    s: torch.Tensor, g: torch.Tensor, r: torch.Tensor, cfg: LadderConfig,
    horizons: tuple, n_paths: int = 2000, seed: int = 0,
) -> torch.Tensor:
    """몬테카를로 oracle (닫힌 형태 교차검증용). 반환 (B, len(horizons))."""
    torch.manual_seed(seed)
    B = s.shape[0]
    sp = s.unsqueeze(1).expand(B, n_paths, -1).contiguous()
    gp = g.unsqueeze(1).expand(B, n_paths).contiguous()
    rp = r.unsqueeze(1).expand(B, n_paths).contiguous()
    sums, cum = [], torch.zeros(B, n_paths)
    for _ in range(max(horizons)):
        sp, gp, rp = _step(sp, gp, rp, cfg)
        m = _vol_multiplier(gp, rp, cfg)
        eps = _sample_eps((B, n_paths), cfg)
        cum = cum + m * (sp.sum(dim=-1) + eps)
        sums.append(cum.mean(dim=1))
    cum_means = torch.stack(sums, dim=-1)                             # (B, max_h)
    return torch.stack([cum_means[:, k - 1] for k in horizons], dim=-1)


def particle_filter_predict(
    delta: torch.Tensor, cfg: LadderConfig, horizons: tuple,
    n_particles: int = 512, seed: int = 0,
) -> torch.Tensor:
    """진짜 생성 모델을 아는 bootstrap particle filter 로 관측 기반 최적 예측 근사.

    delta: (B, n) 관측 증분. 반환 (B, len(horizons)).
    매 스텝 systematic resampling → 마지막 스텝 후 입자 균등 가중, oracle 닫힌 형태로 예측 평균.
    """
    torch.manual_seed(seed)
    B, n = delta.shape
    Np = n_particles
    s, g, r = _init_state((B, Np), cfg)

    for t in range(n):
        if t > 0:
            s, g, r = _step(s, g, r, cfg)
        m = _vol_multiplier(g, r, cfg)
        x = delta[:, t].unsqueeze(1) / m - s.sum(dim=-1)              # (B, Np)
        logw = eps_log_prob(x, cfg) - torch.log(m)

        # systematic resampling
        w = torch.softmax(logw, dim=-1)
        cdf = torch.cumsum(w, dim=-1)
        u = (torch.rand(B, 1) + torch.arange(Np).unsqueeze(0)) / Np
        idx = torch.searchsorted(cdf, u).clamp(max=Np - 1)            # (B, Np)
        s = torch.gather(s, 1, idx.unsqueeze(-1).expand(-1, -1, s.shape[-1]))
        g = torch.gather(g, 1, idx)
        r = torch.gather(r, 1, idx)

    flat_pred = oracle_ladder(
        s.reshape(B * Np, -1), g.reshape(B * Np), r.reshape(B * Np), cfg, horizons
    ).reshape(B, Np, len(horizons))
    return flat_pred.mean(dim=1)
