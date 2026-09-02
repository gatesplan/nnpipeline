"""과제 폭 측정: 이진 예측 2종 (꼬리위험 경보 / 변동성 증감 방향).

1. tail  — 향후 5봉 안에 종목별 문턱 (-2 × 직전 20봉 일변동성 × sqrt(경과일)) 을 밑도는
           누적 하락이 발생하는가. 급락 경보로서의 활용 가치 측정.
2. volchg — 향후 5봉 실현변동성이 직전 5봉 실현변동성보다 커지는가.
           변동성 수준(지속성)이 아닌 변화 방향 — 지속성 가정만으로는 못 맞히는 과제.

모델: A 기저율 / B logistic (변동성·수익률 특징) / D receptor→DecayBank→head (1 logit, BCE).
평가: test AUC.

실행: python -m experiments.daily_forecast.main_topics
"""

from pathlib import Path

from loguru import logger

logger.remove()

import torch
from torch import nn

from candle_data_manager import CandleDataAPI

from experiments.decay_bank_forecast.main_decompose import HL_K4
from experiments.receptor_real_data.cdm_data import LARGE_CAP, SMALL_CAP
from experiments.daily_forecast.data import build_arrays, load_stocks, make_inputs
from experiments.daily_forecast.main import DEVICE, EPOCHS, SEED, WINDOW, YEARS
from experiments.daily_forecast.main_vol import vol_features
from nnpipeline import OHLCVReceptor, Pyramid
from nnpipeline.prototype.decay_bank import DecayBank

RESULTS_DIR = Path(__file__).parent / "results"

BATCH = 4096
LR = 5e-3
LR_LAMBDA = 2e-2
LOOKAHEAD = 5
TAIL_SIGMA = 2.0


class BankClassifier(nn.Module):

    def __init__(self):
        super().__init__()
        self.receptor = OHLCVReceptor()
        self.bank = DecayBank(half_lives=HL_K4, learnable=True)
        self.head = Pyramid(self.bank.out_scales * 3, 1, depth=3, interlayer=[nn.LeakyReLU()])

    def forward(self, hocl, v):
        emb = self.receptor(hocl, v)
        return self.head(self.bank(emb).flatten(start_dim=-2))[..., 0]


def make_tail_labels(hocl_all, batch_starts, window: int):
    """향후 LOOKAHEAD 봉 내 -TAIL_SIGMA × σ_trail20 × sqrt(i) 이하 누적 하락 발생 여부."""
    logc = hocl_all[:, 2]
    end = batch_starts + window - 1
    idx20 = end.unsqueeze(-1) + torch.arange(-19, 1, device=end.device)
    sigma = (logc[idx20] - logc[idx20 - 1]).std(dim=1)                    # (B,)

    fut = end.unsqueeze(-1) + torch.arange(1, LOOKAHEAD + 1, device=end.device)
    cum = logc[fut] - logc[end].unsqueeze(-1)                             # (B, 5)
    scale = torch.sqrt(torch.arange(1, LOOKAHEAD + 1, device=end.device).float())
    breach = cum < (-TAIL_SIGMA * sigma.unsqueeze(-1) * scale)
    return breach.any(dim=1).float()


def make_volchg_labels(hocl_all, batch_starts, window: int):
    """향후 5봉 실현변동성 > 직전 5봉 실현변동성."""
    logc = hocl_all[:, 2]
    end = batch_starts + window - 1
    fut = end.unsqueeze(-1) + torch.arange(1, LOOKAHEAD + 1, device=end.device)
    past = end.unsqueeze(-1) + torch.arange(-LOOKAHEAD + 1, 1, device=end.device)
    rv_fut = ((logc[fut] - logc[fut - 1]) ** 2).sum(dim=1)
    rv_past = ((logc[past] - logc[past - 1]) ** 2).sum(dim=1)
    return (rv_fut > rv_past).float()


def auc_score(scores, labels):
    """순위 통계 기반 AUC."""
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, len(scores) + 1, device=scores.device, dtype=torch.float32)
    pos = labels > 0.5
    n_pos, n_neg = pos.sum().item(), (~pos).sum().item()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return ((ranks[pos].sum().item() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def logistic_baseline(x_tr, y_tr, x_te, epochs: int = 300):
    mu, sd = x_tr.mean(dim=0), x_tr.std(dim=0) + 1e-12
    x_tr, x_te = (x_tr - mu) / sd, (x_te - mu) / sd
    torch.manual_seed(SEED)
    probe = nn.Linear(x_tr.shape[1], 1).to(x_tr.device)
    opt = torch.optim.Adam(probe.parameters(), lr=0.05)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss_fn(probe(x_tr)[:, 0], y_tr).backward()
        opt.step()
    with torch.no_grad():
        return probe(x_te)[:, 0]


def train_classifier(hocl_all, v_all, starts, y):
    torch.manual_seed(SEED)
    model = BankClassifier().to(DEVICE)
    lam = [model.bank.lambda_logit]
    others = [p for p in model.parameters() if p is not model.bank.lambda_logit]
    opt = torch.optim.Adam([{"params": others, "lr": LR}, {"params": lam, "lr": LR_LAMBDA}])
    loss_fn = nn.BCEWithLogitsLoss()

    n_tr = len(starts["train"])
    best_loss, best_state = float("inf"), None
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(n_tr, device=DEVICE)
        for i in range(0, n_tr, BATCH):
            sel = perm[i:i + BATCH]
            h, v = make_inputs(hocl_all, v_all, starts["train"][sel], WINDOW)
            opt.zero_grad()
            loss = loss_fn(model(h, v), y["train"][sel])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            preds = []
            for i in range(0, len(starts["val"]), 8192):
                h, v = make_inputs(hocl_all, v_all, starts["val"][i:i + 8192], WINDOW)
                preds.append(model(h, v))
            val_loss = loss_fn(torch.cat(preds), y["val"]).item()
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, len(starts["test"]), 8192):
            h, v = make_inputs(hocl_all, v_all, starts["test"][i:i + 8192], WINDOW)
            preds.append(model(h, v))
    hls = model.bank.half_lives.detach().tolist()
    return torch.cat(preds), hls


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    torch.manual_seed(SEED)

    api = CandleDataAPI()
    stocks = load_stocks(LARGE_CAP + SMALL_CAP, api, years=YEARS)
    hocl_all, v_all, _, starts, _ = build_arrays(stocks, WINDOW, LOOKAHEAD)
    hocl_all, v_all = hocl_all.to(DEVICE), v_all.to(DEVICE)
    starts = {k: v.to(DEVICE) for k, v in starts.items()}

    tasks = {
        "tail": make_tail_labels,
        "volchg": make_volchg_labels,
    }

    lines = [
        f"이진 과제 폭 측정: {len(stocks)} 종목, window={WINDOW}, lookahead={LOOKAHEAD}, "
        f"epochs={EPOCHS}, device={DEVICE}",
        "",
        f"{'task':>7} | {'양성률':>6} | {'B_logit AUC':>11} | {'D_bank AUC':>10} | learned hl",
        "-" * 85,
    ]

    for name, label_fn in tasks.items():
        y = {k: label_fn(hocl_all, starts[k], WINDOW) for k in starts}
        base_rate = y["test"].mean().item()

        x_tr = vol_features(hocl_all, v_all, starts["train"], WINDOW)
        x_te = vol_features(hocl_all, v_all, starts["test"], WINDOW)
        auc_lin = auc_score(logistic_baseline(x_tr, y["train"], x_te), y["test"])

        scores, hls = train_classifier(hocl_all, v_all, starts, y)
        auc_bank = auc_score(scores, y["test"])

        hl_str = "[" + ", ".join(f"{h:.1f}" for h in hls) + "]"
        lines.append(
            f"{name:>7} | {base_rate:>6.3f} | {auc_lin:>11.4f} | {auc_bank:>10.4f} | {hl_str}"
        )
        print(lines[-1])

    report = "\n".join(lines)
    (RESULTS_DIR / "report_topics.txt").write_text(report, encoding="utf-8")
    print(f"\nsaved -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()
