import torch
import torch.nn as nn
import torch.nn.functional as F


class MarginLoss(nn.Module):
    """Pairwise Margin Ranking Loss.

    Enforces that positive edge scores exceed negative edge scores by at least `margin`:
    L = max(0, score_neg - score_pos + margin)
    """

    def __init__(self, margin: float = 0.2) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pos = scores[targets == 1]
        neg = scores[targets == 0]
        if len(pos) > 0 and len(neg) > 0:
            min_len = min(len(pos), len(neg))
            pos_sampled = pos[:min_len]
            neg_sampled = neg[torch.randperm(len(neg), device=scores.device)[:min_len]]
            return torch.relu(neg_sampled - pos_sampled + self.margin).mean()
        return torch.tensor(0.0, device=scores.device, requires_grad=True)


class BPRLoss(nn.Module):
    """Bayesian Personalized Ranking (BPR) Loss.

    Maximizes the posterior probability that a positive edge is preferred over a negative edge:
    L = -log(sigmoid(score_pos - score_neg) + eps)
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pos = scores[targets == 1]
        neg = scores[targets == 0]
        min_len = min(len(pos), len(neg))
        if min_len == 0:
            return torch.tensor(0.0, device=scores.device, requires_grad=True)
        pos_sampled = pos[:min_len]
        neg_sampled = neg[torch.randperm(len(neg), device=scores.device)[:min_len]]
        return -torch.log(torch.sigmoid(pos_sampled - neg_sampled) + self.eps).mean()


class InfoNCELoss(nn.Module):
    """Contrastive InfoNCE Loss.

    Maximizes similarity between true pairs while repelling sampled negative candidates:
    L = - (score_pos / tau - log(sum_j exp(score_neg_j / tau)))
    """

    def __init__(self, tau: float = 0.1) -> None:
        super().__init__()
        self.tau = tau

    def forward(self, scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pos = scores[targets == 1]
        neg = scores[targets == 0]
        if len(pos) == 0 or len(neg) == 0:
            return torch.tensor(0.0, device=scores.device, requires_grad=True)

        p = pos / self.tau
        n = neg / self.tau
        max_n = torch.max(n)
        sum_exp_n = torch.sum(torch.exp(n - max_n))
        logsumexp = torch.log(torch.exp(p - max_n) + sum_exp_n) + max_n
        return -(p - logsumexp).mean()


class ASL(nn.Module):
    """Asymmetric Loss (ASL) for extreme class imbalance.

    Decouples focusing parameters gamma_pos and gamma_neg:
    L = -y * (1 - p)^gamma_pos * log(p) - (1 - y) * p^gamma_neg * log(1 - p)
    """

    def __init__(self, gamma_pos: float = 1.0, gamma_neg: float = 2.0) -> None:
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.clamp(preds, 1e-7, 1.0 - 1e-7)
        loss_pos = -targets * ((1.0 - p) ** self.gamma_pos) * torch.log(p)
        loss_neg = -(1.0 - targets) * (p ** self.gamma_neg) * torch.log(1.0 - p)
        return torch.mean(loss_pos + loss_neg)


class FL(nn.Module):
    """Focal Loss (FL) for link classification.

    Dynamically modulates cross-entropy loss by (1 - p_t)^gamma.
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy(preds, targets, reduction="none")
        p_t = preds * targets + (1.0 - preds) * (1.0 - targets)
        return (self.alpha * ((1.0 - p_t) ** self.gamma) * bce).mean()
