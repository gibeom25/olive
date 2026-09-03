"""
OLIVE actor-critic network.

입력
----
h : (B, hidden_dim)  — frozen Llama-2 백본 "마지막 레이어"의 hidden state
                       (final RMSNorm 이후, LM head 직전 텐서. hidden_dim=4096)
                       프리필 마지막 위치(빈 토큰)의 출력을 그대로 쓴다.

출력
----
Actor  : h -> z 의 정책 분포 파라미터 (mu_z, std_z),  z ∈ R^{z_dim}
         z 는 논문의 latent code(Eq.5). PPO 정책 분포를 z-space 에 둔다:
             z ~ N(mu_z(h), std_z)
         이후 z -> Δh (action basis projection) -> h' -> decode 는 이 파일 밖에서 한다.
Critic : h -> V(s) ∈ R   (상태가치, Eq.9)

이 파일은 "네트워크"만 담는다. action basis / residual 주입 / 디코딩 / PPO 업데이트 /
LIBERO demo 지도학습(warm-up) 루프는 각각 별도 모듈에서 이 네트워크를 사용한다.
backbone(7B + LM head)은 이 파일에서 전혀 참조하지 않는다 (전 구간 freeze 전제).

두 학습 단계가 이 네트워크를 공유한다:
  1) warm-up (지도학습): LIBERO demo 로 Actor/Critic 을 먼저 1회 학습.
     - Actor : mu_z 를 결정론적으로 써서 decode(h + α·(mu_z @ B)) ≈ expert action 회귀
     - Critic: V(h) -> demo 궤적 return-to-go 회귀
  2) PPO   : z ~ N(mu_z, std_z) 샘플, GAE, clipped objective.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

# ── 차원 ────────────────────────────────────────────────────────────────────
HIDDEN_DIM = 4096   # Llama-2 hidden dim (백본 마지막 레이어 출력)
Z_DIM = 256         # latent code z 차원 (= action basis 행 수, 논문 §3.2)

# ── 정책 표준편차(log_std) ─────────────────────────────────────────────────
# exp() 이전에 clamp 한다. (exp() 이후 std 를 clamp 하면 초기값이 경계에 정확히
# 걸릴 때 entropy bonus gradient 가 영구히 0 이 되는 문제가 있다.)
LOG_STD_INIT = -2.0   # 초기 std = exp(-2) ≈ 0.135
LOG_STD_MIN = -5.0
LOG_STD_MAX = 0.0

# ── 초기화 ─────────────────────────────────────────────────────────────────
# Actor 출력층을 작은 gain 으로 초기화 -> mu_z ≈ 0 -> Δh ≈ 0 -> 시작 시 정책이
# frozen 백본과 거의 동일. warm-up 지도학습을 residual 보정 문제로 만들어 준다.
ACTOR_OUTPUT_GAIN = 0.01


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, out_gain: float | None = None) -> nn.Sequential:
    net = nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    )
    if out_gain is not None:
        nn.init.orthogonal_(net[-1].weight, gain=out_gain)
        nn.init.zeros_(net[-1].bias)
    return net


class Actor(nn.Module):
    """h -> z 정책 분포.  mu_z 는 상태 의존(MLP), log_std 는 상태 무관 학습 파라미터."""

    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        z_dim: int = Z_DIM,
        mlp_hidden: int = Z_DIM,
        output_gain: float = ACTOR_OUTPUT_GAIN,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.mu = _mlp(hidden_dim, mlp_hidden, z_dim, out_gain=output_gain)
        self.log_std = nn.Parameter(torch.full((z_dim,), LOG_STD_INIT))

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """h: (B, hidden_dim) -> (mu_z, std_z), 각각 (B, z_dim)."""
        mu = self.mu(h)
        log_std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp().expand_as(mu)
        return mu, std

    def distribution(self, h: torch.Tensor) -> Normal:
        mu, std = self(h)
        return Normal(mu, std)


class Critic(nn.Module):
    """h -> V(s) ∈ R."""

    def __init__(self, hidden_dim: int = HIDDEN_DIM, mlp_hidden: int = Z_DIM):
        super().__init__()
        self.v = _mlp(hidden_dim, mlp_hidden, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, hidden_dim) -> (B,)."""
        return self.v(h).squeeze(-1)


class ActorCritic(nn.Module):
    """
    Actor + Critic. 입력은 항상 backbone 마지막 레이어 hidden state h (B, hidden_dim).
    backbone 참조 없음 — z -> action 변환은 호출부 책임.
    """

    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        z_dim: int = Z_DIM,
        actor_mlp_hidden: int = Z_DIM,
        critic_mlp_hidden: int = Z_DIM,
        actor_output_gain: float = ACTOR_OUTPUT_GAIN,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.actor = Actor(hidden_dim, z_dim, actor_mlp_hidden, actor_output_gain)
        self.critic = Critic(hidden_dim, critic_mlp_hidden)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """h -> (mu_z, std_z, value).  mu_z/std_z: (B, z_dim),  value: (B,)."""
        mu, std = self.actor(h)
        value = self.critic(h)
        return mu, std, value

    @torch.no_grad()
    def act(self, h: torch.Tensor, deterministic: bool = False):
        """
        롤아웃/추론용.
        Returns: z (B, z_dim), log_prob (B,), value (B,)
        """
        mu, std, value = self(h)
        if deterministic:
            z = mu
        else:
            z = Normal(mu, std).sample()
        log_prob = Normal(mu, std).log_prob(z).sum(-1)
        return z, log_prob, value

    def evaluate_z(self, h: torch.Tensor, z: torch.Tensor):
        """
        PPO 업데이트용: 주어진 z 에 대한 log_prob / entropy / value (gradient 有).
        Returns: log_prob (B,), entropy (B,), value (B,)
        """
        mu, std, value = self(h)
        dist = Normal(mu, std)
        log_prob = dist.log_prob(z).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, value

    # --- 파라미터 헬퍼 ---------------------------------------------------------
    def actor_parameters(self):
        return self.actor.parameters()

    def critic_parameters(self):
        return self.critic.parameters()

    def num_parameters(self) -> dict:
        na = sum(p.numel() for p in self.actor.parameters())
        nc = sum(p.numel() for p in self.critic.parameters())
        return {"actor": na, "critic": nc, "total": na + nc}


if __name__ == "__main__":
    # 스모크: shape / gradient 흐름 확인 (backbone 없이)
    torch.manual_seed(0)
    B = 4
    ac = ActorCritic()
    h = torch.randn(B, HIDDEN_DIM)

    mu, std, v = ac(h)
    assert mu.shape == (B, Z_DIM) and std.shape == (B, Z_DIM) and v.shape == (B,)

    z, logp, val = ac.act(h)
    assert z.shape == (B, Z_DIM) and logp.shape == (B,) and val.shape == (B,)

    logp2, ent, val2 = ac.evaluate_z(h, z)
    loss = -(logp2.mean()) + 0.5 * (val2 - 1.0).pow(2).mean() - 0.01 * ent.mean()
    loss.backward()
    g_actor = any(p.grad is not None and p.grad.abs().sum() > 0 for p in ac.actor.parameters())
    g_critic = any(p.grad is not None and p.grad.abs().sum() > 0 for p in ac.critic.parameters())
    assert g_actor and g_critic

    print("network.py smoke OK")
    print("  params:", ac.num_parameters())
    print(f"  mu_z init |mu|~{mu.abs().mean().item():.4f} (warm start -> ~0), "
          f"std init ~{std.mean().item():.4f}")
