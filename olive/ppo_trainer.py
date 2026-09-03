"""
OLIVE: PPO trainer (논문 §3.4, Algorithm 2 lines 9-13).

z-space 정책 기준. 업데이트는 RolloutBuffer 에 저장된 (h0, z, old_log_prob, adv, ret)
만 쓴다 — frozen backbone 을 다시 돌리지 않는다(효율성 포인트). projection / env 는
여기서 안 쓴다(rollout.py 담당).

    L = L_CLIP - c_v · L_VF + c_e · H              (Algorithm 2 line 11, 부호는 최소화 형태로)
    L_CLIP = E[min(r_t Â_t, clip(r_t, 1-ε, 1+ε) Â_t)]
    r_t    = exp(logπ_new(z|h0) - logπ_old(z|h0))
    L_VF   = E[(V(h0) - R_t)^2]
    H      = E[entropy(π(·|h0))]
"""

from __future__ import annotations

import dataclasses
import math

import torch
import torch.nn as nn


@dataclasses.dataclass
class PPOConfig:
    # 논문 Table 1
    total_steps: int = 50_000          # 총 env step (Table 1). cf. Algorithm 2 본문은 5000 — 불일치, 설정값.
    buffer_size: int = 512             # PPO iteration 당 수집 transition 수
    ppo_epochs: int = 4
    batch_size: int = 64
    lr_start: float = 3e-4
    lr_end: float = 1e-5               # total_steps 에 걸쳐 선형 감소
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    # 옵션
    normalize_adv: bool = True
    clip_value_loss: bool = False      # 논문 미언급 — 기본 off
    target_kl: float | None = None     # approx-KL 조기 종료 (None = 사용 안 함)


class PPOTrainer:
    def __init__(self, actor_critic: nn.Module, cfg: PPOConfig | None = None):
        self.ac = actor_critic
        self.cfg = cfg or PPOConfig()
        params = list(self.ac.actor_parameters()) + list(self.ac.critic_parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.cfg.lr_start, eps=1e-5)
        self.global_step = 0            # 소비한 env step 누계 (LR schedule 기준)
        self._params = params

    # ── LR: total_steps 에 걸쳐 lr_start → lr_end 선형 ──
    def _lr_now(self) -> float:
        frac = min(1.0, self.global_step / max(1, self.cfg.total_steps))
        return self.cfg.lr_start + frac * (self.cfg.lr_end - self.cfg.lr_start)

    def _set_lr(self, lr: float):
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    @torch.no_grad()
    def _diagnostics(self, ratio, adv, new_logp, old_logp) -> dict:
        approx_kl = (old_logp - new_logp).mean().item()          # k1 추정치
        clipfrac = ((ratio - 1.0).abs() > self.cfg.clip_eps).float().mean().item()
        return {"approx_kl": approx_kl, "clip_frac": clipfrac}

    def update(self, buffer, last_value: float = 0.0) -> dict:
        """
        buffer: rollout.RolloutBuffer (add() 로 채워진 상태). 내부에서 GAE 계산.
        last_value: 마지막 state 의 V(s_T) bootstrap (에피소드가 done 으로 끝났으면 0).
        """
        cfg = self.cfg
        buffer.compute_gae(last_value, cfg.gamma, cfg.gae_lambda, normalize_adv=cfg.normalize_adv)
        n = len(buffer)

        lr = self._lr_now()
        self._set_lr(lr)

        logs = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                "approx_kl": 0.0, "clip_frac": 0.0, "grad_norm": 0.0, "n_minibatch": 0}
        stop = False

        for epoch in range(cfg.ppo_epochs):
            if stop:
                break
            for mb in buffer.iter_minibatches(cfg.batch_size, shuffle=True):
                h0, z = mb["h0"], mb["z"]
                old_logp, adv, ret = mb["old_log_prob"], mb["adv"], mb["ret"]

                new_logp, entropy, value = self.ac.evaluate_z(h0, z)   # (b,), (b,), (b,)
                ratio = (new_logp - old_logp).exp()

                # L_CLIP (최대화 대상) → 손실은 음수
                unclipped = ratio * adv
                clipped = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv
                policy_loss = -torch.min(unclipped, clipped).mean()

                # L_VF
                if cfg.clip_value_loss:
                    # old value 를 안 저장했으므로 필요 시 buffer 에서 가져오도록 확장 (현재 off)
                    value_loss = 0.5 * (value - ret).pow(2).mean()
                else:
                    value_loss = 0.5 * (value - ret).pow(2).mean()

                entropy_mean = entropy.mean()
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy_mean

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self._params, cfg.max_grad_norm)
                self.optimizer.step()

                d = self._diagnostics(ratio, adv, new_logp, old_logp)
                logs["policy_loss"] += policy_loss.item()
                logs["value_loss"] += value_loss.item()
                logs["entropy"] += entropy_mean.item()
                logs["approx_kl"] += d["approx_kl"]
                logs["clip_frac"] += d["clip_frac"]
                logs["grad_norm"] += float(grad_norm)
                logs["n_minibatch"] += 1

                if cfg.target_kl is not None and d["approx_kl"] > 1.5 * cfg.target_kl:
                    stop = True
                    break

        m = max(1, logs["n_minibatch"])
        for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "grad_norm"):
            logs[k] /= m

        self.global_step += n
        logs["lr"] = lr
        logs["global_step"] = self.global_step
        logs["buffer_n"] = n
        return logs

    # ── 체크포인트 (Actor/Critic + log_std 만) ──
    def state_dict(self) -> dict:
        return {"actor_critic": self.ac.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "global_step": self.global_step,
                "cfg": dataclasses.asdict(self.cfg)}

    def load_state_dict(self, sd: dict, load_optimizer: bool = True):
        self.ac.load_state_dict(sd["actor_critic"])
        if load_optimizer and "optimizer" in sd:
            self.optimizer.load_state_dict(sd["optimizer"])
        self.global_step = sd.get("global_step", 0)


if __name__ == "__main__":
    # 실제 ActorCritic + 가짜 buffer 로 업데이트 1스텝이 도는지 (env/backbone 없이).
    import numpy as np
    from network import ActorCritic
    from rollout import RolloutBuffer

    torch.manual_seed(0)
    ac = ActorCritic()                      # float32
    trainer = PPOTrainer(ac, PPOConfig(buffer_size=32, batch_size=8, ppo_epochs=2, total_steps=1000))

    buf = RolloutBuffer(device="cpu")
    H = 4096
    for i in range(32):
        h0 = torch.randn(1, 1, H)
        with torch.no_grad():
            z, logp, val = ac.act(h0.float())
        buf.add(h0, z, float(logp), float(val),
                reward=(1.0 if i == 31 else -0.01 * (i + 1)), done=(i == 31))

    before = [p.clone() for p in trainer._params]
    logs = trainer.update(buf, last_value=0.0)
    moved = any(not torch.equal(a, b) for a, b in zip(before, trainer._params))
    assert moved, "파라미터가 안 움직임"
    assert logs["n_minibatch"] == 8 and logs["global_step"] == 32   # 2 epoch * (32/8) = 8
    assert np.isfinite(logs["policy_loss"]) and np.isfinite(logs["value_loss"])
    print("ppo_trainer.py smoke OK")
    for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "grad_norm", "lr"):
        print(f"  {k:12s} = {logs[k]:.5f}")
