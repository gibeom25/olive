"""
OLIVE: 체크포인트 저장/로드.

저장 대상은 학습되는 것만 (~2.2M): ActorCritic + (unconstrained 일 때) projection 의
inject Linear + optimizer + step + config 스냅샷. 7B backbone / action basis buffer 는
저장하지 않는다 (매번 backbone 에서 재추출).
"""

from __future__ import annotations

import dataclasses
import os
import time

import torch


def save(path: str, *, actor_critic, projection=None, optimizer=None,
         step: int = 0, cfg=None, extra: dict | None = None) -> str:
    os.makedirs(os.path.dirname(os.path.expanduser(path)), exist_ok=True)
    payload = {
        "actor_critic": actor_critic.state_dict(),
        "step": int(step),
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if projection is not None and projection.trainable_parameters():
        payload["projection_inject"] = projection._inject_linear.state_dict()
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if cfg is not None:
        payload["config"] = dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else cfg
    if extra:
        payload["extra"] = extra
    path = os.path.expanduser(path)
    torch.save(payload, path)
    return path


def load(path: str, *, actor_critic, projection=None, optimizer=None,
         map_location: str = "cpu", strict: bool = True) -> dict:
    payload = torch.load(os.path.expanduser(path), map_location=map_location, weights_only=False)
    actor_critic.load_state_dict(payload["actor_critic"], strict=strict)
    if projection is not None and "projection_inject" in payload and projection._inject_linear is not None:
        projection._inject_linear.load_state_dict(payload["projection_inject"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload


def latest(dir_path: str) -> str | None:
    d = os.path.expanduser(dir_path)
    if not os.path.isdir(d):
        return None
    cks = [f for f in os.listdir(d) if f.endswith(".pt")]
    if not cks:
        return None
    cks.sort(key=lambda f: os.path.getmtime(os.path.join(d, f)))
    return os.path.join(d, cks[-1])


if __name__ == "__main__":
    import tempfile
    from network import ActorCritic

    ac = ActorCritic()
    opt = torch.optim.Adam(list(ac.parameters()), lr=1e-4)
    with tempfile.TemporaryDirectory() as td:
        p = save(os.path.join(td, "ckpt", "step10.pt"), actor_critic=ac, optimizer=opt, step=10)
        ac2 = ActorCritic()
        before = ac2.actor.mu[0].weight.clone()
        pl = load(p, actor_critic=ac2, optimizer=torch.optim.Adam(list(ac2.parameters())))
        assert pl["step"] == 10
        assert torch.equal(ac.actor.mu[0].weight, ac2.actor.mu[0].weight)
        assert not torch.equal(before, ac2.actor.mu[0].weight)
        assert latest(os.path.join(td, "ckpt")) == p
    print("checkpoint.py smoke OK")
