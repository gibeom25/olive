"""
OLIVE: LIBERO demo 로 Actor/Critic 지도학습 (warm-up).  [원 논문에 없는 보강 단계]

목적
----
PPO 전에 Actor/Critic 을 데모 분포에서 한 번 학습해 시작점을 개선한다.
- Actor : decode_soft( h0 + α·(μ_z(h0) @ B) ) ≈ normalize(expert action)   (BC 회귀)
          → z=0 이면 이미 SFT 정책(≈85%)이므로 "SFT 오차를 고치는 residual" 을 배우는 문제.
- Critic: V(h0) ≈ return-to-go(Eq.12 reward, 데모는 전부 성공)                (value 회귀)

backbone 은 frozen. Actor BC 는 미분가능 soft AR decode(projection.decode_action_soft)를
쓰므로 backbone activation 을 통해 μ_z 까지 gradient 가 흐른다. 프롬프트 길이가 데모마다
달라 배치 prefill 이 번거로우므로 **frame 단위**로 처리하고 batch_size 만큼 gradient 를
누적한 뒤 step 한다. 비용을 줄이려 `frame_stride` 로 서브샘플.

데모 소스
--------
suite.get_task_demonstration(i) 가 주는 상대경로(표준 LIBERO) 를 우선 사용하고,
없으면 config.bc.demo_glob 로 파일을 찾는다. HDF5 는 robomimic 포맷:
    data/demo_k/obs/agentview_rgb   (T, H, W, 3) uint8
    data/demo_k/actions             (T, 7)       float
"""

from __future__ import annotations

import dataclasses
import glob
import os
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F

from libero_env import preprocess_frame, make_task_suite
from reward import compute_reward_paper
from rollout import RolloutConfig, get_prefill_state, normalize_action, _resolve_norm_stats


@dataclasses.dataclass
class _Frame:
    image: np.ndarray          # (rs, rs, 3) uint8  — 전처리 완료
    instruction: str
    target_a_norm: np.ndarray  # (7,) in [-1,1]
    ret_to_go: float           # Critic 타깃


# ─────────────────────────────────────────────────────────────────────────────
def _instruction_from_hdf5(path: str) -> str:
    """data.attrs['problem_info'] JSON 의 language_instruction (가장 신뢰도 높음)."""
    import json
    import h5py
    with h5py.File(path, "r") as f:
        pi = f["data"].attrs.get("problem_info", "{}")
        pi = pi.decode() if isinstance(pi, bytes) else pi
        try:
            return json.loads(pi).get("language_instruction", "").strip()
        except Exception:  # noqa: BLE001
            return ""


def _find_demo_files(cfg) -> list[tuple[str, str]]:
    """[(hdf5_path, instruction), ...]  — suite API 우선, 실패 시 glob. instruction 은 HDF5 attrs 에서."""
    paths = []
    try:
        suite = make_task_suite(cfg.suite)
        from libero.libero import get_libero_path
        ddir = get_libero_path("datasets")
        for i in range(suite.n_tasks):
            p = os.path.join(ddir, suite.get_task_demonstration(i))
            if os.path.isfile(p):
                paths.append(p)
    except Exception as e:  # noqa: BLE001
        print(f"[OLIVE][bc] suite demo API 실패 ({e}) -> demo_glob 사용")
    if not paths:
        paths = sorted(glob.glob(cfg.resolved_demo_glob()))

    out = []
    for p in paths:
        instr = _instruction_from_hdf5(p)
        if not instr:  # 폴백: 파일명
            instr = os.path.basename(p).replace("_demo.hdf5", "").replace("_", " ").strip()
        out.append((p, instr))
    return out


def _iter_frames(cfg, norm_stats: dict) -> Iterator[_Frame]:
    import h5py

    files = _find_demo_files(cfg)
    if cfg.bc.max_demos:
        files = files[: cfg.bc.max_demos]
    print(f"[OLIVE][bc] demo 파일 {len(files)}개, frame_stride={cfg.bc.frame_stride}")

    stride = cfg.bc.frame_stride
    gamma = cfg.bc.gamma
    import time as _t
    _t0 = _t.time()
    _nf = 0
    for fi, (path, instr) in enumerate(files):
        with h5py.File(path, "r") as f:
            demos = list(f["data"].keys())
            print(f"[OLIVE][bc]   loading file {fi+1}/{len(files)}  ({len(demos)} demos)  "
                  f"[{_nf} frames, {_t.time()-_t0:.0f}s]", flush=True)
            for dk in demos:
                g = f["data"][dk]
                obs = g["obs"]
                img_key = "agentview_rgb" if "agentview_rgb" in obs else (
                    "agentview_image" if "agentview_image" in obs else list(obs.keys())[0])
                imgs = obs[img_key][:]                     # (T, H, W, 3)
                acts = np.asarray(g["actions"][:], dtype=np.float64)  # (T, 7)
                T = len(acts)
                # return-to-go (Eq.12): 데모는 성공 -> 마지막 스텝 r=+r_success, 그 전 r=-λ·t
                rtg = np.zeros(T)
                acc = 0.0
                for t in range(T - 1, -1, -1):
                    done = (t == T - 1)
                    r = compute_reward_paper(done, t + 1, cfg.reward.r_success, cfg.reward.lam)
                    acc = r + gamma * acc
                    rtg[t] = acc
                for t in range(0, T, stride):
                    _nf += 1
                    yield _Frame(
                        image=preprocess_frame(imgs[t], cfg.rollout.resize_size),
                        instruction=instr,
                        target_a_norm=normalize_action(acts[t], norm_stats).astype(np.float32),
                        ret_to_go=float(rtg[t]),
                    )
    print(f"[OLIVE][bc]   frame load done: {_nf} frames in {_t.time()-_t0:.0f}s", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
def bc_warmup(*, cfg, vla, processor, actor_critic, projection, wandb_run=None) -> dict:
    """
    Returns: {"n_frames", "actor_loss", "critic_loss", "val_actor_mae", ...}
    학습되는 것: cfg.bc.targets 에 따라 actor / critic / both.
    wandb_run 을 주면 log_every batch 마다 bc/* 지표를 기록 (explicit step 없이 — x축은
    'bc/batch' 메트릭 사용).
    """
    bc = cfg.bc
    device = cfg.device
    actor_critic.train()

    def _wlog(d):
        if wandb_run is not None:
            try:
                wandb_run.log({f"bc/{k}": v for k, v in d.items()})
            except Exception:  # noqa: BLE001
                pass

    norm_stats = _resolve_norm_stats(vla, cfg.suite)
    frames = list(_iter_frames(cfg, norm_stats))
    if not frames:
        print("[OLIVE][bc] frame 0개 — warm-up 건너뜀")
        return {"n_frames": 0}

    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(frames)
    n_val = max(1, int(len(frames) * bc.val_frac))
    val_frames, train_frames = frames[:n_val], frames[n_val:]

    train_actor = bc.targets in ("both", "actor")
    train_critic = bc.targets in ("both", "critic")
    actor_params = (list(actor_critic.actor_parameters()) + projection.trainable_parameters()) if train_actor else []
    critic_params = list(actor_critic.critic_parameters()) if train_critic else []
    params = actor_params + critic_params
    opt = torch.optim.AdamW(params, lr=bc.lr, weight_decay=bc.weight_decay)

    def _gn(ps):
        g = [p.grad.detach().norm() for p in ps if p.grad is not None]
        return float(torch.norm(torch.stack(g))) if g else 0.0

    rcfg = RolloutConfig(unnorm_key=cfg.suite, center_crop=cfg.rollout.center_crop,
                         resize_size=cfg.rollout.resize_size, device=device)
    loss_fn = (lambda a, b: F.mse_loss(a, b)) if bc.action_loss == "mse" else (lambda a, b: F.huber_loss(a, b))

    def _forward(fr: _Frame):
        # backbone prefill (frozen, no_grad)
        with torch.no_grad():
            h0, pkv, mask = get_prefill_state(vla, processor, fr.image, fr.instruction, rcfg)
        h0_f = h0.float()                                    # (1,1,H)
        out = {}
        if train_actor:
            mu_z, _ = actor_critic.actor(h0_f.squeeze(1))    # (1, z_dim)  grad
            a_pred = projection.z_to_action_soft(h0, mu_z.unsqueeze(1), pkv, mask)  # (1,7) grad
            tgt = torch.as_tensor(fr.target_a_norm, device=device).unsqueeze(0)
            out["a_pred"], out["a_tgt"] = a_pred, tgt
        if train_critic:
            v = actor_critic.critic(h0_f.squeeze(1))         # (1,)  grad
            out["v"], out["v_tgt"] = v, torch.tensor([fr.ret_to_go], device=device)
        return out

    import time as _time
    n_batches = (len(train_frames) + bc.batch_size - 1) // bc.batch_size
    print(f"[OLIVE][bc] train {len(train_frames)} frames, {n_batches} batch/epoch x {bc.epochs} epoch, "
          f"targets={bc.targets}", flush=True)

    le = max(1, bc.log_every)
    global_batch = 0
    logs = {"actor_loss": [], "critic_loss": []}
    for epoch in range(bc.epochs):
        rng.shuffle(train_frames)
        _t0 = _time.time()
        for bi, bstart in enumerate(range(0, len(train_frames), bc.batch_size)):
            batch = train_frames[bstart : bstart + bc.batch_size]
            opt.zero_grad(set_to_none=True)
            a_losses, c_losses = [], []
            for fr in batch:
                o = _forward(fr)
                loss = 0.0
                if train_actor:
                    la = loss_fn(o["a_pred"], o["a_tgt"])
                    a_losses.append(la.item()); loss = loss + la
                if train_critic:
                    lc = F.mse_loss(o["v"], o["v_tgt"])
                    c_losses.append(lc.item()); loss = loss + 0.5 * lc
                (loss / len(batch)).backward()
            agn, cgn = _gn(actor_params), _gn(critic_params)   # clip 전 actor/critic 별도 grad norm
            gn = float(torch.nn.utils.clip_grad_norm_(params, bc.grad_clip))
            opt.step()
            global_batch += 1
            if a_losses:
                logs["actor_loss"].append(float(np.mean(a_losses)))
            if c_losses:
                logs["critic_loss"].append(float(np.mean(c_losses)))
            if (bi + 1) % le == 0 or bi + 1 == n_batches:
                al_ = float(np.mean(logs["actor_loss"][-le:])) if logs["actor_loss"] else float("nan")
                cl_ = float(np.mean(logs["critic_loss"][-le:])) if logs["critic_loss"] else float("nan")
                rate = (bi + 1) * bc.batch_size / max(1e-6, _time.time() - _t0)
                eta_ep = (n_batches - bi - 1) * bc.batch_size / max(1e-6, rate)
                done_frac = (epoch + (bi + 1) / n_batches) / bc.epochs
                print(f"[OLIVE][bc]   ep{epoch+1}/{bc.epochs} {bi+1}/{n_batches}  "
                      f"a~{al_:.4f} c~{cl_:.2f}  gn(a/c)~{agn:.2e}/{cgn:.1f}  {rate:.1f} fr/s  "
                      f"ETA_ep {eta_ep/60:.1f}m  ({done_frac*100:.0f}% total)", flush=True)
                _wlog({"batch": global_batch, "epoch": epoch + 1, "actor_loss": al_,
                       "critic_loss": cl_, "actor_grad_norm": agn, "critic_grad_norm": cgn,
                       "frames_per_s": rate, "progress": done_frac})
        al = np.mean(logs["actor_loss"][-50:]) if logs["actor_loss"] else float("nan")
        cl = np.mean(logs["critic_loss"][-50:]) if logs["critic_loss"] else float("nan")
        print(f"[OLIVE][bc] epoch {epoch+1}/{bc.epochs}  actor_mse~{al:.4f}  critic_mse~{cl:.4f}  "
              f"({(_time.time()-_t0)/60:.1f}m)", flush=True)
        _wlog({"epoch_end": epoch + 1, "epoch_actor_mse": float(al), "epoch_critic_mse": float(cl)})
        # epoch 마다 저장 — 중간에 죽어도 마지막 완료 epoch 결과로 eval 가능
        if bc.save_every_epoch:
            try:
                import checkpoint as _ckpt_io
                _p = _ckpt_io.save(os.path.join(cfg.run_dir, "ckpt", f"after_bc_ep{epoch+1}.pt"),
                                   actor_critic=actor_critic, projection=projection, step=0, cfg=cfg,
                                   extra={"bc_epoch": epoch + 1, "actor_mse": float(al), "critic_mse": float(cl)})
                print(f"[OLIVE][bc]   saved {_p}", flush=True)
            except Exception as _e:  # noqa: BLE001
                print(f"[OLIVE][bc]   epoch ckpt 저장 실패: {_e}", flush=True)

    # 검증
    actor_critic.eval()
    val_a_mae, val_c_mae = [], []
    with torch.no_grad():
        for fr in val_frames:
            o = _forward(fr)
            if train_actor:
                val_a_mae.append(float((o["a_pred"] - o["a_tgt"]).abs().mean()))
            if train_critic:
                val_c_mae.append(float((o["v"] - o["v_tgt"]).abs().mean()))
    res = {
        "n_frames": len(frames),
        "n_train": len(train_frames), "n_val": len(val_frames),
        "actor_loss": float(np.mean(logs["actor_loss"][-50:])) if logs["actor_loss"] else None,
        "critic_loss": float(np.mean(logs["critic_loss"][-50:])) if logs["critic_loss"] else None,
        "val_actor_mae": float(np.mean(val_a_mae)) if val_a_mae else None,
        "val_critic_mae": float(np.mean(val_c_mae)) if val_c_mae else None,
    }
    print(f"[OLIVE][bc] done  {res}")
    return res


if __name__ == "__main__":
    # h5py / 경로 유무만 확인 (실학습은 GPU + demo 필요).
    import importlib
    assert importlib.util.find_spec("h5py"), "pip install h5py"
    from config import build_config
    c = build_config(["--suite", "libero_spatial"])
    print("bc_warmup.py import OK  | demo_glob:", c.resolved_demo_glob())
    print("  (실행: train.py 가 bc.enabled=True 일 때 호출)")
