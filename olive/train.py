"""
OLIVE: 메인 드라이버 (논문 Algorithm 2 outer loop + BC warm-up).

흐름
----
    build_config
      -> load_backbone (frozen VLA)
      -> ActorCritic (trainable) + ActionBasisProjection (frozen buffer, 또는 unconstrained 시 학습)
      -> [bc.enabled] bc_warmup(demo 지도학습)
      -> PPO 루프:  collect_rollouts -> PPOTrainer.update -> (주기적) evaluate / checkpoint / wandb
      -> 최종 evaluate + checkpoint

실행
----
    export PYTHONPATH=$HOME/olive/openvla:$HOME/LIBERO/LIBERO
    export LIBERO_CONFIG_PATH=$HOME/.libero_base MUJOCO_GL=egl TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
    python -m olive.train --suite libero_spatial --pretrained_checkpoint <local ckpt> --ablation main
"""

from __future__ import annotations

import math
import os
import random
import time

import numpy as np
import torch

from config import build_config
from backbone import load_backbone
from network import ActorCritic
from projection import ActionBasisProjection
from ppo_trainer import PPOConfig, PPOTrainer
from rollout import RolloutBuffer, RolloutConfig, collect_rollouts
from reward import make_reward_fn
from evaluate import evaluate_suite
from libero_env import make_task_suite, make_env, get_task_init_states, get_image, get_dummy_action
import checkpoint as ckpt_io


def _seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _ppo_config_from(cfg) -> PPOConfig:
    p = cfg.ppo
    return PPOConfig(
        total_steps=p.total_steps, buffer_size=p.buffer_size, ppo_epochs=p.ppo_epochs,
        batch_size=p.batch_size, lr_start=p.lr_start, lr_end=p.lr_end,
        gamma=p.gamma, gae_lambda=p.gae_lambda, clip_eps=p.clip_eps,
        value_coef=p.value_coef,
        entropy_coef=(p.entropy_coef / cfg.net.z_dim) if p.entropy_per_dim else p.entropy_coef,
        max_grad_norm=p.max_grad_norm, normalize_adv=p.normalize_adv,
        clip_value_loss=p.clip_value_loss, target_kl=p.target_kl,
        normalize_value_loss=p.normalize_value_loss, separate_grad_clip=p.separate_grad_clip,
    )


def main(argv=None):
    cfg = build_config(argv)
    _seed_everything(cfg.seed)
    os.makedirs(cfg.run_dir, exist_ok=True)
    print(f"[OLIVE] run_dir = {cfg.run_dir}")
    print(f"[OLIVE] suite={cfg.suite}  ablation-run={cfg.run_name}  bc={cfg.bc.enabled} "
          f"inject={cfg.proj.inject_basis} z_dim={cfg.net.z_dim}")

    run = None
    if cfg.wandb:
        import wandb
        run = wandb.init(project=cfg.wandb_project, entity=cfg.wandb_entity,
                         name=cfg.run_name, dir=cfg.run_dir,
                         config={"suite": cfg.suite, **_flat(cfg)})

    # ── 1. backbone ──
    vla, processor, meta = load_backbone(cfg.pretrained_checkpoint, device=cfg.device,
                                         attn_impl=cfg.attn_impl)

    # ── 2. head ──
    ac = ActorCritic(
        hidden_dim=meta.hidden_dim, z_dim=cfg.net.z_dim,
        actor_mlp_hidden=cfg.net.actor_mlp_hidden, critic_mlp_hidden=cfg.net.critic_mlp_hidden,
        actor_output_gain=cfg.net.actor_output_gain,
    ).to(cfg.device)
    proj = ActionBasisProjection(
        vla, meta.vocab_size, alpha=cfg.proj.alpha, inject_basis=cfg.proj.inject_basis,
        num_bins=cfg.proj.num_bins, action_dim=cfg.proj.action_dim,
        z_dim=cfg.net.z_dim, random_basis_seed=cfg.proj.random_basis_seed,
    ).to(cfg.device)
    print("[OLIVE] trainable params:", ac.num_parameters(),
          "+ proj", sum(p.numel() for p in proj.trainable_parameters()))

    # ── init_ckpt: 이전 단계(보통 BC) 가중치 불러오기 ──
    if cfg.init_ckpt:
        ckpt_io.load(cfg.init_ckpt, actor_critic=ac, projection=proj, map_location=cfg.device)
        print(f"[OLIVE] loaded init head from {cfg.init_ckpt}")

    # ── 3. BC warm-up  (phase in {bc, all}) ──
    do_bc = cfg.bc.enabled and cfg.phase in ("bc", "all")
    if do_bc:
        from bc_warmup import bc_warmup
        bc_res = bc_warmup(cfg=cfg, vla=vla, processor=processor, actor_critic=ac,
                           projection=proj, wandb_run=run)
        if run:
            run.log({f"bc/final_{k}": v for k, v in bc_res.items() if isinstance(v, (int, float))})
        bc_path = ckpt_io.save(os.path.join(cfg.run_dir, "ckpt", "after_bc.pt"),
                               actor_critic=ac, projection=proj, step=0, cfg=cfg, extra={"bc": bc_res})
        print(f"[OLIVE] BC done -> {bc_path}")

    if cfg.phase == "bc":
        print("[OLIVE] phase=bc: PPO 생략하고 종료.")
        if run:
            run.finish()
        return

    # ── 4. PPO 준비 ──
    ppo_cfg = _ppo_config_from(cfg)
    trainer = PPOTrainer(ac, ppo_cfg)
    # unconstrained ablation: projection 의 학습 Linear 도 optimizer 에 추가
    for pg_p in proj.trainable_parameters():
        trainer.optimizer.add_param_group({"params": [pg_p]})
        trainer._params.append(pg_p)

    suite = make_task_suite(cfg.suite)
    tasks = [suite.get_task(i) for i in range(suite.n_tasks)]
    init_map = {id(t): get_task_init_states(suite, i) for i, t in enumerate(tasks)}
    reward_fn = make_reward_fn(cfg.reward.r_success, cfg.reward.lam)
    rcfg = RolloutConfig(
        unnorm_key=cfg.suite, center_crop=cfg.rollout.center_crop,
        resize_size=cfg.rollout.resize_size, max_steps=cfg.rollout.max_steps,
        num_steps_wait=cfg.rollout.num_steps_wait,
        gripper_postprocess=cfg.rollout.gripper_postprocess, device=cfg.device,
    )

    def _make_env(task):
        return make_env(task, resolution=256)

    n_iters = math.ceil(ppo_cfg.total_steps / ppo_cfg.buffer_size)
    print(f"[OLIVE] PPO: {n_iters} iterations x {ppo_cfg.buffer_size} steps "
          f"(total ~{n_iters * ppo_cfg.buffer_size})")

    buffer = RolloutBuffer(device=cfg.device)
    best_sr = -1.0
    t0 = time.time()

    # RL 특성 반영 지표: round-robin 커서(편향 방지) + rollout SR 의 전체/이동평균.
    task_cursor = 0
    init_cursor = {t.language: 0 for t in tasks}         # task 별로 쓴 init state 수
    cum_succ, cum_ep = 0, 0                              # 전체 누적 rollout 성공률
    sr_ema = None                                        # 이동평균 (EMA)
    EMA_A = 0.1

    for it in range(1, n_iters + 1):
        buffer.clear()
        ac.eval()
        _tr = time.time()
        print(f"[OLIVE] it {it}/{n_iters}  rollout 수집 (target {ppo_cfg.buffer_size}, cursor@t{task_cursor % len(tasks)}) ...",
              flush=True)
        rstat = collect_rollouts(
            _make_env, tasks, lambda tk: init_map[id(tk)],
            vla=vla, processor=processor, actor_critic=ac, projection=proj,
            buffer=buffer, cfg=rcfg, reward_fn=reward_fn,
            target_transitions=ppo_cfg.buffer_size,
            rollouts_per_task=cfg.rollout.rollouts_per_task,
            task_cursor=task_cursor,
            init_cursor_of=lambda tk: init_cursor.get(tk.language, 0),
        )
        task_cursor = rstat["task_cursor"]
        for d, v in rstat["per_task"].items():           # 다음 방문 때 다른 init state 쓰도록
            init_cursor[d] = init_cursor.get(d, 0) + v["n"]

        it_sr = rstat["success_rate"]
        cum_succ += sum(v["succ"] for v in rstat["per_task"].values())
        cum_ep += rstat["episodes"]
        cum_sr = cum_succ / max(1, cum_ep)
        sr_ema = it_sr if sr_ema is None else (1 - EMA_A) * sr_ema + EMA_A * it_sr

        print(f"[OLIVE] it {it}/{n_iters}  rollout {rstat['episodes']}ep/{rstat['transitions']}st  "
              f"{rstat['n_tasks_visited']} tasks  SR(iter/ema/cum) "
              f"{it_sr:.2f}/{sr_ema:.2f}/{cum_sr:.2f}  ({(time.time()-_tr)/60:.1f}m) -> PPO update", flush=True)
        ac.train()
        logs = trainer.update(buffer, last_value=0.0)   # timeout=done 로 저장하므로 bootstrap 0 근사

        line = (f"[OLIVE] it {it:3d}/{n_iters}  step {trainer.global_step:6d}  "
                f"SR {it_sr:.2f} ema {sr_ema:.2f} cum {cum_sr:.2f}  "
                f"π {logs['policy_loss']:+.3f}  V {logs['value_loss']:.3f}  "
                f"H {logs['entropy']:+.1f}  KL {logs['approx_kl']:.3f}  "
                f"gn {logs['grad_norm']:.2f}/{logs['grad_norm_critic']:.2f}  "
                f"σR {logs['ret_std']:.1f}  lr {logs['lr']:.1e}  "
                f"({(time.time()-t0)/60:.1f}m)")
        print(line, flush=True)
        _append_jsonl(os.path.join(cfg.run_dir, "metrics.jsonl"),
                      {"kind": "iter", "iter": it, "global_step": trainer.global_step,
                       "rollout_sr": it_sr, "rollout_sr_ema": sr_ema, "rollout_sr_cum": cum_sr,
                       "mean_steps": rstat["mean_steps"], "minutes": (time.time() - t0) / 60,
                       **{k: float(v) for k, v in logs.items()}})
        if run:
            run.log({"ppo/global_step": trainer.global_step,
                     "ppo/iter": it,
                     "train/rollout_sr": it_sr,
                     "train/rollout_sr_ema": sr_ema,
                     "train/rollout_sr_cum": cum_sr,
                     "train/rollout_episodes_cum": cum_ep,
                     "train/n_tasks_visited": rstat["n_tasks_visited"],
                     "train/mean_steps": rstat["mean_steps"],
                     **{f"ppo/{k}": v for k, v in logs.items()}})

        if it % cfg.eval_every == 0 or it == n_iters:
            print(f"[OLIVE] it {it}/{n_iters}  eval {cfg.eval_episodes_per_task} ep/task ...", flush=True)
            eval_res = evaluate_suite(
                vla=vla, processor=processor, actor_critic=ac, projection=proj,
                suite_name=cfg.suite, unnorm_key=cfg.suite, max_steps=cfg.rollout.max_steps,
                episodes_per_task=cfg.eval_episodes_per_task, device=cfg.device,
                center_crop=cfg.rollout.center_crop, verbose=True,
            )
            sr = eval_res["total_sr"]
            print(f"[OLIVE]   eval SR {sr:.1%}", flush=True)
            _append_jsonl(os.path.join(cfg.run_dir, "metrics.jsonl"),
                          {"kind": "eval", "iter": it, "global_step": trainer.global_step,
                           "total_sr": sr, "per_task": eval_res["per_task"]})
            if run:
                run.log({"ppo/global_step": trainer.global_step, "eval/total_sr": sr})
            if sr > best_sr:
                best_sr = sr
                ckpt_io.save(os.path.join(cfg.run_dir, "ckpt", "best.pt"),
                             actor_critic=ac, projection=proj, optimizer=trainer.optimizer,
                             step=trainer.global_step, cfg=cfg, extra={"eval_sr": sr})

        if it % cfg.ckpt_every == 0 or it == n_iters:
            ckpt_io.save(os.path.join(cfg.run_dir, "ckpt", f"it{it:04d}.pt"),
                         actor_critic=ac, projection=proj, optimizer=trainer.optimizer,
                         step=trainer.global_step, cfg=cfg)

    print(f"[OLIVE] done. best eval SR = {best_sr:.1%}  ({(time.time()-t0)/60:.1f}m)")
    if run:
        run.summary["best_eval_sr"] = best_sr
        run.finish()


def _append_jsonl(path, rec):
    import json
    try:
        with open(path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=float) + "\n")
    except Exception as e:  # noqa: BLE001 — 지표 기록 실패로 학습을 죽이지 않음
        print(f"[OLIVE] metrics.jsonl 기록 실패: {e}", flush=True)


def _flat(cfg, prefix=""):
    import dataclasses
    out = {}
    for f in dataclasses.fields(cfg):
        v = getattr(cfg, f.name)
        if dataclasses.is_dataclass(v):
            out.update(_flat(v, f"{prefix}{f.name}."))
        else:
            out[f"{prefix}{f.name}"] = v
    return out


if __name__ == "__main__":
    main()
