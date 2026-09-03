"""
OLIVE: deterministic 평가.

학습된 head 로 각 suite task 를 결정론적(z = μ_z)으로 rollout 해서 success rate 측정.
baseline_libero_eval.md 와 같은 조건(center_crop, 180° 전처리, suite 별 max_steps)이라
숫자를 바로 비교할 수 있다. transition 저장 없음(buffer=None).
"""

from __future__ import annotations

import numpy as np
import torch

from libero_env import make_task_suite, make_env, get_task_init_states, get_image, get_dummy_action
from reward import make_reward_fn
from rollout import RolloutConfig, collect_episode


@torch.no_grad()
def evaluate_suite(
    *,
    vla,
    processor,
    actor_critic,
    projection,
    suite_name: str,
    unnorm_key: str,
    max_steps: int,
    episodes_per_task: int = 20,
    center_crop: bool = True,
    resize_size: int = 224,
    num_steps_wait: int = 10,
    device: str = "cuda:0",
    reward_fn=None,
    verbose: bool = True,
) -> dict:
    actor_critic.eval()
    reward_fn = reward_fn or make_reward_fn()
    rcfg = RolloutConfig(
        unnorm_key=unnorm_key, center_crop=center_crop, resize_size=resize_size,
        max_steps=max_steps, num_steps_wait=num_steps_wait, device=device,
    )

    suite = make_task_suite(suite_name)
    per_task, all_succ = [], []
    for task_id in range(suite.n_tasks):
        task = suite.get_task(task_id)
        inits = get_task_init_states(suite, task_id)
        env, desc = make_env(task)
        succ = []
        for ep in range(episodes_per_task):
            st = collect_episode(
                env, inits[ep % len(inits)], desc,
                vla=vla, processor=processor, actor_critic=actor_critic,
                projection=projection, buffer=None, cfg=rcfg, reward_fn=reward_fn,
                deterministic=True,
                get_image=get_image, dummy_action=get_dummy_action,
            )
            succ.append(float(st["success"]))
        try:
            env.close()
        except Exception:
            pass
        sr = float(np.mean(succ))
        per_task.append({"task_id": task_id, "task": desc, "sr": sr, "n": len(succ)})
        all_succ.extend(succ)
        if verbose:
            print(f"  [{suite_name}] task {task_id:2d}  SR {sr:5.1%}  {desc}")

    total = float(np.mean(all_succ)) if all_succ else 0.0
    if verbose:
        print(f"  [{suite_name}] TOTAL SR {total:.1%}  ({len(all_succ)} episodes)")
    return {"suite": suite_name, "total_sr": total, "n_episodes": len(all_succ), "per_task": per_task}


def _cli():
    import argparse
    from config import TASK_MAX_STEPS
    from backbone import load_backbone
    from network import ActorCritic
    from projection import ActionBasisProjection
    import checkpoint as ckpt_io

    p = argparse.ArgumentParser("olive.evaluate")
    p.add_argument("--suite", required=True)
    p.add_argument("--pretrained_checkpoint", "--vla", required=True)
    p.add_argument("--ckpt", required=True, help="학습된 head .pt")
    p.add_argument("--episodes_per_task", type=int, default=20)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--inject_basis", default="lm_head")
    p.add_argument("--z_dim", type=int, default=256)
    p.add_argument("--wandb_project", default=None, help="주면 결과를 wandb 에 기록 (<name>-eval run)")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_name", default=None)
    a = p.parse_args()

    run = None
    if a.wandb_project:
        try:
            import wandb
            run = wandb.init(project=a.wandb_project, entity=a.wandb_entity,
                             name=a.wandb_name or f"EVAL-{a.suite}",
                             job_type="eval",
                             config={"suite": a.suite, "ckpt": a.ckpt,
                                     "episodes_per_task": a.episodes_per_task,
                                     "inject_basis": a.inject_basis, "z_dim": a.z_dim})
        except Exception as e:  # noqa: BLE001
            print(f"[OLIVE][eval] wandb init 실패 ({e}) — 로깅 없이 진행")

    vla, processor, meta = load_backbone(a.pretrained_checkpoint, device=a.device)
    ac = ActorCritic(hidden_dim=meta.hidden_dim, z_dim=a.z_dim).to(a.device)
    proj = ActionBasisProjection(vla, meta.vocab_size, inject_basis=a.inject_basis, z_dim=a.z_dim).to(a.device)
    ckpt_io.load(a.ckpt, actor_critic=ac, projection=proj, map_location=a.device)

    res = evaluate_suite(
        vla=vla, processor=processor, actor_critic=ac, projection=proj,
        suite_name=a.suite, unnorm_key=a.suite, max_steps=TASK_MAX_STEPS[a.suite],
        episodes_per_task=a.episodes_per_task, device=a.device,
    )
    print(f"\n{a.suite}: {res['total_sr']:.1%}")

    if run is not None:
        run.log({"eval/total_sr": res["total_sr"], "eval/n_episodes": res["n_episodes"],
                 **{f"eval/task_{t['task_id']}_sr": t["sr"] for t in res["per_task"]}})
        run.summary["eval_total_sr"] = res["total_sr"]
        run.finish()


if __name__ == "__main__":
    _cli()
