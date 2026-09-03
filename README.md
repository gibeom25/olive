# OLIVE — Lightweight Latent Residual RL for VLA models

Online PPO adaptation of a **frozen** OpenVLA‑7B backbone via a small **z‑space latent
residual head** (~2.16 M trainable params), constrained to the action‑basis subspace of
the frozen LM head. Paper: `docs/paper/`.

This repo is the working implementation + experiment log as of **2026‑09‑02**.
Result artifacts (checkpoints, rollout videos, wandb runs, logs) were **not** committed —
only code, docs, and the experiment write‑up.

---

## Layout

```
olive/                 main implementation (13 modules, run from inside this dir)
  config.py            all knobs — nested dataclasses, CLI --set a.b.c=v, ablation presets
  backbone.py          load frozen VLA + processor (no openvla import)
  network.py           Actor / Critic MLP  ← trainable head
  projection.py        action basis B, Δh = z·B, h' = h+αΔh, 7-step decode (greedy + soft)
  reward.py            Eq.12 reward
  libero_env.py        LIBERO task/env glue + image preprocessing (openvla parity)
  rollout.py           prefill, RolloutBuffer(+GAE), collect_episode / collect_rollouts
  ppo_trainer.py       PPOConfig + PPOTrainer.update
  bc_warmup.py         LIBERO-demo supervised warm-up (Actor BC + Critic return-to-go)
  evaluate.py          deterministic suite success-rate
  checkpoint.py        save/load head (+optimizer, +config)
  train.py             driver: [BC] → PPO loop → periodic eval / ckpt / wandb
  scripts/             _env.sh, 01_bc_warmup, 02_ppo_train, 03_eval, 04_wandb_report,
                       run_all_suites.sh
eval/                  standalone LIBERO baseline harness (frozen SFT checkpoints)
model_manager/         olive-model.sh — fetch/release the ~15 GB VLA checkpoints on demand
docs/
  experiment_log_2026-09-02.md   full chronology: fixes, shakedown, first runs
  baseline_libero_eval.md        frozen-OpenVLA baseline numbers
  code_flow.html                 architecture / data-flow diagram
  openvla-patches.patch          the 4-file diff applied to a clean openvla clone
  paper/                         the paper PDF
```

`docs/code_flow.html` is the picture; read it first.

---

## Setup

External deps (not vendored): a clone of `openvla/openvla` and `Lifelong-Robot-Learning/LIBERO`.

```bash
conda activate openvla                 # torch 2.7.0+cu128, flash-attn 2.8.3, transformers 4.40.1
git clone https://github.com/openvla/openvla.git
git -C openvla apply /path/to/docs/openvla-patches.patch     # OPENVLA_ATTN, LIBERO_ROLLOUT_DIR, env.close, torch pins
# LIBERO demos: LIBERO-plus datasets dir (see bc.demo_glob)

export PYTHONPATH=$PWD/openvla:$PWD/LIBERO
export LIBERO_CONFIG_PATH=~/.libero_base MUJOCO_GL=egl
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1        # torch>=2.6 torch.load default
```

The `scripts/` wrappers set all of the above via `scripts/_env.sh`.

---

## Run

```bash
cd olive/scripts
# one suite, staged
SUITE=libero_spatial ./01_bc_warmup.sh          # → after_bc.pt
SUITE=libero_spatial ./02_ppo_train.sh          # → best.pt   (auto-inits from after_bc.pt)
SUITE=libero_spatial ./03_eval.sh               # → TOTAL SR

# all suites, each from its own SFT checkpoint
SUITES="libero_goal libero_10" ./run_all_suites.sh
```

Direct: `cd olive && python train.py --suite <s> --pretrained_checkpoint <local ckpt> --phase {bc|ppo|all}`
Ablations: `--ablation {main|no_bc|random_basis|unconstrained|paper_ppo|ent_full|z128|...}` + `--set ppo.total_steps=50000`.

---

## Status (2026‑09‑02)

| suite | frozen-SFT baseline (ours) | OLIVE (BC-critic + PPO 5k) | OLIVE paper |
|---|---|---|---|
| LIBERO-Spatial | 85.0 % | — | 91.9 |
| LIBERO-Goal | 77.5 % | 72.0 % | 71.4 |
| LIBERO-10 | 53.5 % | 48.0 % | 50.0 |

Pipeline verified end-to-end on real hardware. KL explosion from the first PPO run was
fixed (`entropy_per_dim`, `target_kl=0.05`, `lr 1e-4`); PPO is now stable but does **not
yet beat baseline**.

**Next (in `docs/experiment_log`):**
1. Normalize value/return targets — grad-norm 500–2000 is hard-clipped to 1.0, crushing the update.
2. Vectorize rollout + eval (currently batch-1, sequential) so a 50k-step run is feasible.
3. Actor BC soft-decode gradient ≈ 0 (softmax saturation) — revisit.
4. Then `total_steps=50000` + optional Actor BC.
