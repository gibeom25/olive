# Baseline: Frozen OpenVLA-7B on LIBERO (4 suites)

**Run date:** 2026-08-31 22:03 → 2026-09-01 05:55 KST (wall time ~7 h 51 m)
**Purpose:** establish the "no adaptation" reference point for OLIVE — the
official `openvla/openvla-7b-finetuned-libero-*` SFT checkpoints evaluated
out-of-the-box on their matching LIBERO task suite, backbone fully frozen.

---

## Results

`NUM_TRIALS_PER_TASK = 20` → 200 rollouts per suite (10 tasks × 20), seed 7,
`center_crop=True`. All suites exited rc=0, 0 caught exceptions.

| Suite | This run (200 ep) | OpenVLA paper, SFT (3 seeds × 500 ep) |
|-------|:---:|:---:|
| LIBERO-Spatial | **85.0 %** | 84.7 % |
| LIBERO-Object  | **87.0 %** | 88.4 % |
| LIBERO-Goal    | **77.5 %** | 79.2 % |
| LIBERO-Long (10) | **53.5 %** | 53.7 % |
| **Average** | **75.8 %** | 76.5 % |

Our numbers match the **published OpenVLA SFT checkpoint performance** to within
±2 pp (well inside noise for 200 vs 1500 rollouts). This validates the eval
pipeline end-to-end on the new hardware/stack (see Environment below).

> ⚠️ **Open question — which baseline to compare OLIVE against.**
> These results reproduce the OpenVLA-reported SFT performance, **not** the
> OLIVE paper's "Frozen OpenVLA (vanilla)" row (60 / 88 / 70 / 40). The OLIVE
> paper's lower Spatial/Goal/Long numbers presumably come from a different eval
> config (it mentions 180° rotation preprocessing, different seeds, possibly
> fewer rollouts). Before reporting OLIVE gains, decide whether the baseline is
> "as we measured here" or "as in the OLIVE paper" and reconcile the setup.

### Per-task success rate

**LIBERO-Spatial — 85.0 %**

| Task | SR |
|------|:--:|
| pick up the black bowl between the plate and the ramekin and place it on the plate | 0.85 |
| pick up the black bowl next to the ramekin and place it on the plate | 0.95 |
| pick up the black bowl from table center and place it on the plate | 0.80 |
| pick up the black bowl on the cookie box and place it on the plate | 0.95 |
| pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate | 0.75 |
| pick up the black bowl on the ramekin and place it on the plate | 0.90 |
| pick up the black bowl next to the cookie box and place it on the plate | 0.95 |
| pick up the black bowl on the stove and place it on the plate | 0.75 |
| pick up the black bowl next to the plate and place it on the plate | 0.90 |
| pick up the black bowl on the wooden cabinet and place it on the plate | 0.70 |

**LIBERO-Object — 87.0 %**

| Task | SR |
|------|:--:|
| pick up the alphabet soup and place it in the basket | 0.95 |
| pick up the cream cheese and place it in the basket | 0.80 |
| pick up the salad dressing and place it in the basket | 0.95 |
| pick up the bbq sauce and place it in the basket | 0.65 |
| pick up the ketchup and place it in the basket | 1.00 |
| pick up the tomato sauce and place it in the basket | 0.80 |
| pick up the butter and place it in the basket | 0.90 |
| pick up the milk and place it in the basket | 0.90 |
| pick up the chocolate pudding and place it in the basket | 0.90 |
| pick up the orange juice and place it in the basket | 0.85 |

**LIBERO-Goal — 77.5 %**

| Task | SR |
|------|:--:|
| open the middle drawer of the cabinet | 0.60 |
| put the bowl on the stove | 0.95 |
| put the wine bottle on top of the cabinet | 0.90 |
| open the top drawer and put the bowl inside | 0.50 |
| put the bowl on top of the cabinet | 1.00 |
| push the plate to the front of the stove | 0.75 |
| put the cream cheese in the bowl | 0.75 |
| turn on the stove | 0.90 |
| put the bowl on the plate | 0.95 |
| put the wine bottle on the rack | 0.45 |

**LIBERO-Long (LIBERO-10) — 53.5 %**

| Task | SR |
|------|:--:|
| put both the alphabet soup and the tomato sauce in the basket | 0.65 |
| put both the cream cheese box and the butter in the basket | 0.90 |
| turn on the stove and put the moka pot on it | 0.50 |
| put the black bowl in the bottom drawer of the cabinet and close it | 0.40 |
| put the white mug on the left plate and put the yellow and white mug on the right plate | 0.30 |
| pick up the book and place it in the back compartment of the caddy | 0.95 |
| put the white mug on the plate and put the chocolate pudding to the right of the plate | 0.35 |
| put both the alphabet soup and the cream cheese box in the basket | 0.50 |
| put both moka pots on the stove | 0.25 |
| put the yellow and white mug in the microwave and close it | 0.55 |

### Weakest tasks (targets for OLIVE's long-horizon robustness)

| Suite | Task | SR |
|-------|------|:--:|
| Long | put both moka pots on the stove | 0.25 |
| Long | put the white mug on the left plate … right plate | 0.30 |
| Long | put the white mug on the plate and put the chocolate pudding to the right | 0.35 |
| Long | put the black bowl in the bottom drawer of the cabinet and close it | 0.40 |
| Goal | put the wine bottle on the rack | 0.45 |
| Goal | open the top drawer and put the bowl inside | 0.50 |
| Long | turn on the stove and put the moka pot on it | 0.50 |
| Long | put both the alphabet soup and the cream cheese box in the basket | 0.50 |

Failure mode is concentrated in **multi-object / multi-step** tasks — consistent
with error accumulation over long horizons.

---

## Environment

The machine's GPU is an **RTX PRO 6000 Blackwell (sm_120)**. OpenVLA's pinned
`torch 2.2.0+cu121` has **no Blackwell kernels** (`CUDA error: no kernel image
is available`). The `openvla` conda env was moved off the pinned stack:

| package | pinned by OpenVLA | used for this run |
|---------|-------------------|-------------------|
| torch | 2.2.0+cu121 | **2.7.0+cu128** |
| torchvision / torchaudio | 0.17.0 / 2.2.0 | 0.22.0+cu128 / 2.7.0+cu128 |
| flash-attn | 2.5.5 | **2.8.3** (prebuilt `cu12torch2.7cxx11abiTRUE-cp310` wheel) |
| transformers | 4.40.1 | 4.40.1 (unchanged) |
| numpy | — | 1.26.4 |

LIBERO sim stack added to the same env: robosuite 1.4.0, mujoco 3.2.3,
robomimic 0.2.0 (`--no-deps`), bddl 1.0.1, numba 0.59.1, opencv-python 4.6.0.66,
tensorflow-metadata 1.14.0, wandb 0.16.6 (pins chosen to keep numpy < 2 and
protobuf in TF 2.15's range). LIBERO source used from `~/LIBERO/LIBERO` via
`PYTHONPATH` with `LIBERO_CONFIG_PATH=~/.libero_base`. Rendering: `MUJOCO_GL=egl`
(headless).

### Patches to `~/olive/openvla` (vs. upstream commit `c8f03f4`)

| file | change |
|------|--------|
| `pyproject.toml` | `torch`/`torchvision`/`torchaudio` pins `==` → `>=` |
| `experiments/robot/openvla_utils.py` | `attn_implementation` now `os.environ.get("OPENVLA_ATTN", "flash_attention_2")` |
| `experiments/robot/libero/run_libero_eval.py` | `env.close()` at end of each task loop (kills the harmless EGL `__del__` traceback at exit) |
| `experiments/robot/libero/libero_utils.py` | `save_rollout_video()` honors `LIBERO_ROLLOUT_DIR` (was hardcoded `./rollouts`) |

Also required at runtime: `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` — torch ≥ 2.6
flipped `torch.load` default to `weights_only=True`, which breaks LIBERO's
`get_task_init_states()` (pickled numpy).

---

## How it was run

```bash
NUM_TRIALS_PER_TASK=20 ~/olive/eval/run_libero_parallel.sh
```

- `run_libero_parallel.sh` → detached tmux session `libero-eval`, an orchestrator
  that runs suites 2-at-a-time (lane count auto-sized from free VRAM;
  `floor((free − 6000) / 19000)` → 2 on this box), each lane = one
  `run_libero_baseline.sh` invocation.
- Per suite: `~/olive/models/olive-model.sh` fetches the ~15 GB checkpoint,
  `run_libero_eval.py` runs, then the checkpoint is released.
- W&B on by default (`USE_WANDB=true`), project `olive-libero-baseline`.

### Timeline (2 lanes, GPU shared with other users' jobs)

| Suite | eval start | eval end | duration |
|-------|-----------|----------|----------|
| Spatial | 08-31 22:06 | 09-01 00:17 | ~2 h 11 m |
| Object  | 08-31 22:08 | 09-01 00:54 | ~2 h 46 m |
| Goal    | 09-01 00:19 | 09-01 03:00 | ~2 h 40 m |
| Long    | 09-01 00:57 | 09-01 05:55 | ~4 h 58 m |

Wave 1 = Spatial ∥ Object, Wave 2 = Goal ∥ Long. Long is the bottleneck
(520-step episodes, low SR → many run to the step limit).

---

## Artifacts

All under `~/olive/result/baseline/`:

| path | contents |
|------|----------|
| `logs/<suite>-<stamp>/EVAL-*.txt` | per-task + running success rate |
| `logs/<suite>-<stamp>/console.log` | full episode-by-episode stdout |
| `logs/libero-eval-<suite>-<stamp>.out` | per-lane stdout (incl. checkpoint download) |
| `logs/orch-<stamp>.log` | orchestrator (launch / wait / final summary) |
| `rollouts/2026_08_31/`, `rollouts/2026_09_01/` | 803 episode MP4s (~52 MB) |
| `wandb/` | local W&B run files (~7.7 MB) |

**W&B:** <https://wandb.ai/gibeom-real-1526-kyungpook-national-university/olive-libero-baseline>
runs `EVAL-libero_spatial|object|goal|10-openvla-…--baseline`, all `finished`,
`success_rate/total` = 0.85 / 0.87 / 0.775 / 0.535. (One throwaway
`_preflight_check` run in the project is ignorable.)

---

## Next

- Decide the baseline convention for OLIVE comparison (this run vs. OLIVE paper's
  numbers) and, if needed, re-run with the paper's preprocessing.
- Consider a full `NUM_TRIALS_PER_TASK=50` run (3 seeds) for paper-grade numbers
  (~2.5× longer per suite).
- OLIVE adaptation should be measured against the same per-task list above;
  the weak Long/Goal tasks are the ones to watch.
