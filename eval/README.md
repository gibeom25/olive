# olive/eval

Scripts to run the **Frozen OpenVLA (vanilla) LIBERO baseline** from the OLIVE
paper — the `openvla/openvla-7b-finetuned-libero-<suite>` checkpoints evaluated
out-of-the-box on their matching LIBERO suite.

## Files

| file | what |
|---|---|
| `setup_libero_eval_env.sh` | one-time: add the LIBERO sim stack (robosuite/mujoco/bddl/robomimic/...) to the `openvla` conda env. Idempotent — no-ops if already set up. **Already run; env is ready.** |
| `run_libero_baseline.sh`   | sequential: fetch checkpoint → run eval → release, for spatial→object→goal→long |
| `run_libero_parallel.sh`   | same, but runs suites CONCURRENTLY in a detached tmux session, auto-sizing lane count from free GPU memory (`nvidia-smi`) |
| `logs/`                    | per-run logs (created on first run) |

## How long does it take?

Depends heavily on the shared GPU. Rough guide with flash-attn on the RTX PRO 6000,
GPU mostly free:

| mode | trials/task | est. wall time |
|---|---|---|
| sequential, 4 suites | 50 (paper) | ~12–16 h |
| sequential, 4 suites | 20 | ~5–7 h |
| **parallel (2 lanes), 4 suites** | **20** | **~3.5–5 h** |
| parallel (2 lanes), 4 suites | 50 | ~9–11 h |

`long` (LIBERO-10) is the slow pole: ~520-step episodes, ~50% SR so many run full length.
**Calibrate first** with the smoke test below (~15–25 min, mostly the 15 GB download).

## How to run

```bash
# 1) env prep -- ALREADY DONE. Only rerun if the env breaks / you clone it:
#    ~/olive/eval/setup_libero_eval_env.sh
#    MODE=clone ~/olive/eval/setup_libero_eval_env.sh   # builds env `olive-eval` instead

# 2) smoke test (1 rollout/task, ~15-25 min incl. 15 GB download) -- gives a real ETA
KEEP_MODELS=1 NUM_TRIALS_PER_TASK=1 SUITES=spatial ~/olive/eval/run_libero_baseline.sh

# 3a) sequential full run
NUM_TRIALS_PER_TASK=20 ~/olive/eval/run_libero_baseline.sh

# 3b) parallel run in tmux, lane count auto-sized from free VRAM  (recommended)
~/olive/eval/run_libero_parallel.sh
tmux attach -t libero-eval          # watch;  Ctrl-b d to detach
tmux kill-session -t libero-eval    # stop everything
```

## Knobs (env vars)

| var | default | note |
|---|---|---|
| `SUITES` | `spatial object goal long` | subset / reorder |
| `NUM_TRIALS_PER_TASK` | `50` | 50 → 500 rollouts/suite (10 tasks × 50) |
| `SEED` | `7` | |
| `CENTER_CROP` | `True` | leave True — checkpoints trained with random-crop aug |
| `KEEP_MODELS` | `0` | `1` = keep the ~15 GB checkpoint on disk after each suite |
| `CONDA_ENV` | `openvla` | |
| `MUJOCO_GL` | `egl` | switch to `glfw`/`osmesa` if offscreen render fails |

### `run_libero_parallel.sh` extra knobs

| var | default | note |
|---|---|---|
| `MAX_PARALLEL` | `0` | `0` = auto: `floor((free_vram - GPU_RESERVE_MIB) / PER_MODEL_MIB)`; set N to force |
| `PER_MODEL_MIB` | `19000` | VRAM budget per OpenVLA-7B lane (~16–17 GB real + headroom) |
| `GPU_RESERVE_MIB` | `6000` | VRAM left free for other jobs on the shared GPU |
| `HARD_CAP` | `4` | never exceed this many lanes |
| `LAUNCH_STAGGER` | `120` | seconds between lane starts (lets one grab VRAM before the next check) |
| `SESSION` | `libero-eval` | tmux session name |

The orchestrator re-checks `nvidia-smi` before starting each lane and waits until
`free - GPU_RESERVE_MIB >= PER_MODEL_MIB`, so it also backs off when other jobs
are using the GPU. Right now (~55 GB free) it auto-picks **2 lanes**.

## Checkpoint cleanup

Each suite's ~15 GB checkpoint is released automatically on **any** exit:

| how it ends | checkpoint released? |
|---|---|
| suite finishes normally | yes (before next suite) |
| eval script crashes / non-zero exit | yes |
| `Ctrl-C` / `SIGTERM` | yes (EXIT/INT/TERM trap) |
| `tmux kill-session -t libero-eval` | yes (SIGHUP trap; parallel orchestrator forwards TERM to lanes) |
| `SIGKILL` (`kill -9`) / power loss / OOM-killer | **no** — nothing can trap these |
| `KEEP_MODELS=1` | no, kept on purpose |

After a hard kill, recover disk manually:

```bash
~/olive/models/olive-model.sh list                 # see what's still on disk
~/olive/models/olive-model.sh release spatial       # delete one
```

## Notes / caveats

- **Disk**: normally one checkpoint (~15 GB) on disk at a time (2–3 with the
  parallel runner); released between suites unless `KEEP_MODELS=1`.
- **GPU**: single shared GPU (RTX PRO 6000 Blackwell, sm_120), often busy. A full
  4-suite run at 500 rollouts is many hours and contends with other jobs — start
  with the smoke test.
- **torch / flash-attn**: OpenVLA pins torch 2.2.0+cu121 which has **no Blackwell
  kernels** (`CUDA error: no kernel image is available`). The `openvla` env was
  moved to **torch 2.7.0+cu128 + flash-attn 2.8.3** (prebuilt wheel). If flash-attn
  ever breaks after a torch change, run with `OPENVLA_ATTN=sdpa`.
- **LIBERO source** is used from `~/LIBERO/LIBERO` via `PYTHONPATH` (base
  variant), with `LIBERO_CONFIG_PATH=~/.libero_base`. To eval on LIBERO-plus /
  LIBERO-PRO instead, point `LIBERO_SRC` / `LIBERO_CONFIG_PATH` at those.
- Paper reference env: Python 3.10, torch 2.2.0, transformers 4.40.1,
  flash-attn 2.5.5 on an A100. Here torch is 2.7.0+cu128 / flash-attn 2.8.3 on
  Blackwell (forced by the hardware); SR numbers may differ slightly.
- **Results dir**: everything for a run goes under `$RESULT_DIR`
  (default `~/olive/result/baseline`):
  - `logs/<suite>-<timestamp>/` — `console.log` + the eval script's `EVAL-*.txt`
  - `logs/libero-eval-<suite>-<stamp>.out` — per-lane stdout (parallel runner)
  - `logs/orch-<stamp>.log` — orchestrator summary (parallel runner)
  - `rollouts/<DATE>/` — episode MP4s
  - `wandb/` — local W&B run files
  Override with `RESULT_DIR=/path ...`.
- **W&B**: on by default (`USE_WANDB=true`), project `olive-libero-baseline`,
  entity `gibeom-real-1526-kyungpook-national-university`. One run per suite,
  named `EVAL-<suite>-openvla-<datetime>`. Disable with `USE_WANDB=false`.
