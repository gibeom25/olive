# OLIVE (작업 트리)

Frozen OpenVLA-7B backbone 위에서 **z-space latent residual 정책**을 RL 로 학습한다.
`~/olive_ref/` 는 원본 참고용이고, 이 디렉터리(`~/olive/olive/`)가 확장/보강 작업 본체다.

## 규칙

- **`~/olive/openvla` 소스는 수정하지 않는다.** import 해서 쓰는 것만 허용.
  수정이 필요하면 해당 부분을 이 폴더로 **복사(vendor)** 해서 고친다.
  현재 vendor 한 것: `backbone.py`(VLA 로드 로직), `libero_env.py`(이미지 전처리).
  순수 LIBERO 시뮬(`get_libero_env`, benchmark) 은 vendor 불가 → `libero_env.py` 에서만 import.
- **모든 ablation 축은 `config.py` 의 타입 필드로 뺀다.** 모듈은 config 값을 받아
  동작하고 상수를 하드코딩하지 않는다. (아래 "Ablation 축" 참고)

## 원래 논문과 다른 점 (지금까지 확정)

| | 논문 | 여기 |
|---|---|---|
| 정책 분포 | action-space 7-dim Gaussian + 미분가능 soft-decode | **z-space 256-dim Gaussian** (rollout 은 greedy decode, gradient 불필요) |
| 학습 순서 | RL 만 (warm-start = gain 0.01 init) | **LIBERO demo 로 Actor/Critic 지도학습(BC) → 그다음 PPO** |
| backbone | frozen | frozen (동일) |

## 파일 구조 (데이터 흐름 순)

```
config.py        모든 설정 (OliveConfig = 경로 + 네트워크 + BC + PPO + rollout + wandb)
backbone.py      frozen VLA + processor 로드, vocab_size / norm_stats / hidden_dim 추출
network.py       Actor/Critic MLP  — 학습 대상 (~2.2M param, float32)
                   Actor : h(4096) -> (mu_z, std_z)   z_dim=256   (z-space Gaussian)
                   Critic: h(4096) -> V(s)
projection.py    Action Basis Projection — 학습 파라미터 0 (frozen buffer)
                   B = lm_head.weight[vocab-256:vocab]            (Eq.4)
                   delta_h(z) = z @ B_inject                       (Eq.6)
                   apply_residual(h,z) = h + alpha*delta_h         (Eq.7)
                   decode_action(...)      : 7-step greedy (PPO rollout, no_grad)
                   decode_action_soft(...) : 7-step 미분가능 (BC warm-up 전용)
reward.py        r_t = +1 (done) else -lambda*t                    (Eq.12)
libero_env.py    LIBERO task/env glue — LIBERO import 를 여기로 격리
                   make_task_suite / make_env / TASK_MAX_STEPS / get_image / dummy_action
rollout.py       get_prefill_state (frozen backbone 1회 forward -> h0)
                 RolloutBuffer (h0, z, log_prob, value, reward, done + GAE)
                 collect_episode / collect_rollouts  (z 샘플 -> greedy decode -> env.step)
ppo_trainer.py   PPOConfig (Table 1) + PPOTrainer.update(buffer)  (Algorithm 2 line 9-13)
bc_warmup.py     LIBERO demo 지도학습: Actor(BC) + Critic(return-to-go 회귀)
                   LiberoDemoDataset -> (image, instruction, expert_action[7])
evaluate.py      deterministic rollout 로 suite SR 측정 (baseline 과 동일 조건)
checkpoint.py    head(~2.2M) + optimizer + step 저장/로드
train.py         메인 드라이버: [BC warm-up] -> [PPO 루프] -> 주기적 eval / ckpt / wandb
```

## 실행

flat import 구조라 **`~/olive/olive/` 안에서 스크립트로** 실행한다 (`python -m` 아님).

```bash
conda activate openvla
export PYTHONPATH=$HOME/olive/openvla:$HOME/LIBERO/LIBERO
export LIBERO_CONFIG_PATH=$HOME/.libero_base MUJOCO_GL=egl
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
cd ~/olive/olive

CKPT=~/olive/models/openvla-7b-finetuned-libero-spatial   # olive-model.sh fetch spatial 로 먼저 받기

python train.py --suite libero_spatial --pretrained_checkpoint $CKPT --ablation main
python train.py --suite libero_spatial --pretrained_checkpoint $CKPT --no-bc
python train.py --suite libero_spatial --pretrained_checkpoint $CKPT --ablation random_basis
python train.py --suite libero_10 --pretrained_checkpoint $CKPT2 --set ppo.entropy_coef=0.0005

python evaluate.py --suite libero_spatial --vla $CKPT --ckpt ~/olive/result/<run>/ckpt/best.pt
```

각 모듈은 `python <file>.py` 로 자체 스모크 테스트가 돈다 (7B/env 없이 되는 부분만).
config override: `--ablation <name>` (프리셋) + `--set a.b.c=val` (반복 가능, 프리셋 뒤 적용).

### scripts/ (env 세팅 + 단계별 래퍼)

```bash
cd ~/olive/olive/scripts
# 공통 env: SUITE(libero_spatial) ABLATION(main) SEED RUN_NAME  — env var 로 override

SUITE=libero_spatial ABLATION=main ./01_bc_warmup.sh   # LIBERO demo 추가학습 -> ckpt/after_bc.pt
SUITE=libero_spatial ABLATION=main ./02_ppo_train.sh   # rollout PPO (after_bc.pt 자동 init) -> ckpt/best.pt
SUITE=libero_spatial ABLATION=main ./03_eval.sh        # best.pt 고정 후 suite SR 측정
./04_wandb_report.sh                                    # olive-libero 프로젝트 run 표
```

- `_env.sh`: conda + PYTHONPATH + LIBERO + torch env, `fetch_vla`(15GB 체크포인트 자동 fetch).
- `02` 는 `FRESH=1` 로 BC 무시 순수 PPO. `PPO_TOTAL_STEPS` / `ENTROPY_COEF` / `EVAL_EVERY` 등 env override.
- `train.py --phase {bc|ppo|all}` 로 단계 분리, `--init-ckpt` 로 BC 결과를 PPO 초기값으로.
- VLA 체크포인트는 세 단계가 공유하므로 자동 release 안 함 → 끝나면 `~/olive/models/olive-model.sh release <alias>`.

## Ablation 축 (config.py 필드)

| config 필드 | 값 | 검증하는 것 |
|---|---|---|
| `proj.inject_basis` | `lm_head` / `random` / `unconstrained` | action basis 제약의 효과 (논문 §4.4). `unconstrained` = 학습 4096 projection = full-space Residual RL |
| `proj.alpha` | float sweep | residual scale 민감도 |
| `proj.critic_input` | `pre`(h0) / `post`(h0') | critic 이 주입 전/후 어느 state 를 볼지 (논문은 pre) |
| `net.actor_output_gain` | `0.01` / `1.0` | warm-start init 효과 |
| `net.z_dim` | `256` / 128 / 512 | latent 용량 (논문 §4.5: object diversity 에 256 부족 가능성) |
| `bc.enabled` | `True` / `False` | **BC warm-up 자체의 효과 (이번 보강의 핵심)** |
| `bc.targets` | `both` / `actor` / `critic` | 무엇을 사전학습할지 |
| `bc.demo_glob` | 경로 | 표준 LIBERO vs modified_libero_rlds |
| `ppo.entropy_coef` | float | z-space entropy 스케일 (아래) |
| `ppo.reward` | `paper` / ... | reward 설계 |

`train.py --ablation <name>` 로 프리셋 묶음도 제공 (config.py:ABLATIONS).

## 미정 / 결정 필요

- **z-space entropy 스케일**: entropy bonus 가 256-dim 이라 논문(7-dim) 대비 ~37배 셈.
  `PPOConfig.entropy_coef` 를 낮추거나 per-dim 정규화 필요 (train 붙이기 전에).
- **BC 데이터셋**: 표준 LIBERO demo(HDF5) vs `modified_libero_rlds`(SFT 체크포인트 학습본).
  `config.py:BCConfig.demo_glob` 로 지정. 현재 기본 = 표준 LIBERO HDF5.
- **파라미터 수**: 실제 ~2.2M (논문 "300K" 과 불일치). low-rank 병목/공유 투영으로 줄일지.
- **total_steps**: Table 1 은 50000, Algorithm 2 본문은 5000 — `PPOConfig.total_steps` 설정값.
```
