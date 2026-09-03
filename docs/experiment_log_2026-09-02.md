# OLIVE 코드베이스 통합 검증 (shakedown) — 이슈 및 수정

- **일시:** 2026-09-02 09:37 KST 시작 / 09:42 종료
- **대상:** `~/olive/olive/` 전체 파이프라인 (backbone 로드 → BC warm-up → PPO → eval → checkpoint)
- **환경:** conda `openvla`, torch 2.7.0+cu128, flash-attn 2.8.3, RTX PRO 6000 (공유, 여유 ~59GB),
  체크포인트 `openvla-7b-finetuned-libero-spatial` (로컬, `olive-model.sh fetch spatial`)
- **방법:** 2단계
  1. **Stage 1** — 통합 스모크: backbone 로드 + `get_prefill_state` + `ac.act` +
     greedy/soft decode + 역전파
  2. **Stage 2** — `train.py` 축소 실행: BC 1 demo file / PPO 2 iter (buffer 24) /
     eval 10 task × 1 ep, `max_steps=25`

---

## 결과 요약

| Stage | 결과 |
|---|---|
| Stage 1 (통합 스모크) | ✅ PASS (수정 #1 후) |
| Stage 2 (`train.py` 전체) | ✅ PASS (수정 #1, #2 후) — exit 0, 1.3분 |
| 후속: `checkpoint.load` 라운드트립 + `evaluate_suite` 단독 호출 | ✅ PASS |

**파이프라인 전 구간 동작 확인.** SR 은 전부 0.0% 이지만 이건 기능 검증용 축소 실행
(BC 1 epoch / PPO 48 step) 결과이지 성능 지표가 아님.

실행 로그(발췌):
```
[OLIVE][backbone] loaded ... hidden_dim=4096 vocab_size=32000 attn=flash_attention_2
[OLIVE] trainable params: {'actor': 1114880, 'critic': 1049089, 'total': 2163969}
[OLIVE][bc] done  {'n_frames': 224, 'n_train': 213, 'n_val': 11,
                   'actor_loss': 0.441, 'critic_loss': 409.0,
                   'val_actor_mae': 0.411, 'val_critic_mae': 9.49}
[OLIVE] BC done -> .../shakedown/ckpt/after_bc.pt
[OLIVE] it   1/2  step 25  train_SR 0.00  π +0.148  V 15.60  H -148.7  KL 2.830  gn 168.06  lr 3.0e-04
[OLIVE] it   2/2  step 50  train_SR 0.00  π -0.025  V  3.31  H -148.7  KL 0.404  gn 319.39  lr 1.5e-04
[OLIVE]   eval SR 0.0%
[OLIVE] done. best eval SR = 0.0%  (1.3m)
생성: ckpt/{after_bc.pt(8.6MB), best.pt(26MB), it0002.pt(26MB)}
```

---

## 수정한 이슈

### #1  `projection.py` — action 텐서가 bfloat16 → `.numpy()` 불가

- **증상:** `TypeError: Got unsupported ScalarType BFloat16`
  (`rollout.collect_episode`: `projection.z_to_action(...).squeeze(0).cpu().numpy()`)
- **원인:** `ActionBasisProjection` 이 `bin_values` buffer 를 backbone dtype(bf16)로 등록
  → `decode_action` / `decode_argmax` 반환 action 이 bf16 → numpy 는 bf16 미지원.
- **수정:** `bin_values` 를 `float32` 로 등록.
  ```python
  # projection.py  ActionBasisProjection.__init__
  self.register_buffer("bin_values", compute_action_bin_values(num_bins).float(), persistent=False)
  ```
- **검증:** Stage 1 재실행 → greedy/soft decode 모두 `dtype=torch.float32`, unnormalize 통과.

### #2  `config.py` — `Optional[int]` 필드 CLI override 가 float 로 캐스팅

- **증상:** `TypeError: slice indices must be integers or None or have an __index__ method`
  (`bc_warmup._iter_frames`: `files[: cfg.bc.max_demos]`, `max_demos == 1.0`)
- **원인:** `_set_by_path` 가 `cur is None`(= `Optional` 기본값) 인 필드에 대해 무조건
  `float(value)` 시도 → `--set bc.max_demos=1` 이 `1.0` 이 됨.
- **수정:** None 필드는 `none/null` → `None`, 정수꼴 → `int`, 실수꼴 → `float`, 그 외 → `str`.
  ```python
  # config.py  _set_by_path,  elif cur is None:
  if s.lower() in ("none", "null", ""): value = None
  elif s.lstrip("-").isdigit():         value = int(s)
  else:
      try: value = float(s)
      except ValueError: value = s
  ```
- **검증:** `--set bc.max_demos=1` → `int 1`, `--set ppo.target_kl=0.02` → `float`,
  `--set bc.max_demos=none` → `None`.

---

## 관찰 (버그 아님, 후속 조치 필요)

| 항목 | 내용 | 조치 |
|---|---|---|
| **z-space entropy 스케일** | `KL 2.8`, clip 전 `grad_norm 168~319` (→ 1.0 클리핑). 정책 분포가 256-dim 이라 `entropy_coef=0.01` 이 논문(7-dim) 대비 과함이 실측으로 확인됨. | 본 학습 전 `--ablation ent_perdim` 또는 `--set ppo.entropy_coef=0.0003~0.001` 로 조정. `H -148.7` 이 iter 간 거의 안 움직임 = log_std 학습이 미미. |
| `bc.max_demos` 의미 | "HDF5 **파일 수**"만 제한. 파일 하나에 demo 50개 → shakedown 이 224 frame. | "파일당 demo 수" 제한 옵션(`bc.max_demos_per_file`) 추가 검토. 본 학습에선 전체 사용이 맞으니 우선순위 낮음. |
| BC critic 타깃 스케일 | `critic_mse ~409`, `val_critic_mae 9.5`. return-to-go 가 `-λ·t` 누적이라 t=0 에서 수십 규모 → MSE 큼. | 정상. critic 은 random init + 1 epoch. 본 학습에서 epoch 늘리면 수렴. reward normalize 여부는 추후. |
| `ac.act` log_prob 큼 (~171) | 256-dim Gaussian, std=exp(-2). 차원별 log-density 합. | 정상 (PPO 는 ratio = exp(new-old) 라 절대값 무관). |
| `[Warning]: datasets path ... does not exist!` 스팸 | LIBERO(base) 에 demo 없음 → `bc_warmup` 이 `demo_glob`(LIBERO-plus) 로 폴백. eval 중 env 생성마다 출력. | 무해. 필요 시 로그 필터. |
| BC 프레임 전처리 시간 | `list(_iter_frames(...))` 가 224 frame 을 upfront materialize (tf resize). 무출력 구간. | 본 학습(전체 demo, 수천 frame)에선 수 분 소요 예상. tqdm/스트리밍 검토. |

---

## 검증된 것 (7B VLA + LIBERO env 포함)

- `backbone.load_backbone` — bf16, flash_attention_2, `norm_stats` 로드, hidden_dim/vocab_size 추출
- `rollout.get_prefill_state` — 프롬프트 forward, `h0 (1,1,4096)`, KV cache, mask `(1,283)`
- `network.ActorCritic.act` / `.evaluate_z`
- `projection`: `apply_residual`, `decode_action`(greedy 7-step, backbone 루프),
  `decode_action_soft`(미분가능, actor 까지 gradient 확인 grad_sum≈18454)
- `reward.compute_reward_paper`
- `libero_env`: `make_task_suite`, `make_env`(OffScreenRenderEnv), `get_image`(openvla `get_libero_image` 재사용)
- `bc_warmup`: HDF5 로드(`_instruction_from_hdf5`, `agentview_rgb`), BC + critic 회귀, 저장
- `rollout.collect_rollouts` / `collect_episode` — LIBERO env step, gripper 후처리, buffer 적재
- `ppo_trainer.PPOTrainer.update` — GAE, minibatch, clipped surrogate, grad clip, LR schedule
- `evaluate.evaluate_suite` — deterministic rollout, per-task/total SR
- `checkpoint.save` / `load` — head(+optimizer+config) 저장, 로드 라운드트립

## 아직 미검증

- **wandb 로깅 경로** (shakedown 은 `--no-wandb`). `train.py` 의 `wandb.init/log/finish` 는
  코드상 문제 없어 보이나 실제 연결은 다음 실행에서 확인.
- `inject_basis=unconstrained` / `random` ablation 경로 (기본 `lm_head` 만 실행).
- `--phase ppo --init-ckpt` 로 BC→PPO 분리 실행 (shakedown 은 `--phase all`).
  → `checkpoint.load` 는 단독 검증됨, 조합만 미확인.
- 스크립트 `01/02/03` 실제 실행 (wiring/`_env.sh` 는 dry-check 통과).

## 남은 산출물

- `~/olive/result/shakedown/` (58MB) — 검증용, 삭제 가능
- `~/olive/models/openvla-7b-finetuned-libero-spatial` (15GB) — 다음 실학습에 사용, 유지

---

# 첫 실학습: BC warm-up (libero_spatial, ablation=main)

- **일시:** 2026-09-02 10:23 → 13:56 (~3.5h), `--phase bc`, exit 0
- **설정:** frame_stride=5, epochs=3, batch 64, lr 1e-4. 12,648 frames (12,016 train / 632 val)
- **결과:**

  | epoch | actor_mse | critic_mse |
  |---|---|---|
  | 1 | 0.3751 | 159.6 |
  | 2 | 0.3727 | 116.1 |
  | 3 | 0.3718 | 105.0 |

  `val_actor_mae 0.330`, `val_critic_mae 7.58`. → `after_bc.pt` (8.6MB).

- **문제: Actor BC 사실상 미학습.** critic 은 수렴 중(160→105)인데 actor_mse 는 epoch 1
  부터 0.37 정체(3 epoch에 0.3%). 후보 원인:
  1. soft-decode 의 softmax 포화 → `decode_expected` 통과 gradient ≈ 0
     (projection 스모크에서 예고했던 현상)
  2. `bc.lr=1e-4` 낮음 + `actor_output_gain=0.01` → mu_z 극소, 성장 느림
  3. BC 목표 난이도 (SFT 이미 ~85% → residual 신호가 작고 noisy)
- **해석:** `after_bc.pt` 의 actor ≈ warm-start init (mu_z≈0 → Δh≈0) ≈ **frozen SFT 정책**.

## 로깅 상세화 (이 실행 뒤 반영, 다음 실행부터 적용)

| 파일 | 변경 |
|---|---|
| `scripts/_env.sh` | `PYTHONUNBUFFERED=1` — tee redirect 시 실시간 로그 |
| `bc_warmup.py` | frame 로딩 파일별 진행 / `log_every`(20) batch 마다 `a~ c~ gn(a/c)~ fr/s ETA %` + wandb 기록 / epoch 마다 `after_bc_epN.pt` 저장 / actor·critic grad norm 분리 |
| `train.py` | rollout 수집 시작·완료 로그, eval verbose, wandb explicit `step=` 제거 후 `ppo/global_step` 메트릭화, `bc_warmup(wandb_run=run)` |
| `rollout.py` | `collect_rollouts` 에피소드마다 `epN OK/fail steps buffer x/y` 출력 |
| `config.py` | `bc.log_every=20`, `bc.save_every_epoch=True` |

## BC 결과 rollout 평가 (`03_eval.sh`, EVAL_EPISODES=5) — 완료

- 2026-09-02 14:07 → 완료, exit 0. `after_bc.pt` 자동 선택, ckpt config 에서 z_dim=256 / inject=lm_head 추출.
- **결과: TOTAL SR 86.0%** (50 ep = 10 task × 5)

  | task | SR | task | SR |
  |---|---|---|---|
  | 0 | 100% | 5 | 100% |
  | 1 | 100% | 6 | 100% |
  | 2 |  80% | 7 |  80% |
  | 3 | 100% | 8 |  40% |
  | 4 |  60% | 9 | 100% |

- **baseline(spatial, 20 ep/task) 85.0% 대비 86.0% → BC 는 사실상 neutral.**
  - eval 은 `deterministic=True` → z = mu_z. actor 가 BC 에서 안 움직여 mu_z≈0 → Δh≈0 →
    **decode = frozen SFT 정책 그대로**. 즉 이 eval 은 사실상 SFT 정책을 측정한 것.
  - critic 은 BC 로 학습됐지만(mse 160→105) deterministic rollout 엔 영향 없음(PPO advantage
    에서만 쓰임) → 이 평가에선 안 보임.
- task 8 40%(baseline 90%), task 4 60%(baseline 75%) 는 5-ep 노이즈 범위(±~22% 95% CI).

### 이 평가로 확인된 것
- ✅ rollout/eval 파이프라인 실하드웨어 end-to-end 동작 (backbone + prefill + z-space act +
  greedy decode + LIBERO env.step + gripper 후처리 + suite SR 집계)
- ✅ BC 가 정책을 망가뜨리지 않음 (86 ≈ 85)
- ⚠️ BC-actor 는 no-op (actor_mse 정체에서 이미 확인). BC 의 유일한 효과 = critic warm-up (미검증)
- ⏳ PPO 경로는 아직 실하드웨어 미검증

### 다음 fork
- **A. `after_bc.pt` 로 PPO 진행** (`02_ppo_train.sh`). critic warm-start + 논문식 actor
  warm-start(≈identity) 상태. 논문 OLIVE 는 BC 안 함 → 이 상태가 오히려 논문에 가까움.
  첫 시도는 `PPO_TOTAL_STEPS=5000`(Algorithm 2 값, ~10 iter, ~40분) 로 SR 움직임 확인 후 풀런.
- **B. Actor BC 수정 후 재실행**: actor grad norm 로깅(추가함)으로 softmax 포화 확인 →
  lr 상향 / decode softmax temperature / 목표 재설계. ~1~3h 재BC.

---

# 첫 PPO 실행 (짧은 검증, `--phase ppo --init-ckpt after_bc.pt`)

- **일시:** 2026-09-02 14:31 → 17:35 (~183분), `ppo.total_steps=5000` (10 iter × 512),
  `eval_every=5`, `eval_episodes_per_task=20`(기본, 그래서 iter 5·10 eval 이 각 ~30분). exit 0.
- **결과:**

  | | SR (spatial, 200 ep) |
  |---|---|
  | baseline | 85.0% |
  | after_bc (5 ep) | 86.0% |
  | PPO it 5 | 83.5% |
  | **PPO it 10 (best)** | **84.0%** |

  → **5000 step PPO 로 SR 이 안 오름 (85 → 84, 노이즈 범위).**

- **원인: KL divergence 폭발.**

  | iter | 6 | 7 | 8 | 9 | 10 |
  |---|---|---|---|---|---|
  | approx_kl | 1340 | 469 | 299 | 123 | 24 |

  정상 PPO 는 KL ~0.02. iter 6~10 에서 KL 이 줄어드는 건 정책 안정이 아니라 LR decay
  (1.4e-4 → 1.7e-5) 때문. 매 업데이트마다 정책이 완전히 딴 정책으로 튐.
- **근본 원인 (예전부터 플래그, 이번에 실측 확인):**
  1. **z-space entropy 과대** — 정책이 256-dim 이라 `entropy_coef=0.01` 이 논문(7-dim) 대비 ~37x.
     entropy bonus 가 정책을 마구 밀어냄. (`H ~ -140` 은 clamp 로 고정)
  2. **`target_kl` 미설정** — KL 폭발해도 PPO epoch 조기종료 안 함
  3. LR 3e-4 도 다소 높을 수 있음
- **부수:** `02_ppo_train.sh` 를 두 번 띄움(14:30·14:31) → 첫 건 it1 에서 죽고 둘째가 실행됨.
  wandb run 2개 생김(`143039`, `143142`) — 무해하나 clutter.

## PPO 로 확인된 것
- ✅ **PPO 경로 실하드웨어 end-to-end 검증** (rollout 수집 → GAE → clipped update → eval →
  checkpoint). 파이프라인 마지막 미검증 조각 해결.
- ⚠️ 현 하이퍼파라미터로는 RL 이 작동 안 함.

## 다음: PPO 하이퍼파라미터 수정 후 재검증

```bash
# 5000 step 재검증 (빠르게)
ENTROPY_COEF=0.0003 EVAL_EPISODES=5 PPO_TOTAL_STEPS=5000 EVAL_EVERY=5 \
  ./02_ppo_train.sh --set ppo.target_kl=0.05 --set ppo.lr_start=1e-4
# 또는 --ablation ent_perdim (entropy_coef / z_dim)
```
- `entropy_coef` 3e-4 (또는 ÷256), `target_kl` 0.05, `lr_start` 1e-4
- `eval_episodes_per_task` 5 로 (기본 20 이면 iter 당 30분 eval)
- KL 이 0.01~0.1 대로 내려오는지 확인 → 그다음 풀런(50000)

## 산출물

- `~/olive/result/olive-libero_spatial-main/ckpt/` : `after_bc.pt`(8.6MB),
  `best.pt`/`it0010.pt`(26MB, eval_sr 0.84)
- 로그: `logs/01_bc_*.log`, `logs/02_ppo_20260902_143130.log`, `logs/03_eval_*.log`
- wandb: run `olive-libero_spatial-main` (`olive-libero` project)

---

# PPO 수정 + round-robin + SR 지표 + 전체 suite 드라이버 (2026-09-02 17:xx)

## 1. PPO 하이퍼파라미터 기본값 조정 (`config.py:PPOConfig`) — KL 폭발 대응

| 필드 | 이전(논문 action-space) | 지금(z-space 보정) |
|---|---|---|
| `entropy_per_dim` | False | **True** (실효 entropy_coef = 0.01 / z_dim ≈ 3.9e-5) |
| `target_kl` | None | **0.05** (KL > 1.5·target 이면 그 iteration 의 남은 epoch 중단) |
| `lr_start` | 3e-4 | **1e-4** |

- ablation 프리셋 `ent_perdim` 제거(이제 기본), `ent_full`(per_dim 끔) / `paper_ppo`(셋 다 논문값) 추가.
- `_set_by_path` 에 `value is None` 명시 처리 (ABLATIONS dict 의 `None`).

## 2. rollout task 편향 방지 (`rollout.py` / `train.py` / `config.py`)

- **이전 버그:** `collect_rollouts` 가 매 PPO iteration 마다 `tasks[0]` 부터 시작 → 512 buffer 가
  task 0 에피소드 몇 개로 참 → **매 iteration 이 task 0 만 학습** (지난 PPO 로그에 rollout 이
  전부 "black bowl between the plate and the ramekin" 이었던 이유).
- **수정:** `collect_rollouts(task_cursor=, init_cursor_of=)` — `task_cursor` 를 PPO iteration
  간 유지(train.py) → tasks[0] 리셋 안 하고 이어서 순회. `rollouts_per_task` 기본 5 → **2**
  → 한 iteration 이 ~5개 task 를 훑음. `init_cursor` 도 유지 → 재방문 시 다른 init state.
  반환에 `n_tasks_visited`, `per_task {n, succ}`.

## 3. RL 특성 반영 SR 지표 (`train.py`)

iteration 로그: `SR(iter/ema/cum)` = 이번 iter rollout SR / 지수이동평균(α=0.1) / run 전체 누적.
wandb: `train/rollout_sr`, `train/rollout_sr_ema`, `train/rollout_sr_cum`,
`train/rollout_episodes_cum`, `train/n_tasks_visited`.

## 4. 전체 suite 드라이버 `scripts/run_all_suites.sh`

- 4 suite 순차: 각 suite 자기 SFT ckpt 에서 시작(`_env.sh` 매핑 + `fetch_vla`) → BC → PPO → eval.
- suite 끝나면 VLA ckpt(15GB) release. 실패해도 다음 suite 계속. `.done` 마커로 RESUME.
- 요약 `~/olive/result/run_all_<stamp>.md` 실시간 갱신.
- env: `SUITES`, `PPO_TOTAL_STEPS`(기본 5000, 본학습 50000), `EVAL_EPISODES`(iter, 기본 5),
  `EVAL_FINAL_EPISODES`(최종, 기본 20), `BC_FRAME_STRIDE`(기본 10), `RESUME`, `DRY`.
- **예상 시간:** 5000-step + BC stride 10 + final 20ep → suite 당 ~5~6h → 4 suite ~24h.
  첫 pass 는 `EVAL_FINAL_EPISODES=10` 권장.

## 상태
전 모듈 parse + 스모크 통과, `run_all_suites.sh` DRY 확인.

---

# 첫 전체 실행 결과 (long + goal, 2026-09-02 18:16 → 09-03 00:07, ~6h)

설정: `BC_TARGETS=critic`(actor no-op 확인돼서 skip), `BC_FRAME_STRIDE=15`,
`PPO_TOTAL_STEPS=5000`(10 iter), `EVAL_EVERY=10`, `EVAL_FINAL_EPISODES=10`,
PPO 기본값 = entropy_per_dim/target_kl 0.05/lr 1e-4.

| suite | 우리 baseline | OLIVE(여기) | OLIVE 논문 | 논문 baseline | BC critic_mse | 소요 |
|---|---|---|---|---|---|---|
| libero_10 | 53.5% | **48.0%** (best 50.0) | 50.0 | 40.0 | 4587→2197 | 239m |
| libero_goal | 77.5% | **72.0%** (best 70.0) | 71.4 | 70.0 | 1150→518 | 111m |

→ 우리 수치(48, 72) ≈ OLIVE 논문 수치(50, 71.4). 단 **우리 자체 baseline(53.5, 77.5)
대비로는 소폭 하락.**

## ✅ KL 폭발 해결됨

하이퍼파라미터 수정(entropy_per_dim=True, target_kl=0.05, lr 1e-4)이 먹혔다:

| | 첫 PPO(spatial) | 이번(long) | 이번(goal) |
|---|---|---|---|
| approx_kl 범위 | 24 ~ 1340 | 0.011 ~ 0.16 | 0.015 ~ 0.09 |

PPO 가 이제 안정적. (첫 iter 만 살짝 높고 이후 target 내)

## ⚠️ 그래도 baseline 못 넘음 — 원인 분석

1. **value 타깃 미정규화 → grad_norm 폭발 → 하드 클리핑 → 실효 업데이트 극소.**
   `gn` = 500~2000 (max_grad_norm 1.0 로 매 스텝 클립). value loss `V` = long 300~626 /
   goal 77~210. return-to-go(`-λ·t` 누적)가 long 에서 -50~-150 규모라 `(V-ret)²` 가 거대.
   → **returns/value 정규화(running mean/std) 필요.** 현재 advantage 만 정규화.
2. **5000 step 이 너무 짧음.** 논문 Table 1 = 50000. 10 iter 로는 task 당 ~1~2 에피소드
   학습 신호. Algorithm 2 본문의 5000 은 검증용.
3. **long 은 buffer 512 로 task 1개/iter** (에피소드 500~740 스텝). `task_cursor` 유지로
   10 iter 에 10 task 각 1회는 방문하나 per-task 신호 미미. goal 은 2~3 task/iter (OK).
4. **actor 는 순수 warm-start** (critic-only BC). rollout 시 `z ~ N(0, σ≈0.135)` 256-dim
   → `Δh = z@B` 가 SFT 정책을 흔듦 → 탐색 노이즈가 손해, 5000 step 으로는 이득 보는
   μ_z 학습 불가 → 순 하락.

## 다음 (우선순위)

1. **value/return 정규화** (`ppo_trainer.py`: RunningMeanStd 로 return 정규화 후 critic 학습).
   grad_norm 정상화 → 실효 업데이트 회복. **가장 큰 효과 예상.**
2. **batched/vectorized rollout+eval** (사용자 합의) — 없으면 50000-step 이 며칠. 이게 되면
   본 학습 현실화.
3. 그다음 `PPO_TOTAL_STEPS=50000` + (선택) actor BC 재도전.

## 산출물
- `~/olive/result/olive-libero_10-main/` , `olive-libero_goal-main/` : ckpt(`after_bc.pt`,
  `after_bc_ep{1,2,3}.pt`, `best.pt`, `it0010.pt`), logs, `.done`
- 요약: `~/olive/result/run_all_20260902_181601.md`
- wandb: `olive-libero_10-main`, `olive-libero_10-main-eval`, `olive-libero_goal-main`,
  `olive-libero_goal-main-eval`
- VLA 체크포인트 2개 release 됨 (디스크 반환). spatial 것만 15GB 남아있음.

