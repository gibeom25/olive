# 전체 suite 실행 (2026-10-29)

브랜치 `exp/full-run-20261029`. 결과물: `experiments/full_run_20261029/`
(`results.json`, task 별 대표 영상, suite 별 `metrics.jsonl` / `final_eval.json` / 로그).
상세 결과 페이지는 `experiments/full_run_20261029/index.html` (같은 폴더의 `results.json` 을 읽음).

## 결과

| suite | Frozen OpenVLA (200 ep, 09-01) | OLIVE (100 ep) | Δ | 논문 baseline | 논문 OLIVE |
|---|---|---|---|---|---|
| Spatial | 85.0 | **82.0** | −3.0 | 60.0 | 91.9 |
| Object | 87.0 | **88.0** | +1.0 | 88.0 | 38.5 |
| Goal | 77.5 | **73.0** | −4.5 | 70.0 | 71.4 |
| Long (10) | 53.5 | **58.0** | +4.5 | 40.0 | 50.0 |

- 100 에피소드의 95% 구간이 ±7–10pp 라 네 suite 모두 baseline 과 통계적으로 구분되지 않음.
- 9월 run(long 48.0, goal 72.0) 대비 long +10pp. goal 은 비슷.
- Object 하락(논문 38.5%)은 재현되지 않음.
- Baseline 재실행(task 당 1 ep, 영상용): spatial 90 / object 90 / goal 80 / long 50 — 9월 수치와 정합.

## 설정

- `result/run_full_20261029.sh`, `result/run_deferred_20261029.sh` (object/goal 은 우선순위를 뒤로 미뤄 spatial·long 다음에 시작)
- BC: critic 만, frame stride 15, 3 epoch. PPO 5000 step (10 iter), 학습 중 eval 1 ep/task, 최종 eval 10 ep/task.
- 이번 변경: `ppo.normalize_value_loss` (value loss / running Var(return)), `ppo.separate_grad_clip` (actor/critic 개별 클립). `--ablation no_vnorm` 으로 이전 동작.
- 소요: spatial 178분, object 170분, goal 193분, long 354분 (GPU 1장에 2–5 레인 병렬).

## 학습 지표

| suite | actor grad norm | critic grad norm | approx KL | rollout SR (누적) |
|---|---|---|---|---|
| Spatial | 100–184 | 0.61–1.69 | ≤ 0.090 | 0.79 |
| Object | 107–159 | 0.32–1.63 | ≤ 0.168 | 0.90 |
| Goal | 89–330 | 0.58–1.87 | ≤ 0.097 | 0.76 |
| Long | 84–165 | 0.55–2.70 | ≤ 0.171 | 0.67 |

- Critic grad norm 이 9월 500–2000 → 1 근처. value loss 정규화로 해결.
- Actor grad norm 은 여전히 84–330 으로 매 스텝 max 1.0 클리핑 → 실효 업데이트가 작음. 다음 수정 1순위.

## 운영 메모

- 레인 5개 동시 실행 시 GPU 포화로 에피소드당 3배 이상 느려짐 → baseline 은 9월 수치 사용, 학습 중 eval 축소, object/goal 후순위로 조정.
- `openvla` env 는 `PYTHONNOUSERSITE=1` 로 `~/.local` 과 격리해서 실행 (env 설치 때 pip 이 `~/.local` 의 huggingface_hub 2.1.1, wandb 0.28.0, numpy 2.2.6, protobuf 4.25.9, absl-py 2.5.1, tensorflow-metadata 1.21.0 을 제거함).
- push 403: 전역 credential store 토큰 대신 이 레포만 `gh auth git-credential` 사용하도록 로컬 설정.

## 다음

1. Actor gradient 스케일 조정 (log-prob 정규화 또는 actor 전용 clip 값).
2. 50,000 step (논문 Table 1) — batched rollout 필요.
3. Long/Goal 평가 에피소드 확대 (task 단위 차이 검증).
4. 논문 baseline(60/88/70/40)과의 평가 설정 차이 확인.
