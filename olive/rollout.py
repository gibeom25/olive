"""
OLIVE: rollout 수집.

frozen VLA backbone + z-space Actor/Critic(network.py) + ActionBasisProjection
(projection.py) 로 LIBERO 환경에서 에피소드를 굴리고, PPO 업데이트에 필요한
transition 을 RolloutBuffer 에 채운다.

한 timestep
-----------
    raw_obs ──get_prefill_state──▶ h0 (B,1,4096), past_key_values, attention_mask
                                    │  (frozen backbone 1회 forward, no_grad)
    ActorCritic.act(h0.float()) ──▶ z (B,256), log_prob, value       # z-space Gaussian 샘플
    proj.z_to_action(h0, z, pkv, mask) ──▶ a_norm (B,7)  ∈ [-1,1]     # h0+z→h0'→7-step greedy decode
    unnormalize(a_norm, norm_stats) ──▶ a_robot                       # OpenVLA 규약
    normalize_gripper_action / invert_gripper_action                  # LIBERO eval 파이프라인과 동일
    env.step(a_robot) ──▶ obs, _, done, _
    r = reward_fn(done, t)                                            # 논문 Eq.12
    buffer.add(h0, z, log_prob, value, r, done)

dtype
-----
ActorCritic 는 float32 로 두고 h0 를 .float() 해서 먹인다 (Normal 분포 numerics).
z→Δh 의 bfloat16 캐스팅은 projection.py 안에서 처리된다. h0 자체는 backbone dtype
(bfloat16) 그대로 버퍼에 저장하고, 업데이트 때 .float() 한다.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Iterable, Iterator

import numpy as np
import torch
from PIL import Image

# vanilla OpenVLA 규약: 프롬프트가 이 토큰(SentencePiece "▁")으로 끝나지 않으면 붙인다.
# openvla predict_action 과 동일.
EMPTY_TOKEN_ID = 29871

# LIBERO eval(run_libero_eval.py)과 동일한 gripper 후처리를 위해 openvla repo 것을 재사용.
try:
    from experiments.robot.robot_utils import normalize_gripper_action, invert_gripper_action
except Exception:  # openvla repo 가 PYTHONPATH 에 없을 때도 import 자체는 되도록
    normalize_gripper_action = invert_gripper_action = None


# ─────────────────────────────────────────────────────────────────────────────
@dataclasses.dataclass
class RolloutConfig:
    unnorm_key: str = "libero_spatial"      # vla.norm_stats 키 (= task_suite_name)
    center_crop: bool = True                # 체크포인트가 random-crop aug 로 학습됨 → True 유지
    resize_size: int = 224                  # OpenVLA 입력 해상도
    max_steps: int = 220                    # suite 별로 바깥에서 덮어쓴다 (spatial 220 / object 280 / goal 300 / 10 520)
    num_steps_wait: int = 10                # 시뮬 물체 안정화 대기 (그동안 dummy action)
    gripper_postprocess: bool = True        # normalize+invert gripper (LIBERO)
    device: str = "cuda:0"


# ─────────────────────────────────────────────────────────────────────────────
def _center_crop_pil(image: Image.Image, crop_scale: float = 0.9) -> Image.Image:
    w, h = image.size
    cw, ch = int(round(w * (crop_scale ** 0.5))), int(round(h * (crop_scale ** 0.5)))
    left, top = (w - cw) // 2, (h - ch) // 2
    return image.crop((left, top, left + cw, top + ch)).resize((w, h), Image.BILINEAR)


@torch.no_grad()
def get_prefill_state(vla, processor, image: np.ndarray, task_label: str, cfg: RolloutConfig):
    """
    vanilla OpenVLA 의 `.generate()` 가 시작되기 직전과 동일한 프리필(이미지+언어,
    action 토큰 없음) 1회 forward.

    Returns
    -------
    h0              : (1, 1, hidden_dim)  프롬프트 마지막 위치(빈 토큰)의 마지막 레이어 hidden state
    past_key_values : 프롬프트 KV 캐시
    attention_mask  : (1, full_len)       비전 patch 포함 실제 시퀀스 길이만큼 1
    """
    pil = Image.fromarray(image).convert("RGB")
    if cfg.center_crop:
        pil = _center_crop_pil(pil, 0.9)

    prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"
    inputs = processor(prompt, pil).to(cfg.device, dtype=torch.bfloat16)
    input_ids, attention_mask, pixel_values = inputs["input_ids"], inputs["attention_mask"], inputs["pixel_values"]

    if not torch.all(input_ids[:, -1] == EMPTY_TOKEN_ID):
        pad = torch.full((input_ids.shape[0], 1), EMPTY_TOKEN_ID, dtype=input_ids.dtype, device=input_ids.device)
        input_ids = torch.cat((input_ids, pad), dim=1)
        attention_mask = torch.cat(
            (attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)),
            dim=1,
        )

    out = vla(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )
    h0 = out.hidden_states[-1][:, -1:, :]                       # (1, 1, hidden_dim)
    full_len = out.hidden_states[-1].shape[1]                   # 비전 patch 삽입 후 실제 전체 길이
    full_mask = torch.ones((1, full_len), dtype=attention_mask.dtype, device=attention_mask.device)
    return h0, out.past_key_values, full_mask


def unnormalize_action(norm_action: np.ndarray, norm_stats: dict) -> np.ndarray:
    """
    OpenVLA `predict_action` 과 동일한 역정규화.
    norm_stats: {"q01": [7], "q99": [7], "mask": [7](optional)} = vla.norm_stats[unnorm_key]["action"]
    """
    a = np.clip(np.asarray(norm_action, dtype=np.float64), -1.0, 1.0)
    q01 = np.asarray(norm_stats["q01"], dtype=np.float64)
    q99 = np.asarray(norm_stats["q99"], dtype=np.float64)
    mask = np.asarray(norm_stats.get("mask", np.ones_like(q01, dtype=bool)))
    unnorm = 0.5 * (a + 1.0) * (q99 - q01) + q01
    return np.where(mask, unnorm, a)


def normalize_action(robot_action: np.ndarray, norm_stats: dict) -> np.ndarray:
    """
    unnormalize_action 의 역: robot-scale action -> [-1, 1] 정규화 action.
    BC 타깃(expert demo action) 을 우리 decoder 출력과 같은 공간에 놓기 위함.
    주의: gripper convention(부호 뒤집기)까지는 여기서 처리하지 않음 — BC warm-up 근사.
    """
    a = np.asarray(robot_action, dtype=np.float64)
    q01 = np.asarray(norm_stats["q01"], dtype=np.float64)
    q99 = np.asarray(norm_stats["q99"], dtype=np.float64)
    mask = np.asarray(norm_stats.get("mask", np.ones_like(q01, dtype=bool)))
    denom = np.where(np.abs(q99 - q01) < 1e-8, 1.0, q99 - q01)
    norm = 2.0 * (a - q01) / denom - 1.0
    return np.clip(np.where(mask, norm, a), -1.0, 1.0)


def _resolve_norm_stats(vla, unnorm_key: str) -> dict:
    ns = vla.norm_stats
    key = unnorm_key
    if key not in ns and f"{key}_no_noops" in ns:
        key = f"{key}_no_noops"
    if key not in ns:
        raise KeyError(f"unnorm_key {unnorm_key!r} not in vla.norm_stats (have: {list(ns)[:8]}...)")
    stats = ns[key]
    return stats["action"] if "action" in stats else stats


# ─────────────────────────────────────────────────────────────────────────────
class RolloutBuffer:
    """
    z-space PPO transition 저장소. backbone 이 얼려져 있으므로 h0 만 있으면 업데이트
    때 Actor/Critic 을 다시 돌릴 수 있다 (이미지 재인코딩 불필요 = 논문의 효율성 포인트).
    past_key_values / attention_mask 는 rollout 시 action 뽑을 때만 쓰고 저장하지 않는다.
    """

    _FIELDS = ("h0", "z", "log_prob", "value", "reward", "done")

    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self.clear()

    def clear(self):
        self.h0, self.z, self.log_prob, self.value = [], [], [], []
        self.reward, self.done = [], []
        self.adv = None
        self.ret = None

    def __len__(self):
        return len(self.reward)

    def add(self, h0, z, log_prob, value, reward: float, done: bool):
        # h0: (1,1,H) or (1,H) → (H,)  ;  z: (1,256) → (256,)
        self.h0.append(h0.detach().reshape(-1).to(self.device))
        self.z.append(z.detach().reshape(-1).to(self.device))
        self.log_prob.append(float(log_prob))
        self.value.append(float(value))
        self.reward.append(float(reward))
        self.done.append(bool(done))

    # 논문 Algorithm 2 line 9: Â_t ← GAE(γ, λ)
    def compute_gae(self, last_value: float, gamma: float, lam: float, normalize_adv: bool = True):
        n = len(self)
        rew = np.asarray(self.reward, dtype=np.float64)
        val = np.asarray(self.value + [float(last_value)], dtype=np.float64)   # V(s_0..s_n)
        done = np.asarray(self.done, dtype=np.float64)

        adv = np.zeros(n, dtype=np.float64)
        gae = 0.0
        for t in reversed(range(n)):
            nonterminal = 1.0 - done[t]
            delta = rew[t] + gamma * val[t + 1] * nonterminal - val[t]
            gae = delta + gamma * lam * nonterminal * gae
            adv[t] = gae
        ret = adv + val[:n]

        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=self.device)
        if normalize_adv and n > 1:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        self.adv = adv_t
        self.ret = torch.as_tensor(ret, dtype=torch.float32, device=self.device)

    def iter_minibatches(self, batch_size: int, shuffle: bool = True) -> Iterator[dict]:
        assert self.adv is not None, "compute_gae() 를 먼저 호출"
        n = len(self)
        idx = torch.randperm(n, device=self.device) if shuffle else torch.arange(n, device=self.device)
        h0 = torch.stack(self.h0).float()                 # (n, H)  — 업데이트는 float32
        z = torch.stack(self.z).float()                   # (n, 256)
        old_logp = torch.as_tensor(self.log_prob, dtype=torch.float32, device=self.device)
        for s in range(0, n, batch_size):
            b = idx[s : s + batch_size]
            yield {
                "h0": h0[b], "z": z[b],
                "old_log_prob": old_logp[b],
                "adv": self.adv[b], "ret": self.ret[b],
            }


# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def collect_episode(
    env,
    initial_state,
    task_description: str,
    *,
    vla,
    processor,
    actor_critic,
    projection,
    buffer: "RolloutBuffer | None",
    cfg: RolloutConfig,
    reward_fn: Callable[[bool, int], float],
    deterministic: bool = False,
    get_image: Callable[[dict, int], np.ndarray] | None = None,
    dummy_action: Callable[[], list] | None = None,
    frames: list | None = None,
) -> dict:
    """
    한 에피소드 굴리며 transition 을 buffer 에 추가. 성공/스텝수 등 통계 dict 반환.

    get_image(raw_obs, resize_size) -> HxWx3 uint8   (기본: experiments.robot.libero.libero_utils.get_libero_image)
    dummy_action() -> 7-list                          (기본: get_libero_dummy_action("openvla"))
    frames: 주면 매 policy step 의 입력 이미지(HxWx3 uint8)를 append (rollout 영상용)
    """
    if get_image is None:
        from experiments.robot.libero.libero_utils import get_libero_image
        get_image = lambda o, s: get_libero_image(o, s)
    if dummy_action is None:
        from experiments.robot.libero.libero_utils import get_libero_dummy_action
        dummy_action = lambda: get_libero_dummy_action("openvla")

    env.reset()
    obs = env.set_init_state(initial_state)

    norm_stats = _resolve_norm_stats(vla, cfg.unnorm_key)
    success, steps_used = False, 0

    t = 0
    total_limit = cfg.max_steps + cfg.num_steps_wait
    while t < total_limit:
        if t < cfg.num_steps_wait:
            obs, _, done, _ = env.step(dummy_action())
            t += 1
            continue

        img = get_image(obs, cfg.resize_size)
        if frames is not None:
            frames.append(img)
        h0, pkv, mask = get_prefill_state(vla, processor, img, task_description, cfg)

        h0_f = h0.float()
        z, log_prob, value = actor_critic.act(h0_f, deterministic=deterministic)   # z-space
        a_norm = projection.z_to_action(h0, z, pkv, mask).squeeze(0).cpu().numpy()  # (7,) ∈ [-1,1]

        a_robot = unnormalize_action(a_norm, norm_stats)
        if cfg.gripper_postprocess and normalize_gripper_action is not None:
            a_robot = normalize_gripper_action(a_robot, binarize=True)
            a_robot = invert_gripper_action(a_robot)

        obs, _, done, _ = env.step(a_robot.tolist())
        steps_used += 1
        is_last = bool(done) or (t + 1 >= total_limit)   # timeout 도 에피소드 경계로 취급
        r = reward_fn(bool(done), steps_used)
        if buffer is not None:                            # eval 시엔 None
            # GAE 가 에피소드 경계에서 리셋되도록 timeout 도 done=True 로 저장.
            # (reward 는 그대로 구분: 성공 -> +r_success, timeout -> -λ·t)
            buffer.add(h0, z, log_prob, value, r, is_last)

        if done:
            success = True
            break
        t += 1

    return {"success": success, "steps": steps_used, "task": task_description}


@torch.no_grad()
def collect_rollouts(
    make_env: Callable[[object], tuple],      # task -> (env, task_description)
    tasks: Iterable,                          # round-robin 대상 (LIBERO task 객체들)
    init_states_of: Callable[[object], np.ndarray],
    *,
    vla,
    processor,
    actor_critic,
    projection,
    buffer: RolloutBuffer,
    cfg: RolloutConfig,
    reward_fn: Callable[[bool, int], float],
    target_transitions: int,                  # 논문 Table 1 Buffer Size (예: 512)
    rollouts_per_task: int = 2,               # 한 task 방문 시 수집할 에피소드 수
    verbose: bool = True,                     # 에피소드마다 진행 출력
    task_cursor: int = 0,                     # round-robin 커서 (PPO iteration 간 유지 → 편향 방지)
    init_cursor_of: Callable[[object], int] | None = None,  # task -> 이미 쓴 init state 수
) -> dict:
    """
    buffer 가 target_transitions 이상 찰 때까지 task 를 round-robin 하며 에피소드 수집.

    편향 방지: `task_cursor` 를 호출 간(=PPO iteration 간) 유지하면, 매 iteration 이
    tasks[0] 부터 다시 시작하지 않고 이어서 순회한다. rollouts_per_task 를 작게(1~2) 두면
    한 iteration 이 여러 task 를 훑고, 장기적으로 모든 task 가 균등하게 방문된다.
    반환 dict 의 `task_cursor` 를 다음 호출에 넘길 것.
    """
    tasks = list(tasks)
    n_tasks = len(tasks)
    ep_stats, per_task_ep = [], {}
    ti = task_cursor
    used_init = {}  # id(task) -> 다음에 쓸 init index

    while len(buffer) < target_transitions:
        task = tasks[ti % n_tasks]
        ti += 1
        env, desc = make_env(task)
        inits = init_states_of(task)
        n0 = used_init.get(id(task))
        if n0 is None:
            n0 = init_cursor_of(task) if init_cursor_of is not None else 0
        for k in range(rollouts_per_task):
            init = inits[(n0 + k) % len(inits)]
            st = collect_episode(
                env, init, desc,
                vla=vla, processor=processor, actor_critic=actor_critic,
                projection=projection, buffer=buffer, cfg=cfg, reward_fn=reward_fn,
            )
            ep_stats.append(st)
            per_task_ep[desc] = per_task_ep.get(desc, [0, 0])
            per_task_ep[desc][0] += 1
            per_task_ep[desc][1] += int(st["success"])
            if verbose:
                print(f"[OLIVE][rollout]   ep{len(ep_stats)} t{(ti-1) % n_tasks}  "
                      f"{'OK ' if st['success'] else 'fail'} {st['steps']:3d} steps  "
                      f"buffer {len(buffer)}/{target_transitions}  | {desc[:40]}", flush=True)
            if len(buffer) >= target_transitions:
                break
        used_init[id(task)] = n0 + rollouts_per_task
        try:
            env.close()
        except Exception:
            pass

    succ = np.mean([s["success"] for s in ep_stats]) if ep_stats else 0.0
    return {
        "episodes": len(ep_stats),
        "transitions": len(buffer),
        "success_rate": float(succ),
        "n_tasks_visited": len({s["task"] for s in ep_stats}),
        "mean_steps": float(np.mean([s["steps"] for s in ep_stats])) if ep_stats else 0.0,
        "task_cursor": ti,                    # 다음 호출에 넘길 것
        "per_task": {k: {"n": v[0], "succ": v[1]} for k, v in per_task_ep.items()},
        "ep_stats": ep_stats,
    }


if __name__ == "__main__":
    # backbone/env 없이: unnormalize + RolloutBuffer + GAE 만 스모크.
    ns = {"q01": [-1, -1, -1, -1, -1, -1, 0.0], "q99": [1, 1, 1, 1, 1, 1, 1.0],
          "mask": [True] * 6 + [False]}
    a = unnormalize_action(np.array([0.0, 1.0, -1.0, 0.5, -0.5, 0.25, 0.9]), ns)
    assert a.shape == (7,) and abs(a[0] - 0.0) < 1e-9 and abs(a[1] - 1.0) < 1e-9 and a[6] == 0.9  # masked dim 그대로
    print("unnormalize_action  ✓ ", np.round(a, 3))

    buf = RolloutBuffer(device="cpu")
    torch.manual_seed(0)
    for i in range(10):
        buf.add(torch.randn(1, 1, 8), torch.randn(1, 4), log_prob=-1.0 + 0.01 * i,
                value=0.5, reward=(1.0 if i == 9 else -0.01 * (i + 1)), done=(i == 9))
    assert len(buf) == 10
    buf.compute_gae(last_value=0.0, gamma=0.99, lam=0.95, normalize_adv=True)
    assert buf.adv.shape == (10,) and buf.ret.shape == (10,)
    assert abs(float(buf.adv.mean())) < 1e-5 and abs(float(buf.adv.std()) - 1.0) < 1e-1  # normalize 확인
    nb = sum(1 for _ in buf.iter_minibatches(batch_size=4))
    assert nb == 3  # 10 -> 4+4+2
    b0 = next(buf.iter_minibatches(batch_size=4))
    assert b0["h0"].shape[1] == 8 and b0["z"].shape[1] == 4
    print(f"RolloutBuffer + GAE  ✓  (adv mean~{float(buf.adv.mean()):.2e} std~{float(buf.adv.std()):.3f}, minibatches={nb})")
