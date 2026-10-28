"""
OLIVE: 모든 설정을 한곳에.

원칙
----
- ablation 축은 전부 여기 타입 필드로 존재한다. 다른 모듈은 이 값을 "받아서" 동작하고
  상수를 하드코딩하지 않는다.
- dataclass 중첩: OliveConfig ⊃ {NetworkConfig, ProjectionConfig, BCConfig,
  RolloutConfig, PPOConfig}.
- CLI: `build_config()` 가 argparse 로 `--net.z_dim 128` 같은 점표기 override 를 받는다.
- 프리셋: `ABLATIONS` dict 에 이름 -> override dict. `--ablation random_basis` 처럼 사용.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import os
from dataclasses import dataclass, field
from typing import Any, Optional

# suite 이름 -> 에피소드 최대 스텝 (openvla run_libero_eval.py 와 동일)
TASK_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


@dataclass
class NetworkConfig:
    hidden_dim: int = 4096            # Llama-2 hidden dim (백본 고정)
    z_dim: int = 256                 # [ablation] latent code 차원
    actor_mlp_hidden: int = 256
    critic_mlp_hidden: int = 256
    actor_output_gain: float = 0.01  # [ablation] 1.0 = warm-start 없음
    log_std_init: float = -2.0
    log_std_min: float = -5.0
    log_std_max: float = 0.0


@dataclass
class ProjectionConfig:
    alpha: float = 1.0               # [ablation] residual scale (Eq.7)
    inject_basis: str = "lm_head"    # [ablation] lm_head | random | unconstrained
    random_basis_seed: int = 0
    num_bins: int = 256              # action 이산화 bin 수 (= z_dim 이 아니어도 됨, 보통 256)
    critic_input: str = "pre"        # [ablation] pre(h0) | post(h0')
    action_dim: int = 7


@dataclass
class BCConfig:
    enabled: bool = True             # [ablation] BC warm-up on/off  (이번 보강의 핵심)
    targets: str = "both"            # [ablation] both | actor | critic
    # 표준 LIBERO demo (HDF5). modified_libero_rlds 쓰려면 그 경로로.
    demo_glob: str = "~/LIBERO/LIBERO-plus/libero/datasets/{suite}/*.hdf5"
    max_demos: Optional[int] = None
    frame_stride: int = 5            # demo frame 서브샘플 (비용/시간 조절)
    epochs: int = 3
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 0.0
    action_loss: str = "mse"         # mse | huber
    gamma: float = 0.99             # Critic return-to-go 계산용
    grad_clip: float = 1.0
    val_frac: float = 0.05           # demo 중 검증용 비율
    num_workers: int = 4
    log_every: int = 20              # N batch 마다 stdout + wandb 진행 로그
    save_every_epoch: bool = True    # epoch 마다 after_bc_epN.pt 저장


@dataclass
class RolloutConfig:
    center_crop: bool = True
    resize_size: int = 224
    num_steps_wait: int = 10
    gripper_postprocess: bool = True
    # 한 task 방문 시 수집 에피소드 수. 작게(1~2) + round-robin 커서 유지(train.py)로
    # 한 iteration 이 여러 task 를 훑고 장기적으로 균등 방문 → task 편향 방지.
    rollouts_per_task: int = 2
    # max_steps 는 suite 로부터 자동 (TASK_MAX_STEPS)
    max_steps: int = 0               # 0 이면 build 시 suite 기준으로 채움


@dataclass
class RewardConfig:
    kind: str = "paper"              # [ablation] paper(Eq.12) | ...
    r_success: float = 1.0
    lam: float = 0.01               # step penalty 계수 (논문 미명시)


@dataclass
class PPOConfig:
    total_steps: int = 50_000        # Table 1. cf. Algorithm 2 본문 5000 — 설정값
    buffer_size: int = 512
    ppo_epochs: int = 4
    batch_size: int = 64
    # 첫 PPO 검증(2026-09-02)에서 KL 폭발(24~1340) 확인 후 z-space 에 맞게 조정한 기본값.
    # 논문(action-space 7d)값: lr_start 3e-4, entropy_coef 0.01(per-dim 아님), target_kl 없음.
    lr_start: float = 1e-4
    lr_end: float = 1e-5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01       # per_dim=True 면 실효값 = 0.01/z_dim ≈ 3.9e-5
    entropy_per_dim: bool = True     # [ablation] z-space(256d) 라 per-dim 정규화가 기본
    max_grad_norm: float = 1.0
    normalize_adv: bool = True
    clip_value_loss: bool = False
    target_kl: Optional[float] = 0.05   # KL 이 1.5*target 넘으면 그 iteration 의 남은 epoch 중단
    normalize_value_loss: bool = True   # [ablation] value loss / running Var(ret)  (2026-10-29 추가)
    separate_grad_clip: bool = True     # [ablation] actor/critic grad 개별 클립   (2026-10-29 추가)


@dataclass
class OliveConfig:
    # 실행
    suite: str = "libero_spatial"
    pretrained_checkpoint: str = ""  # olive-model.sh 로 받은 로컬 경로 (필수)
    seed: int = 7
    device: str = "cuda:0"
    attn_impl: str = "flash_attention_2"   # flash_attention_2 | sdpa | eager

    # 단계 분리: bc(추가학습만) | ppo(RL 만) | all(둘 다, 기본)
    phase: str = "all"
    init_ckpt: str = ""              # PPO 시작 전 head 가중치를 불러올 .pt (보통 after_bc.pt)

    # 로깅/출력
    run_name: str = "olive"
    result_root: str = "~/olive/result"
    wandb: bool = True
    wandb_project: str = "olive-libero"
    wandb_entity: str = "gibeom-real-1526-kyungpook-national-university"

    # 평가/체크포인트 주기 (PPO iteration 단위)
    eval_every: int = 10
    eval_episodes_per_task: int = 20
    ckpt_every: int = 10

    # 중첩
    net: NetworkConfig = field(default_factory=NetworkConfig)
    proj: ProjectionConfig = field(default_factory=ProjectionConfig)
    bc: BCConfig = field(default_factory=BCConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)

    # ── 파생 경로 ──
    @property
    def run_dir(self) -> str:
        return os.path.expanduser(os.path.join(self.result_root, self.run_name))

    def resolved_demo_glob(self) -> str:
        return os.path.expanduser(self.bc.demo_glob.format(suite=self.suite))

    def finalize(self) -> "OliveConfig":
        """build 마지막에 호출: suite 의존 기본값 채우기 + 검증."""
        if self.rollout.max_steps == 0:
            self.rollout.max_steps = TASK_MAX_STEPS[self.suite]
        assert self.suite in TASK_MAX_STEPS, f"unknown suite {self.suite}"
        assert self.proj.inject_basis in ("lm_head", "random", "unconstrained")
        assert self.proj.critic_input in ("pre", "post")
        assert self.bc.targets in ("both", "actor", "critic")
        # lm_head / random basis 는 z 를 B(=num_bins x H) 에 곱하므로 z_dim == num_bins 강제.
        if self.proj.inject_basis in ("lm_head", "random"):
            assert self.net.z_dim == self.proj.num_bins, (
                f"inject_basis={self.proj.inject_basis} 는 net.z_dim({self.net.z_dim}) "
                f"== proj.num_bins({self.proj.num_bins}) 필요. unconstrained 면 자유."
            )
        return self


# ─────────────────────────────────────────────────────────────────────────────
# ablation 프리셋: 이름 -> {점표기 경로: 값}
ABLATIONS: dict[str, dict[str, Any]] = {
    "main":          {},                                            # 논문 기본 + BC warm-up
    "no_bc":         {"bc.enabled": False},                         # BC 없이 PPO 만
    "bc_actor_only": {"bc.targets": "actor"},
    "bc_critic_only":{"bc.targets": "critic"},
    "random_basis":  {"proj.inject_basis": "random"},               # 논문 §4.4 ablation
    "unconstrained": {"proj.inject_basis": "unconstrained"},        # full-space Residual RL
    "no_warmstart":  {"net.actor_output_gain": 1.0},
    "critic_post":   {"proj.critic_input": "post"},
    # z_dim 을 바꾸면 z 를 256-row action basis 에 곱할 수 없으므로 unconstrained 로.
    "z128":          {"net.z_dim": 128, "proj.inject_basis": "unconstrained"},
    "z512":          {"net.z_dim": 512, "proj.inject_basis": "unconstrained"},
    "ent_full":      {"ppo.entropy_per_dim": False},   # 논문식 (per-dim 정규화 끔) — 비교용
    "paper_ppo":     {"ppo.lr_start": 3e-4, "ppo.entropy_per_dim": False, "ppo.target_kl": None},
    "alpha0.5":      {"proj.alpha": 0.5},
    # 2026-09-02 동작 재현용 (value loss 정규화 / grad 개별 클립 끔)
    "no_vnorm":      {"ppo.normalize_value_loss": False, "ppo.separate_grad_clip": False},
}


def _set_by_path(cfg: OliveConfig, dotted: str, value: Any) -> None:
    obj = cfg
    parts = dotted.split(".")
    for p in parts[:-1]:
        obj = getattr(obj, p)
    leaf = parts[-1]
    cur = getattr(obj, leaf)
    if value is None:                       # ABLATIONS dict 에서 명시적 None
        setattr(obj, leaf, None)
        return
    # 타입 캐스팅 (argparse 는 문자열로 들어옴)
    s = str(value).strip()
    if isinstance(cur, bool):
        value = s.lower() in ("1", "true", "yes", "on")
    elif isinstance(cur, int) and not isinstance(cur, bool):
        value = int(s)
    elif isinstance(cur, float):
        value = float(s)
    elif cur is None:
        # Optional 필드: none/null -> None, 정수꼴 -> int, 실수꼴 -> float, 그 외 -> str 그대로
        if s.lower() in ("none", "null", ""):
            value = None
        elif s.lstrip("-").isdigit():
            value = int(s)
        else:
            try:
                value = float(s)
            except ValueError:
                value = s
    setattr(obj, leaf, value)


def _flat_keys(cfg: OliveConfig, prefix: str = "") -> list[str]:
    keys = []
    for f in dataclasses.fields(cfg):
        v = getattr(cfg, f.name)
        if dataclasses.is_dataclass(v):
            keys += _flat_keys(v, f"{prefix}{f.name}.")
        else:
            keys.append(f"{prefix}{f.name}")
    return keys


def build_config(argv: Optional[list[str]] = None) -> OliveConfig:
    """
    CLI:
        --suite libero_spatial --pretrained_checkpoint <path>
        --ablation random_basis
        --set net.z_dim=128 --set ppo.entropy_coef=0.001   (반복 가능, 프리셋 뒤에 적용)
        --no-bc   (= --set bc.enabled=false 축약)
    """
    p = argparse.ArgumentParser("olive")
    p.add_argument("--suite", default=None)
    p.add_argument("--pretrained_checkpoint", "--ckpt", default=None)
    p.add_argument("--run_name", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--ablation", default="main", choices=list(ABLATIONS))
    p.add_argument("--set", action="append", default=[], metavar="path=value",
                   help="점표기 override, 반복 가능")
    p.add_argument("--no-bc", action="store_true")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--phase", choices=["bc", "ppo", "all"], default=None)
    p.add_argument("--init-ckpt", "--init_ckpt", default=None)
    args = p.parse_args(argv)

    cfg = OliveConfig()
    # 1) ablation 프리셋
    for path, val in ABLATIONS[args.ablation].items():
        _set_by_path(cfg, path, val)
    # 2) 개별 --set override
    for kv in args.set:
        assert "=" in kv, f"--set 형식은 path=value: {kv}"
        path, val = kv.split("=", 1)
        _set_by_path(cfg, path, val)
    # 3) 축약 플래그 / 최상위 인자
    if args.no_bc:
        cfg.bc.enabled = False
    if args.no_wandb:
        cfg.wandb = False
    if args.phase:
        cfg.phase = args.phase
    if args.init_ckpt:
        cfg.init_ckpt = args.init_ckpt
    if args.suite:
        cfg.suite = args.suite
    if args.pretrained_checkpoint:
        cfg.pretrained_checkpoint = args.pretrained_checkpoint
    if args.run_name:
        cfg.run_name = args.run_name
    if args.seed is not None:
        cfg.seed = args.seed
    if cfg.run_name == "olive":
        cfg.run_name = f"olive-{cfg.suite}-{args.ablation}"

    return cfg.finalize()


if __name__ == "__main__":
    c = build_config(["--suite", "libero_10", "--ablation", "z128",
                      "--set", "ppo.entropy_coef=0.0005", "--no-bc"])
    assert c.suite == "libero_10" and c.rollout.max_steps == 520
    assert c.proj.inject_basis == "unconstrained" and c.net.z_dim == 128   # z128 프리셋
    assert abs(c.ppo.entropy_coef - 0.0005) < 1e-12 and c.bc.enabled is False
    assert c.run_name == "olive-libero_10-z128"
    # lm_head basis + z_dim!=256 은 거부되어야
    try:
        build_config(["--ablation", "random_basis", "--set", "net.z_dim=128"])
        raise RuntimeError("should have failed")
    except AssertionError:
        pass
    print("config.py smoke OK")
    print("  run_dir      :", c.run_dir)
    print("  demo_glob    :", c.resolved_demo_glob())
    print("  ablations    :", ", ".join(ABLATIONS))
    print("  #flat fields :", len(_flat_keys(c)))
