"""
OLIVE: LIBERO task / env glue.

LIBERO import 를 이 파일 하나로 격리한다. 이미지 전처리는 openvla `get_libero_image`
와 동일한 로직을 vendor 한다 (openvla 소스 미수정 규칙 + ablation 시 전처리 조작 여지).
순수 시뮬 구성(`OffScreenRenderEnv`, benchmark) 은 LIBERO 에서만 가능하므로 import.

필요 환경변수 (호출 전 설정):
    PYTHONPATH        += $HOME/LIBERO/LIBERO
    LIBERO_CONFIG_PATH = $HOME/.libero_base   (또는 -plus / -pro)
    MUJOCO_GL          = egl
"""

from __future__ import annotations

import numpy as np

from config import TASK_MAX_STEPS  # noqa: F401  (재export 편의)

# vanilla OpenVLA dummy action (7-dim, gripper open). openvla get_libero_dummy_action 과 동일.
DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


# ── task suite / env ────────────────────────────────────────────────────────
def make_task_suite(suite_name: str):
    """benchmark dict 에서 suite 인스턴스 생성. suite.n_tasks, suite.get_task(i) 등."""
    from libero.libero import benchmark
    return benchmark.get_benchmark_dict()[suite_name]()


def get_task_init_states(task_suite, task_id: int) -> np.ndarray:
    """(num_init, state_dim) — torch.load(*.pruned_init). TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 필요."""
    return task_suite.get_task_init_states(task_id)


def make_env(task, resolution: int = 256):
    """
    LIBERO OffScreenRenderEnv 생성.
    Returns: (env, task_description)
      env: .reset(), .set_init_state(state)->obs, .step(action_list)->(obs, r, done, info), .close()
    """
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    import os

    task_bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    return env, task.language


# ── 이미지 전처리 ──────────────────────────────────────────────────────────
# 기본은 openvla `get_libero_image` 를 그대로 써서 baseline 과 픽셀 일치를 보장한다
# (import 만; openvla 소스 미수정). 없거나 ablation 으로 전처리를 바꿀 땐 아래 vendored
# 로직으로 폴백.
try:
    from experiments.robot.libero.libero_utils import get_libero_image as _ov_get_libero_image
except Exception:  # noqa: BLE001
    _ov_get_libero_image = None


def _vendored_preprocess(raw_hwc: np.ndarray, resize_size: int) -> np.ndarray:
    import tensorflow as tf
    img = np.ascontiguousarray(raw_hwc[::-1, ::-1])   # 180° (음수 stride 제거)
    img = tf.image.encode_jpeg(img)
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)
    img = tf.image.resize(img, (resize_size, resize_size), method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    return img.numpy()


def preprocess_frame(raw_hwc: np.ndarray, resize_size: int) -> np.ndarray:
    """raw agentview 프레임(HxWx3 uint8) -> (resize_size, resize_size, 3) uint8. (180° 회전 포함)"""
    if _ov_get_libero_image is not None:
        return _ov_get_libero_image({"agentview_image": np.asarray(raw_hwc)}, resize_size)
    return _vendored_preprocess(raw_hwc, resize_size)


def get_image(obs: dict, resize_size: int) -> np.ndarray:
    """sim obs dict -> 전처리된 프레임."""
    if _ov_get_libero_image is not None:
        return _ov_get_libero_image(obs, resize_size)
    return _vendored_preprocess(obs["agentview_image"], resize_size)


def get_dummy_action() -> list:
    return list(DUMMY_ACTION)


if __name__ == "__main__":
    # LIBERO import 는 env 세팅이 필요. 여기선 상수/시그니처만 확인.
    assert len(DUMMY_ACTION) == 7 and DUMMY_ACTION[-1] == -1.0
    assert TASK_MAX_STEPS["libero_10"] == 520
    print("libero_env.py OK (make_task_suite/make_env 는 LIBERO env 세팅 시 동작)")
