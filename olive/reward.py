"""
OLIVE: reward (논문 §3.5, Eq. 12).

    r_t = { r_success            if task completed (LIBERO sim done)
          { -lambda * t          otherwise                              (step penalty)

t 는 "wait 이후" 경과 스텝 수(1부터). λ·t 형태라 타임아웃까지 끌수록 누적 페널티가
빠르게 커져 timeout 행동을 억제한다 (논문 4.4절: 실패 에피소드는 500 step 경계에서
큰 페널티 누적).

실로봇 배포 시엔 done 을 proprioception 기반 근접 보상으로 대체 가능 (논문 §3.5).
"""

from __future__ import annotations

R_SUCCESS = 1.0
LAMBDA_STEP = 0.01     # 논문에 수치가 명시돼 있지 않음 — 기본값. 실험에서 조정 가능.


def compute_reward_paper(done: bool, t: int, r_success: float = R_SUCCESS, lam: float = LAMBDA_STEP) -> float:
    """Eq. 12 그대로. done 이면 r_success, 아니면 -lam * t."""
    if done:
        return float(r_success)
    return float(-lam * t)


def make_reward_fn(r_success: float = R_SUCCESS, lam: float = LAMBDA_STEP):
    """collect_episode(reward_fn=...) 에 넘길 클로저."""
    def _fn(done: bool, t: int) -> float:
        return compute_reward_paper(done, t, r_success, lam)
    return _fn


if __name__ == "__main__":
    f = make_reward_fn()
    assert f(True, 37) == 1.0
    assert abs(f(False, 100) - (-1.0)) < 1e-9
    assert abs(f(False, 1) - (-0.01)) < 1e-9
    print("reward.py smoke OK  | success=+1.0, step t penalty=-0.01*t")
