"""
OLIVE: z-space projection + action basis projection.

network.py 의 Actor 가 내놓는 latent code  z ∈ R^{z_dim}(=256) 를 frozen backbone 의
hidden-state 공간으로 밀어 넣고, frozen LM head 로 다시 연속 action 으로 디코딩하는
모듈. backbone(7B) 은 여기서 "고정 상수"로만 쓴다 — 학습 파라미터 없음.

논문 수식 대응
--------------
    B   = W[vocab-256 : vocab, :]        (Eq. 4)  frozen LM head 의 action-token 행
    Δh  = z · B_inject                    (Eq. 6)
    h'  = h + α · Δh                      (Eq. 7)
    a'  = LM_Head(h')  의 action 서브슬라이스로 7-step 디코딩   (Eq. 8)

핵심 구현 주의점
----------------
1. B 슬라이스는 반드시 `tokenizer.vocab_size` 기준. `W[-256:, :]` 는 틀리다 —
   LLaMA 계열은 임베딩 행렬을 GPU 정렬을 위해 실제 vocab 보다 크게 패딩하는 일이
   흔하다(예: 32000 -> 32064). 그러면 마지막 256행은 진짜 action token 이 아니라
   일부가 학습된 적도 없는 패딩 행이 되어 Eq.4/8 이 깨진다. OpenVLA `ActionTokenizer`
   가 action token 을 `[vocab - n_bins, vocab)` 에 배치하므로 그 구간을 잘라야 한다.
2. 입력/출력 임베딩은 LLaMA-2 에서 tie 되어 있지 않다. 디코딩(Eq.8, 출력)에는
   lm_head.weight 를, 2~7번째 토큰의 다음-스텝 입력에는 별도의 input embedding 행을
   쓴다.
3. B 의 행 j (0..255) <-> 연속 action 값 매핑은 `ActionTokenizer.
   decode_token_ids_to_actions` 와 정확히 일치해야 한다 (compute_action_bin_values).

정책을 z-space Gaussian 으로 두는 설계(network.py 참고)에서는 디코딩 경로로 gradient 를
흘릴 필요가 없으므로 `decode_action()` 은 greedy(argmax) + no_grad 로 간단하게 둔다.
(action-space 정책으로 바꿔 soft-decode 가 필요해지면 그때 미분 가능 버전을 추가.)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_ACTION_BINS = 256   # = z_dim, = B 행 수
ALPHA = 1.0             # residual scale (Eq. 7)
MIN_ACTION, MAX_ACTION = -1.0, 1.0
ACTION_DIM = 7


# ── frozen backbone 에서 고정 텐서 추출 ────────────────────────────────────
def extract_action_basis(vla, tokenizer_vocab_size: int, num_bins: int = NUM_ACTION_BINS) -> torch.Tensor:
    """Eq. 4:  B = lm_head.weight[vocab - num_bins : vocab, :]   ->  (num_bins, hidden_dim)."""
    lm_head = vla.language_model.get_output_embeddings()      # nn.Linear(hidden, padded_vocab, bias=False)
    W = lm_head.weight.detach()                               # (padded_vocab, hidden)
    lo, hi = tokenizer_vocab_size - num_bins, tokenizer_vocab_size
    assert 0 <= lo < hi <= W.shape[0], f"bad slice [{lo}:{hi}] for lm_head {tuple(W.shape)}"
    return W[lo:hi, :].clone()


def extract_input_embedding_rows(vla, tokenizer_vocab_size: int, num_bins: int = NUM_ACTION_BINS) -> torch.Tensor:
    """2~7번째 토큰의 다음-스텝 입력에 쓰는 input embedding 의 action-token 행 (출력 B 와 별개 행렬)."""
    emb = vla.language_model.get_input_embeddings()           # nn.Embedding(padded_vocab, hidden)
    W = emb.weight.detach()
    lo, hi = tokenizer_vocab_size - num_bins, tokenizer_vocab_size
    return W[lo:hi, :].clone()


def compute_action_bin_values(num_bins: int = NUM_ACTION_BINS) -> torch.Tensor:
    """
    B 의 행 인덱스 j (0..num_bins-1)  ->  연속 action 값  lookup table  (num_bins,).

    OpenVLA `ActionTokenizer` 규약을 그대로 따른다:
        bins         = linspace(MIN, MAX, num_bins)               # num_bins 개
        bin_centers  = (bins[:-1] + bins[1:]) / 2                 # num_bins-1 개
        token_id j 행 -> discretized = num_bins - j
                      -> idx = clip(discretized - 1, 0, num_bins-2)
                      -> bin_centers[idx]
    즉 idx = clip(num_bins - 1 - j, 0, num_bins - 2).
    """
    bins = torch.linspace(MIN_ACTION, MAX_ACTION, num_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0                    # (num_bins - 1,)
    j = torch.arange(num_bins)
    idx = (num_bins - 1 - j).clamp(0, bin_centers.numel() - 1)
    return bin_centers[idx]                                       # (num_bins,)


def _random_basis_like(B: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Ablation 용: shape 은 같되 방향은 임의인 고정 basis. 행별 norm 을 B 평균 norm 에 맞춤."""
    g = torch.Generator().manual_seed(seed)
    R = torch.randn(*B.shape, generator=g).to(B.device)
    R = R / R.norm(dim=-1, keepdim=True)
    return (R * B.float().norm(dim=-1, keepdim=True).mean()).to(B.dtype)


# ── 모듈 ───────────────────────────────────────────────────────────────────
class ActionBasisProjection(nn.Module):
    """
    z  ->  Δh  ->  h'  ->  (7-step) 연속 action.

    학습 파라미터 없음. buffer 만 보유:
        B                : (256, H)  decode 용 (Eq.8) — 항상 진짜 lm_head
        B_inject         : (256, H)  주입 용  (Eq.6) — "lm_head"(기본) 또는 "random"(ablation)
        bin_values       : (256,)    행 j -> 연속 action 값
        input_embed_rows : (256, H)  2~7번째 토큰 입력 임베딩
    """

    def __init__(
        self,
        vla,
        tokenizer_vocab_size: int,
        alpha: float = ALPHA,
        inject_basis: str = "lm_head",      # "lm_head" | "random" | "unconstrained"
        num_bins: int = NUM_ACTION_BINS,
        action_dim: int = ACTION_DIM,
        z_dim: int | None = None,           # unconstrained 일 때 입력 z 차원 (기본 = num_bins)
        random_basis_seed: int = 0,
    ):
        super().__init__()
        # 7B backbone 은 서브모듈로 등록하지 않는다 (state_dict/.to 가 7B 를 끌고 다니지 않도록).
        object.__setattr__(self, "_vla", vla)

        self.alpha = float(alpha)
        self.num_bins = num_bins
        self.action_dim = action_dim
        self.inject_basis = inject_basis
        z_dim = num_bins if z_dim is None else z_dim

        B = extract_action_basis(vla, tokenizer_vocab_size, num_bins)   # decode 용 — 항상 진짜 lm_head
        dtype = B.dtype
        self.register_buffer("B", B, persistent=False)

        # ── 주입 경로 (Eq.6) : ablation 축 ──
        #   lm_head       : Δh = z @ B                (논문 기본, 학습 파라미터 0)
        #   random        : Δh = z @ B_random         (고정 임의 basis, 학습 0)
        #   unconstrained : Δh = Linear(z)            (학습되는 4096-wide projection, full-space Residual RL)
        self._inject_linear = None
        if inject_basis == "lm_head":
            self.register_buffer("B_inject", B.clone(), persistent=False)
            assert z_dim == num_bins, f"lm_head basis 는 z_dim({z_dim}) == num_bins({num_bins}) 필요"
        elif inject_basis == "random":
            self.register_buffer("B_inject", _random_basis_like(B, random_basis_seed), persistent=False)
            assert z_dim == num_bins, f"random basis 는 z_dim({z_dim}) == num_bins({num_bins}) 필요"
        elif inject_basis == "unconstrained":
            self.register_buffer("B_inject", torch.empty(0), persistent=False)  # 미사용 placeholder
            self._inject_linear = nn.Linear(z_dim, B.shape[1], bias=False).to(dtype)
            nn.init.zeros_(self._inject_linear.weight)   # warm start: Δh = 0
        else:
            raise ValueError(f"inject_basis must be lm_head|random|unconstrained, got {inject_basis!r}")
        # bin_values 는 float32 로 둔다 — bf16 텐서는 .numpy() 가 안 되고(action 을 env 로
        # 넘길 때 필요), 작은 action 값이라 precision 도 중요.
        self.register_buffer("bin_values", compute_action_bin_values(num_bins).float(), persistent=False)
        self.register_buffer(
            "input_embed_rows",
            extract_input_embedding_rows(vla, tokenizer_vocab_size, num_bins).to(dtype),
            persistent=False,
        )
        self.hidden_dim = B.shape[1]
        print(f"[OLIVE] ActionBasisProjection: B{tuple(B.shape)} decode(lm_head) / "
              f"inject={inject_basis}"
              f"{' (trainable Linear)' if self._inject_linear is not None else ''}, "
              f"alpha={self.alpha}, bins={num_bins}, z_dim={z_dim}")

    @property
    def dtype(self) -> torch.dtype:
        return self.B.dtype

    def trainable_parameters(self):
        """inject_basis='unconstrained' 일 때만 비어있지 않다 (train.py 가 optimizer 에 추가)."""
        return list(self._inject_linear.parameters()) if self._inject_linear is not None else []

    # ── Eq. 6 / 7 : z-space projection ───────────────────────────────────
    def delta_h(self, z: torch.Tensor) -> torch.Tensor:
        """Δh = z · B_inject  또는  Linear(z).   z: (..., z_dim) -> (..., H)."""
        if self._inject_linear is not None:
            return self._inject_linear(z.to(self._inject_linear.weight.dtype))
        return z.to(self.B_inject.dtype) @ self.B_inject

    def apply_residual(self, h: torch.Tensor, z: torch.Tensor, alpha: float | None = None) -> torch.Tensor:
        """h' = h + α · Δh(z).   h,(out): (..., H)."""
        a = self.alpha if alpha is None else float(alpha)
        return h.to(self.B.dtype) + a * self.delta_h(z)

    # ── Eq. 8 primitive : 한 토큰 위치 디코딩 ────────────────────────────
    def decode_logits(self, h: torch.Tensor) -> torch.Tensor:
        """LM_Head(h) 의 action 서브슬라이스.   h: (..., H) -> (..., 256)."""
        return h.to(self.B.dtype) @ self.B.T

    def decode_probs(self, h: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.decode_logits(h).float(), dim=-1)

    def decode_expected(self, h: torch.Tensor) -> torch.Tensor:
        """softmax 기댓값(soft-argmax) 연속 action.   (..., ) — 미분 가능."""
        return (self.decode_probs(h) * self.bin_values.float()).sum(dim=-1)

    def decode_argmax(self, h: torch.Tensor) -> torch.Tensor:
        """hard argmax 연속 action.   (..., ) — 미분 불가."""
        return self.bin_values[self.decode_logits(h).argmax(dim=-1)]

    # ── Eq. 8 full : 7-step greedy 자기회귀 디코딩 (z-space 정책: no grad) ──
    @torch.no_grad()
    def decode_action(
        self,
        h0_prime: torch.Tensor,          # (B, 1, H)  = h0 + α·Δh
        past_key_values,                 # 프리필 KV 캐시
        attention_mask: torch.Tensor,    # (B, prefill_len)
    ) -> torch.Tensor:
        """
        h0_prime 에서 시작해 frozen backbone 으로 7개 action 토큰을 argmax 로 이어 디코딩.
        Returns: action (B, action_dim) — 연속값 [-1, 1].
        """
        lm = self._vla.language_model
        B = h0_prime.shape[0]
        cur_h = h0_prime.to(self.B.dtype)
        pkv = past_key_values
        mask = attention_mask.to(cur_h.device)

        vals = []
        for step in range(self.action_dim):
            idx = self.decode_logits(cur_h.squeeze(1)).argmax(dim=-1)   # (B,)
            vals.append(self.bin_values[idx])
            if step < self.action_dim - 1:
                emb = self.input_embed_rows[idx].unsqueeze(1)           # (B, 1, H)
                mask = torch.cat(
                    [mask, torch.ones((B, 1), dtype=mask.dtype, device=mask.device)], dim=1
                )
                out = lm(
                    input_ids=None,
                    inputs_embeds=emb,
                    attention_mask=mask,
                    past_key_values=pkv,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
                cur_h = out.hidden_states[-1][:, -1:, :].to(self.B.dtype)
                pkv = out.past_key_values

        return torch.stack(vals, dim=-1)                                # (B, action_dim)

    # ── Eq. 8 full : 7-step 미분가능 디코딩 (BC warm-up 전용) ─────────────
    def decode_action_soft(
        self,
        h0_prime: torch.Tensor,          # (B, 1, H)  = h0 + α·Δh   (grad 有: z -> μ_z 까지)
        past_key_values,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        greedy 대신 soft-argmax + soft-embedding 으로 이어 디코딩 -> action 7-dim 전부에
        대해 z(=μ_z) 까지 gradient 가 흐른다. backbone 파라미터는 frozen 이라 갱신 안 되고
        activation 을 통해서만 gradient 가 지난다. BC(behavior cloning) warm-up 에서만 사용.

        주의: no_grad 아님. 6번의 backbone single-token forward 의 activation 이 메모리에
        남는다 (KV 캐시 재사용이라 토큰 1개씩이므로 프롬프트 forward 보다는 훨씬 가벼움).
        """
        lm = self._vla.language_model
        B = h0_prime.shape[0]
        cur_h = h0_prime.to(self.B.dtype)
        pkv = past_key_values
        mask = attention_mask.to(cur_h.device)

        vals = []
        for step in range(self.action_dim):
            probs = self.decode_probs(cur_h.squeeze(1))                      # (B, 256) — grad
            vals.append((probs * self.bin_values.float()).sum(dim=-1))       # (B,)
            if step < self.action_dim - 1:
                soft_emb = (probs.to(self.input_embed_rows.dtype) @ self.input_embed_rows).unsqueeze(1)  # (B,1,H) — grad
                mask = torch.cat(
                    [mask, torch.ones((B, 1), dtype=mask.dtype, device=mask.device)], dim=1
                )
                out = lm(
                    input_ids=None,
                    inputs_embeds=soft_emb,
                    attention_mask=mask,
                    past_key_values=pkv,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
                cur_h = out.hidden_states[-1][:, -1:, :].to(self.B.dtype)
                pkv = out.past_key_values

        return torch.stack(vals, dim=-1)                                     # (B, action_dim) — grad

    # ── 편의 ───────────────────────────────────────────────────────────
    @torch.no_grad()
    def z_to_action(self, h0, z, past_key_values, attention_mask, alpha: float | None = None):
        """[rollout] h0 + z -> h0' -> 7-step greedy decode -> action (B, action_dim)."""
        h0_prime = self.apply_residual(h0, z, alpha)
        return self.decode_action(h0_prime, past_key_values, attention_mask)

    def z_to_action_soft(self, h0, z, past_key_values, attention_mask, alpha: float | None = None):
        """[BC warm-up] 위와 같지만 미분가능(soft) decode."""
        h0_prime = self.apply_residual(h0, z, alpha)
        return self.decode_action_soft(h0_prime, past_key_values, attention_mask)


if __name__ == "__main__":
    # backbone 없이 되는 부분만 스모크: bin-value 매핑 정확성 + projection shape/grad.
    import numpy as np

    # 1) compute_action_bin_values 가 ActionTokenizer.decode_token_ids_to_actions 와 일치하는지
    #    (임의 vocab_size 로, vocab 독립성까지 확인)
    def _ref_decode(token_ids, vocab_size, num_bins=256):
        bins = np.linspace(MIN_ACTION, MAX_ACTION, num_bins)
        centers = (bins[:-1] + bins[1:]) / 2.0
        disc = vocab_size - np.asarray(token_ids)
        disc = np.clip(disc - 1, 0, centers.shape[0] - 1)
        return centers[disc]

    for vocab_size in (32000, 32064, 128256):
        rows = np.arange(vocab_size - 256, vocab_size)           # B 의 행 j 에 대응하는 token id
        ref = _ref_decode(rows, vocab_size)
        ours = compute_action_bin_values(256).numpy()
        assert np.allclose(ref, ours, atol=1e-6), f"mismatch @ vocab={vocab_size}"
    print("compute_action_bin_values == ActionTokenizer decode  ✓  (vocab-independent)")
    v = compute_action_bin_values(256)
    print(f"  row0 -> {v[0]:+.4f}   row255 -> {v[-1]:+.4f}")

    # 2) projection 수식 shape / grad (가짜 B 로 nn.Module 우회).
    #    실제 hidden state / lm_head 행은 norm 이 크지 않아 softmax 가 포화하지 않는다.
    #    여기서도 작은 스케일로 만들어 soft-decode gradient 까지 확인한다.
    H, Z, Bn = 4096, 256, 3
    proj = ActionBasisProjection.__new__(ActionBasisProjection)
    nn.Module.__init__(proj)
    proj.alpha, proj.num_bins, proj.action_dim = 1.0, 256, 7
    proj._inject_linear = None                     # __new__ 로 __init__ 를 건너뛰므로 수동 설정
    torch.manual_seed(0)
    Bmat = 0.02 * torch.randn(256, H)
    proj.register_buffer("B", Bmat, persistent=False)
    proj.register_buffer("B_inject", Bmat.clone(), persistent=False)
    proj.register_buffer("bin_values", compute_action_bin_values(256), persistent=False)
    proj.register_buffer("input_embed_rows", torch.randn(256, H), persistent=False)
    proj.hidden_dim = H

    h = 0.1 * torch.randn(Bn, H)
    z = (0.1 * torch.randn(Bn, Z)).requires_grad_(True)   # leaf 로 만든다 (0.1*randn 은 non-leaf)

    hp = proj.apply_residual(h, z)
    assert hp.shape == (Bn, H)

    # (a) z -> Δh -> h' -> logits : 선형 경로, gradient 항상 흐른다
    proj.decode_logits(hp).sum().backward(retain_graph=True)
    assert z.grad is not None and z.grad.abs().sum() > 0
    z.grad = None

    # (b) z -> ... -> soft-argmax(expected) : softmax 미포화 스케일에서 gradient 확인
    a_soft = proj.decode_expected(hp)                            # (Bn,)
    a_soft.sum().backward()
    assert z.grad is not None and z.grad.abs().sum() > 0

    a_hard = proj.decode_argmax(hp)
    assert a_hard.shape == (Bn,) and float(a_hard.abs().max()) <= 1.0
    print("projection shape/grad smoke  ✓  (logits 경로 + soft-decode 경로 모두 미분 가능)")
    print(f"  Δh norm ~ {proj.delta_h(z).norm(dim=-1).mean().item():.3f}")
    print("  주의: decode_expected 의 softmax 는 logits 가 크면 포화해 gradient 가 사라진다 "
          "-> z-space 정책은 애초에 decode gradient 에 의존하지 않음(greedy)")
