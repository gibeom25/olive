"""
OLIVE: frozen VLA backbone 로드 (self-contained).

`~/olive/openvla` 의 experiments util 을 import/patch 하지 않고, VLA 로드에 필요한
최소 로직만 여기에 vendor 한다 (openvla 소스는 건드리지 않는다는 규칙).
prismatic HF remote code 자체는 checkpoint 안에 들어있으므로 `trust_remote_code=True`
로 그대로 쓴다.

load_backbone() 이 돌려주는 것
-----------------------------
    vla        : OpenVLAForActionPrediction  (bfloat16, eval, requires_grad=False)
    processor  : PrismaticProcessor
    meta       : BackboneMeta
                   vocab_size   = processor.tokenizer.vocab_size   (패딩 전)
                   hidden_dim   = vla config 의 llm hidden size
                   norm_stats   = vla.norm_stats  (action 역정규화용)
                   device
"""

from __future__ import annotations

import dataclasses
import json
import os

import torch


@dataclasses.dataclass
class BackboneMeta:
    vocab_size: int
    hidden_dim: int
    device: str
    norm_stats: dict


def _freeze(module: torch.nn.Module) -> None:
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)


def load_backbone(pretrained_checkpoint: str, device: str = "cuda:0",
                  attn_impl: str = "flash_attention_2"):
    """
    pretrained_checkpoint: 로컬 경로 (olive-model.sh 로 받은 openvla-7b-finetuned-libero-*).
    HF hub id 도 되지만 로컬 경로 권장.
    """
    from transformers import AutoModelForVision2Seq, AutoProcessor

    ckpt = os.path.expanduser(pretrained_checkpoint)
    assert ckpt, "pretrained_checkpoint 가 비어있음"

    processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)

    vla = AutoModelForVision2Seq.from_pretrained(
        ckpt,
        attn_implementation=attn_impl,       # flash_attention_2 | sdpa | eager
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    vla = vla.to(device)
    _freeze(vla)

    # action 역정규화 통계: 체크포인트의 dataset_statistics.json (openvla get_vla 와 동일)
    norm_stats = getattr(vla, "norm_stats", None)
    if norm_stats is None:
        stats_path = os.path.join(ckpt, "dataset_statistics.json")
        if os.path.isfile(stats_path):
            with open(stats_path) as f:
                norm_stats = json.load(f)
            vla.norm_stats = norm_stats
        else:
            norm_stats = {}
            print("[OLIVE][backbone] WARNING: dataset_statistics.json 없음 — 역정규화 불가")

    # hidden dim: prismatic config 구조가 버전마다 조금 달라 방어적으로 탐색
    hidden_dim = _infer_hidden_dim(vla)
    vocab_size = processor.tokenizer.vocab_size

    meta = BackboneMeta(vocab_size=vocab_size, hidden_dim=hidden_dim,
                        device=device, norm_stats=norm_stats or {})
    print(f"[OLIVE][backbone] loaded {os.path.basename(ckpt)}  "
          f"hidden_dim={hidden_dim} vocab_size={vocab_size} attn={attn_impl} "
          f"norm_stats_keys={list((norm_stats or {}).keys())[:4]}")
    return vla, processor, meta


def _infer_hidden_dim(vla) -> int:
    # 1) language_model config
    for path in ("language_model.config.hidden_size",
                 "config.text_config.hidden_size",
                 "config.hidden_size"):
        obj = vla
        ok = True
        for attr in path.split("."):
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                ok = False
                break
        if ok and isinstance(obj, int):
            return obj
    # 2) lm_head 로부터
    lm_head = vla.language_model.get_output_embeddings()
    return lm_head.weight.shape[1]


if __name__ == "__main__":
    # 실제 로드는 15GB 체크포인트 + GPU 필요. 여기선 import 경로만 확인.
    import importlib
    for m in ("transformers",):
        assert importlib.util.find_spec(m)
    print("backbone.py import-path OK (실로드는 load_backbone(<ckpt>) 호출 필요)")
