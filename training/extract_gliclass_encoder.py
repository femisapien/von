"""Extract the ModernBERT encoder from a knowledgator GLiClass checkpoint.

GLiClass checkpoints (e.g. knowledgator/gliclass-modern-large-v2.0) wrap a plain
ModernBERT encoder (`model.encoder_model.*`) plus two small projectors and a
logit scale. Only the encoder transfers into Von's option-marker model; this
script writes it out as a standalone `ModernBertModel` directory that
`OptionMarkerModel(base_model_id=...)` and `train_option_marker.py --base_model_id`
can load directly.

The GLiClass tokenizer adds two tokens (`<<LABEL>>`, `<<SEP>>`, ids 50368-50369);
the embedding matrix therefore has 50370 rows. Both are kept: they are inert for
Von's packing (which uses [MASK] = 50284) and dropping rows would desync the
saved tokenizer from the weights.

Usage:
    python training/extract_gliclass_encoder.py \
        --src knowledgator/gliclass-modern-large-v2.0 \
        --out checkpoints/gliclass-modern-large-encoder
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from safetensors.torch import load_file, save_file

PREFIX = "model.encoder_model."


def _resolve(src: str) -> str:
    if os.path.isdir(src):
        return src
    from huggingface_hub import snapshot_download
    return snapshot_download(src, allow_patterns=["*.json", "*.safetensors", "tokenizer*"])


def extract(src: str, out: str) -> dict:
    src_dir = _resolve(src)
    with open(os.path.join(src_dir, "config.json"), encoding="utf-8") as f:
        wrapper = json.load(f)
    enc_cfg = wrapper.get("encoder_config")
    if not isinstance(enc_cfg, dict) or enc_cfg.get("model_type") != "modernbert":
        raise ValueError(f"{src}: expected a GLiClass wrapper around a modernbert encoder, got {type(enc_cfg).__name__}")

    tensors = load_file(os.path.join(src_dir, "model.safetensors"))
    enc = {k[len(PREFIX):]: v.contiguous() for k, v in tensors.items() if k.startswith(PREFIX)}
    dropped = sorted(k for k in tensors if not k.startswith(PREFIX))
    if not enc:
        raise ValueError(f"{src}: no tensors under {PREFIX!r}")

    vocab_rows = enc["embeddings.tok_embeddings.weight"].shape[0]
    enc_cfg = dict(enc_cfg)
    enc_cfg["vocab_size"] = int(vocab_rows)
    enc_cfg["architectures"] = ["ModernBertModel"]
    enc_cfg.pop("_name_or_path", None)
    enc_cfg["_von_extracted_from"] = src

    os.makedirs(out, exist_ok=True)
    save_file(enc, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
    with open(os.path.join(out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(enc_cfg, f, indent=2)
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        p = os.path.join(src_dir, name)
        if os.path.exists(p):
            with open(p, "rb") as fi, open(os.path.join(out, name), "wb") as fo:
                fo.write(fi.read())

    return {
        "encoder_tensors": len(enc),
        "dropped": dropped,
        "vocab_rows": int(vocab_rows),
        "hidden_size": enc_cfg.get("hidden_size"),
        "layers": enc_cfg.get("num_hidden_layers"),
        "params_m": round(sum(v.numel() for v in enc.values()) / 1e6, 1),
    }


def verify(out: str) -> None:
    """Round-trip: AutoModel loads with zero missing/unexpected keys, [MASK] survives."""
    from transformers import AutoModel, AutoTokenizer
    model, info = AutoModel.from_pretrained(out, output_loading_info=True)
    bad = {k: v for k, v in info.items() if v}
    if bad:
        raise RuntimeError(f"loading info not clean: {bad}")
    tok = AutoTokenizer.from_pretrained(out)
    if tok.mask_token_id is None or tok.mask_token_id >= model.config.vocab_size:
        raise RuntimeError(f"mask token id {tok.mask_token_id} invalid for vocab {model.config.vocab_size}")
    with torch.no_grad():
        ids = tok("The claim was filed on 2024-02-29. [MASK]", return_tensors="pt")
        h = model(**ids).last_hidden_state
    if not torch.isfinite(h).all():
        raise RuntimeError("non-finite hidden states")
    print(f"verify ok: mask_token_id={tok.mask_token_id} vocab={model.config.vocab_size} hidden={tuple(h.shape)}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", default="knowledgator/gliclass-modern-large-v2.0")
    ap.add_argument("--out", default="checkpoints/gliclass-modern-large-encoder")
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args()
    report = extract(a.src, a.out)
    print(json.dumps(report, indent=2))
    if not a.no_verify:
        verify(a.out)


if __name__ == "__main__":
    main()
