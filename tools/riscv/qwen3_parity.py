#!/usr/bin/env python3
"""Independent numpy oracle for tensorfold-riscv's Qwen3 forward.

Writes a small deterministic f32 Qwen3 checkpoint, computes its forward twice:
once here in float64 (this file's `forward`, written from the HF Qwen3 math
without reference to the Zig code) and once by the tensorfold-riscv binary.
Compares the prompt's last-row logits and the greedy token sequence.

Usage: qwen3_parity.py --bin zig-out/bin/tensorfold-riscv [--work DIR]
Exit 0 when every check passes; prints the receipts either way.
"""
import argparse
import json
import os
import struct
import subprocess
import sys

import numpy as np

SPEC = dict(hidden=32, inter=64, layers=2, heads=4, kv_heads=2, head_dim=8, vocab=97)
EPS = 1e-6
THETA = 1e6


def write_safetensors(path, tensors):
    header = {}
    data = b""
    offset = 0
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        header[name] = {"dtype": "F32", "shape": list(arr.shape), "data_offsets": [offset, offset + arr.nbytes]}
        data += arr.tobytes()
        offset += arr.nbytes
    raw = json.dumps(header).encode()
    hjson = raw + b" " * ((8 - len(raw) % 8) % 8)  # the format pads headers to 8
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        f.write(data)


def build_checkpoint(work):
    os.makedirs(work, exist_ok=True)
    rng = np.random.default_rng(20261115)
    s = SPEC
    w = {}
    w["model.embed_tokens.weight"] = rng.uniform(-0.5, 0.5, (s["vocab"], s["hidden"])).astype(np.float32)
    w["model.norm.weight"] = rng.uniform(0.5, 1.5, (s["hidden"],)).astype(np.float32)
    for i in range(s["layers"]):
        p = f"model.layers.{i}."
        w[p + "self_attn.q_proj.weight"] = rng.uniform(-0.5, 0.5, (s["heads"] * s["head_dim"], s["hidden"])).astype(np.float32)
        w[p + "self_attn.k_proj.weight"] = rng.uniform(-0.5, 0.5, (s["kv_heads"] * s["head_dim"], s["hidden"])).astype(np.float32)
        w[p + "self_attn.v_proj.weight"] = rng.uniform(-0.5, 0.5, (s["kv_heads"] * s["head_dim"], s["hidden"])).astype(np.float32)
        w[p + "self_attn.o_proj.weight"] = rng.uniform(-0.5, 0.5, (s["hidden"], s["heads"] * s["head_dim"])).astype(np.float32)
        w[p + "self_attn.q_norm.weight"] = rng.uniform(0.5, 1.5, (s["head_dim"],)).astype(np.float32)
        w[p + "self_attn.k_norm.weight"] = rng.uniform(0.5, 1.5, (s["head_dim"],)).astype(np.float32)
        w[p + "input_layernorm.weight"] = rng.uniform(0.5, 1.5, (s["hidden"],)).astype(np.float32)
        w[p + "post_attention_layernorm.weight"] = rng.uniform(0.5, 1.5, (s["hidden"],)).astype(np.float32)
        w[p + "mlp.gate_proj.weight"] = rng.uniform(-0.5, 0.5, (s["inter"], s["hidden"])).astype(np.float32)
        w[p + "mlp.up_proj.weight"] = rng.uniform(-0.5, 0.5, (s["inter"], s["hidden"])).astype(np.float32)
        w[p + "mlp.down_proj.weight"] = rng.uniform(-0.5, 0.5, (s["hidden"], s["inter"])).astype(np.float32)
    write_safetensors(os.path.join(work, "model.safetensors"), w)
    config = {
        "architectures": ["Qwen3ForCausalLM"],
        "hidden_size": s["hidden"],
        "intermediate_size": s["inter"],
        "num_hidden_layers": s["layers"],
        "num_attention_heads": s["heads"],
        "num_key_value_heads": s["kv_heads"],
        "head_dim": s["head_dim"],
        "vocab_size": s["vocab"],
        "tie_word_embeddings": True,
        "rms_norm_eps": EPS,
        "rope_theta": THETA,
    }
    with open(os.path.join(work, "config.json"), "w") as f:
        json.dump(config, f)
    return w


def rmsnorm(x, wgt):
    return x / np.sqrt(np.mean(x * x) + EPS) * wgt


def rope(head, pos):
    d = head.shape[-1]
    half = d // 2
    i = np.arange(half)
    freq = THETA ** (-2.0 * i / d)
    c = np.cos(pos * freq)
    s = np.sin(pos * freq)
    out = np.empty_like(head)
    out[:half] = head[:half] * c - head[half:] * s
    out[half:] = head[half:] * c + head[:half] * s
    return out


def forward(w, ids):
    """HF Qwen3 dense forward in float64; returns [len(ids), vocab] logits."""
    s = SPEC
    x = w["model.embed_tokens.weight"][np.array(ids)].astype(np.float64)
    T = len(ids)
    for li in range(s["layers"]):
        p = f"model.layers.{li}."
        normed = np.array([rmsnorm(x[t], w[p + "input_layernorm.weight"]) for t in range(T)])
        q = normed @ w[p + "self_attn.q_proj.weight"].T.astype(np.float64)
        k = normed @ w[p + "self_attn.k_proj.weight"].T.astype(np.float64)
        v = normed @ w[p + "self_attn.v_proj.weight"].T.astype(np.float64)
        cache_k, cache_v = [], []
        attn = np.zeros_like(q)
        for t in range(T):
            qt = q[t].reshape(s["heads"], s["head_dim"])
            kt = k[t].reshape(s["kv_heads"], s["head_dim"])
            vt = v[t].reshape(s["kv_heads"], s["head_dim"])
            for h in range(s["heads"]):
                qt[h] = rmsnorm(qt[h], w[p + "self_attn.q_norm.weight"])
            for h in range(s["kv_heads"]):
                kt[h] = rmsnorm(kt[h], w[p + "self_attn.k_norm.weight"])
            qt = np.stack([rope(qt[h], t) for h in range(s["heads"])])
            kt = np.stack([rope(kt[h], t) for h in range(s["kv_heads"])])
            cache_k.append(kt)
            cache_v.append(vt)
            for h in range(s["heads"]):
                kvh = h // (s["heads"] // s["kv_heads"])
                kk = np.stack([cache_k[j][kvh] for j in range(t + 1)])
                vv = np.stack([cache_v[j][kvh] for j in range(t + 1)])
                scores = kk @ qt[h] / np.sqrt(s["head_dim"])
                scores -= scores.max()
                pr = np.exp(scores)
                pr /= pr.sum()
                attn[t, h * s["head_dim"] : (h + 1) * s["head_dim"]] = pr @ vv
        x = x + attn @ w[p + "self_attn.o_proj.weight"].T.astype(np.float64)
        normed = np.array([rmsnorm(x[t], w[p + "post_attention_layernorm.weight"]) for t in range(T)])
        gate = normed @ w[p + "mlp.gate_proj.weight"].T.astype(np.float64)
        up = normed @ w[p + "mlp.up_proj.weight"].T.astype(np.float64)
        act = (gate / (1.0 + np.exp(-gate))) * up
        x = x + act @ w[p + "mlp.down_proj.weight"].T.astype(np.float64)
    out = np.array([rmsnorm(x[t], w["model.norm.weight"]) for t in range(T)])
    return out @ w["model.embed_tokens.weight"].T.astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True, help="the tensorfold-riscv executable")
    ap.add_argument("--work", default=".zig-cache/riscv-parity")
    ap.add_argument("--tokens", default="3,14,1,5,9,2,6")
    ap.add_argument("--generate", type=int, default=8)
    args = ap.parse_args()

    ids = [int(x) for x in args.tokens.split(",")]
    w = build_checkpoint(args.work)
    ref = forward(w, ids)

    logits_path = os.path.join(args.work, "zig_logits.txt")
    cmd = [args.bin, "run", args.work, "--tokens", args.tokens, "--max-tokens", str(args.generate),
           "--context", "64", "--logits-out", logits_path]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        sys.exit("zig run failed")
    zig_logits = np.array([float(x) for x in open(logits_path)])
    ref_row = ref[-1]

    scale = np.abs(ref_row).max()
    err = np.abs(zig_logits - ref_row).max()
    print(f"logits: max|zig - numpy| = {err:.3e} on scale {scale:.3e} (tol {1e-3 * scale + 1e-4:.3e})")

    # Greedy sequences must match token for token.
    seq = list(ids)
    for _ in range(args.generate):
        nxt = int(np.argmax(forward(w, seq)[-1]))
        seq.append(nxt)
    ref_tokens = seq[len(ids):]

    margin = np.sort(ref_row)[-1] - np.sort(ref_row)[-2]
    print(f"logits: top-1 margin {margin:.3e} (must exceed the tolerance for a meaningful token check)")

    ok = err <= 1e-3 * scale + 1e-4 and margin > 1e-3 * scale + 1e-4
    # Token parity via a run with a report.
    report_path = os.path.join(args.work, "zig_report.json")
    proc2 = subprocess.run(cmd + ["--report", report_path], capture_output=True, text=True)
    if proc2.returncode != 0:
        sys.exit("zig run failed on the report pass")
    zig_tokens = json.loads(open(report_path).read())["tokens"]
    match = zig_tokens == ref_tokens
    print(f"greedy tokens: zig {zig_tokens} vs numpy {ref_tokens} -> {'match' if match else 'MISMATCH'}")
    if not ok or not match:
        sys.exit("parity FAILED")
    print("parity OK")


if __name__ == "__main__":
    main()
