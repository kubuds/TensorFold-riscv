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

# Small dense Qwen3: the f32/q8_0/q4_0 runs use it as-is. q4x needs K to be a
# multiple of its 256-value group, so --spec q4x grows hidden/head_dim: matmul
# K values are hidden (256), inter (512) and heads*head_dim (256).
SPECS = {
    "small": dict(hidden=32, inter=64, layers=2, heads=4, kv_heads=2, head_dim=8, vocab=97),
    "q4x": dict(hidden=256, inter=512, layers=2, heads=4, kv_heads=2, head_dim=64, vocab=97),
}
SPEC = SPECS["small"]
EPS = 1e-6
THETA = 1e6
AQ = False  # q4x quantizes activations to int8; the oracle mirrors it


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


def build_checkpoint(work, spec=None):
    os.makedirs(work, exist_ok=True)
    rng = np.random.default_rng(20261115)
    s = spec or SPEC
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


def round_half_away(x):
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def dequantized(w, mode):
    """The weights the kernels compute with after load-time quantization:
    blocks of 32 along the last axis; d = amax/127 (q8_0) or amax/8 (q4_0)
    rounded to f16, q = round_half_away(w/d) clamped; value = q*d16 resp.
    (q-8)*d16. Mirrors zig/src/riscv/qwen3.zig `quantize` exactly."""
    if mode == "none":
        return w.astype(np.float64)
    out = np.empty_like(w, dtype=np.float64)
    last = w.shape[-1]
    if mode == "q4x":
        # One f16 scale per 256 weights, d = amax/7, q = round(w/d)+8 on
        # [1,15] (zero is exactly 8). Mirrors `quantizeQ4x` in qwen3.zig.
        assert last % 256 == 0
        flat = w.reshape(-1, last)
        outf = out.reshape(-1, last)
        for r in range(flat.shape[0]):
            for g in range(last // 256):
                blk = flat[r, g * 256 : (g + 1) * 256].astype(np.float64)
                amax = np.abs(blk).max()
                d = amax / 7.0
                d16 = float(np.float16(d))
                q = np.clip(round_half_away(blk / d) + 8 if d > 0 else 8, 1, 15)
                outf[r, g * 256 : (g + 1) * 256] = (q - 8) * d16
        return out
    assert last % 32 == 0
    flat = w.reshape(-1, last)
    outf = out.reshape(-1, last)
    for r in range(flat.shape[0]):
        for b in range(last // 32):
            blk = flat[r, b * 32 : (b + 1) * 32].astype(np.float64)
            amax = np.abs(blk).max()
            if mode == "q8_0":
                d = amax / 127.0
                d16 = float(np.float16(d))
                q = np.clip(round_half_away(blk / d) if d > 0 else 0, -127, 127)
                outf[r, b * 32 : (b + 1) * 32] = q * d16
            else:
                d = amax / 8.0
                d16 = float(np.float16(d))
                q = np.clip(round_half_away(blk / d) + 8 if d > 0 else 8, 0, 15)
                outf[r, b * 32 : (b + 1) * 32] = (q - 8) * d16
    return out


def quant_activation(x):
    """int8 per 256-value group, the activation side of every q4x projection
    (d = amax/127, q = round(x/d) on [-127,127], value = q*d with d as f16).
    Decode and prefill share this because both feed vmadot's int8 A operand."""
    k = x.shape[-1]
    if k % 256 != 0:
        return x
    out = x.astype(np.float64).copy()
    for r in range(x.shape[0]):
        for g in range(k // 256):
            blk = x[r, g * 256 : (g + 1) * 256].astype(np.float64)
            amax = np.abs(blk).max()
            d = float(np.float16(amax / 127.0))
            if d > 0:
                out[r, g * 256 : (g + 1) * 256] = np.clip(np.round(blk / d), -127, 127) * d
    return out


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
        a = quant_activation(normed) if AQ else normed
        q = a @ w[p + "self_attn.q_proj.weight"].T.astype(np.float64)
        k = a @ w[p + "self_attn.k_proj.weight"].T.astype(np.float64)
        v = a @ w[p + "self_attn.v_proj.weight"].T.astype(np.float64)
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
        x = x + (quant_activation(attn) if AQ else attn) @ w[p + "self_attn.o_proj.weight"].T.astype(np.float64)
        normed = np.array([rmsnorm(x[t], w[p + "post_attention_layernorm.weight"]) for t in range(T)])
        a2 = quant_activation(normed) if AQ else normed
        gate = a2 @ w[p + "mlp.gate_proj.weight"].T.astype(np.float64)
        up = a2 @ w[p + "mlp.up_proj.weight"].T.astype(np.float64)
        act = (gate / (1.0 + np.exp(-gate))) * up
        x = x + (quant_activation(act) if AQ else act) @ w[p + "mlp.down_proj.weight"].T.astype(np.float64)
    out = np.array([rmsnorm(x[t], w["model.norm.weight"]) for t in range(T)])
    head = w.get("__head__", w["model.embed_tokens.weight"])
    return (quant_activation(out) if AQ else out) @ head.T.astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True, help="the tensorfold-riscv executable")
    ap.add_argument("--work", default=".zig-cache/riscv-parity")
    ap.add_argument("--tokens", default="3,14,1,5,9,2,6")
    ap.add_argument("--generate", type=int, default=8)
    ap.add_argument("--quant", default="none", choices=["none", "q8_0", "q4_0", "q4x"])
    ap.add_argument("--spec", default="small", choices=sorted(SPECS))
    args = ap.parse_args()

    ids = [int(x) for x in args.tokens.split(",")]
    global SPEC, AQ
    SPEC = SPECS[args.spec]
    AQ = args.quant == "q4x"
    w = build_checkpoint(args.work, SPEC)
    if args.quant != "none":
        wq = {}
        for name, arr in w.items():
            # Norms (1-D) and the embedding table never quantize; lm_head -
            # tied to the embedding in this checkpoint - gets its own
            # quantized copy like the Zig loader builds.
            if arr.ndim == 1 or name == "model.embed_tokens.weight":
                wq[name] = arr
            else:
                wq[name] = dequantized(arr, args.quant).astype(np.float32)
        wq["__head__"] = dequantized(w["model.embed_tokens.weight"], args.quant).astype(np.float32)
        w = wq
    ref = forward(w, ids)

    logits_path = os.path.join(args.work, "zig_logits.txt")
    cmd = [args.bin, "run", args.work, "--tokens", args.tokens, "--max-tokens", str(args.generate),
           "--context", "64", "--logits-out", logits_path]
    if args.quant != "none":
        cmd += ["--quant", args.quant]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        sys.exit("zig run failed")
    zig_logits = np.array([float(x) for x in open(logits_path)])
    ref_row = ref[-1]

    scale = np.abs(ref_row).max()
    err = np.abs(zig_logits - ref_row).max()
    print(f"logits: max|zig - numpy| = {err:.3e} on scale {scale:.3e} (tol {1e-3 * scale + 1e-4:.3e}) [quant={args.quant}]")

    # Greedy sequences must match token for token.
    seq = list(ids)
    for _ in range(args.generate):
        nxt = int(np.argmax(forward(w, seq)[-1]))
        seq.append(nxt)
    ref_tokens = seq[len(ids):]

    margin = np.sort(ref_row)[-1] - np.sort(ref_row)[-2]
    print(f"logits: top-1 margin {margin:.3e} (must exceed the tolerance for a meaningful token check)")

    # q4x also quantizes activations to int8 (the vmadot A operand), and the
    # oracle can only approximate the kernel's per-group rounding at the LSB;
    # its logits therefore carry the quantization noise of an independent
    # int8 round trip. Greedy tokens stay the hard gate for that path.
    tol = (5e-2 if args.quant == "q4x" else 1e-3) * scale + 1e-4
    ok = err <= tol and margin > 1e-3 * scale + 1e-4
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
