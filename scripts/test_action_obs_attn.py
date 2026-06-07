"""Numerical check for the `action_attend_obs_only` ablation in CausalWanSelfAttention.

Verifies the core invariant in BOTH code paths used by training and inference:
  - Training teacher-forcing path  (`_process_noisy_action_blocks`)
  - Inference KV-cache path        (`forward(..., kv_cache=...)`)

Property under test: when `action_attend_obs_only=True`, the ACTION/STATE register output must be
*invariant* to any change in the future video tokens (clean future frames + the to-be-generated
noisy/cached video beyond the first frame), and must still *change* when the first frame (current
observation) or the register itself changes. With the flag off (default), the action output MUST
depend on the future video tokens (i.e. behavior is unchanged).

Runs on CPU in float32 (attention is monkeypatched to a plain SDPA) so it needs no GPU and is exact.
"""
import os
import torch
import torch.nn.functional as F

os.environ.setdefault("ATTENTION_BACKEND", "torch")

from groot.vla.model.dreamzero.modules import wan_video_dit_action_casual_chunk as M
from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import CausalWanSelfAttention


class _SDPA(torch.nn.Module):
    """Plain float32 full-attention stand-in for AttentionModule (CPU, exact)."""
    def forward(self, q, k, v, *args, **kwargs):
        # q,k,v: [b, L, n, d] -> [b, n, L, d] -> SDPA -> back.
        qt, kt, vt = (t.transpose(1, 2).float() for t in (q, k, v))
        out = F.scaled_dot_product_attention(qt, kt, vt)
        return out.transpose(1, 2).to(q.dtype)


def _make_attn(flag):
    torch.manual_seed(0)  # identical weights for flag on/off
    attn = CausalWanSelfAttention(
        dim=8, num_heads=2, frame_seqlen=3, local_attn_size=-1, sink_size=0,
        num_frame_per_block=1, qk_norm=False, eps=1e-6,
        num_action_per_block=2, num_state_per_block=1,
        action_attend_obs_only=flag,
    ).eval()
    attn.attn = _SDPA()          # used by action/image block processing
    attn.causal_attn = _SDPA()   # used by clean-image processing (not exercised here)
    return attn


def test_training_path():
    b, n, d, fs = 1, 2, 4, 3
    nfpb, napb, nspb = 1, 2, 1
    noisy_frames, clean_frames = 3, 3
    num_blocks = (noisy_frames - 1) // nfpb              # 2
    action_horizon, state_horizon = num_blocks * napb, num_blocks * nspb  # 4, 2

    g = torch.Generator().manual_seed(1)
    def rnd(L):
        return torch.randn(b, L, n, d, generator=g)

    clean_k, clean_v = rnd(clean_frames * fs), rnd(clean_frames * fs)
    noisy_img_k, noisy_img_v = rnd(noisy_frames * fs), rnd(noisy_frames * fs)
    act_q, act_k, act_v = rnd(action_horizon), rnd(action_horizon), rnd(action_horizon)
    st_k, st_v = rnd(state_horizon), rnd(state_horizon)

    def run(attn, ck, cv, nik, niv):
        return attn._process_noisy_action_blocks(
            act_q, act_k, act_v, ck, cv, nik, niv, st_k, st_v,
            noisy_frames, action_horizon, state_horizon,
        )

    # Perturb ONLY the future video: clean future frames (everything after frame 0) + all noisy img.
    clean_k2, clean_v2 = clean_k.clone(), clean_v.clone()
    clean_k2[:, fs:] += 5.0
    clean_v2[:, fs:] += 5.0
    noisy_img_k2, noisy_img_v2 = noisy_img_k + 5.0, noisy_img_v + 5.0

    for flag in (True, False):
        attn = _make_attn(flag)
        base = run(attn, clean_k, clean_v, noisy_img_k, noisy_img_v)
        pert = run(attn, clean_k2, clean_v2, noisy_img_k2, noisy_img_v2)
        diff = (base - pert).abs().max().item()
        if flag:
            assert diff < 1e-6, f"[train] obs-only action output changed with future video! diff={diff}"
            print(f"[train] flag=True : action invariant to future video (max|d|={diff:.2e})  OK")
        else:
            assert diff > 1e-2, f"[train] default action output should depend on future video, diff={diff}"
            print(f"[train] flag=False: action depends on future video (max|d|={diff:.2e})    OK")

    # Also confirm the obs-only action STILL depends on the first frame (current obs).
    attn = _make_attn(True)
    base = run(attn, clean_k, clean_v, noisy_img_k, noisy_img_v)
    ck3, cv3 = clean_k.clone(), clean_v.clone()
    ck3[:, :fs] += 5.0  # perturb frame 0 only
    cv3[:, :fs] += 5.0
    pert = run(attn, ck3, cv3, noisy_img_k, noisy_img_v)
    diff = (base - pert).abs().max().item()
    assert diff > 1e-2, f"[train] obs-only action should depend on the first frame, diff={diff}"
    print(f"[train] flag=True : action DOES depend on first frame (max|d|={diff:.2e})        OK")


def test_inference_path():
    # Monkeypatch rope to identity so we don't need real RoPE frequency tables.
    orig = M.causal_rope_action_apply
    M.causal_rope_action_apply = lambda x, **kw: x
    try:
        b, n, d, fs = 1, 2, 4, 3
        dim = n * d
        napb, nspb = 2, 1
        action_register_length = napb + nspb          # 3
        img_tokens = fs * 1                            # current block = 1 frame
        s = img_tokens + action_register_length        # 6
        cache_len = 2 * fs                             # 6 (2 cached frames)

        g = torch.Generator().manual_seed(2)
        x = torch.randn(b, s, dim, generator=g)
        kv = torch.randn(2, b, cache_len, n, d, generator=g)
        freqs = torch.zeros(1)  # ignored (rope patched to identity)

        def run(attn, x_in, kv_in):
            out, _ = attn.forward(
                x_in, freqs, freqs, freqs,
                action_register_length=action_register_length,
                kv_cache=kv_in, current_start_frame=1, is_tf=False,
            )
            return out[:, -action_register_length:]  # action/state register output

        # Perturb future video: cached frames beyond frame 0 + the current image tokens in x.
        x2 = x.clone()
        x2[:, :img_tokens] += 5.0
        kv2 = kv.clone()
        kv2[:, :, fs:] += 5.0  # keep cache frame 0 (== current obs) fixed

        for flag in (True, False):
            attn = _make_attn(flag)
            base = run(attn, x, kv)
            pert = run(attn, x2, kv2)
            diff = (base - pert).abs().max().item()
            if flag:
                assert diff < 1e-6, f"[infer] obs-only register changed with future video! diff={diff}"
                print(f"[infer] flag=True : register invariant to future video (max|d|={diff:.2e})  OK")
            else:
                assert diff > 1e-2, f"[infer] default register should depend on future video, diff={diff}"
                print(f"[infer] flag=False: register depends on future video (max|d|={diff:.2e})    OK")

        # obs-only register STILL depends on the first cached frame (current observation).
        attn = _make_attn(True)
        base = run(attn, x, kv)
        kv3 = kv.clone()
        kv3[:, :, :fs] += 5.0  # perturb frame 0 only
        pert = run(attn, x, kv3)
        diff = (base - pert).abs().max().item()
        assert diff > 1e-2, f"[infer] obs-only register should depend on the first frame, diff={diff}"
        print(f"[infer] flag=True : register DOES depend on first frame (max|d|={diff:.2e})        OK")
    finally:
        M.causal_rope_action_apply = orig


if __name__ == "__main__":
    test_training_path()
    test_inference_path()
    print("\nALL CHECKS PASSED")
