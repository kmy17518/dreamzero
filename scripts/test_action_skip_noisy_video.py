"""Numerical check for the `action_skip_noisy_video` ablation in CausalWanSelfAttention.

Verifies the intended behavior in BOTH code paths used by training and inference:
  - Training teacher-forcing path  (`_process_noisy_action_blocks`)
  - Inference KV-cache path        (`forward(..., kv_cache=...)`)

Property under test (flag ON): the action/state register output is *invariant* to the NOISY video
block being denoised (the video-denoising target), but STILL depends on the CLEAN video context
(first frame + clean context blocks / the KV cache history). With the flag OFF (default), the action
output depends on the noisy video block (existing behavior unchanged).

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
        qt, kt, vt = (t.transpose(1, 2).float() for t in (q, k, v))
        out = F.scaled_dot_product_attention(qt, kt, vt)
        return out.transpose(1, 2).to(q.dtype)


def _make_attn(flag):
    torch.manual_seed(0)  # identical weights for flag on/off
    attn = CausalWanSelfAttention(
        dim=8, num_heads=2, frame_seqlen=3, local_attn_size=-1, sink_size=0,
        num_frame_per_block=1, qk_norm=False, eps=1e-6,
        num_action_per_block=2, num_state_per_block=1,
        action_skip_noisy_video=flag,
    ).eval()
    attn.attn = _SDPA()
    attn.causal_attn = _SDPA()
    return attn


def test_training_path():
    b, n, d, fs = 1, 2, 4, 3
    noisy_frames, clean_frames = 3, 3
    num_blocks = (noisy_frames - 1) // 1                 # 2
    action_horizon, state_horizon = num_blocks * 2, num_blocks * 1  # 4, 2

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

    # (1) Perturb ONLY the noisy video block -> flag on: invariant; flag off: changes.
    nik2, niv2 = noisy_img_k + 5.0, noisy_img_v + 5.0
    for flag in (True, False):
        attn = _make_attn(flag)
        base = run(attn, clean_k, clean_v, noisy_img_k, noisy_img_v)
        pert = run(attn, clean_k, clean_v, nik2, niv2)
        diff = (base - pert).abs().max().item()
        if flag:
            assert diff < 1e-6, f"[train] skip-noisy action changed with the noisy block! diff={diff}"
            print(f"[train] flag=True : action invariant to the NOISY video block (max|d|={diff:.2e})  OK")
        else:
            assert diff > 1e-2, f"[train] default action should depend on the noisy block, diff={diff}"
            print(f"[train] flag=False: action depends on the noisy video block (max|d|={diff:.2e})    OK")

    # (2) flag on: action STILL depends on the CLEAN CONTEXT beyond frame 0 (kept).
    attn = _make_attn(True)
    base = run(attn, clean_k, clean_v, noisy_img_k, noisy_img_v)
    ck2, cv2 = clean_k.clone(), clean_v.clone()
    ck2[:, fs:] += 5.0   # clean context blocks 1..i-1 (beyond frame 0)
    cv2[:, fs:] += 5.0
    diff = (base - run(attn, ck2, cv2, noisy_img_k, noisy_img_v)).abs().max().item()
    assert diff > 1e-2, f"[train] skip-noisy action should depend on clean context blocks, diff={diff}"
    print(f"[train] flag=True : action DOES depend on clean context (beyond frame0) (max|d|={diff:.2e}) OK")

    # (3) flag on: action depends on frame 0 (current obs) too.
    ck3, cv3 = clean_k.clone(), clean_v.clone()
    ck3[:, :fs] += 5.0
    cv3[:, :fs] += 5.0
    diff = (base - run(attn, ck3, cv3, noisy_img_k, noisy_img_v)).abs().max().item()
    assert diff > 1e-2, f"[train] skip-noisy action should depend on frame 0, diff={diff}"
    print(f"[train] flag=True : action DOES depend on frame 0 (current obs) (max|d|={diff:.2e})        OK")


def test_inference_path():
    orig = M.causal_rope_action_apply
    M.causal_rope_action_apply = lambda x, **kw: x
    try:
        b, n, d, fs = 1, 2, 4, 3
        dim = n * d
        napb, nspb = 2, 1
        action_register_length = napb + nspb          # 3
        img_tokens = fs * 1                            # current (noisy) block = 1 frame
        s = img_tokens + action_register_length
        cache_len = 2 * fs                            # 2 cached (clean context) frames

        g = torch.Generator().manual_seed(2)
        x = torch.randn(b, s, dim, generator=g)
        kv = torch.randn(2, b, cache_len, n, d, generator=g)
        freqs = torch.zeros(1)

        def run(attn, x_in, kv_in):
            out, _ = attn.forward(
                x_in, freqs, freqs, freqs,
                action_register_length=action_register_length,
                kv_cache=kv_in, current_start_frame=1, is_tf=False,
            )
            return out[:, -action_register_length:]

        # (1) Perturb the CURRENT noisy block (x's image tokens) -> flag on: invariant; off: changes.
        x2 = x.clone()
        x2[:, :img_tokens] += 5.0
        for flag in (True, False):
            attn = _make_attn(flag)
            base = run(attn, x, kv)
            pert = run(attn, x2, kv)
            diff = (base - pert).abs().max().item()
            if flag:
                assert diff < 1e-6, f"[infer] skip-noisy register changed with the noisy block! diff={diff}"
                print(f"[infer] flag=True : register invariant to the CURRENT noisy block (max|d|={diff:.2e})  OK")
            else:
                assert diff > 1e-2, f"[infer] default register should depend on the noisy block, diff={diff}"
                print(f"[infer] flag=False: register depends on the current noisy block (max|d|={diff:.2e})    OK")

        # (2) flag on: register STILL depends on the CLEAN context (the KV cache history).
        attn = _make_attn(True)
        base = run(attn, x, kv)
        kv2 = kv.clone()
        kv2[:, :, :] += 5.0   # perturb the cached clean-context frames
        diff = (base - run(attn, x, kv2)).abs().max().item()
        assert diff > 1e-2, f"[infer] skip-noisy register should depend on the KV-cache context, diff={diff}"
        print(f"[infer] flag=True : register DOES depend on the clean KV-cache context (max|d|={diff:.2e})  OK")
    finally:
        M.causal_rope_action_apply = orig


def test_threading():
    from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import CausalWanModel
    for flag in (True, False):
        m = CausalWanModel(model_type='ti2v', dim=8, num_heads=2, num_layers=2, ffn_dim=16,
                           in_dim=16, out_dim=16, frame_seqlen=4, max_chunk_size=4,
                           num_frame_per_block=2, num_action_per_block=2, num_state_per_block=1,
                           action_skip_noisy_video=flag)
        assert all(blk.self_attn.action_skip_noisy_video == flag for blk in m.blocks)
    m = CausalWanModel(model_type='ti2v', dim=8, num_heads=2, num_layers=1, ffn_dim=16, in_dim=16,
                       out_dim=16, frame_seqlen=4, max_chunk_size=4, num_frame_per_block=2)
    assert m.blocks[0].self_attn.action_skip_noisy_video is False
    print("[thread] CausalWanModel -> block -> self_attn threading (default False)               OK")


if __name__ == "__main__":
    test_training_path()
    test_inference_path()
    test_threading()
    print("\nALL CHECKS PASSED")
