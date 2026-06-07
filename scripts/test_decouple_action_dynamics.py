"""Unit test for the `decouple_action_dynamics` attention flag (LIBERO al/dl-decoupled variant).

This verifies, on CPU with the plain torch SDPA backend, that the DiT self-attention behaves as
intended in BOTH the runtime paths used by LIBERO:

  1. Training (teacher forcing, kv_cache=None, is_tf=True): action tokens must NOT attend to the
     noisy (being-generated / "future") video tokens, but MUST still attend to the clean
     observation video tokens. Video tokens must still attend to action tokens (unchanged).
  2. Inference (kv_cache != None): action tokens must NOT attend to the current noisy video block,
     but MUST still attend to the cached clean observation tokens.

The check is decisive: with the flag ON, perturbing the noisy/generated video leaves the action
output *bitwise* identical (the action's attention inputs are literally independent of it), while
perturbing the clean observation video *does* change the action output. With the flag OFF (the
original joint behavior) the action output depends on the noisy video in both paths.

Run:
    PYTHONPATH=/root/libero_al_dl_decoupled \
    ATTENTION_BACKEND=torch \
    python scripts/test_decouple_action_dynamics.py
"""

import os

# CPU-friendly, deterministic attention (no flash-attn / GPU needed) + polar rope (no TRT path).
os.environ.setdefault("ATTENTION_BACKEND", "torch")
os.environ.setdefault("ENABLE_TENSORRT", "False")

import torch

from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import (
    CausalWanSelfAttention,
)
from groot.vla.model.dreamzero.modules.wan2_1_submodule import rope_params


# ---- tiny but structurally-valid config -------------------------------------------------------
NUM_HEADS = 2
HEAD_DIM = 8
DIM = NUM_HEADS * HEAD_DIM          # 16
FRAME_SEQLEN = 4                    # video tokens per latent frame
NUM_FRAME_PER_BLOCK = 1
NUM_ACTION_PER_BLOCK = 2
NUM_STATE_PER_BLOCK = 1
NOISY_FRAMES = 3                    # -> num_image_blocks = (3-1)//1 = 2
B = 1

torch.manual_seed(0)


def build_attn() -> CausalWanSelfAttention:
    attn = CausalWanSelfAttention(
        dim=DIM,
        num_heads=NUM_HEADS,
        frame_seqlen=FRAME_SEQLEN,
        local_attn_size=-1,
        num_frame_per_block=NUM_FRAME_PER_BLOCK,
        num_action_per_block=NUM_ACTION_PER_BLOCK,
        num_state_per_block=NUM_STATE_PER_BLOCK,
        decouple_action_dynamics=False,  # toggled per-run below
    )
    attn.eval()
    return attn


def video_freqs(num_tokens: int) -> torch.Tensor:
    """[num_tokens, 1, HEAD_DIM//2] complex rope freqs for the video tokens."""
    return rope_params(num_tokens, HEAD_DIM).view(num_tokens, 1, -1)


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def test_training_teacher_forcing() -> None:
    attn = build_attn()

    num_image_blocks = (NOISY_FRAMES - 1) // NUM_FRAME_PER_BLOCK
    seq_len_video = NOISY_FRAMES * FRAME_SEQLEN
    action_horizon = num_image_blocks * NUM_ACTION_PER_BLOCK
    state_horizon = num_image_blocks * NUM_STATE_PER_BLOCK
    R = action_horizon + state_horizon
    s = 2 * seq_len_video + R  # [clean video | noisy video | action+state register]

    freqs = video_freqs(seq_len_video)
    freqs_action = rope_params(1024 * 10, HEAD_DIM)
    freqs_state = rope_params(1024, HEAD_DIM)

    base_x = torch.randn(B, s, DIM)

    clean_sl = slice(0, seq_len_video)
    noisy_sl = slice(seq_len_video, 2 * seq_len_video)
    reg_sl = slice(2 * seq_len_video, s)
    # Output layout: [clean_img | noisy_img | noisy_action | noisy_state]
    action_out_sl = slice(2 * seq_len_video, 2 * seq_len_video + action_horizon)

    def run(x, decouple):
        attn.decouple_action_dynamics = decouple
        with torch.no_grad():
            out, _ = attn(
                x=x, freqs=freqs, freqs_action=freqs_action, freqs_state=freqs_state,
                action_register_length=R, kv_cache=None, is_tf=True,
            )
        return out

    x_perturb_noisy = base_x.clone()
    x_perturb_noisy[:, noisy_sl] += 5.0          # perturb the being-generated video
    x_perturb_clean = base_x.clone()
    x_perturb_clean[:, clean_sl] += 5.0          # perturb the clean observation video
    x_perturb_reg = base_x.clone()
    x_perturb_reg[:, reg_sl] += 5.0              # perturb the action/state register

    for decouple in (False, True):
        base = run(base_x, decouple)
        d_noisy = max_abs_diff(run(x_perturb_noisy, decouple)[:, action_out_sl], base[:, action_out_sl])
        d_clean = max_abs_diff(run(x_perturb_clean, decouple)[:, action_out_sl], base[:, action_out_sl])
        # Does the VIDEO output react to perturbing the action register? (video->action link)
        d_video_from_reg = max_abs_diff(run(x_perturb_reg, decouple)[:, noisy_sl], base[:, noisy_sl])
        tag = "DECOUPLED" if decouple else "coupled  "
        print(f"[train/{tag}] action depends on noisy_video: {d_noisy:.4f} | "
              f"action depends on clean_obs: {d_clean:.4f} | video depends on action: {d_video_from_reg:.4f}")

        if decouple:
            assert d_noisy < 1e-6, f"DECOUPLED: action must NOT see noisy video (got {d_noisy})"
            assert d_clean > 1e-2, f"DECOUPLED: action must still see clean obs (got {d_clean})"
        else:
            assert d_noisy > 1e-2, f"coupled: action should see noisy video (got {d_noisy})"
            assert d_clean > 1e-2, f"coupled: action should see clean obs (got {d_clean})"
        # In both modes, the video should still attend to the action register.
        assert d_video_from_reg > 1e-2, f"video must still attend to action (got {d_video_from_reg})"

    print("  -> training teacher-forcing path OK\n")


def test_inference_kv_cache() -> None:
    attn = build_attn()

    R = NUM_ACTION_PER_BLOCK + NUM_STATE_PER_BLOCK   # inference register == one block
    num_new = NUM_FRAME_PER_BLOCK * FRAME_SEQLEN     # current noisy video block
    cache_len = 2 * FRAME_SEQLEN                      # cached clean observation tokens
    s = num_new + R
    current_start_frame = 1                           # action_state_index = 0

    freqs = video_freqs(num_new)
    freqs_action = rope_params(1024 * 10, HEAD_DIM)
    freqs_state = rope_params(1024, HEAD_DIM)

    base_x = torch.randn(B, s, DIM)
    kv_cache = torch.randn(2, B, cache_len, NUM_HEADS, HEAD_DIM)

    video_sl = slice(0, num_new)
    action_out_sl = slice(num_new, s)  # action register output rows

    def run(x, decouple):
        attn.decouple_action_dynamics = decouple
        with torch.no_grad():
            out, _ = attn(
                x=x, freqs=freqs, freqs_action=freqs_action, freqs_state=freqs_state,
                action_register_length=R, kv_cache=kv_cache.clone(),
                current_start_frame=current_start_frame, is_tf=False,
            )
        return out

    x_perturb_video = base_x.clone()
    x_perturb_video[:, video_sl] += 5.0   # perturb the current (being-generated) noisy video block

    for decouple in (False, True):
        base = run(base_x, decouple)
        d_video = max_abs_diff(run(x_perturb_video, decouple)[:, action_out_sl], base[:, action_out_sl])
        tag = "DECOUPLED" if decouple else "coupled  "
        print(f"[infer/{tag}] action depends on current noisy video block: {d_video:.4f}")
        if decouple:
            assert d_video < 1e-6, f"DECOUPLED: action must NOT see current noisy video (got {d_video})"
        else:
            assert d_video > 1e-2, f"coupled: action should see current noisy video (got {d_video})"

    print("  -> inference kv-cache path OK\n")


def test_dynamics_action_decoupled_training() -> None:
    """Mirror gate (`decouple_dynamics_action`): with the flag ON, the VIDEO output must NOT depend
    on the action tokens, but MUST still depend on the state tokens (we only cut video->action, not
    video->state). With the flag OFF, the video depends on the action (original joint behavior)."""
    attn = build_attn()

    num_image_blocks = (NOISY_FRAMES - 1) // NUM_FRAME_PER_BLOCK
    seq_len_video = NOISY_FRAMES * FRAME_SEQLEN
    action_horizon = num_image_blocks * NUM_ACTION_PER_BLOCK
    state_horizon = num_image_blocks * NUM_STATE_PER_BLOCK
    R = action_horizon + state_horizon
    s = 2 * seq_len_video + R  # [clean video | noisy video | action register | state register]

    freqs = video_freqs(seq_len_video)
    freqs_action = rope_params(1024 * 10, HEAD_DIM)
    freqs_state = rope_params(1024, HEAD_DIM)

    base_x = torch.randn(B, s, DIM)

    noisy_sl = slice(seq_len_video, 2 * seq_len_video)
    # Register layout is [action | state]; perturb each part independently.
    action_sl = slice(2 * seq_len_video, 2 * seq_len_video + action_horizon)
    state_sl = slice(2 * seq_len_video + action_horizon, s)

    def run(x, decouple_dyn_act):
        attn.decouple_action_dynamics = False
        attn.decouple_dynamics_action = decouple_dyn_act
        with torch.no_grad():
            out, _ = attn(
                x=x, freqs=freqs, freqs_action=freqs_action, freqs_state=freqs_state,
                action_register_length=R, kv_cache=None, is_tf=True,
            )
        return out

    x_perturb_action = base_x.clone()
    x_perturb_action[:, action_sl] += 5.0
    x_perturb_state = base_x.clone()
    x_perturb_state[:, state_sl] += 5.0

    for dv in (False, True):
        base = run(base_x, dv)
        d_action = max_abs_diff(run(x_perturb_action, dv)[:, noisy_sl], base[:, noisy_sl])
        d_state = max_abs_diff(run(x_perturb_state, dv)[:, noisy_sl], base[:, noisy_sl])
        tag = "DECOUPLED" if dv else "coupled  "
        print(f"[train/dyn-act {tag}] video depends on action: {d_action:.4f} | "
              f"video depends on state: {d_state:.4f}")
        if dv:
            assert d_action < 1e-6, f"DECOUPLED: video must NOT see action (got {d_action})"
        else:
            assert d_action > 1e-2, f"coupled: video should see action (got {d_action})"
        # In both modes the video must still attend to the state tokens.
        assert d_state > 1e-2, f"video must still attend to state (got {d_state})"

    print("  -> training dynamics->action decoupling OK\n")


def test_dynamics_action_decoupled_inference() -> None:
    attn = build_attn()

    R = NUM_ACTION_PER_BLOCK + NUM_STATE_PER_BLOCK   # inference register == one block
    num_new = NUM_FRAME_PER_BLOCK * FRAME_SEQLEN
    cache_len = 2 * FRAME_SEQLEN
    s = num_new + R
    current_start_frame = 1

    freqs = video_freqs(num_new)
    freqs_action = rope_params(1024 * 10, HEAD_DIM)
    freqs_state = rope_params(1024, HEAD_DIM)

    base_x = torch.randn(B, s, DIM)
    kv_cache = torch.randn(2, B, cache_len, NUM_HEADS, HEAD_DIM)

    video_out_sl = slice(0, num_new)
    # Register layout is [action | state].
    action_sl = slice(num_new, num_new + NUM_ACTION_PER_BLOCK)
    state_sl = slice(num_new + NUM_ACTION_PER_BLOCK, s)

    def run(x, decouple_dyn_act):
        attn.decouple_action_dynamics = False
        attn.decouple_dynamics_action = decouple_dyn_act
        with torch.no_grad():
            out, _ = attn(
                x=x, freqs=freqs, freqs_action=freqs_action, freqs_state=freqs_state,
                action_register_length=R, kv_cache=kv_cache.clone(),
                current_start_frame=current_start_frame, is_tf=False,
            )
        return out

    x_perturb_action = base_x.clone()
    x_perturb_action[:, action_sl] += 5.0
    x_perturb_state = base_x.clone()
    x_perturb_state[:, state_sl] += 5.0

    for dv in (False, True):
        base = run(base_x, dv)
        d_action = max_abs_diff(run(x_perturb_action, dv)[:, video_out_sl], base[:, video_out_sl])
        d_state = max_abs_diff(run(x_perturb_state, dv)[:, video_out_sl], base[:, video_out_sl])
        tag = "DECOUPLED" if dv else "coupled  "
        print(f"[infer/dyn-act {tag}] video depends on action: {d_action:.4f} | "
              f"video depends on state: {d_state:.4f}")
        if dv:
            assert d_action < 1e-6, f"DECOUPLED: video must NOT see action (got {d_action})"
        else:
            assert d_action > 1e-2, f"coupled: video should see action (got {d_action})"
        assert d_state > 1e-2, f"video must still attend to state (got {d_state})"

    print("  -> inference dynamics->action decoupling OK\n")


if __name__ == "__main__":
    print(f"torch={torch.__version__} ATTENTION_BACKEND={os.environ.get('ATTENTION_BACKEND')}\n")
    test_training_teacher_forcing()
    test_inference_kv_cache()
    test_dynamics_action_decoupled_training()
    test_dynamics_action_decoupled_inference()
    print("ALL CHECKS PASSED")
