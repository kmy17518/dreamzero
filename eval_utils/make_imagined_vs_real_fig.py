"""Build the *imagined-vs-real* qualitative figure from an imagine-dump ``.npz``.

For the feedback variant (``context_mode=C``) each query ``t`` conditions on the current real
observation ``O_t`` and emits an imagined next-frame block ``Ĝ_t`` (the frontier it feeds back). The
realized next observation is simply the *next* query's real anchor ``O_{t+1}``. This script overlays
``Ĝ_t`` (model forecast) against ``O_{t+1}`` (what actually happened) across the episode, so the
figure shows how accurate the forecast is and where it drifts.

The model generates a 2-view composite ``[agentview | wristview]`` (each 160x160, seam at the middle).
By default we show the **agentview** (the canonical third-person scene); ``--view composite`` shows
both, ``--view wrist`` the wrist camera.

Rows (top->bottom): realized next obs O_{t+1} | imagined Ĝ_t | 50/50 overlay (ghosting = drift) |
per-pixel error heatmap. Columns are evenly-spaced timesteps, annotated with SSIM / PSNR.

Usage:
  python eval_utils/make_imagined_vs_real_fig.py --npz imagine_dump/checkpoint-40000/<ep>.npz \
      --out eval_outputs/imagined_vs_real/<name>.png --cols 6 --tag SUCCESS --view agent
"""

import argparse
import os

import numpy as np
import cv2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

try:
    from skimage.metrics import structural_similarity as _ssim
    _HAVE_SSIM = True
except Exception:  # noqa: BLE001
    _HAVE_SSIM = False


def _psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse <= 1e-8 else float(10.0 * np.log10((255.0 ** 2) / mse))


def _ssim_rgb(a, b):
    if not _HAVE_SSIM:
        return float("nan")
    try:
        return float(_ssim(a, b, channel_axis=2, data_range=255))
    except TypeError:
        return float(_ssim(a, b, multichannel=True, data_range=255))


def _up(img, scale):
    h, w = img.shape[:2]
    return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST
                      if False else cv2.INTER_LINEAR)


def _view_crop(frame, view):
    """frame is the (H, W=2*S, 3) composite [agent | wrist]. Return the requested view as a square."""
    H, W = frame.shape[:2]
    half = W // 2
    if view == "agent":
        return frame[:, :half]
    if view == "wrist":
        return frame[:, half:]
    return frame  # composite


def _real_view(agent, wrist, view, tile):
    """Build the realized observation in the same layout as the imagined view, sized to `tile`."""
    th, tw = tile
    if view == "agent":
        return cv2.resize(agent, (th, th), interpolation=cv2.INTER_AREA)
    if view == "wrist":
        return cv2.resize(wrist, (th, th), interpolation=cv2.INTER_AREA)
    a = cv2.resize(agent, (th, th), interpolation=cv2.INTER_AREA)
    w = cv2.resize(wrist, (th, th), interpolation=cv2.INTER_AREA)
    return np.concatenate([a, w], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--tag", default="")
    ap.add_argument("--view", default="agent", choices=["agent", "wrist", "composite"])
    ap.add_argument("--replan-steps", type=int, default=24)
    ap.add_argument("--num-steps-wait", type=int, default=10)
    ap.add_argument("--skip-first", type=int, default=1,
                    help="drop the first N query pairs (q0 is the episode-start no-op warmup)")
    ap.add_argument("--upscale", type=int, default=2)
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    imag_full = d["imagined"]
    agent = d["real_agent"] if "real_agent" in d else d["real_anchors"]
    wrist = d["real_wrist"] if "real_wrist" in d else agent
    prompt = str(d["prompt"]) if "prompt" in d else ""
    session = str(d["session_id"]) if "session_id" in d else os.path.basename(args.npz)
    Q = min(len(imag_full), len(agent), len(wrist))

    # square tile size = composite height (= half width)
    S = imag_full.shape[1]

    t_lo = max(0, args.skip_first)
    pairs = [(t, t + 1) for t in range(t_lo, Q - 1)] or [(0, min(1, Q - 1))]
    k = min(args.cols, len(pairs))
    idx = np.linspace(0, len(pairs) - 1, k).round().astype(int)
    sel = [pairs[i] for i in idx]

    n_rows = 4
    tile_w = (2 if args.view == "composite" else 1)
    fig = plt.figure(figsize=(2.4 * tile_w * k, 2.1 * n_rows + 1.0))
    gs = gridspec.GridSpec(n_rows, k, figure=fig, hspace=0.06, wspace=0.05,
                           left=0.075, right=0.995, top=0.85, bottom=0.015)
    row_labels = ["Realized\nnext obs\n$O_{t+1}$", "Imagined\nforecast\n$\\hat{G}_t$",
                  "Overlay\n(50/50)", "Error\n|diff|"]

    ssims, psnrs = [], []
    for c, (ti, tn) in enumerate(sel):
        im = _view_crop(imag_full[ti], args.view).astype(np.uint8)
        ro = _real_view(agent[tn], wrist[tn], args.view, (S, S)).astype(np.uint8)
        if ro.shape != im.shape:
            ro = cv2.resize(ro, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_AREA)
        ov = (0.5 * ro.astype(np.float32) + 0.5 * im.astype(np.float32)).clip(0, 255).astype(np.uint8)
        diff = np.abs(ro.astype(np.int16) - im.astype(np.int16)).mean(axis=2)
        s = _ssim_rgb(ro, im); p = _psnr(ro, im)
        ssims.append(s); psnrs.append(p)
        real_step = args.num_steps_wait + args.replan_steps * tn

        for r, content in enumerate([ro, im, ov, diff]):
            ax = fig.add_subplot(gs[r, c])
            if r < 3:
                ax.imshow(_up(content, args.upscale))
            else:
                ax.imshow(_up(content.astype(np.uint8), args.upscale), cmap="inferno", vmin=0, vmax=110)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("0.55"); sp.set_linewidth(0.6)
            if r == 0:
                sub = (f"SSIM {s:.2f}" if not np.isnan(s) else f"PSNR {p:.1f}dB")
                ax.set_title(f"step ~{real_step}\n{sub}", fontsize=9, pad=3)
            if c == 0:
                ax.set_ylabel(row_labels[r], fontsize=9, rotation=0, ha="right", va="center", labelpad=2)

    mean_ssim = float(np.nanmean(ssims)) if ssims else float("nan")
    mean_psnr = float(np.nanmean(psnrs)) if psnrs else float("nan")
    tag = f"[{args.tag}]  " if args.tag else ""
    mstr = (f"mean SSIM={mean_ssim:.2f}, PSNR={mean_psnr:.1f} dB"
            if not np.isnan(mean_ssim) else f"mean PSNR={mean_psnr:.1f} dB")
    fig.suptitle(
        f"{tag}Imagined future frame  vs.  realized next observation   "
        f"(feedback context_mode=C, ckpt-40000, {args.view} view)\n"
        f"\u201c{prompt}\u201d      \u2014      {mstr}",
        fontsize=12, y=0.965,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}  (Q={Q}, cols={k}, view={args.view}, mean SSIM={mean_ssim:.3f}, PSNR={mean_psnr:.2f})")


if __name__ == "__main__":
    main()
