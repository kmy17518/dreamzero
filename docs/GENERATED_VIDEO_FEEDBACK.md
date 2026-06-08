# Generated-Video Feedback (Explicit Conditioning)

This note explains, at a conceptual level, one form of **explicit conditioning**: a policy that
**predicts a future video frame and feeds its own generated frame back into the action path** as input.
Explicit conditioning comes in two flavors — conditioning the action on the model's **latent** video
tokens (within a single forward pass), and conditioning it on the **generated video frames** themselves
(this variant). The contrasting mechanism is **representation alignment**, where predicting the future
only shapes the shared representation and nothing generated reaches the action. Here the generated future
is not merely a training signal — it is actually consumed at decision time.

## The idea

A joint video + action model can do two things at once: imagine what the scene will look like next, and
decide what to do. *Explicit conditioning* asks whether **letting the action see that imagined future**
helps — i.e., does the policy act better when it is handed a prediction of the next observation, rather
than only the current one?

To test this cleanly we make the model **predict the video one step further into the future than the
action**, and then feed that generated frame back as context for the next decision. We call this a
**future-frame shift**.

## Future-frame shift

The model runs autoregressively over **blocks**. A block is one closed-loop unit: a short window of
video frames together with the action chunk and the proprioceptive state for that window.

- **Aligned (latent-token conditioning).** At block `b` the model jointly predicts the video, the
  action, and the state for the *same* window `b`; the action already attends to the being-generated
  **latent** video tokens for that window. Video and action are time-aligned.
- **Shifted (this variant).** We roll the action/state back one block relative to the video, so at block
  `b` the model predicts the video for window `b` but the **action for the previous window**.
  Equivalently, per closed-loop step `t`:
  - **input:** the current observation `o_t`, concatenated in time with the **previously generated next
    frame** `ô_{t+1}`;
  - **outputs:** the **action for `t+1`** and the **video for `t+2`**.

So the video always leads the action by exactly one block, and the action is conditioned on a frame the
model imagined one step earlier.

| block | video predicted | action predicted | conditioned on (clean context) |
|---|---|---|---|
| 0 | next frame | — (no valid target) | current observation |
| 1 | +1 | action for block 0 | current obs + generated frame 0 |
| 2 | +2 | action for block 1 | current obs + generated frames 0–1 |

### Why the state stays with the action

At inference the model can imagine future *frames* but not future *proprioception*. So the action is
always paired with the **most recent real** proprioceptive state — the one available when that action is
decided. No future proprio is ever fed in, only an imagined future *image*, which keeps training and
inference consistent.

### Cost: one unsupervised block

Because the action is shifted back, the very first block of a sequence has no valid action target and is
**excluded from the action loss**. One action chunk per training clip is therefore unsupervised; using
longer clips (more blocks per sequence) shrinks this fraction.

## Training

Training is **teacher-forced**: the "previously generated next frame" fed back is simply the
ground-truth next frame. The video-prediction loss is unchanged; the action loss is computed on the
shifted action target, with the unsupervised leading block masked out. Nothing else about the model
changes — only the indexing of the action/state relative to the video.

## Inference: feeding the generated future back

At rollout there are two ways to use the prediction, and they differ in whether the generated frame is
*actually* fed back across decisions:

1. **Discarded prediction (no real feedback).** The model still co-generates the next frame inside a
   single forward pass — so the action is conditioned on it for that one step — but the prediction is
   thrown away, and the next decision re-reads only the real camera. This is conditioning *within* a
   step, not true closed-loop feedback.
2. **Grounded feedback (true explicit conditioning).** Each step, the model is re-anchored on the
   **current real observation** and given exactly **one** previously generated frame as the imagined
   "next" context. It predicts the next frame plus the aligned action, then stores that newly generated
   frame to feed back at the following step. The context is therefore always `[real now, one imagined
   next]` and **never accumulates** the model's own outputs.

### Why re-ground on reality every step

If the model is instead allowed to condition on a growing stack of its **own** past predictions, the
imagined world drifts: it stops respecting the real per-step pace and "runs toward the goal," so the
imagined rollout races ahead and re-imagines the task. Keeping only a **single** generated frame in
context, re-grounded on the real observation each step, prevents these errors from compounding.

### Matching the feedback cadence

One decision produces a whole block of motion (one action chunk), and that whole block is what gets fed
back as the imagined next frame. So the controller should **execute a full block before re-querying**;
then the imagined frame lines up with the next real observation (≈ 1:1). If the controller re-queries
much more often than once per block, the imagined future runs ahead of the robot and the conditioning
becomes temporally misaligned.

## Relation to the project question

Across the project's variants, this one is **explicit conditioning on generated video frames** — the
model's own decoded future is fed back as an *input* to the action. It sits alongside two reference
points: **explicit conditioning on latent video tokens** (the aligned joint model, where the action
attends to the being-generated latent video but no decoded frame is fed back), and **representation
alignment** (the decoupled model, where the future-prediction loss shapes the shared representation but
the action→generated-video path is cut, so nothing generated reaches the action). Comparing them isolates
how much of any gain comes from a better representation versus from the policy actually consuming a
generated future — and, among the conditioning variants, whether latent tokens or decoded frames matter.
