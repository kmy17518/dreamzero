"""Targeted unit test for the bs>1 fix in the action-head loss reduction.

Replicates the exact action-loss computation from wan_flow_matching_action_tf.py forward()
(lines ~792-799) and checks, on CPU:
  1. The OLD code `has_real_action[:, None]` raises at batch>1 (the bug).
  2. The FIXED code `has_real_action[:, None, None]` runs and is per-sample independent:
       loss([A, B]) == mean(loss([A]), loss([B]))
  3. has_real_action masks per-sample (has_real_action=[1,0] zeros only sample 1).
"""

import torch

torch.manual_seed(0)
B, T, D = 2, 96, 32  # action register length 96, action_dim 32 (matches the crash shapes)


def action_loss(pred, target, action_mask, has_real_action, train_weight, fixed: bool):
    # mirrors wan_flow_matching_action_tf.py forward()
    action_loss_per_sample = torch.nn.functional.mse_loss(
        pred.float(), target.float(), reduction="none"
    ) * action_mask  # [B, T, D]
    if fixed:
        action_loss_per_sample = has_real_action[:, None, None].float() * action_loss_per_sample
    else:
        action_loss_per_sample = has_real_action[:, None].float() * action_loss_per_sample  # OLD (buggy)
    weight_action = action_loss_per_sample.mean(dim=2) * train_weight  # [B, T]
    return weight_action.mean()


def make(b):
    return dict(
        pred=torch.randn(b, T, D),
        target=torch.randn(b, T, D),
        action_mask=(torch.rand(b, T, D) > 0.2).float(),
        has_real_action=torch.ones(b),
        train_weight=torch.rand(b, T) + 0.5,
    )


# Two distinct samples A and B, plus the bs=2 batch [A, B].
A = make(1)
Bs = make(1)
AB = {k: torch.cat([A[k], Bs[k]], dim=0) for k in A}

# 1) OLD code must fail at bs=2
old_failed = False
try:
    action_loss(**AB, fixed=False)
except RuntimeError as e:
    old_failed = True
    print("OLD code at bs=2 -> RuntimeError (expected):", str(e)[:80])
print("OLD code fails at bs>1:", old_failed)

# 2) FIXED code: per-sample independence
lA = action_loss(**A, fixed=True)
lB = action_loss(**Bs, fixed=True)
lAB = action_loss(**AB, fixed=True)
mean_ab = 0.5 * (lA + lB)
rel = abs(lAB - mean_ab) / max(abs(mean_ab).item(), 1e-8)
print(f"\nFIXED: loss(A)={lA.item():.6f} loss(B)={lB.item():.6f} loss([A,B])={lAB.item():.6f} mean(A,B)={mean_ab.item():.6f}")
print(f"FIXED: |loss([A,B]) - mean(A,B)| rel = {rel:.2e}  (should be ~0; weighted_*_loss is a mean over the batch)")

# 3) has_real_action masks per-sample: [1, 0] -> only sample A's loss survives, scaled by 1/2 (mean over B=2)
AB_mask = {**AB, "has_real_action": torch.tensor([1.0, 0.0])}
lAB_masked = action_loss(**AB_mask, fixed=True)
# Expected: (lossA_contrib + 0) / 2  ; lossA as computed in the 2-batch = same per-sample value as lA, averaged with a zeroed row
expected_masked = 0.5 * lA  # sample 0 keeps its loss, sample 1 zeroed, mean over 2 rows
rel_mask = abs(lAB_masked - expected_masked) / max(abs(expected_masked).item(), 1e-8)
print(f"\nFIXED + has_real_action=[1,0]: loss={lAB_masked.item():.6f}  expected(0.5*loss(A))={expected_masked.item():.6f}  rel={rel_mask:.2e}")

ok = old_failed and rel < 1e-5 and rel_mask < 1e-5
print("\nRESULT:", "PASS" if ok else "FAIL")
