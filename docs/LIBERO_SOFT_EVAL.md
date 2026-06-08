# LIBERO Soft Eval (Progress / Partial-Stage Scoring)

This note explains, at a conceptual level, the **progress-score** ("soft") evaluation used alongside the
standard binary success rate on LIBERO. It covers what the metric is and how to read it — not how to run
it.

## Why a soft score

Binary success is **all-or-nothing**: a policy that grasps the right object and carries it almost to the
target scores the same **0** as a policy that never moves. That makes early training look noisy and hides
real progress. Soft eval gives **partial credit** for how far through a task the policy got, while still
agreeing exactly with binary success at the end.

## How it works

Each task has a formal goal predicate (e.g. *object A is on plate B*). We decompose that goal into an
**ordered sequence of stages** and check, **at every simulation step** (using ground-truth simulator
state), which stage has been reached. Tasks fall into a few families, each with its own stage sequence:

- **Pick-and-place** (put an object on / in a target): `approach source → grasp source → approach target → done`
- **Push** (move an object into a target zone): `approach source → near target → done`
- **Articulation** (open / close / turn on / off): `approach → done`

The **final** stage is exactly the task's goal predicate, so reaching it is identical to a binary
success.

### Latching and ordering

Stages are **strictly ordered** and **latched**: a later stage only counts once the earlier stages have
been reached, and once reached a stage stays credited. The score therefore moves monotonically forward
and, crucially, **progress = 1.0 if and only if the task is a binary success**. The intermediate stages
(approach / near) are detected with simple distance thresholds; the decisive stages (grasp, done) are
exact, so the partial-credit thresholds never change whether a run counts as a success.

## How to read the numbers

For a set of trials on a task, soft eval reports:

- **Per-stage reach fraction** — the fraction of trials that reached *at least* that stage. The last
  stage's reach fraction is exactly the **success rate** (e.g. "every trial grasped the bowl, 80 % got it
  over the plate, 60 % actually placed it").
- **Episode progress** — for a single trial, `(furthest stage reached) / (number of stages)`.
- **Mean episode progress** — episode progress averaged over trials (per task), and again over tasks
  (per suite), giving a single soft score next to the binary success rate.

Goals with several parts (conjunctions) are scored per part, and the trial's progress is the average
across parts.

## Relation to the standard metric

Soft eval is a strict **superset** of binary success: it adds a smooth, partial-credit view of behavior
without changing the success number (the last stage equals the goal predicate). It is most useful for
distinguishing policies that fail in different ways — "never grasps" vs "grasps but doesn't place" — and
for seeing progress earlier in training than binary success can reveal.
