# Report Notes — basic_valid

## Reproducibility

- Eval dir: `/home/tobi/acpl-qwalk/eval/smoke/all_flags_ckpt`
- Checkpoint: `runs/testing/model_last.pt`
- Base condition: `ckpt_policy`
- Seeds: `[0]` (n=1)
- Episodes per seed: `2`
- Device: `cpu`

## Headline claim (write this last)

- **Claim:** …
- **Evidence:** cite Table/Fig refs produced by eval.

## What changed across conditions?

- ckpt_policy: trained ACPL policy
- NoPE: … (expected qualitative effect; sanity-check rationale)
- GlobalCoin: … (expected qualitative effect; sanity-check rationale)
- TimeFrozen: … (expected qualitative effect; sanity-check rationale)
- NodePermute: … (expected qualitative effect; sanity-check rationale)

## Results summary (bullet points)

- Base performance: …
- Ablation deltas: …

## Figures to reference


## Method notes / caveats

- CI method & settings: (copy from eval_ci.txt) …
- Any skipped conditions? …
- Any task-specific interpretability notes (e.g., NodePermute timelines not meaningful) …
