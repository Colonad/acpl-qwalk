# Report Notes — basic_valid

## Reproducibility

- Eval dir: `/mnt/c/Users/Angel/Desktop/CS598/acpl-qwalk/eval/expA`
- Checkpoint: `runs/testing/model_last.pt`
- Base condition: `ckpt_policy`
- Seeds: `[0]` (n=1)
- Episodes per seed: `10`
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

- `figs/Pt__ckpt_policy.png` — Pt timeline (ckpt_policy)
- `figs/Pt__ckpt_policy_abl_GlobalCoin.png` — Pt timeline (ckpt_policy__abl_GlobalCoin)
- `figs/Pt__ckpt_policy_abl_NoPE.png` — Pt timeline (ckpt_policy__abl_NoPE)
- `figs/Pt__ckpt_policy_abl_TimeFrozen.png` — Pt timeline (ckpt_policy__abl_TimeFrozen)

## Method notes / caveats

- CI method & settings: (copy from eval_ci.txt) …
- Any skipped conditions? …
- Any task-specific interpretability notes (e.g., NodePermute timelines not meaningful) …
