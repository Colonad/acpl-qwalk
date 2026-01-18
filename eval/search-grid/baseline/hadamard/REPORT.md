# ACPL Evaluation Report — search-grid

- Generated: `2026-01-11T13:09:46Z`
- Policy: `baseline`
- Baseline kind: `hadamard`
- Device: `cpu`
- Suite: `search-grid`
- Episodes per seed: `256`
- Seeds: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]` (n=10)

## Headline metrics

| Condition | mix.tv | target.tv_vsU | target.success | mix.H | mix.hell | mix.js |
| --- | --- | --- | --- | --- | --- | --- |
| baseline_hadamard | 0.0804 [0.0804, 0.0804] (se=0.0000, n=2560) | 0.0804 [0.0804, 0.0804] (se=0.0000, n=2560) | 0.0157 [0.0156, 0.0158] (se=0.0001, n=2560) | 4.1426 [4.1426, 4.1426] (se=0.0000, n=2560) | 0.0651 [0.0651, 0.0651] (se=0.0000, n=2560) | 0.0042 [0.0042, 0.0042] (se=0.0000, n=2560) |

## Figures

### Other figures

- `figs/confusion_matrix__baseline_hadamard.png`
- `figs/position_timelines__baseline_hadamard.png`
- `figs/tv_to_uniform__baseline_hadamard.png`

## Appendix: per-condition CI tables

### baseline_hadamard

**eval_ci.txt**

```
CI over pooled episodes — baseline_hadamard — SU2 (deg=2)
---------------------------------------------------------
mix.H: 4.1426 [4.1426, 4.1426] @ 95% (n=2560, se=0)
  seed_mean: 4.1426 [4.1426, 4.1426] @ 95% (n=10, se=0)
mix.hell: 0.0651 [0.0651, 0.0651] @ 95% (n=2560, se=0)
  seed_mean: 0.0651 [0.0651, 0.0651] @ 95% (n=10, se=0)
mix.js: 0.0042 [0.0042, 0.0042] @ 95% (n=2560, se=0)
  seed_mean: 0.0042 [0.0042, 0.0042] @ 95% (n=10, se=0)
mix.kl_pu: 0.0163 [0.0163, 0.0163] @ 95% (n=2560, se=0)
  seed_mean: 0.0163 [0.0163, 0.0163] @ 95% (n=10, se=0)
mix.l2: 0.0219 [0.0219, 0.0219] @ 95% (n=2560, se=0)
  seed_mean: 0.0219 [0.0219, 0.0219] @ 95% (n=10, se=0)
mix.tv: 0.0804 [0.0804, 0.0804] @ 95% (n=2560, se=0)
  seed_mean: 0.0804 [0.0804, 0.0804] @ 95% (n=10, se=0)
target.H: 4.1426 [4.1426, 4.1426] @ 95% (n=2560, se=0)
  seed_mean: 4.1426 [4.1426, 4.1426] @ 95% (n=10, se=0)
target.KLpU: 0.0163 [0.0163, 0.0163] @ 95% (n=2560, se=0)
  seed_mean: 0.0163 [0.0163, 0.0163] @ 95% (n=10, se=0)
target.cvar_neglogOmega: 4.1682 [4.1609, 4.1755] @ 95% (n=2560, se=0.003698)
  seed_mean: 4.1682 [4.1576, 4.1776] @ 95% (n=10, se=0.005089)
target.gini: 0.0871 [0.0871, 0.0871] @ 95% (n=2560, se=0)
  seed_mean: 0.0871 [0.0871, 0.0871] @ 95% (n=10, se=0)
target.js_vsU: 0.0042 [0.0042, 0.0042] @ 95% (n=2560, se=0)
  seed_mean: 0.0042 [0.0042, 0.0042] @ 95% (n=10, se=0)
target.maxp: 0.0179 [0.0179, 0.0179] @ 95% (n=2560, se=0)
  seed_mean: 0.0179 [0.0179, 0.0179] @ 95% (n=10, se=0)
target.success: 0.0157 [0.0156, 0.0158] @ 95% (n=2560, se=5.282e-05)
  seed_mean: 0.0157 [0.0156, 0.0159] @ 95% (n=10, se=7.147e-05)
target.tv_vsU: 0.0804 [0.0804, 0.0804] @ 95% (n=2560, se=0)
  seed_mean: 0.0804 [0.0804, 0.0804] @ 95% (n=10, se=0)
```

**eval_ci.json (parsed)**

| Metric | Mean | 95% CI | stderr | n |
| --- | ---: | ---: | ---: | ---: |
| mix.tv | 0.0804 | [0.0804, 0.0804] | 0.0000 | 2560 |
| target.tv_vsU | 0.0804 | [0.0804, 0.0804] | 0.0000 | 2560 |
| target.success | 0.0157 | [0.0156, 0.0158] | 0.0001 | 2560 |
| mix.H | 4.1426 | [4.1426, 4.1426] | 0.0000 | 2560 |
| mix.hell | 0.0651 | [0.0651, 0.0651] | 0.0000 | 2560 |
| mix.js | 0.0042 | [0.0042, 0.0042] | 0.0000 | 2560 |
| mix.kl_pu | 0.0163 | [0.0163, 0.0163] | 0.0000 | 2560 |
| mix.l2 | 0.0219 | [0.0219, 0.0219] | 0.0000 | 2560 |
| target.H | 4.1426 | [4.1426, 4.1426] | 0.0000 | 2560 |
| target.KLpU | 0.0163 | [0.0163, 0.0163] | 0.0000 | 2560 |
| target.cvar_neglogOmega | 4.1682 | [4.1609, 4.1755] | 0.0037 | 2560 |
| target.gini | 0.0871 | [0.0871, 0.0871] | 0.0000 | 2560 |
| target.js_vsU | 0.0042 | [0.0042, 0.0042] | 0.0000 | 2560 |
| target.maxp | 0.0179 | [0.0179, 0.0179] | 0.0000 | 2560 |
