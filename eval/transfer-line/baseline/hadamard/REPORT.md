# ACPL Evaluation Report — transfer-line

- Generated: `2026-01-11T15:26:20Z`
- Policy: `baseline`
- Baseline kind: `hadamard`
- Device: `cpu`
- Suite: `transfer-line`
- Episodes per seed: `256`
- Seeds: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]` (n=10)

## Headline metrics

| Condition | mix.tv | target.tv_vsU | target.success | mix.H | mix.hell | mix.js |
| --- | --- | --- | --- | --- | --- | --- |
| baseline_hadamard | 0.7982 [0.7982, 0.7982] (se=0.0000, n=2560) | 0.7982 [0.7982, 0.7982] (se=0.0000, n=2560) | 0.0000 [0.0000, 0.0000] (se=0.0000, n=2560) | 1.8951 [1.8951, 1.8951] (se=0.0000, n=2560) | 0.7543 [0.7543, 0.7543] (se=0.0000, n=2560) | 0.4465 [0.4465, 0.4465] (se=0.0000, n=2560) |

## Figures

### Other figures

- `figs/position_timelines__baseline_hadamard.png`
- `figs/tv_to_uniform__baseline_hadamard.png`

## Appendix: per-condition CI tables

### baseline_hadamard

**eval_ci.txt**

```
CI over pooled episodes — baseline_hadamard — SU2 (deg=2)
---------------------------------------------------------
mix.H: 1.8951 [1.8951, 1.8951] @ 95% (n=2560, se=0)
  seed_mean: 1.8951 [1.8951, 1.8951] @ 95% (n=10, se=0)
mix.hell: 0.7543 [0.7543, 0.7543] @ 95% (n=2560, se=0)
  seed_mean: 0.7543 [0.7543, 0.7543] @ 95% (n=10, se=0)
mix.js: 0.4465 [0.4465, 0.4465] @ 95% (n=2560, se=0)
  seed_mean: 0.4465 [0.4465, 0.4465] @ 95% (n=10, se=0)
mix.kl_pu: 2.2638 [2.2638, 2.2638] @ 95% (n=2560, se=0)
  seed_mean: 2.2638 [2.2638, 2.2638] @ 95% (n=10, se=0)
mix.l2: 0.4903 [0.4903, 0.4903] @ 95% (n=2560, se=0)
  seed_mean: 0.4903 [0.4903, 0.4903] @ 95% (n=10, se=0)
mix.tv: 0.7982 [0.7982, 0.7982] @ 95% (n=2560, se=0)
  seed_mean: 0.7982 [0.7982, 0.7982] @ 95% (n=10, se=0)
target.H: 1.8951 [1.8951, 1.8951] @ 95% (n=2560, se=0)
  seed_mean: 1.8951 [1.8951, 1.8951] @ 95% (n=10, se=0)
target.KLpU: 2.2638 [2.2638, 2.2638] @ 95% (n=2560, se=0)
  seed_mean: 2.2638 [2.2638, 2.2638] @ 95% (n=10, se=0)
target.cvar_neglogOmega: 18.4207 [18.4207, 18.4207] @ 95% (n=2560, se=0)
  seed_mean: 18.4207 [18.4207, 18.4207] @ 95% (n=10, se=0)
target.gini: 0.9159 [0.9159, 0.9159] @ 95% (n=2560, se=0)
  seed_mean: 0.9159 [0.9159, 0.9159] @ 95% (n=10, se=0)
target.js_vsU: 0.4465 [0.4465, 0.4465] @ 95% (n=2560, se=0)
  seed_mean: 0.4465 [0.4465, 0.4465] @ 95% (n=10, se=0)
target.maxp: 0.4471 [0.4471, 0.4471] @ 95% (n=2560, se=0)
  seed_mean: 0.4471 [0.4471, 0.4471] @ 95% (n=10, se=0)
target.success: 0.0000 [0.0000, 0.0000] @ 95% (n=2560, se=0)
  seed_mean: 0.0000 [0.0000, 0.0000] @ 95% (n=10, se=0)
target.tv_vsU: 0.7982 [0.7982, 0.7982] @ 95% (n=2560, se=0)
  seed_mean: 0.7982 [0.7982, 0.7982] @ 95% (n=10, se=0)
```

**eval_ci.json (parsed)**

| Metric | Mean | 95% CI | stderr | n |
| --- | ---: | ---: | ---: | ---: |
| mix.tv | 0.7982 | [0.7982, 0.7982] | 0.0000 | 2560 |
| target.tv_vsU | 0.7982 | [0.7982, 0.7982] | 0.0000 | 2560 |
| target.success | 0.0000 | [0.0000, 0.0000] | 0.0000 | 2560 |
| mix.H | 1.8951 | [1.8951, 1.8951] | 0.0000 | 2560 |
| mix.hell | 0.7543 | [0.7543, 0.7543] | 0.0000 | 2560 |
| mix.js | 0.4465 | [0.4465, 0.4465] | 0.0000 | 2560 |
| mix.kl_pu | 2.2638 | [2.2638, 2.2638] | 0.0000 | 2560 |
| mix.l2 | 0.4903 | [0.4903, 0.4903] | 0.0000 | 2560 |
| target.H | 1.8951 | [1.8951, 1.8951] | 0.0000 | 2560 |
| target.KLpU | 2.2638 | [2.2638, 2.2638] | 0.0000 | 2560 |
| target.cvar_neglogOmega | 18.4207 | [18.4207, 18.4207] | 0.0000 | 2560 |
| target.gini | 0.9159 | [0.9159, 0.9159] | 0.0000 | 2560 |
| target.js_vsU | 0.4465 | [0.4465, 0.4465] | 0.0000 | 2560 |
| target.maxp | 0.4471 | [0.4471, 0.4471] | 0.0000 | 2560 |
