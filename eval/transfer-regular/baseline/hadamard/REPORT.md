# ACPL Evaluation Report — transfer-regular

- Generated: `2026-01-12T00:14:32Z`
- Policy: `baseline`
- Baseline kind: `hadamard`
- Device: `cpu`
- Suite: `transfer-regular`
- Episodes per seed: `256`
- Seeds: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]` (n=10)

## Headline metrics

| Condition | mix.tv | target.tv_vsU | target.success | mix.H | mix.hell | mix.js |
| --- | --- | --- | --- | --- | --- | --- |
| baseline_hadamard | 0.9844 [0.9844, 0.9844] (se=0.0000, n=2560) | 0.9844 [0.9844, 0.9844] (se=0.0000, n=2560) | 0.0000 [0.0000, 0.0000] (se=0.0000, n=2560) | 0.0001 [0.0001, 0.0001] (se=0.0000, n=2560) | 0.9354 [0.9354, 0.9354] (se=0.0000, n=2560) | 0.6527 [0.6527, 0.6527] (se=0.0000, n=2560) |

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
mix.H: 0.0001 [0.0001, 0.0001] @ 95% (n=2560, se=0)
  seed_mean: 0.0001 [0.0001, 0.0001] @ 95% (n=10, se=0)
mix.hell: 0.9354 [0.9354, 0.9354] @ 95% (n=2560, se=0)
  seed_mean: 0.9354 [0.9354, 0.9354] @ 95% (n=10, se=0)
mix.js: 0.6527 [0.6527, 0.6527] @ 95% (n=2560, se=0)
  seed_mean: 0.6527 [0.6527, 0.6527] @ 95% (n=10, se=0)
mix.kl_pu: 4.1588 [4.1588, 4.1588] @ 95% (n=2560, se=0)
  seed_mean: 4.1588 [4.1588, 4.1588] @ 95% (n=10, se=0)
mix.l2: 0.9922 [0.9922, 0.9922] @ 95% (n=2560, se=0)
  seed_mean: 0.9922 [0.9922, 0.9922] @ 95% (n=10, se=0)
mix.tv: 0.9844 [0.9844, 0.9844] @ 95% (n=2560, se=0)
  seed_mean: 0.9844 [0.9844, 0.9844] @ 95% (n=10, se=0)
target.H: 0.0001 [0.0001, 0.0001] @ 95% (n=2560, se=0)
  seed_mean: 0.0001 [0.0001, 0.0001] @ 95% (n=10, se=0)
target.KLpU: 4.1588 [4.1588, 4.1588] @ 95% (n=2560, se=0)
  seed_mean: 4.1588 [4.1588, 4.1588] @ 95% (n=10, se=0)
target.cvar_neglogOmega: 18.4207 [18.4207, 18.4207] @ 95% (n=2560, se=0)
  seed_mean: 18.4207 [18.4207, 18.4207] @ 95% (n=10, se=0)
target.gini: 0.9844 [0.9844, 0.9844] @ 95% (n=2560, se=0)
  seed_mean: 0.9844 [0.9844, 0.9844] @ 95% (n=10, se=0)
target.js_vsU: 0.6527 [0.6527, 0.6527] @ 95% (n=2560, se=0)
  seed_mean: 0.6527 [0.6527, 0.6527] @ 95% (n=10, se=0)
target.maxp: 1.0000 [1.0000, 1.0000] @ 95% (n=2560, se=0)
  seed_mean: 1.0000 [1.0000, 1.0000] @ 95% (n=10, se=0)
target.success: 0.0000 [0.0000, 0.0000] @ 95% (n=2560, se=0)
  seed_mean: 0.0000 [0.0000, 0.0000] @ 95% (n=10, se=0)
target.tv_vsU: 0.9844 [0.9844, 0.9844] @ 95% (n=2560, se=0)
  seed_mean: 0.9844 [0.9844, 0.9844] @ 95% (n=10, se=0)
```

**eval_ci.json (parsed)**

| Metric | Mean | 95% CI | stderr | n |
| --- | ---: | ---: | ---: | ---: |
| mix.tv | 0.9844 | [0.9844, 0.9844] | 0.0000 | 2560 |
| target.tv_vsU | 0.9844 | [0.9844, 0.9844] | 0.0000 | 2560 |
| target.success | 0.0000 | [0.0000, 0.0000] | 0.0000 | 2560 |
| mix.H | 0.0001 | [0.0001, 0.0001] | 0.0000 | 2560 |
| mix.hell | 0.9354 | [0.9354, 0.9354] | 0.0000 | 2560 |
| mix.js | 0.6527 | [0.6527, 0.6527] | 0.0000 | 2560 |
| mix.kl_pu | 4.1588 | [4.1588, 4.1588] | 0.0000 | 2560 |
| mix.l2 | 0.9922 | [0.9922, 0.9922] | 0.0000 | 2560 |
| target.H | 0.0001 | [0.0001, 0.0001] | 0.0000 | 2560 |
| target.KLpU | 4.1588 | [4.1588, 4.1588] | 0.0000 | 2560 |
| target.cvar_neglogOmega | 18.4207 | [18.4207, 18.4207] | 0.0000 | 2560 |
| target.gini | 0.9844 | [0.9844, 0.9844] | 0.0000 | 2560 |
| target.js_vsU | 0.6527 | [0.6527, 0.6527] | 0.0000 | 2560 |
| target.maxp | 1.0000 | [1.0000, 1.0000] | 0.0000 | 2560 |
