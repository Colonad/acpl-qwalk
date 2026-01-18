# ACPL Evaluation Report — mixing-grid

- Generated: `2026-01-10T18:19:44Z`
- Policy: `baseline`
- Baseline kind: `hadamard`
- Device: `cpu`
- Suite: `mixing-grid`
- Episodes per seed: `256`
- Seeds: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]` (n=10)

## Headline metrics

| Condition | mix.tv | mix.H | mix.hell | mix.js | mix.kl_pu | mix.l2 |
| --- | --- | --- | --- | --- | --- | --- |
| baseline_hadamard | 0.9844 [0.9844, 0.9844] (se=0.0000, n=2560) | 0.0001 [0.0001, 0.0001] (se=0.0000, n=2560) | 0.9354 [0.9354, 0.9354] (se=0.0000, n=2560) | 0.6527 [0.6527, 0.6527] (se=0.0000, n=2560) | 4.1588 [4.1588, 4.1588] (se=0.0000, n=2560) | 0.9922 [0.9922, 0.9922] (se=0.0000, n=2560) |

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
```

**eval_ci.json (parsed)**

| Metric | Mean | 95% CI | stderr | n |
| --- | ---: | ---: | ---: | ---: |
| mix.tv | 0.9844 | [0.9844, 0.9844] | 0.0000 | 2560 |
| mix.H | 0.0001 | [0.0001, 0.0001] | 0.0000 | 2560 |
| mix.hell | 0.9354 | [0.9354, 0.9354] | 0.0000 | 2560 |
| mix.js | 0.6527 | [0.6527, 0.6527] | 0.0000 | 2560 |
| mix.kl_pu | 4.1588 | [4.1588, 4.1588] | 0.0000 | 2560 |
| mix.l2 | 0.9922 | [0.9922, 0.9922] | 0.0000 | 2560 |
