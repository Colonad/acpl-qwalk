# ACPL Evaluation Report — basic_valid

- Generated: `2025-12-27T02:58:06Z`
- Checkpoint: `runs/testing/model_last.pt`
- Policy: `ckpt`
- Device: `cpu`
- Suite: `basic_valid`
- Episodes per seed: `10`
- Seeds: `[0]` (n=1)
- Ablations: `['NoPE', 'GlobalCoin', 'TimeFrozen', 'NodePermute']`

## Headline metrics

| Condition | mix.tv | target.tv_vsU | target.success | mix.H | mix.hell | mix.js |
| --- | --- | --- | --- | --- | --- | --- |
| ckpt_policy | 0.8375 [0.8375, 0.8375] (se=0.0000, n=10) | 0.8375 [0.8375, 0.8375] (se=0.0000, n=10) | 0.0000 [0.0000, 0.0000] (se=0.0000, n=10) | 1.8674 [1.8674, 1.8674] (se=0.0000, n=10) | 0.7763 [0.7763, 0.7763] (se=0.0000, n=10) | 0.4769 [0.4769, 0.4769] (se=0.0000, n=10) |
| ckpt_policy__abl_GlobalCoin | 0.7130 [0.7130, 0.7130] (se=0.0000, n=10) | 0.7130 [0.7130, 0.7130] (se=0.0000, n=10) | 0.0000 [0.0000, 0.0000] (se=0.0000, n=10) | 2.4482 [2.4482, 2.4482] (se=0.0000, n=10) | 0.6903 [0.6903, 0.6903] (se=0.0000, n=10) | 0.3764 [0.3764, 0.3764] (se=0.0000, n=10) |
| ckpt_policy__abl_NoPE | 0.8336 [0.8336, 0.8336] (se=0.0000, n=10) | 0.8336 [0.8336, 0.8336] (se=0.0000, n=10) | 0.0000 [0.0000, 0.0000] (se=0.0000, n=10) | 1.8463 [1.8463, 1.8463] (se=0.0000, n=10) | 0.7926 [0.7926, 0.7926] (se=0.0000, n=10) | 0.4848 [0.4848, 0.4848] (se=0.0000, n=10) |
| ckpt_policy__abl_NodePermute | 0.8297 [0.8297, 0.8297] (se=0.0000, n=10) | 0.8297 [0.8297, 0.8297] (se=0.0000, n=10) | 0.0679 [0.0679, 0.0679] (se=0.0000, n=10) | 1.9393 [1.9393, 1.9393] (se=0.0000, n=10) | 0.7860 [0.7860, 0.7860] (se=0.0000, n=10) | 0.4759 [0.4759, 0.4759] (se=0.0000, n=10) |
| ckpt_policy__abl_TimeFrozen | 0.8665 [0.8665, 0.8665] (se=0.0000, n=10) | 0.8665 [0.8665, 0.8665] (se=0.0000, n=10) | 0.0000 [0.0000, 0.0000] (se=0.0000, n=10) | 1.6178 [1.6178, 1.6178] (se=0.0000, n=10) | 0.8059 [0.8059, 0.8059] (se=0.0000, n=10) | 0.5079 [0.5079, 0.5079] (se=0.0000, n=10) |

## Deltas vs base condition

Base: `ckpt_policy`

| Condition | Metric | Base | Cond | Δ | Δ% |
| --- | --- | ---: | ---: | ---: | ---: |
| ckpt_policy__abl_GlobalCoin | mix.tv | 0.8375 | 0.7130 | -0.1245 | -14.8651% |
| ckpt_policy__abl_GlobalCoin | target.tv_vsU | 0.8375 | 0.7130 | -0.1245 | -14.8651% |
| ckpt_policy__abl_GlobalCoin | target.success | 0.0000 | 0.0000 | +0.0000 |  |
| ckpt_policy__abl_GlobalCoin | mix.H | 1.8674 | 2.4482 | +0.5808 | +31.1027% |
| ckpt_policy__abl_GlobalCoin | mix.hell | 0.7763 | 0.6903 | -0.0861 | -11.0866% |
| ckpt_policy__abl_GlobalCoin | mix.js | 0.4769 | 0.3764 | -0.1006 | -21.0839% |
| ckpt_policy__abl_NoPE | mix.tv | 0.8375 | 0.8336 | -0.0040 | -0.4769% |
| ckpt_policy__abl_NoPE | target.tv_vsU | 0.8375 | 0.8336 | -0.0040 | -0.4769% |
| ckpt_policy__abl_NoPE | target.success | 0.0000 | 0.0000 | +0.0000 |  |
| ckpt_policy__abl_NoPE | mix.H | 1.8674 | 1.8463 | -0.0211 | -1.1279% |
| ckpt_policy__abl_NoPE | mix.hell | 0.7763 | 0.7926 | +0.0162 | +2.0922% |
| ckpt_policy__abl_NoPE | mix.js | 0.4769 | 0.4848 | +0.0078 | +1.6384% |
| ckpt_policy__abl_NodePermute | mix.tv | 0.8375 | 0.8297 | -0.0079 | -0.9426% |
| ckpt_policy__abl_NodePermute | target.tv_vsU | 0.8375 | 0.8297 | -0.0079 | -0.9426% |
| ckpt_policy__abl_NodePermute | target.success | 0.0000 | 0.0679 | +0.0679 |  |
| ckpt_policy__abl_NodePermute | mix.H | 1.8674 | 1.9393 | +0.0719 | +3.8494% |
| ckpt_policy__abl_NodePermute | mix.hell | 0.7763 | 0.7860 | +0.0096 | +1.2412% |
| ckpt_policy__abl_NodePermute | mix.js | 0.4769 | 0.4759 | -0.0010 | -0.2189% |
| ckpt_policy__abl_TimeFrozen | mix.tv | 0.8375 | 0.8665 | +0.0290 | +3.4622% |
| ckpt_policy__abl_TimeFrozen | target.tv_vsU | 0.8375 | 0.8665 | +0.0290 | +3.4622% |
| ckpt_policy__abl_TimeFrozen | target.success | 0.0000 | 0.0000 | +0.0000 |  |
| ckpt_policy__abl_TimeFrozen | mix.H | 1.8674 | 1.6178 | -0.2496 | -13.3681% |
| ckpt_policy__abl_TimeFrozen | mix.hell | 0.7763 | 0.8059 | +0.0295 | +3.8044% |
| ckpt_policy__abl_TimeFrozen | mix.js | 0.4769 | 0.5079 | +0.0310 | +6.4893% |

## Figures

### ckpt_policy — Position timeline

![Pt](figs/Pt__ckpt_policy.png)

### ckpt_policy__abl_GlobalCoin — Position timeline

![Pt](figs/Pt__ckpt_policy_abl_GlobalCoin.png)

### ckpt_policy__abl_NoPE — Position timeline

![Pt](figs/Pt__ckpt_policy_abl_NoPE.png)

### ckpt_policy__abl_TimeFrozen — Position timeline

![Pt](figs/Pt__ckpt_policy_abl_TimeFrozen.png)

## Appendix: per-condition CI tables

### ckpt_policy

**eval_ci.txt**

```
CI over pooled episodes — ckpt_policy — EXP (any degree)
--------------------------------------------------------
mix.H: 1.8674 [1.8674, 1.8674] @ 95% (n=10, se=0)
mix.hell: 0.7763 [0.7763, 0.7763] @ 95% (n=10, se=0)
mix.js: 0.4769 [0.4769, 0.4769] @ 95% (n=10, se=0)
mix.kl_pu: 2.2915 [2.2915, 2.2915] @ 95% (n=10, se=0)
mix.l2: 0.4391 [0.4391, 0.4391] @ 95% (n=10, se=0)
mix.tv: 0.8375 [0.8375, 0.8375] @ 95% (n=10, se=0)
target.H: 1.8674 [1.8674, 1.8674] @ 95% (n=10, se=0)
target.KLpU: 2.2915 [2.2915, 2.2915] @ 95% (n=10, se=0)
target.cvar_neglogOmega: 18.4207 [18.4207, 18.4207] @ 95% (n=10, se=0)
target.gini: 0.9235 [0.9235, 0.9235] @ 95% (n=10, se=0)
target.js_vsU: 0.4769 [0.4769, 0.4769] @ 95% (n=10, se=0)
target.maxp: 0.3159 [0.3159, 0.3159] @ 95% (n=10, se=0)
target.success: 0.0000 [0.0000, 0.0000] @ 95% (n=10, se=0)
target.tv_vsU: 0.8375 [0.8375, 0.8375] @ 95% (n=10, se=0)
```

**eval_ci.json (parsed)**

| Metric | Mean | 95% CI | stderr | n |
| --- | ---: | ---: | ---: | ---: |
| mix.tv | 0.8375 | [0.8375, 0.8375] | 0.0000 | 10 |
| target.tv_vsU | 0.8375 | [0.8375, 0.8375] | 0.0000 | 10 |
| target.success | 0.0000 | [0.0000, 0.0000] | 0.0000 | 10 |
| mix.H | 1.8674 | [1.8674, 1.8674] | 0.0000 | 10 |
| mix.hell | 0.7763 | [0.7763, 0.7763] | 0.0000 | 10 |
| mix.js | 0.4769 | [0.4769, 0.4769] | 0.0000 | 10 |
| mix.kl_pu | 2.2915 | [2.2915, 2.2915] | 0.0000 | 10 |
| mix.l2 | 0.4391 | [0.4391, 0.4391] | 0.0000 | 10 |
| target.H | 1.8674 | [1.8674, 1.8674] | 0.0000 | 10 |
| target.KLpU | 2.2915 | [2.2915, 2.2915] | 0.0000 | 10 |
| target.cvar_neglogOmega | 18.4207 | [18.4207, 18.4207] | 0.0000 | 10 |
| target.gini | 0.9235 | [0.9235, 0.9235] | 0.0000 | 10 |
| target.js_vsU | 0.4769 | [0.4769, 0.4769] | 0.0000 | 10 |
| target.maxp | 0.3159 | [0.3159, 0.3159] | 0.0000 | 10 |

### ckpt_policy__abl_GlobalCoin

**eval_ci.txt**

```
CI over pooled episodes — ckpt_policy__abl_GlobalCoin — EXP (any degree)
------------------------------------------------------------------------
mix.H: 2.4482 [2.4482, 2.4482] @ 95% (n=10, se=0)
mix.hell: 0.6903 [0.6903, 0.6903] @ 95% (n=10, se=0)
mix.js: 0.3764 [0.3764, 0.3764] @ 95% (n=10, se=0)
mix.kl_pu: 1.7107 [1.7107, 1.7107] @ 95% (n=10, se=0)
mix.l2: 0.3536 [0.3536, 0.3536] @ 95% (n=10, se=0)
mix.tv: 0.7130 [0.7130, 0.7130] @ 95% (n=10, se=0)
target.H: 2.4482 [2.4482, 2.4482] @ 95% (n=10, se=0)
target.KLpU: 1.7107 [1.7107, 1.7107] @ 95% (n=10, se=0)
target.cvar_neglogOmega: 18.4207 [18.4207, 18.4207] @ 95% (n=10, se=0)
target.gini: 0.8587 [0.8587, 0.8587] @ 95% (n=10, se=0)
target.js_vsU: 0.3764 [0.3764, 0.3764] @ 95% (n=10, se=0)
target.maxp: 0.2857 [0.2857, 0.2857] @ 95% (n=10, se=0)
target.success: 0.0000 [0.0000, 0.0000] @ 95% (n=10, se=0)
target.tv_vsU: 0.7130 [0.7130, 0.7130] @ 95% (n=10, se=0)
```

**eval_ci.json (parsed)**

| Metric | Mean | 95% CI | stderr | n |
| --- | ---: | ---: | ---: | ---: |
| mix.tv | 0.7130 | [0.7130, 0.7130] | 0.0000 | 10 |
| target.tv_vsU | 0.7130 | [0.7130, 0.7130] | 0.0000 | 10 |
| target.success | 0.0000 | [0.0000, 0.0000] | 0.0000 | 10 |
| mix.H | 2.4482 | [2.4482, 2.4482] | 0.0000 | 10 |
| mix.hell | 0.6903 | [0.6903, 0.6903] | 0.0000 | 10 |
| mix.js | 0.3764 | [0.3764, 0.3764] | 0.0000 | 10 |
| mix.kl_pu | 1.7107 | [1.7107, 1.7107] | 0.0000 | 10 |
| mix.l2 | 0.3536 | [0.3536, 0.3536] | 0.0000 | 10 |
| target.H | 2.4482 | [2.4482, 2.4482] | 0.0000 | 10 |
| target.KLpU | 1.7107 | [1.7107, 1.7107] | 0.0000 | 10 |
| target.cvar_neglogOmega | 18.4207 | [18.4207, 18.4207] | 0.0000 | 10 |
| target.gini | 0.8587 | [0.8587, 0.8587] | 0.0000 | 10 |
| target.js_vsU | 0.3764 | [0.3764, 0.3764] | 0.0000 | 10 |
| target.maxp | 0.2857 | [0.2857, 0.2857] | 0.0000 | 10 |

### ckpt_policy__abl_NoPE

**eval_ci.txt**

```
CI over pooled episodes — ckpt_policy__abl_NoPE — EXP (any degree)
------------------------------------------------------------------
mix.H: 1.8463 [1.8463, 1.8463] @ 95% (n=10, se=0)
mix.hell: 0.7926 [0.7926, 0.7926] @ 95% (n=10, se=0)
mix.js: 0.4848 [0.4848, 0.4848] @ 95% (n=10, se=0)
mix.kl_pu: 2.3126 [2.3126, 2.3126] @ 95% (n=10, se=0)
mix.l2: 0.4568 [0.4568, 0.4568] @ 95% (n=10, se=0)
mix.tv: 0.8336 [0.8336, 0.8336] @ 95% (n=10, se=0)
target.H: 1.8463 [1.8463, 1.8463] @ 95% (n=10, se=0)
target.KLpU: 2.3126 [2.3126, 2.3126] @ 95% (n=10, se=0)
target.cvar_neglogOmega: 18.4207 [18.4207, 18.4207] @ 95% (n=10, se=0)
target.gini: 0.9253 [0.9253, 0.9253] @ 95% (n=10, se=0)
target.js_vsU: 0.4848 [0.4848, 0.4848] @ 95% (n=10, se=0)
target.maxp: 0.3948 [0.3948, 0.3948] @ 95% (n=10, se=0)
target.success: 0.0000 [0.0000, 0.0000] @ 95% (n=10, se=0)
target.tv_vsU: 0.8336 [0.8336, 0.8336] @ 95% (n=10, se=0)
```

**eval_ci.json (parsed)**

| Metric | Mean | 95% CI | stderr | n |
| --- | ---: | ---: | ---: | ---: |
| mix.tv | 0.8336 | [0.8336, 0.8336] | 0.0000 | 10 |
| target.tv_vsU | 0.8336 | [0.8336, 0.8336] | 0.0000 | 10 |
| target.success | 0.0000 | [0.0000, 0.0000] | 0.0000 | 10 |
| mix.H | 1.8463 | [1.8463, 1.8463] | 0.0000 | 10 |
| mix.hell | 0.7926 | [0.7926, 0.7926] | 0.0000 | 10 |
| mix.js | 0.4848 | [0.4848, 0.4848] | 0.0000 | 10 |
| mix.kl_pu | 2.3126 | [2.3126, 2.3126] | 0.0000 | 10 |
| mix.l2 | 0.4568 | [0.4568, 0.4568] | 0.0000 | 10 |
| target.H | 1.8463 | [1.8463, 1.8463] | 0.0000 | 10 |
| target.KLpU | 2.3126 | [2.3126, 2.3126] | 0.0000 | 10 |
| target.cvar_neglogOmega | 18.4207 | [18.4207, 18.4207] | 0.0000 | 10 |
| target.gini | 0.9253 | [0.9253, 0.9253] | 0.0000 | 10 |
| target.js_vsU | 0.4848 | [0.4848, 0.4848] | 0.0000 | 10 |
| target.maxp | 0.3948 | [0.3948, 0.3948] | 0.0000 | 10 |

### ckpt_policy__abl_NodePermute

**eval_ci.txt**

```
CI over pooled episodes — ckpt_policy__abl_NodePermute — EXP (any degree)
-------------------------------------------------------------------------
mix.H: 1.9393 [1.9393, 1.9393] @ 95% (n=10, se=0)
mix.hell: 0.7860 [0.7860, 0.7860] @ 95% (n=10, se=0)
mix.js: 0.4759 [0.4759, 0.4759] @ 95% (n=10, se=0)
mix.kl_pu: 2.2196 [2.2196, 2.2196] @ 95% (n=10, se=0)
mix.l2: 0.4364 [0.4364, 0.4364] @ 95% (n=10, se=0)
mix.tv: 0.8297 [0.8297, 0.8297] @ 95% (n=10, se=0)
target.H: 1.9393 [1.9393, 1.9393] @ 95% (n=10, se=0)
target.KLpU: 2.2196 [2.2196, 2.2196] @ 95% (n=10, se=0)
target.cvar_neglogOmega: 2.6895 [2.6895, 2.6895] @ 95% (n=10, se=0)
target.gini: 0.9146 [0.9146, 0.9146] @ 95% (n=10, se=0)
target.js_vsU: 0.4759 [0.4759, 0.4759] @ 95% (n=10, se=0)
target.maxp: 0.3638 [0.3638, 0.3638] @ 95% (n=10, se=0)
target.success: 0.0679 [0.0679, 0.0679] @ 95% (n=10, se=0)
target.tv_vsU: 0.8297 [0.8297, 0.8297] @ 95% (n=10, se=0)
```

**eval_ci.json (parsed)**

| Metric | Mean | 95% CI | stderr | n |
| --- | ---: | ---: | ---: | ---: |
| mix.tv | 0.8297 | [0.8297, 0.8297] | 0.0000 | 10 |
| target.tv_vsU | 0.8297 | [0.8297, 0.8297] | 0.0000 | 10 |
| target.success | 0.0679 | [0.0679, 0.0679] | 0.0000 | 10 |
| mix.H | 1.9393 | [1.9393, 1.9393] | 0.0000 | 10 |
| mix.hell | 0.7860 | [0.7860, 0.7860] | 0.0000 | 10 |
| mix.js | 0.4759 | [0.4759, 0.4759] | 0.0000 | 10 |
| mix.kl_pu | 2.2196 | [2.2196, 2.2196] | 0.0000 | 10 |
| mix.l2 | 0.4364 | [0.4364, 0.4364] | 0.0000 | 10 |
| target.H | 1.9393 | [1.9393, 1.9393] | 0.0000 | 10 |
| target.KLpU | 2.2196 | [2.2196, 2.2196] | 0.0000 | 10 |
| target.cvar_neglogOmega | 2.6895 | [2.6895, 2.6895] | 0.0000 | 10 |
| target.gini | 0.9146 | [0.9146, 0.9146] | 0.0000 | 10 |
| target.js_vsU | 0.4759 | [0.4759, 0.4759] | 0.0000 | 10 |
| target.maxp | 0.3638 | [0.3638, 0.3638] | 0.0000 | 10 |

### ckpt_policy__abl_TimeFrozen

**eval_ci.txt**

```
CI over pooled episodes — ckpt_policy__abl_TimeFrozen — EXP (any degree)
------------------------------------------------------------------------
mix.H: 1.6178 [1.6178, 1.6178] @ 95% (n=10, se=0)
mix.hell: 0.8059 [0.8059, 0.8059] @ 95% (n=10, se=0)
mix.js: 0.5079 [0.5079, 0.5079] @ 95% (n=10, se=0)
mix.kl_pu: 2.5411 [2.5411, 2.5411] @ 95% (n=10, se=0)
mix.l2: 0.5018 [0.5018, 0.5018] @ 95% (n=10, se=0)
mix.tv: 0.8665 [0.8665, 0.8665] @ 95% (n=10, se=0)
target.H: 1.6178 [1.6178, 1.6178] @ 95% (n=10, se=0)
target.KLpU: 2.5411 [2.5411, 2.5411] @ 95% (n=10, se=0)
target.cvar_neglogOmega: 18.4207 [18.4207, 18.4207] @ 95% (n=10, se=0)
target.gini: 0.9397 [0.9397, 0.9397] @ 95% (n=10, se=0)
target.js_vsU: 0.5079 [0.5079, 0.5079] @ 95% (n=10, se=0)
target.maxp: 0.3723 [0.3723, 0.3723] @ 95% (n=10, se=0)
target.success: 0.0000 [0.0000, 0.0000] @ 95% (n=10, se=0)
target.tv_vsU: 0.8665 [0.8665, 0.8665] @ 95% (n=10, se=0)
```

**eval_ci.json (parsed)**

| Metric | Mean | 95% CI | stderr | n |
| --- | ---: | ---: | ---: | ---: |
| mix.tv | 0.8665 | [0.8665, 0.8665] | 0.0000 | 10 |
| target.tv_vsU | 0.8665 | [0.8665, 0.8665] | 0.0000 | 10 |
| target.success | 0.0000 | [0.0000, 0.0000] | 0.0000 | 10 |
| mix.H | 1.6178 | [1.6178, 1.6178] | 0.0000 | 10 |
| mix.hell | 0.8059 | [0.8059, 0.8059] | 0.0000 | 10 |
| mix.js | 0.5079 | [0.5079, 0.5079] | 0.0000 | 10 |
| mix.kl_pu | 2.5411 | [2.5411, 2.5411] | 0.0000 | 10 |
| mix.l2 | 0.5018 | [0.5018, 0.5018] | 0.0000 | 10 |
| target.H | 1.6178 | [1.6178, 1.6178] | 0.0000 | 10 |
| target.KLpU | 2.5411 | [2.5411, 2.5411] | 0.0000 | 10 |
| target.cvar_neglogOmega | 18.4207 | [18.4207, 18.4207] | 0.0000 | 10 |
| target.gini | 0.9397 | [0.9397, 0.9397] | 0.0000 | 10 |
| target.js_vsU | 0.5079 | [0.5079, 0.5079] | 0.0000 | 10 |
| target.maxp | 0.3723 | [0.3723, 0.3723] | 0.0000 | 10 |
