# W2.2 interim review — fp16 numerical blocker

Independent probe on draft typed APIs:

```python
p = torch.tensor([1., 0., 0., 0.], dtype=torch.float16).reshape(1,4,1,1).requires_grad_()
h = entropy_from_probabilities(p)
h.sum().backward()
# h = NaN; p.grad not all finite
x = torch.tensor([65504., -65504., 0., 0.], dtype=torch.float16).reshape(1,4,1,1)
entropy_from_logits(x)  # NaN despite finite logits
```

float32 and bfloat16 one-hot probes produced finite outputs/gradients.
Root causes: 1e-8 clamp epsilon underflows in fp16; finite fp16 logit difference can
overflow log_softmax to -Inf and 0 * -Inf becomes NaN.

Use stable computation precision (e.g. fp32 for fp16/bfloat16 inputs), preserve gradient
and output dtype contract explicitly. No changes to unrelated training probability callers.
Add regression for valid one-hot fp16 output/gradient and finite extreme fp16 logits;
also keep all required invariance/integration tests. Re-run full suite/compile.
This is part of numerical correctness of newly introduced entropy APIs, not architecture.

Use only repository files/public tools; no provider/account/runtime-directory searches.

## Final numerical review after revision — REVISE

Root full suite 179 passed, 1 warning in 6.04s. However, explicit logits
`[NaN,0,0,0]` or `[+Inf,0,0,0]` now return finite -0 entropy with nonfinite gradients
because nan_to_num/where hides invalid inputs. Add explicit finite-input rejection
BEFORE the C<=1 shortcut. Regression NaN/+Inf/-Inf must raise ValueError.
Keep stable finite-extreme computations, original output dtype and all invariants.
No other scope expansion. Coordinator message msg_201822b8f64c was sent before receipt
msg_cc5a247cac2e, but this correction was not included in that 179-test receipt.
