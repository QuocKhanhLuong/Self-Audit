# Selected diagnostic model

Status: PROPOSED specification. Actual code conformity is independently checked in 08 and 09.

For x∈R^{B×3×H×W}, an encoder produces Fθ(x)∈R^{B×16×h×w}, h≈H/4. There are C=4 classes. A coarse seed is followed by a dynamic read and residual writer to produce A0∈R^{B×4×H×W}. The initial annotation therefore already contains dynamic reading.

## Evidence support and annotation

At each feature-grid query i, Wt(i)={c_{t,i,k}}_{k=1}^K with K=8, c∈[−1,1]^2. Reuse the existing normalized center/radius/orientation/residual-point generator and ordinary QKV reader. With bilinear samples v(F,c),

    r_t(i) = Σ_k softmax_k(q(F_i)^T k(F,c_{t,i,k}) / sqrt(d)) v(F,c_{t,i,k}).
    W_0 = Gω(F, [softmax(A_seed), 0]).
    A_0 = A_seed + Uθ(F,r_0).
    W_1 = Gω(F, [softmax(A_0), stopgrad(g)]).
    A_audit = A_0 + Uθ(F,r_1).
    A_skip = A_0.

The same reader and writer weights are reused. Guidance conditions coordinate generation only. Holding coordinates fixed must eliminate the effect of changing g on the output. Actor probabilities/features remain differentiable inside the actor; the guidance and trigger observations are detached. The evidence is cached image-derived features, not newly acquired MRI measurements. No claim of sensor-level active acquisition is made.

## Explicit reference-free functional

Frozen anchor Z(x) contains central-slice intensity, local average and two Sobel responses, normalized per image. Let b_i=argmax_c A_{ci}, and μ_c be the mean Z over pixels assigned to c (empty classes handled safely). Let v_j denote spatial variance of anchor channel j.

    r_RF(i;x,A) = mean_j [(Z_ij−μ_{b_i,j})²/(v_j+10^−4)]
                 + 0.05 · incident edge-weighted label-boundary penalty.
    Q_RF(x,A) = −mean_i r_RF(i;x,A).
    ΔQ_RF(x,A0,A') = Q_RF(x,A')−Q_RF(x,A0).

The boundary term weights horizontal/vertical class changes by exp(−mean_j adjacent feature-difference²). This is a classical image-region compatibility energy, not ΔDice, anatomical correctness or a novel quality measure. It is invariant under a global permutation of class labels. Its unidentifiable semantic assignments are an intentional red-team fixture.

The learned Auditor is g=qφ(Z(x),stopgrad(softmax(A0)))≥0. It approximates the spatial intrinsic target:

    L_RF(φ) = mean_i [qφ(O)_i − stopgrad(r_RF(O)_i)]².

There is no assumption that named corruptions are harmful. No GT-derived corruption labels, reference masks or learned GT-trained quality regressor enter this loss. The fixed teacher quality Q_RF supplies twin targets; the learned qφ supplies guidance. These are explicitly different functions. The cheap teacher may make distillation unnecessary; direct-energy guidance is a necessary future simplification control if the signal survives.

## Value trigger and counterfactual exploration

Cheap state s contains detached pooled actor features, mean class probabilities, entropy and support statistics. It excludes the full audit output, so deciding to skip does not secretly invoke the Auditor. On independently sampled exploration states, both paths use identical A0 and parameter versions, without gradients:

    v_hat = Q_RF(x,A_audit) − Q_RF(x,A_skip) − c,  c=0.001 intrinsic units.
    L_V(ψ) = E_replay [(Vψ(stopgrad(s))−stopgrad(v_hat))²].

The cost is a declared proxy penalty, not a learned number of seconds. Wall-clock accounting is separate. No claim of calibrated expected Dice utility is made. With a minibatch cap b=floor(Bρ), select at most b positive Vψ values, stable ties by index. This maximizes the predicted sum under that particular cap. It is a batch-allocation policy, not independent Bernoulli decisions. At B=1,ρ=0.5, floor rounding gives zero eligible actions; use the declared B=8 contract for this diagnostic. A deployment policy for variable batch sizes is not established.

Exploration probability ε=0.25 invokes diagnostic audit twins irrespective of the retained decision; it does not force the explored candidate to become the final annotation. Thus the system can discover positive/negative RF effects even when the retained trigger never audits. Count both policy and exploration calls. A fixed upper budget prevents always-audit, but does not prove useful allocation. Negative values and cost can legitimately yield zero retained calls; semantic usefulness still requires evaluation.

With actual paired twins, both potential outputs are observed for a probed state; no inverse-propensity correction across its two actions is needed. Random state sampling is needed to represent the intended state population. Uniform expiring replay avoids priority-induced weighting, but stale actor/q versions still create drift. Values older than eight actor steps expire; this limits rather than proves removal of off-policy bias.

## Two timescales and firewall

Every actor step minimizes CE + foreground soft Dice on the retained annotation, one loss per sample. GT is used here. Every four steps, if replay has entries, update φ on detached observations and ψ on measured RF value (ψ only for the learned policy). Actor, Auditor and Trigger have disjoint parameters/optimizers. No actor objective maximizes Q_RF or qφ. Anchor filters never receive updates.

    ∂L_seg/∂φ = ∂L_seg/∂ψ = 0.
    ∂L_RF/∂θ = ∂L_V/∂θ = 0.

These remove direct cross-loss gradients; they do not guarantee absence of statistical co-adaptation or proxy exploitation through selected training paths. Alternative target auditors, bilevel optimization and prioritized replay are not justified before this simpler firewall is tested.

The minimal system is one optional recurrent transition. Extending to many turns changes value targets, compounding cost, stale replay and stopping; it requires a separate experiment.
