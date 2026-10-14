"""Replay raw anatomical evidence and the fixed linear restart operator.

The readout below is copied from the hash-pinned executed core (lines 188-288).
Only source-ledger bookkeeping and a richer return value are added. The vote
updates and their float32 accumulation order are unchanged. Sources record
algorithmic provenance, not independent observations or verified anatomy.
"""
from __future__ import annotations

import hashlib
import numpy as np
from scipy import ndimage as ndi, sparse

from .runtime import load_runtime

# Fixed before any real-bank/reference comparisons. The pinned teacher already
# returns float16; a mismatching cache does NOT earn a looser tolerance.
# Original solve error <= 1e-6; our tighter
# solve, float32 storage and anchor rounding are covered by 3e-6.
R0_ATOL = 2e-7
Q_ATOL = 3e-6
LINEAR_TOLERANCE = 1e-10
INVARIANT_ATOL = 5e-7
MAX_Z_ERROR = 1e-3


class ReproductionError(RuntimeError):
    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        super().__init__("FAIL_REPRODUCTION: " + str(diagnostics) +
                         "; verify the exact numerical environment and teacher/cache identity; "
                         "exact teacher re-extraction or a lossless feature cache may be required")


def linear_restart(p, rhs, alpha=.35, tolerance=LINEAR_TOLERANCE, max_iterations=10000):
    """K B = alpha (I-(1-alpha)P)^-1 B, without simplex normalization of B.

    Sparse P is kept sparse. B is a bounded-width dense block, e.g. [sV,s];
    callers may process source columns in blocks without constructing dense K.
    The residual certificate applies to arbitrary nonnegative finite masses in
    exact arithmetic for the measured operator. It bounds iterative truncation;
    it is not interval arithmetic for floating roundoff or a semantic guarantee.
    """
    if (not np.isfinite(alpha) or not 0 < alpha <= 1
            or not np.isfinite(tolerance) or not 0 < tolerance < 1
            or type(max_iterations) is not int or max_iterations < 1):
        raise ValueError("invalid linear restart settings")
    p = sparse.csr_matrix(p, dtype=np.float64, copy=True)
    b = rhs.toarray() if sparse.issparse(rhs) else np.asarray(rhs, dtype=np.float64)
    vector = b.ndim == 1
    if vector:
        b = b[:, None]
    if (b.ndim != 2 or min(b.shape) < 1 or p.shape != (len(b), len(b))
            or not np.isfinite(b).all() or (b < 0).any()
            or not np.isfinite(p.data).all() or (p.data < 0).any()):
        raise ValueError("finite nonnegative P and right-hand side required")
    rows = np.asarray(p.sum(axis=1)).ravel()
    if not np.allclose(rows, 1, atol=2e-6, rtol=0):
        raise ValueError("row-stochastic P required")
    # Identical one-time P rounding normalization to the original solver.
    p = (sparse.diags(1 / rows) @ p).tocsr()
    beta_norm = (1-alpha) * float(np.asarray(p.sum(1)).max())
    if beta_norm >= 1:
        raise ValueError("uncertifiable restart contraction")
    q = b.copy()
    for step in range(1, max_iterations + 1):
        q = alpha*b + (1-alpha)*(p @ q)
        residual = float(np.max(np.abs(alpha*b + (1-alpha)*(p @ q)-q)))
        if not np.isfinite(q).all() or not np.isfinite(residual):
            raise ValueError("nonfinite linear restart iterate")
        bound = residual / (1-beta_norm)
        if bound <= tolerance:
            result = q[:, 0] if vector else q
            return result, dict(converged=True, iterations=step, residual=residual,
                                error_bound=bound, tolerance=tolerance,
                                bound_kind="iterative_truncation_exact_arithmetic_no_interval_roundoff",
                                row_sum_max_error=float(np.max(np.abs(rows-1))))
    raise RuntimeError(f"LINEAR_NOT_CONVERGED: residual={residual}, bound={bound}")


def decompose(p, strength, V, *, alpha=.35, min_h=1e-6):
    """Propagate evidence masses; unsupported Z is NaN and must be masked."""
    s = np.asarray(strength, dtype=np.float64)
    v = np.asarray(V, dtype=np.float64)
    if not np.isfinite(min_h) or not 0 < min_h < 1:
        raise ValueError("min_h must be finite in (0,1)")
    if (v.shape != (*s.shape, 4) or not np.isfinite(s).all()
            or not np.isfinite(v).all() or (s < 0).any() or (s > 1).any()
            or (v < 0).any()
            or not np.allclose(v[s > 0].sum(-1), 1, atol=INVARIANT_ATOL, rtol=0)):
        raise ValueError("supported strength and soft class votes required")
    joint = np.concatenate([(s[..., None]*v).reshape(-1, 4), s.reshape(-1, 1)], axis=1)
    propagated, certificate = linear_restart(p, joint, alpha=alpha)
    M = propagated[:, :4].reshape(*s.shape, 4)
    h = propagated[:, 4].reshape(s.shape)
    err = float(np.max(np.abs(M.sum(-1)-h)))
    if (err > INVARIANT_ATOL or (h < 0).any() or (h > 1+INVARIANT_ATOL).any()):
        raise ValueError("mass conservation failed")
    Q = (1-h[..., None])*.25 + M
    # Component-wise residual certificate also bounds each M_c and h. Division
    # amplifies error near zero; certify the quotient, not merely the solver.
    error = certificate["error_bound"]
    quotient = np.zeros(M.shape, dtype=np.float64)
    np.divide(M, h[..., None], out=quotient, where=h[..., None] > 0)
    z_error_bound = np.full(h.shape, np.inf, dtype=np.float64)
    np.divide(error*(1+np.max(np.abs(quotient), axis=-1)), h-error,
              out=z_error_bound, where=h > error)
    valid = np.isfinite(h) & (h > min_h) & (z_error_bound <= MAX_Z_ERROR)
    Z = np.full(M.shape, np.nan, dtype=np.float64)
    Z[valid] = M[valid]/h[valid, None]
    return dict(M=M, h=h, Q=Q, Z=Z, valid=valid, z_error_bound=z_error_bound), {
        "linear_solver": certificate, "sum_M_minus_h_max_error": err,
        "M_error_bound": error, "h_error_bound": error, "max_Z_error_allowed": MAX_Z_ERROR,
        "Z_bound_kind": "quotient_iterative_truncation_no_interval_roundoff_or_semantic_certificate",
        "valid_Z_error_bound_max": float(z_error_bound[valid].max()) if valid.any() else None,
        "min_h": float(min_h), "valid_pixels": int(valid.sum()),
        "positive_contrast_pixels": int((valid & (np.max(Q, -1) > .25+Q_ATOL)).sum()),
        "class_conflict_pixels": int((valid & ((M > 0).sum(-1) > 1)).sum()),
    }


class _SourceLedger:
    """Sparse node/source support, recorded at the original vote update sites."""
    def __init__(self, shape, patient_id):
        self.shape, self.patient_id = shape, patient_id
        self.entries, self.nodes, self.columns, self.values = [], [], [], []

    def add(self, scale_index, k, t, z, candidate, name, cls, mask, score, parents=()):
        mask_digest = hashlib.sha256(np.packbits(mask).tobytes()).hexdigest()
        # scale_index distinguishes repeated configured k; same-mask group records
        # repetitions so no consumer can mistake them for independent evidence.
        group = f"{self.patient_id}:t{t}:z{z}:mask:{mask_digest}"
        identity = f"{self.patient_id}:scale{scale_index}:k{k}:t{t}:z{z}:{name}:{candidate}:c{cls}:{mask_digest}"
        column = len(self.entries)
        self.entries.append(dict(source_id=identity, repeated_mask_group=group,
            scale_index=int(scale_index), clusters=int(k), t=int(t), z=int(z),
            candidate=str(candidate), kind=name, class_id=int(cls),
            score=float(score), mask_sha256=mask_digest, parents=list(parents),
            independent_evidence=False))
        nodes = np.flatnonzero(mask) + (t*self.shape[1]+z)*self.shape[2]*self.shape[3]
        self.nodes.append(nodes)
        self.columns.append(np.full(len(nodes), column, dtype=np.int64))
        self.values.append(np.full(len(nodes), float(score), dtype=np.float64))

    def finish(self, strength, masses):
        shape = (int(np.prod(self.shape)), len(self.entries))
        if not self.entries:
            support = sparse.csr_matrix(shape, dtype=np.float64)
        else:
            nodes, cols, scores = map(np.concatenate, (self.nodes, self.columns, self.values))
            factor = np.zeros(masses.size, dtype=np.float64)
            np.divide(strength.ravel(), masses.ravel(), out=factor, where=masses.ravel() > 0)
            support = sparse.coo_matrix((scores*factor[nodes], (nodes, cols)), shape=shape).tocsr()
        return dict(source_support_data=support.data, source_support_indices=support.indices,
                    source_support_indptr=support.indptr,
                    source_support_shape=np.asarray(support.shape, dtype=np.int64)), self.entries


def reconstruct(image, features, left_axis, cfg, *, left_vector_yx=None,
                patient_id="anonymous", min_h=1e-6, cached_r0=None, cached_q=None,
                cached_reliability=None, cached_labels=None):
    """Use raw votes and original build_graph; never invert cached Q/reliability."""
    runtime = load_runtime()
    replay = anatomical_evidence(image, features, left_axis, cfg,
                                 left_vector_yx=left_vector_yx, patient_id=patient_id)
    p, graph = runtime.core.build_graph(image, features, cfg)
    arrays, report = decompose(p, replay["strength"], replay["V"], alpha=cfg.alpha, min_h=min_h)
    arrays.update({k: v for k, v in replay.items() if isinstance(v, np.ndarray)})
    # Direct original solve is a separate check; it only receives simplex r0.
    original, original_certificate = runtime.core.anchored_solve(
        p, replay["r0"].reshape(-1, 4), cfg.alpha, cfg.tolerance, cfg.max_iterations)
    original = original.reshape(arrays["Q"].shape)
    errors = dict(decomposition_original_q_max_error=float(np.max(np.abs(arrays["Q"]-original))),
                  r0_max_error=None, q_max_error=None, original_q_max_error=None)
    for target, actual, key in ((cached_r0, replay["r0"], "r0_max_error"),
                                (cached_q, arrays["Q"], "q_max_error"),
                                (cached_q, original, "original_q_max_error")):
        if target is not None:
            if target.shape != actual.shape or not np.isfinite(target).all():
                raise ReproductionError({"reason": "cached target shape/finiteness mismatch"})
            errors[key] = float(np.max(np.abs(actual-target)))
    support, _ = runtime.core.anchored_solve(p, np.stack([
        replay["strength"].ravel(), 1-replay["strength"].ravel()], -1),
        cfg.alpha, cfg.tolerance, cfg.max_iterations)
    original_f32 = original.astype(np.float32)
    entropy = -(original_f32*np.log(np.maximum(original_f32, 1e-8))).sum(-1)/np.log(4)
    reliability = (support[:, 0].reshape(image.shape)*(1-entropy)).clip(0, 1).astype(np.float32)
    top = np.sort(original_f32, axis=-1)
    accepted = ((top[..., -1] >= cfg.min_probability)
                & (top[..., -1]-top[..., -2] >= cfg.min_margin)
                & (reliability >= cfg.min_reliability))
    labels = np.where(accepted, original_f32.argmax(-1), 255).astype(np.uint8)
    if cached_reliability is not None:
        if cached_reliability.shape != reliability.shape or not np.isfinite(cached_reliability).all():
            raise ReproductionError({"reason": "cached reliability shape/finiteness mismatch"})
        errors["reliability_max_error"] = float(np.max(np.abs(cached_reliability-reliability)))
    if cached_labels is not None:
        if cached_labels.shape != labels.shape:
            raise ReproductionError({"reason": "cached label shape mismatch"})
        errors["hard_label_mismatches"] = int(np.count_nonzero(cached_labels != labels))
    passed = (errors["decomposition_original_q_max_error"] <= Q_ATOL
              and (errors["r0_max_error"] is None or errors["r0_max_error"] <= R0_ATOL)
              and (errors["q_max_error"] is None or errors["q_max_error"] <= Q_ATOL)
              and (errors["original_q_max_error"] is None or errors["original_q_max_error"] <= Q_ATOL)
              and errors.get("reliability_max_error", 0) <= Q_ATOL
              and errors.get("hard_label_mismatches", 0) == 0)
    report.update(graph=graph, anatomy=replay["anatomy"], original_solver=original_certificate,
                  reproduction={"status": "PASS_REPRODUCTION" if passed else "FAIL_REPRODUCTION",
                    "r0_atol": R0_ATOL, "q_atol": Q_ATOL, "rtol": 0., **errors},
                  source_count=len(replay["sources"]), evidence_independence="not_asserted")
    digest = hashlib.sha256()
    for value in (np.asarray(p.shape, dtype=np.int64), p.indptr, p.indices, p.data):
        digest.update(str(value.dtype).encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    report["graph_csr_sha256"] = digest.hexdigest()
    report["resampled_image_sha256"] = hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
    report["input_features_sha256"] = hashlib.sha256(np.ascontiguousarray(features).tobytes()).hexdigest()
    if not passed:
        raise ReproductionError(report["reproduction"])
    return arrays, report, replay["sources"]


def anatomical_evidence(image,features,left_axis,cfg,*,left_vector_yx=None,patient_id="anonymous"):
    """Experimental multiscale component hypotheses plus soft conflicting votes.

    Priors: an internally supported wall surrounds a cavity; an adjacent cavity
    on the patient's right may be RV. No image-border component can be a wall.
    Atoms have no confidence gate. Partial enclosure is scored, not repaired.
    No missing pixel is filled. These priors do not identify the heart uniquely.
    """
    core = load_runtime().core
    validate_cine, unit_features, _cluster = core.validate_cine, core.unit_features, core._cluster
    ledger = _SourceLedger(image.shape, patient_id)
    validate_cine(image,features)
    # Opt-in continuous world-left covector on BANK pixel coordinates.
    # None preserves the historical cardinal rule; zero means geometry abstains.
    if left_vector_yx is not None:
        left_vector_yx=np.asarray(left_vector_yx,dtype=np.float64)
        if left_vector_yx.shape!=(2,) or not np.isfinite(left_vector_yx).all():
            raise ValueError('finite world-left bank covector required')
    votes=np.zeros((*image.shape,4),np.float32); masses=np.zeros(image.shape,np.float32)
    diag={'wall_cavity_hypotheses':0,'rv_hypotheses':0,'slices_with_hypotheses':0,
          'no_orientation_slices':0,'candidate_cap_slices':0,'hypotheses_before_anatomy':0}
    # Frozen features plus an explicit normalized-intensity term, not semantic logits.
    descriptor=np.concatenate([unit_features(features),image[...,None]*.5],-1)
    for scale_index, k in enumerate(cfg.clusters):
        labels=_cluster(descriptor,k,cfg.kmeans_steps,cfg.max_fit_points)
        for t in range(image.shape[0]):
            for z in range(image.shape[1]):
                lab,im=labels[t,z],image[t,z]
                parts=[]
                for c in np.unique(lab):
                    cc,n=ndi.label(lab==c)
                    sizes=np.bincount(cc.ravel())
                    for j in np.flatnonzero(sizes[1:]>=cfg.min_region_pixels)+1:
                        mask=cc==j
                        parts.append((int(sizes[j]),int(np.flatnonzero(mask)[0]),mask))
                parts.sort(key=lambda x:(-x[0],x[1]))
                if len(parts)>cfg.max_region_candidates:diag['candidate_cap_slices']+=1
                parts=parts[:cfg.max_region_candidates]
                stats=[]
                for n,_,m in parts:
                    border=bool(m[0].any() or m[-1].any() or m[:,0].any() or m[:,-1].any())
                    holes=ndi.binary_fill_holes(m)&~m
                    stats.append((m,holes,border,n))
                found=False
                for wi,(wall,holes,border,wn) in enumerate(stats):
                    if border or not holes.any():continue
                    for ci,(cavity,ch,cb,cn) in enumerate(stats):
                        if ci==wi or cb or ch.any():continue
                        overlap=int((holes&cavity).sum())
                        enclosure=min(overlap/cn,overlap/max(int(holes.sum()),1))
                        if enclosure<cfg.min_enclosure:continue
                        diag['hypotheses_before_anatomy']+=1
                        interface=ndi.binary_dilation(cavity)&wall
                        if not interface.any():continue
                        contrast=abs(float(np.median(im[cavity])-np.median(im[wall])))
                        if contrast<cfg.min_contrast:continue
                        score=float(enclosure*(1-np.exp(-contrast)))
                        # The geometry already observed is evidence; do NOT fill holes to make output masks.
                        votes[t,z,wall,2]+=score;votes[t,z,cavity,3]+=score
                        masses[t,z,wall]+=score;masses[t,z,cavity]+=score
                        ledger.add(scale_index,k,t,z,f"{wi},{ci}","wall",2,wall,score)
                        ledger.add(scale_index,k,t,z,f"{wi},{ci}","cavity",3,cavity,score)
                        diag['wall_cavity_hypotheses']+=1;found=True
                        if left_vector_yx is None:
                            if left_axis not in ('+x','-x','+y','-y'):
                                diag['no_orientation_slices']+=1;continue
                            dim=1 if left_axis.endswith('x') else 0
                            sign=1 if left_axis[0]=='+' else -1
                            left_position=lambda mask: sign*np.nonzero(mask)[dim].mean()
                        else:
                            if not np.any(left_vector_yx):
                                diag['no_orientation_slices']+=1;continue
                            left_position=lambda mask: float(left_vector_yx @ np.argwhere(mask).mean(0))
                        lvpos=left_position(cavity)
                        for ri,(rv,rh,rb,rn) in enumerate(stats):
                            if ri in (wi,ci) or rb or rh.any():continue
                            if not (ndi.binary_dilation(wall)&rv).any():continue
                            if left_position(rv)>=lvpos:continue
                            # Require input support for blood-pool appearance; orientation alone is not a label.
                            gap=abs(float(np.median(im[rv])-np.median(im[cavity])))
                            if gap>max(contrast*.5,.1):continue
                            rs=score*np.exp(-gap/max(contrast,1e-6))
                            votes[t,z,rv,1]+=rs;masses[t,z,rv]+=rs
                            ledger.add(scale_index,k,t,z,f"{wi},{ci},{ri}","rv",1,rv,rs,parents=(wi,ci))
                            diag['rv_hypotheses']+=1
                if found:
                    diag['slices_with_hypotheses']+=1
                    # Border evidence is permitted only in slices with foreground hypotheses.
                    for bi,(m,_,border,_) in enumerate(stats):
                        if border: votes[t,z,m,0]+=.5;masses[t,z,m]+=.5
                        if border: ledger.add(scale_index,k,t,z,bi,"border",0,m,.5)
    r0=np.full(votes.shape,.25,np.float32)
    supported=masses>0
    r0[supported]=votes[supported]/masses[supported,None]
    strength=1-np.exp(-masses)
    r0=strength[...,None]*r0+(1-strength[...,None])*.25
    diag['anchor_supported_pixels']=int(supported.sum())
    if left_vector_yx is not None:
        diag.update(orientation_policy='world_left_v2',
            raw_vote_mass_per_class=votes.sum(axis=(0,1,2,3),dtype=np.float64).tolist(),
            conflicting_vote_pixels=int(((votes>0).sum(-1)>1).sum()),
            evidence_independence='not_asserted_multiscale_votes_can_repeat')
    V = np.zeros(votes.shape, dtype=np.float64)
    np.divide(votes, masses[...,None], out=V, where=masses[...,None]>0)
    support, sources = ledger.finish(strength, masses)
    return dict(r0=r0, strength=strength, raw_votes=votes, masses=masses, V=V,
                anatomy=diag, sources=sources, **support)

