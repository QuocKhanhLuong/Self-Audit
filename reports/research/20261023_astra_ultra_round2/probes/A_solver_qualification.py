"""Finite CPU solver qualification. Research only: never modifies production code."""
from pathlib import Path
from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import product
import hashlib
import json
import platform
import subprocess
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import numpy as np
import scipy
from scipy.optimize import minimize
from self_audit.control import solve_allocation

OUT = Path(__file__).resolve().parents[1] / "evidence"
FEAS_TOL = 1e-10
OBJ_TOL = 1e-12


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_hash(folder):
    files = sorted(p for p in folder.rglob("*") if p.is_file())
    return hashlib.sha256("".join(f"{p.relative_to(folder)}:{sha(p)}\n" for p in files).encode()).hexdigest()


def dec(x):
    return Decimal.from_float(float(x))


def delta(A, b, x):
    return -b @ x + .5*np.einsum("i,gij,j->g", x, A, x)


def obj(A, b, ridge, x):
    return float(delta(A, b, x).mean()+.5*ridge*(x@x))


def verify(A, b, e, eps, ridge, x):
    """Evaluate represented float64 values with exact rational arithmetic."""
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).all():
        return dict(finite=False, declared_feasible=False)
    frac=lambda v:Fraction.from_float(float(v))
    xd = [frac(v) for v in x]
    ds = [-sum(frac(v)*z for v, z in zip(bg, xd))
              +sum(frac(ag[j,k])*xd[j]*xd[k] for j in range(len(x)) for k in range(len(x)))/2
              for ag, bg in zip(A, b)]
    excess = [d-frac(ep) for d, ep in zip(ds, eps)]
    jd = sum(ds)/len(ds)+frac(ridge)*sum(z*z for z in xd)/2
        # This narrow allowance labels roundoff only, independently of production's 1e-10.
    arithmetic_scale = max([sum(abs(frac(v)*z) for v,z in zip(bg,xd))
                                +sum(abs(frac(ag[j,k])*xd[j]*xd[k]) for j in range(len(x)) for k in range(len(x)))/2
                                +abs(frac(ep)) for ag,bg,ep in zip(A,b,eps)]+[frac(1e-300)])
    rounding_allowance = 128*frac(np.finfo(float).eps)*arithmetic_scale
    exact_at_float_point = max(excess) <= 0 and np.all(x >= 0) and np.all(x <= 1)
    roundoff_feasible = max(excess) <= rounding_allowance and np.all(x >= 0) and np.all(x <= 1)
    declared = max(excess) <= frac(FEAS_TOL) and np.all(x >= 0) and np.all(x <= 1)
    production_arithmetic = bool(np.all(e+delta(A,b,x) <= e+eps+FEAS_TOL))
    with localcontext() as ctx:
        ctx.prec=70
        readable_objective=str(Decimal(jd.numerator)/Decimal(jd.denominator))
    return dict(finite=True, declared_feasible=bool(declared), exact_feasible=bool(exact_at_float_point),
                    roundoff_feasible=bool(roundoff_feasible), original_unit_max_excess=float(max(excess)),
                    objective_delta=float(jd), objective_delta_decimal=readable_objective,
                    objective_delta_fraction=str(jd),
                    production_arithmetic_feasible=production_arithmetic,
                    improvement_over_zero=float(-jd), roundoff_allowance=float(rounding_allowance))


def ray_reference(A, b, eps, ridge, direction):
    """High-precision exact scalar formula on a fixed nonnegative box ray."""
    with localcontext() as ctx:
        ctx.prec = 70
        d = [dec(v) for v in direction]
        aa = [sum(dec(ag[j,k])*d[j]*d[k] for j in range(len(d)) for k in range(len(d))) for ag in A]
        bb = [sum(dec(bg[j])*d[j] for j in range(len(d))) for bg in b]
        if any(a < 0 for a in aa):
            return dict(available=False, reason="rounded moment matrix negative along ray")
        upper = min([Decimal(1)/z for z in d if z>0]+[Decimal(1)])
        for a, bv, ep in zip(aa, bb, eps):
            ed = dec(ep)
            if a == 0:
                if bv < 0:
                    upper = min(upper, ed/(-bv))
                continue
            root = (bv*bv+2*a*ed).sqrt()
            bound = 2*ed/(root-bv) if bv<0 else (bv+root)/a
            upper = min(upper, bound)
        denominator = sum(aa)/len(aa)+dec(ridge)*sum(z*z for z in d)
        beta = min(max((sum(bb)/len(bb))/denominator, Decimal(0)), upper)
        bf = float(beta)
        if dec(bf)>beta:
            bf = float(np.nextafter(bf, 0.))
        return dict(available=True, beta=bf, beta_decimal=str(beta),
                    alpha=(np.asarray(direction)*bf).tolist(), ridge_multiplier=float(sum(z*z for z in d)))


def normalized_solve(A, b, e, eps, ridge):
    """Equivalent objective/constraint scaling; postchecks use fixed original units."""
    m = len(b[0]); meanA=A.mean(0); meanb=b.mean(0)
    fs=max(float(np.max(np.abs(meanA))),float(np.max(np.abs(meanb))),ridge,np.finfo(float).tiny)
    cs=np.maximum.reduce([np.max(np.abs(A),axis=(1,2)),np.max(np.abs(b),axis=1),eps,
                          np.full(len(e),np.finfo(float).tiny)])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result=minimize(lambda x:obj(A,b,ridge,x)/fs,np.zeros(m),
                        jac=lambda x:(meanA@x-meanb+ridge*x)/fs, method="SLSQP", bounds=[(0.,1.)]*m,
                        constraints=[dict(type="ineq",fun=lambda x:(eps-delta(A,b,x))/cs,
                                          jac=lambda x:(b-np.einsum("gij,j->gi",A,x))/cs[:,None])],
                        options=dict(ftol=1e-12,maxiter=300))
    x=np.clip(result.x,0.,1.); v=verify(A,b,e,eps,ridge,x)
    accepted=bool(result.success and v["declared_feasible"] and v["objective_delta"]<=OBJ_TOL)
    return dict(raw_alpha=x.tolist(), raw_validation=v, success=bool(result.success),
                status=int(result.status), message=str(result.message), iterations=int(result.nit),
                objective_scale=fs, constraint_scales=cs.tolist(), accepted=accepted,
                alpha=x.tolist() if accepted else np.zeros(m).tolist(),
                warning_messages=sorted(set(str(w.message) for w in caught)))


def box_face_reference(A,b,e,eps,ridge):
    """Enumerate all 3^M box faces. A feasible box minimizer also solves the QCQP.

    This is a float64 numerical reference with recorded box KKT residual, not a formal certificate.
    """
    m=len(b[0]); H=A.mean(0)+ridge*np.eye(m); v=b.mean(0)
    scale=max(np.abs(H).max(),np.abs(v).max(),np.finfo(float).tiny)
    candidates=[]
    for state in product([-1,0,1],repeat=m):
        free=np.array(state)==0; fixed=~free; x=np.array([max(z,0) for z in state],dtype=float)
        try:
            x[free]=np.linalg.solve(H[np.ix_(free,free)]/scale,(v[free]-H[np.ix_(free,fixed)]@x[fixed])/scale)
        except np.linalg.LinAlgError:
            continue
        if np.any(x < -1e-10) or np.any(x > 1+1e-10):continue
        x=np.clip(x,0,1); g=H@x-v
        violation=np.where(x<=0,np.minimum(g,0),np.where(x>=1,np.maximum(g,0),g))
        residual=float(np.abs(violation).max()/scale)
        if residual<=1e-8:candidates.append((obj(A,b,ridge,x),x,residual))
    if not candidates:return dict(available=False)
    _,x,res=min(candidates,key=lambda z:z[0]); val=verify(A,b,e,eps,ridge,x)
    return dict(available=True,alpha=x.tolist(),box_stationarity_relative_residual=res,
                validation=val,qualifies_as_qcqp_reference=val["exact_feasible"],
                qualification="Numerical exhaustive box-face reference; constrained reference only if rational original-budget check passes exactly.")


def grid_comparator(A,b,e,eps,ridge):
    if len(b[0])!=2:return None
    axis=np.linspace(0,1,129);xx=np.array(np.meshgrid(axis,axis)).reshape(2,-1).T
    dd=-xx@b.T+.5*np.einsum("ni,gij,nj->ng",xx,A,xx)
    # Strict float64 grid test, then rational recheck; grid is only a finite feasible comparator.
    valid=np.all(dd<=eps,axis=1);values=dd.mean(1)+.5*ridge*np.sum(xx*xx,axis=1)
    order=np.flatnonzero(valid)[np.argsort(values[valid])]
    for ix in order[:32]:
        val=verify(A,b,e,eps,ridge,xx[ix])
        if val["exact_feasible"]:
            return dict(alpha=xx[ix].tolist(),validation=val,grid_points=129**2,
                        qualification="Finite feasible-grid comparator; NOT a certified/global optimum.")
    return dict(alpha=[0.,0.],validation=verify(A,b,e,eps,ridge,np.zeros(2)),grid_points=129**2)


def make_case(name,V,r,ridge=1e-6,budget=0.,family="",notes=""):
    V=np.asarray(V,float);r=np.asarray(r,float)
    A=np.einsum("gnm,gnk->gmk",V,V); b=np.einsum("gnm,gn->gm",V,r);e=.5*np.sum(r*r,axis=1)
    return dict(name=name,family=family,notes=notes,V=V,r=r,A=A,b=b,e=e,ridge=float(ridge),
                budget=np.broadcast_to(np.asarray(budget,float),e.shape).copy())


def cases():
    out=[]
    for exponent in [-12,-10,-8,-7,-6,-5,-4,-3,-2,0,1,2]:
        s=10.**exponent
        out.append(make_case(f"scalar_response_{exponent}",np.full((3,1,1),s),np.full((3,1),.7),family="scalar_fixed_ridge"))
    for exponent in range(-9,1):
        s=10.**exponent
        for fixed in [False,True]:
            out.append(make_case(f"scalar_uniform_{exponent}_fixed{fixed}",np.full((3,1,1),.4*s),np.full((3,1),.6*s),
                                 ridge=.3 if fixed else .3*s*s,family="uniform_scale_fixed_ridge" if fixed else "uniform_scale_same_problem",
                                 notes="Both V and r scaled; ridge also scales only in same_problem family."))
    for budget in [0.,1e-14,1e-8,.01]:
        out.append(make_case(f"scalar_conflict_budget{budget}",[[[.4]],[[.2]],[[.1]]],[[.6],[-.2],[.2]],budget=budget,family="scalar_budget_conflict"))
    out.append(make_case("scalar_zero_response",np.zeros((3,1,1)),np.full((3,1),.4),family="zero_response"))
    out.append(make_case("scalar_zero_budget_singleton",[[[1.]],[[1.]]],[[1.],[0.]],family="degenerate_singleton"))
    out.append(make_case("scalar_tiny_risk_tolerance",[[[1e-6]]],[[np.sqrt(2e-16)]],family="absolute_tolerance_diagnostic"))
    # B's finite-head case; class 1 response is zero in source 1, not in the entire row.
    out.append(make_case("B_finite_head_local_box_optimum",[[[0.,.025]],[[-.125,.0875]],[[-.125,.0875]]],
                         [[.75],[-.25],[-.25]],family="analytic_multivariate",
                         notes="Known local box minimizer [1,0]; finite guard is a separate experiment owned by B."))
    rng=np.random.default_rng(23023)
    for pattern in ["independent","collinear","near_collinear","opposed","redundant_group","degenerate_coordinate"]:
        base=rng.normal(size=(3,4,2))*.15;r=rng.uniform(-.25,.25,(3,4))
        if pattern=="collinear":base[:,:,1]=base[:,:,0]
        if pattern=="near_collinear":base[:,:,1]=base[:,:,0]+1e-10*base[:,:,1]
        if pattern=="opposed":base[:,:,1]=-base[:,:,0]
        if pattern=="redundant_group":base[0]=0;r[0]=0
        if pattern=="degenerate_coordinate":base[0]=0;base[0,0,0]=.3;r[0]=0
        for exponent in [-8,-6,-4,-2,0,2,3]:
            out.append(make_case(f"m2_{pattern}_{exponent}",base*10.**exponent,r,family=pattern))
    for m in [3,4]:
        for seed in range(4):
            rg=np.random.default_rng(100*m+seed); V=rg.normal(size=(3,4,m))*.15;r=rg.uniform(-.3,.3,(3,4))
            if seed==1:V[:,:,1]=V[:,:,0]
            if seed==2:V[0]=0;r[0]=0
            if seed==3:V[0]=0;V[0,0,0]=.3;r[0]=0
            for exponent in [-8,-6,-3,0]:
                out.append(make_case(f"m{m}_seed{seed}_{exponent}",V*10.**exponent,r,family=f"random_m{m}"))
    return out


def evaluate_case(c):
    A,b,e,eps,ridge=(c[k] for k in ["A","b","e","budget","ridge"]);m=len(b[0])
    original_error=None
    try:
        sol=solve_allocation(A,b,e,ridge=ridge,budget=eps,source_ids=[f"class_{j}" for j in range(m)])
        original=dict(alpha=sol.alpha.tolist(),status=sol.status,validation=verify(A,b,e,eps,ridge,sol.alpha))
    except Exception as exc:
        original_error=repr(exc);original=dict(alpha=np.zeros(m).tolist(),status="EXCEPTION",error=original_error,
                                               validation=verify(A,b,e,eps,ridge,np.zeros(m)))
    normalized=normalized_solve(A,b,e,eps,ridge)
    ray=ray_reference(A,b,eps,ridge,np.ones(m))
    if ray["available"]:ray["validation"]=verify(A,b,e,eps,ridge,np.array(ray["alpha"]))
    coordinates=[]
    for j in range(m):
        direction=np.eye(m)[j];ref=ray_reference(A,b,eps,ridge,direction)
        if ref["available"]:ref["validation"]=verify(A,b,e,eps,ridge,np.array(ref["alpha"]))
        coordinates.append(ref)
    candidates=[("zero",np.zeros(m)),("original",np.array(original["alpha"])),("normalized",np.array(normalized["alpha"]))]
    if ray["available"]:candidates.append(("aggregate_ray",np.array(ray["alpha"])))
    candidates += [(f"coordinate_{j}",np.array(z["alpha"])) for j,z in enumerate(coordinates) if z["available"]]
    checked=[(name,x,verify(A,b,e,eps,ridge,x)) for name,x in candidates]
    admissible=[z for z in checked if z[2]["declared_feasible"] and z[2]["objective_delta"]<=OBJ_TOL]
    name,x,val=min(admissible,key=lambda z:Fraction(z[2]["objective_delta_fraction"]))
    incumbent=dict(selected=name,alpha=x.tolist(),validation=val,
                   qualification="Finite-candidate incumbent with original-unit tolerance; NOT a global or finite-step guarantee.")
    assert val["declared_feasible"]
    if ray["available"] and ray["validation"]["declared_feasible"]:
        assert Fraction(val["objective_delta_fraction"])<=Fraction(ray["validation"]["objective_delta_fraction"])
    strict_admissible=[z for z in checked if z[2]["exact_feasible"] and z[2]["objective_delta"]<=0]
    sn,sx,sv=min(strict_admissible,key=lambda z:Fraction(z[2]["objective_delta_fraction"]))
    strict_incumbent=dict(selected=sn,alpha=sx.tolist(),validation=sv,
                         qualification="Finite-set incumbent under exact rational original-budget check on rounded moments; NOT global/finite-step guarantee.")
    if ray["available"] and ray["validation"]["exact_feasible"]:
        assert Fraction(sv["objective_delta_fraction"])<=Fraction(ray["validation"]["objective_delta_fraction"])
    box=box_face_reference(A,b,e,eps,ridge);grid=grid_comparator(A,b,e,eps,ridge)
    refs=[z[2]["objective_delta"] for z in checked if z[2]["exact_feasible"]]
    if box.get("qualifies_as_qcqp_reference"):refs.append(box["validation"]["objective_delta"])
    if grid:refs.append(grid["validation"]["objective_delta"])
    best=min(refs)
    for result,alpha in [(original,np.array(original["alpha"])),(normalized,np.array(normalized["alpha"]))]:
        result["returned_validation"]=verify(A,b,e,eps,ridge,alpha)
        result["gap_to_best_known_feasible_point"]=max(0.,result["returned_validation"]["objective_delta"]-best)
    result={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in c.items()}
    result.update(original=original,normalized=normalized,aggregate_ray=ray,coordinate_rays=coordinates,
                  safeguarded_incumbent=incumbent,box_reference=box,grid_comparator=grid,
                  exact_budget_incumbent=strict_incumbent,
                  best_known_feasible_objective_delta=best)
    return result


def summarize(rows):
    def counts(which):
        return dict(numerical_failure_or_fallback_count=sum(r[which].get("status") in {"EXCEPTION","SOLVER_FALLBACK_ZERO"} or r[which].get("accepted") is False for r in rows),
                    declared_infeasible_returns=sum(not r[which]["returned_validation"]["declared_feasible"] for r in rows),
                    gap_above_1e_10=sum(r[which]["gap_to_best_known_feasible_point"]>1e-10 for r in rows),
                    max_gap=max(r[which]["gap_to_best_known_feasible_point"] for r in rows))
    improved=[r["name"] for r in rows if r["safeguarded_incumbent"]["validation"]["objective_delta"]<r["original"]["returned_validation"]["objective_delta"]-1e-12]
    return dict(case_count=len(rows),scalar_cases=sum(len(r["b"][0])==1 for r in rows),original=counts("original"),normalized=counts("normalized"),
                scalar_original_coefficient_error_above_1e_3=[r["name"] for r in rows if len(r["b"][0])==1 and r["aggregate_ray"]["available"] and abs(r["original"]["alpha"][0]-r["aggregate_ray"]["beta"])>1e-3],
                scalar_normalized_coefficient_error_above_1e_3=[r["name"] for r in rows if len(r["b"][0])==1 and r["aggregate_ray"]["available"] and abs(r["normalized"]["alpha"][0]-r["aggregate_ray"]["beta"])>1e-3],
                normalized_raw_original_unit_rejections=[r["name"] for r in rows if not r["normalized"]["raw_validation"]["declared_feasible"]],
                normalized_raw_tolerance_only=[r["name"] for r in rows if r["normalized"]["raw_validation"]["declared_feasible"] and not r["normalized"]["raw_validation"]["roundoff_feasible"]],
                normalized_raw_not_exact_budget_feasible=[r["name"] for r in rows if r["normalized"]["raw_validation"]["declared_feasible"] and not r["normalized"]["raw_validation"]["exact_feasible"]],
                incumbent_improved_over_original_1e_12=improved,
                original_worse_than_ray_1e_12=[r["name"] for r in rows if r["aggregate_ray"]["available"] and r["original"]["returned_validation"]["objective_delta"]>r["aggregate_ray"]["validation"]["objective_delta"]+1e-12],
                exceptions={r["name"]:r["original"]["error"] for r in rows if r["original"]["status"]=="EXCEPTION"})


if __name__=="__main__":
    started=time.perf_counter();before=tree_hash(ROOT/"reports/research/20261023_astra_ultra")
    rows=[evaluate_case(c) for c in cases()]
    after=tree_hash(ROOT/"reports/research/20261023_astra_ultra")
    assert before==after
    head=subprocess.check_output(["rtk","proxy","git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    result=dict(kind="EXECUTED_MANUFACTURED_CPU_SOLVER_QUALIFICATION",head=head,
                canonical_source="637f38ced217fe1db2347a87924695533056d8f0",experiment_source="8960b6baac4c6600553d900c303f192c2fcfa4f0",
                command="rtk proxy python reports/research/20261023_astra_ultra_round2/probes/A_solver_qualification.py",
                script_sha256=sha(__file__),production_control_sha256=sha(ROOT/"self_audit/control.py"),
                round1_tree_sha256_before=before,round1_tree_sha256_after=after,
                python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                original_unit_feasibility_tolerance=FEAS_TOL,original_unit_objective_tolerance=OBJ_TOL,
                elapsed_seconds=time.perf_counter()-started,summary=summarize(rows),cases=rows)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"A_solver_qualification.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="cases"},indent=2,allow_nan=False))
