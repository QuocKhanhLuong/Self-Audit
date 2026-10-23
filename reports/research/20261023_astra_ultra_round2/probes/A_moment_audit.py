"""Exact rational audit of saved rounded moments; does not rerun or change solvers."""
from pathlib import Path
from fractions import Fraction as Q
from itertools import combinations
import hashlib,json

BASE=Path(__file__).resolve().parents[1]
INPUT=BASE/"evidence/A_solver_qualification.json"

def determinant(a):
    if len(a)==1:return a[0][0]
    return sum(((-1)**j)*a[0][j]*determinant([r[:j]+r[j+1:] for r in a[1:]]) for j in range(len(a)))

rows=[]
for case in json.loads(INPUT.read_text())["cases"]:
    matrices=[[[Q.from_float(x) for x in row] for row in group] for group in case["A"]]
    n=len(matrices[0]);bad=[]
    for gi,a in enumerate(matrices):
        assert all(a[j][k]==a[k][j] for j in range(n) for k in range(n))
        for size in range(1,n+1):
            for ix in combinations(range(n),size):
                d=determinant([[a[j][k] for k in ix] for j in ix])
                if d<0:bad.append(dict(group=gi,indices=list(ix),determinant_fraction=str(d),determinant_float=float(d)))
    h=[[sum(a[j][k] for a in matrices)/len(matrices)+(Q.from_float(case["ridge"]) if j==k else 0)
        for k in range(n)] for j in range(n)]
    leading=[determinant([row[:size] for row in h[:size]]) for size in range(1,n+1)]
    rows.append(dict(name=case["name"],rounded_group_negative_principal_minors=bad,
                     objective_hessian_exact_positive_definite=all(d>0 for d in leading),
                     objective_hessian_leading_minors=[str(d) for d in leading]))
result=dict(kind="EXACT_RATIONAL_AUDIT_OF_FLOAT64_MOMENT_VALUES",input_sha256=hashlib.sha256(INPUT.read_bytes()).hexdigest(),
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            negative_minor_count=sum(len(r["rounded_group_negative_principal_minors"]) for r in rows),
            affected_cases=[r["name"] for r in rows if r["rounded_group_negative_principal_minors"]],
            objective_hessians_positive_definite=sum(r["objective_hessian_exact_positive_definite"] for r in rows),
            qualification="Constraint Gram rounding can lose exact PSD; objective ridge can still make the represented objective positive definite. This does not certify numerical optimizer accuracy.",cases=rows)
(BASE/"evidence/A_moment_audit.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({k:v for k,v in result.items() if k!="cases"},indent=2))
