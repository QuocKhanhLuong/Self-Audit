"""Independent exact-rational checks of A's saved represented-point feasibility."""
from pathlib import Path
from fractions import Fraction
import importlib.util,json,hashlib,itertools
import numpy as np
base=Path(__file__).parents[1];evidence=base/'evidence'
a=json.loads((evidence/'A_solver_qualification.json').read_text())
def f(x):return Fraction.from_float(float(x))
def check(case,x):
 x=list(map(f,x)); A=case['A'];b=case['b'];eps=case['budget'];m=len(x)
 d=[-sum(f(bg[j])*x[j] for j in range(m))+sum(f(ag[j][k])*x[j]*x[k] for j in range(m) for k in range(m))/2 for ag,bg in zip(A,b)]
 excess=max(v-f(ep) for v,ep in zip(d,eps));box=all(0<=v<=1 for v in x)
 obj=sum(d)/len(d)+f(case['ridge'])*sum(v*v for v in x)/2
 return {'exact_feasible':box and excess<=0,'declared_feasible':box and excess<=f(a['original_unit_feasibility_tolerance']),'max_excess':float(excess),'objective':float(obj)}
count=0;disagreements=[];summary={};examples={}
for case in a['cases']:
 points=[('original',case['original']['alpha'],case['original']['returned_validation']),('normalized_returned',case['normalized']['alpha'],case['normalized']['returned_validation']),('normalized_raw',case['normalized']['raw_alpha'],case['normalized']['raw_validation']),('tolerant_incumbent',case['safeguarded_incumbent']['alpha'],case['safeguarded_incumbent']['validation']),('strict_incumbent',case['exact_budget_incumbent']['alpha'],case['exact_budget_incumbent']['validation'])]
 for name in ['aggregate_ray','box_reference','grid_comparator']:
  z=case.get(name)
  if z and 'alpha' in z:points.append((name,z['alpha'],z['validation']))
 for i,z in enumerate(case['coordinate_rays']):
  if z.get('available'):points.append((f'coordinate_{i}',z['alpha'],z['validation']))
 for name,x,validation in points:
  actual=check(case,x);count+=1
  for key in ['exact_feasible','declared_feasible']:
   if actual[key]!=validation[key]:disagreements.append({'case':case['name'],'point':name,'key':key,'rational':actual[key],'saved_validator':validation[key]})
  if name=='normalized_raw':
   summary['normalized_raw_tolerance_only']=summary.get('normalized_raw_tolerance_only',0)+int(actual['declared_feasible'] and not actual['exact_feasible'])
   summary['normalized_raw_declared_rejections']=summary.get('normalized_raw_declared_rejections',0)+int(not actual['declared_feasible'])
  if case['name'] in ['scalar_response_-6','m2_near_collinear_3','scalar_zero_budget_singleton']:
   examples.setdefault(case['name'],{})[name]=actual
# Exact PSD diagnostic on rounded Gram entries; no theorem about pre-rounding V/r.
def determinant(mat):
 if len(mat)==1:return mat[0][0]
 return sum((-1)**j*mat[0][j]*determinant([row[:j]+row[j+1:] for row in mat[1:]]) for j in range(len(mat)))
negative_minors=[]
for case in a['cases']:
 for g,gram in enumerate(case['A']):
  for size in range(1,len(gram)+1):
   for ix in itertools.combinations(range(len(gram)),size):
    minor=determinant([[f(gram[i][j]) for j in ix] for i in ix])
    if minor<0:negative_minors.append({'case':case['name'],'group':g,'indices':ix,'negative_principal_minor':float(minor)})
(evidence/'F_A_rounded_gram_check.json').write_text(json.dumps({'kind':'EXACT_RATIONAL_ROUNDED_GRAM_PSD_DIAGNOSTIC','negative_minor_count':len(negative_minors),'case_count':len(set(x['case'] for x in negative_minors)),'negative_minors':negative_minors,'scope':'Float64-formed Gram matrices can lose exact PSD under rounding. This does not negate feasibility comparisons against concrete exact-budget points; no patient or model-risk claim.'},indent=2)+'\n')
# Read-only import of A functions; execute3 decisive cases, redirecting no A-owned artifact.
path=base/'probes/A_solver_qualification.py';spec=importlib.util.spec_from_file_location('round2_A_reviewed',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
reruns={}
for case in mod.cases():
 if case['name'] in examples:
  r=mod.evaluate_case(case)
  reruns[case['name']]={'original_alpha':r['original']['alpha'],'normalized_raw_alpha':r['normalized']['raw_alpha'],'normalized_accepted':r['normalized']['accepted'],'normalized_returned_alpha':r['normalized']['alpha'],'normalized_max_excess':r['normalized']['raw_validation']['original_unit_max_excess'],'aggregate_ray_alpha':r['aggregate_ray']['alpha']}
assert not disagreements
assert summary['normalized_raw_tolerance_only']==17
assert summary['normalized_raw_declared_rejections']==1
assert len(reruns)==3 and not reruns['m2_near_collinear_3']['normalized_accepted']
result={'kind':'INDEPENDENT_EXACT_RATIONAL_FEASIBILITY_OF_REPRESENTED_FLOAT_POINTS','A_artifact_sha256':hashlib.sha256((evidence/'A_solver_qualification.json').read_bytes()).hexdigest(),'A_script_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'cases':len(a['cases']),'represented_points_checked':count,'saved_validator_boolean_disagreements':disagreements,'summary':summary,'decisive_examples':examples,'read_only_focused_reruns':reruns,'limitations':['Fraction proves signs only for the saved rounded float64 coefficients and represented points, not pre-rounding V/r or true model risk.','No optimality certificate or patient inference.','The initial Decimal70 validator and revised rational validator both agreed on this finite set; fixed decimal precision is not exact arithmetic in general.']}
(evidence/'F_A_rational_validation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='decisive_examples'},indent=2))
