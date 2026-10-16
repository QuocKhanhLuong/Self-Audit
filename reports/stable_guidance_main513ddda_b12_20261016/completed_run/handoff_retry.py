"""Reuse a completed immutable bank, then run the unchanged main CLI at batch12."""
from pathlib import Path
import fcntl,hashlib,json,os,shlex,signal,shutil,subprocess,sys,time,traceback
JOB=Path(__file__).resolve().parent
M=json.loads((JOB/'LAUNCH_MANIFEST.json').read_text());OLD=Path(M['previous_job'])
ROOT=JOB.parents[1]
def read(p):return json.loads(p.read_text())
def write(p,v):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def status(**kw):
    state=dict(status='WAITING_FOR_FROZEN_BANK',stage='bank_handoff',batch_size=12,epochs=50,updated_at=time.time(),**kw)
    write(JOB/'STATUS.json',state)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def processes():
    found=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            args=(p/'cmdline').read_bytes().split(b'\0')
            if not args or Path(args[0].decode()).name not in {'python','python3'}:continue
            if any(arg.startswith(str(OLD/'source').encode()) or arg==str(OLD/'worker.py').encode() for arg in args):found.append(int(p.name))
        except (OSError,UnicodeError):pass
    return found
try:
    lock=(JOB/'handoff.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    print('[HANDOFF_RETRY] Complete bank retained; batch8 partial epoch was already interrupted and recorded',flush=True)
    last_update=0
    while True:
        try:
            bank=read(OLD/'run/bank/FROZEN.json')
            if bank.get('role')=='guidance_bank' and len(bank.get('patients',[]))==80:break
        except (FileNotFoundError,json.JSONDecodeError):pass
        if time.time()-last_update>10:
            status(completed_bank_patients=len(list((OLD/'run/bank').glob('*.npz'))));last_update=time.time()
            if read(OLD/'STATUS.json').get('status') in {'FAILED','INTERRUPTED','FINISHED'}:raise RuntimeError('Original run stopped before completed bank')
        time.sleep(.2)
    assert not processes(), 'Original process tree still alive'
    stopped_at=read(OLD/'STATUS.json').get('updated_at')
    training_lines=[s for s in (OLD/'launcher.log').read_text().splitlines() if s.startswith('[GUIDANCE_METRIC]')]
    completed_epochs=[s for s in (OLD/'launcher.log').read_text().splitlines() if s.startswith('[STUDENT]')]
    assert not completed_epochs and not (OLD/'run/student/last.pt').exists()
    last_logged=json.loads(training_lines[-1].partition('] ')[2]) if training_lines else None
    write(OLD/'BATCH_HANDOFF.json',dict(reason='User-requested VRAM optimization',superseded_by=str(JOB),
          original_batch_size=8,new_batch_size=12,completed_bank_patients=80,
          last_logged_training_update=last_logged,actual_final_update_count='not known beyond sampled logs',
          completed_epochs=0,partial_student_weights_not_saved=True,new_student_initialization='fresh seed42',stopped_at=stopped_at))
    sys.path.insert(0,str(JOB/'source/src'))
    from self_audit_guidance.io import verify,read_splits
    bank=verify(OLD/'run/bank','guidance_bank');recipe=read(OLD/'run/RECIPE.json')
    assert bank['locked_splits']==recipe['splits']==read_splits(OLD/'SPLIT.json')
    assert bank['config']==recipe['config'] and bank['seed']==M['seed']
    current={p.name:sha(p) for p in (JOB/'source/src/self_audit_guidance').glob('*.py')}
    assert bank['source_sha256']==recipe['source_sha256']==current
    assert recipe['batch_size']==8 and recipe['epochs']==M['epochs']
    original_recipe_sha=sha(OLD/'run/RECIPE.json')
    recipe['batch_size']=M['batch_size']
    dest=JOB/'run';dest.mkdir()
    # This is a new recipe. The full CLI will compare it with its own argument-
    # derived recipe and rerun real preflight before accepting the cached bank.
    write(dest/'RECIPE.json',recipe)
    shutil.copyfile(OLD/'run/EFFECTIVE_SPLIT.json',dest/'EFFECTIVE_SPLIT.json')
    shutil.copytree(OLD/'run/bank',dest/'bank',copy_function=os.link)
    assert sha(dest/'bank/FROZEN.json')==sha(OLD/'run/bank/FROZEN.json')
    write(JOB/'BANK_REUSE.json',dict(status='VERIFIED_COMPLETE_BANK_REUSED',source=str(OLD/'run/bank'),
          manifest_sha256=sha(dest/'bank/FROZEN.json'),source_recipe_sha256=original_recipe_sha,
          new_recipe_sha256=sha(dest/'RECIPE.json'),recipe_changed_fields=['batch_size'],
          storage='hardlinks to immutable hash-verified files; neither copy may be modified',
          teacher_tile_batch_used=1,student_batch_size=12,no_prior_student_updates=False,parent_last_logged_update=last_logged,parent_completed_epochs=0,fresh_student_initialization=True,verified_at=time.time()))
    write(JOB/'STATUS.json',dict(status='STARTING',stage='batch12_preflight',batch_size=12,epochs=50))
    (ROOT/'runs/LATEST_STABLE_GUIDANCE.txt').write_text(str(JOB)+'\n')
    commands={1:shlex.join(['tail','-n','80','-F',str(JOB/'launcher.log')]),
              4:shlex.join(['watch','-n','5','-t',M['python'],str(JOB/'dashboard.py')]),
              5:shlex.join(['watch','-n','10','-t',M['python'],str(JOB/'dashboard.py'),'metrics'])}
    for window,cmd in commands.items():
        target=f'LQK:{window}.0'
        current=subprocess.check_output(['tmux','display-message','-p','-t',target,'#{pane_current_command}'],text=True).strip()
        if current in {'tail','watch'}:subprocess.run(['tmux','respawn-pane','-k','-t',target,cmd],check=True)
    print('[HANDOFF] Complete bank preserved; original main CLI starts batch12 with mandatory preflight',flush=True)
    os.execv(M['python'],[M['python'],'-u',str(JOB/'worker.py')])
except BaseException as exc:
    write(JOB/'STATUS.json',dict(status='HANDOFF_FAILED',error=type(exc).__name__+': '+str(exc),updated_at=time.time()))
    traceback.print_exc();raise
