"""Gated orchestration. Preparation is runnable now; new semantic/model claims require approval.
Each GPU stage is an isolated process. Outcomes distinguish blocked, paused, failed and completed.
"""
from __future__ import annotations
import copy,json,os,sys,time,traceback,subprocess,shutil,re
from pathlib import Path
from collections import Counter
import numpy as np
from odij_core import *
from odij_data import *


def nonfinal_rows(root):
    return flatten([c for s in SPLITS if s!='final' for c in read_json(root/'data'/f'{s}.json')])

def historical_readout(root):
    s=read_json(root/'source'/'paste_back_summary.json')
    out={'status':'historical_only','selection_unchanged':s['recovery']['selected_profile'],
         'source_run':SOURCE_RUN,'results':s['results']['eval_qwen4b_strict']['selected_metrics'],
         'cross_mode':s['cross_mode'],'interpretation':'Completed H final data are now historical/regression evidence. These values do not select a new default.'}
    write_json(root/'history'/'H_summary_for_protocol_design.json',out)


def resolve_revisions(intake):
    from huggingface_hub import HfApi
    p=read_json(intake/'protocol.json')
    if p['approval'].get('approved'):return {'status':'not_changed','reason':'An approved protocol is immutable; unresolved revisions will block it'}
    api=HfApi(token=os.environ.get('HF_TOKEN'));changes=[]
    for m in p['models']:
        if m['revision']:continue
        info=api.model_info(m['id']);sha=info.sha
        if not re.fullmatch('[0-9a-f]{40}',sha or ''):raise IntegrityError('Hub did not resolve immutable revision')
        m['revision']=sha;changes.append({'model':m['id'],'revision':sha})
    write_json(intake/'protocol.json',p)
    prepare_review_template(intake)
    return {'status':'resolved','models':changes,'next_action':'Review the now-pinned protocol and refresh/copy review_template.refreshed.json to review.json before signing.'}


def export_contracts(root):
    native={'schema':'opendecision-native-study-response/v1','not_jev_conformance':True,
        'fields':{'option_probabilities':'map of caller IDs to unconditional probabilities','none_probability':'separate mass for native Choice only, null for Noul/Score','selected_id':'caller option ID or null when native none wins','top_probability':'maximum probability, NOT vendor confidence','noul':'P(yes) only for registered binary tasks','score':'expectation under explicitly supplied numeric levels','engine_profile':'immutable experiment/model identity'},
        'rules':['Option IDs never enter model text; labels and criteria must be present.','No hidden renormalization or injected caller option.','No automatic authorization/irreversible action.','Unsupported primitives/shapes and overlength inputs are errors.']}
    write_json(root/'contracts'/'native_probability_contract.json',native)
    atomic_bytes(root/'contracts'/'SERVICE_HANDOFF.md',b'''# Track S handoff (not Rust service completion)

The included `odij_serve.py` is a resident, bounded-line JSONL prediction worker for the new native study contract.
It loads one locked profile once. It is NOT the pinned Jev HTTP format and does not implement Rust transports or authentication.
The host must supply a bounded queue, tenant authorization, cancellation/admission, and its versioned compatibility adapter.
A deadline is checked before/after inference; an already-dispatched CUDA operation is not preempted.
Use the existing repository's exact wire types and fixtures before claiming S.2/S.3 complete. No public tunnel is opened.
''')


def stage_name(arm,layout,seed):return arm+'__'+layout+'__seed'+str(seed)

def jobs_for(p):
    jobs=[]
    for arm in p['arms']:
        if arm=='lexical':jobs.append({'arm':arm,'layout':'native','seed':p['seeds'][0]});continue
        enc=arm.startswith('modernbert');finite='_finite_' in arm
        layouts=['joint'] if enc else ['finite'] if finite else p['renderers']
        seeds=p['seeds'] if not arm.endswith('finite_frozen') else [p['seeds'][0]]
        for layout in layouts:
            for seed in seeds:jobs.append({'arm':arm,'layout':layout,'seed':seed})
    return jobs

def arm_spec(p,arm):
    key=next((m['key'] for m in p['models'] if arm.startswith(m['key']+'_')),None)
    if key is None:raise GateBlocked('No declared backend for '+arm)
    return next(m for m in p['models'] if m['key']==key)


def feature_rows(rt,rows,layout,cfg,start):
    xs=[]
    for i,r in enumerate(rows):
        xs.append(rt.row_features(r,layout))
        if (i+1)%cfg.checkpoint_steps==0:
            rt.store.backup();print(f'Feature rows {i+1}/{len(rows)}',flush=True)
            if time.monotonic()-start>cfg.worker_budget_minutes*60:raise BudgetStop('Feature cache saved; resume without re-encoding verified rows')
    rt.store.backup();return xs


def sample_benchmark(predict,rows,repeats=3,limit=4):
    from odij_runtime import sync,memory
    from odij_serve import native_response
    import torch
    before=memory()
    if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
    def complete(r):
        values=np.asarray(predict(r));return canonical(native_response(r,values))
    selected=[];seen=set()
    for r in rows:
        if r['split']!='policy_dev' or r['group'] in seen:continue
        selected.append(r);seen.add(r['group'])
        if len(selected)>=limit:break
    ts=[];raw=[]
    for r in selected:
        complete(r);sync()
        for _ in range(repeats):
            sync();t=time.perf_counter();complete(r);sync();dt=(time.perf_counter()-t)*1000;ts.append(dt);raw.append({'id':r['id'],'milliseconds':dt})
    return {'scope':'Q=1 tokenization, model, head, probabilities and native answer JSON; resident model; no policy application, input parsing, HTTP, queue or startup','requests':len(selected),'repeats':repeats,'p50_ms':float(np.median(ts)),'p95_ms':float(np.quantile(ts,.95)),'raw':raw,'memory':memory(),'pre_benchmark_memory':before} if ts else {'status':'not_measured'}


def basic_lexical(r):
    from collections import Counter
    def vector(t):
        t=' '+re.sub(r'\s+',' ',t.casefold())+' ';return Counter(t[i:i+3] for i in range(max(0,len(t)-2)))
    def sim(a,b):
        den=math.sqrt(sum(v*v for v in a.values())*sum(v*v for v in b.values()));return sum(v*b.get(k,0) for k,v in a.items())/den if den else 0.
    a=vector(r['state']+' '+r['q']['instruction']);z=[10*sim(a,vector(option_text(o))) for o in r['q']['options']]
    if r['q']['primitive']=='choice':z.append(0.)
    return np.array(z)


def run_arm(root,cfg,job):
    import torch
    from safetensors.torch import load_file,save_file
    from odij_learning import fit_frozen_variants,load_decision,package_prediction,softmax,Scorer,output_row
    from odij_runtime import Runtime,memory
    from odij_training import online_train
    assert_study_intact(root)
    p=read_json(root/'registered_protocol.json');rows=nonfinal_rows(root);arm,layout,seed=job['arm'],job['layout'],job['seed'];out=root/'arms'/stage_name(arm,layout,seed);out.mkdir(parents=True,exist_ok=True);start=time.monotonic();rt=None
    completion=out/'ARM_DONE.json'
    if completion.exists():
        m=read_json(completion)
        for n,h in m['files'].items():
            if file_sha(out/n)!=h:raise IntegrityError('Completed arm artifact changed')
        return m['profiles']
    identity=digest({'protocol':p,'data':digest(rows),'job':job,'code':{q.name:file_sha(q) for q in Path(__file__).parent.glob('odij_*.py')},'environment':env_identity()})
    if (out/'ARM_IDENTITY.json').exists() and read_json(out/'ARM_IDENTITY.json')['identity']!=identity:raise IntegrityError('Arm resume identity changed')
    write_json(out/'ARM_IDENTITY.json',{'identity':identity,'job':job})
    profiles=[]
    if arm=='lexical':
        logits=[basic_lexical(r) for r in rows];summary,pred=package_prediction(logits,rows,p)
        meta={'name':'lexical','method':'lexical','seed':seed,'layout':'native',**summary,'benchmark':sample_benchmark(lambda r:softmax(basic_lexical(r),summary['calibration']['selected']),rows,p['benchmark']['repeats'])}
        write_json(out/'lexical.json',meta);write_json(out/'lexical_nonfinal_predictions.json',pred)
        profiles=[dict(meta,artifact_dir=out.relative_to(root).as_posix(),head_file=None)]
    else:
        spec=arm_spec(p,arm);training=arm.endswith(('_online','_lora','_full'))
        if not cfg.allow_cpu_test and p['target_device'].casefold() not in (torch.cuda.get_device_name(0).casefold() if torch.cuda.is_available() else ''):raise GateBlocked('This GPU does not match the registered target_device')
        if not cfg.allow_cpu_test and training and torch.cuda.get_device_properties(0).total_memory/2**30<32 and (arm.endswith('_lora') and spec['key']=='qwen4b' or arm.endswith('_online')):
            raise GateBlocked('Matched 4B online/LoRA pair is preregistered as >=32 GiB total GPU memory; preserve FP32, no silent lower-precision substitute')
        rt=Runtime(spec,out,p,training=training,allow_cpu_test=cfg.allow_cpu_test)
        rt.share_feature_store(root)
        try:
            if arm.endswith('_frozen') and '_finite_' not in arm:
                xs=feature_rows(rt,rows,layout,cfg,start);rec=fit_frozen_variants(xs,rows,p,out,seed)
                for m in rec:
                    d=load_decision(out/(m['name']+'.safetensors'),m)
                    def predict(r,d=d):
                        with torch.no_grad():return softmax(d(torch.tensor(rt.row_features(r,layout,memo=False)),r['q']['primitive']=='choice').numpy(),m['calibration']['selected'])
                    m['benchmark']=sample_benchmark(predict,rows,p['benchmark']['repeats']);write_json(out/(m['name']+'.json'),m)
                    profiles.append(dict(m,method='frozen',spec=spec,layout=layout,artifact_dir=out.relative_to(root).as_posix(),head_file=m['name']+'.safetensors'))
            elif arm.endswith('_online') or arm.endswith('_lora') and '_finite_' not in arm:
                mode='online' if arm.endswith('_online') else 'lora'
                initializer=root/'arms'/stage_name(spec['key']+'_frozen',layout,seed)/('rank_seed'+str(seed)+'.safetensors')
                if not initializer.exists():raise GateBlocked('Matching fresh frozen rank not yet completed: '+str(initializer))
                initial=Scorer(rt.hidden);initial.load_state_dict(load_file(str(initializer)))
                head,tr=online_train(rt,rows,p,out/'training',mode,seed,layout,cfg,initial)
                rt.store.close();rt.identity['adaptation_sha256']=tr['sha256'];rt.store=FeatureStore(out/'adapted_features.sqlite',rt.identity);rt.share_feature_store(root)
                xs=feature_rows(rt,rows,layout,cfg,start)
                head=head.cpu();head.change_normalization_preserving_scores(torch.tensor(np.concatenate([x for x,r in zip(xs,rows) if r['split']=='train'])))
                rec=fit_frozen_variants(xs,rows,p,out,seed,rank=head)
                for m in rec:
                    d=load_decision(out/(m['name']+'.safetensors'),m)
                    def predict(r,d=d):
                        with torch.no_grad():return softmax(d(torch.tensor(rt.row_features(r,layout,memo=False)),r['q']['primitive']=='choice').numpy(),m['calibration']['selected'])
                    m['benchmark']=sample_benchmark(predict,rows,p['benchmark']['repeats']);write_json(out/(m['name']+'.json'),m)
                    profiles.append(dict(m,method='frozen',spec=spec,layout=layout,artifact_dir=out.relative_to(root).as_posix(),head_file=m['name']+'.safetensors',training=tr,training_dir='training'))
            elif arm=='modernbert_full' or '_finite_' in arm:
                mode='full_encoder' if arm=='modernbert_full' else 'finite_lora' if arm.endswith('_lora') else 'finite_frozen'
                head=None;tr=None
                if mode!='finite_frozen':head,tr=online_train(rt,rows,p,out/'training',mode,seed,layout,cfg)
                if tr:rt.identity['adaptation_sha256']=tr['sha256']
                def logits_for(r):
                    with torch.no_grad():return (head(rt.marker_features(r)).squeeze(-1) if mode=='full_encoder' else rt.finite_logits(r)).detach().float().cpu().numpy()
                # Resume downstream prediction caching using final trained-weight identity.
                rt.store.close();rt.store=FeatureStore(out/'direct_logits.sqlite',rt.identity)
                logits=[]
                for i,r in enumerate(rows):
                    key=['direct_logits',mode,digest(r)];z=rt.store.get(key)
                    if z is None:z=logits_for(r);rt.store.put(key,z)
                    logits.append(z)
                    if (i+1)%cfg.checkpoint_steps==0:
                        rt.store.backup()
                        if time.monotonic()-start>cfg.worker_budget_minutes*60:raise BudgetStop('Direct logits checkpoint saved')
                summ,pred=package_prediction(logits,rows,p);name=mode+'_seed'+str(seed)
                m={'name':name,'method':'encoder_native' if mode=='full_encoder' else 'finite','layout':layout,'seed':seed,'spec':spec,'training':tr,'training_dir':'training' if tr else None,'head_file':None,**summ}
                m['benchmark']=sample_benchmark(lambda r:softmax(logits_for(r),m['calibration']['selected']),rows,p['benchmark']['repeats']);write_json(out/(name+'.json'),m);write_json(out/(name+'_nonfinal_predictions.json'),pred)
                profiles=[dict(m,artifact_dir=out.relative_to(root).as_posix())]
            else:raise GateBlocked('Arm adapter not implemented: '+arm)
        finally:
            if rt is not None:rt.close()
    # Identity includes exact profiles. No final targets appear in ARM_DONE or selection.
    write_json(out/'profiles.json',profiles)
    files=tree_hashes(out,('__pycache__',));files={n:h for n,h in files.items() if n!='ARM_DONE.json' and not n.endswith(('.sqlite','.sqlite.backup')) and not n.startswith('ckpt-')}
    write_json(completion,{'status':'completed','identity':identity,'job':job,'profiles':profiles,'files':files,'final_data_used':False});return profiles


class Predictor:
    """Loads one locked profile; used by final evaluation, scaling and the resident native worker."""
    def __init__(self,root,p,profile,out,allow_cpu_test=False):
        import torch
        from torch import nn
        from safetensors.torch import load_file
        from odij_learning import load_decision,Scorer
        from odij_runtime import Runtime,inject_lora
        from odij_training import apply_state
        self.root=Path(root);self.p=p;self.profile=profile;self.rt=None;self.head=None;self.d=None
        if profile['method']=='lexical':return
        self.rt=Runtime(profile['spec'],out,p,allow_cpu_test=allow_cpu_test);base=self.root/profile['artifact_dir'];tr=profile.get('training')
        if tr:
            if tr['mode'] in ('lora','finite_lora'):inject_lora(self.rt.model,tr['rank'],tr['alpha'],p['lora_last_blocks'],tr['targets'])
            elif tr['mode']=='full_encoder':
                for v in self.rt.model.parameters():v.requires_grad_(True)
            head=nn.Linear(self.rt.hidden,1).to(self.rt.device) if tr['mode']=='full_encoder' else Scorer(self.rt.hidden).to(self.rt.device) if tr['mode'] in ('online','lora') else None
            if head is not None and isinstance(head,Scorer):head.none.requires_grad_(False)
            path=base/profile['training_dir']/'trained.safetensors'
            if file_sha(path)!=tr['sha256']:raise IntegrityError('Trained artifact changed')
            apply_state(self.rt.model,head,load_file(str(path)));self.head=head
            for v in self.rt.model.parameters():v.requires_grad_(False)
            self.rt.model.eval();self.rt.store.close();self.rt.identity['adaptation_sha256']=tr['sha256'];self.rt.store=FeatureStore(Path(out)/'adapted_inference.sqlite',self.rt.identity)
        self.rt.share_feature_store(self.root)
        if profile['method']=='frozen':self.d=load_decision(base/profile['head_file'],profile)
    def logits(self,row,memo=True):
        import torch
        if self.profile['method']=='lexical':return basic_lexical(row)
        with torch.no_grad():
            if self.profile['method']=='frozen':return self.d(torch.tensor(self.rt.row_features(row,self.profile['layout'],memo=memo)),row['q']['primitive']=='choice').numpy()
            if self.profile['method']=='encoder_native':return self.head(self.rt.marker_features(row)).squeeze(-1).float().cpu().numpy()
            return self.rt.finite_logits(row).float().cpu().numpy()
    def predict(self,row,memo=True):
        from odij_learning import softmax
        return softmax(self.logits(row,memo),self.profile['calibration']['selected'])
    def close(self):
        if self.rt:self.rt.close()


def evaluate_final(root,cfg,profile):
    from odij_learning import output_row,metrics,policy_records
    assert_study_intact(root)
    p=read_json(root/'registered_protocol.json');lock=read_json(root/'MODEL_LOCK.json');pid=profile['profile_id'];out=root/'final'/pid;out.mkdir(parents=True,exist_ok=True)
    for n,h in lock['artifact_hashes'].items():
        if file_sha(root/n)!=h:raise IntegrityError('Model lock artifact changed')
    if (out/'FINAL_DONE.json').exists():
        done=read_json(out/'FINAL_DONE.json')
        if file_sha(out/'predictions.json')!=done['predictions_sha256']:raise IntegrityError('Final predictions changed')
        return done
    exposed=root/'FINAL_EXPOSED.json'
    if not exposed.exists():write_json(exposed,{'time':utc(),'model_lock_sha256':file_sha(root/'MODEL_LOCK.json'),'warning':'Final data now exposed; do not change/reselect models. Resume only this same locked comparison.'})
    elif read_json(exposed)['model_lock_sha256']!=file_sha(root/'MODEL_LOCK.json'):raise IntegrityError('Cannot change model selection after final exposure')
    rows=flatten(read_json(root/'data/final.json'));pr=Predictor(root,p,profile,out/'runtime',cfg.allow_cpu_test);start=time.monotonic();checkpoint=out/'predictions.checkpoint.json';preds=read_json(checkpoint) if checkpoint.exists() else []
    if isinstance(preds,dict):
        if preds.get('model_lock_sha256')!=file_sha(root/'MODEL_LOCK.json') or preds.get('profile_id')!=pid or preds.get('rows_sha256')!=digest(preds.get('rows')):raise IntegrityError('Final checkpoint integrity failed')
        preds=preds['rows']
    if [r['id'] for r in preds]!=[r['id'] for r in rows[:len(preds)]]:raise IntegrityError('Final checkpoint ID/order mismatch')
    try:
        for r in rows[len(preds):]:
            preds.append(output_row(r,pr.predict(r)))
            if len(preds)%cfg.checkpoint_steps==0:
                write_json(checkpoint,{'model_lock_sha256':file_sha(root/'MODEL_LOCK.json'),'profile_id':pid,'rows':preds,'rows_sha256':digest(preds)})
                if pr.rt:pr.rt.store.backup()
                if time.monotonic()-start>cfg.worker_budget_minutes*60:raise BudgetStop('Final prediction checkpoint saved; selection remains locked')
        write_json(out/'predictions.json',preds)
        m=metrics(preds);pol=policy_records(preds,p['policy'],profile['policy']['selected']['threshold'])
        result={'status':'completed','profile_id':pid,'selected_before_final':pid==lock['selected_profile'],'metrics':m,'policy':pol,'predictions_sha256':file_sha(out/'predictions.json'),
          'interpretation':'Final quality is separate from task coverage, numerical equivalence and service readiness.'}
        bounds=p['quality_requirements'];bench=profile.get('benchmark',{});peak=bench.get('memory',{}).get('peak_allocated_gib')
        error=pol['panel_error_among_accepted']
        checks={'family_macro_nll':m['family_macro_nll']<=bounds['max_family_macro_nll'],'accepted_error':error is not None and error<=bounds['max_panel_accepted_error'],'minimum_coverage':pol['family_group_weighted_coverage']>=bounds['min_policy_coverage'],'sample_request_p95':bench.get('p95_ms',float('inf'))<=bounds['max_request_p95_ms'],'peak_allocated_gpu_memory':peak is not None and peak<=bounds['max_peak_allocated_gib']}
        result['registered_point_estimate_checks']=checks;result['meets_registered_point_estimate_limits']=all(checks.values());result['acceptance_limit']='This is not rare-error certification, service p95, arbitrary-question generality or production promotion; it does not select a replacement after seeing final results.'
        write_json(out/'FINAL_DONE.json',result);return result
    finally:pr.close()


def select_and_lock(root,p,profiles):
    assert_study_intact(root)
    if not profiles:raise GateBlocked('No completed profiles')
    normalized=[]
    for m in profiles:
        x=copy.deepcopy(m);x['profile_id']=digest([m['artifact_dir'],m['name']])[:20];normalized.append(x)
    # Preselection stays tied to transfer-aware dev NLL. Limits are reported, never invented or loosened.
    selected=min(normalized,key=lambda x:(x['development']['family_macro_nll'],x['profile_id']))
    hashes={}
    for m in normalized:
        base=root/m['artifact_dir'];names=[m['name']+'.json']
        if m.get('head_file'):names.append(m['head_file'])
        if m.get('training_dir'):names += [m['training_dir']+'/TRAINING_DONE.json',m['training_dir']+'/trained.safetensors']
        for n in names:hashes[(base/n).relative_to(root).as_posix()]=file_sha(base/n)
    lock={'study_identity':read_json(root/'study_lock.json')['identity'],'selected_profile':selected['profile_id'],'selection_metric':'family_macro_nll','selection_partition':'dev',
          'profiles':normalized,'artifact_hashes':hashes,'selection_on_final':False,
          'selection_note':'This is a development-selected research candidate, not automatic production promotion. Final records retain all preregistered feasible arms.'}
    path=root/'MODEL_LOCK.json'
    if path.exists():
        if read_json(path)!=lock:raise IntegrityError('Model selection/weights changed; final lock cannot be replaced')
    else:write_json(path,lock)
    return lock


def status_report(root):
    j=Journal(root);g=read_json(root/'gate_report.json') if (root/'gate_report.json').exists() else {'status':'not_checked'}
    rows=j.report();bad=[r for r in rows if r['status'] in ('failed','paused','blocked')]
    overall='failed' if any(r['status']=='failed' for r in rows) else 'paused' if any(r['status']=='paused' for r in rows) else 'blocked' if g['status']!='passed' or any(r['status']=='blocked' and not r['stage'].startswith('deferred:') for r in rows) else 'completed_requested_scope'
    result={'version':VERSION,'status':overall,'gate_report':g,'stages':rows,'not_production_certification':True}
    write_json(root/'summary.json',result)
    text='# OpenDecision 2I/2J workbench run\n\nOverall workflow status: **'+overall+'**.\n\n'
    text+='| Stage | Status | Detail |\n|---|---|---|\n'
    for r in rows:text+='| '+r['stage']+' | '+r['status']+' | '+str(r.get('reason',r.get('scope',''))).replace('|','/').replace('\n',' ')+' |\n'
    text+='\nDetailed JSON artifacts, registered source hashes, model locks and per-profile results accompany this report. Preparation/checkpoint success is not a quality pass or roadmap completion.\n'
    atomic_bytes(root/'REPORT.md',text.encode());return result


def worker_main(job_path):
    job=read_json(job_path);cfg=Config(**job['cfg']).validate();root=Path(job['root']);stage=job['stage'];result_path=Path(job['result_path'])
    try:
        if not cfg.allow_cpu_test and shutil.disk_usage(root).free/2**30<cfg.minimum_free_disk_gib:
            raise GateBlocked('Insufficient free disk for declared model/checkpoint budget; free space without deleting original results')
        if stage=='arm':result=run_arm(root,cfg,job['arm_job'])
        elif stage=='final':result=evaluate_final(root,cfg,job['profile'])
        elif stage=='mechanics':
            from odij_scaling import mechanics_probe
            result=mechanics_probe(root,cfg)
        elif stage=='scaling':
            from odij_scaling import study_scaling
            result=study_scaling(root,cfg,job['profile'])
        else:raise ValueError('Unknown stage')
        write_json(result_path,{'status':'completed','result':result})
    except GateBlocked as e:write_json(result_path,{'status':'blocked','reason':str(e)})
    except BudgetStop as e:write_json(result_path,{'status':'paused','reason':str(e)})
    except Exception as e:
        # Do not log request text, tokens, secrets, or exception objects containing serialized datasets.
        msg=re.sub(r'hf_[A-Za-z0-9_]+','[REDACTED]',str(e))[:1000]
        write_json(result_path,{'status':'failed','reason':type(e).__name__+': '+msg})
        traceback.print_exc()


class Workbench:
    def __init__(self,cfg):
        self.cfg=cfg.validate();self.root=Path(cfg.work_root)/cfg.study_id;self.drive=Path(cfg.drive_root)/cfg.study_id;self.intake=Path(cfg.intake_root);self.root.mkdir(parents=True,exist_ok=True)
        if not (self.root/'status.json').exists():restore_snapshot(self.drive,self.root)
        self.journal=Journal(self.root)
    def sync(self):
        return commit_snapshot(self.root,self.drive)
    def worker(self,name,stage,**kw):
        remaining=self.cfg.coordinator_budget_minutes-(time.monotonic()-getattr(self,'_invocation_start',time.monotonic()))/60
        if remaining<=0:raise BudgetStop('Notebook invocation budget reached. Resume the same study; no selection rules or completed work change.')
        job_cfg=asdict(self.cfg);job_cfg['worker_budget_minutes']=min(self.cfg.worker_budget_minutes,max(.25,remaining))
        path=self.root/'jobs'/(digest(name)[:16]+'.json');result=self.root/'jobs'/(digest(name)[:16]+'.result.json')
        if result.exists():result.unlink()
        write_json(path,{'cfg':job_cfg,'root':str(self.root),'stage':stage,'result_path':str(result),**kw})
        log=path.with_suffix('.log');self.journal.record(name,'running');last=time.monotonic()
        with open(log,'w') as f:
            env=os.environ.copy();env['PYTHONUNBUFFERED']='1';env['OMP_NUM_THREADS']='2'
            process=subprocess.Popen([sys.executable,'-m','odij_runner','worker',str(path)],stdout=f,stderr=subprocess.STDOUT,env=env)
            try:
                while process.poll() is None:
                    time.sleep(2)
                    if time.monotonic()-last>=self.cfg.sync_seconds:
                        # Child snapshots are individually committed before copying. A partial generation never becomes LATEST.
                        try:self.sync()
                        except Exception as e:print('Snapshot retry needed:',type(e).__name__,flush=True)
                        last=time.monotonic()
            except BaseException:
                process.terminate()
                try:process.wait(timeout=20)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                self.journal.record(name,'paused',reason='Coordinator interrupted; resume last committed checkpoint');self.sync();raise
        res=read_json(result) if result.exists() else {'status':'failed','reason':'Worker exited without a result; inspect '+str(log)}
        self.journal.record(name,res['status'],reason=res.get('reason',''),result_file=str(result.relative_to(self.root)))
        self.sync();return res
    def prepare(self):
        from odij_history import prepare_history
        with local_lock(self.root):
            h,blocked=prepare_history(self.cfg.source_archive,self.root);self.journal.record('historical_H','completed',scope='Read-only lineage, 39 code hashes, saved-head algebra and exclusions; no retraining')
            historical_readout(self.root);write_draft(self.intake);export_contracts(self.root)
            self.journal.record('review_pack','completed',scope='Constructed candidate cases and unapproved protocol/reviewer templates; no independent review asserted')
            self.journal.record('native_contract_draft','completed',scope='Native probability mapping draft and worker contract; not Jev/Rust conformance')
            try:self.journal.record('immutable_model_revisions','completed',result=resolve_revisions(self.intake))
            except Exception as e:self.journal.record('immutable_model_revisions','blocked',reason='Resolve model revisions in protocol.json: '+str(e)[:300])
            blocked.update(previous_final_exclusions(Path(self.cfg.drive_root),self.cfg.study_id))
            p=register_study(self.root,self.intake,blocked);self.journal.record('reviewed_study_gate','completed' if p else 'blocked',reason='' if p else 'Independent review, explicit resource/quality contract and immutable revisions required; see gate_report.json')
            self.sync();return p
    def run_ready(self):
        self._invocation_start=time.monotonic();self.journal.record('invocation_budget','running')
        try:self._run_ready_impl()
        except BudgetStop as e:self.journal.record('invocation_budget','paused',reason=str(e))
        else:self.journal.record('invocation_budget','completed',scope='Requested invocation ended within its stage-boundary budget')
        result=status_report(self.root);self.sync();return result
    def _run_ready_impl(self):
        p=self.prepare()
        with local_lock(self.root):
            if self.cfg.run_gpu_mechanics:
                prior=self.journal.rows.get('gpu_mechanics',{})
                if prior.get('status')!='completed':self.worker('gpu_mechanics','mechanics')
            if self.cfg.run_reviewed_study and p:
                results=[];blocking=[];skipped=[]
                for j in jobs_for(p):
                    name=stage_name(**{'arm':j['arm'],'layout':j['layout'],'seed':j['seed']})
                    res=self.worker('fit:'+name,'arm',arm_job=j)
                    if res['status']=='completed':results.extend(res['result'])
                    elif res['status']=='blocked' and p.get('allow_hardware_skips',False) and ('memory' in res.get('reason','').lower() or 'GPU' in res.get('reason','')):
                        skipped.append({'job':j,'reason':res['reason']});self.journal.record('fit:'+name,'skipped_hardware',reason=res['reason'])
                    else:blocking.append(name)
                write_json(self.root/'preregistered_skips.json',skipped)
                if blocking:self.journal.record('model_lock','blocked',reason='Wait for required paused/failed/dependency stages: '+', '.join(blocking))
                else:
                    lock=select_and_lock(self.root,p,results);self.journal.record('model_lock','completed',scope='Model, calibration, thresholds and all feasible comparisons frozen before final')
                    self.sync()
                    if self.cfg.run_final:
                        record_final_exposure(self.root,Path(self.cfg.drive_root),self.cfg.study_id)
                        for m in lock['profiles']:self.worker('final:'+m['profile_id'],'final',profile=m)
                    if self.cfg.run_scaling:
                        # Sharing requires a TRAINED state-first profile selected using dev, not the final winner.
                        causal=[m for m in lock['profiles'] if m.get('layout')=='state_first' and m.get('method')=='frozen']
                        if causal:
                            m=min(causal,key=lambda x:(x['development']['family_macro_nll'],x['profile_id']));self.worker('multiquestion_scaling','scaling',profile=m)
                        else:self.journal.record('multiquestion_scaling','blocked',reason='No fitted state-first causal profile available; joint encoder batching is not shared-state causal execution')
            self.journal.record('deferred:external_Laya_GLiClass','blocked',reason='Released-checkpoint adapter/provenance review not implemented; no claim of reproduction. Early newly trained ModernBERT arm is implemented separately.')
            self.journal.record('deferred:Rust_Metal_service','blocked',reason='Existing repository/wire-contract integration and named target-platform tests required. Native JSONL worker provided; no HTTP/native parity claimed.')
            self.journal.record('deferred:P2','blocked',reason='Selected-profile and hypothesis gates not satisfied automatically; no low-bit KV sweep, weight quantization, teacher generation or RL launched.')
            result=status_report(self.root);self.sync();return result

if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='worker':worker_main(sys.argv[2])
    else:raise SystemExit('Use through the notebook, or: python -m odij_runner worker job.json')
