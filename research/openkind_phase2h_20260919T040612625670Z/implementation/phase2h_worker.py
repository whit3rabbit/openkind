"""Worker entry points. Fit never reads final payload; final checks the frozen lock first."""
import argparse,copy,gc,json,os,time,traceback,math
from pathlib import Path
from dataclasses import asdict
from collections import defaultdict
import numpy as np
import torch
from safetensors.torch import load_file,save_file
from phase2h_config import Settings,VERSION
from phase2h_data import apply_criteria,contextualize
from phase2h_learning import *
from phase2h_runtime import Runtime,controls,encode,finite_prompt,memory_snapshot
from phase2h_checkpoint import (RowCheckpoint, validate_probability_row, publish_worker,
                                bind_worker_context, verify_final_lock, canonical_hash)
from phase2d_common import CandidateHead,file_sha,json_write,content_hash
from phase2e_precision import sample_parameter_digest,backend_flags
from phase2e_runtime import is_fatal_cuda
from phase2e_core import StageSkip
from phase2h_primitives import fit_fixed,FixedReadout,fixed_metrics

def legacy_scorer(source,hidden):
    m=CandidateHead(hidden);m.load_state_dict(load_file(str(Path(source)/'export/frozen_candidate_head.safetensors')));m.eval();return AffineScorer(hidden).preserve_legacy(m)

def lexical_fit(episodes,library,variant):
    from sklearn.feature_extraction.text import TfidfVectorizer
    docs=[e['state'] for e in episodes]+[c['description'] for e in episodes for c in e['choices']]
    vector=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5));vector.fit(docs)
    return {'vocabulary':{k:int(v) for k,v in vector.vocabulary_.items()},'idf':vector.idf_.tolist(),'fit_scope':'train messages plus offered label definitions; no final message text'}

def lexical_scores(model,episodes):
    from sklearn.feature_extraction.text import TfidfVectorizer
    v=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),vocabulary=model['vocabulary']);v.idf_=np.asarray(model['idf'])
    return [np.asarray((v.transform([e['state']])@v.transform([c['description'] for c in e['choices']]).T).toarray()[0])*10 for e in episodes]

def artifact_profile(name,method,variant,layout,scorer_path,none_model,temp,policies,**extra):
    return dict(name=name,method=method,criteria=variant,layout=layout,scorer_path=scorer_path,none_model=none_model,temperature=temp,policies=policies,**extra)

def fit_all(rt,data,cfg,out,report,only_scorer=None):
    library=data['criteria'];episodes=data['episodes'];profiles=[];features={};score_sets={};layouts=('instruction_first','state_first') if cfg.run_state_first else ('instruction_first',)
    initial=legacy_scorer(rt.source,rt.hidden) if rt.spec['name']=='qwen4b' else None
    old_none=json.loads((rt.source/'export/none_set_linear.json').read_text())
    for variant in library:
        ee={s:[apply_criteria(e,library,variant) for e in episodes[s]] for s in episodes}
        for layout in layouts:
            key=variant+'__'+layout;print('FIT feature contract:',key,flush=True)
            xx={s:rt.episodes(es,layout) for s,es in ee.items()};features[key]=xx
            # Backward-compatible controls exist only at the original 4B hidden width.
            scorers=[]
            if initial is not None:scorers.append(('frozen',copy.deepcopy(initial),{'backbone_updated':False,'candidate_updated':False}))
            for seed in cfg.head_seeds:
                model,training=train_scorer(xx['train'],ee['train'],xx['dev'],ee['dev'],cfg,seed,initial)
                training['initialization']='historical_4B_scorer' if initial is not None else 'random_head'
                scorers.append(('joint_seed'+str(seed),model,training))
                if cfg.run_smaller_model and initial is not None:
                    scratch,scratch_info=train_scorer(xx['train'],ee['train'],xx['dev'],ee['dev'],cfg,seed,None)
                    scratch_info['initialization']='random_head_matched_to_smaller_model'
                    scorers.append(('scratch_joint_seed'+str(seed),scratch,scratch_info))
            for tag,model,training in scorers:
                scores={s:predict_scores(model,x) for s,x in xx.items()};base=key+'__'+tag;path=out/(base+'.safetensors');model.export(path)
                if tag=='frozen':
                    logits={s:add_none(old_none,v) for s,v in scores.items()};p=select_policies(softmaxes(logits['policy_dev']),ee['policy_dev'],cfg)
                    profiles.append(artifact_profile(base+'__unchanged_none','affine',variant,layout,path.name,old_none,1.,p,training=training,calibration={'status':'unchanged_control'},dev_nll=objective(logits['dev'],ee['dev'],cfg.train_mixture),criterion_scope=data['criteria_audit']))
                none,selection=choose_none(scores,ee,cfg,float(model.none.detach()));logits={s:add_none(none,v) for s,v in scores.items()};temp,cal=calibrate(logits,ee,cfg)
                policies=select_policies(softmaxes(logits['policy_dev'],temp),ee['policy_dev'],cfg)
                profile=artifact_profile(base+'__refit_none','affine',variant,layout,path.name,none,temp,policies,training=training,none_selection=selection,calibration=cal,dev_nll=objective(logits['dev'],ee['dev'],cfg.train_mixture),criterion_scope=data['criteria_audit'])
                profiles.append(profile)
            # Checkpoint selection metadata before proceeding to another expensive extraction.
            json_write(out/'profiles_partial.json',profiles)
        lex=lexical_fit(ee['train'],library,variant);ls={s:lexical_scores(lex,es) for s,es in ee.items()};none,selection=choose_none(ls,ee,cfg);logits={s:add_none(none,v) for s,v in ls.items()};temp,cal=calibrate(logits,ee,cfg)
        lex_path=out/(variant+'__lexical.json');json_write(lex_path,lex)
        profiles.append(artifact_profile(variant+'__lexical','lexical',variant,'instruction_first',lex_path.name,none,temp,select_policies(softmaxes(logits['policy_dev'],temp),ee['policy_dev'],cfg),dev_nll=objective(logits['dev'],ee['dev'],cfg.train_mixture),calibration=cal,none_selection=selection,training={'trainable_neural_parameters':0,'score':'char-ngram TF-IDF cosine x10; scale fixed before fitting'}))
        if cfg.run_finite_token:
            fs={s:[rt.finite_scores(e) for e in es] for s,es in ee.items()};temp,cal=calibrate(fs,ee,cfg)
            profiles.append(artifact_profile(variant+'__finite_code','finite_token',variant,'instruction_first',None,None,temp,select_policies(softmaxes(fs['policy_dev'],temp),ee['policy_dev'],cfg),dev_nll=objective(fs['dev'],ee['dev'],cfg.train_mixture),calibration=cal,training={'backbone_updated':False,'readout':'original tied token rows; no answer-generation loop','comparison':'same task/splits, different input rendering; zero candidate-gradient updates, not an equal-training-budget claim'}))
    # No final-set metric is available in this worker. Seal all treatment choices now.
    affines=[p for p in profiles if p['method']=='affine' and 'joint_seed' in p['name']]
    winner=min(affines,key=lambda p:(p['dev_nll'],p['name']))
    report['selected_profile']=winner['name'];report['profiles']=profiles
    # Development-only head fixtures for a future port; not a Rust/Metal validation.
    fm=AffineScorer.load(out/winner['scorer_path']);fx=features[winner['criteria']+'__'+winner['layout']]['dev'][:2]
    fs=predict_scores(fm,fx);fp=softmaxes(add_none(winner['none_model'],fs),winner['temperature'])
    save_file({'episode_'+str(i):torch.as_tensor(x).contiguous() for i,x in enumerate(fx)},str(out/'head_reference_vectors.safetensors'))
    json_write(out/'head_reference_fixtures.json',{'profile':winner['name'],'scorer_path':winner['scorer_path'],'none_model':winner['none_model'],'temperature':winner['temperature'],
        'candidate_scores':[x.tolist() for x in fs],'probabilities':[x.tolist() for x in fp],'scope':'actual selected-head development vectors; supports future head algebra checks only, not backend parity'})
    report['primitive_profiles']=[]
    if cfg.run_primitives:
        for task,rr in data.get('primitives',{}).items():
            ff={s:rt.fixed_features(es) for s,es in rr.items()}
            options=[]
            for seed in cfg.head_seeds:
                model,meta=fit_fixed(ff,rr,cfg,seed,task);name=f'{task}_seed{seed}.safetensors';save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(out/name));meta['path']=name;options.append(meta)
            chosen=min(options,key=lambda x:(x['dev_nll'],x['seed']));report['primitive_profiles'].append({'task':task,'selected_seed':chosen['seed'],'candidates':options})
    json_write(out/'profiles.json',profiles);json_write(out/'selection.json',{'selected_profile':winner['name'],'primitive_profiles':report['primitive_profiles'],'rule':'development weighted NLL only; final not opened'})
    return report

class Predictor:
    def __init__(self,rt,fitroot,profiles,library,cfg):
        self.rt=rt;self.root=Path(fitroot);self.profiles={p['name']:p for p in profiles};self.library=library;self.cfg=cfg;self.models={}
    def episode(self,ep,p):
        return copy.deepcopy(ep) if ep.get('panel')=='natural_documents' else apply_criteria(ep,self.library,p['criteria'])
    def model(self,p):
        key=p['scorer_path']
        if key not in self.models:
            self.models[key]=AffineScorer.load(self.root/key) if p['method']=='affine' else json.loads((self.root/key).read_text())
        return self.models[key]
    def scores(self,ep,p,memo=True,strategy='full_sequential',verify=True):
        e=self.episode(ep,p)
        if p['method']=='lexical':return lexical_scores(self.model(p),[e])[0]
        if p['method']=='finite_token':return self.rt.finite_scores(e,memo)
        seqs=[encode(e,c,self.rt.tokenizer,self.cfg,p['layout']) for c in e['choices']]
        x=np.stack([self.rt.feature(s,memo=True) for s in seqs]) if memo else self.rt.run_sequences(seqs,strategy,verify=verify and strategy.startswith('shared'))
        return predict_scores(self.model(p),[x])[0]
    def predict(self,ep,p,memo=True,strategy='full_sequential',verify=True):
        scores=self.scores(ep,p,memo,strategy,verify);logits=scores if p['method']=='finite_token' else add_none(p['none_model'],[scores])[0]
        return softmaxes([logits],p['temperature'])[0]
    def request(self,ep,p,strategy):
        prob=self.predict(ep,p,False,strategy,verify=False);decisions=[policy_action(prob,x['threshold']) for x in p['policies']]
        return json.dumps({'probabilities':prob.tolist(),'policies':decisions},allow_nan=False),prob

def evaluate_profiles(pred,episodes,profiles,cfg,out,stem):
    # Resume only against the same sealed inputs, profiles and runtime. No refitting.
    checkpoint=RowCheckpoint(out,stem,{'key_fields':['profile','id'],
        'episodes':episodes,'profiles':profiles,'configuration':asdict(cfg)})
    allowed={(p['name'],e['id']) for p in profiles for e in episodes}
    if any((r['profile'],r['id']) not in allowed for r in checkpoint.rows):
        raise ValueError('Unexpected row in '+stem+' checkpoint')
    records=[];summaries={}
    for profile in profiles:
        ps=[]
        for j,ep in enumerate(episodes):
            row=checkpoint.get(profile['name'],ep['id'])
            if row is None:
                prob=pred.predict(ep,profile)
                row={'id':ep['id'],'group':ep['group'],'family':ep['family'],'panel':ep['panel'],
                     'profile':profile['name'],'probabilities':prob.tolist(),'target_index':ep['target_index'],
                     'none_origin':ep['none_origin'],'K':len(ep['choices'])}
                checkpoint.add(row)
            prob=np.asarray(validate_probability_row(row,ep,profile),float)
            ps.append(prob);records.append(row)
            if (j+1)%cfg.checkpoint_every==0:
                checkpoint.save();pred.rt.store.db.commit()
                print(f'  {stem}: {profile["name"]} {j+1}/{len(episodes)} (saved rows reused)',flush=True)
                publish_worker(out,pred.rt.store)
        checkpoint.save()
        # Same original metric definitions, policies, mixture and bootstrap seeds.
        results={}
        for family in sorted({e['family'] for e in episodes}):
            ix=[i for i,e in enumerate(episodes) if e['family']==family];es=[episodes[i] for i in ix];pp=[ps[i] for i in ix]
            results[family]={'quality':quality(pp,es,cfg),'policies':[policy_stats(pp,es,x['threshold'],x['assumed_absent_prior'],x['wrong_cost'],cfg,with_intervals=True) for x in profile['policies']]}
        results['pooled_declared_mixture']={'quality':quality(ps,episodes,cfg),'policies':[policy_stats(ps,episodes,x['threshold'],x['assumed_absent_prior'],x['wrong_cost'],cfg,with_intervals=True) for x in profile['policies']]}
        summaries[profile['name']]=results
        json_write(out/(stem+'_rows.json'),records)
        json_write(out/(stem+'_metrics.json'),summaries)
        publish_worker(out,pred.rt.store)
    if stem=='final':
        details={e['id']:e for e in episodes};errors=[]
        for row in records:
            pr=np.asarray(row['probabilities']);index=int(np.argmax(pr))
            if index!=row['target_index'] and float(pr[index])>=.98:
                e=details[row['id']];p=pred.profiles[row['profile']];rendered=pred.episode(e,p)
                errors.append(dict(row,state=e['state'],choices=rendered['choices'],predicted_index=index,
                                   top_probability=float(pr[index]),original_gold_unchanged=True))
        json_write(out/'high_confidence_errors.json',{'threshold':.98,
            'scope':'Post-evaluation dossier; top_probability is not a separately defined confidence statistic; variants remain grouped by source message',
            'errors':errors})
    return records,summaries

def time_request(fn,repeats=3):
    fn();torch.cuda.synchronize();before=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();times=[];result=None
    for _ in range(repeats):
        torch.cuda.synchronize();start=time.perf_counter();result=fn();torch.cuda.synchronize();times.append((time.perf_counter()-start)*1000)
    return result,{'times_ms':times,'median_ms':float(np.median(times)),'resident_mib':before/1024**2,'peak_extra_mib':max(0,torch.cuda.max_memory_allocated()-before)/1024**2,'timing_scope':'complete in-process request; tokenizer, GPU forward, transfer, head, policy and JSON; no HTTP'}

def subset_by_groups(eps,n,seed):
    chosen=[]
    for family in sorted({e['family'] for e in eps}):
        groups=sorted({e['group'] for e in eps if e['family']==family},key=lambda g:content_hash([seed,g]))[:n]
        chosen.extend(e for e in eps if e['family']==family and e['group'] in groups)
    return chosen

def final_all(rt,data,cfg,out,report,fitroot):
    fitroot=Path(fitroot);profiles=json.loads((fitroot/'profiles.json').read_text());selection=json.loads((fitroot/'selection.json').read_text());pred=Predictor(rt,fitroot,profiles,data['criteria'],cfg)
    eps=data['episodes'];report['final_records'],report['final_metrics']=evaluate_profiles(pred,eps,profiles,cfg,out,'final')
    # Keep summary JSON compact; raw vectors are in a separate inspectable file.
    raw_records=report.pop('final_records',[]);winner=pred.profiles[selection['selected_profile']];report['selected_profile']=winner['name']
    # Matched contrasts are reports, never a second final-set selection rule.
    from phase2h_learning import paired_quality
    vector_map={(r['profile'],r['id']):np.asarray(r['probabilities']) for r in raw_records}
    reference_name='original__instruction_first__frozen__unchanged_none'
    if reference_name in pred.profiles:
        paired={}
        for profile in profiles:
            if profile['name']==reference_name:continue
            paired[profile['name']]=paired_quality([vector_map[(reference_name,e['id'])] for e in eps],[vector_map[(profile['name'],e['id'])] for e in eps],eps,cfg)
        report['paired_vs_original']=paired;json_write(out/'paired_final_contrasts.json',paired)
    report['stage_status']['final_profiles']='completed';json_write(out/'worker_report.json',report)
    panel=subset_by_groups(eps,2,cfg.seed+990)
    parity_checkpoint=RowCheckpoint(out,'parity',{'key_fields':['id','strategy'],'episodes':panel,'profile':winner})
    parity_rows=[]
    for ep in panel:
        ref=pred.predict(ep,winner,memo=True)
        for strategy in ('full_batch4','shared_lossless','shared_kv_fp16'):
            row=parity_checkpoint.get(ep['id'],strategy)
            if row is None:
                p=pred.predict(ep,winner,False,strategy);q=parity([ref],[p],[ep],cfg.probability_tolerance,winner['policies'])
                row=dict(q,id=ep['id'],family=ep['family'],strategy=strategy,reference=ref.tolist(),probabilities=p.tolist())
                parity_checkpoint.add(row);parity_checkpoint.save()
            parity_rows.append(row)
        publish_worker(out,rt.store)
    json_write(out/'parity_rows.json',parity_rows);report['parity_summary']={s:{'episodes':sum(r['strategy']==s for r in parity_rows),'accepted':sum(r['accepted'] for r in parity_rows if r['strategy']==s),'max_delta':max((r['max_probability_delta'] for r in parity_rows if r['strategy']==s),default=0)} for s in ('full_batch4','shared_lossless','shared_kv_fp16')}
    report['stage_status']['parity']='completed';json_write(out/'worker_report.json',report)
    if cfg.run_robustness:
        source=[e for e in subset_by_groups(eps,cfg.sizes()['robust_per_family'],cfg.seed+1001) if e['K']==4 and e['none_origin']!='author_oos']
        contexts=[]
        for ep in source:
            for count in cfg.context_min_tokens:
                for position in ('first','last'):contexts.append(contextualize(ep,position,count,rt.tokenizer,cfg))
            for kind in ('instruction_paraphrase','instruction_in_notes'):contexts.append(contextualize(ep,kind,256,rt.tokenizer,cfg))
        # Compare chosen trained model with its own frozen criteria control, not only a favorable arm.
        controls_p=[p for p in profiles if p['criteria']==winner['criteria'] and p['layout']==winner['layout'] and 'frozen__unchanged_none' in p['name']]
        rr,mm=evaluate_profiles(pred,contexts,[winner]+controls_p,cfg,out,'robustness');report['robustness_metrics']=mm
        json_write(out/'robustness_inputs.json',contexts)
        by=[]
        for p in [winner]+controls_p:
            for tokens in cfg.context_min_tokens:
                for pos in ('first','last'):
                    es=[e for e in contexts if e['context_min_tokens']==tokens and e['evidence_position']==pos];lookup={r['id']:r['probabilities'] for r in rr if r['profile']==p['name']};ps=[lookup[e['id']] for e in es]
                    by.append({'profile':p['name'],'min_tokens':tokens,'position':pos,'quality':quality(ps,es,cfg)})
        report['context_by_position']=by
        report['stage_status']['robustness']='completed';json_write(out/'worker_report.json',report)
        publish_worker(out,rt.store,force=True)
    if data.get('natural_documents'):
        _,report['natural_metrics']=evaluate_profiles(pred,data['natural_documents'],[winner],cfg,out,'natural_documents')
    else:report['natural_documents']={'status':'not_provided','reason':'No natural documents with independent labels supplied; controlled wrappers are not substituted'}
    # Representative request per message, alternating K before looking at results.
    groups=sorted({e['group'] for e in eps},key=lambda g:content_hash([cfg.seed+1002,g]))[:cfg.sizes()['bench']];bench=[]
    for j,g in enumerate(groups):
        options=[e for e in eps if e['group']==g and e['K']==cfg.candidate_counts[j%len(cfg.candidate_counts)]];bench.append(options[j%len(options)])
    bprofiles=[winner]+[p for p in profiles if p['method'] in ('finite_token','lexical') and p['criteria']==winner['criteria']];brows=[]
    bench_checkpoint=RowCheckpoint(out,'benchmark',{'key_fields':['id','profile','strategy'],'episodes':bench,'profiles':bprofiles,'repeats':cfg.benchmark_repeats})
    for ep in bench:
        for profile in bprofiles:
            strategies=('full_sequential','full_batch4','shared_lossless') if profile['method']=='affine' else ('full_sequential',)
            for strategy in strategies:
                row=bench_checkpoint.get(ep['id'],profile['name'],strategy)
                if row is None:
                    # Snapshot I/O is outside this timed call, including its warmup.
                    (payload,pp),timing=time_request(lambda:pred.request(ep,profile,strategy),cfg.benchmark_repeats)
                    reference=pred.predict(ep,profile)
                    row=dict(timing,id=ep['id'],K=ep['K'],profile=profile['name'],method=profile['method'],strategy=strategy,
                             parity=parity([reference],[pp],[ep],cfg.probability_tolerance,profile['policies']),
                             correct=int(np.argmax(pp))==ep['target_index'])
                    bench_checkpoint.add(row);bench_checkpoint.save()
                brows.append(row)
        # Avoid checkpoint file copies while a request measurement is active.
        publish_worker(out,rt.store)
    json_write(out/'benchmark_rows.json',brows);report['benchmark_rows']=brows
    report['stage_status']['benchmarks']='completed';json_write(out/'worker_report.json',report)
    report['primitive_metrics']={}
    if cfg.run_primitives:
        for item in selection.get('primitive_profiles',[]):
            task=item['task'];rr=data['primitives'][task];x=None;results=[]
            primitive_checkpoint=RowCheckpoint(out,task,{'key_fields':['seed'],'rows':rr,'selection':item})
            for meta in item['candidates']:
                saved=primitive_checkpoint.get(meta['seed'])
                if saved is None:
                    if x is None:x=rt.fixed_features(rr)
                    state=load_file(str(fitroot/meta['path']));model=FixedReadout(rt.hidden,state['linear.bias'].numel());model.load_state_dict(state);model.eval()
                    with torch.no_grad():z=model(torch.as_tensor(x)).numpy()
                    m=fixed_metrics(z,rr,meta['temperature'],cfg,task)
                    saved={'seed':meta['seed'],'ids':[r['id'] for r in rr],'metrics':m}
                    primitive_checkpoint.add(saved);primitive_checkpoint.save()
                json_write(out/(task+'_seed'+str(meta['seed'])+'_predictions.json'),saved)
                m=copy.deepcopy(saved['metrics']);m.pop('probabilities',None);m.pop('expected_levels',None);results.append(dict(meta,final=m))
            report['primitive_metrics'][task]={'selected_seed_before_final':item['selected_seed'],'runs':results}
            report['stage_status'][task]='completed';json_write(out/'worker_report.json',report)
            publish_worker(out,rt.store,force=True)
    if cfg.run_state_first:
        from phase2h_extensions import query_mechanics
        report['state_first_query_mechanics']=query_mechanics(rt,pred,eps,profiles,cfg,out)
    return report

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--job',required=True);args=parser.parse_args();job=json.loads(Path(args.job).read_text());cfg=Settings(**job['configuration']);cfg.sizes();out=Path(job['out']);out.mkdir(parents=True,exist_ok=True)
    report={'version':VERSION,'stage':job['stage'],'spec':job['spec'],'mode':job['mode'],'status':'running','stage_status':{},'memory':[]}
    rt=None
    try:
        if job.get('recovery_no_training') and job['stage']!='final':
            raise ValueError('Recovery jobs may not enter fitting code')
        if job['stage']=='final':
            lock=verify_final_lock(job['lock'],job['payload'])
            if 'configuration' in lock and canonical_hash(lock['configuration'])!=canonical_hash(job['configuration']):
                raise ValueError('Evaluation configuration changed after lock')
        data=json.loads(Path(job['payload']).read_text())
        with controls(job['mode']):
            rt=Runtime(job['spec'],job['mode'],job['source'],out,cfg,report);before=sample_parameter_digest(rt.model)
            report['flags']=backend_flags();report['stage_status']['load']='completed'
            if job.get('paired_runtime_identity'):
                clean=lambda identity:{k:v for k,v in identity.items() if k not in ('mode','flags')}
                if clean(rt.identity)!=clean(job['paired_runtime_identity']):
                    raise ValueError('Strict and TF32 workers must use the same GPU/packages/runtime; do not mix machines in an arithmetic comparison')
            if job['stage']=='final':
                report['resume_context_sha256']=bind_worker_context(out,{
                    'runtime':rt.identity,'flags':report['flags'],
                    'lock_sha256':file_sha(job['lock']), 'configuration':job['configuration'],
                    'online_kind':job.get('online_kind'),
                    'code_sha256':{p.name:file_sha(p) for p in Path(__file__).parent.glob('*.py')}})
            json_write(out/'worker_report.json',report)
            publish_worker(out,rt.store,force=True)
            if job.get('online_kind'):
                from phase2h_extensions import online_fit,load_adapter
                if job['stage']=='fit':report=online_fit(rt,data,cfg,out,report,job)
                else:
                    load_adapter(rt,Path(job['fitroot']));report=final_all(rt,data,cfg,out,report,job['fitroot'])
            elif job['stage']=='fit':report=fit_all(rt,data,cfg,out,report)
            else:report=final_all(rt,data,cfg,out,report,job['fitroot'])
            if not job.get('online_kind') and sample_parameter_digest(rt.model)!=before:raise RuntimeError('Frozen backbone sample changed')
            report['weight_integrity']={'sample_unchanged':True if not job.get('online_kind') else None,'scope':'integer-index sampled check, not a full parameter hash; online/LoRA fitting has a separate base-only check, adapter-equipped final path not checked here'}
            # Save core completion before telemetry; missing reporting must not erase fits.
            report['stage_status'][job['stage']]='completed';json_write(out/'worker_report.json',report)
            report['memory'].append(memory_snapshot('after_stage'));report['status']='completed'
            json_write(out/'worker_report.json',report)
            if job['stage']=='final':
                output_hashes={p.name:file_sha(p) for p in out.glob('*.json')
                               if p.name not in ('worker_report.json','job.json','output_inventory.json')}
                json_write(out/'output_inventory.json',{'schema':'openkind-final-output-inventory/v1',
                           'files':output_hashes,'stage':'final','completed':True})
            publish_worker(out,rt.store,force=True)
    except Exception as exc:
        report['status']='skipped' if isinstance(exc,StageSkip) else 'failed';report['error']={'type':type(exc).__name__,'message':str(exc).replace(os.environ.get('HF_TOKEN') or '__NO_TOKEN__','[REDACTED]')[:2000],'fatal_cuda':is_fatal_cuda(exc)}
        tb=traceback.format_exc();token=os.environ.get('HF_TOKEN');tb=tb.replace(token,'[REDACTED]') if token else tb;(out/'traceback.txt').write_text(tb);print(tb,flush=True)
    finally:
        json_write(out/'worker_report.json',report)
        if rt is not None:
            try:
                rt.store.db.commit();publish_worker(out,rt.store,force=True)
            finally:
                rt.close()
    raise SystemExit(0 if report['status']=='completed' else 2)
if __name__=='__main__':main()
