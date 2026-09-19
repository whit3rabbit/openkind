"""Fresh-process execution; semantic labels and execution parity are independently recorded."""
from __future__ import annotations
import argparse, copy, gc, inspect, json, os, traceback
from pathlib import Path
from dataclasses import asdict, fields
from collections import defaultdict
from contextlib import contextmanager
import numpy as np
import torch
from phase2g_config import Settings, VERSION
from phase2g_metrics import parity, quality, summarize_parity
from phase2g_checks import Clock, lifecycle_selftest
from phase2d_common import json_write, content_hash, file_sha, sync, forward_features
from phase2e_cache import make_plan, scalar_from_features, cache_digest
from phase2e_expansion import ExpandedSettings, redact
from phase2e_expand_execution import ScoringEngine, PROTOTYPE, probability_and_policy_views, tiny_expanded_cache_selftest
from phase2e_expand_worker import load_runtime, numerical_controls, memory_snapshot, time_call
from phase2e_runtime import indexing_selftest, is_fatal_cuda
from phase2e_precision import sample_parameter_digest, backend_flags
from phase2e_core import StageSkip, IntegrityError
from phase2f_storage import CodecBank, PrefixLRU
from phase2f_execution import CacheEngine


def light(r):
    return {'scores':np.asarray(r['scores']).tolist(),'distributions':{k:np.asarray(v).tolist() for k,v in r['distributions'].items()},'actions':r['actions']}


def record(ep,strategy):
    return {'id':ep['id'],'group':ep['group'],'family':ep['family'],'panel':ep['panel'],'strategy':strategy,'K':len(ep['choices']),
        'absent':ep['true_intent_omitted'],'none_origin':ep['none_origin'],'sampler':ep['sampler'],'target_index':ep['target_index'],'target_id':ep['target'],
        'domain':ep.get('domain'),'evidence_position':ep.get('evidence_position'),'context_target_min_tokens':ep.get('context_target_min_tokens')}


@contextmanager
def controls(mode):
    # One flag API family, matching the 2E/F implementation; never mix old/new precision APIs.
    with numerical_controls('bf16_default' if mode=='bf16_default' else 'fp32_strict_math'):
        if mode=='fp32_tf32_allowed':
            torch.backends.cuda.matmul.allow_tf32=True
            torch.backends.cudnn.allow_tf32=False
        yield dict(backend_flags(),mode=mode,tf32_scope='CUDA GEMM permission, not proof each operation uses tensor cores; FP32 storage unchanged')


class Evaluator:
    def __init__(self,engine,cfg):self.engine=engine;self.cfg=cfg;self.score_memo={}
    def with_limit(self,ep):
        self.engine.cfg.max_length=ep['max_length'];self.engine.native.cfg.max_length=ep['max_length']
        # Keep prompt identity consistent for cross-request keys when max_length changes.
        from phase2d_common import DYNAMIC_PREFIX,DYNAMIC_STATE_PREFIX,DYNAMIC_CAND_PREFIX,DYNAMIC_TAIL
        self.engine.prompt_sha=content_hash([DYNAMIC_PREFIX,DYNAMIC_STATE_PREFIX,DYNAMIC_CAND_PREFIX,DYNAMIC_TAIL,ep['max_length']])
    @torch.inference_mode()
    def reference(self,ep):
        self.with_limit(ep);engine=self.engine;plan=make_plan(ep,engine.tokenizer,engine.cfg,engine.mode);scores=[]
        # OFFLINE evaluation memoization only. Timed methods never call this function.
        for ids in plan.full_ids:
            key=tuple(ids)
            if key not in self.score_memo:
                batch=engine.native._pack([ids]);h=forward_features(engine.native.model,batch)
                self.score_memo[key]=float(scalar_from_features(engine.native.head,h)[0])
            scores.append(self.score_memo[key])
        s=np.array(scores);d,a,n=engine.views(s,plan)
        return {'scores':s,'distributions':d,'actions':a,'meta':{'prompt_tokens':list(map(len,plan.full_ids)),'truncated':any(plan.truncated),'prefix_tokens':len(plan.prefix_ids),'offline_exact_input_score_memo':True}}
    def run(self,ep,strategy,cache=None,tenant='default',verify=False):
        self.with_limit(ep)
        if strategy=='full_sequential':return self.engine.native.request(ep,'full_sequential',verify=verify)
        if strategy=='full_batch4':return self.engine.native.request(ep,'full_batch4',verify=verify)
        if strategy=='shared_lossless':return self.engine.request(ep,'lossless',cache=cache,tenant=tenant,verify=verify)
        if strategy=='shared_kv_fp16':return self.engine.request(ep,'kv_fp16',cache=cache,tenant=tenant,verify=verify)
        raise ValueError(strategy)


def safe_error(exc):
    if is_fatal_cuda(exc):raise exc
    return {'status':'failed','error':{'type':type(exc).__name__,'message':redact(str(exc),os.environ.get('HF_TOKEN'))[:1500]}}


def evaluate_panel(ev,episodes,parity_ids,cfg,out,report,name):
    rows=[];refs={}
    strategies=('full_batch4','shared_lossless','shared_kv_fp16')
    for j,ep in enumerate(episodes):
        r=record(ep,'full_sequential')
        try:
            result=ev.reference(ep);refs[ep['id']]=light(result)
            r.update(status='completed',outputs=refs[ep['id']],metadata=result['meta'])
            # Truncation can change semantics; retain it, never certify an untruncated task.
            if result['meta']['truncated'] and ep['panel']=='controlled_context':raise IntegrityError('Evidence truncated in context control.')
        except Exception as exc:r.update(safe_error(exc))
        rows.append(r)
        if ep['id'] in parity_ids and ep['id'] in refs:
            for strategy in strategies:
                q=record(ep,strategy)
                try:
                    result=ev.run(ep,strategy,verify=strategy.startswith('shared'))
                    q.update(status='completed',outputs=light(result),comparison=parity(refs[ep['id']],light(result),cfg.cache_probability_tolerance),metadata=result['meta'])
                    if strategy=='shared_kv_fp16':
                        loss=next(x for x in reversed(rows) if x['id']==ep['id'] and x['strategy']=='shared_lossless')
                        if loss['status']=='completed':q['codec_only_comparison']=parity(loss['outputs'],q['outputs'],cfg.cache_probability_tolerance)
                except Exception as exc:q.update(safe_error(exc))
                rows.append(q)
        if j%cfg.checkpoint_every==0 or j+1==len(episodes):
            json_write(out/(name+'_rows.json'),rows);print(f'[{ev.engine.mode}] {name}: {j+1}/{len(episodes)}',flush=True)
    report[name+'_summary']={}
    for family in sorted({e['family'] for e in episodes}):
        report[name+'_summary'][family]={}
        for strategy in ('full_sequential',)+strategies:
            sub=[r for r in rows if r['family']==family and r['strategy']==strategy and r['status']=='completed']
            if sub:
                ids={r['id'] for r in sub}
                matched=[r for r in rows if r['id'] in ids and r['strategy']=='full_sequential' and r['status']=='completed']
                report[name+'_summary'][family][strategy]={'quality':quality(sub,ev.engine.policies,cfg.bootstrap_repeats,cfg.seed),'parity':summarize_parity(sub),
                    'matched_reference_quality':quality(matched,ev.engine.policies,cfg.bootstrap_repeats,cfg.seed),
                    'comparison_scope':'Matched episode IDs; optimization subset is not compared against the larger reference population.'}
    if name=='context':
        by=[]
        for family in sorted({r['family'] for r in rows}):
            for target in cfg.context_min_tokens:
                for pos in ('first','last'):
                    sub=[r for r in rows if r['family']==family and r['context_target_min_tokens']==target and r['evidence_position']==pos and r['strategy']=='full_sequential' and r['status']=='completed']
                    if sub:by.append({'family':family,'context_target':target,'position':pos,'quality':quality(sub,ev.engine.policies,cfg.bootstrap_repeats,cfg.seed)})
        report['context_by_length_position']=by
    report[name+'_failures']=sum(r['status']!='completed' for r in rows)
    json_write(out/(name+'_rows.json'),rows)
    return refs,rows


def benchmarks(ev,episodes,refs,cfg,out):
    rows=[];strategies=['full_sequential','full_batch4','shared_lossless','shared_kv_fp16']
    for j,ep in enumerate(episodes):
        order=list(np.random.default_rng(cfg.seed+j).permutation(strategies))
        for strategy in order:
            r=record(ep,strategy)
            try:
                result,t=time_call(lambda:ev.run(ep,strategy),ev.engine.device,cfg.benchmark_repeats,cfg.benchmark_warmups)
                r.update(status='completed',**t,comparison=parity(refs[ep['id']],light(result),cfg.cache_probability_tolerance),metadata=result['meta'],scope='complete cold request, no offline score memo; fresh prefix and pack/restore included')
            except Exception as exc:r.update(safe_error(exc))
            rows.append(r)
        json_write(out/'benchmark_rows.json',rows)
    return rows


def traffic_events(episodes,cfg):
    # Single-worker trace with controlled virtual arrivals; no sleep or claimed production rate.
    unique={}
    for e in episodes:unique.setdefault(e['group'],[]).append(e)
    groups=sorted(unique,key=lambda g:content_hash([cfg.seed,g]))[:cfg.sizes()['traffic_messages']]
    pool=[sorted(unique[g],key=lambda e:e['id']) for g in groups]
    rng=np.random.default_rng(cfg.seed+101);events=[];now=0.
    for i in range(cfg.sizes()['traffic_requests']):
        # Hot reuse, scan misses and gaps crossing TTL; same trace for every strategy.
        ix=int(rng.integers(0,min(3,len(pool)))) if i%3 else (i//3)%len(pool)
        now+=cfg.prefix_cache_ttl_seconds+1 if i and i%12==0 else 1.
        ep=pool[ix][int(rng.integers(0,len(pool[ix])))];events.append({'id':ep['id'],'tenant':'tenant_b' if i%7==0 else 'tenant_a','arrival':now})
    return events


def traffic(ev,episodes,refs,cfg,out):
    lookup={e['id']:e for e in episodes};events=traffic_events(episodes,cfg);json_write(out/'traffic_events.json',events);rows=[];summary=[]
    variants=[('no_persistent','shared_lossless',False),('gpu_lru_lossless','shared_lossless',True),('gpu_lru_kv_fp16','shared_kv_fp16',True)]
    for repeat in range(cfg.traffic_repeats):
        order=list(np.random.default_rng(cfg.seed+repeat).permutation(len(variants)))
        for vi in order:
            name,strategy,persist=variants[int(vi)];clock=Clock();cache=PrefixLRU(cfg.prefix_cache_mib*1024**2,cfg.prefix_cache_ttl_seconds,clock) if persist else None;current=[]
            for i,event in enumerate(events):
                clock.now=event['arrival'];ep=lookup[event['id']];r=record(ep,strategy);r.update(trace_strategy=name,repeat=repeat,arrival=clock.now,tenant=event['tenant'],trace_index=i)
                try:
                    result,t=time_call(lambda:ev.run(ep,strategy,cache,event['tenant']),ev.engine.device,1,0)
                    r.update(status='completed',**t,outputs=light(result),metadata=result['meta'],comparison=parity(refs[ep['id']],light(result),cfg.cache_probability_tolerance))
                except Exception as exc:r.update(safe_error(exc))
                rows.append(r);current.append(r)
            valid=[r for r in current if r['status']=='completed']
            summary.append({'strategy':name,'repeat':repeat,'requests':len(current),'completed_requests':len(valid),'total_ms':sum(r['p50_ms'] for r in valid) if len(valid)==len(current) else None,
                'cache':cache.stats() if cache is not None else None,'parity':summarize_parity(valid),'scope':'single worker; virtual arrival clock tests expiry without wall-time sleeps; total timings include misses, no network/queue/concurrent requests'})
            if cache is not None:cache.clear()
            del cache;gc.collect()
            if torch.device(ev.engine.device).type=='cuda':torch.cuda.empty_cache()
            json_write(out/'traffic_rows.json',rows);json_write(out/'traffic_summary.json',summary)
            print(f'[{ev.engine.mode}] trace {repeat+1}/{cfg.traffic_repeats}: {name}',flush=True)
    return summary


def run(job):
    cfg=Settings(**job['configuration']);cfg.sizes();out=Path(job['out']);out.mkdir(parents=True,exist_ok=True);mode=job['mode']
    report={'version':VERSION,'mode':mode,'status':'running','stage_status':{},'memory_snapshots':[],'weights_frozen':True,'policies_frozen':True,'no_training':True}
    stage='contracts'
    def save():json_write(out/'worker_report.json',report)
    save()
    try:
        if not torch.__version__.startswith(cfg.require_torch_series):raise StageSkip(f'Reference Torch series {cfg.require_torch_series} required; found {torch.__version__}. Do not silently install a different CUDA stack.')
        report['indexing_selftest']=indexing_selftest();report['lifecycle_selftest']=lifecycle_selftest()
        report['tiny_suffix_batch_selftest']=tiny_expanded_cache_selftest();report['stage_status'][stage]='completed'
        stage='model_load'
        expcfg=ExpandedSettings(**{k:v for k,v in asdict(cfg).items() if k in {f.name for f in fields(ExpandedSettings)}})
        runtime_job=dict(job,configuration=asdict(expcfg),mode='fp32_strict_math' if mode.startswith('fp32') else mode)
        with controls(mode) as flags:
            model,head,tokenizer,none,device=load_runtime(runtime_job,report);report['flags']=flags
            if mode=='fp32_tf32_allowed' and torch.cuda.get_device_capability(device)[0]<8:raise StageSkip('TF32 experiment needs compute capability 8 or later.')
            engine=CacheEngine(ScoringEngine(model,head,tokenizer,copy.deepcopy(cfg),device,mode,none,job['policies']),CodecBank(device),job['tokenizer_sha256'],report['model']['implementation_sha256'])
            ev=Evaluator(engine,cfg);before=sample_parameter_digest(model);report['stage_status'][stage]='completed'
            stage='model_forward_smoke'
            with torch.inference_mode():
                smoke=forward_features(model,engine.native._pack([[17,19,23]]))
                report['model_forward_smoke']={'finite':bool(torch.isfinite(smoke).all()),'shape':list(smoke.shape)}
                if not report['model_forward_smoke']['finite']:raise IntegrityError('Loaded-model smoke produced non-finite features.')
                del smoke
            if before!=sample_parameter_digest(model):raise IntegrityError('Loaded-model smoke changed sampled weights.')
            report['stage_status'][stage]='completed';save()
            inputs=json.loads(Path(job['inputs']).read_text());allrefs={}
            if cfg.run_fresh_quality:
                stage='fresh_quality';ref,rows=evaluate_panel(ev,inputs['episodes'],set(inputs['parity_ids']) if cfg.run_cache_parity else set(),cfg,out,report,'fresh');allrefs.update(ref)
                report['stage_status'][stage]='completed' if not report['fresh_failures'] else 'partial';report['memory_snapshots'].append(memory_snapshot('after_fresh_quality',device));save()
            if cfg.run_labeled_context:
                stage='context';ref,rows=evaluate_panel(ev,inputs['context_episodes'],{e['id'] for e in inputs['context_episodes']} if cfg.run_cache_parity else set(),cfg,out,report,'context');allrefs.update(ref)
                report['stage_status'][stage]='completed' if not report['context_failures'] else 'partial';report['memory_snapshots'].append(memory_snapshot('after_context',device));save()
            lookup={e['id']:e for e in inputs['episodes']+inputs['context_episodes']}
            if cfg.run_request_benchmark:
                stage='benchmarks';eps=[lookup[i] for i in inputs['benchmark_ids']]
                for ep in eps:
                    if ep['id'] not in allrefs:allrefs[ep['id']]=light(ev.reference(ep))
                br=benchmarks(ev,eps,allrefs,cfg,out);report['benchmark_rows']=br;report['stage_status'][stage]='completed' if all(r['status']=='completed' for r in br) else 'partial';save()
            if cfg.run_traffic:
                stage='traffic';eps=[lookup[i] for i in inputs['traffic_ids']]
                for ep in eps:
                    if ep['id'] not in allrefs:allrefs[ep['id']]=light(ev.reference(ep))
                report['traffic_summary']=traffic(ev,eps,allrefs,cfg,out);report['stage_status'][stage]='completed' if all(x['requests']==x['completed_requests'] for x in report['traffic_summary']) else 'partial';save()
            stage='weight_integrity';report['sampled_weights_unchanged']=before==sample_parameter_digest(model)
            if not report['sampled_weights_unchanged']:raise IntegrityError('Sampled weights changed.')
            report['stage_status'][stage]='completed';report['memory_snapshots'].append(memory_snapshot('after_all_stages',device))
            report['status']='completed' if all(x=='completed' for x in report['stage_status'].values()) else 'partial'
    except Exception as exc:
        fatal=is_fatal_cuda(exc);report['status']='skipped' if isinstance(exc,StageSkip) else 'failed';report['stage_status'][stage]=report['status']
        report['error']={'stage':stage,'type':type(exc).__name__,'message':redact(str(exc),os.environ.get('HF_TOKEN'))[:2000],'fatal_cuda':fatal}
        (out/'first_error.txt').write_text(redact(traceback.format_exc(),os.environ.get('HF_TOKEN')));save();print(report['error'],flush=True)
    finally:save()
    return 0 if report['status']=='completed' else 2

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args();raise SystemExit(run(json.loads(Path(a.job).read_text())))
