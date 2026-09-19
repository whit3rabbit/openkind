"""One isolated Qwen precision per process; failures never masquerade as benchmark results."""
from __future__ import annotations
import argparse, copy, gc, json, os, sys, traceback, hashlib
from dataclasses import asdict, fields
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from phase2f_config import Settings, VERSION
from phase2f_upstream import upstream_selftest
from phase2f_storage import CodecBank, PrefixLRU, unique_bytes
from phase2f_execution import CacheEngine, packed_storage_report
from phase2d_common import json_write, content_hash, file_sha, episode_metrics, group_bootstrap, sync
from phase2e_core import IntegrityError, StageSkip
from phase2e_runtime import indexing_selftest, is_fatal_cuda
from phase2e_precision import sample_parameter_digest, backend_flags
from phase2e_cache import make_plan, TokenPlan, cache_digest
from phase2e_expansion import ExpandedSettings, balanced_subset, redact
from phase2e_expand_execution import ScoringEngine, PROTOTYPE, compare_views, tiny_expanded_cache_selftest, ComponentTimer
from phase2e_expand_worker import load_runtime, numerical_controls, memory_snapshot, time_call, brief_result


def comparison(reference,result,cfg):
    c=compare_views(reference['distributions'],result['distributions'],reference['actions'],result['actions'])
    c['accepted_within_sample']=(c['max_probability_delta']<=cfg.cache_probability_tolerance
                               and not c['any_argmax_changed'] and not c['any_policy_output_changed'])
    return c

def info(ep):
    return {'episode_id':ep['id'],'group':ep['group'],'split':ep['evaluation_split'],
            'K':len(ep['choices']),'sampler':ep['sampler'],'absent':bool(ep['true_intent_omitted'])}

def light(result):
    return {'scores':np.asarray(result['scores']).tolist(),
            'distributions':{k:np.asarray(p).tolist() for k,p in result['distributions'].items()},
            'actions':result['actions']}

def fail_record(error,**extra):
    if is_fatal_cuda(error):raise error
    return dict(status='failed',error={'type':type(error).__name__,
        'message':redact(str(error),os.environ.get('HF_TOKEN'))[:1000]},**extra)

def cached_result(engine,plan,root,verify=False):
    scores,meta=engine.suffixes(plan,root,4,verify=verify)
    d,a,_=engine.views(scores,plan)
    return {'scores':scores,'distributions':d,'actions':a,'meta':meta}


def baseline_panel(engine,panel,cfg,out,report):
    refs={};rows=[]
    for j,ep in enumerate(panel):
        full=engine.native.request(ep,'full_sequential')
        row=dict(info(ep),reference=light(full),strategies={})
        for name,fn in [('full_batch4',lambda:engine.native.request(ep,'full_batch4')),
                        ('shared_equal4',lambda:engine.native.request(ep,PROTOTYPE,verify=True)),
                        ('shared_equal8',lambda:engine.equal8(ep,verify=True))]:
            result=fn();row['strategies'][name]={'status':'completed','comparison':comparison(full,result,cfg),
                                              'outputs':light(result),'metadata':result['meta']}
        refs[ep['id']]=full;rows.append(row)
        json_write(out/'baseline_parity.json',rows)
        if j%8==0 or j+1==len(panel):print(f'[{engine.mode}] Uncompressed reference/parity {j+1}/{len(panel)}',flush=True)
    report['baseline_parity']=rows
    return refs


def compression_panel(engine,panel,refs,specs,cfg,out,report):
    rows=[]
    for j,ep in enumerate(panel):
        plan=make_plan(ep,engine.tokenizer,cfg,engine.mode)
        root=engine.prefill(plan);raw_digest=cache_digest(root)
        lossless=cached_result(engine,plan,root,verify=True)
        # Same original prefix for every codec: isolate quantization from prefill/chunking changes.
        for spec in specs:
            row=dict(info(ep),codec=spec,placement='gpu',status='completed')
            try:
                timer=ComponentTimer(engine.device,True)
                snap=timer.call('pack_ms',lambda:engine.bank.pack(root,spec,'gpu',verify=True))
                manifest=packed_storage_report(snap,engine.bank)
                restored=timer.call('restore_ms',lambda:engine.bank.restore(snap,verify=True))
                result=cached_result(engine,plan,restored,verify=True)
                row.update(storage=manifest,component_ms=dict(timer.ms),outputs=light(result),
                           versus_full=comparison(refs[ep['id']],result,cfg),
                           versus_uncompressed_same_chunking=comparison(lossless,result,cfg),
                           scope='KV snapshot round trip plus ordinary floating-point suffix attention; NOT low-bit attention kernels')
                # Re-order every fourth stratum, spread across message groups; same compressed root.
                if len(ep['choices'])==4 and ep['true_intent_omitted'] and ep['sampler']=='label_lexical_hard':
                    repeat_root=engine.bank.restore(snap,verify=True)
                    reverse,meta=engine.suffixes(plan,repeat_root,4,verify=True,order=list(reversed(range(len(plan.full_ids)))))
                    rd,ra,_=engine.views(reverse,plan)
                    c=compare_views(result['distributions'],rd,result['actions'],ra)
                    row['order_check']=c
                    row['order_check_passed']=(c['max_probability_delta']<=cfg.order_probability_tolerance and not c['any_policy_output_changed'])
                    del repeat_root
                if cache_digest(root)!=raw_digest:raise IntegrityError('Compression corrupted the shared original root.')
                del snap,restored,result
            except Exception as e:row.update(fail_record(e))
            rows.append(row);json_write(out/'compression_parity.json',rows)
        del root,lossless
        if j%4==0 or j+1==len(panel):print(f'[{engine.mode}] Packed-cache parity {j+1}/{len(panel)}',flush=True)
    report['compression_rows']=rows
    # Metrics are execution regression on archived labels, not a new training/quality benchmark.
    epmap={e['id']:e for e in panel};quality=[]
    for spec in specs:
        matched=[r for r in rows if r['codec']==spec and r['status']=='completed']
        for split in sorted({e['evaluation_split'] for e in panel}):
            r=[x for x in matched if x['split']==split]
            if not r:continue
            eps=[epmap[x['episode_id']] for x in r]
            for head,none in engine.none_models.items():
                logits=none.logits([np.asarray(x['outputs']['scores']) for x in r])
                quality.append({'codec':spec,'split':split,'none_head':head,
                    'distinct_messages':len({e['group'] for e in eps}),
                    'metrics':episode_metrics(logits,eps),
                    'scope':'frozen-panel execution regression, not independent generalization'})
    report['compression_quality_regression']=quality;json_write(out/'compression_quality.json',quality)


def request_benchmarks(engine,panel,refs,specs,cfg,out,report):
    rows=[];profiles=[]
    for j,ep in enumerate(panel):
        names=['full_sequential','full_batch4','shared_equal4','shared_equal8']
        specs_here=specs if cfg.run_cache_compression else []
        jobs=[(name,None,None) for name in names]+[(f'cold_snapshot/{s}',s,None) for s in specs_here]
        jobs=jobs[j%len(jobs):]+jobs[:j%len(jobs)]
        for name,spec,_ in jobs:
            def call(profile=False):
                if spec is not None:return engine.request(ep,spec,profile=profile)
                if name=='shared_equal8':return engine.equal8(ep)
                return engine.native.request(ep,PROTOTYPE if name=='shared_equal4' else name,profile=profile)
            row=dict(info(ep),strategy=name,temperature_state='cold_prefix',status='completed')
            try:
                result,t=time_call(call,engine.device,cfg.benchmark_repeats,cfg.benchmark_warmups)
                row.update(t,comparison=comparison(refs[ep['id']],result,cfg),metadata=result['meta'],
                    scope='COMPLETE cold request; tokenize, prefill, pack/restore if enabled, suffixes, heads, policies, JSON; codec tables initialized separately')
                if j<4 and spec is not None:
                    measures=[call(profile=True)['meta'] for _ in range(cfg.profile_repeats)]
                    profiles.append(dict(info(ep),strategy=name,raw_profiles=measures,
                       scope='intrusive synchronized wall-time decomposition; never substituted for request timing'))
            except Exception as e:row.update(fail_record(e))
            rows.append(row);json_write(out/'request_benchmarks.json',rows)
        print(f'[{engine.mode}] Cold request timing {j+1}/{len(panel)}',flush=True)
    report['benchmark_rows']=rows;report['component_profiles']=profiles
    json_write(out/'component_profiles.json',profiles)


def trace_panels(episodes,rounds,seed):
    """Controlled traces, not claimed production traffic. Same multiset per workload."""
    by={}
    for e in episodes:by.setdefault(e['group'],[]).append(e)
    locality=[]
    for g in sorted(by):
        seq=sorted(by[g],key=lambda e:e['id'])
        locality.extend(seq*rounds)
    perm=np.random.default_rng(seed).permutation(len(locality))
    return {'grouped_locality':locality,'shuffled_same_requests':[locality[int(i)] for i in perm]}


def cross_request(engine,panel,refs,specs,cfg,out,report):
    variants=[('no_persistent_cache','lossless','gpu',False),('gpu_lru_lossless','lossless','gpu',True)]
    if cfg.run_cpu_offload:variants.append(('cpu_lru_lossless','lossless','cpu',True))
    if 'tq_k3_v4_r32' in specs:variants.append(('gpu_lru_tq_k3_v4_r32','tq_k3_v4_r32','gpu',True))
    rows=[];aggregate=[]
    for workload,trace in trace_panels(panel,cfg.workload_rounds,cfg.seed).items():
        for name,spec,placement,persistent in variants:
            cache=PrefixLRU(cfg.prefix_cache_mib*1024**2,cfg.prefix_cache_ttl_seconds) if persistent else None
            # Warm only kernels with an unrelated request, NEVER prime the measured cache.
            engine.request(trace[0],'lossless')
            start=len(rows)
            for j,ep in enumerate(trace):
                tenant='tenant_a'  # Exact key isolation is tested independently; no cross-tenant reuse.
                row=dict(info(ep),workload=workload,strategy=name,trace_index=j,status='completed')
                try:
                    result,t=time_call(lambda:engine.request(ep,spec,placement,cache,tenant=tenant),engine.device,1,0)
                    row.update(t,comparison=comparison(refs[ep['id']],result,cfg),metadata=result['meta'])
                except Exception as e:row.update(fail_record(e))
                rows.append(row)
                if j%8==0:json_write(out/'cross_request_rows.json',rows)
            measured=rows[start:];valid=[r for r in measured if r['status']=='completed']
            aggregate.append({'workload':workload,'strategy':name,'requests':len(measured),
               'completed_requests':len(valid),'distinct_messages':len({r['group'] for r in measured}),
               'total_request_ms':sum(r['p50_ms'] for r in valid),
               'p50_request_ms':float(np.median([r['p50_ms'] for r in valid])) if valid else None,
               'hit_p50_ms':float(np.median([r['p50_ms'] for r in valid if r['metadata']['cache_hit']])) if any(r['metadata']['cache_hit'] for r in valid) else None,
               'miss_p50_ms':float(np.median([r['p50_ms'] for r in valid if not r['metadata']['cache_hit']])) if any(not r['metadata']['cache_hit'] for r in valid) else None,
               'failed_requests':len(measured)-len(valid),
               'parity':{'maximum_probability_delta':max((r['comparison']['max_probability_delta'] for r in valid),default=None),
                 'requests_with_selected_outcome_change':sum(r['comparison']['any_argmax_changed'] for r in valid),
                 'requests_with_answer_review_change':sum(r['comparison']['any_policy_action_changed'] for r in valid),
                 'requests_with_any_policy_output_change':sum(r['comparison']['any_policy_output_changed'] for r in valid),
                 'accepted_requests':sum(r['comparison']['accepted_within_sample'] for r in valid)},
               'final_cache':cache.stats() if cache else None,'codec_shared_bytes':engine.bank.bytes_for(spec),
               'scope':'single-worker measured trace, one timing per request; cold-start population included; no output/logit memoization'})
            if cache:cache.clear()
            del cache
            gc.collect();torch.cuda.empty_cache() if torch.device(engine.device).type=='cuda' else None
            json_write(out/'cross_request_summary.json',aggregate)
            print(f'[{engine.mode}] Traffic {workload}: {name}',flush=True)
    report['cross_request_summary']=aggregate
    json_write(out/'cross_request_rows.json',rows)


def long_prefix(engine,specs,cfg,out,report):
    rows=[]
    token=engine.tokenizer.encode('information',add_special_tokens=False)[0]
    tails=[engine.tokenizer.encode(f' Candidate {i}: match assessment.',add_special_tokens=False) for i in range(8)]
    for n in cfg.long_prefix_lengths:
        prefix=[token]*n;plan=TokenPlan([prefix+s for s in tails],prefix,tails,[f'c{i}' for i in range(8)],[False]*8,f'long:{n}')
        full,_=engine.native.score_plan(plan,'full_sequential');d,a,_=engine.views(full,plan)
        ref={'distributions':d,'actions':a,'scores':full}
        for strategy in ('full_sequential','full_batch4',PROTOTYPE):
            try:
                (scores,meta),t=time_call(lambda:engine.native.score_plan(plan,strategy),engine.device,cfg.long_prefix_repeats,1)
                dd,aa,_=engine.views(scores,plan)
                rows.append(dict(prefix_tokens=n,K=8,strategy=strategy,status='completed',**t,
                    comparison=comparison(ref,{'distributions':dd,'actions':aa},cfg),metadata=meta))
            except Exception as e:rows.append(fail_record(e,prefix_tokens=n,K=8,strategy=strategy))
        for spec in specs:
            for temp in ('cold','warm'):
                row=dict(prefix_tokens=n,K=8,strategy=f'{temp}_snapshot/{spec}',status='completed')
                cache=PrefixLRU(cfg.prefix_cache_mib*1024**2,cfg.prefix_cache_ttl_seconds) if temp=='warm' else None
                try:
                    fill_ms=None
                    if cache is not None:
                        sync(engine.device);t0=perf_counter();engine.cached_plan(plan,spec,cache=cache);sync(engine.device);fill_ms=(perf_counter()-t0)*1000
                    def call():
                        s,m=engine.cached_plan(plan,spec,cache=cache)
                        dd,aa,_=engine.views(s,plan)
                        return {'scores':s,'distributions':dd,'actions':aa,'meta':m}
                    result,t=time_call(call,engine.device,cfg.long_prefix_repeats,1)
                    row.update(t,comparison=comparison(ref,result,cfg),metadata=result['meta'],
                      population_request_ms=fill_ms,scope='pretokenized synthetic mechanics; not long-context quality; warm population separately reported')
                    if cache is not None and not result['meta']['cache_hit']:row['status']='cache_not_admitted'
                except Exception as e:row.update(fail_record(e))
                rows.append(row)
                if cache:cache.clear()
                del cache
        json_write(out/'long_prefix_rows.json',rows)
        print(f'[{engine.mode}] Long-prefix mechanics: {n}',flush=True)
    report['long_prefix_rows']=rows


def main(job):
    cfg=Settings(**job['configuration']);cfg.sizes();out=Path(job['out']);out.mkdir(parents=True,exist_ok=True)
    inputs=json.loads(Path(job['inputs']).read_text());stage='setup'
    report={'version':VERSION,'mode':job['mode'],'status':'running','stage_status':{},'memory_snapshots':[],
      'frozen_weights':True,'frozen_policies':True,'no_generation':True,'mtp':{'status':'not_applicable','generated_tokens':0,
      'reason':'MTP drafts unknown autoregressive output tokens; every candidate suffix token here is already supplied.'}}
    save=lambda:json_write(out/'worker_report.json',report)
    def completed(name):report['stage_status'][name]='completed';save()
    try:
        stage='indexing_selftest';report[stage]=indexing_selftest();completed(stage)
        stage='tiny_qwen_cache_api';report[stage]=tiny_expanded_cache_selftest();completed(stage)
        tq_ok=False
        stage='upstream_codec_api'
        if cfg.run_turboquant and job.get('upstream_root'):
            try:
                t=perf_counter();report[stage]=upstream_selftest(job['upstream_root']);report[stage]['seconds']=perf_counter()-t;tq_ok=True;completed(stage)
            except Exception as e:
                report[stage]=fail_record(e);report['stage_status'][stage]='failed';save()
        else:report['stage_status'][stage]='disabled' if not cfg.run_turboquant else 'unavailable';save()
        stage='model_load'
        # Reuse the successful E loader; do not mutate its class or install an alternate inference stack.
        allowed={f.name for f in fields(ExpandedSettings)}
        basecfg={k:v for k,v in asdict(cfg).items() if k in allowed}
        loader_job={'configuration':basecfg,'mode':job['mode'],'out':str(out),'source_root':job['source_root']}
        model,head,tok,none,device=load_runtime(loader_job,report)
        native=ScoringEngine(model,head,tok,cfg,device,job['mode'],none,inputs['policies'])
        bank=CodecBank(device,job.get('upstream_root') if tq_ok else None,cfg.quantization_seed)
        engine=CacheEngine(native,bank,inputs['tokenizer_sha256'],content_hash([report['model']['implementation_sha256'],job['code_sha256']]))
        completed(stage)
        with numerical_controls(job['mode']):
            report['numerical_flags']=backend_flags();before=sample_parameter_digest(model)
            stage='model_forward_smoke';native.request(inputs['panel'][0],'full_sequential');completed(stage)
            specs=[s for s in cfg.codec_specs if not s.startswith('tq_') or tq_ok]
            if not cfg.run_cache_compression:specs=['lossless']
            stage='codec_table_initialization';t=perf_counter();bank.warm_for_model(model,specs);sync(device)
            report[stage]={'seconds':perf_counter()-t,'active_specs':specs,'shared_codec_tensor_bytes':unique_bytes(bank.buffers()),
                'cold_setup_excluded_from_steady_state_request_timings':True,
                'packing_note':'Upstream nominal 4-bit Prod uses 3-bit MSE rounded to 4-bit storage plus one sign bit; bytes are measured, not inferred from labels.'};completed(stage)
            stage='reference_parity';refs=baseline_panel(engine,inputs['panel'],cfg,out,report);completed(stage)
            eps={e['id']:e for e in inputs['panel']}
            stage='compression_parity'
            if cfg.run_cache_compression:compression_panel(engine,[eps[i] for i in inputs['compression_ids']],refs,specs,cfg,out,report);completed(stage)
            else:report['stage_status'][stage]='disabled'
            report['memory_snapshots'].append(memory_snapshot('after_cache_parity',device))
            stage='cold_request_benchmarks'
            if cfg.run_request_benchmarks:request_benchmarks(engine,[eps[i] for i in inputs['benchmark_ids']],refs,specs,cfg,out,report);completed(stage)
            else:report['stage_status'][stage]='disabled'
            stage='cross_request_cache'
            if cfg.run_cross_request_cache:cross_request(engine,[eps[i] for i in inputs['benchmark_ids']],refs,specs,cfg,out,report);completed(stage)
            else:report['stage_status'][stage]='disabled'
            stage='long_prefix'
            if cfg.run_long_prefix:long_prefix(engine,specs,cfg,out,report);completed(stage)
            else:report['stage_status'][stage]='disabled'
            stage='weight_integrity'
            if sample_parameter_digest(model)!=before:raise IntegrityError('Frozen model parameter sample changed.')
            completed(stage);report['memory_snapshots'].append(memory_snapshot('before_exit',device))
            failures=any(r.get('status')=='failed' for key in ('compression_rows','benchmark_rows','long_prefix_rows') for r in report.get(key,[]))
            failures=failures or any(r.get('failed_requests',0)>0 for r in report.get('cross_request_summary',[]))
            report['status']='partial' if failures or any(s in ('failed','unavailable') for s in report['stage_status'].values()) else 'completed'
        save();return 0
    except BaseException as e:
        status='skipped' if isinstance(e,StageSkip) else 'failed'
        report['stage_status'][stage]=status;report['status']='partial' if report.get('stage_status',{}).get('reference_parity')=='completed' else status
        report['error']={'stage':stage,'type':type(e).__name__,'message':redact(str(e),os.environ.get('HF_TOKEN'))[:1500],
                        'fatal_cuda':is_fatal_cuda(e),'worker_will_exit':True}
        trace=redact(''.join(traceback.format_exception(type(e),e,e.__traceback__)),os.environ.get('HF_TOKEN'))
        (out/'worker_error.txt').write_text(trace);save();print(trace,flush=True);return 2

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args()
    sys.exit(main(json.loads(Path(a.job).read_text())))
