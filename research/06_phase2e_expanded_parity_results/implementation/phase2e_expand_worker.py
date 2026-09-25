"""Foreground worker entry point. One newly loaded precision mode per process; exit frees CUDA.
This module never forks, loads two large models, or executes code from prior result archives.
"""
from __future__ import annotations
import argparse, gc, hashlib, importlib.metadata as im, inspect, json, os, platform, sys, traceback
from collections import Counter
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
import numpy as np
import torch
from safetensors.torch import load_file
from phase2e_expansion import ExpandedSettings, VERSION, redact
from phase2d_common import CandidateHead, FixedHead, json_write, json_ready, sync, require_finite, file_sha, pack_tokens, forward_features
from phase2d_decisions import NoneModel
from phase2e_core import StageSkip, IntegrityError
from phase2e_cache import make_plan,TokenPlan,common_prefix_length,cache_tensor_records
from phase2e_precision import backend_flags,set_flags,sample_parameter_digest
from phase2e_runtime import indexing_selftest,is_fatal_cuda,validate_token_rows
from phase2e_expand_execution import (ScoringEngine,ComponentTimer,BASE_STRATEGIES,PROTOTYPE,
    probability_and_policy_views,compare_views,tiny_expanded_cache_selftest)


def memory_snapshot(label,device='cuda:0'):
    result={'label':label,'pid':os.getpid()}
    if torch.device(device).type!='cuda':return dict(result,device=str(device),cuda=False)
    sync(device);free,total=torch.cuda.mem_get_info(device)
    result.update(device=str(device),cuda=True,driver_free_gib=free/1024**3,total_gib=total/1024**3,
      allocated_mib=torch.cuda.memory_allocated(device)/1024**2,
      reserved_mib=torch.cuda.memory_reserved(device)/1024**2,
      peak_allocated_mib=torch.cuda.max_memory_allocated(device)/1024**2)
    result['reserved_unallocated_mib']=result['reserved_mib']-result['allocated_mib']
    try:
        import psutil
        result['host_rss_gib']=psutil.Process().memory_info().rss/1024**3
    except ImportError:pass
    return result


def load_preflight(snapshot,parameter_count,dtype_bytes,reserve_gib,allocation_limit_mib):
    required=parameter_count*dtype_bytes/1024**3+reserve_gib
    result={'weight_only_gib':parameter_count*dtype_bytes/1024**3,'workspace_reserve_gib':reserve_gib,
      'required_free_gib':required,'driver_free_gib':snapshot.get('driver_free_gib'),
      'existing_process_allocated_mib':snapshot.get('allocated_mib'),
      'policy':'direct target-dtype load, no BF16 model to convert, no CPU/offload fallback'}
    if snapshot.get('allocated_mib',0)>allocation_limit_mib:
        raise StageSkip('Worker already owns significant CUDA tensors before model loading; isolation check failed.')
    if snapshot.get('driver_free_gib',float('inf'))<required:
        raise StageSkip(f"Fresh-load memory guard: {snapshot['driver_free_gib']:.2f} GiB free, {required:.2f} GiB required (weights + workspace). Use a larger GPU; no offload.")
    return result


@contextmanager
def numerical_controls(mode):
    old=backend_flags()
    try:
        if mode!='bf16_default':
            flags=dict(old);flags['float32_matmul_precision']='highest'
            for k in flags:
                if k.endswith(('allow_tf32','allow_bf16_reduced_precision_reduction','allow_fp16_reduced_precision_reduction','benchmark')):flags[k]=False
                if k.endswith('deterministic'):flags[k]=True
            set_flags(flags)
            from torch.nn.attention import sdpa_kernel,SDPBackend
            context=sdpa_kernel(SDPBackend.MATH)
        else:context=nullcontext()
        with context:yield backend_flags()
    finally:set_flags(old)


def load_runtime(job,report):
    cfg=ExpandedSettings(**job['configuration']);out=Path(job['out']);source=Path(job['source_root'])
    if not torch.cuda.is_available():raise RuntimeError('A GPU Colab runtime is required; no CPU fallback for the 4B experiment.')
    device=torch.device('cuda:0')
    if job['mode'].startswith('bf16') and not torch.cuda.is_bf16_supported(including_emulation=False):
        raise StageSkip('Native BF16 unavailable. Use an L4 or A100 for the reference run.')
    dtype=torch.float32 if job['mode']=='fp32_strict_math' else torch.bfloat16
    report['environment']={'pid':os.getpid(),'python':platform.python_version(),'gpu':torch.cuda.get_device_name(device),
      'torch':torch.__version__,'dtype':str(dtype),'total_gpu_gib':torch.cuda.get_device_properties(device).total_memory/1024**3,
      'packages':{n:im.version(n) for n in ('transformers','accelerate','safetensors','numpy','scipy')}}
    if im.version('transformers')!='5.17.0':raise IntegrityError('This cache adapter is pinned to Transformers 5.17.0.')
    before=memory_snapshot('fresh_worker_before_model',device);report['memory_snapshots'].append(before)
    report['load_preflight']=load_preflight(before,4205751296,4 if dtype==torch.float32 else 2,cfg.fp32_workspace_gib,cfg.initial_gpu_allocation_limit_mib)
    json_write(out/'worker_report.json',report)
    from transformers import AutoTokenizer,AutoModelForCausalLM
    tokenizer=AutoTokenizer.from_pretrained(str(source/'export/tokenizer'),local_files_only=True,trust_remote_code=False)
    print(f"[{job['mode']}] Directly loading ONE text model as {dtype}; fresh free memory {before['driver_free_gib']:.2f} GiB.",flush=True)
    lm,info=AutoModelForCausalLM.from_pretrained(cfg.model_id,revision=cfg.model_revision,token=os.environ.get('HF_TOKEN'),
        trust_remote_code=False,dtype=dtype,device_map={'':0},attn_implementation='sdpa',output_loading_info=True)
    info=json_ready(info);json_write(out/'loading_info.json',info)
    if type(lm).__name__!='Qwen3_5ForCausalLM' or type(lm.model).__name__!='Qwen3_5TextModel':raise IntegrityError('Wrong loaded model graph.')
    if any(info.get(k) for k in ('missing_keys','mismatched_keys','error_msgs')):raise IntegrityError('Incomplete checkpoint load.')
    if any('visual' in name for name,_ in lm.named_modules()):raise IntegrityError('Vision module unexpectedly loaded.')
    model=lm.model;del lm;model.eval()
    for p in model.parameters():p.requires_grad_(False)
    params=sum(p.numel() for p in model.parameters())
    if params!=4205751296:raise IntegrityError('Unexpected text parameter count.')
    if any(p.dtype!=dtype or p.device!=device for p in model.parameters()):raise IntegrityError('Wrong dtype or CPU/offloaded parameter detected.')
    report['model']={'id':cfg.model_id,'revision':cfg.model_revision,'parameters':params,'dtype':str(dtype),
      'class':type(model).__name__,'implementation_sha256':file_sha(inspect.getfile(type(model))),
      'parameter_dtype_counts':dict(Counter(str(p.dtype) for p in model.parameters())),
      'buffer_dtype_counts':dict(Counter(str(b.dtype) for b in model.buffers())),
      'load_strategy':'direct_target_dtype_in_fresh_subprocess','cpu_offload':False,'generation':False}
    head=CandidateHead(model.config.hidden_size);head.load_state_dict(load_file(str(source/'export/frozen_candidate_head.safetensors'),device='cpu'));head.eval()
    for p in head.parameters():p.requires_grad_(False)
    none_models={name:NoneModel(**json.loads((source/f'export/none_{name}.json').read_text())) for name in ('frozen_global','refit_global','set_linear')}
    report['memory_snapshots'].append(memory_snapshot('after_direct_load',device))
    gc.collect();torch.cuda.empty_cache()
    report['memory_snapshots'].append(memory_snapshot('after_releasing_unused_allocator_blocks',device))
    return model,head,tokenizer,none_models,device


def enabled_strategies(cfg,prototype_ok):
    return list(BASE_STRATEGIES)+([PROTOTYPE] if cfg.run_suffix_batch_prototype and prototype_ok else [])


def brief_result(result):
    return {'scores':result['scores'].tolist(),
      'distributions':{n:p.tolist() for n,p in result['distributions'].items()},'actions':result['actions']}


def run_parity(engine,panel,cfg,out,prototype_ok,report):
    rows=[];strategies=enabled_strategies(cfg,prototype_ok)
    # One selected episode per distinct message gets full reversed-order/repeat isolation checks.
    isolation_ids={next(e['id'] for e in panel if e['group']==g and len(e['choices'])==4 and e['sampler']=='label_lexical_hard' and e['true_intent_omitted'])
                   for g in {e['group'] for e in panel}}
    for j,ep in enumerate(panel):
        record={'episode_id':ep['id'],'group':ep['group'],'split':ep['evaluation_split'],'K':len(ep['choices']),
            'sampler':ep['sampler'],'absent':ep['true_intent_omitted'],'strategies':{}}
        reference=engine.request(ep,'full_sequential')
        record['reference']=brief_result(reference)
        plan=make_plan(ep,engine.tokenizer,cfg,engine.mode)
        record['token_ids_sha256']=hashlib.sha256(json.dumps(plan.full_ids,separators=(',',':')).encode()).hexdigest()
        record['any_prompt_truncated']=any(plan.truncated)
        for strategy in strategies[1:]:
            result=engine.request(ep,strategy,verify=strategy.startswith('shared_'))
            compare=compare_views(reference['distributions'],result['distributions'],reference['actions'],result['actions'])
            compare['within_probability_tolerance']=compare['max_probability_delta']<=cfg.cache_probability_tolerance
            compare['same_selected_classes']=not compare['any_argmax_changed']
            compare['same_policy_outputs']=not compare['any_policy_output_changed']
            compare['accepted_within_sample']=compare['within_probability_tolerance'] and compare['same_selected_classes'] and compare['same_policy_outputs']
            record['strategies'][strategy]=dict(compare,outputs=brief_result(result),metadata=result['meta'])
            if strategy.startswith('shared_') and ep['id'] in isolation_ids:
                # New root per replay; every root independently verified after all branches.
                reverse,_=engine.score_plan(plan,strategy,verify=True,order=list(reversed(range(len(plan.full_ids)))))
                repeat,_=engine.score_plan(plan,strategy,verify=True)
                d_reverse,a_reverse=probability_and_policy_views(reverse,engine.none_models,engine.policies,plan.candidate_ids)
                d_repeat,a_repeat=probability_and_policy_views(repeat,engine.none_models,engine.policies,plan.candidate_ids)
                rev=compare_views(result['distributions'],d_reverse,result['actions'],a_reverse)
                rep=compare_views(result['distributions'],d_repeat,result['actions'],a_repeat)
                record['strategies'][strategy]['isolation']={'root_unchanged':True,
                   'reverse_order_max_probability_delta':rev['max_probability_delta'],
                   'repeat_max_probability_delta':rep['max_probability_delta'],
                   'reverse_order_policy_changed':rev['any_policy_output_changed'],
                   'repeat_policy_changed':rep['any_policy_output_changed']}
                if (max(rev['max_probability_delta'],rep['max_probability_delta'])>cfg.order_probability_tolerance or
                    rev['any_policy_output_changed'] or rep['any_policy_output_changed']):
                    record['strategies'][strategy]['accepted_within_sample']=False
        rows.append(record)
        json_write(out/'parity_rows.json',rows)
        if j%8==0 or j+1==len(panel):print(f'[{engine.mode}] Cache/policy parity {j+1}/{len(panel)} episodes.',flush=True)
    report['parity_rows']=rows
    return rows


def run_profiles(engine,episodes,cfg,out,prototype_ok,report):
    rows=[]
    for j,ep in enumerate(episodes):
        for strategy in enabled_strategies(cfg,prototype_ok):
            for _ in range(cfg.benchmark_warmups):engine.request(ep,strategy)
            measures=[]
            for _ in range(cfg.profile_repeats):measures.append(engine.request(ep,strategy,profile=True)['meta'])
            keys=set().union(*(x['component_ms'] for x in measures))
            row={'episode_id':ep['id'],'group':ep['group'],'split':ep['evaluation_split'],
              'K':len(ep['choices']),'sampler':ep['sampler'],'absent':ep['true_intent_omitted'],
              'strategy':strategy,'status':'completed','repeats':cfg.profile_repeats,
              'component_ms':{k:float(np.median([x['component_ms'].get(k,0.) for x in measures])) for k in sorted(keys)},
              'profiled_total_p50_ms':float(np.median([x['profiled_total_ms'] for x in measures])),
              'unattributed_host_p50_ms':float(np.median([x['unattributed_host_ms'] for x in measures])),
              'raw_profiles':measures,
              'scope':'synchronized component WALL times (intrusive); NOT pure kernel durations; not added to independent latency'}
            rows.append(row);json_write(out/'component_profiles.json',rows)
        print(f'[{engine.mode}] Component profile {j+1}/{len(episodes)} episodes.',flush=True)
    report['profile_rows']=rows
    return rows


def time_call(fn,device,repeats,warmups):
    for _ in range(warmups):fn()
    sync(device)
    cuda=torch.device(device).type=='cuda'
    baseline=torch.cuda.memory_allocated(device) if cuda else 0
    if cuda:torch.cuda.reset_peak_memory_stats(device)
    times=[];last=None
    for _ in range(repeats):
        sync(device);t=perf_counter();last=fn();sync(device);times.append((perf_counter()-t)*1000)
    return last,{'p50_ms':float(np.median(times)),'p95_ms':float(np.quantile(times,.95)),
      'times_ms':times,'warmups':warmups,'repeats':repeats,
      'resident_mib':baseline/1024**2 if cuda else None,
      'peak_extra_mib':(torch.cuda.max_memory_allocated(device)-baseline)/1024**2 if cuda else None}


def run_benchmarks(engine,episodes,cfg,out,prototype_ok,report):
    rows=[];strategies=enabled_strategies(cfg,prototype_ok)
    for j,ep in enumerate(episodes):
        reference=engine.request(ep,'full_sequential')
        # Rotate strategy order rather than confounding every strategy with a fixed time position.
        ordered=strategies[j%len(strategies):]+strategies[:j%len(strategies)]
        for strategy in ordered:
            result,timing=time_call(lambda:engine.request(ep,strategy),engine.device,cfg.benchmark_repeats,cfg.benchmark_warmups)
            cmp=compare_views(reference['distributions'],result['distributions'],reference['actions'],result['actions'])
            row={'episode_id':ep['id'],'group':ep['group'],'split':ep['evaluation_split'],
              'K':len(ep['choices']),'sampler':ep['sampler'],'absent':ep['true_intent_omitted'],
              'strategy':strategy,'status':'completed',**timing,'comparison':cmp,'metadata':result['meta'],
              'scope':'uninstrumented cold COMPLETE Python request; tokenizer/transfer/prefill/copy/scoring/policies/JSON included; no HTTP'}
            rows.append(row);json_write(out/'request_benchmarks.json',rows)
        print(f'[{engine.mode}] Complete-request benchmark {j+1}/{len(episodes)} episodes.',flush=True)
    report['benchmark_rows']=rows
    return rows


def run_long_prefix(engine,cfg,out,prototype_ok,report):
    rows=[];k=8
    # Exact token sequences, not decoded/re-tokenized, isolate cache mechanics.
    token=engine.tokenizer.encode('information',add_special_tokens=False)[0]
    tails=[engine.tokenizer.encode(f' Candidate {i}: match assessment.',add_special_tokens=False) for i in range(k)]
    for length in cfg.long_prefix_lengths:
        prefix=[token]*int(length)
        plan=TokenPlan([prefix+t for t in tails],prefix,tails,[f'c{i}' for i in range(k)],[False]*k,f'synthetic:{length}')
        full,_=engine.score_plan(plan,'full_sequential')
        pf,af=probability_and_policy_views(full,engine.none_models,engine.policies,plan.candidate_ids)
        for strategy in enabled_strategies(cfg,prototype_ok):
            (scores,meta),timing=time_call(lambda:engine.score_plan(plan,strategy),engine.device,cfg.long_prefix_repeats,1)
            p,a=probability_and_policy_views(scores,engine.none_models,engine.policies,plan.candidate_ids)
            timer=ComponentTimer(engine.device,True);engine.score_plan(plan,strategy,timer)
            rows.append({'prefix_tokens':length,'K':k,'strategy':strategy,'status':'completed',**timing,
               'comparison':compare_views(pf,p,af,a),'component_profile_ms':dict(timer.ms),'metadata':meta,
               'scope':'pretokenized synthetic mechanics only, cold prefill/copies included; NOT a semantic-quality result'})
            json_write(out/'long_prefix.json',rows)
        print(f'[{engine.mode}] Long-prefix mechanics: {length} tokens.',flush=True)
    report['long_prefix']=rows
    return rows


def summarize_parity(rows,strategies):
    output={}
    for strategy in strategies:
        items=[r['strategies'][strategy] for r in rows if strategy in r['strategies']]
        if not items:continue
        output[strategy]={'episodes':len(items),'messages':len({r['group'] for r in rows if strategy in r['strategies']}),
          'max_probability_delta':max(i['max_probability_delta'] for i in items),
          'episodes_over_probability_tolerance':sum(not i['within_probability_tolerance'] for i in items),
          'episodes_with_class_change':sum(i['any_argmax_changed'] for i in items),
          'episodes_with_answer_review_change':sum(i['any_policy_action_changed'] for i in items),
          'episodes_with_any_policy_output_change':sum(i['any_policy_output_changed'] for i in items),
          'all_accepted_within_sample':all(i['accepted_within_sample'] for i in items)}
    return output


def main(job):
    out=Path(job['out']);out.mkdir(parents=True,exist_ok=True)
    inputs=json.loads(Path(job['inputs']).read_text());job=dict(job,source_root=inputs['source_root'])
    cfg=ExpandedSettings(**job['configuration'])
    report={'version':VERSION,'mode':job['mode'],'status':'running','pid':os.getpid(),
      'stage_status':{},'memory_snapshots':[],'policy_sha256':inputs['policy_sha256'],
      'panel_sha256':inputs['panel_audit']['episodes_sha256'],'weights_frozen':True,
      'policies_frozen':True,'no_training':True}
    def save():json_write(out/'worker_report.json',report)
    active='indexing_selftest';save()
    try:
        report[active]=indexing_selftest();report['stage_status'][active]='completed'
        active='tiny_cache_api_selftest'
        from phase2e_workflow import tiny_qwen_cache_selftest
        report[active]=tiny_qwen_cache_selftest();report['stage_status'][active]='completed'
        active='tiny_suffix_batch_selftest';prototype_ok=False
        if cfg.run_suffix_batch_prototype:
            try:
                report[active]=tiny_expanded_cache_selftest();prototype_ok=True;report['stage_status'][active]='completed'
            except Exception as e:
                report[active]={'status':'failed','error':redact(str(e),os.environ.get('HF_TOKEN'))}
                report['stage_status'][active]='failed'
                print('Suffix-batch prototype disabled after CPU API failure; base comparisons remain enabled.',flush=True)
        else:report['stage_status'][active]='disabled'
        active='model_load';save()
        with numerical_controls(job['mode']) as flags:
            model,head,tokenizer,none_models,device=load_runtime(job,report)
            report['flags']=flags;report['stage_status'][active]='completed';save()
            engine=ScoringEngine(model,head,tokenizer,cfg,device,job['mode'],none_models,inputs['policies'])
            active='model_forward_smoke'
            before=sample_parameter_digest(model)
            smoke=engine.request(inputs['panel'][0],'full_sequential')
            if sample_parameter_digest(model)!=before:raise IntegrityError('Frozen parameter sample changed.')
            report['stage_status'][active]='completed';report['model_forward_smoke']={'finite':True,'weights_sample_unchanged':True}
            report['memory_snapshots'].append(memory_snapshot('after_forward_smoke',device));save()
            active='cache_policy_parity'
            if cfg.run_cache_policy_parity:
                rows=run_parity(engine,inputs['panel'],cfg,out,prototype_ok,report)
                report['parity_summary']=summarize_parity(rows,enabled_strategies(cfg,prototype_ok)[1:])
                report['stage_status'][active]='completed'
            else:report['stage_status'][active]='disabled'
            report['memory_snapshots'].append(memory_snapshot('after_cache_policy_parity',device));save()
            # Profile comes BEFORE optimization benchmarking; no hidden precomputed scores in timers.
            active='component_profile'
            eps={e['id']:e for e in inputs['panel']}
            if cfg.run_component_profile:
                run_profiles(engine,[eps[i] for i in inputs['profile_ids']],cfg,out,prototype_ok,report)
                report['stage_status'][active]='completed'
            else:report['stage_status'][active]='disabled'
            report['memory_snapshots'].append(memory_snapshot('after_component_profile',device));save()
            active='request_benchmark'
            if cfg.run_request_benchmark:
                run_benchmarks(engine,[eps[i] for i in inputs['benchmark_ids']],cfg,out,prototype_ok,report)
                report['stage_status'][active]='completed'
            else:report['stage_status'][active]='disabled'
            report['memory_snapshots'].append(memory_snapshot('after_request_benchmark',device));save()
            active='long_prefix_benchmark'
            if cfg.run_long_prefix_benchmark:
                run_long_prefix(engine,cfg,out,prototype_ok,report);report['stage_status'][active]='completed'
            else:report['stage_status'][active]='disabled'
            active='weight_integrity'
            if sample_parameter_digest(model)!=before:raise IntegrityError('Frozen parameter sample changed during worker.')
            report['stage_status'][active]='completed'
            report['memory_snapshots'].append(memory_snapshot('before_process_exit',device))
            report['status']='partial' if 'failed' in report['stage_status'].values() else 'completed'
        save();return 0
    except BaseException as e:
        status='skipped' if isinstance(e,StageSkip) else 'failed'
        report['stage_status'][active]=status;report['status']='partial' if report['stage_status'].get('model_load')=='completed' else status
        report['error']={'stage':active,'type':type(e).__name__,'message':redact(str(e),os.environ.get('HF_TOKEN'))[:1200],
                         'fatal_cuda':is_fatal_cuda(e),'worker_will_exit':True}
        # No GPU calls after fatal error. CPU partial files are retained and summarized.
        for filename,key in [('parity_rows.json','parity_rows'),('component_profiles.json','profile_rows'),('request_benchmarks.json','benchmark_rows'),('long_prefix.json','long_prefix')]:
            path=out/filename
            if path.exists() and key not in report:report[key]=json.loads(path.read_text())
        if report.get('parity_rows'):report['parity_summary']=summarize_parity(report['parity_rows'],list(BASE_STRATEGIES[1:])+[PROTOTYPE])
        trace=redact(''.join(traceback.format_exception(type(e),e,e.__traceback__)),os.environ.get('HF_TOKEN'))
        (out/'worker_error.txt').write_text(trace);save();print(trace,flush=True)
        return 2

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--job',required=True)
    args=parser.parse_args();sys.exit(main(json.loads(Path(args.job).read_text())))
