"""Measured readout with explicit cold/warm/storage/equivalence scopes; no invented model findings."""
from collections import defaultdict
from pathlib import Path
import json,shutil
import numpy as np
from phase2d_common import json_write


def summarize_comparisons(rows,key):
    if not rows:return {}
    comparisons=[r[key] for r in rows if key in r]
    if not comparisons:return {}
    return {'episodes':len(comparisons),'distinct_messages':len({r['group'] for r in rows}),
       'maximum_probability_delta':max(c['max_probability_delta'] for c in comparisons),
       'episodes_with_selected_outcome_change':sum(c['any_argmax_changed'] for c in comparisons),
       'episodes_with_answer_review_change':sum(c['any_policy_action_changed'] for c in comparisons),
       'episodes_with_any_policy_output_change':sum(c['any_policy_output_changed'] for c in comparisons),
       'accepted_episodes':sum(c['accepted_within_sample'] for c in comparisons),
       'all_accepted_within_sample':all(c['accepted_within_sample'] for c in comparisons)}


def worker_summary(worker):
    out={k:worker.get(k) for k in ('status','stage_status','error','environment','model','memory_snapshots','mtp','upstream_codec_api','codec_table_initialization') if k in worker}
    baseline=defaultdict(list)
    for row in worker.get('baseline_parity',[]):
        for name,item in row['strategies'].items():baseline[name].append({'group':row['group'],'comparison':item['comparison']})
    out['baseline_parity']={k:summarize_comparisons(v,'comparison') for k,v in baseline.items()}
    comp=defaultdict(list)
    for r in worker.get('compression_rows',[]):
        if r.get('status')=='completed':comp[r['codec']].append(r)
    out['compression']={k:{'versus_full':summarize_comparisons(v,'versus_full'),
        'versus_same_chunking':summarize_comparisons(v,'versus_uncompressed_same_chunking'),
        'one_entry_ratio_min':min(r['storage']['one_entry_compression_ratio'] for r in v),
        'one_entry_ratio_max':max(r['storage']['one_entry_compression_ratio'] for r in v),
        'order_checks':sum('order_check_passed' in r for r in v),
        'failed_order_checks':sum(not r['order_check_passed'] for r in v if 'order_check_passed' in r),
        'example_storage':v[0]['storage']} for k,v in comp.items()}
    out['codec_failures']=[{k:r[k] for k in ('codec','episode_id','status','error') if k in r} for r in worker.get('compression_rows',[]) if r.get('status')!='completed']
    out['compression_quality_regression']=worker.get('compression_quality_regression',[])
    groups=defaultdict(list)
    for row in worker.get('benchmark_rows',[]):
        if row.get('status')=='completed':groups[(row['strategy'],row['K'])].append(row)
    out['cold_benchmarks']=[{'strategy':s,'K':k,'episodes':len(rs),'distinct_messages':len({r['group'] for r in rs}),
        'median_of_episode_p50_ms':float(np.median([r['p50_ms'] for r in rs])),
        'max_peak_extra_mib':max([r['peak_extra_mib'] for r in rs if r.get('peak_extra_mib') is not None],default=None),
        'parity':summarize_comparisons(rs,'comparison')} for (s,k),rs in groups.items()]
    out['cross_request_cache']=worker.get('cross_request_summary',[])
    out['long_prefix']=[]
    for r in worker.get('long_prefix_rows',[]):
        item={k:r[k] for k in ('prefix_tokens','K','strategy','status','p50_ms','p95_ms','resident_mib','peak_extra_mib','population_request_ms','error') if k in r}
        if 'comparison' in r:item['comparison']=r['comparison']
        if 'snapshot' in r.get('metadata',{}):item['storage']=r['metadata']['snapshot'];item['codec_shared_bytes']=r['metadata']['codec_shared_bytes']
        out['long_prefix'].append(item)
    out['scope']='Same recorded panel and frozen policies. A lossy-codec failure is not hidden by a faster latency.'
    return out


def cross_precision(report):
    workers=report['workers'];a=workers.get('fp32_strict_math',{});b=workers.get('bf16_default',{})
    aa={r['episode_id']:r for r in a.get('baseline_parity',[])};bb={r['episode_id']:r for r in b.get('baseline_parity',[])}
    if not aa or not bb:return {'status':'unavailable','reason':'Both fresh workers must provide matched reference rows.'}
    from phase2e_expand_execution import compare_views
    rows=[]
    for key in sorted(set(aa)&set(bb)):
        x=aa[key]['reference'];y=bb[key]['reference']
        c=compare_views({k:np.asarray(p) for k,p in x['distributions'].items()},
                        {k:np.asarray(p) for k,p in y['distributions'].items()},x['actions'],y['actions'])
        rows.append(c)
    return {'status':'completed','episodes':len(rows),'max_probability_delta':max(c['max_probability_delta'] for c in rows),
      'selected_outcome_changes':sum(c['any_argmax_changed'] for c in rows),
      'policy_output_changes':sum(c['any_policy_output_changed'] for c in rows),
      'scope':'FP32 versus BF16 full-sequential outputs, distinct from codec error; neither precision is assumed more accurate.'}


def figures(summary,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory=Path(out)/'plots';directory.mkdir(exist_ok=True)
    paths=[]
    label='SYNTHETIC VALIDATION — NOT QWEN RESULTS\n' if summary.get('validation_only') else ''
    for mode,w in summary['workers'].items():
        if not w.get('cold_benchmarks'):continue
        fig,ax=plt.subplots(figsize=(10,5.8))
        groups=defaultdict(list)
        for r in w['cold_benchmarks']:groups[r['strategy']].append(r)
        for name,rows in groups.items():
            rows=sorted(rows,key=lambda r:r['K']);ax.plot([r['K'] for r in rows],[r['median_of_episode_p50_ms'] for r in rows],marker='o',label=name)
        ax.set(xlabel='Real candidates (+ none)',ylabel='Complete cold request median (ms)',title=label+f'Cold request latency — {mode}')
        ax.set_xticks([2,4,8,16]);ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.25)
        ax.legend(fontsize=8,loc='upper left',bbox_to_anchor=(1.01,1));fig.tight_layout()
        p=directory/f'cold_latency_{mode}.png';fig.savefig(p,dpi=150,bbox_inches='tight');plt.close(fig);paths.append(str(p))
        comp=w.get('compression',{})
        if comp:
            fig,ax=plt.subplots(figsize=(9,5))
            names=list(comp);x=np.arange(len(names))
            vals=[comp[n]['example_storage']['one_entry_compression_ratio'] for n in names]
            ax.bar(x,vals);ax.axhline(1,linestyle='--');ax.set_xticks(x,names,rotation=20,ha='right')
            ax.set(ylabel='Original / (stored hybrid state + shared codec tables)',title=label+f'One representative prefix: total cache-storage ratio — {mode}',ylim=(0,max(1.1,max(vals)*1.2)))
            for i,v in enumerate(vals):ax.text(i,v+.02,f'{v:.2f}×',ha='center',fontsize=9)
            fig.text(.01,.01,'One stored entry; includes unchanged recurrent/conv state. Not model compression or peak-memory savings.',fontsize=8)
            fig.tight_layout(rect=(0,.035,1,1));p=directory/f'storage_ratio_{mode}.png';fig.savefig(p,dpi=150,bbox_inches='tight');plt.close(fig);paths.append(str(p))
            fig,ax=plt.subplots(figsize=(9,5))
            vals=[100*comp[n]['versus_full']['maximum_probability_delta'] for n in names]
            ax.bar(x,vals);ax.axhline(.5,linestyle='--',label='0.5 percentage-point diagnostic tolerance')
            ax.set_xticks(x,names,rotation=20,ha='right');ax.set(ylabel='Maximum absolute probability difference (percentage points)',title=label+f'Lossy cache behavior versus full sequential — {mode}')
            ax.legend(fontsize=8);fig.tight_layout();p=directory/f'codec_drift_{mode}.png';fig.savefig(p,dpi=150,bbox_inches='tight');plt.close(fig);paths.append(str(p))
        traffic=w.get('cross_request_cache',[])
        if traffic:
            fig,ax=plt.subplots(figsize=(10,5))
            for workload in sorted({r['workload'] for r in traffic}):
                rows=[r for r in traffic if r['workload']==workload and r['p50_request_ms'] is not None]
                ax.plot([r['strategy'] for r in rows],[r['p50_request_ms'] for r in rows],marker='o',label=workload)
            ax.set(ylabel='Observed trace request median (ms)',title=label+f'Exact-prefix LRU traffic, cold-start misses included — {mode}',ylim=(0,None));ax.tick_params(axis='x',rotation=15);ax.legend();ax.grid(axis='y',alpha=.25)
            fig.tight_layout();p=directory/f'prefix_lru_{mode}.png';fig.savefig(p,dpi=150,bbox_inches='tight');plt.close(fig);paths.append(str(p))
    return paths


def finish_report(experiment):
    report=experiment.report;out=experiment.out
    report['cross_precision']=cross_precision(report)
    statuses=[report['workers'].get(m,{}).get('status','not_run') for m in experiment.cfg.modes]
    report['run_status']='completed' if all(x=='completed' for x in statuses) else ('partial' if any(x in ('completed','partial') for x in statuses) else 'failed')
    summary={k:v for k,v in report.items() if k not in ('workers','code_sha256')}
    summary['workers']={m:worker_summary(w) for m,w in report['workers'].items()}
    summary['acceptance_note']='Completion is execution status. Inspect per-strategy numerical/policy gates before accepting a speed/storage trade-off.'
    summary['mtp']={'status':'not_applicable','generated_tokens':0,'benchmark':None,
      'explanation':'MTP drafts output tokens for autoregressive decoding; it cannot skip already-known candidate input evaluation in this decision-only graph.'}
    summary['artifacts']={'full_report':'opendecision_phase2f_summary.json','summary':'paste_back_summary.json'}
    json_write(out/'opendecision_phase2f_summary.json',report)
    json_write(out/'paste_back_summary.json',summary)
    paths=figures(summary,out)
    (out/'README_results.md').write_text('# OpenDecision Phase 2F\n\nRun '+experiment.run_id+'; status '+report['run_status']+'\n\n'
      'See paste_back_summary.json first. Raw worker rows retain paired scores, frozen policy outputs, actual cache bytes and timing samples.\n\n'
      'TurboQuant is a pinned external GPL-3.0 dependency: standalone codecs adapted to HF prefix snapshots. Native vLLM integration and fused low-bit attention are NOT benchmarked.\n\n'
      'Stored cache compression does not shrink model weights or imply smaller transient inference memory. Warm-cache traces include population misses. MTP is not applicable because no output tokens are generated.\n\n'
      'Prior examples are reused for regression; these are not new independent generalization results.\n')
    impl=out/'implementation';impl.mkdir(exist_ok=True)
    for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,impl/p.name)
    archive=shutil.make_archive(str(out.parent/f'opendecision_phase2f_{experiment.run_id}'),'zip',out)
    if experiment.cfg.copy_results_to_drive:
        dest=Path(experiment.cfg.drive_destination)/experiment.run_id;dest.mkdir(parents=True,exist_ok=True)
        for name in ('paste_back_summary.json','opendecision_phase2f_summary.json','README_results.md'):shutil.copy2(out/name,dest/name)
        shutil.copy2(archive,dest/Path(archive).name)
        print('Copied reports/archive to:',dest)
    print('Local results:',out);print('Archive:',archive)
    print('===== BEGIN_OPENDECISION_PHASE2F_SUMMARY =====')
    print(json.dumps(summary,indent=2,allow_nan=False))
    print('===== END_OPENDECISION_PHASE2F_SUMMARY =====')
    return {'summary':summary,'archive':archive,'plots':paths,'out':str(out)}
