"""Reports computed only from measured workers. No synthetic substitutions for missing modes."""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import json
import numpy as np
from phase2d_common import json_write
from phase2e_expand_execution import compare_views


def aggregate_parity(worker):
    rows=worker.get('parity_rows',[]);groups=defaultdict(list);policies=defaultdict(list)
    for row in rows:
        for strategy,item in row['strategies'].items():
            groups[strategy].append((row,item))
            for key,change in item['policies'].items():policies[strategy,key].append((row,change))
    result={}
    for strategy,items in groups.items():
        deltas=[x['max_probability_delta'] for _,x in items]
        result[strategy]={'episodes':len(items),'distinct_messages':len({r['group'] for r,_ in items}),
           'p50_max_probability_delta':float(np.median(deltas)),
           'p95_max_probability_delta':float(np.quantile(deltas,.95)),
           'max_probability_delta':max(deltas),
           'episodes_over_probability_tolerance':sum(not i['within_probability_tolerance'] for _,i in items),
           'episodes_with_argmax_change':sum(i['any_argmax_changed'] for _,i in items),
           'episodes_with_answer_review_change':sum(i['any_policy_action_changed'] for _,i in items),
           'episodes_with_any_policy_output_change':sum(i['any_policy_output_changed'] for _,i in items),
           'all_accepted_within_sample':all(i['accepted_within_sample'] for _,i in items),
           'isolation_episodes':sum('isolation' in i for _,i in items),
           'policies':{}}
    for (strategy,key),items in policies.items():
        result[strategy]['policies'][key]={'comparisons':len(items),'messages':len({r['group'] for r,_ in items}),
            **{field:sum(bool(c[field]) for _,c in items) for field in
               ('answer_review_changed','review_to_answer','answer_to_review','accepted_candidate_changed','any_output_changed')}}
    return result


def aggregate_benchmarks(worker):
    groups=defaultdict(list)
    for r in worker.get('benchmark_rows',[]):
        if r.get('status')=='completed':groups[r['strategy'],r['K']].append(r)
    result=[]
    for (strategy,k),rs in sorted(groups.items()):
        result.append({'strategy':strategy,'K':k,'episodes':len(rs),'messages':len({r['group'] for r in rs}),
           'median_of_episode_p50_ms':float(np.median([r['p50_ms'] for r in rs])),
           'max_episode_p50_ms':float(np.max([r['p50_ms'] for r in rs])),
           'max_probability_delta':max(r['comparison']['max_probability_delta'] for r in rs),
           'episodes_with_policy_output_change':sum(r['comparison']['any_policy_output_changed'] for r in rs),
           'metric_definition':'median across per-episode synchronized median COMPLETE request times; not a production latency percentile'})
    return result


def cross_precision(workers):
    if not all(m in workers and workers[m].get('parity_rows') for m in ('fp32_strict_math','bf16_default')):
        return {'status':'unavailable','reason':'Both workers need measured parity rows; no substitute FP32 result.'}
    a={r['episode_id']:r for r in workers['fp32_strict_math']['parity_rows']}
    b={r['episode_id']:r for r in workers['bf16_default']['parity_rows']}
    rows=[]
    for key in sorted(set(a)&set(b)):
        if a[key].get('token_ids_sha256')!=b[key].get('token_ids_sha256'):
            raise ValueError('Cross-precision token sequences differ; comparison is invalid.')
        x,y=a[key]['reference'],b[key]['reference']
        xx={n:np.asarray(p) for n,p in x['distributions'].items()};yy={n:np.asarray(p) for n,p in y['distributions'].items()}
        cmp=compare_views(xx,yy,x['actions'],y['actions'])
        rows.append({'episode_id':key,'group':a[key]['group'],**cmp})
    return {'status':'completed','episodes':len(rows),'messages':len({r['group'] for r in rows}),
      'max_probability_delta':max(r['max_probability_delta'] for r in rows),
      'episodes_with_class_change':sum(r['any_argmax_changed'] for r in rows),
      'episodes_with_policy_output_change':sum(r['any_policy_output_changed'] for r in rows),
      'scope':'FP32 full sequential versus BF16 full sequential on identical inputs. Consistency comparison, not proof FP32 is more accurate.',
      'rows':rows}


def make_plots(report,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out);out.mkdir(exist_ok=True)
    modes=report.get('workers',{});prefix='SYNTHETIC VALIDATION — NOT QWEN\n' if report.get('validation_only') else ''
    summaries=[]
    labels={'full_batch4':'Full prompts / batch 4','shared_prefix_chunk':'Prefix / sequential suffixes',
            'shared_prefix_equal_length_batch4':'Prefix / equal-length suffix batches','full_sequential':'Full prompts / sequential'}
    for mode,w in modes.items():
        for strategy,x in w.get('parity_summary',{}).items():summaries.append((mode,strategy,x))
        profiles=w.get('profile_rows',[])
        if profiles:
            strategies=sorted({r['strategy'] for r in profiles});components=sorted(set().union(*(r['component_ms'] for r in profiles)))
            fig,ax=plt.subplots(figsize=(12,5.8));left=np.zeros(len(strategies))
            for name in components:
                values=np.array([np.median([r['component_ms'].get(name,0.) for r in profiles if r['strategy']==s]) for s in strategies])
                ax.barh(np.arange(len(strategies)),values,left=left,label=name.replace('_ms','').replace('_',' '));left+=values
            ax.set_yticks(range(len(strategies)),[labels[s] for s in strategies]);ax.invert_yaxis()
            ax.set_xlabel('Component wall time (ms); median of per-episode component medians')
            ax.set_title(prefix+f'Intrusive synchronized component profiles — {mode}')
            ax.legend(loc='upper left',bbox_to_anchor=(1.01,1),fontsize=9);fig.tight_layout();fig.savefig(out/f'profile_{mode}.png',dpi=150,bbox_inches='tight');plt.close(fig)
        bench=w.get('benchmark_summary',[])
        if bench:
            fig,ax=plt.subplots(figsize=(10,5.4))
            for i,s in enumerate(sorted({r['strategy'] for r in bench})):
                rs=sorted([r for r in bench if r['strategy']==s],key=lambda r:r['K'])
                ax.plot([r['K'] for r in rs],[r['median_of_episode_p50_ms'] for r in rs],marker=['o','s','^','D'][i%4],label=labels[s])
            ax.set_xticks([2,4,8,16]);ax.set_ylim(bottom=0);ax.set_xlabel('Real candidate count')
            ax.set_ylabel('Median of episode median latency (ms)');ax.set_title(prefix+f'Cold complete-request latency — {mode}')
            ax.legend();fig.tight_layout();fig.savefig(out/f'request_latency_{mode}.png',dpi=150);plt.close(fig)
    if summaries:
        names=[f"{m}\n{labels[s]}" for m,s,_ in summaries];n=len(names)
        fig,ax=plt.subplots(figsize=(11,max(4.8,n*.7)))
        ax.barh(range(n),[v['max_probability_delta']*100 for _,_,v in summaries])
        ax.axvline(report['configuration']['cache_probability_tolerance']*100,linestyle='--',label='Declared tolerance')
        ax.set_yticks(range(n),names,fontsize=9);ax.invert_yaxis();ax.set_xlabel('Maximum probability difference (percentage points)')
        ax.set_title(prefix+'Cache/batch agreement against each mode’s full sequential reference');ax.legend()
        fig.tight_layout();fig.savefig(out/'cache_probability_drift.png',dpi=150);plt.close(fig)
        fig,ax=plt.subplots(figsize=(11,max(4.8,n*.7)))
        counts=[v['episodes_with_answer_review_change'] for _,_,v in summaries]
        bars=ax.barh(range(n),counts)
        ax.set_xlim(0,max(1,max(counts))*1.15)
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.bar_label(bars,labels=[str(x) for x in counts],padding=3)
        ax.set_yticks(range(n),names,fontsize=9);ax.invert_yaxis();ax.set_xlabel('Episodes with an answer/review change in any frozen policy')
        ax.set_title(prefix+'Application-action disagreement (policies are not refitted)')
        fig.tight_layout();fig.savefig(out/'policy_action_disagreement.png',dpi=150);plt.close(fig)
    memory=[(m,w['memory_snapshots']) for m,w in modes.items() if w.get('memory_snapshots') and w['memory_snapshots'][0].get('cuda')]
    for mode,rows in memory:
        fig,ax=plt.subplots(figsize=(11,5.5))
        for key,scale,label in [('allocated_mib',1/1024,'Tensor allocations'),('reserved_mib',1/1024,'Allocator reserved'),('driver_free_gib',1,'Driver free')]:
            ax.plot(range(len(rows)),[r[key]*scale for r in rows],marker='o',label=label)
        ax.set_xticks(range(len(rows)),[r['label'].replace('_',' ') for r in rows],rotation=25,ha='right',fontsize=8)
        ax.set_ylim(bottom=0);ax.set_ylabel('GiB');ax.set_title(prefix+f'Isolated worker memory snapshots — {mode}');ax.legend()
        fig.tight_layout();fig.savefig(out/f'memory_{mode}.png',dpi=150);plt.close(fig)


def build_report(report,out,cfg):
    for mode,w in report.get('workers',{}).items():
        w['parity_summary']=aggregate_parity(w)
        w['benchmark_summary']=aggregate_benchmarks(w)
        w['memory_summary']=[{k:r[k] for k in ('label','allocated_mib','reserved_mib','driver_free_gib','peak_allocated_mib') if k in r}
                             for r in w.get('memory_snapshots',[])]
        # Large raw data remain in worker files; compact summaries still retain every policy's counts.
    cross=cross_precision(report.get('workers',{}));json_write(Path(out)/'cross_precision.json',cross)
    report['cross_precision']={k:v for k,v in cross.items() if k!='rows'}
    requested=[m for m in cfg.isolated_modes if m!='fp32_strict_math' or cfg.run_fp32]
    statuses=[report.get('workers',{}).get(m,{}).get('status','not_run') for m in requested]
    report['run_status']='completed' if statuses and all(s=='completed' for s in statuses) else ('partial' if any(s in ('completed','partial') for s in statuses) else 'failed')
    report['acceptance_scope']='A completed worker ran successfully; per-strategy numerical/policy gates can still fail.'
    make_plots(report,Path(out)/'plots')
    lines=['# OpenKind Phase 2E — expanded execution review',f"Version: {report['version']}",f"Run: {report['run_id']}",f"Execution status: {report['run_status']}",'',
      '## Readout',
      'Inspect `parity_summary` for probabilities, selected classes, and answer/review output changes. Completion does not imply equivalence.',
      'Inspect independent complete-request timings separately from intrusive component profiles.',
      'FP32/BF16 workers are separate foreground processes; the notebook coordinator never owns the model.',
      'No head coefficients or acceptance thresholds were fitted in this run. Archived tests are reused for regression only.','',
      '## Files',
      '`experiment_inputs.json`: exact episode panel and frozen policy manifest.',
      '`<mode>/memory_snapshots` in worker_report.json: tensor, allocator, and driver memory snapshots.',
      '`<mode>/parity_rows.json`: raw scores, all three probability models, policy decisions and directed disagreements.',
      '`<mode>/component_profiles.json`: synchronized component wall times.',
      '`<mode>/request_benchmarks.json`: independent uninstrumented cold-request timings.',
      '`cross_precision.json`: between-mode full-sequential comparison; FP32 is not assumed more accurate.','',
      '## Limitations']+['- '+s for s in report['limitations']]
    (Path(out)/'README_results.md').write_text('\n'.join(lines)+'\n')
    return report
