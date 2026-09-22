"""Durable result tables, honest plots, cross-mode comparisons and archive export."""
import copy,json,shutil
from pathlib import Path
import numpy as np
from phase2d_common import json_write
from phase2g_metrics import parity,summarize_parity,quality


def read_rows(path):return json.loads(path.read_text()) if path.exists() else []

def cross_modes(exp):
    ref=read_rows(exp.out/'fp32_strict_math/fresh_rows.json');ref={r['id']:r for r in ref if r['strategy']=='full_sequential' and r['status']=='completed'}
    result={}
    for mode in exp.cfg.modes[1:]:
        rows=[]
        for r in read_rows(exp.out/mode/'fresh_rows.json'):
            if r['status']=='completed' and r['strategy']=='full_sequential' and r['id'] in ref:
                q=dict(r);q['comparison']=parity(ref[r['id']]['outputs'],r['outputs'],exp.cfg.cache_probability_tolerance);rows.append(q)
        result[mode]={'status':'completed' if rows else 'unavailable','summary':summarize_parity(rows),
          'families':{f:summarize_parity([r for r in rows if r['family']==f]) for f in sorted({r['family'] for r in rows})},
          'scope':'identical full-sequential finalized inputs, different arithmetic. Disagreement is not automatically an accuracy loss.'}
        json_write(exp.out/('cross_precision_'+mode+'.json'),rows)
    return result


def tables_and_plots(exp):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    plots=exp.out/'plots';plots.mkdir(exist_ok=True);made=[];quality_rows=[];parity_rows=[];timing_rows=[]
    label='SYNTHETIC VALIDATION — NOT QWEN' if exp.report.get('validation_only') else 'Phase 2G'
    for mode,worker in exp.report['workers'].items():
        for panel in ('fresh','context'):
            for family,strategies in worker.get(panel+'_summary',{}).items():
                for strategy,entry in strategies.items():
                    for head,m in entry['quality']['heads'].items():
                        quality_rows.append({'mode':mode,'panel':panel,'family':family,'strategy':strategy,'head':head,
                          'messages':entry['quality']['independent_message_groups'],'episodes':entry['quality']['episodes'],
                          **{k:m[k] for k in ('accuracy','nll','brier_sum_classes','none_recall','answerable_accuracy','false_none_when_present')}})
                    if entry['parity'].get('evaluated'):parity_rows.append({'mode':mode,'panel':panel,'family':family,'strategy':strategy,**entry['parity']})
        for r in worker.get('benchmark_rows',[]):
            if r['status']=='completed':timing_rows.append({k:r[k] for k in ('id','group','family','strategy','K','p50_ms','peak_extra_mib')}|{'mode':mode,'accepted':r['comparison']['accepted_within_sample']})
    pd.DataFrame(quality_rows).to_csv(exp.out/'quality_metrics.csv',index=False);pd.DataFrame(parity_rows).to_csv(exp.out/'parity_metrics.csv',index=False);pd.DataFrame(timing_rows).to_csv(exp.out/'request_metrics.csv',index=False)
    q=pd.DataFrame(quality_rows)
    if not q.empty:
        for mode in q['mode'].unique():
            d=q[(q['mode']==mode)&(q['panel']=='fresh')&(q['strategy']=='full_sequential')]
            fig,ax=plt.subplots(figsize=(10,5));families=sorted(d.family.unique());xx=np.arange(len(families))
            for j,h in enumerate(('frozen_global','refit_global','set_linear')):
                vals=[float(d[(d.family==f)&(d['head']==h)].accuracy.iloc[0]) for f in families]
                ax.bar(xx+(j-1)*.24,vals,width=.23,label=h)
            ax.set_xticks(xx, [f.replace('_','\n') for f in families]);ax.set_ylim(0,1);ax.set_ylabel('Accuracy on labeled sampled-choice episodes')
            ax.set_title(label+' | frozen-head transfer | '+mode);ax.legend(loc='lower left');ax.grid(axis='y',alpha=.25);fig.tight_layout();p=plots/('fresh_quality_'+mode+'.png');fig.savefig(p,dpi=140);plt.close(fig);made.append(str(p))
    for mode,worker in exp.report['workers'].items():
        ctx=worker.get('context_by_length_position',[])
        if ctx:
            fig,ax=plt.subplots(figsize=(10,5))
            for family in sorted({r['family'] for r in ctx}):
                for pos,ls in [('first','-'),('last','--')]:
                    rs=sorted([r for r in ctx if r['family']==family and r['position']==pos],key=lambda r:r['context_target'])
                    ax.plot([r['context_target'] for r in rs],[r['quality']['heads']['set_linear']['accuracy'] for r in rs],marker='o',linestyle=ls,label=family+' / '+pos)
            ax.set_ylim(0,1);ax.set_xlabel('Minimum constructed state tokens (0 = minimal wrapper)');ax.set_ylabel('Set-linear accuracy');ax.set_title(label+' | controlled context, NOT natural long documents');ax.legend(fontsize=8,bbox_to_anchor=(1.02,1),loc='upper left');fig.tight_layout();p=plots/('controlled_context_'+mode+'.png');fig.savefig(p,dpi=140);plt.close(fig);made.append(str(p))
        ts=worker.get('traffic_summary',[])
        if ts:
            fig,ax=plt.subplots(figsize=(9,5));names=sorted({r['strategy'] for r in ts});x=np.arange(len(names));means=[];low=[];high=[]
            for name in names:
                vals=[r['total_ms']/1000 for r in ts if r['strategy']==name and r['total_ms'] is not None];means.append(float(np.median(vals)) if vals else np.nan);low.append(min(vals) if vals else np.nan);high.append(max(vals) if vals else np.nan)
            ax.bar(x,means);ax.errorbar(x,means,yerr=[np.array(means)-low,np.array(high)-means],fmt='none',capsize=4)
            ax.set_xticks(x,[n.replace('_','\n') for n in names]);ax.set_ylabel('Total measured service time (seconds)');ax.set_title(label+' | controlled TTL trace | '+mode+'\nBars: median; whiskers: replicate range, not confidence interval');fig.tight_layout();p=plots/('traffic_'+mode+'.png');fig.savefig(p,dpi=140);plt.close(fig);made.append(str(p))
    t=pd.DataFrame(timing_rows)
    if not t.empty:
        for mode in t['mode'].unique():
            d=t[t['mode']==mode];fig,ax=plt.subplots(figsize=(9,5))
            for strategy in sorted(d.strategy.unique()):
                g=d[d.strategy==strategy].groupby('K').p50_ms.median();ax.plot(g.index,g.values,marker='o',label=strategy)
            ax.set_xticks(sorted(d.K.unique()));ax.set_xlabel('Real candidates');ax.set_ylabel('Median of request-specific median times (ms)');ax.set_title(label+' | cold complete requests | '+mode);ax.legend();fig.tight_layout();p=plots/('cold_requests_'+mode+'.png');fig.savefig(p,dpi=140);plt.close(fig);made.append(str(p))
    return made


def finish(exp):
    exp.report['cross_precision']=cross_modes(exp)
    statuses=[exp.report['workers'].get(m,{}).get('status','not_run') for m in exp.cfg.modes]
    exp.report['run_status']='completed' if all(s=='completed' for s in statuses) else 'partial' if any(s in ('completed','partial') for s in statuses) else 'failed'
    plots=tables_and_plots(exp)
    # Every completed stage can fail an acceptance gate. Never use run_status as quality approval.
    exp.report['acceptance_scope']='Execution completion and semantic quality/numerical equivalence are reported separately. No production mode automatically selected.'
    json_write(exp.out/'openkind_phase2g_summary.json',exp.report)
    compact=copy.deepcopy(exp.report)
    for worker in compact['workers'].values():
        b=worker.pop('benchmark_rows',[])
        worker['benchmark_compact']=[{'strategy':s,'K':k,'median_of_episode_p50_ms':float(np.median([r['p50_ms'] for r in b if r.get('status')=='completed' and r['strategy']==s and r['K']==k]))}
          for s,k in sorted({(r['strategy'],r['K']) for r in b if r.get('status')=='completed'})]
        for panel in ('fresh','context'):
            for family,strategies in worker.get(panel+'_summary',{}).items():
                for s,entry in strategies.items():
                    entry['quality'].pop('policies',None);entry.get('matched_reference_quality',{}).pop('policies',None)
    json_write(exp.out/'paste_back_summary.json',compact)
    (exp.out/'README_results.md').write_text(f'# OpenKind Phase 2G\n\nRun {exp.run_id}; status {exp.report["run_status"]}.\n\n'+('\n**SYNTHETIC VALIDATION — NOT QWEN RESULTS**\n' if exp.report.get('validation_only') else '')+'\nRead probability quality, frozen-policy outcomes and numerical parity separately. No new trained coefficients. Fresh messages are only project-decontaminated; context extensions are controlled wrappers.\n\nRaw per-episode rows and complete worker errors are retained. The public data and exact message manifest identify the evaluation panel.\n')
    impl=exp.out/'implementation';impl.mkdir(exist_ok=True)
    for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,impl/p.name)
    shutil.copy2(Path(__file__).parent.parent/'assets/prior_exclusions.json',exp.out/'prior_exclusions.json')
    exp.checkpoint()
    archive=Path(shutil.make_archive(str(exp.out.parent/('openkind_phase2g_'+exp.run_id)),'zip',exp.out))
    if exp.cfg.copy_results_to_drive:
        dest=Path(exp.cfg.drive_destination)/exp.run_id;dest.mkdir(parents=True,exist_ok=True)
        for p in (archive,exp.out/'paste_back_summary.json',exp.out/'openkind_phase2g_summary.json',exp.out/'README_results.md',exp.out/'fresh_message_manifest.json'):shutil.copy2(p,dest/p.name)
        print('Copied reports/archive to:',dest)
    print('Local results:',exp.out);print('Result archive:',archive)
    print('===== BEGIN_OPENKIND_PHASE2G_SUMMARY =====');print(json.dumps(compact,indent=2));print('===== END_OPENKIND_PHASE2G_SUMMARY =====')
    return {'out':str(exp.out),'archive':str(archive),'plots':plots,'run_status':exp.report['run_status']}
