"""Readable static experiment figures. No chart is fabricated when a stage is missing."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def save(fig,path):
    if 'SYNTHETIC' in str(path).upper():
        fig.suptitle('SYNTHETIC VALIDATION — NOT QWEN RESULTS', fontsize=11)
        fig.tight_layout(rect=(0,0,1,.94))
    else:
        fig.tight_layout()
    fig.savefig(path,dpi=160,bbox_inches='tight');plt.close(fig)


def plots(report,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    precision=report.get('precision',{}).get('modes',{})
    valid=[(n,r) for n,r in precision.items() if r.get('status')=='completed']
    if valid:
        names=[n.replace('_','\n',1) for n,r in valid]
        fig,ax=plt.subplots(figsize=(10,5));vals=[max(r['max_shape_probability_delta'],1e-12) for _,r in valid]
        ax.bar(names,vals);ax.set_yscale('log');ax.axhline(report['configuration']['probability_tolerance'],linestyle='--',label='Declared diagnostic tolerance')
        ax.set_ylabel('Maximum absolute probability difference (log scale)');ax.set_title('Within-mode shape sensitivity; diagnostic sample\nZero differences plotted at 1e-12')
        ax.legend();save(fig,out/'selective_precision_shape_drift.png')
        fig,ax=plt.subplots(figsize=(9,5))
        for i,(name,r) in enumerate(valid,1):
            x=r['configuration']['parameter_storage_mib']/1024;y=r['single_call_p50_ms']
            ax.scatter([x],[y],label=f'{i}: {name}')
            ax.annotate(str(i),(x,y),xytext=(5,5),textcoords='offset points',fontsize=9)
        ax.margins(x=.15,y=.15)
        ax.legend(loc='upper left',bbox_to_anchor=(1.01,1),fontsize=8)
        ax.set_xlabel('Parameter storage (GiB; not total peak memory)');ax.set_ylabel('Single diagnostic call median (ms)')
        ax.set_title('Precision resource comparison; one fixed input');save(fig,out/'precision_resources.png')
    shared=report.get('shared_prefix',{})
    cache_rows=[r for v in shared.values() for r in v.get('rows',[])]
    if cache_rows:
        fig,ax=plt.subplots(figsize=(8,5))
        for mode in sorted({r['mode'] for r in cache_rows}):
            rows=[r for r in cache_rows if r['mode']==mode]
            ax.scatter([r['prefix_tokens'] for r in rows],[max(r['max_probability_delta_cached_vs_full'],1e-12) for r in rows],label=mode)
        ax.axhline(report['configuration']['cache_probability_tolerance'],linestyle='--',label='Declared cache tolerance')
        ax.set_yscale('log');ax.set_xlabel('Shared prefix tokens');ax.set_ylabel('Cached versus full probability difference')
        ax.set_title('Cache parity by request; zero differences plotted at 1e-12');ax.legend();save(fig,out/'cache_parity.png')
    rows=report.get('request_benchmark',{}).get('rows',[])
    good=[r for r in rows if r.get('status')=='completed' and 'strategy' in r]
    if good:
        for mode in sorted({r['mode'] for r in good}):
            fig,ax=plt.subplots(figsize=(8,5))
            for strategy in ('full_sequential','full_batch4','shared_prefix_chunk'):
                data=[r for r in good if r['strategy']==strategy and r['mode']==mode]
                ks=sorted({r['K'] for r in data})
                if ks:ax.plot(ks,[np.median([r['p50_ms'] for r in data if r['K']==k]) for k in ks],marker='o',label=strategy)
            ax.set_xlabel('Real candidates');ax.set_ylabel('Median of per-request medians (ms)')
            ax.set_title(f'Complete request latency: {mode}\nIncludes cold prefix and clone cost; check parity before comparing')
            ax.legend();save(fig,out/f'request_latency_{mode}.png')
    policies=report.get('policies',{}).get('test_results',[])
    if policies:
        for split in ('test_seen','test_unseen'):
            fig,ax=plt.subplots(figsize=(8,5))
            for cost in report['configuration']['wrong_answer_costs']:
                rows=[r for r in policies if r['split']==split and r['wrong_answer_cost']==cost]
                rows=sorted(rows,key=lambda r:r['absent_prior_assumption'])
                ax.plot([r['absent_prior_assumption'] for r in rows],[r['metrics']['expected_cost'] for r in rows],marker='o',label=f'Wrong-answer cost {cost:g}')
            ax.axhline(report['configuration']['review_cost'],linestyle='--',label='Always review')
            ax.set_xlabel('Assumed missing-answer frequency');ax.set_ylabel('Expected cost in declared scenario units')
            ax.set_title(f'Dev-selected policies: {split}\nScenario reweighting, not observed deployment frequencies');ax.legend();save(fig,out/f'policy_cost_{split}.png')
