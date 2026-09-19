"""Portable per-question charts. No synthetic outcomes enter a real result report."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def render(out,records):
    out=Path(out);plots=out/'plots';plots.mkdir(exist_ok=True)
    for name,r in records.items():
        if not name.startswith('eval_') or r.get('status')!='completed':continue
        banner=('SYNTHETIC VALIDATION — NOT QWEN\n' if r.get('validation_only') else '')
        selected=r['selected_profile'];models=r.get('final_metrics',{})
        families=sorted(f for f in models.get(selected,{}) if f!='pooled_declared_mixture')
        for family in families:
            # Keep a bounded design contrast: legacy original, one lexical/code control, chosen model.
            options=[k for k in models if (k==selected or k.startswith('original__instruction_first__frozen') or k in ('original__lexical','original__finite_code'))]
            options=options[:6];values=[models[k][family]['quality']['accuracy'] for k in options]
            labels=[('Development-selected trained head' if k==selected else 'Lexical control' if k.endswith('__lexical') else 'Finite-code control' if k.endswith('__finite_code') else 'Frozen scorer + old none' if k.endswith('__unchanged_none') else 'Frozen scorer + refitted none') for k in options]
            fig,ax=plt.subplots(figsize=(10,4.5));ax.barh(labels,values);ax.set_xlim(0,1);ax.set_xlabel('Episode accuracy (message-clustered inputs)');ax.set_title(banner+name+' — '+family)
            for i,v in enumerate(values):ax.text(min(v+.012,.94),i,f'{100*v:.1f}%',va='center')
            fig.tight_layout();fig.savefig(plots/(name+'_'+family+'_accuracy.png'),dpi=140);plt.close(fig)
        ctx=[x for x in r.get('context_by_position',[]) if x['profile']==selected]
        if ctx:
            fig,ax=plt.subplots(figsize=(8,4.5))
            for pos in ('first','last'):
                sub=sorted([x for x in ctx if x['position']==pos],key=lambda x:x['min_tokens']);ax.plot([x['min_tokens'] for x in sub],[x['quality']['accuracy'] for x in sub],marker='o',label='Request '+pos)
            ax.set_ylim(0,1);ax.set_xlabel('Minimum constructed state tokens (0 = minimal wrapper)');ax.set_ylabel('Accuracy');ax.set_title(banner+name+' — controlled context, not natural documents');ax.legend();fig.tight_layout();fig.savefig(plots/(name+'_context.png'),dpi=140);plt.close(fig)
        bench=r.get('benchmark_rows',[])
        for k in sorted({x['K'] for x in bench}):
            subset=[x for x in bench if x['K']==k]
            labels=sorted({x['method']+' / '+x['strategy'] for x in subset})
            vals=[np.median([x['median_ms'] for x in subset if x['method']+' / '+x['strategy']==l]) for l in labels]
            fig,ax=plt.subplots(figsize=(9,4.5));ax.barh(labels,vals);ax.set_xlabel('Median of per-request median milliseconds')
            ax.set_title(banner+name+f' — K={k} complete requests; methods use different prompts')
            fig.tight_layout();fig.savefig(plots/(name+f'_latency_k{k}.png'),dpi=140);plt.close(fig)
