"""Separate supervised BoolQ and SST-5 readouts. They are bounded primitive tasks, not generic APIs."""
import json,copy,math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from safetensors.torch import save_file,load_file
from phase2d_common import text_hash,content_hash,json_write
from phase2h_learning import bootstrap
LEVELS=['Very negative: strongly unfavorable overall.','Negative: unfavorable overall.','Neutral: neither favorable nor unfavorable overall.','Positive: favorable overall.','Very positive: strongly favorable overall.']

def prepare_primitives(cfg,blocked,tokenizer):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    sizes=cfg.sizes();out={};sources={};api=HfApi()
    for name,repo in [('noul_boolq','google/boolq'),('score_sst5','SetFit/sst5')]:
        revision=api.dataset_info(repo).sha
        if not revision or len(revision)!=40:raise ValueError('Dataset revision did not resolve')
        ds=load_dataset(repo,revision=revision) # data only; datasets 4.x does not execute legacy remote scripts
        raw={}
        for split in ds:
            rows=[]
            for i,r in enumerate(ds[split]):
                if name=='noul_boolq':
                    if not isinstance(r.get('answer'),bool):raise ValueError('BoolQ requires author boolean labels')
                    state=r['passage'];question=r['question'];y=int(r['answer']);group=text_hash(state)
                    text='Answer the yes/no question using the passage.\nPassage:\n'+state+'\nQuestion:\n'+question+'\nDecision:'
                else:
                    state=r['text'];y=int(r['label']);group=text_hash(state)
                    if y not in range(5):raise ValueError('Unexpected SST-5 label')
                    text='Rate the overall sentiment of this movie-review sentence.\nLevels:\n'+'\n'.join(f'{j+1}: {s}' for j,s in enumerate(LEVELS))+'\nReview:\n'+state+'\nRating:'
                ids=tokenizer.encode(text,add_special_tokens=False)
                if len(ids)>cfg.primitive_max_length:continue # explicit eligibility rule; no silent evidence truncation
                rows.append({'id':name+'_'+split+'_'+str(i),'group':group,'text':text,'state':state,'label':y,'source_index':i,'source_split':split,'task':name,'label_provenance':'original dataset label','token_ids_reference':ids})
            raw[split]=rows
        def take(pool,n,seed):
            chosen=[]
            for r in sorted(pool,key=lambda r:content_hash([seed,r['group'],r['id']])):
                if r['group'] in blocked:continue
                chosen.append(r);blocked.add(r['group'])
                if len(chosen)==n:return chosen
            raise ValueError('Insufficient unused primitive groups for '+name)
        local={}
        for j,s in enumerate(('train','dev','cal_fit','cal_gate')):local[s]=take(raw['train'],sizes['primitive_train'] if s=='train' else sizes['primitive_aux'],cfg.seed+701+j)
        final_source='validation' if name=='noul_boolq' else 'test'
        if final_source not in raw:raise ValueError('Expected labeled author final partition missing')
        local['final']=take(raw[final_source],sizes['primitive_final'],cfg.seed+720)
        out[name]=local;sources[name]={'dataset':repo,'revision':revision,'split_fingerprints':{s:getattr(ds[s],'_fingerprint',None) for s in ds},'rows_before_length_filter':{s:len(ds[s]) for s in ds},'rows_after_length_filter':{s:len(v) for s,v in raw.items()},'final_author_split':final_source,'label_scope':'BoolQ yes/no on supplied passages' if name=='noul_boolq' else 'five-level sentence sentiment; no arbitrary scoring-rubric claim','selection_hash':content_hash(local)}
    return out,sources

class FixedReadout(nn.Module):
    def __init__(self,d,c):
        super().__init__();self.linear=nn.Linear(d,c);self.register_buffer('mean',torch.zeros(d));self.register_buffer('std',torch.ones(d))
    def forward(self,x):return self.linear((x.float()-self.mean)/self.std)

def fit_fixed(features,rows,cfg,seed,task):
    torch.manual_seed(seed);x=torch.as_tensor(features['train']);y=torch.tensor([r['label'] for r in rows['train']]);c=2 if task=='noul_boolq' else 5
    model=FixedReadout(x.shape[1],c);model.mean.copy_(x.mean(0));model.std.copy_(x.std(0,unbiased=False).clamp_min(1e-5));opt=torch.optim.AdamW(model.parameters(),lr=cfg.head_lr,weight_decay=cfg.head_weight_decay)
    best=None;bestloss=float('inf');history=[];stale=0;rng=np.random.default_rng(seed)
    for epoch in range(cfg.head_epochs):
        for ix in np.array_split(rng.permutation(len(x)),max(1,math.ceil(len(x)/cfg.head_batch_size))):
            z=model(x[ix]);loss=nn.functional.cross_entropy(z,y[ix]);opt.zero_grad();loss.backward();opt.step()
        with torch.no_grad():
            dx=torch.as_tensor(features['dev']);dy=torch.tensor([r['label'] for r in rows['dev']]);v=float(nn.functional.cross_entropy(model(dx),dy))
        history.append(v)
        if v<bestloss-1e-6:bestloss=v;best={k:t.detach().clone() for k,t in model.state_dict().items()};stale=0
        else:stale+=1
        if stale>=cfg.head_patience:break
    model.load_state_dict(best);model.eval()
    with torch.no_grad():zs={s:model(torch.as_tensor(features[s])).numpy() for s in ('cal_fit','cal_gate')}
    def loss_at(s,t):
        y=np.array([r['label'] for r in rows[s]]);z=zs[s]/t;return float(np.mean(logsumexp(z,axis=1)-z[np.arange(len(y)),y]))
    fit=minimize_scalar(lambda lt:loss_at('cal_fit',math.exp(lt)),bounds=(math.log(.25),math.log(4)),method='bounded')
    if not fit.success:raise RuntimeError('Primitive temperature fitting failed')
    t=math.exp(fit.x);temperature=t if loss_at('cal_gate',1)-loss_at('cal_gate',t)>=cfg.calibration_min_improvement else 1.
    return model,{'seed':seed,'task':task,'dev_nll':bestloss,'dev_history':history,'temperature':temperature,'label_order':['false','true'] if c==2 else [1,2,3,4,5],'output':'P(yes)' if c==2 else 'complete 5-level distribution and its expected level','limitations':'No dedicated insufficient-evidence ground truth; do not reinterpret unknown as false' if c==2 else 'Hard author sentiment classes, not target soft distributions or arbitrary rubric validation'}

def fixed_metrics(z,rows,temperature,cfg,task):
    z=np.asarray(z,float)/temperature;p=np.exp(z-logsumexp(z,axis=1,keepdims=True));y=np.array([r['label'] for r in rows]);correct=p.argmax(1)==y
    result={'n':len(rows),'groups':len({r['group'] for r in rows}),'accuracy':float(correct.mean()),'nll':float(np.mean(logsumexp(z,axis=1)-z[np.arange(len(y)),y])),'brier_sum_classes':float(np.square(p-np.eye(p.shape[1])[y]).sum(1).mean()),'accuracy_interval':bootstrap(correct,[r['group'] for r in rows],cfg.bootstrap_repeats,cfg.seed),'probabilities':p.tolist()}
    if task=='score_sst5':
        expected=p@np.arange(1,6);rps=np.square(np.cumsum(p,axis=1)[:,:-1]-(y[:,None]<=np.arange(4)[None,:])).mean(1)
        result.update(expected_level_mae=float(np.mean(np.abs(expected-(y+1)))),argmax_level_mae=float(np.mean(np.abs(p.argmax(1)-y))),ranked_probability_score=float(rps.mean()),expected_levels=expected.tolist())
    else:result['binary_brier_yes']=float(np.square(p[:,1]-y).mean())
    return result
