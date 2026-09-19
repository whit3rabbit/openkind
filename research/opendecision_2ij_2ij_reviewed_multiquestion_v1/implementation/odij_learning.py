"""Supervised ragged candidate heads, controlled rejection ablations, calibration and policy.
All fitted quantities use named nonfinal partitions. Final outcomes never choose the winner.
"""
from __future__ import annotations
import math, copy, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import minimize_scalar
from safetensors.torch import save_file,load_file
from odij_core import *
from odij_data import choices_for,option_text

def softmax(x,t=1.):
    x=finite_array(x,1)/float(t)
    if not math.isfinite(t) or t<=0:raise ValueError('Temperature must be positive')
    x=x-x.max();e=np.exp(x);return e/e.sum()
def weights(rows):
    """Equal family mass; equal source-group mass within family; equal variants within group."""
    f=defaultdict(lambda:defaultdict(list))
    for i,r in enumerate(rows):f[r['q']['family']][r['group']].append(i)
    w=np.zeros(len(rows))
    for fam,gs in f.items():
        for g,ix in gs.items():w[ix]=1/len(f)/len(gs)/len(ix)
    return w

def nll(logits,rows,t=1.):
    if not rows or len(logits)!=len(rows):raise ValueError('Missing or mismatched evaluation rows')
    values=[]
    for x,r in zip(logits,rows):
        x=finite_array(x,1)/t;v=x.max();values.append(v+np.log(np.exp(x-v).sum())-x[r['target']])
    return float(weights(rows)@np.asarray(values))

def output_row(r,p):
    p=finite_array(p,1)
    if np.any(p<0) or not np.isclose(p.sum(),1.,atol=1e-6):raise IntegrityError('Not a probability vector')
    opts=choices_for(r['q'])
    if len(opts)!=len(p):raise IntegrityError('Option/probability dimension differs')
    return {'id':r['id'],'group':r['group'],'state_id':r['state_id'],'family':r['q']['family'],'rubric_family':r['q']['rubric_family'],
       'primitive':r['q']['primitive'],'origin':r['q']['origin'],'source_kind':r['source_kind'],
       'candidate_ids':[o['id'] for o in opts],'probabilities':p.tolist(),'target':int(r['target']),
       'prediction':int(p.argmax()),'top_probability':float(p.max()),'correct':bool(p.argmax()==r['target']),
       'nll':float(-np.log(max(p[r['target']],np.finfo(float).tiny))),
       'brier':float(np.square(p-np.eye(len(p))[r['target']]).sum()),
       'values':[o.get('value') for o in opts] if r['q']['primitive']=='score' else None}

def metrics(preds,bootstrap=200):
    if not preds:return {'episodes':0,'messages':0}
    def simple(rr):
        conf=np.array([r['top_probability'] for r in rr]);acc=np.array([r['correct'] for r in rr],float)
        ece=0.
        for i in range(15):
            ix=np.minimum((conf*15).astype(int),14)==i
            if ix.any():ece+=ix.mean()*abs(acc[ix].mean()-conf[ix].mean())
        out={'episodes':len(rr),'messages':len({r['group'] for r in rr}),'accuracy':float(acc.mean()),'nll':float(np.mean([r['nll'] for r in rr])),
             'brier':float(np.mean([r['brier'] for r in rr])),'ece_15bins':float(ece)}
        score=[r for r in rr if r['primitive']=='score']
        if score:
            out['ordinal_argmax_mae']=float(np.mean([abs(r['values'][r['prediction']]-r['values'][r['target']]) for r in score]))
            out['ordinal_expected_value_mae']=float(np.mean([abs(np.asarray(r['values'])@np.asarray(r['probabilities'])-r['values'][r['target']]) for r in score]))
        return out
    out=simple(preds);by=defaultdict(list);cond=defaultdict(list)
    for r in preds:by[r['family']].append(r);cond[r['origin']].append(r)
    out['by_family']={k:simple(v) for k,v in by.items()};out['by_origin']={}
    for k,v in cond.items():
        m=simple(v);m['none_rate']=float(np.mean([r['candidate_ids'][r['prediction']]==NONE for r in v]));out['by_origin'][k]=m
    # Match the registered family/group objective rather than averaging unequal family row counts.
    rr=[{'q':{'family':r['family']},'group':r['group']} for r in preds]
    out['family_macro_nll']=float(weights(rr)@np.array([r['nll'] for r in preds]))
    if bootstrap:
        gs=sorted({r['group'] for r in preds});lookup={g:[r for r in preds if r['group']==g] for g in gs}
        sums=np.array([sum(r['correct'] for r in lookup[g]) for g in gs]);counts=np.array([len(lookup[g]) for g in gs]);rng=np.random.default_rng(2929)
        ix=rng.integers(0,len(gs),(bootstrap,len(gs)));vals=sums[ix].sum(1)/counts[ix].sum(1)
        out['accuracy_cluster_ci95']=list(map(float,np.quantile(vals,[.025,.975])))
        out['uncertainty_limit']='Source-group bootstrap; no pretraining, label-validity, training-seed or rare-error certification.'
    return out

def policy_records(preds,policy,threshold):
    good=[];accepted=[];wrows=[]
    for r in preds:
        acc=r['candidate_ids'][r['prediction']]!=NONE and r['top_probability']>=threshold
        accepted.append(acc);good.append(r['correct']);wrows.append({'q':{'family':r['family']},'group':r['group']})
    a=np.array(accepted);correct=np.array(good);w=weights(wrows)
    return {'threshold':float(threshold),'episodes':len(preds),'accepted_episodes':int(a.sum()),'wrong_accepted':int((a&~correct).sum()),
        'raw_coverage':float(a.mean()),'panel_error_among_accepted':float((~correct[a]).mean()) if a.any() else None,
        'family_group_weighted_coverage':float(w@a),'family_group_weighted_cost':float(w@np.where(a,np.where(correct,0.,policy['wrong_cost']),policy['review_cost'])),
        'always_review_cost':policy['review_cost'],'selection_on_final':False}

def choose_policy(preds,policy):
    rows=[policy_records(preds,policy,t) for t in policy['threshold_grid']]
    # Conservative tie break, declared here and hashed before evaluation.
    best=min(rows,key=lambda x:(x['family_group_weighted_cost'],-x['threshold']))
    return {'selected':best,'candidates':rows,'selected_on':'policy_dev','coverage_is_not_quality':True}

class Scorer(nn.Module):
    def __init__(self,h):
        super().__init__();self.register_buffer('mean',torch.zeros(h));self.register_buffer('std',torch.ones(h));self.linear=nn.Linear(h,1);self.none=nn.Parameter(torch.zeros(()))
    def normalized(self,x):return (x.float()-self.mean)/self.std
    def real(self,x):return self.linear(self.normalized(x)).squeeze(-1)
    def forward(self,x,has_none=True):
        z=self.real(x);return torch.cat([z,self.none.reshape(1)]) if has_none else z
    def fit_normalization(self,x):
        self.mean.copy_(x.mean(0));self.std.copy_(x.std(0,unbiased=False).clamp_min(1e-5))
    def change_normalization_preserving_scores(self,x):
        with torch.no_grad():
            oldm=self.mean.clone();olds=self.std.clone();w=self.linear.weight.clone()
            self.fit_normalization(x);self.linear.weight.copy_(w*(self.std/olds));self.linear.bias.add_((w*(self.mean-oldm)/olds).sum(1))

def stats(z):
    """Symmetric six-score feature vector, matching the concept—not old fitted coefficients."""
    top=torch.topk(z,min(2,len(z))).values
    return torch.stack([top[0],top[0]-top[1] if len(top)>1 else z.new_zeros(()),z.mean(),z.std(unbiased=False),torch.logsumexp(z,0)-math.log(len(z)),z.new_tensor(math.log(len(z)))])

class Rejector(nn.Module):
    def __init__(self,kind,h):
        super().__init__();self.kind=kind
        d=1 if kind=='constant' else 6 if kind=='score_summary' else 6+2*h
        self.register_buffer('mean',torch.zeros(d));self.register_buffer('std',torch.ones(d))
        self.net=nn.Linear(d,1) if kind!='semantic_features' else nn.Sequential(nn.Linear(d,32),nn.Tanh(),nn.Linear(32,1))
        if kind=='constant':self.net.weight.requires_grad_(False);self.net.weight.data.zero_()
    def features(self,z,x):
        if self.kind=='constant':return z.new_zeros(1)
        s=stats(z)
        if self.kind=='score_summary':return s
        # Mean plus score-weighted semantic representation; invariant to candidate permutation.
        return torch.cat([s,x.mean(0),(torch.softmax(z.detach(),0)[:,None]*x).sum(0)])
    def forward(self,z,x):return self.net((self.features(z,x)-self.mean)/self.std).reshape(1)

class FrozenDecision(nn.Module):
    def __init__(self,h,kind='constant'):
        super().__init__();self.rank=Scorer(h);self.reject=Rejector(kind,h)
    def forward(self,x,has_none):
        z=self.rank.real(x)
        return torch.cat([z,self.reject(z,self.rank.normalized(x))]) if has_none else z

def fit_scorer(features,rows,p,seed,checkpoint=None):
    torch.manual_seed(seed);model=Scorer(features[0].shape[1]);xx=[torch.tensor(x,dtype=torch.float32) for x in features]
    train=[i for i,r in enumerate(rows) if r['split']=='train'];dev=[i for i,r in enumerate(rows) if r['split']=='dev']
    if not train or not dev:raise GateBlocked('Train/dev features missing')
    model.fit_normalization(torch.cat([xx[i] for i in train]));opt=torch.optim.AdamW(model.parameters(),lr=p['head_lr'],weight_decay=p['weight_decay'])
    w=weights([rows[i] for i in train]);best=None;bestloss=float('inf');bad=0;history=[]
    for epoch in range(p['head_epochs']):
        order=np.random.default_rng(seed+epoch).permutation(len(train))
        model.train()
        for j in order:
            i=train[j];z=model(xx[i],rows[i]['q']['primitive']=='choice');loss=F.cross_entropy(z[None,:],torch.tensor([rows[i]['target']]))*float(w[j]*len(train))
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        model.eval()
        with torch.no_grad():loss=nll([model(xx[i],rows[i]['q']['primitive']=='choice').numpy() for i in dev],[rows[i] for i in dev])
        history.append({'epoch':epoch+1,'dev_family_macro_nll':loss})
        if loss<bestloss-1e-8:bestloss=loss;best=copy.deepcopy(model.state_dict());bad=0
        else:bad+=1
        if bad>=p['head_patience']:break
    model.load_state_dict(best);return model,history

def fit_rejector(rank,features,rows,p,kind,seed):
    torch.manual_seed(seed);d=FrozenDecision(features[0].shape[1],kind);d.rank.load_state_dict(rank.state_dict())
    for par in d.rank.parameters():par.requires_grad_(False)
    xx=[torch.tensor(x,dtype=torch.float32) for x in features]
    train=[i for i,r in enumerate(rows) if r['split']=='train' and r['q']['primitive']=='choice'];dev=[i for i,r in enumerate(rows) if r['split']=='dev']
    if not train:raise GateBlocked('No Choice rejection training cases')
    with torch.no_grad():
        f=torch.stack([d.reject.features(d.rank.real(xx[i]),d.rank.normalized(xx[i])) for i in train]);d.reject.mean.copy_(f.mean(0));d.reject.std.copy_(f.std(0,unbiased=False).clamp_min(1e-5))
    opt=torch.optim.AdamW([v for v in d.reject.parameters() if v.requires_grad],lr=p['head_lr'],weight_decay=p['weight_decay']);w=weights([rows[i] for i in train]);best=None;bl=float('inf');bad=0;hist=[]
    for epoch in range(p['head_epochs']):
        for j in np.random.default_rng(seed+epoch).permutation(len(train)):
            i=train[j];loss=F.cross_entropy(d(xx[i],True)[None,:],torch.tensor([rows[i]['target']]))*float(w[j]*len(train))
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(d.reject.parameters(),1.);opt.step()
        with torch.no_grad():loss=nll([d(xx[i],rows[i]['q']['primitive']=='choice').numpy() for i in dev],[rows[i] for i in dev])
        hist.append({'epoch':epoch+1,'dev_family_macro_nll':loss})
        if loss<bl-1e-8:best=copy.deepcopy(d.state_dict());bl=loss;bad=0
        else:bad+=1
        if bad>=p['head_patience']:break
    d.load_state_dict(best);d.eval();return d,hist

def calibrate(logits,rows,p):
    ix={s:[i for i,r in enumerate(rows) if r['split']==s] for s in ('cal_fit','cal_gate')}
    if any(not x for x in ix.values()):raise GateBlocked('Missing calibration splits')
    def calc(t,s):return nll([logits[i] for i in ix[s]],[rows[i] for i in ix[s]],t)
    fit=minimize_scalar(lambda lt:calc(float(np.exp(lt)),'cal_fit'),bounds=(math.log(.25),math.log(4.)),method='bounded')
    if not fit.success:raise RuntimeError('Temperature fit failed')
    t=float(np.exp(fit.x));before=calc(1.,'cal_gate');after=calc(t,'cal_gate')
    return {'fitted':t,'selected':t if before-after>=p['calibration_min_nll_gain'] else 1.,'gate_before':before,'gate_after':after,'selected_on':'cal_gate','selection_on_final':False}

def package_prediction(logits,rows,p):
    cal=calibrate(logits,rows,p);pred=[output_row(r,softmax(z,cal['selected'])) for r,z in zip(rows,logits)]
    bysplit={s:[pr for pr,r in zip(pred,rows) if r['split']==s] for s in ('train','dev','cal_fit','cal_gate','policy_dev')}
    return {'calibration':cal,'development':metrics(bysplit['dev'],0),'policy':choose_policy(bysplit['policy_dev'],p['policy'])},pred


def fit_frozen_variants(features,rows,p,out,seed,rank=None):
    out.mkdir(parents=True,exist_ok=True)
    rank_path=out/('rank_seed'+str(seed)+'.safetensors')
    rank_meta=out/('rank_seed'+str(seed)+'.json')
    if rank is None and rank_meta.exists():
        rm=read_json(rank_meta)
        if file_sha(rank_path)!=rm['sha256']:raise IntegrityError('Saved rank changed')
        rank=Scorer(features[0].shape[1]);rank.load_state_dict(load_file(str(rank_path)));history=rm['history']
    elif rank is None:
        rank,history=fit_scorer(features,rows,p,seed)
        save_file({k:v.detach().cpu().contiguous() for k,v in rank.state_dict().items()},str(rank_path));write_json(rank_meta,{'sha256':file_sha(rank_path),'history':history})
    else:history=[{'source':'matched online ranking model; not refitted for this ablation'}]
    records=[]
    for kind in p['rejection_models']:
        name=kind+'_seed'+str(seed);path=out/(name+'.safetensors');meta=out/(name+'.json')
        if meta.exists() and path.exists():
            m=read_json(meta)
            if file_sha(path)!=m['weights_sha256']:raise IntegrityError('Completed head weights changed')
            records.append(m);continue
        d,nh=fit_rejector(rank,features,rows,p,kind,seed)
        with torch.no_grad():logits=[d(torch.tensor(x),r['q']['primitive']=='choice').numpy() for x,r in zip(features,rows)]
        summary,preds=package_prediction(logits,rows,p)
        save_file({k:v.detach().cpu().contiguous() for k,v in d.state_dict().items()},str(path))
        m={'name':name,'kind':kind,'seed':seed,'hidden':features[0].shape[1],'rank_history':history,'rejection_history':nh,'weights_sha256':file_sha(path),**summary}
        write_json(meta,m);write_json(out/(name+'_nonfinal_predictions.json'),preds);records.append(m)
    return records


def load_decision(path,meta):
    d=FrozenDecision(meta['hidden'],meta['kind']);d.load_state_dict(load_file(str(path)));d.eval();return d
