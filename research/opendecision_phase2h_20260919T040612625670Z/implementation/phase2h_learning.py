"""Small supervised interventions. Training, selection, calibration and policy sets are distinct."""
from dataclasses import asdict
import copy,math,json,time
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.special import logsumexp,expit
from scipy.optimize import minimize,minimize_scalar
from safetensors.torch import save_file,load_file
from phase2d_common import json_write,content_hash
from phase2d_decisions import score_features

ORIGINS=('answer_present','annotated_intent_omitted','author_oos')

def weights(episodes,masses=(.60,.25,.15)):
    """Equal message weight within each origin. No episode multiplication masquerades as n."""
    available=[o for o in ORIGINS if any(e['none_origin']==o for e in episodes)]
    mass={o:masses[ORIGINS.index(o)] for o in available};den=sum(mass.values())
    if den<=0:raise ValueError('No supported population strata')
    groups={o:{e['group'] for e in episodes if e['none_origin']==o} for o in available}
    counts=Counter((e['none_origin'],e['group']) for e in episodes)
    w=np.array([mass[e['none_origin']]/den/len(groups[e['none_origin']])/counts[(e['none_origin'],e['group'])] for e in episodes],float)
    if not np.isclose(w.sum(),1):raise AssertionError('Weight normalization')
    return w

def softmaxes(logits,temperature=1.):
    if not np.isfinite(temperature) or temperature<=0:raise ValueError('Temperature must be positive')
    out=[]
    for row in logits:
        x=np.asarray(row,float)/temperature
        if x.ndim!=1 or len(x)<2 or not np.isfinite(x).all():raise ValueError('Invalid logits')
        out.append(np.exp(x-logsumexp(x)))
    return out

def nll_vector(logits,eps,temperature=1.):
    return np.array([logsumexp(np.asarray(x)/temperature)-float(x[e['target_index']])/temperature for x,e in zip(logits,eps)])

def objective(logits,eps,masses=(.60,.25,.15),temperature=1.):return float(weights(eps,masses)@nll_vector(logits,eps,temperature))

class AffineScorer(nn.Module):
    def __init__(self,hidden):
        super().__init__();self.linear=nn.Linear(hidden,1);self.none=nn.Parameter(torch.zeros(()))
        self.register_buffer('mean',torch.zeros(hidden));self.register_buffer('std',torch.ones(hidden))
    def forward(self,x):return self.linear((x.float()-self.mean)/self.std).squeeze(-1)
    def preserve_legacy(self,legacy):
        state=legacy.state_dict();self.mean.copy_(state['scorer.feature_mean']);self.std.copy_(state['scorer.feature_std'])
        self.linear.weight.data.copy_(state['scorer.net.weight']);self.linear.bias.data.copy_(state['scorer.net.bias']);self.none.data.copy_(state['none_logit'])
        return self
    def fit_normalization(self,x,preserve=True):
        x=torch.as_tensor(x,dtype=torch.float32);mu=x.mean(0);sd=x.std(0,unbiased=False).clamp_min(1e-5)
        if preserve:
            effective=self.linear.weight.detach()/self.std;bias=self.linear.bias.detach()-(effective*self.mean).sum(-1)
            self.linear.weight.data.copy_(effective*sd);self.linear.bias.data.copy_(bias+(effective*mu).sum(-1))
        self.mean.copy_(mu);self.std.copy_(sd)
    def export(self,path):save_file({k:v.detach().cpu().contiguous() for k,v in self.state_dict().items()},str(path))
    @classmethod
    def load(cls,path):
        s=load_file(str(path));m=cls(s['mean'].numel());m.load_state_dict(s);m.eval();return m

def padded_features(features,eps,indices):
    width=max(len(eps[i]['choices']) for i in indices);d=features[indices[0]].shape[-1]
    x=torch.zeros((len(indices),width,d));mask=torch.zeros((len(indices),width),dtype=torch.bool);y=[]
    for j,i in enumerate(indices):
        k=len(features[i]);x[j,:k]=torch.as_tensor(features[i]);mask[j,:k]=True;y.append(width if eps[i]['target_index']==k else eps[i]['target_index'])
    return x,mask,torch.tensor(y,dtype=torch.long)

@torch.no_grad()
def predict_scores(model,features):return [model(torch.as_tensor(x,dtype=torch.float32)).cpu().numpy().astype(float) for x in features]

def global_logits(scores,b):return [np.r_[x,float(b)] for x in scores]

def train_scorer(train_x,train_ep,dev_x,dev_ep,cfg,seed,initial=None):
    torch.manual_seed(int(seed));model=copy.deepcopy(initial) if initial is not None else AffineScorer(train_x[0].shape[-1])
    # All normalization is fitted on train candidates only, never dev/calibration/final.
    model.fit_normalization(np.concatenate(train_x),preserve=initial is not None)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.head_lr,weight_decay=cfg.head_weight_decay)
    w=weights(train_ep,cfg.train_mixture);rng=np.random.default_rng(seed);history=[];best=None;best_loss=float('inf');stale=0;t0=time.perf_counter()
    for epoch in range(cfg.head_epochs):
        model.train();order=rng.permutation(len(train_ep));losses=[]
        for start in range(0,len(order),cfg.head_batch_size):
            ix=order[start:start+cfg.head_batch_size];x,mask,y=padded_features(train_x,train_ep,ix)
            s=model(x).masked_fill(~mask,-1e9);z=torch.cat([s,model.none.expand(len(ix),1)],-1)
            per=F.cross_entropy(z,y,reduction='none');ww=torch.as_tensor(w[ix]*len(train_ep),dtype=torch.float32)
            loss=(per*ww).mean();optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            if not torch.isfinite(loss):raise FloatingPointError('Head loss nonfinite')
            losses.append(float(loss.detach()))
        model.eval();ds=predict_scores(model,dev_x);metric=objective(global_logits(ds,float(model.none.detach())),dev_ep,cfg.train_mixture)
        history.append({'epoch':epoch+1,'training_batch_loss_mean':float(np.mean(losses)),'dev_weighted_nll':metric})
        if metric<best_loss-1e-6:best_loss=metric;best={k:v.detach().clone() for k,v in model.state_dict().items()};stale=0
        else:stale+=1
        if stale>=cfg.head_patience:break
    if best is None:raise RuntimeError('No finite head checkpoint')
    model.load_state_dict(best);model.eval()
    return model,{'seed':seed,'history':history,'dev_best_nll':best_loss,'seconds':time.perf_counter()-t0,'trainable_parameters':sum(p.numel() for p in model.parameters()),'backbone_updated':False,'objective':'message/origin-weighted candidate+none cross entropy'}

def none_scores(model,scores):
    f=score_features(scores,model['kind']);mu=np.asarray(model['feature_mean']);sd=np.asarray(model['feature_std']);x=np.column_stack([np.ones(len(f)),(f-mu)/sd]);return x@np.asarray(model['coef'])

def add_none(model,scores):return [np.r_[s,b] for s,b in zip(scores,none_scores(model,scores))]

def fit_none(scores,eps,kind,cfg,initial=0.):
    f=score_features(scores,kind);mu=f.mean(0);sd=np.maximum(f.std(0),1e-5);x=np.column_stack([np.ones(len(f)),(f-mu)/sd]);w=weights(eps,cfg.train_mixture)
    absent=np.array([e['target_index']==len(s) for e,s in zip(eps,scores)],float);lse=np.array([logsumexp(s) for s in scores]);true=np.array([0. if a else s[e['target_index']] for a,s,e in zip(absent,scores,eps)])
    def fun(theta):
        b=x@theta;loss=np.logaddexp(lse,b)-np.where(absent,b,true);g=x.T@(w*(expit(b-lse)-absent));g[1:]+=cfg.none_l2*theta[1:]
        return float(w@loss+.5*cfg.none_l2*np.dot(theta[1:],theta[1:])),g
    opt=minimize(fun,np.r_[initial,np.zeros(f.shape[1])],jac=True,method='L-BFGS-B',options={'maxiter':1000,'ftol':1e-12,'gtol':1e-8})
    if not opt.success or not np.isfinite(opt.fun):raise RuntimeError('None fitting failed: '+str(opt.message))
    return {'kind':kind,'coef':opt.x.tolist(),'feature_mean':mu.tolist(),'feature_std':sd.tolist()}, {'iterations':int(opt.nit),'train_weighted_nll':float(opt.fun),'coefficients':len(opt.x)}

def choose_none(scores,eps,cfg,initial=0.):
    choices=[]
    for kind in ('refit_global','set_linear'):
        m,info=fit_none(scores['train'],eps['train'],kind,cfg,initial);val=objective(add_none(m,scores['dev']),eps['dev'],cfg.train_mixture)
        choices.append({'model':m,'fit':info,'dev_nll':val})
    selected=min(choices,key=lambda r:(r['dev_nll'],len(r['model']['coef'])))
    return selected['model'],{'alternatives':choices,'criterion':'minimum predeclared development weighted NLL; ties prefer fewer coefficients'}

def calibrate(logits,eps,cfg):
    f=lambda lt:objective(logits['cal_fit'],eps['cal_fit'],cfg.train_mixture,math.exp(lt))
    opt=minimize_scalar(f,bounds=(math.log(.25),math.log(4.)),method='bounded')
    if not opt.success:raise RuntimeError('Temperature optimizer failed')
    t=math.exp(opt.x);base=objective(logits['cal_gate'],eps['cal_gate'],cfg.train_mixture);scaled=objective(logits['cal_gate'],eps['cal_gate'],cfg.train_mixture,t)
    chosen=t if base-scaled>=cfg.calibration_min_improvement else 1.
    return chosen,{'fitted':t,'selected':chosen,'gate_raw_nll':base,'gate_scaled_nll':scaled,'minimum_improvement':cfg.calibration_min_improvement,'selection_source':'independent cal_gate, never final'}

def policy_action(p,threshold):
    p=np.asarray(p);i=int(np.argmax(p[:-1]));return i if p[i]>=threshold and p[i]>p[-1] else None

def scenario_weights(eps,prior,oos_share):
    # Explicit policy mixture: present vs absent, with named OOS share within absent.
    return weights(eps,(1-prior,prior*(1-oos_share),prior*oos_share))

def policy_stats(ps,eps,threshold,prior,cost,cfg,with_intervals=False):
    act=[policy_action(p,threshold) for p in ps];answered=np.array([x is not None for x in act]);wrong=np.array([x is not None and x!=e['target_index'] for x,e in zip(act,eps)])
    w=scenario_weights(eps,prior,cfg.oos_share_of_absent);review=~answered
    result = {'threshold':float(threshold),'assumed_absent_prior':prior,'assumed_oos_share_of_absent':cfg.oos_share_of_absent,'wrong_cost':cost,'review_cost':cfg.review_cost,
        'scenario_cost':float(w@(cost*wrong+cfg.review_cost*review)),'always_review_cost':cfg.review_cost,'scenario_coverage':float(w@answered),
        'accepted_episodes':int(answered.sum()),'wrong_accepted_episodes':int(wrong.sum()),'accepted_groups':len({e['group'] for e,a in zip(eps,answered) if a}),
        'panel_error_among_accepted':float(wrong.sum()/answered.sum()) if answered.any() else None,'policy_selection_on_final':False,
        'effective_origin_masses':{o:float(sum(ww for ww,e in zip(w,eps) if e['none_origin']==o)) for o in ORIGINS},
        'complete_population_mixture_available':all(any(e['none_origin']==o for e in eps) for o in ORIGINS),
        'scope':'When a family omits strata, costs are conditioned on available strata and effective masses are reported; pooled mixture is separate'}
    if with_intervals:
        result['intervals']=policy_intervals(eps,answered,wrong,prior,cost,cfg)
    return result

def select_policies(ps,eps,cfg):
    thresholds=sorted(set([0.,.5,.8,.9,.95,.98,.99,.995,.999,1.]))
    out=[]
    for prior in cfg.absent_priors:
        for cost in cfg.wrong_answer_costs:
            candidates=[policy_stats(ps,eps,t,prior,cost,cfg) for t in thresholds]
            pick=min(candidates,key=lambda r:(r['scenario_cost'],-r['threshold']));out.append(pick)
    return out

def bootstrap(values,groups,repeats=500,seed=17):
    unique=sorted(set(groups));sums=np.array([sum(v for v,g in zip(values,groups) if g==u) for u in unique]);counts=np.array([sum(g==u for g in groups) for u in unique])
    if not unique:return {'groups':0,'estimate':None,'ci_low':None,'ci_high':None}
    if len(unique)<2:return {'groups':len(unique),'estimate':float(sums.sum()/counts.sum()),'ci_low':None,'ci_high':None}
    rng=np.random.default_rng(seed);ix=rng.integers(0,len(unique),size=(repeats,len(unique)));v=sums[ix].sum(1)/counts[ix].sum(1)
    return {'groups':len(unique),'estimate':float(sums.sum()/counts.sum()),'ci_low':float(np.quantile(v,.025)),'ci_high':float(np.quantile(v,.975)),'method':'message-cluster bootstrap; not pretraining uncertainty or rare-event certification'}

def quality(ps,eps,cfg):
    pred=np.array([int(np.argmax(p)) for p in ps]);y=np.array([e['target_index'] for e in eps]);ks=np.array([len(e['choices']) for e in eps]);correct=pred==y
    nll=np.array([-math.log(max(float(p[t]),1e-300)) for p,t in zip(ps,y)]);br=np.array([float(np.square(p-np.eye(len(p))[t]).sum()) for p,t in zip(ps,y)]);conf=np.array([max(p) for p in ps]);bucket=np.minimum((conf*15).astype(int),14);ece=0
    for k in range(15):
        sel=bucket==k
        if sel.any():ece+=float(sel.mean()*abs(correct[sel].mean()-conf[sel].mean()))
    result={'episodes':len(eps),'messages':len({e['group'] for e in eps}),'accuracy':float(correct.mean()),'nll':float(nll.mean()),'brier_sum_classes':float(br.mean()),'ece_15bins':ece,
        'accuracy_interval':bootstrap(correct,[e['group'] for e in eps],cfg.bootstrap_repeats,cfg.seed),'assumption_weighted_nll':float(weights(eps,cfg.train_mixture)@nll),'conditional':{}}
    for origin in sorted({e['none_origin'] for e in eps}):
        sel=np.array([e['none_origin']==origin for e in eps]);answer=pred<ks
        result['conditional'][origin]={'episodes':int(sel.sum()),'messages':len({e['group'] for e,s in zip(eps,sel) if s}),'accuracy':float(correct[sel].mean()),'none_rate':float((~answer)[sel].mean()),'offered_answer_error_rate':float((answer&~correct)[sel].mean())}
    return result

def parity(ps,qs,eps,tolerance,policies):
    delta=[float(np.max(np.abs(p-q))) for p,q in zip(ps,qs)];arg=[int(np.argmax(p))!=int(np.argmax(q)) for p,q in zip(ps,qs)];changes=[]
    for p,q in zip(ps,qs):changes.append(any(policy_action(p,r['threshold'])!=policy_action(q,r['threshold']) for r in policies))
    return {'episodes':len(eps),'messages':len({e['group'] for e in eps}),'max_probability_delta':max(delta,default=0.),'over_tolerance':sum(d>tolerance for d in delta),'argmax_changes':sum(arg),'policy_output_changes':sum(changes),'accepted':sum(d<=tolerance and not a and not c for d,a,c in zip(delta,arg,changes))}


def policy_intervals(eps,answered,wrong,prior,cost,cfg):
    """Stratified message-cluster bootstrap preserving paired omission and family sizes.
    Conditional on the sampled families/population mixture. Zero observed errors do
    not produce a valid rare-error upper bound; degenerate bootstrap CIs are marked.
    """
    groups=sorted({e['group'] for e in eps});lookup={g:i for i,g in enumerate(groups)}
    counts=np.zeros((len(groups),3));sums=np.zeros((3,len(groups),3));families={}
    for j,e in enumerate(eps):
        i=lookup[e['group']];o=ORIGINS.index(e['none_origin']);counts[i,o]+=1
        sums[:,i,o]+=[cost*wrong[j]+cfg.review_cost*(not answered[j]),float(answered[j]),float(wrong[j])]
        families[i]=e['family']
    means=np.divide(sums,counts[None,:,:],out=np.zeros_like(sums),where=counts[None,:,:]>0)
    rng=np.random.default_rng(cfg.seed+778);mult=np.zeros((cfg.bootstrap_repeats,len(groups)))
    strata={}
    for i,family in families.items():strata.setdefault((family,tuple(counts[i]>0)),[]).append(i)
    for ids in strata.values():
        ids=sorted(ids);mult[:,ids]=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=cfg.bootstrap_repeats)
    available=(counts>0).sum(0)>0;mass=np.array([1-prior,prior*(1-cfg.oos_share_of_absent),prior*cfg.oos_share_of_absent])*available;mass/=mass.sum()
    den=mult@(counts>0);draw=[]
    for v in means:
        numerator=mult@v;conditional=np.divide(numerator,den,out=np.zeros_like(numerator),where=den>0);draw.append(conditional@mass)
    result={'groups':len(groups),'repeats':cfg.bootstrap_repeats,'method':'95% percentile message-cluster bootstrap stratified by family/origin membership; paired variants remain together; conditional on fixed family/stratum mix',
            'zero_observed_accepted_errors':not bool(np.any(wrong)), 'rare_event_certification':False}
    for label,values in zip(('scenario_cost','scenario_coverage'),draw):
        result[label]={'ci_low':float(np.quantile(values,.025)),'ci_high':float(np.quantile(values,.975))}
    return result


def paired_quality(ps,qs,eps,cfg):
    """Alternative minus reference; raw panel contrasts, not population reweighting."""
    y=[e['target_index'] for e in eps];groups=[e['group'] for e in eps]
    acc=[float(int(np.argmax(q))==t)-float(int(np.argmax(p))==t) for p,q,t in zip(ps,qs,y)]
    nll=[-math.log(max(float(q[t]),1e-300))+math.log(max(float(p[t]),1e-300)) for p,q,t in zip(ps,qs,y)]
    return {'accuracy_alternative_minus_reference':bootstrap(acc,groups,cfg.bootstrap_repeats,cfg.seed),
            'nll_alternative_minus_reference':bootstrap(nll,groups,cfg.bootstrap_repeats,cfg.seed),'scope':'paired final-test report only; does not select or modify any fitted model'}
