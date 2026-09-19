"""Optional matched LoRA/online-head treatment and state-first query mechanics.
No optimized CUDA kernels, Rust/Metal parity or general arbitrary-question validity is claimed.
"""
import copy,json,re,time,math
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import save_file,load_file
from phase2h_learning import AffineScorer,predict_scores,choose_none,add_none,calibrate,select_policies,softmaxes,objective,parity
from phase2h_runtime import encode,FeatureStore
from phase2h_data import apply_criteria
from phase2d_common import json_write,content_hash,forward_features
from phase2e_core import StageSkip

class LowRankLinear(nn.Module):
    """Exact W x + (alpha/r) B A x; B starts at zero. Base weight remains frozen."""
    def __init__(self,base,rank,alpha):
        super().__init__();self.base=base;self.rank=rank;self.alpha=alpha
        for p in self.base.parameters():p.requires_grad_(False)
        self.lora_A=nn.Parameter(torch.empty(rank,base.in_features,device=base.weight.device,dtype=base.weight.dtype))
        self.lora_B=nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device,dtype=base.weight.dtype));nn.init.kaiming_uniform_(self.lora_A,a=math.sqrt(5))
    def forward(self,x):return self.base(x)+(self.alpha/self.rank)*F.linear(F.linear(x,self.lora_A),self.lora_B)

def inject(model,cfg,targets=None):
    if targets is None:
        count=model.config.num_hidden_layers;targets=[]
        for name,module in model.named_modules():
            match=re.search(r'(?:^|\.)layers\.(\d+)\.',name)
            if match and int(match.group(1))>=count-cfg.lora_last_blocks and isinstance(module,nn.Linear):targets.append(name)
    if not targets:raise ValueError('No LoRA projection targets found')
    for name in targets:
        parent,leaf=name.rsplit('.',1);owner=model.get_submodule(parent);module=getattr(owner,leaf)
        if not isinstance(module,nn.Linear):raise ValueError('LoRA target is not an unchanged linear module: '+name)
        setattr(owner,leaf,LowRankLinear(module,cfg.lora_rank,cfg.lora_alpha))
    return targets

def base_sample(model):
    vals={}
    for name,p in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:continue
        canonical=name.replace('.base.','.')
        ids=sorted({(p.numel()-1)*i//16 for i in range(17)}) if p.numel() else []
        vals[canonical]=p.detach().reshape(-1)[torch.tensor(ids,device=p.device,dtype=torch.long)].float().cpu().tolist()
    return content_hash(vals)

def load_adapter(rt,fitroot):
    path=Path(fitroot)/'adapter_manifest.json'
    if not path.exists():return
    spec=json.loads(path.read_text());cfg=copy.copy(rt.cfg);cfg.lora_rank=spec['rank'];cfg.lora_alpha=spec['alpha'];inject(rt.model,cfg,spec['targets'])
    values=load_file(str(Path(fitroot)/'adapter.safetensors'))
    for name,t in rt.model.named_parameters():
        if name in values:t.data.copy_(values[name].to(t.device))
    rt.model.eval();rt.store.close();rt.store=FeatureStore(rt.out/'adapted_features.sqlite',dict(rt.identity,adapter=spec['sha256']))

def online_fit(rt,data,cfg,out,report,job):
    """BCE pair training in both matched arms, not misreported as the main joint-CE treatment."""
    from phase2h_worker import artifact_profile
    if torch.cuda.get_device_properties(0).total_memory/1024**3<cfg.lora_min_gpu_gib:
        raise StageSkip(f'Optional online/LoRA matched treatment requires >= {cfg.lora_min_gpu_gib:g} GiB total GPU memory. No quantized or BF16 substitution.')
    seed=job['online_seed'];torch.manual_seed(seed)
    parent=Path(job['parent_fit']);baseprofiles=json.loads((parent/'profiles.json').read_text())
    options=[p for p in baseprofiles if p['criteria']=='original' and p['layout']=='instruction_first' and f'joint_seed{seed}' in p['name']]
    p=min(options,key=lambda p:p['dev_nll']);head=AffineScorer.load(parent/p['scorer_path']).to(rt.device)
    before=base_sample(rt.model);targets=inject(rt.model,cfg) if job['online_kind']=='lora' else []
    head.none.requires_grad_(False)
    trainable=[x for x in rt.model.parameters() if x.requires_grad]+[x for x in head.parameters() if x.requires_grad];optimizer=torch.optim.AdamW(trainable,lr=cfg.lora_lr,weight_decay=cfg.head_weight_decay)
    episodes=data['episodes']['train'];pairs=[]
    # Same bounded pair set, order, loss, initialization and update budget in both arms.
    for ep in episodes:
        for i,choice in enumerate(ep['choices']):pairs.append((ep,choice,float(i==ep['target_index'])))
    pairs=sorted(pairs,key=lambda r:content_hash([seed,r[0]['id'],r[1]['id']]))[:cfg.lora_pairs_cap]
    actual_updates=0;losses=[];optimizer.zero_grad();t0=time.perf_counter()
    for epoch in range(cfg.lora_epochs):
        order=np.random.default_rng(seed+epoch).permutation(len(pairs));last=len(order)
        for j,idx in enumerate(order):
            ep,c,y=pairs[int(idx)];ids=encode(ep,c,rt.tokenizer,cfg)
            with torch.set_grad_enabled(bool(targets)):
                output=rt.model(**rt.pack([ids]),use_cache=False,return_dict=True);h=output.last_hidden_state[:,-1,:].float()
            z=head(h);loss=F.binary_cross_entropy_with_logits(z,torch.tensor([y],device=rt.device))
            # Each accumulation group, including the last partial group, has its true denominator.
            group_start=(j//cfg.lora_accumulation)*cfg.lora_accumulation;den=min(cfg.lora_accumulation,last-group_start)
            (loss/den).backward();losses.append(float(loss.detach()))
            if (j+1)%cfg.lora_accumulation==0 or j+1==last:
                norm=torch.nn.utils.clip_grad_norm_(trainable,1.)
                if not torch.isfinite(norm):raise FloatingPointError('Nonfinite online gradient')
                optimizer.step();optimizer.zero_grad();actual_updates+=1
            del output,h,z,loss
            if (j+1)%32==0:print(f'{job["online_kind"]} epoch {epoch+1}: {j+1}/{last} pairs',flush=True)
    if base_sample(rt.model)!=before:raise RuntimeError('A frozen base-weight sample changed during LoRA')
    head=head.cpu().eval();head.export(out/'online_head.safetensors');rt.model.eval()
    if targets:
        state={n:t.detach().cpu().contiguous() for n,t in rt.model.named_parameters() if 'lora_' in n};save_file(state,str(out/'adapter.safetensors'))
        from phase2d_common import file_sha
        manifest={'targets':targets,'rank':cfg.lora_rank,'alpha':cfg.lora_alpha,'sha256':file_sha(out/'adapter.safetensors'),'base_revision':rt.spec['revision']};json_write(out/'adapter_manifest.json',manifest)
    rt.store.close();rt.store=FeatureStore(out/'adapted_features.sqlite',dict(rt.identity,online=job['online_kind'],seed=seed,targets=targets,updates=actual_updates))
    ff={s:rt.episodes(es) for s,es in data['episodes'].items()};scores={s:predict_scores(head,x) for s,x in ff.items()};none,sel=choose_none(scores,data['episodes'],cfg,float(head.none.detach()));logits={s:add_none(none,x) for s,x in scores.items()};temp,cal=calibrate(logits,data['episodes'],cfg)
    policies=select_policies(softmaxes(logits['policy_dev'],temp),data['episodes']['policy_dev'],cfg)
    name=job['online_kind']+'_matched_bce_seed'+str(seed);profile=artifact_profile(name,'affine','original','instruction_first','online_head.safetensors',none,temp,policies,dev_nll=objective(logits['dev'],data['episodes']['dev'],cfg.train_mixture),none_selection=sel,calibration=cal,training={'loss':'binary candidate-match loss','pairs':len(pairs),'epochs':cfg.lora_epochs,'updates':actual_updates,'seed':seed,'seconds':time.perf_counter()-t0,'parameters_updated':sum(p.numel() for p in trainable),'base_sample_unchanged':True,'scope':'Compare ONLY against matched online_head arm for the effect of LoRA; not equal optimization to joint-CE head study'})
    json_write(out/'profiles.json',[profile]);json_write(out/'selection.json',{'selected_profile':name,'primitive_profiles':[]})
    report.update(profiles=[profile],selected_profile=name,primitive_profiles=[],online_training=profile['training']);return report

def query_mechanics(rt,pred,episodes,profiles,cfg,out):
    candidates=[p for p in profiles if p['layout']=='state_first' and p['method']=='affine' and 'joint_seed' in p['name']]
    if not candidates:return {'status':'unavailable','reason':'No fitted state-first arm'}
    p=min(candidates,key=lambda p:p['dev_nll']);groups=sorted({e['group'] for e in episodes})[:2];rows=[]
    instructions=['Select the intent that best matches the current request.','Which supplied intent describes this request?','Classify the request using these intent descriptions.','Identify its matching intent; the offered set may omit it.']
    for g in groups:
        for k in cfg.candidate_counts:
            options=[e for e in episodes if e['group']==g and e['K']==k]
            for qcount in (1,2,4):
                queries=[]
                for i in range(qcount):
                    e=pred.episode(options[i%len(options)],p);e['instruction']=instructions[i];queries.append(e)
                seqs=[encode(e,c,rt.tokenizer,cfg,'state_first') for e in queries for c in e['choices']]
                baseline=rt.run_sequences(seqs,'full_sequential');cached=rt.run_sequences(seqs,'shared_lossless',verify=True);head=pred.model(p)
                a=predict_scores(head,[baseline[i*k:(i+1)*k] for i in range(qcount)]);b=predict_scores(head,[cached[i*k:(i+1)*k] for i in range(qcount)])
                aa=softmaxes(add_none(p['none_model'],a),p['temperature']);bb=softmaxes(add_none(p['none_model'],b),p['temperature'])
                # Reverse question execution order, then map branches back; no root mutation allowed.
                revseqs=[s for i in reversed(range(qcount)) for s in seqs[i*k:(i+1)*k]];rev=rt.run_sequences(revseqs,'shared_lossless',verify=True)
                cc=softmaxes(add_none(p['none_model'],predict_scores(head,[rev[(qcount-1-i)*k:(qcount-i)*k] for i in range(qcount)])),p['temperature'])
                from phase2h_worker import time_request
                _,timing=time_request(lambda:rt.run_sequences(seqs,'shared_lossless'),cfg.benchmark_repeats)
                rows.append({'group':g,'Q':qcount,'K':k,'cached_vs_full':parity(aa,bb,queries,cfg.probability_tolerance,p['policies']),'reversed_question_order':parity(bb,cc,queries,cfg.probability_tolerance,p['policies']),'timing':timing,'scope':'Related intent questions and differing offered sets; computational isolation/scaling, not arbitrary independent-question semantic validation; these timings start from finalized tokens'})
    json_write(out/'state_first_queries.json',rows);return {'status':'completed','rows':rows}
