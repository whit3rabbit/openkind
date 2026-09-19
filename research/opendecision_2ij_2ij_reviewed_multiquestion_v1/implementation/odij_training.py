"""Checkpointed online-head/LoRA, full ModernBERT, and selected-token LoRA training.
This is a declared new supervised study, not a rerun or retraining of Phase 2H.
"""
from __future__ import annotations
import copy,io,os,shutil,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import save_file,load_file
from odij_core import *
from odij_data import choices_for
from odij_runtime import inject_lora,segments,memory
from odij_learning import Scorer,weights

def trainable_state(model,head):
    d={f'model.{n}':p.detach().cpu().clone().contiguous() for n,p in model.named_parameters() if p.requires_grad}
    if head is not None:d.update({f'head.{n}':v.detach().cpu().clone().contiguous() for n,v in head.state_dict().items()})
    return d

def apply_state(model,head,d):
    expected={f'model.{n}' for n,p in model.named_parameters() if p.requires_grad}
    if head is not None:expected|={f'head.{n}' for n in head.state_dict()}
    if set(d)!=expected:raise IntegrityError('Training checkpoint parameter keys differ')
    with torch.no_grad():
        for n,p in model.named_parameters():
            k='model.'+n
            if k in d:
                if p.shape!=d[k].shape:raise IntegrityError('Training shape mismatch')
                p.copy_(d[k].to(p.device))
        if head is not None:head.load_state_dict({n[5:]:v.to(next(head.parameters()).device) for n,v in d.items() if n.startswith('head.')})

def save_training(out,identity,step,model,head,opt,extra):
    gen=f'ckpt-{step:08d}-{time.time_ns()}';dst=out/gen;dst.mkdir(parents=True)
    save_file(trainable_state(model,head),str(dst/'trainable.safetensors'))
    rng={'cpu':torch.get_rng_state()}
    if torch.cuda.is_available():rng.update({f'cuda{i}':s for i,s in enumerate(torch.cuda.get_rng_state_all())})
    save_file(rng,str(dst/'rng.safetensors'))
    # Own optimizer file; load only with weights_only=True and after manifest/hash validation.
    torch.save(opt.state_dict(),dst/'optimizer.pt')
    write_json(dst/'progress.json',{'identity':identity,'next_step':step,**extra})
    write_json(dst/'COMPLETE.json',{'files':tree_hashes(dst)})
    write_json(out/'LATEST_TRAIN.json',{'generation':gen,'manifest_sha256':file_sha(dst/'COMPLETE.json')})
    for old in sorted(out.glob('ckpt-*'))[:-2]:
        if (old/'COMPLETE.json').exists():shutil.rmtree(old)

def restore_training(out,identity,model,head,opt):
    if not (out/'LATEST_TRAIN.json').exists():return 0,[]
    ptr=read_json(out/'LATEST_TRAIN.json');gen=ptr['generation']
    if Path(gen).name!=gen or not gen.startswith('ckpt-'):raise IntegrityError('Unsafe training pointer')
    src=out/gen
    if file_sha(src/'COMPLETE.json')!=ptr['manifest_sha256']:raise IntegrityError('Training manifest changed')
    for n,h in read_json(src/'COMPLETE.json')['files'].items():
        if Path(n).is_absolute() or '..' in Path(n).parts or file_sha(src/n)!=h:raise IntegrityError('Incomplete training checkpoint')
    progress=read_json(src/'progress.json')
    if progress['identity']!=identity:raise IntegrityError('Training inputs/code/environment changed; preserve this run and register another profile')
    apply_state(model,head,load_file(str(src/'trainable.safetensors')))
    opt.load_state_dict(torch.load(src/'optimizer.pt',map_location=next(model.parameters()).device,weights_only=True))
    rng=load_file(str(src/'rng.safetensors'));torch.set_rng_state(rng['cpu'].cpu())
    if torch.cuda.is_available():torch.cuda.set_rng_state_all([rng[f'cuda{i}'].cpu() for i in range(torch.cuda.device_count())])
    return int(progress['next_step']),progress.get('history',[])

def online_train(rt,rows,p,out,mode,seed,layout,cfg,initial_rank=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);torch.manual_seed(seed)
    targets=[];head=None
    if mode in ('online','lora'):
        if initial_rank is None:raise GateBlocked('Matched online/LoRA arms require the same completed fresh frozen-rank initializer')
        head=Scorer(rt.hidden).to(rt.device);head.load_state_dict(initial_rank.state_dict());head.none.requires_grad_(False)
    elif mode=='full_encoder':head=nn.Linear(rt.hidden,1).to(rt.device)
    elif mode!='finite_lora':raise ValueError(mode)
    if mode in ('lora','finite_lora'):targets=inject_lora(rt.model,p['lora_rank'],p['lora_alpha'],p['lora_last_blocks'])
    elif mode=='full_encoder':
        for x in rt.model.parameters():x.requires_grad_(True)
    # Fixed epoch/update schedule avoids online checkpoint selection on uncalibrated none probabilities.
    train=[r for r in rows if r['split']=='train'];base=[];rw=weights(train)
    for i,r in enumerate(train):
        if mode in ('online','lora'):
            for j in range(len(r['q']['options'])):base.append((i,j,float(rw[i]/len(r['q']['options']))))
        else:base.append((i,None,float(rw[i])))
    schedule=[]
    for ep in range(p['online_epochs']):
        for ix in np.random.default_rng(seed+ep).permutation(len(base)):schedule.append((ep,*base[int(ix)]))
    schedule=schedule[:p['max_training_steps_per_arm']]
    if not schedule:raise GateBlocked('Empty training schedule')
    groups=[]
    modelpars=[v for v in rt.model.parameters() if v.requires_grad]
    if modelpars:groups.append({'params':modelpars,'lr':p['encoder_lr'] if mode=='full_encoder' else p['adapter_lr']})
    if head is not None:groups.append({'params':[v for v in head.parameters() if v.requires_grad],'lr':p['online_lr']})
    opt=torch.optim.AdamW(groups,weight_decay=p['weight_decay']);trainable=[v for g in groups for v in g['params']]
    identity=digest({'runtime':rt.identity,'mode':mode,'seed':seed,'layout':layout,'protocol':p,'train_rows':digest(train),'schedule':schedule,'targets':targets})
    done=out/'TRAINING_DONE.json'
    if done.exists():
        d=read_json(done)
        if d['identity']!=identity or file_sha(out/'trained.safetensors')!=d['sha256']:raise IntegrityError('Completed training identity changed')
        apply_state(rt.model,head,load_file(str(out/'trained.safetensors')));rt.model.eval();return head,d
    step,history=restore_training(out,identity,rt.model,head,opt);started=time.monotonic();last_loss=[]
    for index in range(step,len(schedule)):
        ep,i,j,weight=schedule[index];r=train[i]
        rt.model.train(mode=='full_encoder') # keep frozen/head-only and LoRA base dropout behavior matched
        opt.zero_grad(set_to_none=True)
        if mode in ('online','lora'):
            _,_,_,seqs=segments(r,rt.tokenizer,layout,p['max_tokens'])
            h=rt.hidden_sequence(seqs[j],grad=mode=='lora')[0,-1,:].float()
            z=head.real(h[None,:]).reshape(())
            loss=F.binary_cross_entropy_with_logits(z,z.new_tensor(float(j==r['target'])))
        elif mode=='full_encoder':
            h=rt.marker_features(r,grad=True);z=head(h).squeeze(-1);loss=F.cross_entropy(z[None,:],torch.tensor([r['target']],device=rt.device))
        else:
            z=rt.finite_logits(r,grad=True);loss=F.cross_entropy(z[None,:],torch.tensor([r['target']],device=rt.device))
        loss=loss*(weight*len(base))
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite supervised loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(trainable,1.)
        if not torch.isfinite(norm):raise FloatingPointError('Nonfinite supervised gradient')
        opt.step();last_loss.append(float(loss.detach()));del loss,z
        if 'h' in locals():del h
        step=index+1
        if step%cfg.checkpoint_steps==0 or step==len(schedule):
            history.append({'through_step':step,'epoch':ep+1,'mean_weighted_loss_since_checkpoint':float(np.mean(last_loss))});last_loss=[]
            save_training(out,identity,step,rt.model,head,opt,{'history':history});rt.store.backup()
            print(f'{mode} seed {seed}: {step}/{len(schedule)} optimizer updates',flush=True)
        if time.monotonic()-started>min(cfg.worker_budget_minutes,p['max_gpu_minutes_per_arm'])*60:
            if step%cfg.checkpoint_steps:save_training(out,identity,step,rt.model,head,opt,{'history':history})
            raise BudgetStop('Training checkpoint saved. Run the notebook again to continue this exact schedule.')
    rt.model.eval();save_file(trainable_state(rt.model,head),str(out/'trained.safetensors'))
    d={'identity':identity,'sha256':file_sha(out/'trained.safetensors'),'mode':mode,'seed':seed,'targets':targets,'rank':p['lora_rank'],'alpha':p['lora_alpha'],
       'updates':len(schedule),'source_groups':len({r['group'] for r in train}),'trainable_parameters':sum(v.numel() for v in trainable),'history':history,
       'loss':'per-candidate BCE' if mode in ('online','lora') else 'categorical NLL',
       'matched_comparison':'online vs lora have same initializer, pairs, schedule, BCE and head learning rate; cross-family/finite-token training is not equal-gradient-budget',
       'final_data_used':False,'training_memory_before_cleanup':memory(),
       'time_cap_scope':'max_gpu_minutes_per_arm is a per-invocation pause cap, not cumulative total; total optimizer updates are fixed by the locked schedule.'}
    write_json(done,d)
    opt.zero_grad(set_to_none=True)
    # Completed immutable weights are sufficient for downstream reuse. Avoid retaining
    # multi-GB optimizer generations after a successful fixed training schedule.
    for old in out.glob('ckpt-*'):
        if (old/'COMPLETE.json').exists():shutil.rmtree(old)
    (out/'LATEST_TRAIN.json').unlink(missing_ok=True)
    return head,d
