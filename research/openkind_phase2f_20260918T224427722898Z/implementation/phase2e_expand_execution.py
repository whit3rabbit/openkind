"""Instrumented full/cache scoring and a conservative equal-length suffix-batch prototype.
The training prompt and archived head algebra are unchanged. No generation or cross-request cache.
"""
from __future__ import annotations
from collections import defaultdict
from contextlib import contextmanager
from time import perf_counter
import json, math
import numpy as np
import torch
from phase2d_common import pack_tokens, forward_features, require_finite, sync
from phase2e_core import IntegrityError
from phase2e_cache import (TokenPlan, make_plan, fork_cache, cache_digest, cache_manifest,
    cache_tensor_records, storage_ids, scalar_from_features, distribution)
from phase2e_runtime import validate_token_rows

BASE_STRATEGIES=('full_sequential','full_batch4','shared_prefix_chunk')
PROTOTYPE='shared_prefix_equal_length_batch4'

class ComponentTimer:
    """Intrusive synchronized wall times. NEVER added to uninstrumented timings as if independent."""
    def __init__(self,device,enabled=False):self.device=device;self.enabled=enabled;self.ms=defaultdict(float);self.calls=defaultdict(int)
    def call(self,name,fn):
        if not self.enabled:return fn()
        sync(self.device);t=perf_counter();result=fn();sync(self.device)
        self.ms[name]+=(perf_counter()-t)*1000;self.calls[name]+=1
        return result


def policy_decision(p, candidate_ids, policy):
    p=np.asarray(p,dtype=np.float64)
    if len(p)!=len(candidate_ids)+1 or not np.isfinite(p).all() or np.any(p<0) or not np.isclose(p.sum(),1,atol=1e-6):
        raise IntegrityError('Invalid policy distribution.')
    top=int(np.argmax(p[:-1]));threshold=float(policy['acceptance_threshold'])
    yes=bool(p[top]>p[-1] and p[top]>=threshold)
    return {'action':'answer' if yes else 'review','candidate_id':candidate_ids[top] if yes else None,
      'top_candidate_id':candidate_ids[top],'top_probability':float(p[top]),'none_probability':float(p[-1]),
      'threshold_margin':float(p[top]-threshold),'none_margin':float(p[top]-p[-1])}


def policy_identity(p):
    return f"prior={p['absent_prior_assumption']}:wrong={p['wrong_answer_cost']}:review={p['review_cost']}"


def probability_and_policy_views(scores,none_models,policies,candidate_ids):
    distributions={n:distribution(scores,m) for n,m in none_models.items()}
    decisions={policy_identity(p):policy_decision(distributions[p['none_head']],candidate_ids,p) for p in policies}
    return distributions,decisions


def compare_views(reference,other,reference_actions,other_actions):
    per_head={}
    for name,p in reference.items():
        q=other[name];per_head[name]={'max_abs_probability_delta':float(np.max(np.abs(p-q))),
           'argmax_changed':bool(int(np.argmax(p))!=int(np.argmax(q)))}
    policies={}
    for key,a in reference_actions.items():
        b=other_actions[key];policies[key]={
          'answer_review_changed':a['action']!=b['action'],
          'review_to_answer':a['action']=='review' and b['action']=='answer',
          'answer_to_review':a['action']=='answer' and b['action']=='review',
          'accepted_candidate_changed':a['action']==b['action']=='answer' and a['candidate_id']!=b['candidate_id'],
          'any_output_changed':(a['action'],a['candidate_id'])!=(b['action'],b['candidate_id'])}
    return {'heads':per_head,'policies':policies,
      'max_probability_delta':max(x['max_abs_probability_delta'] for x in per_head.values()),
      'any_argmax_changed':any(x['argmax_changed'] for x in per_head.values()),
      'any_policy_action_changed':any(x['answer_review_changed'] for x in policies.values()),
      'any_policy_output_changed':any(x['any_output_changed'] for x in policies.values())}


def equal_length_buckets(suffixes,order,max_batch):
    """No padding or content truncation: each batch contains identical suffix token counts."""
    if sorted(order)!=list(range(len(suffixes))):raise ValueError('Invalid candidate permutation.')
    buckets={}
    for i in order:
        if not suffixes[i]:raise ValueError('Empty candidate suffix.')
        buckets.setdefault(len(suffixes[i]),[]).append(i)
    chunks=[]
    for n in sorted(buckets):
        indices=buckets[n]
        chunks.extend(indices[i:i+max_batch] for i in range(0,len(indices),max_batch))
    return chunks


def expand_cache_batch(root,batch_size,device,verify=False):
    """Deepcopy, then duplicate beam index zero. This covers KV AND conv/recurrent state.

    Cache.batch_repeat_interleave is NOT used: the pinned linear-attention mixin does not
    implement that method. reorder_cache is implemented for both cache layer families.
    """
    if batch_size<1:raise ValueError('batch_size must be positive')
    digest=cache_digest(root) if verify else None
    branch=fork_cache(root,verify=verify)
    if batch_size>1:
        if not callable(getattr(branch,'reorder_cache',None)):raise IntegrityError('Cache lacks reorder_cache.')
        branch.reorder_cache(torch.zeros(batch_size,dtype=torch.long,device=device))
    if verify:
        if storage_ids(root)&storage_ids(branch):raise IntegrityError('Expanded branch aliases the reusable root.')
        original=dict(cache_tensor_records(root));expanded=dict(cache_tensor_records(branch))
        if set(original)!=set(expanded):raise IntegrityError('Expanded cache changed its tensor fields.')
        for name,a in original.items():
            b=expanded[name]
            is_batch_state=any(word in name for word in ('conv_states','recurrent_states')) or name.endswith(('.keys','.values'))
            if is_batch_state and a.numel():
                if a.ndim<1 or a.shape[0]!=1 or tuple(b.shape)!=(batch_size,*a.shape[1:]):
                    raise IntegrityError(f'Incorrect hybrid state expansion: {name}')
                if batch_size>1 and b.stride(0)<math.prod(b.shape[1:]):
                    raise IntegrityError(f'Overlapping/expanded-view batch rows: {name}')
                for i in range(batch_size):
                    if not torch.equal(a[0],b[i]):raise IntegrityError(f'Cache expansion value mismatch: {name}')
            elif not torch.equal(a,b):raise IntegrityError(f'Non-batch cache metadata tensor changed: {name}')
        if cache_digest(root)!=digest:raise IntegrityError('Expansion mutated root.')
    return branch


class ScoringEngine:
    def __init__(self,model,head,tokenizer,cfg,device,mode,none_models,policies):
        self.model=model;self.head=head;self.tokenizer=tokenizer;self.cfg=cfg;self.device=device;self.mode=mode
        self.none_models=none_models;self.policies=policies;self.vocab=model.get_input_embeddings().num_embeddings

    def _pack(self,seqs,prefix_length=0):
        validate_token_rows(seqs,self.vocab,self.tokenizer.pad_token_id)
        if prefix_length:
            if len({len(s) for s in seqs})!=1:raise IntegrityError('Suffix batch must be padding-free.')
            ids=torch.tensor(seqs,dtype=torch.long,device=self.device)
            n=ids.shape[1];b=ids.shape[0]
            return {'input_ids':ids,'attention_mask':torch.ones((b,prefix_length+n),dtype=torch.long,device=self.device),
              'position_ids':torch.arange(prefix_length,prefix_length+n,device=self.device)[None,:].expand(b,-1)}
        return pack_tokens(seqs,self.tokenizer.pad_token_id,self.device,explicit_positions=True)

    def _prefill_forward(self,batch,prefix_ids):
        out=self.model(**batch,use_cache=True,output_hidden_states=False,return_dict=True)
        root=out.past_key_values
        if root is None or int(root.get_seq_length())!=len(prefix_ids):raise IntegrityError('Prefill cache length mismatch.')
        root._openkind_prefix_ids=tuple(prefix_ids)
        root._openkind_model_identity=id(self.model)
        root._openkind_parameter_dtypes=tuple(str(p.dtype) for p in self.model.parameters())
        return root

    def _suffix_forward(self,batch,branch,prefix_length):
        out=self.model(**batch,past_key_values=branch,use_cache=True,output_hidden_states=False,return_dict=True)
        if int(out.past_key_values.get_seq_length())!=prefix_length+batch['input_ids'].shape[1]:
            raise IntegrityError('Suffix continuation has incorrect cache length.')
        features=out.last_hidden_state[:,-1,:].float();require_finite(features,'suffix representation')
        return features

    @torch.inference_mode()
    def score_plan(self,plan,strategy,timer=None,verify=False,order=None):
        timer=timer or ComponentTimer(self.device)
        k=len(plan.full_ids);order=list(range(k)) if order is None else list(order)
        if sorted(order)!=list(range(k)):raise ValueError('Invalid scoring order')
        scores=np.empty(k,dtype=np.float64);meta={'model_calls':0,'clone_calls':0,'suffix_batch_sizes':[]}
        if strategy in ('full_sequential','full_batch4'):
            size=1 if strategy=='full_sequential' else 4
            for start in range(0,k,size):
                ix=order[start:start+size]
                batch=timer.call('input_pack_transfer_ms',lambda:self._pack([plan.full_ids[i] for i in ix]))
                features=timer.call('full_forward_ms',lambda:forward_features(self.model,batch))
                values=timer.call('score_transfer_head_ms',lambda:scalar_from_features(self.head,features))
                scores[ix]=values;meta['model_calls']+=1
                del batch,features,values
        elif strategy in ('shared_prefix_chunk',PROTOTYPE):
            batch=timer.call('input_pack_transfer_ms',lambda:self._pack([plan.prefix_ids]))
            root=timer.call('prefix_forward_ms',lambda:self._prefill_forward(batch,plan.prefix_ids));del batch
            digest=cache_digest(root) if verify else None
            if verify:
                manifest=cache_manifest(root);meta['cache_layout']=manifest
                if type(self.model).__name__=='Qwen3_5TextModel' and not all(manifest[n] for n in ('includes_recurrent','includes_convolution','includes_kv')):
                    raise IntegrityError('Missing hybrid cache state family.')
            chunks=([[i] for i in order] if strategy=='shared_prefix_chunk' else
                    equal_length_buckets(plan.suffix_ids,order,self.cfg.suffix_batch_size))
            meta['model_calls']=1
            for ix in chunks:
                # In the uninstrumented path clone+expand execute without explicit synchronization.
                branch=timer.call('cache_clone_ms',lambda:fork_cache(root,verify=verify))
                if len(ix)>1:
                    def expand():
                        if not callable(getattr(branch,'reorder_cache',None)):raise IntegrityError('Cache lacks reorder_cache.')
                        branch.reorder_cache(torch.zeros(len(ix),dtype=torch.long,device=self.device))
                        return branch
                    branch=timer.call('cache_expand_ms',expand)
                    if verify:
                        # Check shape and row independence; tiny selftest checks exact copied values too.
                        if storage_ids(root)&storage_ids(branch):raise IntegrityError('Batched branch aliases root.')
                        for name,t in cache_tensor_records(branch):
                            if t.numel() and (name.endswith(('.keys','.values')) or 'conv_states' in name or 'recurrent_states' in name):
                                if t.shape[0]!=len(ix) or t.stride(0)<math.prod(t.shape[1:]):raise IntegrityError('Invalid expanded row storage.')
                batch=timer.call('input_pack_transfer_ms',lambda:self._pack([plan.suffix_ids[i] for i in ix],len(plan.prefix_ids)))
                features=timer.call('suffix_forward_ms',lambda:self._suffix_forward(batch,branch,len(plan.prefix_ids)))
                values=timer.call('score_transfer_head_ms',lambda:scalar_from_features(self.head,features))
                scores[ix]=values;meta['model_calls']+=1;meta['clone_calls']+=1;meta['suffix_batch_sizes'].append(len(ix))
                del features,values,batch,branch
            if verify:
                if cache_digest(root)!=digest:raise IntegrityError('Branch mutated the reusable prefix.')
                meta['cache_root_unchanged']=True
            del root
        else:raise ValueError(strategy)
        if not np.isfinite(scores).all():raise FloatingPointError('Non-finite candidate scores.')
        meta.update(prefix_tokens=len(plan.prefix_ids),full_token_sum=sum(map(len,plan.full_ids)),
          logical_input_tokens=(len(plan.prefix_ids)+sum(map(len,plan.suffix_ids))) if strategy.startswith('shared_') else sum(map(len,plan.full_ids)),
          candidate_count=k,padding_in_execution=strategy=='full_batch4',cold_prefix=True)
        return scores,meta

    @torch.inference_mode()
    def request(self,episode,strategy,profile=False,verify=False):
        timer=ComponentTimer(self.device,profile);sync(self.device) if profile else None;t=perf_counter()
        plan=timer.call('tokenization_plan_ms',lambda:make_plan(episode,self.tokenizer,self.cfg,self.mode))
        scores,meta=self.score_plan(plan,strategy,timer,verify)
        def finalize():
            distributions,actions=probability_and_policy_views(scores,self.none_models,self.policies,plan.candidate_ids)
            body={'candidate_ids':plan.candidate_ids,'distributions':{k:list(map(float,v)) for k,v in distributions.items()},
                  'policy_actions':actions}
            payload=json.dumps(body,allow_nan=False)
            return distributions,actions,len(payload.encode())
        distributions,actions,nbytes=timer.call('probability_policy_json_ms',finalize)
        if profile:sync(self.device)
        wall=(perf_counter()-t)*1000
        meta.update(json_bytes=nbytes,component_ms=dict(timer.ms),component_calls=dict(timer.calls),
          profiled_total_ms=wall if profile else None,profile_instrumented=profile)
        if profile:meta['unattributed_host_ms']=max(0.,wall-sum(timer.ms.values()))
        return {'scores':scores,'distributions':distributions,'actions':actions,'meta':meta}


@torch.inference_mode()
def tiny_expanded_cache_selftest():
    """Runs on real installed Transformers, tiny random CPU model, before the 4B load."""
    from types import SimpleNamespace
    from transformers import Qwen3_5TextConfig,Qwen3_5TextModel
    from phase2d_common import CandidateHead
    from phase2e_cache import prefill
    from phase2d_decisions import NoneModel
    config=Qwen3_5TextConfig(vocab_size=256,hidden_size=64,intermediate_size=128,num_hidden_layers=4,
      num_attention_heads=4,num_key_value_heads=2,head_dim=16,linear_key_head_dim=16,linear_value_head_dim=16,
      linear_num_key_heads=2,linear_num_value_heads=4,linear_conv_kernel_dim=4,
      layer_types=['linear_attention']*3+['full_attention'],max_position_embeddings=256,pad_token_id=0,
      rope_parameters={'rope_type':'default','rope_theta':10000.,'partial_rotary_factor':1.,'mrope_section':[1,1,2]})
    config._attn_implementation='sdpa'
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7);model=Qwen3_5TextModel(config).float().eval();head=CandidateHead(64).eval()
    for p in model.parameters():p.requires_grad_(False)
    prefix=[3,4,5,6,7,8,9];suffixes=[[11,12,13],[21,22,23],[31,32],[41,42]]
    plan=TokenPlan([prefix+s for s in suffixes],prefix,suffixes,list('abcd'),[False]*4,'tiny')
    engine=ScoringEngine(model,head,SimpleNamespace(pad_token_id=0),SimpleNamespace(suffix_batch_size=4),'cpu','tiny',{},[])
    a,_=engine.score_plan(plan,'full_sequential');b,_=engine.score_plan(plan,'shared_prefix_chunk',verify=True)
    c,_=engine.score_plan(plan,PROTOTYPE,verify=True)
    d,_=engine.score_plan(plan,PROTOTYPE,verify=True,order=[3,2,1,0])
    root=prefill(model,prefix,'cpu');branch=expand_cache_batch(root,2,'cpu',verify=True)
    for name,t in cache_tensor_records(branch):
        if t.numel() and t.ndim>1 and t.shape[0]==2:
            saved=t[1].clone();t[0].add_(1)
            if not torch.equal(t[1],saved):raise IntegrityError('Expanded branch row alias.')
    delta=max(float(np.max(np.abs(a-b))),float(np.max(np.abs(a-c))))
    if delta>1e-3 or not np.allclose(c,d,atol=1e-5,rtol=0):raise IntegrityError('Tiny Qwen suffix-batch API mismatch.')
    return {'status':'passed','max_score_delta_vs_full':delta,
      'scope':'random tiny CPU Qwen: cache API/row isolation only, not real checkpoint quality or GPU performance'}
