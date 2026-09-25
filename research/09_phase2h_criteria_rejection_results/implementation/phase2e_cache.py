"""Within-request shared token prefix. Deep-copy ALL cache state, not just attention KV.
No generation, no global cache, no rollback/cropping, no cross-question reuse.
"""
from __future__ import annotations
import copy, hashlib, json, math
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from phase2d_common import encode_candidate, pack_tokens, forward_features, require_finite, probs, sync
from phase2e_core import IntegrityError, StageSkip
from phase2e_runtime import raise_if_fatal_cuda, validate_token_rows

@dataclass
class TokenPlan:
    full_ids: list
    prefix_ids: list
    suffix_ids: list
    candidate_ids: list
    truncated: list
    identity: str


def common_prefix_length(seqs):
    if len(seqs)<2 or any(not s for s in seqs):raise ValueError('At least two nonempty candidate sequences required')
    n=min(map(len,seqs))-1  # Keep a nonempty suffix even for duplicate descriptions.
    for j in range(n):
        if any(s[j]!=seqs[0][j] for s in seqs[1:]):return j
    return n


def make_plan(ep,tokenizer,cfg,mode):
    rows=[encode_candidate(ep,c,tokenizer,cfg.max_length) for c in ep['choices']]
    full=[r['input_ids'] for r in rows];n=common_prefix_length(full)
    if n<1:raise StageSkip('No common token prefix: a shared-cache comparison is not applicable')
    prefix=full[0][:n];suffix=[s[n:] for s in full]
    if any(prefix+s!=f for s,f in zip(suffix,full)):raise IntegrityError('Cached versus uncached token mismatch')
    keys=[c['id'] for c in ep['choices']]
    if len(set(keys))!=len(keys):raise ValueError('Candidate keys must be unique')
    identity=hashlib.sha256(json.dumps({'model':cfg.model_revision,'mode':mode,'prefix':prefix,
        'positions':'zero-based unpadded','max_length':cfg.max_length,'scope':'one request'},sort_keys=True).encode()).hexdigest()
    return TokenPlan(full,prefix,suffix,keys,[r['state_truncated'] for r in rows],identity)


def walk_cache(value,path='cache',visited=None):
    """Enumerate actual cache-owned tensors and metadata, including lists/dicts of recurrent state."""
    if visited is None:visited=set()
    if isinstance(value,torch.Tensor):
        yield path,value;return
    if isinstance(value,(str,bytes,int,float,bool,type(None),torch.dtype,torch.device)):
        yield path,value;return
    if id(value) in visited:return
    visited.add(id(value))
    if isinstance(value,dict):
        for k,v in sorted(value.items(),key=lambda kv:str(kv[0])):yield from walk_cache(v,f'{path}.{k}',visited)
    elif isinstance(value,(list,tuple)):
        for i,v in enumerate(value):yield from walk_cache(v,f'{path}[{i}]',visited)
    elif hasattr(value,'__dict__'):
        for k,v in sorted(vars(value).items()):
            if k in ('prefetch_stream',):continue  # Stream identity is not model state.
            yield from walk_cache(v,path+'.'+k,visited)


def cache_tensor_records(cache):
    return [(p,t) for p,t in walk_cache(cache) if isinstance(t,torch.Tensor)]


def storage_ids(cache):
    return {(str(t.device),t.untyped_storage().data_ptr()) for _,t in cache_tensor_records(cache) if t.numel()}


def cache_digest(cache):
    """Full diagnostic hash, including every cache tensor. Never called inside benchmark timing."""
    h=hashlib.sha256()
    for name,value in walk_cache(cache):
        h.update(name.encode())
        if isinstance(value,torch.Tensor):
            h.update(str((value.dtype,list(value.shape))).encode())
            h.update(value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
        else:h.update(str(value).encode())
    return h.hexdigest()


def cache_manifest(cache):
    rows=[];seen=set();size=0
    for name,t in cache_tensor_records(cache):
        rows.append({'name':name,'shape':list(t.shape),'dtype':str(t.dtype),'bytes':t.numel()*t.element_size()})
        key=(str(t.device),t.untyped_storage().data_ptr())
        if key not in seen:
            size+=t.untyped_storage().nbytes();seen.add(key)
    if not rows:raise IntegrityError('No tensors in returned cache; not a valid cache test')
    return {'class':type(cache).__name__,'tensor_count':len(rows),'unique_storage_mib':size/1024**2,'tensors':rows,
            'includes_recurrent':any('recurrent' in r['name'] for r in rows),
            'includes_convolution':any('conv' in r['name'] for r in rows),
            'includes_kv':any(r['name'].endswith(('.keys','.values')) for r in rows)}


def fork_cache(root,verify=False):
    fork=copy.deepcopy(root)
    if verify:
        if storage_ids(root)&storage_ids(fork):raise IntegrityError('Cache branch shares mutable storage with prefix')
        if cache_digest(root)!=cache_digest(fork):raise IntegrityError('Deep-copied cache differs from root')
    return fork

@torch.inference_mode()
def prefill(model,prefix_ids,device):
    validate_token_rows([prefix_ids], model.get_input_embeddings().num_embeddings)
    ids=torch.tensor([prefix_ids],dtype=torch.long,device=device)
    pos=torch.arange(len(prefix_ids),device=device)[None,:]
    out=model(input_ids=ids,attention_mask=torch.ones_like(ids),position_ids=pos,
              use_cache=True,output_hidden_states=False,return_dict=True)
    cache=out.past_key_values
    if cache is None:raise IntegrityError('Model did not return past_key_values')
    if int(cache.get_seq_length())!=len(prefix_ids):raise IntegrityError('Unexpected cached prefix length')
    cache._openkind_prefix_ids=tuple(prefix_ids)
    cache._openkind_model_identity=id(model)
    cache._openkind_parameter_dtypes=tuple(str(p.dtype) for p in model.parameters())
    return cache

@torch.inference_mode()
def suffix_feature(model,cache,suffix_ids,prefix_length,device,tokenwise=False):
    if not suffix_ids:raise ValueError('Empty suffix is unsupported')
    validate_token_rows([suffix_ids], model.get_input_embeddings().num_embeddings)
    chunks=[[i] for i in suffix_ids] if tokenwise else [suffix_ids]
    position=prefix_length;feature=None
    for chunk in chunks:
        ids=torch.tensor([chunk],dtype=torch.long,device=device)
        mask=torch.ones((1,position+len(chunk)),dtype=torch.long,device=device)
        pos=torch.arange(position,position+len(chunk),device=device)[None,:]
        out=model(input_ids=ids,attention_mask=mask,position_ids=pos,past_key_values=cache,
                  use_cache=True,output_hidden_states=False,return_dict=True)
        cache=out.past_key_values;position+=len(chunk)
        if int(cache.get_seq_length())!=position:raise IntegrityError('Suffix cache length mismatch')
        feature=out.last_hidden_state[:,-1,:].float()
    require_finite(feature,'cached suffix representation')
    return feature

@torch.inference_mode()
def scalar_from_features(head,features):
    scores=head.scorer(features.float().cpu()).squeeze(-1)
    require_finite(scores,'candidate logits')
    return scores.numpy().astype(np.float64)

@torch.inference_mode()
def full_scores(model,head,plan,pad_id,device,batch_size=1):
    validate_token_rows(plan.full_ids, model.get_input_embeddings().num_embeddings, pad_id)
    result=[]
    for start in range(0,len(plan.full_ids),batch_size):
        batch=pack_tokens(plan.full_ids[start:start+batch_size],pad_id,device,explicit_positions=True)
        result.extend(scalar_from_features(head,forward_features(model,batch)).tolist())
    return np.asarray(result)

@torch.inference_mode()
def shared_scores(model,head,plan,device,root=None,order=None,verify=False,tokenwise=False):
    if root is None:root=prefill(model,plan.prefix_ids,device)
    if (getattr(root,'_openkind_prefix_ids',None)!=tuple(plan.prefix_ids)
        or getattr(root,'_openkind_model_identity',None)!=id(model)
        or getattr(root,'_openkind_parameter_dtypes',None)!=tuple(str(p.dtype) for p in model.parameters())):
        raise IntegrityError('Cache root belongs to a different prefix, model, or parameter-dtype policy')
    original=cache_digest(root) if verify else None
    order=list(range(len(plan.suffix_ids))) if order is None else list(order)
    if sorted(order)!=list(range(len(plan.suffix_ids))):raise ValueError('Invalid candidate permutation')
    values=np.empty(len(order));clone_ms=[]
    for i in order:
        if verify:sync(device)
        t=perf_counter();branch=fork_cache(root,verify=verify)
        if verify:sync(device)
        clone_ms.append((perf_counter()-t)*1000 if verify else None)
        feature=suffix_feature(model,branch,plan.suffix_ids[i],len(plan.prefix_ids),device,tokenwise)
        values[i]=scalar_from_features(head,feature)[0]
        del feature,branch
    if verify and cache_digest(root)!=original:raise IntegrityError('A candidate mutated the reusable prefix cache')
    return values,{'clone_ms':clone_ms,'root_unchanged':True if verify else None}


def distribution(scores,none_model):
    logits=none_model.logits([np.asarray(scores)])[0]
    return probs(np.asarray(logits)[None,:])[0]


def parity_episode(model,head,ep,tokenizer,none_model,cfg,mode,device,tokenwise_probe=False):
    plan=make_plan(ep,tokenizer,cfg,mode)
    full=full_scores(model,head,plan,tokenizer.pad_token_id,device,1)
    full4=full_scores(model,head,plan,tokenizer.pad_token_id,device,4)
    root=prefill(model,plan.prefix_ids,device);manifest=cache_manifest(root)
    if type(model).__name__=='Qwen3_5TextModel' and not (manifest['includes_recurrent'] and manifest['includes_convolution'] and manifest['includes_kv']):
        raise IntegrityError('Hybrid cache audit did not find all expected state families')
    a,meta=shared_scores(model,head,plan,device,root=root,verify=True)
    b,_=shared_scores(model,head,plan,device,root=root,order=list(reversed(range(len(plan.full_ids)))),verify=True)
    again,_=shared_scores(model,head,plan,device,root=root,verify=True)
    pf,pc,p4=distribution(full,none_model),distribution(a,none_model),distribution(full4,none_model)
    order_delta=float(np.abs(pc-distribution(b,none_model)).max())
    repeat_delta=float(np.abs(pc-distribution(again,none_model)).max())
    delta=float(np.abs(pc-pf).max());changed=bool(pc.argmax()!=pf.argmax())
    result={'episode_id':ep['id'],'mode':mode,'sampler':ep.get('sampler'),'true_intent_omitted':ep.get('true_intent_omitted'),
       'K':len(full),'prefix_tokens':len(plan.prefix_ids),'suffix_tokens':list(map(len,plan.suffix_ids)),
       'input_tokens_exact':True,'prefix_identity':plan.identity,'state_truncated':any(plan.truncated),
       'max_score_delta_cached_vs_full':float(np.abs(a-full).max()),'max_probability_delta_cached_vs_full':delta,
       'choice_changed_cached_vs_full':changed,'max_probability_delta_batch4_vs_full':float(np.abs(p4-pf).max()),
       'branch_order_probability_delta':order_delta,'repeated_branch_probability_delta':repeat_delta,
       'cache_root_unchanged':True,'cache':manifest,'clone_ms':meta['clone_ms'],
       'within_sampled_tolerance':delta<=cfg.cache_probability_tolerance and not changed and order_delta<=cfg.order_probability_tolerance and repeat_delta<=cfg.order_probability_tolerance,
       'full_scores':full,'shared_scores':a,'full_probabilities':pf,'shared_probabilities':pc,
       'reference_correct':bool(pf.argmax()==ep['target_index']),'cached_correct':bool(pc.argmax()==ep['target_index'])}
    if tokenwise_probe:
        try:
            tok,_=shared_scores(model,head,plan,device,root=root,verify=True,tokenwise=True)
            pt=distribution(tok,none_model)
            result['tokenwise_input_replay']={'status':'completed','probability_delta_vs_full':float(np.abs(pt-pf).max()),
                'probability_delta_vs_chunk':float(np.abs(pt-pc).max()),'choice_changed_vs_full':bool(pt.argmax()!=pf.argmax()),
                'note':'replays existing input tokens; does not sample or generate output text; not an automatic fallback'}
        except IntegrityError:
            raise
        except Exception as e:
            raise_if_fatal_cuda(e)
            from phase2d_common import sanitize_error
            result['tokenwise_input_replay']={'status':'failed','error':sanitize_error(e)}
    del root
    return result

@torch.inference_mode()
def request(model,head,ep,tokenizer,none_model,cfg,mode,device,strategy):
    """All tokenization/transfer/cache cloning/scoring/serialization occur INSIDE this call."""
    plan=make_plan(ep,tokenizer,cfg,mode)
    if strategy=='full_sequential':s=full_scores(model,head,plan,tokenizer.pad_token_id,device,1);calls=len(s)
    elif strategy=='full_batch4':s=full_scores(model,head,plan,tokenizer.pad_token_id,device,4);calls=math.ceil(len(s)/4)
    elif strategy=='shared_prefix_chunk':s,_=shared_scores(model,head,plan,device,verify=False);calls=1+len(s)
    else:raise ValueError(strategy)
    p=distribution(s,none_model)
    response={'choice':plan.candidate_ids[int(p.argmax())] if int(p.argmax())<len(s) else '__none_of_these__',
       'probabilities':dict(zip(plan.candidate_ids+['__none_of_these__'],map(float,p)))}
    payload=json.dumps(response,allow_nan=False)
    full_token_count=sum(map(len,plan.full_ids))
    evaluated_tokens=(len(plan.prefix_ids)+sum(map(len,plan.suffix_ids))) if strategy=='shared_prefix_chunk' else full_token_count
    return p,{'json_bytes':len(payload.encode()),'model_calls':calls,'prefix_tokens':len(plan.prefix_ids),
              'full_prompt_token_sum':full_token_count,'logical_tokens_evaluated':evaluated_tokens,
              'padding_compute_not_in_token_count':strategy=='full_batch4'}


def time_request(fn,device,repeats,warmups):
    for _ in range(warmups):fn()
    sync(device);base=torch.cuda.memory_allocated(device) if torch.device(device).type=='cuda' else 0
    if torch.device(device).type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    times=[];p=None;meta=None
    for _ in range(repeats):
        sync(device);t=perf_counter();p,meta=fn();sync(device);times.append(1000*(perf_counter()-t))
    return p,dict(meta,p50_ms=float(np.median(times)),p95_ms=float(np.quantile(times,.95)),times_ms=times,
      peak_extra_mib=(torch.cuda.max_memory_allocated(device)-base)/1024**2 if torch.device(device).type=='cuda' else None,
      resident_mib=base/1024**2 if torch.device(device).type=='cuda' else None,
      warmups=warmups,repeats=repeats,cold_prefix=True,
      scope='complete in-process request; prefill included on every cached call; deepcopy overhead included; no HTTP; diagnostic full-cache hashes excluded')
