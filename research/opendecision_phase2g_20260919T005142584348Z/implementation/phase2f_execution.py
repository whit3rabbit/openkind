"""Decision requests using the validated Phase 2E graph plus storage-tier and warm-cache ablations."""
from __future__ import annotations
import hashlib, json, math
from time import perf_counter
import numpy as np
import torch
from phase2d_common import sync, file_sha, content_hash, require_finite
from phase2e_core import IntegrityError
from phase2e_cache import make_plan, fork_cache, cache_digest, storage_ids, scalar_from_features
from phase2e_expand_execution import ScoringEngine, ComponentTimer, equal_length_buckets, probability_and_policy_views
from phase2f_storage import CodecBank, PrefixLRU, prefix_key

class CacheEngine:
    def __init__(self,native,bank,tokenizer_sha,backend_sha):
        self.native=native;self.bank=bank;self.device=native.device;self.mode=native.mode;self.cfg=native.cfg
        self.tokenizer=native.tokenizer;self.none_models=native.none_models;self.policies=native.policies
        self.tokenizer_sha=tokenizer_sha;self.backend_sha=backend_sha
        from phase2d_common import DYNAMIC_PREFIX,DYNAMIC_STATE_PREFIX,DYNAMIC_CAND_PREFIX,DYNAMIC_TAIL
        self.prompt_sha=content_hash([DYNAMIC_PREFIX,DYNAMIC_STATE_PREFIX,DYNAMIC_CAND_PREFIX,DYNAMIC_TAIL,self.cfg.max_length])

    def key(self,plan,spec,tenant='research-default'):
        return prefix_key(plan.prefix_ids,tenant=tenant,model_revision=self.cfg.model_revision,
          tokenizer_sha=self.tokenizer_sha,prompt_sha=self.prompt_sha,mode=self.mode,
          backend_sha=self.backend_sha,codec=f'{spec}:seed={self.cfg.quantization_seed}:snapshot-v1')

    @torch.inference_mode()
    def prefill(self,plan,timer=None):
        timer=timer or ComponentTimer(self.device)
        batch=timer.call('input_pack_transfer_ms',lambda:self.native._pack([plan.prefix_ids]))
        return timer.call('prefix_forward_ms',lambda:self.native._prefill_forward(batch,plan.prefix_ids))

    @torch.inference_mode()
    def suffixes(self,plan,root,batch_size=4,timer=None,verify=False,order=None):
        timer=timer or ComponentTimer(self.device)
        if tuple(plan.prefix_ids)!=getattr(root,'_opendecision_prefix_ids',None):raise IntegrityError('Snapshot prefix identity mismatch.')
        if id(self.native.model)!=getattr(root,'_opendecision_model_identity',None):raise IntegrityError('Snapshot belongs to another model instance.')
        order=list(range(len(plan.full_ids))) if order is None else list(order)
        groups=equal_length_buckets(plan.suffix_ids,order,batch_size)
        scores=np.empty(len(plan.full_ids),dtype=np.float64)
        digest=cache_digest(root) if verify else None
        for ix in groups:
            branch=timer.call('cache_clone_ms',lambda:fork_cache(root,verify=verify))
            if len(ix)>1:
                def expand():
                    branch.reorder_cache(torch.zeros(len(ix),dtype=torch.long,device=self.device));return branch
                branch=timer.call('cache_expand_ms',expand)
                if verify and storage_ids(root)&storage_ids(branch):raise IntegrityError('Expanded suffix cache aliases root.')
            batch=timer.call('input_pack_transfer_ms',lambda:self.native._pack([plan.suffix_ids[i] for i in ix],len(plan.prefix_ids)))
            features=timer.call('suffix_forward_ms',lambda:self.native._suffix_forward(batch,branch,len(plan.prefix_ids)))
            scores[ix]=timer.call('score_transfer_head_ms',lambda:scalar_from_features(self.native.head,features))
            del branch,batch,features
        if not np.isfinite(scores).all():raise FloatingPointError('Nonfinite decision scores.')
        if verify and cache_digest(root)!=digest:raise IntegrityError('Suffix branch mutated root.')
        return scores,{'suffix_batches':len(groups),'suffix_batch_sizes':[len(ix) for ix in groups],
                       'padding_in_suffixes':False,'cache_root_unchanged':True if verify else None}

    def views(self,scores,plan):
        d,a=probability_and_policy_views(scores,self.none_models,self.policies,plan.candidate_ids)
        body={'candidate_ids':plan.candidate_ids,'distributions':{k:p.tolist() for k,p in d.items()},'policy_actions':a}
        return d,a,len(json.dumps(body,allow_nan=False,separators=(',',':')).encode())

    @torch.inference_mode()
    def cached_plan(self,plan,spec='lossless',placement='gpu',cache=None,batch_size=4,
                    timer=None,verify=False,tenant='research-default',order=None):
        timer=timer or ComponentTimer(self.device);key=self.key(plan,spec,tenant)
        snap=timer.call('prefix_lookup_ms',lambda:cache.get(key)) if cache is not None else None
        hit=snap is not None;inserted=False
        if snap is None:
            root=self.prefill(plan,timer)
            snap=timer.call('snapshot_pack_ms',lambda:self.bank.pack(root,spec,placement,verify))
            del root
            if cache is not None:inserted=timer.call('cache_insert_ms',lambda:cache.put(key,snap))
        before=cache_digest(snap.state) if verify else None
        root=timer.call('snapshot_restore_ms',lambda:self.bank.restore(snap,verify))
        scores,meta=self.suffixes(plan,root,batch_size,timer,verify,order)
        del root
        if verify and cache_digest(snap.state)!=before:raise IntegrityError('Inference changed a stored snapshot.')
        meta.update(cache_hit=hit,cache_inserted=inserted,prefix_tokens=len(plan.prefix_ids),
          candidate_count=len(plan.full_ids),snapshot=snap.manifest(),codec_shared_bytes=self.bank.bytes_for(spec),
          model_calls=meta['suffix_batches']+(0 if hit else 1),
          logical_input_tokens=sum(map(len,plan.suffix_ids))+(0 if hit else len(plan.prefix_ids)),
          cache_key_sha256=key,cache_stats=cache.stats() if cache is not None else None)
        return scores,meta

    @torch.inference_mode()
    def request(self,ep,spec='lossless',placement='gpu',cache=None,batch_size=4,
                profile=False,verify=False,tenant='research-default'):
        timer=ComponentTimer(self.device,profile)
        if profile:sync(self.device)
        t=perf_counter()
        plan=timer.call('tokenization_plan_ms',lambda:make_plan(ep,self.tokenizer,self.cfg,self.mode))
        scores,meta=self.cached_plan(plan,spec,placement,cache,batch_size,timer,verify,tenant)
        d,a,n=timer.call('probability_policy_json_ms',lambda:self.views(scores,plan))
        if profile:sync(self.device)
        wall=(perf_counter()-t)*1000
        meta.update(json_bytes=n,component_ms=dict(timer.ms),component_calls=dict(timer.calls),
                    profiled_total_ms=wall if profile else None,profile_instrumented=profile,
                    cold_prefix=not meta['cache_hit'],result_memoization=False)
        return {'scores':scores,'distributions':d,'actions':a,'meta':meta}

    @torch.inference_mode()
    def equal8(self,ep,verify=False):
        plan=make_plan(ep,self.tokenizer,self.cfg,self.mode);root=self.prefill(plan)
        scores,meta=self.suffixes(plan,root,8,verify=verify)
        d,a,n=self.views(scores,plan)
        meta.update(prefix_tokens=len(plan.prefix_ids),model_calls=1+meta['suffix_batches'],json_bytes=n,cold_prefix=True)
        return {'scores':scores,'distributions':d,'actions':a,'meta':meta}


def packed_storage_report(snapshot,bank):
    m=snapshot.manifest();overhead=bank.bytes_for(snapshot.spec)
    m['shared_codec_tables_bytes']=overhead
    m['one_entry_total_tensor_bytes']=m['stored_tensor_bytes']+overhead
    m['one_entry_compression_ratio']=m['original_tensor_bytes']/max(1,m['one_entry_total_tensor_bytes'])
    m['amortized_ratios']={str(n):n*m['original_tensor_bytes']/max(1,n*m['stored_tensor_bytes']+overhead) for n in (1,8,32)}
    m['warning']='Stored prefix only. Model weights unchanged; restoration materializes full floating-point cache and transient memory is measured separately.'
    return m
