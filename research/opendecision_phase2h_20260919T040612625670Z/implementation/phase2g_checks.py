"""Cache lifecycle checks use the actual retained implementation, virtual time and tiny tensors."""
import copy, hashlib, json
from types import SimpleNamespace
import torch
from phase2f_storage import PrefixLRU, prefix_key, CodecBank
from phase2e_cache import cache_digest, storage_ids
from phase2e_core import IntegrityError

class Clock:
    def __init__(self):self.now=0.
    def __call__(self):return self.now
    def advance(self,n):
        if n<0:raise ValueError('Virtual arrival clock cannot go backward.')
        self.now+=n

class TinyEntry:
    def __init__(self,n):self.nbytes=n


@torch.inference_mode()
def lifecycle_selftest():
    passed=[]
    c=Clock();l=PrefixLRU(100,10,c)
    assert l.put('a',TinyEntry(60));assert l.get('a') is not None
    c.advance(10);assert l.get('a') is None and l.expired==1;passed.append('expiry_at_exact_boundary')
    assert l.put('a',TinyEntry(40));assert l.put('b',TinyEntry(40));assert l.get('a') is not None
    assert l.put('c',TinyEntry(40));assert l.get('b') is None and l.get('a') is not None;passed.append('byte_bounded_lru_eviction')
    assert not l.put('oversize',TinyEntry(101));assert l.bypasses==1 and l.bytes<=100;passed.append('oversize_bypass')
    kw=dict(tenant='a',model_revision='m',tokenizer_sha='t',prompt_sha='p',mode='f32',backend_sha='b',codec='lossless')
    k=prefix_key([1,2],**kw)
    for field in kw:
        q=dict(kw);q[field]+='changed';assert prefix_key([1,2],**q)!=k
    assert prefix_key([1,3],**kw)!=k;passed.append('tenant_and_execution_identity_separation')
    root=SimpleNamespace(layers=[SimpleNamespace(keys=torch.randn(1,2,8,4),values=torch.randn(1,2,8,4)),
             SimpleNamespace(conv_states=[torch.randn(1,4,3)],recurrent_states=[torch.randn(1,2,4,4)])])
    bank=CodecBank('cpu');d=cache_digest(root)
    for spec in ('lossless','kv_fp16'):
        snap=bank.pack(root,spec,'cpu',True);a=bank.restore(snap,True);b=bank.restore(snap,True)
        assert not storage_ids(a)&storage_ids(b);assert not storage_ids(root)&storage_ids(a)
        a.layers[0].keys.add_(100)
        assert cache_digest(root)==d;assert not torch.equal(a.layers[0].keys,b.layers[0].keys)
        assert torch.equal(root.layers[1].recurrent_states[0],b.layers[1].recurrent_states[0])
    passed+=['restored_reader_mutation_isolation','recurrent_state_remains_exact']
    # Equivalent to releasing a cancelled caller's private branch; does NOT cancel a GPU kernel.
    del a;restored=bank.restore(snap,True);assert cache_digest(root)==d;passed.append('abandoned_reader_does_not_mutate_root')
    return {'status':'passed','checks':passed,'count':len(passed),'scope':'actual single-worker cache contracts with tiny CPU tensors; no GPU cancellation, concurrency, HTTP or authorization validation'}
