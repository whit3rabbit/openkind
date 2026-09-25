"""Immutable prefix snapshots, real packed KV storage, and byte-bounded exact-prefix LRU.

TurboQuant mode uses PINNED UPSTREAM primitives (not a home-made codec): packed Prod keys
and group-quantized values. Attention KV alone is lossy; all DeltaNet and convolution
state remains exact. Snapshot restoration reconstructs floating-point KV before ordinary
attention. It is NOT a fused low-bit attention implementation or model-weight quantizer.
"""
from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import copy, hashlib, json, re, time
import torch
from phase2e_cache import cache_tensor_records, cache_digest, storage_ids
from phase2e_core import IntegrityError
from phase2f_upstream import load_upstream


def unique_bytes(value):
    """Actual tensor STORAGE bytes, deduplicated by storage identity; excludes Python heap."""
    seen=set();total=0
    for _,t in cache_tensor_records(value):
        key=(str(t.device),t.untyped_storage().data_ptr())
        if key not in seen and t.numel():
            seen.add(key);total+=t.untyped_storage().nbytes()
    return total


def storage_breakdown(value):
    sizes={'attention_kv':0,'recurrent':0,'convolution':0,'other':0};seen=set()
    for name,t in cache_tensor_records(value):
        ident=(str(t.device),t.untyped_storage().data_ptr())
        if ident in seen or not t.numel(): continue
        seen.add(ident)
        kind=('recurrent' if 'recurrent_states' in name else 'convolution' if 'conv_states' in name
              else 'attention_kv' if re.search(r'\.(keys|values)(?:\.|\[|$)',name) else 'other')
        sizes[kind]+=t.untyped_storage().nbytes()
    return sizes


def move_owned_tree(value,device):
    # A copy is intentional even when already on device: never return mutable alias storage.
    memo={id(t):t.detach().to(device=device,copy=True).contiguous() for _,t in cache_tensor_records(value)}
    return copy.deepcopy(value,memo)

@dataclass
class PackedKV:
    family: str
    shape: tuple
    original_dtype: torch.dtype
    layer: int
    n_compressed: int
    key_bits: int = 0
    value_bits: int = 0
    encoded: object = None
    tail: object = None
    quantized_dtype: str = 'float32'

@dataclass
class Snapshot:
    state: object
    spec: str
    original_bytes: int
    original_breakdown: dict
    original_digest: str | None
    placement: str
    compressed_tokens_per_layer: dict

    @property
    def nbytes(self): return unique_bytes(self.state)

    def manifest(self):
        return {'codec':self.spec,'placement':self.placement,
            'original_tensor_bytes':self.original_bytes,'stored_tensor_bytes':self.nbytes,
            'original_by_family':self.original_breakdown,'stored_by_family':storage_breakdown(self.state),
            'compressed_tokens_per_layer':self.compressed_tokens_per_layer,
            'python_metadata_included':False,'codec_tables_included':False,
            'restored_working_cache_is_floating_point':True,
            'recurrent_and_convolution_lossy':False}

class CodecBank:
    """Shared per-layer rotation/codebook buffers; charged ONCE, never omitted from net ratios."""
    def __init__(self,device='cpu',upstream_root=None,seed=42,primitives=None):
        self.device=torch.device(device);self.seed=seed;self.codecs={}
        self.quant=self.kv=None
        if primitives is not None:self.quant,self.kv=primitives
        elif upstream_root is not None:self.quant,self.kv=load_upstream(upstream_root)

    def key_codec(self,layer,dim,bits):
        if self.quant is None: raise RuntimeError('Verified upstream TurboQuant dependency unavailable; no substitute codec is used.')
        key=(layer,dim,bits)
        if key not in self.codecs:
            self.codecs[key]=self.quant.TurboQuantProd(dim=dim,bits=bits,device=self.device,
                dtype=torch.float32,seed=self.seed+layer*7).eval()
        return self.codecs[key]

    def buffers(self):
        result={f'{l}_{d}_{b}':dict(m.named_buffers()) for (l,d,b),m in self.codecs.items()}
        # Upstream keeps an eight-byte bit-packing constant per device outside nn.Module.
        result['upstream_global_buffers']=getattr(self.quant,'_POWERS_CACHE',{})
        return result

    def bytes_for(self,spec):
        if not spec.startswith('tq_'):return 0
        bits=int(re.fullmatch(r'tq_k([34])_v([24])_r(\d+)',spec)[1])
        selected={str(k):dict(v.named_buffers()) for k,v in self.codecs.items() if k[2]==bits}
        selected['upstream_global_buffers']=getattr(self.quant,'_POWERS_CACHE',{})
        return unique_bytes(selected)

    def warm_for_model(self,model,specs):
        """Codec setup reported separately; no model-data-dependent fitting or test-data tuning."""
        if not any(s.startswith('tq_') for s in specs): return
        for layer_id,layer in enumerate(model.layers):
            attn=getattr(layer,'self_attn',None)
            if attn is None: continue
            dim=getattr(attn,'head_dim',None) or getattr(model.config,'head_dim',None)
            if not dim:raise IntegrityError('Could not determine full-attention head dimension.')
            for spec in specs:
                if spec.startswith('tq_'):self.key_codec(layer_id,int(dim),int(spec.split('_')[1][1:]))

    @torch.inference_mode()
    def pack(self,root,spec='lossless',placement='gpu',verify=False):
        if placement not in ('gpu','cpu'):raise ValueError('placement must be gpu or cpu')
        destination=self.device if placement=='gpu' else torch.device('cpu')
        rows=cache_tensor_records(root);memo={};by_layer={};targeted=0
        original_bytes=unique_bytes(root);breakdown=storage_breakdown(root)
        digest=cache_digest(root) if verify else None
        match=re.fullmatch(r'tq_k([34])_v([24])_r(\d+)',spec)
        if spec not in ('lossless','kv_fp16') and match is None:raise ValueError('Unsupported snapshot codec: '+spec)
        for name,t in rows:
            if id(t) in memo: continue
            is_kv=name.endswith(('.keys','.values')) and t.numel()>0
            if not is_kv or spec=='lossless':
                memo[id(t)]=t.detach().to(destination,copy=True).contiguous();continue
            if t.ndim!=4: raise IntegrityError('Expected [batch, KV_heads, tokens, dim] attention cache.')
            layer_match=re.search(r'layers\[(\d+)\]',name)
            if not layer_match:raise IntegrityError('Unrecognized cache path: '+name)
            lid=int(layer_match[1]);targeted+=1
            if spec=='kv_fp16':
                value=t.to(dtype=torch.float16,device=destination,copy=True).contiguous()
                if not torch.isfinite(value).all():raise FloatingPointError('FP16 snapshot overflow; not a usable cache representation.')
                memo[id(t)]=PackedKV('cast',tuple(t.shape),t.dtype,lid,t.shape[-2],encoded=value)
                by_layer[str(lid)]=t.shape[-2];continue
            bits,vbits,tailn=map(int,match.groups());n=max(0,t.shape[-2]-tailn)
            by_layer[str(lid)]=n
            if not n:
                memo[id(t)]=t.detach().to(destination,copy=True).contiguous();continue
            x=t[...,:n,:].float().contiguous()
            if not torch.isfinite(x).all():raise FloatingPointError('Non-finite KV supplied to codec.')
            if name.endswith('.keys'):
                encoded=self.key_codec(lid,t.shape[-1],bits).quantize(x);family='key'
            else:
                if self.kv is None:raise RuntimeError('TurboQuant value codec unavailable.')
                encoded=self.kv.quantize_values(x,bits=vbits,group_size=32);family='value'
            # Clone the tail so it cannot retain the original full-length tensor storage.
            tail=t[...,n:,:].detach().to(destination,copy=True).contiguous()
            memo[id(t)]=PackedKV(family,tuple(t.shape),t.dtype,lid,n,bits,vbits,
                                  move_owned_tree(encoded,destination),tail)
            del x,encoded
        if spec!='lossless' and targeted==0:raise IntegrityError('No attention KV found; cannot claim a compression experiment.')
        state=copy.deepcopy(root,memo)
        snap=Snapshot(state,spec,original_bytes,breakdown,digest,placement,by_layer)
        if verify:
            if storage_ids(state)&storage_ids(root):raise IntegrityError('Stored snapshot aliases mutable original cache.')
            if cache_digest(root)!=digest:raise IntegrityError('Packing modified the original cache.')
            for name,t in rows:
                if ('conv_states' in name or 'recurrent_states' in name):
                    other=dict(cache_tensor_records(state))[name]
                    if other.dtype!=t.dtype or not torch.equal(other.to(t.device),t):raise IntegrityError('Non-KV state was changed by a KV codec.')
        return snap

    @torch.inference_mode()
    def restore(self,snap,verify=False):
        memo={}
        # Generic object walk stops at PackedKV to avoid copying its compressed children first.
        def visit(x,seen):
            if isinstance(x,PackedKV):
                if id(x) in memo:return
                enc=move_owned_tree(x.encoded,self.device)
                if x.family=='cast':value=enc.to(x.original_dtype)
                elif x.family=='key':value=self.key_codec(x.layer,x.shape[-1],x.key_bits).dequantize(enc).to(x.original_dtype)
                elif x.family=='value':value=self.kv.dequantize_values(enc,group_size=32).to(x.original_dtype)
                else:raise IntegrityError('Unknown packed cache family.')
                if x.tail is not None:value=torch.cat((value,x.tail.to(self.device)),dim=-2)
                if tuple(value.shape)!=x.shape or not torch.isfinite(value).all():raise IntegrityError('Invalid restored KV tensor.')
                memo[id(x)]=value.contiguous();return
            if isinstance(x,torch.Tensor):
                if id(x) not in memo:memo[id(x)]=x.detach().to(self.device,copy=True).contiguous()
                return
            if isinstance(x,(str,int,float,bool,bytes,type(None),torch.dtype,torch.device)):return
            if id(x) in seen:return
            seen.add(id(x))
            if isinstance(x,dict):children=x.values()
            elif isinstance(x,(list,tuple)):children=x
            elif hasattr(x,'__dict__'):children=vars(x).values()
            else:return
            for child in children:visit(child,seen)
        visit(snap.state,set());root=copy.deepcopy(snap.state,memo)
        if verify:
            if storage_ids(root)&storage_ids(snap.state):raise IntegrityError('Restored cache aliases stored snapshot.')
            if snap.spec=='lossless' and snap.original_digest is not None and cache_digest(root)!=snap.original_digest:
                raise IntegrityError('Lossless cache round trip changed state.')
        return root


def prefix_key(prefix_ids,*,tenant,model_revision,tokenizer_sha,prompt_sha,mode,backend_sha,codec,position_policy='zero-based-unpadded'):
    if not tenant or not prefix_ids:raise ValueError('Tenant namespace and nonempty exact prefix are required.')
    obj={'tenant':tenant,'model_revision':model_revision,'tokenizer_sha256':tokenizer_sha,
         'prompt_sha256':prompt_sha,'numerical_mode':mode,'backend_sha256':backend_sha,
         'snapshot_codec':codec,'position_policy':position_policy,'tokens':list(map(int,prefix_ids))}
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()

class PrefixLRU:
    """Single-worker, exact-key, TTL and byte-bounded immutable snapshot store. Not a concurrent server."""
    def __init__(self,max_bytes,ttl_seconds=600.,clock=time.monotonic):
        if max_bytes<1 or ttl_seconds<=0:raise ValueError('Invalid LRU configuration.')
        self.max_bytes=int(max_bytes);self.ttl=float(ttl_seconds);self.clock=clock
        self.items=OrderedDict();self.bytes=0;self.peak_bytes=0;self.hits=0;self.misses=0;self.evictions=0;self.expired=0;self.bypasses=0
    def _remove(self,key):
        size=self.items[key][2];del self.items[key];self.bytes-=size
    def get(self,key):
        record=self.items.get(key)
        if record is None:self.misses+=1;return None
        value,expires,_=record
        if expires<=self.clock():
            self._remove(key);self.expired+=1;self.misses+=1;return None
        self.items.move_to_end(key);self.hits+=1;return value
    def put(self,key,value):
        size=value.nbytes
        if key in self.items:self._remove(key)
        # Expired entries must not displace live entries or stay counted forever.
        now=self.clock()
        for k in list(self.items):
            if self.items[k][1]<=now:self._remove(k);self.expired+=1
        if size>self.max_bytes:self.bypasses+=1;return False
        while self.bytes+size>self.max_bytes:
            self._remove(next(iter(self.items)));self.evictions+=1
        self.items[key]=(value,now+self.ttl,size);self.bytes+=size;self.peak_bytes=max(self.peak_bytes,self.bytes)
        return True
    def clear(self):self.items.clear();self.bytes=0
    def stats(self):
        return {'hits':self.hits,'misses':self.misses,'evictions':self.evictions,'expired':self.expired,
          'oversize_bypasses':self.bypasses,'entries':len(self.items),'tensor_bytes':self.bytes,
          'peak_tensor_bytes':self.peak_bytes,'budget_bytes':self.max_bytes,
          'scope':'cache-entry tensor storage only; codec tables, model and temporary branches separately reported'}
