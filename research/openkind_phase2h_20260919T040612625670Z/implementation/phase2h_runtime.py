"""One model per worker. Immutable execution identities, finite checks and persistent CPU features."""
import copy,json,hashlib,sqlite3,os,time,math,platform,gc,inspect
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
from phase2d_common import json_write,json_ready,content_hash,file_sha,pack_tokens,forward_features,require_finite
from phase2e_expand_worker import memory_snapshot,load_preflight,numerical_controls
from phase2e_runtime import validate_token_rows,indexing_selftest
from phase2e_cache import common_prefix_length,cache_digest,fork_cache
from phase2e_expand_execution import expand_cache_batch,equal_length_buckets
from phase2h_learning import AffineScorer

@contextmanager
def controls(mode):
    with numerical_controls('fp32_strict_math'):
        if mode=='fp32_tf32_allowed':torch.backends.cuda.matmul.allow_tf32=True
        elif mode!='fp32_strict_math':raise ValueError('Unregistered numerical mode')
        yield

def encode(ep,choice,tokenizer,cfg,layout='instruction_first'):
    """Retain the original segmented encoder, but reject rather than silently truncate."""
    from phase2d_common import DYNAMIC_PREFIX,DYNAMIC_STATE_PREFIX,DYNAMIC_CAND_PREFIX,DYNAMIC_TAIL
    enc=lambda s:tokenizer.encode(s,add_special_tokens=False)
    if layout=='instruction_first':ids=enc(DYNAMIC_PREFIX)+enc(ep['instruction'])+enc(DYNAMIC_STATE_PREFIX)+enc(ep['state'])+enc(DYNAMIC_CAND_PREFIX)+enc(choice['description'])+enc(DYNAMIC_TAIL)
    elif layout=='state_first':ids=enc('Decision state:\n')+enc(ep['state'])+enc('\nQuestion instruction:\n')+enc(ep['instruction'])+enc(DYNAMIC_CAND_PREFIX)+enc(choice['description'])+enc(DYNAMIC_TAIL)
    else:raise ValueError(layout)
    limit=ep.get('max_length',cfg.max_length)
    if not ids or len(ids)>limit:raise ValueError(f'Input exceeds registered limit ({len(ids)} > {limit}); no state or criterion is silently truncated. id={ep["id"]}')
    return ids

def finite_prompt(ep,tokenizer,cfg):
    codes=list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')[:len(ep['choices'])+1]
    if len(codes)!=len(ep['choices'])+1:raise ValueError('Finite-code baseline supports at most 25 candidates')
    tid=[tokenizer.encode(' '+c,add_special_tokens=False) for c in codes]
    if any(len(x)!=1 for x in tid) or len({x[0] for x in tid})!=len(tid):raise ValueError('Answer codes are not distinct single tokens')
    text='Choose the matching option, or none when no option matches.\nInstruction:\n'+ep['instruction']+'\nState:\n'+ep['state']+'\nOptions:\n'+'\n'.join(c+': '+o['description'] for c,o in zip(codes,ep['choices']))+'\n'+codes[-1]+': None of these options matches.\nReturn only its code.\nAnswer:'
    ids=tokenizer.encode(text,add_special_tokens=False)
    if len(ids)>4096:raise ValueError('Finite-code prompt exceeds 4096-token cap; no truncation')
    return ids,[x[0] for x in tid]

class FeatureStore:
    def __init__(self,path,identity):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True);self.identity=content_hash(identity);self.db=sqlite3.connect(str(path))
        self.db.execute('PRAGMA journal_mode=WAL');self.db.execute('CREATE TABLE IF NOT EXISTS features (key TEXT PRIMARY KEY, n INTEGER, data BLOB, sha TEXT)');self.pending=0
    def key(self,ids):return content_hash([self.identity,ids])
    def get(self,ids):
        r=self.db.execute('SELECT n,data,sha FROM features WHERE key=?',(self.key(ids),)).fetchone()
        if r is None:return None
        n,b,s=r
        if hashlib.sha256(b).hexdigest()!=s:raise ValueError('Feature cache checksum failed')
        x=np.frombuffer(b,dtype='<f4').copy()
        if len(x)!=n or not np.isfinite(x).all():raise ValueError('Invalid feature cache')
        return x
    def put(self,ids,x):
        x=np.asarray(x,dtype='<f4').ravel()
        if not np.isfinite(x).all():raise FloatingPointError('Cannot cache nonfinite features')
        b=x.tobytes();self.db.execute('INSERT OR REPLACE INTO features VALUES (?,?,?,?)',(self.key(ids),len(x),b,hashlib.sha256(b).hexdigest()));self.pending+=1
        if self.pending>=16:self.db.commit();self.pending=0
    def close(self):self.db.commit();self.db.close()

class Runtime:
    def __init__(self,spec,mode,source,out,cfg,report):
        if not torch.cuda.is_available():raise RuntimeError('GPU required; no large-model CPU fallback')
        import importlib.metadata as im
        if im.version('transformers')!='5.17.0':raise RuntimeError('Install pinned Transformers 5.17.0')
        if not torch.__version__.startswith('2.11.'):
            raise RuntimeError('This reference notebook requires PyTorch 2.11.x. Do not silently replace CUDA/Torch; use a separately versioned environment for another release.')
        report['indexing_selftest']=indexing_selftest()
        from phase2e_expand_execution import tiny_expanded_cache_selftest
        report['tiny_cache_api_selftest']=tiny_expanded_cache_selftest()
        self.cfg=cfg;self.device=torch.device('cuda:0');self.mode=mode;self.spec=spec;self.out=Path(out);self.source=Path(source)
        before=memory_snapshot('fresh_worker');report['memory']=[before]
        report['preflight']=load_preflight(before,spec['parameter_estimate'],4,2.,64.)
        from transformers import AutoTokenizer,AutoModelForCausalLM
        tokpath=str(self.source/'export/tokenizer') if spec['name']=='qwen4b' else spec['model_id']
        self.tokenizer=AutoTokenizer.from_pretrained(tokpath,revision=spec['revision'] if spec['name']!='qwen4b' else None,token=os.environ.get('HF_TOKEN'),trust_remote_code=False)
        model,info=AutoModelForCausalLM.from_pretrained(spec['model_id'],revision=spec['revision'],token=os.environ.get('HF_TOKEN'),dtype=torch.float32,device_map={'':0},attn_implementation='sdpa',trust_remote_code=False,output_loading_info=True)
        info=json_ready(info);json_write(self.out/'loading_info.json',info)
        if any(info.get(k) for k in ('missing_keys','mismatched_keys','error_msgs')):raise RuntimeError('Incomplete checkpoint load')
        if type(model).__name__!='Qwen3_5ForCausalLM' or type(model.model).__name__!='Qwen3_5TextModel':raise RuntimeError('Unexpected model graph')
        if any('visual' in n for n,_ in model.named_modules()):raise RuntimeError('Vision graph loaded unexpectedly')
        self.tied_output=(model.get_output_embeddings().weight.data_ptr()==model.get_input_embeddings().weight.data_ptr())
        self.model=model.model;del model;self.model.eval()
        for p in self.model.parameters():p.requires_grad_(False)
        self.vocab=self.model.get_input_embeddings().num_embeddings;self.hidden=self.model.config.hidden_size
        params=sum(p.numel() for p in self.model.parameters())
        if spec['name']=='qwen4b' and params!=4205751296:raise RuntimeError('4B parameter count changed')
        if any(p.dtype!=torch.float32 or p.device!=self.device for p in self.model.parameters()):raise RuntimeError('Wrong storage dtype or unexpected offload')
        if self.tokenizer.pad_token_id is None:self.tokenizer.pad_token=self.tokenizer.eos_token
        # Cache identity includes actual packages/device/tokenizer/implementation.
        from phase2e_precision import backend_flags
        packages={}
        for name in ('transformers','torch','numpy','scipy','scikit-learn','safetensors','flash-linear-attention','fla-core','causal-conv1d','flash-attn'):
            try:packages[name]=im.version(name)
            except im.PackageNotFoundError:packages[name]=None
        tokenizer_identity={p.name:file_sha(p) for p in (self.source/'export/tokenizer').glob('*') if p.is_file()} if spec['name']=='qwen4b' else {'revision':spec['revision']}
        self.identity={'spec':spec,'mode':mode,'transformers':im.version('transformers'),'torch':torch.__version__,
            'source_sha256':file_sha(inspect.getfile(type(self.model))),'no_padding_feature_extraction':True,'max_length':cfg.max_length,
            'runtime_revision':'2h.1.2','packages':packages,'gpu':torch.cuda.get_device_name(),
            'capability':list(torch.cuda.get_device_capability()),'cuda':torch.version.cuda,
            'flags':backend_flags(),'tokenizer':tokenizer_identity,
            'runtime_source':file_sha(__file__),'common_source':file_sha(inspect.getfile(forward_features))}
        report['model']={'id':spec['model_id'],'revision':spec['revision'],'parameters':params,'hidden_size':self.hidden,'storage_dtype':'float32','tied_embedding':self.tied_output,'generation':False,'cpu_offload':False}
        report['environment']={'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'transformers':im.version('transformers'),'python':platform.python_version()};report['memory'].append(memory_snapshot('model_loaded'))
        self.store=FeatureStore(self.out/'features.sqlite',self.identity);self.calls=0
        sm=self.feature(self.tokenizer.encode('A short model integrity check.',add_special_tokens=False),memo=False)
        if len(sm)!=self.hidden:raise RuntimeError('Hidden size mismatch')
        report['model_forward_smoke']={'finite':bool(np.isfinite(sm).all()),'shape':list(sm.shape)}
    def pack(self,seqs,prefix=0):
        validate_token_rows(seqs,self.vocab,self.tokenizer.pad_token_id)
        if prefix:
            if len(set(map(len,seqs)))!=1:raise ValueError('Suffix batch padding is forbidden')
            b=len(seqs);n=len(seqs[0]);return {'input_ids':torch.tensor(seqs,dtype=torch.long,device=self.device),'attention_mask':torch.ones((b,prefix+n),dtype=torch.long,device=self.device),'position_ids':torch.arange(prefix,prefix+n,device=self.device)[None,:].expand(b,-1)}
        return pack_tokens(seqs,self.tokenizer.pad_token_id,self.device,explicit_positions=True)
    @torch.no_grad()
    def feature(self,ids,memo=True):
        x=self.store.get(ids) if memo else None
        if x is not None:
            if len(x)!=self.hidden:raise ValueError('Feature cache hidden width mismatch')
            return x
        x=forward_features(self.model,self.pack([ids]))[0].float().cpu().numpy();self.calls+=1
        if memo:self.store.put(ids,x)
        return x
    def episodes(self,eps,layout='instruction_first'):
        result=[]
        for j,e in enumerate(eps):
            result.append(np.stack([self.feature(encode(e,c,self.tokenizer,self.cfg,layout)) for c in e['choices']]))
            if (j+1)%self.cfg.checkpoint_every==0:print(f'  features {j+1}/{len(eps)}; actual forwards {self.calls}',flush=True)
        self.store.db.commit();return result
    def fixed_features(self,rows):return np.stack([self.feature(self.tokenizer.encode(r['text'],add_special_tokens=False)) for r in rows])
    @torch.no_grad()
    def finite_scores(self,e,memo=True):
        if not self.tied_output:raise RuntimeError('Finite-token baseline requires verified tied output weights')
        ids,codes=finite_prompt(e,self.tokenizer,self.cfg);h=torch.tensor(self.feature(ids,memo),device=self.device)
        w=self.model.get_input_embeddings().weight[torch.tensor(codes,device=self.device)]
        scores=(w@h).float().cpu().numpy();require_finite(torch.as_tensor(scores),'finite-token scores');return scores
    @torch.no_grad()
    def run_sequences(self,seqs,strategy='full_sequential',verify=False):
        """Never use the offline feature cache in request benchmarks/parity tests."""
        if strategy=='full_sequential':return np.stack([self.feature(s,False) for s in seqs])
        if strategy=='full_batch4':
            return torch.cat([forward_features(self.model,self.pack(seqs[i:i+4])) for i in range(0,len(seqs),4)]).float().cpu().numpy()
        if strategy not in ('shared_lossless','shared_kv_fp16'):raise ValueError(strategy)
        n=common_prefix_length(seqs);n=min(n,min(map(len,seqs))-1)
        if n<1:return self.run_sequences(seqs,'full_sequential',verify)
        suffixes=[s[n:] for s in seqs];root=self.model(**self.pack([seqs[0][:n]]),use_cache=True,return_dict=True).past_key_values
        before=cache_digest(root) if verify else None
        if strategy=='shared_kv_fp16':
            root=fork_cache(root)
            for layer in root.layers:
                for key in ('keys','values'):
                    value=getattr(layer,key,None)
                    if isinstance(value,torch.Tensor):setattr(layer,key,value.half())
        stored_digest=cache_digest(root) if verify else None
        outputs=[None]*len(seqs)
        for ix in equal_length_buckets(suffixes,list(range(len(suffixes))),4):
            branch=expand_cache_batch(root,len(ix),self.device,verify=False)
            if strategy=='shared_kv_fp16':
                for layer in branch.layers:
                    for key in ('keys','values'):
                        v=getattr(layer,key,None)
                        if isinstance(v,torch.Tensor):setattr(layer,key,v.float())
            o=self.model(**self.pack([suffixes[i] for i in ix],n),past_key_values=branch,use_cache=True,return_dict=True)
            h=o.last_hidden_state[:,-1,:].float().cpu().numpy()
            for i,x in zip(ix,h):outputs[i]=x
            del o,branch
        if verify and cache_digest(root)!=stored_digest:raise RuntimeError('Prefix root mutated')
        result=np.stack(outputs)
        if not np.isfinite(result).all():raise FloatingPointError('Nonfinite branch output')
        return result
    def close(self):self.store.close()
