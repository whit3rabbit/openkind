"""Pinned external TurboQuant primitives. No pip setup hooks, vLLM install, or monkey patching.
Files are downloaded without HF credentials. Git blob hashes are verified before import.
The upstream dependency is GPL-3.0; its license is downloaded alongside its source.
This adapter does NOT reproduce its vLLM backend or claim fused quantized attention.
"""
import hashlib, importlib, json, os, sys, time
from pathlib import Path
from urllib.request import Request, urlopen
from phase2f_config import TURBOQUANT_COMMIT

BLOBS = {
 'LICENSE':'f288702d2fa16d3cdf0035b15a9fcbc552cd88e7',
 'turboquant/__init__.py':'0b82c963f1bc967ced76e23d010e388b5b7b15db',
 'turboquant/codebook.py':'5527b1cc9cb53527997cedf96bc116ac7f2ef1e9',
 'turboquant/rotation.py':'fe570b144d9d0094063a90531a03a463121479b5',
 'turboquant/quantizer.py':'574486d525fd175c782b7f2ef4462804d66b64c6',
 'turboquant/kv_cache.py':'ab00d9b52074f3935aa4667033568dfd7d4eba7e',
 'turboquant/capture.py':'d10263918f3862bd3360dc5550bec9a9c151eb18',
 'turboquant/store.py':'970d4151f09e17886a62bd7d294c0cedc2721cb3',
 'turboquant/score.py':'4df41207fc484ed04fb1ce9dfb02158ee16e773c'}

def git_blob_sha(data):
    return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()

def verify_upstream(root):
    root=Path(root); hashes={}
    for name,expected in BLOBS.items():
        path=root/name
        if not path.is_file() or path.is_symlink(): raise ValueError('Missing or symlinked TurboQuant source: '+name)
        data=path.read_bytes()
        if git_blob_sha(data)!=expected: raise ValueError('TurboQuant source blob mismatch: '+name)
        hashes[name]=hashlib.sha256(data).hexdigest()
    return hashes

def download_upstream(cache_dir):
    root=Path(cache_dir)/('turboquant_'+TURBOQUANT_COMMIT)
    root.mkdir(parents=True,exist_ok=True)
    for name,expected in BLOBS.items():
        path=root/name
        if path.is_file() and not path.is_symlink() and git_blob_sha(path.read_bytes())==expected: continue
        url=f'https://raw.githubusercontent.com/0xSero/turboquant/{TURBOQUANT_COMMIT}/{name}'
        last=None
        for attempt in range(3):
            try:
                request=Request(url,headers={'User-Agent':'OpenKind-Phase2F/1.0'})
                with urlopen(request,timeout=45) as response: data=response.read(1_000_001)
                if len(data)>1_000_000 or git_blob_sha(data)!=expected: raise ValueError('Wrong/oversized pinned upstream bytes: '+name)
                path.parent.mkdir(parents=True,exist_ok=True)
                tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_bytes(data);tmp.replace(path)
                break
            except Exception as e:
                last=e
                if attempt==2: raise RuntimeError(f'Could not fetch verified TurboQuant file {name}: {type(e).__name__}') from e
                time.sleep(attempt+1)
    hashes=verify_upstream(root)
    return {'status':'verified','root':str(root),'repository':'https://github.com/0xSero/turboquant',
       'commit':TURBOQUANT_COMMIT,'license':'GPL-3.0','source_sha256':hashes,
       'integration':'upstream standalone quantizer and value codec through a snapshot adapter; NOT upstream vLLM backend',
       'network_credentials_sent':False,'upstream_setup_executed':False}

def load_upstream(root):
    root=Path(root).resolve();verify_upstream(root)
    if 'turboquant' in sys.modules:
        origin=Path(sys.modules['turboquant'].__file__).resolve()
        if root not in origin.parents: raise RuntimeError('Another turboquant package is already imported.')
    if str(root) not in sys.path: sys.path.insert(0,str(root))
    quant=importlib.import_module('turboquant.quantizer');kv=importlib.import_module('turboquant.kv_cache')
    for module in (quant,kv):
        if root not in Path(module.__file__).resolve().parents: raise RuntimeError('Wrong upstream module imported.')
    return quant,kv

def upstream_selftest(root):
    """Executed in Colab before Qwen load; real codecs, zero and random vectors, Qwen head_dim 256."""
    import torch
    quant,kv=load_upstream(root)
    gen=torch.Generator().manual_seed(97)
    rows=[]
    for bits in (3,4):
        codec=quant.TurboQuantProd(dim=256,bits=bits,device=torch.device('cpu'),dtype=torch.float32,seed=42)
        x=torch.randn((1,2,9,256),generator=gen);x[...,0,:]=0.
        encoded=codec.quantize(x);recon=codec.dequantize(encoded)
        if recon.shape!=x.shape or not torch.isfinite(recon).all() or not torch.equal(recon[...,0,:],x[...,0,:]): raise ValueError('TurboQuant key API/zero-vector test failed.')
        q=torch.randn((1,2,3,256),generator=gen)
        score=codec.attention_score(q,encoded);direct=q@recon.transpose(-2,-1)
        if not torch.allclose(score,direct,atol=3e-5,rtol=3e-5): raise ValueError('Upstream reconstruction/inner-product API mismatch.')
        for vb in (2,4):
            v=kv.quantize_values(x,bits=vb,group_size=32);vr=kv.dequantize_values(v,group_size=32)
            if vr.shape!=x.shape or not torch.isfinite(vr).all(): raise ValueError('TurboQuant value API test failed.')
        mse=encoded.mse_indices.numel();signs=encoded.qjl_signs.numel()
        rows.append({'key_nominal_bits':bits,'index_and_sign_storage_bits_per_coordinate':8*(mse+signs)/x.numel(),
           'reconstruction_rmse':float((x-recon).square().mean().sqrt()),
           'scope':'random CPU vectors; API/packing check, not model quality or GPU timing'})
    return {'status':'passed','rows':rows,'qwen_head_dim':256,'source':verify_upstream(root)}
