"""Selective-precision experiments. Promotions are temporary; no trained weights are updated."""
from __future__ import annotations
import gc, hashlib, inspect, json
from contextlib import contextmanager, nullcontext
from collections import Counter
import numpy as np
import torch
from phase2d_common import (json_ready, json_write, pack_tokens, last_nonpad,
    forward_features, require_finite, sync, gpu_memory, metrics, sanitize_error)
from phase2e_core import StageSkip, IntegrityError
from phase2e_runtime import (integer_sample_indices, is_fatal_cuda,
    raise_if_fatal_cuda, aggregate_outcomes)

MODES=('bf16_default','bf16_strict_math','first4_blocks_fp32','delta_modules_fp32','mlp_modules_fp32','fp32_strict_math')

def backend_flags():
    d={'float32_matmul_precision':torch.get_float32_matmul_precision()}
    for prefix,obj in [('matmul',torch.backends.cuda.matmul),('cudnn',torch.backends.cudnn)]:
        for name in ('allow_tf32','allow_bf16_reduced_precision_reduction','allow_fp16_reduced_precision_reduction','benchmark','deterministic'):
            if hasattr(obj,name):d[f'{prefix}.{name}']=getattr(obj,name)
    return d

def set_flags(flags):
    torch.set_float32_matmul_precision(flags['float32_matmul_precision'])
    for prefix,obj in [('matmul',torch.backends.cuda.matmul),('cudnn',torch.backends.cudnn)]:
        for name in ('allow_tf32','allow_bf16_reduced_precision_reduction','allow_fp16_reduced_precision_reduction','benchmark','deterministic'):
            if f'{prefix}.{name}' in flags:setattr(obj,name,flags[f'{prefix}.{name}'])

def sample_parameter_digest(model):
    """Lightweight mutation alarm, explicitly NOT a full checkpoint checksum."""
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        h.update(name.encode());flat=p.detach().reshape(-1)
        indices = integer_sample_indices(flat.numel())
        if not indices:
            continue
        ix = torch.tensor(indices, dtype=torch.long, device=flat.device)
        v = flat.index_select(0, ix).float().cpu().numpy()
        h.update(v.tobytes())
    return h.hexdigest()

def cast_floats(value,dtype):
    if isinstance(value,torch.Tensor):return value.to(dtype=dtype) if value.is_floating_point() else value
    if isinstance(value,tuple):return tuple(cast_floats(v,dtype) for v in value)
    if isinstance(value,list):return [cast_floats(v,dtype) for v in value]
    if isinstance(value,dict):return {k:cast_floats(v,dtype) for k,v in value.items()}
    # Cache objects are deliberately not walked or cast here.
    return value

def promoted_forward(original):
    def run(*args,**kwargs):
        x=args[0] if args else kwargs.get('hidden_states')
        if not isinstance(x,torch.Tensor):raise TypeError('Promoted module requires hidden_states as first tensor')
        incoming=x.dtype
        # Cast rotary embeddings and floating attention masks with hidden states.
        out=original(*cast_floats(args,torch.float32),**cast_floats(kwargs,torch.float32))
        return cast_floats(out,incoming)
    return run

def target_modules(model,mode):
    if mode=='first4_blocks_fp32':return [(f'layers.{i}',m) for i,m in enumerate(model.layers[:4])]
    if mode=='delta_modules_fp32':return [(n,m) for n,m in model.named_modules() if type(m).__name__=='Qwen3_5GatedDeltaNet']
    if mode=='mlp_modules_fp32':return [(n,m) for n,m in model.named_modules() if type(m).__name__=='Qwen3_5MLP']
    return []

@contextmanager
def precision_mode(model,mode,device,cfg):
    if mode not in MODES:raise ValueError(mode)
    if mode=='fp32_strict_math' and not cfg.run_fp32:raise StageSkip('FP32 disabled in settings')
    flags=backend_flags();before=sample_parameter_digest(model)
    param_dtypes={n:p.dtype for n,p in model.named_parameters()}
    # Preserve exact original small buffers, including FP32 rotary frequencies.
    buffers={n:(b.detach().cpu().clone(),b.dtype,b.device) for n,b in model.named_buffers()}
    targets=target_modules(model,mode)
    if mode in ('delta_modules_fp32','mlp_modules_fp32') and not targets:raise StageSkip(f'No expected modules for {mode}')
    all_fp32=mode=='fp32_strict_math'
    selected=list(model.parameters()) if all_fp32 else list({id(p):p for _,m in targets for p in m.parameters()}.values())
    if torch.device(device).type=='cuda' and selected:
        gc.collect();torch.cuda.empty_cache();sync(device)
        free,_=torch.cuda.mem_get_info(device)
        extra=sum(p.numel()*max(0,4-p.element_size()) for p in selected)
        largest=max((p.numel()*4 for p in selected),default=0)
        if free<extra+largest+cfg.fp32_workspace_gib*1024**3:
            raise StageSkip(f'Precision memory guard: free {free/1024**3:.2f} GiB, conservative additional requirement {(extra+largest)/1024**3+cfg.fp32_workspace_gib:.2f} GiB. No CPU offload.')
    changes=[]
    primary_error = None
    try:
        if mode!='bf16_default':
            strict=dict(flags);strict['float32_matmul_precision']='highest'
            for k in strict:
                if k.endswith(('allow_tf32','allow_bf16_reduced_precision_reduction','allow_fp16_reduced_precision_reduction','benchmark')):strict[k]=False
                if k.endswith('deterministic'):strict[k]=True
            set_flags(strict)
        if all_fp32:model.to(dtype=torch.float32)
        else:
            for name,module in targets:
                old=module.forward;changes.append((module,old));module.to(dtype=torch.float32)
                module.forward=promoted_forward(old)
        ctx=nullcontext()
        if mode!='bf16_default':
            from torch.nn.attention import sdpa_kernel,SDPBackend
            ctx=sdpa_kernel(SDPBackend.MATH)
        model.eval()
        with ctx:
            yield {'mode':mode,'promoted_modules':[n for n,_ in targets],
                'parameter_dtypes':dict(Counter(str(p.dtype) for p in model.parameters())),
                'parameter_storage_mib':sum(p.numel()*p.element_size() for p in model.parameters())/1024**2,
                'flags':backend_flags(),'forced_math_sdpa':mode!='bf16_default',
                'scope':'module-level FP32 islands cast outputs back at boundary; not a custom accumulator kernel; no parameter learning'}
    except BaseException as error:
        primary_error = error
        raise
    finally:
        # Restoring Python callables is safe even after a device assertion.
        for m, old in reversed(changes):
            m.forward = old
        # CUDA asserts invalidate all device allocations. Do not copy weights,
        # hash GPU tensors, synchronize, or call empty_cache in that context.
        if not (primary_error is not None and is_fatal_cuda(primary_error)):
            try:
                with torch.no_grad():
                    for n, p in model.named_parameters():
                        if p.dtype != param_dtypes[n]:
                            p.data = p.data.to(dtype=param_dtypes[n])
                    for n, (value, dtype, dev) in buffers.items():
                        parent, _, leaf = n.rpartition('.')
                        mod = model.get_submodule(parent) if parent else model
                        old = mod._buffers[leaf]
                        if old.shape != value.shape:
                            raise IntegrityError('A model buffer changed shape during diagnostics')
                        mod._buffers[leaf] = value.to(device=dev, dtype=dtype)
                set_flags(flags)
                gc.collect()
                if torch.device(device).type == 'cuda':
                    torch.cuda.empty_cache()
                    sync(device)
                if sample_parameter_digest(model) != before:
                    raise IntegrityError('Parameter sample mutation after precision restoration')
            except BaseException as cleanup_error:
                # Keep both errors chained, and never hide a fatal cleanup error.
                raise_if_fatal_cuda(cleanup_error)
                if primary_error is not None:
                    raise IntegrityError('Precision restoration failed; model state is not reusable') from cleanup_error
                raise


def dtype_signature(x):
    if isinstance(x,torch.Tensor):return {'dtype':str(x.dtype),'shape':list(x.shape)}
    if isinstance(x,(tuple,list)):return [dtype_signature(v) for v in x if isinstance(v,(tuple,list,torch.Tensor))]
    if isinstance(x,dict):return {k:dtype_signature(v) for k,v in x.items() if isinstance(v,(tuple,list,torch.Tensor))}
    return type(x).__name__

@torch.inference_mode()
def capture_internal(model,batch):
    """Trace module boundaries in first four blocks; do not claim internal op dtypes from hooks."""
    rows={};handles=[];idx=int(torch.nonzero(batch['attention_mask'][0])[-1])
    targets={}
    for i,block in enumerate(model.layers[:4]):
        targets[f'block{i}']=block
        for n,m in block.named_modules():
            if n and ('.' not in n or n.endswith(('in_proj_qkv','out_proj','gate_proj','down_proj'))):targets[f'block{i}.{n}']=m
    for name,module in targets.items():
        def hook(m,args,out,name=name):
            tensor=out[0] if isinstance(out,tuple) else out
            item={'input':dtype_signature(args),'output':dtype_signature(out)}
            if isinstance(tensor,torch.Tensor) and tensor.ndim==3 and tensor.shape[1]==batch['input_ids'].shape[1]:
                item['target_vector']=tensor[0,idx].detach().float().cpu().numpy()
            rows[name]=item
        handles.append(module.register_forward_hook(hook))
    try:forward_features(model,batch)
    finally:
        for h in handles:h.remove()
    return rows

@torch.inference_mode()
def probability_call(model,head,batch):
    x=forward_features(model,batch).cpu()
    logits=head(x).float();require_finite(logits,'NLI head')
    return logits.numpy(),torch.softmax(logits,-1).numpy()

def shape_batches(row,short,long,pad,device):
    ids=row['input_ids'];n=len(ids)
    return {
      'repeat_alone':pack_tokens([ids],pad,device,explicit_positions=True),
      'duplicate_batch2':pack_tokens([ids,ids],pad,device,explicit_positions=True),
      'right_pad32':pack_tokens([ids],pad,device,pad_to=n+32,explicit_positions=True),
      'mixed_batch3':pack_tokens([ids,short['input_ids'],long['input_ids']],pad,device,explicit_positions=True),
      'left_pad32_explicit':pack_tokens([ids],pad,device,pad_to=n+32,side='left',explicit_positions=True)}

def run_precision(model,head,rows,regression,tokenizer,cfg,device,out):
    from time import perf_counter
    out.mkdir(parents=True,exist_ok=True)
    records=[];reports={};trace_rows=[];reference_prob={};pad=tokenizer.pad_token_id
    short=min(rows,key=lambda r:len(r['input_ids']));long=max(rows,key=lambda r:len(r['input_ids']))
    for mode in cfg.precision_modes:
        try:
            with precision_mode(model,mode,device,cfg) as meta:
                print('Precision:',mode,flush=True);start=perf_counter();alone=[]
                for j,row in enumerate(rows):
                    batch=pack_tokens([row],pad,device,explicit_positions=True)
                    _,p=probability_call(model,head,batch);p=p[0];alone.append(p)
                    baseline=capture_internal(model,batch) if j<cfg.sizes()['trace'] else None
                    for name,b in shape_batches(row,short,long,pad,device).items():
                        _,other=probability_call(model,head,b);delta=float(np.max(np.abs(other[0]-p)))
                        records.append({'mode':mode,'scenario':name,'example':j,'delta':delta,
                           'class_changed':bool(other[0].argmax()!=p.argmax()),'over_tolerance':delta>cfg.probability_tolerance})
                        if baseline is not None and name=='duplicate_batch2':
                            trace=capture_internal(model,b)
                            for m,a in baseline.items():
                                if 'target_vector' in a and 'target_vector' in trace.get(m,{}):
                                    diff=np.abs(a['target_vector']-trace[m]['target_vector'])
                                    trace_rows.append({'mode':mode,'example':j,'module':m,'max_abs':float(diff.max()),'mean_abs':float(diff.mean()),
                                                       'alone_io':{k:v for k,v in a.items() if k!='target_vector'}})
                reference_prob[mode]=np.asarray(alone)
                quality={}
                for split,data in regression.items():
                    logits=[]
                    for row in data:
                        l,_=probability_call(model,head,pack_tokens([row],pad,device,explicit_positions=True));logits.append(l[0])
                    arr=np.asarray(logits);labels=np.array([r['label'] for r in data])
                    quality[split]=metrics(arr,labels)
                    np.savez_compressed(out/f'{mode}_{split}.npz',logits=arr,labels=labels)
                durations=[]
                b=pack_tokens([rows[0]],pad,device,explicit_positions=True)
                for _ in range(cfg.benchmark_warmups):probability_call(model,head,b)
                for _ in range(cfg.benchmark_repeats):
                    sync(device);t=perf_counter();probability_call(model,head,b);sync(device);durations.append(1000*(perf_counter()-t))
                rr=[r for r in records if r['mode']==mode]
                reports[mode]={'status':'completed','configuration':meta,'quality_regression':quality,
                    'max_shape_probability_delta':max(r['delta'] for r in rr),'class_changes':sum(r['class_changed'] for r in rr),
                    'within_sampled_tolerance':all(not r['over_tolerance'] and not r['class_changed'] for r in rr),
                    'single_call_p50_ms':float(np.median(durations)),'sample_timings_ms':durations,
                    'resident_memory':gpu_memory(device),'seconds':perf_counter()-start}
        except IntegrityError:raise
        except Exception as e:
            reports[mode]={'status':'skipped' if isinstance(e,StageSkip) else 'failed','error':sanitize_error(e)}
            json_write(out/'partial.json', {'modes': reports, 'shape_records': records, 'internal_traces': trace_rows})
            raise_if_fatal_cuda(e)
            print(mode,reports[mode]['status'],reports[mode]['error'],flush=True)
        json_write(out/'partial.json',{'modes':reports,'shape_records':records,'internal_traces':trace_rows})
    if 'fp32_strict_math' in reference_prob:
        p0=reference_prob['fp32_strict_math']
        for mode,p in reference_prob.items():
            reports[mode]['versus_fp32_single']={'max_probability_delta':float(np.abs(p-p0).max()),'class_changes':int((p.argmax(1)!=p0.argmax(1)).sum())}
    result={'status':aggregate_outcomes(reports),'modes':reports,'shape_records':records,'internal_traces':trace_rows,
            'tolerance':cfg.probability_tolerance,'selection':'no automatic production mode selected',
            'sample_scope':'reused D diagnostic inputs, enriched for drift; NLI is fixed C test-regression, not a fresh benchmark',
            'audit_scope':'module I/O hooks do not reveal every internal accumulation dtype; source files and hashes recorded separately'}
    json_write(out/'precision_report.json',result)
    return result
