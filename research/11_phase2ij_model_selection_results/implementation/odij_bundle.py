"""Portable, checksummed selected-model handoff without training data or final labels.
The bundled Python reference remains CUDA/PyTorch, not a claimed Metal/GGUF export.
Base weights are pinned external dependencies; small heads/adapters (or learned encoder
weights) are carried. A separate hydration operation can make the base dependency local.
"""
from __future__ import annotations
import argparse, copy, json, os, shutil, sys, tempfile, time, zipfile
from pathlib import Path, PurePosixPath
import numpy as np
from odij_core import *

BUNDLE_SCHEMA='openkind-model-bundle/v1'


def checked_path(root,name):
    rel=PurePosixPath(name)
    if not name or rel.is_absolute() or '..' in rel.parts or '\\' in name:
        raise IntegrityError('Unsafe bundle member')
    path=Path(root).joinpath(*rel.parts)
    if not path.resolve().is_relative_to(Path(root).resolve()) or path.is_symlink():
        raise IntegrityError('Bundle member escapes root or is a symlink')
    return path


def verify_bundle(root):
    root=Path(root);m=read_json(root/'BUNDLE_MANIFEST.json')
    if m.get('schema')!=BUNDLE_SCHEMA:raise IntegrityError('Unknown model bundle schema')
    required={'profile.json','inference_protocol.json','MODEL_CONTRACT.json','golden.json'}
    if not required.issubset(m.get('files',{})):raise IntegrityError('Incomplete bundle manifest')
    for n,h in m['files'].items():
        p=checked_path(root,n)
        if not p.is_file() or file_sha(p)!=h:raise IntegrityError('Missing or changed bundle artifact: '+n)
    for folder in ('model','tokenizer','implementation'):
        for path in (root/folder).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.relative_to(root).as_posix() not in m['files']:
                raise IntegrityError('Unexpected unmanifested inference artifact')
    profile=read_json(root/'profile.json');base=checked_path(root,profile['artifact_dir'])
    if profile.get('head_file') and not (profile['artifact_dir']+'/'+profile['head_file']) in m['files']:
        raise IntegrityError('Profile head not locked in bundle')
    if profile.get('training_dir'):
        n=profile['artifact_dir']+'/'+profile['training_dir']+'/trained.safetensors'
        if n not in m['files'] or m['files'][n]!=profile['training']['sha256']:
            raise IntegrityError('Trainable parameters not locked in bundle')
    if not profile.get('spec',{}).get('revision') and profile.get('method')!='lexical':raise IntegrityError('Missing immutable base revision')
    return m


def inference_protocol(p):
    # This copy is a model-input contract. No case data, human names, or final labels.
    keys=('scope','train_families','dev_transfer_families','final_transfer_families','max_tokens',
          'max_questions','max_candidates','truncation','probability_tolerance','order_tolerance',
          'lora_last_blocks','target_device','deployment_target','quality_requirements')
    return {k:copy.deepcopy(p[k]) for k in keys if k in p}


def native_request_for(case):
    questions=[]
    for q in case['questions']:
        questions.append({k:copy.deepcopy(q[k]) for k in ('id','family','rubric_family','primitive','instruction','options')})
    return {'request_id':'fixture-'+case['id'],'state':case['state'],'questions':questions}


def fixture_cases(root):
    """Choose a small nonfinal interface set by coverage only, never by prediction outcome."""
    cases=read_json(Path(root)/'data/policy_dev.json');chosen=[];covered=set()
    for case in cases:
        kinds={q['primitive'] for q in case['questions']};natural=case['source']['kind']=='natural'
        tags=kinds|({'natural'} if natural else {'constructed'})
        if tags-covered:
            c=copy.deepcopy(case);qs=[];used=set()
            for q in c['questions']:
                if q['primitive'] not in used:qs.append(q);used.add(q['primitive'])
            c['questions']=qs[:3];chosen.append(c);covered.update(tags)
        if len(chosen)>=3:break
    if not chosen:raise IntegrityError('No nonfinal fixtures available')
    return chosen


class BundlePredictor:
    def __init__(self,bundle,runtime_dir,allow_cpu_test=False,local_base=None):
        from odij_runner import Predictor
        self.bundle=Path(bundle);self.manifest=verify_bundle(bundle)
        self.profile=read_json(self.bundle/'profile.json');self.p=read_json(self.bundle/'inference_protocol.json')
        spec=self.profile.get('spec')
        if local_base:
            base=Path(local_base);receipt=read_json(base/'BASE_HYDRATION.json')
            if receipt.get('model_id')!=spec['id'] or receipt.get('revision')!=spec['revision']:
                raise IntegrityError('Local base dependency has another checkpoint identity')
            for n,h in receipt['files'].items():
                if file_sha(checked_path(base,n))!=h:raise IntegrityError('Hydrated base weight changed')
            spec['local_base_path']=str(base)
        if spec and (self.bundle/'tokenizer').is_dir():spec['local_tokenizer_path']=str(self.bundle/'tokenizer')
        self.predictor=Predictor(self.bundle,self.p,self.profile,Path(runtime_dir),allow_cpu_test)
    def evaluate(self,req):
        from odij_serve import request_rows,native_response
        rows=request_rows(req,self.p)
        return {'schema':'openkind-native-study-response/v1','request_id':req['request_id'],
                'engine_profile':self.profile['profile_id'],'experimental':True,
                'answers':[native_response(r,self.predictor.predict(r,memo=False)) for r in rows]}
    def close(self):self.predictor.close()


def portable_contract(p,profile):
    from odij_runtime import PREFIX
    runtime=profile.get('runtime',{});spec=profile.get('spec',{})
    return {'schema':'openkind-model-contract/v1','profile_id':profile['profile_id'],
      'backbone':spec,'layout':profile.get('layout'),'method':profile['method'],'head_file':profile.get('head_file'),
      'training':profile.get('training'),'rejection':profile.get('kind',profile.get('name')),
      'calibration':profile['calibration'],'policy':profile['policy'],
      'model_config':runtime.get('model_config'),
      'feature_contract':{'pooling':'last candidate-conditioned token for Qwen; tracked option marker for ModernBERT',
          'causal_tokenization':'segments encoded independently with add_special_tokens=False; concatenate exact IDs',
          'prefix':PREFIX,'no_generation':True,'tied_vocab_matrix_not_removed':True,
          'implementation':'implementation/odij_runtime.py and odij_learning.py'},
      'artifact_format':'safetensors for fitted parameters; tokenizer assets and immutable base checkpoint reference',
      'base_weights_included':False,
      'probability_contract':{'choice':'K caller candidates plus separate native semantic none mass; no renormalization',
        'noul':'P(true) from ordered [false,true] candidates; no general insufficient-evidence guarantee',
        'score':'probabilities over explicit levels; sum(level_value * probability)',
        'confidence':'Not vendor-compatible; native confidence is null, top_probability separately named',
        'host_arithmetic':'Serialize Python float / Rust f64; model reference arithmetic is FP32',
        'option_ids':'Opaque; label and criteria are required semantic inputs'},
      'execution_contract':{'reference':'full sequential resident Python/PyTorch strict FP32',
        'nested_cache':'state-first causal only; attention KV, convolution and recurrent state all require isolated copies',
        'model_config_is_not_backend_support':'No ONNX, Candle, GGUF or Metal availability is inferred from file format',
        'batching_or_cache_promotion':'Requires the profile-specific probability/argmax/policy evidence',
        'capabilities':inference_protocol(p)},
      'rust_attachment':{'engine_trait':'DecisionEngine: Send + Sync; evaluate(SystemRequest) -> EngineResult<SystemResponse>',
        'boundary':'Keep server/client and registry; implement the selected probability model behind the engine boundary.',
        'wire_gaps':['Choice probabilities currently must cover only caller criteria and sum to one; native none needs an explicit versioned mapping.',
          'Choice/Score confidence is required by Rust wire types; native top_probability must not silently stand in for an unspecified confidence statistic.',
          'Rust accepts null descriptions and structured state/instructions; semantic rendering/adapter must be explicitly versioned.',
          'Mock transport tests are not evidence that the exported model is semantically validated.']},
      'evidence_scope':p['scope'],'production_ready':False,'native_target_parity':'not measured'}


def copy_profile(root,dest,profile):
    base=Path(root)/profile['artifact_dir'];target=dest/'model';target.mkdir(parents=True,exist_ok=True)
    names=[profile['name']+'.json']
    if profile.get('head_file'):names.append(profile['head_file'])
    if profile.get('training_dir'):names += [profile['training_dir']+'/trained.safetensors',profile['training_dir']+'/TRAINING_DONE.json']
    for name in names:atomic_copy(checked_path(base,name),checked_path(target,name))
    m=copy.deepcopy(profile);m['artifact_dir']='model'
    write_json(dest/'profile.json',m)
    return m


def tensor_inventory(root):
    from safetensors import safe_open
    out={}
    for p in Path(root).rglob('*.safetensors'):
        with safe_open(p,framework='pt',device='cpu') as f:
            out[p.relative_to(root).as_posix()]={k:{'shape':list(f.get_slice(k).get_shape()),'dtype':f.get_slice(k).get_dtype()} for k in f.keys()}
    return out


def build_and_verify_bundle(root,cfg):
    from odij_data import assert_study_intact
    from odij_runner import Predictor
    from odij_serve import request_rows,native_response
    from odij_runtime import segments,joint_ids,finite_ids
    from odij_selection import decision_report
    root=Path(root);assert_study_intact(root);lock=read_json(root/'MODEL_LOCK.json');pid=lock.get('selected_profile')
    if not pid:raise GateBlocked('No eligible model selected; no substitute is silently exported')
    profile=next(m for m in lock['profiles'] if m['profile_id']==pid);p=read_json(root/'registered_protocol.json')
    for n,h in lock['artifact_hashes'].items():
        if file_sha(root/n)!=h:raise IntegrityError('Model lock changed before export')
    dest=root/'model_bundle'
    if (dest/'BUNDLE_MANIFEST.json').exists():
        m=verify_bundle(dest)
        if m['model_lock_sha256']!=file_sha(root/'MODEL_LOCK.json'):raise IntegrityError('Bundle belongs to another model lock')
        if (dest/'RELOAD_CHECK.json').exists():
            check=read_json(dest/'RELOAD_CHECK.json')
            if check.get('bundle_manifest_sha256')!=file_sha(dest/'BUNDLE_MANIFEST.json'):raise IntegrityError('Reload receipt belongs to another bundle')
            if check.get('passed'):return publish_bundle(root,cfg)
    elif dest.exists():shutil.rmtree(dest)  # Only the incomplete derived bundle, never training/results.
    if not (dest/'BUNDLE_MANIFEST.json').exists():
        dest.mkdir(parents=True,exist_ok=True);copied=copy_profile(root,dest,profile)
        write_json(dest/'inference_protocol.json',inference_protocol(p));write_json(dest/'MODEL_CONTRACT.json',portable_contract(p,copied))
        write_json(dest/'MODEL_DECISION.json',decision_report(root))
        from odij_head_reference import head_graph
        write_json(dest/'HEAD_GRAPH.json',head_graph(copied))
        golden=[];pr=Predictor(root,p,profile,root/'bundle_source_runtime',cfg.allow_cpu_test)
        try:
            if pr.rt and hasattr(pr.rt.tokenizer,'save_pretrained'):pr.rt.tokenizer.save_pretrained(dest/'tokenizer')
            elif not cfg.allow_cpu_test and profile['method']!='lexical':raise IntegrityError('Selected tokenizer cannot be exported')
            for case in fixture_cases(root):
                req=native_request_for(case);rows=request_rows(req,p);answers=[];features=[]
                for r in rows:
                    z=pr.logits(r,memo=False);probs=pr.predict(r,memo=False);answers.append(native_response(r,probs))
                    item={'question_id':r['q']['id'],'logits':np.asarray(z).tolist(),'probabilities':probs.tolist()}
                    if pr.rt:
                        tok=pr.rt.tokenizer
                        if profile['method']=='finite':item['input_ids'],item['code_token_ids']=finite_ids(r,tok,p['max_tokens'])
                        elif profile.get('spec',{}).get('kind')=='encoder':item['input_ids'],item['option_marker_indices']=joint_ids(r,tok,p['max_tokens'])
                        else:
                            a,b,c,d=segments(r,tok,profile['layout'],p['max_tokens']);item.update(root_ids=a,question_ids=b,candidate_suffix_ids=c,full_candidate_ids=d)
                        if profile['method']=='frozen':item['candidate_features']=pr.rt.row_features(r,profile['layout'],memo=False).tolist()
                        elif profile['method']=='encoder_native':
                            import torch
                            with torch.no_grad():item['marker_features']=pr.rt.marker_features(r).detach().float().cpu().tolist()
                    features.append(item)
                golden.append({'request':req,'expected_native_answers':answers,'head_and_token_fixtures':features,'source_partition':'policy_dev','ground_truth_labels_exported':False})
            write_json(dest/'golden.json',golden)
        finally:pr.close()
        write_json(dest/'tensor_inventory.json',tensor_inventory(dest/'model'))
        for module in Path(__file__).parent.glob('odij_*.py'):atomic_copy(module,dest/'implementation'/module.name)
        atomic_bytes(dest/'serve_bundle.py',b"from pathlib import Path\nimport sys\nsys.path.insert(0,str(Path(__file__).parent/'implementation'))\nfrom odij_bundle import serve_main\nif __name__=='__main__': serve_main()\n")
        atomic_bytes(dest/'hydrate_base.py',b"from pathlib import Path\nimport sys\nsys.path.insert(0,str(Path(__file__).parent/'implementation'))\nfrom odij_bundle import hydrate_main\nif __name__=='__main__': hydrate_main()\n")
        atomic_bytes(dest/'requirements.txt',b'transformers==5.17.0\nsafetensors\nnumpy\nscipy\naccelerate\nhuggingface-hub\npsutil\n# Install the compatible Torch 2.11.x/CUDA build separately.\n')
        atomic_bytes(dest/'RUST_MODEL_HANDOFF.md',handoff_text(copied,p).encode())
        files=tree_hashes(dest,('__pycache__',))
        write_json(dest/'BUNDLE_MANIFEST.json',{'schema':BUNDLE_SCHEMA,'version':VERSION,'profile_id':pid,
          'model_lock_sha256':file_sha(root/'MODEL_LOCK.json'),'files':files,'base_weights_included':False,
          'base':profile.get('spec'),'data_scope':p['scope'],'cpu_toy_only':cfg.allow_cpu_test,
          'hashes_are_integrity_checks_not_signatures':True,'production_ready':False})
    bundle=BundlePredictor(dest,root/'bundle_reload_runtime',cfg.allow_cpu_test)
    maximum=0.;changes=0;n=0
    try:
        for f in read_json(dest/'golden.json'):
            actual=bundle.evaluate(f['request'])['answers']
            for a,b in zip(actual,f['expected_native_answers']):
                if set(a['option_probabilities'])!=set(b['option_probabilities']):raise IntegrityError('Reloaded option IDs differ')
                delta=max(abs(a['option_probabilities'][k]-b['option_probabilities'][k]) for k in a['option_probabilities'])
                if a['none_probability'] is not None:delta=max(delta,abs(a['none_probability']-b['none_probability']))
                maximum=max(maximum,delta);changes+=a['selected_id']!=b['selected_id'];n+=1
    finally:bundle.close()
    passed=maximum<=p['probability_tolerance'] and changes==0
    from odij_head_reference import check_head_fixtures
    algebra=check_head_fixtures(dest)
    if algebra.get('status')=='failed':passed=False
    write_json(dest/'HEAD_ALGEBRA_CHECK.json',algebra)
    receipt={'head_algebra':algebra,'passed':bool(passed),'questions':n,'max_probability_delta':maximum,'selected_id_changes':int(changes),
        'bundle_manifest_sha256':file_sha(dest/'BUNDLE_MANIFEST.json'),'cpu_toy_only':cfg.allow_cpu_test,
        'scope':'Freshly loaded exported artifacts vs the same selected Python reference on nonfinal fixtures. Not Rust/Metal or arbitrary input certification.'}
    write_json(dest/'RELOAD_CHECK.json',receipt)
    if not passed:raise IntegrityError('Bundle reload does not preserve selected profile on its fixtures')
    return publish_bundle(root,cfg)


def publish_bundle(root,cfg):
    root=Path(root);dest=root/'model_bundle';verify_bundle(dest)
    archive=root/'exports'/('model_bundle_'+root.name+'.zip');archive.parent.mkdir(parents=True,exist_ok=True)
    tmp=archive.with_suffix('.tmp.zip')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
        m=read_json(dest/'BUNDLE_MANIFEST.json')
        for n in sorted(set(m['files'])|{'BUNDLE_MANIFEST.json','RELOAD_CHECK.json','HEAD_ALGEBRA_CHECK.json'}):z.write(dest/n,n)
    os.replace(tmp,archive);drive=Path(cfg.drive_root)/cfg.study_id;atomic_copy(archive,drive/archive.name)
    result={'status':'exported_reference_bundle','profile_id':read_json(dest/'profile.json')['profile_id'],
        'archive':str(archive),'drive_archive':str(drive/archive.name),'sha256':file_sha(archive),
        'base_weights_included':False,'reload_check':read_json(dest/'RELOAD_CHECK.json'),'production_ready':False}
    write_json(root/'bundle_export.json',result);return result


def handoff_text(profile,p):
    return f'''# Selected model → Rust handoff

Profile: `{profile['profile_id']}`. Method: `{profile['method']}`. Layout: `{profile.get('layout')}`.
Scope: `{p['scope']}`. This is a reference bundle, not a claim of native/production readiness.

## Build around this contract, not a guessed backbone

Read MODEL_CONTRACT.json, HEAD_GRAPH.json, tensor_inventory.json, inference_protocol.json, and golden.json.
The independent NumPy readout checker and HEAD_ALGEBRA_CHECK.json make head porting testable without rerunning the backbone.
Keep your existing DecisionEngine and server/client. Implement the chosen model behind that boundary.
For Qwen the executable graph is token embeddings → hybrid DeltaNet/attention backbone → last
candidate token features → the fitted rank/rejection modules → calibrated probabilities. Stateful
branching requires recurrent, convolution, and attention-KV isolation. For ModernBERT use the joint
candidate renderer and exact marker indices; do not graft Qwen cache rules onto an encoder. For the
finite-code model gather the selected vocabulary rows; no answer-generation loop is required.

All custom readout weights are in model/. LoRA training tensors and exact targets are recorded
when applicable. These are custom low-rank modules, not a PEFT adapter export. A generic GGUF
converter or text-generation endpoint is not automatically capable of this decision graph.

## Loading locally

`python serve_bundle.py --bundle .` loads the pinned base and fitted parameters once and reads native
JSONL on stdin. Torch 2.11.x + Transformers 5.17.0 and a supported NVIDIA CUDA GPU are required by this
reference loader. No HTTP tunnel/server is opened. The bundle does not require cases.jsonl or final
labels. Base weights are deliberately not duplicated in this archive. To make that dependency local,
run `python hydrate_base.py --bundle . --destination ./base_weights` and later pass
`--local-base ./base_weights`. That operation downloads potentially many GiB and writes its own digest
receipt. Verify source/model rights before redistributing weights. This is not an MLX/Metal loader.

## Existing Rust wire contract needs an explicit adapter

Your ChoiceAnswer requires caller option keys only, total mass one, a selected key, and confidence.
The experimental model returns separate native none mass and null selected_id when none wins.
Do not drop that mass, renormalize silently, invent an unrequested option, or substitute max probability
for an unspecified confidence statistic. Choose a versioned native extension or caller-explicit none
contract; test that mapping separately. Rust f64 wire values do not require the backbone to execute f64.
Null descriptions and structured inputs also need a declared semantic renderer, not a transport rewrite.

## Parity order

1. Verify every artifact digest and tensor name/shape.
2. Match golden token IDs and marker/segment positions.
3. Replay the exported feature → logits → probability fixtures in Rust.
4. Match full-backbone features on the actual deployment device.
5. Only then approve batching/caching/alternate precision on that profile.

RELOAD_CHECK.json checks a fresh Python load. It is not step 3 or 4 in Rust, not a Mac benchmark,
and not vendor Jev conformance. Numerical/semantic failures remain separate from execution completion.
'''


def hydrate_base(bundle,destination):
    from huggingface_hub import snapshot_download
    m=verify_bundle(bundle);spec=m['base'];out=Path(destination)
    if out.exists() and any(out.iterdir()):raise IntegrityError('Base destination must be empty; do not mix checkpoints')
    out.mkdir(parents=True,exist_ok=True)
    cached=Path(snapshot_download(spec['id'],revision=spec['revision'],token=os.environ.get('HF_TOKEN'),
        allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.tiktoken','LICENSE*','README*']))
    for path in cached.rglob('*'):
        if path.is_file():atomic_copy(path,out/path.relative_to(cached))
    if not list(out.glob('*.safetensors')):raise IntegrityError('No safetensors base weights downloaded')
    receipt={'model_id':spec['id'],'revision':spec['revision'],'files':tree_hashes(out),'rights_cleared':False}
    write_json(out/'BASE_HYDRATION.json',receipt);return receipt


def hydrate_main():
    ap=argparse.ArgumentParser();ap.add_argument('--bundle',default='.');ap.add_argument('--destination',required=True);a=ap.parse_args()
    hydrate_base(a.bundle,a.destination);print('Pinned base downloaded. Source terms and target-device parity remain separate.')


def serve_main():
    from contextlib import redirect_stdout
    from odij_serve import MAX_LINE_BYTES
    ap=argparse.ArgumentParser();ap.add_argument('--bundle',default='.');ap.add_argument('--local-base');a=ap.parse_args()
    # Loader diagnostics must not corrupt the machine-readable stdout channel.
    with tempfile.TemporaryDirectory(prefix='openkind-runtime-') as tmp:
        with redirect_stdout(sys.stderr):model=BundlePredictor(a.bundle,tmp,local_base=a.local_base)
        try:
            while True:
                line=sys.stdin.buffer.readline(MAX_LINE_BYTES+1)
                if not line:break
                if len(line)>MAX_LINE_BYTES:
                    print(json.dumps({'error':{'code':'request_too_large'}}),flush=True);break
                rid=None;start=time.monotonic()
                try:
                    req=json.loads(line);rid=req.get('request_id');budget=req.get('deadline_ms',60000)
                    if not isinstance(budget,(int,float)) or not 0<budget<=600000:raise ValueError('Invalid deadline')
                    with redirect_stdout(sys.stderr):result=model.evaluate(req)
                    elapsed=(time.monotonic()-start)*1000
                    if elapsed>budget:raise TimeoutError('Deadline after nonpreemptive model operation')
                    result['elapsed_ms']=elapsed
                except Exception as e:result={'request_id':rid,'error':{'code':type(e).__name__,'message':'Request rejected; no partial decision returned.'}}
                print(json.dumps(result,allow_nan=False),flush=True)
        finally:model.close()
