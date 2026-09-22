"""Read-only completed-H validation and historical regression handoff; never a new final selection."""
from __future__ import annotations
import io,re,zipfile
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
from safetensors.numpy import load as load_safetensors
from odij_core import *
from odij_learning import softmax

def collect_hashes(obj,out):
    if isinstance(obj,dict):
        for k,v in obj.items():
            if k in ('group','group_id') and isinstance(v,str):out.add(v)
            if k in ('state','text') and isinstance(v,str):out.add(text_hash(v))
            if re.fullmatch('[0-9a-f]{64}',str(k)):out.add(k)
            collect_hashes(v,out)
    elif isinstance(obj,list):
        for v in obj:collect_hashes(v,out)
    elif isinstance(obj,str) and re.fullmatch('[0-9a-f]{64}',obj):out.add(obj)

def prepare_history(archive,root):
    source=root/'source';safe_extract(archive,source,SOURCE_SHA256)
    s=read_json(source/'openkind_phase2h_summary.json')
    if s['status']!='completed' or any(s['workers'][w]['status']!='completed' for w in ('eval_qwen4b_strict','eval_qwen4b_tf32')):raise IntegrityError('Expected completed H continuation')
    if s.get('original_fits_retrained') is not False:raise IntegrityError('Expected no-retraining lineage')
    if file_sha(source/'original_attempt.zip')!='51c772ab248d8949fe73bbbd91056826975eebb3594d03ca693438355d9a4cda':raise IntegrityError('Original attempt archive differs')
    if file_sha(source/'final_lock.json')!='3b30ced9af995812aeb23db12fdedfb7ba3247324c184e56871dd9e4108c3d14':raise IntegrityError('Historical final lock differs')
    rec=read_json(source/'recovery_manifest.json');checks=[]
    for name,h in rec['code_sha256'].items():
        if file_sha(source/'implementation'/name)!=h:raise IntegrityError('Historical source hash differs '+name)
        checks.append(name)
    z=zipfile.ZipFile(source/'original_attempt.zip');fits=[p for p in (source/'fit_qwen4b').iterdir() if p.is_file()]
    preserved=0
    for p in fits:
        n='fit_qwen4b/'+p.name
        if n in z.namelist():
            if hashlib.sha256(z.read(n)).hexdigest()!=file_sha(p):raise IntegrityError('Recovered fit differs '+n)
            preserved+=1
    f=read_json(source/'fit_qwen4b/head_reference_fixtures.json')
    head=load_safetensors((source/'fit_qwen4b'/f['scorer_path']).read_bytes());vectors=load_safetensors((source/'fit_qwen4b/head_reference_vectors.safetensors').read_bytes());scoreerr=0.;perr=0.
    for i in range(len(f['candidate_scores'])):
        x=vectors[f'episode_{i}'].astype(float);scores=(((x-head['mean'])/head['std'])@head['linear.weight'].T+head['linear.bias']).ravel()
        mx=np.max(scores);st=np.array([mx,np.sort(scores)[-1]-np.sort(scores)[-2],scores.mean(),scores.std(),mx+np.log(np.exp(scores-mx).mean()),np.log(len(scores))]);nh=f['none_model']
        null=np.asarray(nh['coef'])@np.r_[1.,(st-np.asarray(nh['feature_mean']))/np.asarray(nh['feature_std'])]
        p=softmax(np.r_[scores,null],f['temperature']);scoreerr=max(scoreerr,float(np.max(np.abs(scores-f['candidate_scores'][i]))));perr=max(perr,float(np.max(np.abs(p-f['probabilities'][i]))))
    if scoreerr>1e-5 or perr>1e-6:raise IntegrityError('Historical exported-head algebra failed')
    blocked=set()
    for n in ('split_manifest.json','fit_payload.json','final_payload_reserved.json','criteria_support.json'):
        collect_hashes(read_json(source/n),blocked)
    # Never mutate the old registry. Its path is a source setting, not executable code.
    config=read_json(source/'source_configuration.json')
    paths=[]
    for key in ('registry_path','reference_archive'):
        raw=config.get(key)
        if not raw:continue
        path=Path(raw)
        if not str(path).startswith('/content/drive/MyDrive/'):continue
        if not path.exists():continue
        if path.suffix=='.json':collect_hashes(read_json(path),blocked);paths.append(str(path))
        elif path.suffix=='.zip':
            with zipfile.ZipFile(path) as older:
                for n in older.namelist():
                    if n.endswith(('prior_exclusions.json','fresh_message_manifest.json','split_manifest.json','experiment_inputs.json')) and older.getinfo(n).file_size<20*1024**2:
                        collect_hashes(json.loads(older.read(n)),blocked)
                paths.append(str(path))
    write_json(root/'history'/'excluded_group_hashes.json',sorted(blocked))
    errors=read_json(source/'eval_qwen4b_strict/high_confidence_errors.json')
    write_json(root/'history'/'H_error_review_dossier.json',{'scope':'Historical already-inspected H cases only; forbidden as new final/selection data. Do not silently relabel.','source_sha256':SOURCE_SHA256,'errors':errors})
    summary={'status':'verified_historical_artifacts','source_run':SOURCE_RUN,'source_sha256':SOURCE_SHA256,'implementation_hashes_checked':len(checks),'preserved_fit_files':preserved,
      'head_score_max_residual':scoreerr,'head_probability_max_residual':perr,'exclusion_hash_count':len(blocked),'additional_manifests_read':paths,
      'freshness_limit':'Exact groups/text from available H and earlier manifests; no paraphrase or pretraining decontamination.',
      'final_selected_model_unchanged':read_json(source/'fit_qwen4b/selection.json')['selected_profile'],
      'purpose':'Lineage/regression and protocol-design evidence; not a repeated final model-selection exercise.'}
    write_json(root/'history'/'source_audit.json',summary)
    return summary,blocked
