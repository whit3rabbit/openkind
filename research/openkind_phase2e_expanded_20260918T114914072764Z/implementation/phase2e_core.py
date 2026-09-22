"""Phase 2E contracts and read-only provenance. No remote execution on import."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import Counter
import hashlib, json, os, zipfile, stat, time, copy
import numpy as np
import torch
from safetensors.torch import load_file
from phase2d_common import (FixedHead, CandidateHead, json_ready, json_write, file_sha,
    load_heads_and_check, encode_candidate, encode_nli, pack_tokens, forward_features,
    normalize_nli, recover_rows, text_hash, choose_banking, load_banking_without_scripts,
    DYNAMIC_PREFIX, DYNAMIC_STATE_PREFIX, DYNAMIC_CAND_PREFIX, DYNAMIC_TAIL, LABELS)
from phase2d_decisions import NoneModel, make_episode, prior_weights

VERSION = '2e.1.1'
SOURCES = {
 'model': 'https://huggingface.co/Qwen/Qwen3.5-4B-Base',
 'transformers_model': 'https://raw.githubusercontent.com/huggingface/transformers/v5.17.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py',
 'cache': 'https://raw.githubusercontent.com/huggingface/transformers/v5.17.0/src/transformers/cache_utils.py',
 'numerics': 'https://docs.pytorch.org/docs/2.11/notes/numerical_accuracy.html',
}
@dataclass
class Settings:
    preset: str = 'standard'
    model_id: str = 'Qwen/Qwen3.5-4B-Base'
    model_revision: str = '1001bb4d826a52d1f399e183466143f4da7b741b'
    nli_revision: str = 'da70db2af9d09693783c3320c4249840212ee221'
    banking_revision: str = '90d4e2ee5521c04fc1488f065b8b083658768c57'
    drive_notebooks: str = '/content/drive/MyDrive/Colab Notebooks'
    reference_c: str = ''
    reference_d: str = ''
    expected_c_sha: str = '2952c2fcca6638c0cc1d1e1060a567b849e46424eb2d16a078d1945233bffb63'
    expected_d_sha: str = '88a40170bd7b6a44bc41439c14163c5f0f816121201dd4106a2ffa7c51df3050'
    output_root: str = '/content/openkind_phase2e'
    cache_root: str = '/content/openkind_phase2e_cache'
    max_length: int = 256
    seed: int = 71
    run_precision: bool = True
    run_shared_prefix: bool = True
    run_policy_evaluation: bool = True
    run_request_benchmark: bool = True
    run_fp32: bool = True
    run_long_prefix_benchmark: bool = True
    copy_results_to_drive: bool = True
    precision_modes: tuple = ('bf16_default','bf16_strict_math','first4_blocks_fp32',
                              'delta_modules_fp32','mlp_modules_fp32','fp32_strict_math')
    cache_modes: tuple = ('bf16_default','fp32_strict_math')
    policy_execution_mode: str = 'bf16_default'
    candidate_counts: tuple = (2,4,8,16)
    absent_priors: tuple = (0.05,0.25,0.5)
    wrong_answer_costs: tuple = (1.,5.,20.)
    review_cost: float = 0.1
    probability_tolerance: float = 0.005
    cache_probability_tolerance: float = 0.005
    order_probability_tolerance: float = 1e-6
    benchmark_repeats: int = 5
    benchmark_warmups: int = 2
    fp32_workspace_gib: float = 2.0
    long_prefix_lengths: tuple = (64,256,1024)
    cache_reference_method: str = 'deepcopy_all_state'
    resume_score_cache: bool = True

    def sizes(self):
        choices={
          'smoke': dict(numerical=2,trace=1,nli_per_split=3,dev=4,test_seen=4,test_unseen=4,cache_episodes=2,bench_messages=1),
          'quick': dict(numerical=8,trace=1,nli_per_split=24,dev=16,test_seen=16,test_unseen=16,cache_episodes=4,bench_messages=1),
          'standard': dict(numerical=24,trace=3,nli_per_split=128,dev=64,test_seen=64,test_unseen=64,cache_episodes=8,bench_messages=2)}
        if self.preset not in choices: raise ValueError('preset must be smoke, quick or standard')
        if self.max_length != 256: raise ValueError('Preserve the trained 256-token scoring contract')
        if any(k < 2 or k > 16 for k in self.candidate_counts): raise ValueError('This experiment supports K=2..16')
        if self.policy_execution_mode != 'bf16_default':
            raise ValueError('Policy validation remains on the frozen BF16 reference; precision comparisons are separate')
        return choices[self.preset]

    def archive_paths(self):
        d=Path(self.drive_notebooks)
        return (Path(self.reference_c) if self.reference_c else d/'OpenKind_Phase2C_results/20260917T222948Z/openkind_phase2c_20260917T222948Z.zip',
                Path(self.reference_d) if self.reference_d else d/'OpenKind_Phase2D_results/20260917T234417Z/openkind_phase2d_20260917T234417Z.zip')

class StageSkip(RuntimeError): pass
class IntegrityError(RuntimeError): pass

def safe_extract(archive, root, expected):
    """Validate before extraction; archives are data and are never imported/executed."""
    archive,root=Path(archive),Path(root)
    if not archive.is_file(): raise FileNotFoundError(f'Required prior result archive is missing: {archive}')
    actual=file_sha(archive)
    if actual != expected: raise IntegrityError(f'Prior archive checksum mismatch: {archive.name}')
    root.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        infos=z.infolist()
        if len(infos)>2000 or sum(i.file_size for i in infos)>512*1024**2: raise IntegrityError('Oversized archive')
        if len({i.filename for i in infos}) != len(infos): raise IntegrityError('Duplicate ZIP member names')
        for i in infos:
            name=Path(i.filename)
            if name.is_absolute() or '..' in name.parts or '\\' in i.filename or stat.S_ISLNK(i.external_attr>>16):
                raise IntegrityError('Unsafe archive path')
            if i.file_size>80*1024**2: raise IntegrityError('Oversized archive member')
        for i in infos:
            if i.is_dir(): continue
            p=root/i.filename; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(z.read(i))
    return root

def read_sources(cfg, out):
    cpath,dpath=cfg.archive_paths()
    c=safe_extract(cpath,Path(cfg.cache_root)/'source_c',cfg.expected_c_sha)
    d=safe_extract(dpath,Path(cfg.cache_root)/'source_d',cfg.expected_d_sha)
    read=lambda root,name:json.loads((root/name).read_text())
    ce=read(c,'dynamic_export/manifest.json'); ne=read(c,'frozen_nli_export/manifest.json')
    de=read(d,'export/manifest.json')
    for m in (ce,ne,de):
        if (m['model_id'],m['model_revision'])!=(cfg.model_id,cfg.model_revision): raise IntegrityError('Model identity mismatch')
    expected={'prefix':DYNAMIC_PREFIX,'state_prefix':DYNAMIC_STATE_PREFIX,'candidate_prefix':DYNAMIC_CAND_PREFIX,'tail':DYNAMIC_TAIL}
    if ce['prompt_segments'] != expected or de['candidate_contract']['prompt_segments'] != expected:
        raise IntegrityError('Candidate prompt contract changed')
    ref={'root':c,'nli_export':ne,'dynamic_export':ce}
    nli,candidate,parity=load_heads_and_check(ref)
    if file_sha(d/'export/frozen_candidate_head.safetensors') != file_sha(c/'dynamic_export/candidate_head.safetensors'):
        raise IntegrityError('Candidate head differs between phases')
    models={name:NoneModel(**read(d,f'export/none_{name}.json')) for name in ('frozen_global','refit_global','set_linear')}
    fixtures=read(d,'export/golden_none_inputs.json')
    for row in fixtures:
        got=models['set_linear'].logits([row['candidate_scores']])[0]
        if not np.allclose(got,row['all_logits'],rtol=0,atol=1e-10): raise IntegrityError('None head fixture mismatch')
    report={'source_c_sha256':cfg.expected_c_sha,'source_d_sha256':cfg.expected_d_sha,
            'source_c_run':'20260917T222948Z','source_d_run':'20260917T234417Z',
            'head_fixture_parity':parity,'none_fixtures_passed':len(fixtures),
            'heads_frozen':True,'source_paths_read_only':True}
    json_write(Path(out)/'source_audit.json',report)
    return dict(c=c,d=d,nli=nli,candidate=candidate,none_models=models,
                data_c=read(c,'dynamic_data_audit.json'),episodes_c=read(c,'dynamic_episodes.json'),
                episodes_d=read(d,'episodes.json'),nli_manifest=read(c,'nli_split_manifest.json'),
                numerical=read(d,'numerical_inputs.json'),report=report)

def fresh_banking_episodes(raw, ref, cfg):
    """Fresh development/test messages; model weights stay fixed. Paired conditions per message."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    names=list(raw['train'].features['label'].names)
    if names != ref['data_c']['class_names']: raise IntegrityError('Banking77 class order changed')
    seen=ref['data_c']['training_label_ids']; unseen=ref['data_c']['heldout_label_ids']
    blocked={e['group'] for source in (ref['episodes_c'],ref['episodes_d']) for es in source.values() for e in es}
    used=set(blocked); all_eps={}; groups={}
    similarity=np.zeros((77,77))
    for pool in (seen,unseen):
        v=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5)).fit_transform([names[i].replace('_',' ') for i in pool])
        similarity[np.ix_(pool,pool)]=(v@v.T).toarray()
    for j,split in enumerate(('dev','test_seen','test_unseen')):
        pool=unseen if split=='test_unseen' else seen
        rows=choose_banking(raw,'test' if split.startswith('test') else 'train',cfg.sizes()[split],set(pool),used,cfg.seed+100+j)
        groups[split]={r['group'] for r in rows}; eps=[]
        for row in rows:
            rng=np.random.default_rng(cfg.seed+int(row['group'][:12],16)); negatives=[i for i in pool if i!=row['label']]
            u=list(map(int,rng.permutation(negatives))); ties={i:float(rng.random()) for i in negatives}
            h=sorted(negatives,key=lambda i:(-similarity[row['label'],i],ties[i]))
            for sampler,ordered in [('uniform',u),('label_lexical_hard',h)]:
                for k in cfg.candidate_counts:
                    for absent in (False,True):
                        ids=ordered[:k] if absent else [row['label']]+ordered[:k-1]
                        ids=list(map(int,rng.permutation(ids)))
                        eps.append(make_episode(row,names,ids,absent,k,sampler))
        all_eps[split]=eps
    for a,ga in groups.items():
        if ga&blocked:raise IntegrityError('Previously used message leaked into new evaluation')
        for b,gb in groups.items():
            if a!=b and ga&gb:raise IntegrityError('New split overlap')
    audit={'previous_messages_excluded':len(blocked),'split_counts':{s:{'messages':len(groups[s]),'episodes':len(es)} for s,es in all_eps.items()},
       'test_none_rate':.5,'candidate_counts':cfg.candidate_counts,'samplers':['uniform','label_lexical_hard'],
       'prior_scenarios':cfg.absent_priors,'source_selection':'fresh relative to ALL C and D messages; repeat seed reuses E messages',
       'holdout_scope':'57/20 head-training label split retained; one domain; Qwen pretraining contamination unknown',
       'policy_fit_scope':'only model/acceptance-threshold selection on new dev; no neural/none weights refitted'}
    return all_eps,audit

def select_complete_pairs(episodes,n):
    """A bounded suite, prioritizing both hard/absent and uniform/present, without using predictions."""
    queues={}
    for e in episodes:
        key=(e['sampler'],bool(e['true_intent_omitted']),len(e['choices']))
        queues.setdefault(key,[]).append(e)
    out=[]
    keys=sorted(queues)
    while len(out)<n and any(queues.values()):
        for key in keys:
            if queues[key]:out.append(queues[key].pop(0))
            if len(out)==n:break
    return out
