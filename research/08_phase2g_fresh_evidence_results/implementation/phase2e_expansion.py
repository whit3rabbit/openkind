"""Phase 2E 2e.2.0: CPU coordinator, immutable inputs and isolated GPU workers.
No model is loaded, CUDA initialized, or network request made at import time.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import hashlib, json, os, re, shutil, subprocess, sys
import numpy as np
from phase2e_core import Settings as LegacySettings, IntegrityError, safe_extract, read_sources
from phase2d_common import json_write, file_sha, content_hash

VERSION = '2e.2.0'
PRIOR_E_RUN = '20260918T032049180933Z'
PRIOR_E_SHA = 'a264e5f8fcee2c0eb8dcb7e1cba48e655ab917e5fae8a7d623857db671716631'

@dataclass
class ExpandedSettings(LegacySettings):
    reference_e: str = ''
    expected_e_sha: str = PRIOR_E_SHA
    output_root: str = '/content/openkind_phase2e_expanded'
    cache_root: str = '/content/openkind_phase2e_expanded_cache'
    # Each mode gets a newly executed Python interpreter; never fork a CUDA model.
    isolated_modes: tuple = ('fp32_strict_math', 'bf16_default')
    run_fp32: bool = True
    run_cache_policy_parity: bool = True
    run_component_profile: bool = True
    run_suffix_batch_prototype: bool = True
    run_request_benchmark: bool = True
    run_long_prefix_benchmark: bool = True
    # Prior selective-precision/policy-fit results are imported, not rerun by this focused expansion.
    run_precision: bool = False
    run_policy_evaluation: bool = False
    benchmark_repeats: int = 5
    benchmark_warmups: int = 2
    profile_repeats: int = 3
    suffix_batch_size: int = 4
    fp32_workspace_gib: float = 2.0
    initial_gpu_allocation_limit_mib: float = 64.0
    long_prefix_lengths: tuple = (64, 256, 1024)
    long_prefix_repeats: int = 5
    copy_results_to_drive: bool = True
    diagnostics_seed: int = 83

    def expanded_sizes(self):
        values = {
          'smoke': dict(messages_per_split=1, benchmark_per_stratum=1, profile_per_stratum=1),
          'quick': dict(messages_per_split=2, benchmark_per_stratum=1, profile_per_stratum=1),
          'standard': dict(messages_per_split=4, benchmark_per_stratum=2, profile_per_stratum=1)}
        if self.preset not in values: raise ValueError('preset must be smoke, quick, or standard')
        if tuple(self.candidate_counts) != (2, 4, 8, 16):
            raise ValueError('The balanced expansion retains the archived 2/4/8/16 candidate grid.')
        if self.max_length != 256: raise ValueError('Preserve the trained 256-token prompt contract.')
        if self.suffix_batch_size < 1 or self.suffix_batch_size > 8:
            raise ValueError('suffix_batch_size must be 1..8.')
        for name in ('benchmark_repeats','profile_repeats','long_prefix_repeats'):
            if getattr(self,name) < 1: raise ValueError(f'{name} must be positive')
        if self.benchmark_warmups < 0 or self.fp32_workspace_gib < 1.0:
            raise ValueError('Use nonnegative warmups and at least 1 GiB workspace reserve.')
        allowed = {'fp32_strict_math','bf16_default','bf16_strict_math'}
        if len(set(self.isolated_modes)) != len(self.isolated_modes) or set(self.isolated_modes)-allowed:
            raise ValueError('Duplicate or unsupported isolated mode.')
        return values[self.preset]

    def e_archive_path(self):
        return (Path(self.reference_e) if self.reference_e else Path(self.drive_notebooks)/
                f'OpenKind_Phase2E_results/{PRIOR_E_RUN}/openkind_phase2e_{PRIOR_E_RUN}.zip')


def stable_rank(value, seed):
    return hashlib.sha256(f'{seed}:{value}'.encode()).hexdigest()


def select_diverse_panel(episodes_by_split, cfg):
    """Select groups before episodes. No scores, errors, or test outcomes used for selection."""
    selected=[]; chosen={}; n=cfg.expanded_sizes()['messages_per_split']
    samplers=('uniform','label_lexical_hard')
    expected={(k,s,a) for k in cfg.candidate_counts for s in samplers for a in (False,True)}
    for split in ('test_seen','test_unseen'):
        buckets={}
        for ep in episodes_by_split[split]: buckets.setdefault(ep['group'],[]).append(ep)
        groups=sorted(buckets,key=lambda g:stable_rank(split+g,cfg.diagnostics_seed))[:n]
        if len(groups)!=n: raise IntegrityError(f'Not enough distinct {split} messages.')
        chosen[split]=groups
        for group in groups:
            entries={}
            for ep in buckets[group]:
                key=(len(ep['choices']),ep['sampler'],bool(ep['true_intent_omitted']))
                if key in entries: raise IntegrityError('Duplicate episode in factorial group.')
                entries[key]=ep
            if set(entries)!=expected: raise IntegrityError('Incomplete archived candidate/sampler/presence grid.')
            for key in sorted(expected): selected.append(dict(entries[key],evaluation_split=split))
    if set(chosen['test_seen']) & set(chosen['test_unseen']): raise IntegrityError('Seen/unseen group overlap.')
    ids=[e['id'] for e in selected]
    if len(set(ids))!=len(ids): raise IntegrityError('Duplicate episode ID.')
    return selected, {'selection':'hash-ranked message groups first, then COMPLETE factorial episodes; no outcome-based selection',
        'source':'archived Phase 2E tests: regression/parity, NOT new generalization data',
        'messages_by_split':{k:len(v) for k,v in chosen.items()},'groups_by_split':chosen,
        'episode_count':len(selected),'distinct_messages':len(set(e['group'] for e in selected)),
        'candidate_counts':list(cfg.candidate_counts),'samplers':list(samplers),
        'present_episodes':sum(not e['true_intent_omitted'] for e in selected),
        'absent_episodes':sum(e['true_intent_omitted'] for e in selected),
        'episodes_sha256':content_hash(selected)}


def balanced_subset(panel, per_stratum, seed):
    """Paired presence states; rotate across all selected groups for each K/sampler stratum."""
    groups=sorted({e['group'] for e in panel}, key=lambda g:stable_rank(g,seed))
    lookup={(e['group'],len(e['choices']),e['sampler'],bool(e['true_intent_omitted'])):e for e in panel}
    if not groups or per_stratum<1 or per_stratum>len(groups): raise ValueError('Invalid profile/benchmark panel.')
    pairs=sorted({(len(e['choices']),e['sampler']) for e in panel})
    result=[]
    for j,(k,sampler) in enumerate(pairs):
        for off in range(per_stratum):
            group=groups[(j*per_stratum+off)%len(groups)]
            for absent in (False,True): result.append(lookup[group,k,sampler,absent])
    return result


def redact(text, token=None):
    if token: text=text.replace(token,'[REDACTED]')
    return re.sub(r'hf_[A-Za-z0-9_]+','[REDACTED]',text)


class ExpandedExperiment:
    """Notebook controller remains CPU-only. Every GPU worker terminates before the next starts."""
    def __init__(self,cfg):
        cfg.expanded_sizes(); self.cfg=cfg
        self.run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        self.out=Path(cfg.output_root)/self.run_id;self.out.mkdir(parents=True,exist_ok=False)
        self.prepared=False;self.source=None
        self.report={'schema':'openkind-phase2e-expanded-summary/v1','version':VERSION,
          'run_id':self.run_id,'configuration':asdict(cfg),'stage_status':{},'workers':{},
          'limitations':[
            'All Qwen weights, C/D heads and E acceptance policies are frozen; no refitting or threshold search.',
            'Tests reuse archived E messages and selected policies: execution regression, not independent generalization.',
            'Complete factorial variants share messages; episode counts are not independent sample counts.',
            'FP32 loads in an independent fresh subprocess; no GPU offload and no in-place precision sweep before it.',
            'Fresh-process isolation does not guarantee enough physical GPU memory; preflight and OOM are reported.',
            'Equal-length suffix batching expands all hybrid state with repeated beam indices; no suffix padding.',
            'Profile component times include synchronization and are intrusive; uninstrumented timings are reported separately.',
            'All request timings use a cold prefix; no cross-request cache, Rust/Metal or HTTP benchmark.',
            'A passed numerical tolerance is not a safety guarantee; mode completion is distinct from equivalence acceptance.',
            'Long-prefix mechanics use synthetic token sequences; they do not measure long-context decision quality.']}
        self.report['code_sha256']={p.name:file_sha(p) for p in Path(__file__).parent.glob('*.py')}
        self.checkpoint()

    def checkpoint(self): json_write(self.out/'partial_report.json',self.report)

    def prepare(self):
        import torch
        if torch.cuda.is_initialized():
            raise RuntimeError('Restart the Colab session. The notebook coordinator must not already own a CUDA context/model.')
        from phase2e_runtime import indexing_selftest
        self.report['indexing_selftest']=indexing_selftest()
        self.ref=read_sources(self.cfg,self.out)
        root=safe_extract(self.cfg.e_archive_path(),Path(self.cfg.cache_root)/('source_e_'+self.cfg.expected_e_sha[:12]),self.cfg.expected_e_sha)
        old=json.loads((root/'openkind_phase2e_summary.json').read_text())
        if old['run_id']!=PRIOR_E_RUN or old['version']!='2e.1.1': raise IntegrityError('Wrong Phase 2E source run.')
        if old['model']['revision']!=self.cfg.model_revision: raise IntegrityError('Checkpoint identity differs.')
        for name,cpath in [('frozen_candidate_head.safetensors',self.ref['c']/'dynamic_export/candidate_head.safetensors'),
                           ('nli_head.safetensors',self.ref['c']/'frozen_nli_export/head.safetensors')]:
            # NLI export filename can differ; actual tensor fixtures are replayed below.
            if file_sha(root/'export'/name)!=file_sha(cpath):
                raise IntegrityError('E candidate weights differ from verified C/D head.')
        for name in ('frozen_global','refit_global','set_linear'):
            if file_sha(root/f'export/none_{name}.json') != file_sha(self.ref['d']/f'export/none_{name}.json'):
                raise IntegrityError('E none coefficients differ from verified D export.')
        policy=json.loads((root/'policy_report.json').read_text())['selections']
        if len(policy)!=9: raise IntegrityError('Expected the nine archived policy scenarios.')
        for p in policy:
            if p['none_head'] not in self.ref['none_models'] or not 0<=p['acceptance_threshold']<=1.000001:
                raise IntegrityError('Invalid archived policy.')
        episodes=json.loads((root/'episodes.json').read_text())
        panel,audit=select_diverse_panel(episodes,self.cfg)
        benchmark=balanced_subset(panel,self.cfg.expanded_sizes()['benchmark_per_stratum'],self.cfg.diagnostics_seed+1)
        profile=balanced_subset(panel,self.cfg.expanded_sizes()['profile_per_stratum'],self.cfg.diagnostics_seed+2)
        self.source=root
        payload={'panel':panel,'benchmark_ids':[e['id'] for e in benchmark],
          'profile_ids':[e['id'] for e in profile],'policies':policy,'source_root':str(root),
          'source_c_root':str(self.ref['c']),'source_d_root':str(self.ref['d']),
          'policy_sha256':content_hash(policy),'panel_audit':audit}
        json_write(self.out/'experiment_inputs.json',payload)
        self.report['source']={'e_run_id':PRIOR_E_RUN,'e_archive_sha256':self.cfg.expected_e_sha,
          'c_archive_sha256':self.cfg.expected_c_sha,'d_archive_sha256':self.cfg.expected_d_sha,
          'source_not_modified':True,'heads_and_policies_frozen':True,
          'policy_sha256':content_hash(policy),'head_fixtures':self.ref['report']['head_fixture_parity']}
        self.report['sampling']=dict(audit,benchmark_episodes=len(benchmark),profile_episodes=len(profile),
          benchmark_distinct_messages=len({e['group'] for e in benchmark}),
          profile_distinct_messages=len({e['group'] for e in profile}))
        self.report['historical_results']={'source_only_not_rerun':True,
          'precision':old.get('precision',{}),'policy_selections':policy}
        # Summary avoids embedding all historical results; full report retains this context.
        self.report['stage_status']['prepare']='completed';self.prepared=True;self.checkpoint()
        print(json.dumps(self.report['sampling'],indent=2),flush=True)
        print('Saved policies frozen. No test-driven threshold selection. GPU not initialized by coordinator.',flush=True)
        return audit

    def run_mode(self, mode):
        import torch
        if not self.prepared: raise RuntimeError('Run prepare() first.')
        if torch.cuda.is_initialized(): raise RuntimeError('Coordinator owns CUDA; restart session before isolated workers.')
        if mode not in self.cfg.isolated_modes: raise ValueError('Mode not enabled in isolated_modes.')
        if mode=='fp32_strict_math' and not self.cfg.run_fp32:
            self.report['workers'][mode]={'status':'disabled'};self.checkpoint();return
        jobdir=self.out/mode
        if jobdir.exists(): raise RuntimeError('This mode already has a run folder. Create a new experiment, not a mixed rerun.')
        jobdir.mkdir()
        job={'configuration':asdict(self.cfg),'mode':mode,'out':str(jobdir),
             'inputs':str(self.out/'experiment_inputs.json'),'version':VERSION}
        json_write(jobdir/'job.json',job)
        env=dict(os.environ);env['PYTHONUNBUFFERED']='1'
        # Token is passed in the child environment only, never in argv or saved JSON.
        token=env.get('HF_TOKEN')
        if not token:
            try:
                from google.colab import userdata
                token=userdata.get('HF_TOKEN')
            except Exception: token=None
        if token: env['HF_TOKEN']=token
        env['PYTHONPATH']=str(Path(__file__).parent)+os.pathsep+env.get('PYTHONPATH','')
        cmd=[sys.executable,'-u','-m','phase2e_expand_worker','--job',str(jobdir/'job.json')]
        self.report['stage_status'][mode]='running';self.checkpoint()
        proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=env,bufsize=1)
        try:
            with (jobdir/'worker.log').open('w') as log:
                for line in proc.stdout:
                    line=redact(line,token);log.write(line);log.flush();print(line,end='',flush=True)
            code=proc.wait()
        except BaseException:
            proc.terminate()
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
            self.report['stage_status'][mode]='interrupted';self.checkpoint();raise
        resultfile=jobdir/'worker_report.json'
        result=json.loads(resultfile.read_text()) if resultfile.exists() else {'status':'failed','error':'Worker exited before report.'}
        result['exit_code']=code
        if code!=0 and result.get('status')=='completed':result['status']='failed'
        self.report['workers'][mode]=result
        self.report['stage_status'][mode]=result.get('status','failed');self.checkpoint()
        print(f'Worker {mode} exited ({code}); its model/context are no longer resident.',flush=True)
        return result

    def finish(self):
        from phase2e_expand_reporting import build_report
        result=build_report(self.report,self.out,self.cfg)
        self.report=result;self.checkpoint()
        json_write(self.out/'openkind_phase2e_expanded_summary.json',result)
        compact={k:v for k,v in result.items() if k not in ('historical_results',)}
        compact['workers']={k:{kk:vv for kk,vv in v.items() if kk not in ('parity_rows','profile_rows','benchmark_rows','memory_snapshots')}
                            for k,v in result.get('workers',{}).items()}
        json_write(self.out/'paste_back_summary.json',compact)
        if self.source is not None:
            shutil.copytree(self.source/'export',self.out/'frozen_export',dirs_exist_ok=True)
        source_dir=self.out/'implementation';source_dir.mkdir(exist_ok=True)
        for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,source_dir/p.name)
        archive=Path(shutil.make_archive(str(self.out.parent/f'openkind_phase2e_expanded_{self.run_id}'),'zip',self.out))
        if self.cfg.copy_results_to_drive:
            dest=Path(self.cfg.drive_notebooks)/'OpenKind_Phase2E_expanded_results'/self.run_id
            dest.mkdir(parents=True,exist_ok=False)
            for p in [archive,self.out/'paste_back_summary.json',self.out/'openkind_phase2e_expanded_summary.json',self.out/'README_results.md']:
                shutil.copy2(p,dest/p.name)
            print('Copied reports/archive to:',dest)
        print('Local results:',self.out,'\nResult archive:',archive)
        print('===== BEGIN_OPENKIND_PHASE2E_EXPANDED_SUMMARY =====')
        print(json.dumps(compact,indent=2,allow_nan=False))
        print('===== END_OPENKIND_PHASE2E_EXPANDED_SUMMARY =====')
        return self.out,archive
