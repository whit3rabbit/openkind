"""CPU coordinator: provenance checks, fresh labels, isolated workers, incremental Drive reports."""
from __future__ import annotations
import copy, hashlib, json, os, shutil, subprocess, sys, zipfile
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import asdict
from collections import Counter
import numpy as np
from phase2g_config import Settings, VERSION, F_RUN, F_SHA
from phase2g_data import load_sources,prepare_fresh,select_message_panel,context_variants,read_custom
from phase2g_checks import lifecycle_selftest
from phase2g_metrics import parity,summarize_parity
from phase2d_common import file_sha,json_write,content_hash
from phase2e_core import safe_extract,IntegrityError
from phase2e_expansion import redact


class Experiment:
    def __init__(self,cfg):
        cfg.sizes();self.cfg=cfg;self.run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        self.out=Path(cfg.output_root)/self.run_id;self.out.mkdir(parents=True,exist_ok=False)
        self.prepared=False
        self.report={'schema':'opendecision-phase2g-summary/v1','version':VERSION,'run_id':self.run_id,'configuration':asdict(cfg),
          'stage_status':{},'workers':{},'sources':{},'limitations':[
          'No neural, none-head or policy training. Existing heads and thresholds are frozen.',
          'Fresh means absent from the recorded C/D/E project message manifests, not absent from Qwen pretraining or semantically deduplicated.',
          'Banking and CLINC author test labels are preserved. Candidate sets are sampled, not full 77/150-way benchmarks.',
          'CLINC non-financial domains and author OOS are distinct transfer panels; banking policies are not automatically calibrated there.',
          'Controlled context wraps a real labeled request in generated administrative notes. It is not a natural long-document benchmark.',
          'TF32-allowed keeps FP32 tensor storage; it is an arithmetic permission, not a guaranteed speedup, kernel choice or memory reduction.',
          'Numerical gates compare same-precision alternatives and cross-mode references separately. They do not certify semantic correctness.',
          'Cold request timings exclude offline score memoization. Trace timestamps are simulated arrival times; service timings are measured wall times.',
          'Cache lifecycle CPU tests do not prove concurrent-reader, GPU-cancellation, authorization, HTTP or Rust/Metal behavior.',
          'Prior failed low-bit snapshot configurations are historical results, not silently tuned on this new test.',
          'No new independent-question shared-state contract, Noul/Score training, MTP generation or model-size comparison in this notebook.']}
        self.report['code_sha256']={p.name:file_sha(p) for p in Path(__file__).parent.glob('*.py')};self.checkpoint()
    def checkpoint(self):json_write(self.out/'partial_report.json',self.report)
    def progress_to_drive(self):
        if not self.cfg.copy_results_to_drive:return
        dest=Path(self.cfg.drive_destination)/self.run_id;dest.mkdir(parents=True,exist_ok=True)
        shutil.copy2(self.out/'partial_report.json',dest/'progress_summary.json')
        for mode in self.report['workers']:
            p=self.out/mode/'worker_report.json'
            if p.exists():shutil.copy2(p,dest/(mode+'_progress.json'))
    def prepare(self,source_override=None,tokenizer_override=None):
        import torch
        if torch.cuda.is_initialized():raise RuntimeError('Restart Colab: coordinator must not own CUDA state.')
        root=safe_extract(self.cfg.reference_archive,Path(self.cfg.cache_root)/'f_reference',F_SHA)
        old=json.loads((root/'opendecision_phase2f_summary.json').read_text());inputs=json.loads((root/'experiment_inputs.json').read_text())
        if old['run_id']!=F_RUN or old['run_status']!='completed':raise IntegrityError('A completed Phase 2F reference is required.')
        if content_hash(inputs['policies'])!=inputs['policy_sha256']:raise IntegrityError('Frozen policies changed.')
        source=root/'runtime_source';manifest=json.loads((source/'export/manifest.json').read_text())
        if manifest['candidate_head_sha256']!=file_sha(source/'export/frozen_candidate_head.safetensors'):raise IntegrityError('Frozen candidate weights changed.')
        if manifest['model_revision']!=self.cfg.model_revision:raise IntegrityError('Checkpoint revision mismatch.')
        self.runtime_source=self.out/'runtime_source';shutil.copytree(source,self.runtime_source)
        self.report['source']={'run_id':F_RUN,'archive_sha256':F_SHA,'policy_sha256':inputs['policy_sha256'],'candidate_head_sha256':manifest['candidate_head_sha256'],'source_unchanged':True}
        from phase2d_decisions import NoneModel
        from phase2e_cache import distribution
        none=NoneModel(**json.loads((source/'export/none_set_linear.json').read_text()));gold=json.loads((source/'export/golden_none_inputs.json').read_text())
        delta=max(float(np.max(np.abs(distribution(g['candidate_scores'],none)-np.asarray(g['probabilities'])))) for g in gold)
        if delta>1e-10:raise IntegrityError('Archived head algebra fixtures failed.')
        self.report['head_fixture_replay']={'examples':len(gold),'max_abs_probability_delta':delta,'scope':'head algebra only'}
        self.report['lifecycle_contracts']=lifecycle_selftest()
        assets=Path(__file__).parent.parent/'assets'
        # Test override is explicit and permanently marks every resulting report non-Qwen/non-production.
        if source_override is not None:
            data=source_override;self.report['validation_only']=True;self.report['validation_warning']='SYNTHETIC VALIDATION, NOT QWEN RESULTS'
        else:data=load_sources(Path(self.cfg.cache_root)/'public_data',assets/'prior_exclusions.json')
        bank,clinc,oos,domains,old_exclusions,provenance=data
        eps,audit=prepare_fresh(bank,clinc,oos,domains,old_exclusions,self.cfg)
        blocked=set(old_exclusions['normalized_message_hashes'])|{e['group'] for e in eps}
        eps+=read_custom(self.cfg.custom_jsonl,blocked)
        if tokenizer_override is None:
            from transformers import AutoTokenizer
            tok=AutoTokenizer.from_pretrained(str(source/'export/tokenizer'),local_files_only=True,trust_remote_code=False)
        else:tok=tokenizer_override;self.report['validation_only']=True
        contexts=context_variants(eps,tok,self.cfg) if self.cfg.run_labeled_context else []
        panel=select_message_panel(eps,self.cfg.sizes()['parity_per_family'],self.cfg.seed+20)
        # Round-robin families, one request per message; candidate K and answer presence alternate.
        by_family={}
        for ep in eps:by_family.setdefault(ep['family'],{}).setdefault(ep['group'],[]).append(ep)
        benches=[];families=sorted(by_family);round_index=0
        while len(benches)<self.cfg.sizes()['benchmark_messages']:
            changed=False
            for family in families:
                gg=sorted(by_family[family],key=lambda g:content_hash([self.cfg.seed+21,g]))
                if round_index>=len(gg):continue
                options=sorted(by_family[family][gg[round_index]],key=lambda e:e['id'])
                target_k=self.cfg.candidate_counts[round_index%len(self.cfg.candidate_counts)]
                target_absent=bool((families.index(family)+round_index)%2)
                candidates=[e for e in options if len(e['choices'])==target_k and e['true_intent_omitted']==target_absent]
                if not candidates:candidates=[e for e in options if len(e['choices'])==target_k] or options
                benches.append(candidates[0]);changed=True
                if len(benches)==self.cfg.sizes()['benchmark_messages']:break
            if not changed:break
            round_index+=1
        # Preserve all fresh message IDs in export; no training occurs on these evaluation-only rows.
        self.inputs={'episodes':eps,'context_episodes':contexts,'parity_ids':[e['id'] for e in panel],
             'benchmark_ids':[e['id'] for e in benches],'traffic_ids':[e['id'] for e in eps],
             'policies':inputs['policies'],'policy_sha256':inputs['policy_sha256'],'tokenizer_sha256':file_sha(source/'export/tokenizer/tokenizer.json')}
        json_write(self.out/'experiment_inputs.json',self.inputs);json_write(self.out/'fresh_message_manifest.json',{'groups':sorted({e['group'] for e in eps}),'audit':audit})
        self.report['data']=dict(audit,custom_episodes=sum(e['family']=='custom' for e in eps),parity_episodes=len(panel),parity_messages=len({e['group'] for e in panel}),benchmark_episodes=len(benches),controlled_context_episodes=len(contexts),controlled_context_messages=len({e['group'] for e in contexts}))
        self.report['sources'].update(provenance)
        self.report['stage_status']['prepare']='completed';self.prepared=True;self.checkpoint();self.progress_to_drive()
        print(json.dumps(self.report['data'],indent=2));return self.report['data']
    def run_mode(self,mode):
        import torch
        if not self.prepared:raise RuntimeError('Prepare the experiment first.')
        if torch.cuda.is_initialized():raise RuntimeError('Restart: coordinator already owns CUDA.')
        if mode not in self.cfg.modes:raise ValueError(mode)
        out=self.out/mode
        if out.exists():raise RuntimeError('Worker directory exists; create a new experiment rather than mixing outputs.')
        out.mkdir();job={'configuration':asdict(self.cfg),'out':str(out),'mode':mode,'inputs':str(self.out/'experiment_inputs.json'),
              'source_root':str(self.runtime_source),'policies':self.inputs['policies'],'tokenizer_sha256':self.inputs['tokenizer_sha256']}
        json_write(out/'job.json',job);env=dict(os.environ);env['PYTHONUNBUFFERED']='1';env['TOKENIZERS_PARALLELISM']='false'
        token=env.get('HF_TOKEN')
        if not token:
            try:
                from google.colab import userdata
                token=userdata.get('HF_TOKEN')
            except Exception:token=None
        if token:env['HF_TOKEN']=token
        env['PYTHONPATH']=str(Path(__file__).parent)+os.pathsep+env.get('PYTHONPATH','')
        self.report['stage_status'][mode]='running';self.checkpoint();self.progress_to_drive()
        proc=subprocess.Popen([sys.executable,'-u','-m','phase2g_worker','--job',str(out/'job.json')],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=env,bufsize=1)
        try:
            with (out/'worker.log').open('w') as log:
                for line in proc.stdout:
                    line=redact(line,token);log.write(line);log.flush();print(line,end='',flush=True)
            rc=proc.wait()
        except BaseException:
            proc.terminate()
            try:proc.wait(timeout=15)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
            self.report['stage_status'][mode]='interrupted';self.checkpoint();self.progress_to_drive();raise
        p=out/'worker_report.json';r=json.loads(p.read_text()) if p.exists() else {'status':'failed','error':'worker exited before writing report'}
        r['exit_code']=rc
        if rc and r['status']=='completed':r['status']='failed'
        self.report['workers'][mode]=r;self.report['stage_status'][mode]=r['status'];self.checkpoint();self.progress_to_drive()
        print(f'\n{mode}: {r["status"]}. Successful execution does not imply parity or semantic acceptance.')
        return r
    def finish(self):
        from phase2g_report import finish
        return finish(self)
