"""CPU coordinator with pre-final artifact lock, exclusion registry, and isolated CUDA workers."""
import copy,hashlib,json,os,shutil,subprocess,sys,time,zipfile
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
from collections import Counter
import numpy as np
from phase2h_config import Settings,VERSION,G_SHA,G_RUN
from phase2h_data import *
from phase2h_learning import parity
from phase2d_common import file_sha,json_write,content_hash
from phase2e_core import safe_extract
from phase2e_expansion import redact

def copy_report_file(source, destination):
    """Publish one report; recreate its parent at the actual copy boundary.

    Bounded retries cover a disappearing destination on a mounted filesystem.
    Missing source files and permission errors are never treated as success.
    """
    source, destination = Path(source), Path(destination)
    if not source.is_file():
        raise FileNotFoundError(f"Local artifact is missing: {source}")
    for attempt in range(3):
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            if file_sha(source) != file_sha(destination):
                raise OSError(f"Copied artifact checksum mismatch: {destination}")
            return destination
        except FileNotFoundError:
            if not source.is_file() or attempt == 2:
                raise
            time.sleep(0.25 * (attempt + 1))


class Experiment:
    def __init__(self,cfg):
        cfg.sizes();self.cfg=cfg;self.run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');self.out=Path(cfg.output_root)/self.run_id;self.out.mkdir(parents=True,exist_ok=False)
        self.report={'schema':'openkind-phase2h-summary/v1','version':VERSION,'run_id':self.run_id,'configuration':asdict(cfg),'status':'initialized','stages':{},'workers':{},'limitations':[
            'Criteria support examples are author-training data, not independently reviewed definitions. Review attestations are not independently verified by code.',
            'Final evaluation is read only after fitting and artifact lock. This is a workflow guard, not a security boundary against manually opening local files.',
            'Exact normalized-message exclusion is not paraphrase or Qwen pretraining decontamination.',
            'Omitted in-scope labels, author OOS, and application review are reported separately.',
            'Training population weights and policy error costs are preregistered assumptions, not deployment measurements.',
            'None heads, normalizers, calibration and policy thresholds use distinct named source-training partitions; final output cannot refit them.',
            'All multiple candidate, context and question variants remain clustered under their original message group.',
            'Finite-token baseline is unadapted vocabulary scoring plus held-out calibration; it is not an equal-gradient-budget SALSA reproduction.',
            'BoolQ and SST-5 are bounded Noul/Score tests; no arbitrary yes/no, insufficient-evidence or ordinal-rubric validation follows.',
            'Natural-document evidence requires user-supplied independent labels; controlled context wrappers are kept separate.',
            'State-first rendering is a new input/model contract; related-query cache mechanics do not establish arbitrary-question semantics.',
            'LoRA uses a separate matched online-head BCE control, not an unqualified comparison to joint-CE training.',
            'No Rust/Metal, real concurrent serving, HTTP or cross-platform latency has been measured by this notebook.'],
            'code_sha256':{p.name:file_sha(p) for p in Path(__file__).parent.glob('*.py')}}
        json_write(self.out/'config.json',asdict(cfg));self.save()
    def save(self):
        json_write(self.out/'progress_summary.json',self.report)
        if self.cfg.copy_results_to_drive:
            dest=Path(self.cfg.drive_destination)/self.run_id;dest.mkdir(parents=True,exist_ok=True);copy_report_file(self.out/'progress_summary.json',dest/'progress_summary.json')
    def prepare(self,source_override=None,tokenizer_override=None):
        import torch
        if torch.cuda.is_initialized():raise RuntimeError('Restart: coordinator must not own a CUDA context')
        if (self.out/'prepare_checkpoint.json').is_file():
            return self.resume_prepare_export()
        root=safe_extract(self.cfg.reference_archive,Path(self.cfg.cache_root)/'reference_g',G_SHA)
        prior=json.loads((root/'openkind_phase2g_summary.json').read_text())
        if prior['run_id']!=G_RUN or prior['run_status']!='completed':raise ValueError('Completed G reference required')
        self.source=self.out/'runtime_source';shutil.copytree(root/'runtime_source',self.source)
        prior_input=json.loads((root/'experiment_inputs.json').read_text());old=json.loads((root/'prior_exclusions.json').read_text());prior_groups=set(old['normalized_message_hashes'])|set(json.loads((root/'fresh_message_manifest.json').read_text())['groups'])
        manifest=json.loads((self.source/'export/manifest.json').read_text())
        if file_sha(self.source/'export/frozen_candidate_head.safetensors')!=manifest['candidate_head_sha256']:raise ValueError('Prior head hash mismatch')
        from phase2h_learning import add_none,softmaxes
        golden=json.loads((self.source/'export/golden_none_inputs.json').read_text())
        none=json.loads((self.source/'export/none_set_linear.json').read_text())
        largest=0.
        for fixture in golden:
            got=softmaxes(add_none(none,[np.asarray(fixture['candidate_scores'],float)]))[0]
            largest=max(largest,float(np.max(np.abs(got-np.asarray(fixture['probabilities'])))))
        if largest>1e-10:raise ValueError('Historical none-head fixture mismatch')
        self.report['source_fixture_replay']={'none_fixtures':len(golden),'maximum_probability_residual':largest,'scope':'saved CPU head algebra only, not Qwen/CUDA'}
        self.report['reference']={'run_id':G_RUN,'archive_sha256':G_SHA,'candidate_head_sha256':manifest['candidate_head_sha256'],'historical_outputs_unchanged':True}
        if tokenizer_override is None:
            from transformers import AutoTokenizer
            tokenizer=AutoTokenizer.from_pretrained(str(self.source/'export/tokenizer'),local_files_only=True,trust_remote_code=False)
        else:tokenizer=tokenizer_override;self.report['validation_only']=True
        public=source_override or load_public(Path(self.cfg.cache_root)/'public_data')
        if source_override is not None:self.report['validation_only']=True
        banks,clinc,domains,provenance=public
        self.specs=[{'name':'qwen4b','model_id':'Qwen/Qwen3.5-4B-Base','revision':'1001bb4d826a52d1f399e183466143f4da7b741b','parameter_estimate':4205751296}]
        if self.cfg.run_smaller_model:
            from huggingface_hub import HfApi
            info=HfApi().model_info(self.cfg.smaller_model_id)
            if not info.sha or len(info.sha)!=40:raise ValueError('Smaller-model revision not resolved')
            self.specs.append({'name':'qwen2b','model_id':self.cfg.smaller_model_id,'revision':info.sha,'parameter_estimate':2500000000})
        registry=Path(self.cfg.registry_path);registry.parent.mkdir(parents=True,exist_ok=True)
        # Local-process lock, not a distributed lock across Colab machines.
        # Drive FUSE need not support flock, so lock locally and detect observed registry changes.
        import fcntl,tempfile
        lock_path=Path(tempfile.gettempdir())/('openkind_h_'+content_hash(str(registry))+'.lock')
        with lock_path.open('w') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX)
            registry_before=file_sha(registry) if registry.exists() else None
            saved=json.loads(registry.read_text()) if registry.exists() else {'runs':{},'reserved_groups':[]}
            previous=prior_groups|set(saved['reserved_groups'])
            splits,support,pools,audit=build_splits(banks,clinc,domains,old,self.cfg,previous)
            library,template,criteria_audit=criteria_library(support,pools,self.cfg.criteria_review_json)
            episodes={s:make_episodes(rows,pools,self.cfg,s) for s,rows in splits.items()}
            allgroups={r['group'] for rr in splits.values() for r in rr}|{r['group'] for r in support};blocked=previous|allgroups
            primitives={};primitive_sources={}
            if self.cfg.run_primitives:
                from phase2h_primitives import prepare_primitives
                primitives,primitive_sources=prepare_primitives(self.cfg,blocked,tokenizer)
                allgroups|={r['group'] for task in primitives.values() for rr in task.values() for r in rr}
            natural=read_natural(self.cfg.natural_documents_jsonl,blocked,self.cfg);allgroups|={r['group'] for r in natural}
            # Reject too-long content BEFORE any model treatment runs; do not give arms different truncated evidence.
            from phase2h_runtime import encode,finite_prompt
            for s,eps in episodes.items():
                for e in eps:
                    for variant in library:
                        v=apply_criteria(e,library,variant)
                        for c in v['choices']:
                            encode(v,c,tokenizer,self.cfg)
                            if self.cfg.run_state_first:encode(v,c,tokenizer,self.cfg,'state_first')
                        if self.cfg.run_finite_token:finite_prompt(v,tokenizer,self.cfg)
            if (file_sha(registry) if registry.exists() else None)!=registry_before:
                raise RuntimeError('Reservation registry changed during preparation. Do not run multiple H preparation sessions concurrently.')
            saved['reserved_groups']=sorted(set(saved['reserved_groups'])|allgroups);saved['runs'][self.run_id]={'groups':sorted(allgroups),'config_sha256':content_hash(asdict(self.cfg)),'status':'reserved_before_fit'};json_write(registry,saved)
            fcntl.flock(handle,fcntl.LOCK_UN)
        self.report['data']=dict(audit,new_total_reserved_groups=len(allgroups),episode_counts={s:len(v) for s,v in episodes.items()},criteria_audit=criteria_audit,primitive_sources=primitive_sources,natural_documents=len(natural),registry_scope='all new reserved groups excluded from later 2H studies; single-coordinator workflow, not distributed reservation safety')
        self.report['sources']=provenance;self.report['model_specs']=self.specs
        self.fit_payload={'episodes':{s:e for s,e in episodes.items() if s!='final'},'criteria':library,'criteria_audit':criteria_audit,'primitives':{task:{s:v for s,v in rows.items() if s!='final'} for task,rows in primitives.items()}}
        self.final_payload={'episodes':episodes['final'],'criteria':library,'criteria_audit':criteria_audit,'primitives':{task:rows['final'] for task,rows in primitives.items()},'natural_documents':natural}
        json_write(self.out/'fit_payload.json',self.fit_payload);json_write(self.out/'final_payload_reserved.json',self.final_payload);json_write(self.out/'criteria_review_template.json',template)
        json_write(self.out/'split_manifest.json',{s:[{'group':r['group'],'family':r['family'],'source_split':r['source_split'],'source_index':r['source_index']} for r in rows] for s,rows in splits.items()})
        json_write(self.out/'criteria_support.json',support);json_write(self.out/'frozen_criteria.json',library)
        # Pair review order depends only on label names, never model test errors.
        pairs=[]
        for family,pool in pools.items():
            similarity=label_similarity(pool)
            for i,label in enumerate(pool):
                j=int(np.argsort(similarity[i])[-2]);pairs.append({'family':family,'label_a':label,'label_b':pool[j],'lexical_similarity':float(similarity[i,j]),'review':'pending','gold_labels_changed':False})
        json_write(self.out/'criteria_pair_review_queue.json',pairs)
        # Preserve the prepared study locally BEFORE attempting any Drive copy.
        self.report['stages']['prepare']='data_prepared_export_pending'
        self.report['status']='data_prepared_export_pending'
        prepared_names=('fit_payload.json','final_payload_reserved.json','criteria_review_template.json',
                        'criteria_pair_review_queue.json','split_manifest.json','frozen_criteria.json','criteria_support.json')
        json_write(self.out/'prepare_checkpoint.json',{
            'run_id':self.run_id,'configuration_sha256':content_hash(asdict(self.cfg)),
            'files':{name:file_sha(self.out/name) for name in prepared_names},'report':self.report})
        json_write(self.out/'progress_summary.json',self.report)
        return self.resume_prepare_export()
    def resume_prepare_export(self):
        # Complete only the export tail; never redraw splits or edit reservations.
        if self.report.get('workers'):
            raise RuntimeError('Model workers already started; do not rerun preparation.')
        checkpoint_path=self.out/'prepare_checkpoint.json'
        checkpoint=json.loads(checkpoint_path.read_text()) if checkpoint_path.is_file() else None
        if checkpoint is not None:
            if checkpoint['run_id']!=self.run_id or checkpoint['configuration_sha256']!=content_hash(asdict(self.cfg)):
                raise ValueError('Prepared run/configuration changed; refusing to resume with different settings.')
            for name,expected in checkpoint['files'].items():
                if file_sha(self.out/name)!=expected:
                    raise ValueError('Prepared artifact changed: '+name)
            self.report=copy.deepcopy(checkpoint['report'])
        elif 'data' not in self.report:
            raise RuntimeError('No prepared data/checkpoint available; export-only recovery cannot continue.')
        names=('fit_payload.json','final_payload_reserved.json','criteria_review_template.json',
               'criteria_pair_review_queue.json','split_manifest.json','frozen_criteria.json','criteria_support.json')
        for name in names:
            if not (self.out/name).is_file():raise FileNotFoundError('Missing prepared artifact: '+name)
        # Validate identity without rewriting the persistent reservation registry.
        registry=json.loads(Path(self.cfg.registry_path).read_text())
        reservation=registry.get('runs',{}).get(self.run_id)
        if reservation is None or reservation['config_sha256']!=content_hash(asdict(self.cfg)):
            raise ValueError('Original reservation/configuration not found; do not create a replacement study.')
        self.source=self.out/'runtime_source'
        if not self.source.is_dir():raise FileNotFoundError('Prepared runtime_source is missing.')
        self.specs=self.report['model_specs']
        # Loading already-written payloads does not display labels, refit, or evaluate final data.
        self.fit_payload=json.loads((self.out/'fit_payload.json').read_text())
        self.final_payload=json.loads((self.out/'final_payload_reserved.json').read_text())
        if self.cfg.copy_results_to_drive:
            dest=Path(self.cfg.drive_destination)/self.run_id
            for name in ('criteria_review_template.json','criteria_pair_review_queue.json','split_manifest.json','frozen_criteria.json'):
                copy_report_file(self.out/name,dest/name)
        self.report['stages']['prepare']='completed';self.report['status']='prepared';self.save()
        print(json.dumps(self.report['data'],indent=2))
        print('Criteria-review status:',self.report['data']['criteria_audit']['independent_review'])
        return self.report['data']
    def worker(self,label,stage,spec,mode='fp32_strict_math',fitroot=None,extra=None):
        import torch
        if torch.cuda.is_initialized():raise RuntimeError('Restart: coordinator CUDA state violates worker isolation')
        out=self.out/label
        if out.exists():raise RuntimeError('Worker directory exists; do not mix repeated outputs. Resume feature extraction with its saved job or use a new study.')
        out.mkdir();job={'stage':stage,'spec':spec,'mode':mode,'configuration':asdict(self.cfg),'out':str(out),'source':str(self.source),'payload':str(self.out/('fit_payload.json' if stage=='fit' else 'final_payload_reserved.json'))}
        if fitroot:job['fitroot']=str(fitroot)
        if stage=='final':job['lock']=str(self.out/'final_lock.json')
        if extra:job.update(extra)
        json_write(out/'job.json',job);env=dict(os.environ);env['PYTHONPATH']=str(Path(__file__).parent)+os.pathsep+env.get('PYTHONPATH','');env['PYTHONUNBUFFERED']='1';env['TOKENIZERS_PARALLELISM']='false'
        token=env.get('HF_TOKEN')
        if not token:
            try:
                from google.colab import userdata
                token=userdata.get('HF_TOKEN')
            except Exception:token=None
        if token:env['HF_TOKEN']=token
        self.report['stages'][label]='running';self.save()
        process=subprocess.Popen([sys.executable,'-u','-m','phase2h_worker','--job',str(out/'job.json')],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        try:
            with (out/'worker.log').open('w') as log:
                for line in process.stdout:
                    line=redact(line,token);log.write(line);log.flush();print(line,end='',flush=True)
                rc=process.wait()
        except BaseException:
            process.terminate()
            try:process.wait(timeout=15)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            self.report['stages'][label]='interrupted';self.save();raise
        record=json.loads((out/'worker_report.json').read_text()) if (out/'worker_report.json').exists() else {'status':'failed','error':'Worker exited without a report'}
        self.report['stages'][label]=record['status'];self.report['workers'][label]={'status':record['status'],'exit_code':rc,'report_path':str(out/'worker_report.json'),'error':record.get('error')};self.save();return record
    def fit(self):
        self.fit_jobs=[]
        for spec in self.specs:
            label='fit_'+spec['name'];r=self.worker(label,'fit',spec)
            if r['status']=='completed':self.fit_jobs.append({'label':label,'spec':spec,'extra':{}})
        if not any(x['spec']['name']=='qwen4b' for x in self.fit_jobs):raise RuntimeError('The 4B baseline must complete before additional treatments/final evaluation')
        if self.cfg.run_lora:
            for seed in self.cfg.head_seeds:
                for kind in ('online_head','lora'):
                    label=f'fit_{kind}_{seed}';extra={'online_kind':kind,'online_seed':seed,'parent_fit':str(self.out/'fit_qwen4b')};r=self.worker(label,'fit',self.specs[0],extra=extra)
                    if r['status']=='completed':self.fit_jobs.append({'label':label,'spec':self.specs[0],'extra':extra})
        json_write(self.out/'fit_jobs.json',self.fit_jobs)
        return self.report['stages']
    def seal(self):
        if (self.out/'final_lock.json').exists():
            from phase2h_checkpoint import verify_final_lock
            verify_final_lock(self.out/'final_lock.json',self.out/'final_payload_reserved.json')
            return
        if self.cfg.require_independent_review_for_final and self.report['data']['criteria_audit']['independent_review']=='not_performed':
            self.report['status']='awaiting_independent_criteria_review';self.save();raise RuntimeError('Reviewed definitions are required by configuration. Fit results are preserved; final remains unopened.')
        records={}
        for job in self.fit_jobs:
            for path in (self.out/job['label']).glob('*'):
                if path.suffix in ('.json','.safetensors') and path.name not in ('worker_report.json','job.json','loading_info.json'):records[str(path)]=file_sha(path)
        lock={'schema':'openkind-final-lock/v1','run_id':self.run_id,'final_payload_sha256':file_sha(self.out/'final_payload_reserved.json'),'fit_payload_sha256':file_sha(self.out/'fit_payload.json'),'criteria_sha256':file_sha(self.out/'frozen_criteria.json'),'fit_files':records,'criteria_review':self.report['data']['criteria_audit'],'created_before_final_predictions':True,'no_security_boundary_claim':True}
        json_write(self.out/'final_lock.json',lock);self.report['final_lock_sha256']=file_sha(self.out/'final_lock.json');self.save()
    def evaluate(self):
        if not self.cfg.run_final:self.report['status']='development_only_final_not_opened';self.save();return
        self.seal()
        for job in self.fit_jobs:
            label='eval_'+job['label'][4:]+'_strict';self.worker(label,'final',job['spec'],fitroot=self.out/job['label'],extra=job['extra'])
        if self.cfg.run_tf32:self.worker('eval_qwen4b_tf32','final',self.specs[0],mode='fp32_tf32_allowed',fitroot=self.out/'fit_qwen4b')
        self.report['final_evaluation_exposed']=True;self.save()
    def finish(self):
        records={}
        for name,w in self.report['workers'].items():
            p=Path(w['report_path']);records[name]=json.loads(p.read_text()) if p.exists() else w
        cross={}
        if 'eval_qwen4b_strict' in records and 'eval_qwen4b_tf32' in records:
            a=self.out/'eval_qwen4b_strict/final_rows.json';b=self.out/'eval_qwen4b_tf32/final_rows.json'
            if a.exists() and b.exists():
                aa=json.loads(a.read_text());bb=json.loads(b.read_text());lookup={(r['profile'],r['id']):r for r in bb};profiles={p['name']:p for p in records['fit_qwen4b']['profiles']};ep={e['id']:e for e in self.final_payload['episodes']}
                for profile in profiles:
                    rr=[r for r in aa if r['profile']==profile and (profile,r['id']) in lookup]
                    cross[profile]=parity([np.array(r['probabilities']) for r in rr],[np.array(lookup[(profile,r['id'])]['probabilities']) for r in rr],[ep[r['id']] for r in rr],self.cfg.probability_tolerance,profiles[profile]['policies'])
        self.report['cross_mode']=cross
        self.report['open_question_status']={
            'criteria':'support-example ablation measured if completed; independent domain review '+self.report.get('data',{}).get('criteria_audit',{}).get('independent_review','not_performed'),
            'rejection_and_multidomain_head':'fitted on new source-training partitions; final status is per worker',
            'robustness':'controlled context and instruction-in-state tests; natural data separately supplied or explicitly absent',
            'tf32':'comparison requested' if self.cfg.run_tf32 else 'disabled',
            'state_first':'related-query mechanics and adapted-head study requested' if self.cfg.run_state_first else 'optional, disabled',
            'finite_token':'one-step conditional code scoring requested' if self.cfg.run_finite_token else 'disabled',
            'smaller_model':'separate refitted-head model-size treatment requested' if self.cfg.run_smaller_model else 'optional, disabled',
            'lora':'matched online-head/LoRA BCE controls requested; memory skips remain explicit' if self.cfg.run_lora else 'optional, disabled',
            'noul_score':'BoolQ/SST-5 supervised domain probes requested' if self.cfg.run_primitives else 'optional, disabled',
            'rust_metal_service':'deferred; export fixtures only, no cross-platform or concurrency evidence'}
        failures=any(w['status']!='completed' for w in self.report['workers'].values())
        if self.cfg.run_final and not self.report.get('final_evaluation_exposed'):
            failures=True
        self.report['status']='partial' if failures else ('completed' if self.report.get('final_evaluation_exposed') else 'development_only')
        full=dict(self.report,worker_results=records);json_write(self.out/'openkind_phase2h_summary.json',full)
        compact=dict(self.report);compact['results']={}
        for name,r in records.items():
            if name.startswith('eval_') and r.get('status')=='completed':
                winner=r['selected_profile'];compact['results'][name]={'selected_before_final':winner,'selected_metrics':r.get('final_metrics',{}).get(winner,{}),'parity':r.get('parity_summary'),'model':r.get('model'),'primitive_metrics':r.get('primitive_metrics'),'benchmark_rows':r.get('benchmark_rows'), 'state_first_requested':self.cfg.run_state_first}
            elif name.startswith('fit_'):compact['results'][name]={'status':r.get('status'),'selected_profile':r.get('selected_profile'),'profile_dev_nll':{p['name']:p['dev_nll'] for p in r.get('profiles',[])}}
        json_write(self.out/'paste_back_summary.json',compact)
        from phase2h_visuals import render
        render(self.out,records)
        impl=self.out/'implementation';impl.mkdir(exist_ok=True)
        for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,impl/p.name)
        (self.out/'README_results.md').write_text('Phase 2H saved experiment. Run completion is separate from task quality, independent criteria review, and numerical acceptance.\nFull raw vectors, split/source manifests and locks accompany the compact summary. Features.sqlite files are excluded from the archive but retained in the live runtime.\n')
        archive=Path(self.cfg.output_root)/('openkind_phase2h_'+self.run_id+'.zip')
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for p in self.out.rglob('*'):
                if p.is_file() and not any(x in p.name for x in ('.sqlite','__pycache__')) and p.suffix!='.pyc':z.write(p,p.relative_to(self.out))
        if self.cfg.copy_results_to_drive:
            dest=Path(self.cfg.drive_destination)/self.run_id;dest.mkdir(parents=True,exist_ok=True)
            for name in ('paste_back_summary.json','openkind_phase2h_summary.json','README_results.md','split_manifest.json','criteria_review_template.json','criteria_pair_review_queue.json','final_lock.json'):
                p=self.out/name
                if p.exists():copy_report_file(p,dest/name)
            copy_report_file(archive,dest/archive.name);print('Copied reports/archive to:',dest)
        print('Local results:',self.out);print('Result archive:',archive)
        print('===== BEGIN_OPENKIND_PHASE2H_SUMMARY =====');print(json.dumps(compact,indent=2));print('===== END_OPENKIND_PHASE2H_SUMMARY =====');return compact
