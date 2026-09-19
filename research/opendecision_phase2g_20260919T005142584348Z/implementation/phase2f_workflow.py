"""CPU-only notebook coordinator. New results; all prior Drive sources remain read-only."""
from __future__ import annotations
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,zipfile
import numpy as np
from phase2f_config import Settings, VERSION, BASELINE_RUN
from phase2f_upstream import download_upstream
from phase2d_common import json_write,file_sha,content_hash
from phase2e_core import safe_extract,IntegrityError
from phase2e_expansion import balanced_subset,redact


def choose_panel(original,cfg):
    """Reuse regression rows with complete cells, never select on codec outcomes."""
    take=cfg.sizes()['groups_per_split'];selected=[];counts={}
    expected={(k,s,a) for k in (2,4,8,16) for s in ('uniform','label_lexical_hard') for a in (False,True)}
    for split in ('test_seen','test_unseen'):
        rows=[e for e in original if e['evaluation_split']==split]
        groups=sorted({e['group'] for e in rows},key=lambda g:hashlib.sha256((str(cfg.seed)+g).encode()).hexdigest())[:take]
        if len(groups)!=take:raise IntegrityError('Not enough distinct messages in baseline panel.')
        for group in groups:
            sub=[e for e in rows if e['group']==group]
            cells={(len(e['choices']),e['sampler'],bool(e['true_intent_omitted'])) for e in sub}
            if cells!=expected or len(sub)!=len(expected):raise IntegrityError('Missing/duplicate regression factorial cell.')
            selected.extend(sorted(sub,key=lambda e:e['id']))
        counts[split]=len(groups)
    if len({e['id'] for e in selected})!=len(selected):raise IntegrityError('Duplicate episode IDs.')
    return selected,{'source':'successful expanded E regression panel; not fresh generalization data',
       'distinct_messages':len({e['group'] for e in selected}),'episodes':len(selected),
       'messages_per_split':counts,'outcome_based_selection':False,'sha256':content_hash(selected)}

class Experiment:
    def __init__(self,cfg):
        cfg.sizes();self.cfg=cfg
        self.run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        self.out=Path(cfg.output_root)/self.run_id;self.out.mkdir(parents=True,exist_ok=False)
        self.prepared=False
        self.report={'schema':'opendecision-phase2f-summary/v1','version':VERSION,'run_id':self.run_id,
           'configuration':asdict(cfg),'stage_status':{},'workers':{},'sources':{},
           'capabilities':{'MTP':{'status':'not_applicable','generated_output_tokens':0,
              'reason':'Native multi-token prediction drafts autoregressive output tokens. Known candidate suffix inputs are already processed in parallel; no decoding loop exists.'},
              'native_vllm_turboquant':{'status':'not_tested','reason':'This notebook retains the custom HF decision graph; it uses pinned upstream codec primitives, not the repo vLLM attention backend.'},
              'fused_low_bit_attention':{'status':'not_implemented','reason':'Compressed prefix snapshots are restored to floating point before suffix attention.'}},
           'limitations':[
             'Frozen checkpoint, neural heads and application policies. No training, no threshold refit, no output generation.',
             'Regression panel reuses E messages. Factorial episodes share message groups; no new cross-domain or long-context quality claim.',
             'TurboQuant code is the pinned 0xSero implementation, not an official paper-author implementation or a claimed paper replication.',
             'Only full-attention keys and values are quantized. Convolution and recurrent state remain exact and count toward total memory.',
             'Cache entries store real packed tensors. Model weights remain unquantized; active attention uses restored floating-point tensors.',
             'Reported cache tensor bytes exclude Python heap and allocator fragmentation; CUDA allocated/reserved peaks are reported separately.',
             'Warm-cache latency cannot replace cold-cache latency. Controlled request traces include their misses and evictions.',
             'No Rust/Metal, HTTP, concurrent serving, automatic policy safety, or MTP generation benchmark is performed.',
             'CPU codec selftests verify API/packing only. Qwen/GPU numerical and policy gates must pass independently.']}
        self.report['code_sha256']={p.name:file_sha(p) for p in Path(__file__).parent.glob('*.py')}
        self.checkpoint()
    def checkpoint(self):json_write(self.out/'partial_report.json',self.report)
    def prepare(self):
        import torch
        if torch.cuda.is_initialized():raise RuntimeError('Restart Colab: the coordinator must not already own a CUDA model/context.')
        cfg=self.cfg
        archive=Path(cfg.reference_archive)
        if not archive.is_file():raise FileNotFoundError(f'Missing successful expanded E archive: {archive}')
        root=safe_extract(archive,Path(cfg.cache_root)/('reference_'+cfg.expected_reference_sha256[:16]),cfg.expected_reference_sha256)
        old=json.loads((root/'opendecision_phase2e_expanded_summary.json').read_text())
        if old.get('run_id')!=BASELINE_RUN or any(old.get('workers',{}).get(m,{}).get('status')!='completed' for m in ('fp32_strict_math','bf16_default')):
            raise IntegrityError('Baseline must have two completed workers; failed/skipped T4 reports are not a baseline.')
        inputs=json.loads((root/'experiment_inputs.json').read_text());panel,audit=choose_panel(inputs['panel'],cfg)
        if old['configuration']['model_revision']!=cfg.model_revision:raise IntegrityError('Wrong checkpoint revision.')
        if content_hash(inputs['policies'])!=inputs['policy_sha256']:raise IntegrityError('Policy checksum mismatch.')
        manifest=json.loads((root/'frozen_export/manifest.json').read_text())
        if manifest['candidate_head_sha256']!=file_sha(root/'frozen_export/frozen_candidate_head.safetensors'):raise IntegrityError('Candidate head checksum mismatch.')
        # Only data and weights are imported. Prior archive implementation files are never executed.
        runtime_source=self.out/'runtime_source';shutil.copytree(root/'frozen_export',runtime_source/'export')
        benchmark=balanced_subset(panel,cfg.sizes()['benchmark_per_stratum'],cfg.seed+1)
        compression=balanced_subset(panel,cfg.sizes()['compression_per_stratum'],cfg.seed+2)
        self.inputs={'panel':panel,'benchmark_ids':[e['id'] for e in benchmark],
          'compression_ids':[e['id'] for e in compression],'policies':inputs['policies'],
          'policy_sha256':inputs['policy_sha256'],'tokenizer_sha256':file_sha(root/'frozen_export/tokenizer/tokenizer.json')}
        json_write(self.out/'experiment_inputs.json',self.inputs)
        # Only source diagnostics needed for provenance are retained, not misleading old aggregate speedups.
        self.report['source']={'run_id':BASELINE_RUN,'archive_sha256':cfg.expected_reference_sha256,
          'source_not_modified':True,'weights_frozen':True,'policies_frozen':True,
          'candidate_head_sha256':manifest['candidate_head_sha256'],'policy_sha256':inputs['policy_sha256']}
        self.report['sampling']=dict(audit,compression_episodes=len(compression),benchmark_episodes=len(benchmark),
          compression_messages=len({e['group'] for e in compression}),benchmark_messages=len({e['group'] for e in benchmark}))
        self.upstream=None
        if cfg.run_turboquant:
            try:self.upstream=download_upstream(cfg.cache_root);self.report['sources']['turboquant']=self.upstream
            except Exception as e:
                self.report['sources']['turboquant']={'status':'unavailable','error':redact(str(e))[:1000],
                    'fallback':'No substitute quantizer. Uncompressed and FP16-storage controls still run.'}
        else:self.report['sources']['turboquant']={'status':'disabled'}
        self.report['sources']['research']={
          'TurboQuant_repository':'https://github.com/0xSero/turboquant',
          'TurboQuant_paper':'https://arxiv.org/abs/2504.19874',
          'MTP':'https://docs.vllm.ai/en/v0.18.0/features/speculative_decoding/mtp/',
          'HF_cache_strategies':'https://huggingface.co/docs/transformers/v5.17.0/kv_cache',
          'PyTorch_numerics':'https://docs.pytorch.org/docs/2.11/notes/numerical_accuracy.html'}
        # Re-evaluate the actual archived none-head goldens, not a synthetic expected result.
        from phase2d_decisions import NoneModel
        from phase2e_cache import distribution
        import numpy as np
        golden=json.loads((runtime_source/'export/golden_none_inputs.json').read_text())
        selected_none=NoneModel(**json.loads((runtime_source/'export/none_set_linear.json').read_text()))
        max_delta=max(float(np.max(np.abs(distribution(g['candidate_scores'],selected_none)-np.asarray(g['probabilities'])))) for g in golden)
        if max_delta>1e-10:raise IntegrityError('Archived none-head probability fixtures do not replay.')
        self.report['sources']['none_fixture_replay']={'examples':len(golden),'maximum_probability_difference':max_delta,
            'scope':'actual saved decision-head algebra; not Qwen GPU parity'}
        self.runtime_source=runtime_source;self.root=root;self.prepared=True
        self.report['stage_status']['prepare']='completed';self.checkpoint()
        print(json.dumps(self.report['sampling'],indent=2))
        print('Frozen policies and exact token rendering retained. No GPU initialized by coordinator.')
        return self.report['sampling']
    def run_mode(self,mode):
        import torch
        if not self.prepared:raise RuntimeError('Run prepare first.')
        if torch.cuda.is_initialized():raise RuntimeError('Coordinator owns CUDA; restart runtime before isolated workers.')
        if mode not in self.cfg.modes:raise ValueError('Mode not configured.')
        out=self.out/mode
        if out.exists():raise RuntimeError('Worker output already exists. Create a new Experiment instead of mixing runs.')
        out.mkdir()
        job={'mode':mode,'configuration':asdict(self.cfg),'out':str(out),
          'source_root':str(self.runtime_source),'inputs':str(self.out/'experiment_inputs.json'),
          'upstream_root':self.upstream['root'] if self.upstream else None,
          'code_sha256':self.report['code_sha256']}
        json_write(out/'job.json',job)
        env=dict(os.environ);env['PYTHONUNBUFFERED']='1';env['TOKENIZERS_PARALLELISM']='false'
        token=env.get('HF_TOKEN')
        if not token:
            try:
                from google.colab import userdata
                token=userdata.get('HF_TOKEN')
            except Exception:token=None
        if token:env['HF_TOKEN']=token
        env['PYTHONPATH']=str(Path(__file__).parent)+os.pathsep+env.get('PYTHONPATH','')
        self.report['stage_status'][mode]='running';self.checkpoint()
        proc=subprocess.Popen([sys.executable,'-u','-m','phase2f_worker','--job',str(out/'job.json')],
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=env,bufsize=1)
        try:
            with (out/'worker.log').open('w') as log:
                for line in proc.stdout:
                    line=redact(line,token);log.write(line);log.flush();print(line,end='',flush=True)
            code=proc.wait()
        except BaseException:
            proc.terminate()
            try:proc.wait(timeout=20)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
            self.report['stage_status'][mode]='interrupted';self.checkpoint();raise
        report_path=out/'worker_report.json'
        worker=json.loads(report_path.read_text()) if report_path.exists() else {'status':'failed','error':{'message':'Worker exited without a report.'}}
        worker['exit_code']=code
        if code!=0 and worker.get('status')=='completed':worker['status']='failed'
        self.report['workers'][mode]=worker;self.report['stage_status'][mode]=worker.get('status','failed');self.checkpoint()
        print(f'\n{mode}: {worker["status"]}. Completion is not a numerical-acceptance verdict.')
        return worker
    def finish(self):
        from phase2f_report import finish_report
        return finish_report(self)
