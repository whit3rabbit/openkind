"""Colab workflow for Phase 2E. GPU research is run by the user; no synthetic fallback."""
from __future__ import annotations
import copy, gc, hashlib, traceback, re, importlib.metadata as im, inspect, json, os, platform, shutil, sys, time
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from phase2d_common import (json_write,json_ready,sanitize_error,seed_everything,file_sha,pack_tokens,
    forward_features,encode_candidate,encode_nli,recover_rows,load_banking_without_scripts,FixedHead,CandidateHead,sync)
from phase2e_core import (Settings,VERSION,SOURCES,read_sources,fresh_banking_episodes,select_complete_pairs,StageSkip,IntegrityError)
from phase2e_precision import run_precision,precision_mode,backend_flags,sample_parameter_digest
from phase2e_runtime import (FatalCUDAError, raise_if_fatal_cuda, is_fatal_cuda,
    aggregate_outcomes, indexing_selftest, validate_token_rows)
from phase2e_cache import (make_plan,parity_episode,request,time_request,TokenPlan,prefill,cache_manifest,
    full_scores,shared_scores,distribution,common_prefix_length)
from phase2e_policy import score_episodes,evaluate_policies

LIMITATIONS=[
 'Frozen Qwen3.5-4B, NLI/candidate heads, and all Phase 2D none-head coefficients. No LoRA or neural training.',
 'Precision islands promote whole modules and cast outputs back. This is not a custom FP32 accumulation kernel.',
 'NLI accuracy is a regression subset of previously reviewed Phase 2C tests, not new evidence of generalization.',
 'Shape inputs reuse the high-drift Phase 2D debugging sample; they cannot estimate normal production failure frequency.',
 'Shared cache is scoped to ONE question/request and deep-copies every mutable state tensor. No cross-request cache or question batching.',
 'Cache reuse changes the chunking/execution graph; numerical agreement must be measured, not assumed.',
 'Banking policy dev/test messages exclude C and D messages. Repeating the same seed reuses E messages.',
 'Banking held-out labels remain in one domain; pretraining decontamination is unknown; none is omitted annotated intent, not a safety detector.',
 'Absent priors and error/review costs are declared scenarios, not measured deployment rates or universal policies.',
 'Synthetic long-prefix results concern execution mechanics only, not long-context classification accuracy.',
 'Timing is single-process Python without server/network; full prefix creation and deep-copy work included; no production latency guarantee.',
 'No quantization, Rust/Metal, Jev HTTP, ordinal Score training, or proprietary Jev reproduction is established.']


def tiny_qwen_cache_selftest():
    """Runs in Colab with installed Transformers, before downloading/loading 4B weights.
    Random tiny model tests API plumbing only; it does NOT measure the real checkpoint.
    """
    from transformers import Qwen3_5TextConfig,Qwen3_5TextModel
    from phase2e_cache import TokenPlan,full_scores,shared_scores,prefill,fork_cache
    config=Qwen3_5TextConfig(vocab_size=256,hidden_size=64,intermediate_size=128,num_hidden_layers=4,
        num_attention_heads=4,num_key_value_heads=2,head_dim=16,
        linear_key_head_dim=16,linear_value_head_dim=16,linear_num_key_heads=2,linear_num_value_heads=4,
        linear_conv_kernel_dim=4,layer_types=['linear_attention']*3+['full_attention'],
        max_position_embeddings=256,pad_token_id=0,
        rope_parameters={'rope_type':'default','rope_theta':10000.,'partial_rotary_factor':1.,'mrope_section':[1,1,2]})
    config._attn_implementation='sdpa'
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7);model=Qwen3_5TextModel(config).float().eval();head=CandidateHead(64).eval()
    for p in model.parameters():p.requires_grad_(False)
    prefix=[3,4,5,6,7,8,9];suffixes=[[11,12,13],[21,22]]
    plan=TokenPlan([prefix+s for s in suffixes],prefix,suffixes,['a','b'],[False]*2,'tiny')
    full=full_scores(model,head,plan,0,'cpu');root=prefill(model,prefix,'cpu');fork_cache(root,True)
    cached,_=shared_scores(model,head,plan,'cpu',root,verify=True)
    delta=float(np.abs(full-cached).max())
    if delta>1e-3:raise IntegrityError(f'Tiny random-Qwen cache API test exceeds 1e-3 score tolerance: {delta}')
    manifest=cache_manifest(root)
    del model,root,head;gc.collect()
    return {'status':'passed','max_scalar_delta':delta,'cache_family_checks':manifest,
            'scope':'random tiny CPU model API test only; not Qwen3.5-4B accuracy or performance'}


class Experiment:
    def __init__(self,cfg:Settings):
        self.cfg=cfg;cfg.sizes();self.run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        self.out=Path(cfg.output_root)/self.run_id;self.out.mkdir(parents=True,exist_ok=False)
        self.cache=Path(cfg.cache_root);self.cache.mkdir(parents=True,exist_ok=True)
        self.ref=None;self.model=None;self.episodes={};self.regression={};self.tokenizer=None
        self._fatal_cuda = False
        self.report={'schema':'opendecision-phase2e-summary/v1','version':VERSION,'run_id':self.run_id,
            'stage_status':{},'limitations':LIMITATIONS,'sources':SOURCES,'configuration':asdict(cfg),'warnings':[]}
        json_write(self.out/'configuration.json',asdict(cfg));self.checkpoint()

    def checkpoint(self):json_write(self.out/'partial_report.json',self.report)

    def _block_after_cuda_failure(self, stage, error):
        """Persist the first failure using CPU I/O only; never continue on poisoned CUDA."""
        self._fatal_cuda = True
        original = sanitize_error(error)
        self.report['fatal_cuda'] = {'stage': stage, 'restart_required': True,
            'first_error': original,
            'message': 'GPU work stopped. Restart the Colab session, then Run all. Do not use empty_cache as recovery.'}
        # Full chained traceback is saved locally with credentials redacted.
        trace = ''.join(traceback.format_exception(type(error), error, error.__traceback__))
        trace = re.sub(r'hf_[A-Za-z0-9_]+', '[REDACTED]', trace)
        token = getattr(self, 'token', None)
        if token:
            trace = trace.replace(token, '[REDACTED]')
        (self.out/'fatal_cuda_traceback.txt').write_text(trace, encoding='utf-8')
        for name in ('model', 'model_forward_smoke', 'precision', 'shared_prefix', 'request_benchmark', 'policies'):
            if name != stage and self.report['stage_status'].get(name) not in ('completed', 'partial', 'failed', 'disabled', 'skipped'):
                self.report['stage_status'][name] = 'blocked_after_fatal_cuda'
        self.report['warnings'].append('Fatal CUDA failure: no results should be inferred for blocked stages.')
        # Preserve any partial precision measurements saved before unwinding.
        path = self.out/'precision'/'partial.json'
        if path.is_file() and 'precision' not in self.report:
            self.report['precision'] = json.loads(path.read_text())
            self.report['precision']['status'] = 'failed'
        self.checkpoint()
        try:
            # finish() only serializes CPU metadata, plots, and copies files.
            self.finish()
        except Exception as report_error:
            print('CPU failure report is at:', self.out, sanitize_error(report_error), flush=True)

    def _stage_outcome(self, name):
        value = self.report.get(name)
        if name == 'precision' and isinstance(value, dict):
            return aggregate_outcomes(value.get('modes', {}))
        if name == 'shared_prefix' and isinstance(value, dict):
            return aggregate_outcomes(value)
        if name == 'request_benchmark' and isinstance(value, dict):
            outcomes = list(value.get('rows', []))
            if self.cfg.run_long_prefix_benchmark:
                outcomes += self.report.get('long_prefix_benchmark', {}).get('rows', [])
            return aggregate_outcomes(outcomes)
        return 'completed'

    @contextmanager
    def stage(self, name, required=False):
        if self._fatal_cuda and name not in ('export',):
            raise FatalCUDAError('This experiment has a fatal CUDA failure. Restart the Colab session; do not resume GPU cells.')
        self.report['stage_status'][name] = 'running'
        self.checkpoint()
        t = time.perf_counter()
        try:
            yield
        except BaseException as error:
            self.report['stage_status'][name] = 'skipped' if isinstance(error, StageSkip) else 'failed'
            self.report.setdefault('errors', {})[name] = sanitize_error(error)
            self.checkpoint()
            if is_fatal_cuda(error):
                self.report.setdefault('stage_seconds', {})[name] = time.perf_counter() - t
                self._block_after_cuda_failure(name, error)
                raise_if_fatal_cuda(error)
            if isinstance(error, (IntegrityError, KeyboardInterrupt, SystemExit)) or required:
                raise
            print(name, self.report['stage_status'][name], sanitize_error(error), flush=True)
        else:
            self.report['stage_status'][name] = self._stage_outcome(name)
        finally:
            self.report.setdefault('stage_seconds', {})[name] = time.perf_counter() - t
            self.checkpoint()

    def setup(self):
        with self.stage('indexing_selftest', required=True):
            self.report['indexing_selftest'] = indexing_selftest()
            print('Integer-only parameter sampling self-test passed (including 635,699,200 elements).')
        with self.stage('setup',required=True):
            if not torch.cuda.is_available():raise RuntimeError('Use a fresh GPU Colab runtime; L4/A100 with native BF16 is required for the reference experiment.')
            self.device=torch.device('cuda:0')
            if not torch.cuda.is_bf16_supported(including_emulation=False):raise RuntimeError('Native BF16 unavailable. Use L4/A100; do not mix T4 emulation/FP16 with this reference run.')
            if torch.cuda.memory_allocated(self.device)>512*1024**2:raise RuntimeError('Existing GPU tensors detected. Start a fresh runtime; do not keep another large model resident.')
            if im.version('transformers')!='5.17.0':raise RuntimeError('This notebook pins Transformers 5.17.0. Run the install cell in a fresh runtime.')
            self.token=os.environ.get('HF_TOKEN')
            if not self.token:
                try:
                    from google.colab import userdata
                    self.token=userdata.get('HF_TOKEN')
                except Exception:self.token=None
            seed_everything(self.cfg.seed)
            versions={p:im.version(p) for p in ('torch','transformers','accelerate','datasets','safetensors','numpy','scipy','scikit-learn')}
            optional={}
            for p in ('flash-attn','flash-linear-attention','fla-core','causal-conv1d'):
                try:optional[p]=im.version(p)
                except im.PackageNotFoundError:optional[p]=None
            self.report['environment']={'python':sys.version.split()[0],'platform':platform.platform(),
                'gpu':torch.cuda.get_device_name(self.device),'native_bf16':True,'capability':list(torch.cuda.get_device_capability(self.device)),
                'gpu_total_gib':torch.cuda.get_device_properties(self.device).total_memory/1024**3,'packages':versions,
                'optional_kernels':optional,'backend_flags':backend_flags()}
            self.report['code_sha256']={p.name:file_sha(p) for p in Path(__file__).parent.glob('phase2*.py')}
            self.ref=read_sources(self.cfg,self.out);self.report['source']=self.ref['report']
            print(json.dumps(self.report['environment'],indent=2));print('Source fixtures verified; no archived Python code is executed.')
            json_write(self.out/'environment.json',self.report['environment'])
        with self.stage('tiny_cache_api_selftest'):
            self.report['tiny_cache_api_selftest']=tiny_qwen_cache_selftest()
            print('Tiny random-model cache API self-test passed. This is not a checkpoint benchmark.')

    def load_data(self):
        with self.stage('data',required=True):
            from transformers import AutoTokenizer
            self.tokenizer=AutoTokenizer.from_pretrained(self.cfg.model_id,revision=self.cfg.model_revision,token=self.token,trust_remote_code=False)
            if self.tokenizer.pad_token_id is None:self.tokenizer.pad_token=self.tokenizer.eos_token
            if self.tokenizer.pad_token_id is None:raise RuntimeError('Tokenizer has no usable padding token')
            self.tokenizer.padding_side='right'
            if self.cfg.run_precision:
                from datasets import load_dataset
                self.numerical=self.ref['numerical'][:self.cfg.sizes()['numerical']]
                for split in ('test_matched','test_mismatched'):
                    entries=self.ref['nli_manifest'][split][:self.cfg.sizes()['nli_per_split']]
                    required={e['source_split'] for e in entries}
                    raw={s:load_dataset('nyu-mll/multi_nli',revision=self.cfg.nli_revision,split=s,token=self.token) for s in required}
                    recovered=recover_rows(raw,entries,len(entries));self.regression[split]=[encode_nli(r,self.tokenizer,self.cfg.max_length) for r in recovered]
                    del raw
                json_write(self.out/'nli_regression_manifest.json',{s:[{k:v for k,v in r.items() if k not in ('premise','hypothesis')} for r in rows] for s,rows in self.regression.items()})
            if self.cfg.run_shared_prefix or self.cfg.run_policy_evaluation or self.cfg.run_request_benchmark:
                raw,sources=load_banking_without_scripts('PolyAI/banking77',self.cfg.banking_revision,self.token,self.cache)
                self.episodes,audit=fresh_banking_episodes(raw,self.ref,self.cfg);audit['sources']=sources
                self.report['data']=audit;json_write(self.out/'data_audit.json',audit);json_write(self.out/'episodes.json',self.episodes)
                print(json.dumps(audit['split_counts'],indent=2));del raw
            gc.collect()

    def load_model(self):
        with self.stage('model',required=True):
            from transformers import AutoModelForCausalLM
            free,_=torch.cuda.mem_get_info(self.device)
            if free<10*1024**3:raise RuntimeError('Less than 10 GiB free before the BF16 model load. Use a fresh larger GPU runtime.')
            print('Loading ONE text-only model at the pinned checkpoint revision.',flush=True)
            lm,info=AutoModelForCausalLM.from_pretrained(self.cfg.model_id,revision=self.cfg.model_revision,token=self.token,
                 trust_remote_code=False,dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa',output_loading_info=True)
            info=json_ready(info);json_write(self.out/'loading_info.json',info)
            if type(lm).__name__!='Qwen3_5ForCausalLM' or type(lm.model).__name__!='Qwen3_5TextModel':raise IntegrityError('Wrong model classes')
            if any(info.get(k) for k in ('missing_keys','mismatched_keys','error_msgs')):raise IntegrityError('Incomplete model weights')
            if any('visual' in n for n,_ in lm.named_modules()):raise IntegrityError('Unexpected vision tower')
            self.model=lm.model;del lm;self.model.eval()
            for p in self.model.parameters():p.requires_grad_(False)
            implementation=inspect.getfile(type(self.model));module=sys.modules[type(self.model).__module__]
            self.report['model']={'id':self.cfg.model_id,'revision':self.cfg.model_revision,'class':type(self.model).__name__,
                'hidden_size':self.model.config.hidden_size,'parameters':sum(p.numel() for p in self.model.parameters()),
                'trainable_parameters':sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                'implementation_sha256':file_sha(implementation),'layer_types':list(self.model.config.layer_types),
                'generation':False,'vision':False}
            audit={}
            for name in ('torch_chunk_gated_delta_rule','torch_recurrent_gated_delta_rule','Qwen3_5RMSNorm','Qwen3_5RMSNormGated','Qwen3_5GatedDeltaNet'):
                fn=getattr(module,name,None)
                if fn is None:continue
                try:
                    source=inspect.getsource(fn)
                    audit[name]={'source_sha256':hashlib.sha256(source.encode()).hexdigest(),
                        'explicit_float32_source_lines':[line.strip() for line in source.splitlines() if 'torch.float32' in line or '.float()' in line][:20],
                        'scope':'source inspection; not proof of every dispatched kernel runtime dtype'}
                except (TypeError,OSError):audit[name]={'status':'source_unavailable'}
            self.report['source_dtype_audit']=audit
            json_write(self.out/'model.json',self.report['model']);json_write(self.out/'source_dtype_audit.json',audit)
            gc.collect();torch.cuda.empty_cache()
        with self.stage('model_forward_smoke', required=True):
            vocab = int(self.model.get_input_embeddings().num_embeddings)
            checked = validate_token_rows(getattr(self, 'numerical', []), vocab, self.tokenizer.pad_token_id)
            for rows in self.regression.values():
                checked += validate_token_rows(rows, vocab, self.tokenizer.pad_token_id)
            # This metadata-only preflight catches the original index bug without GPU indexing.
            for parameter in self.model.parameters():
                from phase2e_runtime import integer_sample_indices
                integer_sample_indices(parameter.numel())
            sample = self.tokenizer.encode('A short model forward smoke test.', add_special_tokens=False)
            validate_token_rows([sample], vocab, self.tokenizer.pad_token_id)
            before = sample_parameter_digest(self.model)
            batch = pack_tokens([sample], self.tokenizer.pad_token_id, self.device, explicit_positions=True)
            vector = forward_features(self.model, batch).cpu()
            sync(self.device)
            after = sample_parameter_digest(self.model)
            if before != after:
                raise IntegrityError('Model forward mutated sampled frozen parameters')
            self.report['model_forward_smoke'] = {'status': 'passed', 'hidden_shape': list(vector.shape),
                'finite': bool(torch.isfinite(vector).all()), 'input_rows_bounds_checked': checked,
                'parameter_sampling': 'integer_only', 'parameter_sample_unchanged': True}
            json_write(self.out/'model_forward_smoke.json', self.report['model_forward_smoke'])
            print('Loaded model passed real forward smoke test and integer-only parameter audit.', flush=True)

    def precision(self):
        if not self.cfg.run_precision:self.report['stage_status']['precision']='disabled';self.checkpoint();return
        with self.stage('precision'):
            self.report['precision']=run_precision(self.model,self.ref['nli'],self.numerical,self.regression,self.tokenizer,self.cfg,self.device,self.out/'precision')

    def shared_prefix(self):
        if not self.cfg.run_shared_prefix:self.report['stage_status']['shared_prefix']='disabled';self.checkpoint();return
        results={}
        with self.stage('shared_prefix'):
            examples=select_complete_pairs(self.episodes['dev'],self.cfg.sizes()['cache_episodes'])
            for mode in self.cfg.cache_modes:
                try:
                    with precision_mode(self.model,mode,self.device,self.cfg):
                        rows=[]
                        for i,ep in enumerate(examples):
                            print('Cache parity:',mode,i+1,'/',len(examples),flush=True)
                            rows.append(parity_episode(self.model,self.ref['candidate'],ep,self.tokenizer,self.ref['none_models']['set_linear'],self.cfg,mode,self.device,tokenwise_probe=i==0))
                            json_write(self.out/'cache'/f'{mode}_partial.json',rows)
                        results[mode]={'status':'completed','rows':rows,'all_within_sampled_tolerance':all(r['within_sampled_tolerance'] for r in rows),
                           'max_probability_delta':max(r['max_probability_delta_cached_vs_full'] for r in rows),
                           'choice_changes':sum(r['choice_changed_cached_vs_full'] for r in rows),
                           'scope':'deep-copy all hybrid cache states per branch, unpadded suffix chunks; root hashes and order checks outside timing'}
                except IntegrityError:raise
                except Exception as e:
                    results[mode]={'status':'skipped' if isinstance(e,StageSkip) else 'failed','error':sanitize_error(e)}
                    self.report['shared_prefix'] = results
                    json_write(self.out/'cache'/'report.json', results)
                    raise_if_fatal_cuda(e)
                json_write(self.out/'cache'/'report.json',results)
            self.report['shared_prefix']=results

    def benchmark(self):
        if not self.cfg.run_request_benchmark:self.report['stage_status']['request_benchmark']='disabled';self.checkpoint();return
        rows=[]
        with self.stage('request_benchmark'):
            # Same dev messages, both samplers and both answer-presence conditions at every K.
            groups=list(dict.fromkeys(e['group'] for e in self.episodes['dev']))[:self.cfg.sizes()['bench_messages']]
            eps=[e for e in self.episodes['dev'] if e['group'] in groups]
            for mode in self.cfg.cache_modes:
                try:
                    with precision_mode(self.model,mode,self.device,self.cfg):
                        for ep in eps:
                            reference,_=request(self.model,self.ref['candidate'],ep,self.tokenizer,self.ref['none_models']['set_linear'],self.cfg,mode,self.device,'full_sequential')
                            for strategy in ('full_sequential','full_batch4','shared_prefix_chunk'):
                                if strategy=='shared_prefix_chunk' and self.report.get('shared_prefix',{}).get(mode,{}).get('status')!='completed':
                                    rows.append({'mode':mode,'strategy':strategy,'episode':ep['id'],'status':'skipped','reason':'cache API/integrity stage not completed'});continue
                                try:
                                    fn=lambda:request(self.model,self.ref['candidate'],ep,self.tokenizer,self.ref['none_models']['set_linear'],self.cfg,mode,self.device,strategy)
                                    p,meta=time_request(fn,self.device,self.cfg.benchmark_repeats,self.cfg.benchmark_warmups)
                                    delta=float(np.abs(reference-p).max());flip=bool(reference.argmax()!=p.argmax())
                                    row={'mode':mode,'strategy':strategy,'episode':ep['id'],'K':len(ep['choices']),'sampler':ep['sampler'],
                                         'absent':ep['true_intent_omitted'],'status':'completed','max_probability_delta_vs_full':delta,
                                         'choice_changed':flip,'within_request_tolerance':delta<=self.cfg.cache_probability_tolerance and not flip,**meta}
                                    if strategy=='shared_prefix_chunk':row['cache_stage_passed_sample']=self.report['shared_prefix'][mode]['all_within_sampled_tolerance']
                                    rows.append(row)
                                except IntegrityError:raise
                                except Exception as e:
                                    rows.append({'mode':mode,'strategy':strategy,'episode':ep['id'],'status':'failed','error':sanitize_error(e)})
                                    self.report['request_benchmark'] = {'rows': rows}
                                    json_write(self.out/'request_benchmark.json', rows)
                                    raise_if_fatal_cuda(e)
                            json_write(self.out/'request_benchmark.json',rows)
                except IntegrityError:raise
                except Exception as e:
                    rows.append({'mode':mode,'status':'skipped' if isinstance(e,StageSkip) else 'failed','error':sanitize_error(e)})
                    self.report['request_benchmark'] = {'rows': rows}
                    json_write(self.out/'request_benchmark.json', rows)
                    raise_if_fatal_cuda(e)
            self.report['request_benchmark']={'rows':rows,'scope':'cold-prefix complete requests; same inputs; both presence conditions and distractor types; sequential branch cloning, NOT parallel suffix batches'}
            self.long_prefix_benchmark()

    def long_prefix_benchmark(self):
        if not self.cfg.run_long_prefix_benchmark:self.report['long_prefix_benchmark']={'status':'disabled'};return
        results=[];head=self.ref['candidate'];none=self.ref['none_models']['set_linear']
        # Engineering-only stress case: deterministic valid tokens, no accuracy interpretation.
        token_seed=self.tokenizer.encode('Archived context for a cache mechanics benchmark. ',add_special_tokens=False)
        suffixes=[self.tokenizer.encode(f'\nCandidate intent {i}: review this support request.\nMatch assessment:',add_special_tokens=False) for i in range(8)]
        for mode in self.cfg.cache_modes:
            try:
                with precision_mode(self.model,mode,self.device,self.cfg):
                    for length in self.cfg.long_prefix_lengths:
                        prefix=(token_seed*(length//len(token_seed)+1))[:length]
                        plan=TokenPlan([prefix+s for s in suffixes],prefix,suffixes,[f'c{i}' for i in range(8)],[False]*8,'synthetic_mechanics')
                        pf=distribution(full_scores(self.model,head,plan,self.tokenizer.pad_token_id,self.device,1),none)
                        for strategy in ('full_sequential','full_batch4','shared_prefix_chunk'):
                            if strategy=='shared_prefix_chunk' and self.report.get('shared_prefix',{}).get(mode,{}).get('status')!='completed':continue
                            def fn(strategy=strategy):
                                if strategy=='shared_prefix_chunk':s,_=shared_scores(self.model,head,plan,self.device,verify=False)
                                else:s=full_scores(self.model,head,plan,self.tokenizer.pad_token_id,self.device,4 if strategy=='full_batch4' else 1)
                                return distribution(s,none),{}
                            try:
                                p,meta=time_request(fn,self.device,max(2,self.cfg.benchmark_repeats//2),1)
                                results.append({'mode':mode,'prefix_length':length,'K':8,'strategy':strategy,'status':'completed',
                                   'max_probability_delta':float(np.abs(p-pf).max()),**meta,
                                   'scope':'pretokenized synthetic long-prefix mechanics only; excludes tokenization; includes fresh prefill/clone; NOT a quality test'})
                            except IntegrityError:
                                raise
                            except Exception as e:
                                results.append({'mode':mode,'prefix_length':length,'strategy':strategy,'status':'failed','error':sanitize_error(e)})
                                self.report['long_prefix_benchmark'] = {'status': 'failed', 'rows': results}
                                raise_if_fatal_cuda(e)
            except IntegrityError:
                raise
            except Exception as e:
                results.append({'mode':mode,'status':'skipped' if isinstance(e,StageSkip) else 'failed','error':sanitize_error(e)})
                self.report['long_prefix_benchmark'] = {'status': 'failed', 'rows': results}
                raise_if_fatal_cuda(e)
        self.report['long_prefix_benchmark']={'status':aggregate_outcomes(results),'rows':results};json_write(self.out/'long_prefix_benchmark.json',results)

    def policies(self):
        if not self.cfg.run_policy_evaluation:self.report['stage_status']['policies']='disabled';self.checkpoint();return
        with self.stage('policies'):
            identity={'model':self.report['model'],'environment':self.report['environment'],'code':self.report['code_sha256'],
                      'precision':self.cfg.policy_execution_mode,'flags':backend_flags(),'positions':'explicit_zero_based','max_length':self.cfg.max_length,
                      'candidate_head_sha256':file_sha(self.ref['d']/'export/frozen_candidate_head.safetensors')}
            stores={};scoring={}
            with precision_mode(self.model,self.cfg.policy_execution_mode,self.device,self.cfg):
                for split,episodes in self.episodes.items():
                    stores[split],scoring[split]=score_episodes(episodes,self.model,self.ref['candidate'],self.tokenizer,self.cfg,self.device,self.cache/'candidate_scores.sqlite3',identity)
                    json_write(self.out/f'scores_{split}.json',{'ids':[e['id'] for e in episodes],'scores':stores[split]})
            report,predictions=evaluate_policies(stores,self.episodes,self.ref['none_models'],self.cfg)
            self.report['policies']=report;self.report['policy_scoring']=scoring
            json_write(self.out/'policy_report.json',report)
            for split,p in predictions.items():json_write(self.out/f'probabilities_{split}.json',p)
            self.policy_predictions=predictions

    def export(self):
        if self.ref is None:return
        with self.stage('export'):
            export=self.out/'export';export.mkdir(exist_ok=True)
            for file in self.ref['d'].joinpath('export').glob('*'):
                if file.is_file():shutil.copy2(file,export/file.name)
            shutil.copy2(self.ref['c']/'frozen_nli_export/head.safetensors',export/'nli_head.safetensors')
            self.tokenizer.save_pretrained(str(export/'tokenizer'))
            json_write(export/'phase2e_manifest.json',{'format':'opendecision-research-reference/v2e','model':self.report.get('model'),
              'source':self.report['source'],'original_none_manifest':'manifest.json','policy_selections':self.report.get('policies',{}).get('selections',[]),
              'precision_results':'../precision/precision_report.json','cache_results':'../cache/report.json',
              'shared_prefix_method':'exact token longest common prefix; all-state deepcopy; no sampled text generation',
              'cache_key_fields':['checkpoint revision','tokenizer/prompt identity','precision mode','exact prefix ids','positions','request isolation'],
              'scope':'Python evidence/fixtures only; no Rust/Metal implementation or production certification'})
            # New saved token/logit fixtures are diagnostic, without inventing successful GPU results.
            examples=[]
            for mode,report in self.report.get('shared_prefix',{}).items():
                for row in report.get('rows',[])[:2]:examples.append({k:v for k,v in row.items() if k not in ('cache','clone_ms')})
            json_write(export/'cache_result_fixtures.json',examples)

    def compact(self):
        r=copy.deepcopy(self.report)
        for mode,result in r.get('precision',{}).get('modes',{}).items():
            config=result.get('configuration',{});config['promoted_module_count']=len(config.pop('promoted_modules',[]))
        if 'precision' in r:
            r['precision'].pop('shape_records',None);r['precision'].pop('internal_traces',None)
        for mode,v in r.get('shared_prefix',{}).items():
            v['cache_layout_examples']=[row['cache'] for row in v.get('rows',[])[:1]]
            v['rows']=[{k:x for k,x in row.items() if k not in ('cache','full_scores','shared_scores','full_probabilities','shared_probabilities','clone_ms')} for row in v.get('rows',[])]
        # Timings retain per-request outcomes but omit individual timing samples in the paste-back file.
        for section in ('request_benchmark','long_prefix_benchmark'):
            for row in r.get(section,{}).get('rows',[]):row.pop('times_ms',None)
        r.pop('configuration',None)
        return r

    def finish(self,copy_to_drive=None):
        """Can be rerun after an optional-stage error to preserve completed stages."""
        active = [v for k, v in self.report['stage_status'].items() if k not in ('export', 'report')]
        self.report['run_status'] = ('failed' if self._fatal_cuda or 'failed' in active or 'blocked_after_fatal_cuda' in active
                                    else 'partial' if 'partial' in active or 'skipped' in active or 'running' in active
                                    else 'completed')
        self.report['stage_status']['report']='completed';self.checkpoint()
        try:
            from phase2e_visuals import plots
            plots(self.report,self.out/'plots')
        except Exception as e:self.report.setdefault('errors',{})['plotting']=sanitize_error(e)
        json_write(self.out/'opendecision_phase2e_summary.json',self.report)
        small=self.compact();json_write(self.out/'paste_back_summary.json',small)
        (self.out/'README_results.md').write_text('# OpenDecision Phase 2E\n\nRun: '+self.run_id+'\n\n'+
            'Stage status:\n```json\n'+json.dumps(self.report['stage_status'],indent=2)+'\n```\n\n'+
            '\n'.join('- '+x for x in LIMITATIONS),encoding='utf-8')
        archive=shutil.make_archive(str(self.out.parent/f'opendecision_phase2e_{self.run_id}'),'zip',self.out)
        use_drive=self.cfg.copy_results_to_drive if copy_to_drive is None else copy_to_drive
        if use_drive:
            try:
                target=Path(self.cfg.drive_notebooks)/'OpenDecision_Phase2E_results'/self.run_id;target.mkdir(parents=True,exist_ok=True)
                for f in ('paste_back_summary.json','opendecision_phase2e_summary.json','README_results.md'):shutil.copy2(self.out/f,target/f)
                shutil.copy2(archive,target/Path(archive).name)
                print('Copied reports/archive to:',target)
            except Exception as e:print('Drive copy failed; local outputs remain:',sanitize_error(e))
        print('RUN STATUS:', self.report['run_status']);print('Local results:',self.out);print('Result archive:',archive)
        print('===== BEGIN_OPENDECISION_PHASE2E_SUMMARY =====')
        print(json.dumps(json_ready(small),indent=2,allow_nan=False))
        print('===== END_OPENDECISION_PHASE2E_SUMMARY =====')
        return self.out,Path(archive)
