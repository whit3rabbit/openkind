"""Recover saved Phase 2H fits and execute missing final stages; never train.

The old worker remains failed in the original archive and copied worker_report.
A separate recovered-artifacts record authorizes evaluation only after validation.
The original reservation registry, original result directory and archive are read-only.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import time
import zipfile
from dataclasses import asdict
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file as load_numpy
from phase2h_config import Settings, VERSION
from phase2h_checkpoint import (atomic_json, atomic_copy, sha256_file, canonical_hash,
                                resolve_under, verify_final_lock)
from phase2d_common import content_hash

SOURCE_RUN = '20260919T040612625670Z'
SOURCE_SHA256 = '51c772ab248d8949fe73bbbd91056826975eebb3594d03ca693438355d9a4cda'
SOURCE_ARCHIVE = ('/content/drive/MyDrive/Colab Notebooks/OpenDecision_Phase2H_results/'
                  + SOURCE_RUN + '/opendecision_phase2h_' + SOURCE_RUN + '.zip')
DEFAULT_DESTINATION = '/content/drive/MyDrive/Colab Notebooks/OpenDecision_Phase2H_Finish_results'
MODEL_REVISION = '1001bb4d826a52d1f399e183466143f4da7b741b'


def code_hashes():
    return {p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def safe_extract_data(archive, destination, expected_sha256=None):
    """Extract data only. Bundled reviewed modules, never archived .py, are imported."""
    archive, destination = Path(archive), Path(destination)
    if expected_sha256 and sha256_file(archive) != expected_sha256:
        raise ValueError('Source archive checksum changed; refusing unreviewed recovery')
    with zipfile.ZipFile(archive) as z:
        infos = z.infolist()
        if len(infos) > 10000 or sum(i.file_size for i in infos) > 2 * 1024**3:
            raise ValueError('Archive exceeds recovery size bounds')
        seen = set()
        for i in infos:
            if i.filename in seen:
                raise ValueError('Duplicate ZIP entry: ' + i.filename)
            seen.add(i.filename)
            resolve_under(destination, i.filename)
            if stat.S_ISLNK(i.external_attr >> 16) or '\\' in i.filename:
                raise ValueError('ZIP links and ambiguous paths are forbidden')
        destination.mkdir(parents=True, exist_ok=True)
        for i in infos:
            p = resolve_under(destination, i.filename)
            if i.is_dir():
                p.mkdir(parents=True, exist_ok=True)
            else:
                p.parent.mkdir(parents=True, exist_ok=True)
                with z.open(i) as source, p.open('wb') as target:
                    shutil.copyfileobj(source, target, 1024 * 1024)
    return destination


def inspect_saved_fit(root):
    """CPU-only validation. Do not parse final_payload_reserved.json here."""
    root = Path(root)
    summary = json.loads((root / 'opendecision_phase2h_summary.json').read_text())
    original_cfg = json.loads((root / 'config.json').read_text())
    cfg = Settings(**original_cfg)
    cfg.sizes()
    if summary['run_id'] != SOURCE_RUN or summary['version'] != '2h.1.1':
        raise ValueError('This recovery is for the reviewed 2h.1.1 source attempt')
    if summary['status'] != 'partial':
        raise ValueError('Unexpected source status; inspect rather than replacing evidence')
    for optional in ('run_lora', 'run_smaller_model', 'run_state_first'):
        if original_cfg[optional]:
            raise ValueError('This artifact-only recovery does not synthesize optional fitted arms')
    if not cfg.run_final or not cfg.run_tf32 or not cfg.run_primitives:
        raise ValueError('Source stage contract differs from the reviewed attempt')
    if any(root.glob('eval_*')) or (root / 'final_lock.json').exists():
        raise ValueError('Source already has evaluation/lock artifacts; inspect before recovery')
    prepared = json.loads((root / 'prepare_checkpoint.json').read_text())
    if prepared['run_id'] != SOURCE_RUN or prepared['configuration_sha256'] != content_hash(original_cfg):
        raise ValueError('Prepared configuration identity mismatch')
    for name, expected in prepared['files'].items():
        if sha256_file(resolve_under(root, name)) != expected:
            raise ValueError('Prepared data changed: ' + name)
    for name, expected in summary['code_sha256'].items():
        if sha256_file(root / 'implementation' / name) != expected:
            raise ValueError('Archived source hash mismatch: ' + name)
    if original_cfg != summary['configuration']:
        raise ValueError('Configuration copies disagree')
    fit = root / 'fit_qwen4b'
    report = json.loads((fit / 'worker_report.json').read_text())
    if report.get('status') != 'failed' or report.get('error') != {
        'type': 'NameError', 'message': "name 'memory_snapshot' is not defined", 'fatal_cuda': False}:
        raise ValueError('Not the reviewed reporting-only failure; do not bypass a model failure')
    if not report.get('weight_integrity', {}).get('sample_unchanged'):
        raise ValueError('Missing saved frozen-backbone integrity evidence')
    if report['spec']['revision'] != MODEL_REVISION or report['model']['hidden_size'] != 2560:
        raise ValueError('Model identity or hidden width changed')
    profiles = json.loads((fit / 'profiles.json').read_text())
    selection = json.loads((fit / 'selection.json').read_text())
    if profiles != report['profiles'] or selection['primitive_profiles'] != report['primitive_profiles']:
        raise ValueError('Saved profile/selection copies disagree')
    expected_names = set()
    for variant in ('original', 'support_examples'):
        expected_names |= {f'{variant}__instruction_first__frozen__{none}' for none in ('unchanged_none', 'refit_none')}
        expected_names |= {f'{variant}__instruction_first__joint_seed{s}__refit_none' for s in cfg.head_seeds}
        expected_names |= {f'{variant}__lexical', f'{variant}__finite_code'}
    if len(profiles) != 14 or {p['name'] for p in profiles} != expected_names:
        raise ValueError('Missing or unexpected predefined development profiles')
    candidates = [p for p in profiles if p['method'] == 'affine' and 'joint_seed' in p['name']]
    winner = min(candidates, key=lambda p: (p['dev_nll'], p['name']))
    if selection['selected_profile'] != winner['name'] or report['selected_profile'] != winner['name']:
        raise ValueError('Saved development selection disagrees with its registered rule')
    tensor_files = set()
    for p in profiles:
        if not math.isfinite(p['dev_nll']) or not math.isfinite(p['temperature']) or p['temperature'] <= 0:
            raise ValueError('Invalid saved development loss/temperature')
        if len(p['policies']) != 9 or any(not 0 <= r['threshold'] <= 1 for r in p['policies']):
            raise ValueError('Incomplete saved policy selection')
        if p['method'] == 'affine':
            path = resolve_under(fit, p['scorer_path']);tensor_files.add(path.name)
            s = load_numpy(str(path))
            shapes = {'mean': (2560,), 'std': (2560,), 'linear.weight': (1, 2560), 'linear.bias': (1,), 'none': ()}
            if set(s) != set(shapes) or any(s[k].shape != shape for k, shape in shapes.items()):
                raise ValueError('Unexpected affine readout tensors')
            if any(not np.isfinite(v).all() for v in s.values()) or np.any(s['std'] <= 0):
                raise ValueError('Invalid fitted readout values')
            none = p['none_model']
            if not all(np.isfinite(none[k]).all() for k in ('coef', 'feature_mean', 'feature_std')) or np.any(np.asarray(none['feature_std']) <= 0):
                raise ValueError('Invalid none-head coefficients')
        elif p['method'] == 'lexical':
            lex = json.loads(resolve_under(fit, p['scorer_path']).read_text())
            if len(lex['vocabulary']) != len(lex['idf']) or not np.isfinite(lex['idf']).all():
                raise ValueError('Incomplete lexical control')
    primitive_count = 0
    if {x['task'] for x in selection['primitive_profiles']} != {'noul_boolq', 'score_sst5'}:
        raise ValueError('Missing primitive fits')
    for item in selection['primitive_profiles']:
        if {p['seed'] for p in item['candidates']} != set(cfg.head_seeds):
            raise ValueError('Missing primitive seeds')
        best = min(item['candidates'], key=lambda x: (x['dev_nll'], x['seed']))
        if best['seed'] != item['selected_seed']:
            raise ValueError('Primitive selection changed')
        classes = 2 if item['task'] == 'noul_boolq' else 5
        for meta in item['candidates']:
            s = load_numpy(str(resolve_under(fit, meta['path'])))
            shapes = {'mean': (2560,), 'std': (2560,), 'linear.weight': (classes, 2560), 'linear.bias': (classes,)}
            if set(s) != set(shapes) or any(s[k].shape != shape for k, shape in shapes.items()) or any(not np.isfinite(v).all() for v in s.values()) or np.any(s['std'] <= 0):
                raise ValueError('Invalid primitive readout')
            primitive_count += 1
    # Replay actual exported DEVELOPMENT vectors; never open final stimuli.
    fixture = json.loads((fit / 'head_reference_fixtures.json').read_text())
    if fixture['profile'] != winner['name'] or fixture['none_model'] != winner['none_model'] or fixture['temperature'] != winner['temperature']:
        raise ValueError('Development fixture does not identify the selected model')
    vectors = load_numpy(str(fit / 'head_reference_vectors.safetensors'))
    state = load_numpy(str(fit / winner['scorer_path']))
    from phase2h_learning import add_none, softmaxes
    scores = []
    for i in range(len(fixture['candidate_scores'])):
        x = vectors['episode_' + str(i)].astype(np.float64)
        scores.append((((x-state['mean'])/state['std']) @ state['linear.weight'].astype(np.float64).T + state['linear.bias']).ravel())
    ps = softmaxes(add_none(winner['none_model'], scores), winner['temperature'])
    score_error = max(float(np.max(np.abs(a-b))) for a,b in zip(scores, fixture['candidate_scores']))
    probability_error = max(float(np.max(np.abs(a-b))) for a,b in zip(ps, fixture['probabilities']))
    if score_error > 1e-5 or probability_error > 1e-6:
        raise ValueError('Saved development-head algebra does not match the exported fixture')
    immutable = {}
    for name in prepared['files']:
        immutable[name] = sha256_file(root / name)
    for directory in ('fit_qwen4b', 'runtime_source'):
        for p in sorted((root / directory).rglob('*')):
            if p.is_file() and p.suffix in ('.json', '.safetensors', '.jinja'):
                immutable[p.relative_to(root).as_posix()] = sha256_file(p)
    audit = {'status': 'recovered_artifacts_validated', 'source_run_id': SOURCE_RUN,
             'source_worker_status': 'failed', 'source_error': report['error'],
             'recovery_scope': 'Saved artifacts and development head algebra only; no retraining or backbone inference',
             'source_backbone_sample_unchanged': True, 'profiles': 14,
             'affine_weight_files': len(tensor_files), 'primitive_weight_files': primitive_count,
             'saved_policy_records': sum(len(p['policies']) for p in profiles),
             'selected_profile': winner['name'], 'head_fixture_count': len(scores),
             'max_score_residual': score_error, 'max_probability_residual': probability_error,
             'final_stimuli_parsed_by_recovery_validation': False,
             'training_or_calibration_recomputed': False}
    return summary, original_cfg, report, audit, immutable


class FinishExperiment:
    """Single-coordinator artifact-only continuation of the recorded experiment."""
    def __init__(self, archive=SOURCE_ARCHIVE, *, expected_sha256=SOURCE_SHA256,
                 work_root='/content/opendecision_phase2h_finish',
                 destination_root=DEFAULT_DESTINATION, finish_tag='finish_2h_1_2',
                 copy_to_drive=True, save_features=True, checkpoint_seconds=120):
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', finish_tag):
            raise ValueError('FINISH_TAG must be a short filename-safe identifier')
        self.archive = Path(archive)
        self.run_id = SOURCE_RUN + '__' + finish_tag
        self.out = Path(work_root) / self.run_id
        self.remote = Path(destination_root) / self.run_id
        if self.out.resolve() == self.archive.parent.resolve() or self.remote.resolve() == self.archive.parent.resolve():
            raise ValueError('Recovery must not overwrite the original result folder')
        self.copy_to_drive = copy_to_drive
        self.expected_sha = expected_sha256
        self._prepared = False
        self.save_features = bool(save_features)
        self.checkpoint_seconds = max(15, int(checkpoint_seconds))
        self.manifest_identity = {'schema': 'opendecision-phase2h-finish/v1',
            'source_run_id': SOURCE_RUN, 'recovery_run_id': self.run_id,
            'source_archive_sha256': expected_sha256, 'version': VERSION,
            'code_sha256': code_hashes(), 'no_training': True}

    def prepare(self):
        if sha256_file(self.archive) != self.expected_sha:
            raise ValueError('Source archive hash differs from the reviewed attempt')
        was_present = self.out.exists()
        for base in ([self.out, self.remote] if self.copy_to_drive else [self.out]):
            path = base / 'recovery_manifest.json'
            if path.exists() and json.loads(path.read_text()) != self.manifest_identity:
                raise ValueError('Different source/code already uses this FINISH_TAG. Do not mix runs.')
        if was_present and not (self.out / 'recovery_manifest.json').exists():
            raise ValueError('Unrecognized existing working directory; choose an empty WORK_ROOT')
        self.out.mkdir(parents=True, exist_ok=True)
        atomic_json(self.out / 'recovery_manifest.json', self.manifest_identity)
        # The archive is a read-only evidence input, even though copies are extracted.
        self.snapshot = self.out / 'source_snapshot'
        ready=self.snapshot / '.source_archive_sha256'
        if not ready.is_file() or ready.read_text()!=self.expected_sha:
            if self.snapshot.exists():
                shutil.rmtree(self.snapshot)  # incomplete local extraction only
            safe_extract_data(self.archive, self.snapshot, self.expected_sha)
            ready.write_text(self.expected_sha)
        summary, original_cfg, old_worker, audit, immutable = inspect_saved_fit(self.snapshot)
        self.immutable = immutable
        for name, expected in immutable.items():
            src, dst = self.snapshot / name, self.out / name
            if dst.exists() and sha256_file(dst) != expected:
                raise ValueError('Saved source artifact was altered in recovery: ' + name)
            if not dst.exists():
                atomic_copy(src, dst)
        self.source = self.out / 'runtime_source'
        self.original_configuration = original_cfg
        self.cfg = Settings(**dict(original_cfg, output_root=str(self.out.parent),
            drive_destination=str(self.remote.parent), copy_results_to_drive=self.copy_to_drive))
        atomic_json(self.out / 'config.json', asdict(self.cfg))
        atomic_json(self.out / 'source_configuration.json', original_cfg)
        atomic_json(self.out / 'fitted_artifact_recovery.json', audit)
        self.specs = summary['model_specs']
        self.fit_jobs = [{'label': 'fit_qwen4b', 'spec': self.specs[0], 'extra': {}}]
        self.report = {k: copy.deepcopy(summary[k]) for k in
                       ('schema', 'data', 'sources', 'reference', 'limitations', 'model_specs') if k in summary}
        recovered = dict(copy.deepcopy(old_worker), status='recovered_artifacts_validated',
                         recovery=audit, original_worker_status='failed')
        atomic_json(self.out / 'fit_recovery_report.json', recovered)
        self.report.update(version=VERSION, run_id=self.run_id, configuration=asdict(self.cfg),
            status='recovered_fits_final_pending', code_sha256=code_hashes(),
            source_run_id=SOURCE_RUN, source_status='partial', recovery=audit,
            stages={'prepare': 'reused_original_reservation', 'fit_qwen4b': 'recovered_artifacts_validated'},
            workers={'fit_qwen4b': {'status': 'recovered_artifacts_validated', 'exit_code': None,
                      'report_path': str(self.out / 'fit_recovery_report.json'),
                      'original_worker_status': 'failed', 'training_executed': False}})
        if self.copy_to_drive and not was_present:
            self._restore_remote()
        for label in self.expected_workers():
            path = self.out / label / 'worker_report.json'
            if path.exists():
                worker = json.loads(path.read_text())
                self.report['workers'][label] = {'status': worker.get('status', 'interrupted'),
                    'report_path': str(path), 'exit_code': None, 'restored': True}
                self.report['stages'][label] = worker.get('status', 'interrupted')
        self._prepared = True
        self.save()
        if self.copy_to_drive:
            for name in ('recovery_manifest.json', 'fitted_artifact_recovery.json', 'fit_recovery_report.json'):
                atomic_copy(self.out / name, self.remote / name)
        print('Validated saved fits:', audit['profiles'], 'profiles;', audit['primitive_weight_files'], 'primitive heads')
        print('No data resampling, fitting, calibration selection or policy selection will run.')
        print('Continuation:', self.run_id)
        return self.report['data']

    def _restore_remote(self):
        if not self.remote.exists():
            return
        # Only the last completely published per-worker ZIP is a checkpoint authority.
        lock = self.remote / 'final_lock.json'
        if lock.exists():
            atomic_copy(lock, self.out / 'final_lock.json')
        for label in self.expected_workers():
            zpath = self.remote / label / 'worker_checkpoint.zip'
            if not zpath.exists():
                continue
            if not lock.exists():
                raise ValueError('A remote evaluation checkpoint has no final lock')
            temp = self.out / ('.restore_' + label)
            if temp.exists():
                shutil.rmtree(temp)
            safe_extract_data(zpath, temp)
            inventory = json.loads((temp / 'CHECKPOINT_INVENTORY.json').read_text())
            if inventory['worker'] != label:
                raise ValueError('Checkpoint worker identity mismatch')
            for name, expected in inventory['files'].items():
                if sha256_file(resolve_under(temp, name)) != expected:
                    raise ValueError('Corrupt remote checkpoint member: ' + name)
            target = self.out / label
            target.mkdir(exist_ok=True)
            for name in inventory['files']:
                atomic_copy(temp / name, target / name)
            shutil.rmtree(temp)
            print('Restored durable checkpoint:', label)

    def expected_workers(self):
        return ['eval_qwen4b_strict', 'eval_qwen4b_tf32']

    def save(self):
        atomic_json(self.out / 'progress_summary.json', self.report)
        if self.copy_to_drive:
            atomic_copy(self.out / 'progress_summary.json', self.remote / 'progress_summary.json')

    def fit(self):
        """Compatibility with the original notebook UI; deliberately does not fit."""
        if not self._prepared:
            raise RuntimeError('Call prepare() before reusing fitted artifacts')
        self._verify_immutable()
        print('SKIPPED all training: using the validated saved fitting artifacts.')
        return self.report['stages']

    def _verify_immutable(self):
        for name, expected in self.immutable.items():
            if sha256_file(resolve_under(self.out, name)) != expected:
                raise ValueError('Original source artifact changed: ' + name)

    def seal(self):
        self._verify_immutable()
        path = self.out / 'final_lock.json'
        if path.exists():
            verify_final_lock(path, self.out / 'final_payload_reserved.json')
            return
        if any((self.out / label / 'resume_context.json').exists() for label in self.expected_workers()):
            raise ValueError('Evaluation state exists without its original final lock')
        if self.cfg.require_independent_review_for_final and self.report['data']['criteria_audit']['independent_review'] == 'not_performed':
            raise RuntimeError('Registered independent review requirement is not satisfied')
        lock = {'schema': 'opendecision-final-lock/v2', 'run_id': self.run_id,
            'source_run_id': SOURCE_RUN, 'source_archive_sha256': self.expected_sha,
            'final_payload_sha256': sha256_file(self.out / 'final_payload_reserved.json'),
            'fit_payload_sha256': sha256_file(self.out / 'fit_payload.json'),
            'criteria_sha256': sha256_file(self.out / 'frozen_criteria.json'),
            'fit_files': {k:v for k,v in self.immutable.items() if k.startswith('fit_qwen4b/')},
            'runtime_source_files': {k:v for k,v in self.immutable.items() if k.startswith('runtime_source/')},
            'execution_code_sha256': code_hashes(), 'configuration': asdict(self.cfg),
            'created_before_final_predictions': True, 'no_security_boundary_claim': True}
        atomic_json(path, lock)
        if self.copy_to_drive:
            atomic_copy(path, self.remote / path.name)
        self.report['final_lock_sha256'] = sha256_file(path)
        self.save()

    def _completed(self, label):
        root = self.out / label
        report_path = root / 'worker_report.json'
        if not report_path.exists():
            return False
        record = json.loads(report_path.read_text())
        if record.get('status') != 'completed':
            return False
        required = ['final_rows.json', 'final_metrics.json', 'parity_rows.json', 'benchmark_rows.json',
                    'resume_context.json', 'output_inventory.json']
        if self.cfg.run_robustness:
            required += ['robustness_rows.json', 'robustness_metrics.json']
        for item in json.loads((self.out / 'fit_qwen4b/selection.json').read_text())['primitive_profiles']:
            required += [item['task'] + '_seed' + str(p['seed']) + '_predictions.json' for p in item['candidates']]
        if any(not (root / name).is_file() for name in required):
            raise ValueError('Completed worker is missing expected outputs: ' + label)
        if record.get('stage_status', {}).get('final') != 'completed':
            raise ValueError('Worker did not finish the final stage: ' + label)
        inventory = json.loads((root / 'output_inventory.json').read_text())
        for name, expected in inventory['files'].items():
            if sha256_file(resolve_under(root, name)) != expected:
                raise ValueError('Completed output hash changed: ' + name)
        return True

    def _run_worker(self, label, mode):
        verify_final_lock(self.out / 'final_lock.json', self.out / 'final_payload_reserved.json')
        if self._completed(label):
            print('Already completed and verified; skipping:', label)
            self.report['workers'][label] = {'status': 'completed', 'exit_code': 0,
                'report_path': str(self.out / label / 'worker_report.json'), 'reused_completed': True}
            self.report['stages'][label] = 'completed'
            self.save()
            return
        out = self.out / label
        out.mkdir(exist_ok=True)
        job = {'stage': 'final', 'spec': self.specs[0], 'mode': mode,
               'configuration': asdict(self.cfg), 'out': str(out), 'source': str(self.source),
               'payload': str(self.out / 'final_payload_reserved.json'),
               'fitroot': str(self.out / 'fit_qwen4b'), 'lock': str(self.out / 'final_lock.json'),
               'recovery_no_training': True}
        if mode == 'fp32_tf32_allowed':
            strict_context = json.loads((self.out / 'eval_qwen4b_strict/resume_context.json').read_text())
            job['paired_runtime_identity'] = strict_context['context']['runtime']
        atomic_json(out / 'job.json', job)
        env = dict(os.environ)
        env['PYTHONPATH'] = str(Path(__file__).parent) + os.pathsep + env.get('PYTHONPATH', '')
        env['PYTHONUNBUFFERED'] = '1'
        env['TOKENIZERS_PARALLELISM'] = 'false'
        env['OPENDECISION_2H_CHECKPOINT_SECONDS'] = str(self.checkpoint_seconds)
        env['OPENDECISION_2H_SAVE_FEATURES'] = '1' if self.save_features else '0'
        if self.copy_to_drive:
            env['OPENDECISION_2H_CHECKPOINT_ROOT'] = str(self.remote)
        else:
            env.pop('OPENDECISION_2H_CHECKPOINT_ROOT', None)
        token = env.get('HF_TOKEN')
        if not token:
            raise RuntimeError('Enable the HF_TOKEN Colab secret before model evaluation')
        self.report['stages'][label] = 'running';self.save()
        print('Starting missing evaluation:', label, flush=True)
        process = subprocess.Popen([sys.executable, '-u', '-m', 'phase2h_worker', '--job', str(out / 'job.json')],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            with (out / 'worker.log').open('a') as log:
                for line in process.stdout:
                    clean = line.replace(token, '[REDACTED]')
                    clean = re.sub(r'hf_[A-Za-z0-9_]+', '[REDACTED]', clean)
                    log.write(clean);log.flush();print(clean, end='', flush=True)
                code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill();process.wait()
            self.report['stages'][label] = 'interrupted';self.save()
            raise
        record_path = out / 'worker_report.json'
        record = json.loads(record_path.read_text()) if record_path.exists() else {'status': 'failed', 'error': 'No worker report'}
        self.report['workers'][label] = {'status': record['status'], 'exit_code': code,
            'report_path': str(record_path), 'error': record.get('error')}
        self.report['stages'][label] = record['status'];self.save()
        if code != 0 or not self._completed(label):
            raise RuntimeError(label + ' did not complete. Saved checkpoints remain; inspect its traceback, then rerun this notebook without changing the source or FINISH_TAG.')

    def evaluate(self):
        if not self._prepared:
            raise RuntimeError('prepare() must validate source artifacts first')
        self.seal()
        # Prevent two notebook coordinators sharing one local output directory.
        import fcntl
        with (self.out / '.coordinator.lock').open('w') as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('Another local coordinator is already running this continuation')
            self._run_worker('eval_qwen4b_strict', 'fp32_strict_math')
            self._run_worker('eval_qwen4b_tf32', 'fp32_tf32_allowed')
        self._verify_immutable()
        self.report['final_evaluation_exposed'] = True
        self.save()

    def finish(self):
        """Publish what exists. Completion requires both full evaluation workers."""
        self._verify_immutable()
        records = {'fit_qwen4b': json.loads((self.out / 'fit_recovery_report.json').read_text())}
        for label in self.expected_workers():
            p = self.out / label / 'worker_report.json'
            if p.exists():
                records[label] = json.loads(p.read_text())
        completed = all(self._completed(label) for label in self.expected_workers())
        self.report['status'] = 'completed' if completed else 'partial'
        self.report['final_evaluation_exposed'] = any((self.out / label / 'resume_context.json').exists() for label in self.expected_workers())
        self.report['source_attempt_status'] = 'partial; fit worker failed during memory reporting'
        self.report['original_fits_retrained'] = False
        self.report['cross_mode'] = {}
        if completed:
            from phase2h_learning import parity
            a = json.loads((self.out / 'eval_qwen4b_strict/final_rows.json').read_text())
            b = json.loads((self.out / 'eval_qwen4b_tf32/final_rows.json').read_text())
            aa = {(r['profile'], r['id']): r for r in a};bb = {(r['profile'], r['id']): r for r in b}
            if len(aa) != len(a) or len(bb) != len(b) or set(aa) != set(bb):
                raise ValueError('Cross-mode result sets are not exactly matched')
            for p in records['fit_qwen4b']['profiles']:
                keys = [k for k in aa if k[0] == p['name']]
                self.report['cross_mode'][p['name']] = parity(
                    [np.asarray(aa[k]['probabilities']) for k in keys],
                    [np.asarray(bb[k]['probabilities']) for k in keys],
                    [aa[k] for k in keys], self.cfg.probability_tolerance, p['policies'])
        self.report['open_question_status'] = {
            'criteria': 'Support-example ablation evaluated' if completed else 'Saved development evidence; final incomplete',
            'independent_criteria_review': self.report['data']['criteria_audit']['independent_review'],
            'rejection_and_multidomain_head': 'Original saved fits reused; final evaluated' if completed else 'Original saved fits reused; final pending/partial',
            'robustness': 'Controlled context and instruction-in-state evaluated' if completed else 'Pending/partial',
            'tf32': 'Executed; read numerical and policy gates separately' if completed else 'Pending/partial',
            'finite_token': 'Unadapted one-step control evaluated' if completed else 'Development available; final pending/partial',
            'noul_score': 'Bounded BoolQ/SST-5 final probes evaluated' if completed else 'Saved primitive fits; final pending/partial',
            'state_first': 'Optional, disabled in original study',
            'smaller_model': 'Optional, disabled in original study',
            'lora': 'Optional, disabled in original study',
            'natural_documents': 'Not provided in original study',
            'rust_metal_service': 'Deferred; not measured by this notebook'}
        self.report['completion_is_not_acceptance'] = True
        full = dict(self.report, worker_results=records)
        compact = dict(self.report, results={})
        compact['results']['fit_qwen4b'] = {'status': 'recovered_artifacts_validated',
            'selected_profile': records['fit_qwen4b']['selected_profile'],
            'profile_dev_nll': {p['name']: p['dev_nll'] for p in records['fit_qwen4b']['profiles']}}
        for label, r in records.items():
            if not label.startswith('eval_'):
                continue
            winner = r.get('selected_profile')
            compact['results'][label] = {'status': r['status'], 'selected_before_final': winner,
                'selected_metrics': r.get('final_metrics', {}).get(winner, {}),
                'parity': r.get('parity_summary'), 'primitive_metrics': r.get('primitive_metrics', {}),
                'benchmark_rows': r.get('benchmark_rows', []), 'error': r.get('error')}
        atomic_json(self.out / 'opendecision_phase2h_summary.json', full)
        atomic_json(self.out / 'paste_back_summary.json', compact)
        from phase2h_visuals import render
        try:
            render(self.out, records)
        except Exception as exc:
            atomic_json(self.out / 'plot_warning.json', {'error': str(exc), 'metrics_preserved': True})
        (self.out / 'README_results.md').write_text(
            'Phase 2H continuation from ' + SOURCE_RUN + '. Original fits were not retrained.\n'
            'Original failed worker status is preserved in fit_qwen4b/worker_report.json.\n'
            'Recovery validation is recorded separately in fitted_artifact_recovery.json.\n'
            'Completion requires both strict and TF32 final workers; equivalence/quality are separate.\n'
            'Evaluation features are in periodic Drive worker checkpoints when enabled, not this compact archive.\n')
        implementation = self.out / 'implementation'
        implementation.mkdir(exist_ok=True)
        for p in Path(__file__).parent.glob('*.py'):
            atomic_copy(p, implementation / p.name)
        archive = self.out.parent / ('opendecision_phase2h_' + self.run_id + '.zip')
        temp = archive.with_name(archive.name + '.tmp')
        with zipfile.ZipFile(temp, 'w', zipfile.ZIP_DEFLATED, compresslevel=3) as z:
            z.write(self.archive, 'original_attempt.zip')
            for p in sorted(self.out.rglob('*')):
                relative = p.relative_to(self.out)
                if not p.is_file() or 'source_snapshot' in relative.parts or '__pycache__' in relative.parts:
                    continue
                if any(part.startswith('.') for part in relative.parts) or p.name.endswith(('.sqlite', '-wal', '-shm', '.pyc', '.tmp', '.uploading', '.zip')):
                    continue
                z.write(p, relative.as_posix())
        os.replace(temp, archive)
        self.save()
        if self.copy_to_drive:
            for name in ('recovery_manifest.json', 'fitted_artifact_recovery.json', 'fit_recovery_report.json',
                         'paste_back_summary.json', 'opendecision_phase2h_summary.json', 'README_results.md',
                         'final_lock.json', 'split_manifest.json'):
                if (self.out / name).exists():
                    atomic_copy(self.out / name, self.remote / name)
            atomic_copy(archive, self.remote / archive.name)
        print('Overall execution status:', self.report['status'])
        print('Original fits retrained: False')
        print('Result archive:', archive)
        print('Drive continuation folder:', self.remote if self.copy_to_drive else '(disabled)')
        return compact
