"""Atomic, identity-bound evaluation checkpoints; no training or model imports.

The same inputs/profile definitions/runtime identity are required when reusing rows.
Drive copies happen only outside timed requests. SQLite backups use SQLite's backup
API, not an unsafe copy of a database with an outstanding WAL.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
import shutil
import sqlite3
import time
import warnings
import zipfile
from pathlib import Path


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(',', ':')).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def resolve_under(root, relative):
    """Only portable relative paths are accepted in new recovery manifests."""
    root = Path(root).resolve()
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Manifest path must be relative and contained: ' + str(relative))
    p = (root / relative).resolve()
    if not p.is_relative_to(root):
        raise ValueError('Manifest path escaped its root')
    return p


def atomic_copy(source, destination):
    source, destination = Path(source), Path(destination)
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.resolve() == destination.resolve():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and sha256_file(source) == sha256_file(destination):
        return
    tmp = destination.with_name(destination.name + '.uploading')
    for attempt in range(3):
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, tmp)
            if sha256_file(source) != sha256_file(tmp):
                raise OSError('Copy checksum mismatch: ' + str(destination))
            os.replace(tmp, destination)
            return
        except (OSError, FileNotFoundError):
            if not source.is_file() or attempt == 2:
                raise
            time.sleep(0.3 * (attempt + 1))


def verify_final_lock(lock_path, payload_path):
    """Read hashes before parsing reserved final examples; old v1 locks supported."""
    lock_path = Path(lock_path)
    lock = json.loads(lock_path.read_text())
    if sha256_file(payload_path) != lock['final_payload_sha256']:
        raise ValueError('Final payload changed after lock')
    root = lock_path.parent
    for name, field in (('fit_payload.json', 'fit_payload_sha256'),
                        ('frozen_criteria.json', 'criteria_sha256')):
        if field in lock and sha256_file(root / name) != lock[field]:
            raise ValueError('Prepared artifact changed after lock: ' + name)
    for name, expected in lock['fit_files'].items():
        path = Path(name) if Path(name).is_absolute() and lock.get('schema', '').endswith('/v1') else resolve_under(root, name)
        if sha256_file(path) != expected:
            raise ValueError('Fitted artifact changed after final lock: ' + name)
    for name, expected in lock.get('runtime_source_files', {}).items():
        if sha256_file(resolve_under(root, name)) != expected:
            raise ValueError('Reference/tokenizer artifact changed: ' + name)
    for name, expected in lock.get('execution_code_sha256', {}).items():
        if sha256_file(Path(__file__).parent / name) != expected:
            raise ValueError('Execution code changed after lock: ' + name)
    return lock


def bind_worker_context(out, context):
    """Do not combine predictions/timings from changed engines or machines."""
    out = Path(out)
    path = out / 'resume_context.json'
    record = {'schema': 'openkind-worker-resume/v1', 'context': context,
              'context_sha256': canonical_hash(context)}
    if path.exists():
        old = json.loads(path.read_text())
        if old != record:
            raise ValueError('Resume context changed (inputs, code, model, packages or GPU). '
                             'Do not mix outputs. Use a new FINISH_TAG for a separately '
                             'reported evaluation, or restore the original environment.')
    else:
        # New workers never silently adopt old unbound output files.
        if any(out.glob('*_rows.json')) or any(out.glob('*.checkpoint.json')):
            raise ValueError('Found unbound evaluation outputs; cannot trust as resume data')
        atomic_json(path, record)
    return record['context_sha256']


class RowCheckpoint:
    """One atomic envelope binds progress to the exact workload and execution."""
    def __init__(self, out, name, workload):
        self.out = Path(out)
        self.path = self.out / (name + '.checkpoint.json')
        context_path = self.out / 'resume_context.json'
        context = json.loads(context_path.read_text())['context_sha256'] if context_path.exists() else None
        if context is None:
            raise ValueError('Bind worker context before loading evaluation checkpoints')
        self.signature = canonical_hash({'context': context, 'workload': workload})
        self.rows = []
        if self.path.exists():
            old = json.loads(self.path.read_text())
            if old.get('signature') != self.signature:
                raise ValueError('Checkpoint workload changed: ' + name)
            if canonical_hash(old['rows']) != old.get('rows_sha256'):
                raise ValueError('Checkpoint row checksum mismatch: ' + name)
            self.rows = old['rows']
        self._keys = {}
        for row in self.rows:
            key = tuple(row[k] for k in workload['key_fields'])
            if key in self._keys:
                raise ValueError('Duplicate checkpoint key: ' + repr(key))
            self._keys[key] = row
        self.key_fields = workload['key_fields']

    def get(self, *key):
        return self._keys.get(tuple(key))

    def add(self, row):
        key = tuple(row[k] for k in self.key_fields)
        if key in self._keys:
            if self._keys[key] != row:
                raise ValueError('Conflicting checkpoint result for ' + repr(key))
            return
        self.rows.append(row)
        self._keys[key] = row

    def save(self):
        atomic_json(self.path, {'schema': 'openkind-row-checkpoint/v1',
                               'signature': self.signature, 'rows': self.rows,
                               'rows_sha256': canonical_hash(self.rows)})


def validate_probability_row(row, episode, profile):
    for key, expected in (('id', episode['id']), ('group', episode['group']),
                          ('profile', profile['name']), ('target_index', episode['target_index']),
                          ('K', len(episode['choices'])), ('none_origin', episode['none_origin']),
                          ('family', episode['family']), ('panel', episode['panel'])):
        if row.get(key) != expected:
            raise ValueError('Checkpoint metadata mismatch for ' + str(episode['id']) + ': ' + key)
    p = row['probabilities']
    if len(p) != len(episode['choices']) + 1 or any(not math.isfinite(x) or x < 0 for x in p) or abs(sum(p)-1) > 1e-6:
        raise ValueError('Invalid checkpoint probability distribution')
    return p


_LAST_PUBLISH = {}

def publish_worker(out, store=None, force=False):
    """Best-effort periodic Drive checkpoint. Final export still verifies all copies.

    Intended call sites are BETWEEN evaluations, never within time_request().
    A runtime loss can require replay since the last verified checkpoint.
    """
    root = os.environ.get('OPENKIND_2H_CHECKPOINT_ROOT')
    if not root:
        return {'status': 'local_only'}
    out = Path(out)
    now = time.monotonic()
    interval = max(15., float(os.environ.get('OPENKIND_2H_CHECKPOINT_SECONDS', '120')))
    if not force and now - _LAST_PUBLISH.get(str(out), -1e30) < interval:
        return {'status': 'not_due'}
    destination = Path(root) / out.name
    backup_path = out / '.feature_backup.sqlite'
    archive_path = out / '.drive_checkpoint.zip'
    try:
        if store is not None:
            store.db.commit()
        members = {}
        for p in sorted(out.iterdir()):
            if not p.is_file() or p.name.startswith('.') or p.name.endswith(('.tmp', '.uploading', '-wal', '-shm', '.sqlite', '.zip')):
                continue
            if p.name in ('worker.log', 'job.json'):
                continue
            members[p.name] = p
        if store is not None and os.environ.get('OPENKIND_2H_SAVE_FEATURES', '1') == '1':
            dest = sqlite3.connect(str(backup_path))
            try:
                store.db.backup(dest)
            finally:
                dest.close()
            members[store.path.name] = backup_path
        hashes = {name: sha256_file(path) for name, path in members.items()}
        inventory = {'schema': 'openkind-drive-checkpoint/v1', 'worker': out.name,
                     'files': hashes, 'checkpoint_utc_epoch': time.time()}
        # One atomic ZIP publication: a interrupted copy leaves the last complete ZIP.
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
            for name, path in members.items():
                z.write(path, name)
            z.writestr('CHECKPOINT_INVENTORY.json', json.dumps(inventory, allow_nan=False))
        atomic_copy(archive_path, destination / 'worker_checkpoint.zip')
        _LAST_PUBLISH[str(out)] = now
        print('Checkpoint saved to Drive:', destination / 'worker_checkpoint.zip', flush=True)
        return {'status': 'saved', 'files': len(hashes)}
    except Exception as exc:
        warnings.warn('Drive checkpoint failed; local outputs remain: ' + str(exc), RuntimeWarning)
        return {'status': 'failed', 'error': str(exc)}
    finally:
        backup_path.unlink(missing_ok=True)
        archive_path.unlink(missing_ok=True)
