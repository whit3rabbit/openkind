"""OpenDecision 2I/2J workbench: immutable inputs, safe persistence, and explicit gates.
No experiment archives are imported or executed. A workflow lock is not an authorization boundary.
"""
from __future__ import annotations
import contextlib, hashlib, json, math, os, platform, shutil, sqlite3, stat, time, uuid, zipfile
from pathlib import Path, PurePosixPath
from dataclasses import dataclass, asdict
from typing import Any
import numpy as np

VERSION = '2ij.2.0'
SOURCE_RUN = '20260919T040612625670Z__finish_2h_1_2'
SOURCE_SHA256 = '7df9de857e4683386f4b687244e0bbc28a4377855d5d696d1699cb973e76fa8f'
QWEN4_REV = '1001bb4d826a52d1f399e183466143f4da7b741b'
SPLITS = ('train', 'dev', 'cal_fit', 'cal_gate', 'policy_dev', 'final')
NONE = '__none__'
class IntegrityError(RuntimeError): pass
class GateBlocked(RuntimeError): pass
class HardwareSkip(GateBlocked): pass
class BudgetStop(RuntimeError): pass

def canonical(x: Any) -> bytes:
    return json.dumps(x, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()
def digest(x: Any) -> str: return hashlib.sha256(canonical(x)).hexdigest()
def file_sha(p: str | Path) -> str:
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()
def text_hash(s: str) -> str: return hashlib.sha256(' '.join(s.split()).casefold().encode()).hexdigest()
def read_json(p: str | Path) -> Any:
    with open(p,encoding='utf-8') as f: return json.load(f)
def atomic_bytes(p: str | Path,b: bytes) -> None:
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp-'+uuid.uuid4().hex)
    try:
        with open(tmp,'xb') as f: f.write(b);f.flush();os.fsync(f.fileno())
        os.replace(tmp,p)
    finally:
        if tmp.exists():tmp.unlink()
def write_json(p: str | Path,x: Any) -> None:
    atomic_bytes(p,json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False).encode()+b'\n')
def atomic_copy(src: str | Path,dst: str | Path) -> None:
    src,dst=Path(src),Path(dst);dst.parent.mkdir(parents=True,exist_ok=True)
    tmp=dst.with_name(dst.name+'.tmp-'+uuid.uuid4().hex)
    try:
        shutil.copyfile(src,tmp)
        if file_sha(src)!=file_sha(tmp):raise IntegrityError('Copy checksum mismatch')
        os.replace(tmp,dst)
    finally:
        if tmp.exists():tmp.unlink()
def nonempty(x: Any) -> bool:return isinstance(x,str) and bool(x.strip())
def utc():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def finite_array(x, ndim=None):
    x=np.asarray(x,dtype=np.float64)
    if not np.isfinite(x).all() or (ndim is not None and x.ndim!=ndim):raise IntegrityError('Nonfinite or wrong-shaped array')
    return x

def safe_extract(path: str | Path,dest: str | Path,expected: str | None=None) -> None:
    path,dest=Path(path),Path(dest)
    if expected and file_sha(path)!=expected:raise IntegrityError('Source archive changed; refuse silently changing experiment lineage')
    with zipfile.ZipFile(path) as z:
        if len(z.infolist())>10000 or sum(i.file_size for i in z.infolist())>2*1024**3:raise IntegrityError('Archive bounds exceeded')
        seen=set()
        for i in z.infolist():
            p=PurePosixPath(i.filename)
            if p.is_absolute() or '..' in p.parts or '\\' in i.filename or ':' in i.filename or i.filename in seen:raise IntegrityError('Unsafe/duplicate archive path')
            seen.add(i.filename)
            if stat.S_ISLNK(i.external_attr>>16):raise IntegrityError('Archive symlinks forbidden')
        dest.mkdir(parents=True,exist_ok=True)
        for i in z.infolist():
            if i.is_dir():continue
            p=dest/i.filename
            if p.exists():
                if hashlib.sha256(z.read(i)).hexdigest()!=file_sha(p):raise IntegrityError('Existing source differs')
            else:atomic_bytes(p,z.read(i))

@contextlib.contextmanager
def local_lock(root: Path):
    """OS-held single-coordinator lock. Does not claim distributed Drive locking."""
    import fcntl
    root.mkdir(parents=True,exist_ok=True)
    with open(root/'.coordinator.lock','a+') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise GateBlocked('Another coordinator owns this local study')
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)

def tree_hashes(root: Path,exclude=()):
    return {p.relative_to(root).as_posix():file_sha(p) for p in sorted(root.rglob('*')) if p.is_file() and not any(s in p.parts for s in exclude) and '.tmp-' not in p.name and not p.name.endswith(('-wal','-shm'))}

def commit_snapshot(root: Path,drive: Path):
    """Copy immutable snapshot; only complete SHA-checked generations can be restored.
    SQLite WAL files are excluded; callers must make online backups first.
    """
    drive.mkdir(parents=True,exist_ok=True)
    gen='snapshot-'+str(time.time_ns())
    dst=drive/gen;dst.mkdir()
    try:
        files=tree_hashes(root,('source','__pycache__','snapshots','exports'))
        files={n:h for n,h in files.items() if n!='.coordinator.lock' and not n.endswith('.sqlite')}
        for n,h in files.items():
            atomic_copy(root/n,dst/n)
            if file_sha(dst/n)!=h:raise IntegrityError('Drive snapshot mismatch')
        manifest={'version':VERSION,'created_at':utc(),'files':files}
        write_json(dst/'SNAPSHOT_COMPLETE.json',manifest)
        write_json(drive/'LATEST.json',{'generation':gen,'manifest_sha256':file_sha(dst/'SNAPSHOT_COMPLETE.json')})
    except BaseException:
        # A concurrent worker may rotate a checkpoint while it is copied. Retry later;
        # never publish that generation or accumulate abandoned large model copies.
        shutil.rmtree(dst,ignore_errors=True)
        raise
    # Keep the newest two committed generations. Never delete original H sources.
    complete=sorted(p for p in drive.glob('snapshot-*') if (p/'SNAPSHOT_COMPLETE.json').exists())
    for old in complete[:-2]:shutil.rmtree(old)
    return gen

def restore_snapshot(drive: Path,root: Path):
    if not (drive/'LATEST.json').exists():return False
    ptr=read_json(drive/'LATEST.json');gen=ptr['generation']
    if Path(gen).name!=gen or not gen.startswith('snapshot-'):raise IntegrityError('Unsafe snapshot pointer')
    src=drive/gen
    if file_sha(src/'SNAPSHOT_COMPLETE.json')!=ptr['manifest_sha256']:raise IntegrityError('Snapshot manifest changed')
    m=read_json(src/'SNAPSHOT_COMPLETE.json')
    for n,h in m['files'].items():
        if PurePosixPath(n).is_absolute() or '..' in PurePosixPath(n).parts:raise IntegrityError('Unsafe snapshot member')
        if file_sha(src/n)!=h:raise IntegrityError('Incomplete/corrupt snapshot')
    for n in m['files']:atomic_copy(src/n,root/n)
    return True

class FeatureStore:
    """A memoized feature is identified by actual model/code/environment + finalized token IDs."""
    def __init__(self,path:Path,identity:dict):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True);self.identity=digest(identity)
        backup=self.path.with_suffix('.sqlite.backup')
        if not self.path.exists() and backup.exists():atomic_copy(backup,self.path)
        self.db=sqlite3.connect(self.path);self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('CREATE TABLE IF NOT EXISTS features (key TEXT PRIMARY KEY, shape TEXT, data BLOB, sha TEXT)')
    def key(self,token_identity):return digest([self.identity,token_identity])
    def get(self,token_identity):
        r=self.db.execute('SELECT shape,data,sha FROM features WHERE key=?',(self.key(token_identity),)).fetchone()
        if r is None:return None
        shape,b,h=r
        if hashlib.sha256(b).hexdigest()!=h:raise IntegrityError('Feature checksum failed')
        x=np.frombuffer(b,dtype='<f4').copy().reshape(json.loads(shape))
        if not np.isfinite(x).all():raise IntegrityError('Invalid cached feature')
        return x
    def put(self,token_identity,x):
        x=np.asarray(x,dtype='<f4');finite_array(x)
        b=x.tobytes();self.db.execute('INSERT OR REPLACE INTO features VALUES (?,?,?,?)',(self.key(token_identity),json.dumps(x.shape),b,hashlib.sha256(b).hexdigest()));self.db.commit()
    def backup(self):
        self.db.commit();p=self.path.with_suffix('.sqlite.backup');tmp=p.with_name(p.name+'.tmp-'+uuid.uuid4().hex)
        try:
            with sqlite3.connect(tmp) as dst:self.db.backup(dst)
            os.replace(tmp,p)
        finally:
            if tmp.exists():tmp.unlink()
    def close(self):self.backup();self.db.close()

@dataclass
class Config:
    work_root:str='/content/opendecision_2ij'
    drive_root:str='/content/drive/MyDrive/Colab Notebooks/OpenDecision_Phase2IJ_results'
    study_id:str='2ij_reviewed_multiquestion_v1'
    source_archive:str='/content/drive/MyDrive/Colab Notebooks/OpenDecision_Phase2H_Finish_results/'+SOURCE_RUN+'/opendecision_phase2h_'+SOURCE_RUN+'.zip'
    # Human-editable intake is separate from immutable registered studies and never overwritten.
    intake_root:str='/content/drive/MyDrive/Colab Notebooks/OpenDecision_Phase2IJ_review'
    study_mode:str='reviewed'
    comparison_preset:str='screen'
    include_multirc:bool=True
    run_model_export:bool=True
    run_gpu_mechanics:bool=True
    run_reviewed_study:bool=True
    run_final:bool=True
    run_scaling:bool=True
    coordinator_budget_minutes:float=120.0
    worker_budget_minutes:float=45.0
    checkpoint_steps:int=32
    sync_seconds:int=180
    minimum_free_disk_gib:float=20.0
    allow_cpu_test:bool=False
    def validate(self):
        if not self.study_id or self.study_id in ('.','..') or Path(self.study_id).name!=self.study_id or not __import__('re').fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,100}',self.study_id):raise ValueError('study_id must be a safe basename')
        if self.study_mode not in ('reviewed','exploratory_pilot'):raise ValueError('Unknown study mode')
        if self.comparison_preset not in ('screen','confirm','full'):raise ValueError('Unknown comparison preset')
        if not math.isfinite(self.coordinator_budget_minutes) or self.coordinator_budget_minutes<=0 or not math.isfinite(self.worker_budget_minutes) or self.worker_budget_minutes<=0 or self.minimum_free_disk_gib<0 or self.checkpoint_steps<1 or self.sync_seconds<10:raise ValueError('Invalid work budget')
        if Path(self.work_root).resolve()==Path(self.drive_root).resolve():raise ValueError('Local work and Drive destination must differ')
        return self

class Journal:
    def __init__(self,root:Path):self.root=root;self.path=root/'status.json';self.rows=read_json(self.path) if self.path.exists() else {}
    def record(self,stage,status,**detail):
        self.rows[stage]={'status':status,'updated_at':utc(),**detail};write_json(self.path,self.rows)
    def report(self):return [{'stage':k,**v} for k,v in self.rows.items()]

def env_identity():
    import importlib.metadata as im
    versions={}
    for n in ['torch','transformers','numpy','scipy','safetensors','peft','accelerate','huggingface-hub','flash-linear-attention','fla-core','causal-conv1d','flash-attn']:
        try:versions[n]=im.version(n)
        except im.PackageNotFoundError:versions[n]=None
    out={'python':platform.python_version(),'packages':versions,'workbench':VERSION}
    try:
        import torch
        out.update(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except ImportError:pass
    return out
