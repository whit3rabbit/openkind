"""Phase 2D common contracts. Prompt, head, and metric definitions retained from Phase 2C.
No model download or training on import. Imported archives are data, never Python code.
"""
from __future__ import annotations
import copy, gc, hashlib, importlib.metadata as im, json, math, os, random, re, shutil, time, zipfile
from collections import Counter
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp, expit
from sklearn.metrics import f1_score, roc_auc_score
from safetensors.torch import load_file, save_file
from tqdm.auto import tqdm
VERSION = "2d.1.0"


LABELS = ["entailment", "neutral", "contradiction"]

NLI_PREFIX = "Decide the relationship between the premise and hypothesis.\nPremise:\n"

NLI_HYP_PREFIX = "\nHypothesis:\n"

NLI_TAIL = ("\nChoices:\nA: entailment. The premise makes the hypothesis true."
            "\nB: neutral. The premise does not establish whether the hypothesis is true or false."
            "\nC: contradiction. The premise makes the hypothesis false."
            "\nReturn the choice code.\nAnswer:")

DYNAMIC_INSTRUCTION = "Select the banking support intent that best describes the customer's message."

DYNAMIC_PREFIX = "Evaluate how well the candidate intent matches the message.\nInstruction:\n"

DYNAMIC_STATE_PREFIX = "\nMessage:\n"

DYNAMIC_CAND_PREFIX = "\nCandidate intent:\n"

DYNAMIC_TAIL = "\nMatch assessment:"

NONE_ID = "__none_of_these__"

def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return [json_ready(x) for x in sorted(value, key=str)]
    if isinstance(value, (list, tuple)):
        return [json_ready(x) for x in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, (Path, torch.dtype, torch.device)):
        return str(value)
    return value

def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(json_ready(value), indent=2, allow_nan=False)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(payload, encoding="utf-8")
    temp.replace(path)

def sanitize_error(error):
    text = re.sub(r"hf_[A-Za-z0-9_]+", "[REDACTED]", str(error))
    return {"type": type(error).__name__, "message": text[:700]}

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def normalized_text(text):
    return " ".join(text.split()).casefold()

def text_hash(text):
    return hashlib.sha256(normalized_text(text).encode("utf-8")).hexdigest()

def content_hash(value):
    return hashlib.sha256(json.dumps(json_ready(value), sort_keys=True, allow_nan=False).encode()).hexdigest()

def require_finite(tensor, context):
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"Non-finite values in {context}; stop rather than replacing them.")

def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)

def gpu_memory(device):
    if torch.device(device).type != "cuda":
        return {"allocated_mib": None, "reserved_mib": None, "peak_allocated_mib": None}
    return {"allocated_mib": torch.cuda.memory_allocated(device)/1024**2,
            "reserved_mib": torch.cuda.memory_reserved(device)/1024**2,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device)/1024**2}

def normalize_nli(row, split, index):
    premise, hypothesis = row["premise"].strip(), row["hypothesis"].strip()
    label = int(row["label"])
    if not premise or not hypothesis or label not in range(3):
        return None
    return dict(premise=premise, hypothesis=hypothesis, label=label,
                genre=row.get("genre", "unknown"), group=text_hash(premise),
                pair_hash=text_hash(premise + "\n<HYPOTHESIS>\n" + hypothesis),
                source_split=split, source_index=int(index), pair_id=str(row.get("pairID", index)))

def recover_rows(raw, manifest_rows, n):
    rows = []
    if n > len(manifest_rows):
        raise ValueError("Replication may not add training examples beyond the saved Phase 2B manifest")
    for entry in manifest_rows[:n]:
        row = normalize_nli(raw[entry["source_split"]][entry["source_index"]],
                            entry["source_split"], entry["source_index"])
        if row is None or row["pair_hash"] != entry["pair_hash"] or row["label"] != entry["label"]:
            raise ValueError("Pinned dataset no longer matches the saved source indices/hashes")
        rows.append(row)
    return rows

def encode_nli(row, tokenizer, max_length=256):
    enc = lambda text: tokenizer.encode(text, add_special_tokens=False)
    prefix, hp, tail = enc(NLI_PREFIX), enc(NLI_HYP_PREFIX), enc(NLI_TAIL)
    p, h = enc(row["premise"]), enc(row["hypothesis"])
    budget = max_length-len(prefix)-len(hp)-len(tail)
    if budget < 32:
        raise ValueError("NLI prompt leaves too little content budget")
    hk = h[:max(1, min(len(h), budget-min(24, budget//3)))]
    pk = p[:budget-len(hk)]
    return dict(row, input_ids=prefix+pk+hp+hk+tail,
                premise_truncated=len(pk)<len(p), hypothesis_truncated=len(hk)<len(h))

def pack_tokens(rows, pad_id, device="cpu", pad_to=None, side="right", explicit_positions=False):
    seqs = [r["input_ids"] if isinstance(r, dict) else r for r in rows]
    if not seqs or any(len(x) == 0 for x in seqs):
        raise ValueError("Every input must have at least one non-padding token")
    length = max(map(len, seqs)) if pad_to is None else int(pad_to)
    if length < max(map(len, seqs)) or side not in ("left", "right"):
        raise ValueError("Invalid padding length/side")
    ids = torch.full((len(seqs), length), int(pad_id), dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, seq in enumerate(seqs):
        sl = slice(0, len(seq)) if side == "right" else slice(length-len(seq), length)
        ids[i, sl] = torch.tensor(seq, dtype=torch.long, device=device)
        mask[i, sl] = 1
    output = {"input_ids": ids, "attention_mask": mask}
    if explicit_positions:
        output["position_ids"] = (mask.cumsum(-1)-1).clamp_min(0)
    return output

def last_nonpad(hidden, mask):
    if hidden.ndim != 3 or mask.shape != hidden.shape[:2] or not (mask.sum(-1)>0).all():
        raise ValueError("Invalid hidden/mask shapes or all-padding row")
    indices = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    last = indices.masked_fill(mask == 0, -1).max(-1).values
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), last]

@torch.inference_mode()
def forward_features(backbone, batch):
    hidden = backbone(**batch, use_cache=False, output_hidden_states=False, return_dict=True).last_hidden_state
    result = last_nonpad(hidden, batch["attention_mask"]).float()
    require_finite(result, "last non-padding hidden representation")
    return result

class FixedHead(nn.Module):
    """State-dict compatible with Phase 2B's last-token linear and MLP heads."""
    def __init__(self, hidden, classes=3, architecture="linear", width=256):
        super().__init__()
        self.architecture = architecture
        self.register_buffer("feature_mean", torch.zeros(hidden))
        self.register_buffer("feature_std", torch.ones(hidden))
        if architecture == "linear":
            self.net = nn.Linear(hidden, classes)
        elif architecture == "mlp":
            self.net = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, width), nn.GELU(),
                                     nn.Dropout(0.1), nn.Linear(width, classes))
        else:
            raise ValueError(architecture)
    def forward(self, x):
        return self.net((x.float()-self.feature_mean)/self.feature_std)
    def fit_normalization(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        self.feature_mean.copy_(x.mean(0))
        self.feature_std.copy_(x.std(0).clamp_min(1e-5))

def cpu_state(model):
    return {k: v.detach().cpu().clone().contiguous() for k, v in model.state_dict().items()}

def probs(logits, temperature=1.):
    x = np.asarray(logits, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] < 2 or not np.isfinite(x).all() or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Expected finite [N,K>=2] logits and a positive finite temperature")
    z = x/temperature
    return np.exp(z-logsumexp(z, axis=1, keepdims=True))

def probability_metrics(p, y, bins=15):
    p, y = np.asarray(p, dtype=np.float64), np.asarray(y, dtype=np.int64)
    if p.ndim != 2 or len(p)!=len(y) or len(y)==0 or not np.isfinite(p).all():
        raise ValueError("Invalid probability arrays")
    if np.any(p<0) or not np.allclose(p.sum(1),1,atol=1e-6) or np.any(y<0) or np.any(y>=p.shape[1]):
        raise ValueError("Probability normalization/label check failed")
    pred = p.argmax(1)
    conf = p.max(1)
    correct = pred == y
    ece = 0.
    reliability = []
    bucket = np.minimum((conf*bins).astype(int), bins-1)
    for b in range(bins):
        ix = bucket == b
        if ix.any():
            accuracy, confidence = float(correct[ix].mean()), float(conf[ix].mean())
            ece += ix.mean()*abs(accuracy-confidence)
            reliability.append(dict(bin=b, n=int(ix.sum()), accuracy=accuracy, confidence=confidence))
    m = {"n": len(y), "accuracy": float(correct.mean()),
         "macro_f1": float(f1_score(y,pred,labels=list(range(p.shape[1])),average="macro",zero_division=0)),
         "nll": float(-np.log(np.maximum(p[np.arange(len(y)),y], np.finfo(float).tiny)).mean()),
         "brier_sum_classes": float(np.square(p-np.eye(p.shape[1])[y]).sum(1).mean()),
         "ece_toplabel_15bins": float(ece)}
    return m, reliability

def metrics(logits, y, temperature=1.):
    return probability_metrics(probs(logits, temperature), y)[0]

def fit_temperature(logits, labels):
    x, y = np.asarray(logits,dtype=np.float64), np.asarray(labels,dtype=np.int64)
    # Direct log-softmax NLL: no probability clipping in fitting.
    objective = lambda log_t: float(np.mean(logsumexp(x/np.exp(log_t),axis=1)-(x/np.exp(log_t))[np.arange(len(y)),y]))
    opt = minimize_scalar(objective,bounds=(math.log(.25),math.log(4.)),method="bounded")
    if not opt.success or not np.isfinite(opt.fun):
        raise RuntimeError("Temperature optimizer failed")
    return float(np.exp(opt.x))

def calibration_gate(fit_logits, fit_y, gate_logits, gate_y, minimum_improvement=.002):
    fitted = fit_temperature(fit_logits, fit_y)
    raw = metrics(gate_logits, gate_y)
    scaled = metrics(gate_logits, gate_y, fitted)
    improvement = raw["nll"]-scaled["nll"]
    selected = fitted if improvement >= minimum_improvement else 1.
    return {"fitted_temperature": fitted, "selected_temperature": selected,
            "selected_on": "separate calibration_gate NLL, never test data",
            "gate_nll_improvement": improvement, "minimum_required_improvement": minimum_improvement,
            "gate_raw": raw, "gate_temperature_scaled": scaled,
            "temperature_scaling_selected": selected != 1.}

def group_bootstrap(values, groups, repeats=500, seed=17):
    values = np.asarray(values,dtype=np.float64)
    groups = np.asarray(groups)
    unique, inv = np.unique(groups,return_inverse=True)
    totals = np.bincount(inv,weights=values)
    counts = np.bincount(inv)
    if len(unique)<2:
        return {"estimate":float(values.mean()), "ci_low":None,"ci_high":None,"groups":len(unique)}
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(repeats):
        ids = rng.integers(0,len(unique),len(unique))
        estimates.append(totals[ids].sum()/counts[ids].sum())
    low, high = np.quantile(estimates,[.025,.975])
    return dict(estimate=float(values.mean()),ci_low=float(low),ci_high=float(high),groups=len(unique),
                method="95% group bootstrap; sampling uncertainty, not pretraining or training-seed uncertainty")

class CandidateHead(nn.Module):
    """One shared scalar function per (state,instruction,candidate), plus learned null logit.

    No class-specific output matrix and no arbitrary candidate key enters the model.
    The none outcome is a learned baseline, NOT a proven out-of-distribution detector.
    """
    def __init__(self,hidden):
        super().__init__()
        self.scorer=FixedHead(hidden,1,"linear")
        self.none_logit=nn.Parameter(torch.zeros(()))
    def forward(self,x,mask):
        score=self.scorer(x).squeeze(-1).masked_fill(~mask,-1e9)
        return torch.cat([score,self.none_logit.expand(len(x),1)],dim=1)

def parse_banking_csv(content, names):
    import csv
    import io
    mapping = {name: i for i, name in enumerate(names)}
    reader = csv.reader(io.StringIO(content.decode("utf-8-sig")), skipinitialspace=True)
    next(reader)  # original author's file has a two-column header
    rows = []
    for row in reader:
        if len(row) != 2 or row[1] not in mapping:
            raise ValueError("Unexpected Banking77 CSV row or class name")
        rows.append({"text": row[0], "label": mapping[row[1]]})
    return rows

def load_banking_without_scripts(dataset_id, revision, token, cache_root):
    """Read original-author CSVs with SHA-256 checks, never execute legacy dataset scripts.

    The pinned Hugging Face dataset_infos.json supplies class order, URLs and
    expected content hashes. HTTP requests to GitHub never contain HF_TOKEN.
    """
    from huggingface_hub import hf_hub_download
    from datasets import Dataset, DatasetDict, Features, Value, ClassLabel
    from urllib.request import Request, urlopen
    metadata_path = hf_hub_download(dataset_id, "dataset_infos.json", repo_type="dataset",
                                    revision=revision, token=token)
    info = json.loads(Path(metadata_path).read_text())["default"]
    names = info["features"]["label"]["names"]
    if len(names) != 77:
        raise ValueError("Expected the author's 77-label manifest")
    features = Features({"text": Value("string"), "label": ClassLabel(names=names)})
    cache_root = Path(cache_root)/"banking77_csv"
    cache_root.mkdir(parents=True, exist_ok=True)
    datasets, sources = {}, {}
    for split in ("train", "test"):
        urls = [url for url in info["download_checksums"] if url.endswith(f"/banking_data/{split}.csv")]
        if len(urls) != 1:
            raise ValueError("Cannot uniquely resolve original Banking77 CSV from pinned metadata")
        url = urls[0]
        if not url.startswith("https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"):
            raise ValueError("Unexpected Banking77 data origin; inspect metadata rather than executing it")
        expected = info["download_checksums"][url]["checksum"]
        path = cache_root/f"{expected}.csv"
        if path.exists():
            content = path.read_bytes()
        else:
            request = Request(url, headers={"User-Agent": "OpenDecision-Phase2D/1.0"})
            with urlopen(request, timeout=90) as response:
                content = response.read(10*1024**2+1)
            if len(content) > 10*1024**2:
                raise ValueError("Banking77 CSV exceeds the bounded download size")
        actual = hashlib.sha256(content).hexdigest()
        if actual != expected:
            raise ValueError(f"Banking77 {split} content hash changed; do not silently use a different dataset")
        if not path.exists():path.write_bytes(content)
        rows = parse_banking_csv(content, names)
        expected_n = int(info["splits"][split]["num_examples"])
        if len(rows) != expected_n:
            raise ValueError(f"Banking77 {split} row count mismatch")
        datasets[split] = Dataset.from_list(rows, features=features)
        sources[split] = {"url": url, "sha256": actual, "examples": len(rows)}
    return DatasetDict(datasets), {"hf_metadata_revision": revision, "loader": "verified original CSVs; no dataset script",
                                  "csv_sources": sources, "license": info.get("license")}

def normalize_banking(row,source,index):
    text=row["text"].strip()
    if not text:
        return None
    return dict(state=text,label=int(row["label"]),group=text_hash(text),source_split=source,source_index=int(index))

def choose_banking(raw,source,n,classes,blocked,seed):
    rows=[]
    for idx in np.random.default_rng(seed).permutation(len(raw[source])):
        row=normalize_banking(raw[source][int(idx)],source,int(idx))
        if row is None or row["label"] not in classes or row["group"] in blocked:
            continue
        rows.append(row); blocked.add(row["group"])
        if len(rows)==n:
            return rows
    raise ValueError(f"Insufficient unique Banking77 messages for {source}: {len(rows)}/{n}")

def encode_candidate(ep,choice,tokenizer,max_length,instruction_override=None):
    enc=lambda s:tokenizer.encode(s,add_special_tokens=False)
    instruction=instruction_override or ep["instruction"]
    pre=enc(DYNAMIC_PREFIX)+enc(instruction)+enc(DYNAMIC_STATE_PREFIX)
    post=enc(DYNAMIC_CAND_PREFIX)+enc(choice["description"])+enc(DYNAMIC_TAIL)
    budget=max_length-len(pre)-len(post)
    if budget<16:
        raise ValueError("Instruction/candidate text too long; never silently truncate the candidate definition")
    state=enc(ep["state"])
    return {"input_ids":pre+state[:budget]+post,"state_truncated":len(state)>budget,
            "group":ep["group"]}

def pad_episode_logits(logits,labels):
    width=max(map(len,logits))
    x=np.full((len(logits),width),-1e9,dtype=np.float64)
    y=[]
    for i,(row,label) in enumerate(zip(logits,labels)):
        k=len(row)-1
        x[i,:k]=row[:k];x[i,-1]=row[-1]
        y.append(width-1 if label==k else label)
    return x,np.asarray(y,dtype=np.int64)

def episode_metrics(logits,episodes,temperature=1.):
    labels=[e["target_index"] for e in episodes]
    matrix,targets=pad_episode_logits(logits,labels)
    m=metrics(matrix,targets,temperature)
    m.pop("macro_f1")  # position-specific macro-F1 is not meaningful for dynamic labels
    p=[probs(np.asarray(row)[None,:],temperature)[0] for row in logits]
    pred=np.array([r.argmax() for r in p])
    correct=pred==np.array(labels)
    absent=np.array([e["target_index"]==len(e["choices"]) for e in episodes])
    abstain=np.array([pred[i]==len(e["choices"]) for i,e in enumerate(episodes)])
    m.update(coverage=float((~abstain).mean()),none_target_rate=float(absent.mean()),
             none_recall=float(abstain[absent].mean()) if absent.any() else None,
             false_abstention_rate=float(abstain[~absent].mean()) if (~absent).any() else None,
             answered_accuracy=float(correct[~abstain].mean()) if (~abstain).any() else None,
             answerable_accuracy=float(correct[~absent].mean()) if (~absent).any() else None,
             none_auroc=float(roc_auc_score(absent,[r[-1] for r in p])) if len(np.unique(absent))==2 else None)
    return m

@dataclass
class Settings:
    preset: str = 'standard'  # smoke / quick / standard; smoke is plumbing only
    prior_archive: str = '/content/drive/MyDrive/Colab Notebooks/OpenDecision_Phase2C_results/20260917T222948Z/opendecision_phase2c_20260917T222948Z.zip'
    expected_archive_sha256: str = '2952c2fcca6638c0cc1d1e1060a567b849e46424eb2d16a078d1945233bffb63'
    model_id: str = 'Qwen/Qwen3.5-4B-Base'
    model_revision: str = '1001bb4d826a52d1f399e183466143f4da7b741b'
    nli_revision: str = 'da70db2af9d09693783c3320c4249840212ee221'
    banking_revision: str = '90d4e2ee5521c04fc1488f065b8b083658768c57'
    output_root: str = '/content/opendecision_phase2d'
    resume_run_id: str = ''  # same-runtime resume only; must match saved config and source hash
    mount_drive: bool = True
    save_to_drive: bool = True
    run_numerics: bool = True
    run_fp32: bool = True   # one model converted temporarily, with a memory guard
    run_none_ablation: bool = True
    run_request_benchmark: bool = True
    diagnostic_seed: int = 17041
    data_seed: int = 17043
    max_length: int = 256
    fit_candidate_counts: tuple = (2,4,8,16)
    eval_candidate_counts: tuple = (2,4,8,16)
    samplers: tuple = ('uniform', 'label_lexical_hard')
    benchmark_batches: tuple = (1,2,4)
    benchmark_repeats: int = 5
    benchmark_warmups: int = 1
    probability_tolerance: float = 0.005
    thresholds: tuple = (0.5,0.8,0.9,0.95)
    none_l2: float = 0.001
    calibration_gate_minimum: float = 0.002
    bootstrap_repeats: int = 500
    score_cache_commit_every: int = 50

    def sizes(self):
        return {
            'smoke': dict(numerics=2, trace=1, train=12,dev=6,cal_fit=6,cal_gate=6,test_seen=4,test_unseen=4,benchmark_messages=1),
            'quick': dict(numerics=8,trace=2,train=120,dev=40,cal_fit=60,cal_gate=40,test_seen=24,test_unseen=24,benchmark_messages=2),
            'standard': dict(numerics=32,trace=4,train=500,dev=150,cal_fit=200,cal_gate=150,test_seen=100,test_unseen=100,benchmark_messages=4),
        }[self.preset]

    def validate(self):
        if self.preset not in ('smoke','quick','standard'): raise ValueError('Unknown preset')
        if not (128 <= self.max_length <= 512): raise ValueError('max_length must be 128..512')
        if not self.fit_candidate_counts or not self.eval_candidate_counts: raise ValueError('Empty candidate counts')
        if min(self.fit_candidate_counts+self.eval_candidate_counts)<2: raise ValueError('At least two real candidates')
        if max(self.fit_candidate_counts+self.eval_candidate_counts)>19: raise ValueError('20-label holdout requires K<=19 for omitted-answer episodes')
        if set(self.samplers)-{'uniform','label_lexical_hard'}: raise ValueError('Unknown sampler')
        if not self.samplers or len(set(self.samplers))!=len(self.samplers): raise ValueError('Unique nonempty samplers required')
        if min(self.benchmark_batches)<1 or self.benchmark_repeats<1 or self.benchmark_warmups<0: raise ValueError('Invalid benchmarking settings')
        if self.none_l2<0 or self.bootstrap_repeats<2 or self.score_cache_commit_every<1: raise ValueError('Invalid optimization/cache setting')
        if self.resume_run_id and not re.fullmatch(r'[A-Za-z0-9_-]+',self.resume_run_id): raise ValueError('Invalid resume ID')
        if not re.fullmatch(r'[0-9a-f]{64}',self.expected_archive_sha256): raise ValueError('Require pinned reference archive SHA256')


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def read_reference(cfg, destination):
    """Allowlisted bounded extraction; no pickle or archived source-code execution."""
    archive=Path(cfg.prior_archive)
    if not archive.is_file(): raise FileNotFoundError(f'Mount Drive or set prior_archive: {archive}')
    sha=file_sha(archive)
    if sha!=cfg.expected_archive_sha256: raise ValueError('Reference ZIP checksum mismatch. Use the specified Phase 2C run, not an arbitrary archive.')
    names=['paste_back_summary.json','nli_split_manifest.json','dynamic_episodes.json','dynamic_data_audit.json','stability.json',
           'frozen_nli_export/manifest.json','frozen_nli_export/head.safetensors','frozen_nli_export/golden_head_inputs.npz','frozen_nli_export/golden_token_inputs.json',
           'dynamic_export/manifest.json','dynamic_export/candidate_head.safetensors','dynamic_export/golden_candidate_inputs.npz']
    root=Path(destination);root.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        if len(z.namelist())!=len(set(z.namelist())):raise ValueError('Duplicate ZIP members')
        for name in names:
            info=z.getinfo(name)
            if info.file_size>40*1024**2:raise ValueError('Oversized reference member')
            p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(info))
    ref={'root':root,'sha256':sha}
    for key,name in [('summary','paste_back_summary.json'),('nli_manifest','nli_split_manifest.json'),('episodes','dynamic_episodes.json'),('data','dynamic_data_audit.json'),('stability','stability.json'),('nli_export','frozen_nli_export/manifest.json'),('dynamic_export','dynamic_export/manifest.json')]:
        ref[key]=json.loads((root/name).read_text())
    if ref['summary']['run_id']!='20260917T222948Z':raise ValueError('Wrong reference run')
    for m in [ref['nli_export'],ref['dynamic_export']]:
        if m['model_revision']!=cfg.model_revision or m['model_id']!=cfg.model_id:raise ValueError('Model reference mismatch')
    if ref['nli_export']['label_order']!=LABELS:raise ValueError('NLI class order mismatch')
    if cfg.max_length!=ref['dynamic_export']['max_length']:raise ValueError('Keep the trained prompt max_length for Phase 2D')
    return ref


def load_heads_and_check(ref):
    h=FixedHead(ref['nli_export']['hidden_size']).eval()
    h.load_state_dict(load_file(str(ref['root']/'frozen_nli_export/head.safetensors')),strict=True)
    c=CandidateHead(ref['dynamic_export']['hidden_size']).eval()
    c.load_state_dict(load_file(str(ref['root']/'dynamic_export/candidate_head.safetensors')),strict=True)
    for m in (h,c):
        for p in m.parameters():p.requires_grad_(False)
    with torch.no_grad():
        with np.load(ref['root']/'frozen_nli_export/golden_head_inputs.npz',allow_pickle=False) as z:
            got=h(torch.from_numpy(z['features'])).numpy();expected=z['raw_logits']
            hd=float(np.max(np.abs(got-expected)))
        with np.load(ref['root']/'dynamic_export/golden_candidate_inputs.npz',allow_pickle=False) as z:
            got=c(torch.from_numpy(z['features']),torch.from_numpy(z['candidate_mask'])).numpy();expected=z['logits']
            cd=float(np.max(np.abs(got-expected)))
    if hd>1e-4 or cd>1e-4:raise ValueError(f'Exported head fixtures do not reproduce: NLI={hd}, candidate={cd}')
    for head in [h,c.scorer]:
        if not torch.isfinite(head.feature_std).all() or not (head.feature_std>0).all():raise ValueError('Invalid normalization buffers')
    return h,c,{'nli_max_abs_logit_difference':hd,'candidate_max_abs_logit_difference':cd,
                'scope':'CPU head fixture parity only, not GPU backbone parity','frozen':True}
