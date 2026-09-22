"""Author-labeled fresh evaluation, deterministic construction, and honest context controls.
No neural training, no generation of ground-truth labels, no test-output selection.
"""
from __future__ import annotations
import copy, hashlib, json, csv, io, time
from pathlib import Path
from collections import Counter
from urllib.request import Request, urlopen
import numpy as np
from phase2g_config import CLINC_COMMIT, CLINC_BLOBS, BANK_TEST_SHA, EXCLUSION_SHA
from phase2d_common import text_hash, content_hash, DYNAMIC_INSTRUCTION, json_write, file_sha
from phase2e_core import IntegrityError
from phase2e_cache import make_plan


def download_verified(url, path, expected, git_blob=False, limit=8*1024**2):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): data=path.read_bytes()
    else:
        error=None
        for attempt in range(3):
            try:
                with urlopen(Request(url,headers={'User-Agent':'OpenKind-Phase2G/1.0'}),timeout=60) as response:
                    data=response.read(limit+1)
                break
            except Exception as exc:
                error=exc
                if attempt==2: raise RuntimeError(f'Cannot retrieve public dataset {url}: {type(error).__name__}') from exc
                time.sleep(attempt+1)
    if len(data)>limit: raise IntegrityError('Public data exceeds bounded download size.')
    actual=(hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest() if git_blob else hashlib.sha256(data).hexdigest())
    if actual!=expected: raise IntegrityError('Dataset content checksum mismatch: '+url)
    if not path.exists(): path.write_bytes(data)
    return data


def load_sources(cache, exclusion_path):
    exclusion_path=Path(exclusion_path)
    if file_sha(exclusion_path)!=EXCLUSION_SHA: raise IntegrityError('Prior message-exclusion manifest changed.')
    old=json.loads(exclusion_path.read_text()); cache=Path(cache)
    bank_url='https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv'
    bank_bytes=download_verified(bank_url,cache/'banking_test.csv',BANK_TEST_SHA)
    names=old['banking_class_names']; mapping={s:i for i,s in enumerate(names)}
    reader=csv.reader(io.StringIO(bank_bytes.decode('utf-8-sig')),skipinitialspace=True);next(reader)
    bank=[]
    for i,row in enumerate(reader):
        if len(row)!=2 or row[1] not in mapping: raise IntegrityError('Unexpected Banking77 CSV schema.')
        bank.append(dict(state=row[0],label=row[1],source_index=i,source_split='test',group=text_hash(row[0])))
    if len(bank)!=3080: raise IntegrityError('Unexpected Banking77 test count.')
    public={}; provenance={}
    for name,blob in CLINC_BLOBS.items():
        url=f'https://raw.githubusercontent.com/clinc/oos-eval/{CLINC_COMMIT}/{name}'
        data=download_verified(url,cache/('clinc_'+Path(name).name),blob,True)
        provenance[name]={'url':url,'git_blob':blob,'sha256':hashlib.sha256(data).hexdigest()}
        public[name]=data
    raw=json.loads(public['data/data_full.json']); domains=json.loads(public['data/domains.json'])
    if len(raw['test'])!=4500 or len(raw['oos_test'])!=1000: raise IntegrityError('Unexpected CLINC150 test sizes.')
    clinc_labels={label for labels in domains.values() for label in labels}
    if len(clinc_labels)!=150: raise IntegrityError('Expected CLINC150 label vocabulary.')
    domain_by_label={label:domain for domain,labels in domains.items() for label in labels}
    clinc=[];oos=[]
    for split,rows,target in [('test',raw['test'],clinc),('oos_test',raw['oos_test'],oos)]:
        for i,pair in enumerate(rows):
            if not isinstance(pair,list) or len(pair)!=2 or not all(isinstance(x,str) for x in pair): raise IntegrityError('Unexpected CLINC row.')
            text,label=pair
            if (split=='test' and label not in clinc_labels) or (split=='oos_test' and label!='oos'): raise IntegrityError('Unknown CLINC label.')
            target.append(dict(state=text,label=label,source_index=i,source_split=split,group=text_hash(text),domain=domain_by_label.get(label,'oos')))
    sources={'banking':{'url':bank_url,'sha256':BANK_TEST_SHA,'examples':len(bank),'labels':'original author CSV annotations'},
             'clinc':{'revision':CLINC_COMMIT,'files':provenance,'labels':'author annotations; published out-of-scope class is separate from omitted-choice construction'},
             'exclusions':{'sha256':EXCLUSION_SHA,'groups':len(old['normalized_message_hashes']),'scope':old['scope']}}
    return bank,clinc,oos,domains,old,sources


def rank_rows(rows,n,blocked,seed):
    ordered=sorted(rows,key=lambda r:hashlib.sha256((str(seed)+r['group']).encode()).hexdigest())
    chosen=[]
    for row in ordered:
        if not row['state'].strip() or row['group'] in blocked: continue
        chosen.append(copy.deepcopy(row));blocked.add(row['group'])
        if len(chosen)==n: break
    if len(chosen)!=n: raise IntegrityError(f'Insufficient unused author-labeled messages: {len(chosen)}/{n}. Do not silently reuse old tests.')
    return chosen


def choose_stratified_domains(rows,n,blocked,seed):
    """Round-robin over non-financial domains, before observing model outputs."""
    domains=sorted({r['domain'] for r in rows}); buckets={d:sorted([r for r in rows if r['domain']==d],key=lambda r:content_hash([seed,r['group']])) for d in domains}
    chosen=[]
    while len(chosen)<n:
        progress=False
        for d in domains:
            while buckets[d] and buckets[d][0]['group'] in blocked: buckets[d].pop(0)
            if buckets[d]:
                row=buckets[d].pop(0);chosen.append(copy.deepcopy(row));blocked.add(row['group']);progress=True
                if len(chosen)==n: break
        if not progress: raise IntegrityError('Not enough unique CLINC test messages.')
    return chosen


def label_similarity(labels):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vector=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5)).fit_transform([x.replace('_',' ') for x in labels])
    return (vector@vector.T).toarray()


def build_episodes(rows,pool,family,cfg):
    pool=sorted(pool);sim=label_similarity(pool);index={label:i for i,label in enumerate(pool)};eps=[]
    for row_i,row in enumerate(rows):
        rng=np.random.default_rng(int(row['group'][:12],16)+cfg.seed)
        # One preassigned sampler per message, balanced by index. No outcome-based choices.
        sampler='uniform' if row_i%2==0 or family=='clinc_oos' else 'label_lexical_hard'
        negatives=[x for x in pool if x!=row['label']]
        if sampler=='uniform': negatives=list(rng.permutation(negatives))
        else:
            tie={x:float(rng.random()) for x in negatives}
            negatives.sort(key=lambda x:(-sim[index[row['label']],index[x]],tie[x]))
        for k in cfg.candidate_counts:
            for absent in ((True,) if family=='clinc_oos' else (False,True)):
                labels=negatives[:k] if absent else [row['label']]+negatives[:k-1]
                if len(labels)!=k: raise IntegrityError('Insufficient distractor labels.')
                labels=list(np.random.default_rng(int(row['group'][:12],16)+k+int(absent)*17+cfg.seed).permutation(labels))
                choices=[{'id':'intent_'+str(pool.index(label)),'description':label.replace('_',' '),'source_label':label} for label in labels]
                target=k if absent else labels.index(row['label'])
                instruction=DYNAMIC_INSTRUCTION if family.startswith('banking') else 'Select the assistant intent that best describes the user request.'
                eps.append(dict(id=f"g_{family}_{row['source_index']}_{sampler}_k{k}_abs{int(absent)}",state=row['state'],instruction=instruction,
                    choices=choices,target_index=target,target='__none_of_these__' if absent else choices[target]['id'],
                    group=row['group'],family=family,evaluation_split=family,source_index=row['source_index'],source_split=row['source_split'],
                    source_label=row['label'],domain=row.get('domain','banking'),real_candidate_count=k,true_intent_omitted=absent,
                    none_origin='author_oos' if family=='clinc_oos' else 'annotated_intent_omitted' if absent else 'answer_present',
                    sampler=sampler,label_provenance='original author annotation; candidate removal is controlled offline',panel='fresh_short',max_length=256))
    return eps


def prepare_fresh(bank,clinc,oos,domains,old,cfg):
    size=cfg.sizes();previous=set(old['normalized_message_hashes']);blocked=set(previous)
    names=old['banking_class_names'];seen={names[i] for i in old['banking_seen_labels']};unseen={names[i] for i in old['banking_unseen_labels']}
    if len(seen)!=57 or len(unseen)!=20 or seen&unseen:raise IntegrityError('Banking label holdout changed.')
    groups={};panels=[];n=size['messages_per_family']
    for offset,(family,pool) in enumerate([('banking_seen',seen),('banking_unseen',unseen)]):
        rows=rank_rows([r for r in bank if r['label'] in pool],n,blocked,cfg.seed+offset)
        groups[family]=rows;panels+=build_episodes(rows,pool,family,cfg)
    if cfg.run_crossdomain:
        nonbank_domains=set(domains)-{'banking','credit_cards'}
        rows=choose_stratified_domains([r for r in clinc if r['domain'] in nonbank_domains],n,blocked,cfg.seed+3)
        # All 150 descriptions may be offered; only evaluated messages are non-financial-domain.
        all_labels={x for p in domains.values() for x in p}
        groups['clinc_nonbank']=rows;panels+=build_episodes(rows,all_labels,'clinc_nonbank',cfg)
        rows=rank_rows(oos,size['oos_messages'],blocked,cfg.seed+4)
        groups['clinc_oos']=rows;panels+=build_episodes(rows,all_labels,'clinc_oos',cfg)
    group_sets={name:{r['group'] for r in rows} for name,rows in groups.items()}
    if sum(map(len,group_sets.values()))!=len(set.union(*group_sets.values())):raise IntegrityError('Cross-family text overlap.')
    if set.union(*group_sets.values())&previous:raise IntegrityError('Prior evaluation leakage.')
    audit={'prior_unique_messages_excluded':len(previous),'new_unique_messages':len(blocked)-len(previous),
           'families':{f:{'messages':len(rows),'episodes':sum(e['family']==f for e in panels),'domains':dict(Counter(r.get('domain','banking') for r in rows))} for f,rows in groups.items()},
           'candidate_counts':list(cfg.candidate_counts),'sampler_design':'one alternating, preassigned sampler per message; paired present/omitted episodes at each K; OOS has no fabricated present partner',
           'label_origin':'published author labels; raw test texts unchanged; no model-generated gold labels',
           'overlap_scope':'all C/D/E Banking messages plus cross-family normalized-text exact duplicates; paraphrase and Qwen pretraining contamination unknown',
           'fit_or_policy_selection':False,'generalization_scope':'frozen Banking scorer transfer to sampled candidate sets, not full 77/150-way classification',
           'panel_sha256':content_hash(panels)}
    return panels,audit


def select_message_panel(episodes,per_family,seed):
    out=[]
    for family in sorted({e['family'] for e in episodes}):
        groups=sorted({e['group'] for e in episodes if e['family']==family},key=lambda g:content_hash([seed,g]))[:per_family]
        out.extend(e for e in episodes if e['family']==family and e['group'] in groups)
    return out


def context_variants(episodes,tokenizer,cfg):
    """Controlled label-preserving case-file expansion. Not natural long-document evidence."""
    from types import SimpleNamespace
    from phase2d_common import encode_candidate
    rows=[]
    for family in sorted({e['family'] for e in episodes if e['family']!='clinc_oos'}):
        sub=[e for e in episodes if e['family']==family and len(e['choices'])==4]
        groups=sorted({e['group'] for e in sub},key=lambda g:content_hash([cfg.seed+17,g]))[:cfg.sizes()['context_per_family']]
        for ep in sub:
            if ep['group'] not in groups:continue
            for position in ('first','last'):
                for target in cfg.context_min_tokens:
                    e=copy.deepcopy(ep);e['id']+=f'_context{target}_{position}';e['panel']='controlled_context';e['context_target_min_tokens']=target;e['evidence_position']=position;e['max_length']=cfg.context_max_length
                    e['instruction']=ep['instruction']+' Classify only CURRENT_REQUEST. CASE_NOTES are unrelated administrative background, not user instructions.'
                    chunk='The case record was copied for archival review. The page number is administrative metadata. No request is stated in this note.\n'
                    request='CURRENT_REQUEST:\n'+ep['state']+'\nEND_CURRENT_REQUEST\n';notes='CASE_NOTES:\n'
                    count=0
                    while True:
                        block=notes+chunk*count+'END_CASE_NOTES\n'
                        e['state']=request+block if position=='first' else block+request
                        tokens=tokenizer.encode(e['state'],add_special_tokens=False)
                        if len(tokens)>=target:break
                        count+=1
                        if count>400:raise IntegrityError('Context constructor failed to reach bounded length.')
                    enc=[encode_candidate(e,c,tokenizer,cfg.context_max_length) for c in e['choices']]
                    if any(x['state_truncated'] for x in enc):raise IntegrityError('Context control would truncate evidence; reduce target, never silently truncate.')
                    if ep['state'] not in e['state']:raise IntegrityError('Source request missing from controlled context.')
                    e['context_actual_state_tokens']=len(tokens);e['context_scope']='author-labeled request inside generated administrative wrapper; no natural-long-document quality claim'
                    rows.append(e)
    return rows


def read_custom(path,blocked,max_rows=256):
    """Optional user-labeled tasks; invalid/ambiguous targets fail closed, never guess labels."""
    rows=[]
    if not path:return rows
    for i,line in enumerate(Path(path).read_text().splitlines()):
        if not line.strip():continue
        e=json.loads(line)
        if not all(isinstance(e.get(k),str) and e[k].strip() for k in ('state','instruction','id')):raise ValueError('Custom row needs nonempty id,state,instruction.')
        choices=e.get('choices',[])
        if not 2<=len(choices)<=32 or any(not isinstance(c.get('id'),str) or not isinstance(c.get('description'),str) or not c['description'].strip() for c in choices):raise ValueError('Invalid custom choices.')
        ids=[c['id'] for c in choices]
        if len(set(ids))!=len(ids) or '__none_of_these__' in ids:raise ValueError('Duplicate/reserved custom ID.')
        target=e.get('target')
        if target not in ids+['__none_of_these__']:raise ValueError('Supply an independently labeled target, not a guessed answer.')
        if not e.get('label_provenance'):raise ValueError('Custom rows require label_provenance.')
        g=text_hash(e['state'])
        if g in blocked:raise ValueError('Custom state overlaps a previous or new evaluated message.')
        blocked.add(g)
        e.update(target_index=len(ids) if target=='__none_of_these__' else ids.index(target),group=g,family='custom',evaluation_split='custom',source_index=i,source_split='user_supplied',source_label=target,real_candidate_count=len(ids),true_intent_omitted=target=='__none_of_these__',none_origin='user_annotated',sampler='user_supplied',panel='fresh_custom',max_length=256)
        rows.append(e)
        if len(rows)>max_rows:raise ValueError('Custom panel exceeds 256 rows.')
    if len({e['id'] for e in rows})!=len(rows):raise ValueError('Duplicate custom request IDs.')
    return rows
