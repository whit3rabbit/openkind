"""Model-first intake. Exploration is explicitly not independent review or release approval.
A new exploratory study uses a new namespace and a machine-verifiable provenance receipt.
The existing independently reviewed path and its locked data remain unchanged.
"""
from __future__ import annotations
import copy, hashlib, html, math, random, re
from collections import defaultdict, Counter
from pathlib import Path
from odij_core import *
from odij_data import protocol_template, validate_case, load_cases, protocol_body, flatten

MULTIRC = {
    'repo':'aps/super_glue',
    'files':{
        'train':{'path':'multirc/train-00000-of-00001.parquet','sha256':'ced6d731a335a0e4fbd29f6f382f291518952c0f449448361d294000c39e6728'},
        'validation':{'path':'multirc/validation-00000-of-00001.parquet','sha256':'5eb64893afc5af7828b2f3b41d6a4574eb7a82cbb17c86c75842f5f95f679ada'},
    },
    'source':'https://cogcomp.seas.upenn.edu/multirc/',
    'paper':'https://aclanthology.org/N18-1023/',
    'purpose':'Per-answer binary correctness; multiple correct answers are NOT forced into a single-choice target.',
    'rights':'Source terms must be checked before redistribution or deployment; no license clearance asserted.'
}
PINNED_MODELS={'qwen2b':'b1485b2fa6dfa1287294f269f5fb618e03d52d7c',
               'modernbert':'45bb4654a4d5aaff24dd11d4781fa46d39bf8c13'}


def clean_markup(text):
    # Explicit input transformation, not semantic paraphrasing or label correction.
    return html.unescape(re.sub(r'<[^>]+>',' ',str(text))).strip()


def fetch_multirc(cache: Path):
    """Download only the two known labeled source partitions; never the unlabeled test.
    The content digests, not a mutable main branch, decide admissibility.
    """
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq
    cache.mkdir(parents=True,exist_ok=True)
    lock=cache/'SOURCE_LOCK.json'
    if lock.exists():
        meta=read_json(lock);rev=meta['revision']
        if meta['source_spec']!=MULTIRC:raise IntegrityError('MultiRC source specification changed')
    else:
        rev=HfApi(token=os.environ.get('HF_TOKEN')).dataset_info(MULTIRC['repo']).sha
        if not re.fullmatch('[0-9a-f]{40}',rev or ''):raise IntegrityError('Dataset revision is not immutable')
        meta={'source_spec':MULTIRC,'revision':rev,'files':{}}
    records={}
    for part,desc in MULTIRC['files'].items():
        destination=cache/(part+'.parquet')
        if not destination.exists():
            source=hf_hub_download(MULTIRC['repo'],desc['path'],repo_type='dataset',revision=rev,token=os.environ.get('HF_TOKEN'))
            if file_sha(source)!=desc['sha256']:raise IntegrityError('MultiRC content differs from reviewed source identity; do not silently substitute')
            atomic_copy(source,destination)
        if file_sha(destination)!=desc['sha256']:raise IntegrityError('MultiRC cache changed')
        records[part]=pq.read_table(destination).to_pylist()
        meta['files'][part]={'sha256':file_sha(destination),'rows':len(records[part])}
    write_json(lock,meta)
    return records,meta


def build_multirc_cases(records,seed,blocked=(),counts=None):
    """Keep all rows from a source paragraph together. Choose by content hash, not label.
    Natural binary tasks accompany, but do not replace, controlled Choice/Score cases.
    """
    counts=counts or {'train':48,'dev':12,'cal_fit':8,'cal_gate':8,'policy_dev':12,'final':24}
    pools={};eligibility={}
    for source_part,raw in records.items():
        groups=defaultdict(list)
        for r in raw:
            if r.get('label') not in (0,1):raise IntegrityError('MultiRC requires original binary labels')
            if not isinstance(r.get('idx'),dict):raise IntegrityError('Unexpected MultiRC index schema')
            groups[int(r['idx']['paragraph'])].append(r)
        pool=[];skipped=Counter()
        for gid,rs in groups.items():
            states={clean_markup(r['paragraph']) for r in rs}
            if len(states)!=1:raise IntegrityError('One paragraph ID maps to different states')
            state=next(iter(states));gh=text_hash(state)
            if gh in blocked:skipped['historical_or_duplicate']+=1;continue
            # Predeclared eligibility screen, never truncate the evidence.
            if len(state)>2400:skipped['over_2400_state_characters']+=1;continue
            stems=defaultdict(list)
            for r in rs:
                if len(clean_markup(r['question']))+len(clean_markup(r['answer']))<=600:
                    stems[int(r['idx']['question'])].append(r)
            if len(stems)<2:skipped['fewer_than_two_distinct_source_questions']+=1;continue
            ordered=sorted(stems,key=lambda k:digest([seed,source_part,gid,k]))
            selected=[]
            # Cycle source questions before adding second answer hypotheses. No label balancing.
            queues={k:sorted(stems[k],key=lambda r:digest([seed,r['idx'],r['question'],r['answer']])) for k in ordered}
            while len(selected)<4 and any(queues.values()):
                for k in ordered:
                    if queues[k] and len(selected)<4:selected.append(queues[k].pop(0))
            qs=[]
            for r in selected:
                source_q=int(r['idx']['question']);source_a=int(r['idx']['answer'])
                qs.append({'id':f'q{source_q}_a{source_a}','family':'multirc_answer_correctness','rubric_family':'multirc_binary_v1',
                  'primitive':'noul','instruction':'Using only the passage, is the proposed answer a correct answer to the question?\nQuestion: '+clean_markup(r['question'])+'\nProposed answer: '+clean_markup(r['answer']),
                  'options':[{'id':'no','label':'No','criteria':'The proposed answer is not a correct answer under the passage.','value':False},
                             {'id':'yes','label':'Yes','criteria':'The proposed answer is a correct answer under the passage.','value':True}],
                  'target_id':'yes' if r['label']==1 else 'no','origin':'answer_present','evidence_quotes':[],
                  'source_annotation':{'dataset':'MultiRC/SuperGLUE','partition':source_part,'idx':r['idx'],'label':int(r['label'])}})
            pool.append({'id':f'multirc-{source_part}-{gid}','group_id':'multirc-'+gh,'state':state,'questions':qs,
                         'source':{'kind':'natural','reference':f"{MULTIRC['repo']}/multirc/{source_part}/paragraph={gid}",
                                   'author':'MultiRC original annotations distributed through SuperGLUE; not independently re-reviewed here'}})
        pools[source_part]=sorted(pool,key=lambda c:digest([seed,c['group_id']]))
        eligibility[source_part]={'eligible_paragraphs':len(pool),'excluded':dict(skipped)}
    out=[];seen=set(blocked);cursor=0
    for split,n in counts.items():
        pool=pools['validation'] if split=='final' else pools['train']
        available=[c for c in pool if text_hash(c['state']) not in seen]
        if len(available)<n:raise GateBlocked(f'MultiRC has {len(available)} eligible unused paragraphs for {split}, requires {n}; source counts are not silently reduced')
        for c in available[:n]:
            seen.add(text_hash(c['state']));out.append(validate_case(dict(c,split=split)))
    return out,{'selection_seed':seed,'paragraph_counts':counts,'max_hypotheses_per_paragraph':4,
                'input_transform':'Strip HTML tags to spaces; HTML unescape; no paraphrasing',
                'eligibility':eligibility,'official_benchmark_score':False,
                'limitation':'This is a sampled binary correctness probe, not the official MultiRC F1/EM benchmark or a held-out semantic-family proof.'}


def build_constructed_cases(seed,counts=None,blocked=()):
    """Oracle-derived fictional facts, randomized clause order and IDs, mixed answerability.
    The same fact/wording generator spans splits: this tests bounded transfer, not novel-world generality.
    """
    counts=counts or {'train':64,'dev':24,'cal_fit':16,'cal_gate':16,'policy_dev':24,'final':32}
    rng=random.Random(seed);out=[];seen_facts=set()
    labels=['workstation','application server','network appliance']
    def options(texts):return [{'id':f'o{j}','label':x,'criteria':x} for j,x in enumerate(texts)]
    for split,n in counts.items():
        for i in range(n):
            asset=rng.choice(labels);external=bool(rng.getrandbits(1));comp=bool(rng.getrandbits(1));owner=bool(rng.getrandbits(1));conflict=bool(rng.getrandbits(1));impact=rng.randrange(3)
            os_name=None if i%8==3 else rng.choice([None,'Linux','Windows'])
            facts=(asset,external,comp,owner,conflict,impact,os_name)
            attempts=0
            while facts in seen_facts or 'latent-'+digest(facts) in blocked:
                attempts+=1
                if attempts>10000:raise GateBlocked('Requested constructed corpus exhausts distinct latent facts; use a new generator rather than leaking variants across splits')
                asset=rng.choice(labels);external=bool(rng.getrandbits(1));comp=bool(rng.getrandbits(1));owner=bool(rng.getrandbits(1));conflict=bool(rng.getrandbits(1));impact=rng.randrange(3)
                os_name=None if i%8==3 else rng.choice([None,'Linux','Windows']);facts=(asset,external,comp,owner,conflict,impact,os_name)
            seen_facts.add(facts)
            case_id=f'pilot-{seed}-{split}-{i:04d}'
            clauses=[f'Affected asset: {asset}.',f'An external destination was {"observed" if external else "not observed"}.',
                     f'Compromise was {"confirmed" if comp else "not confirmed"}.',f'Owner information is {"present" if owner else "missing"}.',
                     f'The analyst reports are {"contradictory" if conflict else "consistent"}.',f'The documented impact level is {impact}.']
            if os_name:clauses.append('The asset operating system is '+os_name+'.')
            else:clauses.append('The operating system is not specified.')
            rng.shuffle(clauses)
            state=f'Fictional incident {rng.getrandbits(64):016x}. '+' '.join(clauses)+' These statements are the full evidence record; do not infer missing facts.'
            qs=[]
            def add(family,instruction,opts,target,primitive='choice',rubric=None,origin='answer_present'):
                qs.append({'id':family,'family':family,'rubric_family':rubric or family+'_v2','primitive':primitive,'instruction':instruction,
                           'options':copy.deepcopy(opts),'target_id':target,'origin':origin,'evidence_quotes':[]})
            yn=[{'id':'no','label':'No','criteria':'The stated proposition is false according to the record.','value':False},
                {'id':'yes','label':'Yes','criteria':'The stated proposition is true according to the record.','value':True}]
            op=options(labels);rng.shuffle(op)
            add('asset','Which asset type is explicitly recorded?',op,next(o['id'] for o in op if o['label']==asset))
            add('external','Does the record explicitly state that an external destination was observed?',yn,'yes' if external else 'no','noul')
            add('compromise','Does the record explicitly state that compromise was confirmed?',yn,'yes' if comp else 'no','noul')
            mode=i%4
            if mode==0:add('routing','Which supplied category describes this record?',options(['A documented technical incident','A dining reservation']),'o0')
            elif mode==1:add('routing','Choose the stated affected asset type. If no offered option matches, none is correct.',options(['mobile handset','printer']),NONE,origin='annotated_intent_omitted')
            elif mode==2:add('routing','Classify a restaurant reservation. Technical incidents are out of scope for this question.',options(['Reserve a table','Cancel a booking']),NONE,origin='author_oos')
            else:add('routing','Which operating system is explicitly identified?',options(['Linux','Windows']),'o0' if os_name=='Linux' else 'o1' if os_name=='Windows' else NONE,origin='answer_present' if os_name else 'insufficient_evidence')
            levels=[{'id':f'l{j}','label':str(j),'criteria':f'The record explicitly states impact level {j}.','value':j} for j in range(3)]
            add('urgency','Return the explicitly documented impact level under this numeric rubric.',levels,f'l{impact}','score','urgency_v2')
            if split not in ('train','final'):
                add('conflicting_reports','Are the reports explicitly described as contradictory?',yn,'yes' if conflict else 'no','noul')
                add('owner_evidence','Is owner information explicitly present?',yn,'yes' if owner else 'no','noul')
            if split=='final':
                add('impact_rubric','Which level does the explicit impact statement support? Do not infer a level from the other facts.',levels,f'l{impact}','score','impact_v2')
                add('evidence_sufficient','Is a specific operating system named in the record?',yn,'yes' if os_name else 'no','noul')
            out.append(validate_case({'id':case_id,'group_id':'latent-'+digest(facts),'split':split,'state':state,'questions':qs,
                    'source':{'kind':'constructed','reference':'odij-oracle-facts-v2','author':'Programmatic fictional facts; not independent human annotation'},
                    'oracle_facts':{'asset':asset,'external':external,'compromise':comp,'owner':owner,'conflict':conflict,'impact':impact,'os_name':os_name}}))
    return out


def pilot_receipt(cases,p):
    return {'schema':'opendecision-exploratory-provenance/v1','protocol_sha256':digest(protocol_body(p)),
            'cases_sha256':digest(cases),'case_hashes':{c['id']:digest(c) for c in cases},
            'independent_human_review':False,'production_approval':False,
            'source_labels':'Programmatic constructed facts and optional unchanged MultiRC answer annotations.',
            'receipt_kind':'Machine consistency/provenance only; not an approval signature.'}


def pilot_receipt_errors(cases,p,r):
    errors=[]
    if p.get('scope')!='exploratory_pilot':errors.append('Exploratory receipt cannot approve a reviewed study')
    expected=pilot_receipt(cases,p)
    if r!=expected:errors.append('Exploratory provenance receipt does not match exact cases/protocol or falsely claims review')
    source=p.get('pilot_data',{})
    if source.get('generator')!='odij-oracle-facts-v2':errors.append('Unknown exploratory corpus generator')
    if source.get('combined_cases_sha256')!=digest(cases):errors.append('Pilot dataset hash differs')
    generated=[c for c in cases if c['source']['kind']=='constructed']
    if digest(generated)!=digest(build_constructed_cases(source.get('seed',0),source.get('constructed_counts'),source.get('excluded_group_ids',[]))):errors.append('Constructed cases differ from declared deterministic oracle')
    natural=[c for c in cases if c['source']['kind']=='natural']
    if natural and not source.get('multirc'):errors.append('Unregistered natural source cannot use a pilot receipt')
    if digest(natural)!=source.get('natural_cases_sha256'):errors.append('Natural annotation subset changed')
    for c in natural:
        for q in c['questions']:
            ann=q.get('source_annotation',{})
            if ann.get('dataset')!='MultiRC/SuperGLUE' or ann.get('label') not in (0,1) or q['target_id']!=('yes' if ann['label']==1 else 'no'):
                errors.append('Natural target differs from its retained MultiRC annotation')
    if p.get('approval',{}).get('approved'):errors.append('Exploratory receipt must not claim independent approval')
    return sorted(set(errors))


def prepare_pilot(intake,cfg,blocked=()):
    """Only initializes a separate new pilot intake. Existing review files are not migrated or signed."""
    intake=Path(intake);intake.mkdir(parents=True,exist_ok=True)
    if (intake/'protocol.json').exists():
        p=read_json(intake/'protocol.json')
        if p.get('scope')!='exploratory_pilot':raise GateBlocked('Pilot mode must use a separate intake; the reviewed protocol will not be changed')
        if p['pilot_data'].get('study_id')!=cfg.study_id or p['pilot_data'].get('preset')!=cfg.comparison_preset or bool(p['pilot_data'].get('multirc'))!=cfg.include_multirc:
            raise IntegrityError('Pilot controls changed. Use a new study ID and intake; do not overwrite a registered experiment')
        return p
    import torch
    seed=int(digest(['model-selection-pilot',cfg.study_id])[:8],16)
    counts={'train':64,'dev':24,'cal_fit':16,'cal_gate':16,'policy_dev':24,'final':32}
    constructed=build_constructed_cases(seed,counts,blocked);natural=[];source=None
    if cfg.include_multirc:
        records,source=fetch_multirc(intake/'source_cache')
        natural,selection=build_multirc_cases(records,seed,blocked)
        source['selection']=selection
    cases=constructed+natural
    p=protocol_template();p.update(scope='exploratory_pilot',title='Bounded model selection for a Rust backend',
      description='Exploratory author-labeled / oracle-generated pilot. No independent human approval, arbitrary-domain validity, or deployment acceptance claimed.',
      target_device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else ('CPU unit test' if cfg.allow_cpu_test else None),
      final_rubric_families=['impact_v2'],seeds=[17] if cfg.comparison_preset=='screen' else [17,29,43],
      head_epochs=12,head_patience=3,max_training_steps_per_arm=600 if cfg.comparison_preset=='screen' else 2000)
    if natural:p['train_families'].append('multirc_answer_correctness')
    for m in p['models']:
        if m['key'] in PINNED_MODELS:m['revision']=PINNED_MODELS[m['key']]
    p['arms']=['qwen2b_frozen','qwen2b_online','qwen2b_lora','qwen4b_frozen','modernbert_frozen','modernbert_full','qwen2b_finite_frozen','qwen2b_finite_lora','lexical']
    if cfg.comparison_preset=='full':p['arms'][4:4]=['qwen4b_online','qwen4b_lora']
    p['selection']={'rule':'dev_constraints_then_nll_band_then_latency','nll_band':0.02,
                    'allow_lexical_as_model':False,'missing_limits':'provisional_only',
                    'benchmark':'policy_dev_Q4_full_sequential_complete_native_response',
                    'purpose':'Choose a measured integration candidate; no post-final replacement or universal model claim.'}
    p['deployment_target']={'platform':'to_be_declared_by_owner','validation':'not_measured','colab_device_is_not_deployment_evidence':True}
    p['pilot_data']={'study_id':cfg.study_id,'preset':cfg.comparison_preset,'generator':'odij-oracle-facts-v2','seed':seed,
                     'constructed_counts':counts,'excluded_group_ids':sorted(blocked),'multirc':source,'natural_cases_sha256':digest(natural),'combined_cases_sha256':digest(cases),
                     'limitations':['Constructed splits keep distinct latent fact combinations, but rubric holdouts still share templates and rules; not genuinely novel semantic tasks.',
                                    'MultiRC adds natural binary questions, not natural Choice/Score annotation.',
                                    'No pretraining decontamination; all source hypotheses on a paragraph share a group.',
                                    'Research data/source rights need review before release.']}
    atomic_bytes(intake/'cases.jsonl',b''.join(canonical(c)+b'\n' for c in cases))
    write_json(intake/'protocol.json',p);write_json(intake/'review.json',pilot_receipt(cases,p))
    atomic_bytes(intake/'REVIEW_REQUIRED.md',b'''# Exploratory pilot, not a reviewed release\n\nThis namespace allows bounded research without fabricating independent review.\nThe provenance receipt is not an approval. All final labels are held out of fitting/selection.\nFor a release decision, create a new independently reviewed study with suitable fresh tasks,\nexplicit risk/resource bounds, and validation on the actual deployment device.\n''')
    return p


def intake_summary(intake):
    p=read_json(Path(intake)/'protocol.json');cases=load_cases(Path(intake)/'cases.jsonl')
    out={'scope':p['scope'],'device':p['target_device'],'seeds':p['seeds'],'arms':p['arms'],
         'cases':len(cases),'groups_by_split':dict(Counter(c['split'] for c in cases)),
         'source_kinds':dict(Counter(c['source']['kind'] for c in cases)),
         'unfilled_promotion_bounds':[k for k,v in p['quality_requirements'].items() if v is None],
         'human_review_claimed':p['scope']!='exploratory_pilot' and p.get('approval',{}).get('approved') is True,
         'final_text_displayed':False}
    return out
