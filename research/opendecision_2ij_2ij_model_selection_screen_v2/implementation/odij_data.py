"""Native study schema and review gate. Draft cases are constructed fixtures, never auto-approved.
Opaque IDs are excluded from rendering. All held-out families and all state variants stay grouped.
"""
from __future__ import annotations
import copy, json, re, random
from collections import Counter,defaultdict
from pathlib import Path
from odij_core import *

ORIGINS=('answer_present','annotated_intent_omitted','author_oos','insufficient_evidence')

def option_text(o):return o['label']+' — '+o['criteria']
def choices_for(q):
    opts=copy.deepcopy(q['options'])
    if q['primitive']=='choice':opts.append({'id':NONE,'label':'None of the offered options','criteria':'No offered option is justified by the state and question criteria.'})
    return opts

def target_index(q):return [o['id'] for o in choices_for(q)].index(q['target_id'])

def flatten(cases,split=None):
    out=[]
    for c in cases:
        if split is not None and c['split']!=split:continue
        for q in c['questions']:
            out.append({'id':c['id']+'/'+q['id'],'group':c['group_id'],'state_id':c['id'],'state':c['state'],
                'split':c['split'],'source_kind':c['source']['kind'],'q':q,'target':target_index(q)})
    return out

def protocol_template():
    """Settings are proposed planning values, not inherited scientific acceptance thresholds."""
    return {'schema':'opendecision-study-protocol/v1','version':VERSION,
      'title':'Reviewed multi-question transfer and model comparison','scope':'constructed_pilot',
      'description':'REVIEW REQUIRED: replace or independently review the proposed corpus. No operational generality claim.',
      'train_families':['asset','external','compromise','routing','urgency'],
      'dev_transfer_families':['conflicting_reports','owner_evidence'],
      'final_transfer_families':['impact_rubric','evidence_sufficient'],
      'final_rubric_families':['impact_v1'],
      'required_origins':list(ORIGINS),'natural_documents_required':False,
      'minimum_groups':{'train':24,'dev':8,'cal_fit':8,'cal_gate':8,'policy_dev':8,'final':8},
      'min_questions_per_state':2,'max_questions':16,'max_candidates':16,'max_tokens':1792,
      'truncation':'reject','selection_metric':'family_macro_nll',
      'target_device':None,'quality_requirements':{'max_family_macro_nll':None,'max_panel_accepted_error':None,'min_policy_coverage':None,'max_request_p95_ms':None,'max_peak_allocated_gib':None},
      'policy':{'wrong_cost':5.0,'review_cost':0.1,'threshold_grid':[0.,.5,.7,.8,.9,.95,.98,.99,1.]},
      'calibration_min_nll_gain':0.002,'probability_tolerance':0.005,'order_tolerance':1e-5,
      'seeds':[17,29,43],'head_epochs':20,'head_patience':4,'head_lr':0.001,'weight_decay':0.001,
      'online_epochs':2,'online_lr':0.0001,'adapter_lr':0.00005,'encoder_lr':0.00002,
      'lora_rank':8,'lora_alpha':16,'lora_last_blocks':4,
      'renderers':['instruction_first','state_first'],
      'rejection_models':['constant','score_summary','semantic_features'],
      'models':[
        {'key':'qwen4b','id':'Qwen/Qwen3.5-4B-Base','revision':QWEN4_REV,'kind':'causal','parameter_hint':4205751296},
        {'key':'qwen2b','id':'Qwen/Qwen3.5-2B-Base','revision':None,'kind':'causal','parameter_hint':2200000000},
        {'key':'modernbert','id':'answerdotai/ModernBERT-large','revision':None,'kind':'encoder','parameter_hint':400000000}],
      'arms':['qwen4b_frozen','qwen2b_frozen','qwen4b_online','qwen4b_lora','qwen2b_online','qwen2b_lora','modernbert_frozen','modernbert_full','qwen2b_finite_frozen','qwen2b_finite_lora','lexical'],
      'allow_hardware_skips':True,'max_training_steps_per_arm':2000,'max_gpu_minutes_per_arm':120,
      'benchmark':{'repeats':3,'state_tokens':[64,256,1024],'Q':[1,4,16],'K':[2,4,8,16],'max_grid_cells':12,'grid':[[L,Q,K] for i,L in enumerate((64,256,1024)) for Q,K in ((1,2),(4,8 if i==1 else 4),(16,2),(1,16))]},
      'approval':{'approved':False,'owner':'','independent_reviewer':'','reviewed_at':'','final_not_used_for_selection':False,'human_labels_and_criteria_reviewed':False,'hardware_and_quality_limits_accepted':False}}

def validate_case(c):
    for k in ('id','group_id','state'):
        if not nonempty(c.get(k)):raise ValueError('Case requires '+k)
    if c.get('split') not in SPLITS:raise ValueError('Unknown split')
    if len(c['state'])>100000:raise ValueError('State character bound')
    src=c.get('source',{})
    if src.get('kind') not in ('constructed','natural') or not nonempty(src.get('reference')) or not nonempty(src.get('author')):raise ValueError('Source kind/reference/author required')
    if not isinstance(c.get('questions'),list) or not 1<=len(c['questions'])<=64:raise ValueError('Question count')
    seen=set()
    for q in c['questions']:
        for k in ('id','family','rubric_family','instruction','target_id'):
            if not nonempty(q.get(k)):raise ValueError('Question requires '+k)
        if q['id'] in seen:raise ValueError('Duplicate question ID')
        seen.add(q['id'])
        if q.get('primitive') not in ('choice','noul','score'):raise ValueError('Unsupported primitive')
        opts=q.get('options',[])
        if not 2<=len(opts)<=25:raise ValueError('Candidate count')
        ids=[o.get('id') for o in opts]
        if any(not nonempty(x) for x in ids) or len(set(ids))!=len(ids) or NONE in ids:raise ValueError('Candidate IDs must be unique, nonempty and opaque; internal none is reserved')
        if any(not nonempty(o.get('label')) or not nonempty(o.get('criteria')) for o in opts):raise ValueError('Every option requires semantic label and criteria, not only an ID')
        if len({option_text(o).strip().casefold() for o in opts})!=len(opts):raise ValueError('Identical candidate meanings require adjudication')
        if q['target_id'] not in ids+[NONE]:raise ValueError('Target is not a declared option')
        if q.get('origin') not in ORIGINS:raise ValueError('Explicit answerability origin required')
        if q['primitive']!='choice' and (q['target_id']==NONE or q['origin']!='answer_present'):raise ValueError('Noul/Score do not silently acquire a none outcome; use a separate evidence question')
        if q['primitive']=='choice' and ((q['target_id']==NONE)!=(q['origin']!='answer_present')):raise ValueError('None origin and target disagree')
        if q['primitive']=='noul' and (len(opts)!=2 or [o.get('value') for o in opts]!=[False,True]):raise ValueError('Noul options must have values false, true in that order')
        if q['primitive']=='score':
            v=[o.get('value') for o in opts]
            if any(not isinstance(x,(float,int)) or isinstance(x,bool) or not math.isfinite(x) for x in v) or v!=sorted(set(v)):raise ValueError('Score requires distinct ascending finite level values')
        # This checks declared evidence locations, not semantic truth.
        for quote in q.get('evidence_quotes',[]):
            if not nonempty(quote) or quote not in c['state']:raise ValueError('Evidence quote not present in state')
    return c

def load_cases(path):
    out=[]
    with open(path,encoding='utf-8') as f:
        for i,line in enumerate(f,1):
            if line.strip():
                try:out.append(validate_case(json.loads(line)))
                except Exception as e:raise ValueError(f'Invalid case at line {i}: {e}') from e
    if not out:raise ValueError('Empty corpus')
    return out

def protocol_body(p):return {k:v for k,v in p.items() if k!='approval'}

def validate_study(cases,p,blocked_hashes=()):
    """Fail closed. Returned violations are repair instructions, never implicit approvals."""
    errors=[]; add=errors.append
    ids=[c['id'] for c in cases]
    if len(ids)!=len(set(ids)):add('Duplicate case IDs')
    groups=defaultdict(set);texts=defaultdict(set)
    for c in cases:
        groups[c['group_id']].add(c['split']);texts[text_hash(c['state'])].add(c['split'])
        if text_hash(c['state']) in blocked_hashes or c['group_id'] in blocked_hashes:add('Historical/excluded state reused: '+c['id'])
        if len(c['questions'])<p.get('min_questions_per_state',2):add('Not multi-question: '+c['id'])
    if any(len(s)>1 for s in groups.values()):add('Related group crosses splits')
    if any(len(s)>1 for s in texts.values()):add('Exact normalized state crosses splits')
    tf=set(p.get('train_families',[]));df=set(p.get('dev_transfer_families',[]));ff=set(p.get('final_transfer_families',[]));rf=set(p.get('final_rubric_families',[]))
    if not tf or not df or not ff or tf&df or tf&ff or df&ff:add('Train/dev-transfer/final-transfer families must be explicit, nonempty and disjoint')
    if not rf:add('A final rubric holdout must be declared')
    rows=flatten(cases)
    for s in SPLITS:
        rows_s=[r for r in rows if r['split']==s];gs={r['group'] for r in rows_s}
        if len(gs)<p.get('minimum_groups',{}).get(s,1):add('Insufficient groups in '+s)
        fam={r['q']['family'] for r in rows_s}
        allowed=tf|(ff if s=='final' else (df if s!='train' else set()))
        if fam-allowed:add('Unregistered or held-out family in '+s+': '+str(sorted(fam-allowed)))
        if s=='dev' and not df.issubset(fam):add('Development lacks intended transfer families')
        if s=='final' and not ff.issubset(fam):add('Final lacks declared held-out families')
        for r in rows_s:
            if r['q']['rubric_family'] in rf and s!='final':add('Final rubric exposed in '+s)
            if len(r['q']['options'])>p.get('max_candidates',16):add('Candidate work bound exceeded')
        if s=='final' and not rf.issubset({r['q']['rubric_family'] for r in rows_s}):add('Final rubric holdouts absent')
    for s in ('train','dev','final'):
        have={r['q']['origin'] for r in rows if r['split']==s}
        if set(p.get('required_origins',ORIGINS))-have:add('Missing separate answerability strata in '+s)
    if p.get('natural_documents_required') and not any(c['source']['kind']=='natural' and c['split']=='final' for c in cases):add('Independent natural-document final cases missing')
    if p.get('scope')=='operational' and not p.get('natural_documents_required'):add('Operational scope cannot bypass natural-document requirement')
    if p.get('scope') not in ('constructed_pilot','operational','exploratory_pilot'):add('Scope must be constructed_pilot or operational')
    if p.get('truncation')!='reject':add('Only explicit reject-on-overlength is implemented')
    if p.get('selection_metric')!='family_macro_nll':add('Unsupported preselection metric')
    if not nonempty(p.get('target_device')):add('Target GPU/device name must be specified before selection')
    exploratory=p.get('scope')=='exploratory_pilot'
    qr=p.get('quality_requirements',{})
    for k in ('max_family_macro_nll','max_panel_accepted_error','min_policy_coverage','max_request_p95_ms','max_peak_allocated_gib'):
        v=qr.get(k)
        if v is None and exploratory:continue
        if not isinstance(v,(int,float)) or isinstance(v,bool) or not math.isfinite(v) or v<0:add('Set a finite predeclared quality/resource bound: '+k)
    for k in ('max_panel_accepted_error','min_policy_coverage'):
        if isinstance(qr.get(k),(int,float)) and not 0<=qr[k]<=1:add('Probability bound outside [0,1]: '+k)
    if not exploratory:
        a=p.get('approval',{})
        for k in ('approved','final_not_used_for_selection','human_labels_and_criteria_reviewed','hardware_and_quality_limits_accepted'):
            if a.get(k) is not True:add('Review attestation required: '+k)
        for k in ('owner','independent_reviewer','reviewed_at'):
            if not nonempty(a.get(k)):add('Reviewer identity/date required: '+k)
        if a.get('owner')==a.get('independent_reviewer'):add('Independent reviewer must differ from protocol owner')
    if not p.get('seeds') or any(not isinstance(s,int) for s in p['seeds']) or len(set(p['seeds']))!=len(p['seeds']):add('Explicit unique integer seeds required')
    if p.get('selection'):
        rule=p['selection'];band=rule.get('nll_band')
        if rule.get('rule')!='dev_constraints_then_nll_band_then_latency':add('Unknown explicit model selection rule')
        if not isinstance(band,(int,float)) or isinstance(band,bool) or not math.isfinite(band) or band<0:add('Selection NLL band must be finite and nonnegative')
    allowed_arms={'qwen4b_frozen','qwen2b_frozen','qwen4b_online','qwen4b_lora','qwen2b_online','qwen2b_lora','modernbert_frozen','modernbert_full','qwen2b_finite_frozen','qwen2b_finite_lora','lexical'}
    arms=p.get('arms',[])
    if not arms or len(set(arms))!=len(arms) or set(arms)-allowed_arms:add('Declare a unique implemented arm matrix')
    for key in ('head_epochs','head_patience','online_epochs','max_training_steps_per_arm','lora_rank','lora_last_blocks','max_tokens','max_candidates','max_questions'):
        if not isinstance(p.get(key),int) or isinstance(p.get(key),bool) or p[key]<1:add('Positive integer budget required: '+key)
    for key in ('head_lr','online_lr','adapter_lr','encoder_lr','lora_alpha','max_gpu_minutes_per_arm'):
        v=p.get(key)
        if not isinstance(v,(int,float)) or isinstance(v,bool) or not math.isfinite(v) or v<=0:add('Positive finite budget required: '+key)
    if not set(p.get('renderers',[])).issubset({'instruction_first','state_first'}) or not p.get('renderers'):add('Declared supported renderers required')
    if not set(p.get('rejection_models',[])).issubset({'constant','score_summary','semantic_features'}) or not p.get('rejection_models'):add('Declared supported rejection controls required')
    if 'qwen4b_lora' in arms and 'qwen4b_online' not in arms:add('4B LoRA requires matched online-head control')
    if set(arms)&{'qwen4b_lora','qwen4b_online'} and 'qwen4b_frozen' not in arms:add('Online pair requires the fresh frozen rank initializer')
    if 'qwen2b_lora' in arms and 'qwen2b_online' not in arms:add('2B LoRA requires matched online-head control')
    if set(arms)&{'qwen2b_lora','qwen2b_online'} and 'qwen2b_frozen' not in arms:add('2B online pair requires fresh frozen initializer')
    keys=[m.get('key') for m in p.get('models',[])]
    if len(keys)!=len(set(keys)):add('Duplicate model keys')
    for arm in arms:
        if arm!='lexical' and not any(arm.startswith(str(k)+'_') for k in keys):add('Model spec missing for arm '+arm)
    for c in cases:
        if len(c['questions'])>p.get('max_questions',16):add('Case exceeds registered Q bound')
    # Models are resolved before registration, never left on mutable main.
    for m in p.get('models',[]):
        if not re.fullmatch(r'[0-9a-f]{40}',m.get('revision') or ''):add('Pin immutable model revision: '+m['key'])
    return sorted(set(errors))

def review_template(cases,p):
    return {'schema':'opendecision-review/v1','protocol_sha256':digest(protocol_body(p)),
        'cases_sha256':digest(cases),'approved':False,'reviewer':'','reviewed_at':'',
        'case_hashes':{c['id']:digest(c) for c in cases},
        'labels_and_criteria_checked':False,'all_cases_independently_reviewed':False,
        'final_cases_kept_out_of_model_selection':False,
        'limitation':'An attestation is not a verified reviewer identity or cryptographic secrecy.'}

def review_errors(cases,p,review):
    if p.get('scope')=='exploratory_pilot':
        from odij_intake import pilot_receipt_errors
        return pilot_receipt_errors(cases,p,review)
    errors=[]
    if review.get('cases_sha256')!=digest(cases):errors.append('Review refers to different cases')
    if review.get('protocol_sha256')!=digest(protocol_body(p)):errors.append('Review refers to a different protocol (including model revisions)')
    if review.get('case_hashes')!={c['id']:digest(c) for c in cases}:errors.append('Case-level review manifest changed')
    for k in ('approved','labels_and_criteria_checked','all_cases_independently_reviewed','final_cases_kept_out_of_model_selection'):
        if review.get(k) is not True:errors.append('Missing review decision: '+k)
    if not nonempty(review.get('reviewer')) or review.get('reviewer')!=p.get('approval',{}).get('independent_reviewer'):errors.append('Reviewer does not match protocol')
    if not nonempty(review.get('reviewed_at')):errors.append('Review timestamp required')
    return errors

def write_draft(intake:Path):
    """Useful, explicitly synthetic cases; users supply/approve real cases before a study runs."""
    intake.mkdir(parents=True,exist_ok=True)
    if (intake/'cases.jsonl').exists() or (intake/'protocol.json').exists():return
    p=protocol_template();cases=[];rng=random.Random(906191)
    counts={'train':32,'dev':12,'cal_fit':12,'cal_gate':12,'policy_dev':12,'final':12}
    def options(labels):return [{'id':f'o{i}','label':t,'criteria':t} for i,t in enumerate(labels)]
    for sp,n in counts.items():
        for i in range(n):
            serial=f'{sp}-{i:04d}';case_marker=f'{rng.getrandbits(48):012x}';asset=rng.choice(['workstation','application server','network appliance']);ext=rng.choice([True,False]);comp=rng.choice([True,False]);owner=rng.choice([True,False]);conflict=rng.choice([True,False]);sev=rng.randrange(3)
            state=(f'Fictional incident record {case_marker}. Affected asset: {asset}. '
              f'An external destination was {"observed" if ext else "not observed"}. '
              f'Compromise was {"confirmed" if comp else "not confirmed"}. '
              f'Owner information is {"present" if owner else "missing"}. '
              f'The analyst reports are {"contradictory" if conflict else "consistent"}. '
              f'The documented impact level is {sev}. These statements are the full record; do not infer unrecorded facts.')
            qs=[]
            def q(fam,instr,opts,t,prim='choice',rub=None,origin='answer_present'):
                qs.append({'id':fam,'family':fam,'rubric_family':rub or fam+'_v1','primitive':prim,'instruction':instr,'options':opts,'target_id':t,'origin':origin,'evidence_quotes':[]})
            op=options(['workstation','application server','network appliance']);q('asset','Select the explicitly stated affected asset type.',op,op[['workstation','application server','network appliance'].index(asset)]['id'])
            yn=[{'id':'n','label':'No','criteria':'The proposition is explicitly false.','value':False},{'id':'y','label':'Yes','criteria':'The proposition is explicitly true.','value':True}]
            q('external','Does the record explicitly say that an external destination was observed?',copy.deepcopy(yn),'y' if ext else 'n','noul')
            q('compromise','Does the record explicitly say that compromise was confirmed?',copy.deepcopy(yn),'y' if comp else 'n','noul')
            # Dedicated omission/OOS/insufficiency examples are declared constructed; no author-OOS corpus claim.
            mode=i%4
            if mode==0:q('routing','Choose a category that applies to this incident record.',options(['A documented incident','A personal travel request']),'o0')
            elif mode==1:q('routing','Choose the explicitly stated asset type. No offered category may be invented.',options(['mobile handset','printer']),NONE,origin='annotated_intent_omitted')
            elif mode==2:q('routing','Choose the documented restaurant-reservation intent; technical incidents are out of this task scope.',options(['Reserve a table','Cancel a dining reservation']),NONE,origin='author_oos')
            else:q('routing','Choose the affected asset operating system only if the record states it.',options(['Linux','Windows']),NONE,origin='insufficient_evidence')
            levels=[{'id':f'l{j}','label':str(j),'criteria':f'The record explicitly states impact level {j}.','value':j} for j in range(3)]
            q('urgency','Use the explicit impact level as this bounded three-level urgency rubric; do not invent severity.',levels,f'l{sev}','score','urgency_v1')
            if sp not in ('train','final'):
                q('conflicting_reports','Are the analyst reports explicitly described as contradictory?',copy.deepcopy(yn),'y' if conflict else 'n','noul')
                q('owner_evidence','Is owner information explicitly present?',copy.deepcopy(yn),'y' if owner else 'n','noul')
            if sp=='final':
                q('impact_rubric','Apply the rubric levels to the explicitly documented impact, not inferred likelihood.',copy.deepcopy(levels),f'l{sev}','score','impact_v1')
                q('evidence_sufficient','Is the record sufficient to identify an operating system? Sufficient means a named operating system appears in the record.',copy.deepcopy(yn),'n','noul')
            c={'id':serial,'group_id':'synthetic-'+serial,'split':sp,'state':state,'source':{'kind':'constructed','reference':'workbench deterministic fictional fixture generator v1','author':'OpenDecision fixture generator; not an independently annotated operational source'},'questions':qs}
            cases.append(validate_case(c))
    atomic_bytes(intake/'cases.jsonl',b''.join(canonical(c)+b'\n' for c in cases));write_json(intake/'protocol.json',p);write_json(intake/'review.json',review_template(cases,p))
    atomic_bytes(intake/'REVIEW_REQUIRED.md',('''# Review required before new model training

These cases are **constructed draft fixtures**, not a collected or independently reviewed dataset.
The `author_oos` tag here is a declared out-of-task construction, not evidence from CLINC or a real author-OOS corpus.
Replace cases.jsonl with reviewed natural/mixed examples for operational scope. Preserve the schema and split boundaries.

1. Review task definitions, candidates, labels, family/rubric holdouts and source grouping.
2. Supply target_device and all quality/resource requirements in protocol.json; review the proposed budgets.
3. Run the notebook's revision-resolution and review-template refresh cell. Revisions become immutable before approval.
4. Have a distinct reviewer inspect all cases and record the requested attestations in review.json and protocol.json.
5. Run all again with the SAME study_id. A registered study's inputs cannot change; a changed design requires a new study_id.

No approval is prefilled. These attestations are workflow checks, not identity verification or protection against opening final files manually.
The notebook never prints final question text before model lock. An independent data reviewer may inspect final annotations without using them for model selection.
''').encode())


def prepare_review_template(intake):
    p=read_json(intake/'protocol.json');cases=load_cases(intake/'cases.jsonl')
    out=intake/'review_template.refreshed.json';write_json(out,review_template(cases,p));return out


def register_study(root:Path,intake:Path,blocked_hashes=(),source_hash=SOURCE_SHA256):
    p=read_json(intake/'protocol.json');cases=load_cases(intake/'cases.jsonl');review=read_json(intake/'review.json')
    errors=validate_study(cases,p,blocked_hashes)+review_errors(cases,p,review)
    if errors:write_json(root/'gate_report.json',{'status':'blocked','requirements':sorted(set(errors))});return None
    source_code={q.name:file_sha(q) for q in sorted(Path(__file__).parent.glob('odij_*.py'))}
    identity={'version':VERSION,'protocol':digest(p),'cases':digest(cases),'review':digest(review),'historical_archive':source_hash,'implementation':source_code}
    lock=root/'study_lock.json'
    if lock.exists():
        if read_json(lock)['identity']!=identity:raise IntegrityError('Registered study changed. Preserve it and choose a NEW study_id; never replace final results.')
        assert_study_intact(root)
    else:
        # Full cases may be validated mechanically before the lock. Modeling accesses final only after model lock.
        write_json(root/'registered_protocol.json',p)
        for s in SPLITS:write_json(root/'data'/f'{s}.json',[c for c in cases if c['split']==s])
        write_json(root/'review_attestation.json',review)
        write_json(lock,{'created_at':utc(),'identity':identity,'files':tree_hashes(root/'data')})
    write_json(root/'gate_report.json',{'status':'passed','scope':p['scope'],'identity':digest(identity),'human_review':p.get('scope')!='exploratory_pilot','attestation_limit':'Pilot receipt checks provenance/structure only; independent human review is not asserted. Reviewed attestations are workflow records, not identity verification.'})
    return p


def assert_study_intact(root):
    lock=read_json(root/'study_lock.json')
    if digest(read_json(root/'registered_protocol.json'))!=lock['identity']['protocol']:raise IntegrityError('Registered protocol changed')
    if digest(read_json(root/'review_attestation.json'))!=lock['identity']['review']:raise IntegrityError('Review attestation changed')
    for n,h in lock['files'].items():
        if file_sha(root/'data'/n)!=h:raise IntegrityError('Registered split changed: '+n)
    actual={q.name:file_sha(q) for q in sorted(Path(__file__).parent.glob('odij_*.py'))}
    if actual!=lock['identity']['implementation']:raise IntegrityError('Study implementation changed; register a new study rather than mixing results')
    return lock


def previous_final_exclusions(drive_root,study_id):
    """Read-only exact-identity exclusions from previously exposed workbench finals.
    Single-coordinator convention, not distributed locking or paraphrase decontamination.
    """
    blocked=set()
    for path in sorted((Path(drive_root)/'_final_exposures').glob('*.json')):
        m=read_json(path)
        if m.get('study_id')==study_id:continue
        if m.get('schema')!='opendecision-final-exposure/v1':raise IntegrityError('Unknown final-exposure manifest')
        blocked.update(m['group_ids']);blocked.update(m['normalized_state_hashes'])
    return blocked


def record_final_exposure(root,drive_root,study_id):
    """Conservatively reserve even if the subsequent final worker is interrupted."""
    lock=root/'MODEL_LOCK.json'
    if not lock.exists():raise IntegrityError('Cannot expose final before model lock')
    cases=read_json(root/'data/final.json')
    m={'schema':'opendecision-final-exposure/v1','study_id':study_id,'model_lock_sha256':file_sha(lock),
       'group_ids':sorted({c['group_id'] for c in cases}),
       'normalized_state_hashes':sorted({text_hash(c['state']) for c in cases})}
    path=Path(drive_root)/'_final_exposures'/(study_id+'.json')
    if path.exists() and read_json(path)!=m:raise IntegrityError('Exposed final/model identity cannot be replaced')
    write_json(path,m)
