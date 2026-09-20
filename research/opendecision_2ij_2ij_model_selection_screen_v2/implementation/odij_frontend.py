"""Notebook preflight, safe intake editing, and explicit human review UI.
No human approval is manufactured. Exploratory provenance and reviewed studies
are separate scopes/identities. Final annotation review requires an explicit UI role.
"""
from __future__ import annotations
import copy, html, math, re
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from odij_core import *
from odij_data import *


def config_summary(intake):
    intake=Path(intake);p=read_json(intake/'protocol.json');cases=load_cases(intake/'cases.jsonl')
    rows=flatten(cases)
    return {'scope':p['scope'],'target_device':p.get('target_device'),
      'independent_review_claimed':p.get('approval',{}).get('approved',False),
      'cases_by_split':dict(Counter(c['split'] for c in cases)),
      'questions_by_split':dict(Counter(r['split'] for r in rows)),
      'questions_by_primitive':dict(Counter(r['q']['primitive'] for r in rows)),
      'source_kinds':dict(Counter(c['source']['kind'] for c in cases)),
      'quality_requirements':p['quality_requirements'],'selection':p.get('selection',{'rule':'family_macro_nll'}),
      'arms':p['arms'],'seeds':p['seeds'],'renderers':p['renderers'],
      'protocol_sha256':digest(protocol_body(p)),'cases_sha256':digest(cases),
      'limits':'Metadata only. No final input/label text is displayed; no study is approved by this summary.'}


def preflight_inputs(intake,output,tokenizer_loader=None,allow_cpu_test=False):
    """Common input-length contract for every model arm, before costly fitting.
    This validates final *shape* under frozen rules, not semantic labels or outcomes.
    Returns counts/max lengths only; never prints final stimuli. Overlength rejects
    the common study; it never silently truncates, drops examples, or alters criteria.
    """
    import torch
    from odij_runtime import segments,joint_ids,finite_ids
    p=read_json(Path(intake)/'protocol.json');cases=load_cases(Path(intake)/'cases.jsonl');rows=flatten(cases)
    ident={'protocol':digest(protocol_body(p)),'cases':digest(cases),'implementation':file_sha(__file__),
           'renderer':file_sha(Path(__file__).parent/'odij_runtime.py'),'environment':env_identity()}
    path=Path(output)/'input_preflight.json'
    if path.exists():
        previous=read_json(path)
        if previous.get('identity')==ident and previous.get('status')=='passed':return previous
    problems=[];reports=[]
    if tokenizer_loader is None:
        from transformers import AutoTokenizer
        tokenizer_loader=lambda spec:AutoTokenizer.from_pretrained(spec['id'],revision=spec['revision'],trust_remote_code=False,token=os.environ.get('HF_TOKEN'))
    for spec in p['models']:
        arms=[a for a in p['arms'] if a.startswith(spec['key']+'_')]
        if not arms:continue
        token=tokenizer_loader(spec);variants=set()
        if spec['kind']=='encoder':variants.add('joint')
        else:
            if any('_finite_' not in a for a in arms):variants.update(p['renderers'])
            if any('_finite_' in a for a in arms):variants.add('finite')
        for layout in sorted(variants):
            lengths=defaultdict(list);errors=Counter()
            for r in rows:
                try:
                    if layout=='joint':ids,_=joint_ids(r,token,p['max_tokens']);n=len(ids)
                    elif layout=='finite':ids,_=finite_ids(r,token,p['max_tokens']);n=len(ids)
                    else:_,_,_,ids=segments(r,token,layout,p['max_tokens']);n=max(map(len,ids))
                    lengths[r['split']].append(n)
                except (ValueError,GateBlocked) as e:
                    errors[r['split']]+=1
            reports.append({'model':spec['id'],'revision':spec['revision'],'layout':layout,
              'max_tokens_by_split':{s:max(v) for s,v in lengths.items()},'questions_by_split':{s:len(v) for s,v in lengths.items()},
              'rejected_counts_by_split':dict(errors)})
            if errors:problems.append(spec['key']+'/'+layout+': overlength or unsupported encoding; inspect nonfinal cases/renderer, do not silently filter the final set')
    device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    if not allow_cpu_test:
        if device is None:problems.append('CUDA GPU is required for real model workers')
        elif str(p.get('target_device') or '').casefold() not in device.casefold():problems.append('Runtime differs from declared target_device')
        if not torch.__version__.startswith('2.11.'):problems.append('Real model workers require Torch 2.11.x; do not silently replace Torch/CUDA')
        import importlib.metadata as im
        if im.version('transformers')!='5.17.0':problems.append('Real model workers require Transformers 5.17.0')
    result={'status':'blocked' if problems else 'passed','identity':ident,'models':reports,'requirements':problems,
      'gpu':device,'final_inputs':'Only encoding/length checked before training; no final prediction or label-based tuning',
      'common_data_kept':True,'truncation':'reject'}
    write_json(path,result);return result


def configure_intake(intake,study_root,updates):
    """Explicit owner edits before registration. Never invalidate a live registered study silently."""
    intake,study_root=Path(intake),Path(study_root)
    if (study_root/'study_lock.json').exists():raise IntegrityError('Registered inputs are immutable; a changed design needs a new study ID/intake')
    allowed={'target_device','quality_requirements','arms','seeds','renderers','selection','deployment_target',
      'max_tokens','max_training_steps_per_arm','head_epochs','head_patience','online_epochs','benchmark'}
    if set(updates)-allowed:raise ValueError('Unsupported preflight edit; change task data through a separately reviewed intake')
    p=read_json(intake/'protocol.json');before=copy.deepcopy(p)
    p.update(copy.deepcopy(updates))
    for k in p['approval']:
        p['approval'][k]='' if k in ('owner','independent_reviewer','reviewed_at') else False
    cases=load_cases(intake/'cases.jsonl')
    errors=validate_study(cases,p)
    relevant=[e for e in errors if not ('attestation' in e or 'Reviewer' in e or 'Independent reviewer' in e)]
    if relevant:raise GateBlocked('; '.join(relevant))
    if before==p:return p
    backup=intake/'edit_history'/('protocol-'+digest(before)+'.json')
    if not backup.exists():write_json(backup,before)
    write_json(intake/'protocol.json',p)
    if p['scope']=='exploratory_pilot':
        from odij_intake import pilot_receipt
        write_json(intake/'review.json',pilot_receipt(cases,p))
    else:
        prepare_review_template(intake)
        # Previous approvals must not survive changed semantics or device/limits.
        write_json(intake/'review.json',read_json(intake/'review_template.refreshed.json'))
    return p


def record_case_review(intake,case_id,reviewer,accepted,comment='',allow_final=False):
    """A real user action records the current case hash; edits invalidate the mark."""
    intake=Path(intake);p=read_json(intake/'protocol.json')
    if p['scope']=='exploratory_pilot':raise GateBlocked('Exploratory receipts are not human review. Use a separate reviewed intake to claim independent approval')
    if not nonempty(reviewer):raise ValueError('Reviewer name is required')
    case=next(c for c in load_cases(intake/'cases.jsonl') if c['id']==case_id)
    if case['split']=='final' and not allow_final:raise GateBlocked('Final annotation view is restricted to an explicitly acknowledged independent reviewer role')
    path=intake/'case_review_progress.json';v=read_json(path) if path.exists() else {'schema':'opendecision-case-review-progress/v1','cases':{}}
    v['cases'][case_id]={'case_sha256':digest(case),'reviewer':reviewer.strip(),'accepted':bool(accepted),'comment':comment,'reviewed_at':utc()}
    write_json(path,v);return v


def sign_review(intake,study_root,owner,reviewer,attest_final_unused=False,attest_limits=False):
    intake,study_root=Path(intake),Path(study_root)
    if (study_root/'study_lock.json').exists():raise IntegrityError('Cannot alter review after study registration')
    p=read_json(intake/'protocol.json');cases=load_cases(intake/'cases.jsonl')
    if p['scope']=='exploratory_pilot':raise GateBlocked('Cannot relabel an exploratory pilot as independently reviewed')
    owner,reviewer=owner.strip(),reviewer.strip()
    if not owner or not reviewer or owner==reviewer:raise ValueError('Owner and distinct independent reviewer required')
    marks=read_json(intake/'case_review_progress.json')['cases'] if (intake/'case_review_progress.json').exists() else {}
    missing=[c['id'] for c in cases if not (marks.get(c['id'],{}).get('accepted') is True and marks[c['id']]['case_sha256']==digest(c) and marks[c['id']]['reviewer']==reviewer)]
    if missing:raise GateBlocked(f'{len(missing)} cases lack a current accepted review by this reviewer')
    if not attest_final_unused or not attest_limits:raise GateBlocked('Final holdout and hardware/limits attestations must be explicitly provided')
    stamp=utc();p['approval']={'approved':True,'owner':owner,'independent_reviewer':reviewer,'reviewed_at':stamp,
      'final_not_used_for_selection':True,'human_labels_and_criteria_reviewed':True,'hardware_and_quality_limits_accepted':True}
    errors=validate_study(cases,p)
    if errors:raise GateBlocked('; '.join(errors))
    review=review_template(cases,p);review.update(approved=True,reviewer=reviewer,reviewed_at=stamp,
      labels_and_criteria_checked=True,all_cases_independently_reviewed=True,final_cases_kept_out_of_model_selection=True)
    write_json(intake/'protocol.json',p);write_json(intake/'review.json',review)
    return {'status':'recorded_human_attestation','identity_verified':False,'case_count':len(cases),'protocol_sha256':review['protocol_sha256']}


def review_ui(intake,study_root):
    """Display optional annotation review in-place in Colab. Run-all never clicks/signs."""
    import ipywidgets as w
    from IPython.display import display,clear_output
    intake=Path(intake);p=read_json(intake/'protocol.json')
    if p['scope']=='exploratory_pilot':
        display(w.HTML('<b>Exploratory model-selection mode.</b> No human approval is claimed. Use the optional configuration cell before registering; a release confirmation uses a separate reviewed intake.'))
        return
    cases=load_cases(intake/'cases.jsonl');by_id={c['id']:c for c in cases}
    owner=w.Text(description='Owner');reviewer=w.Text(description='Reviewer')
    final_role=w.Checkbox(value=False,description='Independent annotation reviewer: allow final case inspection')
    picker=w.Dropdown(description='Case',options=[c['id'] for c in cases if c['split']!='final'])
    body=w.Output();message=w.Output();comment=w.Textarea(description='Notes');accept=w.Button(description='Accept this case');reject=w.Button(description='Needs correction')
    final_unused=w.Checkbox(value=False,description='Final cases have not been used for model selection')
    limits=w.Checkbox(value=False,description='I reviewed and accept the protocol, target and quality/resource limits')
    sign=w.Button(description='Record completed independent review',layout=w.Layout(width='310px'))
    def render(*_):
        with body:
            clear_output(wait=True)
            c=by_id[picker.value]
            display(w.HTML('<pre style="white-space:pre-wrap">'+html.escape(json.dumps(c,indent=2))+'</pre>'))
    def role_change(*_):
        picker.options=[c['id'] for c in cases if c['split']!='final' or final_role.value]
        render()
    def mark(ok):
        with message:
            clear_output(wait=True)
            try:
                record_case_review(intake,picker.value,reviewer.value,ok,comment.value,final_role.value)
                print('Recorded current-case review; this is not final protocol approval.')
            except Exception as e:print(type(e).__name__+': '+str(e))
    def finish(_):
        with message:
            clear_output(wait=True)
            try:print(sign_review(intake,study_root,owner.value,reviewer.value,final_unused.value,limits.value))
            except Exception as e:print(type(e).__name__+': '+str(e))
    picker.observe(render,names='value');final_role.observe(role_change,names='value')
    accept.on_click(lambda _:mark(True));reject.on_click(lambda _:mark(False));sign.on_click(finish)
    display(w.VBox([w.HTML('<b>Independent review.</b> Inspect each case and the protocol. No approve-all control is provided.'),owner,reviewer,final_role,picker,body,comment,w.HBox([accept,reject]),final_unused,limits,sign,message]));render()
