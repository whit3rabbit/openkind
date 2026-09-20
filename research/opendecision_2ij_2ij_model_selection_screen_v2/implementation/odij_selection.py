"""A model decision, not an endless leaderboard. Selection never reads final results.
New-mode selection: apply supplied nonfinal bounds, retain an NLL band, choose lower
measured Q<=4 p95 latency (then memory/NLL/ID). Missing limits permit a provisional
research candidate only; they never become a passed release gate.
"""
from __future__ import annotations
import math
from pathlib import Path
from odij_core import *


def point_checks(metrics,policy,benchmark,bounds):
    values={'max_family_macro_nll':metrics.get('family_macro_nll'),
            'max_panel_accepted_error':policy.get('panel_error_among_accepted'),
            'min_policy_coverage':policy.get('family_group_weighted_coverage'),
            'max_request_p95_ms':benchmark.get('p95_ms'),
            'max_peak_allocated_gib':benchmark.get('memory',{}).get('peak_allocated_gib')}
    checks={}
    for k,v in values.items():
        bound=bounds.get(k)
        if bound is None:checks[k]=None
        elif v is None or not isinstance(v,(int,float)) or not math.isfinite(v):checks[k]=False
        else:checks[k]=bool(v>=bound if k.startswith('min_') else v<=bound)
    return checks,values


def choose_candidate(profiles,p):
    rule=p.get('selection')
    if not rule:
        # Legacy independently reviewed protocol retains its exact declared preselection.
        best=min(profiles,key=lambda m:(m['development']['family_macro_nll'],m['profile_id']))
        return best,{'rule':'legacy_family_macro_nll','selected_on':'dev','uses_final':False}
    if rule.get('rule')!='dev_constraints_then_nll_band_then_latency':raise GateBlocked('Unknown model-selection rule')
    eps=rule.get('nll_band',0.)
    if not isinstance(eps,(int,float)) or not math.isfinite(eps) or eps<0:raise GateBlocked('Invalid predeclared NLL band')
    rows=[];eligible=[]
    for m in profiles:
        checks,values=point_checks(m['development'],m['policy']['selected'],m.get('benchmark',{}),p['quality_requirements'])
        reasons=[]
        if m['method']=='lexical' and not rule.get('allow_lexical_as_model',False):reasons.append('lexical_control_not_neural_deployment_candidate')
        reasons += ['nonfinal_limit:'+k for k,v in checks.items() if v is False]
        if not math.isfinite(m['development'].get('family_macro_nll',float('inf'))):reasons.append('nonfinite_development_score')
        latency=m.get('benchmark',{}).get('p95_ms')
        if not isinstance(latency,(int,float)) or not math.isfinite(latency) or latency<=0:reasons.append('missing_or_invalid_measured_latency')
        rows.append({'profile_id':m['profile_id'],'nonfinal_checks':checks,'nonfinal_values':values,'excluded_reasons':reasons})
        if not reasons:eligible.append(m)
    audit={'rule':rule,'selected_on':['dev','policy_dev','nonfinal_Q4_benchmark'],'uses_final':False,'eligibility':rows,
           'missing_limits':[k for k,v in p['quality_requirements'].items() if v is None],
           'timing_scope':'Resident Q<=4 full-sequential offline requests; Q=1 diagnostic retained. Not optimized Q scaling or service p95.'}
    if not eligible:
        audit['status']='no_candidate_meets_declared_nonfinal_requirements';return None,audit
    best_nll=min(m['development']['family_macro_nll'] for m in eligible)
    band=[m for m in eligible if m['development']['family_macro_nll']<=best_nll+eps]
    def key(m):
        b=m['benchmark'];mem=b.get('memory',{}).get('peak_allocated_gib')
        return b['p95_ms'],mem if mem is not None else float('inf'),m['development']['family_macro_nll'],m['profile_id']
    selected=min(band,key=key)
    audit.update(status='candidate_selected_before_final',best_nonfinal_nll=best_nll,band_profile_ids=[m['profile_id'] for m in band],selected_profile=selected['profile_id'])
    return selected,audit


def decision_report(root):
    root=Path(root);lock_path=root/'MODEL_LOCK.json'
    if not lock_path.exists():
        result={'status':'pending_model_comparison','selected_profile':None,'production_ready':False}
    else:
        lock=read_json(lock_path);p=read_json(root/'registered_protocol.json');pid=lock['selected_profile']
        profile=next((m for m in lock['profiles'] if m['profile_id']==pid),None)
        final_path=root/'final'/str(pid)/'FINAL_DONE.json';final=read_json(final_path) if final_path.exists() else None
        all_final=all((root/'final'/m['profile_id']/'FINAL_DONE.json').exists() for m in lock['profiles'])
        missing=[k for k,v in p['quality_requirements'].items() if v is None]
        reasons=[]
        if p['scope']=='exploratory_pilot':reasons.append('exploratory_data_not_independently_reviewed')
        if missing:reasons.append('promotion_bounds_not_fully_declared')
        if not all_final:reasons.append('one_or_more_required_final_comparisons_pending')
        if final and not final.get('meets_registered_point_estimate_limits',False):reasons.append('final_acceptance_not_established')
        if profile is None:status='no_candidate_meets_declared_nonfinal_requirements'
        elif final is None:status='candidate_locked_final_pending'
        elif p['scope']=='exploratory_pilot' or missing:status='provisional_integration_candidate'
        elif final['meets_registered_point_estimate_limits']:status='passes_declared_offline_point_limits'
        else:status='selected_candidate_did_not_pass_final_limits'
        result={'schema':'opendecision-model-decision/v1','status':status,'selected_profile':pid,
                'selected_model':profile.get('spec',{}).get('id') if profile else None,
                'selected_layout':profile.get('layout') if profile else None,
                'selected_rejection':profile.get('kind') if profile else None,
                'scope':p['scope'],'all_final_comparisons_complete':all_final,
                'selection':lock.get('selection_audit'),'selected_final':final,
                'production_ready':False,'deployment_target':p.get('deployment_target',{'validation':'not_measured'}),
                'reasons_not_release_ready':reasons+['Rust/native target-device parity and model-specific deployment evaluation remain separate.'],
                'post_final_reselection':False,
                'next_decision':'Build against the exported model contract; use a fresh reviewed confirmation study before a release claim.'}
    write_json(root/'MODEL_DECISION.json',result)
    lines=['# Model decision','',f"**Status:** {result['status']}",f"**Profile locked before final:** {result.get('selected_profile')}",
           f"**Backbone:** {result.get('selected_model')}",f"**Rendering:** {result.get('selected_layout')}",
           '', 'The selected profile is never replaced using the exposed final leaderboard. A provisional integration candidate is not a certified general decision model.',
           '', '## Remaining release evidence']
    lines += ['- '+x for x in result.get('reasons_not_release_ready', ['Model comparison has not completed.'])]
    atomic_bytes(root/'MODEL_DECISION.md',('\n'.join(lines)+'\n').encode())
    return result


def roadmap_evidence(root):
    root=Path(root);decision=decision_report(root)
    trained=(root/'MODEL_LOCK.json').exists();has_final=bool(decision.get('all_final_comparisons_complete'))
    mechanics=read_json(root/'mechanics/DONE.json') if (root/'mechanics/DONE.json').exists() else {}
    scaling=read_json(root/'scaling/DONE.json') if (root/'scaling/DONE.json').exists() else {}
    result={'schema':'opendecision-roadmap-evidence/v1','implementation_version':VERSION,'automatic_parent_task_closure':False,
      '2I.1-2I.2':{'status':'pilot_registered_not_reviewed' if trained and decision.get('scope')=='exploratory_pilot' else 'see_review_gate','human_approval_fabricated':False},
      '2I.6_fixture_fix':{'implementation':'distinct-question-v2','execution':'measured' if mechanics.get('distinct_extra_question_asserted') else 'pending','result':'mechanics/DONE.json'},
      '2I.3-2I.6':{'status':'bounded_readout_available' if scaling else 'pending','result':'scaling/DONE.json','full_generality_closed':False},
      '2J.1-2J.6':{'status':'bounded_comparison_available' if has_final else 'in_progress' if trained else 'pending','result':'MODEL_DECISION.json','release_selected':False},
      'model_bundle':{'status':'exported' if (root/'model_bundle/BUNDLE_MANIFEST.json').exists() else 'pending'},
      'S_and_Phase3':{'status':'model_contract_handoff_only','server_work_does_not_gate_model_selection':True},
      'P2':{'status':'deferred_hypothesis_and_model_evidence_required'},
      'H':{'status':'closed_unchanged'}}
    write_json(root/'roadmap_evidence.json',result);return result
