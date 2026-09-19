"""Independent, message-clustered summaries; parity and semantic quality are separate."""
import math
import numpy as np
from phase2d_common import group_bootstrap
from phase2e_expand_execution import compare_views


def parity(ref,result,tolerance=.005):
    p={k:np.asarray(v,dtype=np.float64) for k,v in ref['distributions'].items()}
    q={k:np.asarray(v,dtype=np.float64) for k,v in result['distributions'].items()}
    c=compare_views(p,q,ref['actions'],result['actions'])
    c['accepted_within_sample']=bool(c['max_probability_delta']<=tolerance and not c['any_argmax_changed'] and not c['any_policy_output_changed'])
    return c


def binary_upper_zero(n,confidence=.95):
    # Independent message groups only; no claim for exchangeability or hidden correlated groups.
    return float(1-(1-confidence)**(1/n)) if n>0 else None


def quality(rows,policies,repeats=500,seed=111):
    if not rows:return {}
    groups=np.array([r['group'] for r in rows]);target=np.array([r['target_index'] for r in rows]);absent=np.array([r['absent'] for r in rows]);ks=np.array([r['K'] for r in rows]);out={'episodes':len(rows),'independent_message_groups':len(set(groups)),'heads':{},'policies':{}}
    for head in rows[0]['outputs']['distributions']:
        ps=[np.asarray(r['outputs']['distributions'][head],dtype=float) for r in rows]
        if any(not np.isfinite(p).all() or (p<0).any() or not np.isclose(p.sum(),1,atol=1e-7) for p in ps):raise ValueError('Invalid probability distribution.')
        pred=np.array([p.argmax() for p in ps]);correct=pred==target;none=pred==ks
        ll=np.array([-math.log(max(p[y],np.finfo(float).tiny)) for p,y in zip(ps,target)])
        brier=np.array([float((p*p).sum()-2*p[y]+1) for p,y in zip(ps,target)])
        conf=np.array([p.max() for p in ps]);bins=np.minimum((15*conf).astype(int),14)
        ece=sum(float((bins==b).mean())*abs(float(correct[bins==b].mean()-conf[bins==b].mean())) for b in range(15) if (bins==b).any())
        per_group_bad=[not bool(correct[groups==g].all()) for g in sorted(set(groups))]
        out['heads'][head]={'accuracy':float(correct.mean()),'nll':float(ll.mean()),'brier_sum_classes':float(brier.mean()),'ece_15bins':ece,
          'answerable_accuracy':float(correct[~absent].mean()) if (~absent).any() else None,
          'none_recall':float(none[absent].mean()) if absent.any() else None,
          'false_none_when_present':float(none[~absent].mean()) if (~absent).any() else None,
          'accuracy_message_bootstrap':group_bootstrap(correct.astype(float),groups,repeats,seed),
          'nll_message_bootstrap':group_bootstrap(ll,groups,repeats,seed+1),
          'message_any_error_count':int(sum(per_group_bad))}
    for key in rows[0]['outputs']['actions']:
        actions=[r['outputs']['actions'][key] for r in rows]
        answer=np.array([a['action']=='answer' for a in actions])
        good=np.array([a['action']=='answer' and not r['absent'] and a['candidate_id']==r['target_id'] for a,r in zip(actions,rows)])
        wrong=answer&~good
        selected=next(p for p in policies if f"prior={p['absent_prior_assumption']}:wrong={p['wrong_answer_cost']}:review={p['review_cost']}"==key)
        costs=wrong*selected['wrong_answer_cost']+(~answer)*selected['review_cost']
        weighted=None
        if absent.any() and (~absent).any():
            prior=selected['absent_prior_assumption'];weighted=float(prior*costs[absent].mean()+(1-prior)*costs[~absent].mean())
        # Actual accepted labels are checked, not inferred from agreement with reference.
        out['policies'][key]={'panel_answer_rate':float(answer.mean()),'panel_error_among_answers':float(wrong.sum()/answer.sum()) if answer.any() else None,
            'false_answer_rate_when_absent':float(answer[absent].mean()) if absent.any() else None,
            'review_rate_when_present':float((~answer[~absent]).mean()) if (~absent).any() else None,
            'scenario_weighted_cost':weighted,'always_review_cost':selected['review_cost'],
            'accepted_episodes':int(answer.sum()),'wrong_accepted_episodes':int(wrong.sum()),
            'accepted_message_groups':len(set(groups[answer])),
            'semantic_correctness_source':'published label or explicitly attributed context transformation; NOT reference parity',
            'scope':'thresholds frozen from Banking development; CLINC/custom transfer is diagnostic, not recalibrated policy certification'}
    return out


def summarize_parity(rows):
    done=[r for r in rows if r.get('status')=='completed' and 'comparison' in r]
    if not done:return {'evaluated':0,'status':'unavailable'}
    c=[r['comparison'] for r in done];gs={r['group'] for r in done};badgs={r['group'] for r in done if not r['comparison']['accepted_within_sample']}
    return {'evaluated':len(done),'distinct_message_groups':len(gs),'maximum_probability_delta':max(x['max_probability_delta'] for x in c),
        'episodes_over_0_005':sum(x['max_probability_delta']>.005 for x in c),'argmax_changed':sum(x['any_argmax_changed'] for x in c),
        'policy_output_changed':sum(x['any_policy_output_changed'] for x in c),'accepted':sum(x['accepted_within_sample'] for x in c),
        'message_groups_with_any_failure':len(badgs),
        'zero_failure_message_upper_bound_95':binary_upper_zero(len(gs)) if not badgs else None,
        'upper_bound_scope':'one-sided zero-event binomial bound on message-level any-failure, only under independent/exchangeable message assumptions; not safety certification'}
