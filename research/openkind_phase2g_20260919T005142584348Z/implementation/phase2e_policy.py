"""Policy selection is separate from probabilities. Frozen heads; fresh dev/test messages."""
from __future__ import annotations
import sqlite3, json, hashlib
from pathlib import Path
from collections import Counter
import numpy as np
from tqdm.auto import tqdm
from phase2d_common import json_ready, json_write, encode_candidate, pack_tokens, forward_features, probs, group_bootstrap
from phase2d_decisions import prior_weights, extended_metrics
from phase2e_cache import scalar_from_features


def score_episodes(episodes,model,head,tokenizer,cfg,device,cache_path,identity):
    """Only full, unpadded, batch-one reference calls. Cache scalar logits by full execution identity."""
    Path(cache_path).parent.mkdir(parents=True,exist_ok=True)
    conn=sqlite3.connect(str(cache_path))
    conn.execute('CREATE TABLE IF NOT EXISTS scores (key TEXT PRIMARY KEY,value REAL NOT NULL)')
    outputs=[];hits=0;misses=0;truncated=0
    identity_bytes=json.dumps(json_ready(identity),sort_keys=True).encode()
    try:
        for ep in tqdm(episodes,desc='Reference candidate scores'):
            result=[]
            for choice in ep['choices']:
                row=encode_candidate(ep,choice,tokenizer,cfg.max_length)
                key=hashlib.sha256(identity_bytes+json.dumps(row['input_ids']).encode()).hexdigest()
                cached=conn.execute('SELECT value FROM scores WHERE key=?',(key,)).fetchone() if cfg.resume_score_cache else None
                if cached is not None:value=float(cached[0]);hits+=1
                else:
                    batch=pack_tokens([row],tokenizer.pad_token_id,device,explicit_positions=True)
                    value=float(scalar_from_features(head,forward_features(model,batch))[0]);misses+=1
                    conn.execute('INSERT OR REPLACE INTO scores VALUES (?,?)',(key,value));conn.commit()
                if not np.isfinite(value):raise FloatingPointError('Cached or computed score is nonfinite')
                truncated+=int(row['state_truncated']);result.append(value)
            outputs.append(result)
    finally:conn.close()
    return outputs,{'hits':hits,'misses':misses,'truncated_candidate_prompts':truncated,
                   'execution':'full prompt; unpadded batch one; explicit positions; no prefix reuse'}


def model_probabilities(scores,none_model):
    return [probs(np.asarray(x)[None,:])[0] for x in none_model.logits(scores)]


def policy_outcomes(probabilities,episodes,threshold):
    """Offer top real candidate only when it beats none AND its probability >= threshold.
    Otherwise route to manual review, regardless of whether a correct candidate is present.
    """
    accepted=[];wrong=[];correct=[];absent=[]
    for p,e in zip(probabilities,episodes):
        k=len(e['choices']);best=int(np.argmax(p[:k]));yes=bool(p[best]>p[k] and p[best]>=threshold)
        is_correct=yes and best==e['target_index']
        accepted.append(yes);correct.append(is_correct);wrong.append(yes and not is_correct);absent.append(e['true_intent_omitted'])
    return {'accepted':np.asarray(accepted,bool),'wrong':np.asarray(wrong,bool),'correct':np.asarray(correct,bool),'absent':np.asarray(absent,bool)}


def cost_result(probabilities,episodes,threshold,prior,wrong_cost,review_cost):
    if wrong_cost<0 or review_cost<0:raise ValueError('Costs must be nonnegative')
    o=policy_outcomes(probabilities,episodes,threshold);w=prior_weights(episodes,prior)
    losses=wrong_cost*o['wrong']+review_cost*(~o['accepted'])
    answer_mass=float(w@o['accepted']);absent=o['absent'];present=~absent
    return {'expected_cost':float(w@losses),'weighted_answer_rate':answer_mass,
      'weighted_error_among_answers':float((w@o['wrong'])/answer_mass) if answer_mass else None,
      'false_answer_rate_when_absent':float(o['accepted'][absent].mean()) if absent.any() else None,
      'review_rate_when_present':float((~o['accepted'][present]).mean()) if present.any() else None,
      'correct_answer_rate_when_present':float(o['correct'][present].mean()) if present.any() else None,
      'reference_always_review_cost':review_cost,'episode_count':len(episodes),'message_count':len({e['group'] for e in episodes})},losses


def grouped_scenario_values(values,episodes,prior):
    buckets={}
    for v,e in zip(values,episodes):buckets.setdefault((e['group'],e['true_intent_omitted']),[]).append(float(v))
    groups=sorted({g for g,a in buckets})
    result=[(1-prior)*np.mean(buckets[g,False])+prior*np.mean(buckets[g,True]) for g in groups]
    return result,groups


def evaluate_policies(stores,episodes,models,cfg):
    """Models and thresholds selected on new dev only, separately for each assumed cost/prior scenario."""
    predictions={s:{name:model_probabilities(scores,m) for name,m in models.items()} for s,scores in stores.items()}
    thresholds=np.r_[np.linspace(0,1,101),1.000001]  # Last point explicitly permits always-review.
    selections=[];results=[];surfaces=[]
    for prior in cfg.absent_priors:
        for wrong_cost in cfg.wrong_answer_costs:
            options=[]
            for name in models:
                for threshold in thresholds:
                    r,_=cost_result(predictions['dev'][name],episodes['dev'],float(threshold),prior,wrong_cost,cfg.review_cost)
                    options.append((r['expected_cost'],name,float(threshold)))
            loss,name,threshold=min(options,key=lambda x:(x[0],list(models).index(x[1]),x[2]))
            select={'absent_prior_assumption':prior,'wrong_answer_cost':wrong_cost,'review_cost':cfg.review_cost,
                    'none_head':name,'acceptance_threshold':threshold,'dev_expected_cost':loss,
                    'selected_on':'fresh development messages; test outcomes never used'}
            selections.append(select)
            for split in ('test_seen','test_unseen'):
                metrics,losses=cost_result(predictions[split][name],episodes[split],threshold,prior,wrong_cost,cfg.review_cost)
                vals,groups=grouped_scenario_values(losses,episodes[split],prior)
                ci=group_bootstrap(vals,groups,repeats=400,seed=cfg.seed)
                results.append(dict(select,split=split,metrics=metrics,cost_interval=ci))
            # Raw argmax/none policies are retained for all heads, not only the selected policy.
            for split in ('test_seen','test_unseen'):
                for baseline in models:
                    r,_=cost_result(predictions[split][baseline],episodes[split],0.,prior,wrong_cost,cfg.review_cost)
                    surfaces.append(dict(split=split,head=baseline,threshold=0.,prior=prior,wrong_cost=wrong_cost,**r))
    probability_metrics={}
    for split in ('test_seen','test_unseen'):
        probability_metrics[split]={name:extended_metrics(model.logits(stores[split]),episodes[split]) for name,model in models.items()}
    report={'selections':selections,'test_results':results,'unthresholded_baselines':surfaces,
        'probability_metrics':probability_metrics,
        'assumptions':{'absent_priors':cfg.absent_priors,'wrong_costs':cfg.wrong_answer_costs,'review_cost':cfg.review_cost,
            'correct_answer_cost':0.,'all_review_has_cost':True,'scenario_reweighting':'equal message weights, conditional present/absent mass; not a measured deployment frequency'},
        'scope':'frozen probability models plus separately selected acceptance policies; banking sampled choices only; no universal safety/rejection claim'}
    return report,predictions
