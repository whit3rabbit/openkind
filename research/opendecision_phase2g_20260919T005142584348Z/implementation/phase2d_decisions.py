"""Frozen candidate scoring, controlled missing-answer ablations, complete-request timing."""
from phase2d_common import *
import sqlite3

FEATURE_NAMES=['max_score','top_two_gap','mean_score','std_score','logmeanexp','log_K']


def overlap_audit(splits,blocked):
    groups={s:{r['group'] for r in rows} for s,rows in splits.items()}
    for s,g in groups.items():
        if g & blocked:raise AssertionError(f'Phase 2C message overlap in {s}')
    comparisons={}
    for i,(a,ga) in enumerate(groups.items()):
        for b,gb in list(groups.items())[i+1:]:
            comparisons[a+'__'+b]=len(ga & gb)
            if ga & gb:raise AssertionError('New phase data overlap')
    return comparisons


def make_episode(row,names,candidate_ids,missing,k,sampler):
    ids=list(map(int,candidate_ids))
    choices=[{'id':f'intent_{i}','description':names[i].replace('_',' '),'source_label':i} for i in ids]
    target=len(ids) if missing else ids.index(row['label'])
    return dict(id=f"{row['source_split']}_{row['source_index']}_{sampler}_k{k}_abs{int(missing)}",
                state=row['state'],instruction=DYNAMIC_INSTRUCTION,choices=choices,
                target_index=target,target=NONE_ID if missing else choices[target]['id'],
                group=row['group'],source_split=row['source_split'],source_index=row['source_index'],
                source_label=row['label'],real_candidate_count=k,true_intent_omitted=bool(missing),sampler=sampler)


def build_fresh_episodes(raw,ref,cfg):
    """Disjoint new messages; same 57/20 label partition; nested distractors across K."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    names=list(raw['train'].features['label'].names)
    if names!=ref['data']['class_names']:raise ValueError('Banking label order changed')
    seen=ref['data']['training_label_ids'];unseen=ref['data']['heldout_label_ids']
    if set(seen)&set(unseen) or len(seen)!=57 or len(unseen)!=20:raise ValueError('Unexpected reference label split')
    blocked={ep['group'] for es in ref['episodes'].values() for ep in es}
    # Lexical difficulty uses only the label descriptions, not learned test-message features.
    # The true label is used OFFLINE to choose plausible distractors; it is not a production retriever.
    similarity=np.zeros((len(names),len(names)),dtype=np.float64)
    # Separate vocabularies prevent unseen descriptions influencing fit-time IDF weights.
    for label_pool in (seen,unseen):
        vectors=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5)).fit_transform([names[i].replace('_',' ') for i in label_pool])
        similarity[np.ix_(label_pool,label_pool)]=(vectors@vectors.T).toarray()
    all_used=set(blocked);message_splits={};episodes={}
    for j,s in enumerate(('train','dev','cal_fit','cal_gate','test_seen','test_unseen')):
        pool=unseen if s=='test_unseen' else seen
        source='test' if s.startswith('test') else 'train'
        rows=choose_banking(raw,source,cfg.sizes()[s],set(pool),all_used,cfg.data_seed+j)
        message_splits[s]=rows;eps=[]
        for row in rows:
            seed=int(row['group'][:12],16)+cfg.data_seed
            rng=np.random.default_rng(seed)
            negatives=[c for c in pool if c!=row['label']]
            uniform=list(rng.permutation(negatives))
            # Deterministic random tie-breaks; never rely on integer class position.
            tie={int(c):float(rng.random()) for c in negatives}
            hard=sorted(negatives,key=lambda c:(-similarity[row['label'],c],tie[c]))
            orders={'uniform':uniform,'label_lexical_hard':hard}
            ks=cfg.eval_candidate_counts if s.startswith('test') else (int(rng.choice(cfg.fit_candidate_counts)),)
            samplers=cfg.samplers if s.startswith('test') else (str(rng.choice(cfg.samplers)),)
            for sampler in samplers:
                for k in ks:
                    for missing in (False,True):
                        ids=orders[sampler][:k] if missing else [row['label']]+orders[sampler][:k-1]
                        if len(ids)!=k:raise ValueError('Not enough negative categories')
                        order_rng=np.random.default_rng(seed+k*101+int(missing)*37+(1 if sampler=='uniform' else 2))
                        ids=list(order_rng.permutation(ids))
                        eps.append(make_episode(row,names,ids,missing,k,sampler))
        episodes[s]=eps
    audit=overlap_audit(message_splits,blocked)
    for s in ('train','dev','cal_fit','cal_gate','test_seen'):
        if any(c['source_label'] in unseen for ep in episodes[s] for c in ep['choices']):raise AssertionError('Held-out label leaked')
    if any(c['source_label'] not in unseen for ep in episodes['test_unseen'] for c in ep['choices']):raise AssertionError('Incorrect unseen pool')
    return episodes,{'prior_messages_excluded':len(blocked),'overlaps':audit,
                     'label_holdout_scope':'same 20 labels excluded from Phase 2C scorer training and Phase 2D rejection fitting/calibration, not Qwen pretraining',
                     'label_ids_seen':seen,'label_ids_unseen':unseen,
                     'splits':{s:{'messages':len(rows),'episodes':len(episodes[s]),'true_answer_absent_fraction':0.5,
                                  'candidate_counts':dict(Counter(len(e['choices']) for e in episodes[s])),
                                  'samplers':dict(Counter(e['sampler'] for e in episodes[s]))} for s,rows in message_splits.items()},
                     'construction':'paired present/absent episodes; nested negative lists across K within message and sampler',
                     'fit_dev_calibration_absent_prior':0.25,
                     'hard_negative_definition':'character 3..5 gram TF-IDF similarity between true-label and negative-label descriptions; random tie-break. Not model-mined hard negatives.',
                     'scope':'one domain and sampled candidates; evaluation labels are unseen only to head stages; not a 77-way benchmark or universal OOD detection'}


@dataclass
class ScoredEpisodes:
    episodes: list
    scores: list

    def subset(self,indices):return ScoredEpisodes([self.episodes[i] for i in indices],[self.scores[i] for i in indices])


def cache_identity(cfg,ref,environment):
    return {'version':VERSION,'model':cfg.model_id,'revision':cfg.model_revision,
            'candidate_head_sha256':file_sha(ref['root']/'dynamic_export/candidate_head.safetensors'),
            'dtype':'torch.bfloat16','batch':1,'backend':'default_sdpa','environment':environment,
            'max_length':cfg.max_length,'prompt_contract':ref['dynamic_export']['prompt_segments']}


@torch.inference_mode()
def score_episodes(episodes,model,candidate_head,tokenizer,cfg,cache_path,identity,device):
    """Cache scalar logits, not all token features. One unpadded prompt per model call.
    SQLite transactions survive an interrupted cell; never reuse other precision/backend fingerprints.
    """
    unique={};references=[];trunc=0
    for ep in episodes:
        keys=[]
        for c in ep['choices']:
            row=encode_candidate(ep,c,tokenizer,cfg.max_length)
            key=content_hash({'identity':identity,'tokens':row['input_ids']})
            keys.append(key)
            if key not in unique:
                unique[key]=row;trunc+=int(row['state_truncated'])
        references.append(keys)
    Path(cache_path).parent.mkdir(parents=True,exist_ok=True)
    connection=sqlite3.connect(str(cache_path));connection.execute('PRAGMA journal_mode=WAL')
    connection.execute('CREATE TABLE IF NOT EXISTS scores (key TEXT PRIMARY KEY, score REAL NOT NULL)')
    values={};misses=[]
    for key in unique:
        entry=connection.execute('SELECT score FROM scores WHERE key=?',(key,)).fetchone()
        if entry is None:misses.append(key)
        elif math.isfinite(entry[0]):values[key]=float(entry[0])
        else:raise ValueError('Nonfinite cached score')
    candidate_head.to(device).eval();began=time.perf_counter()
    try:
        for i,key in enumerate(tqdm(misses,desc='Unpadded frozen candidate scores')):
            row=unique[key]
            batch=pack_tokens([row],tokenizer.pad_token_id,device)
            h=forward_features(model,batch)
            s=candidate_head.scorer(h).reshape(-1)
            require_finite(s,'frozen candidate score')
            value=float(s.item());values[key]=value
            connection.execute('INSERT INTO scores(key,score) VALUES (?,?)',(key,value))
            if (i+1)%cfg.score_cache_commit_every==0:connection.commit()
        connection.commit()
    finally:
        connection.commit();connection.close()
        candidate_head.to('cpu')
    scores=[np.array([values[k] for k in ks],dtype=np.float64) for ks in references]
    return ScoredEpisodes(episodes,scores),{'candidate_appearances':sum(map(len,references)),'unique_prompts':len(unique),
        'cache_hits':len(unique)-len(misses),'computed':len(misses),'seconds':time.perf_counter()-began,
        'unique_truncated_prompts':trunc,'cache_identity':content_hash(identity),'feature_batch_size':1,
        'state_reencoded_per_candidate':True}


def score_features(scores,kind):
    if kind in ('frozen_global','refit_global'):return np.empty((len(scores),0),dtype=np.float64)
    if kind=='count_bias':return np.array([[np.log(len(s))] for s in scores],dtype=np.float64)
    if kind!='set_linear':raise ValueError(kind)
    result=[]
    for s in scores:
        s=np.asarray(s,dtype=np.float64)
        if len(s)<2 or not np.isfinite(s).all():raise ValueError('Need finite >=2 candidate scores')
        top=np.sort(s)[-2:]
        result.append([top[-1],top[-1]-top[-2],s.mean(),s.std(),logsumexp(s)-np.log(len(s)),np.log(len(s))])
    return np.asarray(result)


def prior_weights(episodes,prior=0.25):
    """Equal message weight; within each message the explicit absent mass is prior.
    This reweights a controlled paired stress set. It is NOT an observed deployment prior.
    """
    if not 0<prior<1:raise ValueError('Prior must lie strictly between zero and one')
    counts=Counter((e['group'],bool(e['true_intent_omitted'])) for e in episodes)
    weights=[]
    for e in episodes:
        g=e['group'];absent=bool(e['true_intent_omitted'])
        if (g,True) not in counts or (g,False) not in counts:raise ValueError('Both paired conditions required per message')
        weights.append((prior if absent else 1-prior)/counts[g,absent])
    w=np.asarray(weights);return w/w.sum()


@dataclass
class NoneModel:
    kind: str
    coef: list
    feature_mean: list
    feature_std: list

    def none_logits(self,scores):
        f=score_features(scores,self.kind)
        if f.shape[1]:f=(f-np.asarray(self.feature_mean))/np.asarray(self.feature_std)
        x=np.column_stack([np.ones(len(scores)),f])
        z=x@np.asarray(self.coef)
        if not np.isfinite(z).all():raise FloatingPointError('Nonfinite none scores')
        return z

    def logits(self,scores):
        return [np.r_[s,b] for s,b in zip(scores,self.none_logits(scores))]


def per_episode_nll(logits,episodes,temperature=1.):
    if not math.isfinite(temperature) or temperature<=0:raise ValueError('Invalid temperature')
    return np.array([logsumexp(np.asarray(s)/temperature)-s[e['target_index']]/temperature for s,e in zip(logits,episodes)])


def weighted_nll(logits,episodes,prior=.25,temperature=1.):
    return float(prior_weights(episodes,prior)@per_episode_nll(logits,episodes,temperature))


def fit_none_model(train,kind,initial_none,l2=.001):
    f=score_features(train.scores,kind)
    mean=f.mean(0);std=np.maximum(f.std(0),1e-5)
    x=np.column_stack([np.ones(len(f)),(f-mean)/std])
    weights=prior_weights(train.episodes)
    absent=np.array([e['true_intent_omitted'] for e in train.episodes],dtype=np.float64)
    lse=np.array([logsumexp(s) for s in train.scores])
    true_score=np.array([0. if e['true_intent_omitted'] else s[e['target_index']] for s,e in zip(train.scores,train.episodes)])
    def objective(theta):
        b=x@theta
        loss=np.logaddexp(lse,b)-np.where(absent>0,b,true_score)
        penalty=0.5*l2*np.dot(theta[1:],theta[1:])
        grad=x.T@(weights*(expit(b-lse)-absent));grad[1:]+=l2*theta[1:]
        return float(weights@loss+penalty),grad
    initial=np.r_[initial_none,np.zeros(f.shape[1])]
    opt=minimize(objective,initial,jac=True,method='L-BFGS-B',options={'maxiter':1000,'ftol':1e-12,'gtol':1e-8})
    if not opt.success or not np.isfinite(opt.fun):raise RuntimeError(f'None optimizer failed: {opt.message}')
    model=NoneModel(kind,opt.x.tolist(),mean.tolist(),std.tolist())
    return model,{'optimizer':'L-BFGS-B, analytic gradient','success':bool(opt.success),'iterations':int(opt.nit),
                  'parameters':len(opt.x),'training_weighted_nll':weighted_nll(model.logits(train.scores),train.episodes),
                  'l2_non_intercept':l2,'candidate_scorer_changed':False,'assumed_absent_prior':.25}


def extended_metrics(logits,episodes,temperature=1.):
    result=episode_metrics(logits,episodes,temperature)
    absent=np.array([e['true_intent_omitted'] for e in episodes])
    pred=np.array([int(np.argmax(s)) for s in logits])
    k=np.array([len(e['choices']) for e in episodes]);reject=pred==k
    correct=pred==np.array([e['target_index'] for e in episodes])
    result.update({'absent_n':int(absent.sum()),'present_n':int((~absent).sum()),
                   'false_answers_when_absent':int((absent&~reject).sum()),
                   'false_answer_rate_when_absent':float((~reject)[absent].mean()) if absent.any() else None,
                   'false_abstentions_when_present':int((~absent&reject).sum()),
                   'wrong_real_choice_when_present':int((~absent&~reject&~correct).sum()),
                   'messages':len({e['group'] for e in episodes})})
    return result


def select_rejection_models(train,dev,candidate_head,cfg):
    initial=float(candidate_head.none_logit.detach().cpu())
    models={'frozen_global':NoneModel('frozen_global',[initial],[],[])}
    fits={'frozen_global':{'status':'unchanged_phase2c','parameters_trained':0}}
    for kind in ('refit_global','count_bias','set_linear'):
        models[kind],fits[kind]=fit_none_model(train,kind,initial,cfg.none_l2)
    comparison={}
    for name,model in models.items():
        logits=model.logits(dev.scores)
        comparison[name]={'weighted_nll_prior025':weighted_nll(logits,dev.episodes),
                          'paired_stress_metrics':extended_metrics(logits,dev.episodes)}
    winner=min(comparison,key=lambda k:(comparison[k]['weighted_nll_prior025'],len(models[k].coef),k))
    return models,{'selected':winner,'criterion':'lowest dev message-weighted NLL at assumed omitted-intent prior 0.25; ties prefer fewer parameters',
                   'development':comparison,'training':fits,'frozen_candidate_scores':True,
                   'scope':'selected head is not a validated safety policy; held-out errors and tradeoffs must still be inspected'}


def weighted_temperature_gate(model,fit,gate,minimum=.002):
    fit_z=model.logits(fit.scores);gate_z=model.logits(gate.scores)
    objective=lambda t:weighted_nll(fit_z,fit.episodes,temperature=float(np.exp(t)))
    opt=minimize_scalar(objective,bounds=(np.log(.25),np.log(4.)),method='bounded')
    if not opt.success or not np.isfinite(opt.fun):raise RuntimeError('Temperature fitting failed')
    t=float(np.exp(opt.x));raw=weighted_nll(gate_z,gate.episodes);scaled=weighted_nll(gate_z,gate.episodes,temperature=t)
    return {'fitted_temperature':t,'selected_temperature':t if raw-scaled>=minimum else 1.,
            'gate_raw_weighted_nll':raw,'gate_scaled_weighted_nll':scaled,'improvement':raw-scaled,
            'minimum_improvement':minimum,'assumed_absent_prior':.25,'selected_on':'separate calibration-gate messages, never test',
            'temperature_changes_argmax':False}


def evaluate_rejection(store,models,selected,temperature,cfg):
    report={};predictions={}
    for name,model in models.items():
        logits=model.logits(store.scores);t=temperature if name==selected else 1.
        predictions[name]=[{'episode_id':e['id'],'group':e['group'],'sampler':e['sampler'],'K':len(e['choices']),
                            'absent':e['true_intent_omitted'],'target_index':e['target_index'],
                            'logits':s.tolist(),'probabilities':probs(s[None,:],t)[0].tolist(),
                            'prediction_index':int(s.argmax())} for e,s in zip(store.episodes,logits)]
        by={}
        for k in sorted(set(len(e['choices']) for e in store.episodes)):
            for sampler in sorted(set(e['sampler'] for e in store.episodes)):
                ix=[i for i,e in enumerate(store.episodes) if len(e['choices'])==k and e['sampler']==sampler]
                if ix:by[f'k{k}/{sampler}']=extended_metrics([logits[i] for i in ix],[store.episodes[i] for i in ix],t)
        correctness=[int(np.argmax(s)==e['target_index']) for s,e in zip(logits,store.episodes)]
        report[name]={'raw':extended_metrics(logits,store.episodes),
                      'reported_temperature':t,'temperature_scaled':extended_metrics(logits,store.episodes,t),
                      'weighted_nll_prior_scenarios':{str(p):weighted_nll(logits,store.episodes,p,t) for p in (.05,.25,.5)},
                      'by_K_and_sampler':by,'accuracy_interval_message_bootstrap':group_bootstrap(correctness,[e['group'] for e in store.episodes],cfg.bootstrap_repeats,cfg.data_seed)}
    # Paired estimate on the SAME test episodes; messages, not candidate-count copies, are resampled.
    base=models['frozen_global'].logits(store.scores);new=models[selected].logits(store.scores)
    difference=np.array([int(a.argmax()==e['target_index'])-int(b.argmax()==e['target_index']) for a,b,e in zip(new,base,store.episodes)])
    absent=np.array([e['true_intent_omitted'] for e in store.episodes]);groups=np.array([e['group'] for e in store.episodes])
    delta_absent=np.array([int(a.argmax()<len(e['choices']))-int(b.argmax()<len(e['choices'])) for a,b,e in zip(new,base,store.episodes)])
    paired={'selected_minus_frozen_accuracy':group_bootstrap(difference,groups,cfg.bootstrap_repeats,cfg.data_seed),
            'selected_minus_frozen_false_answer_rate_when_absent':group_bootstrap(delta_absent[absent],groups[absent],cfg.bootstrap_repeats,cfg.data_seed)}
    return {'models':report,'selected':selected,'paired':paired,
            'denominator_note':'paired stress set is 50% absent; weighted NLL scenarios are assumptions, not observed deployment priors'},predictions


@torch.inference_mode()
def real_request(ep,model,candidate_head,tokenizer,none_model,cfg,device,batch_size,temperature):
    # Include all tokenization, H2D, K candidate forwards, none/head computation, CPU sync and JSON serialization.
    rows=[encode_candidate(ep,c,tokenizer,cfg.max_length) for c in ep['choices']]
    scores=[];calls=0
    for start in range(0,len(rows),batch_size):
        batch=pack_tokens(rows[start:start+batch_size],tokenizer.pad_token_id,device)
        x=forward_features(model,batch)
        z=candidate_head.scorer(x).reshape(-1)
        require_finite(z,'full request candidate scores')
        scores.extend(z.float().cpu().numpy().tolist());calls+=1
    z=none_model.logits([np.asarray(scores,dtype=np.float64)])[0]
    p=probs(z[None,:],temperature)[0]
    ids=[c['id'] for c in ep['choices']]+[NONE_ID]
    response=json.dumps({'selected':ids[int(p.argmax())],'probabilities':dict(zip(ids,p.tolist()))},allow_nan=False)
    return response,p,{'model_calls':calls,'input_tokens_total':sum(len(r['input_ids']) for r in rows),
                       'candidate_count':len(rows),'max_candidate_tokens':max(len(r['input_ids']) for r in rows),
                       'state_reencoded':len(rows)}


def benchmark_requests(episodes,model,candidate_head,tokenizer,none_model,cfg,device,temperature):
    candidate_head.to(device).eval();rows=[]
    # One small fixed set of DEV messages, reused at every K and candidate batch size.
    messages={}
    for ep in episodes:
        if not ep['true_intent_omitted'] and ep['sampler']=='uniform':messages.setdefault(ep['group'],[]).append(ep)
    selected=list(messages)[:cfg.sizes()['benchmark_messages']]
    chosen=[]
    for g in selected:
        byk={len(e['choices']):e for e in messages[g]}
        chosen.extend(byk[k] for k in cfg.eval_candidate_counts if k in byk)
    if not chosen:raise ValueError('No benchmark episodes')
    try:
        for ep in tqdm(chosen,desc='Complete dynamic requests'):
            _,refp,_=real_request(ep,model,candidate_head,tokenizer,none_model,cfg,device,1,temperature)
            for batch_size in cfg.benchmark_batches:
                try:
                    for _ in range(cfg.benchmark_warmups):real_request(ep,model,candidate_head,tokenizer,none_model,cfg,device,batch_size,temperature)
                    sync(device)
                    baseline=torch.cuda.memory_allocated(device) if torch.device(device).type=='cuda' else None
                    if baseline is not None:torch.cuda.reset_peak_memory_stats(device)
                    times=[];deltas=[];flips=[]
                    for _ in range(cfg.benchmark_repeats):
                        sync(device);t0=time.perf_counter()
                        response,p,info=real_request(ep,model,candidate_head,tokenizer,none_model,cfg,device,batch_size,temperature)
                        sync(device);times.append(1000*(time.perf_counter()-t0))
                        del response
                        deltas.append(float(np.max(np.abs(p-refp))));flips.append(bool(p.argmax()!=refp.argmax()))
                    rows.append(dict(episode_id=ep['id'],group=ep['group'],candidate_batch_size=batch_size,**info,status='completed',
                                     p50_ms=float(np.median(times)),p95_ms=float(np.quantile(times,.95)),times_ms=times,
                                     max_probability_delta_vs_unpadded=max(deltas),selection_changed=any(flips),
                                     requests_per_second=1000/float(np.median(times)),candidates_per_second=len(ep['choices'])*1000/float(np.median(times)),
                                     resident_mib=baseline/1024**2 if baseline is not None else None,
                                     peak_extra_mib=(torch.cuda.max_memory_allocated(device)-baseline)/1024**2 if baseline is not None else None))
                except torch.OutOfMemoryError as exc:
                    rows.append({'episode_id':ep['id'],'candidate_batch_size':batch_size,'status':'failed_oom','error':sanitize_error(exc)})
                    gc.collect();torch.cuda.empty_cache()
    finally:candidate_head.to('cpu')
    return {'rows':rows,'distinct_messages':len(selected),'scope':'complete single Python request, no server/network; real dev messages; no caches of hidden state or logits in timing; K full state encodings',
            'p95_warning':'few repeated calls: descriptive timing quantile, not a production tail-latency guarantee',
            'shared_prefill':False,'generation':False}


def benchmark_episodes(dev_episodes,ref,cfg):
    # Expand ONLY dev messages, with reproducible nested uniform negatives. No test-set timing selection.
    names=ref['data']['class_names'];pool=ref['data']['training_label_ids'];unique={}
    for e in dev_episodes:unique.setdefault(e['group'],e)
    result=[]
    for g,e in list(unique.items())[:cfg.sizes()['benchmark_messages']]:
        row={'state':e['state'],'label':e['source_label'],'group':g,'source_split':e['source_split'],'source_index':e['source_index']}
        rng=np.random.default_rng(int(g[:12],16)+cfg.data_seed)
        negatives=list(rng.permutation([i for i in pool if i!=row['label']]))
        for k in cfg.eval_candidate_counts:
            ids=list(rng.permutation([row['label']]+negatives[:k-1]))
            result.append(make_episode(row,names,ids,False,k,'uniform'))
    return result
