"""Separate genuine reviewed questions from synthetic Q/K execution mechanics.
The state-first profile is always compared to its own full-token reference.
"""
from __future__ import annotations
import copy,time,itertools
from pathlib import Path
import numpy as np
import torch
from odij_core import *
from odij_data import flatten,choices_for,protocol_template
from odij_runtime import Runtime,segments,tensor_digest,cache_bytes,sync,memory
from odij_learning import softmax,output_row,metrics


def synthetic_rows(qn,kn,state='A fictional record for execution testing only.',offset=0):
    out=[]
    for i in range(offset,offset+qn):
        q={'id':f'q{i}','family':'synthetic_mechanics','rubric_family':'not_a_semantic_task','primitive':'choice','instruction':f'Mechanical isolation probe number {i}. Evaluate each supplied marker.',
           'options':[{'id':f'c{j}','label':f'Option {j}','criteria':f'Mechanical marker number {j}, not a real-world judgment.'} for j in range(kn)],'target_id':'c0','origin':'answer_present'}
        out.append({'id':f'm{i}','group':'mechanics','state_id':'mechanics','state':state,'split':'mechanics','source_kind':'constructed','q':q,'target':0})
    return out


def distinct_extra_question(rows):
    """A different instruction/options set, not a restart of synthetic q0 numbering."""
    extra=synthetic_rows(1,2,rows[0]['state'],offset=len(rows)+17)[0]
    extra['q']['instruction']='Determine whether the supplied marker describes a geometric shape. This is an independent execution-only query.'
    extra['q']['options']=[{'id':'shape_round','label':'Round shape','criteria':'The marker explicitly names a circular shape.'},
                          {'id':'shape_angular','label':'Angular shape','criteria':'The marker explicitly names an angular shape.'}]
    extra['q']['target_id']='shape_round'
    assert extra['q']['id'] not in {r['q']['id'] for r in rows}
    meaning=lambda r:digest([r['q']['instruction'],r['q']['options']])
    if meaning(extra) in {meaning(r) for r in rows}:raise IntegrityError('Unrelated fixture must not duplicate a question')
    return extra


def batched_full(rt,rows,batch_size=4):
    seqs=[];spans=[]
    for r in rows:
        full=segments(r,rt.tokenizer,'state_first',rt.protocol['max_tokens'])[3];spans.append((len(seqs),len(full)));seqs.extend(full)
    groups={}
    for i,s in enumerate(seqs):groups.setdefault(len(s),[]).append(i)
    out=[None]*len(seqs)
    for length,indices in groups.items():
        for start in range(0,len(indices),batch_size):
            ix=indices[start:start+batch_size];ids=torch.tensor([seqs[i] for i in ix],device=rt.device,dtype=torch.long);rt.calls+=1;rt.tokens+=ids.numel()
            with torch.no_grad():
                h=rt.model(input_ids=ids,attention_mask=torch.ones_like(ids),position_ids=torch.arange(length,device=rt.device)[None,:].expand(len(ix),-1),use_cache=False,return_dict=True).last_hidden_state[:,-1,:].float().cpu().numpy()
            if not np.isfinite(h).all():raise FloatingPointError('Batched features nonfinite')
            for i,v in zip(ix,h):out[i]=v
    return [np.stack(out[a:a+n]) for a,n in spans]


def timing(fn,repeats=3,rt=None):
    fn();sync();times=[];work=[]
    if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
    for _ in range(repeats):
        sync();before=(rt.calls,rt.tokens) if rt is not None else None;t=time.perf_counter();fn();sync();times.append((time.perf_counter()-t)*1000)
        if before is not None:work.append({'forward_calls':rt.calls-before[0],'token_rows':rt.tokens-before[1]})
    return {'raw_ms':times,'median_ms':float(np.median(times)),'p95_ms':float(np.quantile(times,.95)),'memory':memory(),'model_work_per_repeat':work,'scope':'Resident model; model/head work as named; excludes input parsing, native JSON, application policy, server queue and network'}


def compare(a,b,tol):
    if len(a)!=len(b) or any(x.shape!=y.shape for x,y in zip(a,b)):raise IntegrityError('Comparison shape differs')
    absmax=max(float(np.abs(x-y).max()) for x,y in zip(a,b));rel=max(float(np.abs(x-y).max()/max(1.,np.abs(x).max())) for x,y in zip(a,b))
    return {'maximum_absolute_feature_delta':absmax,'maximum_scaled_feature_delta':rel,'within_declared_scaled_feature_tolerance':rel<=tol}


def mechanics_probe(root,cfg):
    p=protocol_template();spec=p['models'][0];out=root/'mechanics';out.mkdir(parents=True,exist_ok=True)
    if (out/'DONE.json').exists():
        prior=read_json(out/'DONE.json')
        if prior.get('fixture_version')!='distinct-question-v2':raise IntegrityError('Old mechanics fixture: preserve the old run and use a new study ID')
        return prior
    rt=Runtime(spec,out,p,allow_cpu_test=cfg.allow_cpu_test)
    try:
        rows=synthetic_rows(2,2)
        full=[rt.row_features(r,'state_first',memo=False) for r in rows]
        nested,c=rt.nested_features(rows,verify=True);digest_before=tensor_digest(c)
        reverse,_=rt.nested_features(list(reversed(rows)),root=c,verify=True);reverse=list(reversed(reverse))
        extra=distinct_extra_question(rows)
        added,_=rt.nested_features(rows+[extra],root=c,verify=True)
        duplicate,_=rt.nested_features(rows+[copy.deepcopy(rows[0])],root=c,verify=True)
        bat=batched_full(rt,rows)
        result={'status':'completed_mechanics_only','semantic_quality_evaluated':False,'training_performed':False,
          'new_input_contract':'state_first segmented v1','Q':2,'K':2,'model_revision':spec['revision'],'cache_bytes':cache_bytes(c),
          'cache_tensor_families':[n for n,_ in __import__('odij_runtime').walk_tensors(c)],
          'nested_vs_full':compare(full,nested,1e-4),'question_reordering':compare(nested,reverse,1e-5),'unrelated_question_addition':compare(nested,added[:2],1e-5),
          'duplicate_question_insertion':compare(nested,duplicate[:2],1e-5),'distinct_extra_question_asserted':True,'fixture_version':'distinct-question-v2',
          'full_equal_length_batch4_vs_sequential':compare(full,bat,1e-4),'root_unchanged':tensor_digest(c)==digest_before,
          'interpretation':'An untrained synthetic feature-equivalence probe, not generalized decisions or completion of 2I. It deliberately does not reuse H labels for fitting.'}
        write_json(out/'DONE.json',result);return result
    finally:rt.close()


def native_probs(pred,rows,features):
    with torch.no_grad():return [softmax(pred.d(torch.tensor(x),r['q']['primitive']=='choice').numpy(),pred.profile['calibration']['selected']) for r,x in zip(rows,features)]


def parity_probs(a,b,rows,tol,threshold):
    wrong=[];pd=0;cd=0
    for i,(x,y,r) in enumerate(zip(a,b,rows)):
        delta=float(np.abs(x-y).max());ci=int(x.argmax())!=int(y.argmax());opts=choices_for(r['q'])
        def action(p):
            j=int(p.argmax());return opts[j]['id'] if opts[j]['id']!=NONE and p[j]>=threshold else None
        pi=action(x)!=action(y);pd+=pi;cd+=ci
        wrong.append({'id':r['id'],'max_probability_delta':delta,'argmax_changed':ci,'policy_changed':pi,'accepted_equivalence':delta<=tol and not ci and not pi})
    return {'questions':len(rows),'max_probability_delta':max(x['max_probability_delta'] for x in wrong),'argmax_changes':cd,'policy_changes':pd,'accepted':sum(x['accepted_equivalence'] for x in wrong),'rows':wrong}


def study_scaling(root,cfg,profile):
    from odij_runner import Predictor
    from odij_data import assert_study_intact
    assert_study_intact(root)
    p=read_json(root/'registered_protocol.json');out=root/'scaling';out.mkdir(parents=True,exist_ok=True);pred=Predictor(root,p,profile,out/'runtime',cfg.allow_cpu_test);rt=pred.rt
    if pred.d is None or profile['layout']!='state_first':raise GateBlocked('Scaling requires a fitted causal state-first head')
    done=out/'DONE.json';start=time.monotonic();results=read_json(out/'progress.json') if (out/'progress.json').exists() else {'semantic':[],'mechanical':[]}
    if done.exists():pred.close();return read_json(done)
    try:
        cases=read_json(root/'data/policy_dev.json')[:3] # named nonfinal semantic diagnostic, not new final optimization data
        for case in cases:
            for Q in p['benchmark']['Q']:
                key=case['id']+f'/Q{Q}'
                if any(x['key']==key for x in results['semantic']):continue
                if Q>len(case['questions']):
                    results['semantic'].append({'key':key,'status':'not_applicable','reason':'Not enough independently reviewed questions; no duplicates invented to meet Q'});continue
                rows=flatten([dict(case,questions=case['questions'][:Q])]);full=[rt.row_features(r,'state_first',memo=False) for r in rows];nested,rootcache=rt.nested_features(rows,verify=True)
                a=native_probs(pred,rows,full);b=native_probs(pred,rows,nested);bp=native_probs(pred,rows,batched_full(rt,rows));threshold=profile['policy']['selected']['threshold']
                reverse,_=rt.nested_features(list(reversed(rows)),root=rootcache,verify=True);rev=native_probs(pred,rows,list(reversed(reverse)))
                add_check={'status':'not_applicable','reason':'No additional reviewed question on this state'}
                if len(case['questions'])>Q:
                    added_rows=flatten([dict(case,questions=case['questions'][:Q+1])]);af,_=rt.nested_features(added_rows,root=rootcache,verify=True);ap=native_probs(pred,added_rows,af)
                    add_check=parity_probs(b,ap[:Q],rows,p['order_tolerance'],threshold)
                # A machine-ID rename must not change tokenized semantics.
                renamed=copy.deepcopy(rows)
                for r in renamed:
                    for j,o in enumerate(r['q']['options']):o['id']='opaque_'+str(j)
                rfeatures,_=rt.nested_features(renamed,root=rootcache,verify=True);rp=native_probs(pred,rows,rfeatures)
                # Candidate permutation is compared after re-aligning the probability vector.
                perm=copy.deepcopy(rows)
                for r in perm:r['q']['options'].reverse()
                pf,_=rt.nested_features(perm,root=rootcache,verify=True);pp=native_probs(pred,perm,pf);pp=[np.r_[x[:-1][::-1],x[-1:]] if r['q']['primitive']=='choice' else x[::-1] for x,r in zip(pp,rows)]
                def full_fn():return native_probs(pred,rows,[rt.row_features(r,'state_first',memo=False) for r in rows])
                def batch_fn():return native_probs(pred,rows,batched_full(rt,rows))
                def cold_fn():return native_probs(pred,rows,rt.nested_features(rows)[0])
                def warm_fn():return native_probs(pred,rows,rt.nested_features(rows,root=rootcache)[0])
                item={'key':key,'status':'completed','split':'policy_dev','Q':Q,'K':[len(r['q']['options']) for r in rows],
                  'quality_full':metrics([output_row(r,x) for r,x in zip(rows,a)],0),'nested_vs_full':parity_probs(a,b,rows,p['probability_tolerance'],threshold),
                  'full_batch_vs_full':parity_probs(a,bp,rows,p['probability_tolerance'],threshold),'adding_reviewed_question':add_check,
                  'question_order':parity_probs(b,rev,rows,p['order_tolerance'],threshold),'opaque_ids':parity_probs(b,rp,rows,p['order_tolerance'],threshold),
                  'candidate_order':parity_probs(b,pp,rows,p['order_tolerance'],threshold),
                  'timing':{name:timing(fn,p['benchmark']['repeats'],rt) for name,fn in [('full_sequential',full_fn),('full_exact_length_batch4',batch_fn),('nested_cold',cold_fn),('nested_warm',warm_fn)]},
                  'cache_bytes':cache_bytes(rootcache),'claim_scope':'Reviewed same-state questions in this nonfinal panel; not broad generality.'}
                results['semantic'].append(item);write_json(out/'progress.json',results);rt.store.backup();del rootcache
                if time.monotonic()-start>cfg.worker_budget_minutes*60:raise BudgetStop('Scaling progress saved')
        # Balanced bounded screen covers each requested dimension without pretending synthetic labels are real.
        grid=p['benchmark'].get('grid',[[L,Q,K] for L in p['benchmark']['state_tokens'] for Q in p['benchmark']['Q'] for K in p['benchmark']['K']])
        for L,Q,K in grid[:p['benchmark']['max_grid_cells']]:
            key=f'L{L}_Q{Q}_K{K}'
            if any(x['key']==key for x in results['mechanical']):continue
            base=rt.tokenizer.encode('Administrative background with no requested decision. '*max(1,L),add_special_tokens=False)[:L]
            state=rt.tokenizer.decode(base);rows=synthetic_rows(Q,K,state)
            # Any unsupported length fails explicitly; no silent truncation or automatic budget expansion.
            try:
                full=[rt.row_features(r,'state_first',memo=False) for r in rows];nested,c=rt.nested_features(rows,verify=True)
                item={'key':key,'status':'completed','semantic_quality_evaluated':False,'actual_state_tokens':len(rt.tokenizer.encode(state,add_special_tokens=False)),
                   'feature_parity':compare(full,nested,1e-4),'cache_bytes':cache_bytes(c),'Q':Q,'K':K,
                   'timing':{'full':timing(lambda:[rt.row_features(r,'state_first',memo=False) for r in rows],p['benchmark']['repeats'],rt),
                            'full_exact_length_batch4':timing(lambda:batched_full(rt,rows),p['benchmark']['repeats'],rt),
                            'nested_cold':timing(lambda:rt.nested_features(rows),p['benchmark']['repeats'],rt),
                            'nested_warm':timing(lambda:rt.nested_features(rows,root=c),p['benchmark']['repeats'],rt)}}
                del c
            except ValueError as e:item={'key':key,'status':'unsupported_input','reason':str(e),'semantic_quality_evaluated':False}
            results['mechanical'].append(item);write_json(out/'progress.json',results)
            if time.monotonic()-start>cfg.worker_budget_minutes*60:raise BudgetStop('Synthetic mechanics grid progress saved')
        results['profile_selected_on']='dev only';results['profile_id']=profile['profile_id'];write_json(done,results);return results
    finally:pred.close()
