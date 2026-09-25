"""Independent NumPy/f64 readout reference for the Rust parity handoff.
This does not run a backbone. It validates exported candidate/marker features
against the saved numerical function and preserves explicit semantic none mass.
"""
from __future__ import annotations
import json,math
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file


def sm(z,t=1.):
    z=np.asarray(z,dtype=np.float64)/t;z=z-z.max();v=np.exp(z);return v/v.sum()


def frozen_logits(x,w,kind,has_none):
    x=np.asarray(x,dtype=np.float64)
    v=(x-w['rank.mean'])/w['rank.std']
    z=(v@w['rank.linear.weight'].astype('float64').T+w['rank.linear.bias']).reshape(-1)
    if not has_none:return z
    if kind=='constant':f=np.zeros(1)
    else:
        top=np.sort(z)[::-1];m=z.max()
        f=np.array([top[0],top[0]-top[1] if len(top)>1 else 0.,z.mean(),z.std(ddof=0),m+np.log(np.exp(z-m).sum())-math.log(len(z)),math.log(len(z))])
        if kind=='semantic_features':f=np.concatenate([f,v.mean(0),(sm(z)[:,None]*v).sum(0)])
    f=(f-w['reject.mean'])/w['reject.std']
    if kind=='semantic_features':
        hidden=np.tanh(f@w['reject.net.0.weight'].astype('float64').T+w['reject.net.0.bias'])
        none=hidden@w['reject.net.2.weight'].astype('float64').T+w['reject.net.2.bias']
    else:none=f@w['reject.net.weight'].astype('float64').T+w['reject.net.bias']
    return np.concatenate([z,np.asarray(none).reshape(1)])


def head_graph(profile):
    return {'method':profile['method'],'reference_implementation':'implementation/odij_head_reference.py',
      'frozen_readout_steps':[
        'V = (candidate_features - rank.mean) / rank.std; population-normalization buffers are already fitted.',
        'z = V @ rank.linear.weight.T + rank.linear.bias; one real logit per candidate.',
        'For Choice only: compute rejector features. Constant: [0]. Score-summary: [max, top-two gap, mean, population std, logmeanexp, log(K)].',
        'Semantic rejector: concatenate score-summary, mean(V), sum(softmax(z) * V).',
        'Standardize rejector features using reject.mean/std; apply stored linear, or linear→tanh→linear network.',
        'Append rejector output as native none logit ONLY for Choice. rank.none is an unused training parameter in the exported FrozenDecision graph.',
        'Noul and Score use only real logits. Divide all relevant logits by selected temperature; stable softmax.',
        'Host argmax breaks exact ties by first candidate index. Remap to original supplied IDs; policy threshold is a separate application action.'
      ],
      'encoder_native_steps':['Joint option-marker features include the explicit native none marker for Choice.',
        'z = marker_features @ head.weight.T + head.bias from training/trained.safetensors.',
        'Selected temperature then stable softmax.'],
      'finite_steps':['Renderer produces distinct single-token option codes; gather those output-embedding rows.',
        'z = selected_vocabulary_weights @ final_hidden + selected_bias, if present.',
        'Selected temperature then stable softmax. No free-text decoding; backbone/base matrix parity remains required.'],
      'arithmetic_note':'NumPy/f64 is an independent algebra check. It is not bitwise FP32 parity or Rust/Metal execution.'}


def check_head_fixtures(bundle):
    from odij_bundle import verify_bundle
    from odij_core import read_json,write_json
    b=Path(bundle);verify_bundle(b);p=read_json(b/'profile.json');contract=read_json(b/'inference_protocol.json');gold=read_json(b/'golden.json')
    records=[];base=b/p['artifact_dir']
    if p['method']=='frozen':w=load_file(base/p['head_file'])
    elif p['method']=='encoder_native':w=load_file(base/p['training_dir']/'trained.safetensors')
    else:return {'status':'not_applicable','reason':'Selected-vocabulary backbone projection requires base weights; retained token/logit fixtures are the next parity input.'}
    for item in gold:
        qs={q['id']:q for q in item['request']['questions']}
        for f in item['head_and_token_fixtures']:
            q=qs[f['question_id']]
            if p['method']=='frozen':z=frozen_logits(f['candidate_features'],w,p['kind'],q['primitive']=='choice')
            else:z=(np.asarray(f['marker_features'],dtype=np.float64)@w['head.weight'].astype('float64').T+w['head.bias']).reshape(-1)
            predicted=sm(z,p['calibration']['selected']);expected=np.asarray(f['probabilities'])
            delta=float(np.abs(predicted-expected).max())
            records.append({'question_id':q['id'],'max_logit_delta':float(np.abs(z-np.asarray(f['logits'])).max()),
                            'max_probability_delta':delta,'argmax_changed':bool(predicted.argmax()!=expected.argmax()),
                            'accepted':bool(delta<=contract['probability_tolerance'] and predicted.argmax()==expected.argmax())})
    return {'status':'passed' if records and all(r['accepted'] for r in records) else 'failed','records':records,
       'scope':'Independent NumPy/f64 algebra on nonfinal saved features. Not an independent backbone implementation or Rust test.'}
