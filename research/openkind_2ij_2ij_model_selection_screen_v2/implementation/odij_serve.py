"""Resident native JSONL study worker. NOT a Jev HTTP server or Rust integration.
One request at a time; host owns admission/auth/tenant routing. No public socket/tunnel is opened.
"""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
from odij_core import *
from odij_data import validate_case,flatten,choices_for
from odij_runner import Predictor

MAX_LINE_BYTES=256*1024

def native_response(row,p):
    opts=choices_for(row['q']);ix=int(p.argmax());q=row['q'];out={'question_id':q['id'],'primitive':q['primitive'],
        'selected_id':None if opts[ix]['id']==NONE else opts[ix]['id'],
        'option_probabilities':{o['id']:float(p[i]) for i,o in enumerate(q['options'])},
        'none_probability':float(p[-1]) if q['primitive']=='choice' else None,'top_probability':float(p.max()),
        'confidence':None,'confidence_note':'Vendor confidence is not implemented; top_probability is explicitly separate.'}
    if q['primitive']=='noul':out['noul']=float(p[1])
    if q['primitive']=='score':out['score']=float(sum(float(o['value'])*p[i] for i,o in enumerate(q['options'])))
    return out

def request_rows(req,p):
    if not isinstance(req,dict) or not nonempty(req.get('request_id')) or not nonempty(req.get('state')):raise ValueError('request_id and state required')
    qs=req.get('questions',[])
    if not isinstance(qs,list) or not 1<=len(qs)<=p['max_questions']:raise ValueError('Request question bound exceeded')
    native=[];families=set(p['train_families']+p['dev_transfer_families']+p['final_transfer_families'])
    for q in qs:
        if q.get('family') not in families:raise ValueError('Unsupported task family')
        if 'target_id' in q or 'origin' in q:raise ValueError('Inference requests must not provide labels or answerability targets')
        if not 2<=len(q.get('options',[]))<=p['max_candidates']:raise ValueError('Candidate bound exceeded')
        # Schema validation uses a local placeholder only. It is never rendered, scored, or returned as truth.
        v=dict(q,target_id=q['options'][0]['id'],origin='answer_present');native.append(v)
    c=validate_case({'id':req['request_id'],'group_id':req['request_id'],'state':req['state'],'split':'dev',
                   'source':{'kind':'natural','reference':'caller input; unvalidated truth','author':'caller'},'questions':native})
    return flatten([c])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study-root',required=True);ap.add_argument('--profile-id');args=ap.parse_args()
    from odij_data import assert_study_intact
    root=Path(args.study_root);assert_study_intact(root);lock=read_json(root/'MODEL_LOCK.json');p=read_json(root/'registered_protocol.json')
    for n,h in lock['artifact_hashes'].items():
        if file_sha(root/n)!=h:raise IntegrityError('Locked service profile changed')
    pid=args.profile_id or lock['selected_profile'];profile=next(x for x in lock['profiles'] if x['profile_id']==pid)
    pred=Predictor(root,p,profile,root/'native_worker_runtime')
    try:
        while True:
            line=sys.stdin.buffer.readline(MAX_LINE_BYTES+1)
            if not line:break
            if len(line)>MAX_LINE_BYTES:
                # Reject and exit instead of interpreting the remaining bytes as another request.
                print(json.dumps({'error':{'code':'request_too_large'}}),flush=True);break
            rid=None;started=time.monotonic()
            try:
                req=json.loads(line);rid=req.get('request_id');budget=req.get('deadline_ms',60000)
                if not isinstance(budget,(int,float)) or not 0<budget<=600000:raise ValueError('Invalid deadline_ms')
                rows=request_rows(req,p);answers=[]
                for r in rows:
                    if (time.monotonic()-started)*1000>budget:raise TimeoutError('Deadline before dispatch')
                    answers.append(native_response(r,pred.predict(r,memo=False)))
                elapsed=(time.monotonic()-started)*1000
                if elapsed>budget:raise TimeoutError('Deadline after nonpreemptive model operation')
                out={'schema':'openkind-native-study-response/v1','request_id':rid,'engine_profile':pid,'experimental':True,'answers':answers,'elapsed_ms':elapsed}
            except Exception as e:
                out={'request_id':rid,'error':{'code':type(e).__name__,'message':'Request rejected; no partial decision returned.'}}
            print(json.dumps(out,allow_nan=False),flush=True)
    finally:pred.close()
if __name__=='__main__':main()
