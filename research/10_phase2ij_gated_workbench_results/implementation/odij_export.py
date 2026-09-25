"""Compact scientific handoff; big weights/cache/optimizer snapshots stay in resumable Drive storage."""
from __future__ import annotations
import os,zipfile
from pathlib import Path
from odij_core import *
from odij_runner import status_report


def export_results(root,drive):
    root,drive=Path(root),Path(drive)
    summary=status_report(root);profiles=[];final=[]
    if (root/'MODEL_LOCK.json').exists():
        lock=read_json(root/'MODEL_LOCK.json')
        profiles=[{'profile_id':m['profile_id'],'name':m['name'],'method':m['method'],'spec':m.get('spec'),'layout':m.get('layout'),'development':m['development'],'benchmark':m.get('benchmark'),'selected_before_final':m['profile_id']==lock['selected_profile']} for m in lock['profiles']]
    for p in sorted((root/'final').glob('*/FINAL_DONE.json')):
        row=read_json(p);final.append(row)
    compact={'schema':'openkind-2ij-paste-back/v1','version':VERSION,'source_H_archive_sha256':SOURCE_SHA256,
        'study_id':root.name,'overall_execution_status':summary['status'],'gates':summary['gate_report'],'stages':summary['stages'],
        'registered_scope':read_json(root/'registered_protocol.json')['scope'] if (root/'registered_protocol.json').exists() else None,
        'model_profiles':profiles,'final_results':final,
        'limits':['New human review is not auto-approved.','Historical H is not retrained or used for new model selection.','Model/renderer/none methods and supervision differ; matched claims are arm-specific.','Construction/probe completion is not all roadmap work completed.','No external Laya/GLiClass adapter, native Rust/Metal, or production service claim.']}
    write_json(root/'paste_back_summary.json',compact)
    out=root/'exports';out.mkdir(exist_ok=True);archive=out/('openkind_2ij_'+root.name+'.zip');tmp=archive.with_suffix('.tmp.zip')
    selected=[]
    for p in sorted(root.rglob('*')):
        rel=p.relative_to(root)
        if not p.is_file() or any(x in rel.parts for x in ('source','exports','feature_cache','__pycache__')):continue
        if any(part.startswith('ckpt-') for part in rel.parts) or '.tmp-' in p.name:continue
        if p.suffix not in ('.json','.jsonl','.md','.py','.log'):continue
        selected.append(p)
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
        for p in selected:z.write(p,p.relative_to(root).as_posix())
        for p in sorted(Path(__file__).parent.glob('odij_*.py')):z.write(p,'implementation/'+p.name)
        z.writestr('COMPACT_ARCHIVE_SCOPE.txt','This compact report excludes source H archives, model weights, optimizer states and SQLite feature caches. Resume from the separate complete Drive snapshot generations, not this ZIP.\n')
    os.replace(tmp,archive);drive.mkdir(parents=True,exist_ok=True)
    for p in (archive,root/'paste_back_summary.json',root/'REPORT.md',root/'status.json'):
        atomic_copy(p,drive/p.name)
    return {'archive':str(archive),'drive_archive':str(drive/archive.name),'paste_back_summary':str(root/'paste_back_summary.json'),'status':summary['status']}
