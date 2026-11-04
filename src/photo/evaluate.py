"""One frozen evaluation, followed by source-disjoint robustness and saved figures."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from PIL import Image,ImageEnhance,ImageFilter

from .common import DATA,ROOT,LIMITS,save_json


def fingerprint():
    return hashlib.sha256((ROOT/'src/photo/review.py').read_bytes()).hexdigest()


def job(row):
    from .review import review_photo_pair
    start=time.perf_counter()
    result=review_photo_pair(ROOT/row['path1'],ROOT/row['path2'],row['reported_lat'],row['reported_lon'],**LIMITS)
    return {**row,**result,'elapsed_s':time.perf_counter()-start}


def summary(frame):
    normal=frame.expected_verdict.eq('요건 충족');positive=frame.verdict.eq('요건 충족')
    correct=frame.verdict.eq(frame.expected_verdict)
    exact1=frame.plate_1.fillna('').eq(frame.plate_true_1)
    exact2=frame.plate_2.fillna('').eq(frame.plate_true_2)
    visible2=frame['case'].ne('번호판 가림')
    return {'n_pairs':len(frame),'verdict_accuracy':float(correct.mean()),
            'normal_recall':float(positive[normal].mean()) if normal.any() else None,
            'rejection_recall':float((~positive[~normal]).mean()) if (~normal).any() else None,
            'balanced_accuracy':float((positive[normal].mean()+(~positive[~normal]).mean())/2) if normal.any() and (~normal).any() else None,
            'plate_exact_all':float((exact1.sum()+exact2.sum())/(2*len(frame))),
            'plate_exact_visible':float((exact1.sum()+exact2[visible2].sum())/(len(frame)+visible2.sum())),
            'n_visible_images':int(len(frame)+visible2.sum()),'false_accepts':int((positive&~normal).sum()),
            'false_rejections':int((~positive&normal).sum()),'majority_reject_baseline':float((~normal).mean())}


def write_results(results,path):
    frame=pd.DataFrame(results)
    for col in ['reasons','box_1','box_2','gps_1','gps_2']:
        if col in frame:
            frame[col]=frame[col].map(lambda x:json.dumps(x,ensure_ascii=False))
    frame.to_csv(path,index=False,encoding='utf-8-sig')


def perturb(path,kind,level,target):
    with Image.open(path) as raw:
        exif=raw.info.get('exif',b'')
        image=raw.convert('RGB')
        if kind=='blur':image=image.filter(ImageFilter.GaussianBlur(level))
        elif kind=='brightness':image=ImageEnhance.Brightness(image).enhance(level)
        elif kind=='tilt':image=image.rotate(level,resample=Image.Resampling.BICUBIC,expand=True,fillcolor=(35,35,35))
        else:raise ValueError('Unknown perturbation')
        target.parent.mkdir(parents=True,exist_ok=True)
        image.save(target,quality=95,subsampling=0,exif=exif)


def evaluate(workers=1,resume=False):
    if (DATA/'evaluation_finished.json').exists():
        raise RuntimeError('Final evaluation completed. Reuse saved results; --resume is only for an interrupted run.')
    output=DATA/'evaluation_results.csv'
    if output.exists() and not resume:
        raise RuntimeError('Final evaluation already exists. Reuse saved results; do not tune on the final set.')
    manifest=pd.read_csv(DATA/'manifest.csv',keep_default_na=False)
    manifest=manifest[manifest.evaluation_include.astype(str).str.lower().eq('true')].copy()
    dev=json.loads((DATA/'dev_protocol.json').read_text())
    assert not set(manifest.source_id)&set(dev['development_source_ids'])
    assert len(manifest)==204 and manifest['case'].value_counts().eq(34).all()
    before=fingerprint()
    freeze={'ocr_source_sha256':before,'thresholds':LIMITS,'n_pairs':len(manifest),
            'development_source_ids':dev['development_source_ids'],'evaluation_source_ids':sorted(manifest.source_id.unique().tolist()),
            'robustness':{'pairs':24,'sample_seed':20261030,'blur_px':[0,1.5,3],'brightness_multiplier':[1,.6,.3],'tilt_degrees':[0,10,20]},
            'workers':workers,'per_process_cpu_threads':1,'cuda_visible_devices':'0','torch_device_required':'cuda:0 RTX 5090',
            'stage':'frozen_before_final_evaluation','no_final_tuning':True}
    frozen_path=DATA/'evaluation_frozen.json'
    if frozen_path.exists():
        if not resume:
            raise RuntimeError('A frozen attempt exists; use --resume without tuning or altering inputs')
        previous=json.loads(frozen_path.read_text())
        assert previous['ocr_source_sha256']==before,'Cannot resume with a changed OCR implementation'
        assert previous['thresholds']==LIMITS and previous['evaluation_source_ids']==freeze['evaluation_source_ids']
    else:
        save_json(freeze,frozen_path)
    allowed=set(os.sched_getaffinity(0));cap=allowed&(set(range(16))|set(range(32,48)))
    if cap:os.sched_setaffinity(0,cap)
    started=time.perf_counter();results=[]
    progress=DATA/'evaluation_progress.jsonl'
    if resume and progress.exists():
        results=[json.loads(line) for line in progress.read_text().splitlines() if line.strip()]
    assert len({row['pair_id'] for row in results})==len(results),'Duplicate completed pair records'
    truth=manifest.set_index('pair_id')
    for row in results:
        assert row['pair_id'] in truth.index
        for key in ['plate_true_1','plate_true_2','case','reported_lat','reported_lon']:
            if key.startswith('reported_'):
                assert abs(float(row[key])-float(truth.loc[row['pair_id'],key]))<1e-9
            else:
                assert row[key]==truth.loc[row['pair_id'],key]
    completed={row['pair_id'] for row in results}
    with (DATA/'evaluation_attempts.jsonl').open('a') as handle:
        handle.write(json.dumps({'resume':resume,'workers':workers,'reused_completed_pairs':len(completed),'ocr_source_sha256':before,'reason':'Resource-only recovery from CUDA OOM; fixed inputs and model'})+'\n')
    print(f'reusing {len(completed)} completed pairs; workers={workers}',flush=True)
    context=multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=workers,mp_context=context) as pool:
        futures={pool.submit(job,row):row['pair_id'] for row in manifest.to_dict('records') if row['pair_id'] not in completed}
        for future in as_completed(futures):
            result=future.result();results.append(result)
            with (DATA/'evaluation_progress.jsonl').open('a') as handle:handle.write(json.dumps(result,ensure_ascii=False)+'\n')
            if len(results)%12==0:print(f'final pairs {len(results)}/{len(manifest)}',flush=True)
        results.sort(key=lambda x:x['pair_id'])
        assert fingerprint()==before,'OCR implementation changed during the frozen evaluation'
        write_results(results,output)
        base=pd.DataFrame(results);save_json(summary(base),DATA/'evaluation_summary.json')
        cases=[]
        for case,group in base.groupby('case',sort=False):cases.append({'case':case,**summary(group)})
        pd.DataFrame(cases).to_csv(DATA/'evaluation_by_case.csv',index=False,encoding='utf-8-sig')
        classes=['요건 충족','보완 필요'];cells=[]
        for expected in classes:
            for predicted in classes:
                cells.append({'expected':expected,'predicted':predicted,'count':int((base.expected_verdict.eq(expected)&base.verdict.eq(predicted)).sum())})
        pd.DataFrame(cells).to_csv(DATA/'confusion.csv',index=False,encoding='utf-8-sig')
        selected=pd.concat([part.sample(n=4,random_state=20261030) for _,part in manifest.groupby('case')],ignore_index=True)
        selected.to_csv(DATA/'robustness_selection.csv',index=False,encoding='utf-8-sig')
        robust_progress=DATA/'robustness_progress.jsonl'
        robust_results=[json.loads(line) for line in robust_progress.read_text().splitlines() if line.strip()] if resume and robust_progress.exists() else []
        robust_completed={(row['perturbation'],row['level'],row['pair_id']) for row in robust_results};jobs=[]
        for kind,levels in [('blur',[1.5,3]),('brightness',[.6,.3]),('tilt',[10,20])]:
            for level in levels:
                for row in selected.to_dict('records'):
                    transformed={**row,'perturbation':kind,'level':level}
                    for field in ['path1','path2']:
                        target=DATA/'robustness'/f'{kind}_{level}'/Path(row[field]).name
                        perturb(ROOT/row[field],kind,level,target)
                        transformed[field]=str(target.relative_to(ROOT))
                    if (kind,level,row['pair_id']) not in robust_completed:
                        jobs.append(transformed)
        futures={pool.submit(job,row):row for row in jobs}
        for future in as_completed(futures):
            robust_results.append(future.result())
            with robust_progress.open('a') as handle:handle.write(json.dumps(robust_results[-1],ensure_ascii=False)+'\n')
            if len(robust_results)%24==0:print(f'robustness pairs {len(robust_results)}/144',flush=True)
        write_results(robust_results,DATA/'robustness_results.csv')
        rows=[];selected_base=base[base.pair_id.isin(selected.pair_id)]
        for kind,zero in [('blur',0),('brightness',1),('tilt',0)]:
            rows.append({'perturbation':kind,'level':zero,**summary(selected_base)})
        for (kind,level),group in pd.DataFrame(robust_results).groupby(['perturbation','level']):
            rows.append({'perturbation':kind,'level':level,**summary(group)})
        pd.DataFrame(rows).to_csv(DATA/'robustness_summary.csv',index=False,encoding='utf-8-sig')
    # Qualitative originals here are display-model signs; retain them as explicit
    # negative controls rather than calling them real registration-plate accuracy.
    from .review import read_plate
    qualitative=pd.read_csv(DATA/'qualitative_manifest.csv',keep_default_na=False);out=[]
    for row in qualitative.to_dict('records'):
        with Image.open(ROOT/row['path']) as image:prediction=read_plate(image)
        out.append({**row,**prediction})
    pd.DataFrame(out).to_csv(DATA/'qualitative_results.csv',index=False,encoding='utf-8-sig')
    if (DATA/'ai_original_manifest.csv').exists():
        originals=pd.read_csv(DATA/'ai_original_manifest.csv',keep_default_na=False);raw_results=[]
        for row in originals.to_dict('records'):
            with Image.open(ROOT/row['path']) as image:prediction=read_plate(image)
            raw_results.append({**row,**prediction,'exact_match':prediction['plate']==row['raw_expected_plate'],'subset':'AI 원본 번호판'})
        raw_frame=pd.DataFrame(raw_results)
        raw_frame.to_csv(DATA/'ai_original_results.csv',index=False,encoding='utf-8-sig')
        strata=[('전체',raw_frame),('낮',raw_frame[raw_frame.time_of_day.eq('낮')]),('밤',raw_frame[raw_frame.time_of_day.eq('밤')]),('비',raw_frame[raw_frame.rain.astype(str).str.lower().eq('true')])]
        pd.DataFrame([{'condition':label,'n_images':len(group),'n_exact':int(group.exact_match.sum()),'exact_accuracy':float(group.exact_match.mean())} for label,group in strata]).to_csv(DATA/'ai_original_by_condition.csv',index=False,encoding='utf-8-sig')
    from .plots import render
    render()
    assert fingerprint()==before
    save_json({'execution_finished':True,'n_final_pairs':len(results),'n_robustness_pairs':len(robust_results),
               'n_qualitative_negative_controls':len(out),'elapsed_s':time.perf_counter()-started,'source_disjoint':True,
               'final_tuning':False,'ocr_source_sha256':before,'artifacts':['evaluation_results.csv','evaluation_summary.json','evaluation_by_case.csv','confusion.csv','robustness_summary.csv','qualitative_results.csv']},DATA/'evaluation_finished.json')
    print(json.dumps(json.loads((DATA/'evaluation_summary.json').read_text()),ensure_ascii=False),flush=True)
    print('Photo evaluation and figures finished',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=1);p.add_argument('--resume',action='store_true');args=p.parse_args();evaluate(args.workers,args.resume)
