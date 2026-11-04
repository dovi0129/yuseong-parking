"""Check completed photo artifacts, inputs, and fixed evaluation provenance."""
import ast
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .common import DATA,FIG,ROOT,LIMITS,save_json
from .evaluate import fingerprint


def run():
    checks=[]
    def require(value,label):
        if not bool(value):raise AssertionError(label)
        checks.append(label)
    finished=json.loads((DATA/'evaluation_finished.json').read_text())
    frozen=json.loads((DATA/'evaluation_frozen.json').read_text())
    require(finished['execution_finished'],'complete runner marker')
    require(finished['ocr_source_sha256']==fingerprint()==frozen['ocr_source_sha256'],'OCR code unchanged across frozen evaluation and handoff')
    require(frozen['thresholds']==LIMITS=={'min_gap_s':60,'max_move_m':15,'max_report_m':50},'pre-fixed 60s/15m/50m criteria')
    manifest=pd.read_csv(DATA/'manifest.csv',keep_default_na=False)
    evaluation=manifest[manifest.evaluation_include.astype(str).str.lower().eq('true')]
    require(len(manifest)==205 and len(evaluation)==204,'204 evaluation pairs and one separate UI demo')
    require(evaluation['case'].value_counts().eq(34).all(),'six balanced case types')
    require(evaluation.source_id.nunique()==10 and evaluation.source_type.eq('AI 생성(ChatGPT)').all(),'ten AI evaluation backgrounds')
    dev=json.loads((DATA/'dev_protocol.json').read_text())
    require(not set(dev['development_source_ids'])&set(evaluation.source_id),'development/evaluation background disjointness')
    require(set(evaluation.source_id)==set(dev['evaluation_source_ids'])==set(frozen['evaluation_source_ids']),'actual source lists agree')
    results=pd.read_csv(DATA/'evaluation_results.csv',keep_default_na=False)
    require(len(results)==204 and results.pair_id.is_unique,'one persisted prediction per evaluation pair')
    require(set(results.pair_id)==set(evaluation.pair_id),'all and only final pair IDs evaluated')
    require(results.verdict.isin(['요건 충족','보완 필요']).all(),'valid verdict vocabulary')
    require((results.gap_s==results.gap_true_s).all(),'serialized EXIF gaps match the generated truth')
    normal=results['case'].eq('정상')
    require((results.loc[normal,'gap_s']>=60).all() and (results.loc[normal,'move_m'].astype(float)<=15).all(),'normal EXIF cases satisfy time/move bounds')
    require(results.loc[results['case'].eq('간격 미달'),'gap_s'].lt(60).all(),'short-gap cases stay below fixed threshold')
    for field in ['plate_1','plate_2','box_1','box_2','taken_1','taken_2','gps_1','gps_2','gap_s','move_m','report_m','verdict','reasons']:
        require(field in results.columns,f'public result key: {field}')
    cells=pd.read_csv(DATA/'confusion.csv')
    for row in cells.itertuples(index=False):
        require(row.count==int((results.expected_verdict.eq(row.expected)&results.verdict.eq(row.predicted)).sum()),'confusion cell agrees with raw predictions')
    summary=json.loads((DATA/'evaluation_summary.json').read_text())
    require(np.isclose(summary['verdict_accuracy'],results.verdict.eq(results.expected_verdict).mean()),'decision accuracy recomputed from raw predictions')
    exact1=results.plate_1.eq(results.plate_true_1);exact2=results.plate_2.eq(results.plate_true_2)
    visible=results['case'].ne('번호판 가림')
    require(np.isclose(summary['plate_exact_all'],(exact1.sum()+exact2.sum())/408),'all-image exact plate accuracy')
    require(np.isclose(summary['plate_exact_visible'],(exact1.sum()+exact2[visible].sum())/374),'visible-image exact plate denominator')
    robust=pd.read_csv(DATA/'robustness_results.csv')
    require(len(robust)==144 and not robust[['perturbation','level','pair_id']].duplicated().any(),'144 unique fixed robustness variants')
    require(pd.read_csv(DATA/'robustness_summary.csv').n_pairs.eq(24).all(),'same 24 pairs in every robustness condition')
    originals=pd.read_csv(DATA/'ai_original_results.csv')
    require(len(originals)==10 and originals.source_id.is_unique,'ten unpainted original AI images evaluated separately')
    require(originals.raw_expected_plate.eq('123가4567').all(),'original AI truth fixed independently of OCR')
    require((DATA/'real_plate_results.csv').exists() and len(pd.read_csv(DATA/'real_plate_results.csv'))==5,'five real-photo qualitative results kept separately')
    demo=json.loads((DATA/'ai01_demo_result.json').read_text())
    require(demo['plate_1']==demo['plate_2']=='123가4567' and demo['verdict']=='요건 충족','original ai01 normal UI demo really runs')
    sources=pd.read_csv(DATA/'sources.csv',keep_default_na=False)
    used=set(manifest.source_id)|set(dev['development_source_ids'])|set(pd.read_csv(DATA/'real_plate_results.csv').source_id)
    require(used.issubset(set(sources.source_id)),'used sources have attribution/license records')
    require(sources[sources.source_type.eq('위키미디어 공용')].license.str.startswith(('CC BY','CC0','Public domain')).all(),'Wikimedia sources use allowed licenses')
    constraints={line.split('==')[0].lower().replace('_','-'):line.split('==')[1] for line in (DATA/'install_constraints.txt').read_text().splitlines()}
    current={dist.metadata['Name'].lower().replace('_','-'):dist.version for dist in metadata.distributions() if dist.metadata.get('Name')}
    require(all(current[name]==version for name,version in constraints.items()),'all 63 original package versions preserved')
    for stem in ['confusion','robustness','examples','ai_original']:
        path=FIG/f'photo_{stem}.png'
        with Image.open(path) as image:
            require(min(image.info.get('dpi',(0,0)))>=199.8,f'{path.name}: requested 200dpi')
            image.verify()
    for path in (ROOT/'src/photo').glob('*.py'):ast.parse(path.read_text())
    checks.append('all owned photo Python files parse')
    weights={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in (DATA/'models/easyocr').glob('*.pth')}
    save_json({'checks':checks,'n_checks':len(checks),'weight_sha256':weights,
               'reused_validation':'20 known-answer rule/EXIF/orientation tests; byte-identical AI copies checked before inference; four figure files inspected visually',
               'failed_attempt':'CUDA OOM with two processes, retained in evaluation_run.log and evaluation_attempts.jsonl; resumed with one process and reused two completed records',
               'no_final_retuning':True},DATA/'photo_checks.json')
    print(f'PASS: {len(checks)} photo artifact checks; fixed OCR, inputs, criteria and source split')


if __name__=='__main__':run()
