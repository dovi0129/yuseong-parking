"""Generate source-attributed, perspective-composited demonstration pairs."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
import math
import re

import cv2
import numpy as np
import pandas as pd
import piexif
from PIL import Image, ImageDraw, ImageFont
from matplotlib.font_manager import FontProperties, findfont
from pyproj import Geod

from .common import DATA, LETTERS, LIMITS, ROOT, save_json
from .review import PLATE

GEOD = Geod(ellps="WGS84")
CASES = ["정상", "간격 미달", "위치 불일치", "다른 번호", "번호판 가림", "위치 정보 없음"]
DEV_SOURCES={"wm_99597207","wm_134022527"}
NEAR_DUPLICATE_EXCLUDED={"wm_109681829"}
# Corners inspected on downloaded source images, in clockwise normalized order.
QUADS = {
    "wm_99597207": [[.774,.711],[.887,.671],[.887,.731],[.773,.775]],
    "wm_134022527": [[.035,.616],[.150,.650],[.150,.714],[.035,.681]],
    "wm_199386154": [[.758,.590],[.913,.581],[.914,.624],[.758,.639]],
    "wm_154775355": [[.694,.697],[.766,.659],[.767,.716],[.697,.757]],
    "wm_135544162": [[.774,.751],[.941,.713],[.941,.789],[.775,.829]],
    "wm_135544164": [[.916,.690],[.987,.667],[.989,.720],[.916,.747]],
    "wm_136651582": [[.052,.656],[.194,.686],[.194,.755],[.052,.722]],
    "wm_109681829": [[.092,.557],[.252,.583],[.252,.643],[.091,.615]],
    "wm_174447438": [[.660,.650],[.840,.620],[.846,.690],[.664,.726]],
    "wm_146353915": [[.551,.575],[.677,.567],[.683,.612],[.555,.620]],
    "wm_133742075": [[.861,.500],[.937,.481],[.941,.542],[.864,.563]],
    "wm_108205201": [[.775,.454],[.875,.442],[.880,.531],[.779,.544]],
    "wm_109681849": [[.112,.588],[.257,.642],[.258,.717],[.112,.671]],
}


def random_plate(rng):
    prefix = str(int(rng.integers(100, 1000))) if rng.random() < .5 else str(int(rng.integers(10,100)))
    return prefix + str(rng.choice(list(LETTERS))) + f"{rng.integers(1000,10000):04d}"


def plate_image(number, occluded=False):
    image = Image.new("RGB", (520, 110), (247, 247, 245))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((1,1,518,108), radius=5, outline=(48,48,48), width=3)
    font = ImageFont.truetype(findfont(FontProperties(family="NanumGothic", weight="bold")), 88)
    glyph = Image.new("RGBA", (650, 115), (255,255,255,0))
    gd = ImageDraw.Draw(glyph)
    bbox = gd.textbbox((0,0),number,font=font,stroke_width=1)
    gd.text((-bbox[0],-bbox[1]),number,font=font,fill=(15,15,15),stroke_width=1)
    glyph = glyph.crop((0,0,bbox[2]-bbox[0],bbox[3]-bbox[1]))
    glyph = glyph.resize((min(glyph.width,470),76),Image.Resampling.LANCZOS)
    image.paste(glyph, ((520-glyph.width)//2,(110-glyph.height)//2),glyph)
    if occluded:
        ImageDraw.Draw(image).rectangle((145,3,355,107),fill=(48,48,48))
    return np.asarray(image)


def overlay(base, number, quad, occluded=False):
    h,w=base.shape[:2]
    target=np.asarray(quad,np.float32)*[w,h]
    source=np.array([[0,0],[519,0],[519,109],[0,109]],np.float32)
    matrix=cv2.getPerspectiveTransform(source,target.astype(np.float32))
    patch=cv2.warpPerspective(plate_image(number,occluded),matrix,(w,h))
    mask=cv2.warpPerspective(np.ones((110,520),np.uint8)*255,matrix,(w,h))
    result=base.copy();result[mask>0]=patch[mask>0]
    return result,target.tolist()


def exif_bytes(timestamp, gps=None):
    stamp=timestamp.strftime("%Y:%m:%d %H:%M:%S").encode()
    exif={"0th":{piexif.ImageIFD.Orientation:1},"Exif":{piexif.ExifIFD.DateTimeOriginal:stamp},"GPS":{},"1st":{},"thumbnail":None}
    if gps is not None:
        def dms(value):
            value=abs(value);degrees=int(value);minutes=int((value-degrees)*60);seconds=((value-degrees)*60-minutes)*60
            return ((degrees,1),(minutes,1),(round(seconds*1000000),1000000))
        lat,lon=gps
        exif["GPS"]={piexif.GPSIFD.GPSLatitudeRef:b'N' if lat>=0 else b'S',piexif.GPSIFD.GPSLatitude:dms(lat),
                     piexif.GPSIFD.GPSLongitudeRef:b'E' if lon>=0 else b'W',piexif.GPSIFD.GPSLongitude:dms(lon),
                     piexif.GPSIFD.GPSVersionID:(2,3,0,0)}
    return piexif.dump(exif)


def offset(gps, metres, azimuth=90):
    lon,lat,_=GEOD.fwd(gps[1],gps[0],azimuth,metres)
    return lat,lon


def save_photo(array, path, timestamp, gps):
    path.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(array).save(path, quality=95, subsampling=0, exif=exif_bytes(timestamp,gps))


def ingest_ai():
    """Filename truth is used exclusively by the generator, never by OCR."""
    directory=DATA/'ai_generated';directory.mkdir(parents=True,exist_ok=True)
    registry=DATA/'ai_inputs.json'
    if registry.exists():
        quads=json.loads((DATA/'ai_plate_quads.json').read_text())
        scenes=['횡단보도','소화전','버스정류장','교차로 모퉁이','어린이보호구역','인도','횡단보도','소화전','버스정류장','교차로 모퉁이']
        rows=[]
        for item in json.loads(registry.read_text())['images']:
            index=item['index'];sid=item['source_id']
            rows.append({'source_id':sid,'source_type':'AI 생성(ChatGPT)','path':item['path'],'original_path':item['original_path'],
                         'source_url':'사용자 제공 ChatGPT 생성 사진','image_url':'','author':'사용자 제공(ChatGPT)',
                         'license':'AI 생성(ChatGPT)','license_url':'','adaptation':'평가 쌍은 무작위 합성 번호판·합성 EXIF, AI 원본 판독은 별도 부분집합',
                         'title':sid,'raw_expected_plate':item['plate_true'],'quad':quads[sid],
                         'scene':scenes[index-1],'time_of_day':'밤' if index in [3,4,7,8] else '낮','rain':index in [4,8]})
        save_json({'pending':[],'ready':len(rows),'copies':'ai_bases/ai01~ai10, 원본 바이트 유지'},DATA/'ai_ingest_status.json')
        return rows
    custom_path=DATA/'ai_regions.json'
    custom=json.loads(custom_path.read_text()) if custom_path.exists() else {}
    rows=[];unannotated=[]
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in ('.jpg','.jpeg','.png','.webp'):
            continue
        match=re.search(r'[0-9]{2,3}['+LETTERS+r'][0-9]{4}',path.stem)
        if not match:
            unannotated.append({'file':path.name,'reason':'파일명 정답 번호 없음'});continue
        if path.name not in custom:
            unannotated.append({'file':path.name,'reason':'번호판 사각형 시각 확인 필요'});continue
        sid='ai_'+path.stem
        rows.append({'source_id':sid,'source_type':'AI 생성','path':str(path.relative_to(ROOT)),
                     'source_url':'사용자 제공 AI 생성 사진','image_url':'','author':'사용자 제공',
                     'license':'AI 생성','license_url':'','adaptation':'시연용 합성 번호판으로 정답 정규화; 시각·GPS는 합성 EXIF',
                     'title':path.name,'expected_plate':match.group(0),'quad':custom[path.name]})
    save_json({'pending':unannotated,'ready':len(rows)},DATA/'ai_ingest_status.json')
    return rows


def generate(n=204, seed=20261030, dev=False):
    if not dev and (DATA/'evaluation_frozen.json').exists():
        raise RuntimeError('Final photo inputs are frozen. Preserve this version; prepare a separately authorized version for new inputs.')
    rng=np.random.default_rng(seed)
    sources=pd.read_csv(DATA/'sources.csv',keep_default_na=False)
    wm=sources[sources.source_type.eq('위키미디어 공용') & sources.source_id.isin(QUADS)].to_dict('records')
    for row in wm:row['quad']=QUADS[row['source_id']]
    ai=ingest_ai()
    eval_wm=[row for row in wm if row['source_id'] not in DEV_SOURCES|NEAR_DUPLICATE_EXCLUDED]
    backgrounds=[row for row in wm if row['source_id'] in DEV_SOURCES] if dev else (ai if ai else eval_wm)
    if ai:
        keep=sources[~sources.source_type.str.startswith('AI 생성')]
        export=pd.DataFrame([{k:v for k,v in row.items() if k not in ('quad','expected_plate')} for row in ai])
        pd.concat([keep,export],ignore_index=True).to_csv(DATA/'sources.csv',index=False,encoding='utf-8-sig')
    arrays={row['source_id']:np.asarray(Image.open(ROOT/row['path']).convert('RGB')) for row in backgrounds}
    phase='dev' if dev else 'pairs'
    rows=[]
    for index in range(n):
        case=CASES[index%len(CASES)];background=backgrounds[(index//len(CASES))%len(backgrounds)]
        plate1=random_plate(rng)
        plate2=random_plate(rng) if case=='다른 번호' else plate1
        while case=='다른 번호' and plate2==plate1:plate2=random_plate(rng)
        base=arrays[background['source_id']]
        first,quad=overlay(base,plate1,background['quad'])
        second,_=overlay(base,plate2,background['quad'],occluded=case=='번호판 가림')
        h,w=base.shape[:2]
        angle=float(rng.uniform(-1.3,1.3));scale=float(rng.uniform(.985,1.015))
        matrix=cv2.getRotationMatrix2D((w/2,h/2),angle,scale);matrix[:,2]+=rng.uniform(-3,3,2)
        second=cv2.warpAffine(second,matrix,(w,h),borderMode=cv2.BORDER_REFLECT)
        second=np.clip(second.astype(float)*rng.uniform(.88,1.08),0,255).astype(np.uint8)
        t1=datetime(2024,7,1,20 if background.get('time_of_day')=='밤' else 12,0,0)+timedelta(days=index%28,minutes=index*2)
        gap=int(rng.integers(5,55)) if case=='간격 미달' else int(rng.integers(65,181))
        reported=(36.3567+float(rng.uniform(-.02,.02)),127.3486+float(rng.uniform(-.02,.02)))
        gps1=reported;gps2=offset(gps1,float(rng.uniform(1,10)),float(rng.uniform(0,360)))
        subtype=''
        if case=='위치 불일치':
            if (index//6)%2==0:
                gps2=offset(gps1,float(rng.uniform(75,150)));subtype='촬영 위치 불일치'
            else:
                gps1=offset(reported,float(rng.uniform(80,140)))
                gps2=offset(gps1,float(rng.uniform(1,8)));subtype='신고 위치 불일치'
        if case=='위치 정보 없음':gps2=None
        pair_id=f'{phase}_{index+1:04d}'
        path1=DATA/phase/f'{pair_id}_a.jpg';path2=DATA/phase/f'{pair_id}_b.jpg'
        save_photo(first,path1,t1,gps1);save_photo(second,path2,t1+timedelta(seconds=gap),gps2)
        rows.append({'pair_id':pair_id,'path1':str(path1.relative_to(ROOT)),'path2':str(path2.relative_to(ROOT)),
                     'case':case,'subcase':subtype,'reported_lat':reported[0],'reported_lon':reported[1],
                     'plate_true_1':plate1,'plate_true_2':plate2,'expected_verdict':'요건 충족' if case=='정상' else '보완 필요',
                     'source_id':background['source_id'],'source_type':background['source_type'],
                     'synthetic_exif':True,'gap_true_s':gap,'quad_1':json.dumps(quad),'evaluation_include':not dev,
                     'subset':'합성 번호판','scene':background.get('scene',''),'time_of_day':background.get('time_of_day',''),'rain':background.get('rain',False)})
    frame=pd.DataFrame(rows);target=DATA/('dev_manifest.csv' if dev else 'manifest.csv')
    frame.to_csv(target,index=False,encoding='utf-8-sig')
    protocol={'thresholds':LIMITS,'source_preference':'시각 확인한 AI 생성 사진은 평가 쪽에만 사용; 개발은 별도 위키미디어 두 사진',
               'seed':seed,'n_pairs':n,'case_counts':frame['case'].value_counts().to_dict(),
               'metadata':'모든 촬영 시각·GPS는 시연용으로 합성; 실제 촬영 증거가 아님',
               'ground_truth_is_not_an_ocr_input':True,'plate_format':'자가용 2~3자리 숫자+허용 한글 1글자+4자리 숫자',
               'source_type':backgrounds[0]['source_type'],
               'development_source_ids':sorted(DEV_SOURCES),'evaluation_source_ids':[row['source_id'] for row in (ai if ai else eval_wm)],
               'near_duplicate_excluded':sorted(NEAR_DUPLICATE_EXCLUDED),
               'split_rule':'동일 위키미디어 파일·근접 장면을 개발/평가 사이에 공유하지 않음. 사용자 AI 사진은 평가에만 추가.',
               'used_source_ids':sorted(frame.source_id.unique().tolist())}
    save_json(protocol,DATA/('dev_protocol.json' if dev else 'protocol.json'))
    if not dev:
        previous=json.loads((DATA/'dev_protocol.json').read_text())
        previous.update({k:protocol[k] for k in ['development_source_ids','evaluation_source_ids','near_duplicate_excluded','split_rule']})
        save_json(previous,DATA/'dev_protocol.json')
        if ai:
            first=next(row for row in ai if row['source_id']=='ai01')
            raw=Image.open(ROOT/first['path']).convert('RGB');array=np.asarray(raw)
            second=np.clip(array.astype(float)*.99,0,255).astype(np.uint8)
            gps=(36.3567,127.3486);t=datetime(2024,7,1,12)
            path1=DATA/'demo_original/ai01_a.png';path2=DATA/'demo_original/ai01_b.png';path1.parent.mkdir(parents=True,exist_ok=True)
            raw.save(path1,exif=exif_bytes(t,gps));Image.fromarray(second).save(path2,exif=exif_bytes(t+timedelta(seconds=90),offset(gps,5)))
            demo={'pair_id':'demo_ai01_original_normal','path1':str(path1.relative_to(ROOT)),'path2':str(path2.relative_to(ROOT)),
                  'case':'정상','reported_lat':gps[0],'reported_lon':gps[1],'plate_true_1':'123가4567','plate_true_2':'123가4567',
                  'expected_verdict':'요건 충족','source_id':'ai01','source_type':'AI 생성(ChatGPT)','evaluation_include':False,
                  'subset':'AI 원본 시연','scene':first['scene'],'time_of_day':first['time_of_day'],'rain':first['rain']}
            frame=pd.concat([pd.DataFrame([demo]),frame],ignore_index=True)
            frame.to_csv(target,index=False,encoding='utf-8-sig')
            originals=pd.DataFrame([{k:v for k,v in row.items() if k!='quad'} for row in ai])
            originals.to_csv(DATA/'ai_original_manifest.csv',index=False,encoding='utf-8-sig')
    print(f'{phase}: generated {n} pairs, source={backgrounds[0]["source_type"]}',flush=True)
    return frame


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--n',type=int,default=204);parser.add_argument('--seed',type=int,default=20261030);parser.add_argument('--dev',action='store_true')
    args=parser.parse_args();generate(args.n,args.seed,args.dev)
