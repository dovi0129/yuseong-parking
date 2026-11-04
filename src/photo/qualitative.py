"""Unretouched real-photo observations, separate from synthetic final scores."""
import json
import pandas as pd
from PIL import Image

from .common import DATA,ROOT,save_json
from .review import read_plate,review_photo_pair


def run():
    path=DATA/'real_plate_results.csv'
    if path.exists():
        raise RuntimeError('Qualitative results exist; reuse them without retuning')
    records=pd.read_csv(DATA/'real_plate_manifest.csv',keep_default_na=False).to_dict('records')
    out=[]
    for row in records:
        with Image.open(ROOT/row['path']) as image:predicted=read_plate(image)
        truth=row.get('plate_true','')
        out.append({**row,**predicted,'exact_match':predicted['plate']==truth if truth else None})
    pd.DataFrame(out).to_csv(path,index=False,encoding='utf-8-sig')
    manifest=pd.read_csv(DATA/'manifest.csv',keep_default_na=False)
    demo=manifest[manifest.pair_id.eq('demo_ai01_original_normal')].iloc[0]
    result=review_photo_pair(ROOT/demo.path1,ROOT/demo.path2,demo.reported_lat,demo.reported_lon)
    save_json(result,DATA/'ai01_demo_result.json')
    print('real qualitative photos',len(out),'ai01 original demo',result,flush=True)


if __name__=='__main__':run()
