"""Proposal figures from frozen photo-evaluation tables, at 200 dpi."""
from __future__ import annotations

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image

from .common import DATA,FIG,ROOT,save_json
from src.pipeline import paper_figstyle as ps


def save(fig,name,caption):
    FIG.mkdir(parents=True,exist_ok=True)
    fig.savefig(FIG/f'photo_{name}.png',dpi=200,bbox_inches='tight');plt.close(fig)
    path=DATA/'figure_captions.json';captions=json.loads(path.read_text()) if path.exists() else {}
    captions[f'photo_{name}.png']=caption;save_json(captions,path)


def confusion():
    rows=pd.read_csv(DATA/'confusion.csv');classes=['요건 충족','보완 필요']
    matrix=np.array([[int(rows[rows.expected.eq(a)&rows.predicted.eq(b)]['count'].iloc[0]) for b in classes] for a in classes])
    ps.apply(.78);fig,ax=plt.subplots();plot=ax.imshow(matrix,cmap='Blues',vmin=0)
    for i in range(2):
        for j in range(2):ax.text(j,i,str(matrix[i,j]),ha='center',va='center',fontsize=18,color='white' if matrix[i,j]>matrix.max()*.6 else '#222222')
    ax.set(xticks=[0,1],xticklabels=classes,yticks=[0,1],yticklabels=classes,xlabel='모듈 판정',ylabel='정답 판정')
    fig.colorbar(plot,ax=ax,label='사진 쌍 (개)',shrink=.8)
    save(fig,'confusion','바탕 사진이 개발 표본과 겹치지 않는 합성 번호판 204쌍. 정상 34쌍, 보완 필요 170쌍. 셀은 사진 쌍 수이다.')


def robustness():
    frame=pd.read_csv(DATA/'robustness_summary.csv')
    ps.apply(.38);fig,axes=plt.subplots(1,3,figsize=(9.2,3.5),sharey=True)
    labels={'blur':'흐림 반경 (px)','brightness':'밝기 배수','tilt':'기울기 (°)'}
    for ax,kind in zip(axes,['blur','brightness','tilt']):
        rows=frame[frame.perturbation.eq(kind)].sort_values('level')
        ax.plot(rows.level,rows.plate_exact_visible*100,'o-',color='#0072B2',label='번호 완전 일치')
        ax.plot(rows.level,rows.balanced_accuracy*100,'s--',color='#D55E00',label='요건 균형 정확도')
        ax.set(xlabel=labels[kind],ylim=(0,105),xticks=rows.level)
        if kind=='brightness':ax.invert_xaxis()
    axes[0].set_ylabel('정확도 (%)');axes[0].legend(loc='lower left',fontsize=7)
    save(fig,'robustness','평가 표본의 유형별 4쌍, 총 24쌍을 고정해 강도만 바꿨다. 번호 일치율은 가린 두 번째 번호판을 제외하며, 균형 정확도는 정상 수용률과 거부 사례 검출률의 평균이다. 각 강도는 같은 24쌍이다.')


def examples():
    frame=pd.read_csv(DATA/'evaluation_results.csv',keep_default_na=False)
    ps.apply(1.2);fig,axes=plt.subplots(3,2,figsize=(9.2,10.6))
    for index,(ax,(case,rows)) in enumerate(zip(axes.flat,frame.groupby('case',sort=False)),start=1):
        preferred=rows[rows.source_id.eq(f'ai{index:02d}')]
        row=(preferred if len(preferred) else rows).sort_values('pair_id').iloc[0]
        with Image.open(ROOT/row.path2) as raw:image=raw.convert('RGB')
        ax.imshow(image);box=json.loads(row.box_2)
        if box:
            x1,y1,x2,y2=box;ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,color='#D55E00',linewidth=1.5))
        ax.set_axis_off()
        plate=row.plate_2 if row.plate_2 else '판독 불가'
        ax.text(.01,.98,f'{case} · {row.verdict}',transform=ax.transAxes,va='top',fontsize=8,
                bbox={'facecolor':'white','edgecolor':'none','alpha':.9,'pad':2})
        ax.text(.01,.02,plate,transform=ax.transAxes,fontsize=9,bbox={'facecolor':'white','edgecolor':'none','alpha':.9,'pad':2})
    save(fig,'examples','사례 유형별로 ai01~ai06의 서로 다른 배경에서 pair_id가 가장 빠른 쌍을 표시하였다. 예측 성공 여부로 예제를 고르지 않았다. 상자는 모듈이 읽은 번호 영역이며 사진은 AI 장면에 합성 번호판을 덮은 시연 자료이다.')


def originals():
    path=DATA/'ai_original_by_condition.csv'
    if not path.exists():return
    frame=pd.read_csv(path)
    ps.apply(.65);fig,ax=plt.subplots()
    ax.bar(frame.condition,frame.exact_accuracy*100,color='#0072B2',width=.55)
    for i,row in enumerate(frame.itertuples(index=False)):
        ax.text(i,row.exact_accuracy*100+2,f'{row.n_exact}/{row.n_images}',ha='center',fontsize=9)
    ax.set(ylabel='번호 완전 일치율 (%)',ylim=(0,113),xlabel='AI 원본 조건')
    save(fig,'ai_original','번호판을 덮지 않은 AI 원본 10장, 정답은 모두 123가4567이다. 낮 6장·밤 4장·비 2장이며 비는 밤의 부분집합이다. 막대 위 숫자는 일치 수/사진 수이다.')


def render():
    confusion();robustness();examples();originals()
