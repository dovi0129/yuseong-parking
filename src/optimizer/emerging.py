"""Save descriptive annual-vs-operational grade differences; never used to plan."""
import pandas as pd

from .domain import ROOT
from .processing import load_catalog,grade,save_csv


def main():
    reports=pd.read_parquet(ROOT/"data/processed/reports.parquet",columns=["ts","spot_id"])
    reports["ts"]=pd.to_datetime(reports.ts)
    year=reports.loc[reports.ts.between("2024-01-01","2025-01-01",inclusive="left")]
    annual=year.groupby("spot_id").size().rename("연간 신고 수")
    result=load_catalog().join(annual,on="spot_id")
    result=result.loc[(result["연간 신고 수"]>=20)&(result.n_reports_train<10)].copy()
    result["10~12월 신고 수"]=result["연간 신고 수"]-result.n_reports_train
    result["연간 지점 등급"]=result["연간 신고 수"].map(grade)
    result["운영 기준 등급"]=result.n_reports_train.map(grade)
    result=result.rename(columns={"spot_id":"지점 ID","label":"위치","dong":"동","lat":"위도","lon":"경도","n_reports_train":"1~9월 신고 수"})
    columns=["지점 ID","위치","동","위도","경도","연간 신고 수","1~9월 신고 수","10~12월 신고 수","연간 지점 등급","운영 기준 등급"]
    result=result[columns].sort_values(["연간 신고 수","지점 ID"],ascending=[False,True])
    print(save_csv(result,ROOT/"data/processed/emerging_spots.csv"))


if __name__=="__main__":main()
