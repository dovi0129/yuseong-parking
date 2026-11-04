"""Bounded local launcher with an explicit stage selector and completion marker."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from .domain import ROOT,ROUTES


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--only",choices=("evaluate","b3","figures","exports","all"),default="all")
    parser.add_argument("--date",default="2024-10-02")
    args=parser.parse_args()
    logdir=ROOT/"artifacts/logs";logdir.mkdir(parents=True,exist_ok=True)
    done=ROUTES/"evaluation.done";done.unlink(missing_ok=True)
    status=ROUTES/"run_status.json"
    commands={"evaluate":["-m","src.optimizer.evaluate","--solver-seconds","30"],
              "b3":["-m","src.optimizer.evaluate_b3"],
              "figures":["-m","src.optimizer.figures"],
              "exports":["-m","src.optimizer.processing","--date",args.date,"--start","17:00","--end","21:00","--vehicles","2"]}
    stages=list(commands) if args.only=="all" else [args.only]
    for stage in stages:
        status.write_text(json.dumps({"state":"running","stage":stage,"started_unix":time.time()}))
        log=logdir/f"optimizer_{stage}_launcher_v1.log"
        with log.open("w") as stream:
            completed=subprocess.run([sys.executable,"-u"]+commands[stage],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        if completed.returncode:
            status.write_text(json.dumps({"state":"failed","stage":stage,"log":str(log),"returncode":completed.returncode}))
            raise SystemExit(completed.returncode)
    status.write_text(json.dumps({"state":"complete","stages":stages,"finished_unix":time.time()}))
    done.touch()
    print("completed",','.join(stages),status)


if __name__=="__main__":main()
