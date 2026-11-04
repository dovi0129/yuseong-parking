# 신고 사진 검토 모듈

```python
from src.photo import review_photo_pair
result = review_photo_pair(path1, path2, reported_lat, reported_lon)
```

`plate_1`, `plate_2`는 2~3자리 숫자·자가용 한글 1글자·숫자 4자리 형식으로 인식된 번호이다. `box_1`, `box_2`는 원본 픽셀의 `[x1,y1,x2,y2]`이며, 읽지 못하면 번호와 상자를 `None`으로 반환한다. EXIF 회전이 있으면 상자를 원본 픽셀 좌표로 돌려놓는다. 파일명·정답표·합성 위치 좌표는 판독 입력으로 사용하지 않는다.

`taken_1`, `taken_2`는 EXIF DateTimeOriginal이고, `gps_1`, `gps_2`는 EXIF GPS의 `(lat,lon)`이다. `gap_s`는 두 시각 차이의 절댓값, `move_m`은 사진 간 WGS84 측지거리, `report_m`은 두 사진에서 신고 지점까지 거리의 최댓값이다. `verdict`는 `요건 충족` 또는 `보완 필요`이며, `reasons`는 항목별 사유이다.

기준은 `min_gap_s=60`, `max_move_m=15`, `max_report_m=50`으로 평가 전에 고정하였다. 같은 번호가 판독되고 60초 이상·15m 이내·신고 위치 50m 이내일 때만 요건 충족이다. 신고 좌표를 생략해도 요건 충족으로 처리하지 않는다. 사진을 올린 순서와 촬영 시각의 순서는 달라도 된다.

사유 값은 `번호 판독 불가`, `번호 불일치`, `촬영 시각 정보 없음`, `촬영 간격 미달`, `위치 정보 없음`, `촬영 위치 불일치`, `신고 위치 정보 없음`, `신고 위치 불일치`이다. 두 사진 간 거리와 신고 지점 거리는 서로 다른 사유를 반환하므로 화면에서 기준을 다시 계산하지 않는다.

## 실행과 자료

- 모델은 EasyOCR 한국어 모델이며, OpenCV 밝은 사각형 검출과 CRAFT 검출을 사용한다. 코드 사용권과 출처는 `data/photo_demo/model_provenance.json`, `licenses/`에 있다.
- `CUDA_VISIBLE_DEVICES=0`으로 실행하고 `torch.cuda.get_device_name(0)`가 RTX 5090인지 확인한다. 다른 GPU이면 실행을 거부한다. 각 GPU 프로세스의 CPU 스레드는 1개이다.
- 추가 패키지는 `requirements_Worker_2.txt`에 기록하였다. 설치 전 기존 63개 패키지 버전은 유지하였다.
- 개발 사진은 위키미디어 2장이고, 사용자 AI 사진은 평가 쪽에만 사용한다. 분리 목록은 `data/photo_demo/dev_protocol.json`에 있다.
- `data/photo_demo/ai_generated/`의 사용자 원본은 보존하며, `ai_bases/ai01`~`ai10`은 동일 바이트 복사본이다. `sources.csv`에 `AI 생성(ChatGPT)`, 장면, 낮·밤·비 조건을 기록하였다.
- `manifest.csv`의 `evaluation_include=True` 204행은 합성 번호판 평가 자료이다. 첫 행 `demo_ai01_original_normal`은 번호판을 덮지 않은 ai01로 만든 화면 시연이며 통계에서 제외한다. 시각·GPS는 이 시연에도 합성하였다.
- `ai_original_results.csv`는 번호판을 덮지 않은 AI 원본 10장의 별도 판독 결과이다. 정답은 모두 `123가4567`이고, 낮·밤·비별 수치는 `ai_original_by_condition.csv`에 있다.
- `qualitative_results.csv`의 위키미디어 전시 표지 5장은 번호 없음 음성 검사이다. 별도의 `real_plate_results.csv`는 공개 실차 사진 5장의 원판독 결과이며, 수기로 정답을 확정한 3장·가림 1장·수기 확정 불가 1장을 구분한다.

실행 명령은 프로젝트 루트에서 다음과 같다.

```bash
.venv/bin/python -m unittest src.photo.test_review -v
.venv/bin/python -m src.photo.generate --n 204 --seed 20261030
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m src.photo.evaluate --workers 1
```

최종 평가가 이미 있으면 평가 명령은 중단한다. 자원 오류로 중단된 고정 실행만 `--resume`으로 이어갈 수 있고 기록된 완료 쌍은 재사용한다. 최초 병렬 2프로세스 실행은 공유 GPU 메모리 부족으로 중단되었으며 1프로세스로 재개하였다. 원본·예측·판정 기준은 바꾸지 않았다. 저장된 결과로 그림만 다시 만들려면 `.venv/bin/python -c 'from src.photo.plots import render; render()'`를 사용한다. 고정 후에는 생성 명령도 기존 평가 입력 덮어쓰기를 거부한다.

## 한계

EXIF는 읽힌 시각·GPS의 일관성을 검사하는 입력이며 진위를 인증하는 증거가 아니다. 합성 번호판 204쌍과 AI 원본 10장은 실제 주민 신고 사진과 구별한다. AI 원본 10장의 정답 번호는 모두 같으므로 다양한 실번호판의 판독 성능을 대표하지 않는다. 공개 실차 정성 사진의 수기 정답 3장 중 1장은 완전 일치, 2장은 판독 불가였다. 작은 번호판·어둠·기울기에서 오독과 판독 실패가 발생하며 실차 성능을 합성 평가 수치로 대신하지 않는다.
