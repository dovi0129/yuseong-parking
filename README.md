# 유성 스마트 주정차 대응 시스템

2026학년도 전국 대학(원)생 빅데이터 분석 경진대회 출품작입니다(팀 밭밭스팟). 유성구 불법주정차 신고 자료(2024)로 동작하며, 화면은 세 가지입니다.

- **현황:** 조회 기간의 반복·상습 지점과 급증 지점(직전 한 달 신고가 5건 이상이고 그 앞 석 달 월평균의 2배 이상인 지점)을 지도와 명단으로 보여 줍니다.
- **순회 계획:** 고른 날짜가 속한 달의 전달 말일까지의 신고로 만든 월별 순회 동선입니다. 근무 구간, 단속 차량 수, 교통안전 우선(학교·어린이보호구역·교통사고 가중치) 적용 여부를 고르면 경로 지도와 방문표(계획 도착 시각, 과거 평균 신고 수)를 보여 줍니다. 같은 달의 평일(또는 주말)은 같은 동선을 씁니다.
- **민원 접수:** 하루의 신고를 접수 순서대로 보여 주며, 신고마다 지점 등급, 금일 순번, 묶음 번호(같은 지점·같은 날 신고), 순회 구분(이달 순회 지점 / 다음 달 순회 검토 / 일반), 시설 개선 검토, 회신 초안을 붙입니다. 사진 검토 탭에서는 신고 사진 두 장의 차량 번호·촬영 간격·촬영 위치를 판정합니다.

## 설치

Python 3.11이 필요합니다.

```bash
python3.11 -m venv .venv            # Windows: py -3.11 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\Activate.ps1

# PyTorch(사진 검토용): GPU가 없거나 macOS이면 cpu, NVIDIA GPU가 있으면 cu128
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## 실행

macOS·Linux에서는 `ui/run.sh`를 실행합니다. Windows에서는 `ui` 폴더에서 창 두 개를 띄웁니다.

```powershell
..\.venv\Scripts\python.exe tile_proxy.py --port 8503
..\.venv\Scripts\streamlit.exe run main.py --server.port 8502
```

브라우저에서 http://localhost:8502 를 엽니다. 지도 타일은 OpenStreetMap에서 받으므로 인터넷 연결이 필요합니다. 사진 검토는 처음 실행할 때 EasyOCR 모델(약 95 MB)을 내려받습니다.

## 월별 계획 다시 만들기

순회 계획 화면은 10~12월의 모든 조합(근무 구간 4 × 평일·주말 × 차량 1~3대 × 교통안전 우선 적용 여부)의 계획을 미리 만들어 둔 파일에서 읽습니다. 자료를 바꾼 뒤에는 다음 명령으로 다시 만듭니다. 계획 하나에 약 30초가 걸리며, 이미 있는 계획은 건너뜁니다.

```bash
python -m src.optimizer.monthly --workers 4
```

달마다 입력 자료는 `data/processed/routes/plan_cache/monthly/<연월>/`에, 계획은 `data/processed/routes/plan_cache/`에 저장합니다.

## 폴더

| 폴더 | 내용 |
|---|---|
| `ui/` | 업무 화면(Streamlit), 지도 타일 중계 서버 |
| `src/pipeline/` | 원자료 정제, 좌표 변환, 교통안전 가중치, 우선순위 점수, 평가 |
| `src/optimizer/` | 순회 경로 최적화(OR-Tools), 월별 계획, 급증 지점, 민원 분류 규칙, CSV 출력 |
| `src/photo/` | 신고 사진 번호판 판독과 요건 판정 |
| `data/raw/` | 공공데이터포털 원자료 |
| `data/processed/` | 정제 자료, 우선순위 점수, 이동시간 행렬, 월별 계획 |
| `data/photo_demo/` | AI로 생성한 신고 장면 사진과 평가용 합성 사진 |
