# 유성 스마트 주정차 대응 시스템

2026학년도 전국 대학(원)생 빅데이터 분석 경진대회 출품작입니다(팀 밭밭스팟). 유성구 불법주정차 신고 자료(2024)로 동작하며, 화면은 세 가지입니다.

- **현황:** 반복·상습 신고 지점 지도와 명단
- **순회 계획:** 근무 구간과 차량 수에 맞춘 이동형 단속 차량의 순회 경로와 방문표
- **민원 접수:** 신고 분류, 같은 지점·같은 날 신고 묶음, 회신 초안, 신고 사진 검토

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

## 폴더

| 폴더 | 내용 |
|---|---|
| `ui/` | 업무 화면(Streamlit), 지도 타일 중계 서버 |
| `src/pipeline/` | 원자료 정제, 좌표 변환, 위험 가중치, 우선순위 점수, 평가 |
| `src/optimizer/` | 순회 경로 최적화(OR-Tools), 민원 분류 규칙, CSV 출력 |
| `src/photo/` | 신고 사진 번호판 판독과 요건 판정 |
| `data/raw/` | 공공데이터포털 원자료 |
| `data/processed/` | 정제 자료, 우선순위 점수, 경로 계산 결과 |
| `data/photo_demo/` | AI로 생성한 신고 장면 사진과 평가용 합성 사진 |
