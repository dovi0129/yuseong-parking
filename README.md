# 유성 스마트 주정차 대응 시스템

2026학년도 전국 대학(원)생 빅데이터 분석 경진대회 출품작이다(팀 백민우·최해근). 유성구 불법주정차 신고(2024)를 분석해 반복 신고 지점 현황, 이동형 단속 차량 순회 계획, 민원 접수 처리(분류·묶음·회신 초안·신고 사진 자동 검토)를 제공하는 업무 화면이다.

## 1. 설치

Python 3.11이 필요하다. 아래 명령은 저장소 폴더에서 실행한다.

### 가상환경 만들기

Windows(PowerShell):

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

macOS·Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### PyTorch 먼저 설치

사진 검토(EasyOCR)가 PyTorch를 쓴다. PC에 맞는 것 하나만 설치한다.

```bash
# NVIDIA GPU가 없는 PC, macOS
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# NVIDIA GPU가 있는 PC(최신 드라이버)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

macOS에서는 `--index-url` 없이 `pip install torch torchvision`으로 설치한다.

### 나머지 패키지

```bash
pip install -r requirements.txt
```

## 2. 실행

화면(Streamlit)과 지도 타일 중계 서버를 함께 띄운다. 둘 다 `ui` 폴더 안에서 실행해야 화면 테마가 적용된다.

Windows(PowerShell 창 두 개):

```powershell
# 창 1: 지도 타일 중계 서버
cd ui
..\.venv\Scripts\python.exe tile_proxy.py --port 8503

# 창 2: 화면
cd ui
..\.venv\Scripts\streamlit.exe run main.py --server.port 8502
```

macOS·Linux:

```bash
ui/run.sh
```

브라우저에서 http://localhost:8502 를 연다.

- 지도 타일은 처음 볼 때 OpenStreetMap에서 받아 `data/processed/tile_cache/`에 저장하므로 인터넷이 필요하다.
- 사진 검토는 처음 실행할 때 EasyOCR 모델(약 95 MB)을 `data/photo_demo/models/`에 내려받는다. GPU가 없으면 CPU로 판독하며, 사진 한 쌍에 10초 안팎이 걸린다.
- 순회 계획, 도로 이동시간, 좌표는 저장된 계산 결과를 읽으므로 카카오 API 키가 필요 없다.

## 3. 시험

```bash
python -m unittest src.optimizer.test_contract src.optimizer.test_api -v
python -m unittest src.photo.test_review -v
cd ui && python -m unittest test_ui -v
```

## 4. 폴더 구성

| 폴더 | 내용 |
|---|---|
| `ui/` | 업무 화면(Streamlit), 지도 타일 중계 서버 |
| `src/pipeline/` | 원자료 정제, 좌표 변환, 위험 가중치, 우선순위 점수, 평가, 분석 그림 |
| `src/optimizer/` | 순회 경로 최적화, 민원 접수 분류 규칙, CSV 출력, 화면용 함수(`api.py`) |
| `src/photo/` | 신고 사진 번호판 판독과 판정 |
| `data/raw/` | 공공데이터포털 원자료 |
| `data/processed/` | 정제 자료, 점수, 경로 캐시, 평가 결과 |
| `data/photo_demo/` | AI로 생성한 신고 장면 사진과 평가용 합성 사진 |

## 5. 저장소에 넣지 않은 것

- 카카오 API 키(`.env`), 가상환경
- 대회 제출 문서, 분석 보고서, 그림·로그 같은 실행 결과물(`artifacts/`는 실행하면 만들어진다)
- 참가신청서 작성본과 연락처 메모(개인 정보)
- EasyOCR 모델, 지도 타일 캐시(실행할 때 자동으로 받는다)
- 사진 판독 강건성 시험용 변형 사진(`data/photo_demo/robustness/`)
- 카카오 Local API의 좌표 응답 원본(`data/processed/geocode_*`). 화면과 경로 계산은 지점별 좌표(`data/processed/spots.csv`)만 쓴다.

자료를 처음부터 다시 만들려면(`python -m src.pipeline.run --only all`) 좌표 변환을 위해 프로젝트 폴더의 `.env`에 `KAKAO_REST_API_KEY=<키>`를 넣어야 한다.
