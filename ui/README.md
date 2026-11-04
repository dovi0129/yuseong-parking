# 유성 스마트 주정차 대응 시스템 화면 (ui/)

## 실행

```bash
ui/run.sh                 # 화면 http://localhost:8502 + 타일 중계 8503, 포트는 PORT=8600 TILE_PORT=8601 ui/run.sh 처럼 바꾼다
```

`run.sh`는 타일 중계 서버(`tile_proxy.py`)와 화면을 함께 띄우고, 끝낼 때 둘 다 내린다. 브라우저는 페이지와 같은 호스트의 8503에서 타일을 받으므로, SSH 포트 전달로 볼 때는 8502와 8503을 함께 전달한다.

`ui/` 안에서 실행해야 `ui/.streamlit/config.toml`의 테마가 적용된다. 파이썬은 `.venv/bin/python`(3.11)이다. 시스템명과 팀명은 `ui/site.toml`의 설정값이며, 이 파일만 고치면 상단 로고·바닥글·브라우저 제목이 함께 바뀐다.

## 시험과 캡처

```bash
cd ui && ../.venv/bin/python -m unittest test_ui -v                # 세 메뉴·탭이 예외 없이 열리는지 확인한다
.venv/bin/python ui/capture.py --url http://127.0.0.1:8502         # 떠 있는 서버를 찍는다(--url 없으면 8765에 새로 띄운다)
.venv/bin/python ui/capture.py --url http://127.0.0.1:8502 --width 1280 --scale 2 --out artifacts/screens/print --only overview  # 인쇄용(2560 px 폭), 타일 검사는 ui_tile_check_print.json. 표 글자를 키울 때는 --width 914 --scale 2.8 (--zoom은 앱 화면 크기, 100~140%이지만 표 폭 결함이 있다)
```

- 캡처: `artifacts/screens/ui_<상태>.png`(1600×1000)와 `ui_<상태>_full.png`(전체 페이지). 상태는 overview, overview_list, overview_focus(명단에서 행을 고른 뒤 지도 강조), plan, plan_list, plan_focus, reception, reception_photo다.
- 타일 검사: 지도 타일 응답마다 HTTP 상태, 바이트 크기, SHA-256, 중계 출처(cache·fetched·gray)를 `artifacts/logs/ui_tile_check.json`에 남기고, `ui/reference_tiles/`의 두 대체 이미지(CARTO 'API KEY REQUIRED' 2,049 B, OSM 'Access blocked' 6,933 B)와 해시가 같은 타일을 센다.
- 보이는 글자: `ui/visible_strings.txt`. 상태마다 화면 글자를 뽑고(지명·숫자·회신 본문 같은 데이터 값은 뺀다), 표·선택지·배지 값은 코드 기준으로 덧붙인다.

## 구성

| 파일 | 내용 |
|---|---|
| `main.py` | 머리(로고, 화면 크기), 가로 메뉴, 경로 표시줄, 제목, 분할 탭, 조회 조건 표, 결과, 바닥글 |
| `backend.py` | `src.optimizer.api` 호출과 캐시. 계산·분류·CSV 인코딩은 API가 맡는다 |
| `maps.py` | folium 지도(중계 서버의 OSM 타일 회색 처리, 전체 화면·확대 단추, 휠 확대, 선택 지점 강조, 범례) |
| `tile_proxy.py` | OSM 표준 타일 중계·캐시. 화면이 실제로 요청한 타일만 받아 `data/processed/tile_cache/`에 둔다. User-Agent는 `YuseongParkingDemo/0.1 (student competition prototype)`, OSM 동시 요청은 2개까지다. 대체 이미지(해시 대조)는 버리고 2초·5초 뒤 최대 2회 다시 받으며, 그래도 막히면 회색 타일을 주고 `artifacts/logs/tile_proxy.log`에 남긴다. 지역 일괄 다운로드는 하지 않는다 |
| `photo.py` | 사진 검토. `src.photo.review_photo_pair`(Worker_2)를 부르고 시연 목록은 `manifest.csv`, 없으면 `dev_manifest.csv`를 쓴다 |
| `brand.py`, `site.toml` | 설정값, SVG 글자 로고(지도 핀 + 주차 표시), 바닥글 문구 |
| `theme.css` | 참고 화면(위택스 목록 화면)에서 잰 값으로 쓴 스타일. 원본 CSS는 복사하지 않았다 |

## 화면

| 메뉴 | 분할 탭 | 조회 조건 | 결과 |
|---|---|---|---|
| 현황 | 지도 / 명단 | 기간, 동, 지점 등급 | 숫자 칸 4개, 반복 지점 지도 또는 지점 명단 |
| 순회 계획 | 지도 / 방문표 | 날짜, 근무 구간, 차량 수, 위험 가중 | 단속 차량별 경로 지도 또는 방문표 |
| 민원 접수 | 접수 목록 / 사진 검토 | 날짜, 기준 시각 / 시연 사진, 사진 파일 | 접수 목록과 상세, 또는 사진 판정 |

- 주소창의 `?menu=overview|plan|reception&tab=map|list|photo`로 메뉴와 탭을 바로 연다.
- 명단·방문표에서 행을 고르면 지도 탭으로 넘어가 그 지점을 확대하고 테두리로 강조한다.
- 지도는 처음 열 때 현황은 유성구 경계(`data/processed/routes/yuseong_boundary.geojson`), 순회 계획은 경로 범위에 맞춘다(확대 단계 0.25 단위). 유성구 밖은 흰색으로 반쯤 덮고 경계를 점선으로 그린다.
- 머리 오른쪽의 '<user_label> | 로그아웃'은 표시용이다. 로그아웃은 세션 상태와 주소창 값을 지워 현황 첫 화면으로 돌아갈 뿐 인증은 없다. 팀명(team_name)이 비어 있으면 저작권 줄은 '© 2026'만 쓴다.
- 사진 검토의 시연 사진 이름은 '사례 번호 · 장면 · 유형'이다(장면은 manifest의 scene, 없으면 sources.csv). 인식한 번호판 상자는 남색 테두리(흰 바깥선)로 그린다.
- 조회 조건은 '조회'를 눌러야 적용된다. 결과는 '총 N건', 표(페이지 나눔), 오른쪽 위 '엑셀 내려받기'(utf-8-sig CSV) 순서다. 빈 결과에는 '조회 결과가 없습니다.'만 쓴다.
- 현황의 지점 등급은 조회 기간 안의 신고 수로 매긴다(반복 10건 이상, 상습 20건 이상, `overview_filtered(grade_basis='period')`). 순회 계획과 민원 접수의 등급은 10월 1일 운영 시작을 재현하는 운영 기준(1~9월 신고 수)을 따른다.
- 민원 접수는 순회 계획에서 마지막으로 조회한 근무 구간·차량 수·위험 가중으로 그날의 순회 경로를 만든 뒤, 평가 기간(2024-10~12) 하루의 신고를 기준 시각까지 분류한다. 재생은 1초마다 기준 시각을 10분씩 옮긴다.
- 사진 검토는 접수 목록에서 고른 건(고르지 않으면 가장 최근 건)에 시연 사진 쌍이나 올린 사진 2장을 붙여 판정하고, 결과를 접수 목록과 CSV의 '사진 검토' 열에 '요건 충족' 또는 '보완 필요: 사유'로 남긴다. 신고 위치는 시연 쌍이면 manifest의 좌표, 올린 사진이면 접수 지점의 좌표다. 기준값은 60초, 15 m, 50 m다.
- 바닥글의 사진 문구는 시연 사진 출처(`sources.csv`의 출처 유형)에 'AI 생성'이 있으면 'AI로 생성한', 없으면 '합성한'으로 쓴다.
- 화면 캐시 키에는 입력 파일 수정 시각, `api.PLANNER_CACHE_VERSION`, `processing.PROCESSING_RULES_VERSION`이 들어간다.
- 회신 내용은 고정 문장 틀로 만든 미발송 초안이다. 글상자에서 고친 내용은 같은 세션의 CSV에 반영된다.

## 조치 구분 규칙 (화면에는 싣지 않는다)

| 열 | 값 | 조건 |
|---|---|---|
| 조치 구분 | 금일 순회 포함 | 오늘 순회 경로에 포함되고 접수 시각이 도착 예정보다 이르다 |
| | 순회 경로 반영 검토 | 반복·상습 지점인데 오늘 경로에 없다 |
| | 일반 처리 | 그 밖의 경우 |
| 시설 개선 검토 | 해당 / 비해당 | 상습 지점이거나 보호구역 지정시설 300m 이내이면 해당 |

보호구역 지정시설 300m 이내는 시설 좌표 반경으로 만든 대리지표이며 법적 구역 경계가 아니다. 기대 신고 수는 학습 기간으로 추정한 구간당 기대 신고 수다.
