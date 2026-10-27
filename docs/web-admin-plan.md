# 웹 관리자 페이지 계획 (SER5, `/renju/admin/`)

배포 라인(`feat/Web_SER5-tbv3-fast5`)의 렌주 웹에 관리자 페이지를 더하는 계획이다. 1단계 읽기 전용 진단 도구, 2단계 관리자 서버·로그인·배포 템플릿, 3단계 대국 목록·번호 기보·기보 검증,
4단계 재생·자동 갱신을 구현했다. 실제 SER5 운영 데이터 확인과 Caddy/systemd 적용·게임 smoke는 아직 실행하지 않았다.
5~7단계(통계·백업·SER5 적용)는 후속 작업이다.
현재 실행·검증 절차와 한계는 [관리자 배포 안내](../deploy/README-admin.md)를 따른다.
SQLite(`data/renju_web.db`, `src/weblog/store.py`, 스키마 v2)는 그대로 쓴다.

## 1. 원안에서 바뀐 점과 근거

| # | 원안 | 실제 | 계획 |
|---|---|---|---|
| 1 | 경로 `/admin` | 도메인 루트는 `test-web`이고 렌주는 `/renju/` 아래다(`docs/web-ser5.md` §9.14). | `/renju/admin/` |
| 2 | 관리자 API 경로 미정 | 게임 서버는 경로에 `/api/`가 있으면 앞을 떼고 처리한다(`scripts/run_web_play.py` `_route`). Caddy 설정이 어긋나면 관리자 요청이 게임 API로 간다. | 관리자 API는 `/renju/admin/data/...`(`/api/` 없음). Caddy에서 `handle_path /renju/admin/*`를 `/renju/*`보다 앞에 둔다. |
| 3 | 로그인 하나 | Caddy `basic_auth /renju*`가 이미 있다. 플레이어와 같이 쓰는 베타 비밀번호다. | 두 겹: Basic Auth(그대로) → 관리자 로그인. |
| 4 | 진행 중 기보를 위해 착수별 저장 보완 | DB는 이미 착수마다 기록한다(writer thread, 큐). 대국 종료 때만 쓰는 것은 `logs/web_play/` 원본 로그다. | 항목 삭제. |
| 5 | 진행 중 / 정상 종료 / 중단 | `status`: `in_progress` / `finished` / `unfinished` / `abandoned`, `result`: `HUMAN_WIN` / `AI_WIN` / `DRAW` / `UNFINISHED`. 서버가 죽으면 `in_progress`로 남는다. | 실제 값 표시. 마지막 착수가 세션 TTL(1시간)보다 오래된 `in_progress`는 화면에서 "중단 추정"으로 표시한다(DB는 고치지 않는다). |
| 6 | 무르기 언급 없음 | 무른 수는 `undone_at`을 표시하고 남긴다. `seq`는 무른 수를 포함한 기록 순서, `ply`는 판 위에 남은 수의 순서다. | 재생은 `undone_at IS NULL ORDER BY ply`. 무르기 지점을 수순 목록에 표시한다. **무르기 대국도 승률에 포함한다**(무르기 횟수를 목록·통계에 표시하고, 제외 필터를 둔다). |
| 7 | 금수·거부 기록 | `veto_positions`는 AI 후보 수가 VCT2로 패배가 증명되어 바꾼 기록이다. 사람의 금수 시도는 어디에도 기록되지 않는다(브라우저가 금수 자리를 막고, 서버는 400만 돌려준다). | 이름을 "AI VCT2 거부 기록"으로 바꾼다. 사람 금수 시도 기록은 첫 범위에서 뺀다. TB-V3 대국에서 이 기록이 생기는지 1단계에서 확인한다. |
| 8 | AI 버전 `rt30` 등 | 기본 상대는 `tbv3`, `rt30`은 이전 모델(legacy)이다. `agent_key`만으로는 배포 간 엔진 변화를 구분하지 못한다. | `agent_key`와 `engine_config_hash` 앞 12자리로 묶는다. 상세 화면에 `engine_commit`, runtime 해시를 보인다. |
| 9 | AI 계산시간 하나 | `engine_seconds`(탐색), `queue_seconds`(대기열), `wall_seconds`(사용자가 기다린 시간). 30초 soft limit 초과는 `rt_json`의 `soft_limit_exceeded`. | 계산시간과 대기시간을 나눠 보이고, 30초 초과 건수를 보인다. `actor`가 `HUMAN`, `OPENING_RULE`인 수는 뺀다. |
| 10 | 관리자 조회가 게임 잠금을 기다리지 않게 | 별도 프로세스라 게임 서버의 잠금과 무관하다. WAL이라 읽기가 쓰기를 막지 않는다. 긴 읽기 트랜잭션은 checkpoint를 밀어 `-wal` 파일을 키운다. | 읽기는 `open_readonly`로 짧게 한다. 관리자 프로세스는 게임 서버와 같은 `deploy` 사용자로 실행한다(`-shm` 권한). |
| 11 | 실시간 상태 | AI 생각 중, 대기열, 세션 수, DB 쓰기 실패는 게임 서버 메모리에만 있다. | 관리자 서버가 `127.0.0.1:8000/api/health`, `/api/engines`를 호출해 "서버 상태" 패널을 만든다. |
| 12 | 백업 새로 구현 | `deploy/backup_db.py`가 매일 04:17에 백업 API로 백업하고 integrity_check 후 14개를 보관한다. | 그 로직을 재사용하고 `manifest.json`을 더한다. `logs/web_play/` 원본 로그 포함은 선택이다. |

## 2. 구성

```text
Caddy (HTTPS, basic_auth /renju*)
 ├─ /renju/admin/*  → 127.0.0.1:8010  renju-admin.service (표준 라이브러리만, 엔진·torch import 없음)
 │     └─ SQLite 읽기 전용(mode=ro) + 게임 서버 /api/health 내부 호출
 └─ /renju/*        → 127.0.0.1:8000  renju-web.service (변경 없음)
```

- 관리자 프로세스는 DB에 쓰지 않는다. 게임 서버, 동결 엔진 파일, `run_web_play.py`의 엔진 경로는 건드리지 않는다.
- 작업은 배포 브랜치에서만 한다. 시작 전에 `scripts/check_track_b_frozen.py`, 끝난 뒤 `scripts/ser5/tbv3_smoke.py`(세션 연속성 포함)가 통과해야 한다.
- 8010은 UFW에서 열지 않는다.

## 3. 로그인과 접근 보호

- 관리자 계정 1개. 비밀번호 해시(`hashlib.scrypt`)는 `/etc/renju-admin.env`(권한 600)에 둔다. 저장소와 DB에는 넣지 않는다.
- 세션은 메모리에만 둔다(재시작하면 다시 로그인).
  - 쿠키 `renju_admin`, `Path=/renju/admin`, `HttpOnly; Secure; SameSite=Strict`
  - 유휴 30분, 최대 12시간
- 로그인 실패 제한: IP별 5회 실패 시 15분 잠금. IP는 Caddy(127.0.0.1)가 보낸 `X-Forwarded-For`만 믿는다.
- POST(로그인, 로그아웃, 백업 생성)는 `Origin`이 같은 사이트일 때만 받는다. 모든 응답에 `Cache-Control: no-store`.
- 관리자 화면, 조회 API, 백업 다운로드 모두 인증을 거친다.
- 계정과 세션은 DB 밖에 있으므로 백업에 들어가지 않는다.

## 4. 대국 목록

| 항목 | 내용 |
|---|---|
| 시작·종료 시각 | KST |
| AI | `agent_key` + 엔진 해시 앞 12자리 |
| 사람 색 | 흑 / 백 |
| 상태 | `status` 값, 오래된 `in_progress`는 "중단 추정" |
| 결과 | 사람 승 / AI 승 / 무승부 / 미완료 |
| 착수 수 | 무르지 않은 수의 개수(진행 중 대국도; `number_of_moves`는 종료 때만 채워진다) |
| 무르기 | `undo_count` |
| AI 시간 | 계산시간 평균·최대, 30초 초과 건수 |

- 필터: 기간(KST), AI, 색, 상태, 결과, 무르기 여부.
- 최신순, `started_at` 기준 커서 페이지(인덱스 `games_started` 있음).
- 행을 누르면 상세 기보 화면으로 간다.

## 5. 번호 기보와 재생

15×15 바둑판과 착수 번호가 적힌 돌을 화면 중심에 둔다. SVG로 그린다.

| 기능 | 동작 |
|---|---|
| 착수 번호 | 흑돌에 흰 숫자, 백돌에 검은 숫자. 225수까지 읽히는 크기 |
| 번호 표시 | 전체 / 마지막 수만 / 숨김 |
| 현재 수 | `현재 수 / 전체 수` (예: `29 / 56`) |
| 이동 | 처음·이전·다음·마지막, 수순 슬라이더 |
| 자동 재생 | 재생·일시정지, 속도 선택 |
| 마지막 착수 강조 | 번호를 가리지 않는 테두리 |
| 좌표 | 가로 A–O, 세로 1–15 |
| 수순 목록 연동 | 목록의 수를 누르면 그 국면으로 이동 |

- N수를 고르면 1수부터 N수까지만 표시한다. 마지막 수에서 전체 기보가 보인다.
- PC는 바둑판 옆, 모바일은 바둑판 아래에 착수 목록과 시간을 둔다.
- 돌을 누르면 좌표, 둔 쪽(사람 / 중앙 첫 수 / AI), 계산시간·대기시간을 보인다.
- 시간 그래프의 점을 누르면 그 국면으로 간다.
- 무른 수는 수순 목록에 무르기 지점으로 표시하고, 선택하면 회색 점선 돌로 보인다.
- AI VCT2 거부 기록을 누르면 그 시점 보드에 원래 후보 수(`tree_move`)와 실제로 둔 수(`played`)를 함께 보인다.
- 진행 중 대국은 "최신 수 따라가기". 과거 수를 보는 중에는 새 착수가 와도 위치를 유지한다.

**좌표와 검증.**

- DB는 0 기준 `row0` / `col0`, row 0이 위쪽이다. 표기는 `web/index.html`의 `notation()`과 같은 `열 문자 + (15 - row)` 하나로 쓴다.
  게임 메시지의 `(row+1, col+1)` 내부 좌표와 섞지 않는다.
- 기보는 저장된 착수를 재현한다. AI를 다시 실행하지 않는다.
- 서버에서 수순을 `renju.Game`으로 다시 둔다. 수마다 합법인지, `board_hash`가 맞는지, 마지막에 `final_board_json`과 같은지 본다.
  첫 수 중앙, 누락·중복·범위 밖 좌표도 같이 본다. 어긋나면 고치지 않고 그 대국에 경고를 표시한다.

## 6. 실시간 조회

- 기본 5초 간격, 탭이 숨겨지면 30초.
- 새 대국, 추가 착수, 종료 후 결과·통계를 반영한다.
- 상세 화면은 `?since=<version>`으로 묻는다. 기록이 그대로면 `unchanged`만 오고, 바뀌었으면 대국 행·착수·거부 기록 전체를
  한 읽기 트랜잭션에서 읽어 온다. `version`은 마지막 `seq`, 착수 행 수, 무른 수, 거부 기록 수, 상태·결과·종료 시각,
  중단 추정 여부로 만든다. (계획의 `?after_seq=`는 무르기가 앞선 수의 `undone_at`을 바꾸는 것을 놓치므로 바꿨다.
  착수는 무른 수를 합쳐도 수백 행이라 전체를 다시 받아도 작다.)
- 마지막 갱신 시각, 연결 상태, 게임 서버 상태(세션, AI 대기열, DB `failures` / `last_error`)를 보인다.

## 7. 통계

| 통계 | 표시 |
|---|---|
| 대국 현황 | `finished` / `in_progress`(중단 추정 별도) / `unfinished` / `abandoned` 건수 |
| AI 승률 | 승·무·패 건수와 승률 |
| AI 버전별 | 엔진 해시 단위 비교 표와 막대그래프 |
| 흑·백별 | 사람 색 기준 |
| 날짜별 대국 수 | KST 일별 추이 |
| AI 시간 | 계산시간·대기시간 각각 평균·중앙값·p95·최댓값, 30초 초과 건수 |
| 대국별 시간 | 착수 번호별 그래프 |

- AI 승률 = AI 승 ÷ `status='finished'`인 대국. 무승부는 분모에 넣는다. `unfinished`, `abandoned`, `in_progress`는 뺀다.
- **무르기 대국은 승률에 포함한다.** 무르기가 있었던 대국 수와 그 대국들의 결과를 따로 보이고, 필터로 제외해 볼 수 있다.
- 날짜 집계는 `date(started_at, '+9 hours')`.
- 시간 통계는 `actor`가 `HUMAN`, `OPENING_RULE`이 아닌 수만 쓴다. 값이 없으면 `미기록`. 백분위는 값을 가져와 Python에서 계산한다.

## 8. SQLite 운영

- 지금 PRAGMA: `journal_mode=WAL`, `synchronous=NORMAL`, `foreign_keys=ON`, `busy_timeout=5000`(`store.configure`). 바꾸지 않는다.
- 관리자는 `store.open_readonly`(`mode=ro`)만 쓴다. 쓰기는 게임 서버의 writer thread 하나뿐이다.
- 읽기 트랜잭션은 짧게, 목록은 필요한 범위만 읽는다.
- 인덱스: 있는 것(`games_started`, `games_agent`, `moves_live_ply`, `veto_game`)으로 시작하고, 실제 쿼리 계획을 본 뒤에 더한다.
  인덱스 추가는 스키마 변경(v2 → v3 migration)이라 게임 서버 쪽 작업이다.

## 9. 백업 생성과 다운로드

```text
renju_web_full.db
renju_web_full.sql
games.csv
moves.csv
veto_positions.csv
manifest.json        (생성 시각, 스키마 버전, 테이블별 건수, 파일별 SHA-256·크기, 서버 커밋)
[선택] logs/web_play/
```

```text
운영 SQLite → 백업 API로 스냅샷 .db → 스냅샷에서 .sql(iterdump)·CSV 추출
→ integrity_check·건수 확인 → manifest.json → .tar.gz → 인증된 다운로드
```

- 모든 추출물은 같은 스냅샷에서 만든다. 운영 DB 파일을 복사하지 않는다.
- CSV는 `utf-8-sig`(기존 `moves.csv`와 같다), JSON 컬럼도 그대로 넣는다.
- 위치는 `data/admin_exports/`(git 무시, 웹 경로 밖). 한 번에 하나만, `nice`로 낮은 우선순위, 시작 전 디스크 여유 확인.
- 상태: 생성 중 / 완료 / 실패. 최근 N개만 남기고 24시간 넘은 임시 파일은 지운다.
- 화면에 생성 시각, 테이블별 건수, 파일 크기를 보인다.
- 매일 도는 `renju-db-backup.timer`는 그대로 둔다.

## 10. 구현 순서와 완료 기준

| 단계 | 작업 | 완료 기준 |
|---|---|---|
| 1 | SER5 확인: `git rev-parse HEAD`, `PRAGMA user_version`(=2), 상태·결과 분포, 남은 `in_progress` 수, TB-V3 대국의 veto 행 | 실제 값으로 표시 규칙 확정 |
| 2 | 관리자 서버 골격, 로그인, Caddy 블록, `renju-admin.service` | 비로그인 시 모든 경로 차단, 게임 smoke 통과 |
| 3 | 목록, 번호 기보 | 재현 검증(합법, `board_hash`, 최종 보드) 통과, 225수 표시, 무르기 대국 정상 |
| 4 | 재생, 실시간 | 앞뒤 이동·자동 재생, 진행 중 대국 따라가기, 과거 수 보기 중 위치 유지 |
| 5 | 통계 | 화면 값과 직접 SQL 집계가 일치 |
| 6 | 백업 | 압축 해제 → integrity ok → `.sql` 복원 후 건수 일치 → CSV 건수 일치 |
| 7 | SER5 적용 | 대국 중 조회·백업에도 AI 응답시간 변화 없음, 재시작 후 동작 |

첫 범위: 로그인, 실시간 대국 목록, 번호 기보와 재생, 승률·착수시간 통계, AI VCT2 거부 기록 조회, 전체 백업 다운로드.
범위 밖: 사람 금수 시도 기록, AI 오류(worker hang 등) 기록. 둘 다 DB에 없어 스키마 v3이 필요하다.
