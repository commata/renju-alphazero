# SER5 관리자: 1~6단계

기준 계획: `docs/web-admin-plan.md`. 지금 제공하는 것: 읽기 전용 진단, 인증된 관리자 서버,
대국 목록, 번호 기보와 기보 검증, 재생, 자동 갱신(§5), 통계(§6), 전체 백업·다운로드(§7).
운영 SER5 배포(7단계)와 운영 DB 수치 확인은 이 클라우드에서 실행하지 않았다.
게임 서버·엔진·DB 스키마·기존 백업 timer는 변경하지 않는다.

## 1. 운영 데이터 확인 (SER5에서 실행)

```bash
cd /home/deploy/renju-alphazero
.venv-ser5/bin/python scripts/check_track_b_frozen.py
.venv-ser5/bin/python scripts/ser5/admin_inspect.py --db data/renju_web.db
```

두 명령은 실제 SER5에서 실행해야 한다. 진단 JSON에 checkout HEAD, 스키마(2), 상태·결과
분포, `in_progress` 및 중단 추정 수, AI/엔진 해시별 VCT2 거부 행 수, 무르기 대국 결과,
무르기를 포함하는 승률을 기록한다. 운영 DB나 상태를 수정하지 않는다. 존재하지 않는 DB를
만들지 않으며 스키마가 다르면 exit 1이다. TB-V3 행이 없거나 veto가 0이면 그대로 보고한다.
이 보고서는 실행 시점의 DB 상태이고 이후에도 고정된 값이 아니다. 개인정보·세션 해시는 출력하지 않는다.

승률은 `AI_WIN / status='finished'`이다. 무르기와 무승부를 포함한다. 완료 결과가 NULL이거나
예상 밖 값이면 `unknown_results`로 드러내며 분모에서 임의로 빼지 않는다. 과거 DB에 없는
값은 만들지 않는다. 중단 추정은 마지막 착수·무르기(없으면 시작) 이후 1시간 경과 기준이며
재시작, 긴 생각 등과 구분할 수 없는 표시용 추정이다.

## 2. 관리자 계정 설정

SER5에서 기존 `.venv-ser5`를 그대로 사용한다. 추가 패키지는 필요 없다.

```bash
.venv-ser5/bin/python scripts/run_web_admin.py --hash-password
if ! sudo test -e /etc/renju-admin.env; then
    sudo install -m 600 /dev/null /etc/renju-admin.env
fi
sudoedit /etc/renju-admin.env
```

비밀번호를 두 번 입력해 얻은 해시를 아래 필드에 넣는다. 실제 비밀번호는 넣지 않는다.

```text
ADMIN_USER=원하는관리자아이디
ADMIN_PASSWORD_HASH=scrypt$16384$8$1$생성한salt$생성한digest
ADMIN_ORIGIN=https://ser5tack.duckdns.org
```

이 파일은 systemd가 읽으므로 `root:root`, 600으로 유지한다. 셸로 source할 필요가 없다.
`ADMIN_ORIGIN`은 경로·끝 슬래시 없는 정확한 브라우저 origin이다. Caddy의 기존 Basic Auth는
유지하며, 그 아이디·비밀번호와 별도의 관리자 계정이다. 비밀번호 변경 후에는 관리자 서비스만
재시작하면 기존 세션도 모두 폐기된다. 해시나 환경 파일은 커밋·백업에 포함하지 않는다.

## 3. 서비스와 Caddy (운영 적용 시)

1. 기존 `/etc/caddy/Caddyfile`을 백업한다. `deploy/Caddyfile.admin`의 두 handler를 기존
   도메인 블록에 합친다. 기존 `/renju/*` 블록을 대체하고 `/renju/admin/*`를 그 앞에 둔다.
   루트 test-web, 기존 `basic_auth /renju*`, TLS·로그 설정은 보존한다. 예제 전체로 덮어쓰지 않는다.
2. 아래 명령으로 검증·적용한다. 게임 서비스는 재시작하지 않는다.

```bash
sudo cp deploy/renju-admin.service /etc/systemd/system/
sudo systemd-analyze verify /etc/systemd/system/renju-admin.service
sudo systemctl daemon-reload
sudo systemctl enable --now renju-admin
sudo caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile
sudo systemctl reload caddy
sudo systemctl status renju-admin --no-pager
```

주소는 `/renju/admin/`. 8010은 loopback에만 바인딩하며 UFW에 추가하지 않는다. `deploy` 사용자로
게임 서버와 같은 SQLite WAL/-shm에 접근한다. 관리자 연결은 `open_readonly(mode=ro)`만 사용한다.

관리자 화면은 운영 DB 진단과 게임 서버 `/api/health`, `/api/engines` 상태를 표시한다.
게임 서버가 꺼져 있으면 unavailable로 표시하고 관리자 로그인은 유지한다. 5초마다 갱신하며
숨겨진 탭은 30초로 줄인다. 백업 생성은 Origin을 확인한 POST로만 요청하며, 다운로드도 로그인 세션이 필요하다. 운영 DB에는 쓰지 않는다.
공개 경로는 로그인 페이지·로그인용 CSS/JS와 로그인 POST뿐이며 다른 경로는 인증을 요구한다.

## 4. 확인과 되돌리기

- Basic Auth만 통과하고 관리자 로그인하지 않은 상태: `/renju/admin/`는 로그인으로 이동,
  `/renju/admin/data/status`와 임의 다운로드 경로는 401.
- 로그인: `renju_admin` 쿠키에 Path=/renju/admin, HttpOnly, Secure, SameSite=Strict 확인.
- 누락·다른 Origin의 POST는 403. 실패 5회면 IP별 15분 제한. 유휴 30분·최대 12시간 만료.
- 로그아웃·관리자 서비스 재시작 후 이전 쿠키로 접근하면 401.
- 운영 기보가 새로 기록되는 동안 진단 결과가 갱신되는지 확인.
- 기존 `/renju/` 대국과 루트 test-web을 확인. 게임 smoke는 HTTPS 경유로 실행하고 세션 연속성까지 확인한다.
  `scripts/ser5/tbv3_smoke.py` 사용법은 `deploy/README.md` §3.1 참조.

되돌릴 때는 이전 Caddyfile을 복구해 validate/reload하고 `sudo systemctl disable --now renju-admin`.
게임 서비스와 운영 DB, 기존 일일 백업은 그대로 둔다.

## 5. 대국 목록과 기보 (3·4단계)

화면은 `/renju/admin/` 한 페이지에서 주소 뒤 `#`로 나뉜다: `#/`(현황), `#/games`(목록), `#/games/<game_id>`(기보), `#/statistics`(통계), `#/exports`(백업).
데이터 API는 모두 로그인이 필요하고 DB는 `mode=ro`로만 읽는다.

| 경로 | 내용 |
|---|---|
| `GET /data/games` | 목록. `from`/`to`(KST 날짜, 포함), `agent`, `engine`(엔진 전체 hash 또는 none), `color`(BLACK/WHITE), `status`(in_progress/finished/unfinished/abandoned/suspected), `result`(HUMAN_WIN/AI_WIN/DRAW/UNFINISHED/none), `undo`(yes/no), `limit`(1..200, 기본 50), `cursor` |
| `GET /data/games/options` | 목록 필터의 AI 선택지(DB에 있는 `agent_key`, 엔진 hash) |
| `GET /data/games/<game_id>[?since=<version>]` | 대국 행, 착수(무른 수 포함), VCT2 거부 기록, 기보 검증. `since`가 지금 `version`과 같으면 `{"unchanged": true}` |

잘못된 필터 값·반복된 파라미터는 400, 없는 대국은 404, DB 오류는 503이다.

**목록.** 최신순(`started_at`, `game_id`), cursor 페이지다. 착수 수는 판 위의 수(무른 수 제외)이고, 계산·총 응답 시간과
soft limit 초과 기록 수는 기록된 AI 착수 전체(무른 수 포함: 엔진이 실제로 탐색했다)로 센다. `중단 추정`은 진행 중이면서 마지막
착수·무르기(없으면 시작)가 1시간보다 오래된 대국이며, DB의 상태는 바꾸지 않는다.

**기보.** SVG 15×15, 흑돌 흰 번호·백돌 검은 번호, 마지막 수는 번호를 가리지 않는 빨간 테두리, 가로 A–O·세로 15–1
(row 0이 위쪽, 대국 화면의 표기와 같다). N수를 고르면 1..N수만 보인다.

- 이동: 처음·이전·다음·마지막, 슬라이더, 좌우 화살표 키, 자동 재생(2 / 1 / 0.4초)
- 번호: 전체 / 마지막 수만 / 숨김
- 돌을 누르면 수 번호·좌표·색·둔 쪽·계산·큐 대기·총 응답 시간·기록 시각
- 수순 목록은 기록 순서다. 무른 수는 줄을 긋고 `(n)`으로 보인다. 누르면 기록 순서(`seq`)와 각 수의 `ply`로 당시 분기를 재구성해 점선 돌로 보인다. 여러 번 무른 뒤 다른 곳에 둔 대국도 현재 분기와 섞지 않는다.
- AI 계산시간 그래프: 판 위의 AI 수의 `engine_seconds`. 누르면 가장 가까운 AI 수의 국면으로 간다. 값이 20초 이상이면 30초 점선을 그린다.
- VCT2 거부 기록: 누르면 그 국면(`ply` = 거부 당시 판 위의 수)에 거부된 후보(빨간 X)와 실제로 둔 수(파란 점선)를 보인다. 저장된 `payload.history`를 사용하며 없으면 국면 미기록을 표시한다.
- 진행 중인 대국: `최신 수 따라가기`(기본 켬)이면 새 수가 오면 마지막 수로 간다. 앞선 수를 보는 중이면 위치를 유지한다.
- 5초마다(숨긴 탭은 30초) 다시 묻는다. 기보는 `version`이 바뀔 때만 다시 그린다. 무르기 횟수 등 대국 행의 변경도 version에 반영한다. 화면 이동·필터 변경 시 이전 요청을 취소하고 늦은 응답을 무시한다.

**기보 검증.** 저장된 착수를 `renju.Game`(서버와 같은 규칙, 흑 금수 포함)으로 다시 둔다. AI는 실행하지 않는다.
수 번호 1..N 연속, 기록 순서, 중앙 첫 수(`OPENING_RULE`), 흑부터 번갈아, 범위, 합법 여부, 착수 전 국면 hash(서버의
`board_hash`), 최종 보드(`final_board_json`), 저장된 착수 수, 정상 종료 대국의 결과를 본다. 어긋나면 고치지 않고
기보 위에 항목별로 보인다(예: 한 칸 밀린 수는 이후 수의 `illegal`, 상하 반전은 `final_board`/`board_hash`).

## 6. 통계 (5단계)

`GET /data/statistics`는 목록과 같은 필터(페이지·cursor 제외)를 적용한다. 기본값은 무르기 포함이다.
한 읽기 트랜잭션에서 대국·AI 시간을 읽고 연결을 닫은 후 그룹과 백분위를 계산한다.

- AI 승률 = AI 승 / 정상 종료 수. 무승부와 결과 미기록/오류도 분모에 포함하며 후자는 별도 경고한다.
- 전체 상태, 중단 추정(진행 중의 부분집합), 무르기가 있었던 대국 수와 승·무·패
- `agent_key` + **전체** `engine_config_hash`별 표와 승률 막대, 사람 색별 결과, KST 일별 대국 수
- 시간은 무른 수까지 포함한 AI 착수 전체가 대상이다. 사람·중앙 자동 착수는 제외한다.
- `engine_seconds`(계산), `queue_seconds`(큐 대기), `wall_seconds`(총 응답) 각각 평균·중앙값·p95·최대
- NULL은 미기록, 음수·비수치·비유한 값은 잘못된 값으로 센다. 모두 시간 집계에서 제외한다.
- p95는 정렬 후 `(n-1)*0.95` 위치의 선형 보간이다. 표본이 없으면 `미기록`이다.
- 30초 초과 건수는 `rt_json.soft_limit_exceeded=true`로 **기록된** AI 수다. 없는 플래그를 추정하지 않는다.

## 7. 전체 백업 (6단계)

| 경로 | 내용 |
|---|---|
| `POST /data/exports` | JSON `{"include_raw_logs": false}`(기본)로 전체 백업 요청. 접수 202, 생성 중이면 409. 인증·같은 Origin 필수 |
| `GET /data/exports` | 생성 중/완료/실패, 생성 시각·건수·크기·SHA-256, 보관 개수 |
| `GET /data/exports/<id>/download` | 완료된 압축만 첨부 파일로 전송. UUID 이외 경로 거부, 미완료·없는 파일 404 |

작업은 별도 Python 프로세스에서 `nice +10`으로 실행한다. filesystem flock을 worker 종료까지 유지해
별도 관리자 인스턴스나 재시작과 겹쳐도 동시 생성은 하나다. 파일을 전부 만든 후 원자적으로 공개한다.
동작에는 Linux `fcntl`을 사용한다(SER5 대상). 관리자 서버가 죽고 worker도 중단된 작업은 다음 조회에서
실패로 표시한다. 살아 있는 worker의 임시 파일은 정리하지 않는다.

압축 구성은 `renju_web_full.db`, `renju_web_full.sql`, CSV 3개와 `manifest.json`이다.
DB는 기존 일일 백업과 공유하는 SQLite backup API + integrity_check로 만든다. SQL/CSV는 그 사본만 읽는다.
SQL dump는 `user_version=2`도 보존하고 CSV는 utf-8-sig다. manifest에는 건수·스키마·서버 커밋과 구성 파일별
해시·크기가 있다(manifest 자신의 해시는 제외). 내려받는 압축 자체의 해시도 목록 API에 제공한다.

`include_raw_logs=true`는 스냅샷의 정상 종료 대국에 연결된 **game.json·moves.csv만** 추가한다.
DB가 가리킨 경로가 설정된 로그 디렉터리 밖이거나 심볼릭 링크이면 포함하지 않고 누락 수를 기록한다.
이 원본 파일들은 보조 사본이며 DB/SQL/CSV와 원자적 스냅샷이 아니다. 계정 환경 파일·세션 메모리·임의 경로는
내보내지 않는다. 백업 worker에는 `ADMIN_*` 환경 변수를 전달하지 않는다.

기본 저장 경로는 `data/admin_exports/`(공개 웹 경로 밖, git 무시)다. disk 여유를 검사하고 부족하면 실패한다.
완료본은 최근 5개를 보관하고 24시간이 지난 임시 파일은 다음 상태 조회·생성 시 정리한다. 바꾸려면 관리자
ExecStart에 `--export-dir <path> --export-keep <N>`을 지정한다. 기존 `renju-db-backup.timer`와 보관 14개는 유지한다.
백업은 목록/통계 필터에 관계없이 **전체 DB**를 포함한다. DB 복원·삭제 기능은 제공하지 않는다.

## 로컬 검증

```bash
python -m unittest tests.test_web_admin tests.test_web_admin_games tests.test_web_admin_statistics tests.test_web_admin_exports -v
for f in web/admin/*.js; do node --check "$f"; done
```

운영 파일 대신 임시 schema-v2 DB와 임시 계정을 쓰는 HTTP 통합 테스트다. 실서비스의 HTTPS,
Caddy 라우팅, SER5 응답시간, 인증된 TB-V3 모델 smoke를 대체하지 않는다. 로컬 수동 테스트에서만
`ADMIN_ORIGIN=http://127.0.0.1:8010`을 쓰면 Secure 없는 쿠키를 발급한다. 운영 서비스의 HTTPS
origin을 바꾸거나 게임 서버의 `--secure-cookie`를 빼지 않는다.

선택 브라우저 회귀(운영 의존성 아님):

```bash
python -m pip install playwright
python tests/admin_browser_smoke.py --chromium /usr/bin/chromium
```

별도 브라우저 설치 대신 기존 Chromium 경로를 지정한다. 임시 계정과 실제 PlaySession/LogStore 기보로
무른 분기·VCT2 당시 국면, 늦은 응답, 재생, 진행 중 따라가기/위치 유지, 통계 필터, 백업 다운로드와
모바일 가로 넘침을 검사한다. 캡처는 git에서 제외된 `logs/admin-ui/`에 둔다. 제품 CSP는 완화하지 않는다.
