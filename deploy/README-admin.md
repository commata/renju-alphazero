# SER5 관리자: 1~4단계

기준 계획: `docs/web-admin-plan.md`. 지금 제공하는 것: 읽기 전용 진단, 인증된 관리자 서버,
대국 목록, 번호 기보와 기보 검증, 재생, 자동 갱신(§5). 통계 차트·백업은 5·6단계이며 아직 제공하지 않는다.
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
숨겨진 탭은 30초로 줄인다. 로그인·로그아웃 외의 변경 API나 다운로드 API는 아직 없다.
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

화면은 `/renju/admin/` 한 페이지에서 주소 뒤 `#`로 나뉜다: `#/`(현황), `#/games`(목록), `#/games/<game_id>`(기보).
데이터 API는 모두 로그인이 필요하고 DB는 `mode=ro`로만 읽는다.

| 경로 | 내용 |
|---|---|
| `GET /data/games` | 목록. `from`/`to`(KST 날짜, 포함), `agent`, `color`(BLACK/WHITE), `status`(in_progress/finished/unfinished/abandoned/suspected), `result`(HUMAN_WIN/AI_WIN/DRAW/UNFINISHED/none), `undo`(yes/no), `limit`(1..200, 기본 50), `cursor` |
| `GET /data/games/options` | 목록 필터의 AI 선택지(DB에 있는 `agent_key`, 엔진 hash) |
| `GET /data/games/<game_id>[?since=<version>]` | 대국 행, 착수(무른 수 포함), VCT2 거부 기록, 기보 검증. `since`가 지금 `version`과 같으면 `{"unchanged": true}` |

잘못된 필터 값·반복된 파라미터는 400, 없는 대국은 404, DB 오류는 503이다.

**목록.** 최신순(`started_at`, `game_id`), cursor 페이지다. 착수 수는 판 위의 수(무른 수 제외)이고, 계산·대기 시간과
30초 초과 수는 기록된 AI 착수 전체(무른 수 포함: 엔진이 실제로 탐색했다)로 센다. `중단 추정`은 진행 중이면서 마지막
착수·무르기(없으면 시작)가 1시간보다 오래된 대국이며, DB의 상태는 바꾸지 않는다.

**기보.** SVG 15×15, 흑돌 흰 번호·백돌 검은 번호, 마지막 수는 번호를 가리지 않는 빨간 테두리, 가로 A–O·세로 15–1
(row 0이 위쪽, 대국 화면의 표기와 같다). N수를 고르면 1..N수만 보인다.

- 이동: 처음·이전·다음·마지막, 슬라이더, 좌우 화살표 키, 자동 재생(2 / 1 / 0.4초)
- 번호: 전체 / 마지막 수만 / 숨김
- 돌을 누르면 수 번호·좌표·색·둔 쪽·계산·대기 시간·기록 시각
- 수순 목록은 기록 순서다. 무른 수는 줄을 긋고 `(n)`으로 보인다. 누르면 그 수가 놓였던 국면에 점선 돌로 보인다.
- AI 계산시간 그래프: 판 위의 AI 수의 `engine_seconds`. 누르면 가장 가까운 AI 수의 국면으로 간다. 값이 20초 이상이면 30초 점선을 그린다.
- VCT2 거부 기록: 누르면 그 국면(`ply` = 거부 당시 판 위의 수)에 거부된 후보(빨간 X)와 실제로 둔 수(파란 점선)를 보인다.
- 진행 중인 대국: `최신 수 따라가기`(기본 켬)이면 새 수가 오면 마지막 수로 간다. 앞선 수를 보는 중이면 위치를 유지한다.
- 5초마다(숨긴 탭은 30초) 다시 묻는다. 기보는 `version`이 바뀔 때만 다시 그린다.

**기보 검증.** 저장된 착수를 `renju.Game`(서버와 같은 규칙, 흑 금수 포함)으로 다시 둔다. AI는 실행하지 않는다.
수 번호 1..N 연속, 기록 순서, 중앙 첫 수(`OPENING_RULE`), 흑부터 번갈아, 범위, 합법 여부, 착수 전 국면 hash(서버의
`board_hash`), 최종 보드(`final_board_json`), 저장된 착수 수, 정상 종료 대국의 결과를 본다. 어긋나면 고치지 않고
기보 위에 항목별로 보인다(예: 한 칸 밀린 수는 이후 수의 `illegal`, 상하 반전은 `final_board`/`board_hash`).

## 로컬 검증

```bash
python -m unittest tests.test_web_admin tests.test_web_admin_games -v
for f in web/admin/*.js; do node --check "$f"; done
```

운영 파일 대신 임시 schema-v2 DB와 임시 계정을 쓰는 HTTP 통합 테스트다. 실서비스의 HTTPS,
Caddy 라우팅, SER5 응답시간, 인증된 TB-V3 모델 smoke를 대체하지 않는다. 로컬 수동 테스트에서만
`ADMIN_ORIGIN=http://127.0.0.1:8010`을 쓰면 Secure 없는 쿠키를 발급한다. 운영 서비스의 HTTPS
origin을 바꾸거나 게임 서버의 `--secure-cookie`를 빼지 않는다.
