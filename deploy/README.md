# SER5 웹 운영 배포 (TB-V3 Resolve / fast5, CPU value)

미니PC(SER5) 한 대에서 렌주 웹대국을 `https://ser5tack.duckdns.org/`로 운영하는 구성이다.
웹 구조는 `docs/web-ser5.md` §8(멀티 세션)을 따른다.

기본 상대는 **latest-certified** 엔진이다(`src/rt30/tb_v3_slot.py`).
- 뜻: 가장 최근에 만든 checkpoint가 아니라, 가장 최근에 승급 gate와 배포 gate를 모두 통과한 엔진·실행판·모델이다.
- 지금 기본 상대: TB-V3 Resolve / fast5 / 동결 H3 policy / 동결 H6 value / CPU value
- 근거
  - 승급: TB-V3 promotion replay, fast5 G0 · G1 · G2 desktop
  - 배포: fast5 G2 SER5 `SER5_CERTIFIED`(worker 1), fast5 CPU 인증 PASS(seed 8451 / 8452)
- 학습 중인 H7-c checkpoint는 자동으로 바뀌지 않는다. 새 승급·배포 단계가 `LATEST_CERTIFIED`를 바꿀 때만 바뀐다.
- **이전 웹 모델**은 TB-RT30-v4 Legacy다. v4 = 예전 TB-RT30-v3 + TB-V3 native builder.
  - same-move gate(§3.2)를 통과한 뒤에만 선택지에 나온다. 그 전에는 서버가 "not served"를 로그로 남기고 TB-V3만 서비스한다.
  - 기본값이 되지는 않는다.
  - worker는 처음 선택될 때 뜬다. 세션 TTL(1시간) 동안 쓰이지 않으면 닫힌다.

## 1. 환경

| 항목 | 값 |
|---|---|
| 서버 | `ser5-server`, Ubuntu Server, LAN `192.168.10.200` (`enp1s0`) |
| 도메인 | `ser5tack.duckdns.org` → 공인 IP (DuckDNS가 변동 IP를 갱신) |
| 공유기 | 80/tcp, 443/tcp → `192.168.10.200`; 외부 SSH 22222 → 22 |
| HTTPS | **Caddy 자동 HTTPS**(인증서 발급·갱신 모두 Caddy). Certbot은 쓰지 않는다. 설정은 `/etc/caddy/Caddyfile` |
| 앱 | `renju-web.service` → `127.0.0.1:8000` (외부에서 직접 열리지 않는다) |
| 데이터 | `data/renju_web.db`(SQLite, WAL), 백업 `data/backups/`, 대국 원본 로그 `logs/web_play/` |
| 방화벽 | UFW: 80/443 허용(`Nginx Full`), 22. **8000은 열지 않는다** |

```text
Internet ─ https://ser5tack.duckdns.org ─ 공유기 80/443 ─ Caddy :443 (자동 TLS, Basic Auth)
                                                              └─ http://127.0.0.1:8000  renju-web.service
                                                                   ├─ slot 0: TB-V3 engine worker (CPU value, + 예비 worker)
                                                                   └─ SQLite data/renju_web.db
```

- `caddy.service`는 지금처럼 HTTPS와 인증서를 맡는다. `renju-web.service`는 Python 서버만 맡는다. 둘은 따로 켜고 끈다.
- Caddy가 받는 인증서의 실제 발급 CA는 Caddy가 고른다(기본은 Let's Encrypt, 실패하면 ZeroSSL). 여기서는 따로 정하지 않는다.
- 앱은 Caddy가 보내는 `X-Forwarded-Proto: https`를 보고 세션 쿠키에 `Secure`를 붙인다. 서비스는 `--secure-cookie`로도 강제한다.

## 2. 설치 전 점검 (읽기만 한다)

```bash
cd ~/renju-alphazero && git pull
sudo -v                      # 포트를 잡은 프로세스 이름까지 보려면
deploy/check.sh
```

아래를 확인한다. `[WARN]`이 있으면 해당 절을 먼저 본다.

- DuckDNS가 공인 IP를 가리키는가.
- 80/443을 누가 잡고 있는가. Caddy여야 한다.
  - Nginx가 잡고 있으면 §4.3을 본다.
  - Docker 컨테이너가 잡고 있으면 §4.2를 본다.
- 8000을 누가 잡고 있는가. 수동으로 띄운 예전 서버라면 §3에서 끈다.

## 3. 웹 서버 서비스

```bash
cd ~/renju-alphazero
# 수동으로 띄운 서버가 있으면 끈다 (systemd가 같은 포트를 쓴다)
pkill -f "scripts/run_web_pla[y]"; sleep 2

git rev-parse --is-shallow-repository   # false 여야 한다 (근거 검사가 G2 측정 커밋을 git에서 읽는다)
sudo cp deploy/renju-web.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now renju-web
journalctl -u renju-web -f      # "deployment check PASS", "worker ready", "Renju local web" 확인 후 Ctrl-C
curl -s localhost:8000/api/health; echo     # "default_agent": "tbv3", "engine": {"engine": "TB-V3", ...}
curl -s localhost:8000/api/engine; echo     # 엔진 · runtime 해시, 모델 SHA-256 4개, 근거 파일
curl -s localhost:8000/api/engines; echo    # 제공 중인 상대: tbv3 (latest-certified), rt30 (legacy-certified), worker 상태
```

- 시작 전에 `rt30.native.build --if-stale`와 `rt30.vcfcore.build --if-stale`가 돈다. C 소스나 Python이 바뀌었을 때만 다시 빌드한다.
- 서버는 엔진 worker를 띄우기 전에 배포 검사를 한다(fail-closed). 하나라도 실패하면 시작하지 않는다.
  - canonical = TB-V3 / fast5
  - fast5 근거(G0 두 파일, G1, G2 desktop, G2 SER5, CPU 인증)가 이 checkout에서 성립(runtime 해시, 안전 위반 0)
  - TB-V3 identity, 동결 policy / value 파일 SHA, 두 native 빌드가 최신
- TB-V3는 deadline이 없다(노드 예산, SER5 최대 6.9초). 30초는 soft monitor다: 넘으면 로그(`[tbv3] soft limit`)와 수 기록(`rt.soft_limit_exceeded`)만 남기고 그 수를 그대로 둔다.
- 대체 수(fallback)는 없다. worker가 300초 동안 답하지 않거나 죽으면 그 수는 오류로 끝나고(브라우저에 오류 표시), worker는 교체된다.
- 시간 기준 세 가지
  - 30초: 엔진 성능 soft monitor. 기록만 한다.
  - 120초: smoke 스크립트의 응답 한계. smoke 성능 timeout이지 엔진 장애가 아니다. worker는 아직 탐색 중일 수 있다.
  - 300초: 서버의 worker hang 기준. 이 수는 오류가 되고 worker를 교체한다.

### 3.1 smoke test

운영 서비스는 `--secure-cookie`다. 그래서 `Secure` 세션 쿠키는 `http://`로는 돌아오지 않는다(브라우저도, smoke도 같다).
- `http://127.0.0.1:8000`(운영 서비스)에 smoke를 돌리면 요청마다 새 세션이 된다.
- smoke는 이것을 세션 검사에서 바로 거부한다. 운영 서비스의 `--secure-cookie`는 빼지 않는다.

```bash
# A. 배포 전: --secure-cookie 없는 임시 로컬 서버 (서비스와 다른 포트)
PYTHONPATH=src .venv-ser5/bin/python scripts/run_web_play.py --port 8001 --tbv3 --ai-slots 1 &
sleep 60; .venv-ser5/bin/python scripts/ser5/tbv3_smoke.py --base http://127.0.0.1:8001; kill %1
# B. 운영: Caddy HTTPS 경유 (Basic Auth)
.venv-ser5/bin/python scripts/ser5/tbv3_smoke.py --base https://ser5tack.duckdns.org --user '<id>' --password '<pw>'
```

- smoke가 확인하는 것
  - health · engine 정보
  - 세션 연속성: `/api/new`가 준 쿠키가 다음 요청에 돌아와 같은 판(seed)이 보이는지
  - AI 응답 6수(모두 엔진 수, 시간)
- 실패하면 마지막 판 상태(status, move_count, last_error)와 서버 AI 큐(running, queued, done, failed), sessions, secure_cookie를 함께 출력한다.

### 3.2 TB-RT30-v4 same-move gate (이전 웹 모델을 다시 열기 전에 한 번)

v4는 v3와 설정 · 코드가 같고 `src/rt30/native/build.py`만 다르다(`_forbidden` 추가).
- 판정: deadline 없이(결정에 시간이 들어가지 않게) 같은 국면을 다시 두어 수와 탐색 기록이 v3와 같은지 본다.
  - 수, route, 노드 수, 검사 상태, tree 수
- 대상
  - v3의 커밋된 동등성 기록(H3 8국면, frozen / fast3 / rt3)
  - SER5 대표 국면 묶음 두 개(`slow-stratified-v1` 60, `critical-regressions-v1` 6): v3 checkout과 이 checkout에서

```bash
cd ~/renju-alphazero && export PYTHONPATH=src CK=runs/h3_policy_64x4/best.pt
mkdir -p runs/v4_gate
git worktree add ../renju-v3 3b8746e          # 마지막 TB-RT30-v3 웹 커밋 (feat/Web_SER5)
(cd ../renju-v3 && PYTHONPATH=src ../renju-alphazero/.venv-ser5/bin/python -m rt30.native.build)
KEYS="puct_policy_vct2_50k_8415/2/14 puct_policy_vct2_50k_8415/9/18 puct_policy_vct2_50k_8415/9/26
puct_policy_vct2_50k_8415/10/18 puct_policy_vct2_50k_8415/10/20 puct_policy_vct2_50k_8415/11/9
puct_policy_vct2_50k_8415/11/11 puct_policy_vct2_50k_8415/11/13"
.venv-ser5/bin/python scripts/run_rt30_replay.py --checkpoint $CK --arms frozen,fast3,rt3 --keys $KEYS \
  --workers 4 --output runs/v4_gate/v4_equivalence_h3_8.json
for SET in slow-stratified-v1 critical-regressions-v1; do
  (cd ../renju-v3 && PYTHONPATH=src ../renju-alphazero/.venv-ser5/bin/python scripts/run_rt30_replay.py \
     --checkpoint ../renju-alphazero/$CK --arms rt3 --keys-file docs/rt30-eval/$SET.json --workers 4 \
     --output ../renju-alphazero/runs/v4_gate/v3_$SET.json)
  .venv-ser5/bin/python scripts/run_rt30_replay.py --checkpoint $CK --arms rt3 --keys-file docs/rt30-eval/$SET.json \
     --workers 4 --output runs/v4_gate/v4_$SET.json
done
.venv-ser5/bin/python scripts/check_rt30_v4_gate.py --v4-equivalence runs/v4_gate/v4_equivalence_h3_8.json \
  --pair runs/v4_gate/v3_slow-stratified-v1.json runs/v4_gate/v4_slow-stratified-v1.json \
  --pair runs/v4_gate/v3_critical-regressions-v1.json runs/v4_gate/v4_critical-regressions-v1.json \
  --output docs/rt30-results/v4_same_move.json
git add docs/rt30-results/v4_same_move.json && git commit -m "docs: TB-RT30-v4 same-move gate" && git push
sudo systemctl restart renju-web     # 로그: "TB-RT30-v4: offered as legacy-certified"
```

- `V4_SAME_MOVE_PASS`이면 v4는 `legacy-certified`가 된다(`rt30.legacy.status`가 기록과 지금 v4 잠금 파일을 대조한다).
- FAIL이면 기록을 커밋하지 않는다. v4는 후보로 남고 웹에 나오지 않는다.
- 죽으면 5초 뒤 다시 뜬다(10분에 5번까지). 끄기(`systemctl stop`)는 SIGTERM이다. 대기 중인 AI 수를 취소하고, 진행 중인 판을 `abandoned`로 남기고, 엔진 worker와 helper를 닫는다. 탐색 중이면 최대 40초 걸린다.
- 설정 값은 `deploy/renju-web.service`의 `ExecStart` 한 줄이다. 바꾸면 `sudo systemctl daemon-reload && sudo systemctl restart renju-web`.

## 4. Caddy

### 4.1 Caddy가 systemd 서비스일 때 (기본)

```bash
sudo cp /etc/caddy/Caddyfile /etc/caddy/Caddyfile.bak-$(date +%Y%m%d-%H%M%S)   # 먼저 백업
caddy hash-password --plaintext '<베타 비밀번호>'                                  # Basic Auth hash
sudoedit /etc/caddy/Caddyfile
```

`deploy/Caddyfile.renju`의 `ser5tack.duckdns.org { ... }` 블록을 **기존 파일에 합친다.** 파일을 통째로 바꾸지 않는다.

- 이 도메인 블록이 이미 있으면 그 안의 `reverse_proxy` 대상을 `127.0.0.1:8000`으로 바꾸고, 나머지 줄을 더한다.
- `<PASTE-HASH-HERE>`를 위의 hash로 바꾼다.
- Caddy 2.8 미만이면 `basic_auth` 대신 `basicauth`라고 쓴다(`caddy version`으로 확인).
- 접근 로그 폴더는 `/var/log/caddy/`다. 없으면 `sudo mkdir -p /var/log/caddy && sudo chown caddy:caddy /var/log/caddy`.

```bash
caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile   # "Valid configuration"
sudo systemctl reload caddy                                         # 끊김 없이 적용
deploy/check.sh --after
```

`https://ser5tack.duckdns.org/`에 접속하면 브라우저가 아이디(`renju`)와 비밀번호를 묻는다. 그 뒤 대국 화면이 나와야 한다.

**같은 도메인에 다른 사이트가 있을 때(경로로 나누기).** 예를 들어 `test-web`(Docker nginx, `127.0.0.1:8080`)이 루트를 쓰고 있다면, 렌주를 `/renju/` 아래에 둔다.

```caddyfile
ser5tack.duckdns.org {
	encode zstd gzip
	basic_auth /renju* {
		renju <hash>
	}
	redir /renju /renju/
	handle_path /renju/* {
		reverse_proxy 127.0.0.1:8000
	}
	handle {
		reverse_proxy 127.0.0.1:8080
	}
}
```

- 페이지는 API를 자기 위치 기준 상대 경로로 부른다. 그래서 루트에서도, `/renju/` 아래에서도 동작한다.
  - 앱은 `handle_path`(prefix 제거)도 `handle`(prefix 유지)도 받는다.
  - 주소는 끝에 `/`를 붙여 `/renju/`로 연다. `redir`이 `/renju`를 `/renju/`로 보낸다.
- `basic_auth /renju*`는 렌주 경로에만 로그인을 건다.
- 점검: `BASE=/renju deploy/check.sh --after`.

### 4.2 Caddy가 Docker 컨테이너일 때

컨테이너 안의 `127.0.0.1`은 이 서버가 아니다. 그래서 4.1의 `reverse_proxy 127.0.0.1:8000`이 닿지 않는다. 둘 중 하나로 한다.

- **권장: 컨테이너를 host 네트워크로 돌린다.** compose의 caddy 서비스에 `network_mode: host`를 두고 `ports:`를 뺀다. 그러면 4.1의 블록이 그대로 동작하고, 앱은 계속 127.0.0.1에만 열린다.
- **bridge 네트워크를 유지할 때:**
  - 앱을 Docker 브리지 게이트웨이 주소에 연다. 서비스 `ExecStart`의 `--host`를 `172.18.0.1`로 바꾼다(`ip -br addr`의 `br-…` 주소).
  - Caddyfile은 `reverse_proxy 172.18.0.1:8000`으로 쓴다.
  - UFW에서 컨테이너 대역만 허용한다: `sudo ufw allow from 172.18.0.0/16 to any port 8000 proto tcp`.
  - 이때도 공유기는 8000을 포워딩하지 않으므로 외부에서는 열리지 않는다.

### 4.3 Nginx와의 관계

80/443은 한 프로그램만 잡을 수 있다. TLS는 Caddy가 맡는다.

- Nginx가 다른 용도 없이 켜져 있다면 끈다: `sudo systemctl disable --now nginx`.
  UFW의 `Nginx Full` 규칙은 80/443을 여는 규칙일 뿐이라 그대로 둬도 된다.
- Nginx에 다른 사이트가 있다면 그 사이트도 Caddy 블록으로 옮긴다. 또는 Nginx를 내부 포트로 내리고 Caddy가 그쪽으로 프록시한다.
  Caddy와 Nginx가 동시에 TLS를 맡게 하지는 않는다.

## 5. 업데이트와 되돌리기

```bash
cd ~/renju-alphazero && git pull
sudo systemctl restart renju-web          # 진행 중인 판은 abandoned로 남는다: 사용자가 적을 때
journalctl -u renju-web -n 30
```

되돌리기:

```bash
git log --oneline -5
git checkout <이전 commit>
sudo systemctl restart renju-web
```

Caddy 설정을 되돌리려면 4.1에서 만든 백업을 `/etc/caddy/Caddyfile`로 복사하고 `sudo systemctl reload caddy`.

## 6. 백업과 로그

```bash
sudo cp deploy/renju-db-backup.service deploy/renju-db-backup.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now renju-db-backup.timer     # 매일 04:17, 최근 14개 보관
sudo systemctl start renju-db-backup.service          # 지금 한 번
ls -l data/backups/
```

- 백업은 서버가 쓰는 중에도 일관된 사본을 만든다(SQLite backup API). 사본마다 `integrity_check`를 한다.
- 백업은 같은 디스크에 있다. 디스크 고장에 대비하려면 `data/backups/`를 가끔 다른 PC로 복사한다.
  예: `scp -P 22222 deploy@ser5tack.duckdns.org:renju-alphazero/data/backups/*.db .`
- 앱 로그: `journalctl -u renju-web`(systemd가 보관·회전한다). 접근 로그: `/var/log/caddy/renju-access.log`(Caddy가 20MiB마다 회전, 10개 보관).
- 대국 원본 로그(`logs/web_play/`)와 DB는 RenjuNet 파생 데이터가 아니지만 **사용자 대국 기록이다.** 커밋하지 않는다.

## 7. 운영 확인

```bash
deploy/check.sh --after
curl -s localhost:8000/api/health; echo
```

`/api/health`에서 볼 것:

- `ai.slots` 2
- `ai.failed` 0
- `ai.moved`: slot을 옮긴 수(정상이며, 많아지면 B3.1과 비교)
- `sessions`
- `db.failures` 0

DB 확인(읽기 전용):

```bash
.venv-ser5/bin/python - <<'PY'
import sys; sys.path.insert(0, 'src')
from weblog.store import open_readonly
c = open_readonly('data/renju_web.db')
print(dict(c.execute("SELECT count(*) n, sum(status='finished') finished FROM games").fetchone()))
print(dict(c.execute("SELECT max(engine_seconds) max_engine, sum(engine_seconds > 27.5) over_deadline, "
                     "sum(emergency_used) emergency FROM moves WHERE actor NOT IN ('HUMAN','OPENING_RULE')").fetchone()))
PY
```

엔진 27.5초 초과와 비상수는 0이어야 한다. 0이 아니면 `rt_json`에서 그 수의 `source`·`watchdog`·`attack_helper_failed`를 본다.
