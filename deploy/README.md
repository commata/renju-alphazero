# SER5 웹 운영 배포 (TB-RT30-v3)

미니PC(SER5) 한 대에서 렌주 웹대국을 `https://ser5tack.duckdns.org/`로 운영하는 구성이다.
엔진과 웹 구조는 `docs/web-ser5.md` §8(멀티 세션), §9.11(v3 연결), §9.12(B3, `--ai-slots 2`)를 따른다.

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
                                                                   ├─ slot 0: engine worker + V8-B helper (+ 예비 짝)
                                                                   ├─ slot 1: engine worker + V8-B helper (+ 예비 짝)
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

sudo cp deploy/renju-web.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now renju-web
journalctl -u renju-web -f      # "TB-RT30-v3: ... + V8-B helper each", "worker ready", "Renju local web" 확인 후 Ctrl-C
curl -s localhost:8000/api/health; echo     # "ai": {"slots": 2, ...}
```

- 시작 전에 `rt30.native.build --if-stale`가 돈다. `git pull`로 C 소스가 바뀌었거나 Python이 바뀌었을 때만 다시 빌드한다.
  native scan이 맞지 않으면 서버는 시작 단계에서 멈춘다(v3가 v1 시간 배분으로 몰래 돌지 않게).
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
