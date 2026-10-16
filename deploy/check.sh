#!/usr/bin/env bash
# Read-only check of the SER5 web deployment (deploy/README.md). Changes nothing; prints what it finds.
#   deploy/check.sh            # before installing: who holds the ports, how Caddy runs, DNS
#   deploy/check.sh --after    # after installing: also the service, the local and the public endpoint
DOMAIN=${DOMAIN:-ser5tack.duckdns.org}
PORT=${PORT:-8000}
APP=${APP:-/home/deploy/renju-alphazero}
after=0; [ "${1:-}" = "--after" ] && after=1
ok() { printf '  [ok]   %s\n' "$*"; }
warn() { printf '  [WARN] %s\n' "$*"; }
info() { printf '  [..]   %s\n' "$*"; }
SUDO=""; sudo -n true 2>/dev/null && SUDO="sudo -n"

echo "== DNS (DuckDNS) =="
public=$(curl -4 -s --max-time 5 icanhazip.com || true)
resolved=$(dig +short "$DOMAIN" 2>/dev/null | tail -1)
info "public IP $public, $DOMAIN -> $resolved"
[ -n "$public" ] && [ "$public" = "$resolved" ] && ok "DuckDNS points at this connection" || warn "DuckDNS does not match the public IP (updater running?)"

echo "== listening ports 80 / 443 / $PORT =="
for p in 80 443 "$PORT"; do
    line=$($SUDO ss -ltnpH "sport = :$p" 2>/dev/null | head -3)
    if [ -z "$line" ]; then info "$p: nobody listens"; else echo "$line" | sed "s/^/  [..]   $p: /"; fi
done
if $SUDO ss -ltnH "sport = :$PORT" 2>/dev/null | grep -qvE '127\.0\.0\.1:|\[::1\]:'; then
    warn "port $PORT listens beyond localhost: the app must bind 127.0.0.1 (Caddy is the only way in)"
fi
[ -z "$SUDO" ] && info "(process names need sudo: run 'sudo -v' first for them)"

echo "== Caddy =="
if systemctl list-unit-files caddy.service >/dev/null 2>&1 && systemctl is-active --quiet caddy; then
    ok "caddy.service is active (systemd): reverse_proxy 127.0.0.1:$PORT works as written"
    caddy version 2>/dev/null | sed 's/^/  [..]   caddy /'
    [ -r /etc/caddy/Caddyfile ] && grep -n "$DOMAIN\|reverse_proxy\|basic_*auth" /etc/caddy/Caddyfile | sed 's/^/  [..]   Caddyfile:/'
else
    info "caddy.service is not active"
fi
if command -v docker >/dev/null 2>&1; then
    containers=$($SUDO docker ps --format '{{.Names}} {{.Image}} {{.Ports}}' 2>/dev/null)
    if [ -n "$containers" ]; then
        echo "$containers" | sed 's/^/  [..]   docker: /'
        echo "$containers" | grep -qi caddy && warn "Caddy runs in Docker: use the Docker variant of deploy/README.md §4 (127.0.0.1 in the container is not this host)"
    else
        info "docker: no running container visible (or no permission: sudo -v)"
    fi
fi
if systemctl is-active --quiet nginx 2>/dev/null; then
    warn "nginx is active: only one of Caddy / Nginx may hold 80/443 (see deploy/README.md §4)"
else
    info "nginx is not active"
fi

echo "== firewall =="
$SUDO ufw status 2>/dev/null | sed 's/^/  [..]   /' || info "ufw status needs sudo"

echo "== application =="
[ -x "$APP/.venv-ser5/bin/python" ] && ok "venv $("$APP/.venv-ser5/bin/python" --version 2>&1)" || warn "no $APP/.venv-ser5"
[ -f "$APP/runs/h3_policy_64x4/best.pt" ] && ok "H3 checkpoint present" || warn "no runs/h3_policy_64x4/best.pt"
(cd "$APP" && PYTHONPATH=src .venv-ser5/bin/python -c 'from rt30.native import build; import sys; r = build.stale(); print("  [ok]   native scan built for this checkout" if r is None else f"  [WARN] native scan: {r} (python -m rt30.native.build --if-stale)")') 2>/dev/null
if [ "$after" = 1 ]; then
    echo "== service =="
    systemctl is-active --quiet renju-web && ok "renju-web.service active" || warn "renju-web.service not active (journalctl -u renju-web -n 50)"
    health=$(curl -s --max-time 5 "http://127.0.0.1:$PORT/api/health")
    echo "$health" | grep -q '"ai"' && ok "local health: $health" || warn "local health failed: $health"
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "https://$DOMAIN/api/health")
    case "$code" in
        200) ok "https://$DOMAIN/api/health -> 200" ;;
        401) ok "https://$DOMAIN/api/health -> 401 (Basic Auth is on: expected)" ;;
        *) warn "https://$DOMAIN/api/health -> $code" ;;
    esac
fi
