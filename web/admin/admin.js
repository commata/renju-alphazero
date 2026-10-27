'use strict';
// Shell of the admin page: hash routes (#/, #/games, #/games/<id>), one refresh timer, the status view.
const Admin = {
  base: '/renju/admin',
  views: {},
  current: null,
  timer: null,
  loading: false,
  fmt: value => value === null || value === undefined ? '미기록' : String(value),
  seconds: value => value === null || value === undefined ? '미기록' : Number(value).toFixed(1),
  kst(iso, withSeconds = false) {
    if (!iso) return '—';
    const stamp = new Date(iso);
    if (Number.isNaN(stamp.getTime())) return iso;
    return stamp.toLocaleString('ko-KR', {timeZone: 'Asia/Seoul', year: 'numeric', month: '2-digit', day: '2-digit',
      hour: '2-digit', minute: '2-digit', second: withSeconds ? '2-digit' : undefined, hour12: false});
  },
  LETTERS: 'ABCDEFGHIJKLMNO',
  notation(row, col) { return `${Admin.LETTERS[col]}${15 - row}`; },  // the game page's notation()
  STATUS: {in_progress: '진행 중', finished: '정상 종료', unfinished: '미완료', abandoned: '중단'},
  RESULT: {HUMAN_WIN: '사람 승', AI_WIN: 'AI 승', DRAW: '무승부', UNFINISHED: '미완료'},
  COLOR: {BLACK: '흑', WHITE: '백'},
  statusText(game) { return game.suspected ? '중단 추정' : (Admin.STATUS[game.status] || game.status); },
  el(tag, props = {}, ...children) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(props)) {
      if (key === 'class') node.className = value;
      else if (key === 'dataset') Object.assign(node.dataset, value);
      else if (key === 'role' || key.startsWith('aria-')) node.setAttribute(key, value);
      else if (key.startsWith('on')) node.addEventListener(key.slice(2), value);
      else node[key] = value;
    }
    node.append(...children.filter(child => child !== null && child !== undefined));
    return node;
  },
  async get(path) {
    const response = await fetch(Admin.base + path, {cache: 'no-store', credentials: 'same-origin'});
    if (response.status === 401) { location.replace(Admin.base + '/login'); throw new Error('login required'); }
    let data = null;
    try { data = await response.json(); } catch { data = null; }
    if (!response.ok) throw new Error((data && data.error) || `HTTP ${response.status}`);
    return data;
  },
  connected(text) { document.querySelector('#connection').textContent = text; },
  failed(error) {
    if (error.message !== 'login required') Admin.connected(`연결 오류 · 이전 값 유지 · ${error.message}`);
  },
  // One timer for the visible view: 5 s, 30 s while the tab is hidden. A slow request never overlaps.
  schedule() {
    clearTimeout(Admin.timer);
    Admin.timer = setTimeout(Admin.refresh, document.hidden ? 30000 : 5000);
  },
  async refresh() {
    clearTimeout(Admin.timer);
    if (Admin.loading || !Admin.current) { Admin.schedule(); return; }
    Admin.loading = true;
    const view = Admin.current;
    try { await view.refresh(); } catch (error) { Admin.failed(error); } finally {
      Admin.loading = false;
      if (view === Admin.current) Admin.schedule();
    }
  },
  route() {
    const hash = location.hash.replace(/^#/, '') || '/';
    let name = 'status', arg = null;
    const match = hash.match(/^\/games\/([0-9A-Za-z_-]{1,64})$/);
    if (match) { name = 'record'; arg = match[1]; } else if (hash === '/games') name = 'games';
    for (const view of document.querySelectorAll('.view')) view.hidden = view.id !== `view-${name}`;
    for (const link of document.querySelectorAll('nav a')) {
      link.toggleAttribute('aria-current', link.dataset.view === name || (name === 'record' && link.dataset.view === 'games'));
    }
    if (Admin.current && Admin.current.leave) Admin.current.leave();
    Admin.current = Admin.views[name];
    Admin.current.enter(arg);
    Admin.loading = false;
    Admin.refresh();
  },
};

Admin.views.status = {
  enter() {},
  async refresh() {
    const data = await Admin.get('/data/status');
    const db = data.database;
    const summary = document.querySelector('#summary');
    summary.replaceChildren();
    const rate = db.finished_summary.ai_win_rate;
    const rows = {
      '스키마': db.schema_version, '전체 대국': db.counts.games, '기록된 착수 (무른 수 포함)': db.counts.moves,
      'AI VCT2 거부 기록': db.counts.veto_positions, '진행 중': db.in_progress,
      '중단 추정 (DB 변경 없음)': db.suspected_stopped, '무르기가 있었던 대국': db.undo_games.games,
      'AI 승률 (무르기·무승부 포함)': rate === null ? '완료된 대국 없음' : `${(rate * 100).toFixed(1)}%`,
      '정상 종료 표본 수': db.finished_summary.finished
    };
    for (const [label, value] of Object.entries(rows)) {
      summary.append(Admin.el('dt', {textContent: label}), Admin.el('dd', {textContent: Admin.fmt(value)}));
    }
    document.querySelector('#details').textContent = JSON.stringify(db, null, 2);
    document.querySelector('#server').textContent = JSON.stringify(data.game_server, null, 2);
    Admin.connected(`DB 연결됨 · 마지막 갱신 ${Admin.kst(db.observed_at, true)} KST`);
  },
};

document.addEventListener('visibilitychange', () => { if (!document.hidden) Admin.refresh(); else Admin.schedule(); });
window.addEventListener('hashchange', Admin.route);
document.addEventListener('DOMContentLoaded', Admin.route);  // after the deferred view scripts
document.querySelector('#logout').addEventListener('click', async () => {
  try {
    const response = await fetch(Admin.base + '/logout', {method: 'POST', credentials: 'same-origin',
      headers: {'Content-Type': 'application/json'}, body: '{}'});
    if (response.ok || response.status === 401) { clearTimeout(Admin.timer); location.replace(Admin.base + '/login'); }
    else Admin.connected('로그아웃에 실패했습니다. 다시 시도하세요.');
  } catch { Admin.connected('연결 오류로 로그아웃하지 못했습니다.'); }
});
