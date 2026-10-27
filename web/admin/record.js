'use strict';
// One game (#/games/<id>): numbered record on an SVG board, replay controls, move list, AI time chart,
// VCT2 veto positions. It replays the stored moves only; no engine runs. Coordinates: row0 / col0 from the
// database, row 0 at the top shown as 15 (Admin.notation, the same as the game page).
Admin.views.record = (() => {
  const SVG = 'http://www.w3.org/2000/svg';
  const SIZE = 15, MARGIN = 50, GAP = 40, STONE = 18.5, VIEW = 2 * MARGIN + (SIZE - 1) * GAP;
  const STARS = [[3, 3], [3, 11], [7, 7], [11, 3], [11, 11]];
  const KIND = {HUMAN: '사람', OPENING: '중앙 첫 수', AI: 'AI'};
  const $ = selector => document.querySelector(selector);
  const board = $('#board'), chart = $('#time-chart'), slider = $('#ply-slider');
  let gameId = null, data = null, version = null, live = [], ply = 0;
  let preview = null;   // an undone move shown dashed on the position before it
  let veto = null;      // a veto row shown on its position
  let player = null;    // autoplay interval

  const pos = index => MARGIN + index * GAP;
  function svg(tag, attrs = {}, parent = null, text = null) {
    const node = document.createElementNS(SVG, tag);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
    if (text !== null) node.textContent = text;
    if (parent) parent.append(node);
    return node;
  }
  const where = move => Admin.notation(move.row0, move.col0);
  const color = move => Admin.COLOR[move.player] || move.player;

  function describe(move) {
    const parts = [`${move.ply}수${move.undone_at ? ' (무름)' : ''}`, where(move), color(move), KIND[move.kind] || move.actor];
    if (move.kind === 'AI') {
      parts.push(`계산 ${Admin.seconds(move.engine_seconds)}초`, `대기 ${Admin.seconds(move.wall_seconds)}초`);
      if (move.soft_limit_exceeded) parts.push('30초 초과');
    }
    parts.push(`기록 ${Admin.kst(move.recorded_at, true)}`);
    if (move.undone_at) parts.push(`무른 시각 ${Admin.kst(move.undone_at, true)}`);
    return parts.join(' · ');
  }

  // ---- board -------------------------------------------------------------------------------------
  function drawBoard() {
    board.replaceChildren();
    svg('rect', {x: 0, y: 0, width: VIEW, height: VIEW, rx: 10, class: 'wood'}, board);
    for (let i = 0; i < SIZE; i += 1) {
      svg('line', {x1: pos(0), y1: pos(i), x2: pos(SIZE - 1), y2: pos(i), class: 'grid'}, board);
      svg('line', {x1: pos(i), y1: pos(0), x2: pos(i), y2: pos(SIZE - 1), class: 'grid'}, board);
      const letter = Admin.LETTERS[i], number = String(SIZE - i);
      svg('text', {x: pos(i), y: 17, class: 'coord'}, board, letter);
      svg('text', {x: pos(i), y: VIEW - 17, class: 'coord'}, board, letter);
      svg('text', {x: 17, y: pos(i), class: 'coord'}, board, number);
      svg('text', {x: VIEW - 17, y: pos(i), class: 'coord'}, board, number);
    }
    for (const [r, c] of STARS) svg('circle', {cx: pos(c), cy: pos(r), r: 3.5, class: 'star'}, board);
    const mode = $('#labels').value;
    const shown = live.slice(0, ply);
    shown.forEach((move, index) => {
      const last = index === shown.length - 1;
      const group = svg('g', {class: 'stone-hit'}, board);
      group.addEventListener('click', () => { $('#stone-info').textContent = describe(move); });
      svg('title', {}, group, describe(move));
      svg('circle', {cx: pos(move.col0), cy: pos(move.row0), r: STONE, class: move.player === 'BLACK' ? 'stone black' : 'stone white'}, group);
      if (mode === 'all' || (mode === 'last' && last)) {
        svg('text', {x: pos(move.col0), y: pos(move.row0), class: `num ${move.player === 'BLACK' ? 'on-black' : 'on-white'}${move.ply >= 100 ? ' small' : ''}`}, group, String(move.ply));
      }
      if (last && !preview) svg('circle', {cx: pos(move.col0), cy: pos(move.row0), r: STONE + 3.5, class: 'last-ring'}, board);
    });
    if (preview) {
      const group = svg('g', {class: 'stone-hit'}, board);
      group.addEventListener('click', () => { $('#stone-info').textContent = describe(preview); });
      svg('title', {}, group, describe(preview));
      svg('circle', {cx: pos(preview.col0), cy: pos(preview.row0), r: STONE, class: 'stone undone'}, group);
      svg('text', {x: pos(preview.col0), y: pos(preview.row0), class: `num on-undone${preview.ply >= 100 ? ' small' : ''}`}, group, String(preview.ply));
    }
    if (veto) {
      if (veto.tree_move) {
        const [r, c] = veto.tree_move, d = 11;
        const mark = svg('g', {class: 'veto-x'}, board);
        svg('title', {}, mark, `거부된 후보 ${Admin.notation(r, c)}`);
        svg('line', {x1: pos(c) - d, y1: pos(r) - d, x2: pos(c) + d, y2: pos(r) + d}, mark);
        svg('line', {x1: pos(c) - d, y1: pos(r) + d, x2: pos(c) + d, y2: pos(r) - d}, mark);
      }
      if (veto.played) {
        const [r, c] = veto.played;
        const ring = svg('circle', {cx: pos(c), cy: pos(r), r: STONE + 1, class: 'veto-played'}, board);
        svg('title', {}, ring, `실제로 둔 수 ${Admin.notation(r, c)}`);
      }
    }
  }

  // ---- AI time chart (one series: engine seconds of the AI moves on the board) ---------------------
  function drawChart() {
    chart.replaceChildren();
    const W = 480, H = 190, L = 40, R = 12, T = 12, B = 28;
    chart.setAttribute('viewBox', `0 0 ${W} ${H}`);
    const points = live.filter(m => m.kind === 'AI' && m.engine_seconds !== null && m.engine_seconds !== undefined);
    if (!points.length) {
      svg('text', {x: W / 2, y: H / 2, class: 'chart-empty'}, chart, live.some(m => m.kind === 'AI') ? 'AI 계산시간 미기록' : 'AI 착수 없음');
      $('#chart-note').textContent = '';
      return;
    }
    const peak = Math.max(...points.map(m => m.engine_seconds));
    const top = peak >= 20 ? Math.max(peak, 30) * 1.08 : Math.max(peak * 1.15, 1);
    const last = Math.max(live.length, 2);
    const x = p => L + (p - 1) / (last - 1) * (W - L - R);
    const y = s => T + (1 - s / top) * (H - T - B);
    for (const tick of [0, top / 2, top]) {
      svg('line', {x1: L, x2: W - R, y1: y(tick), y2: y(tick), class: tick === 0 ? 'axis' : 'gridline'}, chart);
      svg('text', {x: L - 6, y: y(tick), class: 'tick y'}, chart, tick < 10 ? tick.toFixed(1) : tick.toFixed(0));
    }
    for (const p of [1, last]) svg('text', {x: x(p), y: H - 8, class: 'tick x'}, chart, `${p}수`);
    if (top >= 30) svg('line', {x1: L, x2: W - R, y1: y(30), y2: y(30), class: 'limit'}, chart);
    svg('polyline', {points: points.map(m => `${x(m.ply)},${y(m.engine_seconds)}`).join(' '), class: 'chart-line'}, chart);
    const radius = points.length > 60 ? 3 : 4.5;
    for (const move of points) {
      svg('circle', {cx: x(move.ply), cy: y(move.engine_seconds), r: move.ply === ply ? radius + 2 : radius,
        class: move.ply === ply ? 'dot current' : 'dot'}, chart);
    }
    const note = top >= 30 ? '점선은 30초 soft limit입니다. 그래프를 누르면 가장 가까운 AI 수의 국면으로 이동합니다.'
      : '그래프를 누르면 가장 가까운 AI 수의 국면으로 이동합니다.';
    $('#chart-note').textContent = note;
    // One hit area for the whole plot: the nearest AI move to the pointer (points can be 3 px apart).
    const cross = svg('line', {y1: T, y2: H - B, class: 'crosshair', visibility: 'hidden'}, chart);
    const hover = svg('circle', {r: radius + 3, class: 'dot hover', visibility: 'hidden'}, chart);
    const area = svg('rect', {x: L, y: 0, width: W - L - R, height: H - B, class: 'hit-area'}, chart);
    const nearest = event => {
      const point = chart.createSVGPoint();
      point.x = event.clientX;
      point.y = event.clientY;
      const at = point.matrixTransform(chart.getScreenCTM().inverse()).x;
      return points.reduce((best, m) => Math.abs(x(m.ply) - at) < Math.abs(x(best.ply) - at) ? m : best);
    };
    area.addEventListener('pointermove', event => {
      const move = nearest(event);
      for (const [node, attrs] of [[cross, {x1: x(move.ply), x2: x(move.ply)}], [hover, {cx: x(move.ply), cy: y(move.engine_seconds)}]]) {
        for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
        node.setAttribute('visibility', 'visible');
      }
      $('#chart-note').textContent = describe(move);
    });
    area.addEventListener('pointerleave', () => {
      cross.setAttribute('visibility', 'hidden');
      hover.setAttribute('visibility', 'hidden');
      $('#chart-note').textContent = note;
    });
    area.addEventListener('click', event => go(nearest(event).ply));
  }

  // ---- move list, vetoes, header ------------------------------------------------------------------
  function drawMoves() {
    const body = $('#moves-table tbody');
    const rows = data.moves.map(move => {
      const notes = [];
      if (move.undone_at) notes.push('무름');
      if (move.soft_limit_exceeded) notes.push('30초 초과');
      if (move.timed_out) notes.push('시간 초과');
      if (move.emergency_used) notes.push('비상수');
      const ai = move.kind === 'AI';
      const current = !move.undone_at && move.ply === ply && !preview;
      const tr = Admin.el('tr', {class: `clickable${move.undone_at ? ' undone' : ''}${current || preview === move ? ' current' : ''}`,
        onclick: () => move.undone_at ? showUndone(move) : go(move.ply)},
      Admin.el('td', {className: 'num', textContent: move.undone_at ? `(${move.ply})` : String(move.ply)}),
      Admin.el('td', {textContent: where(move)}),
      Admin.el('td', {textContent: `${color(move)} · ${KIND[move.kind] || move.actor}`}),
      Admin.el('td', {className: 'num', textContent: ai ? Admin.seconds(move.engine_seconds) : ''}),
      Admin.el('td', {className: 'num', textContent: ai ? Admin.seconds(move.wall_seconds) : ''}),
      Admin.el('td', {className: notes.length ? 'warn' : '', textContent: notes.join(' · ')}));
      if (current || preview === move) tr.setAttribute('aria-current', 'true');
      return tr;
    });
    body.replaceChildren(...rows);
    const marked = body.querySelector('tr.current');
    const wrap = $('.moves-wrap');
    if (marked && (marked.offsetTop < wrap.scrollTop || marked.offsetTop > wrap.scrollTop + wrap.clientHeight - 40)) {
      wrap.scrollTop = marked.offsetTop - wrap.clientHeight / 2;
    }
  }

  function drawVetoes() {
    $('#veto-section').hidden = !data.vetoes.length;
    $('#veto-list').replaceChildren(...data.vetoes.map(row => {
      const tree = row.tree_move ? Admin.notation(...row.tree_move) : '미기록';
      const played = row.played ? Admin.notation(...row.played) : '미기록';
      return Admin.el('li', {}, Admin.el('button', {className: `plain${veto === row ? ' active' : ''}`,
        textContent: `${row.ply}수 뒤 · 후보 ${tree} → 둔 수 ${played} · ${row.reason}`,
        onclick: () => { stop(); veto = row; preview = null; ply = Math.min(row.ply, live.length); draw(); }}));
    }));
  }

  function drawHead() {
    const game = data.game;
    $('#record-title').textContent = `${Admin.kst(game.started_at)} · ${game.agent_name} · 사람 ${Admin.COLOR[game.human_color] || game.human_color}`;
    const short = value => value ? String(value).slice(0, 12) : '—';
    const rows = [
      ['상태', Admin.statusText(game)], ['결과', Admin.RESULT[game.result] || game.result || '—'],
      ['시작 (KST)', Admin.kst(game.started_at, true)], ['종료 (KST)', Admin.kst(game.finished_at, true)],
      ['착수 (판 위)', String(live.length)], ['무르기', String(game.undo_count)],
      ['AI', `${game.agent_key} · ${game.agent_name}`], ['엔진 hash', short(game.engine_config_hash)],
      ['runtime hash', short(game.runtime_hash)], ['커밋', short(game.engine_commit) + (game.git_dirty ? ' (dirty)' : '')],
      ['seed', Admin.fmt(game.seed)], ['게임 ID', game.game_id], ['원본 로그', game.raw_log_dir || '—'],
    ];
    $('#record-meta').replaceChildren(...rows.flatMap(([k, v]) => [Admin.el('dt', {textContent: k}), Admin.el('dd', {textContent: v})]));
    const check = data.verification;
    const box = $('#record-issues');
    if (check.ok) {
      box.replaceChildren(Admin.el('p', {className: 'ok', textContent: `기보 검증 통과 · ${check.replayed}수 재현 (합법 수, 착수 전 국면 hash, 최종 보드, 결과)`}));
    } else {
      box.replaceChildren(Admin.el('div', {className: 'issues', role: 'alert'},
        Admin.el('strong', {textContent: '기보 기록에 문제가 있습니다 (보정하지 않고 그대로 표시합니다)'}),
        Admin.el('ul', {}, ...check.issues.map(issue => Admin.el('li', {
          textContent: `${issue.ply ? `${issue.ply}수: ` : ''}${issue.message} [${issue.code}]`})))));
    }
    $('#follow-wrap').hidden = game.status !== 'in_progress';
  }

  function draw() {
    slider.max = String(live.length);
    slider.value = String(ply);
    $('#ply-label').textContent = preview ? `${preview.ply} (무른 수) / ${live.length}` : `${ply} / ${live.length}`;
    $('#go-first').disabled = $('#go-prev').disabled = ply === 0 && !preview;
    $('#go-next').disabled = $('#go-last').disabled = ply === live.length && !preview;
    drawBoard();
    drawChart();
    drawMoves();
    drawVetoes();
  }

  // ---- navigation and autoplay ----------------------------------------------------------------------
  function go(target) {
    preview = null;
    veto = null;
    ply = Math.max(0, Math.min(live.length, target));
    if (ply > 0) $('#stone-info').textContent = describe(live[ply - 1]);
    draw();
  }
  function showUndone(move) {
    stop();
    veto = null;
    preview = move;
    ply = Math.max(0, Math.min(live.length, move.ply - 1));  // the position it was played on (plies before undo)
    $('#stone-info').textContent = describe(move);
    draw();
  }
  function stop() {
    clearInterval(player);
    player = null;
    $('#play').textContent = '재생';
  }
  function play() {
    if (player) { stop(); return; }
    if (ply >= live.length) go(0);
    $('#play').textContent = '일시정지';
    player = setInterval(() => {
      if (ply >= live.length) { stop(); return; }
      go(ply + 1);
    }, Number($('#speed').value));
  }

  $('#go-first').addEventListener('click', () => { stop(); go(0); });
  $('#go-prev').addEventListener('click', () => { stop(); go(preview ? ply : ply - 1); });
  $('#go-next').addEventListener('click', () => { stop(); go(ply + 1); });
  $('#go-last').addEventListener('click', () => { stop(); go(live.length); });
  slider.addEventListener('input', () => { stop(); go(Number(slider.value)); });
  $('#play').addEventListener('click', play);
  $('#speed').addEventListener('change', () => { if (player) { stop(); play(); } });
  $('#labels').addEventListener('change', drawBoard);
  document.addEventListener('keydown', event => {
    if (Admin.current !== Admin.views.record || !data || event.target.closest('input, select, textarea')) return;
    if (event.key === 'ArrowLeft') { stop(); go(preview ? ply : ply - 1); event.preventDefault(); }
    if (event.key === 'ArrowRight') { stop(); go(ply + 1); event.preventDefault(); }
  });

  function apply(next) {
    const oldTotal = live.length;
    const atEnd = !data || ply === oldTotal;
    data = next;
    version = next.version;
    live = next.moves.filter(m => !m.undone_at).sort((a, b) => a.ply - b.ply || a.seq - b.seq);
    if (preview) preview = next.moves.find(m => m.seq === preview.seq && m.undone_at) || null;
    if (veto) veto = next.vetoes.find(v => v.id === veto.id) || null;
    if (atEnd && (oldTotal === 0 || $('#follow').checked || next.game.status !== 'in_progress')) {
      ply = live.length;   // first load, or following the latest move
    } else {
      ply = Math.min(ply, live.length);   // looking at an earlier move: stay there
    }
    drawHead();
    draw();
  }

  return {
    enter(id) {
      if (id === gameId) return;
      stop();
      gameId = id; data = null; version = null; live = []; ply = 0; preview = null; veto = null;
      $('#follow').checked = true;
      $('#record-title').textContent = '불러오는 중…';
      $('#record-meta').replaceChildren();
      $('#record-issues').replaceChildren();
      $('#stone-info').textContent = '돌을 누르면 그 수의 정보를 보여 줍니다.';
      board.replaceChildren();
      chart.replaceChildren();
      $('#moves-table tbody').replaceChildren();
      $('#veto-section').hidden = true;
    },
    leave() { stop(); },
    async refresh() {
      const id = gameId;
      const query = version ? `?since=${encodeURIComponent(version)}` : '';
      const next = await Admin.get(`/data/games/${id}${query}`);
      if (id !== gameId) return;  // the page moved to another game meanwhile
      if (!next.unchanged) apply(next);
      Admin.connected(`DB 연결됨 · 마지막 갱신 ${Admin.kst(next.observed_at, true)} KST`);
    },
  };
})();
