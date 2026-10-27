'use strict';
// Game list (#/games): filters, newest first, cursor pages; the current page is re-read every refresh.
Admin.views.games = (() => {
  const form = document.querySelector('#filters');
  const body = document.querySelector('#games-table tbody');
  let cursors = [null];  // cursor of each page seen; the last one is the current page
  let next = null;

  function query() {
    const params = Admin.filterQuery(form);
    const cursor = cursors[cursors.length - 1];
    if (cursor) params.set('cursor', cursor);
    return params.toString();
  }

  function cell(text, className) { return Admin.el('td', {textContent: text, className: className || ''}); }

  function row(game) {
    const link = Admin.el('a', {href: `#/games/${game.game_id}`, textContent: Admin.kst(game.started_at)});
    const ai = `${game.agent_key}${game.engine_hash ? ` · ${game.engine_hash}` : ''}`;
    const engine = game.ai_moves ? `${Admin.seconds(game.engine_avg)} / ${Admin.seconds(game.engine_max)}` : '—';
    const tr = Admin.el('tr', {class: 'clickable', onclick: event => {
      if (event.target.tagName !== 'A') location.hash = `#/games/${game.game_id}`;
    }},
    Admin.el('td', {}, link), cell(Admin.kst(game.finished_at)), cell(ai), cell(Admin.COLOR[game.human_color] || game.human_color),
    Admin.el('td', {}, Admin.el('span', {class: `tag ${game.suspected ? 'suspected' : game.status}`, textContent: Admin.statusText(game)})),
    cell(Admin.RESULT[game.result] || (game.result || '—')), cell(String(game.moves), 'num'),
    cell(game.undo_count ? String(game.undo_count) : '—', 'num'), cell(engine, 'num'),
    cell(game.ai_moves ? Admin.seconds(game.wall_max) : '—', 'num'),
    cell(game.soft_limit_exceeded ? String(game.soft_limit_exceeded) : '—', game.soft_limit_exceeded ? 'num warn' : 'num'));
    return tr;
  }

  function restart() { cursors = [null]; Admin.refresh(); }
  form.addEventListener('change', restart);
  form.addEventListener('reset', () => setTimeout(restart));
  form.addEventListener('submit', event => { event.preventDefault(); restart(); });
  document.querySelector('#page-first').addEventListener('click', restart);
  document.querySelector('#page-prev').addEventListener('click', () => { if (cursors.length > 1) { cursors.pop(); Admin.refresh(); } });
  document.querySelector('#page-next').addEventListener('click', () => { if (next) { cursors.push(next); Admin.refresh(); } });

  return {
    enter() {},
    async refresh(context) {
      await Admin.filterOptions(form, context);
      const data = await Admin.get(`/data/games?${query()}`, context);
      next = data.next_cursor;
      body.replaceChildren(...data.games.map(row));
      if (!data.games.length) body.append(Admin.el('tr', {}, Admin.el('td', {colSpan: 11, className: 'muted', textContent: '조건에 맞는 대국이 없습니다.'})));
      document.querySelector('#games-summary').textContent =
        `조건에 맞는 대국 ${data.total}건 · 최신순 · 착수 수는 판 위의 수(무른 수 제외), 시간은 기록된 AI 착수 전체 기준`;
      document.querySelector('#page-label').textContent = `${cursors.length}쪽`;
      document.querySelector('#page-first').disabled = document.querySelector('#page-prev').disabled = cursors.length === 1;
      document.querySelector('#page-next').disabled = !next;
      Admin.connected(`DB 연결됨 · 마지막 갱신 ${Admin.kst(data.observed_at, true)} KST`);
    },
  };
})();
