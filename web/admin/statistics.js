'use strict';
// Statistics share the list's filters; all aggregates come from one server-side snapshot.
Admin.views.statistics = (() => {
  const form = document.querySelector('#filters').cloneNode(true);
  form.id = 'statistics-form';
  document.querySelector('#statistics-filters').append(form);
  const rate = value => value === null ? '표본 없음' : `${(value * 100).toFixed(1)}%`;
  const cell = value => Admin.el('td', {textContent: value, className: 'num'});
  function table(target, headings, rows) {
    const head = Admin.el('thead', {}, Admin.el('tr', {}, ...headings.map(text => Admin.el('th', {textContent: text}))));
    const body = Admin.el('tbody', {}, ...rows);
    if (!rows.length) body.append(Admin.el('tr', {}, Admin.el('td', {colSpan: headings.length, textContent: '조건에 맞는 기록이 없습니다.'})));
    document.querySelector(target).replaceChildren(Admin.el('table', {}, head, body));
  }
  function resultRow(label, row) {
    return Admin.el('tr', {}, Admin.el('td', {textContent: label}), cell(row.games), cell(row.finished),
      cell(row.ai_wins), cell(row.human_wins), cell(row.draws), cell(row.unknown_results), cell(rate(row.ai_win_rate)));
  }
  function bar(label, value, maximum, caption) {
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', '0 0 300 18');
    svg.setAttribute('role', 'img');
    svg.setAttribute('aria-label', `${label}: ${caption}`);
    for (const [width, className] of [[300, 'bar-background'], [maximum ? 300 * value / maximum : 0, 'bar-value']]) {
      const rect = document.createElementNS(svg.namespaceURI, 'rect');
      for (const [key, val] of Object.entries({width, height: 18, rx: 3, class: className})) rect.setAttribute(key, val);
      svg.append(rect);
    }
    return Admin.el('div', {className: 'bar-row'}, Admin.el('span', {textContent: label}), svg,
      Admin.el('span', {textContent: caption, className: 'num'}));
  }
  function card(label, value, note) {
    return Admin.el('section', {className: 'card'}, Admin.el('div', {textContent: label, className: 'muted small'}),
      Admin.el('strong', {textContent: value}), Admin.el('div', {textContent: note, className: 'small muted'}));
  }
  const headings = ['구분', '전체', '정상 종료', 'AI 승', '사람 승', '무승부', '결과 미기록/오류', 'AI 승률'];
  form.addEventListener('change', Admin.refresh);
  form.addEventListener('reset', () => setTimeout(Admin.refresh));
  form.addEventListener('submit', event => { event.preventDefault(); Admin.refresh(); });
  return {
    enter() {},
    async refresh(context) {
      await Admin.filterOptions(form, context);
      const data = await Admin.get(`/data/statistics?${Admin.filterQuery(form)}`, context);
      const total = data.totals;
      const states = total.statuses;
      document.querySelector('#statistics-cards').replaceChildren(
        card('대국 수', total.games, `정상 종료 ${total.finished} · 미완료 ${states.unfinished} · 중단 ${states.abandoned}`),
        card('AI 승률', rate(total.ai_win_rate), `AI ${total.ai_wins}승 · 사람 ${total.human_wins}승 · 무승부 ${total.draws}`),
        card('진행 중', states.in_progress, `중단 추정 ${total.suspected}건 포함`),
        card('무르기 대국', data.undo.games, `완료 ${data.undo.finished} · AI ${data.undo.ai_wins}승 · 사람 ${data.undo.human_wins}승 · 무 ${data.undo.draws}`),
        card('30초 soft limit 초과', data.soft_limit_exceeded, `기록된 AI 착수 ${data.ai_moves}수 기준`));
      document.querySelector('#statistics-warning').textContent = [
        total.unknown_results ? `정상 종료 중 결과 미기록/오류 ${total.unknown_results}건도 승률 분모에 포함됩니다.` : '',
        data.invalid_date_games ? `날짜 오류 ${data.invalid_date_games}건은 날짜별 그래프에서 제외됩니다.` : ''
      ].filter(Boolean).join(' ');
      const engineLabel = row => `${row.agent_key} · ${row.engine_config_hash ? row.engine_config_hash.slice(0, 12) : '해시 미기록'}`;
      const bars = data.by_engine.map(row => bar(engineLabel(row), row.ai_win_rate || 0, 1,
        `${rate(row.ai_win_rate)} (${row.finished}국)`));
      document.querySelector('#engine-bars').replaceChildren(...bars);
      table('#engine-statistics', headings, data.by_engine.map(row => resultRow(engineLabel(row), row)));
      table('#color-statistics', headings, data.by_color.map(row => resultRow(`사람 ${Admin.COLOR[row.human_color] || row.human_color}`, row)));
      const maxDay = Math.max(1, ...data.daily.map(row => row.games));
      const daily = data.daily.map(row => bar(row.day, row.games, maxDay, `${row.games}국`));
      document.querySelector('#daily-chart').replaceChildren(...daily);
      if (!daily.length) document.querySelector('#daily-chart').textContent = '조건에 맞는 기록이 없습니다.';
      const names = {engine_seconds: '계산', queue_seconds: '큐 대기', wall_seconds: '총 응답'};
      table('#timing-statistics', ['시간', '측정 수', '미기록', '잘못된 값', '평균 (초)', '중앙값 (초)', 'p95 (초)', '최대 (초)'],
        Object.entries(data.timing).map(([key, summary]) => Admin.el('tr', {},
          Admin.el('td', {textContent: names[key]}), cell(summary.count), cell(summary.missing), cell(summary.invalid),
          ...['mean', 'median', 'p95', 'max'].map(name => cell(Admin.seconds(summary[name]))))));
      Admin.connected(`통계 갱신 ${Admin.kst(data.observed_at, true)} KST · 시작일 기준 필터 적용`);
    }
  };
})();
