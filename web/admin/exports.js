'use strict';
Admin.views.exports = (() => {
  const start = document.querySelector('#export-start');
  const message = document.querySelector('#export-message');
  const labels = {creating: '생성 중', ready: '완료', failed: '실패'};
  let submitting = false;
  const bytes = value => value === undefined ? '—' : `${(value / 1024 / 1024).toFixed(2)} MiB`;
  start.addEventListener('click', async () => {
    submitting = true;
    start.disabled = true;
    message.textContent = '백업 생성을 요청하고 있습니다…';
    try {
      await Admin.post('/data/exports', {include_raw_logs: document.querySelector('#export-raw').checked});
      message.textContent = '백업 생성 중입니다. 완료되면 다운로드 버튼이 표시됩니다.';
    } catch (error) { message.textContent = error.message; }
    finally { submitting = false; if (Admin.current === Admin.views.exports) Admin.refresh(); }
  });
  function row(job) {
    const counts = job.counts;
    const summary = counts ? `대국 ${counts.games} · 착수 ${counts.moves} · VCT2 ${counts.veto_positions}` : '—';
    let action;
    if (job.state === 'ready') action = Admin.el('a', {href: `${Admin.base}/data/exports/${job.id}/download`, textContent: '압축파일 다운로드'});
    else action = Admin.el('span', {textContent: job.error || (job.state === 'creating' ? '잠시 기다려 주세요.' : '실패')});
    return Admin.el('tr', {},
      Admin.el('td', {textContent: Admin.kst(job.created_at, true)}),
      Admin.el('td', {textContent: labels[job.state] || job.state}),
      Admin.el('td', {textContent: summary}),
      Admin.el('td', {textContent: job.include_raw_logs ? `포함 ${job.raw_logs_included ?? '…'}개 · 누락 ${job.raw_logs_skipped ?? '…'}개` : '미포함'}),
      Admin.el('td', {className: 'num', textContent: bytes(job.bytes)}),
      Admin.el('td', {}, action));
  }
  return {
    enter() {},
    async refresh(context) {
      const data = await Admin.get('/data/exports', context);
      start.disabled = submitting || data.busy;
      document.querySelector('#export-retention').textContent = `완료된 백업은 최근 ${data.keep}개를 보관합니다. 필요한 파일은 PC에 다운로드하세요.`;
      const table = Admin.el('table', {}, Admin.el('thead', {}, Admin.el('tr', {},
        ...['생성 시각 (KST)', '상태', '스냅샷 건수', '원본 로그', '크기', '다운로드 / 오류'].map(text => Admin.el('th', {textContent: text})))),
      Admin.el('tbody', {}, ...data.jobs.map(row)));
      document.querySelector('#exports-table').replaceChildren(table);
      if (!data.jobs.length) document.querySelector('#exports-table').append(Admin.el('p', {textContent: '아직 생성한 백업이 없습니다.'}));
      Admin.connected(`백업 상태 확인 ${Admin.kst(new Date().toISOString(), true)} KST${data.busy ? ' · 생성 중' : ''}`);
    }
  };
})();
