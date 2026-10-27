'use strict';
let timer;
let loading = false;
const fmt = value => value === null || value === undefined ? '미기록' : String(value);
function redirectLogin() { location.replace('/renju/admin/login'); }
async function refresh() {
  clearTimeout(timer);
  if (loading) return;
  loading = true;
  const connection = document.querySelector('#connection');
  try {
    const response = await fetch('/renju/admin/data/status', {cache: 'no-store', credentials: 'same-origin'});
    if (response.status === 401) { redirectLogin(); return; }
    if (!response.ok) throw new Error('DB 또는 서버 상태 확인에 실패했습니다.');
    const data = await response.json();
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
      const term = document.createElement('dt'); term.textContent = label;
      const detail = document.createElement('dd'); detail.textContent = fmt(value);
      summary.append(term, detail);
    }
    document.querySelector('#details').textContent = JSON.stringify(db, null, 2);
    document.querySelector('#server').textContent = JSON.stringify(data.game_server, null, 2);
    const observed = new Date(db.observed_at).toLocaleString('ko-KR', {timeZone: 'Asia/Seoul'});
    connection.textContent = `DB 연결됨 · 마지막 갱신 ${observed} KST`;
  } catch (error) {
    connection.textContent = `연결 오류 · 이전 값 유지 · ${error.message}`;
  } finally {
    loading = false;
    timer = setTimeout(refresh, document.hidden ? 30000 : 5000);
  }
}
document.addEventListener('visibilitychange', () => { if (!loading) refresh(); });
document.querySelector('#logout').addEventListener('click', async () => {
  try {
    const response = await fetch('/renju/admin/logout', {method: 'POST', credentials: 'same-origin',
      headers: {'Content-Type': 'application/json'}, body: '{}'});
    if (response.ok || response.status === 401) { clearTimeout(timer); redirectLogin(); }
    else document.querySelector('#connection').textContent = '로그아웃에 실패했습니다. 다시 시도하세요.';
  } catch { document.querySelector('#connection').textContent = '연결 오류로 로그아웃하지 못했습니다.'; }
});
refresh();
