"""Optional Chromium regression for admin stages 3..6 (not part of unittest discovery).

    python -m pip install playwright
    python tests/admin_browser_smoke.py --chromium /usr/bin/chromium

Uses a disposable real PlaySession/LogStore database and account, never SER5.
"""
from __future__ import annotations

import argparse
import re
import asyncio
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from playwright.async_api import async_playwright, expect  # noqa: E402
from tests.test_web_admin_games import build_database, web  # noqa: E402
from renju import BLACK, WHITE  # noqa: E402
from webadmin.auth import Auth, hash_password  # noqa: E402
from webadmin.exports import ExportManager  # noqa: E402
from webadmin.games import game_detail  # noqa: E402
from webadmin.server import AdminServer  # noqa: E402
from weblog.store import LogStore  # noqa: E402


async def run(args):
    with TemporaryDirectory() as tmp:
        folder = Path(tmp)
        db, ids = build_database(folder)
        store = LogStore(db)
        branch = web.PlaySession(store=store, log_root=folder / 'logs')
        branch.reset(agent_key='corner', human_color=BLACK, seed=21)
        branch.play_human(6, 6)
        history = [list(move) for move in branch.game.history]
        store.record_veto(branch.db_game_id, ply=4, payload={'history': history, 'tree_move': [8, 7], 'played': [6, 7]})
        branch.play_human(6, 7)
        branch.undo()
        branch.undo()
        branch.play_human(8, 8)
        branch.play_human(8, 9)
        live = web.PlaySession(store=store, log_root=folder / 'logs')
        live.reset(agent_key='corner', human_color=WHITE, seed=22)
        live.play_human(6, 7)
        store.flush()
        detail = game_detail(db, branch.db_game_id)
        undone_seq = next(m['seq'] for m in detail['moves'] if m['undone_at'] and m['ply'] == 5)
        manager = ExportManager(db, folder / 'exports', repo=ROOT, log_root=folder / 'logs')
        server = AdminServer(('127.0.0.1', 0), db=db, origin='http://127.0.0.1',
                             auth=Auth('admin', hash_password('ui-only-password')), assets=ROOT / 'web/admin', exports=manager)
        base = f'http://127.0.0.1:{server.server_address[1]}'
        server.origin = base
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            async with async_playwright() as playwright:
                browser = await playwright.chromium.launch(executable_path=args.chromium, headless=True,
                                                            args=['--no-sandbox'])
                context = await browser.new_context(viewport={'width': 1440, 'height': 1000}, accept_downloads=True)
                page = await context.new_page()
                errors = []
                page.on('pageerror', lambda error: errors.append(str(error)))
                await page.goto(base + '/renju/admin/login')
                await page.locator('[name=username]').fill('admin')
                await page.locator('[name=password]').fill('ui-only-password')
                await page.get_by_role('button', name='로그인', exact=True).click()
                await page.wait_for_url('**/renju/admin/')
                await page.goto(base + '/renju/admin/#/games')
                await page.locator('#games-table tbody tr').first.wait_for()
                await page.goto(base + f'/renju/admin/#/games/{branch.db_game_id}')
                await expect(page.locator('#ply-label')).to_have_text('6 / 6')
                await page.locator(f'#moves-table tr[data-seq="{undone_seq}"]').click()
                # The old branch has a black stone at (6,6), not the replacement at (8,8).
                assert await page.locator('#board circle.black[cx="290"][cy="290"]').count() == 1
                assert await page.locator('#board circle.black[cx="370"][cy="370"]').count() == 0
                await page.locator('#veto-list button').first.click()
                assert await page.locator('#board circle.black[cx="290"][cy="290"]').count() == 1
                await page.locator('#go-first').click()
                await page.locator('#play').click()
                await expect(page.locator('#ply-label')).to_have_text(re.compile(r'^1 /'))
                await page.locator('#play').click()
                # Route away during an intentionally delayed old record request.
                started = asyncio.Event()
                async def delayed(route):
                    started.set()
                    await asyncio.sleep(.5)
                    await route.continue_()
                await page.route(f'**/data/games/{ids["win"]}*', delayed)
                await page.goto(base + f'/renju/admin/#/games/{ids["win"]}')
                await asyncio.wait_for(started.wait(), timeout=5)
                await page.goto(base + f'/renju/admin/#/games/{live.db_game_id}')
                await expect(page.locator('#ply-label')).to_have_text('3 / 3')
                await page.wait_for_timeout(700)
                assert live.db_game_id in await page.locator('#record-provenance').text_content()
                live.play_human(6, 8)
                store.flush()
                await expect(page.locator('#ply-label')).to_have_text('5 / 5', timeout=10000)
                await page.locator('#go-prev').click()
                live.play_human(6, 9)
                store.flush()
                await expect(page.locator('#ply-label')).to_have_text('4 / 7', timeout=10000)
                await page.goto(base + '/renju/admin/#/statistics')
                await page.wait_for_selector('#statistics-cards .card')
                await page.locator('#statistics-form [name=undo]').select_option('no')
                await expect(page.locator('#statistics-cards .card').nth(3).locator('strong')).to_have_text('0')
                args.screenshots.mkdir(parents=True, exist_ok=True)
                await page.screenshot(path=str(args.screenshots / 'statistics-desktop.png'), full_page=True)
                await page.set_viewport_size({'width': 390, 'height': 844})
                await page.screenshot(path=str(args.screenshots / 'statistics-mobile.png'), full_page=True)
                assert await page.evaluate('() => document.documentElement.scrollWidth <= innerWidth')
                await page.goto(base + '/renju/admin/#/exports')
                await page.locator('#export-start').click()
                download_link = page.get_by_role('link', name='압축파일 다운로드')
                await download_link.wait_for(timeout=15000)
                async with page.expect_download() as download:
                    await download_link.click()
                assert (await download.value).suggested_filename.endswith('.tar.gz')
                await page.screenshot(path=str(args.screenshots / 'exports-mobile.png'), full_page=True)
                await page.goto(base + f'/renju/admin/#/games/{branch.db_game_id}')
                await expect(page.locator('#ply-label')).to_have_text('6 / 6')
                await page.screenshot(path=str(args.screenshots / 'record-mobile.png'), full_page=True)
                assert await page.evaluate('() => document.documentElement.scrollWidth <= innerWidth')
                assert not errors, errors
                await browser.close()
                print('PASS Chromium: historic undo/veto positions, replay, delayed responses, live follow/stay, statistics filters, export download, mobile layout; no page errors')
        finally:
            server.shutdown()
            thread.join()
            server.server_close()
            store.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chromium', default='/usr/bin/chromium')
    parser.add_argument('--screenshots', type=Path, default=ROOT / 'logs/admin-ui')
    asyncio.run(run(parser.parse_args()))
