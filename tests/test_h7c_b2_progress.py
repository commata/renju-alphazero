"""B2 stdout heartbeat and out-of-order drift persistence. No long games or wall-clock minute sleeps."""
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from threading import Event
import unittest
from unittest import mock

from hybrid.progress import Progress
from scripts import h7c_drift as drift, h7c_loop as loop


class HeartbeatTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.log = self.root / 'c4-selfplay.log'

    def test_bounded_tail_prefers_progress_over_json_and_game_keys(self):
        self.log.write_text('x' * 20000 + '\n[old] 4/4 games 100% ETA 0 min\n', encoding='utf-8')
        offset = self.log.stat().st_size
        with self.log.open('a', encoding='utf-8') as handle:
            handle.write('[selfplay] 7/1000 games 0.7% 1 min ETA 100 min\n')
            handle.write('h7c:c4@rapfi/8714/0/black result=win\n{"games": 1000}\n')
        line = loop.last_progress(self.log, offset)
        self.assertIn('7/1000', line)
        self.assertNotIn('4/4', line)
        self.assertIn('unknown', loop.last_progress(self.log, self.log.stat().st_size))

    def test_missing_log_or_no_progress_is_reported_honestly(self):
        self.assertIn('unknown', loop.last_progress(self.log))
        self.log.write_text('{"summary": 1}\n')
        self.assertIn('unknown', loop.last_progress(self.log))

    def test_value_step_can_be_forwarded(self):
        self.log.write_text('[h7c-value] step 10/100 objective 0.4\n')
        self.assertIn('step 10/100', loop.last_progress(self.log))

    def test_wait_timeout_prints_stage_elapsed_and_child_progress(self):
        process = mock.Mock(pid=456)
        process.wait.side_effect = [subprocess.TimeoutExpired(['python'], 60), 0]
        messages, pids = [], []
        def start(*args, **kwargs):
            # Simulate a flushed child progress line, after the parent opened its append log.
            with self.log.open('a') as handle:
                handle.write('[selfplay] 60/1000 games 6% 1 min ETA 16 min\n')
            return process
        with mock.patch.object(loop.subprocess, 'Popen', side_effect=start), \
                mock.patch.object(loop, 'perf_counter', side_effect=[0.0, 61.0]):
            code = loop.execute(['python'], self.log, self.root, pids.append,
                                log_progress=lambda text, **kw: messages.append((text, kw)))
        self.assertEqual((code, pids), (0, [456]))
        self.assertEqual(len(messages), 1)
        self.assertIn('b2:c4-selfplay', messages[0][0])
        self.assertIn('elapsed 1.0 min', messages[0][0])
        self.assertIn('60/1000', messages[0][0])
        self.assertTrue(messages[0][1]['flush'])
        self.assertEqual(process.wait.call_args_list, [mock.call(timeout=60.0), mock.call(timeout=60.0)])

    def test_quick_exit_has_no_spurious_heartbeat(self):
        process = mock.Mock(pid=789)
        process.wait.return_value = 0
        log = mock.Mock()
        with mock.patch.object(loop.subprocess, 'Popen', return_value=process):
            self.assertEqual(loop.execute(['python'], self.log, self.root, lambda pid: None, log_progress=log), 0)
        log.assert_not_called()

    def test_interrupt_still_terminates_child(self):
        process = mock.Mock(pid=789)
        process.wait.side_effect = [KeyboardInterrupt, 0]
        with mock.patch.object(loop.subprocess, 'Popen', return_value=process), self.assertRaises(KeyboardInterrupt):
            loop.execute(['python'], self.log, self.root, lambda pid: None)
        process.terminate.assert_called_once()


class DriftCollectionTest(unittest.TestCase):
    def test_resume_eta_uses_only_newly_finished_games(self):
        progress = drift.DriftProgress(100, 90)
        progress.started = 10.0
        self.assertIn('ETA unknown', progress.line(now=70.0))
        progress.done = 92  # two new games in a minute; eight remain: ETA four minutes
        line = progress.line(now=70.0)
        self.assertIn('92/100 games', line)
        self.assertIn('92.0%', line)
        self.assertIn('ETA 4 min', line)

    def test_fast_later_game_is_written_before_blocked_first_game(self):
        fast_written = Event()
        class Writer(io.StringIO):
            flushes = 0
            def write(self, value):
                result = super().write(value)
                if json.loads(value)['key'] == 'fast':
                    fast_written.set()
                return result
            def flush(self):
                self.flushes += 1
        def worker(tasks):
            key = tasks[0]['key']
            if key == 'slow' and not fast_written.wait(2):
                raise AssertionError('later completed game was blocked by ordered collection')
            return [{'key': key}]
        writer, done, messages = Writer(), {'previous-1': {}, 'previous-2': {}}, []
        progress = Progress('h7c-drift', 4, 'games', every_seconds=.01,
                            log=lambda line, **kw: messages.append(line))
        progress.done = 2
        with ThreadPoolExecutor(max_workers=2) as pool, mock.patch.object(drift.ar, '_play_chunk', worker):
            # Any accidental reintroduction of ordered map must fail this regression test.
            with mock.patch.object(pool, 'map', side_effect=AssertionError('do not use ordered pool.map')):
                result = drift.collect_games(pool, [{'key':'slow'}, {'key':'fast'}], writer, done, progress)
        rows = [json.loads(line) for line in writer.getvalue().splitlines()]
        self.assertEqual([r['key'] for r in rows], ['fast', 'slow'])
        self.assertTrue(all(r['arm'] == 'drift' for r in rows))
        self.assertEqual([rows[0]['key'], result[0][0]['key']], ['fast', 'slow'])
        self.assertEqual(writer.flushes, 2)
        self.assertEqual(progress.done, 4)
        self.assertTrue(any('4/4' in line and '100.0%' in line and 'ETA' in line for line in messages))

    def test_no_result_yet_still_emits_progress_heartbeat(self):
        release, messages = Event(), []
        def worker(tasks):
            if not release.wait(2):
                raise AssertionError('test release timed out')
            return [{'key': tasks[0]['key']}]
        def log(line, **kw):
            messages.append(line)
            if 'pending' in line:
                release.set()  # only a real no-result heartbeat releases the first game
        progress = Progress('h7c-drift', 1, 'games', every_seconds=.01, log=log)
        try:
            with ThreadPoolExecutor(max_workers=1) as pool, mock.patch.object(drift.ar, '_play_chunk', worker):
                drift.collect_games(pool, [{'key': 'slow'}], io.StringIO(), {}, progress)
        finally:
            release.set()
        self.assertTrue(any('pending' in line and '0/1' in line and 'ETA' in line for line in messages))


if __name__ == '__main__':
    unittest.main()
