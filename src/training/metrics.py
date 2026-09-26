"""metrics.jsonl events, run metadata segments, and resume-time log truncation.

Every metrics event carries ``generation`` = the generation that produced it. On
resume from a checkpoint whose ``generation`` is G, events with generation >= G belong
to the generation being re-run, so they are dropped (after backing up the file).
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def timestamp() -> str:
    return datetime.now().strftime('%Y%m%d-%H%M%S')


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
                    encoding='utf-8')
    temp.replace(path)


class MetricsLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: dict) -> None:
        if 'type' not in event or 'generation' not in event:
            raise ValueError('metrics events need type and generation')
        line = json.dumps(event, ensure_ascii=False, allow_nan=False, sort_keys=False)
        with open(self.path, 'a', encoding='utf-8') as handle:
            handle.write(line + '\n')
            handle.flush()


def read_metrics(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()
            if line.strip()]


def truncate_for_resume(run_dir: Path, generation: int, stamp: str) -> dict:
    """Keep only generation < ``generation`` in metrics and per-generation result files.

    The original metrics file is preserved as ``metrics.jsonl.bak-<stamp>``; per-generation
    files being re-run are renamed with the same suffix.
    """
    moved = []
    metrics = run_dir / 'metrics.jsonl'
    kept = dropped = 0
    if metrics.exists():
        backup = metrics.with_name(f'metrics.jsonl.bak-{stamp}')
        shutil.copyfile(metrics, backup)
        events = read_metrics(metrics)
        keep = [e for e in events if e['generation'] < generation]
        kept, dropped = len(keep), len(events) - len(keep)
        metrics.write_text(''.join(json.dumps(e, ensure_ascii=False) + '\n' for e in keep),
                           encoding='utf-8')
        moved.append(str(backup.name))
    for sub in ('self_play', 'evaluation'):
        directory = run_dir / sub
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob('gen*.json')):
            if int(path.stem[3:]) >= generation:
                target = path.with_name(f'{path.name}.bak-{stamp}')
                path.replace(target)
                moved.append(f'{sub}/{target.name}')
    return {'kept_events': kept, 'dropped_events': dropped, 'backups': moved}


class RunMetadata:
    """metadata.json with an append-only list of execution segments."""

    def __init__(self, path: Path, base: dict | None = None):
        self.path = path
        if path.exists():
            self.data = json.loads(path.read_text(encoding='utf-8'))
        else:
            self.data = {**(base or {}), 'segments': []}
        self.segment: dict | None = None

    def start_segment(self, *, start_generation: int, resumed: bool, extra: dict) -> None:
        self.segment = {'start': utc_now(), 'end': None, 'duration_seconds': None,
                        'start_generation': start_generation, 'end_generation': None,
                        'resumed': resumed, 'status': 'running', **extra}
        self.data['segments'].append(self.segment)
        write_json(self.path, self.data)

    def finish_segment(self, *, end_generation: int, status: str, duration: float,
                       error: str | None = None) -> None:
        self.segment.update(end=utc_now(), duration_seconds=duration,
                            end_generation=end_generation, status=status)
        if error:
            self.segment['error'] = error
        write_json(self.path, self.data)
