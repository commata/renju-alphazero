"""R2-D4 record copies: desktop logs copied with a manifest, and an exact check of what a records-only commit stages.

    copy-logs      copy the matching logs into a new records folder (never over a file, never deleting or moving an
                   original), compare every copy's SHA-256 with its original and write ``log_manifest.json``
                   (original path, copy path, bytes, SHA-256). An existing SHA256SUMS is never touched.
    manifest       ``records_manifest.json`` of a records folder: every file below it (relative path, bytes, SHA-256),
                   the check result and the log copies included; written once, never over an existing manifest.
    verify-staged  before a records-only commit: every expected path exists and is not ignored, no tracked file has
                   unstaged changes, and the staged set is exactly the expected set (a missing, an unexpected or an
                   ignored file stops the commit).

    With --resume, copy-logs reuses matching copies after an interruption; manifest verifies an existing manifest.
    Both commands refuse changed contents and preserve every existing file.

    python scripts/h7r_d4_records.py copy-logs --src-dir runs/h7r/R2-D4-BASELINE-1/logs --pattern 'smoke-*.log' \\
        --dest-dir docs/h7r/r2/R2-D4-BASELINE-1/records/smoke/run-logs \\
        --manifest docs/h7r/r2/R2-D4-BASELINE-1/records/smoke/log_manifest.json
    python scripts/h7r_d4_records.py verify-staged --expect <path> [--expect <path> ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / 'src'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

MANIFEST_FORMAT = 'h7r-d4-log-manifest-v1'
RECORDS_FORMAT = 'h7r-d4-records-manifest-v1'


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def copy_logs(src_dir, pattern: str, dest_dir, manifest, root=ROOT, *, resume: bool = False) -> dict:
    from hybrid.h7r_ledger import publish_json

    root, src_dir, dest_dir, manifest = Path(root), Path(src_dir), Path(dest_dir), Path(manifest)
    sources = sorted(p for p in src_dir.glob(pattern) if p.is_file())
    if not sources:
        raise ValueError(f'no {pattern} in {src_dir}: nothing copied')
    if manifest.exists() and not resume:
        raise ValueError(f'{manifest} exists')
    targets = [dest_dir / p.name for p in sources]
    clash = [str(t) for t in targets if t.exists()]
    if clash and not resume:
        raise ValueError(f'copies exist already (never overwritten): {clash}')
    for source, target in zip(sources, targets):
        if target.exists() and sha256(source) != sha256(target):
            raise ValueError(f'{target}: existing copy SHA-256 differs from {source}')
    dest_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for source, target in zip(sources, targets):
        before = sha256(source)
        if not target.exists():
            shutil.copy2(source, target)
        after, copied = sha256(source), sha256(target)
        if not before == after == copied:
            raise ValueError(f'{target}: copy SHA-256 differs from {source}')
        rel = lambda p: str(Path(p).resolve().relative_to(root.resolve())) if Path(p).resolve().is_relative_to(root.resolve()) else str(p)  # noqa: E731
        entries.append({'original': rel(source), 'copy': rel(target), 'bytes': target.stat().st_size,
                        'sha256': copied})
    body = {'format': MANIFEST_FORMAT, 'pattern': pattern, 'logs': entries,
            'note': 'copies of the desktop run logs; the originals stay in place, SHA256SUMS is not changed'}
    if manifest.exists():
        if json.loads(manifest.read_text(encoding='utf-8')) != body:
            raise ValueError(f'{manifest}: existing manifest differs from the checked log copies')
    else:
        publish_json(manifest, body)
    return body


def records_manifest(folder, output, *, resume: bool = False) -> dict:
    from hybrid.h7r_ledger import publish_json

    folder, output = Path(folder), Path(output)
    if output.exists() and not resume:
        raise ValueError(f'{output} exists')
    files = [p for p in sorted(folder.rglob('*')) if p.is_file() and p.resolve() != output.resolve()]
    if not files:
        raise ValueError(f'{folder}: no files')
    forbidden = [str(p) for p in files if any(s in ('.jsonl', '.pt', '.pth', '.ckpt', '.onnx', '.npz', '.npy', '.h5')
                                             for s in p.suffixes)]
    if forbidden:
        raise ValueError(f'raw journal / model / tensor cannot enter a records manifest: {forbidden}')
    body = {'format': RECORDS_FORMAT, 'folder': str(folder),
            'files': [{'path': str(p.relative_to(folder)), 'bytes': p.stat().st_size, 'sha256': sha256(p)}
                      for p in files]}
    if output.exists():
        if json.loads(output.read_text(encoding='utf-8')) != body:
            raise ValueError(f'{output}: existing manifest differs from the current records')
    else:
        publish_json(output, body)
    return body


def _git(root, *args) -> str:
    return subprocess.run(['git', *args], cwd=root, capture_output=True, text=True, check=True).stdout


def verify_staged(expected: list[str], root=ROOT) -> list[str]:
    """Problems with a records-only commit (empty when the staged set is exactly ``expected``)."""
    root = Path(root)
    expected = sorted(set(expected))
    problems = []
    for path in expected:
        if not (root / path).is_file():
            problems.append(f'missing: {path}')
        elif subprocess.run(['git', 'check-ignore', '-q', '--no-index', path], cwd=root).returncode == 0:
            problems.append(f'ignored by .gitignore: {path}')
    unstaged = [line for line in _git(root, 'diff', '--name-only').splitlines() if line]
    if unstaged:
        problems.append(f'unstaged tracked changes: {unstaged}')
    staged = sorted(line for line in _git(root, 'diff', '--cached', '--name-only').splitlines() if line)
    if staged != expected:
        extra, absent = sorted(set(staged) - set(expected)), sorted(set(expected) - set(staged))
        problems.append(f'staged set differs: unexpected {extra}, not staged {absent}')
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    c = sub.add_parser('copy-logs')
    c.add_argument('--src-dir', required=True)
    c.add_argument('--pattern', required=True)
    c.add_argument('--dest-dir', required=True)
    c.add_argument('--manifest', required=True)
    c.add_argument('--resume', action='store_true', help='reuse byte-identical copies and verify an existing manifest')
    m = sub.add_parser('manifest')
    m.add_argument('--dir', required=True)
    m.add_argument('--output', required=True)
    m.add_argument('--resume', action='store_true', help='verify an existing manifest without rewriting it')
    v = sub.add_parser('verify-staged')
    v.add_argument('--expect', action='append', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'copy-logs':
            body = copy_logs(args.src_dir, args.pattern, args.dest_dir, args.manifest, resume=args.resume)
            for e in body['logs']:
                print(f"copied {e['original']} -> {e['copy']} ({e['bytes']} bytes, {e['sha256'][:12]})", flush=True)
            return 0
        if args.command == 'manifest':
            body = records_manifest(args.dir, args.output, resume=args.resume)
            print(f"records manifest: {len(body['files'])} files -> {args.output}", flush=True)
            return 0
        problems = verify_staged(args.expect)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f'STOP: {exc}', file=sys.stderr, flush=True)
        return 1
    for problem in problems:
        print(f'STOP: {problem}', flush=True)
    if not problems:
        print(f'staged set OK: {len(args.expect)} files', flush=True)
    return 1 if problems else 0


if __name__ == '__main__':
    raise SystemExit(main())
