"""Audit persisted Stage 6 evaluation files against the Renju engine and metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from renju import BLACK, WHITE, Game  # noqa: E402
from training.metrics import read_metrics  # noqa: E402


class EvaluationAuditError(RuntimeError):
    pass


def _expected_result(winner: int | None, model_color: str) -> str:
    color = BLACK if model_color == 'black' else WHITE if model_color == 'white' else None
    if color is None:
        raise EvaluationAuditError(f'unknown model_color: {model_color!r}')
    if winner is None:
        return 'draw'
    return 'win' if winner == color else 'loss'


def _wld(results: list[str]) -> dict:
    return {
        'wins': results.count('win'),
        'losses': results.count('loss'),
        'draws': results.count('draw'),
        'games': len(results),
    }


def verify_run(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    evaluation_dir = run_dir / 'evaluation'
    files = sorted(evaluation_dir.glob('gen*.json'))
    if not files:
        raise EvaluationAuditError(f'no evaluation files under {evaluation_dir}')

    metrics = read_metrics(run_dir / 'metrics.jsonl')
    metric_index = {}
    for event in metrics:
        if event.get('type') != 'evaluation':
            continue
        key = (event['generation'], event['opponent'])
        if key in metric_index:
            raise EvaluationAuditError(f'duplicate evaluation metric: {key}')
        metric_index[key] = event

    checked_games = 0
    checked_opponents = 0
    generations = []

    for path in files:
        payload = json.loads(path.read_text(encoding='utf-8'))
        generation = payload['generation']
        generations.append(generation)
        for opponent_name, opponent in payload['opponents'].items():
            checked_opponents += 1
            overall = []
            by_color = {'black': [], 'white': []}
            for record in opponent['games']:
                game = Game()
                for move in record['moves']:
                    game.play(*move)
                if not game.done:
                    raise EvaluationAuditError(
                        f'gen {generation} {opponent_name}: replay did not terminate')
                if game.winner != record['winner']:
                    raise EvaluationAuditError(
                        f'gen {generation} {opponent_name}: winner mismatch '
                        f'{game.winner} != {record["winner"]}')
                expected = _expected_result(game.winner, record['model_color'])
                if record['result'] != expected:
                    raise EvaluationAuditError(
                        f'gen {generation} {opponent_name}: result mismatch '
                        f'{record["result"]!r} != {expected!r}')
                overall.append(expected)
                by_color[record['model_color']].append(expected)
                checked_games += 1

            expected_summary = _wld(overall)
            summary = opponent['summary']
            for key in ('wins', 'losses', 'draws', 'games'):
                if summary[key] != expected_summary[key]:
                    raise EvaluationAuditError(
                        f'gen {generation} {opponent_name}: summary {key} mismatch')
            for color in ('black', 'white'):
                expected_color = _wld(by_color[color])
                for key in ('wins', 'losses', 'draws', 'games'):
                    if summary[color][key] != expected_color[key]:
                        raise EvaluationAuditError(
                            f'gen {generation} {opponent_name} {color}: {key} mismatch')

            metric = metric_index.get((generation, opponent_name))
            if metric is None:
                raise EvaluationAuditError(
                    f'gen {generation} {opponent_name}: missing metrics event')
            for key in ('wins', 'losses', 'draws', 'games'):
                if metric[key] != expected_summary[key]:
                    raise EvaluationAuditError(
                        f'gen {generation} {opponent_name}: metrics {key} mismatch')
            for color in ('black', 'white'):
                if metric[color] != summary[color]:
                    raise EvaluationAuditError(
                        f'gen {generation} {opponent_name}: metrics {color} mismatch')

    return {
        'run_dir': str(run_dir),
        'generations': generations,
        'opponents': checked_opponents,
        'games': checked_games,
        'status': 'PASS',
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Replay Stage 6 evaluation games and verify W/L/D attribution.')
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_run(args.run_dir), indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
