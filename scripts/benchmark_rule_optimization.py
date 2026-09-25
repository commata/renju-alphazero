"""Alternating perf_counter benchmarks and complete seeded V6 oracle smoke games."""
import argparse
from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests'))
from agents import MCTSV6Agent, RandomAgent
from evaluation import play_game
from renju import rules
import reference_rules
from rule_validation import fixtures
from benchmark_engine import prepare_midgame


@contextmanager
def implementation(reference):
    """Replace already-imported rule aliases for a complete reference call graph.

    Test/benchmark only, sequential, restored even after an exception. This also
    covers search modules that imported private rule helpers directly.
    """
    replacements = {id(value): getattr(reference_rules, name)
                    for name, value in vars(rules).items()
                    if callable(value) and hasattr(reference_rules, name)}
    saved = []
    if reference:
        for module in list(sys.modules.values()):
            name = getattr(module, '__name__', '')
            if name == 'renju.rules' or name.startswith(('renju.', 'search.', 'agents.')):
                for key, value in list(vars(module).items()):
                    if id(value) in replacements:
                        saved.append((module, key, value))
                        setattr(module, key, replacements[id(value)])
    try:
        yield
    finally:
        for module, key, value in reversed(saved):
            setattr(module, key, value)


def summarize(samples):
    return dict(mean_ms=statistics.mean(samples)*1000,
                median_ms=statistics.median(samples)*1000,
                min_ms=min(samples)*1000, max_ms=max(samples)*1000,
                ops_sec=1/statistics.mean(samples))


def smoke(seed, fixture_path=None):
    results = []
    for index in range(2):
        histories = []
        for reference in (True, False):
            with implementation(reference):
                v6 = MCTSV6Agent(seed=seed+index, simulations=1, tactical_simulations=1)
                random = RandomAgent(seed+100+index)
                result = play_game(v6, random) if index == 0 else play_game(random, v6)
            histories.append((result.winner, result.history))
        if histories[0] != histories[1]:
            raise AssertionError('V6 winner/history mismatch')
        digest = sha256(json.dumps(histories[0]).encode()).hexdigest()
        results.append(dict(seed=seed+index, v6_color='black' if index == 0 else 'white',
                            winner=histories[0][0], moves=len(histories[0][1]), sha256=digest, history=histories[0][1]))
        print(json.dumps({k: v for k, v in results[-1].items() if k != 'history'}), flush=True)
    if fixture_path:
        Path(fixture_path).write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--games', type=int, default=2)
    parser.add_argument('--smoke', action='store_true', help='compare two complete V6 vs Random games, both colors, 1/1 simulations')
    parser.add_argument('--write-smoke-fixture', help='save verified baseline histories for corpus replay')
    args = parser.parse_args()
    if min(args.iterations, args.repeats, args.games) <= 0:
        parser.error('iterations, repeats and games must be positive')
    if args.smoke:
        smoke(args.seed, args.write_smoke_fixture)
        return
    boards = [prepare_midgame(args.seed+i)[0] for i in range(4)]
    probes = [(game.board, next((r,c) for r in range(15) for c in range(15) if game.board[r][c] == 0))
              for game in boards]
    probes += [(game.board, (7,7)) for _, game in fixtures() if game.board[7][7] == 0]
    samples = {key: {'reference': [], 'optimized': []}
               for key in ('black_legal_moves', 'forbidden_reason', 'random_game')}
    moves = {'reference': 0, 'optimized': 0}
    durations = {'reference': 0., 'optimized': 0.}
    for repeat in range(args.repeats):
        for ref in (True, False):
            name = 'reference' if ref else 'optimized'
            with implementation(ref):
                for game in boards:
                    game.legal_moves()  # warm-up
                for board, point in probes:
                    rules.forbidden_reason(board, *point)
                start = perf_counter()
                for _ in range(args.iterations):
                    for game in boards:
                        game.legal_moves()
                samples['black_legal_moves'][name].append((perf_counter()-start)/(args.iterations*len(boards)))
                start = perf_counter()
                for _ in range(args.iterations):
                    for board, point in probes:
                        rules.forbidden_reason(board, *point)
                samples['forbidden_reason'][name].append((perf_counter()-start)/(args.iterations*len(probes)))
                play_game(RandomAgent(args.seed), RandomAgent(args.seed+1))  # warm-up
                for i in range(args.games):
                    start = perf_counter()
                    result = play_game(RandomAgent(args.seed+i*2), RandomAgent(args.seed+i*2+1))
                    elapsed = perf_counter()-start
                    samples['random_game'][name].append(elapsed)
                    moves[name] += len(result.history)
                    durations[name] += elapsed
        print(f'repeat {repeat+1}/{args.repeats} complete', flush=True)
    report = {key: {name: summarize(values) for name, values in groups.items()}
              for key, groups in samples.items()}
    report['random_moves_sec'] = {name: moves[name]/durations[name] for name in moves}
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
