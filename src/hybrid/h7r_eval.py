"""H7-R R1 evaluation: contracts, pair statistics, the anchor difference, sizing and the promotion rule v3
(docs/h7-rl-plan.md, R1). Measurement code only: nothing here plays, trains or changes a champion.

Contract (``CONTRACT_FORMAT``, one JSON file under ``docs/h7r/contracts/``; seeds from ``SEED_BLOCK``, reserved in
``hybrid.h6_labels.RESERVED_SEEDS``, each used by one contract only):
    purpose   ``pilot`` (measure the suites and model pairs; never a verdict), ``confirm`` (one promotion judgment;
              played only once ``status`` is ``fixed`` and every rule value is set) or ``holdout`` (a self-play
              sample for evaluation only)
    models    name -> policy / value paths and their four SHA-256
    arms      tested and opponent (a model and a budget factor: 1 = TB-V3 S50, 8 = S400), pairs, seed, suite file
              and SHA-256; arms with the same ``pairing`` play the same openings (needed for the anchor difference)
    stats     one-sided alpha, the pair bootstrap (resampling unit: the opening pair, percentile, seed, repetitions),
              the error budget over cycles, the sizing target (delta, power)
    confirm   ``informative`` (which pairs count towards m: ``move_split`` or ``score_split``, and m), ``anchor``
              (model and the non-inferiority margin), ``selection`` (margin, alpha), ``gates`` (time ratios, tactical
              rule, forbidden errors)

Pairs: the two games of one opening with colours swapped. Every valid pair counts in the score and every effect;
``move_split`` (the two games' move sequences differ) and ``score_split`` (pair score != 0.5) are counted apart and
only used for the minimum-information rule. A truncated game, a missing colour, a repeated or unknown opening is
refused; an opponent's illegal move makes the arm invalid.
"""
from __future__ import annotations

import hashlib
import json
import re
import math
from math import ceil
from pathlib import Path
from random import Random
from statistics import NormalDist, fmean, pstdev

CONTRACT_FORMAT = 'h7r-eval-contract-v1'
SEED_BLOCK = range(8781, 8800)  # hybrid.h6_labels.RESERVED_SEEDS, registered 2026-10-28 for H7-R evaluation
PURPOSES = ('pilot', 'confirm', 'holdout')
BUDGETS = (1, 2, 4, 8)
SCORE = {'win': 1.0, 'draw': 0.5, 'loss': 0.0}
ROLE_ARMS = ('candidate_vs_champion', 'candidate_vs_anchor', 'champion_vs_anchor')
MODEL_KEYS = ('policy_checkpoint_sha256', 'policy_metadata_sha256', 'value_checkpoint_sha256', 'value_metadata_sha256')


def check_model_identity(identity):
    require(isinstance(identity, dict) and set(identity) == set(MODEL_KEYS)
            and all(isinstance(v, str) and re.fullmatch(r'[0-9a-f]{64}', v) for v in identity.values()),
            'model identity must contain four SHA-256 values')


CONFIRM_KEYS = {'informative': ('kind', 'm'), 'anchor': ('model', 'margin'),
                'selection': ('margin', 'alpha', 'min_observations', 'holdout_contract_sha256', 'engine'),
                'gates': ('time', 'tactical_rule', 'forbidden_errors')}


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message) -> None:
    if not ok:
        raise ValueError(message)


# --- contracts ---------------------------------------------------------------------------------------------------

def model_identity(policy, value) -> dict | None:
    """The four SHA-256 of a model pair, as the candidate factory's ``model_identity`` keys them (computed here: the
    factory is used only by the ``scripts/h7c_*`` tools); None when a file is missing."""
    files = {'policy_checkpoint_sha256': Path(policy), 'policy_metadata_sha256': Path(policy).with_suffix('.json'),
             'value_checkpoint_sha256': Path(value), 'value_metadata_sha256': Path(value).with_name('best.json')}
    if not all(f.is_file() for f in files.values()):
        return None
    return {k: file_sha256(f) for k, f in files.items()}


def unset(contract: dict) -> list[str]:
    """The confirm rule values still missing (a confirm contract is played and judged only without any)."""
    out = [] if contract.get('status') == 'fixed' else ['status']
    for block, keys in CONFIRM_KEYS.items():
        values = contract.get(block) or {}
        out += [f'{block}.{k}' for k in keys if values.get(k) is None]
    time = (contract.get('gates') or {}).get('time') or {}
    out += [f'gates.time.{k}' for k in ('mean_ratio_max', 'p95_ratio_max') if time.get(k) is None]
    return out


def check_contract(contract: dict, path, root, check_files: bool = True) -> None:
    from hybrid.h6_labels import RESERVED_SEEDS

    root = Path(root)
    require(contract.get('format') == CONTRACT_FORMAT, f'{path}: not an {CONTRACT_FORMAT} contract')
    require(isinstance(contract.get('contract_id'), str) and contract['contract_id'], f'{path}: no contract id')
    require(contract.get('purpose') in PURPOSES and contract.get('status') in ('draft', 'fixed'),
            f'{path}: purpose / status')
    models = contract.get('models') or {}
    for name, m in models.items():
        require(set(m) >= {'policy', 'value', 'identity'}, f'{path}: model {name} needs policy, value, identity')
        check_model_identity(m['identity'])
        if check_files:
            actual = model_identity(root / m['policy'], root / m['value'])
            require(actual is not None, f'{path}: model {name} identity files are missing')
            require(actual == m['identity'],
                    f'{path}: model {name} files are missing or not its recorded identity')
    seeds = []
    names = set()
    for arm in contract.get('arms', []):
        require(arm.get('name') not in names, f"{path}: arm {arm.get('name')} twice")
        names.add(arm['name'])
        for side in ('tested', 'opponent'):
            require(arm[side]['model'] in models and arm[side]['budget'] in BUDGETS,
                    f"{path}: arm {arm['name']} {side} model / budget")
        require(isinstance(arm.get('pairs'), int) and arm['pairs'] > 0, f"{path}: arm {arm['name']} pairs")
        require(file_sha256(root / arm['suite']['path']) == arm['suite']['sha256'],
                f"{path}: arm {arm['name']} suite file is not its SHA-256")
        seeds.append(arm['seed'])
    if contract['purpose'] == 'holdout':
        h = contract.get('holdout') or {}
        require(h.get('split') in ('dev', 'conf') and isinstance(h.get('games'), int) and h['games'] > 0
                and h.get('actor') in models and isinstance(h.get('cycle'), int), f'{path}: holdout block')
        seeds.append(h['seed'])
    for seed in seeds:
        require(seed in SEED_BLOCK and seed in RESERVED_SEEDS, f'{path}: seed {seed} is not an H7-R reserved seed')
    require(len(set(seeds)) == len(seeds) or _paired_seeds(contract), f'{path}: a seed used twice outside a pairing')
    stats = contract.get('stats') or {}
    b = stats.get('bootstrap') or {}
    require(0 < stats.get('alpha_one_sided', 0) < 0.5 and b.get('method') == 'percentile'
            and b.get('unit') == 'opening pair' and isinstance(b.get('samples'), int) and b['samples'] >= 1000
            and isinstance(b.get('seed'), int), f'{path}: stats')
    if contract['purpose'] == 'confirm':
        roles = contract.get('roles') or {}
        require(set(roles) == {'candidate', 'champion', 'anchor'} and all(r in models for r in roles.values()),
                f'{path}: roles')
        require(names >= set(ROLE_ARMS), f'{path}: a confirm contract plays {ROLE_ARMS}')
        informative = contract.get('informative') or {}
        require(informative.get('kind') in (None, 'move_split', 'score_split'), 'invalid informative kind')
        if informative.get('m') is not None:
            require(type(informative['m']) is int and informative['m'] > 0, 'invalid minimum information count')
        for name in ('selection', 'anchor'):
            block = contract.get(name) or {}
            margin = block.get('margin')
            if margin is not None:
                require(type(margin) in (float, int) and math.isfinite(margin) and -1 <= margin <= 0,
                        f'{name}: invalid non-inferiority margin')
        selection = contract.get('selection') or {}
        if selection.get('alpha') is not None:
            require(0 < selection['alpha'] <= stats['alpha_one_sided'], 'selection alpha exceeds error budget')
        if selection.get('min_observations') is not None:
            require(type(selection['min_observations']) is int and selection['min_observations'] > 0,
                    'invalid selection minimum observations')
        for limit in ((contract.get('gates') or {}).get('time') or {}).values():
            if limit is not None:
                require(type(limit) in (int, float) and math.isfinite(limit) and limit > 0, 'invalid time limit')


def _paired_seeds(contract: dict) -> bool:
    """Arms of one pairing share their seed (the same openings); every other seed is distinct."""
    by_seed = {}
    for arm in contract['arms']:
        by_seed.setdefault(arm['seed'], set()).add(arm.get('pairing'))
    return all(len(p) == 1 and None not in p for s, p in by_seed.items()
               if sum(a['seed'] == s for a in contract['arms']) > 1)


def contract_seeds(contract: dict) -> set:
    out = {a['seed'] for a in contract.get('arms', [])}
    if contract.get('purpose') == 'holdout':
        out.add(contract['holdout']['seed'])
    return out


def load_contract(path, root, check_files: bool = True, contracts_dir=None) -> tuple[dict, str]:
    """A valid contract and its SHA-256; its seeds used by no other contract of ``contracts_dir``."""
    path = Path(path)
    contract = json.loads(path.read_text(encoding='utf-8'))
    check_contract(contract, path, root, check_files)
    folders = {Path(root) / 'docs/h7r/contracts', path.parent}
    if contracts_dir is not None:
        folders.add(Path(contracts_dir))
    others = {p.resolve() for folder in folders if folder.exists() for p in folder.glob('*.json')}
    mine = contract_seeds(contract)
    for other in sorted(others):
        if other.resolve() == path.resolve():
            continue
        theirs = json.loads(other.read_text(encoding='utf-8'))
        require(theirs.get('contract_id') != contract['contract_id'], f'{other}: the same contract id')
        shared = mine & contract_seeds(theirs)
        require(not shared, f'{path}: seeds {sorted(shared)} are already used by {other.name}')
    return contract, file_sha256(path)


# --- pairs -------------------------------------------------------------------------------------------------------

def validate_game(record: dict, opening: list) -> None:
    """Replay without an engine; never score a terminal flag without checking the stones."""
    from renju import Game, IllegalMove, BLACK, WHITE
    moves = record.get('moves')
    require(isinstance(moves, list) and moves[:len(opening)] == opening,
            'game does not start with its contracted opening')
    require(record.get('tested_color') in ('black', 'white'), 'unknown tested colour')
    require(record.get('length') == len(moves), 'game length differs from its moves')
    board = Game()
    try:
        for move in moves:
            require(isinstance(move, (list, tuple)) and len(move) == 2
                    and all(type(x) is int and 0 <= x < 15 for x in move), 'invalid move coordinate')
            board.play(*move)
    except IllegalMove as exc:
        raise ValueError(f'illegal recorded move: {exc}') from exc
    illegal = record.get('illegal')
    if illegal is not None:
        require(not board.done and illegal.get('ply') == len(moves), 'illegal event at wrong position')
        tested = BLACK if record['tested_color'] == 'black' else WHITE
        require(illegal.get('side') == ('tested' if board.to_play == tested else 'opponent'),
                'illegal event has wrong side')
        move = illegal.get('move')
        require(isinstance(move, list) and len(move) == 2 and all(type(x) is int for x in move),
                'illegal event has no attempted move')
        try:
            board.play(*move)
        except (IllegalMove, IndexError):
            pass
        else:
            raise ValueError('recorded illegal move is legal')
        expected = 'loss' if illegal['side'] == 'tested' else 'opponent_illegal'
    else:
        require(board.done, 'game is not terminal (truncated or missing moves)')
        tested = BLACK if record['tested_color'] == 'black' else WHITE
        expected = 'draw' if board.winner is None else 'win' if board.winner == tested else 'loss'
    require(record.get('result') == expected and record.get('winner') == board.winner,
            'recorded result/winner differs from replay')


def pair_table(games: list[dict], openings: list[dict]) -> dict:
    """The colour-swapped pairs of one arm: every opening played as black and as white by the tested side.
    Refused: a truncated game, a missing colour, a game twice, an opening outside the arm. ``valid`` is False when
    the opponent made an illegal move (no score is given to the tested side for it)."""
    wanted = {o['history_id']: o for o in openings}
    by = {}
    for g in games:
        require(g.get('opening_id') in wanted, f"{g.get('key')}: an opening outside this arm")
        require(not g.get('truncated') and g['result'] != 'truncated', f"{g['key']}: a truncated game")
        validate_game(g, wanted[g['opening_id']]['moves'])
        key = (g['opening_id'], g['tested_color'])
        require(key not in by, f"{g['key']}: the same opening and colour twice")
        by[key] = g
    rows, invalid = [], []
    for o in openings:
        black, white = by.get((o['history_id'], 'black')), by.get((o['history_id'], 'white'))
        require(black is not None and white is not None, f"opening {o['history_id'][:12]}: a colour is missing")
        if 'opponent_illegal' in (black['result'], white['result']):
            invalid.append(o['history_id'])
            continue
        score = (SCORE[black['result']] + SCORE[white['result']]) / 2
        rows.append({'opening_id': o['history_id'], 'class_id': o['class_id'], 'black': black['result'],
                     'white': white['result'], 'pair_score': score,
                     'move_split': black['moves'] != white['moves'], 'score_split': score != 0.5})
    return {'pairs': rows, 'opponent_illegal_pairs': invalid, 'valid': not invalid,
            'tested_illegal': sum(bool(g.get('illegal')) and g['illegal'].get('side') == 'tested' for g in games)}


def _resampled_means(values: list[float], bootstrap: dict) -> list[float]:
    rng, n = Random(bootstrap['seed']), len(values)
    return sorted(fmean(values[rng.randrange(n)] for _ in range(n)) for _ in range(bootstrap['samples']))


def _q(sorted_values: list[float], q: float) -> float:
    return round(sorted_values[min(len(sorted_values) - 1, max(0, int(q * (len(sorted_values) - 1))))], 5)


def arm_summary(table: dict, stats: dict) -> dict:
    """Score over every valid pair, its pair bootstrap (two-sided 95% and the one-sided lower bound at alpha), the
    two information counts and the colour split."""
    pairs = table['pairs']
    scores = [p['pair_score'] for p in pairs]
    out = {'pairs': len(pairs), 'valid': table['valid'], 'opponent_illegal_pairs': table['opponent_illegal_pairs'],
           'tested_illegal': table['tested_illegal'],
           'move_split': sum(p['move_split'] for p in pairs), 'score_split': sum(p['score_split'] for p in pairs),
           'score_split_positive': sum(p['pair_score'] > 0.5 for p in pairs),
           'score_split_negative': sum(p['pair_score'] < 0.5 for p in pairs),
           'distinct_classes': len({p['class_id'] for p in pairs})}
    if not scores:
        return {**out, 'score': None, 'improvement': None, 'ci95': None, 'lower_one_sided': None, 'pair_sd': None}
    means = _resampled_means(scores, stats['bootstrap'])
    alpha = stats['alpha_one_sided']
    by = {c: [SCORE[p[c]] for p in pairs] for c in ('black', 'white')}
    return {**out, 'score': round(fmean(scores), 5), 'improvement': round(fmean(scores) - 0.5, 5),
            'ci95': [round(_q(means, 0.025) - 0.5, 5), round(_q(means, 0.975) - 0.5, 5)],
            'lower_one_sided': round(_q(means, alpha) - 0.5, 5), 'alpha_one_sided': alpha,
            'pair_sd': round(pstdev(scores), 5),
            'score_by_tested_colour': {c: round(fmean(v), 5) for c, v in by.items()},
            'bootstrap': stats['bootstrap']}


def anchor_difference(table_a: dict, table_b: dict, stats: dict, margin: float | None) -> dict:
    """Candidate vs anchor minus champion vs anchor, pair by pair on the same openings (A - B); non-inferior when
    the one-sided lower bound of the mean difference is at least ``margin`` (a non-positive number)."""
    a = {p['opening_id']: p['pair_score'] for p in table_a['pairs']}
    b = {p['opening_id']: p['pair_score'] for p in table_b['pairs']}
    require(set(a) == set(b) and a, 'the anchor arms were not played on the same openings')
    d = [a[k] - b[k] for k in sorted(a)]
    means = _resampled_means(d, stats['bootstrap'])
    alpha = stats['alpha_one_sided']
    lower, upper = _q(means, alpha), _q(means, 1 - alpha)
    status = None if margin is None else 'NONINFERIOR' if lower >= margin else 'INFERIOR' if upper < margin \
        else 'INCONCLUSIVE'
    return {'pairs': len(d), 'mean_difference': round(fmean(d), 5), 'lower_one_sided': lower,
            'upper_one_sided': upper, 'margin': margin, 'status': status or 'UNSET',
            'valid': table_a['valid'] and table_b['valid']}


def cluster_noninferiority(rows: list[dict], margin: float | None, alpha: float, bootstrap: dict,
                           min_observations: int) -> dict:
    """Selection quality: rows ``{'game', 'delta'}`` (candidate minus reference per observation); the mean delta's
    bootstrap over source games (observations of one game move together). INCONCLUSIVE is never a pass."""
    groups = {}
    for r in rows:
        groups.setdefault(r['game'], []).append(r['delta'])
    n = len(rows)
    if margin is None:
        return {'observations': n, 'status': 'UNSET'}
    if n < min_observations or len(groups) < 2:
        return {'observations': n, 'games': len(groups), 'status': 'INCONCLUSIVE', 'reason': 'TOO_FEW'}
    ids = sorted(groups)
    rng = Random(bootstrap['seed'])
    means = []
    for _ in range(bootstrap['samples']):
        draw = [x for _ in ids for x in groups[ids[rng.randrange(len(ids))]]]
        means.append(fmean(draw))
    means.sort()
    lower, upper = _q(means, alpha), _q(means, 1 - alpha)
    status = 'NONINFERIOR' if lower >= margin else 'INFERIOR' if upper < margin else 'INCONCLUSIVE'
    return {'observations': n, 'games': len(groups), 'mean_delta': round(fmean(x['delta'] for x in rows), 5),
            'lower_one_sided': lower, 'upper_one_sided': upper, 'margin': margin, 'status': status}


# --- sizing ------------------------------------------------------------------------------------------------------

def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    centre, half = p + z * z / (2 * n), z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5
    return [round(max(0.0, (centre - half) / (1 + z * z / n)), 5), round(min(1.0, (centre + half) / (1 + z * z / n)), 5)]


def binomial_at_least(m: int, n: int, p: float) -> float:
    """P(X >= m) for X ~ Binomial(n, p)."""
    from math import comb
    return round(1 - sum(comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(min(m, n + 1))), 5)


def sizing(summary_pairs: list[dict], stats: dict, delta: float, power: float, m_candidates: list[int],
           seconds_per_pair: float | None) -> dict:
    """A proposal (never a start): pairs for ``delta`` at one-sided alpha and ``power`` with the pair SD at its
    upper bootstrap bound, the chance of reaching m informative pairs (at the lower bound of each rate), the cost."""
    scores = [p['pair_score'] for p in summary_pairs]
    n0 = len(scores)
    require(n0 >= 2, 'sizing needs a pilot arm with pairs')
    rng = Random(stats['bootstrap']['seed'])
    sds = sorted(pstdev([scores[rng.randrange(n0)] for _ in range(n0)]) for _ in range(stats['bootstrap']['samples']))
    sd_upper = max(_q(sds, 0.975), 1e-6)
    z = NormalDist().inv_cdf(1 - stats['alpha_one_sided']) + NormalDist().inv_cdf(power)
    need = ceil((z * sd_upper / delta) ** 2)
    rates = {k: {'count': sum(p[k] for p in summary_pairs), 'rate_ci95': wilson(sum(p[k] for p in summary_pairs), n0)}
             for k in ('move_split', 'score_split')}
    chance = {k: {str(m): binomial_at_least(m, need, rates[k]['rate_ci95'][0]) for m in m_candidates}
              for k in rates}
    return {'pilot_pairs': n0, 'pair_sd': round(pstdev(scores), 5), 'pair_sd_upper': round(sd_upper, 5),
            'delta': delta, 'power': power, 'alpha_one_sided': stats['alpha_one_sided'], 'pairs_needed': need,
            'informative_rates': rates, 'chance_of_m_at_rate_lower_bound': chance,
            'seconds_per_pair': seconds_per_pair,
            'estimated_seconds': None if seconds_per_pair is None else round(need * seconds_per_pair),
            'note': 'a planning proposal: it starts nothing and is not a guarantee of the effect size'}


# --- the promotion rule v3 ---------------------------------------------------------------------------------------

def judge(contract: dict, arms: dict, selection: dict | None, tactical: dict | None, timing: dict | None) -> dict:
    """PROMOTE only when every rule value is set and every item passes; anything else keeps the champion.
    ``arms``: role -> {'table', 'summary'} for ``ROLE_ARMS``."""
    reasons = []
    missing = unset(contract)
    if missing:
        reasons.append('CONFIG_UNSET')
    summary = arms['candidate_vs_champion']['summary']
    informative = contract.get('informative') or {}
    if not all(arms[r]['summary']['valid'] for r in ROLE_ARMS):
        reasons.append('ARM_INVALID')
    kind, m = informative.get('kind'), informative.get('m')
    if kind is not None and m is not None and summary[kind] < m:
        reasons.append('TOO_FEW_INFORMATIVE')
    if summary['lower_one_sided'] is None or not summary['lower_one_sided'] > 0:
        reasons.append('NOT_SUPERIOR')
    anchor = anchor_difference(arms['candidate_vs_anchor']['table'], arms['champion_vs_anchor']['table'],
                               contract['stats'], (contract.get('anchor') or {}).get('margin'))
    if anchor['status'] != 'NONINFERIOR':
        reasons.append(f"ANCHOR_{anchor['status']}")
    bound = lambda r: isinstance(r, dict) and r.get('evidence_verified') is True and r.get('models') == {
        role: contract.get('models', {}).get(name, {}).get('identity') for role, name in contract.get('roles', {}).items()}
    sel = (selection or {}).get('status', 'MISSING') if bound(selection) else 'MISSING'
    if sel != 'NONINFERIOR':
        reasons.append(f'SELECTION_{sel}')
    gates = contract.get('gates') or {}
    gate = {'tactical': bound(tactical) and (tactical or {}).get('verdict') == 'PASS',
            'forbidden': summary['tested_illegal'] == 0 and all(arms[r]['summary']['tested_illegal'] == 0
                                                                 for r in ROLE_ARMS),
            'time': bound(timing) and (timing or {}).get('gate', {}).get('pass') is True
            and (timing or {}).get('gate', {}).get('limits') == gates.get('time')}
    reasons += [f'GATE_{k.upper()}' for k, ok in gate.items() if not ok]
    return {'decision': 'KEEP_CHAMPION' if reasons else 'PROMOTE', 'reasons': reasons, 'unset': missing,
            'candidate_vs_champion': {k: summary[k] for k in ('pairs', 'score', 'improvement', 'ci95',
                                                               'lower_one_sided', 'move_split', 'score_split')},
            'anchor': anchor, 'selection': selection, 'gates': gate,
            'rule': 'promotion v3 (docs/h7-rl-plan.md): one-sided lower > 0, m informative pairs, anchor difference '
                    'non-inferior, selection non-inferior, tactical / forbidden / time gates; any unset value, '
                    'missing record or INCONCLUSIVE keeps the champion'}
