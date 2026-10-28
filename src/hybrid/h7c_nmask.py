"""H7C-NMASK-1 (``VALUE_ONLY_NETWORK_MASK``): one value training that differs from the c8 candidate in one variable.

The contract is docs/h7-c1.md ("VALUE_ONLY_NETWORK_MASK 계약"). It is a separate experiment, not a B2 cycle: B2 is
not resumed, no cycle record is changed, and its verdict changes no champion, canonical engine or program.

    parent      c5 (the c8 verdict's parent); control: the c8 candidate (the c8 verdict's candidate); policy: the frozen
                H3, byte for byte; the value head only (trunk and policy head frozen)
    same as c8  value rows, cache (splits, union masking), replay anchor, config (seed, batch, steps, learning rate,
                D4 augmentation, evaluations, guard): every input is checked against the c8 value record
    the change  every cycle cache row whose position ``analysis.puct_value_v1.exact_leaf`` settles (the search never
                asks the network there) contributes nothing to the loss and to the cycle val metric:
                    L_B = sum_{i in B} k_i w_i (v_i - t_i)^2 / sum_{i in B} w_i,  k_i = 0 masked cycle row, else 1
                (B the mixed batch, w the rows' original weights: the denominator is never renormalized, so a removed
                row's share is not given to the others; a batch whose cycle rows are all masked has a cycle term 0 and
                its replay term as usual). Replay rows are unchanged. Rows are never deleted.

Files: the mask (``MASK_FORMAT``: one bool per cycle cache row and split, ``runs/``) and its manifest (counts, weights,
SHA-256, the row-by-row link to the label rows; committed), the confirmatory sample (``SAMPLE_FORMAT``: the c8
self-play games the E2 sensitivity did not use; committed), and the decision (``DECISION_FORMAT``: everything above
plus the evaluation contract, fixed and committed before any training or evaluation).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = 'H7C-NMASK-1'
RECIPE = 'VALUE_ONLY_NETWORK_MASK'
MASK_FORMAT = 'h7c-nmask-mask-v1'
SAMPLE_FORMAT = 'h7c-nmask-sample-v1'
DECISION_FORMAT = 'h7c-nmask-decision-v1'
SOURCE_CYCLE = 8
# hybrid.h6_labels.RESERVED_SEEDS 8761-8770 (registered 2026-10-27 before any H7C-NMASK-1 run); 8762, 8765-8770 unused
SEEDS = {'arena_official': 8761, 'adjudication_order': 8763, 'adjudication_bootstrap': 8764}
SEED_BLOCK = range(8761, 8771)
MASK_DEFINITION = 'analysis.puct_value_v1.exact_leaf(position) is not None'
LOSS = ('L_B = sum_{i in B} k_i w_i (v_i - t_i)^2 / sum_{i in B} w_i; k_i = 0 for a masked cycle row, else 1; '
        'w the original row weights (no renormalization, no redistribution); replay rows k = 1')
CYCLE_VAL_METRIC = ('sum_{val} k_i w_i (v_i - t_i)^2 / sum_{val} w_i over the effective cycle val set (after the '
                    'union masking); the selection objective is its mean with the replay val WMSE; the H6 guard is '
                    'unchanged (replay val WMSE <= 1.05 x the parent\'s)')
RECORDS = {
    'c8_verdict': 'docs/h7c/c8/arena_verdict.json',
    'c8_evidence': 'docs/h7c/c8/evidence_manifest.json',
    'c8_value_record': 'docs/h7c/c8/value_record.json',
    'c8_cache_manifest': 'docs/h7c/c8/value_cache_manifest.json',
    'b2_program': 'docs/h7c/b2_program.json',
    'd4_c8': 'docs/h7c/e2/d4_c8_vs_c5.json',
    'e2_manifest': 'docs/h7c/e2/evidence_manifest.json',
    'e2_summary': 'docs/h7c/e2/sensitivity_c8.json',
    'check': 'docs/h7c/e2_adjudicate_check/check.json',
    'mask_manifest': 'docs/h7c/nmask/mask_manifest.json',
    'sample': 'docs/h7c/nmask/confirm_sample.json',
    'decision': 'docs/h7c/nmask/decision.json',
}
PATHS = {  # the local files (runs/, data/), checked by SHA-256 against the records
    'value_rows': 'runs/h7c/c8/value_rows.jsonl.gz', 'cycle_cache': 'runs/h7c/c8/value_cache',
    'replay_cache': 'data/tactical/h6/value_cache_v3', 'selfplay': 'runs/h7c/c8/selfplay.jsonl',
    'parent_value': 'runs/h7c/c5/value/best.pt', 'control_value': 'runs/h7c/c8/value/best.pt',
    'policy': 'runs/h3_policy_64x4/best.pt', 'mask': 'runs/h7c/nmask/mask.pt',
}
CONFIRM = {  # the confirmatory evaluation (docs/h7-c1.md), fixed in the decision
    'comparisons': [['mask', 'c8'], ['mask', 'c5'], ['c8', 'c5']], 'primary': 'mask:c8',
    'models': {'mask': 'candidate', 'c8': 'control', 'c5': 'parent'}, 'actor': 'c5',
    'positions': 'every recorded ply of every sample game (the self-play record\'s plies), each model set searching '
                 'the actor\'s position with the actor\'s seeds',
    'observation': 'a ply where the two model sets\' engine moves differ (per comparison; never deduplicated)',
    'bootstrap': {'samples': 10_000, 'seed': SEEDS['adjudication_bootstrap'], 'method': 'percentile',
                  'cluster': 'source game'},
    'report_level': 0.95, 'reading_level': 1 - 0.05 / 3,
    'min_decisive': 20, 'min_decisive_games': 10, 'max_invalid_share': 0.05,
    'order_seed': SEEDS['adjudication_order'],
    'invalid': ['timeout', 'engine_error', 'engine_illegal', 'forbid_mismatch', 'fixed_move_illegal', 'truncated',
                'abnormal_end'],
    'sensitivity': 'abnormal_end scored 0.5, reported apart, never a reading',
    'inconclusive': ['INCOMPLETE', 'INVALID_RATE', 'TOO_FEW_DECISIVE', 'DEGENERATE_BOOTSTRAP',
                     'INTERVAL_CONTAINS_HALF'],
    'stop': 'any input, model, sample or engine identity mismatch stops the run before a branch is played; no '
            'branch, position, budget or repetition is added after a result',
}


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def sha_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read(relative, root=None) -> dict:
    return json.loads((Path(root or ROOT) / relative).read_text(encoding='utf-8'))


def require(ok, message) -> None:
    if not ok:
        raise ValueError(message)


def committed(path, root=None) -> bool:
    """The file is tracked and the same as in HEAD (a pre-registration must be in the history before it is used)."""
    root = Path(root or ROOT)
    if not Path(path).resolve().is_relative_to(root.resolve()):
        return False
    relative = Path(path).resolve().relative_to(root.resolve()).as_posix()
    try:
        subprocess.run(['git', 'ls-files', '--error-unmatch', relative], cwd=root, capture_output=True, check=True)
        subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', relative], cwd=root, capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return False
    return True


# --- the mask ----------------------------------------------------------------------------------------------------

def build_mask(value_rows: Path, cache_dir: Path, progress=None) -> tuple[dict, dict]:
    """One ``keep`` bool per cycle cache row and split (False = ``exact_leaf`` settles the position), linked row by
    row: the label rows are grouped and replayed as the cache builder does, and every cache row's game, ply, value,
    weight, kind and stones must be that label row's. Returns (keep by split, the manifest body)."""
    import torch

    from analysis.puct_value_v1 import exact_leaf
    from renju import Game

    from .h3_cache import tensor_sha256
    from .h6_value_cache import SPLITS, compact_game, game_records, iter_games, load_value_cache

    digest = hashlib.sha256()
    rows = {s: [] for s in SPLITS}
    games = {s: [] for s in SPLITS}
    kinds = {}
    for game_id, label_rows in iter_games(Path(value_rows), digest):
        _, split, moves, labelled, _ = compact_game(label_rows)
        detail = {r['ply']: r['label_kind'] for r in label_rows if not r['value_mask']}
        records = game_records(moves, labelled)
        game, index = Game(), len(games[split])
        games[split].append(game_id)
        for (ply, value, weight, kind), packed in zip(labelled, records):
            while len(game.history) < ply:
                game.play(*moves[len(game.history)])
            exact = exact_leaf(game) is not None
            rows[split].append({'game': index, 'ply': ply, 'value': value, 'weight': weight, 'kind': kind,
                                'packed': packed[:3], 'exact': exact})
            route = 'exact_leaf' if exact else 'network'
            k = kinds.setdefault(detail[ply], {'exact_leaf': [0, 0.0], 'network': [0, 0.0]})
            k[route][0] += 1
            k[route][1] += weight
        if progress is not None:
            progress.update(1)
    manifest = json.loads((Path(cache_dir) / 'manifest.json').read_text(encoding='utf-8'))
    require(digest.hexdigest() == manifest['labels_sha256'],
            f'{value_rows} is not the label file of {cache_dir} ({digest.hexdigest()[:12]})')
    keep, splits = {}, {}
    for split in SPLITS:
        data, _ = load_value_cache(Path(cache_dir), split, verify_hash=True)
        listed = json.loads((Path(cache_dir) / f'{split}_games.json').read_text(encoding='utf-8'))
        require(listed == games[split], f'{split}: the cache game list is not the label rows\' game order')
        mine = rows[split]
        require(data['value'].shape[0] == len(mine), f'{split}: {data["value"].shape[0]} cache rows, '
                                                     f'{len(mine)} labelled rows')
        for i, r in enumerate(mine):
            same = (int(data['game'][i]) == r['game'] and int(data['ply'][i]) == r['ply']
                    and float(data['value'][i]) == float(r['value'])
                    and float(data['weight'][i]) == float(torch.tensor(r['weight'], dtype=torch.float32))
                    and int(data['kind'][i]) == r['kind']
                    and all(bytes(data[f][i].tolist()) == r['packed'][j]
                            for j, f in enumerate(('black', 'white', 'legal'))))
            require(same, f'{split} row {i}: the cache row is not the label row {games[split][r["game"]]} '
                          f'ply {r["ply"]}')
        keep[split] = torch.tensor([not r['exact'] for r in mine], dtype=torch.bool)
        w = data['weight'].double()
        k = keep[split]
        splits[split] = {'rows': len(mine), 'masked_rows': int((~k).sum()), 'kept_rows': int(k.sum()),
                         'weight': float(w.sum()), 'masked_weight': float(w[~k].sum()),
                         'kept_weight': float(w[k].sum()), 'cache_sha256': manifest['splits'][split]['sha256']}
    body = {'format': MASK_FORMAT, 'experiment': EXPERIMENT, 'definition': MASK_DEFINITION,
            'code_sha256': {'analysis/puct_value_v1.py': file_sha256(ROOT / 'src/analysis/puct_value_v1.py')},
            'labels_sha256': manifest['labels_sha256'], 'cache_manifest_sha256': file_sha256(
                Path(cache_dir) / 'manifest.json'),
            'splits': splits, 'keep_sha256': tensor_sha256({s: keep[s].to(torch.uint8) for s in SPLITS}),
            'by_kind': {kind: {route: {'rows': v[0], 'weight': round(v[1], 3)} for route, v in sorted(b.items())}
                        for kind, b in sorted(kinds.items())},
            'link': 'each cache row = its label row (game id order, ply, value, weight, kind, stones, legal mask)'}
    return keep, body


def check_mask_against_d4(body: dict, d4: dict) -> None:
    """The mask's exact_leaf rows and weights by label kind = the committed c8 D4 classification of the same rows."""
    recorded = d4['parts']['d4']['exact_leaf']['by_kind']
    require(body['by_kind'] == recorded, 'the mask does not classify the label rows as the c8 D4 record does')


def save_mask(path: Path, keep: dict, body: dict) -> None:
    import torch

    from .h3_cache import tensor_sha256

    require(tensor_sha256({s: keep[s].to(torch.uint8) for s in keep}) == body['keep_sha256'], 'mask hash')
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'format': MASK_FORMAT, 'keep': keep}, path)


def load_mask(path: Path, body: dict) -> dict:
    """The keep tensors of ``path``, their SHA-256 = the manifest's, one per cache row of each split."""
    import torch

    from .h3_cache import tensor_sha256

    data = torch.load(path, weights_only=True)
    require(data.get('format') == MASK_FORMAT, f'{path}: not a {MASK_FORMAT} file')
    keep = data['keep']
    require(set(keep) == set(body['splits']) and all(keep[s].dtype == torch.bool for s in keep),
            f'{path}: not one bool tensor per split')
    require(tensor_sha256({s: keep[s].to(torch.uint8) for s in keep}) == body['keep_sha256'],
            f'{path} is not the mask its manifest names')
    for split, info in body['splits'].items():
        require(keep[split].shape[0] == info['rows'] and int((~keep[split]).sum()) == info['masked_rows'],
                f'{path}: {split} counts differ from the manifest')
    return keep


# --- the confirmatory sample -------------------------------------------------------------------------------------

def confirm_sample(games: list[dict], e2_summary: dict) -> dict:
    """The c8 self-play games the E2 sensitivity did not use: the E2 sample reproduced (its SHA-256), then the rest
    by game index. Overlap with the E2 sample is checked to be empty."""
    import h7c_labels as lab
    import h7c_sensitivity as sn

    identity = e2_summary['identity']
    e2 = sn.sample_games(games, identity['games_sample'])
    e2_ids = [lab.game_id(g) for g in e2]
    require(hashlib.sha256(json.dumps(e2_ids).encode()).hexdigest() == identity['sample_sha256'],
            'the self-play file does not give the E2 sample')
    used = set(e2_ids)
    rest = sorted((g for g in games if lab.game_id(g) not in used), key=lambda g: g['game_index'])
    ids = [lab.game_id(g) for g in rest]
    require(len(set(ids)) == len(ids) and not set(ids) & used and len(ids) + len(used) == len(games),
            'the confirmatory sample overlaps the E2 sample or repeats a game')
    return {'format': SAMPLE_FORMAT, 'experiment': EXPERIMENT, 'selfplay_jsonl_sha256': identity['selfplay_jsonl_sha256'],
            'selfplay_games': len(games), 'e2_sample': {'games': len(e2_ids), 'sha256': identity['sample_sha256']},
            'rule': 'every game of the c8 self-play not in the E2 sensitivity sample, by game index',
            'games': len(ids), 'game_ids': ids, 'game_indices': [g['game_index'] for g in rest],
            'plies': sum(len(g['plies']) for g in rest),
            'ids_sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            'use': 'confirmatory only: the E2 200 stay development / exploration'}


def check_sample(sample: dict) -> None:
    require(sample.get('format') == SAMPLE_FORMAT and sample.get('experiment') == EXPERIMENT, 'not the sample file')
    ids = sample['game_ids']
    require(hashlib.sha256(json.dumps(ids).encode()).hexdigest() == sample['ids_sha256']
            and len(ids) == len(set(ids)) == sample['games'], 'the sample game list is not its SHA-256')


# --- the decision ------------------------------------------------------------------------------------------------

def expected_decision(root=None) -> dict:
    """The decision as the committed records give it (parent, control, inputs, mask, sample, check, evaluation)."""
    from analysis import tb_v3

    from .h7c_recipe import MODEL_KEYS

    root = Path(root or ROOT)
    shas = {name: file_sha256(root / rel) for name, rel in RECORDS.items() if name != 'decision'}
    verdict, evidence, record = (read(RECORDS[k], root) for k in ('c8_verdict', 'c8_evidence', 'c8_value_record'))
    program, check, mask, sample = (read(RECORDS[k], root) for k in ('b2_program', 'check', 'mask_manifest', 'sample'))
    require(check.get('format') == 'h7c-adjudicate-check-v1' and check.get('verdict') == 'PASS',
            f"{RECORDS['check']}: the read-only check did not pass ({check.get('failed')})")
    require(verdict.get('format') == 'h7c-arena-v1-verdict' and verdict.get('cycle') == SOURCE_CYCLE
            and verdict['promotion']['decision'] == 'KEEP_CHAMPION', 'not the c8 verdict')
    require(evidence.get('verdict_sha256') == shas['c8_verdict'], 'the c8 evidence does not name the c8 verdict')
    require(set(verdict['parent']) == set(MODEL_KEYS) and verdict['candidate']['value_checkpoint_sha256']
            == record['candidate']['checkpoint_sha256'], 'the c8 value record is not the verdict\'s candidate')
    require(all(verdict['parent'][k] == tb_v3.MODELS[k] for k in MODEL_KEYS[:2]), 'the parent policy is not H3')
    files = evidence['files_sha256']
    inputs = record['inputs']
    require(mask.get('format') == MASK_FORMAT and mask['labels_sha256'] == inputs['cycle_labels_sha256']
            and {s: v['cache_sha256'] for s, v in mask['splits'].items()} == inputs['cycle_cache_splits'],
            'the mask manifest is not of the c8 cycle cache')
    check_mask_against_d4(mask, read(RECORDS['d4_c8'], root))
    check_sample(sample)
    require(sample['selfplay_jsonl_sha256'] == files[PATHS['selfplay']], 'the sample is not of the c8 self-play')
    return {
        'format': DECISION_FORMAT, 'experiment': EXPERIMENT, 'recipe': RECIPE,
        'parent': verdict['parent'], 'control': verdict['candidate'],
        'policy': {'action': 'freeze_parent', 'training': False, 'checkpoint_sha256':
                   verdict['parent']['policy_checkpoint_sha256']},
        'source': {'cycle': SOURCE_CYCLE, 'selfplay_run_seed': inputs['selfplay_run_seed'],
                   'value_rows_sha256': files[PATHS['value_rows']], 'cycle_labels_sha256': inputs['cycle_labels_sha256'],
                   'cycle_cache_splits': inputs['cycle_cache_splits'],
                   'replay_labels_sha256': inputs['replay_labels_sha256'],
                   'replay_cache_splits': inputs['replay_cache_splits'],
                   'selfplay_jsonl_sha256': files[PATHS['selfplay']]},
        'value': {'config': program['value']['config'], 'config_sha256': inputs['config_sha256'],
                  'settings': record['config'], 'replay_anchor_labels_sha256': inputs['replay_labels_sha256'],
                  'steps': {k: record[k] for k in ('total_steps', 'steps_per_epoch', 'warmup_steps',
                                                    'train_rows_per_epoch')},
                  'frozen_state_sha256': record['frozen_state_sha256'],
                  'control_parent_val': {k: record['parent_val'][k] for k in ('cycle_wmse', 'replay_wmse')},
                  'control_best': {k: record['best'][k] for k in ('step', 'cycle_wmse', 'replay_wmse')}},
        'mask': {'definition': MASK_DEFINITION, 'manifest_sha256': shas['mask_manifest'],
                 'keep_sha256': mask['keep_sha256'], 'applies_to': 'cycle rows of every split; replay rows unchanged',
                 'masked_rows': {s: v['masked_rows'] for s, v in mask['splits'].items()},
                 'masked_weight': {s: v['masked_weight'] for s, v in mask['splits'].items()}},
        'loss': LOSS, 'cycle_val_metric': CYCLE_VAL_METRIC,
        'zero_cycle_weight_batch': 'cycle term 0, replay term as usual, counted in the record',
        'fail_closed': ['a non-finite loss or objective', 'no cycle train row with weight', 'a mask of another cache '
                        'or with other counts', 'any input other than the decision\'s'],
        'records_sha256': shas,
        'evaluation': {
            'official': {'opponent': 'parent (c5)', 'pairs': 50, 'seed': SEEDS['arena_official'],
                         'promotion': {'score_above': 0.5, 'improvement_lower_min': -0.05,
                                       'tactical_rule': 'h7c-tactical-v2'},
                         'meaning': 'the existing rule; a PROMOTE changes no champion, program or canonical engine'},
            'confirm': {**CONFIRM, 'sample_sha256': shas['sample'], 'sample_ids_sha256': sample['ids_sha256'],
                        'games': sample['games'],
                        'engine': {'registered': program['external'], 'rule': 4, 'move_timeout': 600.0}},
        },
        'seeds': {**SEEDS, 'reserved_block': [SEED_BLOCK.start, SEED_BLOCK.stop - 1],
                  'training': record['config']['seed']},
        'out_of_scope': ['V9', 'B2 c9', 'head / trunk growth', 'D4 checkpoint selection', 'more self-play',
                         'S400 retarget', 'reusing the E2 200 for selection'],
        'approved_by': 'user, 2026-10-27 (docs/h7-c1.md)',
    }


def load_decision(path=None, root=None, require_committed: bool = True) -> tuple[dict, str]:
    """The decision file = ``expected_decision`` of the committed records, and committed itself."""
    root = Path(root or ROOT)
    path = Path(path or root / RECORDS['decision'])
    decision = json.loads(path.read_text(encoding='utf-8'))
    require(decision == expected_decision(root), f'{path} is not the decision the committed records give')
    if require_committed:
        require(committed(path, root) and all(committed(root / RECORDS[k], root) for k in
                                              ('check', 'mask_manifest', 'sample')),
                f'{path} and its records must be committed before they are used')
    return decision, file_sha256(path)
