"""R3-POLICY-LOOP-1 (``scripts/h7r_r3_loop.py``): the contract; the decision table; the S400 rows rule (train and val,
S50 kept without S400 mass, counted per split); the pairing of two cycles by opening id; the training path (initial
model and teacher from their own files); and on a temporary repository with the R2-S400-TRAIN fixture: prepare, the
gate, then cycle 1 end to end (self-play -> S50 reproduction -> S400 targets -> rows loaded as training tensors ->
training -> Arena -> judge) and cycle 2 started from the cycle-1 candidate only (self-play, search prior and training
initialisation name its SHA-256); the refusals of the chain; the desktop blocks under real bash.

The engine (self-play moves, searches, Arena games) and the trainer are stand-ins; the game records, label step, row
rule, tensors, checks, statistics and records are the tools'."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import h7c_labels as lab
import h7c_retarget as rt
import h7c_selfplay as sp
import h7r_eval as cli
import h7r_p1 as p1
import h7r_r3_loop as r3
import h7r_s400_train as st
from hybrid import h7c_retarget as hr
from hybrid import h7c_targets as targets
from hybrid.h6_labels import split_for
from renju import BLACK
from tests.test_h7c_labels import MOVES, PLAN
from tests.test_h7r_preserve import PreserveBase
from tests.test_h7r_s400_train import QUIET, git

ROOT = Path(__file__).resolve().parents[1]
GAMES = 6
SMALL_SEARCH = {**r3.SEARCH, 'reproduction_games': 2}
TEST_LABELS = {**r3.LABELS, 'fast': 'off', 'workers': 1}


def cycle_game(index, cycle, run_seed, models, git_state):
    """A complete legal self-play record (``tests/test_h7c_labels.game``) of this cycle, seed and actor."""
    identity = sp.run_identity(cycle, run_seed, models, 'cuda')
    plies = []
    for ply in range(3, len(MOVES)):
        move, (route, event) = MOVES[ply], PLAN[ply]
        visits = [[*move, 40, 0.1], [3, 3, 10, 0.0]] if route == 'tree' else []
        plies.append({'ply': ply, 'color': 'black' if ply % 2 == 0 else 'white', 'route': route,
                      'engine_move': move, 'played': move, 'event': event, 'seconds': 0.1, 'visits': visits,
                      'proven_loss': [], 'target': targets.policy_target(route, move, visits, []), 'e4': {}})
    return {**identity, 'generator_git': git_state, 'game_index': index, 'game_seed': 1000 + index,
            'opening_seed': 2, 'opening': MOVES[:3], 'moves': [list(m) for m in MOVES], 'length': len(MOVES),
            'result': 'black', 'winner': BLACK, 'real_draw': False, 'truncated': False, 'events': {}, 'plies': plies,
            'game_seconds': 1.0}


class FakeAgent:
    def __init__(self, seed):
        from analysis.e4v2_attack import Cooldown
        self.seed, self.cooldown = seed, Cooldown()

    def new_game(self):
        from analysis.e4v2_attack import Cooldown
        self.cooldown = Cooldown()


def fake_search_factory(games_by_moves, factor, no_mass=()):
    """At factor 1 the actor's own record; at factor 8 another visit split (``no_mass``: plies whose S400 visits
    all fall on a proven loss: no S400 target)."""
    def search(agent, position):
        n = len(position.history)
        p = next(p for p in games_by_moves['plies'] if p['ply'] == n)
        if factor == 1:
            return {'route': p['route'], 'engine_move': p['engine_move'], 'visits': copy.deepcopy(p['visits']),
                    'proven_loss': [], 'e4': {}, 'tree_budget': 50, 'seconds': 0.01}
        move = p['engine_move']
        if n in no_mass:
            return {'route': 'tree', 'engine_move': move, 'visits': [[3, 3, 400, -1.0]], 'proven_loss': [[3, 3]],
                    'e4': {}, 'tree_budget': 400, 'seconds': 0.02}
        return {'route': 'tree', 'engine_move': move, 'visits': [[*move, 300, 0.3], [3, 3, 100, 0.0]],
                'proven_loss': [], 'e4': {}, 'tree_budget': 400, 'seconds': 0.02}
    return search


def fake_execute(no_mass=()):
    """``h7c_retarget._execute`` in process: ``retarget_game`` with the fake search; the records appended as the
    real one appends them (resumed records kept)."""
    def execute(args, games, identity, modes, factor, log, gate=None, validate=True):
        finished = sp.load_finished(args.jsonl, {'identity': identity}, log=log) if args.jsonl.exists() else {}
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.jsonl.open('a', encoding='utf-8') as out:
            for g in games:
                if g['game_index'] in finished:
                    continue
                with mock.patch.object(rt, 'search', fake_search_factory(g, factor, no_mass)):
                    rows = rt.retarget_game(g, modes[0], make=FakeAgent)
                for p, row in zip(hr.eligible_plies(g), rows):
                    if validate:
                        hr.check_row(g, p, row)
                rec = {'identity': identity, 'game_index': g['game_index'], 'rows': rows}
                out.write(json.dumps(rec, separators=(',', ':')) + '\n')
                finished[g['game_index']] = rec
        execute.calls.append({'factor': factor, 'policy': str(args.policy_checkpoint), 'games': len(games)})
        return finished
    execute.calls = []
    return execute


def fake_arena_game(task, make_tested=None, make_opponent=None, max_plies=None):
    """A complete game decided by the opening and the arm: the tested side wins both colours at a sixth of the
    openings, loses both at another sixth, and otherwise wins as black only (a score near 0.5: CONTINUE)."""
    from tests.h7r_helpers import winning_moves

    ctx = task['context']
    black = task['tested_color'] == BLACK
    h = int(hashlib.sha256((task['opening_id'] + ctx['arm']).encode()).hexdigest(), 16)
    result = 'win' if h % 6 == 0 or (black and h % 6 != 1) else 'loss'
    tested = BLACK if black else -BLACK
    winner = tested if result == 'win' else -tested
    moves = winning_moves(task['opening'], winner)
    return {'key': task['key'], 'signature': ctx['signature'], 'arm': task['arm'], 'pair': task['pair'],
            'seed': task['seed'], 'tested_color': 'black' if black else 'white', 'result': result,
            'winner': winner, 'length': len(moves), 'moves': moves, 'illegal': None, 'truncated': False,
            'tested_moves': [], 'opponent_routes': {}, 'game_seconds': 1.0}


# --- rules ---------------------------------------------------------------------------------------------------------

class RuleTest(unittest.TestCase):
    def test_the_decision_table(self):
        b = lambda lo, hi: {'estimate': (lo + hi) / 2, 'lower_one_sided': lo, 'upper_one_sided': hi}  # noqa: E731
        self.assertEqual(r3.decide(b(0.001, 0.05), 1), 'END_STRONGER')
        self.assertEqual(r3.decide(b(-0.06, -0.0201), 1), 'STOP_REGRESSION')
        for k, want in ((1, 'CONTINUE'), (2, 'CONTINUE'), (3, 'END_BUDGET')):
            self.assertEqual(r3.decide(b(-0.04, -0.02), k), want)  # -2%p exactly is not below -2%p
            self.assertEqual(r3.decide(b(-0.057, -0.0183), k), want)  # PRESERVE S2: upper -1.83%p, continues
            self.assertEqual(r3.decide(b(0.0, 0.03), k), want)  # a lower bound of exactly 0 is not above 0
        self.assertEqual(r3.label(b(-0.057, -0.0183)), 'WEAKER')
        self.assertEqual(r3.label(b(-0.02, 0.01)), 'NOT_ESTABLISHED')

    def test_cycles_are_paired_by_opening_id(self):
        def table(order, scores):
            return {'table': {'pairs': [{'opening_id': f'o{i}', 'class_id': f'c{i}', 'pair_score': scores[i]}
                                        for i in order]}}
        arena = {'arms': [{'name': 'A2', 'pairing': 'O1-dev-c'}]}
        prev = {'arms': [{'name': 'A1', 'pairing': 'O1-dev-c'}]}
        now, before = table([2, 0, 1], [1.0, 0.5, 0.0]), table([0, 1, 2], [0.5, 0.5, 0.5])
        rows = r3.change_strata(arena, {'A2': now}, prev, {'A1': before})['O1-dev-c']
        self.assertEqual(rows, [(0.0, 0.5, 'c2'), (1.0, 0.5, 'c0'), (0.5, 0.5, 'c1')])  # never by array position
        before['table']['pairs'][0]['opening_id'] = 'oX'
        with self.assertRaisesRegex(ValueError, 'did not play the same openings'):
            r3.change_strata(arena, {'A2': now}, prev, {'A1': before})

    def test_s400_targets_for_train_and_val(self):
        models = {k: 'a' * 64 for k in ('policy_checkpoint_sha256', 'policy_metadata_sha256',
                                        'value_checkpoint_sha256', 'value_metadata_sha256')}
        git_state = {'commit': 'abc', 'dirty': False}
        val = next(i for i in range(200) if split_for(f'h7c:c1:8809:{i}', fractions=lab.SPLIT_FRACTIONS,
                                                       salt=lab.SPLIT_SALT) == 'val')
        games = [cycle_game(i, 1, 8809, models, git_state) for i in (0, val)]
        _, base, _ = lab.build(games, probes={}, **TEST_LABELS)
        ident = {'x': 1}
        execute = fake_execute(no_mass=(7,))
        with TemporaryDirectory() as tmp:
            ns = argparse.Namespace(jsonl=Path(tmp) / 's400.jsonl', policy_checkpoint='p')
            finished = execute(ns, games, ident, ('restored',), 8, QUIET)
        s400 = {(gi, r['ply']): r for gi, rec in finished.items() for r in rec['rows']}
        by_id = {lab.game_id(g): g for g in games}
        cand, counts = r3.replace_targets(base, by_id, s400)
        self.assertEqual(set(counts), {'train', 'val'})
        for split in ('train', 'val'):
            self.assertEqual(counts[split]['visit_rows'], 3)
            self.assertEqual((counts[split]['s400'], counts[split]['s50_kept_no_s400_mass']), (2, 1))  # ply 7 kept
        self.assertEqual(len(cand), len(base))
        for a, b in zip(base, cand):
            self.assertEqual({k: v for k, v in a.items() if k not in st.ROW_CHANGED},
                             {k: v for k, v in b.items() if k not in st.ROW_CHANGED})
            if a['kind'] == 'visits' and a['ply'] != 7:
                g = by_id[a['game_id']]
                want = hr.retarget_target(next(q for q in g['plies'] if q['ply'] == a['ply']),
                                          s400[(g['game_index'], a['ply'])])[0]['target']
                self.assertEqual((b['target'], b['s400']), (want, {'target': 's400'}))
                self.assertNotEqual(b['target'], a['target'])
            elif a['kind'] == 'visits':
                self.assertEqual((b['target'], b['s400']['target']), (a['target'], 's50_kept'))
            else:
                self.assertEqual(a, b)  # proof rows untouched
        self.assertEqual(r3.check_replacement(base, cand, by_id, s400), [])
        i = next(i for i, b in enumerate(cand) if (b.get('s400') or {}).get('target') == 's400')
        for change in (lambda r: r.update(target=base[i]['target']), lambda r: r.update(split='val'),
                       lambda r: r.update(mask=not r['mask'])):
            bad = copy.deepcopy(cand)
            change(bad[i])
            self.assertTrue(r3.check_replacement(base, bad, by_id, s400))
        self.assertTrue(r3.check_replacement(base, cand[:-1], by_id, s400))


class TrainingPathTest(unittest.TestCase):
    """The initial model and the teacher come from their own files (``h7r_p1a1.execute`` loads both from H3)."""

    def test_initial_model_and_teacher_are_loaded_separately(self):
        import contextlib
        loaded = []

        class Net:
            def __init__(self, path):
                self.path = path

            def state_dict(self):
                return {'path': self.path}

        def load(path, device):
            loaded.append(str(path))
            return Net(str(path)), {}

        trained = {}

        def train_policy(cfg, parent, teacher, *a, **k):
            trained.update(parent=parent.path, teacher=teacher.path, replay=k['replay'])
            return {'record': {}, 'p1': {'replay': {}}, 'state': None}

        @contextlib.contextmanager
        def deterministic(c, device):
            yield {'threads': 1}

        class Fp:
            def __init__(self, data):
                pass

            def install(self, stack):
                pass

            def summary(self):
                return {}

        s4 = json.loads((ROOT / st.CONTRACT).read_text())
        checked = {'rows': {'train': [], 'val': []}, 'cache': {'train': 'replay', 'val': 'guard'}}
        with mock.patch('h7c_train_policy.load_policy_net', load), \
                mock.patch('hybrid.h7r_p1_train.train_policy', train_policy), \
                mock.patch.object(r3.a1, 'deterministic', deterministic), \
                mock.patch('h7r_p1_diag.Fingerprints', Fp), \
                mock.patch('h7r_p1_diag.state_sha256', lambda s: hashlib.sha256(s['path'].encode()).hexdigest()), \
                mock.patch('hybrid.h7c_train.policy_tensors', lambda rows: {}):
            run = r3.run_training(s4, 20261201, 'cpu', checked, 'runs/c1/model/best.pt', 'runs/h3/best.pt', QUIET)
        self.assertEqual(loaded, ['runs/c1/model/best.pt', 'runs/h3/best.pt'])
        self.assertEqual((trained['parent'], trained['teacher']), ('runs/c1/model/best.pt', 'runs/h3/best.pt'))
        self.assertEqual(trained['replay'].mu, 0.5)
        self.assertEqual(run['summary']['initial_state_sha256'],
                         {'parent': hashlib.sha256(b'runs/c1/model/best.pt').hexdigest(),
                          'teacher': hashlib.sha256(b'runs/h3/best.pt').hexdigest()})

    def test_the_two_epoch_rule(self):
        self.assertEqual(r3.steps_rule(24077, 512), 96)
        self.assertEqual(r3.steps_rule(25000, 512), 98)
        self.assertEqual(r3.steps_rule(10, 512), 2)


# --- the loop on the R2-S400-TRAIN fixture ----------------------------------------------------------------------------

class LoopBase(PreserveBase):
    def setUp(self):
        patches = [mock.patch.object(r3, 'GAMES', GAMES), mock.patch.object(r3, 'SEARCH', SMALL_SEARCH),
                   mock.patch.object(r3, 'LABELS', TEST_LABELS)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        super().setUp()
        root = self.root
        for rel in (r3.SUITE_RECORD, *(path for _, path in r3.SUITES)):  # the committed suites of the checkout
            shutil.copy(ROOT / rel, root / rel)
        self.h3 = {'policy_checkpoint_sha256': p1.file_sha256(root / p1.PATHS['h3']),
                   'policy_metadata_sha256': p1.file_sha256(root / p1.h3_meta_path())}
        self.cache = {s: {'target': __import__('torch').zeros(3, 225)} for s in ('train', 'val')}
        for target, name, value in ((p1, 'check_inputs', lambda c, r: {'rows': {}, 'h3': self.h3, 'cache': self.cache,
                                                                         'source': {'models': {}}}),):
            patch = mock.patch.object(target, name, value)
            patch.start()
            self.addCleanup(patch.stop)
        tb = mock.patch.dict('analysis.tb_v3.MODELS', self.h3)
        tb.start()
        self.addCleanup(tb.stop)
        self.c3 = r3.contract_body(root, sources=())
        self.write_loop(self.c3)
        git(root, 'add', '-A')
        git(root, 'commit', '-qm', 'R3 contract')
        self.base = json.loads((root / st.run_record_path(20261024)).read_text())
        self.selected = {}  # model path -> selected state (the fake trainer's)
        self.trained = []
        self.selfplays = []

    def write_loop(self, c):
        path = self.root / r3.CONTRACT
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(c, indent=1) + '\n')

    def approve_loop(self, *stages):
        for s in stages:
            self.c3['approved'][s] = True
        self.write_loop(self.c3)
        git(self.root, 'commit', '-qam', 'approve ' + ' '.join(stages))

    def args(self, **k):
        return argparse.Namespace(contract=None, device='cuda', log_name=None, **k)

    # the stand-ins ------------------------------------------------------------------------------------------------

    def fake_selfplay(self, ns, log=print):
        from analysis import tb_v3_net
        models = tb_v3_net.model_identity(ns.policy_checkpoint, ns.value_checkpoint)
        git_state = {'commit': git(self.root, 'rev-parse', 'HEAD'), 'dirty': False}
        self.selfplays.append({'cycle': ns.cycle, 'policy': Path(ns.policy_checkpoint).relative_to(self.root).as_posix(),
                               'run_seed': ns.run_seed})
        games = [cycle_game(i, ns.cycle, ns.run_seed, models, git_state) for i in range(ns.games)]
        ns.jsonl.parent.mkdir(parents=True, exist_ok=True)
        ns.jsonl.write_text(''.join(json.dumps(g) + '\n' for g in games))
        identity = {**sp.run_identity(ns.cycle, ns.run_seed, models, ns.device), 'generator_git': git_state}
        return {'identity': identity, 'summary': sp.summarize(games), 'wall_seconds': 1.0}

    def fake_training(self, s4, seed, device, checked, init_path, teacher_path, log=print, expect_settings=None):
        from tests.test_h7c_train import small_model
        import torch
        init = Path(init_path).relative_to(self.root).as_posix()
        self.trained.append({'seed': seed, 'init': init, 'teacher': Path(teacher_path).relative_to(self.root).as_posix(),
                             'train_rows': len(checked['rows']['train']), 'val_rows': len(checked['rows']['val'])})
        from hybrid import h7c_train as tr
        run = self.preserve_run(seed, 0.5)
        n = len(checked['rows']['train'])
        tensors = tr.policy_tensors(checked['rows']['train'])  # the S400 rows load as training tensors
        assert tensors['target'].shape == (n, 225) and n > 0
        assert all((r.get('s400') or {}).get('target') == 's400' for r in checked['rows']['train'] if r['kind'] == 'visits')
        run['result']['record'].update(total_steps=r3.steps_rule(n, 512), train_rows=n,
                                       val_rows=len(checked['rows']['val']))
        parent = self.selected.get(init, self.base['initial_state_sha256']['parent'])
        run['summary']['initial_state_sha256'] = {'parent': parent,
                                                  'teacher': self.base['initial_state_sha256']['teacher']}
        sel = hashlib.sha256(f'{seed}'.encode()).hexdigest()
        run['summary']['selected_state_sha256'] = sel
        run['result']['state'] = {}
        run['parent'] = small_model(seed % 7)
        run['logits'] = torch.zeros(3)
        self.selected[r3.model_dir(len(self.trained)) + '/best.pt'] = sel
        if getattr(self, 'no_candidate', False):
            run['result']['state'] = None
            run['result']['record'].update(verdict='NO_CANDIDATE', best=None)
            run['summary']['selected_state_sha256'] = None
        return run

    def fake_gate_training(self, s4, seed, device, checked, init_path, teacher_path, log=print, expect_settings=None):
        self.gate_calls = (Path(init_path).relative_to(self.root).as_posix(),
                           Path(teacher_path).relative_to(self.root).as_posix(), expect_settings)
        return self.preserve_run(seed, 0.5, getattr(self, 'gate_change', None))

    def through_gate(self):
        self.approve_loop('prepare', 'gate')
        r3.prepare(self.args(), self.root)
        self.commit(r3.D + '/prepare.json')
        with mock.patch.object(r3, 'run_training', self.fake_gate_training):
            rec = r3.gate(self.args(), self.root, log=QUIET)
        self.commit(r3.D + '/gate.json')
        return rec

    def step(self, k, name, commit=True, **patches):
        execute = getattr(self, 'execute', None) or fake_execute()
        self.execute = execute
        state = {'commit': git(self.root, 'rev-parse', 'HEAD'), 'dirty': False}
        with mock.patch.object(sp, 'run', self.fake_selfplay), mock.patch.object(rt, '_execute', execute), \
                mock.patch.object(r3, 'run_training', self.fake_training), \
                mock.patch('h7c_arena.play_game', fake_arena_game), mock.patch.object(cli, '_git', lambda _: dict(state)):
            rec, code = r3.step(self.args(cycle=k, name=name, threads_for_tests=True, workers=1), self.root, log=QUIET)
        r3.verify(self.args(cycle=k, name=name), self.root)
        if commit:
            for f in r3.files_of(k, name):
                git(self.root, 'add', f)
            git(self.root, 'commit', '-qm', f'c{k} {name}')
        return rec, code

    def cycle(self, k):
        out = {}
        for name in r3.STEPS:
            out[name] = self.step(k, name)[0]
        return out


class LoopTest(LoopBase):
    def test_cycle_one_then_cycle_two_from_its_candidate(self):
        gate = self.through_gate()
        self.assertEqual((gate['passed'], gate['differences']), (True, []))
        self.assertEqual(self.gate_calls[:2], (p1.PATHS['h3'], p1.PATHS['h3']))  # both H3 in the gate
        self.approve_loop('loop')
        self.assertEqual(r3.next_step(self.args(), self.root),
                         {'action': 'run', 'cycle': 1, 'step': 'selfplay', 'record': r3.record_path(1, 'selfplay'),
                          'partial': False})
        with self.assertRaisesRegex(ValueError, 'the next step is'):
            r3.step(self.args(cycle=1, name='train'), self.root, log=QUIET)
        c1 = self.cycle(1)
        # cycle 1: H3 played, was the search prior and the initial model; S50 reproduced, S400 on every eligible ply
        self.assertEqual(self.selfplays[0], {'cycle': 1, 'policy': p1.PATHS['h3'], 'run_seed': 8809})
        self.assertTrue(c1['search']['s50_reproduction']['reproduced'])
        self.assertEqual(c1['search']['s50_reproduction']['compared'], 2 * 3)
        self.assertEqual([(x['factor'], x['policy'].endswith(p1.PATHS['h3'])) for x in self.execute.calls],
                         [(1, True), (8, True)])
        self.assertEqual(c1['search']['s400']['rows'], GAMES * 3)
        by = c1['rows']['replacement_by_split']
        self.assertEqual(sum(v['s400'] for v in by.values()), sum(v['visit_rows'] for v in by.values()))
        self.assertEqual(self.trained[0]['init'], p1.PATHS['h3'])
        self.assertEqual(self.trained[0]['teacher'], p1.PATHS['h3'])
        self.assertEqual(self.trained[0]['train_rows'], c1['rows']['unmasked']['train'])
        self.assertEqual(c1['train']['steps_rule']['total_steps'], r3.steps_rule(self.trained[0]['train_rows'], 512))
        self.assertEqual(c1['train']['verdict'], 'CANDIDATE')
        arena = json.loads((self.root / r3.arena_contract_path(1)).read_text())
        self.assertEqual({(a['pairing'], a['seed'], a['pairs']) for a in arena['arms']},
                         {('O1-dev-c', 9105, 300), ('O1-dev-d', 9106, 300)})
        self.assertEqual(arena['models']['L']['identity']['policy_checkpoint_sha256'],
                         c1['train']['candidate']['checkpoint_sha256'])
        self.assertEqual(c1['arena']['models']['L']['policy_checkpoint_sha256'], c1['train']['candidate']['checkpoint_sha256'])
        self.assertIsNone(c1['arena']['change_from_previous_cycle'])
        self.assertEqual(c1['arena']['decision'], r3.decide(c1['arena']['bounds']['P'], 1))
        if c1['arena']['decision'] != 'CONTINUE':
            self.skipTest(f"the stand-in game decided {c1['arena']['decision']}")
        # cycle 2: every step names the cycle-1 candidate
        cand = r3.model_dir(1) + '/best.pt'
        self.step(2, 'selfplay')
        self.assertEqual(self.selfplays[1], {'cycle': 2, 'policy': cand, 'run_seed': 8810})
        rec, _ = self.step(2, 'search')
        self.assertTrue(all(x['policy'].endswith(cand) for x in self.execute.calls[2:]))
        self.assertEqual(rec['actor']['checkpoint_sha256'], c1['train']['candidate']['checkpoint_sha256'])
        self.step(2, 'rows')
        rec, _ = self.step(2, 'train')
        self.assertEqual((self.trained[1]['init'], self.trained[1]['teacher'], self.trained[1]['seed']),
                         (cand, p1.PATHS['h3'], 20261202))
        self.assertEqual(rec['initial_model']['checkpoint_sha256'], c1['train']['candidate']['checkpoint_sha256'])
        self.assertEqual(rec['initial_state_sha256']['parent'], c1['train']['selected_state_sha256'])
        self.step(2, 'arena-contract')
        rec, _ = self.step(2, 'arena')
        ch = rec['change_from_previous_cycle']
        self.assertEqual((ch['paired_by'], ch['units']), ('opening id', {'O1-dev-c': 300, 'O1-dev-d': 300}))

    def test_cycle_two_needs_the_candidate_files_and_a_continue(self):
        self.through_gate()
        self.approve_loop('loop')
        c1 = self.cycle(1)
        if c1['arena']['decision'] != 'CONTINUE':
            self.skipTest('the stand-in game did not continue')
        (self.root / r3.model_dir(1) / 'best.pt').write_text('another checkpoint')
        with self.assertRaisesRegex(ValueError, 'the cycle 1 candidate: .* is not the recorded file'):
            self.step(2, 'selfplay')
        self.assertEqual(len(self.selfplays), 1)  # no second self-play

    def test_a_stop_ends_the_loop(self):
        self.through_gate()
        self.approve_loop('loop')
        for name in r3.STEPS[:4]:
            self.step(1, name)
        self.step(1, 'arena-contract')
        with mock.patch.object(r3, 'decide', lambda b, k: 'STOP_REGRESSION'):
            self.step(1, 'arena')
        self.assertEqual(r3.next_step(self.args(), self.root),
                         {'action': 'done', 'cycle': 1, 'step': 'arena', 'decision': 'STOP_REGRESSION'})
        with self.assertRaisesRegex(ValueError, 'the next step is'):
            self.step(2, 'selfplay')

    def test_no_candidate_ends_the_loop(self):
        self.through_gate()
        self.approve_loop('loop')
        for name in r3.STEPS[:3]:
            self.step(1, name)
        self.no_candidate = True
        rec, _ = self.step(1, 'train')
        self.assertEqual(rec['verdict'], 'NO_CANDIDATE')
        self.assertFalse((self.root / r3.model_dir(1) / 'best.pt').exists())
        self.assertEqual(r3.next_step(self.args(), self.root)['decision'], 'END_NO_CANDIDATE')

    def test_s50_not_reproduced_stops_before_s400(self):
        self.through_gate()
        self.approve_loop('loop')
        self.step(1, 'selfplay')
        execute = fake_execute()

        def other(args, games, identity, modes, factor, log, gate=None, validate=True):
            out = execute(args, games, identity, modes, factor, log, gate, validate)
            if factor == 1:
                out[games[0]['game_index']]['rows'][0]['engine_move'] = [3, 3]
            return out
        other.calls = execute.calls
        self.execute = other
        rec, code = self.step(1, 'search')
        self.assertEqual((rec['status'], code), ('S50_NOT_REPRODUCED', 2))
        self.assertEqual([x['factor'] for x in execute.calls], [1])  # no S400 search
        self.assertEqual(r3.next_step(self.args(), self.root)['decision'], 'STOP_S50_NOT_REPRODUCED')

    def test_a_wrong_initial_state_publishes_nothing(self):
        self.through_gate()
        self.approve_loop('loop')
        for name in r3.STEPS[:3]:
            self.step(1, name)
        self.selected[p1.PATHS['h3']] = '0' * 64  # the trainer started from another state
        with self.assertRaisesRegex(ValueError, 'is not the cycle 0 policy'):
            self.step(1, 'train')
        self.assertFalse((self.root / r3.record_path(1, 'train')).exists())

    def test_the_gate_is_exact(self):
        self.approve_loop('prepare', 'gate')
        r3.prepare(self.args(), self.root)
        self.commit(r3.D + '/prepare.json')
        self.gate_change = lambda r: r['evaluations'][0]['val'].update(ce=r['evaluations'][0]['val']['ce'] + 1e-12)
        with mock.patch.object(r3, 'run_training', self.fake_gate_training):
            rec = r3.gate(self.args(), self.root, log=QUIET)
        self.assertFalse(rec['passed'])
        self.commit(r3.D + '/gate.json')
        self.approve_loop('loop')
        with self.assertRaisesRegex(ValueError, 'the training-path gate did not pass'):
            r3.next_step(self.args(), self.root)

    def test_prepare_refusals(self):
        with self.assertRaisesRegex(ValueError, 'the prepare stage is not approved'):
            r3.prepare(self.args(), self.root)
        self.approve_loop('prepare')
        other = {'format': 'h7r-eval-contract-v2', 'contract_id': 'OTHER', 'arms': [{'seed': 9107}]}
        self.commit_record('docs/h7r/contracts-v2/OTHER.json', other)
        with self.assertRaisesRegex(ValueError, r'the contract seeds are already used: \{9107'):
            r3.prepare(self.args(), self.root)
        git(self.root, 'rm', '-q', 'docs/h7r/contracts-v2/OTHER.json')
        git(self.root, 'commit', '-qm', 'removed')
        self.assertTrue(r3.prepare(self.args(), self.root)['passed'])

    def test_the_contract(self):
        c, _, _ = r3.load_contract(None, self.root)
        self.assertEqual(c['seeds']['arena'], {'1': {'O1-dev-c': 9105, 'O1-dev-d': 9106},
                                               '2': {'O1-dev-c': 9107, 'O1-dev-d': 9108},
                                               '3': {'O1-dev-c': 9109, 'O1-dev-d': 9110}})
        self.write_loop({**self.c3, 'rules': {**self.c3['rules'], 'delta': 0.03}})
        git(self.root, 'commit', '-qam', 'delta changed')
        with self.assertRaisesRegex(ValueError, 'the rules, seeds, budgets or statistics differ'):
            r3.load_contract(None, self.root)
        self.write_loop({**self.c3, 'approved': {'prepare': False, 'gate': False, 'loop': True}})
        git(self.root, 'commit', '-qam', 'out of order')
        with self.assertRaisesRegex(ValueError, 'loop approved before'):
            r3.load_contract(None, self.root)


class FixedContractTest(unittest.TestCase):
    def test_the_committed_contract(self):
        c, _, eid = r3.load_contract(None, ROOT)
        self.assertEqual(list(c['approved']), list(r3.STAGES))  # approvals change over time, never the id
        self.assertEqual({k: v for k, v in c.items() if k not in ('sources', 'approved')},
                         {k: v for k, v in r3.contract_body(ROOT).items() if k not in ('sources', 'approved')})
        self.assertEqual([s['path'] for s in c['sources']], list(r3.SOURCES))
        self.assertEqual(eid, 'd464e0447642aacb83934c1a00b8a7958832a91f1d7aa807eac3f5a0d3b09fc3')


# --- the desktop blocks ----------------------------------------------------------------------------------------------

R3_STUB = r'''import hashlib, json, os, subprocess, sys
from pathlib import Path
D = 'docs/h7r/r3/R3-POLICY-LOOP-1'
R = 'runs/h7r/R3-POLICY-LOOP-1'
STEPS = ['selfplay', 'search', 'rows', 'train', 'arena-contract', 'arena']
cmd = sys.argv[1]
arg = lambda n: sys.argv[sys.argv.index(n) + 1] if n in sys.argv else None
log = arg('--log-name')
cublas = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
tracked = lambda rel: subprocess.run(['git', 'cat-file', '-e', 'HEAD:' + rel], capture_output=True).returncode == 0
rec_path = lambda k, s: f"{D}/c{k}/{'judge' if s == 'arena' else s}.json"
def stop(msg, code=1):
    print('STOP: ValueError: ' + msg); sys.exit(code)
def outcome(s, r):
    if s == 'search' and r.get('status') != 'SEARCHED': return 'STOP_S50_NOT_REPRODUCED'
    if s == 'train' and r.get('verdict') != 'CANDIDATE': return 'END_NO_CANDIDATE'
    if s == 'arena' and r.get('decision') != 'CONTINUE': return r.get('decision')
def state():
    for k in (1, 2, 3):
        for s in STEPS:
            rel = rec_path(k, s)
            if tracked(rel):
                end = outcome(s, json.loads(Path(rel).read_text()))
                if end: return {'action': 'done', 'cycle': k, 'step': s, 'decision': end}
            elif Path(rel).exists():
                return {'action': 'record', 'cycle': k, 'step': s, 'record': rel}
            else:
                return {'action': 'run', 'cycle': k, 'step': s, 'record': rel,
                        'partial': Path(R, f'c{k}', s + '.partial').exists()}
    return {'action': 'done', 'cycle': 3, 'step': 'arena', 'decision': 'END_BUDGET'}
def files(k, s):
    return [rec_path(k, s)] + ([f'docs/h7r/contracts-v2/R3-POLICY-LOOP-ARENA-{k}.json'] if s == 'arena-contract' else [])
if cmd in ('prepare', 'gate'):
    if os.environ.get('R3_FORBID') == cmd: stop('must not run again', 9)
    if cmd == 'gate' and cublas != ':4096:8': stop('CUBLAS_WORKSPACE_CONFIG', 7)
    passed = not (cmd == 'gate' and os.environ.get('R3_GATE_FAIL'))
    Path(D, cmd + '.json').write_text(json.dumps({'format': 'h7r-r3-loop-' + cmd + '-v1', 'passed': passed,
                                                  'log_name': log}))
    print(cmd, passed); sys.exit(0 if passed else 2)
if cmd == 'next':
    print(json.dumps(state())); sys.exit(0)
k = int(arg('--cycle')); s = arg('--name')
if cmd == 'files':
    print('\n'.join(files(k, s))); sys.exit(0)
if cmd == 'verify':
    r = json.loads(Path(rec_path(k, s)).read_text())
    if s == 'train' and sha(Path(R, f'c{k}', 'best.pt')) != r['candidate']:
        stop('best.pt: its SHA-256 is not the one the record names')
    print(json.dumps({'log_name': r['log_name'], 'exit_code': 2 if s == 'search' and r['status'] != 'SEARCHED' else 0,
                      'stage': files(k, s)})); sys.exit(0)
# step
if os.environ.get('R3_FORBID') == f'c{k}-{s}': stop('must not run again', 9)
if state() != {'action': 'run', 'cycle': k, 'step': s, 'record': rec_path(k, s), 'partial': state().get('partial')}:
    stop('not the next step')
if (s == 'train') != (cublas == ':4096:8'): stop('CUBLAS_WORKSPACE_CONFIG for ' + s, 7)
if os.environ.get('R3_FAIL') == f'c{k}-{s}': stop('stand-in failure')
Path(R, f'c{k}').mkdir(parents=True, exist_ok=True)
body, code = {'format': 'h7r-r3-loop-' + s + '-v1', 'cycle': k, 'log_name': log}, 0
if s == 'search':
    failed = os.environ.get('R3_S50_FAIL') == str(k)
    body['status'], code = ('S50_NOT_REPRODUCED', 2) if failed else ('SEARCHED', 0)
if s == 'train':
    Path(R, f'c{k}', 'best.pt').write_text(f'candidate {k}')
    body.update(verdict='CANDIDATE', candidate=sha(Path(R, f'c{k}', 'best.pt')))
if s == 'arena-contract':
    Path('docs/h7r/contracts-v2').mkdir(parents=True, exist_ok=True)
    Path(f'docs/h7r/contracts-v2/R3-POLICY-LOOP-ARENA-{k}.json').write_text(json.dumps({'cycle': k}))
if s == 'arena':
    body['decision'] = os.environ.get(f'R3_DECISION_{k}', 'CONTINUE' if k < 3 else 'END_BUDGET')
Path(rec_path(k, s)).parent.mkdir(parents=True, exist_ok=True)
Path(rec_path(k, s)).write_text(json.dumps(body))
if s == 'train' and os.environ.get('R3_TAMPER') == str(k):
    Path(R, f'c{k}', 'best.pt').write_text('replaced')
print(s, code); sys.exit(code)
'''


class BlockTest(unittest.TestCase):
    """The desktop blocks of the draft (section 13) run by bash on a real temporary repository with a bare origin, the
    real .gitignore and record tool; the loop tool is a stand-in."""

    def setUp(self):
        import re
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        self.origin, self.work = base / 'origin.git', base / 'work'
        git(base, 'init', '-q', '--bare', str(self.origin))
        git(base, 'init', '-q', '-b', 'feat/h7-c1', str(self.work))
        w = self.work
        for k, v in (('user.email', 't@t'), ('user.name', 't')):
            git(w, 'config', k, v)
        shutil.copy(ROOT / '.gitignore', w / '.gitignore')
        for rel in ('scripts/h7r_d4_records.py', 'src/hybrid/h7r_ledger.py', 'scripts/h7r_attempts.py'):
            (w / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / rel, w / rel)
        (w / 'src/hybrid/__init__.py').write_text('')
        (w / 'scripts/check_track_b_frozen.py').write_text('print("frozen PASS")' + chr(10))
        (w / 'scripts/h7r_r3_loop.py').write_text(R3_STUB)
        (w / '.venv/bin').mkdir(parents=True)
        (w / '.venv/bin/activate').write_text('')
        (w / r3.D).mkdir(parents=True)
        (w / r3.CONTRACT).write_text(json.dumps({'format': 'stand-in', 'contract_id': r3.CONTRACT_ID}))
        git(w, 'add', '-A')
        git(w, 'commit', '-qm', 'init')
        git(w, 'remote', 'add', 'origin', str(self.origin))
        git(w, 'push', '-q', 'origin', 'feat/h7-c1')
        doc = (ROOT / 'docs/h7r/r3/R3-POLICY-LOOP-1-draft.md').read_text()
        blocks = re.findall(r'```bash' + chr(10) + r'(.*?)```', doc, re.S)
        self.assertEqual(len(blocks), 3)
        self.blocks = dict(zip(('r3-P', 'r3-G', 'r3-L'), (b.replace('~/renju-h6', w.as_posix()) for b in blocks)))

    def bash(self, name, **env):
        import os
        import subprocess
        e = {k: v for k, v in os.environ.items() if k != 'CUBLAS_WORKSPACE_CONFIG'}
        e = {**e, 'PYTHONDONTWRITEBYTECODE': '1', **env}
        r = subprocess.run(['bash', '-c', self.blocks[name]], cwd=self.work, capture_output=True, text=True, env=e,
                           start_new_session=True)
        return r.returncode, r.stdout + r.stderr

    def files(self, rel):
        return git(self.work, 'ls-files', rel).splitlines()

    def subjects(self):
        return git(self.work, 'log', '--format=%s').splitlines()

    def through(self, *blocks, **env):
        for b in blocks:
            code, out = self.bash(b, **env)
            self.assertEqual(code, 0, f'{b}: {out}')

    def assert_pushed(self):
        self.assertEqual(git(self.work, 'rev-parse', 'HEAD'), git(self.origin, 'rev-parse', 'feat/h7-c1'))
        self.assertEqual(git(self.work, 'status', '--porcelain', '--untracked-files=no'), '')

    def step_commits(self):
        return [s for s in self.subjects() if s.startswith('H7-R R3 LOOP: c')]

    def test_prepare_then_gate(self):
        code, out = self.bash('r3-G')
        self.assertEqual(code, 1)
        self.assertIn('prepare is not recorded', out)
        self.through('r3-P', 'r3-G')
        self.assertTrue(self.files(f'{r3.D}/gate.json'))
        self.assert_pushed()
        code, out = self.bash('r3-G', R3_GATE_FAIL='1')
        self.assertEqual(code, 1)
        self.assertIn('already recorded', out)

    def test_the_loop_runs_to_the_budget_one_commit_per_step(self):
        code, out = self.bash('r3-L')
        self.assertEqual(code, 1)
        self.assertIn('the gate is not recorded', out)
        self.through('r3-P', 'r3-G')
        code, out = self.bash('r3-L')
        self.assertEqual(code, 0, out)
        self.assertIn('__DONE__ END_BUDGET (cycle 3, arena)', out)
        want = [f'H7-R R3 LOOP: c{k} {s} exit 0 (records only)' for k in (1, 2, 3) for s in r3.STEPS]
        self.assertEqual(self.step_commits()[::-1], want)
        self.assertEqual(len([f for f in self.files(f'{r3.D}/stage-logs') if f.endswith('.log') and '/c' in f]), 18)
        self.assertTrue(self.files(r3.arena_contract_path(2)))
        self.assert_pushed()
        code, out = self.bash('r3-L')  # done: nothing runs again
        self.assertEqual(code, 0, out)
        self.assertIn('__DONE__ END_BUDGET', out)
        self.assertEqual(len(self.step_commits()), 18)

    def test_a_decision_ends_the_loop(self):
        self.through('r3-P', 'r3-G')
        code, out = self.bash('r3-L', R3_DECISION_1='STOP_REGRESSION')
        self.assertEqual(code, 0, out)
        self.assertIn('__DONE__ STOP_REGRESSION (cycle 1, arena)', out)
        self.assertEqual(len(self.step_commits()), 6)
        self.assertFalse(self.files(f'{r3.D}/c2'))

    def test_s50_not_reproduced_is_recorded_and_ends(self):
        self.through('r3-P', 'r3-G')
        code, out = self.bash('r3-L', R3_S50_FAIL='1')
        self.assertEqual(code, 0, out)
        self.assertIn('__DONE__ STOP_S50_NOT_REPRODUCED (cycle 1, search)', out)
        self.assertEqual(self.step_commits()[0], 'H7-R R3 LOOP: c1 search exit 2 (records only)')

    def test_a_checkpoint_changed_before_staging_is_not_committed(self):
        self.through('r3-P', 'r3-G')
        code, out = self.bash('r3-L', R3_TAMPER='1')
        self.assertEqual(code, 1, out)
        self.assertIn('c1 train: the record does not check against its files; nothing committed', out)
        self.assertEqual(self.step_commits()[0], 'H7-R R3 LOOP: c1 rows exit 0 (records only)')
        self.assertEqual(git(self.work, 'diff', '--cached', '--name-only'), '')
        self.assert_pushed()

    def test_a_failed_step_commits_its_log_and_the_rerun_resumes(self):
        self.through('r3-P', 'r3-G')
        code, out = self.bash('r3-L', R3_FAIL='c1-rows')
        self.assertEqual(code, 1, out)
        self.assertIn('STOP: c1 rows failed (exit 1)', out)
        self.assertEqual(self.step_commits()[0], 'H7-R R3 LOOP: c1 rows exit 1 (records only)')
        self.assertFalse(self.files(r3.record_path(1, 'rows')))
        code, out = self.bash('r3-L', R3_DECISION_1='END_STRONGER')
        self.assertEqual(code, 0, out)
        self.assertIn('__DONE__ END_STRONGER (cycle 1, arena)', out)
        self.assertEqual(self.step_commits()[3], 'H7-R R3 LOOP: c1 rows exit 0 (records only)')

    def test_an_interrupted_step_is_recorded_without_a_new_run(self):
        self.through('r3-P', 'r3-G')
        L = self.work / r3.RUNS / 'logs'
        L.mkdir(parents=True, exist_ok=True)
        name = 'c1-selfplay-AbC123xy.log'
        (L / name).write_text('selfplay 0' + chr(10))  # killed before its exit line
        rec = self.work / r3.record_path(1, 'selfplay')
        rec.parent.mkdir(parents=True, exist_ok=True)
        rec.write_text(json.dumps({'format': 'h7r-r3-loop-selfplay-v1', 'cycle': 1, 'log_name': name}))
        code, out = self.bash('r3-L', R3_FORBID='c1-selfplay', R3_DECISION_1='STOP_REGRESSION')
        self.assertEqual(code, 0, out)
        self.assertIn('recording it (no new run)', out)
        self.assertIn('Recovered exit code', (self.work / r3.D / 'stage-logs' / name).read_text())
        self.assertEqual(self.step_commits()[-1], 'H7-R R3 LOOP: c1 selfplay exit 0 (records only)')
