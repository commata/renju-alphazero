"""Stage 7 tactical/value probes: raw network policy/value on fixed positions.

Measures the network itself (no search) so learning signal is visible before it is
strong enough to win games. All policy numbers use the same masked softmax as the
PUCT evaluator; the black-forbidden diagnostic is the only unmasked quantity.

Per kind (``stage7-probes-v1``):
- ``top1`` / ``top3``: fraction of probes whose top-1 / top-3 masked policy action is a
  correct move;
- ``correct_mass``: mean masked probability on the correct move set;
- ``uniform_mass``: mean probability a uniform policy over legal moves would give the
  same set (``|correct| / |legal|``); ``mass_lift = correct_mass / uniform_mass`` is the
  learning signal while absolute mass is still tiny;
- ``value_mean`` / ``value_sign_accuracy`` for kinds with a value label (value is from
  the side to move). A constant-sign value head scores 1.0 on one side and 0.0 on the
  other, so the headline value metrics are the cross-kind ``value_overall``:
  ``separation = mean(value | win label) - mean(value | loss label)`` and
  ``balanced_sign_accuracy`` (mean of the win-side and loss-side accuracies);
- ``avoid_mass`` / ``avoid_lift`` for the ``avoid`` kind (lower is better).

``forbidden_diagnostic``: on BLACK-to-move probes that have forbidden empty points,
the unmasked softmax mass on those points and its lift over a uniform distribution on
empty points. The engine never allows these moves, so this only shows whether the raw
logits still point at them.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch

from model.config import ModelConfig, coordinate_to_action
from model.encoding import encode_game
from model.masking import legal_moves_to_mask, masked_softmax
from model.network import PolicyValueNet
from renju import BLACK, EMPTY, Game
from renju.rules import forbidden_reason

from .training_checkpoint import load_checkpoint_payload

PROBE_FORMAT = 'stage7-probes-v1'
RESULT_FORMAT = 'stage7-probe-results-v1'
POLICY_KINDS = ('immediate_win', 'must_block', 'vcf')


@dataclass(frozen=True)
class Probe:
    id: str
    kind: str
    to_play: str
    moves: tuple[tuple[int, int], ...]
    correct: frozenset[int]
    avoid: frozenset[int]
    value_sign: int | None


def load_probe_set(path: str | Path) -> tuple[list[Probe], str]:
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    if data.get('format') != PROBE_FORMAT:
        raise ValueError(f"unsupported probe format: {data.get('format')!r}")
    probes = [
        Probe(id=item['id'], kind=item['kind'], to_play=item['to_play'],
              moves=tuple(tuple(m) for m in item['moves']),
              correct=frozenset(coordinate_to_action(*m) for m in item['correct_moves']),
              avoid=frozenset(coordinate_to_action(*m) for m in item['avoid_moves']),
              value_sign=item['value_sign'])
        for item in data['probes']
    ]
    return probes, hashlib.sha256(raw).hexdigest()


def load_model_from_training_checkpoint(path: str | Path) -> tuple[PolicyValueNet, dict]:
    payload = load_checkpoint_payload(path)
    model = PolicyValueNet(ModelConfig(**payload['model_config']))
    model.load_state_dict(payload['model_state_dict'])
    model.eval()
    info = {
        'checkpoint': str(path),
        'checkpoint_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        'generation': payload['generation'],
        'global_step': payload['global_step'],
        'critical_config_hash': payload.get('critical_config_hash'),
        'git_commit': payload.get('git_commit'),
    }
    return model, info


def _replay(moves) -> Game:
    game = Game()
    for move in moves:
        game.play(*move)
    return game


def _forbidden_actions(game: Game) -> list[int]:
    return [coordinate_to_action(r, c) for r in range(15) for c in range(15)
            if game.board[r][c] == EMPTY and forbidden_reason(game.board, r, c) is not None]


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def evaluate_probes(model: PolicyValueNet, probes: list[Probe], *, batch_size: int = 64
                    ) -> dict:
    """Return per-probe rows plus per-kind and forbidden summaries."""
    rows = []
    forbidden_rows = []
    was_training = model.training
    model.eval()
    try:
        for start in range(0, len(probes), batch_size):
            chunk = probes[start:start + batch_size]
            games = [_replay(p.moves) for p in chunk]
            masks = [legal_moves_to_mask(g.legal_moves()) for g in games]
            planes = torch.stack([encode_game(g, m) for g, m in zip(games, masks)])
            mask_batch = torch.stack(masks)
            with torch.inference_mode():
                logits, values = model(planes)
                policy = masked_softmax(logits, mask_batch)
                raw = torch.softmax(logits, dim=-1)
            for probe, game, mask, pol, raw_row, value in zip(
                    chunk, games, masks, policy, raw, values.reshape(-1)):
                legal_count = int(mask.sum())
                order = torch.argsort(pol, descending=True, stable=True).tolist()
                row = {'id': probe.id, 'kind': probe.kind, 'to_play': probe.to_play,
                       'value': float(value), 'legal_moves': legal_count,
                       'top1_action': order[0]}
                if probe.correct:
                    correct = sorted(probe.correct)
                    row.update(
                        top1=order[0] in probe.correct,
                        top3=any(a in probe.correct for a in order[:3]),
                        correct_mass=float(pol[correct].sum()),
                        uniform_mass=len(correct) / legal_count)
                if probe.avoid:
                    avoid = sorted(probe.avoid)
                    row.update(avoid_mass=float(pol[avoid].sum()),
                               avoid_uniform_mass=len(avoid) / legal_count)
                if probe.value_sign is not None:
                    row['value_sign'] = probe.value_sign
                    row['value_sign_correct'] = float(value) * probe.value_sign > 0
                rows.append(row)
                if game.to_play == BLACK:
                    forbidden = _forbidden_actions(game)
                    if forbidden:
                        empty = sum(v == EMPTY for line in game.board for v in line)
                        forbidden_rows.append({
                            'id': probe.id,
                            'raw_mass': float(raw_row[forbidden].sum()),
                            'uniform_mass': len(forbidden) / empty,
                        })
    finally:
        model.train(was_training)
    return {'rows': rows, 'summary': summarize(rows),
            'value_overall': summarize_value(rows),
            'forbidden_diagnostic': summarize_forbidden(forbidden_rows)}


def summarize(rows: list[dict]) -> dict:
    out = {}
    for kind in sorted({r['kind'] for r in rows}):
        subset = [r for r in rows if r['kind'] == kind]
        summary = {'probes': len(subset)}
        policy = [r for r in subset if 'correct_mass' in r]
        if policy:
            correct_mass = _mean(r['correct_mass'] for r in policy)
            uniform_mass = _mean(r['uniform_mass'] for r in policy)
            summary.update(top1=_mean(float(r['top1']) for r in policy),
                           top3=_mean(float(r['top3']) for r in policy),
                           correct_mass=correct_mass, uniform_mass=uniform_mass,
                           mass_lift=correct_mass / uniform_mass)
        avoid = [r for r in subset if 'avoid_mass' in r]
        if avoid:
            avoid_mass = _mean(r['avoid_mass'] for r in avoid)
            avoid_uniform = _mean(r['avoid_uniform_mass'] for r in avoid)
            summary.update(avoid_mass=avoid_mass, avoid_uniform_mass=avoid_uniform,
                           avoid_lift=avoid_mass / avoid_uniform)
        valued = [r for r in subset if 'value_sign' in r]
        if valued:
            summary.update(value_mean=_mean(r['value'] for r in valued),
                           value_sign_accuracy=_mean(float(r['value_sign_correct'])
                                                     for r in valued))
        for color in ('BLACK', 'WHITE'):
            part = [r for r in policy if r['to_play'] == color]
            if part:
                summary[f'top1_{color.lower()}'] = _mean(float(r['top1']) for r in part)
        out[kind] = summary
    return out


def summarize_value(rows: list[dict]) -> dict:
    wins = [r for r in rows if r.get('value_sign') == 1]
    losses = [r for r in rows if r.get('value_sign') == -1]
    if not wins or not losses:
        return {'win_probes': len(wins), 'loss_probes': len(losses)}
    win_acc = _mean(float(r['value_sign_correct']) for r in wins)
    loss_acc = _mean(float(r['value_sign_correct']) for r in losses)
    win_mean = _mean(r['value'] for r in wins)
    loss_mean = _mean(r['value'] for r in losses)
    return {'win_probes': len(wins), 'loss_probes': len(losses),
            'win_value_mean': win_mean, 'loss_value_mean': loss_mean,
            'separation': win_mean - loss_mean,
            'win_sign_accuracy': win_acc, 'loss_sign_accuracy': loss_acc,
            'balanced_sign_accuracy': (win_acc + loss_acc) / 2}


def summarize_forbidden(rows: list[dict]) -> dict:
    if not rows:
        return {'positions': 0}
    raw = _mean(r['raw_mass'] for r in rows)
    uniform = _mean(r['uniform_mass'] for r in rows)
    return {'positions': len(rows), 'raw_mass': raw, 'uniform_mass': uniform,
            'raw_lift': raw / uniform}


def run_probe_file(checkpoint: str | Path, probe_path: str | Path) -> dict:
    probes, probe_sha = load_probe_set(probe_path)
    model, info = load_model_from_training_checkpoint(checkpoint)
    result = evaluate_probes(model, probes)
    return {'format_version': RESULT_FORMAT, 'probe_set': str(probe_path),
            'probe_set_sha256': probe_sha, **info, **result}
