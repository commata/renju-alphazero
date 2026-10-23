"""H7-c1 policy targets and the opening-sampling rule (docs/h7-c1.md; contract in docs/h6-value.md, H7-c1 approval).

Pure functions of one engine move's record, shared by the self-play generator (``scripts/h7c_selfplay.py``, which
samples with them) and the label step (which writes the targets), so the two can never disagree.

Routes of TB-V3 (``analysis.mcts_v8`` + E4-v2):
    proven win   stage1 (a five now), stage3 (an unstoppable four), own_vcf (VCF), own_vct (V8-B, depth-1 VCT),
                 own_vct2 (E4-v2): the played move is a proven win
    tree         the PUCT tree ran: ``v8_root_visits`` holds (move, visits, mean value) of the root children
    forced       stage2 (block a five), stage4 / stage5 (forced defences): no tree, no visit distribution

Policy target (``TARGET_FORMAT``):
    proof_one_hot  proven-win routes: the played (proven) move, probability 1; no KL to the H3 teacher there
    visits         tree: root visits normalised over the children, the proven-loss children (``UNSAFE`` in the
                   root, VCT2 or stage-VCT checks) set to 0 and the rest renormalised; KL to H3 applies
    none           forced routes, or a tree whose mass is all on proven losses: no policy loss for the position
The raw visits are kept beside the target (``raw_visits``), so a later target definition needs no new self-play.

Sampling of the first ``SAMPLE_PLIES`` moves after the opening (T = 1):
    proven win     never sampled: the proven move is played (event ``proven_win``)
    tree           a move drawn from the visit distribution above (event ``sampled``), proven losses excluded
    otherwise      the engine's own move (event ``no_distribution``, logged)
"""
from __future__ import annotations

TARGET_FORMAT = 'h7c-target-v1'
SAMPLING_FORMAT = 'h7c-sampling-v1'
SAMPLE_PLIES = 4
TEMPERATURE = 1.0
PROVEN_WIN_ROUTES = frozenset({'stage1', 'stage3', 'own_vcf', 'own_vct', 'own_vct2'})
FORCED_ROUTES = frozenset({'stage2', 'stage4', 'stage5'})
LOSS = 'UNSAFE'  # PROVEN_LOSS_VCT1 in analysis.mcts_v8's stored strings


def _move(m) -> tuple[int, int]:
    return int(m[0]), int(m[1])


def raw_visits(diagnostics) -> list[list]:
    """``[[row, col, visits, mean_value], ...]`` of the root children (empty off the tree)."""
    return [[int(m[0]), int(m[1]), int(n), round(float(q), 6)] for m, n, q in diagnostics.v8_root_visits]


def proven_losses(record: dict) -> list[list[int]]:
    """Root candidates the engine proved losing on this move (root, VCT2 and stage-VCT checks), sorted."""
    out = set()
    for part in ('root', 'vct2', 'vct'):
        for move, status in (record.get(part) or {}).get('checked', []):
            if status == LOSS:
                out.add(_move(move))
    return [list(m) for m in sorted(out)]


def visit_distribution(visits: list[list], losses: list[list[int]]) -> list[list] | None:
    """``[[row, col, p], ...]`` (row-major order) from the visits without the proven losses; None if no mass."""
    excluded = {_move(m) for m in losses}
    kept = sorted((_move(v[:2]), v[2]) for v in visits if v[2] > 0 and _move(v[:2]) not in excluded)
    total = sum(n for _, n in kept)
    if total <= 0:
        return None
    return [[m[0], m[1], n / total] for m, n in kept]


def policy_target(route: str, played, visits: list[list], losses: list[list[int]]) -> dict:
    """The policy target of one engine move (see the module docstring)."""
    if route in PROVEN_WIN_ROUTES:
        return {'format': TARGET_FORMAT, 'kind': 'proof_one_hot', 'kl': False,
                'target': [[int(played[0]), int(played[1]), 1.0]]}
    if route not in FORCED_ROUTES and visits:
        dist = visit_distribution(visits, losses)
        if dist is not None:
            return {'format': TARGET_FORMAT, 'kind': 'visits', 'kl': True, 'target': dist}
    return {'format': TARGET_FORMAT, 'kind': 'none', 'kl': False, 'target': None}


def sample_move(route: str, engine_move, visits: list[list], losses: list[list[int]], rng) -> tuple[list[int], str]:
    """The move to play during the sampled opening plies, and its event (see the module docstring)."""
    engine_move = [int(engine_move[0]), int(engine_move[1])]
    if route in PROVEN_WIN_ROUTES:
        return engine_move, 'proven_win'
    dist = visit_distribution(visits, losses) if route not in FORCED_ROUTES else None
    if dist is None:
        return engine_move, 'no_distribution'
    x, edge = rng.random(), 0.0
    for row, col, p in dist:
        edge += p
        if x < edge:
            return [row, col], 'sampled'
    return [dist[-1][0], dist[-1][1]], 'sampled'  # float rounding at the top edge
