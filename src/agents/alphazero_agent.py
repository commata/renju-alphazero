"""AlphaZero PUCT agent for human play, with per-move root diagnostics.

Deterministic (noise off, temperature 0), the same search as external evaluation
(``training.evaluation.PUCTAgent``). Each move records enough of the root to tell
apart, when a human wins, whether the policy missed a move (prior), the value was
wrong (NN value / Q) or PUCT never visited it (visits):

- ``root_visits``: every visited child, ``"row,col" -> visits`` (0-based);
- ``top_visits``: the ``top_k`` most visited children with visits, prior and Q
  (Q from the mover's view, i.e. the side to move at the root);
- ``top_priors``: the ``top_k`` root priors;
- ``root_value``: the evaluator value of the root (side to move's view);
- ``root_q``: visit-weighted mean child Q (side to move's view);
- ``tactical_rules`` / ``tactical_allowed`` / ``tactical_proven``: whether PUCT v2
  rules were on, how many root children they allowed and the rule-proven root value.

The evaluator is injected, so this module stays torch-free; ``from_checkpoint``
imports torch lazily.
"""
from __future__ import annotations

from pathlib import Path
from time import perf_counter

from model.config import action_to_coordinate
from search.alphazero import SearchConfig, search_with_tree, select_action
from search.tactics import tactical_filter


def _key(action: int) -> str:
    row, col = action_to_coordinate(action)
    return f'{row},{col}'


class AlphaZeroAgent:
    def __init__(self, evaluator, search_config: SearchConfig, *, name: str = 'AlphaZero',
                 top_k: int = 8, info: dict | None = None):
        if search_config.noise_enabled or search_config.temperature_moves != 0:
            raise ValueError('play PUCT requires noise OFF and temperature 0')
        self.evaluator = evaluator
        self.config = search_config
        self.name = name
        self.top_k = top_k
        self.info = dict(info or {})
        self.diagnostics: dict = {}

    @classmethod
    def from_checkpoint(cls, path: str | Path, *, simulations: int | None = None,
                        tactical_rules: str = 'auto', device: str = 'cpu',
                        name: str | None = None) -> 'AlphaZeroAgent':
        """Load a Stage 6+ training checkpoint (model + its evaluation search settings)."""
        from model.evaluator import PolicyValueEvaluator
        from training.probes import load_model_from_training_checkpoint
        from training.training_checkpoint import load_checkpoint_payload

        model, info = load_model_from_training_checkpoint(path)
        evaluation = load_checkpoint_payload(path)['config']['evaluation']
        rules = (evaluation.get('tactical_rules', False) if tactical_rules == 'auto'
                 else tactical_rules == 'on')
        config = SearchConfig(
            num_simulations=simulations or evaluation['puct_simulations'],
            c_puct=evaluation['c_puct'], temperature_moves=0, noise_enabled=False,
            tactical_rules=rules)
        label = name or f"AlphaZero-gen{info['generation']}"
        return cls(PolicyValueEvaluator(model, device=device), config, name=label,
                   info={**info, 'search': config.to_dict()})

    def select_move(self, game) -> tuple[int, int]:
        started = perf_counter()
        result, root = search_with_tree(game, self.evaluator, self.config, None)
        action = select_action(result, len(game.history), self.config, None)
        diagnostics = {
            'simulations': self.config.num_simulations,
            'fast_path': result.fast_path,
            'chosen': list(action_to_coordinate(action)),
            'tactical_rules': self.config.tactical_rules,
            'search_seconds': perf_counter() - started,
        }
        if self.config.tactical_rules:
            allowed, proven = tactical_filter(game, game.legal_moves())
            diagnostics['tactical_allowed'] = len(allowed)
            diagnostics['tactical_proven'] = proven
        if root is not None:
            visited = {a: child for a, child in root.children.items() if child.visit_count}
            total = sum(child.visit_count for child in visited.values())
            diagnostics['root_value'] = root.nn_value
            diagnostics['root_q'] = (sum(c.visit_count * c.q for c in visited.values()) / total
                                     if total else None)
            diagnostics['root_visits'] = {_key(a): c.visit_count for a, c in sorted(visited.items())}
            by_visits = sorted(visited.items(),
                               key=lambda item: (-item[1].visit_count, -item[1].prior, item[0]))
            diagnostics['top_visits'] = [
                {'move': list(action_to_coordinate(a)), 'visits': c.visit_count,
                 'prior': c.prior, 'q': c.q}
                for a, c in by_visits[:self.top_k]]
            by_prior = sorted(root.children.items(), key=lambda item: (-item[1].prior, item[0]))
            diagnostics['top_priors'] = [
                {'move': list(action_to_coordinate(a)), 'prior': c.prior,
                 'visits': c.visit_count}
                for a, c in by_prior[:self.top_k]]
        self.diagnostics = diagnostics
        return action_to_coordinate(action)
