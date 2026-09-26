"""Search package with lazy (PEP 562) exports.

Submodules are imported only when an exported name is first accessed, so that
``import search.alphazero`` (Stage 5) does not load the frozen Stage 3 V2~V6
search modules or their threat planners. The exported objects are the exact
submodule objects; only the import timing changed.
"""
from importlib import import_module

_EXPORTS = {
    "MCTSNode": "mcts",
    "mcts_search": "mcts",
    "mcts_search_v3": "mcts_v3",
    "mcts_search_v32": "mcts_v32",
    "mcts_search_v321": "mcts_v321",
    "mcts_search_v4": "mcts_v4",
    "mcts_search_v5": "mcts_v5",
    "mcts_search_v6": "mcts_v6",
    "mcts_search_v7": "mcts_v7",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module_name}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
