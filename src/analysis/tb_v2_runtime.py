"""TB-V2-Value + fast4: the frozen runtime profile of the frozen engine (docs/h6-value.md, P1a-1 / P3).

Two identities, kept apart:
    engine   ``analysis.tb_v2_value`` (``ENGINE_HASH``): what the engine decides. Unchanged by P1a-1.
    runtime  this file (``runtime_hash(device)``): how it is run. The exact speed-up mode ``fast4`` (fast3's
             native threat windows + the native black forbidden points, P1a-1 PASS on the desktop paired
             gate and the SER5 stop rule), the native C sources and the Python modules that bind them
             (SHA-256 below, checked by ``tests/test_tb_v2_runtime.py``), and the value device.
The same engine run on CUDA (canonical: E3 ran its value on CUDA) and on CPU (the SER5 web machine) are two
runtime profiles with two runtime hashes. CPU is not move-identical to CUDA (3 / 827 corpus moves differ),
so it is used for the web only once its own certification passes (``CERTIFICATION``).

Built modules (``.so``) differ per machine (Python, compiler), so their SHA-256 is recorded, not gated:
``provenance`` returns it with the compiler and Python of the build. ``install`` puts this process on fast4
and checks what is actually bound; ``make_agent`` refuses to build an agent on any other binding.
"""
from __future__ import annotations

NAME = 'TB-V2-Value/fast4'
ENGINE = 'TB-V2-Value'
ENGINE_HASH = '256e5abe028c694891d33b9be1b9f58287fe18b6d4af278822b03cf09b7ec5c1'
FAST_MODE = 'fast4'
NATIVE_SOURCES = {
    'threat_windows.c': '2a1e309ca5d34b2d313cd86d3adc412613fc8b5ae671181a055315f8400fefe5',
    'forbidden.c': '211fd89d65db9963b8d1c31f09c4d235f927ad6744010672b0ae6c70d3435822',
}
# The Python side of fast4: what binds the native functions (and fast2's exact replacements under them).
BINDING_SOURCES = {
    'src/rt30/fast.py': '7a3ce59f4abf9f4cc7fe8c7044b71a41632f6ead7caf04b209282263a630cfee',
    'src/rt30/fast_windows.py': 'b82d55957f9a8ccbb4c18c7a42bddac56613b1c092db3781d9b2f4977405d609',
    'src/rt30/fast_extensions.py': 'd935eb813db49d4c777ed20fc92f6723549c16ddafe63bb6b49ee9af6599662d',
    'src/rt30/fast3.py': '36cc689639e7f5d67ef7b4a9969582f69ab2a7d7d8b3deae7c790428bb9d7f96',
    'src/rt30/fast3_windows.py': '309473704e7ecdda1628043bc85f083857dabbf33c9ff44cd163388fe80181bc',
    'src/rt30/fast4_forbidden.py': '2b268c5f17fcee7090e836129bac761155209f467f38cffdb77197f8112ee66d',
}
CANONICAL_DEVICE = 'cuda'
# Which value devices may serve: CUDA is the canonical backend (E3); CPU needs the CPU-backend
# certification (docs/h6-value.md, seeds 8431 + 8432) before web use.
CERTIFICATION = {'cuda': 'canonical (E3 ran its value on CUDA)', 'cpu': None}
EVIDENCE = {
    'function_check': 'docs/h7/h7_p1a_check.json',
    'desktop_p2_cpu': 'docs/h7/h7_p1a_p2_cpu.json',
    'desktop_paired_cuda': 'docs/h7/h7_p1a_paired_cuda.json',
    'ser5_r1': 'docs/h7/h7_p1a_ser5_fast4_r1_summary.json',
    'ser5_r2': 'docs/h7/h7_p1a_ser5_fast4_r2_summary.json',
}


def device_kind(device: str) -> str:
    """'cuda' for any CUDA device ('cuda', 'cuda:0'), else 'cpu'."""
    return 'cuda' if str(device).startswith('cuda') else 'cpu'


def runtime_hash(device: str) -> str:
    """SHA-256 of the canonical JSON of the engine hash, the fast mode, every gated source and the device."""
    import hashlib
    import json

    payload = {'engine_hash': ENGINE_HASH, 'fast_mode': FAST_MODE, 'native_sources': NATIVE_SOURCES,
               'binding_sources': BINDING_SOURCES, 'value_device': device_kind(device)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def check_bound() -> None:
    """Raise unless this process runs exactly fast4: the native threat windows and forbidden points bound
    where the engine looks them up, fast2's exact extensions, and a fresh build of the gated sources."""
    from hybrid import h6_fast
    from rt30 import fast3_windows, fast4_forbidden
    from rt30.native import build
    from search import mcts, mcts_v5, mcts_v7

    state = h6_fast.state()
    if state['active'] != h6_fast.expected_active(FAST_MODE):
        raise RuntimeError(f'{NAME} needs fast4 bound, this process runs {state["active"]}')
    if not (mcts_v5._threat_windows is fast3_windows.threat_windows_native
            and mcts_v7._threat_windows is fast3_windows.threat_windows_native
            and mcts.forbidden_reason is fast4_forbidden.forbidden_native
            and mcts_v7.forbidden_reason is fast4_forbidden.forbidden_native):
        raise RuntimeError(f'{NAME}: the engine modules do not bind the native functions')
    reason = build.stale()
    if reason is not None:
        raise RuntimeError(f'{NAME}: the native build is stale ({reason}); python -m rt30.native.build --if-stale')
    sources = (build.build_info() or {}).get('sources', {})
    if {k: sources.get(k) for k in NATIVE_SOURCES} != NATIVE_SOURCES:
        raise RuntimeError(f'{NAME}: the native sources are not the frozen ones: {sources}')


def install() -> dict:
    """Put this process on fast4 and check it (``check_bound``); returns ``h6_fast.state()``."""
    from hybrid import h6_fast

    state = h6_fast.install(FAST_MODE)
    check_bound()
    return state


def make_agent(policy, leaf_value, seed: int = 42):
    """TB-V2-Value's agent, refused unless this process runs the frozen fast4 profile."""
    from . import tb_v2_value

    if tb_v2_value.config_hash() != ENGINE_HASH:
        raise RuntimeError(f'{NAME}: analysis.tb_v2_value is not the frozen engine')
    check_bound()
    agent = tb_v2_value.make_agent(policy, leaf_value, seed=seed)
    agent.name = NAME
    return agent


def provenance(device: str) -> dict:
    """For game and web logs: both identities, the gated sources and the machine's build record."""
    from rt30.native import build

    info = build.build_info() or {}
    return {'engine': ENGINE, 'engine_hash': ENGINE_HASH, 'runtime': NAME, 'runtime_hash': runtime_hash(device),
            'fast_mode': FAST_MODE, 'value_device': device_kind(device),
            'certified': CERTIFICATION.get(device_kind(device)), 'native_sources': NATIVE_SOURCES,
            'native_binaries': info.get('modules'), 'native_python': info.get('python'),
            'native_compiler': info.get('compiler_version')}
