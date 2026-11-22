"""Finish the original fast5 plan after the Linux session ended during B3."""
import hashlib
import json
from pathlib import Path

import h7r_budget_diag_fast5 as run

folder = run.ROOT / 'docs/h7r/budget/TB-V3-BUDGET-DIAG-FAST5-1'
plan = json.loads((folder / 'deepen-plan.json').read_bytes())
calibration = (folder / 'calibrate.json').read_bytes()
assert hashlib.sha256(calibration).hexdigest() == plan['calibration_sha256']
assert hashlib.sha256(Path(run.__file__).read_bytes()).hexdigest() == plan['tool_sha256']
_, expected, inputs = run.evidence()
assert plan['inputs'] == inputs and plan['expected_b0'] == expected
runtime = json.loads(calibration)['B0']['runtime']
results = {}
for name, limit in plan['limits_seconds'].items():
    path = folder / f'{name}.json'
    if path.exists():
        stored = json.loads(path.read_bytes())
        assert all(stored[key] == value for key, value in plan.items())
        row = {key: value for key, value in stored.items() if key not in plan}
    else:
        assert name == 'B3'
        row = run.check(name, limit)
        run.write_new(path, {**plan, **row})
    assert row['budget'] == run.budgets()[name] and row['limit_seconds'] == limit
    assert row.get('runtime', runtime) == runtime
    results[name] = row
    print(f"{name} {row['result']['status']} ({row['wall_seconds']} s)", flush=True)

statuses = {row['result']['status'] for row in results.values()}
proven = statuses & {'PROVEN_LOSS', 'VCT2_CLEAR'}
final = ('CONFLICT' if len(proven) > 1 else next(iter(proven)) if proven else
         'INCOMPLETE' if 'TIMEOUT' in statuses else 'UNKNOWN')
run.write_new(folder / 'deepen.json',
              {**plan, 'results': results, 'final': final,
               'interrupted_attempt': {'budget': 'B3', 'reason': 'Linux session exited without a result'},
               'resume_tool_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
print(final, flush=True)
