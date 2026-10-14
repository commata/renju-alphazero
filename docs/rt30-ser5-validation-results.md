# SER5 RT30 Validation Results

- 작성 시각: 2026-10-14T23:30:27+09:00
- 서버: Beelink SER5 MAX / Ubuntu
- Git branch: `feat/Web_SER5`
- Git commit at collection: `c9856c518fa64ecfc017ea4a95046a9c7a7edc89`

## H3 Policy Checkpoint

TB-Engine-v1에 동결된 H3 policy checkpoint를 사용한다.

```text
runs/h3_policy_64x4/best.pt
SHA256: efee483284dfdd87eb6caa59d0ea3fd015c9dd5862e5e26925b9a23095d0ce61

runs/h3_policy_64x4/best.json
SHA256: 5648b7fbe71dfa4ab92f6c9bc505ee9ec0af68414f86b97ea22744c9fc022609
```

## H3 TB-Engine-v1 Equivalence

- Positions: **8**
- Identical: **8**
- Different: `[]`

검증 대상:

```text
frozen == fast == rt(deadline=None)
```

## Uniform-prior Supervisor 40-position

```json
{
  "seconds": {
    "n": 40,
    "median": 12.958,
    "p95": 26.001,
    "p99": 26.002,
    "max": 26.002,
    "sum": 551.7,
    "over_27.5s": 0
  },
  "routes": {
    "tree": 36,
    "stage4": 4
  },
  "sup_wall": {
    "sources": {
      "engine": 40
    },
    "watchdogs": 0,
    "over_30s": 0,
    "ipc_overhead": {
      "n": 40,
      "median": 0.0,
      "p95": 0.001,
      "p99": 0.001,
      "max": 0.001,
      "sum": 0.0,
      "over_27.5s": 0
    }
  },
  "rt_deadline": {
    "timeouts": {
      "attack": 12,
      "stage_vct": 0,
      "root": 7,
      "vct2": 6
    },
    "skipped": {
      "puct": 0,
      "vct2": 0
    },
    "moves_with_timeout_or_skip": 17
  }
}
```

## Uniform-prior Fast Baseline v2

```json
{
  "seconds": {
    "n": 40,
    "median": 13.344,
    "p95": 99.379,
    "p99": 127.817,
    "max": 127.817,
    "sum": 1109.6,
    "over_27.5s": 12
  },
  "routes": {
    "tree": 33,
    "own_vct": 3,
    "stage4": 4
  },
  "phases": {
    "attack": {
      "n": 40,
      "median": 0.181,
      "p95": 42.188,
      "p99": 43.129,
      "max": 43.129,
      "sum": 372.1,
      "over_27.5s": 8
    },
    "stage_vct": {
      "n": 4,
      "median": 5.91,
      "p95": 13.159,
      "p99": 13.159,
      "max": 13.159,
      "sum": 25.8,
      "over_27.5s": 0
    },
    "tree": {
      "n": 33,
      "median": 1.153,
      "p95": 2.251,
      "p99": 2.261,
      "max": 2.261,
      "sum": 42.7,
      "over_27.5s": 0
    },
    "root": {
      "n": 33,
      "median": 0.741,
      "p95": 85.153,
      "p99": 85.183,
      "max": 85.183,
      "sum": 422.3,
      "over_27.5s": 5
    },
    "vct2": {
      "n": 33,
      "median": 9.763,
      "p95": 11.835,
      "p99": 12.623,
      "max": 12.623,
      "sum": 239.0,
      "over_27.5s": 0
    },
    "v7_modules": {
      "n": 40,
      "median": 0.374,
      "p95": 42.429,
      "p99": 43.259,
      "max": 43.259,
      "sum": 379.6,
      "over_27.5s": 8
    },
    "v7_only": {
      "n": 40,
      "median": 0.064,
      "p95": 1.102,
      "p99": 1.462,
      "max": 1.462,
      "sum": 7.6,
      "over_27.5s": 0
    }
  }
}
```

## Uniform-prior RT30 27.5s v2

### RT latency / phases

```json
{
  "seconds": {
    "n": 40,
    "median": 12.885,
    "p95": 26.0,
    "p99": 26.001,
    "max": 26.001,
    "sum": 549.6,
    "over_27.5s": 0
  },
  "routes": {
    "tree": 36,
    "stage4": 4
  },
  "phases": {
    "attack": {
      "n": 40,
      "median": 0.185,
      "p95": 4.0,
      "p99": 4.0,
      "max": 4.0,
      "sum": 56.9,
      "over_27.5s": 0
    },
    "stage_vct": {
      "n": 4,
      "median": 6.079,
      "p95": 13.589,
      "p99": 13.589,
      "max": 13.589,
      "sum": 26.6,
      "over_27.5s": 0
    },
    "tree": {
      "n": 36,
      "median": 1.154,
      "p95": 2.231,
      "p99": 2.279,
      "max": 2.279,
      "sum": 46.0,
      "over_27.5s": 0
    },
    "root": {
      "n": 36,
      "median": 0.746,
      "p95": 20.104,
      "p99": 20.482,
      "max": 20.482,
      "sum": 186.2,
      "over_27.5s": 0
    },
    "vct2": {
      "n": 36,
      "median": 5.888,
      "p95": 12.175,
      "p99": 12.988,
      "max": 12.988,
      "sum": 226.1,
      "over_27.5s": 0
    },
    "v7_modules": {
      "n": 40,
      "median": 0.379,
      "p95": 4.6,
      "p99": 5.095,
      "max": 5.095,
      "sum": 64.5,
      "over_27.5s": 0
    },
    "v7_only": {
      "n": 40,
      "median": 0.064,
      "p95": 1.095,
      "p99": 1.45,
      "max": 1.45,
      "sum": 7.6,
      "over_27.5s": 0
    }
  }
}
```

### Deadline

```json
{
  "timeouts": {
    "attack": 12,
    "stage_vct": 0,
    "root": 7,
    "vct2": 6
  },
  "skipped": {
    "puct": 0,
    "vct2": 0
  },
  "moves_with_timeout_or_skip": 17
}
```

### Fast baseline comparison

```json
{
  "baseline": "runs/rt30/fast_uniform_40_v2.json",
  "baseline_arm": "fast",
  "positions": 40,
  "missing": 0,
  "same_move": 35,
  "changed_kinds": {
    "lost_win": 3,
    "played_baseline_unsafe": 1,
    "kept_tree_move": 2
  },
  "changed": [
    {
      "key": "puct_policy_vct2_50k_8415/9/26",
      "baseline": [
        "own_vct",
        [
          2,
          7
        ]
      ],
      "run": [
        "tree",
        [
          2,
          7
        ]
      ],
      "kind": "lost_win",
      "played_in_baseline": null
    },
    {
      "key": "puct_policy_vct2_50k_8415/10/20",
      "baseline": [
        "own_vct",
        [
          10,
          9
        ]
      ],
      "run": [
        "tree",
        [
          5,
          9
        ]
      ],
      "kind": "lost_win",
      "played_in_baseline": null
    },
    {
      "key": "puct_policy_vct2_50k_8415/14/15",
      "baseline": [
        "tree",
        [
          5,
          4
        ]
      ],
      "run": [
        "tree",
        [
          5,
          6
        ]
      ],
      "kind": "played_baseline_unsafe",
      "played_in_baseline": "UNSAFE"
    },
    {
      "key": "puct_policy_vct2_50k_8415/15/32",
      "baseline": [
        "own_vct",
        [
          5,
          11
        ]
      ],
      "run": [
        "tree",
        [
          4,
          8
        ]
      ],
      "kind": "lost_win",
      "played_in_baseline": null
    },
    {
      "key": "puct_policy_vct2_50k_8415/17/23",
      "baseline": [
        "tree",
        [
          9,
          11
        ]
      ],
      "run": [
        "tree",
        [
          8,
          11
        ]
      ],
      "kind": "kept_tree_move",
      "played_in_baseline": "UNKNOWN"
    },
    {
      "key": "puct_policy_vct2_50k_8415/22/37",
      "baseline": [
        "tree",
        [
          4,
          4
        ]
      ],
      "run": [
        "tree",
        [
          2,
          8
        ]
      ],
      "kind": "kept_tree_move",
      "played_in_baseline": "UNKNOWN"
    }
  ],
  "lost_own_vct": [
    {
      "key": "puct_policy_vct2_50k_8415/9/26",
      "win": [
        2,
        7
      ],
      "played": [
        2,
        7
      ],
      "same_move": true
    },
    {
      "key": "puct_policy_vct2_50k_8415/10/20",
      "win": [
        10,
        9
      ],
      "played": [
        5,
        9
      ],
      "same_move": false
    },
    {
      "key": "puct_policy_vct2_50k_8415/15/32",
      "win": [
        5,
        11
      ],
      "played": [
        4,
        8
      ],
      "same_move": false
    }
  ],
  "lost_own_vct_other_move": 2,
  "played_baseline_unsafe": [
    "puct_policy_vct2_50k_8415/14/15"
  ]
}
```

## V8-B benchmark: `v8b_cap4.json`

```json
{
  "positives": 67,
  "negatives": 0,
  "combos": {
    "frozen/fair": {
      "recall": "36/67",
      "won": 36,
      "same_move": 36,
      "positive_seconds": {
        "n": 67,
        "median": 2.898,
        "p95": 4.002,
        "max": 4.003,
        "sum": 163.6
      },
      "positive_seconds_when_won": {
        "n": 36,
        "median": 0.952,
        "p95": 3.244,
        "max": 3.555,
        "sum": 39.5
      },
      "negative_seconds": {},
      "timeouts": {
        "positive": 31,
        "negative": 0
      },
      "false_wins_on_negatives": 0
    },
    "frozen/widen": {
      "recall": "38/67",
      "won": 38,
      "same_move": 38,
      "positive_seconds": {
        "n": 67,
        "median": 1.865,
        "p95": 4.003,
        "max": 4.004,
        "sum": 156.7
      },
      "positive_seconds_when_won": {
        "n": 38,
        "median": 0.942,
        "p95": 3.151,
        "max": 3.662,
        "sum": 40.7
      },
      "negative_seconds": {},
      "timeouts": {
        "positive": 29,
        "negative": 0
      },
      "false_wins_on_negatives": 0
    }
  }
}
```

## V8-B benchmark: `v8b_cap8_orders.json`

```json
{
  "positives": 67,
  "negatives": 0,
  "combos": {
    "frozen/fair": {
      "recall": "48/67",
      "won": 48,
      "same_move": 48,
      "positive_seconds": {
        "n": 67,
        "median": 2.912,
        "p95": 8.004,
        "max": 8.007,
        "sum": 262.9
      },
      "positive_seconds_when_won": {
        "n": 48,
        "median": 1.418,
        "p95": 6.41,
        "max": 7.652,
        "sum": 110.8
      },
      "negative_seconds": {},
      "timeouts": {
        "positive": 19,
        "negative": 0
      },
      "false_wins_on_negatives": 0
    },
    "frozen/widen": {
      "recall": "48/67",
      "won": 48,
      "same_move": 46,
      "positive_seconds": {
        "n": 67,
        "median": 1.867,
        "p95": 8.003,
        "max": 8.005,
        "sum": 252.3
      },
      "positive_seconds_when_won": {
        "n": 48,
        "median": 1.318,
        "p95": 6.562,
        "max": 7.651,
        "sum": 100.3
      },
      "negative_seconds": {},
      "timeouts": {
        "positive": 19,
        "negative": 0
      },
      "false_wins_on_negatives": 0
    },
    "threes_first/fair": {
      "recall": "45/67",
      "won": 45,
      "same_move": 45,
      "positive_seconds": {
        "n": 67,
        "median": 4.531,
        "p95": 8.004,
        "max": 8.007,
        "sum": 281.1
      },
      "positive_seconds_when_won": {
        "n": 45,
        "median": 1.304,
        "p95": 6.867,
        "max": 7.652,
        "sum": 105.1
      },
      "negative_seconds": {},
      "timeouts": {
        "positive": 22,
        "negative": 0
      },
      "false_wins_on_negatives": 0
    },
    "threes_first/widen": {
      "recall": "47/67",
      "won": 47,
      "same_move": 46,
      "positive_seconds": {
        "n": 67,
        "median": 1.853,
        "p95": 8.001,
        "max": 8.007,
        "sum": 251.2
      },
      "positive_seconds_when_won": {
        "n": 47,
        "median": 1.304,
        "p95": 6.353,
        "max": 7.65,
        "sum": 91.2
      },
      "negative_seconds": {},
      "timeouts": {
        "positive": 20,
        "negative": 0
      },
      "false_wins_on_negatives": 0
    }
  }
}
```

## Raw Result Files

- `docs/rt30-results/fast_uniform_40_v2.json`
- `docs/rt30-results/fast_uniform_40_v2.log`
- `docs/rt30-results/fast_uniform_40_w1.json`
- `docs/rt30-results/fast_uniform_40_w1.log`
- `docs/rt30-results/h3_equivalence_8.json`
- `docs/rt30-results/h3_equivalence_8.log`
- `docs/rt30-results/rt27_uniform_40_v2.json`
- `docs/rt30-results/rt27_uniform_40_v2.log`
- `docs/rt30-results/rt27_uniform_40_w1.json`
- `docs/rt30-results/rt27_uniform_40_w1.log`
- `docs/rt30-results/rt27_uniform_8_w1.json`
- `docs/rt30-results/rt27_uniform_8_w1.log`
- `docs/rt30-results/smoke_uniform.json`
- `docs/rt30-results/smoke_uniform.log`
- `docs/rt30-results/sup27_uniform_40.json`
- `docs/rt30-results/sup27_uniform_40.log`
- `docs/rt30-results/v8b_cap4.json`
- `docs/rt30-results/v8b_cap4.log`
- `docs/rt30-results/v8b_cap8_orders.json`
- `docs/rt30-results/v8b_cap8_orders.log`

## Notes

- `uniform-prior` 결과는 tactical/deadline/SER5 성능 검증용이며 실제 TB-Engine-v1 policy prior와 동일하지 않다.
- H3 checkpoint를 사용한 결과가 실제 frozen TB-Engine-v1 기준 최종 검증이다.
- checkpoint binary 자체는 저장소에 커밋하지 않는다.
- 웹 최종 SLA는 supervisor의 request wall time 기준 `< 30s`, 위반 0을 목표로 한다.
