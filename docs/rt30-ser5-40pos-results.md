# SER5 RT30 40-Position Results

- 작성 시각: 2026-10-14T20:26:31+09:00
- 서버: SER5 Ubuntu
- Git branch: `feat/Web_SER5`
- Git commit: `7e9239744b7473b3a0ad31e3fd40a8f5b8b2a10d`
- H3 checkpoint: 없음
- PUCT prior: `uniform`
- workers: `1`
- seed: `42`

> 이 결과는 H3 checkpoint가 없는 uniform-prior 사전 검증이다. 전술 탐색, latency, deadline 동작과 fast 대비 RT30 변화 분석에는 사용할 수 있지만, 실제 TB-Engine-v1의 최종 기력 평가는 checkpoint 적용 후 다시 수행해야 한다.

## 1. Fast baseline

```bash
python scripts/run_rt30_replay.py \
  --uniform-prior \
  --arms fast \
  --select slow \
  --limit 40 \
  --workers 1 \
  --seed 42 \
  --output runs/rt30/fast_uniform_40_w1.json
```

| metric | value |
|---|---:|
| n | 40 |
| median | 13.35 |
| p95 | 99.936 |
| p99 | 127.949 |
| max | 127.949 |
| sum | 1111.9 |
| over_27.5s | 12 |

## 2. RT30 deadline=27.5s

```bash
python scripts/run_rt30_replay.py \
  --uniform-prior \
  --arms rt \
  --deadline 27.5 \
  --select slow \
  --limit 40 \
  --workers 1 \
  --seed 42 \
  --baseline runs/rt30/fast_uniform_40_w1.json \
  --output runs/rt30/rt27_uniform_40_w1.json
```

### Latency

| metric | value |
|---|---:|
| n | 40 |
| median | 13.049 |
| p95 | 26.001 |
| p99 | 26.001 |
| max | 26.001 |
| sum | 553.3 |
| over_27.5s | 0 |

### Routes

```json
{
  "tree": 36,
  "stage4": 4
}
```

### V7 only

```json
{
  "n": 40,
  "median": 0.065,
  "p95": 1.104,
  "p99": 1.467,
  "max": 1.467,
  "sum": 7.7,
  "over_27.5s": 0
}
```

### Deadline events

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

## 3. Fast baseline vs RT30

```json
{
  "baseline": "runs/rt30/fast_uniform_40_w1.json",
  "baseline_arm": "fast",
  "positions": 40,
  "missing": 0,
  "same_move": 35,
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
      ]
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
      ]
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
      ]
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
      ]
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
      ]
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
      ]
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
  "lost_own_vct_other_move": 2
}
```

### Lost proven wins

| position | fast move | RT move | fast route | RT route |
|---|---|---|---|---|
| `puct_policy_vct2_50k_8415/9/26` | `None` | `None` | None | None |
| `puct_policy_vct2_50k_8415/10/20` | `None` | `None` | None | None |
| `puct_policy_vct2_50k_8415/15/32` | `None` | `None` | None | None |

## 4. 해석 기준

- `fast`는 exact 최적화만 적용한 baseline이다.
- `rt`는 같은 엔진 흐름에 27.5초 absolute deadline을 적용한 실시간 후보이다.
- `vs_baseline.lost_own_vct`는 fast에서는 증명된 승리였으나 RT30에서는 시간 제한 때문에 그 증명을 잃은 국면이다.
- `rt.seconds.max`와 `over_27.5s`는 engine-level latency 검증 기준이다.
- 실제 웹 30초 SLA는 이후 supervisor 단계에서 `wall_seconds`로 최종 검증한다.
- `v7_only`를 순수 V7 비용으로 사용하며, 과거 `v7_modules`는 V8-B 시간이 포함돼 직접 비교하지 않는다.
- 이 결과는 slow position 40개에 대한 stress-oriented sample이며 전체 게임 분포와 동일하지 않다.

## 5. 원본 결과

- `runs/rt30/fast_uniform_40_w1.json`
- `runs/rt30/rt27_uniform_40_w1.json`

`runs/`는 Git ignore 대상이므로 저장소에는 이 Markdown 요약만 커밋한다.
