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

| position | fast(증명된 승리) | RT move | 같은 수 |
|---|---|---|---|
| `puct_policy_vct2_50k_8415/9/26` | `[2, 7]` | `[2, 7]` | 예(증명 없음) |
| `puct_policy_vct2_50k_8415/10/20` | `[10, 9]` | `[5, 9]` | 아니오 |
| `puct_policy_vct2_50k_8415/15/32` | `[5, 11]` | `[4, 8]` | 아니오 |

(첫 판의 표는 생성 스크립트가 없는 키를 읽어 `None`으로 나왔다. 값은 위 JSON의 `win`/`played`다.)

## 4. 해석 기준

- `fast`는 exact 최적화만 적용한 baseline이다.
- `rt`는 같은 엔진 흐름에 27.5초 absolute deadline을 적용한 실시간 후보이다.
- `vs_baseline.lost_own_vct`는 fast에서는 증명된 승리였으나 RT30에서는 시간 제한 때문에 그 증명을 잃은 국면이다.
- `rt.seconds.max`와 `over_27.5s`는 engine-level latency 검증 기준이다.
- 실제 웹 30초 SLA는 이후 supervisor 단계에서 `wall_seconds`로 최종 검증한다.
- `v7_only`를 순수 V7 비용으로 사용하며, 과거 `v7_modules`는 V8-B 시간이 포함돼 직접 비교하지 않는다.
- 이 결과는 slow position 40개에 대한 stress-oriented sample이며 전체 게임 분포와 동일하지 않다.

## 5. 검토 메모 (2026-10-14)

### 5.1 바뀐 수 5개의 분류

fast(seed 42)로 같은 국면을 다시 돌려 fast 자신의 검사 결과(V8-C·VCT2 status)로 RT의 수를 분류했다.
V8-B와 PUCT는 같은 seed에서 결정적이라, 컨테이너에서도 SER5와 같은 수가 나왔다.

| 국면 | 분류 | fast | RT | fast의 검사에서 RT 수의 상태 |
|---|---|---|---|---|
| `10/20` | 승리 손실 | own_vct `[10, 9]` | `[5, 9]` | (V8-B 증명 대상 아님. 오프라인: REFUTED·SAFE) |
| `15/32` | 승리 손실 | own_vct `[5, 11]` | `[4, 8]` | — |
| `14/15` | **증명된 패배 수를 둠** | `[5, 4]`(SAFE) | `[5, 6]` | **UNSAFE** |
| `17/23` | tree 수 유지 | `[9, 11]`(SAFE로 교체) | `[8, 11]` | UNKNOWN |
| `22/37` | tree 수 유지 | `[4, 4]`(SAFE로 교체) | `[2, 8]` | UNKNOWN |

(`9/26`은 수가 같고 route만 다르다.)

- **`14/15`가 가장 심각하다.** tree 수 `[5, 8]`이 VCT1 패배로 증명된 뒤, RT는 자식 검사 도중 deadline에 걸렸다.
  동결 fallback 규칙("첫 UNKNOWN 자식, 즉시 지지 않는 수")대로 `[5, 6]`을 뒀는데, fast는 이 수를 **UNSAFE로 증명했다**.
  timeout 아래에서 UNKNOWN은 대부분 "검사하지 못함"이라, 이 fallback은 안전성 면에서 거의 무작위다.
  V8-B와 별도로 다뤄야 할 V8-C 문제다.
- `17/23`, `22/37`은 aggressive 모드의 "증명된 SAFE 자식으로 교체"를 시간 안에 못 한 경우다. 둔 수는 패배로 증명되지 않았다.
  H1(§12.9)에서도 이런 수가 실제로는 안전했던 사례가 있었다. 기력 영향은 체크포인트를 넣은 대국 gate로만 판단할 수 있다.
- replay `--baseline`은 이제 이 분류(`changed_kinds`, `played_baseline_unsafe`)를 자동으로 낸다.
  baseline 행에 검사 status(`checked`)와 tree 수가 기록되므로 baseline도 새 코드로 다시 만들어야 한다.

### 5.2 deadline 개입 빈도

40국면 중 17수(42.5%)에서 timeout 또는 skip이 있었다. 이 표본은 기록상 30초를 넘은 국면만 모은 stress set이라,
대국 전체의 비율이 아니다. 대국 전체에서의 비율은 웹 DB의 `timed_out`으로 본다.

### 5.3 supervisor 경유 측정 (컨테이너, 4국면)

`--arms sup`(worker 프로세스 + IPC + watchdog, 웹 서버와 같은 경로)으로 위 4국면을 돌렸다.
결과는 wall 최대 26.004초, watchdog 0, IPC overhead 최대 3ms다. 바뀐 수의 분류도 위 표와 같다.

## 6. 원본 결과

- `runs/rt30/fast_uniform_40_w1.json`
- `runs/rt30/rt27_uniform_40_w1.json`

`runs/`는 Git ignore 대상이므로 저장소에는 이 Markdown 요약만 커밋한다.
