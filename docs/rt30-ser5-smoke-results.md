# SER5 RT30 Smoke Test Results

- 작성 시각: 2026-10-14T16:47:00+09:00
- 서버: SER5 Ubuntu
- Git branch: `feat/Web_SER5`
- Git commit: `c19f31f965e90b86c9585477a27a6b456e1a3483`
- H3 checkpoint: 없음
- PUCT prior: `uniform`

> 이 결과는 H3 policy checkpoint가 없는 상태의 사전 검증이다. 전술 탐색 및 RT30 흐름 검증에는 사용할 수 있지만, 실제 TB-Engine-v1의 최종 성능/기력 평가는 checkpoint를 넣은 뒤 다시 수행해야 한다.

## 1. Frozen / Fast / RT 동일성 및 고속화

실행 조건:

```bash
python scripts/run_rt30_replay.py \
  --uniform-prior \
  --arms frozen,fast,rt \
  --select slow \
  --limit 8 \
  --workers 1 \
  --output runs/rt30/smoke_uniform.json
```

- Positions: **8**
- Identical: **8/8**
- Different: `[]`
- Fast speedup vs frozen: **2.476x**
- RT(no deadline) speedup vs frozen: **2.481x**

### frozen

| metric | value |
|---|---:|
| n | 8 |
| median | 33.556 |
| p95 | 110.735 |
| p99 | 110.735 |
| max | 110.735 |
| sum | 340.5 |
| over_27.5s | 6 |

### fast

| metric | value |
|---|---:|
| n | 8 |
| median | 13.072 |
| p95 | 45.028 |
| p99 | 45.028 |
| max | 45.028 |
| sum | 137.5 |
| over_27.5s | 1 |

### 단계별 주요 관찰

| phase | frozen median | fast median | frozen max | fast max |
|---|---:|---:|---:|---:|
| attack | 1.24 | 0.496 | 108.4 | 43.193 |
| tree | 1.078 | 1.089 | 1.176 | 1.191 |
| root | 0.847 | 0.419 | 20.295 | 8.275 |
| vct2 | 21.259 | 8.817 | 36.711 | 11.859 |
| v7_modules | 1.373 | 0.583 | 108.686 | 43.323 |

## 2. RT30 27.5초 Deadline Smoke Test

실행 조건:

```bash
python scripts/run_rt30_replay.py \
  --uniform-prior \
  --arms rt \
  --deadline 27.5 \
  --select slow \
  --limit 8 \
  --workers 1 \
  --output runs/rt30/rt27_uniform_8_w1.json
```

### 실제 착수 시간

| metric | value |
|---|---:|
| n | 8 |
| median | 12.462 |
| p95 | 26.001 |
| p99 | 26.001 |
| max | 26.001 |
| over_27_5 | 0 |
| over_30 | 0 |

### Deadline 동작

```json
{
  "timeouts": {
    "attack": 3,
    "stage_vct": 0,
    "root": 1,
    "vct2": 1
  },
  "skipped": {
    "puct": 0,
    "vct2": 0
  },
  "moves_with_timeout_or_skip": 3
}
```

### 국면별 결과

| position | route | seconds | move | timeouts | skipped |
|---|---|---:|---|---|---|
| `puct_policy_vct2_50k_8415/2/14` | tree | 13.0563 | `[6, 6]` | `[]` | `[]` |
| `puct_policy_vct2_50k_8415/9/18` | tree | 5.8351 | `[4, 10]` | `['attack']` | `[]` |
| `puct_policy_vct2_50k_8415/9/26` | tree | 6.4348 | `[2, 7]` | `['attack']` | `[]` |
| `puct_policy_vct2_50k_8415/10/18` | tree | 12.9682 | `[5, 9]` | `[]` | `[]` |
| `puct_policy_vct2_50k_8415/10/20` | tree | 26.0011 | `[5, 9]` | `['attack', 'root', 'vct2']` | `[]` |
| `puct_policy_vct2_50k_8415/11/9` | tree | 9.6677 | `[8, 11]` | `[]` | `[]` |
| `puct_policy_vct2_50k_8415/11/11` | tree | 11.9566 | `[4, 6]` | `[]` | `[]` |
| `puct_policy_vct2_50k_8415/11/13` | tree | 13.9303 | `[8, 7]` | `[]` | `[]` |

## 3. 해석

- 8개 slow position에서 `frozen`, `fast`, `rt(deadline 없음)`의 결과가 모두 동일했다.
- SER5에서 exact `_threat_windows` 최적화는 이 표본 기준 frozen 대비 약 2.48배의 전체 속도 향상을 보였다.
- PUCT tree 시간은 거의 변하지 않아 현재 최적화가 주로 전술 탐색 비용을 줄이고 있음을 확인했다.
- Deadline 실행에서 timeout은 증명을 SAFE/WIN/UNSAFE로 잘못 확정하지 않고 UNKNOWN/fallback 경로로 처리된다.
- 이 표본은 파일 순서상 처음 8개 slow position이므로 전체 성능을 일반화하지 않는다.
- H3 checkpoint가 없으므로 실제 TB-Engine-v1의 V8-C/VCT2 순서 및 최종 성능은 checkpoint 적용 후 다시 측정해야 한다.

## 4. 원본 결과 파일

- `runs/rt30/smoke_uniform.json`
- `runs/rt30/rt27_uniform_8_w1.json`

두 파일은 `runs/`가 Git ignore 대상이므로 저장소에는 이 Markdown 요약만 커밋한다.
