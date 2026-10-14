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

> `v7_modules`(`v7_module_seconds`)는 own VCF 직전부터 V7 root 준비가 끝날 때까지 재는 타이머라 **V8-B 시간이 들어 있다**.
> 순수 V7 시간으로 읽지 않는다. `c19f31f`부터 replay 출력에 `v7_only`(= v7_modules − attack)가 따로 나온다.
> 컨테이너 같은 8국면에서 `v7_only`는 0.04~0.31초였다.

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

## 4. 검토 메모 (2026-10-14)

### 4.1 deadline이 V8-B 승리 증명을 끊은 2국면

V8-B는 tree보다 먼저 돌고 난수를 쓰지 않는다. 그래서 같은 국면이면 기계·seed와 관계없이 결과가 같다.
이 8국면 중 fast(=frozen)가 VCT1 승리를 증명한 국면은 `8415/9/26`과 `8415/10/20` 두 개다. RT는 두 국면 모두 V8-B가 잘려 tree route로 갔다.

| 국면 | fast(증명된 승리) | SER5 RT(seed 42) | 컨테이너 RT(seed 7) |
|---|---|---|---|
| `8415/9/26` | own_vct `[2, 7]` | tree `[2, 7]`(같은 수, 증명 없음) | tree `[1, 5]` |
| `8415/10/20` | own_vct `[10, 9]` | tree `[5, 9]` | tree `[4, 9]` |

- SER5에서 `9/26`이 같은 수였던 것은 PUCT seed에 따른 우연이다. 같은 seed로 비교하면 두 국면 모두 다른 수를 둔다.
- `10/20`의 `[5, 9]`를 오프라인으로 다시 확인했다. VCT1 공격은 REFUTED(승리 아님)이고, 안전성은 SAFE다(75k 노드).
  `[10, 9]`는 WIN이다. 즉 **증명된 승리를 놓친 실제 기력 손실**이다.
- `[10, 9]` 한 수의 증명은 18k 노드, 7.1초면 끝난다. 그런데 V8-B 전체가 32초(SER5 fast 최대 43초)나 걸리는 이유는
  공정 배분 스케줄러가 예산을 여러 후보에 나눠 쓰기 때문이다. 따라서 cap만 4초 → 8초로 올려서는 부족할 수 있다.
  후보 순서와 배분 방식을 함께 봐야 한다(동작 변경이므로 gate 필요).
- 컨테이너에서 `--timing attack_seconds=12`(seed 7)로 돌려도 두 승리를 모두 놓쳤다. 최대 26.0초, 27.5초 초과 0이다.
  cap만 올리는 방법의 한계를 보여준다.
- 컨테이너에서 같은 seed(7)로 fast와 RT를 비교하면, 나머지 6국면은 RT가 deadline 상황에서도 fast와 **같은 수**를 뒀다.
  다만 그중 timeout이 난 수는 `9/18`(attack) 하나뿐이다.
- 앞으로는 `--baseline <fast run JSON>`(같은 `--seed` 필수)으로 돌리면 `vs_baseline.lost_own_vct`가 자동으로 나온다.

### 4.2 26.0초는 설계상의 상한이다

마지막 단계(VCT2, V8-A)는 `deadline − final_margin(1.5초)`까지 쓴다. 그래서 timeout이 난 수는 27.5 − 1.5 = 26.0초 근처에서
끝난다(26.0011초). 행운이 아니라 설계값이다. 여유 1.5초는 응답·IPC 몫으로 남겨 둔 것이다.

### 4.3 엔진 deadline과 웹 30초 SLA는 다르다

이 값은 replay의 `select_move` 시간이다. 사용자가 체감하는 값은 DB의 `wall_seconds`(요청 수신 → 수 준비)다.

- 대기열 시간은 deadline을 늘리지 않는다. deadline이 **요청 수신 시점의 절대 시각**이기 때문이다(§3.2).
  대기 시간만큼 탐색 시간이 줄어든다.
- 그래서 30초를 넘을 수 있는 경로는 셋이다.
  1. 대기만으로 PUCT guard 아래까지 남은 시간이 줄어든 경우(짧은 응답 경로가 필요)
  2. deadline이 닿지 않는 frozen 구간의 overrun(V7 모듈, PUCT)
  3. 응답·IPC 지연
- 이 셋은 supervisor 구현 후 포화 측정에서 `wall_seconds`로 확인한다.

### 4.4 통계의 한계

- 8국면이라 p95 = p99 = max다. 분포는 40국면 이상에서 본다.
- 파일 순서상 첫 8개 slow 국면이고, uniform prior다(V8-C·VCT2의 자식 순서는 진짜 TB1과 다를 수 있다).

## 5. 원본 결과 파일

- `runs/rt30/smoke_uniform.json`
- `runs/rt30/rt27_uniform_8_w1.json`

두 파일은 `runs/`가 Git ignore 대상이므로 저장소에는 이 Markdown 요약만 커밋한다.
