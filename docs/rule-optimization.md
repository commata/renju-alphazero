# 렌주 규칙 엔진 최적화 검증

## 범위와 기준

2026-09-25, `perf/renju-legal-moves`, Python 3.13.14 / Windows 11에서 측정했다.
기준 커밋은 `9f48d146111bd1c1a6154c58dbd10b7aaa26da13`이다.
CPU 식별 출력은 `Intel64 Family 6 Model 170 Stepping 4, GenuineIntel`이다.

`tests/reference_rules.py`는 기준 `src/renju/rules.py`의 독립적인 전체 복사본이다.
두 파일의 기준 Git blob hash는 `7bdd91285ffdb4eb39e007c79485a731da20c69b`로 같다.
oracle은 production 함수를 import하지 않으며, production도 oracle을 import하지 않는다.
동결 이후 oracle은 수정하지 않았다. 합법수 기대 목록은 oracle의 빈칸별 판정에
기존 첫 수 중앙 고정, 종료 상태, 백 차례 규칙을 적용해 행 우선으로 구성한다.

## 변경과 정확성 근거

1. `_open_three()` 방향 prefilter: move와 한 개의 extension이 길이 4의 연속
   흑돌 구간을 만들려면 기존 흑돌 두 개가 move의 해당 방향 ±3 안에 있어야 한다.
   이 필요조건을 만족하지 못하는 방향만 제외한다. 만족하면 기존 재귀 판정을 실행한다.
2. `_quiet_black_point()`: 각 방향 ±4 안의 다른 흑돌 수가 모두 2 이하이고,
   두 개 있는 방향이 하나 이하일 때만 안전하다고 판정한다.
   - move를 지나는 장목은 ±4 안에 적어도 네 개의 기존 흑돌이 필요하다.
   - move를 포함한 four의 5칸 window에는 다른 흑돌 세 개가 필요하다.
   - 열린 삼은 같은 방향 ±3 안에 다른 흑돌 두 개가 필요하므로 삼삼에는 그런 방향이 두 개 필요하다.
   백돌에 막혀 있어도 흑돌을 모두 세므로 과대 계산만 가능하다. False는 기존 정확 판정기로 보낸다.
3. `_NEIGHBORS4`는 보드 내용과 무관한 좌표 tuple만 저장한다.
   fast-path는 `forbidden_reason()`의 기존 입력 검사 뒤, 임시 착수 전에 위치한다.
   반환값·예외 메시지와 보드 복원 동작을 유지한다.

production 변경은 `rules.py`의 38줄 추가뿐이다. 모든 빈칸이 여전히 합법수 검사 대상이다.
`Game.legal_moves()`, `Game.play()`, `has_legal_move()`와 검색/agent 코드는 그대로다.
candidate pruning, 근사 판정, locality 기반 합법수 제한, heuristic 의존, board cache는 없다.
재귀적 extension은 이미 four를 만들었으므로 quiet 조건을 만족할 수 없다.
해당 내부 경로에 fast-path를 중복 적용하지 않고 방향 prefilter로 재귀 비용을 줄였다.

목표 성능을 충족하여 삼삼 루프 조기 종료, straight-four/four window geometry 사전 계산은
추가하지 않았다. Policy-Value Network 등 Stage 4 구현도 이번 변경에 포함하지 않았다.

## 검증 구성과 재현

```powershell
python -m unittest discover -s tests -v
python scripts/validate_rule_optimization.py --seed 42 --positions 10000
python scripts/validate_rule_optimization.py --seed 2026 --positions 1000
python scripts/benchmark_rule_optimization.py --seed 42 --smoke
python scripts/benchmark_rule_optimization.py --seed 42 --iterations 20 --repeats 5 --games 2
python scripts/benchmark_engine.py --iterations 100 --games 10 --seed 42 --profile
python -m compileall -q src tests scripts
git diff 9f48d14 --check
```

기본 differential test는 seed 42의 240개 상태를 검사한다. 추가된 다른 test는
음수·범위 밖·이미 점유된 좌표의 예외 유형/메시지와 보드 보존을 비교한다.
기존 147개 테스트는 삭제하거나 약화하지 않았다.

corpus는 다음으로 구성된다.

- Random/Tactical 합법 대국의 다양한 ply. 게임이 끝나면 새 게임을 시작하며 seeded RNG 스트림은 계속 진행한다.
- 최적화 전에 저장한 seed 42/43의 V6 대국 전체 기보를 재생한 19개 상태.
- 중앙 밀집 보드와 전체 보드 synthetic 상태. 점유율 20~80%, 점유 칸 중 흑돌 확률 80%.
- 기존 규칙/위협 test에서 가져온 12개 패턴의 4회 회전: double-three, blocked-three,
  recursive double-three, double-four, same-axis multiple four, overline/exact-five 우선순위,
  exact-five/fork 우선순위, edge three/four, cross-overline completion,
  사사/재귀 삼삼에 의해 금지되는 extension.

각 상태의 **모든 빈칸**에 oracle과 production을 각각 호출하고, 각 호출 직후
전체 보드가 같은지 확인한다. 원래 차례와 무관하게 흑·백 양쪽 `Game.legal_moves()`를
oracle 기대 **list 전체**와 비교한다. 종료 상태도 포함한다.
실패하면 위치·통계와 함께 즉시 비정상 종료한다. 금수 유형별 100건 미만이면 경고한다.
유한 differential 검증만으로 모든 보드에 대한 증명을 주장하지 않는다.
위 필요조건에 대한 논리와 회귀·다양한 보드 검증을 함께 정확성 근거로 삼는다.

## 측정 방법

`benchmark_rule_optimization.py`는 warm-up과 준비 작업을 제외한 `perf_counter()`를 쓴다.
60-ply 보드 4개(seed 42, 43, 44, 45), midgame과 회전 fixture의 대표 금수 검사 52개,
seeded Random 완결 대국 2개를 사용한다. reference → optimized 순서로 5회 교대한다.
합법수·금수 검사는 각 묶음을 20회 반복한다. 출력 mean/median/min/max는 반복 묶음의
평균 호출 시간에 대한 통계이고, 완결 대국은 10회 개별 대국 시간에 대한 통계이다.

reference 측정은 테스트 전용 context manager로 현재 import된 규칙 함수 alias들을
독립 oracle 함수로 교체한다. `finally`에서 복원하며 production 코드는 수정하지 않는다.
V6 smoke도 동일 방식으로 direct/private rule helper alias까지 교체한다.

기존 `benchmark_engine.py`의 10판 대국과 교대 benchmark의 2판 반복은 seed 구성과
대국 길이가 다르므로 서로의 moves/sec를 혼합하지 않는다. cProfile은 별도 75-ply
Random 대국에서 호출 수와 병목만 확인하는 데 사용한다. 측정 중 다른 검증 프로세스는 실행하지 않았다.
사용자 제공 과거 9.851 ms/239.2 moves/sec와 달리 이번 세션 baseline은
8.157 ms/260.918 moves/sec였다. 배수는 이번 세션의 비교 가능한 값으로 계산한다.

## 결과

### 단계별 결과

| 단계 | 전체 tests | differential | 단일 midgame 흑 ms | 4개 midgame 흑 평균 ms | Random 10판 moves/sec |
|---|---:|---|---:|---:|---:|
| 최적화 전 | 149 PASS | unit 240 상태 PASS | 8.157 | 8.589 | 260.918 |
| open-three prefilter | 149 PASS | 1,000 상태 / 166,515 빈칸 PASS | 5.148 | 5.186 | 398.453 |
| + safe fast-path | 149 PASS | 1,000 상태 / 166,515 빈칸 PASS | 2.204 | 2.404 | 898.869 |
| 최종 재검증 | 149 PASS | 10,000 + 별도 seed 1,000 상태 PASS | 2.504 | 2.435 | 855.209 |

모든 단계에서 규칙/합법수 불일치와 복원 실패는 0이었다. 기준 147개와 추가 2개 모두 통과했다.
최종 전체 unit suite는 9.503초였다. compileall과 diff 공백 검사도 통과했다.
패키지 매니페스트에 별도 lint/typecheck 명령은 없다.

### 대규모 differential

| 항목 | seed 42 | seed 2026 |
|---|---:|---:|
| positions | 10,000 | 1,000 |
| cells | 1,615,570 | 171,673 |
| None | 1,558,091 | 166,350 |
| 삼삼 | 8,778 | 933 |
| 사사 | 19,537 | 1,757 |
| 장목 | 29,164 | 2,633 |
| mismatches | 0 | 0 |
| legal_mismatches | 0 | 0 |
| restoration_failures | 0 | 0 |

seed 42의 10,000개는 fixture 48개, V6 replay 19개,
Random/Tactical/synthetic 각 3,311개이다. 각 금수 유형이 100건 이상으로 경고는 없었다.
흑·백 legal list를 각 상태마다 모두 비교했으므로 주 검증에서 20,000개 목록을 비교했다.

### 최종 교대 측정 (profiler 없음)

| Metric / 구현 | mean ms | median ms | min ms | max ms | ops/sec |
|---|---:|---:|---:|---:|---:|
| black legal_moves / reference | 8.473 | 8.399 | 8.051 | 9.070 | 118.022 |
| black legal_moves / optimized | 2.435 | 2.467 | 2.226 | 2.568 | 410.658 |
| forbidden_reason / reference | 0.119 | 0.117 | 0.112 | 0.132 | 8,400.664 |
| forbidden_reason / optimized | 0.082 | 0.083 | 0.075 | 0.086 | 12,254.469 |
| random_game / reference | 355.391 | 357.340 | 327.486 | 376.820 | 2.814 |
| random_game / optimized | 48.279 | 48.071 | 40.081 | 59.214 | 20.713 |

흑 legal_moves **3.48배**, fixture를 포함한 대표 forbidden_reason **1.46배** 개선이다.
교대 측정의 Random throughput은 **218.1 → 1,605.3 moves/sec (7.36배)**이다.
기존 benchmark_engine의 10판 비교는 **260.918 → 855.209 moves/sec (3.28배)**이며,
양쪽 모두 총 1,252수였다. 해당 단일 midgame은 **8.157 → 2.504 ms (3.26배)**이다.
백 midgame은 0.026474 → 0.015861 ms였지만 백 코드 변경이 없으므로 작은 시간 차이를
알고리즘 개선으로 해석하지 않는다. thermal/scheduler 변동을 감안해 교대 측정을 주 근거로 삼는다.

### cProfile (누적 시간, 별도 한 판)

| 함수 | baseline calls | final calls | baseline cumulative s | final cumulative s |
|---|---:|---:|---:|---:|
| `forbidden_reason` | 6,994 | 6,994 | 0.947 | 0.095 |
| `_forbidden_after_black_move` | 7,347/6,994 | 766/635 | 0.941 | 0.077 |
| `_open_three` | 29,324/27,948 | 3,028/2,512 | 0.670 | 0.043 |
| `_straight_four_after_extension` | 150,711 | 3,181 | 0.565 | 0.020 |
| `_fours` | 29,354 | 3,044 | 0.219 | 0.035 |

`calls`의 슬래시는 재귀를 포함한 전체/최초 호출 수이다.
straight-four 호출은 **97.9% 감소**했다. `_fours`가 남은 비용 중 비중은 크지만
목표를 이미 충족했으므로 geometry window 최적화는 보류했다.
profiler의 누적 시간 비율을 실제 실행 성능 배수로 사용하지 않았다.

### V6 determinism

두 판 모두 V6 vs Random, simulations=1 / tactical_simulations=1의 완결 smoke이다.
시간은 제외하고 `(winner, history)` 전체를 직접 비교하고 JSON 직렬화 SHA256도 기록했다.
최적화 전 fixture, 최종 reference, 최종 optimized가 모두 일치했다.
이 smoke는 기본 50/100 simulations의 기력 평가를 의미하지 않는다.

| seed | V6 색 | winner | 수 | SHA256 |
|---|---|---:|---:|---|
| 42 | black | 1 | 9 | `7000f108b14d5e10cb95490dc9d99697b9b8448a324552f1c9e25497f48f526f` |
| 43 | white | -1 | 10 | `8a0a596fdd1cbb6ff1e0f87b044a0657bea6942611117e2e46c31d8b198c7188` |

## 최종 판단

**PASS** — 규칙 및 전체 합법수/순서 보존 검증 통과, 5 ms 목표와 3 ms stretch 달성.
Stage 4 Policy-Value Network 개발을 진행할 수 있다. 이번 작업에는 다음 단계 코드를 추가하지 않았다.

상세 수치와 단계별 profile/differential 결과는
[기계 판독 결과](rule-optimization-results.json)에 보존한다.
