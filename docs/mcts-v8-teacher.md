# MCTS-v8 Teacher 설계서 (v1 확정, 개발 중)

> 상태: 설계 v1 확정(2026-09-30). **V8-1(골격)과 V8-A(stage 4/5 VCT1 safety) 구현 완료**, 구현·측정 기록은 §11.
> **V8-B(자기 VCT1 공격) 구현 완료, 조건부 통과**(s0 4/4, false positive 0, D4 26/32는 측정값) — §4.2.9, §11.10. 다음은 V8-4 실전 측정(§11.11).
> 개발 브랜치는 `ccr-ba71e723-f37v8m`이다. V8_TEACHER 동결 전까지 `V8_DEFAULTS` 값은 바뀔 수 있다.

작성 2026-09-30. 입력: "AlphaZero 연구 라인과 teacher 제작 라인을 분리하고 V8 teacher를 새로 만든다"는 제안(V8-M1~M6,
teacher root record, 3계층 데이터, V8-D/V8-T4 arm), 사람 대 v7 웹 대국 6판(`tests/fixtures/web_play_v7_human_games_v1.json`),
[v7-nn-integration-review.md](v7-nn-integration-review.md), [mcts-v7.md](mcts-v7.md), [Stage 8 계획 §12.10](stage8-plan.md).
재현: `python scripts/check_v8_branch_points.py`(§2, 읽기 전용).

좌표는 따로 적지 않으면 0-based (행, 열)이다. v7 검토서 §6.1 표는 1-based다.

## 0. 결론

**큰 방향은 맞다.** V7은 그대로 두고, 기력·정답 품질 우선의 V8을 따로 만든다. AlphaZero(AZ) 실험(대조군·S4·T1)은 V8을
기다리지 않는다. 다만 사람 대국 로그를 V8 모듈 기준으로 다시 돌려 보니 **모듈 우선순위가 제안과 다르다.**

| 제안 | 판정 | 근거 |
|---|---|---|
| V7은 benchmark이고, 동결 후 변경은 새 버전으로 | **맞음** | `mcts-v7.md` §1. `tests/frozen_baseline.sha256`은 base MCTS와 V3/V3.2/V3.2.1/V4/V5/V6/V7 search, threat planner/patterns, V6/V7 agent를 잠근다 |
| V7은 VCT 방어를 범위 밖에 뒀다 | **맞음** | `mcts-v7.md` §1·§13.3(패배 21판 모두 VCF 범위 밖) |
| AZ는 V8을 기다리지 않고 진행 | **맞음** | arm이 모두 gen 400 상태에서 분기하므로 V8 arm을 늦게 붙여도 짝지은 비교가 유지된다(검토서 §7-7) |
| V8-M1: V7이 흑이면 백의 미래 공격 분석이 없다 | **코드상 사실, 그러나 사람 패배와 무관** | `mcts_v6.plan_root`는 흑 차례에 흑 setup만 본다. 하지만 백 차례(양쪽을 봄)였던 2판에서도, 흑 차례 2판에 상대 색 planner를 돌려도 사람의 승리 위협을 **0/4** 잡았다(§2) |
| V8-M2: 좋은 방어수가 후보 20개 밖에 있을 수 있다 | **사람 패배에서는 아님** | stage 4 3판 모두 VCT1-SAFE 수가 stage 4 방어 집합(4개)과 V5/V6 root 20개 **안에 있었다**(§2). 문제는 후보 생성이 아니라 **선택**이다 |
| V8-M3: VCT safety 추가 | **맞음, 최우선** | 100k-node 오프라인 검사에서는 stage 4 방어 4개에 VCT1을 붙였을 때 3판 모두 SAFE 수를 찾았다(§2). 단 실제 V8 예산에서도 같은 결과를 재현하는지는 게이트에서 확인해야 한다 |
| VCT depth 2 | **온라인/teacher 경로에는 아직 채택 근거 없음** | depth 1만으로 4후보 10~48초, 20후보는 24분이 걸렸다(§2.3). depth 2는 아직 실측하지 않았으므로 “불가능”이 아니라 오프라인 비용 측정 후 판단한다 |
| V8-M4: 금수 유도 강화 | **근거 약함, 후순위** | 사람 패배 4판의 분기점은 금수와 무관한 VCT였다. v7 M3는 V6/V5 각각 100판에서 2회씩(총 4/200) 발동했고, 설계 근거 fixture는 1개였다 |
| V8-M5: 위험 국면 400회 탐색 | **stage 4를 못 고친다** | 사람 패배 3판의 분기 착수는 stage 4(탐색 0회)였다. 탐색 횟수는 MCTS 경로에만 영향이 있다. `164723`(MCTS 100회)도 탐색만으로 풀린다는 근거가 없다(검토서 §1.4) |
| V8-M6 / teacher root record(visits) | **새 코드로는 가능, 해상도 주의** | 동결 v7은 root visit을 내보내지 않는다. V5 트리는 progressive widening이라 50회에는 최대 15개, 100회에는 최대 17개의 root child만 열 수 있다. 어떤 후보가 안 열리는지는 top-k 가중 무작위 확장에 따라 달라진다(§4.5) |
| 사람-v7 패배 4판을 regression suite로 | **이미 있음** | `vct_probes_v1.json`(기본 11개, D4 88개)과 가지치기 없는 solver 재검증 `SAME`(검토서 §8). V8 게이트로 그대로 쓴다 |
| probe 게임 전체 제외, D4 dedup | **이미 있음** | `build_tactical_dataset.py`(검토서 §7-1) |
| V8 soft policy로 AZ를 증류 | **새 arm으로 가능, 이전 결정의 변경** | 검토서 §3.2는 v7 MCTS 착수를 target에서 뺐다. 지금은 AZ(B400~B880)가 v5/v6/v7에 0~1/20이어서 classical teacher가 훨씬 강하다(§12.10). 그래서 **T1과 별도 arm**으로 검증한다(§6) |
| V8-T4(지속 혼합) | **코드 없음** | 현재는 fine-tune 1회(`make_teacher_branch.py`)뿐이다. 매 세대 혼합은 학습 루프 변경(training-critical)이다 |
| V8 파일을 `src/search/mcts_v8.py`에 | **그대로는 테스트 실패** | `tests/test_analysis_threats.py`가 `src` 안 어디서든 `analysis` import를 금지한다. VCT solver는 `analysis.threats`에 있다(§4.8) |

**V8의 1순위는 "stage 4/5 강제수와 root 선택에 VCT1 safety를 붙이는 것"이다.** 양방향 planner와 후보 확장은 사람 로그에서
근거가 나오지 않았으므로 뒤로 미루고, 필요성을 V8 게이트의 측정으로 다시 판단한다.

## 1. 목적과 원칙

```
Track A (계속)                      Track B (새로)
AZ: 대조군 / S4 / T1                V7 동결 유지  →  V8 teacher 개발 → V8_TEACHER 동결
      │                                                          │
      │                                                   teacher dataset
      └──────────── gen 400 분기점에서 V8 arm 추가 ◄─────────────┘
                            │
                 대조군·S4·T1·V8 arm 직접 대국
```

- V7은 AZ 측정용 동결 benchmark로 남는다. `mcts_v7.py`, `mcts_v6.py`, `threat_planning.py` 등 잠긴 파일은 수정하지 않는다.
  V8은 이들을 import해서 쓰고, 바꿔야 하는 함수는 V8 파일에 새로 쓴다.
- V8의 목표는 **정답 품질**이다. 속도는 두 번째다. 대신 비용은 반드시 측정해서 기록한다(데이터 생성량을 정하는 값이다).
- V8도 **동결 후 데이터를 만든다**(`V8_TEACHER`). teacher가 바뀌면 dataset 해시가 바뀌고, 이전 arm과 비교할 수 없게 된다.
- V8의 파라미터는 AZ checkpoint 결과를 보고 조정하지 않는다(V7과 같은 원칙). V8 게이트(§7)만 본다.
- **정답은 증명이 우선이다.** proof label > V8 탐색 분포 > 대국 결과 z. V8이 UNSAFE로 증명된 수에 방문을 줬다면 그 수는
  soft target에서 지운다(§5.2).
- AZ 검색(`search.alphazero`, PUCT v2)에는 VCF/VCT를 넣지 않는다(Stage 7-B 결정 유지). V8은 teacher·평가 상대일 뿐이다.

## 2. 사람 대국 로그 재검토 (V8 관점)

### 2.1 로그 사실

6판, 사람 4승(백 2·흑 2), v7 2승. v7 착수 75수 중 44수(59%)가 탐색 0회였다(검토서 §1.1).
패배 4판은 모두 "마지막 SAFE 착수 → 사람의 VCT1 위협 1수 → 모든 응수 VCF 패배"로 끝났다. 이 분기점과 VCT1 정답은
검토서 §6.1과 `vct_probes_v1.json`에 증명과 함께 있다(노드 한도 도달 0회).

### 2.2 V8 모듈별 반사실 검사

`scripts/check_v8_branch_points.py`로 각 분기 국면에서 "그 모듈이 있었다면 다른 수를 골랐을까"를 확인했다.

| 판(v7 색) | 경로 | v7 수 | 사람 다음 수 | stage 4 방어 집합 | VCT1-SAFE | SAFE가 stage 4 집합에 | SAFE가 V6 root 20개에 | 상대 색 planner(`future_setups`) | stage 4 집합 VCT1 검사 |
|---|---|---|---|---|---|---|---|---|---|
| `163810`(흑) | stage 4 | (11,6) | (10,9) | (6,11) (7,10) (11,6) (12,5) | (4,7) (7,10) | (7,10) | 둘 다 | setup 0개 | (7,10)만 SAFE, 36.6초 |
| `163903`(흑) | stage 4 | (6,8) | (8,5) | (6,3) (6,4) (6,8) (6,9) | (6,4) (11,8) | (6,4) | 둘 다 | setup 0개 | (6,4)만 SAFE, 10.1초 |
| `164445`(백) | stage 4 | (7,3) | (5,6) | (2,8) (3,7) (7,3) (8,2) | (3,7) | (3,7) | 있음 | setup 0개 | (3,7)만 SAFE, 48.3초 |
| `164723`(백) | MCTS 100회 | (10,8) | (5,9) | 없음 | 미열거(VCF-safe 206개) | — | 기존 스크립트 후보 20개 중 12개 SAFE¹ | setup 0개 | 기존 후보 20개: SAFE 12 / UNSAFE 8, 1,452초¹ |

(VCT1 검사: `analysis.threats.ThreatSolver`, 노드 한도 100,000, 이 컨테이너 기준 시간.)

¹ 처음 스크립트는 `_forced_v5_move` 뒤 root 후보를 만들 때 새 `_RootContext`를 생성해 Stage-5의 `context.injected`를
버리는 재현 오차가 있었다(실제 V6/V7은 같은 context를 이어 쓴다). 고친 스크립트의 `--structure-only` 재검증에서 **네 판 모두
`stage5 injection False: []`, V5/V6 root 20개**였다. 따라서 오차가 root 후보를 바꿀 수 없었고, 1,452초 분류 결과(12 SAFE / 8 UNSAFE)는
실제 V6 root 20개에 대한 결과로 **확정**한다. V7은 이 20개에 M2(VCF tier)와 M3를 적용해 순서를 바꾸므로 "V7 root 1위"가 아니라
"V6 순서 1위 (6,7)이 SAFE"로 적는다. (10,8)의 VCT1-UNSAFE 증명은 probe fixture에도 독립적으로 있다.

읽는 법:

1. **3판은 stage 4 방어 집합 안에 유일한 VCT1-SAFE 수가 있었다.** V7 stage 4는 "남는 막을 수 없는 4의 수 → 쌍위협 수 →
   VCF" 순으로 골라서 VCT를 보지 않았다. VCT1 검사를 붙인 stage 4는 세 판 모두 다른(SAFE) 수를 고른다.
   이것이 V8-M3의 가장 직접적인 근거다.
2. **stage 4 세 판은 후보 부족이 아니다.** VCT1-SAFE 수가 stage 4 방어 집합과 counterfactual V5/V6 root 안에 이미
   있었다. `164723`도 기존 분석 후보에는 SAFE가 있었지만, 실제 V7 context와의 동일성은 위 구조 재검증 뒤 확정한다.
   따라서 현재 로그에는 후보를 25개로 늘려야 한다는 **양의 근거가 없다**는 결론까지만 둔다.
3. **양방향 planner(V8-M1)는 이 위협을 못 본다.** `future_setups`는 "다음 수에 33/43/44 같은 복합 위협을 만드는 한 수"를
   찾는다. 사람의 수는 복합 위협이 아니라 **"위협 1수 뒤 VCF"**였다. 네 판 모두 상대 색으로 planner를 돌려도 setup이 0개였다.
   `164445`·`164723`은 v7이 백이라 원래 흑 planner가 돌던 판인데도 같았다.
4. **`164723`은 MCTS 경로다.** 이 판만은 stage 4 수정으로 고쳐지지 않는다. v7 수 (10,8)은 VCT1-UNSAFE이고, V6 root 20개 중
   12개가 SAFE, 8개가 UNSAFE다(¹, 확정). V6 순서 1위 (6,7)이 SAFE이므로 "순서대로 검사하다 멈춤" 방식(V8-C)이면 비용도 작다.
   V8-C의 게이트는 **(10,8)을 제거하고 VCT1-SAFE 수를 고르는지**다.

### 2.3 VCT1 비용

- stage 4 방어 4개: 10~48초(VCF 호출 881~2,012회). 사람 패배 3판 모두 모든 후보를 끝까지 판정했다(UNKNOWN 0).
- `164723`의 V6 root 후보 20개 전부: **1,452초(약 24분).** 한 후보 평균 73초다(root 구조는 §2.2 ¹에서 재검증됨).
  검토서 §6.1에서도 "VCT가 없음을 증명"하는 쪽이
  훨씬 비쌌다(승리 위협 전부 찾기 29분).
- 따라서 현재 비용으로 "모든 root 후보에 VCT1"을 매 착수 수행하는 것은 teacher 생성 경로에 실용적이지 않다.
  **우선순위 순 검사 + 조기 종료**, **착수당 노드 예산**, **UNKNOWN tier**가 필요하다. depth 2는 V8 온라인 경로에서는
  보류하고, 먼저 오프라인에서 비용을 재는 대상으로 둔다.

### 2.4 한계

- 6판, 사람 한 명, 비슷한 수법이다. "VCT1 방어가 가장 큰 약점"은 이 표본과 V7 benchmark 패배 21판(§13.3, 모두 VCT)에서만
  나온 결론이다.
- SAFE는 depth 1 기준이다. SAFE 수가 depth 2 이상에서 질 수 있는지는 확인하지 않았다.
- VCF 단계는 동결 solver를 쓰므로 "방어자의 강제 방어가 4를 만드는 수순"을 건너뛰는 한계가 그대로 있다(검토서 §1.4).
- 사람 대 AZ 로그는 아직 없다. AZ의 약점이 같은 종류인지 모른다(§8).

## 3. V8 결정 흐름

```text
Stage 1   내 즉시 승리                                   V5 그대로 (사실)
Stage 2   상대 즉시 승리 차단                            V5 그대로 (승리점 1개면 사실)
Stage 3   내 unstoppable four                            V5 그대로 (사실)
Stage 3V  내 VCF                                         V7 M1 그대로
Stage 3T  내 VCT1 (예산 안에서만)                        V8-B
Stage 4   상대 unstoppable four 방어                     V5 후보 + V8-A (VCT1 safety 정렬)
Stage 5   상대 이중위협 방어                              V5 후보 + V8-A (VCT1 safety 확인)
Stage 6   V6 root 후보 → V7 M2(VCF) → V8-C(VCT1) → V7 M3 → 트리 탐색(V8-E 예산)
기록      root record(V8-F)
```

stage 1~3과 V7 M1은 증명된 수라서 그대로 둔다. stage 2에서 승리점이 2개 이상이면 이미 진 국면이므로 무엇을 둬도 같다.

## 4. V8 모듈 (우선순위 순)

### 4.1 V8-A — stage 4/5 VCT1 safety (필수, 1순위)

- stage 4 방어 집합은 V5와 같게 만든다(creator ∪ creator가 걸린 threat window의 빈 점).
- 정렬 key: `(VCT1 tier, 남는 unstoppable four 수, 쌍위협 수, V7 VCF tier, V5 root key)`.
  VCT1 tier는 **SAFE < UNKNOWN < UNSAFE**다. 남는 4의 수보다 앞에 두는 이유: 사람 패배 3판에서 v7 수와 SAFE 수가 모두
  남는 4 = 0이었다. 남는 4가 1 이상인 수는 VCT1 검사에서 UNSAFE(상대 즉시 4 → VCF)로 먼저 걸러진다.
- 검사는 V5 key 순이 아니라 **기존 V7 stage 4 정렬 순**(V7이 고른 수 먼저)으로 하고 첫 SAFE에서 멈춘다. 첫 후보가 SAFE면 추가 비용은 한 번뿐이다.
- **예산은 같은 몫의 라운드로 나눠 쓴다(구현 중 발견, §11.3·§11.5~§11.7).** 예산은 두 가지다: 총 VCF 노드와 캐시되지 않은
  VCF 호출 수. 첫 목표 몫은 남은 양의 `1/(2n)`이고 라운드마다 두 배가 되지만, 매 라운드의 상한은
  `남은 예산 // 미결 후보 수`다. 여기서 **공평한 몫이 1 이상이면 `1/(2n)`이 정수 나눗셈으로 0이 되더라도 모든 후보가
  최소 1노드·1호출씩 받는다.** 반대로 노드나 호출 중 하나라도 모든 미결 후보에게 1단위를 줄 수 없으면 특정 후보에게 잔여를
  몰아주지 않는다. 모든 미결 후보를 `node_share=0, call_share=0`으로 한 번만 확인해 VCF를 새로 쓰지 않고 끝나는 구조적 판정만
  허용한 뒤 탐색을 멈추고 fallback한다. 이 pass 뒤에는 **예산을 소진으로 보고한다**(`exhausted = True`). 남은 양을 공평하게 쓸
  방법이 없어 쓰지 않았을 뿐, 검사는 더 할 수 없기 때문이다. 이 pass 도중 전역 한도에 걸려 소진이 먼저 확정되면 남은 후보는
  건너뛴다(검사 기록 없음 → fallback에서는 미검사로 취급). 따라서 후보 순서가 작은 잔여 예산을 독점하지 않는다. 끝난 하위 결과는
  캐시에 남고, per-VCF 노드 한도로 끝난 UNKNOWN은 다시 검사하지 않는다.
- **"첫 SAFE"의 정확한 뜻:** 라운드 분배 때문에 선택되는 수는 "정렬 순서상 첫 SAFE"가 아니라 **"가장 먼저 증명된 SAFE"**다.
  같은 라운드 안에서는 정렬 순서대로 검사하므로, V7 수가 SAFE이고 첫 라운드 몫 안에서 증명되면 V7 수가 선택된다. 그러나 V7 수가
  SAFE인데 증명이 첫 몫을 넘고 뒤 후보가 싸게 SAFE로 증명되면 뒤 후보가 선택된다. 두 수 모두 증명된 SAFE라 안전성은 같지만
  V7과 수가 달라진다(`v8_changed`). 사람 패배 24개 probe에서는 V7 수가 모두 UNSAFE라 이 경우가 없었다. V8-4 실측에서
  "V7 수가 SAFE로 확인됐는데 V8이 다른 수를 둔" 횟수를 따로 센다(원칙: 불필요한 변경은 teacher 분포를 흔든다).
- stage 4 집합 전체가 UNSAFE이면 **root 후보 전체로 넓혀** VCT1 검사를 계속한다(§2.2에서 `163810`·`163903`은 stage 4 밖에도
  SAFE가 하나씩 있었다: (4,7), (11,8)). **예산이 이미 다 떨어졌어도 root 후보 목록은 만든다**(fallback 후보로 쓰기 위해서다).
- **SAFE를 하나도 증명하지 못했을 때(fallback):** 증명된 패배수를 미증명 수보다 앞에 두지 않는다.
  1. V7 수가 UNSAFE로 증명되지 않았으면 V7 수.
  2. 아니면 판정 못 한(UNKNOWN 또는 미검사) 강제 방어 중 첫 수.
  3. 그다음 root 후보 중 반박되지 않은(UNKNOWN 또는 **미검사**) 첫 수. 남은 예산이 root 후보 수보다 작아 한 수도 검사하지
     못한 경우도 여기에 들어간다(§11.6).
  4. 2·3의 후보는 `_not_immediately_lost`(둔 뒤 상대 즉시 5목·막을 수 없는 4가 없음)를 통과해야 한다. 한 수 만에 지는 수를
     "미증명"이라는 이유로 고르지 않기 위해서다.
  5. 검사한 수가 모두 UNSAFE로 증명됐을 때만 V7 선택을 그대로 둔다.
- stage 5도 같다. 단일 쌍위협 차단점이 VCT1-UNSAFE면 root 후보로 넓힌다.
- 게이트: `vct_probes_v1`의 `must_defend_vct` 24개(D4 포함)에서 V8의 수가 `correct_moves` 안, `avoid_moves` 밖이고,
  **V8 자신이 그 수를 VCT1-SAFE로 증명했을 것**(fallback으로 우연히 맞은 수는 실패). 예산 소진 횟수와 시간 분포(중앙값·p95·최대)는
  따로 기록한다.

### 4.2 V8-B — 자기 VCT1 공격 (2순위, V8-A scheduler 수정 후 재설계)

목표는 V7 M1(VCF)이 찾지 못한 **조용한 한 수 뒤의 VCT1 강제승**만 추가하는 것이다. 사람 패배 4판에서 사람의 결정타가 모두
이 형태였고, `vct_attack`은 기본 4국면 × D4 8 = 32개다. V8-B는 공격을 **증명했을 때만** 착수하며 UNKNOWN에는 절대
공격하지 않는다. 아직 구현하지 않는다.

#### 4.2.1 사실과 한계

- fixture의 `correct_moves`는 100,000-node `ThreatSolver`가 UNKNOWN 없이 전수 확인한 **비종료 quiet VCT1 위협의 완전 집합**이다.
  `winning_threats()`는 착수 즉시 게임이 끝나는 5목은 제외하지만 V8-B는 Stage 1 뒤에 있으므로 게이트 의미에는 문제가 없다.
- 빠른 후보는 합법수 중 `_fast_pattern_features_for_move`가 `four_directions > 0` 또는 `open_three_directions > 0`인 수다.
  현재 32개 probe에서 후보 ∩ 정답 recall은 32/32다. 하지만 `164723`의 정답 (3,9)는 이 특징이 없어 후보 밖이고 (5,9)만 잡힌다.
  따라서 **후보 생성은 실제로 불완전**하며, V8-B가 못 찾았다고 VCT1 공격이 없다고 결론내리면 안 된다.
- 공격 증명은 V8-A의 `status_after(depth=1)`와 방향이 반대다. 후보 h를 둔 뒤 상대 차례에서
  `decision(game, vct_depth=0, stop_at_safe=True)`를 실행한다.
  - 상대의 모든 확인된 응수가 UNSAFE이고 UNKNOWN이 없음 → **WIN**
  - SAFE 응수 하나를 찾음 → **REFUTED** (즉시 중단 가능)
  - SAFE는 없지만 예산/개별 VCF 한도 때문에 UNKNOWN이 남음 → **UNKNOWN**
- `stop_at_safe=True`는 결과를 바꾸지 않는다. REFUTED에는 살아남는 응수 하나면 충분하고, WIN만 모든 응수의 패배 증명이 필요하다.
- 동결 VCF solver의 범위 한계(방어자의 강제 방어가 4를 만드는 일부 수순)는 그대로 승계한다. 따라서 V8-B의 WIN은
  **현재 solver class 안에서의 증명**이다.

#### 4.2.2 후보와 결정 순서

1. Stage 1~3과 V7 M1을 먼저 실행한다. 여기서 이미 즉시승/unstoppable four/VCF가 있으면 V8-B는 돌지 않는다.
2. 남은 합법수에서 위 fast feature 후보를 만든다. 후보 수 상한은 두지 않는다(사람 대국 실측 최대 19, 정답이 6·8·10위에도 있음).
3. 정렬은 `_v321_priority_score` 내림차순, 동점은 V5 root key로 고정한다.
4. 후보 밖은 **미검사**로 기록한다. teacher record에서 "후보 없음"과 "VCT1 승리 없음"을 구분한다.

```text
Stage 1   즉시 승리
Stage 2   상대 즉시 승리 차단
Stage 3   내 unstoppable four
Stage 3V  V7 own VCF (M1)
Stage 3T  V8-B own VCT1 공격
Stage 4   V8-A
Stage 5   V8-A
Stage 6   V6/V7 tree
```

V8-B가 WIN을 증명하면 첫 WIN을 즉시 둔다. WIN이 없으면 REFUTED·UNKNOWN 여부와 무관하게 아무 수도 강제하지 않고 Stage 4/5 또는
tree로 내려간다. **공격 쪽에는 fallback이 없다.**

#### 4.2.3 공용 예산 scheduler

V8-B는 V8-A와 **같은 공정 분배 함수**를 재사용한다. 두 모듈이 각자 scheduler를 가지면 이번에 고친 rounding/fairness 버그가
다시 갈라질 수 있기 때문이다.

- V8-A의 현재 규칙을 일반화해 "상태 함수 + 목표 상태"를 받는 helper로 만든다.
- 각 모듈은 solver/cache/예산을 별도로 가진다. V8-B가 V8-A 예산을 미리 소모하지 않는다.
- 첫 목표 몫은 각 예산의 `remaining // (2*n)`, 이후 두 배씩 증가한다.
- 매 라운드 `fair = remaining // pending`을 계산한다. **fair ≥ 1이면 초기 half-share가 0이어도 최소 1을 모든 pending 후보에 준다.**
- node 또는 call 중 하나라도 fair가 0이면 어느 후보에도 잔여를 독점시키지 않는다. 모든 pending 후보를 `0/0` share로 한 번만
  호출해 새 VCF 자원을 쓰지 않고 끝나는 구조적 결과만 회수한 뒤 UNKNOWN으로 남기고 종료한다.
- share 때문에 잘린 결과는 retryable UNKNOWN이며 완결된 하위 VCF 결과만 cache에 남긴다. 개별 `attack_vcf_node_limit`까지
  실제로 탐색하고 끝난 UNKNOWN은 final UNKNOWN이라 같은 착수에서 다시 돌리지 않는다.

개발 기본값은 측정용으로 다음에서 시작한다.

| 키 | 개발값 | 의미 |
|---|---:|---|
| `own_vct_attack` | True | V8-B on/off |
| `attack_vcf_node_limit` | 20,000 | VCF 한 번의 노드 한도 |
| `attack_call_limit` | ~~3,000~~ → **10,000** | V8-B 한 착수의 uncached VCF 호출 상한(구현 중 변경, §4.2.9) |
| `attack_node_budget` | 200,000 | V8-B 한 착수의 총 VCF 노드 상한 |

이 값은 V8-5 전에 고정하지 않는다. 예산 없는 실측에서 WIN 하나의 증명이 606~1,034 VCF 호출, 5~72초였고, 실제 국면 하나는
10분을 넘겨 중단했으므로 bounded scheduler는 필수다.

#### 4.2.4 구현 인터페이스

- `_BudgetedSolver.attack_status(game, move, node_share, call_share)` → `WIN | REFUTED | UNKNOWN`.
- 새 진단:
  `v8_attack_status`, `v8_attack_move`, `v8_attack_candidates`, `v8_attack_checked`,
  `v8_attack_calls`, `v8_attack_nodes`, `v8_attack_budget_exhausted`, `v8_attack_seconds`.
- V8-B 선택 시 `v8_route = "own_vct"`.
- budget diagnostics에는 가능하면 `round_count`, `node_share_cuts`, `call_share_cuts`,
  `per_vcf_limit_unknowns`를 추가한다. 현재 V8-A의 p95 악화는 call-share 재진입과 node-share cut을 구분하지 못해 원인을
  단정할 수 없기 때문이다.
- 모든 `play`는 `try/finally`로 복원한다.

#### 4.2.5 게이트

**1단계 — candidate recall**

32개 `vct_attack` 각각에서 `candidate_set ∩ correct_moves != ∅`. 현재 측정은 32/32이며 코드화한다.

**2단계 — bounded proof**

32개 모두 다음을 동시에 만족해야 통과한다.

- 선택 수 ∈ `correct_moves`
- `v8_route == "own_vct"`
- `v8_attack_checked[선택 수] == WIN`
- 선택 수는 V8의 bounded solver가 직접 WIN으로 증명

예산 소진 자체는 오류가 아니지만, 그 때문에 WIN을 못 찾으면 해당 probe는 proof gate 실패다.

**3단계 — 독립 재검증**

V8-B가 WIN이라고 한 수는 같은 solver 객체/cache를 재사용하지 않고 **fresh `ThreatSolver(node_limit=100_000)`**로 상대의 모든 응수를
다시 분류한다. SAFE 또는 UNKNOWN이 하나라도 나오면 실패한다. `vcf_loss`, `must_defend_vct`, 사람 대국 144국면에서도
V8-B가 새로 WIN을 선언한 수를 같은 방식으로 검사해 false-positive가 없는지 본다.

#### 4.2.6 회귀 조건

- `own_vct_attack=False` + `stage_vct_safety=False` → 경로별 대표 국면에서 V7과 동일.
- `own_vct_attack=False` + V8-A on → 현재 V8-A 동작과 동일, `must_defend_vct` 24/24 proven SAFE 유지.
- 기존 테스트의 `OFF` 설정은 V8-B 기본값이 True가 되면 **두 모듈을 모두 False로 명시**한다.
- 불법 착수 0, board/history mutation 0, 동결 V2~V7 hash 불변.
- 비용은 평균·median·p95·max, node/call 사용률, budget exhaustion, scheduler cut 종류를 V8-A와 같은 형식으로 남긴다.

#### 4.2.7 현재 실측 해석

- 기본 4국면 후보 수: 7 / 8 / 13 / 10.
- 정답 후보 순위: 1 / 6 / 6·10 / 8(다른 정답 (3,9)는 후보 밖).
- WIN 증명: 606~1,034 VCF 호출, 약 5~72초.
- 반박은 대부분 싸지만 예외적으로 648호출·28초가 걸린 후보가 있었다.
- 사람 대국 표본에서 후보는 평균 5.7개, 최대 19개였다. 대부분 빠르지만 10분 초과 국면이 있었으므로 **평균보다 tail을 먼저 본다.**
  (정정, §4.2.9: 그 10분 초과 국면(3번 판 31수)은 실제 흐름에서는 stage 2가 처리해 V8-B가 돌지 않는다. 측정 스크립트가 stage를
  건너뛰지 않고 공격 검사를 돌렸다. 다만 `vct_attack` 대칭 변형에서 예산 없이 10분을 넘는 증명이 따로 확인됐다.)

#### 4.2.8 구현 전 보완 사항 (4차 검토)

1. **"첫 WIN" = 가장 먼저 증명된 WIN.** 공용 scheduler를 쓰므로 V8-A와 같다(§4.1). 두 수가 모두 WIN이면 안전성은 같다. 다만
   진단에 `v8_attack_rank`(선택 수의 후보 정렬 순위)를 남겨, 우선순위가 높은 WIN을 제치고 뒤 WIN을 둔 빈도를 본다.
2. **착수당 최악 비용은 두 모듈의 합이다.** V8-B는 stage 1~3·M1이 아닌 거의 모든 착수에서 돌고, stage 4/5 국면에서는 V8-B가 먼저
   예산을 쓴 뒤 V8-A가 돈다. 개발값(각 200k 노드)과 실측 처리량(약 1,100 노드/초)이면 한 착수가 최악 **약 3분 + 3분**이다.
   이 값은 게이트 측정용이다. `V8_TEACHER`/`V8_PLAY` 예산은 V8-4에서 "게이트 통과를 유지하는 최소 예산"과 teacher 처리량을 보고
   정한다.
3. **zero-budget pass의 의미는 V8-A와 같다.** pass 뒤 예산을 소진으로 보고한다. V8-B에서는 UNKNOWN이 곧 "공격 안 함"이므로
   이 pass는 사실상 캐시된 결과를 회수하는 역할만 한다.
4. **독립 재검증은 버그 검출용이다.** 동결 VCF solver는 결정론적 DFS라서 20,000노드 안에서 찾은 VCF는 100,000노드에서도 같은
   순서로 찾는다. 따라서 재검증 불일치는 탐색 한도 차이가 아니라 **예산·캐시 코드의 버그**를 뜻한다. 재검증은 이미 있는
   `analysis.tactical_labels.proves_threat`(모든 응수 분류, `stop_at_safe` 없음)를 fresh `ThreatSolver(node_limit=100_000)`로
   호출한다. 새 검증 코드를 만들지 않는다.
5. **예산 중단은 결과를 오염시키지 않는다.** `attack_status`는 `decision` 도중 `_BudgetExhausted`가 나면 UNKNOWN을 돌려준다.
   예산에 잘린 VCF는 캐시되지 않고, per-VCF 한도로 끝난 VCF(`VCF_UNKNOWN`)는 캐시되지만 그 응수를 UNKNOWN으로 만들 뿐이라
   WIN으로 바뀌지 않는다. 단위 테스트로 "예산 중단 뒤 같은 solver로 재호출해도 WIN 판정이 증명 없이 나오지 않음"을 확인한다.
6. **teacher label:** V8-B의 WIN은 `vct_attack` kind로 쓰되 첫 WIN 하나만 증명하므로 **one-hot(불완전 집합)**이다. 기존 T2의
   `vct_attack`과 같은 의미다(검토서 §7-3). value는 +1(증명).
7. **흑 금수:** 후보는 `game.legal_moves()`에서만 만든다(흑 금수점 제외). 방어자의 응수도 `decision`이 합법수만 두므로 금수 효과는
   엔진 규칙대로 처리된다.

#### 4.2.9 구현 기록과 게이트 결과 (구현 완료, 게이트 미통과)

**구현(§4.2.2~§4.2.8 그대로):**

- `_first_safe`를 `_first_proven(game, moves, statuses, solver, evaluate, target)`로 일반화했다. V8-A는
  `(status_after, SAFE)`, V8-B는 `(attack_status, WIN)`으로 같은 함수를 쓴다. `_first_safe`는 V8-A용 얇은 래퍼로 남겼다.
- `_BudgetedSolver._bounded`가 몫 설정·착수·복원·예산 중단 처리를 맡고, `status_after`와 `attack_status`는 그 위의 판정 함수만
  다르다. `attack_status`는 `decision(game, 0, stop_at_safe=True)`의 결과를 UNSAFE→WIN, SAFE→REFUTED, UNKNOWN→UNKNOWN으로
  바꾼다. 합법수가 없는 판(가득 참)은 REFUTED다(`decision_status`가 빈 집계를 UNSAFE로 읽는 것을 막는다).
- `_attack_candidates`: `context.legal`(흑 금수 제외) 중 4 또는 열린 3을 만드는 수, `_v321_priority_score` 내림차순 → V5 root key.
- `_own_vct_attack`은 M1(own VCF) 뒤, stage 4/5 앞에서 돈다. 전용 `_BudgetedSolver`를 쓴다(V8-A와 캐시·예산 분리).
  WIN이면 `v8_route = "own_vct"`, `v8_changed = True`, `v8_v7_move = None`(V7의 수는 전체 V7 탐색을 돌려야 알 수 있어서
  계산하지 않는다). WIN이 없으면 아무것도 두지 않고 V8-A/tree로 넘어간다. V8-B는 난수를 쓰지 않으므로 뒤의 트리 탐색 난수열은
  V8-B를 끈 경우와 같다.
- 새 설정 4개(`own_vct_attack`, `attack_vcf_node_limit`, `attack_call_limit`, `attack_node_budget`)와 진단 9개
  (`v8_attack_status/move/rank/candidates/checked/calls/nodes/budget_exhausted/seconds`). §4.2.4의 "가능하면" 진단
  (`round_count`, cut 종류 구분)은 아직 넣지 않았다.
- **`attack_call_limit` 3,000 → 10,000:** 3,000에서는 `164445` 15수(s0)가 호출 2,999개를 쓰고 WIN을 못 찾았다. 6,000이면 2,116호출·43.7초,
  10,000이면 1,728호출·27.6초에 WIN을 찾았다. 한도가 커지면 몫이 커져 라운드 사이에 잘렸다가 다시 도는 호출이 줄기 때문이다.
  시간은 노드 예산이 제한하므로 호출 한도는 폭주 방지용으로 둔다.

**테스트:** V8 단위 테스트 24개 → 31개. `OFF`는 두 모듈 모두 끔(V7과 동일), V8-A 테스트는 `A_ONLY`(B 끔)로 격리했다. 새 테스트:
실제 공격 국면(163903 14수)에서 (8,5)를 6위 후보로 증명하고 fresh 100k solver로 재검증, 50노드 예산이면 공격하지 않고 B를 끈 경우와
같은 수, 예산에 잘린 뒤 재호출해도 끝까지 증명해야 WIN, 후보 생성(8개, (8,5)는 6번째, 모두 합법), 후보 0개면 비용 0, 공용 scheduler에서
비싼 REFUTED 후보가 WIN 후보의 예산을 뺏지 못함, B가 WIN을 못 찾으면 A만 켠 것과 같은 수. 전체 테스트 495개 통과(torch 131개 skip),
동결 해시 불변.

**게이트 (`python scripts/run_v8_gates.py --gate attack`, 이 컨테이너):**

| 항목 | 결과 |
|---|---|
| candidate recall | **32/32** |
| bounded proof | **26/32** |
| 독립 재검증(fresh 100k, 모든 응수 분류) | 26/26 일치(**false positive 0**) |
| 예산 소진 | 5 |
| 시간 | 평균 52.3초, 중앙값 18.1초, p95 164.0초, 최대 169.4초. 최대 호출 2,525 |

| 기본 국면 | 결과 | 실패한 대칭 | 실패 이유 |
|---|---|---|---|
| `001`(163810 16수) | 8/8, 모두 1위 후보 | — | — |
| `004`(163903 14수) | 8/8, 모두 6위 후보 | — | — |
| `007`(164445 15수) | 5/8 | s3, s6, s7 | 노드 예산 200k 소진. 정답 후보 2개 포함 2~3개가 UNKNOWN으로 남음 |
| `009`(164723 21수) | 5/8 | s1, s4, s7 | s4·s7은 노드 예산 소진. **s1은 예산이 남았는데**(129,456노드) 정답 (5,5)가 per-VCF 한도 20k에 걸려 최종 UNKNOWN |

- **실패 6개는 모두 "WIN을 못 찾음"이고, 틀린 WIN은 0개다.** V8-B는 이때 아무것도 두지 않고 V8-A/tree로 넘어간다(설계대로).
- **원인은 대칭 변형 사이의 비용 차이다.** 동결 VCF solver와 `ordered_moves`의 탐색 순서가 D4에 대해 불변이 아니어서, 같은 국면이라도
  방향에 따라 증명 비용이 크게 다르다. `007` s0은 예산 없이 (5,6) 증명에 12.5초였지만, s3은 정답 후보 (6,9) 하나를 예산 없이
  증명하는 데 **21분 넘게 끝나지 않아 중단했다**(이 컨테이너, per-VCF 한도 20k). 1M 노드·per-VCF 100k로 다시 돌린 V8에서도
  `007-s3`(645초), `007-s6`(839초)이 예산을 다 쓰고 WIN을 찾지 못했다. 그래서 예산을 늘리는 것만으로는 해결되지 않는다.
- 실패 후보를 **예산 없이 단독으로** 증명한 비용(fresh solver, 순차):

| probe | 후보 | per-VCF 20k | per-VCF 100k |
|---|---|---|---|
| `009-s1` | (5,5) | UNKNOWN(per-VCF 한도 3회), 106,760노드, 93.3초 | **WIN**, 158,784노드, 132.8초 |
| `009-s7` | (5,9) | UNKNOWN(per-VCF 한도 3회), 498,565노드, 456.7초 | 미측정(중단) |
| `007-s3` | (6,9) | 21분 넘게 미종료(중단) | 미측정 |

  즉 실패는 두 종류다. `009-s1`은 **per-VCF 한도**가 원인이라 한도를 100k로 올리면 풀린다. 나머지는 증명 자체가 수십만 노드·수 분
  이상 드는 방향이다. 비교: 같은 국면의 s0 방향은 (5,9) 증명이 122,118노드·100.5초, `007` s0은 33,108노드·26.6초였다.

**사람 대국 144국면 (V8-A 끔, V8-B만):** V8-B는 102국면에서 돌았다(나머지는 stage 1~3·own VCF). WIN 5개, 모두 fresh 100k 재검증과
일치(false positive 0). 이 중 4개는 사람이 이긴 판의 결정타 국면(`vct_attack` 기본 국면과 같은 국면)이고, 1개는 v7이 이긴 판(2번 판
17수)이다. V8-B 시간: 중앙값 0.09초, 평균 3.1초, p95 12.5초, 최대 118.9초. 예산 소진 0회.

**V8-A 회귀 (A 켬, B 끔):** 24/24, 증명 SAFE 24, 예산 소진 0. 24개 모두 이전 실행과 수·노드·호출이 같다(scheduler 일반화는 동작을
바꾸지 않았다).

**V8-B 완료 판정: 조건부 완료(§11.10).** 기본 방향(s0) 4/4와 false positive 0을 필수 게이트로 두고, D4 32개의 증명률(26/32)은
측정값으로 기록한다. 아래 1~3 중 최종 선택은 V8-4 실전 측정(§11.11)에서 대칭 미탐이 실제로 얼마나 문제가 되는지 본 뒤 한다.
§4.2.5의 bounded proof 32/32는 채우지 못했다.

1. **게이트를 방향 한정으로 바꾼다:** 기본 방향(s0) 4/4와 false positive 0을 필수로, 대칭 변형의 증명률은 측정값으로 기록한다.
   현재 결과로 통과한다. 대신 "같은 국면인데 방향에 따라 공격을 못 찾는다"는 한계가 V8에 남는다.
2. **예산을 늘린다:** `007-s3`은 1M 노드로도 부족했으므로 이것만으로는 32/32가 안 된다. 착수 시간도 크게 늘어난다.
3. **탐색 순서를 대칭에 강하게 만든다:** 동결 VCF solver는 바꿀 수 없으므로 V8 전용 VCF(순서 규칙만 다른 사본)를 만든다.
   큰 작업이고, V8-A의 SAFE 판정도 같은 solver를 쓰므로 V8-A 게이트를 다시 돌려야 한다.

### 4.3 V8-C — root VCT1 safety (3순위, 비용 측정 후 범위 결정)

- V7 M2 뒤에 붙인다. 먼저 M2의 verified-SAFE tier를 검사한다. 여기서 VCT1-SAFE를 하나도 못 찾고 그 tier를
  모두 VCT1-UNSAFE로 소진했을 때만 M2 inconclusive tier를 두 번째 pool로 검사한다. 확인된 VCF-UNSAFE 후보를
  VCT 검사가 되살리지는 않는다.
- 전 후보를 매 착수 완전 분류하는 것은 현재 비용으로 실용적이지 않다(§2.3). 대신:
  1. **싼 사전 우선순위 필터:** 후보를 둔 뒤 상대의 "4 또는 열린 3 생성수" 중 VCF 후속이 보이는 후보를 먼저 검사한다.
     이것은 완전한 VCT1 필터가 아니다. `ThreatSolver`는 모든 합법 quiet move와 금수 변화까지 보므로, 필터에서 빠진 후보도
     예산이 남으면 뒤이어 검사한다. 끝까지 못 본 후보는 "VCT1 미검사"로 표시하고 SAFE로 올리지 않는다.
  2. 검사 대상을 V6 순서(사전 필터 hit 우선, 각 그룹 안에서 V6 순서)로 검사하고, **SAFE가 K개(기본 3) 나오면 멈춘다.**
     **SAFE가 하나라도 증명되면 tree root는 그 proven-SAFE 후보들로만 제한한다.** SAFE를 찾아 놓고 UNKNOWN/미검사를 같은
     tree에 남기면 MCTS가 다시 미검사 패착을 선택할 수 있어 safety filter의 의미가 사라진다.
  3. 착수당 노드 예산을 둔다. 예산 소진까지 SAFE를 하나도 못 찾았을 때만 UNKNOWN + 미검사 후보를 V6 순서로 fallback하고,
     확인된 UNSAFE는 제외한다. 모든 후보가 UNSAFE로 확인된 경우에만 원래 V7/V6 선택을 최종 fallback으로 허용한다.
- `164723` 20수에서 v7 수 (10,8)은 VCT1-UNSAFE로 증명되어 있다(검토서 §6.1). V8-C가 이 수를 빼는지가 1차 게이트다.
- 착수 시간이 한 수 수십 초가 되면 teacher 생성량이 크게 줄어든다. 그래서 V8-C는 **V8-A/B 구현 후 비용을 재고** 범위(K,
  예산, 필터)를 정한다. 측정 전에는 숫자를 확정하지 않는다.

### 4.4 V8-D — 양방향 future-threat planner (보류)

- 제안의 V8-M1이다. 코드 사실(흑 차례에 백 setup을 안 봄)은 맞다.
- 하지만 사람 패배 4판에는 효과가 없었고(§2.2), V7 benchmark 패배 21판의 원인도 VCT였다. 이 모듈의 효과를 보여 주는 국면이
  아직 없다.
- V8-A~C 구현 후 V8 대 V7 대국의 V8 패배를 분석해 "상대 복합 위협 setup을 못 막아 진" 사례가 나오면 그때 넣는다.
  넣는다면 `plan_root`를 V8 파일에 새로 써야 한다(`mcts_v6.py` 동결).

### 4.5 V8-E — 탐색 예산과 root 후보 (4순위)

- **후보 확장(제안 V8-M2)은 일단 하지 않는다.** 사람 로그에서 후보 부족 근거가 없고(§2.2), V6은 planner 주입 후보를
  baseline 20개에 합쳐 root가 20개를 넘을 수도 있다(`_root_candidates_v6`의 `set(baseline).union(reasons)`).
  따라서 예산은 고정 20이 아니라 **실제 root 길이**를 기준으로 계산한다.
- 대신 **root에서 모든 후보가 최소 한 번은 열릴 수 있는 simulation 수를 보장**한다. V5의
  `_allowed_children`는 simulation 시작 시점의 `node.visits`에 대해
  `initial_width + floor(sqrt(visits))`를 허용하고, visit 증가는 simulation 끝의 backprop에서 일어난다.
  그래서 width=8일 때 50회는 최대 15개, 100회는 최대 17개, 144회도 최대 19개이며 **20개를 모두 열려면 최소 145회**다.
  일반적으로 root 후보가 R개이고 R>W(initial_width)이면 최소
  `max(R, (R-W)^2 + 1)`회를 둔다(R≤W면 R회면 충분).
- 또 expansion은 V6 순서대로 하나씩 여는 것이 아니다. `_pop_ranked_untried`가 남은 후보의 상위
  `priority_top_k=8` 안에서 rank-weighted random 선택을 한다. 따라서 “50회면 16~20위가 방문되지 않는다”는 주장은
  성립하지 않는다. 정확한 표현은 **20후보/50회에서는 적어도 5개, 100회에서는 적어도 3개 후보가 미방문**이고,
  어느 후보인지는 seed에 따라 달라진다는 것이다.
- 제안의 100/200/400 단계는 실험 시작점으로 쓴다. 착수 시간과 V8 대 V7 점수로 정한다. `V8_PLAY`(사람 대국·비교)와
  `V8_TEACHER`(데이터 생성)를 같은 알고리즘, 다른 예산으로 둔다.

### 4.6 V8-F — teacher root record

착수마다 다음을 남긴다(대국 로그 JSON, 기존 `game.json` 구조에 추가).

```json
{
  "ply": 20, "to_play": "WHITE", "route": "stage4|stage6|own_vcf|own_vct|...",
  "chosen": [3, 7],
  "candidates": [
    {"move": [3, 7], "visits": null, "vcf": "SAFE", "vct1": "SAFE", "reasons": ["stage4"]},
    {"move": [7, 3], "visits": null, "vcf": "SAFE", "vct1": "UNSAFE", "witness": [[5, 6], "..."]}
  ],
  "proof": {"kind": "must_defend_vct", "complete": false},
  "nodes": {"own_vcf": 12, "vcf_safety": 3400, "vct1": 41000}, "seconds": 18.2
}
```

- `vcf`/`vct1`은 SAFE/UNSAFE/UNKNOWN/미검사(`null`)다. **미검사를 SAFE로 쓰지 않는다.**
- stage 경로 착수에는 visits가 없다. soft target은 트리 탐색 착수에만 생긴다.
- `proof.complete`는 모든 합법수를 분류했을 때만 true다. false면 정답 **집합**이 아니라 "이 수는 안전하다/지는 수다"만 쓸 수 있다.

### 4.7 V8-G — 금수 함정 (5순위)

- V7 M3(`self_forbidden_min_white 3`, 감점만)를 그대로 쓴다. 2단계 점수화(제안 V8-M4)는 근거 국면이 생기면 넣는다.
- 금수 효과는 VCT1 검사가 이미 다룬다. ThreatSolver는 방어자의 응수를 실제로 두므로 "흑이 막을 자리가 금수가 됨"과
  "백 돌이 흑 금수를 풀어 줌"을 엔진 규칙으로 처리한다(검토서 §3.3). 즉 V8-A/C가 금수를 이용한 VCT1 공격도 막는다.

### 4.8 코드 배치와 격리

- `tests/test_analysis_threats.py`는 (1) AZ 경로(`search.alphazero`, `search.tactics`, `training.*`, `model.*`)가 `analysis`를
  불러오지 않는지, (2) `src`의 **analysis 패키지 밖** 파일이 `analysis`를 import하지 않는지를 검사한다. V8은
  `analysis.threats`가 필요하므로 `src/search/mcts_v8.py`에 두면 현재 격리 테스트를 깨뜨린다.
- 이 테스트를 V8 예외 allowlist로 약화시키지 않는다. V8은 학습 탐색이 아니라 **오프라인 teacher/평가 도구**이므로
  `src/analysis/mcts_v8.py`와 `src/analysis/mcts_v8_agent.py`에 둔다. 그러면 기존 의존 방향
  `analysis.threats -> search.mcts_v7`을 유지하고, `search.mcts_v8 -> analysis.threats -> search.mcts_v7` 같은 역방향
  의존도 만들지 않는다.
- `agents/__init__.py`는 모든 agent를 즉시 import하고, `training/evaluation.py`가 `agents`를 불러온다. 따라서 V8 agent는
  **`agents/__init__.py`에 넣지 않는다.** `run_web_play.py`, benchmark, teacher 생성 스크립트가 V8을 요청할 때만
  `analysis.mcts_v8_agent`를 조건부 import한다. `search/__init__.py`에도 export하지 않는다.
- 파일: `src/analysis/mcts_v8.py`, `src/analysis/mcts_v8_agent.py`, `scripts/run_mcts_v8_benchmark.py`,
  `scripts/generate_v8_teacher_games.py`, `tests/test_mcts_v8.py`. 기존 analysis 격리 테스트는 **변경 없이 유지**한다.
  (구현된 파일은 §11.)

## 5. Teacher 데이터

### 5.1 3계층

| 계층 | 내용 | policy target | value target | 신뢰도 |
|---|---|---|---|---|
| A. 증명 | immediate_win, must_block, unstoppable_four, vcf, vcf_loss, forced_loss, vct_attack, must_defend_vct, vct_loss | 증명된 수(집합 또는 one-hot) | 증명값 | 최고 |
| B. V8 탐색 | V8 트리 탐색 착수 | root visits 분포(§5.2 정리 후) | 없음 또는 약한 가중 | 중간 |
| C. 대국 결과 | V8 대 V8 / V8 대 V7 등 | B와 같음 | z(대국 결과) | 낮음 |

- A는 이미 있는 `analysis.tactical_labels`(T1/T2 kind)를 그대로 쓴다. V8 대국은 A의 **국면 공급원**이 하나 늘어나는 것이다.
- 한 국면에 A label이 있으면 B/C target은 쓰지 않는다(증명 우선).

### 5.2 B 계층 soft target 정리 규칙

1. VCF-UNSAFE, VCT1-UNSAFE로 증명된 수의 방문은 지우고 다시 정규화한다.
2. 방문이 0인 후보는 target에 넣지 않는다(열리지 않은 후보는 "나쁜 수"가 아니다, §4.5).
3. root 방문 합이 `V8_TEACHER` simulations의 절반 미만이면(stage/모듈 조기 종료 등) 쓰지 않는다.
4. 온도는 1(방문 비율)로 저장하고, 학습 쪽에서 바꾼다.

주의: V5 트리는 휴리스틱 순서와 random rollout에 의존한다. B 계층은 "V8 스타일"을 담는다. 그래서 B 계층의 효과는
A만 쓰는 arm(T1/T2)과 **별도 arm**으로만 판단한다(§6).

### 5.3 국면 공급원

| 공급원 | 이미 있음 | 비고 |
|---|---|---|
| V7 benchmark 기보(200판) | 있음 | A 계층 전용(v7 수는 target이 아님) |
| AZ self-play(gen 160~400) | 있음 | A 계층. 대부분 9~10수 게임이라 전술 국면 비율이 낮다(§12.10) |
| 사람 대 v7 6판 | 있음 | probe 원천이라 **학습에서 제외**(게임 단위) |
| V8 대 V8, V8 대 V7 | 새로 | A + B(+C). 무작위 2수 오프닝, 색 교환 |
| V8 대 AZ(B400 계열) | 새로 | AZ가 실제로 가는 국면에서 V8 답을 얻는다. 분포 이동을 줄인다 |
| 사람 대 V8 / 사람 대 AZ | 새로 | 새 probe 원천(§8). 학습에서 제외 |

dataset `stats`에는 `source|color|phase|kind` 개수를 남긴다. 기존 `build_tactical_dataset.py::_phase`와 비교 가능하게
phase는 **ply < 12 opening, ply < 30 middle, 그 외 late**를 그대로 사용한다. 새 경계를 쓰려면 dataset format/version부터
분리한다.

### 5.4 누수 방지

- probe 국면(D4 포함)을 지나는 게임은 통째로 뺀다(기존 규칙). V8 게이트용 사람 패배 fixture도 같은 probe set이므로 자동으로 빠진다.
- **V8 게이트 국면으로 V8을 조정하지 않는다.** V8-A의 게이트가 `must_defend_vct` probe이므로, 이 probe는 V8의 "개발 기준"이 되고
  AZ 평가에는 여전히 held-out이다(AZ 학습 데이터에는 들어가지 않는다). 새 사람 대국 probe(§8)는 V8 개발에도 쓰지 않는 held-out으로 둔다.

## 6. AZ 실험 arm

모든 arm은 `stage8_g3_b` gen 400에서 분기하고, gen 480/560/640에서 대조군(`stage8_b400_long`)과 직접 대국 100판을 둔다
(검토서 §9·§10과 같은 절차). 100판의 표준오차는 약 5%p라서 10%p 안쪽 차이는 구분하지 못한다.

| arm | 내용 | 새 코드 | 시작 조건 |
|---|---|---|---|
| 대조군 | B400 continuation | 없음 | 완료(gen 880) |
| S4 | temperature_moves 4 | 없음 | 진행 중 |
| T1 | 증명 label fine-tune 1회 | 없음 | 진행 중 |
| **V8-A1** | T1 dataset + **V8 대국에서 나온 A 계층 label** fine-tune 1회 | V8, 생성 스크립트 | V8_TEACHER 동결 후 |
| **V8-D** | V8-A1 + **B 계층 soft policy** fine-tune 1회 | 위 + soft target 학습 | V8-A1과 동시 |
| V8-T4 | 매 세대 batch에 teacher 행 혼합 | 학습 루프 변경 | V8-D 또는 T1이 이득을 보이고 640에서 씻겨 나갈 때 |
| S4 + (가장 좋은 teacher) | 조합 | 없음 | S4와 teacher arm 결과 후 |

- 제안의 "V8-D 대 T1"을 둘로 나눴다. **V8-A1 대 T1**은 "V8 대국이라는 새 국면 공급원"의 효과이고, **V8-D 대 V8-A1**은
  "soft policy"의 효과다. 하나로 합치면 어느 쪽 덕인지 알 수 없다.
- fine-tune 투입량은 T1과 같게 고정한다(1,000 step × 64 × 0.5, 검토서 §7-4).
- soft target 학습은 `make_teacher_branch.py`에 target 형식(집합 균등 vs 분포)을 추가해야 한다. 현재 dataset의 `policy`는
  수 목록(균등)뿐이다.
- V8-T4는 학습 루프에 teacher sampler를 넣는 training-critical 변경이다. 비율(24:8, 20:12 등)과 proof/soft 비율은 이때 정한다.
  T1/V8-D에서 "초반 이득 → 640 소멸"이 보여야 시작한다(검토서 §7-6).
- 판정 우선순위는 검토서 §3.6을 따른다. **V8을 본 arm은 V7·V8 대국 점수를 주 지표로 쓰지 않는다.**
- Stage 9(network 확대)는 이 arm들 이후, raw 전술 probe가 기준선(must_block 0.6) 가까이 오른 뒤에도 정체할 때 판단한다(§12.10과 같다).

## 7. V8 게이트 (`V8_TEACHER` 동결 전)

### 기능

- 전체 테스트와 frozen hash 통과(V2~V7 파일 불변). 격리 테스트(§4.8) 통과.
- 불법 착수 0, 같은 seed 결정성.
- `vct_probes_v1`: `must_defend_vct` 24개 전부 `correct_moves` 안(V8-A). `vct_attack` 32개에서 첫 수가 증명된 위협 집합 안(V8-B).
- `mcts_v7_positions.json`, `mcts_v7_vcf_regression.json`(124국면): V7과 같거나 더 나은 결과(V7 M1/M2 동작 유지).
- 사람 패배 4판 분기 국면: 1~3판은 SAFE 수 선택. `164723` 20수는 (10,8)을 제거하고 VCT1-SAFE 수를 선택.
  root 구조 재검증은 끝났다(§2.2 ¹). 12 SAFE / 8 UNSAFE 분류를 V8-C 게이트의 기준으로 쓴다.

### 비용 (사용자 PC에서 V7과 같은 조건으로)

- 평균·중앙값·최악 착수 시간, 모듈별 노드(`v8_vct_nodes` 등), 예산 소진 비율.
- teacher 생성 처리량: 한 코어당 시간당 대국 수. 이 값으로 dataset 규모를 정한다.

### 기력

- V8 대 V7 100판(50쌍, 무작위 2수 오프닝, 색 교환). 전체·색별 score, 불법 0.
- V8 패배 대국을 `analysis.threats`로(웹 대국의 `analyze_web_play_losses.py --vct-depth 1`과 같은 방식) 분석해 남은 약점 종류를
  기록한다(§4.4 판단 근거). benchmark 결과 파일용 입력은 새로 붙여야 한다.
- 목표 승률에 맞추지 않는다. V7보다 유의하게 약하면 동결하지 않는다.

## 8. 사람 대국 로그 운영 (추가)

1. **사람 대 AZ(B400 이후)를 먼저 둔다.** 도구는 이미 있다(`run_web_play.py --az-checkpoint`, 착수별 root visits·prior·Q).
   AZ가 v7과 같은 VCT1 약점을 보이는지, 아니면 더 기초적인 방어(must_block·열린 3)에서 지는지가 V8 teacher의 우선순위를 바꾼다.
   AZ raw must_block top-1이 0.10~0.28(§12.10)이므로 후자일 가능성이 높다. 그렇다면 A 계층(T1)만으로 충분하고 B 계층의 이득은
   작을 수 있다.
2. **사람 대 V8**은 V8 게이트 뒤에 둔다. 사람이 V8을 이긴 판은 V8의 남은 약점(§4.4 결정 근거)이고, 증명된 분기점은
   `vct_probes_v2` 후보다.
3. 새 로그는 `tests/fixtures/web_play_human_games_v2.json`(여러 agent)로 모으고, probe를 만들면 **그 게임은 학습에서 뺀다.**
4. 한 로그로 V8 설계를 바꾸기 전에 `check_v8_branch_points.py`처럼 "그 모듈이 있었으면 수가 바뀌었나"를 먼저 재현한다.

공유해 줄 것: `logs/web_play/` 중 AZ·V8 대국 폴더(`game.json`, `moves.csv`).

## 9. 작업 순서

| # | 작업 | 산출물 | 선행 |
|---|---|---|---|
| V8-0 | V7 동결 유지, 사람 패배 fixture | **완료**(`vct_probes_v1`, `SAME`), 반사실 검사(§2) | — |
| V8-1 | 기존 격리 규칙 유지, `analysis/mcts_v8.py` 골격(V7 위임), root record | **완료**(§11.1): 모듈 끔 = V7과 144/144 동일 | — |
| V8-2 | V8-A stage 4/5 VCT1 | **구현 완료**(§11.2, 경계조건 §11.5~§11.7). stage 4: 실제 probe 24개(§11.3). stage 5: 실제 국면 5개는 모두 SAFE 경로, UNSAFE→확장 경로는 scripted solver 단위 테스트만(§11.4) | V8-1 |
| V8-3 | V8-B 자기 VCT1 | **구현 완료, 게이트 미통과**(§4.2.9): recall 32/32, bounded proof 26/32, 독립 재검증 26/26(false positive 0). 실패 6개는 대칭 변형의 증명 비용. 게이트 기준 결정 필요 | V8-2 |
| V8-4 | V8 대 V7 실전 측정(§11.11) → V8-B D4 기준·V8-C 범위 결정 → V8-C | 착수 시간 분포, 모듈 발동률, 반사실 기록, 승패 | V8-2, V8-3 |
| V8-5 | V8-E 예산(`V8_PLAY`/`V8_TEACHER`) | V8 대 V7 100판 | V8-4 |
| V8-6 | V8_TEACHER 동결(해시, fingerprint) | 문서 확정 | V8-5 |
| T8-1 | V8 대국 생성 + A/B 계층 dataset | dataset, stats | V8-6 |
| T8-2 | V8-A1, V8-D arm | 직접 대국, 판정 | T8-1, T1 결과 |
| T8-3 | V8-T4 | 학습 루프 변경 | T8-2 결과 |
| T8-4 | S4 + 가장 좋은 teacher | 64×4 최종 후보 | S4·T8 결과 |
| — | Stage 9 판단 | §12.10 기준 | 위 전체 |

V8-D(양방향 planner)와 V8-G 강화는 V8-5 패배 분석에서 근거가 나오면 V8-6 전에 넣는다.

## 10. 열린 질문

- V8-A가 stage 4 집합이 모두 UNSAFE일 때 root로 넓히는 비용: 사람 패배 3판은 stage 4 안에 SAFE가 있었다. 모두 UNSAFE인 국면의
  빈도와 비용은 V8-4에서 잰다.
- V8-C의 사전 우선순위 필터가 `164723` 같은 국면에서 SAFE를 얼마나 빨리 찾는지. 필터 miss를 SAFE로 간주하지 않으며,
  proven SAFE가 생기면 root를 그 집합으로 제한하므로 필터 완전성에 안전성을 맡기지 않는다.
- B 계층 soft target이 V5 트리의 휴리스틱 순서를 그대로 옮기는지: V8-D 대 V8-A1에서만 판단한다.
- depth 2: V8 온라인/teacher 경로에서는 보류. 오프라인 분석(`build_vct_probes.py --vct-depth 2`)으로 비용과
  UNKNOWN 비율을 먼저 재고, 그 결과가 실용적일 때만 범위를 다시 연다.

## 11. 구현 기록

### 11.1 V8-1 골격 (완료)

- 파일: `src/analysis/mcts_v8.py`(`mcts_search_v8`, `V8_DEFAULTS`, `SearchDiagnostics`), `src/analysis/mcts_v8_agent.py`
  (`MCTSV8Agent`, `agents`에서 export하지 않음), `tests/test_mcts_v8.py`, `scripts/run_v8_gates.py`.
- 동결 파일(V2~V7, planner, agent)은 수정하지 않았다. `mcts_search_v7`의 흐름을 V8 파일에 옮기고, 필요한 동결 함수만 import한다.
  root 방문 기록을 위해 `_search_v5_tree`를 `_search_tree_v8`로 옮겼다(루프와 random 호출 순서 동일, 자식 목록만 반환).
- **모듈을 끄면 V7과 같은 수를 둔다.** 사람 대국 6판의 모든 비종료 국면 144개에서 같은 `Random(seed)`로
  `mcts_search_v7`과 144/144 일치했다(경로: tree 74, stage 4 24, stage 2 22, own VCF 8, stage 1·3 각 6, stage 5 4).
  단위 테스트는 그중 12개 국면(한 판), **경로별 대표 국면 8개**(tree normal·tactical, stage 1~5, own VCF), agent 수준 비교를 돈다.
- 진단: `v8_route`, `v8_v7_move`(V7이 뒀을 수), `v8_changed`, `v8_vct_checked`(검사 순서대로 (수, 상태)),
  `v8_vct_widened`, `v8_vct_calls`, `v8_vct_nodes`, `v8_vct_budget_exhausted`, `v8_vct_seconds`,
  `v8_root_visits`(tree 착수의 root 자식별 (수, 방문, 평균값)). §4.6 root record의 원자료다.

### 11.2 V8-A stage 4/5 VCT1 safety (구현 완료)

- `_BudgetedSolver`(ThreatSolver 하위 클래스): 착수당 **VCF 호출 수**와 **총 VCF 노드**를 제한한다. 개별 VCF 호출 한도
  (`vct_vcf_node_limit`)에 걸리면 기존처럼 UNKNOWN이고, 총 예산에 잘린 호출은 캐시에 넣지 않고 중단한다(잘린 탐색을
  "끝난 탐색"으로 오인하지 않는다). 중단돼도 판은 원상 복구된다(테스트).
- 개발 기본값(`V8_DEFAULTS`, V8-5에서 확정):

| 키 | 값 | 의미 |
|---|---|---|
| `stage_vct_safety` | True | V8-A 켜기(False면 V7과 동일) |
| `vct_vcf_node_limit` | 20,000 | VCF 한 번의 노드 한도(오프라인 fixture는 100,000) |
| `vct_call_limit` | 3,000 | 착수당 VCF 호출 수 |
| `vct_node_budget` | 200,000 | 착수당 총 VCF 노드 |

- 처음에는 호출 수만 제한했다. 그런데 VCF 한 번이 20,000노드에 11초까지 걸려서 `006-s3`이 10분 넘게 끝나지 않았다.
  그래서 총 노드 예산을 추가했다(이 컨테이너에서 약 0.5~0.7 ms/노드).

### 11.3 게이트 결과: `must_defend_vct` (V8-A)

`python scripts/run_v8_gates.py`(기본 `V8_DEFAULTS`, seed 1, 이 컨테이너). 최신 실행은 §11.6 수정(호출 수도 같은 몫으로 분배)
후의 결과다. **24/24 통과, 24개 모두 V8이 직접 VCT1-SAFE로 증명한 수, 예산 소진 0회**(예산은 썼지만 다 쓰지는 않았다). 24개 모두
stage 4 경로이고, V7이 뒀을 수(`avoid_moves`)를 버리고 `correct_moves`의 수를 골랐다.

| probe(기본 국면) | D4 8개 결과 | 착수 시간 | VCF 노드 | VCF 호출 |
|---|---|---|---|---|
| `000`(163810 15수, v7 흑) | 8/8 | 11.0~52.6초 | 10,707~58,840 | 1,141~1,181 |
| `003`(163903 13수, v7 흑) | 8/8 | 8.7~14.2초 | 8,580~14,323 | 596~605 |
| `006`(164445 14수, v7 백) | 8/8 | 12.9~30.3초 | 14,591~34,651 | 573~668 |
| 전체 | **24/24(증명 SAFE 24)** | 평균 19.9초, 중앙값 13.8초, p95 48.6초, 최대 52.6초 | 최대 58,840(예산의 29%) | 최대 1,181(한도의 39%) |

실행별 요약(모두 24/24 통과, 표의 중앙값은 `statistics.median`, 이전 두 실행은 짝수 표본의 위쪽 중앙값):

| 실행 | 예산 분배 | 증명 SAFE 확인 | 평균 | 중앙값 | p95 | 최대 | 최대 노드 |
|---|---|---|---|---|---|---|---|
| `425ce40` | 노드만 2회 분할 | 안 함 | 21.9초 | — | — | 37.2초 | 41,480 |
| `bcafad4` | 노드만 같은 몫 라운드 | 24/24 | 21.7초 | 22.4초 | 36.3초 | 37.6초 | 41,480 |
| `cc7ba7f` | 노드 + 호출 같은 몫 라운드 | 24/24 | 19.9초 | 13.8초 | 48.6초 | 52.6초 | 58,840 |
| 현재(§11.9) | 위 + 반올림 경계 수정 | 24/24 | 20.6초 | 14.2초 | 50.4초 | 55.5초 | 58,840 |

- 호출 수 분배로 `003`·`006`은 빨라졌다(호출 870~1,479 → 573~668). 그러나 `000`의 대칭 2개(s2, s6)는 30초 → 49~53초로 느려졌다.
  첫 라운드 호출 몫(3,000/8 = 375)에 판정이 잘려 다음 라운드에서 상위 VCT 탐색을 다시 진입한 것은 확인됐지만, **call-share cut은
  새 VCF 호출을 시작하기 전에 발생하므로 그 자체가 진행 중 VCF의 부분 node를 버리는 것은 아니다.** 실제 증가분이 node-share cut,
  재귀 재진입/순회 비용, 둘의 조합 중 무엇인지는 현재 진단만으로 분리할 수 없다. V8-B 구현 전후로 round/cut 종류를 기록해
  V8-4에서 원인을 측정한다.
- 대칭 변형끼리 비용이 5배까지 다르다. 동결 VCF solver와 `ordered_moves`의 후보 순서가 대칭에 대해 불변이 아니기 때문이다.
  결과(고른 수)는 대칭과 맞게 바뀌었다.
- **1회 예산 방식에서는 21/24였다.** V7의 수부터 끝까지 증명하던 첫 구현은 `006`의 대칭 3개에서 V7 수의 UNSAFE 증명에 20만 노드를
  다 쓰고(최대 168초) SAFE 수를 검사하지 못해 V7 수를 그대로 뒀다. 1회 방식에서 V7 수 증명에만 최대 73,324노드가 든 국면도
  있었으므로, 예산만 키우는 것보다 분배가 낫다.
- 이 수치는 사람 패배 3판(같은 수법)에서 나온 국면뿐이다. 실제 대국의 stage 4/5 빈도와 비용은 V8-4에서 잰다. 사람 대국
  144국면(§11.1)에서는 V8-A가 도는 stage 4/5가 28개(19%)였다. 이 비율과 위 비용이면 평균 착수 시간이 크게 오를 수 있다.
- 벽시계 기준 VCF 노드 처리량은 1,109 노드/초다(VCF 밖의 수 생성·금수 판정 시간 포함). 노드 예산은 시간 상한이 아니다.

### 11.4 stage 5 실측

사람 대국 6판의 stage 5 국면 4개와 V5 단위 테스트의 합성 국면 1개에서 V8-A를 돌렸다. 5개 모두 V7의 차단점이 VCT1-SAFE로
증명되어 그대로 뒀다(0.5~22.1초, 최대 24,091노드, 확장 없음). 사람 패배 분석(§2)에서도 stage 5 착수는 모두 SAFE였다.
그래서 **"stage 5 차단점이 UNSAFE → root 확장"** 경로는 실제 국면이 아직 없다. 이 경로는 실제 stage 5 국면에 scripted solver를
붙인 단위 테스트로만 검증한다(§11.5·§11.6).

### 11.5 외부 검토 반영 (`425ce40` 이후)

| 지적 | 사실 확인 | 조치 |
|---|---|---|
| root로 넓힌 뒤 fallback이 증명된 UNSAFE인 V7 수를 고를 수 있다 | **맞음.** 확장 조건이 "강제 방어가 모두 UNSAFE"인데, fallback은 강제 방어 중 UNKNOWN만 찾았다. 그래서 root 후보가 UNKNOWN이어도 V7 수(UNSAFE)로 돌아갔다 | fallback 순서를 §4.1처럼 고쳤다(V7 수가 UNSAFE가 아니면 V7 수 → 미판정 강제 방어 → 미판정 root 후보, 2·3은 `_not_immediately_lost` 통과 필요 → 모두 UNSAFE일 때만 V7). 테스트 `test_widened_unknown_beats_a_proven_loss` |
| 2차 pass에서 첫 UNKNOWN 후보가 남은 예산을 모두 쓸 수 있다 | **맞음.** 2차는 몫 제한 없이 순서대로 돌았다. 또 첫 수정안(몫을 두 배로 늘리다 "나머지 전부"로 바꾸는 방식)도 같은 문제가 남아 테스트에서 실패했다 | 모든 라운드를 같은 몫으로 돌리고, 몫은 두 배씩 늘리되 "남은 예산 ÷ 미결 후보 수"를 넘지 않게 했다. 테스트: A(15만 노드, UNSAFE)·B(6만 노드, SAFE)·예산 20만에서 B를 고름 |
| gate가 fixture 정답만 보고 V8의 SAFE 증명은 보지 않는다 | **맞음** | `run_v8_gates.py`가 `v8_vct_checked[move] == SAFE`도 요구한다. 예산 소진 수, 시간 중앙값·p95·최대, 노드/초를 요약에 남긴다. 재실행 결과 24/24(증명 SAFE 24) |
| stage 5 전용 V8 테스트가 없다 | **맞음** | 실제 stage 5 국면 테스트 1개(SAFE 유지), scripted solver 정책 테스트 4개(SAFE 유지, UNSAFE→확장→SAFE, 미판정 방어 우선, 모두 UNSAFE면 V7). 모두 판 복원 확인 |
| V7 동등성 CI 테스트는 한 판 12국면뿐이다 | **맞음** | 경로별 대표 국면 8개(tree normal·tactical, stage 1~5, own VCF) 동등성 테스트를 추가했다. 경로 자체도 검사한다. 144국면 전체는 여전히 수동 실행이다 |
| 노드 예산은 시간 상한이 아니다 | **맞음** | 코드 변경 없음. gate가 시간 분포와 노드/초를 기록한다. V8-4에서 같은 지표를 실제 대국으로 잰다 |
| GitHub에 `425ce40`의 CI 기록이 없다 | **맞음** | `.github/workflows/ci.yml`은 `pull_request`와 `main` push에서만 돈다. 이 브랜치의 테스트 결과는 컨테이너 로컬 실행이다. PR을 열면 CI가 돈다 |

### 11.6 2차 외부 검토 반영 (`bcafad4` 이후)

| 지적 | 사실 확인 | 조치 |
|---|---|---|
| root로 넓힌 직후 예산이 거의 없으면 증명된 UNSAFE인 V7 수로 돌아간다 | **맞음, 범위는 더 넓음.** (a) 남은 노드가 root 후보 수의 2배보다 작으면 첫 목표 몫이 0이라 root를 제대로 공평하게 검사하지 못했고, fallback은 `statuses`에 있는 UNKNOWN root만 받았다. (b) 마지막 강제 방어를 UNSAFE로 증명하면서 예산이 딱 떨어지면 `not solver.exhausted` 조건 때문에 root 목록 자체를 만들지 않았다 | root 목록은 강제 방어가 모두 UNSAFE이면 예산과 상관없이 만든다. fallback은 미검사 root 후보도 받는다(`_not_immediately_lost` 통과 필요). §11.7에서 half-share 0과 fair-share 0을 분리해 잔여 독점도 제거했다 |
| 같은 몫 분배가 노드에만 적용되고 VCF 호출 한도는 전역 공유다 | **맞음** | 호출 수도 노드와 같은 규칙(첫 라운드 `1/(2n)`, 두 배씩, 남은 양의 균등 분할 이하)으로 나눈다. 테스트: 노드는 넉넉하고 A가 호출 2,900개, B가 400개를 써야 할 때 B를 고름 |
| gate 중앙값이 짝수 표본에서 위쪽 값이다 | **맞음** | `statistics.median` 사용 |
| §11.3 표의 시간 범위와 전체 요약이 다른 실행에서 나왔다 | **맞음**(표는 `425ce40`, 요약은 `bcafad4` 실행) | 표를 최신 실행 하나로 다시 만들고, 실행별 요약 표를 따로 뒀다 |
| "예산을 사용하지 않았다"가 아니라 "소진하지 않았다" | 이 문서에는 "예산 소진 0회"로 적혀 있었다. 뜻을 분명히 하려고 사용률(노드 최대 29%, 호출 최대 39%)을 같이 적었다 | 문구 보강 |

새 테스트 3개(호출 몫, 몫 0, 예산 딱 떨어짐)는 `bcafad4` 코드에서 실패하고 수정 후 통과하는 것을 확인했다. V8 단위 테스트는
18개 → 21개다.

### 11.7 3차 외부 검토 반영 (`cc7ba7f` 이후)

남은 scheduler 경계조건도 사실이었다. 기존 코드는 첫 목표 몫 `remaining // (2*n)`이 0이면 곧바로 마지막 잔여 경로로 들어갔기
때문에, 실제로는 `remaining // n >= 1`이라 모든 후보에게 1단위씩 줄 수 있는 `n <= remaining < 2n` 구간에서도 첫 후보가
남은 예산을 독점할 수 있었다. node와 call 양쪽에 같은 문제가 있었다.

수정 규칙:

1. 매 라운드 `fair_node = remaining_nodes // pending`, `fair_call = remaining_calls // pending`을 먼저 계산한다.
2. 두 fair share가 모두 1 이상이면 초기 half-share가 0이어도 실제 share를 최소 1로 올린다. 어느 후보도 다음 후보의 1단위를 뺏지 못한다.
3. 둘 중 하나라도 fair share가 0이면 특정 후보에게 leftover를 주지 않는다. 모든 pending 후보를 `0 node / 0 call`로 한 번만
   실행해 새 VCF 자원 없이 끝나는 구조적 판정만 허용하고 종료한다.
4. fallback은 기존처럼 UNKNOWN/미검사 수를 proven-UNSAFE보다 앞에 둔다.

회귀 테스트를 두 개 추가했다: 6 calls/4 candidates와 6 nodes/4 candidates에서 첫 후보가 전체 6을 요구하고 두 번째 후보가 1만으로
SAFE가 되는 경우다. 두 테스트는 `cc7ba7f`의 scheduler라면 첫 후보가 잔여를 독점하는 경계를 직접 겨냥한다.

### 11.8 다음 작업

1. V8-4 pilot 실행(§11.11.4.1, 데스크톱) → 비용 확인 → 본 측정(50쌍 100판). runner는 구현 완료.
2. V8-4 결과로 V8-B D4 기준(§4.2.9의 1~3)과 V8-C 범위를 결정한다.
3. 웹 대국 V8 선택지는 추가했다(`python scripts/run_web_play.py` → 상대 "MCTS V8 (개발)"). 사람 대 V8 로그는 §8대로 모은다.

### 11.9 `6da1435` 검증 (실행)

`6da1435`(반올림 경계 수정)를 이 컨테이너에서 직접 실행해 확인했다.

- **수정 자체는 맞다.** 새 테스트 2개(6 calls / 4 후보, 6 nodes / 4 후보)는 `cc7ba7f` 코드에서 실패하고 `6da1435`에서 통과한다.
  즉 `n ≤ remaining < 2n` 구간에서 첫 후보가 잔여를 독점하던 버그는 실제로 있었고, 이번에 고쳐졌다. §11.6에 "남은 양은 후보당
  1단위도 안 된다"고 적은 것은 이 구간에서 틀렸다.
- **그러나 `6da1435`에서 기존 테스트 1개가 실패했다.** `test_exhausted_budget_keeps_unrefuted_v7_move`(VCF 호출 한도 1). 강제 방어 4개에
  `1 // 4 = 0`이라 zero-budget pass만 돌고, 호출 1개는 쓰이지 않은 채 `exhausted = False`로 남았다. 고른 수(V7 수)는 맞지만,
  진단과 게이트 통계는 "예산 소진 없음"으로 보고했다. 실제로는 예산 부족으로 아무것도 검사하지 못한 상태다.
- **보완:** zero-budget pass 뒤에 `exhausted = True`로 표시한다(§4.1). 테스트 1개 추가: 3 calls / 4 후보에서 누구도 잔여를 받지 않고,
  4개 모두 0/0 몫으로 한 번씩 호출되며, 예산이 소진으로 보고된다. 기존 실패 테스트는 수정 없이 통과한다.
- 단위 테스트 24개 통과(V8). 동결 V2~V7 해시 불변.
- 게이트 재실행: **24/24, 증명 SAFE 24, 예산 소진 0회.** 노드·호출 수는 `cc7ba7f` 실행과 24개 모두 같다(이 24개에서는 반올림
  경계에 걸리지 않는다). 시간은 평균 20.6초, 중앙값 14.2초, p95 50.4초, 최대 55.5초로, 같은 노드 수에서 ±5% 안의 측정 잡음이다.

### 11.10 5차 검토 반영 (`8d8bc54` 이후)

| 지적 | 사실 확인 | 조치 |
|---|---|---|
| `_attack_check()`의 `game.done` 판정이 `Game.play()`의 `to_play` 동작과 반대 | **맞음.** `Game.play()`는 5목이 나면 `to_play`를 바꾸지 않는다(끝난 판의 `to_play`는 이긴 쪽). 그런데 `winner == -to_play`로 비교해서, 5목을 만드는 공격수를 REFUTED로 읽었다. 5목 수는 공격 후보가 아니고(`_fast_pattern_features_for_move`가 4·열린 3 개수를 0으로 준다) stage 1이 먼저 두므로 실제 흐름에는 닿지 않았다 | `winner is not None`이면 WIN, 가득 찬 판이면 REFUTED. 테스트 추가 |
| (추가 발견) 같은 가정이 공용 solver `ThreatSolver._after_move`에도 있다 | **맞음, 더 중요함.** "방금 둔 쪽"의 상태를 볼 때 `opponent = game.to_play`로 두고 `game.done`이면 `winner == opponent`를 UNSAFE로 읽었다. 끝난 판에서는 `to_play`가 방금 둔 쪽이라 **자기 5목을 패배(UNSAFE)로 판정**했다. 실험: 흑이 열린 4에서 5목을 만드는 두 수가 모두 UNSAFE | 방금 둔 쪽의 착수로 판이 끝나면(5목 또는 판 가득 참) 항상 SAFE. 테스트 2개 추가(끝난 판 SAFE, `decision`에서 5목 수 SAFE) |
| attack gate에서 fixture 밖 WIN도 fresh 100k 재검증 | **맞음.** 기존에는 `bounded`(정답 집합 안) 조건 뒤에만 재검증했다 | V8이 WIN을 선언하면 항상 재검증한다. 정답 집합은 완전 집합이므로 집합 밖 WIN이나 재검증 실패는 `false_positive`로 센다 |

**solver 수정의 영향 범위.** 오판은 "둘 차례인 쪽에게 즉시 5목 수가 있는 국면"을 `decision`으로 분류할 때만 생긴다.

- solver 재귀 안에서는 생기지 않는다. `_after_move`는 그 전에 `ours`(방금 둔 쪽의 즉시 5목 수)를 검사해 1개면 강제 차단 수순,
  2개 이상이면 SAFE로 끝낸다. 그래서 재귀 속 `decision`은 둘 차례인 쪽에게 5목 수가 없는 국면에서만 호출된다.
- V8에서는 stage 1(내 5목)과 stage 2(상대 5목 차단)가 먼저 처리한다. 상대 5목 자리가 흑 금수라 stage 2가 막지 못하고 V8-B가
  도는 경우에도, 백의 다른 응수들은 `ours` 검사에서 SAFE가 되므로 틀린 WIN이 나오지 않는다.
- 확인: 수정 후 V8-A s0 3개와 V8-B s0 4개를 다시 돌렸다. **7개 모두 이전 실행과 수·노드·호출이 같고, false positive 0.**
  단위 테스트 3개(새 테스트)는 수정 전 코드에서 실패하고 수정 후 통과한다. `vct_probes_v1.json`과 T1/T2 dataset을 만든 코드도 같은
  solver를 쓰지만, 위 이유로 결과가 바뀌지 않는다고 판단해 다시 만들지 않았다(재생성은 1~2시간).

### 11.11 V8-4 설계 — 실전 측정

#### 11.11.1 목적

게이트는 고정 국면에서 "맞는 수를 증명하는가"만 본다. V8-4는 **실제 대국에서의 비용과 효과**를 잰다. 결과로 세 가지를 정한다.
(a) V8-B D4 기준(§4.2.9의 1~3), (b) V8-C 범위, (c) V8-5의 `V8_PLAY`/`V8_TEACHER` 예산.

#### 11.11.2 측정 항목과 측정 방법

| 항목 | 측정 방법 | 한계 |
|---|---|---|
| 승패 | V8 대 V7, 무작위 2수 오프닝(반경 2)을 색 교환으로 두 번. score와 색별 score | 100판의 표준오차 약 5%p |
| 모듈 발동률 | 착수마다 `v8_route` 기록. `own_vct`(V8-B), `stage4`/`stage5`에서 `v8_changed`(V8-A가 V7 수를 바꿈) | — |
| 착수 시간 | V8·V7 각각 전체 시간, V8은 `v8_attack_seconds`, `v8_vct_seconds`를 따로. 평균·중앙값·p95·최대, 예산 소진률 | 이 컨테이너와 데스크톱의 절대 시간은 다르다. 비율로 비교 |
| V8-B가 수를 바꿨는가 | `own_vct` 착수에서 **V7이 같은 국면에 뒀을 수**를 따로 계산해 기록(`--counterfactual`). V7 수가 같은 WIN인지는 `proves_threat`로 확인 | V7 탐색 1회(1~3초)가 추가된다. 선택 옵션 |
| 그 변경이 승패를 바꿨는가 | 한 판에서는 알 수 없다. **짝지은 ablation**: 같은 오프닝·seed로 (A+B) 대 V7, (A만) 대 V7을 둔다. 두 arm의 score 차가 V8-B의 기여다 | 판수가 2배. pilot에서 V8-B 발동이 거의 없으면 생략 |
| V8-A 변경의 효과 | V8-A가 수를 바꾼 착수에서 V7의 수는 `v8_v7_move`로 이미 기록되고, 그 수의 상태도 `v8_vct_checked`에 있다(대개 UNSAFE) | — |
| 대칭 미탐의 실전 빈도 | 실전에는 정답이 없다. **사후 분석**: V8-B가 돌았지만 WIN이 없던 착수 중 표본(전부 또는 무작위 N개)을 큰 예산(per-VCF 100k, 총 예산 없음, 국면당 시간 상한)으로 다시 검사해 "V8-B가 놓친 증명된 WIN" 비율을 센다. V8이 진 판은 전부 검사한다 | 오프라인 비용이 크다. 데스크톱에서 별도 실행 |
| 결정성 | 같은 seed로 몇 판을 다시 두어 기보가 같은지 확인 | — |

#### 11.11.3 규모와 비용 추정

- 사람 대국 144국면 기준으로 V8-B는 착수당 평균 3.1초(중앙값 0.09초)다. V8-A는 stage 4/5 착수(약 19%)에서 10~57초다. V7 자체는
  착수당 약 1.3초다(mcts-v7.md §13). 대략 V8 착수당 평균 5~10초로 잡으면, 한 판(V8 착수 약 15~30수)은 2~5분이다.
- pilot 10쌍(20판): 1~2시간(1 프로세스). 본 측정 50쌍(100판): 6~8시간(1 프로세스), 4 프로세스면 2시간 안팎. 숫자는 추정이며
  pilot으로 확인한다.
- **최악 착수는 V8-B 200k + V8-A 200k 노드로 약 6분이다.** pilot에서 최악값이 이 근처면 본 측정 전에 예산을 줄이는 것을 먼저 검토한다.

#### 11.11.4 runner (`scripts/run_mcts_v8_benchmark.py`, 구현 완료)

- `scripts/run_mcts_v7_benchmark.py`의 구조(오프닝 생성, 색 교환, `derive_seed`, 결과 해시)를 그대로 따른다. V8 agent는
  `analysis.mcts_v8_agent`에서 직접 import한다(`agents`에 넣지 않는다, §4.8).
- 옵션(설계와 이름이 다른 것은 괄호): `--arm full|a_only|b_only|off`(설계의 `--v8-config`), `--pairs`, `--seed`(기본 8401),
  `--opening-random-plies 2`, `--opening-radius 2`, `--workers N`, `--counterfactual`, `--games-jsonl`(설계의 `--resume`:
  같은 파일로 다시 실행하면 끝난 판을 건너뛴다), `--output`. `--simulations`/`--tactical-simulations`는 smoke 테스트 전용이다
  (값이 출력에 기록된다).
- **짝지은 ablation:** 오프닝과 판 seed는 `--seed`와 쌍 번호로만 정해지고 arm과 무관하다. 같은 seed로 arm만 바꿔 돌리면 같은
  오프닝·같은 색·같은 seed에서 V8 설정만 다르다(테스트로 확인).
- 판 기록(JSONL 한 줄): 오프닝, seed, 색, 결과, 기보, 판 시간, V7 착수 시간, V8 착수별 `{ply, seconds, route, changed, v7_move,
  attack{status, rank, candidates, calls, nodes, exhausted, seconds}, vct{checked, widened, calls, nodes, exhausted, seconds}}`.
  `--counterfactual`이면 `own_vct` 착수에 `{v7_move, v7_move_attack_status, seconds}`(V7이 뒀을 수와 그 수의 VCT1 공격 판정)를
  더한다. 반사실 계산은 별도 난수열을 써서 대국 자체는 바뀌지 않는다.
- 요약: score, V8 색별 score, 평균 판 길이, route별 발동 수, route별 `changed` 수, V8/V7 착수 시간 분포(평균·중앙값·p95·최대),
  V8-B(돈 착수 수, WIN 수, 예산 소진, 시간 분포), V8-A(돈 착수 수, 수 변경, root 확장, 예산 소진, 시간 분포), 반사실 요약,
  결과 해시, git 커밋, 실제 V8/V7 설정.
- 확인(이 컨테이너, smoke 설정 simulations 4/8, 1쌍): full arm 2판 약 1분. 같은 JSONL로 다시 실행하면 0판을 두고 같은 결과 해시를
  낸다. `--workers 2`의 결과 해시가 1 프로세스와 같다. 테스트 3개(`tests/test_mcts_v8_benchmark.py`), 웹 대국 V8 테스트 1개.
- 사후 분석 스크립트(`scripts/analyze_v8_benchmark.py`: 패배 판 분기점, V8-B 미탐 표본 검사)는 pilot 결과를 본 뒤 만든다.

#### 11.11.4.1 pilot 실행 명령 (데스크톱 PowerShell)

```powershell
# 0. 최신 코드
git fetch origin ccr-ba71e723-f37v8m
git checkout ccr-ba71e723-f37v8m
git pull origin ccr-ba71e723-f37v8m

# 1. 빠른 동작 확인 (약 1~2분, 실제 설정 아님)
python scripts/run_mcts_v8_benchmark.py --arm full --pairs 1 --seed 1 --simulations 4 `
    --tactical-simulations 8 --counterfactual --output runs/v8_4/smoke.json

# 2. pilot: full arm 10쌍 20판, 4 프로세스, 반사실 기록 (추정 30분~1시간)
python scripts/run_mcts_v8_benchmark.py --arm full --pairs 10 --seed 8401 --workers 4 `
    --counterfactual --games-jsonl runs/v8_4/pilot_full.jsonl --output runs/v8_4/pilot_full.json `
    2>&1 | Tee-Object -FilePath runs/v8_4/pilot_full.log

# 3. (선택) 같은 오프닝으로 V8-A만: V8-B 기여를 보는 짝 비교
python scripts/run_mcts_v8_benchmark.py --arm a_only --pairs 10 --seed 8401 --workers 4 `
    --games-jsonl runs/v8_4/pilot_a_only.jsonl --output runs/v8_4/pilot_a_only.json `
    2>&1 | Tee-Object -FilePath runs/v8_4/pilot_a_only.log
```

중단되면 같은 명령을 다시 실행하면 끝난 판은 건너뛴다(`--games-jsonl`). 공유할 파일: `runs/v8_4/pilot_*.json`과 `.log`.

#### 11.11.5 판정 기준(미리 정해 둠)

| 관찰 | 결정 |
|---|---|
| V8 대 V7 score가 V7 대 V6(0.90)처럼 명확히 높고, 최악 착수 시간이 teacher 생성에 감당 가능 | V8-5(예산 확정)로 간다 |
| V8-B 미탐(대칭 포함)이 V8 패배 분기점에 반복해서 나온다 | §4.2.9의 3(V8 전용 VCF 순서)을 검토한다 |
| V8-B 발동이 드물고 ablation 차이도 없다 | V8-B 예산을 줄여 비용을 아낀다(§4.2.9의 1 확정) |
| V8 패배가 V8-A·V8-B 범위 밖(루트 선택)에서 난다 | V8-C를 시작한다 |
| 최악 착수 시간이 너무 길다 | 예산을 줄이고 게이트(s0 4/4, 24/24)를 유지하는 최소 예산을 찾는다 |
