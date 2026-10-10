# MCTS-v8 Teacher 설계서 (v1 확정, 개발 중)

> 상태: 설계 v1 확정(2026-09-30). **V8-1(골격)과 V8-A(stage 4/5 VCT1 safety) 구현 완료**, 구현·측정 기록은 §11.
> **V8-B(자기 VCT1 공격) 구현 완료, 조건부 통과**(s0 4/4, false positive 0, D4 26/32는 측정값) — §4.2.9, §11.10.
> V8-4 pilot 완료(§11.12). **V8-C(root VCT1 safety, 트리 후 검증) 구현 완료**, root 게이트 3/3 — §4.3.
> **2026-10-01: 두 트랙 분리(§12).** V8은 Track B(Hybrid)의 전술 모듈이 됐다. V8-C는 `aggressive`(기본)·`veto` 두 모드, runner에 `--opponent`.
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

### 4.3 V8-C — root VCT1 safety (구현, "트리 후 검증" 방식)

#### 4.3.1 근거

V8-4 pilot의 full 패배 3판 중 2판(쌍 6 흑 56수, 쌍 7 백 29수)이 트리 경로에서 VCF-SAFE·VCT1-UNSAFE 수를 둔 것이었다(§11.12).
사람 패배 `164723` 20수와 같은 종류다. §11.11.5에 **pilot 전에** 적어 둔 기준("V8 패배가 V8-A·V8-B 범위 밖(root 선택)에서 난다
→ V8-C 시작")에 해당한다.

#### 4.3.2 처음 설계(트리 전 root 필터)를 버린 이유 — 측정

처음 설계는 트리 전에 root 후보를 V7 순서로 VCT1 검사해 SAFE가 K개(기본 3) 나오면 멈추고, SAFE가 하나라도 있으면 **root를 증명된 SAFE로만
제한**하는 것이었다. 구현해서 pilot full의 트리 착수 무작위 20개(seed 5)와 위 3국면에서 쟀다.

| 방식 | V8-C 시간(20착수) | 꼬리 | 행동 |
|---|---|---|---|
| 트리 전 root 필터(K=3) | 평균 약 28초 | 36.9·94.0·186.6(소진)·201.0(소진)초 | **거의 모든 착수에서 root 20개 → 3개.** 조용한 국면에서도 V7 트리의 폭을 "가장 먼저 증명된 3개"로 자른다 |
| **트리 후 검증(채택)** | 평균 약 19초 | 106.7·225.6(소진)초 | **18/20은 V7의 수가 SAFE로 증명되어 V7과 같은 수.** 1착수는 V7 수 (6,5)가 UNSAFE로 증명되어 방문 5위 (4,10)(SAFE)로 바꿨다 |

트리 전 필터는 비용도 더 크고, V7의 판단을 거의 매 착수 바꾼다. 트리 후 검증은 보통 착수에서 V7과 같은 수를 두고(트리의 root와 난수열이 V7과
같다), 검증 1회만 한다. 덤으로 V7이 뒀을 수를 정확히 알 수 있어 `v8_changed`가 의미를 가진다.

#### 4.3.3 알고리즘(구현: `_verify_root_choice`)

1. V7의 트리를 그대로 돈다(M2 VCF tier, M3 금수 감점 포함). 고른 수 = V7의 수.
2. 그 수를 **예산의 절반**으로 혼자 VCT1 검사한다. SAFE면 그대로 둔다(보통의 경우).
3. 아니면 트리 선호 순서(방문 수 → 평균값)의 상위 `root_max_children`개(0이면 전부)를, V7 수가 아직 미결이면 그것도 포함해 공용 공정 분배
   (`_first_proven`)로 검사한다. 처음 증명된 SAFE를 둔다.
4. 상위 K개가 **모두 UNSAFE로 증명되면** 나머지 자식으로 검사를 넓힌다(K개만 보고 "전부 UNSAFE"로 결론 내리지 않는다, §11.13).
5. SAFE가 없으면: V7 수가 UNSAFE로 증명되지 않았으면 V7 수, 증명됐으면 미판정(UNKNOWN 또는 미검사)이고 즉시 지지 않는(`_not_immediately_lost`)
   첫 자식. 모두 UNSAFE면 V7 수.
6. 전용 `_BudgetedSolver`(캐시·예산 분리). 진단: `v8_root_checked`, `v8_root_rank`(1 = V7 수), `v8_root_calls/nodes/budget_exhausted/seconds`.

설정(개발값, V8-5에서 확정): `root_vct_safety` True, `root_vcf_node_limit` 20,000, `root_call_limit` 10,000, `root_node_budget` 200,000,
`root_max_children` §4.3.5.

#### 4.3.4 게이트 (`python scripts/run_v8_gates.py --gate root`, fixture `tests/fixtures/v8c_root_probes_v1.json`)

fixture는 패배 분기 3국면이다(pilot 쌍 6 흑 56수, 쌍 7 백 29수, 사람 `164723` 20수). 각 국면을 두 가지로 검사한다.

- **엔진 검사:** V8 기본값으로 둔 수가 트리 경로이고, 기록된 패배수가 아니며, V8-C가 SAFE로 증명했고, fresh `ThreatSolver(100k)`도 SAFE.
- **강제 검사:** 트리 선택은 원래 대국의 난수 상태에 달려 있어 재현되지 않는다(실제로 pilot 국면을 다시 두면 V7이 다른 수를 고르기도 한다).
  그래서 기록된 패배수를 "V7의 수"로 `_verify_root_choice`에 직접 넣는다(자식은 V6 root 순서). V8-C가 그 수를 UNSAFE로 증명하고,
  SAFE로 증명한 다른 수로 바꾸고, fresh solver도 SAFE여야 한다.

결과는 §4.3.5.

#### 4.3.5 측정 결과와 남은 문제

같은 3국면(엔진 검사 seed 1, 강제 검사)으로 `root_max_children`(K)와 `root_node_budget`(B)를 바꿔 쟀다(이 컨테이너, 3개 동시 실행이라
시간은 부풀려져 있다). "통과"는 엔진·강제 검사 모두 통과다.

| K / B | pilot-6 흑 56수 | pilot-7 백 29수 | 164723 20수 | 통과 |
|---|---|---|---|---|
| 전부 / 200k | 통과 | 실패(예산 소진, 강제 검사도 SAFE 못 찾음) | 실패(예산 소진, 미판정 fallback (7,11)) | 1/3 |
| 4 / 200k | 통과 | 실패(엔진: V7 수 (4,9) 미판정) | 실패(엔진: 예산 소진) | 1/3 |
| 8 / 200k | 통과 | 실패 | 통과 | 2/3 |
| **4 / 400k (채택)** | **통과** | **통과**(V7 수 (4,9) SAFE 증명에 128,917노드) | **통과**((10,8) UNSAFE 증명 → (5,6) SAFE) | **3/3** |

- **K 제한이 필요하다.** 날카로운 국면에서는 SAFE 증명 하나가 수만~십수만 노드다. 예산을 17~20개 자식에 공평하게 나누면 아무것도 증명되지 않는다.
- **예산 400k가 필요하다.** pilot-7은 V7 수 자체의 SAFE 증명에 128,917노드가 든다. 처음 단독 검사 몫(예산의 절반)이 이것보다 커야 한다.
- **K 제한의 함정(구현 중 발견, 고침):** 처음 K 구현은 상위 K개가 모두 UNSAFE로 증명되면 "전부 UNSAFE면 V7 수"로 끝나서, 검사하지 않은
  5위 이하 자식이 남아 있는데도 **증명된 패배수 (3,4)를 그대로 뒀다**(pilot-6, K=4). 지금은 상위 K개가 모두 UNSAFE면 나머지로 넓히고,
  fallback도 K 밖의 미검사 자식을 받는다(테스트 2개).
- **비용:** 채택 설정에서 세 국면의 V8-C 착수 시간은 평균 214초, 최대 350초다(동시 실행). 최대 285,472노드, 예산 소진 0. 조용한 국면에서는
  보통 V7 수 검증 1회(트리 표본 20착수 중 16착수가 0.0~8.2초, 200k 설정 기준)다. 꼬리는 V8-5에서 다룬다.
- **예산 부족의 위험(실제 확인):** 200k에서 `164723`은 V7 수 (10,8)을 UNSAFE로 증명해 피했지만, SAFE를 못 찾아 미판정 fallback으로 (7,11)을
  뒀다. 이 수는 §2.2의 오프라인 전수 분류에서 **VCT1-UNSAFE였다.** "증명된 패배수를 피한다"는 규칙만으로는 예산이 모자랄 때 실제 패배를 막지
  못한다. 그래서 V8-C는 예산 부족 자체를 줄이는 쪽(K 제한·예산)으로 맞췄다.

#### 4.3.6 착수당 비용 합

pilot의 최악 착수(168.5초)는 같은 착수에서 V8-B와 V8-A가 둘 다 비싼 경우였다. 트리 경로에서는 V8-B와 V8-C가 같은 착수에서 돈다.
지금은 모듈마다 예산이 따로라 최악값이 더해진다(V8-B 200k + V8-C 200k 노드). 모듈별 진단(calls/nodes/exhausted/seconds)은 따로 남기므로,
V8-5에서 **착수 전체 공유 상한**(예: 남은 착수 예산을 다음 모듈에 넘기는 방식)을 넣을 수 있다. 구조는 각 모듈이 `_BudgetedSolver`를
새로 만드는 지점 하나뿐이라, 거기서 남은 착수 예산을 `node_budget`으로 넘기면 된다.

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
| V8-4 | V8 대 V7 실전 측정(§11.11) → V8-B D4 기준·V8-C 범위 결정 → V8-C | **pilot 완료**(§11.12, §11.13). **V8-C 구현 완료**(§4.3): root 게이트 3/3(K=4, 400k). 본 측정은 V8-C 포함 후보로 | V8-2, V8-3 |
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

1. V8-C 비용 확인 완료(§11.14). V7 상대 점수로는 V8-C 효과를 잴 수 없어 50쌍 본 측정은 보류. 다음은 runner에 상대 선택 옵션을 넣고
   VCT를 찌를 수 있는 상대(`v8:b_only`)로 A+B+C 대 A+B를 짝 비교한다.
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

# 결과 폴더 (Tee-Object는 python보다 먼저 로그 파일을 열므로 폴더가 미리 있어야 한다)
New-Item -ItemType Directory -Force runs/v8_4 | Out-Null

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

#### 11.11.4.2 V8-C 포함 측정 명령 (§11.13 이후)

arm 이름이 바뀌었다: `full` = A+B+C, `ab` = A+B(pilot의 full), `a_only` = A만(pilot과 같음). 같은 seed면 오프닝·색·seed가 같아 짝 비교가 된다.
V8-C 때문에 착수당 최악 비용이 커졌으므로(수 분) 먼저 5쌍으로 비용을 본다.

```powershell
New-Item -ItemType Directory -Force runs/v8_4c | Out-Null

# 1. 비용 확인: V8-C 포함 5쌍 10판
python scripts/run_mcts_v8_benchmark.py --arm full --pairs 5 --seed 8401 --workers 4 --counterfactual `
    --games-jsonl runs/v8_4c/full.jsonl --output runs/v8_4c/full_p5.json 2>&1 | Tee-Object -FilePath runs/v8_4c/full_p5.log

# 2. 비용이 감당되면 같은 파일로 50쌍까지 이어서 (앞의 10판은 건너뜀)
python scripts/run_mcts_v8_benchmark.py --arm full --pairs 50 --seed 8401 --workers 4 --counterfactual `
    --games-jsonl runs/v8_4c/full.jsonl --output runs/v8_4c/full.json 2>&1 | Tee-Object -FilePath runs/v8_4c/full.log

# 3. 짝 비교: V8-C 없는 V8(A+B), 같은 seed
python scripts/run_mcts_v8_benchmark.py --arm ab --pairs 50 --seed 8401 --workers 4 `
    --games-jsonl runs/v8_4c/ab.jsonl --output runs/v8_4c/ab.json 2>&1 | Tee-Object -FilePath runs/v8_4c/ab.log

# 4. 사후 분석 (짝 반사실 + 패배 분기점)
python scripts/analyze_v8_benchmark.py --run runs/v8_4c/full.json --baseline runs/v8_4c/ab.json `
    --output runs/v8_4c/analysis.json
```

#### 11.11.5 판정 기준(미리 정해 둠)

| 관찰 | 결정 |
|---|---|
| V8 대 V7 score가 V7 대 V6(0.90)처럼 명확히 높고, 최악 착수 시간이 teacher 생성에 감당 가능 | V8-5(예산 확정)로 간다 |
| V8-B 미탐(대칭 포함)이 V8 패배 분기점에 반복해서 나온다 | §4.2.9의 3(V8 전용 VCF 순서)을 검토한다 |
| V8-B 발동이 드물고 ablation 차이도 없다 | V8-B 예산을 줄여 비용을 아낀다(§4.2.9의 1 확정) |
| V8 패배가 V8-A·V8-B 범위 밖(루트 선택)에서 난다 | V8-C를 시작한다 |
| 최악 착수 시간이 너무 길다 | 예산을 줄이고 게이트(s0 4/4, 24/24)를 유지하는 최소 예산을 찾는다 |

### 11.12 V8-4 pilot 결과 (데스크톱, `a02c3ab`, seed 8401, 10쌍 20판, 4 프로세스)

입력: `runs/v8_4/pilot_full.json`(full, `--counterfactual`), `runs/v8_4/pilot_a_only.json`(a_only). 두 arm은 같은 오프닝·색·seed다.

#### 승패

| arm | 승/무/패 | score | V8 흑 / 백 | 평균 판 길이 |
|---|---|---|---|---|
| full (V8-A + V8-B) | 16 / 1 / 3 | **0.825** | 0.85 / 0.80 | 59.0수 |
| a_only (V8-A만) | 13 / 2 / 5 | 0.700 | 0.60 / 0.80 | 95.0수 |

- **짝 비교:** 20쌍 중 9판은 두 arm의 기보가 완전히 같다(V8-B가 한 번도 발동하지 않았거나, V8-B가 둔 수가 a_only의 수와 같았다).
  결과가 달라진 판은 3판이고 **모두 full이 낫다**(패→승 2판: 쌍 2·8 흑, 무→승 1판: 쌍 5 흑). 나빠진 판은 0이다.
  표본이 작아 통계적으로는 유의하지 않다(3:0 부호 검정 p = 0.25).
- full이 진 3판(쌍 1 백, 쌍 6 흑, 쌍 7 백)은 a_only와 기보가 같다. **V8-B와 무관한 패배다.**

#### V8-B (full)

- 338착수에서 돌았고 **13번 WIN을 증명해 두었다.** WIN 착수부터 판 끝까지 5~19수였다.
- **짝 효과(기계적 ablation):** V8-B가 발동한 13판에서 full은 13승, 같은 판의 a_only는 10승 1무 2패다. a_only도 이기던 10판은
  승리를 유지했고, a_only가 진 2판과 비긴 1판을 full이 이겼다. 13판 중 11판은 **정확히 V8-B가 둔 착수에서 처음** 두 arm의 기보가
  갈라졌다(나머지 2판은 a_only도 같은 수를 둬 기보가 같다). V8-B가 발동하지 않은 7판 중 두 arm이 갈라진 판은 0이다.
- **반사실(정정):** runner의 `--counterfactual` 기록(V7 재탐색 수: REFUTED 11, WIN 2)은 **독립 난수열로 V7을 다시 돌린 것**이라
  실제 a_only 수와 13번 중 2번만 같았다. 그래서 그 기록만으로 "V7이 11번의 강제승을 놓쳤다"고 쓰면 안 된다(처음 기록이 틀렸다).
  대신 **실제 a_only가 같은 국면에서 둔 수**를 공격 판정했다(`scripts/analyze_v8_benchmark.py`, 결과
  `docs/mcts-v8-results/pilot_analysis.json`): **WIN 2(V8과 같은 수), REFUTED 11, UNKNOWN 0.** 즉 실제 baseline은 13번 중 11번
  그 자리에서 VCT1 강제승을 두지 않았고, V8-B가 그것을 찾았다. (그 11판 중 a_only가 결국 이긴 판은 8판이다.)
- 선택 수의 후보 순위: 1~10위(후보 4~13개). 10위·호출 3,427개인 경우도 있어 호출 한도 10,000의 근거가 실전에서도 확인됐다.
- 시간: 중앙값 0.08초, 평균 4.6초, p95 24.8초, 최대 135.9초. 예산 소진 9회(2.7%).

#### V8-A

| arm | 돈 착수 | 수 변경 | root 확장 | 예산 소진 | 시간 평균 / p95 / 최대 |
|---|---|---|---|---|---|
| full | 95 | 10 | 14 | **9 (9.5%)** | 11.7 / 74.6 / 154.4초 |
| a_only | 151 | 18 | 19 | **21 (13.9%)** | 15.7 / 77.0 / 92.1초 |

- **수 변경의 대부분은 "V7 수가 UNKNOWN인데 다른 수가 먼저 SAFE로 증명된" 경우다**(full 10번 중 8번, a_only 18번 중 14번).
  §4.1에서 예상한 "가장 먼저 증명된 SAFE" 효과다. V7 수가 지는 수라는 증명은 없었다. 나머지(full 2, a_only 4)는 V7 수가 UNSAFE로
  증명되고 SAFE가 없어 미판정 수로 fallback한 경우다(모두 root 확장 + 예산 소진).
- 실전의 stage 4/5 국면은 게이트 표본(0/24)보다 예산 소진이 훨씬 잦았다. full은 stage 4에서 7회, stage 5에서 2회, a_only는
  stage 4에서 15회, stage 5에서 6회다.

#### 비용

| | full | a_only |
|---|---|---|
| V8 착수 시간 평균 / 중앙값 / p95 / 최대 | 5.45 / 0.95 / 23.9 / **168.5초** | 3.49 / 0.97 / 11.2 / 92.1초 |
| V7 착수 시간 평균 | 0.83초 | 1.08초 |
| 판당 시간(1 프로세스 기준) | 178초 | 211초 |

- full의 최악 착수(168.5초, 쌍 1 흑 90수)는 stage 4 국면에서 V8-B(133.4초, 199,997노드, 예산 소진) 뒤에 V8-A(31.3초)가 돈 경우다.
  §4.2.8의 "두 모듈 합" 우려가 실제로 나왔다. 이 국면은 V8-5 예산 조정의 기준 국면으로 `tests/fixtures/v8_budget_probes_v1.json`에 남겼다.
- **비용의 대부분은 꼬리다.** full의 V8 사고 시간 3,069초 중 V8-B가 1,564초(51%), V8-A가 1,113초(36%), 나머지(V7 탐색·전술)가 약 13%다.
  V8-B는 중앙값 0.08초지만 평균 4.6초·p95 24.8초·최대 135.9초인 heavy-tail 구조다. "중앙값이 작아서 싸다"고 읽으면 안 된다.
- 그런데 판당 시간은 full(178초)이 a_only(211초)보다 짧다. V8-B가 강제승을 찾아 판을 평균 95수에서 59수로 줄였기 때문이다. 이 pilot에서는
  V8-B 때문에 teacher 처리량이 나빠지지 않았다. 다만 한 착수 136~168초는 사람 대국과 대량 생성에서 관리해야 할 꼬리다.
- 판당 약 3분이다. 4 프로세스면 시간당 약 80판이다.

#### 패배 3판의 분기점 (V8 착수를 per-VCF 100k·총 2M 노드로 VCF/VCT1 판정, 마지막 8수)

증명 기록(국면 해시, 검사 수, 상태, 노드·호출·시간)은 `docs/mcts-v8-results/pilot_analysis.json`의 `losses`에 있다
(`python scripts/analyze_v8_benchmark.py --run docs/mcts-v8-results/pilot_full.json --baseline docs/mcts-v8-results/pilot_a_only.json --output ...`로 재현).

| 판 | 처음 지는 V8 수 | 경로 | 그 직전 V8 수 | 해석 |
|---|---|---|---|---|
| 쌍 6, V8 흑 | 56수 (10,7): VCF-SAFE, **VCT1-UNSAFE** | tree | 54수 VCT1-SAFE | **V8-C 범위**(root에 VCT1 검사가 없음) |
| 쌍 7, V8 백 | 29수 (4,5): VCF-SAFE, **VCT1-UNSAFE** | tree | 27수 VCT1-SAFE | **V8-C 범위** |
| 쌍 1, V8 백 | 59수 VCF-UNSAFE. 57수 (12,3)은 오프라인 100k·8분으로도 UNKNOWN | stage 4 (V8-A 예산 소진, root 확장 후 fallback) | — | V8-A가 강제 방어 18개를 모두 UNSAFE로 증명하고 root까지 넓혔지만 SAFE를 못 찾은 어려운 국면. 이전 분기점은 미분석 |

#### 판정 (§11.11.5)

1. **V8-B: 유지.** 짝 비교에서 결과가 달라진 3쌍이 모두 개선이고 나빠진 판이 0이며, 실제 baseline 수 기준으로도 11번의 VCT1 강제승을
   새로 찾았다. 틀린 공격은 0이다. 다만 **유망한 것이지 통계적으로 입증된 것은 아니다**(짝 score 차 +12.5%p, bootstrap 95% 구간이
   대략 0~+27.5%p라 0을 확실히 배제하지 못한다). 비용은 heavy-tail이다(위).
   D4 26/32 기준은 §4.2.9의 1(방향 한정 게이트)을 유지한다. 패배 3판 중 V8-B 미탐이 원인인 판은 없었다.
2. **V8-C: 시작 근거가 생겼다.** full 패배 3판 중 2판이 tree 경로에서 VCT1-UNSAFE 수를 둔 것이다. 사람 패배 `164723`과 같은 종류다.
   두 분기 국면(쌍 6 흑 56수, 쌍 7 백 29수)을 V8-C 회귀 fixture로 쓴다.
3. **예산:** 최악 168초와 V8-A 소진 9~14%는 V8-5에서 다룬다. 같은 착수에서 V8-B와 V8-A가 둘 다 비싼 경우가 최악값을 만든다.
4. **강도 판단은 보류:** 20판은 작다(score 표준오차 1 SE가 약 8~10%p, 95% 불확실성은 그보다 훨씬 크다). 본 측정 50쌍은 V8-C까지 넣은 후보로 돌리는 것이 데스크톱 시간을 아낀다.

### 11.13 6차 검토 반영 (`facfaa5` 이후, pilot 분석 검토)

| 지적 | 사실 확인 | 조치 |
|---|---|---|
| "11/13은 V7이 놓친 강제승" 문장은 근거가 없다 | **맞음.** `--counterfactual`은 독립 난수열로 V7을 다시 돌린 것이고, 실제 a_only 수와 2/13만 같았다 | 문장을 고쳤다. 실제 a_only 수를 같은 국면에서 공격 판정하는 분석을 새로 했다(`scripts/analyze_v8_benchmark.py`): **WIN 2(V8과 같은 수), REFUTED 11.** 실제 baseline 기준으로도 11번은 그 자리에서 VCT1 강제승을 두지 않았다 |
| V8-B 효과는 "13판 13승"보다 짝 비교로 써야 한다 | **맞음.** 같은 13판에서 a_only는 10승 1무 2패, 11판은 V8-B 착수에서 정확히 갈라졌고, V8-B가 발동하지 않은 판은 갈라진 판이 0 | §11.12에 기계적 ablation으로 다시 썼다 |
| 비용은 중앙값보다 꼬리 | **맞음.** V8-B 1,564초(51%), V8-A 1,113초(36%) | §11.12 비용 절에 비중과 heavy-tail을 적었다. 판당 시간은 full이 더 짧다는 점(판 길이 95→59수)도 함께 |
| "실전 stage 4가 더 어렵다"보다 "stage 4/5에서 소진이 잦았다" | **맞음.** full stage 4 7회·stage 5 2회, a_only 15회·6회 | 문장 수정 |
| 최악 착수(쌍 1 흑 90수)를 예산 회귀 fixture로 | 채택 | `tests/fixtures/v8_budget_probes_v1.json`(통과 기준 없음, V8-5용 기록) |
| 증명 산출물을 JSON으로 남겨라 | 채택 | `docs/mcts-v8-results/pilot_analysis.json`(국면 해시, 검사 수, 상태, 노드·호출·시간, 예산 소진). pilot 결과 파일 2개도 같은 폴더에 넣었다 |
| V8-C부터 모듈별 비용 기록과 착수 공유 상한 구조 | 채택 | V8-C 진단 6개(`v8_root_*`), runner `root` 기록과 `v8_c_root` 요약, §4.3.6 |
| "±9%p" | **맞음** | "1 SE 약 8~10%p"로 고쳤다 |
| V8-C 시작은 사전 기준에 따른 것 | 확인 | §4.3.1에 명시 |

**V8-C 구현 중 바뀐 것:** 처음 설계(트리 전 root 필터)는 측정 후 버리고 "트리 후 검증"으로 바꿨다(§4.3.2). K 제한의 함정을 고쳤다(§4.3.5).
runner arm은 `full`(A+B+C), `ab`(pilot의 full), `a_only`, `b_only`, `c_only`, `off`가 됐다. pilot과 비교할 때는 `ab`·`a_only`가 같은 설정이다.

### 11.14 V8-C 포함 비용 확인 (데스크톱, `eb88b46`, seed 8401, 5쌍 10판)

입력: `docs/mcts-v8-results/v8c_full_p5.json`(arm `full` = A+B+C), 분석 `docs/mcts-v8-results/v8c_full_p5_analysis.json`.
오프닝이 pilot(§11.12)의 쌍 0~4와 같으므로 pilot의 `full`(= 지금의 `ab`, A+B)·`a_only`와 짝 비교했다.

#### 결과

| 설정 (같은 10판) | 승/무/패 | score |
|---|---|---|
| A+B+C (이번) | 7 / 1 / 2 | 0.75 |
| A+B (pilot full) | 8 / 1 / 1 | 0.85 |
| A만 (pilot a_only) | 7 / 1 / 2 | 0.75 |

- **V8-C가 바꾼 수에서 기보가 갈라진 판은 5판이다.** 결과는 개선 1(쌍 1 백: 패→승), **악화 2(쌍 1 흑, 쌍 3 백: 승→패)**, 같음 2.
  나머지 5판은 A+B와 기보가 같다.
- V8-C 수 변경 9회: V7 수가 **VCT1-UNSAFE로 증명**되어 SAFE 수로 바꾼 것 6회, V7 수가 미판정인데 다른 수가 먼저 SAFE로 증명된 것 3회.
  **악화된 2판은 둘 다 "V7 수 UNSAFE → SAFE 수" 교체였다.** 같은 국면에서 A+B는 그 VCT1-UNSAFE 수를 두고도 이겼다.

#### 패배 2판의 분기점 (per-VCF 100k, 총 2M 노드, V8 마지막 10수)

| 판 | V8-C 교체 | 이후 | 처음 지는 V8 수 |
|---|---|---|---|
| 쌍 3 백 | 13수 (10,9) UNSAFE → (8,8) SAFE (사후 분석도 SAFE) | 15수 stage 2도 VCT1-SAFE | **17수 stage 4: 강제 방어와 root 후보 26개가 전부 VCT1-UNSAFE** |
| 쌍 1 흑 | 18수 (8,5) UNSAFE → (7,5) SAFE | 26·28·30수 모두 SAFE | **32수 stage 4: 후보 21개 중 20개 UNSAFE, 1개 미판정(예산 소진)** |

V8-C가 고른 수 자체는 VCT1-SAFE였다. 그런데 상대(V7)가 두 수 둔 뒤에는 모든 수가 지는 국면이 됐다. **깊이 1 검사로는 보이지 않는 2수 이상의
공격**이다. V8-A·V8-C의 SAFE는 "VCT1으로는 지지 않는다"는 뜻이지 안전을 보장하지 않는다(§2.4에 적은 한계가 실전에서 나왔다).

#### 해석

1. **V7은 VCT를 찌르지 못한다.** V8-C가 피한 VCT1-UNSAFE 수는 객관적으로는 지는 수지만, V7 상대로는 대개 지지 않는다. 그래서 V8 대 V7 점수는
   V8-C의 가치를 과소평가하고, 교체 수(트리 순위 2~5위)가 덜 좋은 수면 오히려 손해로 나타날 수 있다. 이번 2판이 그런 경우인지, 우연인지는
   10판으로 가를 수 없다.
2. **teacher 품질의 기준은 "V7을 이기는가"만이 아니다.** V8은 AlphaZero의 teacher라서 객관적으로 지는 수를 가르치면 안 된다. VCT1-UNSAFE 수를 피하는
   것은 이 목적에 맞다. 다만 그 대가(트리가 더 좋다고 본 수를 버림)가 실제로 나쁜지는 **VCT를 찌를 수 있는 상대**로 재야 한다.
3. **깊이 2 이상의 공격이 다음 약점이다.** 이번 패배 2판과 pilot 패배의 일부가 여기에 해당한다. 온라인 depth 2는 비용상 아직 불가능하다(§2.3).

#### 비용

| 같은 10판 | V8 착수당 평균 | 판당 시간 | V8-B / V8-A / V8-C / 나머지 |
|---|---|---|---|
| A+B+C | 7.41초 | 249초 | 392 / 408 / **1,230** / 192초 |
| A+B | 6.48초 | 248초 | 1,168 / 719 / 0 / 251초 |

- V8-C가 V8 사고 시간의 55%를 쓴다. 대신 판이 갈라져 V8-B·V8-A 시간이 줄어, 판당 시간은 거의 같다.
- V8-C 시간은 중앙값 0.46초, p95 49.3초, 최대 182.5초다. 최악 착수는 186.8초(V8-C)이고, 예산 소진은 157회 중 1회다.

#### 판정

- **50쌍 본 측정은 보류한다.** V7 상대 점수로는 V8-C의 효과를 제대로 잴 수 없다는 근거가 생겼다(해석 1).
- 다음 측정에는 **VCT를 찌를 수 있는 상대**가 필요하다. 후보는 V8-B를 켠 V8(`b_only`: V7 + 자기 VCT1 공격)다. runner에 상대 선택
  옵션(`--opponent v7 | v8:<arm>`)을 넣고, 같은 오프닝으로 A+B+C와 A+B를 그 상대와 짝 비교하는 것을 제안한다.

## 12. 두 트랙 분리와 V8-C 모드 (2026-10-01 결정)

입력: RenjuNet 기보 DB(`renjunet_v10_20260930.rif`)와 "Pure AlphaZero / V8 Hybrid 두 트랙" 제안, 그 검토(§12.6), 사용자 결정.
이 절은 §0·§1의 "V8 = AlphaZero의 teacher"라는 위치를 바꾼다. **V8은 이제 Track B(Hybrid 엔진)의 전술 모듈이다.**
V8을 교사로 쓰는 AZ arm(§5·§6의 V8-D, V8-T4, V8-A1)은 Track B로 옮긴다.

### 12.1 트랙 정의

| | Track A — Self-play-only AZ (AZ-Tactical baseline) | Track B — Strong Hybrid |
|---|---|---|
| 목적 | 자기대국만으로 어디까지 가는지 재는 연구 기준선 | 최강 기력 |
| 허용 | 게임 규칙, 즉승·즉방 필터(`tactical_rules: true`, PUCT v2), random init NN, PUCT, self-play | RenjuNet 사전학습, V8 VCF/VCT 모듈, solver/V8 증명 라벨, Hybrid self-play·fine-tune |
| 금지 | 외부 기보(RenjuNet), V8/V7 teacher 라벨·모방, solver supervision(T1 포함), 다른 트랙 checkpoint로 init | — |
| 계보 | Stage 6 MVP(random init) → 7-A → 7-B arm B → 7-C B → 7-D → Stage 8(B400, S4) | 새로 시작(H 단계) |

- Track A를 **"Pure AlphaZero"라고 부르지 않는다.** `tactical_rules: true`는 사람이 만든 전술 지식(즉승·즉방)이다.
- **엄격한 AZ(`tactical_rules: false`)는 ablation으로만 보존한다.** 새 설정은 만들지 않는다. 이미
  `configs/stage7b_a_scale.yaml`·`configs/stage7c_a_scale_long.yaml`(Stage 7-B/7-C arm A)이 그 설정이고, 계보도 Stage 6 MVP부터 규칙 필터 없이 이어진다.
- **T1(증명 label fine-tune, `stage8-plan.md` §12.10)은 Track B arm이다.** solver supervision이기 때문이다. Track A의 다음 arm은 S4(레시피)다.
- RenjuNet 국면은 Track A에서 **평가 전용**으로만 쓸 수 있다(학습·init 금지).
- 데이터·체크포인트 경로: `data/selfplay/pure_az`(Track A), `data/external/renjunet`, `data/tactical/v8_labels`, `checkpoints/pure_az|hybrid`.
  `data/`·`checkpoints/`는 이미 `.gitignore` 대상이다. **RenjuNet 원본·변환본은 커밋하지 않는다.**
- RenjuNet 라이선스(파일 안 문구): 비상업·OFFLINE 데이터베이스 형태로만 사용, 웹사이트·ONLINE 시스템에서 내용이나 변형물 사용 금지.
  그래서 RenjuNet으로 학습한 모델은 `run_web_play`를 **로컬에서만** 쓰고 인터넷에 공개하지 않는다(보수적 운용, 법률 판단 아님).

### 12.2 상태 용어

V8-A·V8-C의 `SAFE`는 안전 증명이 아니다(§11.14: V8-C가 (8,5) UNSAFE → (7,5) SAFE로 바꾼 판도 이후 깊이 2 이상의 공격으로 졌다).
문서에서는 다음 이름을 쓴다. 코드의 문자열 값은 이전 결과 파일과 비교할 수 있도록 그대로 둔다
(`analysis.mcts_v8`에 별칭 `NOT_REFUTED_VCT1`, `PROVEN_LOSS_VCT1`).

| 문서 이름 | 저장 문자열 | 뜻 |
|---|---|---|
| PROVEN_LOSS_VCT1 | `UNSAFE` | 상대의 VCF 또는 "조용한 수 1개 + VCF" 승리가 증명됨(동결 VCF solver의 범위 안에서) |
| NOT_REFUTED_VCT1 | `SAFE` | 예산 안에서 깊이 1 VCT 탐색을 끝까지 했고 반박을 못 찾음. 깊이 2 이상 공격은 보지 않음 |
| UNKNOWN | `UNKNOWN` | 예산·노드 한도로 끝나지 않음 |
| PROVEN_WIN_VCT1 | `WIN`(V8-B) | 우리 수 뒤 상대의 모든 응수가 우리 VCF로 짐 |

**PROVEN_LOSS는 veto다.** 나중에 NN을 붙여도 NN 점수가 PROVEN_LOSS를 뒤집지 못한다.

### 12.3 V8-C 두 모드

기존 동작을 덮어쓰지 않고 `root_vct_mode`로 나눈다. 기본값은 기존 동작(`aggressive`)이라 §11.12~§11.14 결과와 이어진다.

| 모드 | V7(트리 1위) 수가 NOT_REFUTED | UNKNOWN | PROVEN_LOSS |
|---|---|---|---|
| `aggressive` (기본, arm `full`) | 유지 | 트리 상위 K=4개 중 첫 NOT_REFUTED 자식으로 **교체** | 같은 방식으로 교체, 모두 PROVEN_LOSS면 넓혀서 계속 |
| `veto` (arm `full_veto`) | 유지 | **유지**(다른 수는 검사도 안 함) | aggressive와 똑같이 검사한 뒤, 트리 순서로 처음 나오는 PROVEN_LOSS가 아니고 즉시 지지도 않는 수로 교체. 모두 PROVEN_LOSS면 V7 수 |

- 두 모드 모두 V7 수에 예산 절반을 준다. V7 수가 PROVEN_LOSS면 자식 검사(공정 몫, 상위 4개, 넓힘)도 같고, 고르는 규칙만 다르다(§12.8).
  `5042938`의 첫 veto는 자식마다 남은 예산의 절반을 순서대로 줬다. §12.8에서 바꿨다.
- 진단: `v8_root_switch`가 `proven_loss`(V7 수가 PROVEN_LOSS라 교체) 또는 `unknown`(aggressive에서 UNKNOWN 때문에 교체)을 기록한다.
  runner 요약 `v8_c_root`에 `switched_on_proven_loss`, `switched_on_unknown`, `v7_move_status`, `children_checked`가 추가됐다.
- root 게이트(`run_v8_gates.py --gate root`)는 기본 모드(aggressive)를 잰다. 게이트 probe의 V7 수는 PROVEN_LOSS라 두 모드 모두 교체가 일어나지만,
  veto는 교체 수가 NOT_REFUTED라는 보장이 없다(UNKNOWN 수도 받는다). 게이트의 "교체 수가 독립 검사로 SAFE" 조건은 aggressive용이다.

### 12.4 VCT를 찌를 수 있는 상대 (`--opponent`)

`run_mcts_v8_benchmark.py --opponent v7 | v8:<arm>`. 상대를 바꿔도 오프닝·시드는 그대로다(같은 `--seed`면 arm끼리 짝이 유지된다).
V8 상대는 V7과 같은 난수 스트림을 쓴다(`v8:off`는 V7과 같은 수를 두는 것을 테스트로 확인). 상대가 `v7`이 아니면 게임 key가 `full_veto@v8:b_only/...`처럼
바뀌므로 같은 JSONL에 섞여도 충돌하지 않는다. 요약에는 상대의 경로 분포(`opponent_routes`, 특히 `own_vct` = 상대가 VCT1 공격을 둔 횟수)와
상대 착수 시간(`opponent_move_seconds`)이 들어간다. `analyze_v8_benchmark.py --baseline`은 상대가 다르면 거부한다.

### 12.5 다음 측정: aggressive 대 veto (짧은 짝 비교)

상대 `v8:b_only`(V7 + 자기 VCT1 공격), seed 8401, **5쌍(10판)부터**. 방향이 분명하지 않으면 20~25쌍으로 늘린다. **50쌍은 아직 돌리지 않는다.**

| 비교 | 볼 것 |
|---|---|
| 점수 | 짝 비교(같은 오프닝·색), 부호 검정은 참고만(10판은 1 SE ≈ 16%p) |
| PROVEN_LOSS 회피 | `switched_on_proven_loss` |
| UNKNOWN 교체 | `switched_on_unknown`(aggressive만, veto는 0이어야 함) |
| 비용 | `v8_c_root.seconds`(평균·p95·최대), `children_checked`, `v8_move_seconds` |
| 상대가 실제로 찔렀나 | `opponent_routes.own_vct` |

선택(같은 오프닝): A+B(arm `ab`)도 같은 상대로 돌려 두면 "C가 있어야 하는가"까지 볼 수 있다.

실행(Windows PowerShell, 저장소 루트, `.venv-cpu`):

```powershell
New-Item -ItemType Directory -Force runs\v8_h1 | Out-Null
python scripts/run_mcts_v8_benchmark.py --arm full --opponent v8:b_only --pairs 5 --seed 8401 --workers 4 `
  --games-jsonl runs\v8_h1\full_vs_b.jsonl --output runs\v8_h1\full_vs_b.json 2>&1 | Tee-Object runs\v8_h1\full_vs_b.log
python scripts/run_mcts_v8_benchmark.py --arm full_veto --opponent v8:b_only --pairs 5 --seed 8401 --workers 4 `
  --games-jsonl runs\v8_h1\veto_vs_b.jsonl --output runs\v8_h1\veto_vs_b.json 2>&1 | Tee-Object runs\v8_h1\veto_vs_b.log
# 선택: python scripts/run_mcts_v8_benchmark.py --arm ab --opponent v8:b_only ... (같은 seed)
# 늘릴 때: --pairs 20 으로 같은 jsonl에 이어서 실행(끝난 판은 건너뜀)
```

### 12.6 RenjuNet 데이터 사실 확인 (파서 구현 전, 2026-10-01)

RIF XML, 165,115판(평균 27.2수). 검토용 표본 재생은 우리 `Game` 엔진으로 했다.

| 항목 | 값 | 파이프라인 규칙 |
|---|---|---|
| 규칙 category | 렌주(category 1) 138,808판, **고모쿠(category 2·3) 26,307판** | category 1만 사용(rule id가 아니라 category로 거름) |
| 10수 이상 렌주 대국 | 109,264판 | 최소 10수 |
| 재생(렌주 표본 3,000판) | 98.5% 성공, 흑 금수 착수 1.3%(삼삼 31·사사 7·장목 1), 첫 수가 중앙이 아님 0.2% | 금수가 마지막 기록이면 직전 국면까지 사용, 중간이면 판 전체 제외; 첫 수가 중앙이 아니면 제외 |
| 종료 방식 | 오목 완성 18%, 기권·시간·합의 81% | 승패 결과는 약한 value 신호(작은 λ 또는 미사용) |
| 오목 완성 판의 기록 결과 일치 | 543/550 | 불일치 판은 제외 |
| 개국 | 26개 공식 개국 + swap + 5수째 두 곳 제시 | 1~5수는 국면 복원에만 쓰고 policy loss weight 0 |

분할: 대회 단위 train/val/test → D4 정규형 hash → 세 집합 사이에 같은 정규형 국면이 있으면 val/test에서 제거
(다른 대회의 같은 공식 개국 국면이 평가에 새지 않게).

### 12.7 Track B 로드맵 (H 단계)

| 단계 | 내용 | 판정 |
|---|---|---|
| H0 | §12.1~12.4 (이 커밋) | 문서·코드·테스트 |
| H1 | §12.5 aggressive 대 veto 짧은 짝 비교 → V8-C 기본 모드 결정 | 사용자 실행 |
| H2 | RenjuNet 파서 + 검증 리포트(§12.6 규칙) | 개수·제외 사유 표 |
| H3 | Policy-only 사전학습 | held-out top-1/3/5, legal rate, **must_block 전술 세트가 떨어지면 채택하지 않음** |
| H4 | V8 후보 순서에 policy 사용, V8 대 V8+Policy를 **같은 wall-clock**(CPU 추론 포함)으로 비교 | 짝 비교 |
| H5 | Hybrid 구조: legal/terminal → proven tactical win → PROVEN_LOSS veto → AZ PUCT(policy/value). VCF/VCT는 root·전술 trigger에서만 | V5 rollout 트리에 value를 붙이는 방향은 확장하지 않음 |
| H6 | value: V8 증명 라벨(PROVEN_WIN/LOSS_VCT1) 우선, RenjuNet 결과는 작은 weight, 이후 Hybrid self-play 결과 | 출처별 weight |
| H7 | Hybrid self-play / fine-tune | |
| H8 | 최종 비교(Track A 최신, V7, V8, Hybrid) | 고정 예산·양색 균형 |

**2026-10-09 갱신.** H7은 반복 학습 주기(H7-c1, c2, …), H8은 봉인 holdout에서의 최종 평가로 쓴다.
Track B는 B1-Champion / B1-Matched / B2로 나눈다. 정의와 순서는 [track-b-b1-b2.md](track-b-b1-b2.md).

### 12.8 H1 1차 결과(5쌍, 상대 `v8:b_only`)와 후속 조치 (2026-10-02)

결과 파일: `docs/mcts-v8-results/h1_full_vs_b.json`, `h1_veto_vs_b.json`(커밋 `5042938`, seed 8401).

| | `full` (aggressive) | `full_veto` (순차 배분, `5042938`) |
|---|---|---|
| 승/무/패 | 5 / 1 / 4 (0.55) | 5 / 0 / 5 (0.50) |
| V8-C 교체 | 7 (PROVEN_LOSS 4, UNKNOWN 3) | 8 (PROVEN_LOSS 8) |
| 상대 `own_vct` | 3 | 4 |

**10판 중 8판은 착수열이 완전히 같다.** 갈린 곳은 두 국면뿐이고(`scripts/compare_v8_divergence.py`), 둘 다 오프라인 큰 예산
(VCF당 200k 노드)으로 다시 판정했다.

| 국면 | V7 수 | aggressive | veto | 오프라인 판정 |
|---|---|---|---|---|
| 쌍 1 백 41수 | `[10,8]` UNSAFE | `[11,5]`(14위) 승 | `[6,10]`(5위, UNKNOWN) 승 | `[10,8]` UNSAFE(10초), `[11,5]` SAFE(14초), `[6,10]` **SAFE(1,226초)** |
| 쌍 3 흑 90수 | `[14,12]` UNKNOWN | `[13,5]`(2위) → 225수 무 | `[14,12]` 유지 → 128수 패 | `[14,12]` **SAFE(310초)**, `[13,5]` SAFE(6초) |

해석:

1. **두 국면 모두 양쪽 선택이 NOT_REFUTED_VCT1이다.** 90수에서 veto가 "지는 수를 유지했다"는 근거는 없다. aggressive는 더 안전한 수가 아니라
   **증명하기 쉬운 수**를 골랐다(트리 1위 대신 2위). veto 쪽 패배는 38수 뒤(124수 stage 4 전부 UNSAFE)라 90수 선택과의 인과는 이 데이터로 세울 수 없다.
2. **41수에서 veto가 더 비쌌던 이유는 `5042938`의 veto 예산 배분이다.** V7 수 검사는 같고, 그 뒤 veto는 자식마다 "남은 예산의 절반"을 순서대로 줬다.
   `[6,10]`은 증명이 아주 비싼 수라 여기에 약 10만 노드가 들어갔다가 잘렸다(212k 노드). aggressive의 공정 몫 라운드는 이 수에 작은 몫만 쓰고
   싼 `[11,5]`를 찾았다(156k).
3. **착수 시간 분포(p95, 60초 초과 횟수)는 모드 비교로 쓸 수 없다.** veto의 60초 초과 13회 중 12회가 갈린 뒤의 다른 판에서 나왔다. 두 모드 공통의
   큰 비용은 "V7 수가 UNKNOWN일 때 V7 수 하나에 예산 절반(400k 기준 200k 노드, 데스크톱 65~73초)"이다.
4. 점수 차이(0.05)는 한 판(무 → 패)이다. 5쌍으로는 모드를 가를 수 없고, 판 결과보다 **갈림 국면 단위 비교**가 효율적이다.

조치(이 커밋):

- **veto는 aggressive와 똑같이 자식을 검사한다**(공정 몫, 상위 4개, 모두 UNSAFE면 넓힘). 다른 것은 고르는 규칙뿐이다:
  aggressive = 처음 증명된 SAFE, veto = 트리 순서로 처음 나오는 PROVEN_LOSS가 아닌 수. 그래서 두 모드의 비용 차이는 "V7 수가 UNKNOWN이면 veto는
  거기서 멈춘다"는 것 하나다. `5042938`의 veto 결과(`h1_veto_vs_b.json`)는 **이전 배분 규칙**의 결과이고, 새 veto는 JSONL을 새로 만들어 돌려야 한다.
- **arm `full_r250`**: `root_node_budget` 400k → 250k(V7 수 몫 200k → 125k).
- **fixture `tests/fixtures/v8c_divergence_probes_v1.json`** + `run_v8_gates.py --gate divergence`(`all`에는 포함하지 않음, 2국면 × 2모드 × 2예산).
  41수는 `[10,8]`을 반드시 바꿔야 하고, 90수는 어떤 NOT_REFUTED 수든 정답이며 노드 수를 감시한다.
- **`scripts/compare_v8_divergence.py`**: 두 결과 파일의 첫 갈림 국면, 양쪽 결정 진단, 양쪽 수의 오프라인 VCT1 판정, 같은 판들의 V8 시간 합.

divergence 게이트(이 컨테이너, 1,444 VCF 노드/초; 노드 수가 비교 기준):

| 국면 | 모드 | 400k | 250k |
|---|---|---|---|
| 41수 | aggressive | `[11,5]` SAFE, 165,546 노드 | `[11,5]` SAFE, 201,426 |
| 41수 | veto(새 배분) | `[6,10]` UNKNOWN, **165,546**(이전 212,231) | `[6,10]` UNKNOWN, 201,426 |
| 90수 | aggressive | `[13,5]` SAFE, 257,826 | `[13,5]` SAFE, **164,076** |
| 90수 | veto | `[14,12]` UNKNOWN, 200,000 | `[14,12]` UNKNOWN, **125,000** |

- 250k에서도 aggressive는 두 국면 모두 같은 수를 찾는다. V7 수가 UNKNOWN인 국면의 비용이 줄어든다(90수 258k → 164k).
- 다만 교체 국면의 비용은 줄지 않을 수 있다(41수 166k → 201k: V7 수 몫이 작아지면 공정 몫 라운드 구성이 바뀐다). 250k의 효과는 실전 측정으로 본다.
- 예산을 줄이면 V7 수가 UNKNOWN으로 끝나는 일이 늘고, aggressive는 그때 트리 1위 수를 버린다. `switched_on_unknown`을 함께 본다.

다음 측정(같은 seed 8401, 상대 `v8:b_only`, 5쌍): `full_veto`(새 배분, 새 JSONL), `full_r250`. 각각 `full`과 `compare_v8_divergence.py`로 갈림 국면을 비교한다.

### 12.9 H1 2차 결과: 새 veto와 `full_r250` (2026-10-02, 검산 반영)

결과 파일(데스크톱, `66b194d`): `docs/mcts-v8-results/h1_veto2_vs_b.json`, `h1_r250_vs_b.json`, `h1_div_full_veto2.json`,
`h1_div_full_r250.json`, `h1_div_gate.json`(divergence 게이트 2국면 × 2모드 × 2예산, 250k 포함 8/8 PASS).
`full`은 §12.8의 `h1_full_vs_b.json`을 다시 썼다. root 게이트(3 probe)는 이 컨테이너에서 따로 돌렸다:
`h1_root_gate_250k.json`, `h1_root_gate_400k.json`(`run_v8_gates.py --gate root --root-node-budget N`).
**divergence 게이트와 root 게이트는 다른 probe 집합이다.** 250k가 divergence 게이트를 통과한 것은 root 게이트 결과와 상충하지 않는다.

| arm | 승/무/패 | 착수열이 `full`과 같은 판 | V8-C 노드 합 | V8-C 교체(PROVEN_LOSS / UNKNOWN) |
|---|---|---|---|---|
| `full` (aggressive, 400k, K=4) | 5/1/4 | — | 2,432,757 | 4 / 3 |
| `full_veto` 새 배분 | 5/0/5 | 8/10 (10판 모두 이전 veto와 같은 착수열) | 2,239,298 (이전 veto 2,478,732) | 8 / 0 |
| `full_r250` | 5/1/4 | **10/10** | **2,023,739 (−16.82%)** | 4 / 3 |

#### 시간은 비교에 쓰지 않는다

| run | V8-C 노드/초 | 상대 착수 시간 합 |
|---|---|---|
| `full` | 2,544 | 271초 |
| `full_veto` 새 배분 | 1,762 | 419초 |
| `full_r250` | 1,179 | 616초 |

`full_r250`은 10판 전체 수순이 `full`과 같고, V8-B 노드(765,679)와 V8-A 노드(695,552)도 정확히 같은데 시간은 크게 늘었다.
처리 속도가 run마다 달랐다는 것까지만 확인된다. 원인(동시 실행, 다른 프로세스, 클럭·전력·열 상태 등)은 원자료로 특정할 수 없다.
H1 비교는 노드 수로 하고, 앞으로 wall-clock 비교는 같은 기계 상태에서 하나씩 실행한다.

#### aggressive 대 veto

- 새 veto의 배분 수정은 실전에서 확인됐다. 41수 V8-C 노드는 aggressive와 같은 155,843이다(이전 veto 212,231).
- 두 모드가 갈린 국면은 41수와 90수이고, **고르는 규칙이 두 경우 모두에서 다르다**(이 절의 첫 판 `89c2afe`에 쓴 "차이는 V7 수가 UNKNOWN일 때 바꾸는가 하나"는 틀렸다. §12.8의 "하나"는 비용 차이에 대한 말이다).
  1. **V7 수가 UNKNOWN일 때**(90수): veto는 V7 수를 유지하고, aggressive는 다른 자식이 SAFE로 증명되면 바꾼다.
  2. **V7 수가 UNSAFE일 때**(41수): 둘 다 같은 자식 검사를 하지만, veto는 트리 순서로 처음 나오는 PROVEN_LOSS가 아닌 수(UNKNOWN 포함)를,
     aggressive는 증명된 SAFE 수를 고른다.
  - 비용 차이는 1번에서만 생긴다(veto는 V7 수 검사 뒤 멈춘다).
- 오프라인 재판정(VCF당 200k, 검사당 2M 노드):
  - 41수: veto의 `[6,10]`은 2M 노드로도 UNKNOWN이다(612초). aggressive의 `[11,5]`는 SAFE다.
  - 90수: veto가 유지한 `[14,12]`는 SAFE지만 증명에 436,673 노드가 든다. 온라인에서 V7 수 몫(200k)으로는 증명할 수 없다.
    aggressive는 예산 안에서 `[13,5]` SAFE를 찾았다.
- 승패 표본(10판, 갈린 판 2개)으로 우열은 가릴 수 없다. 그래도 **제한된 예산 안에서 증명된 NOT_REFUTED 수를 우선한다는 V8-C의 목적에는
  aggressive가 더 직접적으로 맞는다.** 다만 aggressive가 바꾼 수는 "더 안전한 수"가 아니라 "증명된 수"다. 90수의 `[14,12]`도 실제로는 NOT_REFUTED였고,
  aggressive는 트리 1위 대신 2위를 뒀다.
- `full`에서 V7 수가 UNKNOWN이었던 root 국면은 142회 중 4회였고, 그중 3회 교체했다. 이 경로의 발생 빈도는 낮다.
- **기본 모드는 aggressive를 유지한다.**

#### `full_r250`

- 10판 모두 `full`과 전체 수순이 같다. V8-C 노드가 다른 국면은 6개다.
  - V7 수가 UNKNOWN이던 4개는 모두 줄었다(400,000→250,000, 284,729→181,604, 232,826→148,451, 350,390→230,638).
  - 41수의 교체 탐색은 늘었다(155,843→201,859).
  - V7 수 SAFE였던 68수도 조금 늘었다(132,998→135,216. 마지막 VCF 호출이 몫을 넘는 만큼).
- `full`의 root 검사 142회 중 V7 수의 SAFE 증명에 125k 노드를 넘게 쓴 경우는 1회(68수)였다. 벤치마크만 보면 250k로 대부분 충분하다.
- 기본값을 내리려면 regression 게이트도 통과해야 한다. **root 게이트(3 probe)는 250k에서 2/3이다.**
  - `human-164723-ply20`이 249,999 노드를 쓰고도 SAFE를 증명하지 못했다. 피해야 할 수는 피했지만 증명된 수가 아니다.
  - 400k는 3/3이다. 같은 probe는 `[5,6]`(트리 3위)을 SAFE로 증명하는 데 285,472 노드를 썼다. 250k로는 원래 닿지 않는 양이다. §4.3.5의 판정이 재현됐다.
  - 결과 파일은 `h1_root_gate_250k.json`, `h1_root_gate_400k.json`이다.
- **그래서 `root_node_budget`은 400k를 유지한다.**

#### K=4

H1은 `root_max_children`를 다른 값과 비교하지 않았다. K=4는 "최적으로 검증됨"이 아니라 "회귀가 발견되지 않았고, 바꿀 근거도 없음"이다.
필요하면 별도 ablation으로 다룬다.

#### H1 판정

- V8-C 기본값은 **aggressive / root budget 400k / K=4**로 고정한다.
- V8-B 상대 5쌍에서 설정 변경이 실제 착수를 바꾼 판은 0~2개였다. 같은 벤치마크의 쌍 수를 늘리는 것은 정보 효율이 낮다.
- 남은 비용 수단은 V7 수 몫(예산의 1/2)이다. 줄이면 UNKNOWN이 늘어 aggressive의 교체가 늘 수 있으므로 H1에서는 바꾸지 않는다.
- root 게이트 결과 파일을 보존한 상태에서 **H1을 종료하고 H2(RenjuNet 파서·검증 리포트)로 넘어간다.**

### 12.10 H2: RenjuNet 파서와 검증 리포트 (2026-10-02)

H2는 모델 학습을 하지 않는다. RenjuNet RIF를 우리 규칙 엔진 기준으로 설명 가능한 Track B 대국 집합으로 만들고, 그 기준을 동결한다.
코드: `src/hybrid/renjunet.py`, `scripts/build_renjunet_dataset.py`, `tests/test_renjunet.py`.
결과: `docs/mcts-v8-results/h2_renjunet_validation.json`(개수만, 착수 없음). 대국 데이터는 `data/external/renjunet/`(커밋하지 않음).

#### 설계안 사실 확인과 수정

| 설계안 | 판정 | 반영 |
|---|---|---|
| category로 렌주를 거름 | 맞음 | `rule` → `<rule category>`. category 2·3(고모쿠) 26,307판 제외 |
| 종료 원인을 five / resign / time / agreement로 보존 | **원자료로 불가** | RIF에는 종료 필드가 없다. `<info>`는 14,069판(8.5%)에만 있고 대부분 개국 절차 메모다(`b=1245`, `y`, `yamaguchi` 등). 판 위에서 알 수 있는 `five` / `forbidden`(마지막 흑 금수) / `unknown`만 남긴다 |
| 오목 완성 판의 결과 교차검증 | 맞음, **확장** | 마지막 흑 금수(흑 패)도 검사한다. 흑 오목인데 `bresult` 0, 백 오목인데 1, 오목인데 무승부, 금수 종료인데 흑승이면 `result_mismatch` |
| 마지막 금수는 직전까지 사용, 중간 금수는 판 전체 제외 | 맞음 | `end=forbidden`으로 잘라 둔다. 그 금수 수는 policy target이 아니다 |
| 1~5수는 국면 복원에만, policy loss 0 | 맞음 | `POLICY_FROM_PLY=5`(ply 0~4 제외). 학습 시 적용한다(H3) |
| 대회 단위 split → D4 정규형 → val/test에서 제거 | 맞음, **구체화** | 국면 단위로 가린다(`masked_plies`). 대국은 남기고 해당 국면만 policy·평가에서 뺀다. train 안의 중복 국면은 사람 착수 분포라서 그대로 둔다 |
| (설계안에 없음) 같은 대국의 중복 기록 | **추가** | 전체 착수열이 D4로 같은 대국은 id가 작은 것만 남긴다(`duplicate_game` 540) |
| (설계안에 없음) Track A 격리 | **추가** | Track B 코드는 `src/hybrid` 패키지. Track A(`search`, `training`, `model`, `agents`)가 import하면 테스트가 실패한다 |
| 산출물 `train.* / val.* / test.*` | **변경** | 대국 단위 `games.jsonl.gz` 하나(대국마다 `split`, `masked_plies`). 텐서 변환은 H3에서 한다. 원자료 라이선스상 이 파일과 RIF는 커밋하지 않는다 |

좌표: 글자 a~o가 열, 숫자 1~15가 아래에서 센 행이다(h8이 중앙). `row = 15 - 숫자`로 읽는다.

#### 개수와 제외 사유 (각 판에 사유 하나, 아래 순서로 처음 걸리는 것)

| 사유 | 판 수 | 설명 |
|---|---:|---|
| `non_renju_rule` | 26,307 | category 2·3(고모쿠) |
| `bad_coordinate` | 0 | a1~o15 밖 좌표 |
| `occupied_point` | 2 | 이미 돌이 있는 자리에 착수 |
| `non_center_first` | 291 | 첫 수가 h8이 아님 |
| `midgame_forbidden` | 166 | 흑 금수 뒤에도 착수가 이어짐(삼삼 125, 사사 40, 장목 1) |
| `moves_after_five` | 14 | 오목으로 끝난 뒤 착수가 더 있음 |
| `too_short` | 29,537 | 금수 절단 후 10수 미만 |
| `result_mismatch` | 247 | 백 오목+흑승 115, 흑 오목+백승 92, 금수 종료+흑승 38, 오목+무승부 2 |
| `duplicate_game` | 540 | D4로 같은 착수열이 더 작은 id에 있음 |
| **`accepted`** | **108,011** | |
| 합계 | 165,115 | 원본 `<game>` 수와 일치 |

- 채택 대국의 끝: `five` 19,806(18.3%), `forbidden` 1,558, `unknown` 86,647. 기록 결과: 흑승 51,997, 백승 47,411, 무 8,603.
- §12.6 사전 조사와 맞는다. category 1은 138,808판(165,115 − 26,307)이다. `too_short` 29,537은 사전 조사의 "10수 미만 29,544"와 7판 차이다. 사전 조사는 자르기 전 길이를 셌고, 여기서는 앞 순서의 사유(첫 수 비중앙 등)가 먼저 걸리고 금수 절단 뒤 길이를 센다.
- 분할(대회 단위, 해시 순서로 test 5%, val 5%, 나머지 train):

| split | 대회 | 대국 | 국면(policy 상태) | 가린 국면 | 그중 ply ≥ 5 |
|---|---:|---:|---:|---:|---:|
| train | 2,195 | 97,098 | 3,663,581 | 0 | 0 |
| val | 137 | 5,442 | 202,390 | 75,057 | 47,864 / 175,180 (27%) |
| test | 105 | 5,471 | 200,919 | 75,864 | 48,524 / 173,564 (28%) |

- 가린 비율(val+test): ply 0~4는 100%, 5~9는 90%, 10~14는 54%, 15~19는 23%, 20~24는 10%, 30 이후는 2% 이하다.
  공식 개국 때문에 초반은 거의 전부 train과 겹친다. 그래서 held-out 지표는 사실상 **ply 10 이후 국면**을 잰다.

#### H2 게이트 (전부 통과)

1. 각 판의 사유가 하나뿐이고, 합계가 원본과 같다(`sum_check`).
2. 출력 대국 108,011판을 `Game`으로 다시 재생해도 실패 0.
3. 가린 뒤 split 사이에 같은 D4 정규형 국면이 0(train∩val, train∩test, val∩test). 마스킹과 별도로 다시 계산한다.
4. 결정성: `--check-determinism`이 같은 파일을 두 번 빌드해 출력 SHA-256을 비교한다.

빌드 시간은 이 컨테이너에서 약 2.5분이다(결정성 검사 포함 시 두 배).
기준값(코드 `9e74e79`): 원본 `renjunet_v10_20260930.rif.gz` SHA-256 `3bbd8d0c…a6ba`, 출력 `games.jsonl.gz` 내용 SHA-256 `228f14b2…7e74`(`deterministic: true`).
같은 원본과 코드로 빌드하면 이 값이 나와야 한다.

**H2 판정: 완료.** H3(policy-only 사전학습)는 `games.jsonl.gz`를 읽어 ply ≥ 5이고 `masked_plies`가 아닌 국면만 policy target으로 쓴다.

### 12.11 H3: policy-only 사전학습 구현 (2026-10-03, 본 학습은 RTX 5070 설치 후)

구현은 지금 하고, 본 학습은 RTX 5070이 오면 한다. 코드: `src/hybrid/h3_cache.py`, `src/hybrid/h3_train.py`,
`scripts/build_h3_cache.py`, `scripts/train_h3_policy.py`, `scripts/eval_h3_gate.py`, `configs/hybrid/h3_policy_64x4.yaml`,
`configs/hybrid/h3_smoke_cpu.yaml`, `tests/test_h3.py`.

#### 설계 검토(2차) 사실 확인

| 주장 | 판정 | 근거 / 반영 |
|---|---|---|
| RTX 5070(sm_120)은 CUDA 12.8 이상 PyTorch 2.7 이상이 필요 | 맞음 | — |
| PyTorch 2.14가 나와 있고 기본 빌드는 CUDA 13.0 | **맞음** | 데스크톱 `.venv-cpu`가 `2.14.0+cpu`(`policy-value-overfit-results.json`). 이 컨테이너에서 PyPI `torch`는 `2.14.1+cu130`이고 `get_arch_list()`에 `sm_120`이 있다 |
| cu132를 권장 | **고정하지 않음** | PyTorch 공지상 CUDA 13.2는 2.12부터 실험 빌드(nightly)로 들어왔다. 2.14 정식 cu130 빌드에 이미 `sm_120`이 있으므로, 설치 시점의 공식 선택기(Windows, pip, CUDA 13.x)를 따르고 `get_arch_list()`에 `sm_120`이 있는지로 판정한다. Windows의 PyPI 기본 `torch`는 CPU 빌드이므로 반드시 CUDA index URL로 설치한다 |
| CUDA 13.x는 드라이버 580 이상 | 맞음(NVIDIA 문서) | 설치 시점 최신 드라이버 |
| "50 step 6.3초"는 6.224 ms를 혼동한 것, 기록은 9.44초 | **둘 다 기록이 있다** | 6.3초는 `stage8-plan.md` 822행(Stage 8 학습, batch 32, 50 step, denormal 수정 후)이다. 9.44초는 Stage 4 과적합 측정(`policy-value-overfit-results.json`, 중간 평가 포함)이다. 결론은 같다: **CPU epoch 시간은 아직 측정되지 않았다.** smoke로 잰다 |
| `legal_actions` plane 때문에 입력 계산이 병목 | 맞음 | `encode_game`은 mask가 없으면 `game.legal_moves()`를 부른다. cache에서 한 번만 계산한다 |
| float 텐서가 아니라 bit-packed cache | 맞음 | 국면당 흑·백·합법 각 29바이트 + 메타데이터 약 95바이트 |
| cache에 provenance, 다르면 로딩 거부 | 맞음 | manifest에 H2 출력 SHA, encoder/action 버전, 규칙 소스(`rules.py`, `game.py`) SHA, policy 시작 ply. 현재 코드와 다르면 `load_cache`가 거부한다 |
| `value_weight=0`이어도 value head BatchNorm 통계가 바뀐다 | **맞음** | `value_head`에 `BatchNorm2d`가 있고 train 모드 forward는 running 통계를 갱신한다. H3는 `policy_head(trunk(x))`만 부른다. 테스트가 학습 후 value head의 가중치와 BN 버퍼가 그대로인지 확인한다 |
| must_block 기준은 "최신 Track A"가 아니라 B400 고정 | 맞음 | B400 raw must_block top-1 0.23, `stage7_probes_v1.json`의 must_block 40개 → **9/40**. B880은 0.15라 기준이 낮아진다 |
| must_block은 체크포인트 선택에 쓰지 않음 | 맞음 | 선택은 val(masked top-1, 같으면 낮은 CE). must_block은 선택된 `best.pt`에 한 번만 `eval_h3_gate.py`로 |
| mask 후 legal rate는 항상 100% | 맞음 | 지표는 masked top-1/3/5와 CE. raw top-1 합법률과 raw 비합법 확률 질량은 진단값(게이트 아님) |
| D4 변환과 금수 mask 조합 검증 | 맞음, **추가** | `verify_d4`: 표본 국면마다 8개 대칭 모두에서 "변환한 cache mask == 변환한 판의 `Game` 합법수". `verify_encoding`: cache planes == 실제 대국을 재생한 `encode_game` |

#### 확정 설계

| 항목 | 내용 |
|---|---|
| 모델 | `PolicyValueNet` 64×4(Track A와 같음), random init. B400에서 시작하지 않는다 |
| 입력 | H2 `games.jsonl.gz` → `build_h3_cache.py` → `h3_cache/{train,val,test}.pt` + `manifest.json`(커밋 금지) |
| 표본 | ply ≥ 5. val/test는 `masked_plies` 제외 |
| 증강 | 학습 시 표본마다 무작위 D4(`augment`, `model.symmetry`와 같은 변환을 테스트로 확인) |
| loss | masked cross-entropy(사람 착수 one-hot). value head는 forward하지 않는다 |
| 최적화 | AdamW, lr 2e-3, weight decay 1e-4, warmup 1,000 step 후 cosine(최저 2%), batch 1,024, 10 epoch(첫 계획값, 미조정) |
| 재개 | `snapshot.pt`(모델, optimizer, scheduler, step, 난수 상태)를 원자적으로 저장. 데이터 순서는 (seed, epoch)로 정해져 이어서 같다. 학습에 영향을 주는 설정이 바뀌면 재개를 거부한다. 중단 후 재개가 끊김 없이 한 번에 돈 것과 같은 가중치를 내는지 테스트로 확인한다 |
| 산출물 | `best.pt`/`last.pt`(기존 weights-only 형식, H4에서 그대로 로드) + `best.json`(`policy_trained: true`, `value_trained: false`, H2·cache SHA, val 지표) |
| 평가 | val/test: masked top-1/3/5, CE, ply 구간별 top-1. 진단: raw top-1 합법률, raw 비합법 질량 |
| 게이트 | `eval_h3_gate.py`: test 지표 + must_block ≥ B400(9/40, `--anchor`로 B400 체크포인트를 주면 같은 코드로 측정한 값). 증명 라벨 모델 0.72는 참고값 |
| 격리 | `hybrid → model, training.probes`만 허용. Track A는 `hybrid`를 import하지 않는다(기존 테스트) |
| H4 주의 | H3 체크포인트의 value 출력은 학습되지 않은 값이므로 쓰지 않는다 |

#### 실행 순서

1. **지금(데스크톱 CPU):** `build_h3_cache.py`(1회) → `train_h3_policy.py --config configs/hybrid/h3_smoke_cpu.yaml`을 batch 32/128/256으로 실행해 samples/s 기록.
2. **RTX 5070 설치 후:**
   - 드라이버 설치
   - 새 venv `.venv-cuda`에 PyTorch CUDA 13.x 빌드 설치
   - `torch.cuda.get_arch_list()`에 `sm_120`이 있고 실제 행렬곱이 되는지 확인
   - GPU smoke(`--device cuda --max-steps 300`)
   - 본 학습(`h3_policy_64x4.yaml`)
   - `eval_h3_gate.py`

#### 이 컨테이너에서 확인한 것 (2026-10-03, 4코어 CPU, PyTorch 2.14.1)

**cache.** 결과 파일은 `docs/mcts-v8-results/h3_cache_report.json`이다(개수와 hash만 있다).

| split | 국면 | 대국 | 표본 검사(encode / D4 8대칭) |
|---|---:|---:|---|
| train | 3,178,091 | 97,098 | 300 / 300 불일치 0 |
| val | 127,316 | 5,354 | 300 / 300 불일치 0 |
| test | 125,040 | 5,374 | 300 / 300 불일치 0 |

- 입력은 H2 출력 `228f14b2…`이다. 국면 수는 §12.10의 policy 상태 수에서 가린 국면을 뺀 값과 같다(val 175,180 − 47,864, test 173,564 − 48,524).
- 크기는 318 MB이고, 4 worker로 329초 걸렸다.

**CPU smoke(100 step, 64×4, D4 증강).**

| batch | samples/s | 100 step 뒤 val top-1(5,000 국면) |
|---:|---:|---:|
| 32 | 237 | 0.038 |
| 128 | 429 | 0.063 |
| 256 | 499 | 0.082 |

- batch 하나의 입력 준비(cache 풀기 + D4)는 batch 256 기준 2.4 ms다. CPU에서는 모델 계산이 시간의 거의 전부다.
- 500 samples/s면 1 epoch(318만 국면)에 약 1.8시간이 걸린다. 10 epoch면 이 컨테이너에서 하루 가까이다. 그래서 본 학습은 GPU에서 한다.
- 데스크톱 CPU 값은 같은 smoke로 따로 잰다.
- 전체 테스트는 torch를 설치한 이 환경에서 539개 모두 통과했다(skip 0).

#### 데스크톱 CPU smoke (i5-12600K, Windows, `.venv-cpu`, `torch_threads: 0` = PyTorch 기본 스레드 수)

| batch | 50 step 구간 samples/s | 100 step 구간 | 전체(평가 포함) | 100 step 뒤 val top-1 |
|---:|---:|---:|---:|---:|
| 32 | 378 | 380 | 297 | 0.038 |
| 128 | 609 | 326 | 277 | 0.063 |
| 256 | 200 | 212 | 201 | 0.080 |

- loss와 val 지표는 이 컨테이너와 거의 같다. 50 step loss는 4.6796 대 4.6793, batch 256 top-1은 0.080 대 0.082다. 같은 데이터와 같은 코드라는 확인이다.
- **속도는 batch를 키울수록 오히려 떨어졌고, 같은 run 안에서도 흔들렸다**(batch 128: 609 → 326). 4코어 컨테이너(batch 256에서 499)보다 느리다.
- 가장 그럴듯한 원인은 기본 스레드 수가 P코어와 E코어를 모두 쓰는 것이다(12600K는 P 6 + E 4코어, 16스레드). 아직 확인한 사실은 아니고 가설이다.
  `--torch-threads` 1/4/6/8로 다시 잰다. 본 학습은 GPU에서 하므로 이 값은 참고용이다.

**스레드 수 측정(batch 256, 100 step, 데스크톱).**

| `--torch-threads` | samples/s(50 / 100 step) | val top-1 |
|---:|---:|---:|
| 1 | 237 / 239 | 0.082 |
| 4 | **611 / 613** | 0.081 |
| 6 | 546 / 542 | 0.081 |
| 8 | 609 / 611 | 0.086 |
| 기본(16) | 200 / 212 | 0.080 |

- **원인은 기본 스레드 수였다.** 스레드를 4 이상 명시하면 기본값보다 약 3배 빠르고, run 안에서도 흔들리지 않는다. 6이 4·8보다 조금 낮은 건 측정 잡음 범위로 본다.
  P코어·E코어 가설과 맞지만, 4~8 사이 차이가 작아서 E코어 탓인지까지는 가르지 못한다.
- 그래서 두 config 모두 `torch_threads: 4`로 바꿨다. GPU 학습에서도 CPU 쪽 입력 준비가 같은 문제를 피한다.
- 데스크톱 CPU로 1 epoch은 약 87분(318만 / 610), 10 epoch은 약 14.5시간이다. GPU가 늦어지면 CPU로 본 학습을 하는 것도 가능하다.
- 스레드 수에 따라 loss가 소수점 셋째 자리에서 다르다(4.4685 대 4.4706). 병렬 합산 순서가 달라 생기는 정상 차이다. 재개는 같은 스레드 수로 해야 bit 단위로 같다.

#### RTX 5070 smoke (2026-10-05, `.venv-cuda`, PyTorch 2.14.1+cu130, batch 1,024, fp32, 300 step)

- 설치 확인: `sm_120`이 arch 목록에 있고 GPU 행렬곱이 정상이다. `.venv-cuda`에서 회귀 테스트 540개가 모두 OK다.
- 로그의 samples/s는 시작부터의 누적 평균이었다(2,616 → 7,916). 50 step마다의 구간 속도로 다시 계산하면 다음과 같다.
  - 처음 50 step은 19.6초로, CUDA 초기화와 cuDNN 준비 시간이 들어 있다.
  - 그 뒤는 50 step마다 3.83~3.87초로 일정했다. 약 **13,300 samples/s**이고, 데스크톱 CPU(4스레드, 610)의 약 22배다.
  - 이후 로그는 구간 속도를 찍도록 바꿨다. 학습 결과에는 영향이 없다.
- 300 step 뒤 val(20,000 국면): top-1 0.110, top-3 0.279, top-5 0.404, CE 3.460. 이때 lr은 아직 warmup 중(6e-4)이었다.
- 예상 본 학습 시간: 1 epoch 약 4분(318만 / 13,300), 10 epoch 약 40분에 평가 시간이 더해진다.
- bf16은 쓰지 않는다. fp32로 충분히 빠르고 비교 기준이 단순하다.
- 64×4 모델에서는 CPU 쪽 입력 준비가 step 시간의 10% 남짓을 차지한다(batch 1,024 기준 약 10 ms / 77 ms). 지금은 최적화할 필요가 없다.

### 12.12 H3 결과(RTX 5070 본 학습)와 H4 설계 (2026-10-05)

결과 파일: `docs/mcts-v8-results/h3_best.json`, `h3_gate.json`, `h3_metrics.jsonl`, `h3_train.jsonl`(지표만, 착수 없음).
10 epoch, 31,030 step. 실제 학습 시간은 약 48분이다.
- 100 step 구간 속도의 중앙값은 13,034 samples/s다.
- step 3,700~16,700 사이에 9,500 안팎으로 느린 구간이 길게 있었다. 학습 설정과 무관하고, 같은 PC의 다른 부하로 보인다(원인 미확인).

#### 결과와 사실 확인

| 지표 | val(127,316) | test(125,040) |
|---|---:|---:|
| top-1 | 52.41% | 52.04%(65,076개) |
| top-3 | 76.27% | 76.38%(95,510개) |
| top-5 | 85.22% | 85.22%(106,554개) |
| CE | 1.592 | 1.598 |

- val top-1은 46.60%(step 3,000)에서 52.41%로 올랐고, 마지막 4,000 step의 증가는 0.06%p다. **수렴했다.**
- 마지막 train loss(D4 증강 포함)는 약 1.48로 val CE 1.59보다 0.11 낮다. 과적합 신호는 작다.
  평탄해진 원인은 데이터 과적합보다 64×4 용량일 가능성이 있지만, 아직 검증하지 않은 가설이다.
- val과 test의 차이는 0.37%p다. 대회 단위 분할과 D4 마스킹을 한 held-out이므로 **일반화가 확인됐다.**
- ply 구간별 test top-1은 49.7~53.1%로 고르다. 단, 5~9 구간은 2,653개뿐이다(초반 국면은 대부분 train과 겹쳐 가려졌다, §12.10).
- `best.pt`는 step 31,030이다. step 30,000(52.397%)과 사실상 같다.

**전술 probe(raw policy, mask 적용, 탐색 없음)**

| 종류 | top-1 | top-3 | 정답 확률 질량 |
|---|---:|---:|---:|
| must_block | **40/40** | 40/40 | 0.970 |
| immediate_win | 38/40(흑 0.90, 백 1.00) | 40/40 | 0.902 |
| vcf | 28/40 | 37/40 | 0.551 |
| avoid(낮을수록 좋음) | — | — | 0.036(균등 분포 0.0075의 4.8배) |

**H3 게이트: PASS**(must_block 40 ≥ B400 9).

해석에서 바로잡을 점:

1. **B400의 9/40과 직접 비교해 "전술이 4배 강하다"고 읽으면 안 된다.**
   Track A는 PUCT v2 즉승·즉방 필터(`tactical_rules: true`)가 막기를 대신한다. 그래서 raw policy가 막기를 배울 필요가 적었다.
   H3는 사람 기보에서 "막는 수"를 직접 모방했다. 게이트는 "H3 policy가 Track A 기준선 아래로 떨어지지 않았다"는 확인이다.
   증명 라벨 모델의 0.72는 다른 held-out 집합에서 잰 값이라 참고값일 뿐이다.
2. **VCF는 V8에서 solver가 맡는다**(V7 M1 own VCF, V8-B). policy의 VCF top-1 70%가 V8의 VCF 실력을 정하지 않는다.
   policy의 쓰임은 solver가 끝내지 못하는 국면의 후보 순서다. "MCTS와 결합하면 VCF가 좋아진다"는 아직 측정하지 않았다.
3. **value 관련 숫자는 해석하지 않는다**(`value_trained: false`). 학습하지 않은 head의 출력이다.
4. **raw 비합법 질량 45%는 학습 방식상 정상이다.** masked CE는 비합법 칸(이미 돌이 있는 칸, 흑 금수)의 logit에 학습 신호를 주지 않는다.
   이 policy는 어디서든 `masked_softmax`(합법수 mask)를 거쳐야 한다. 기존 Track A 평가기와 PUCT도 같은 규칙이다.
5. avoid 4.8배는 게이트가 아니고 관찰 지표다. H4에서 policy가 지는 수를 위로 올리지 않는지는 V8-A/C의 PROVEN_LOSS veto가 막는다.

#### H4: V8 후보 순서에 H3 policy를 쓴다

목표: policy가 V8의 탐색 대상을 바꿀 때 기력이 오르는지 확인한다. **solver의 판정(WIN / PROVEN_LOSS veto)은 policy가 절대 뒤집지 않는다.**

V8에서 순서가 결과를 바꾸는 곳(코드 기준):

| 위치 | 현재 순서 | policy 적용 |
|---|---|---|
| tree root 후보 `root_moves` | V6 root 후보 20개 → V7 VCF 안전 tier → tier 안 흑 자기 금수점 감점 | **H4-a:** tier 구조는 그대로 두고 tier 안에서 policy 확률 순으로 정렬. 순서가 progressive widening(top-k 순위 가중)으로 열리는 자식을 정한다(50 sims에 최대 15개) |
| root 후보 집합 | V6 root 후보 20개 | **H4-b:** policy top-k(예: 8) 중 V6 후보 밖의 수를 추가한다(recall). 추가 수도 V7 VCF 안전 tier 검사를 똑같이 거친다 |
| V8-B 공격 후보 | V3.2.1 priority 순 | H4-c(후순위): policy 순. WIN 증명은 그대로라 결과보다 속도에 영향 |
| V8-A 넓히기, V8-C 자식 순서 | V6 root 순, 방문 수 순 | 바꾸지 않는다(veto 경로는 그대로) |
| 트리 안쪽 노드, rollout | V3.2.1 휴리스틱 | 바꾸지 않는다(노드마다 NN을 부르면 비용이 커진다. H5에서 PUCT로 다룬다) |

설계 원칙:

- **root에서만 NN을 1번 부른다.** CPU 64×4는 국면당 수 ms 수준이다(Stage 4 측정 B1 추론 6.2 ms). V8 착수 시간(중앙값 1초대)에 비하면 작다.
  그래서 "같은 wall-clock 비교"는 같은 시뮬레이션 수로 비교하고, 착수 시간을 함께 기록하는 것으로 충분하다. 시간 차이가 크게 나면 그때 시간 예산 모드를 넣는다.
- V8은 frozen V5/V6/V7 파일을 고치지 않는다. 기존처럼 V8 쪽 복사본(`_search_tree_v8`)과 root 준비 단계에서만 바꾼다.
  policy를 켜면 tree 결과가 V7과 같아야 한다는 성질이 깨진다. 진단의 `v8_v7_move`는 "policy 정렬 후 tree가 고른 수"로 이름을 분명히 한다.
- **격리 규칙 변경이 필요하다.** 지금은 `analysis` 밖의 어떤 코드도 `analysis`를 import할 수 없다(`test_no_source_outside_analysis_imports_it`).
  둘 다 Track B이므로 규칙을 **"Track A(`search`, `training`, `model`, `agents`)는 `analysis`·`hybrid`를 import하지 않는다. `hybrid → analysis`는 허용"**으로 바꾼다.
  policy 로딩(torch)은 `hybrid` 쪽에 두고, V8에는 "후보 → 점수" 함수만 넘긴다. torch가 없어도 V8은 지금처럼 돈다.
- H3 체크포인트는 `best.json`의 `value_trained: false`를 확인하고 policy 출력만 쓴다.

측정 순서:

1. **오프라인 recall(학습·대국 없음, 몇 분):** RenjuNet test 국면에서 사람 착수가 (i) V6 root 후보 20개 안, (ii) policy top-8/20 안, (iii) 둘의 합집합 안에 드는 비율.
   V8 자기 대국 국면에서는 V8이 실제로 둔 수에 대해 같은 비율을 본다. (ii)·(iii)이 (i)보다 크게 높아야 H4-b의 의미가 있다.
2. **짝 비교 벤치마크**(`run_mcts_v8_benchmark.py`, seed 8401, 5쌍부터):
   - 기준선: `full`(H1 확정 설정).
   - 비교 arm: `full_policy`(H4-a), `full_policy_recall`(H4-a + H4-b).
   - 상대: `v7`과 `v8:b_only` 둘 다.
   - 판 결과와 함께 `compare_v8_divergence.py`로 갈림 국면을 오프라인 판정한다. policy 때문에 PROVEN_LOSS 수를 둔 일이 0인지가 안전 확인이다.
3. 1~2에서 이득이 보이면 20~25쌍으로 늘린다. 이득이 없으면 V5 rollout 트리 안에서는 policy의 쓸모가 작다는 결론으로 H5(PUCT + policy + V8 모듈)로 간다.

H3 개선(더 큰 모델, 더 긴 학습)은 H4 결과를 본 뒤 정한다. policy 품질이 아니라 쓰는 방식이 병목일 수 있기 때문이다.

### 12.13 H4 구현 (2026-10-06): 검토 반영

§12.12 설계에 대한 검토의 세 가지 수정을 모두 반영했다.

1. **오프라인 기준을 바꿨다.** "policy top-8이 V6 후보 20개보다 높아야 한다"는 조건은 필요 없다.
   - H4-b의 가치는 **증분 recall**(합집합 후보 recall − V6 후보 recall)이다.
   - H4-a의 가치는 **실제로 열리는 자식 안의 recall**이다. V5 tree는 50 sims에 root 자식 15개, 100 sims에 17개를 연다.
     `root_opening_count`로 계산한 값이 실제 tree와 같다는 것을 확인했다.
   - 각 자식은 상위 `priority_top_k`(8) 창에서 순위 가중 무작위로 뽑힌다. 그래서 "앞 15개"가 아니라, 이 뽑기를 `--draws`번 모사한 **열릴 확률**로 잰다.
   - 사람 착수는 최선수가 아니므로 이 지표는 진단용이다. 기력 판정은 짝 비교가 한다.
   - 매개변수(extra 8)는 미리 고정했다. 조정할 일이 생기면 val로 하고, test는 최종 보고에 한 번만 쓴다.
2. **solver 불변 조건을 테스트로 고정했다**(`tests/test_mcts_v8.py::RootPolicyTest`).
   - stage 1(오목)과 own VCF 국면에서 policy가 다른 수에 확률 1을 줘도 결과와 경로가 그대로이고, policy는 호출조차 되지 않는다.
   - pilot-6 국면에서 policy가 VCT1-UNSAFE 수 `(10,7)`에 확률 1을 주면 그 수가 root 순서 1위가 된다.
     그래도 V8은 V8-C가 UNSAFE로 증명한 수를 두지 않는다.
   - tree가 그 수를 고르도록 강제한 경우에도 V8-C가 `proven_loss`로 교체한다.
   - policy 정렬은 안전 tier를 섞지 않는다(tier 순서 유지, tier 안에서만 확률 순).
   - H4-b 추가 수는 tier 검사 **전에** 후보에 붙어서 같은 VCF 안전 검사를 받는다.
3. **V8은 torch를 쓰지 않는다.**
   - `analysis.mcts_v8`은 `root_policy`(국면 → {수: 확률}을 돌려주는 함수)만 받는다.
   - torch와 H3 체크포인트는 `hybrid.h4_policy.RootPolicy`가 맡는다. 이 어댑터는 trunk와 policy head만 부르고 value head는 쓰지 않는다.
   - `analysis`를 import해도 torch와 `hybrid`가 로드되지 않는다는 것을 테스트로 확인한다.
   - 격리 규칙은 "Track A는 `analysis`·`hybrid`를 import하지 않는다. `hybrid → analysis`는 허용"으로 바꿨다.

그 밖의 반영:

- **fail fast.** `root_policy_order`나 `root_policy_extra`를 켰는데 policy가 없으면 agent와 search 모두 `ValueError`를 낸다.
  러너는 policy arm일 때 게임을 시작하기 전에 체크포인트를 실제로 로드해 본다.
  체크포인트의 메타데이터가 `policy_trained: true`, `value_trained: false`가 아니면 로딩을 거부한다.
- **결정적 tie-break.** 정렬 키는 (safety tier, policy 확률 내림차순, 기존 V7 순서)다.
- **기록.** 착수마다 다음을 남긴다(러너 `policy` 블록과 요약).
  - policy 호출 시간
  - 추가된 수, tier를 통과한 추가 수, 그중 실제로 열린 수
  - 기존 열릴 순위에서 밀려난 V6 수(`displaced`)
  - tree가 고른 수의 policy 순위·확률, root 순서 순위
  - policy를 켜면 `v8_v7_move`는 "policy로 정렬한 root에서 tree가 고른 수(V8-C 전)"다.
- **H4-b의 부작용(문서화).** 추가 수도 V7 VCF 안전 예산(8,000 노드)을 나눠 쓴다. 추가 수는 V6 후보 뒤에 검사하므로 예산이 모자라면 추가 수가 먼저 inconclusive가 된다.
- **arm:** `full_policy`(H4-a), `full_policy_recall`(H4-a+b, extra 8).
- **seed:** 러너의 `--seed 8401 --pairs N`은 쌍마다 오프닝과 시드를 따로 만들고 색을 바꿔 두 판을 둔다. 검토에서 제안한 seed 8401~8405 방식과 같은 설계라서 바꾸지 않는다.
- **H4-c는 미룬다.** 원인을 하나씩 분리하기 위해서다.

판정 단계:

| 단계 | 질문 |
|---|---|
| 오프라인 A | policy 정렬이 열린 자식 안의 recall(opened recall)을 높이나 |
| 오프라인 B | 추가 수가 V6 누락을 얼마나 보완하나(증분 recall) |
| 5쌍 smoke | 불변 조건 위반이나 심한 회귀가 없나 |
| 갈림 국면 | policy가 실제 선택을 어떻게 바꿨나(`compare_v8_divergence.py`) |
| 20~25쌍 | 이득이 재현되나 |

결과 해석 기준: `full_policy`만 좋아지면 병목은 순서, `full_policy_recall`만 좋아지면 후보 집합이다. 둘 다 무효면 H5(PUCT)로 간다.

### 12.14 H4 1차 결과(5쌍)와 확대 측정 설계 (2026-10-07)

결과 파일:
- 대국: `docs/mcts-v8-results/h4_policy_vs_b.json`, `h4_recall_vs_b.json`
- 오프라인 recall: `h4_recall_val.json`
- 갈림 국면: `h4_div_policy.json`, `h4_div_recall.json`

요약은 `scripts/summarize_h4.py`로 만든다. 기준선은 H1의 `h1_full_vs_b.json`(커밋 `5042938`)이다.

**오프라인 recall**(val 2,000 표본 중 tree 경로 1,056 국면, extra 8):

| 방식 | 후보 recall | 상위 8 창 | 열릴 확률 |
|---|---:|---:|---:|
| baseline | 69.4% | 49.9% | 63.3% |
| H4-a 정렬 | 69.4% | 66.1% | 67.9% |
| H4-a+b | **89.8%** | 84.5% | **86.9%** |

- 증분 recall은 +20.4%p다. policy 단독 top-8 recall(86.0%)이 V6 후보 20개 recall(69.4%)보다 높다.
- H4-a는 창 recall을 +16.2%p 올리지만 열릴 확률은 +4.5%p에 그친다. 순위 가중 무작위 열기가 정렬 효과를 희석한다.

**대국**(상대 `v8:b_only`, seed 8401, 5쌍):

| arm | W/D/L | score(95% Wilson) | 흑 | 백 | 기준선 대비 판별 +/−/= | 안전 위반 | 상대 VCT가 있던 패배 |
|---|---|---|---:|---:|---|---:|---:|
| full | 5/1/4 | 0.55 [0.27, 0.80] | 0.5 | 0.6 | — | 0 | 3/4 |
| full_policy | 7/0/3 | 0.70 [0.40, 0.89] | 0.6 | 0.8 | 3/1/6 | 0 | 3/3 |
| full_policy_recall | 8/0/2 | 0.80 [0.49, 0.94] | 1.0 | 0.6 | 4/1/5 | 0 | 2/2 |

사실 확인과 해석:

- **10판 모두 첫 tree 착수(3·4수)에서 기준선과 갈렸다.**
  - 갈린 수는 양쪽 다 오프라인 VCT1 SAFE다.
  - 그 뒤로는 완전히 다른 대국이라, 짝 비교가 분산을 줄여 주지 않는다. "같은 오프닝"만 공유하는 독립 표본에 가깝다.
  - 그래서 판별 +/−(3/1, 4/1)도 잡음 범위다.
- **안전 불변 조건 위반 0.**
  - tree 경로에서 V8-C가 UNSAFE로 판정한 수를 둔 일은 0이다.
  - stage 4/5에서 UNSAFE 수를 둔 경우는 full_policy 4번, full_policy_recall 3번, full 6번 있었다. 모두 검사한 후보 20~26개가 전부 UNSAFE였던 **이미 진 국면**이다. policy와 무관하다(stage 4/5는 policy를 쓰지 않는다).
- **policy arm의 패배 5판은 모두 상대가 VCT1 공격을 둔 판이다.** 패배 원인은 여전히 깊이 1 이상의 VCT 공격이다(§11.14와 같다).
- H4-b는 실제로 탐색을 바꾼다.
  - tree 착수 92회 중 61회에서 추가된 수가 열렸다.
  - 14회는 tree가 추가된 수를 골랐다(V8-C 검증 전 기준).
- **NN 호출은 착수당 3 ms(p95 3.5 ms, CPU 1스레드)다.** 병목이 아니다.
- 시간 꼬리(p95: 기준선 44초, H4-a 38초, H4-b 83초)는 판이 다르기 때문에 비교할 수 없다. 큰 시간은 V8-B/A/C solver에서 나온다.
- 흑 1.0 / 백 0.6(H4-b)은 5판씩이라 의미를 두지 않는다. 확대 측정에서 색별로 본다.

#### 확대 측정(25쌍 = 50판/arm)

1. **기준선 재현과 확대.** `full`을 현재 코드로 25쌍 새로 돌린다(새 JSONL).
   - 앞 5쌍 10판이 H1의 기록과 착수까지 같아야 한다(`summarize_h4.py`의 `same_moves`).
   - 같다면 H4 코드 변경이 policy를 끈 V8을 바꾸지 않았다는 확인이다.
2. **policy arm 두 개를 `--pairs 25`로 이어서 돌린다.** 같은 JSONL을 쓰면 끝난 10판은 건너뛴다.
3. 세 run을 하나씩 차례로 돌린다(동시에 돌리면 시간 값을 못 쓴다).
4. `summarize_h4.py`로 표를 만든다.

미리 정한 판정 기준(50판, 1 SE ≈ 7%p):

| 조건 | 판정 |
|---|---|
| 안전 위반 > 0 | 즉시 중단, 원인 분석 |
| arm − full ≥ +0.10이고, 흑·백 어느 쪽도 full보다 0.10 넘게 낮지 않음 | 채택 후보 |
| 두 arm 모두 채택 후보 | H4-b가 H4-a보다 +0.05 이상이면 H4-b, 아니면 H4-a(더 단순하고 시간 꼬리가 짧다) |
| 0 ≤ 차이 < +0.10 | 효과 미확정. 기력 이득 없음으로 보고 H5로 간다. H5는 policy를 PUCT prior로 직접 쓰므로 H4 채택 여부와 무관하게 진행할 수 있다 |
| 차이 < 0 | 해당 arm 폐기 |

선택: 채택된 arm만 `--opponent v7` 10쌍으로 다른 상대에서도 같은 방향인지 본다.

### 12.15 H4 확대 결과(25쌍)와 판정 (2026-10-08)

결과 파일: `docs/mcts-v8-results/h4_full25_vs_b.json`, `h4_policy25_vs_b.json`, `h4_recall25_vs_b.json`, 요약 `h4_summary25.json`. 세 run 모두 커밋 `9c6e570`, seed 8401, 상대 `v8:b_only`.

| arm | W/D/L | score(95% Wilson) | 흑 | 백 | full 대비 판별 +/− | 안전 위반 | 상대 VCT가 있던 패배 | V8 착수 s median / p95 | 대국 시간 합 |
|---|---|---|---:|---:|---|---:|---:|---|---:|
| full | 41/3/6 | 0.85 [0.73, 0.92] | 0.80 | 0.90 | — | 0 | 4/6 | 1.27 / 79.3 | 296분 |
| full_policy | 32/1/17 | 0.65 [0.51, 0.77] | 0.60 | 0.70 | 6/15 | 0 | 16/17 | 1.33 / 89.9 | 346분 |
| full_policy_recall | 37/3/10 | 0.77 [0.64, 0.87] | 0.78 | 0.76 | 7/10 | 0 | 9/10 | 2.34 / 147.6 | 618분 |

**판정(§12.14 기준): H4-a −0.20, H4-b −0.08 → 둘 다 폐기.** `full`이 기준선으로 남는다.

- **기준선 재현.** `h1_full_vs_b.json`과 겹치는 10판에서 `same_moves` 10/10이다. policy 옵션을 끈 V8은 H4 코드 변경 뒤에도 착수까지 같다. 그래서 H4 코드는 기본값 off로 그대로 둔다(ablation용).
- **opening 쌍 bootstrap(25쌍, 95%).**
  - H4-a − full: −0.20 [−0.37, −0.02]
  - H4-b − full: −0.08 [−0.23, +0.07]
  - H4-b − H4-a: +0.12 [−0.04, +0.29]
- **처음 10판 대 나중 40판.**
  - full 0.55 → 0.925, H4-a 0.70 → 0.64, H4-b 0.80 → 0.76.
  - 5쌍 결과의 순서(H4-b > H4-a > full)는 opening 표본 효과였다.
- **갈림.** 50판 모두 첫 tree 착수에서 기준선과 갈린다(ply 3: 23판, ply 4: 24판, ply 5: 2판, ply 8: 1판). 판별 +/−는 독립 표본에 가깝다.
- **H4-b는 실제로 작동한다.**
  - tree 착수 777회 중 543회에서 추가된 수가 열렸다.
  - 124회는 tree가 추가된 수를 골랐다(V8-C 검증 전).
  - 그중 119회가 실제로 두어졌다.
- **policy가 고른 수는 V8-C에서 덜 안전하다.** V7 tree 선택이 SAFE로 확인되지 않은(UNSAFE 또는 UNKNOWN) 비율:
  - full 43/636 (6.8%), H4-a 48/510 (9.4%), H4-b 79/777 (10.2%).
  - H4-b에서 추가된 수를 고른 경우는 18/124 (14.5%)다.
  - 그 결과 V8-C 교체가 늘었고(full 26회, H4-b 47회), root 검증 시간도 늘었다(합계 129분 → 307분).
  - H4-b의 시간 증가는 대부분 이 검증 비용과 더 긴 대국(평균 51.3수 → 56.5수)에서 나온다. NN 호출은 7 ms다.
- **이미 진 국면에서 둔 증명된 패배수**(검사한 대안이 모두 UNSAFE이거나 즉시 진다): full 9, H4-a 28, H4-b 17. 이는 더 나쁜 국면에 자주 들어갔다는 결과이지 원인이 아니다.

**안전 위반 1건(full, `pair 10 / black / ply 50 / stage4`)은 요약 스크립트의 오탐이었다.**

- 검사한 22수 중 21수가 UNSAFE였다. 유일한 UNKNOWN `(6,11)`은 예산이 소진돼 끝나지 않은 수였다.
- 그 수는 두는 즉시 상대가 5목 또는 막을 수 없는 4를 만든다(`_not_immediately_lost` = False). V8의 fallback은 설계대로 이 수를 건너뛰고 V7의 수를 둔다.
- 엔진 수정은 필요 없다. `summarize_h4.py`의 불변 조건을 엔진과 같게 고쳤다: 즉시 지는 대안은 대안이 아니다. 회귀 테스트는 `tests/test_summarize_h4.py`다.
- 재계산 결과 세 arm 모두 안전 위반 0이다. full의 이미 진 국면은 8 → 9로 바뀐다.

#### H5 설계 조정

H4의 교훈은 두 가지다. 첫째, policy 순서를 V5 rollout 트리의 root에 덧씌우면 tree가 VCT1에서 덜 안전한 수를 더 자주 고른다. 둘째, 5쌍은 판정에 쓸 수 없다. H5는 이를 반영해 다음과 같이 바꾼다.

1. **두 가지를 한 번에 바꾸지 않는다.** H5는 트리 알고리즘(V5 widening → PUCT)과 prior(휴리스틱 → policy)를 함께 바꾼다. 그래서 대조 arm을 둔다.
   - `puct_heur`: PUCT, prior는 V6 후보 점수의 softmax(NN 없음).
   - `puct_policy`: PUCT, prior는 H3 policy.
   - 두 arm 모두 바깥 V8 모듈(Stage 1–5, own VCF, V8-B, V8-A, V8-C aggressive 400k K=4)은 그대로 둔다.
2. **leaf 값.** H3에는 value가 없다(`value_trained: false`). H5의 leaf는 기존 `_rollout_v321` 결과를 쓴다. value net은 H6에서 이 자리를 바꾼다. 이렇게 하면 H5에서 바뀌는 것은 선택 규칙과 prior뿐이다.
3. **구현 위치.** `src/hybrid/`에 둔다. Track A의 `search/alphazero.py`의 PUCT 선택과 backup을 재사용할 수 있는지 먼저 확인한다. Track A 코드는 수정하지 않는다.
4. **평가 프로토콜 강화.**
   - 처음부터 25쌍으로 돌린다.
   - opening 쌍 bootstrap CI와 색별 score를 보고한다.
   - 시간 비율(arm/full 대국 시간 합)을 보고한다. 1.5배를 넘으면 같은 시간 예산의 full과 다시 비교한다.
   - `v8:b_only` 상대에서 full이 0.85(나중 40판 0.925)여서 천장에 가깝다. 그래서 H5 주 비교는 **`--opponent v8:full`**(기준선과 직접 대국, 0.5 = 동등)로 하고, `v8:b_only`는 보조로 둔다.
   - 채택 후보는 다른 seed(8402) 25쌍에서 한 번 더 확인한다.
5. **판정.**
   - `puct_policy` 대 `v8:full` ≥ 0.60, 그리고 `puct_policy − puct_heur` ≥ +0.10이면 policy prior 채택.
   - `puct_heur`만 좋으면 이득은 트리 구조에서 나온 것이다.
   - 둘 다 0.5 미만이면 H6(value)을 먼저 하고 H5를 다시 측정한다.

### 12.16 H5 구현: PUCT 트리와 prior 분리 (2026-10-08, 검토 반영)

§12.15의 H5 조정안에 대한 검토 6가지를 반영했다.

**구조.**
- V8의 tree 경로에서 트리만 바꾼다(`tree_mode="puct"`). Stage 1–5, own VCF, V8-B, V8-A, V8-C(aggressive, 400k, K=4)는 그대로다.
- PUCT 트리는 `src/analysis/puct_v8.py`(torch 없음)에 있다.
  - §12.15에서는 `src/hybrid/`라고 했지만, V8이 이 트리를 직접 부르고 `analysis`는 `hybrid`를 import하지 않는다.
  - 그래서 트리는 `analysis`에 두고, NN prior는 H4와 같은 callable(`hybrid.h4_policy.RootPolicy`)로 주입한다.

**arm.** 세 arm 모두 같은 c_puct를 쓴다.

| arm | 트리 | prior | 보는 것 |
|---|---|---|---|
| `full` | V5 widening | (rank 가중 열기) | 기준선(상대 `v8:full`) |
| `puct_uniform` | PUCT | 1/n | PUCT 선택 규칙 자체 |
| `puct_heur` | PUCT | 1/순위(V8 후보 순서) | heuristic prior |
| `puct_policy` | PUCT | H3 policy(후보로 제한 후 재정규화) | NN prior |

**같은 행동 집합(검토 2).**
- root 자식은 V8이 준비한 root 목록이다: V6 → VCF 안전 tier → M3, H4 옵션 없음.
- 내부 노드 자식은 `_search_candidates_v321`이다. V5 트리가 widening하는 목록과 같다.
- PUCT는 모든 자식을 처음부터 두고, 방문 배분은 prior와 Q가 정한다.
- `tree_mode="puct"`와 H4 옵션(`root_policy_order` / `root_policy_extra`)을 함께 쓰면 오류가 난다.
- 테스트는 세 prior에서 root와 내부 자식 목록이 똑같은지 확인한다.

**값과 선택(검토의 Track A 재사용 확인 4가지).**
1. 값 관점.
   - 노드 값은 그 노드로 들어온 수를 둔 쪽 관점이다(`search.mcts._backpropagate` 재사용). 부모는 자식의 평균값을 최대화한다.
   - Track A의 `search.alphazero`와 같은 규약이다(`player_who_moved` 관점).
   - 부호 테스트:
     - 5목 수의 평균값은 흑 관점 +1이다.
     - 백의 5목을 허용하는 수의 평균값은 −1이다(prior 0.99여도 그렇다).
     - uniform prior는 값만으로 이기는 수와 막는 수를 고른다.
2. 금수 마스크.
   - policy는 `RootPolicy`가 합법 수(흑 금수 제외)에 대해 masked softmax로 계산한다.
   - 그다음 자식 목록으로 제한하고 재정규화한다.
   - 자식에 질량이 없으면 uniform으로 대체하고 그 횟수를 센다(`prior_fallbacks`).
3. 같은 후보 집합: 위와 같다.
4. leaf.
   - `_rollout_v321`로 끝까지 둔 결과(+1/−1/0)다. value net은 H6에서 이 자리를 바꾼다.
   - 선택식은 Track A의 `puct_score`와 같다: `Q + c·P·sqrt(max(1,N))/(1+n)`, FPU 0.
   - 최종 수는 방문 수, 평균값, prior, 좌표 순으로 고른다.

**c_puct 보정(검토 6).** `scripts/h5_calibrate_puct.py`가 맡는다.
- RenjuNet val의 tree 경로 국면 120개에서 PUCT 트리만 돌린다. 대국도 승률도 보지 않는다.
- 후보 c ∈ {0.5, 1.0, 1.5, 2.0}. 미리 정한 통과 조건:

| 조건 | 기준 |
|---|---|
| 붕괴 없음 | 모든 prior에서 방문된 root 자식 ≥ 3, 최다 방문 비율 ≤ 0.75 |
| prior가 혼자 결정하지 않음 | policy prior에서 policy 1순위를 고르는 비율 ≤ 0.80 |
| prior가 무시되지 않음 | 그 비율이 uniform prior보다 +0.05 이상 |

- 통과한 c 중 1.5(Track A 기본값)에 가장 가까운 값을 쓴다. 통과하는 값이 없으면 벤치마크를 하지 않는다.
- 벤치마크는 `--puct-c`가 없으면 PUCT arm을 실행하지 않는다. c는 게임 키에도 들어가서, 다른 c로 이어 돌리면 섞이지 않는다.

**측정(검토 3, 4, 5).**
- 상대는 `v8:full`, 25쌍이다. 0.5가 기준선과 동등이다.
- 연구용 비교는 같은 시뮬레이션 수(50/100)다.
- 실사용 판단은 시간 비율(같은 대국 안에서 V8 초 / 상대 초)로 한다. 1.5를 넘으면 `--arm-simulations 25 --arm-tactical-simulations 50`으로 시간을 맞춘 재측정을 하고, 그 결과로 판정한다.
- 수마다 기록하는 것(`tree` 블록): 트리 모드, 시간, 시뮬레이션 수, NN 호출, prior fallback, root prior 엔트로피, prior 1순위 수와 확률.
- `scripts/summarize_h5.py`가 보고하는 것:
  - score: Wilson 구간과 opening 쌍 bootstrap 구간
  - 흑/백 score
  - 안전 위반
  - 상대 VCT가 있던 패배
  - 이미 진 국면의 패배수 비율
  - tree 수의 SAFE/UNKNOWN/UNSAFE와 not-SAFE 비율
  - V8-C 교체 수, root 검사 시간
  - 시뮬레이션/초
  - prior 엔트로피, prior 1순위 선택 비율
  - 시간 비율
  - arm 간 쌍 차이(bootstrap)

**판정(미리 고정).**

| 결과 | 판정 |
|---|---|
| 안전 위반 > 0 | 중단 |
| 8401: policy ≥ 0.60, policy − heur ≥ +0.10, 흑·백 ≥ 0.50, tree not-SAFE ≤ heur + 0.03, 시간 비율 ≤ 1.5 | **8402 재검증 후보** (채택 아님) |
| 8401+8402(arm당 100판): policy ≥ 0.58, (policy − 0.5)의 쌍 bootstrap 하한 > 0, policy − heur ≥ +0.05, 같은 guard | policy prior 채택 |
| heur > uniform, policy ≤ heur | "PUCT + heuristic prior가 유효". 트리 구조 효과라고 단정하지 않는다 |
| uniform > 0.5, 나머지 이득 없음 | PUCT 선택 규칙 자체의 효과일 가능성 |
| 셋 다 < 0.5 | H6(value) 후 다시 측정 |
| policy가 VCT 패배·tree not-SAFE·V8-C 교체를 악화 | 승률과 무관하게 기각 또는 c 재보정 |

`puct_uniform`은 원인 분석용이다. 8402에서는 `puct_heur`와 `puct_policy`만 돌려도 된다.

#### c_puct 보정 결과 (사용자 실행, val 120 국면, 817초)

| prior | c | 방문 자식 | 최다 방문 비율 | policy 1순위 선택 | 사람 수 일치(진단) | 초/탐색 |
|---|---:|---:|---:|---:|---:|---:|
| uniform | 0.5 / 1.0 / **1.5** / 2.0 | 9.97 / 11.04 / **13.03** / 15.07 | 0.630 / 0.557 / **0.484** / 0.403 | 0.117 / 0.100 / **0.092** / 0.108 | 0.050 / 0.042 / **0.050** / 0.083 | 1.85–2.03 |
| heuristic | 0.5 / 1.0 / **1.5** / 2.0 | 8.93 / 9.85 / **10.99** / 11.12 | 0.675 / 0.605 / **0.550** / 0.513 | 0.083 / 0.108 / **0.108** / 0.158 | 0.092 / 0.108 / **0.100** / 0.133 | 1.83–1.93 |
| policy | 0.5 / 1.0 / **1.5** / 2.0 | 7.53 / 7.58 / **7.62** / 7.68 | 0.715 / 0.661 / **0.620** / 0.577 | 0.283 / 0.392 / **0.475** / 0.525 | 0.167 / 0.275 / **0.258** / 0.308 | 2.42–2.58 |

- 네 값이 모두 통과했다. 규칙대로 1.5에 가장 가까운 **c_puct = 1.5**로 고정한다.
- c = 1.5에서 policy prior의 영향:
  - 트리가 policy 1순위를 고르는 비율은 47.5%다. uniform prior는 9.2%다.
  - 그러니 prior는 분명히 쓰인다. 그래도 나머지 절반은 Q가 뒤집으므로 H4처럼 prior가 혼자 결정하지는 않는다.
- policy prior는 방문을 약 7.6개 자식에 모은다. uniform은 13개, heuristic은 11개다.
- c = 0.5의 policy 최다 방문 비율 0.715는 붕괴 기준 0.75에 가까웠다. 1.5에서는 0.620이다.
- 탐색당 시간: policy가 약 2.5초, uniform과 heuristic이 약 1.9초다(+0.6초, NN 호출).
- 사람 수 일치는 진단용이다(사람 수 ≠ 최선 수). 순서는 policy 26% > heuristic 10% > uniform 5%다.
- 보정은 통과 조건만 확인했다. 어느 c가 더 센지는 보지 않았다.

### 12.17 H5 1차 결과 (seed 8401, 상대 `v8:full`, 25쌍)

결과 파일:
- `docs/mcts-v8-results/h5_uniform_8401.json`, `h5_heur_8401.json`, `h5_policy_8401.json`
- 요약 `h5_summary_8401.json`, 보정 `h5_calibration.json`
- 세 run 모두 커밋 `56cd9c0`, c_puct 1.5다.
- 상대 설정: V5 트리, V8-A/B/C 켬, aggressive.

| arm | W/D/L | score (Wilson / 쌍 bootstrap) | 흑 | 백 | 안전 위반 | 상대 VCT 패배 | tree not-SAFE | V8-C 교체 | 시간 비율 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| puct_uniform | 21/3/26 | 0.45 [0.32, 0.59] / [0.32, 0.58] | 0.42 | 0.48 | 0 | 25/26 | 11.5% | 45 | 1.05 |
| puct_heur | 25/7/18 | 0.57 [0.43, 0.70] / [0.43, 0.71] | 0.50 | 0.64 | 0 | 18/18 | 6.8% | 52 | 1.09 |
| puct_policy | **41/6/3** | **0.88** [0.76, 0.94] / [0.78, 0.96] | 0.92 | 0.84 | 0 | 3/3 | **4.7%** | 30 | **0.65** |

쌍 차이(25 opening 쌍 bootstrap):
- heur − uniform: +0.12 [−0.08, +0.32]
- policy − heur: **+0.31 [+0.14, +0.48]**
- policy − uniform: +0.43 [+0.26, +0.59]

**판정: 8401 후보 조건을 모두 통과했다 → seed 8402 재검증.** 채택은 아직 아니다.

사실 확인:
- 모든 승패는 5목으로 끝났다. 무승부는 판이 다 찬 경우(225수)뿐이다.
- PUCT 트리가 실제로 돌았다.
  - tree 수는 전부 `mode=puct`다.
  - policy arm의 NN 호출은 tree 수당 62.8회다(시뮬레이션 수 + 1에 맞는다).
  - prior fallback은 0이다.
- **이기는 방식.**
  - policy arm의 41승 중 38승은 V8-B(own VCT)가 처음 찾은 강제승이다. 2승은 own VCF다.
  - 첫 강제승은 중앙값 ply 18에서 나왔다. uniform은 22, heur는 21이다.
  - 상대(full)는 policy arm과의 대국에서 own VCT를 3번만 두었다. uniform 상대로는 25번이다. 대신 stage 4(방어 강제)로 299번 몰렸다.
  - 즉 policy prior가 트리를 전술적 주도권을 쥐는 국면으로 이끌고, 마무리는 V8의 증명 모듈이 한다. Track B가 노린 조합이다.
- **tree 수 안전성은 오히려 좋아졌다.** not-SAFE 비율 4.7%다. heur는 6.8%, uniform은 11.5%, H4-b는 10.2%였다. H4의 실패 패턴(policy가 덜 안전한 수를 고름)이 PUCT에서는 나타나지 않았다.
- **시간 비율 0.65.** V8 383분 대 상대 587분이다. 상대가 방어 solver에 시간을 더 쓴다.
- **PUCT 규칙 자체(uniform)는 기준선보다 약하다(0.45).** heur는 기준선과 구분되지 않는다(0.57, 구간이 0.5를 포함). 이득은 트리 구조가 아니라 NN prior에서 나온다.

한계:
- 상대가 `v8:full` 하나뿐이다(같은 엔진 계열).
- opening은 무작위 2수 25개다. H4의 교훈(5쌍 → 25쌍에서 역전)이 있으므로 8402 재현이 필요하다.
- 8402가 통과하면 다른 상대(`v8:b_only`, H4와 같은 조건)에서도 방향을 확인한다. 이것은 판정 기준이 아니라 보조 확인이다.

### 12.18 웹 대국 패배 분석과 VCT2·H6 안전 설계 (2026-10-07)

입력: `run_web_play` 대국 1판(사람 백 승, 100수, 상대는 웹 기본 `v8` = V5 트리 `full`). 기보와 probe 정의는
`docs/mcts-v8-results/probe_web_v8_loss_20261007.json`. 좌표는 이 절에서 1-indexed(row0+1, col0+1)다.
처음 분석안("조용한 장기 포석에 졌다")을 로그·코드·solver 재실행으로 검토하고, 그 검토를 한 번 더 교차 검토한 최종본이다.

#### 사실 (로그와 solver 재실행으로 확인)

| 항목 | 값 |
|---|---|
| V8 착수 경로 | 49수: tree 46, stage4 2(ply 95, 97), stage2 1(ply 99). own VCF 0, V8-B WIN 0 |
| 시뮬레이션 | tree 46수는 50회, 강제수(95/97/99)는 0회 |
| V8-B | 27개 후보 전부 REFUTED, **모두 ply 5–33**. ply 35부터 공격 후보(4·열린 3을 만드는 수) 0개 |
| root 평균값 | ply 41 무렵부터 음수. ply 57, 61, 63, 69, 75–83, 87–93에서 15개 자식 모두 −1.0(ply 85는 최고 −0.67). 자식당 방문 3–4회 |
| ply 89–93 root 후보 | 15개 전부 흑 진영. (11,13), (12,12), (14,14) 등 우하단 대응점은 없음 |
| 승부 수순 | 백 92 (13,13) → 흑 93 (3,4) → 백 94 (12,13) 13열 위협(10·12·13·15) → 흑 95 (11,13) → **백 96 (12,12) 삼삼**(12행 12–14 + 대각 11,11–13,13) → 흑 97 (10,10) → 백 98 (12,11) 열린 4 → 흑 99 (12,10) → 백 100 (12,15) |
| ply 95 V8-A | 강제방어 3개((11,13), (14,13), (9,13)) + 넓힘 20개, 모두 UNSAFE. 19,583 nodes, VCF 443회 |
| ply 97 V8-A | 강제방어 6개 + 넓힘 18개, 모두 UNSAFE |
| 백 94 직후 | 흑 합법수 131개 전부 `ThreatSolver` depth 1에서 UNSAFE(14초) |
| 백 96 직후 | 흑 합법수 129개 전부 depth 0(VCF)에서 UNSAFE |
| 흑 85·87·89·91 직후 | depth 1·2 모두 SAFE(= depth-2 VCT 클래스 안에서 반박 없음. 전체 게임 안전이 아님) |

**결론.** 흑 93 (3,4)는 **PROVEN_LOSS_VCT2**다(witness: 백 94, 그 뒤 모든 응수가 PROVEN_LOSS_VCT1).
V8-C의 `SAFE`(NOT_REFUTED_VCT1)는 정의대로 맞았고, 패배는 그 탐색 깊이에서 정확히 준비수 1개 바깥에 있었다.
"30수 뒤 장기 판세" 문제가 아니라 VCT1 지평선 바로 밖의 짧은 전술이다.

**미확인.** 흑 93에서 버티는 수가 있었는지(P92의 참값). depth-2 전체 검사는 후보 하나에 30분을 넘겨 끝내지 못했다.

#### 원인 정리

| # | 원인 | 근거 |
|---|---|---|
| C1 | 안전 검사가 VCT1까지 | (3,4)가 SAFE 통과 후 VCT2로 짐 |
| C2 | 상대 준비수 감지 없음, 방어는 Stage 4부터 | Stage 4 첫 발동(95)은 이미 진 뒤 |
| C3 | root 후보가 핵심 방어점을 놓침 | 후보 pool은 모든 돌 주변(`search/mcts_v3._neighborhood_pool`)이지만, 점수가 자기 돌을 약간 우대하고(`search/mcts._move_score`: 거리 1에서 자기 +6 / 상대 +5) shortlist가 제한돼 우하단 점이 올라오지 못함 |
| C4 | rollout 값 포화 | 50회를 15개에 나눠 모두 −1이면 비교 정보가 없고, 최종 선택은 방문 수 → 평균값 → **무작위**(`search/mcts.py`) |
| C5 | 공격을 쌓지 못함 | V8-B 후보는 즉시 4·열린 3을 만드는 수뿐 |
| C6 | V8-C는 트리 1위가 SAFE면 다른 수를 보지 않음 | ply 93에서 (3,4) 하나만 검사 |

−1.0은 "그 자식의 rollout 표본이 모두 흑 패배"라는 나쁜 신호이지만, 방문 3–4회라 value로 해석할 수 없다.

#### 설계 원칙 (H5 이후 단계 전체에 적용)

1. **기준선은 움직이지 않는다.** `V8_DEFAULTS`와 `v8:full`의 정의는 바꾸지 않는다. 새 기능은 기본값 off 옵션과 새 arm으로만 넣고,
   off 상태에서 기존 결과와 착수가 같은지 회귀 테스트한다(H4 방식). 채택되면 새 이름의 config로 고정한다.
2. **증명만 라벨이 된다.** UNSAFE/WIN witness가 있는 결과만 value 타깃이다. SAFE, NOT_FOUND, UNKNOWN, 라벨 없는 국면은 value loss를 mask한다(0으로 채우지 않는다).
3. **가지치기는 공격자 쪽만.** selective VCT2는 공격자 준비수만 위협 수(4, 3, 끊긴 3)로 제한하고, 방어자 응수는 항상 전체 합법수(흑 금수 반영)를 검사한다.
   이렇게 하면 찾은 witness는 증명이고, "못 찾음"은 불완전할 뿐이다. 방어자 응수를 줄이면 거짓 PROVEN_LOSS가 생기므로 금지한다.
4. **한 번에 하나만 바꾼다.** VCT2 veto, root 후보 주입, value leaf는 각각 별도 arm으로 측정한다.
5. **결과를 본 뒤 판정 기준을 바꾸지 않는다.** seed를 추가해 기준을 맞추지 않는다.
6. **예산은 노드·호출 수로만.** 시간 cut은 쓰지 않는다(재현성). 후보 순서도 결정적이어야 한다.

#### 상태 이름

| 이름 | 뜻 | value 라벨 |
|---|---|---|
| `PROVEN_LOSS_VCT2` | 상대의 "준비수 ≤ 2 + VCF" 승리가 witness로 증명됨 | 둘 차례 관점 −1 |
| `NO_VCT2_FOUND_WITHIN_HORIZON` | depth-2 전체 검사를 끝냈고 반박 없음. **방어가 있다는 뜻이 아님** | 없음 |
| `NO_TARGETED_VCT2_FOUND` | selective 검사에서 못 찾음 | 없음 |
| `UNKNOWN` | 예산 소진 | 없음 |

"HAS_DEFENSE" 같은 이름은 쓰지 않는다.

#### 라벨 계약 (H6)

- **value는 항상 둘 차례(side to move) 관점.** 행마다 `side_to_move`, `value_target`, `value_perspective="side_to_move"`를 저장한다.
  - 둘 차례의 증명된 승리 +1, 증명된 패배 −1, 실제로 끝난 무승부만 0.
  - self-play 결과는 각 국면의 둘 차례로 변환(흑 승 → 흑 차례 국면 +1, 백 차례 국면 −1).
- 행마다 출처 정보: `game_id`, `ply`, `canonical_hash`(D4), `source`, `solver_class`, `solver_depth`, `solver_budget`, `engine_config_hash`, `model_hash`, `git_commit`.
- **분할과 누출 방지.** probe 국면(D4 정규형)이 하나라도 들어 있는 **게임 전체**를 학습에서 뺀다(앞뒤 수로 결과가 새지 않게).
  그다음 `game_id` 단위로 train/val/test를 나누고, 같은 정규형 국면이 집합 사이에 걸치지 않게 검사한다. 위반이 있으면 학습을 거부한다.
- **H7 provenance.** 한 게임 안에서는 model·config가 고정이어야 한다(위반 시 거부). 서로 다른 게임 사이의 세대 혼합은 replay buffer에서 허용하되,
  게임마다 생성 model·config를 기록하고 buffer가 허용하는 세대 범위를 명시한다. 벤치마크 데이터처럼 프로토콜이 동질성을 요구할 때만 혼합을 금지한다.

#### 탐색 부호 계약 (H6 value leaf)

- `analysis/puct_v8.py`와 `search/mcts._backpropagate`의 노드 값은 **그 노드로 들어온 수를 둔 쪽(player_just_moved)** 관점이다.
  value net이 leaf 국면의 둘 차례 관점 값 `v`를 주면 leaf 노드 값은 `−v`이고, 위로 갈수록 부호가 번갈아 바뀐다.
- `_backpropagate`는 승자(+1/−1/0 결과)를 받는다. H6에서는 실수 값을 받는 backup을 새로 두되 변환은 한 함수에서만 한다.
- 필수 테스트: 즉시 5목 승리, 상대 5목 허용, 1수·2수 강제승, 부모·자식 부호 반전, 실제 무승부(0). H5의 부호 테스트를 value-leaf 버전으로 확장한다.
- PROVEN_LOSS veto는 value보다 위에 둔다(§12.2 유지).

#### Probe

| probe | 국면 | 둘 차례 | 성격 | 기대 |
|---|---|---|---|---|
| P92 | 백 92 직후 | 흑 | 순위 probe, 참값 UNRESOLVED | (3,4)가 우하단 대응수보다 아래, root 후보·visit이 우하단으로. **value 부호는 기준으로 쓰지 않는다** |
| P93 | 흑 93 (3,4) 직후 | 백 | 증명 국면(백 VCT2 승리) | **둘 차례(백) 관점 value > 0**, 흑 관점 < 0 |
| P94 | 백 94 직후 | 흑 | 회귀 fixture | 흑 합법수 131개 전부 PROVEN_LOSS_VCT1, 둘 차례 관점 value < 0 |

P93은 백이 이기는 국면이다. 검토 과정에서 "둘 차례(백) 관점 < 0"이라는 제안이 있었지만 반대다. 백 94가 witness이므로 백(둘 차례)이 +1이다.
P92 참값은 데스크톱에서 depth-2 전체 계산으로 확정을 시도한다. 결과는 위 상태 이름으로만 기록한다.

#### 단계

| 단계 | 내용 | 통과/진행 조건 |
|---|---|---|
| A0 | 8402 실행 환경 확인. `56cd9c0..8dd3a8c`는 문서·결과 파일만 바꿨다(`src` `scripts` `configs` `tests` 변경 없음, 확인함). policy checkpoint SHA-256 기록(8401 결과에는 경로만 있다) | 차이 없음, 같은 checkpoint, `--puct-c 1.5`, 상대 `v8:full`, 25쌍 |
| B | seed 8402: `puct_heur` → 끝난 뒤 `puct_policy`(동시에 돌리지 않는다. 시간 비율 오염 방지). 이 동안 V8·H6 코드를 바꾸지 않는다 | — |
| C | 8401+8402 요약, §12.16 판정표 그대로 | PASS → `puct_policy` 채택, 커밋·checkpoint hash·c·`v8:full`·프로토콜 고정. FAIL → 실패 원인 분석(seed·opening 민감도, prior fallback, not-SAFE, V8-C 교체, 결정적 패착) |
| S1 | probe 등록·측정 + 빈도 조사. 기존 V8 JSONL 전체에서 "tree 수 SAFE → 상대 수 뒤 Stage 4 → 강제방어 전부 UNSAFE" 패턴을 `game_id`, opening, seed, ply, 승패와 함께 센다(같은 opening의 반복은 하나로) | `full`, `puct_heur`, `puct_policy`를 P92·P93·P94에서 측정 |
| 결정 | `puct_policy`가 P92를 해결하고 패턴이 드물면 S2·S3 생략 → H6. 문제가 남고 반복되면 S2 | 미리 고정 |
| S2 | selective VCT2를 `analysis/`의 분석 전용 함수로(착수 경로에서 호출 안 함). 원칙 3, 상태 이름 표 적용 | P93 UNSAFE, witness를 전체 `ThreatSolver` depth 2로 재검증해 일치, 응수 누락 없음, 노드 예산만 사용. S1 패배 국면으로 재현율·비용 측정 |
| S3 | 엔진 arm 두 개를 따로: (a) V8-C 상위 K 자식에 VCT2 veto(PROVEN_LOSS일 때만), (b) 상대 위협 지역 방어점을 root 후보에 주입. 기본값 off | off에서 기존 결과와 같은 착수(10/10) + 고정 fixture(종국, quiet, VCF, VCT, 금수, P92·P93 인접). H5 프로토콜(25쌍, `v8:full`, bootstrap, 시간 비율 ≤ 1.5, 안전 위반 0, 다른 seed 재확인) |
| H6 | 라벨 파이프라인(위 계약) → value 학습 → value leaf 통합 + 부호 테스트 → 벤치마크. S3 채택 시 VCT2 on/off × value on/off | P93·P94 value 부호, 증명 held-out 부호 정확도, RenjuNet 결과 calibration, P92 순위 |
| H7 | Hybrid self-play | provenance 규칙 |

### 12.19 H5 판정(PASS)과 S1·H6 설계 (2026-10-08)

결과 파일:
- `docs/mcts-v8-results/h5_heur_8402.json`, `h5_policy_8402.json`(커밋 `660b78f`, 실행 코드는 `56cd9c0`과 같음)
- 통합 요약 `h5_summary_all.json`(8401+8402, arm당 100판)
- 기준선 고정 `h5_baseline_manifest.json`

#### H5 판정

| arm | W/D/L | score (쌍 bootstrap) | 흑 | 백 | tree not-SAFE | 시간 비율 | 안전 위반 |
|---|---|---|---:|---:|---:|---:|---:|
| puct_heur | 46/15/39 | 0.535 [0.435, 0.63] | 0.48 | 0.59 | 6.48% | 1.159 | 0 |
| **puct_policy** | **78/12/10** | **0.84 [0.77, 0.905]** | 0.81 | 0.87 | 5.75% | 0.757 | 0 |

- policy − heur(50 opening 쌍): **+0.305 [+0.18, +0.425]**. seed별로는 8401 +0.31, 8402 +0.30이다.
- §12.16의 채택 조건(policy ≥ 0.58, (policy − 0.5) 하한 > 0, policy − heur ≥ +0.05)과 guard(흑·백 ≥ 0.50, not-SAFE ≤ heur + 0.03, 시간 비율 ≤ 1.5, 안전 위반 0)를 모두 통과했다. **H5 = `puct_policy` 채택.**
- 기준선 고정(`h5_baseline_manifest.json`): 커밋 `660b78f`, H3 checkpoint SHA-256 `efee4832…0ce61`, c_puct 1.5, V8 config. `v8:full`의 정의는 그대로다.

**결과를 읽을 때 주의할 점.**
- **시간.** 통합 대국 시간 합은 policy 1,649.6분, heur 1,345.5분이다. 하지만 늘어난 시간은 상대(`v8:full`)가 쓴 것이다. policy에 밀린 상대가 Stage 4 방어 solver를 많이 돌렸다(8401 policy 판에서 상대 586분, heur 판에서 312분).
  V8 자신의 시간은 policy 710.8분, heur 722.4분으로 비슷하다. 시간 판단은 같은 대국 안의 시간 비율로만 한다. 실행이 다르면 부하 차이가 있어서 wall-clock(tree 중앙값 등)을 비교하지 않는다.
- **넓히기.** heur에서 root 검사가 최대 20개까지 간 것은 설계대로다. 상위 4개가 모두 UNSAFE면 나머지 자식으로 넓힌다(`_search_root_children`).
- **policy와 UNKNOWN.** policy는 증명된 위험 후보(V7 수 UNSAFE)를 덜 낸다(8402: 0.81% 대 2.15%). 대신 UNKNOWN은 조금 더 많다(5.85% 대 4.05%). 실제로 둔 수가 UNSAFE인데 검사된 SAFE 대안이 있었던 경우는 두 arm 모두 0이다.
- **checkpoint 동일성.** 8401 결과에는 checkpoint 경로만 있다. 그래서 8401과 8402가 같은 바이트를 썼다는 것은 끝까지 증명할 수 없다. 앞으로는 결과 JSON의 `provenance`(아래)가 기록한다.

#### 기록 인프라 (이 커밋)

- `run_mcts_v8_benchmark.py` 결과에 `provenance`를 추가했다: `git_dirty`, `policy_checkpoint_sha256`, `policy_metadata_sha256`. 착수에는 영향이 없다.
- `scripts/run_s1_probes.py`, `scripts/s1_loss_analysis.py`를 추가했다(분석 전용). 테스트는 `tests/test_s1_tools.py`.
- **H6-0에서 할 일:** 지금 runner는 PUCT arm을 상대로 쓰는 것을 거부한다("its c_puct would be unset"). `v8:puct_policy`를 상대로 쓰려면 상대용 c_puct 인자(`--opponent-puct-c`)와 `--value-checkpoint`(후보에만 적용)가 필요하다.

#### S1-1 VCT2-like horizon signature

정의: V8-C가 SAFE로 본 tree 수 바로 다음 V8 수가 Stage 4이고, 검사한 방어가 모두 UNSAFE인 경우.
**바로 다음 수에 드러나는 경우만 잡으므로 하한이다. "VCT2 발생률"이라고 부르지 않는다.** 한 opening은 두 색으로 두 판이므로, opening 단위는 한 번만 센다(seed가 다르면 다른 opening).

| arm | 게임 | opening |
|---|---|---|
| puct_uniform (8401) | 5/50 (10%) | 5/25 (20%) |
| puct_heur (8401+8402) | 13/100 (13%) | 13/50 (26%) |
| **puct_policy (8401+8402)** | **1/100 (1%)** | **1/50 (2%)** |

#### S1-2 Probe (P92/P93/P94)

- **대상:** `full`, `puct_heur`, `puct_policy`를 seed 10개씩(`--base-seed 9201`). 결과는 `run_s1_probes.py`가 기록한다.
- **영역 R:** 9–15행 × 9–15열(1-indexed). 백 92·94의 위협이 모두 이 안에 있다.
- **P92 지표:**
  - (3,4)를 최종 선택한 횟수
  - R의 수가 root 후보에 들어간 비율
  - R 최상위 수가 (3,4)보다 순위(방문 수 → 평균값 순)와 방문 수에서 앞서는지
  - 두 수의 평균 visit share와 raw policy 확률
- **P93·P94:** 정상동작 확인용으로 경로와 V8-A/B/C 상태만 기록한다. value 부호(P93 백 차례 > 0, P94 흑 차례 < 0)는 H6에서 확인한다.
- **C4 기준값:** root 자식 평균값의 range, std, `saturated`(range ≤ 0.05)를 기록한다.

#### S1-3 패배 원인 분류

방법:
- 패배한 판의 V8 수를 끝에서부터 거꾸로 본다.
- 각 수 직후 국면을 node 예산이 있는 solver로 depth 0 → 1 → 2 순서로 검사하고, 처음 증명되는 깊이를 기록한다.
- depth 2 안에서 증명되지 않는 수를 만나면 멈춘다. 증명된 패배가 이어진 구간에서 가장 이른 수가 결정적 수다.
- 예산은 node·호출 수로만 정한다(재현 가능).

| primary (판마다 하나) | 뜻 |
|---|---|
| `VCT2_HORIZON` | 결정적 수가 depth 2에서만 증명된 패배 |
| `VCT1_LOSS` | 결정적 수가 depth ≤ 1에서 증명된 패배 |
| `DEEPER_OR_POSITIONAL` | 결정적 수가 강제된 수였다(5목 차단 Stage 2, 또는 검사한 대안이 모두 UNSAFE). 그 전에 depth 2보다 깊게 이미 졌다 |
| `UNRESOLVED` | 마지막 V8 수조차 예산 안에서 증명되지 않음 |

flags(여러 개 가능): `ENGINE_SAID_SAFE`, `ENGINE_UNKNOWN`, `BOUNDARY_UNKNOWN`(패배 구간 직전 수가 UNKNOWN이라 결정적 수가 더 이를 수 있음), `ROOT_UNKNOWN`, `ROOT_BUDGET_EXHAUSTED`, `STAGE_BUDGET_EXHAUSTED`, `OPP_OWN_VCT`, `V8C_SWITCHED`, `LONG_GAME`(≥ 150수), `SIGNATURE`.

#### S1 결정 (미리 고정)

아래 네 조건을 **모두** 만족하면 S2·S3를 건너뛰고 H6로 간다.
1. P92에서 `puct_policy`가 (3,4)를 고른 seed ≤ 2/10
2. P92에서 R 최상위 수가 (3,4)보다 순위와 방문 모두 앞선 seed ≥ 8/10
3. signature opening rate ≤ 2/50(현재 1/50)
4. policy 패배 중 primary `VCT2_HORIZON` ≤ 1판, 그리고 같은 opening에서 반복되지 않음

하나라도 실패하면 S2(§12.18)로 간다. S3는 S2가 놓친 패배를 잡는 재현율과 감당할 만한 비용을 보여줄 때만 진행한다.

#### H6 설계 (S1 통과 후)

| 단계 | 내용 | 통과 조건 |
|---|---|---|
| H6-0 | §12.18 라벨 계약 구현(행 스키마, probe가 든 게임 전체 제외, `game_id` 분할, D4 교차 검사). runner에 `--value-checkpoint`(후보에만)와 `--opponent-puct-c` 추가. 평가 seed는 **8403, 8404, 예비 8405**. 8401·8402 대국은 학습에 쓰지 않는다 | 스키마·누출 테스트. 옵션을 끄면 기존과 같은 착수 |
| H6-1 | RenjuNet **train split**의 증명 라벨(VCF, 옵션으로 VCT1. 둘 차례 관점 ±1) + RenjuNet 승패(작은 weight). 라벨이 없는 국면은 mask | 출처·깊이별 개수와 ±1 비율 보고 |
| H6-2a | **trunk를 고정하고 value head만 학습.** policy 출력은 H3와 비트 단위로 같아야 한다 | 샘플 국면에서 logits가 같다는 테스트 |
| H6-2b | (조건부) 2a가 학습 부족일 때만: trunk 미세조정(낮은 LR + H3 policy 증류). **별도 arm(H6-joint)이고, isolation arm이 채택된 뒤에만 평가** | H3 gate(top-1, must_block 40/40) 유지 |
| H6-3 | 오프라인 gate: 증명 held-out 부호 정확도, RenjuNet 결과 calibration(ECE), D4 일관성, **P93 > 0(백 차례), P94 < 0(흑 차례)**. P92는 순위만 본다 | 미리 고정. 실패하면 대국 평가 없이 H6-1/2로 돌아간다 |
| λ 고정 | gate 통과이고 ECE ≤ 0.10이면 λ = 1.0(value만), ECE > 0.10이면 λ = 0.5(value와 rollout 반씩). **대국 후보는 하나만** | 대국 전에 결정 |
| H6-4 | arm `puct_policy_value`: policy는 H3 그대로, leaf만 value. 부호는 §12.18 계약(leaf 노드 값 = −v)을 한 함수에서 처리. PROVEN_LOSS veto 유지. 수마다 C4 지표(range, std, saturated) 기록 | 부호 테스트(즉시 5목, 상대 5목 허용, 1·2수 강제승, 부모·자식 부호, 무승부) |
| H6-5 | 주 상대는 고정된 `v8:puct_policy`(0.5 = 동등), 보조 상대는 `v8:full`. 각 25쌍 | 아래 표 |

H6-5 판정:

| 단계 | 조건 | 판정 |
|---|---|---|
| 8403 | guard 실패 또는 score < 0.50 | REJECT |
| 8403 | score ≥ 0.50 | 8404 진행 |
| 8403+8404 | score ≥ 0.55, 쌍 bootstrap 하한 > 0.5 | **ADOPT** |
| 8403+8404 | score ≥ 0.55, 하한 ≤ 0.5 | **INCONCLUSIVE** → 8405를 한 번만 추가하고 3 seed에 같은 기준을 적용. 그래도 안 되면 REJECT |
| 8403+8404 | 0.50 ≤ score < 0.55 | REJECT(이득이 작음) |

- guard: 안전 위반 0, tree not-SAFE ≤ 기준선 + 0.03, 시간 비율 ≤ 1.5, 흑·백 각 ≥ 0.45, P93·P94 부호 유지.
- 보고에 넣을 것: saturated root 비율이 rollout 기준선보다 줄었는지, P92 순위 변화.

순서: H5-F(이 커밋) → S1 실행 → 결정 → (S2 → 조건부 S3) → H6-0 → H6-1 → H6-2a → H6-3(λ 고정) → H6-4 → H6-5 → (ADOPT 뒤에만 H6-2b) → H7.

### 12.20 S1 probe 결과(P92/P93/P94)와 S2 목표 (2026-10-08)

결과 파일:
- `docs/mcts-v8-results/s1_probes_full.json`, `s1_probes_puct_heur.json`, `s1_probes_puct_policy.json`
  - 커밋 `b648489`, `git_dirty: false`, seed 10개(`--base-seed 9201`)
  - 형식 v1이라 포화 지표가 미방문 자식까지 포함한다. 아래 표의 visited 기준 값은 `root_children`에서 다시 계산했다.
- `s1_p93_verification.json`(`scripts/s1_verify_p93.py`)
- policy metadata SHA-256 `5648b7fb…2609`를 기준선 manifest에 채웠다.

#### 결과

| probe | full (V5) | puct_heur | puct_policy |
|---|---|---|---|
| P92 (3,4) 선택 | 2/10 | 0/10 | 0/10 |
| P92 실제 수 | 7가지로 흩어짐 | (9,7) 10/10 | (4,4) 10/10 |
| P92 R 후보가 root에 있음 | 6/10 | 10/10 | 10/10 |
| P92 R 후보와 (3,4)가 둘 다 있음 | 4/10 | 10/10 | 10/10 |
| P92 둘 다 있을 때 R이 순위·방문 모두 앞섬 | 0/4 | 0/10 | 0/10 |
| P92 §12.19 gate 집계(R은 있고 (3,4)는 없는 경우도 "앞섬"으로 셈) | 2/10 | 0/10 | 0/10 |
| P92 방문 자식 / 자식, 방문 Q | 15/15, 전부 −1 | 20/20, 전부 −1 | 20/20, 전부 −1 |
| P93 둔 수 | 7가지로 흩어짐 | (12,13) 10/10 | (13,11) 10/10 |
| **P93 강제승 유지**(solver 검증) | **3/10** | **10/10** | **10/10** |
| P93 방문 자식 / 자식, 방문 Q | 15/15, 전부 +1 | 2/22, 전부 +1 | 4/22, 전부 +1 |
| P94 | Stage 4 (11,13), 방어 23개 전부 UNSAFE, 10/10 | 같음 | 같음 |

- **P93 검증.** 백 (12,12), (12,13), (13,11) 뒤에는 흑의 131개 응수가 depth 1에서 모두 UNSAFE다. 즉 백이 강제승을 유지한다.
  (4,3), (4,4), (8,3), (8,10), (9,7) 뒤에는 흑에게 SAFE 응수가 있다. 이 탐색 클래스 안에서 반박되며, 더 깊은 승리가 있는지는 모른다.
- full의 §12.19 gate 집계 2/10은 R이 (3,4)를 실제로 앞선 경우가 아니다. (3,4)가 root 후보에 아예 없었던 경우다. 앞으로의 probe(형식 v2)는 `both_present`, `far_absent_region_present`, `region_strictly_beats_far`를 따로 기록한다.

#### S1 판정

조건 1은 통과했다(puct_policy 0/10). **조건 2는 실패했다(0/10). §12.19 규칙에 따라 S2로 진행한다.** 조건 4(패배 분류)의 결과와 관계없이 이 판정은 그대로다.

#### 해석 (확인된 것과 아직 모르는 것)

1. **C3(후보 recall)는 확인된 첫 번째 병목이다.** 세 arm의 P92 실행 30번에서 R 영역의 root 후보는 (10,9) 하나뿐이었다.
   이후 진행에서 방어와 관련된 것으로 보이는 점들(continuation-derived defence candidates: (11,13), (12,12), (14,14) 등)은 한 번도 후보에 오르지 않았다. H5는 행동 집합을 고정했으므로(§12.16) 이 결과는 예상된 것이다.
2. **policy의 방어점 평가 품질은 아직 확인되지 않았다.** 후보 안에서 raw policy 확률은 (10,9) 0.00174 < (3,4) 0.00403이었다. 후보 밖의 점들에 policy가 확률을 얼마나 주는지는 아직 측정하지 않았다.
3. **C4(값 포화)도 함께 있다.** P92에서 방문된 root 자식의 Q는 세 arm 모두 전부 −1이다. 방어 후보를 root에 넣어도 rollout이 −1을 돌려주면 결국 다시 prior가 결정한다. S3-B와 H6 value는 서로 대신하는 해결책이 아니라, 서로 다른 병목을 고치는 단계다.
4. **P93에서 PUCT가 C4를 해결한 것은 아니다.** v1 집계에서 PUCT arm의 포화가 0으로 나온 것은 미방문 자식의 Q 초깃값 0 때문이었다. 방문된 자식은 세 arm 모두 전부 +1이다.
   PUCT의 개선(강제승 유지 10/10, full은 3/10)은 prior가 방문을 소수 후보에 모은 데서 왔다. full은 값이 후보를 구별하지 못해 방문 분포와 tie-break(무작위)에 크게 의존했다.
5. **"실제 방어점"이라고 아직 단정할 수 없다.** P92의 참값은 UNRESOLVED다. P92가 이미 강제 패배 국면이라면 조건 2는 "정답 수를 얼마나 찾는가"를 잰 것이 아니게 된다.

#### 지표 수정 (§12.19의 C4 정의를 대체)

- **C4 포화는 방문된 자식(visits > 0)만으로 판정한다.**
  - 기록하는 값: `range_visited`, `std_visited`, `visited / children`, `saturated_visited`(range_visited ≤ 0.05). `range_all`은 참고용이다.
  - H6-4/H6-5에서 rollout 기준선과 비교할 때도 이 정의를 쓴다. `run_s1_probes.py`(형식 v2)가 이렇게 기록한다.

#### S2 목표 (분석 전용, 엔진 변경 없음)

1. **P92의 참값.** 흑의 모든 합법수를 `ThreatSolver` depth 2로 분류한다(node 예산 고정, 시간 cut 없음).
   결과는 셋 중 하나로 기록한다: `PROVEN_LOSS`(모든 수 UNSAFE), 패배를 피하는 수의 집합(depth-2 클래스 기준), UNKNOWN.
2. **saving-defence 집합의 후보 recall.** 1에서 찾은 수 가운데 현재 V8 root 후보(V6 → safety tier)에 들어가는 비율.
3. **raw policy 진단(필수).** P92에서 H3 policy가 saving-defence에 주는 확률과 순위(전체 합법수 기준).
4. **S3-B 방식은 1–3의 결과로 정한다(미리 고정).**

   | S2 결과 | S3-B |
   |---|---|
   | P92가 `PROVEN_LOSS` | 이 probe를 근거로 S3-B를 만들지 않는다. 다른 근거가 필요하다(S1-3 패배 분류, 추가 사례) |
   | saving-defence가 있고, policy 상위 K(K ≤ 8) 안에 있음 | policy 상위 K를 root 후보에 주입 |
   | saving-defence가 있지만 policy가 낮게 봄 | 위협 영역 또는 solver 기반 후보 주입 |

5. S1-3 패배 분류(조건 4)의 결과도 S2 보고에 합친다.

### 12.21 S1 패배 분류 결과와 S2 진행 (2026-10-08)

결과 파일:
- `docs/mcts-v8-results/s1_losses_policy_v1.json`: 데스크톱 원본 출력(커밋 `b648489`, 예산 node_limit 20,000 / call_limit 50,000 / node_budget 3,000,000)
- `s1_losses_policy.json`: v1의 walk를 그대로 쓰고 `--reuse-classification`으로 다시 집계한 v2. `causal_status`, `boundary_status`, 입력 파일 SHA-256이 추가됐다
- `s2_witness_positions.json`: VCT2 다섯 판의 결정적 수 직전 국면(S2 시험 세트)

재현성: 확정 2판(8401 pair 4, 16)을 클라우드에서 다시 계산했고 walk가 데스크톱 결과와 완전히 같았다.

#### 결과 (puct_policy, 패배 10판)

| primary | 판 수 | causal_status |
|---|---:|---|
| `VCT2_HORIZON` | 5 | **CONFIRMED 2**(8401 p4, p16), TENTATIVE 3(8401 p17, 8402 p7 흑, 8402 p17) |
| `VCT1_LOSS` | 3 | 3판 모두 직전 수가 UNKNOWN, 3판 모두 stage 예산 소진 |
| `DEEPER_OR_POSITIONAL` | 2 | 강제 방어에서 검사한 수가 모두 UNSAFE(8402 p11: 23/23, p14: 21/21) |

- **CONFIRMED**: 패배 구간 직전의 V8 수가 SAFE다. 그래서 결정적 수가 처음 패배로 넘어간 수라는 것이 확인된다.
- **TENTATIVE**: 직전 수가 UNKNOWN(예산 소진)이다. 실제 첫 패착은 더 앞에 있을 수 있다.

**S1 판정: 조건 4 실패.** VCT2_HORIZON ≤ 1이 기준인데, 확정된 2판만으로도 기준을 넘는다. 5판 모두 opening이 다르다(반복 없음). S1의 조건 2와 4가 실패했으므로 **S2 진행이 확정**됐다.

#### 해석

1. **VCT2 지평선 불일치가 가장 큰 국소 실패 유형이다.**
   - 10판 중 5판에서 엔진이 SAFE로 판정한 수(V8-C 4판, V8-A 1판)가 depth 2에서 UNSAFE였다.
   - 최초 패배 전이가 확정된 것은 2판이고, 나머지 3판은 원인 귀속이 잠정적이다.
2. **빈도는 두 층으로 적는다.**
   - 국소 VCT2 witness: 게임 5/100, opening 5/50
   - 원인이 확정된 하한: 게임 2/100, opening 2/50
   - signature(§12.19)는 이 유형 중 1건만 잡았다. 국소 witness 기준 recall 1/5, 확정 기준 1/2다. 바로 다음 수가 Stage 4인 좁은 패턴이라 일반적인 VCT2 지표가 아니고, S1 판정 근거로는 패배 분류를 우선한다.
3. **예산 소진은 가장 흔하게 함께 나타나는 운영 요인이다.**
   - `STAGE_BUDGET_EXHAUSTED`가 10판 중 7판에 있지만, 7판 모두에서 원인이라는 증거는 없다(VCT2·DEEPER 판에도 섞여 있다).
   - VCT1_LOSS 3판에서는 직접적인 실패 메커니즘으로 의심된다.
   - 대표 사례는 **8402 p3(흑, ply 24, stage5)**다. V8-A가 첫 후보 하나를 검사하다 예산이 바닥났고(검사 1개, UNKNOWN), 증명되지 않은 수를 뒀는데 그 수는 depth 1에서 지는 수였다.
   - 8402 p7 백(UNSAFE 19 + UNKNOWN 2)과 p22 흑(UNSAFE 19 + UNKNOWN 1)은 거의 진 국면에서 예산 때문에 증명을 끝내지 못한 경우다.
4. **초반 백의 패턴.**
   - 8401의 VCT2 3판은 모두 V8이 백이고, 결정적 수가 ply 11(V8의 다섯 번째 수)이다.
   - VCT2 5판 모두 `ENGINE_SAID_SAFE`와 `OPP_OWN_VCT`가 함께 붙어 있다.
   - 상대의 V8-B가 원인이라고 단정하지 않는다. "depth 1에서 SAFE로 통과한 국면이 이후 상대의 강제 공격으로 이어지는 패턴이 반복됐다"까지만 말할 수 있다.

#### S2 (분석 전용, 엔진 변경 없음)

| 항목 | 내용 | 스크립트 |
|---|---|---|
| S2-1 P92 참값 | 흑의 모든 합법수를 depth 2까지 분류(수마다 node·call 예산, 시간 cut 없음, 끝난 수는 바로 저장해서 이어서 실행 가능). 판정: `PROVEN_LOSS` / `SAVING_MOVES_FOUND`(depth-2 클래스) / `UNRESOLVED`. saving 수가 각 arm의 root 후보에 들어 있던 비율도 낸다 | `s2_position_truth.py` |
| S2-2 raw policy | P92의 전체 합법수 분포(나중에 S2-1과 결합), witness 다섯 국면에서 V8이 실제로 둔 결정적 수의 policy 순위와 확률, top 10 | `s2_policy_diag.py`(torch 필요) |
| S2-3 selective VCT2 탐지기 | §12.18 원칙(공격자 쪽만 가지치기, 방어자는 전체 합법수, 시간 cut 없음)으로 구현한다. 측정은 두 층으로 나눈다: **탐지 recall은 witness 5국면 전체**(국소 witness가 있는 양성 표본), **S3 효과의 근거는 확정 2국면**(잠정 3국면은 보조). S2-3이 5/5를 잡아도 "실제 패배 5판을 막는다"고 말하지 않는다 | S2-1·S2-2 결과를 본 뒤 구현 |
| S2-4 S3-B 방식 | §12.20 표대로 S2-1·S2-2 결과로 정한다 | — |

#### S3 arm 분리 (미리 고정)

- **S3-VCT2:** selective depth-2 탐지와 방어(veto 또는 후보 주입).
- **S3-Budget:** stage safety 예산 배분 변경(예산 확대, 또는 후보별 공정 배분으로 첫 후보 독점 방지).
- 두 arm은 따로 측정한다. 이 데이터에는 VCT2와 예산 소진이 같은 판에 함께 나타나는 경우가 있으므로, 같이 바꾸면 원인을 나눌 수 없다.

#### 데이터 규칙

- witness 국면과 8401·8402의 모든 국면은 **학습에 쓰지 않는다.** probe와 평가에만 쓴다(§12.19 H6-0).

### 12.22 S2-2 policy 진단과 확정 2판의 대안 수 (2026-10-08)

결과 파일:
- `docs/mcts-v8-results/s2_policy_diag.json`: 데스크톱 실행, 커밋 `69a3249`, `git_dirty: false`, H3 checkpoint와 metadata SHA-256이 기준선 manifest와 같음
- `s2_witness_alternatives.json`(`scripts/s2_witness_alternatives.py`, 노드 예산 3M, 시간 cut 없음)

#### P92 raw policy (합법수 133개)

| 항목 | 값 |
|---|---|
| R 영역(9–15행 × 9–15열) policy 확률 합 | **0.763** |
| 상위 10개 중 R 영역 수 | 8개: (12,12) 2위 0.090, (13,11) 3위, (14,11) 4위, (11,13) 5위, (14,12) 6위, (12,13) 7위, (14,9) 8위, (11,12) 10위 |
| P92 root 후보 20개(§12.20)에 실린 확률 합 | **0.207**. 확률의 79.3%가 PUCT 행동 집합 밖에 있음 |
| 상위 10개 중 root 후보에 든 수 | (4,4) 1위 0.125, (9,7) 9위. R 영역 수는 하나도 없음 |
| (3,4) / (10,9) | 33위 0.0040 / 45위 0.0017 |

**해석.**
- §12.20에서 미확인으로 남겼던 "policy의 방어점 평가 품질"은 방향 면에서 확인됐다. H3 policy는 우하단 위협 지역을 강하게 본다. 하지만 V8 후보 생성 단계가 그 정보를 PUCT에 넘기지 못한다. C3의 강한 증거다.
- **그렇다고 "C3 때문에 졌다"까지는 아니다.** S2-1에서 우하단 상위 수들이 모두 PROVEN_LOSS로 나오면, policy가 그 지역을 잘 보는 것은 P92를 구하는 데 의미가 없다. 결론은 S2-1의 133개 전체 결과로 낸다.
- 후보를 주입해도 P92에서는 방문된 Q가 모두 −1이다(C4). 그러면 선택 기준은 prior인데, policy 1위가 R 밖의 (4,4)라서 주입만으로는 우하단 수가 선택되지 않을 수 있다. S3-B 평가에서 이 점을 따로 측정한다.

#### VCT2 witness 국면의 결정적 수와 policy 순위

| 판 | causal | 둔 수 (policy 순위, 확률) | policy 1위 | 2위 |
|---|---|---|---|---|
| 8401 p4 (백) | CONFIRMED | (9,8) 7위 0.011, **depth 2에서 패배** | (8,7) 0.579, **depth 2 안에서 패배 없음**(107초) | (6,7) 0.194, depth 2에서 패배 |
| 8401 p16 (백) | CONFIRMED | (8,9) 4위 0.014, **depth 2에서 패배** | (10,8) 0.731, **depth 2 안에서 패배 없음**(164초) | (8,7) 0.136, depth 2에서 패배 |
| 8401 p17 (백) | TENTATIVE | (9,6) 1위 0.540 | 같은 수 | — |
| 8402 p7 (흑) | TENTATIVE | (7,5) 2위 0.194 | (8,6) 0.229 | — |
| 8402 p17 (흑) | TENTATIVE | (6,5) 2위 0.305 | (5,6) 0.350 | — |

- 확정 2판에서는 강한 policy prior(0.58, 0.73)가 최종 tree 선택으로 이어지지 않았다. 두 판 모두 V8-C 교체가 없었고, tree가 고른 수가 그대로 두어졌다.
- H5 PUCT에서 prior 말고 선택에 영향을 주는 주요 신호는 rollout으로 얻은 Q다. 그래서 **rollout 평가가 prior를 뒤집은 것이 유력한 원인**이다.
  - 아직 증명은 아니다. 증명하려면 그 root에서 policy 1위 수와 실제 수의 prior, visit, Q, PUCT 점수 변화를 대조해야 한다(H6 전에 해당 국면에서 다시 측정할 항목).
- 버려진 policy 1위 수는 depth 2 클래스 안에서 패배가 발견되지 않았다(실제로 방어된다는 증명은 아님). 이 패턴은 **S3-VCT2**(고른 수의 depth-2 패배를 veto하면 대안이 남아 있음)와 **H6 value**(rollout 노이즈 감소)로 모두 줄일 수 있다. 그래서 둘을 따로 측정한다(§12.21).
- TENTATIVE 3판은 결정적 수가 policy 1–2위다. 확정 2판과는 다른 실패 유형일 수 있지만, 원인 귀속이 잠정이라 결론을 내지 않는다.

#### S2-1 운영: 범위를 줄이지 않고 133개 전체를 유지

- **133개 전체를 유지하는 이유.** "policy 상위 수 중에 saving 수가 있는가"와 "P92에 saving 수가 있는가"는 다른 질문이다. 상위 수가 모두 PROVEN_LOSS로 나오면 범위를 줄인 검사로는 아무것도 확정할 수 없다. policy가 낮게 본 saving 수가 있다면 S3-B 방식이 policy 주입에서 위협 기반 주입으로 바뀐다.
  - 결과가 어느 쪽이든 쓸모 있다. 모두 PROVEN_LOSS면 P92를 S3-B 근거에서 빼고, 확정 2판을 S2/S3-VCT2의 핵심 시험으로 쓴다. saving 수가 있으면 그 집합의 policy 순위와 root recall로 S3-B 방식을 정한다.
- **병렬화.** 수 하나하나는 서로 독립이라 `--workers N`으로 동시에 계산한다. 결과는 부모 프로세스 하나만 파일에 쓴다.
  - 이미 계산된 수는 jsonl에서 건너뛰므로, 중간에 worker 수를 바꿔도 남은 수만 계산된다. 중단할 때 잃는 건 계산 중이던 수뿐이다.
- **`s2_position_truth.py` 형식 v2.**
  - 줄마다 예산을 기록한다. 예산이 다른 재실행은 거부해서, 한 파일에 서로 다른 예산의 결과가 섞이지 않는다. 예산이 기록되기 전의 줄은 그 실행이 쓴 기본 예산으로 본다.
  - 중단 때문에 잘린 마지막 줄은 건너뛰고 그 수를 다시 계산한다.
  - 진행 줄에 경과 시간과 남은 시간 추정을 출력한다.
- worker 수: 데스크톱(i5-12600K, 32GB)에서 4로 시작한다. CPU와 메모리 사용을 확인한 뒤 6–8로 늘린다. 늘린다고 선형으로 빨라지지는 않는다(P/E 코어 차이, 메모리 경쟁).

### 12.23 S2-1 P92 참값과 S2-3 selective VCT2 탐지기 (2026-10-09)

#### S2-1 P92 결과

결과 파일:
- `docs/mcts-v8-results/s2_p92_truth.json`, `.jsonl`(데스크톱, 커밋 `6a75c12`, 예산 node_limit 20,000 / call_limit 100,000 / node_budget 10,000,000, workers 4 → 8)
- 결합 결과 `s2_p92_joined.json`(`scripts/s2_join_p92.py`: truth + `s2_policy_diag.json`의 policy 분포 + S1 probe의 root 후보)

provenance:
- 예산이 기록된 92개 행은 모두 위 예산으로 같다.
- 앞서 실행해서 이어 붙인 41개 legacy 행에는 행별 예산 기록이 없다. 그 실행은 기본 예산을 썼다(`resume.legacy_rows = 41`).

**결론.** P92의 합법수 133개를 모두 처리했다.
- **129개(96.99%)는 상대의 VCT2 승리가 증명됐다.** 나머지 4개 (10,9), (11,7), (14,6), (15,4)는 이 예산 안에서 결론이 나지 않아 UNKNOWN이다.
- 국면 판정은 **`UNRESOLVED`**다. 패배를 피하는 수는 발견되지 않았다. 하지만 UNKNOWN 4개 중 하나가 패배를 피하는 수일 가능성은 배제하지 못한다. 그래서 "수 133개 중 대부분이 진다"를 "국면이 확정적으로 졌다"로 해석하지 않는다.
- UNKNOWN의 원인(전체 예산, 호출 한도, VCF 단일 호출의 node_limit 중 무엇인지)은 이번 기록에 없다. 이후 `s2_position_truth.py`(형식 v3)가 행마다 `search`로 기록한다.
- 증명된 129개의 `lost_depth`는 모두 2다. 깊이 0이나 1에서 진 수는 없다.

**policy와 후보 생성.**
- policy 상위 12개 수는 모두 VCT2 패배다. 패배가 증명된 수에 실린 policy 확률은 0.9958이다.
- 패배로 증명되지 않은 수 중 policy 순위가 가장 높은 것은 45위다. R 영역은 37개 중 36개가 패배로 증명됐고, (10,9)는 UNKNOWN이다.
- UNKNOWN 4개 중 (10,9)와 (11,7)은 이미 root 후보에 반복해서 들어 있었다. puct 두 arm은 10/10, full은 6/10과 10/10이다. 그러니 이 둘이 패배를 피하는 수라 해도 후보 누락 문제가 아니다.
- (14,6)과 (15,4)는 root 후보에 없었지만 policy 순위가 66위, 112위다. 그러니 policy 상위 K개를 주입하는 방식의 근거가 되지 않는다.
- 결론: P92는 "policy는 패배를 피하는 수를 알았는데 후보 생성기가 버렸다"는 가설을 지지하지 않는다.

**S3-B 보류(규칙 일반화).** P92를 포함해 지금까지 근거가 없으므로 S3-B는 구현하지 않는다.
- 다시 검토하는 조건: **policy가 높게 본 수이면서 깊이 2에서 패배가 발견되지 않은 수(NO_VCT2_FOUND)가 root 후보에서 반복해서 빠지는 사례**가 확인될 때.
- 이 규칙은 §12.20 표의 S3-B 행을 대체한다.

**흑 91 → 백 92 해석.**
- 흑 91 직후에는 백의 깊이 2 승리가 확인되지 않았다(§12.18). 백 92 뒤에는 흑의 합법수 대부분이 깊이 2에서 진다.
- 이는 백 92가 조용한 준비수 하나이고 그 뒤에 VCT2가 이어지는, **흑 91 기준 VCT3급 지평선 실패 가능성**과 일치한다.
- 다만 아직 증명되지 않은 것이 두 가지 있다.
  - UNKNOWN 4개가 있으므로, 백 92가 흑의 모든 응수에 대해 강제승을 만든다는 것.
  - 흑 91을 실제 패착으로 특정하는 것. 그러려면 흑 91 자리에서 다른 수 중 지지 않는 대안이 있었는지 확인해야 한다.
- 그래서 "흑 91에서 시작되는 VCT3급 지평선 실패 가능성이 강하게 제기됐다"까지만 기록한다.

#### S2-3 selective VCT2 탐지기

구현: `src/analysis/selective_vct.py`, 테스트 `tests/test_selective_vct.py`.
- `ThreatSolver.quiet_moves`를 hook으로 분리했다. 기본값은 기존과 같이 모든 합법수라서 동작이 바뀌지 않는다(threats·V8 테스트 69개 통과).
- **깊이 0–1**은 기존 전체 solver를 쓴다(V8-C와 같은 클래스, 예산 400k).
- **깊이 2**는 `SelectiveSolver`를 쓴다. 공격자의 조용한 수는 위협 수(4, 열린 3, 끊긴 3)로만 제한하고, 방어자의 응수는 항상 전체 합법수를 본다. 그래서 PROVEN_LOSS는 증명이고, "못 찾음"은 불완전할 뿐이다.
- 상태는 `PROVEN_LOSS`(깊이와 증거 포함) / `NO_TARGETED_VCT2_FOUND` / `UNKNOWN`(원인 포함) 세 가지다. "SAFE"나 "PROVEN_SAFE"는 쓰지 않는다. 깊이 2 탐색으로는 그보다 깊은 패배를 배제할 수 없기 때문이다.
- `verify_witness`는 깊이 2 PROVEN_LOSS를 전체 `ThreatSolver`로 다시 검증한다. 증거로 나온 위협 수 뒤에서 방어자의 모든 응수를 depth 1로 확인한다.
- 예산은 node와 call 수로만 정한다(재현 가능).

평가: `scripts/s2_vct2_detector_eval.py`. 세트와 gate는 측정 전에 고정했다.
- **A 핵심:** 확정 2국면. 결정적 수 2/2를 탐지해야 하고, 깊이 2 패배가 없는 대안을 PROVEN_LOSS로 내면 안 된다.
- **B 탐색용:** 잠정 3국면.
- **C stress:** P92 수 133개.
- **D control:** H5 policy가 이긴 대국의 tree 수 표본. 평가에만 쓰고 학습에는 쓰지 않는다.
- **gate:** A 결정적 수 2/2, soundness 위반 0, D 비용 중앙값 ≤ 2초이고 p95 ≤ 10초.
  - D 비용은 두 가지로 잰다. 전체 비용, 그리고 추가 비용(selective 단계). 깊이 0–1 단계는 V8-C가 tree 경로에서 이미 검사하는 클래스라서, 탐지기를 붙였을 때 실제로 늘어나는 비용은 selective 단계다.

결과(클라우드, workers 4, D는 표본 40개, `s2_detector_eval_b{1000k,10k,30k}.json`):

| selective 예산 | A 결정적 수 | A 양성 전체 | A 오탐 | soundness 위반 | D 개입 | D UNKNOWN | D 추가 비용 중앙값 / p95 | D 전체 p95 |
|---|---|---|---|---|---|---|---|---|
| 1,000k(기본) | **2/2** | 4/4 | 0 | 0 | 0/40 | 2 | 0.32초 / **291.5초** | 292.1초 |
| 30k | **2/2** | 4/4 | 0 | 0 | 0/40 | 5 | 0.30초 / **22.6초** | 99.7초 |
| 10k | **2/2** | 3/4 | 0 | 0 | 0/40 | 7 | 0.35초 / **11.3초** | 115.6초 |

- **B(기본 예산):** 2/3. 8401 p17 (9,6)은 161초 뒤 NO_TARGETED가 나왔다. 백의 승리에 위협이 아닌 조용한 수가 필요한 사례로 보인다(selective 탐색의 불완전성). 탐지한 2개는 모두 전체 solver 검증을 통과했다.
- **P92 (3,4):** 기본 예산에서 PROVEN_LOSS(73초, 증거는 백 (11,13)). 같은 수를 전체 탐색으로 판정했을 때는 1,043초였다.
- **판정: 미리 정한 gate를 세 예산 모두 통과하지 못했다.** A 결정적 수와 soundness는 모두 통과했지만, D 비용 p95에서 걸렸다.
  - 10k가 가장 가깝다(추가 비용 p95 11.3초 > 10초). 추가 비용 중앙값은 세 예산 모두 0.3초대다.
  - 전체 p95가 큰 이유는 깊이 0–1 단계(400k)의 긴 꼬리다. 이 비용은 V8-C가 이미 쓰고 있는 비용이다.
  - 처리 속도는 약 1,000 노드/초(순수 Python VCF)다.
- **참고:** H5 policy arm의 V8 착수 시간 p95는 이미 약 80초다(§12.19). 10초 기준은 엔진 전체 비용과 비교해 정한 값이 아니었다. 기준을 바꿀지는 결과를 본 뒤 정할 일이 아니므로 사용자 결정 사항으로 둔다(아래).

#### 다음 결정 (사용자)

| 선택지 | 내용 |
|---|---|
| (a) 기준 유지 | 10초를 지킨다. selective 예산을 더 낮춰(예: 8k) 다시 측정한다. A 결정적 수 중 하나가 selective 7.7k 노드에서 잡혔으므로 recall이 떨어질 위험이 있다 |
| (b) 비용 기준을 엔진 수준으로 교체 | D gate 대신, S3-VCT2 arm을 벤치마크에 넣었을 때의 **시간 비율 ≤ 1.5**(H5와 같은 guard)를 비용 기준으로 쓴다. 10k 또는 30k로 S3에 진입한다 |
| (c) 속도 개선 먼저 | VCF 캐시 공유나 위협 수 정렬 같은 엔지니어링으로 꼬리를 줄인 뒤 다시 측정한다 |

C(P92 129개 recall)와 D 100개 전체 측정은 선택한 예산으로 데스크톱에서 실행한다.

### 12.24 S3-VCT2: 비용 gate 개정과 엔진 벤치마크 (2026-10-09)

#### 프로토콜 개정 (사용자 결정: §12.23 선택지 (b), 실패 시 (c))

- **개정 내용.** §12.23의 D 비용 gate(탐지기 단독 중앙값 ≤ 2초, p95 ≤ 10초)는 **공식 비용 gate에서 내린다.** 이 기준은 엔진 전체 지연(H5 policy arm의 착수 시간 p95 ≈ 80초)과 무관하게 정한 절대값이었다. 대신 **S3-VCT2 arm을 실제로 붙였을 때의 end-to-end 시간 비율 ≤ 1.5**를 공식 비용 gate로 쓴다(H5와 같은 guard).
- **소급 금지.** 이미 측정한 D 결과(`s2_detector_eval_b*.json`)를 새 기준으로 다시 읽어 PASS를 선언하지 않는다. 새 gate는 아래의 **새 벤치마크**에서만 판정한다. §12.23의 표와 "gate 미통과" 기록은 그대로 둔다.
- **예산.** selective 깊이 2 예산은 10k(node_budget 10,000, node_limit 20,000, call_limit 20,000)로 고정한다. 세 예산 중 추가 비용 꼬리가 가장 작았고, A 결정적 수 2/2를 탐지했다.
- **실패 시.** 비용 때문에 실패하면(아래 OPTIMIZE) (c) 속도 개선으로 간다. (a) 예산 추가 축소는 하지 않는다.

#### S3-VCT2 엔진 동작 (`root_vct2_check`, 기본 꺼짐)

구현: `src/analysis/mcts_v8.py`의 `_vct2_veto`, 벤치마크 arm `puct_policy_vct2` = `puct_policy` + `root_vct2_check=True`.
- **위치.** tree 경로에서 V8-C(`_verify_root_choice`) 다음에 한 번 실행한다. stage 경로와 own VCF/VCT 경로는 건드리지 않는다.
- **검사.** 둘 수(V8-C 이후의 선택)를 새 `SelectiveSolver`(10k 예산)로 깊이 2까지 본다. 공격자는 위협 수만, 방어자는 모든 응수를 본다. 그래서 UNSAFE는 증명된 패배다.
- **veto.** 증명된 패배일 때만 tree 순서(방문 수, 평균 가치)로 다음 자식을 최대 `vct2_max_children = 4`개까지(둘 수 포함) 본다. V8-C가 UNSAFE로 판정한 수와 즉시 지는 수는 건너뛰고, 증명된 패배가 아닌 첫 수를 둔다.
- **UNKNOWN 규칙.** UNKNOWN(예산 초과)은 안전 판정이 아니다. 둘 수가 UNKNOWN이면 veto하지 않는다(증명이 없으면 tree의 선택을 바꾸지 않는다). 대안을 고를 때 UNKNOWN 수는 "증명된 패배가 아님"으로 허용한다. 모든 대안이 증명된 패배이면 원래 선택을 그대로 둔다(V8-C와 같은 규칙).
- **기록.** 착수마다 `vct2: {checked, switched, nodes, seconds}`. 요약에 `vct2` 블록(검사 수, 둘 수가 증명된 패배였던 수, 교체 수, 시간 분포).
- 꺼져 있으면 동작은 이전과 같다(V8 회귀 테스트, frozen baseline 확인).

#### 벤치마크 설계 (실행 전 고정)

- arm: `puct_policy`(기준) vs `puct_policy_vct2`. 상대 `v8:full`, `--puct-c 1.5`, H3 policy checkpoint, 25 opening pair(50판)씩.
- seed: **8411, 8412**(새 seed). 8401/8402는 H5에 썼고, 8403–8405는 H6용으로 남겨 둔다.
- 기준 arm도 새 seed로 다시 돌린다. 시간 비율은 같은 기계, 같은 부하에서 재야 하므로 **같은 seed의 두 arm을 동시에 같은 workers 수로** 실행한다.
- 비교: `scripts/s3_compare.py`.
  - R = Σ V8 착수 시간(vct2) / Σ V8 착수 시간(기준), 대국 시간 비율, 착수 시간 중앙값·p95
  - opening pair 단위 점수 차(vct2 − 기준)와 bootstrap 95% 구간
  - 메커니즘: 검사한 수, 둘 수가 증명된 패배였던 수, 교체 수
- **판정 규칙(고정).**

| 판정 | 조건 |
|---|---|
| REJECT | 안전 불변식 위반이 있거나, 점수 차 < −0.05 |
| OPTIMIZE → (c) | REJECT가 아니지만 R > 1.5, 또는 어느 arm이든 상대 대비 시간 비율 > 1.5 |
| ADOPT | 위 둘 다 아님 (R ≤ 1.5, 시간 비율 ≤ 1.5, 위반 0, 점수 차 ≥ −0.05) |

- ADOPT는 "비용 안에서 해가 없다"는 뜻이다. 점수 향상은 요구하지 않는다(교체가 드문 장치라 50판×2로는 검정력이 낮다). 교체 사례는 메커니즘 기록으로 따로 본다.
- C(P92 129개 recall)와 D 100개 전체 측정은 선택 사항이다. 판정에는 쓰지 않는다.
- 실행: `scripts/run_s3.ps1`(데스크톱). seed마다 두 arm을 동시에 workers 4로 돌리고, 1분마다 python 프로세스 우선순위를 높음으로 올린다. 다시 실행하면 `--games-jsonl`에서 이어 간다. 끝나면 `s3_compare.py`를 실행해 `runs/s3/s3_compare.json`을 만든다.

### 12.25 S3-VCT2 결과: ADOPT(안전·비용), 효능은 미입증 (2026-10-09)

결과 파일(데스크톱, 커밋 `b25aa3e`, git_dirty = False):
- 원본: `docs/mcts-v8-results/s3_base_8411.json`, `s3_base_8412.json`, `s3_vct2_8411.json`, `s3_vct2_8412.json`
- 비교: `s3_compare.json`(클라우드에서 같은 결과로 다시 계산해 확인함)
- veto 재검증: `s3_vct2_full_recheck.json`(`scripts/s3_vct2_recheck.py`). veto가 일어난 국면에서 검사한 수마다 다음을 기록한다.
  - 엔진 판정(10k)
  - selective 기본 예산(1M) 판정과 `verify_witness` 결과
  - 전체 클래스(공격자의 모든 조용한 수, P92 참값 예산 10M) 판정
  - 노드 수, 시간, 입력 파일 SHA-256, git commit

#### 판정

**ADOPT (safety / non-inferiority / cost 통과; efficacy not established).**
- 미리 정한 세 gate를 모두 통과했으므로 통합 대상으로 채택한다.
  - 안전 불변식 위반 0
  - 점수 차 −0.01(95% 구간 [−0.03, 0.0]) ≥ −0.05
  - end-to-end 시간 비율 1.054 ≤ 1.5. 상대 대비 시간 비율은 기준 0.896, vct2 0.941이다.
- **ADOPT는 S3-VCT2가 기존 엔진보다 강하다는 뜻이 아니다.**
  - 점수는 기준 0.825(77승 11무 12패), S3-VCT2 0.815(76승 11무 13패)이고 향상은 관측되지 않았다.
  - 통과한 것은 우월성(superiority)이 아니라 비열등성(non-inferiority) 기준이다.
- 비용: 검사 1회는 중앙값 0.13초, p95 5.2초, 최대 9.4초이고 노드 중앙값은 46이다. §12.23의 탐지기 단독 측정(추가 비용 p95 11.3초, 전체 115초)보다 훨씬 작다. 깊이 0–1을 V8-C가 먼저 거르기 때문이다. (c) 속도 개선은 지금 필요 없다.

#### 개입 빈도와 재검증

| 항목 | 값 |
|---|---|
| 검사한 tree 수 | 1,910 (vct2 arm의 tree 수 전부) |
| veto 발생(둔 수가 PROVEN_LOSS) | 5 (0.262%) |
| 실제 교체 | 2 (0.105%) |
| 수순이 달라진 대국 | 2 / 100 (나머지 98판은 수순이 완전히 같다) |
| 결과가 달라진 대국 | 1 / 100 (8411 p8, 승 → 패) |
| veto precision(전체 클래스로 확인) | **5/5**. 엔진이 UNSAFE로 본 수 7개 모두 전체 클래스에서 PROVEN_LOSS |
| 불건전 증거(`verify_witness` False) | 0 |
| 교체 수의 엔진 판정 | SAFE 1, UNKNOWN 1 |
| 교체 수의 전체 클래스 판정 | SAFE 1(깊이 2 안 패배 없음), PROVEN_LOSS 1 |
| replacement VCT2 escape rate | **1/2** |
| false switch(전체 클래스 SAFE인 수를 바꿈) | 0/2 |

veto 국면(좌표는 1-indexed, ply는 0부터 센다):

| 대국 | ply | 둔 수(엔진 → 전체 클래스) | 대안 | 결과(기준 → vct2) |
|---|---|---|---|---|
| 8411 p8 백 | 11 | (5,9) UNSAFE → 깊이 2 패배 | (7,7) 엔진 UNKNOWN → **전체 클래스 깊이 2 패배**(1M selective 102초, 전체 206초) | 승 → **패** |
| 8412 p4 백 | 11 | (5,9) UNSAFE → 깊이 2 패배 | (9,7) 엔진 SAFE → 전체 클래스 SAFE(133초) | 승 → 승 |
| 8411 p22 백 | 13 | (5,8) UNSAFE → 깊이 2 패배 | (8,11), (10,8) 모두 깊이 1 패배 → 교체 없음 | 승 → 승 |
| 8411 p19 흑 | 156 | (7,14) 깊이 0 패배 | 나머지는 V8-C UNSAFE → 검사 대상 없음 | 패 → 패 |
| 8412 p5 백 | 23 | (4,7) 깊이 0 패배 | 위와 같음 | 패 → 패 |

- 앞의 두 국면이 실제로 개입한 경우다. p22, p19, p5는 이미 진 국면이거나 검사한 후보가 모두 진 국면이다.
- 8411 p22와 두 교체 국면의 기준 arm에서는 VCT2 패배가 증명된 수를 두고도 이겼다. 상대가 그 강제승을 쓰지 못했다.

#### 8411 p8 해석

- **empirical regression은 있다.** S3-VCT2의 교체가 실제 대국 수순의 분기점이었고, 결과는 승에서 패로 나빠졌다. 그러니 관측된 회귀를 장치와 무관하다고 볼 수 없다.
- **proof unsoundness는 없다.** 전체 클래스 재검증에서 원래 수와 교체 수가 모두 VCT2 패배였다. 그러니 이것은 "VCT2-safe한 수를 잘못 veto해서 패배를 만든 오류"가 아니다.
- **핵심 원인:** 10k selective 예산으로는 교체 후보의 패배를 증명하지 못했다(UNKNOWN). 검사한 대안 중 NO_TARGETED_VCT2_FOUND가 없었고, 규칙대로 UNKNOWN 수를 골랐다.
- **"UNKNOWN보다 NO_TARGETED 우선" 규칙은 이 사례를 고치지 못한다.** 8412 p4가 보여 주듯, NO_TARGETED 대안이 있으면 이미 그쪽으로 교체한다. p8에서는 검사한 대안이 UNKNOWN 하나뿐이었다. 그래서 정책 문제는 상태 우선순위가 아니라 다음 둘이다.
  - 대안 탐색 폭이나 대안 확인 예산을 늘릴 것인가
  - 검사한 대안이 모두 UNKNOWN일 때 어떤 fallback을 쓸 것인가

#### 두 가지를 나눠서 해석한다

- **Detection correctness:** 재검증한 PROVEN_LOSS 7개 중 잘못된 증명은 없다(false positive 0, 재현 파일 `s3_vct2_full_recheck.json`).
- **Replacement effectiveness:** PROVEN_LOSS를 찾은 뒤 고른 대안이 실제로 패배를 피하는지는 충분히 검증되지 않았다(1/2, 표본 2개).

**효능을 측정하지 못한 이유.**
- 100판 벤치마크에서는 S3-VCT2의 개입 빈도(veto 0.26%, 교체 0.1%, 결과가 달라진 대국 1판)가 너무 낮다. 효능을 판별할 통계적 검출력이 부족하다.
- `v8:full` 상대의 공격 horizon 제한(자기 공격은 VCT1까지)도 VCT2 회피의 이득 신호를 약하게 하는 추가 요인이다.
- 그러니 상대만 강하게 바꿔서는 해결되지 않는다. 개입 국면을 직접 겨냥한 평가가 필요하다.

**새 연구 질문.** 탐지 정확도가 아니라 **"VCT2 패배를 정확히 찾은 뒤, 제한된 계산량으로 더 나은 대안을 어떻게 고를 것인가"**다. 탐지기가 실패한 것이 아니라, replacement / adjudication 단계가 다음 병목으로 드러났다.

#### 동결

- **S3-VCT2-v1**을 다음 설정으로 동결한다. 아래 평가가 끝날 때까지 바꾸지 않는다.
  - `root_vct2_check = True`
  - `vct2_node_budget = 10,000`, `vct2_node_limit = 20,000`, `vct2_call_limit = 20,000`
  - `vct2_max_children = 4`
  - V8-C 뒤에 실행, UNKNOWN 대안 허용
- replacement 정책 변경은 아래 E1·E2가 끝난 뒤에, 새 seed로 검증한다. 원래 수가 PROVEN_LOSS이고 검사한 대안이 모두 UNKNOWN인 경우는 별도 정책 문제로 다룬다.

#### 다음 단계 (targeted efficacy 평가, 지표는 실행 전 고정)

| 단계 | 내용 | 지표 |
|---|---|---|
| E0 | 벤치마크 veto 국면 재검증 | **완료**(위 표, `s3_vct2_full_recheck.json`) |
| E1 | **rescue suite.** 원래 수(기준 엔진이 둔 수)는 전체 클래스 VCT2 패배이고, 전체 클래스 SAFE 대안이 하나 이상 있는 확정 국면 세트. 출처: §12.22 확정 2국면(대안의 참값 있음), 8412 p4 ply 11, 그리고 H5·S3 대국(8401/8402/8411/8412, 평가 전용)에서 캐낸 국면. 각 국면에서 `puct_policy`와 `puct_policy_vct2`의 착수를 여러 random seed로 기록한다 | PROVEN_LOSS precision, detection rate(원래 수를 veto한 비율), **replacement VCT2 escape rate**, UNKNOWN replacement rate, false switch rate, intervention cost |
| E2 | **VCT2를 응징할 수 있는 상대.** selective VCT2 공격(증명된 WIN만)을 켠 V8을 상대로, 새 seed 8413/8414에서 `puct_policy` 대 `puct_policy_vct2`를 각 25쌍 | 일반 승률, paired diff, 시간 비율, 개입 빈도 |
| E3 | E1·E2 결과를 보고 replacement 정책 후보(대안 폭, 대안 예산, 모두 UNKNOWN일 때의 fallback)를 정하고, 새 seed로 검증 | 미리 고정 |

- E1 국면을 캐는 데 쓴 대국은 학습에 쓰지 않는다(§12.19의 8401/8402 규칙과 같음). H6용 seed 8403–8405는 계속 예약해 둔다.
- 순서: E1 → E2 → (E3). H6에서 S3 채택에 따른 VCT2 on/off 요인(§12.18)은 S3-VCT2-v1 설정으로 넣는다.

### 12.26 E1 rescue suite 설계와 웹 실전 stress test (2026-10-09, 검토 반영)

**S3-VCT2-v1은 E1 동안 한 줄도 바꾸지 않는다.**
- 설정은 `src/analysis/s3_vct2_v1.py`에 전부 적어 두었다(`V8_DEFAULTS`에서 파생하지 않음). 테스트가 S3 결과 파일의 `v8_config`와 같은지 확인한다.
- policy checkpoint도 S3가 쓴 바이트(SHA-256)와 다르면 거부한다.
- 10k → 20k나 K4 → K6 같은 변경은 모두 E3에서 한다.
- **E1은 개발 실험이 아니라 진단 실험이다.**

#### E1의 목적

baseline이 고른 수가 `PROVEN_LOSS_VCT2`이고, 현재 엔진이 접근할 수 있는 대안 중 적어도 하나가 전체 깊이 2 탐색에서 VCT2 패배가 배제된 국면을 모은다. 그런 국면에서 동결된 S3-VCT2-v1이 실제로 패배를 피하는 수를 고르는지 측정한다.

- "안전한 수"라고 부르지 않는다. 이 라벨은 **`VCT2_CLEAR`**(깊이 2 안에 패배 없음)다. 게임 전체에서 안전하다는 뜻이 아니다.
- `VCT2_CLEAR`는 E1 평가용으로만 쓴다. **H6 value 라벨로 쓰지 않는다**(§12.18: 증명된 WIN/LOSS만 라벨).
- 성공률 하나를 재는 것이 목적이 아니다. **실패하면 아래 파이프라인의 어느 단계가 병목인지 찾는 것**이 목적이다.

```
baseline 수가 실제 VCT2 패배
 → ① 10k 탐지기가 패배를 찾았나          (DETECT_MISS)
 → ② 살 수 있는 대안이 candidate pool에 있나 (POOL_MISS)
 → ③ 그 대안이 veto 순서 상위 4개 안에 있나  (K4_MISS)
 → ④ 10k에서 그 대안을 UNKNOWN이 아닌 것으로 판별했나 (BUDGET_AMBIGUITY)
 → ⑤ 최종 선택 규칙이 그 대안을 골랐나       (SELECTION_ERROR)
 → ⑥ 최종 수가 실제로 VCT2를 피했나          (RESCUED)
```

#### Suite 구축 (`scripts/e1_build_suite.py`, builder와 evaluator 분리)

- **출처(고정).** baseline `puct_policy` 대국 로그 네 개: `h5_policy_8401/8402.json`, `s3_base_8411/8412.json`.
  - 대상은 tree 경로 수 중 V8-C가 이미 패배로 증명하지 않은 수 전부(약 3,980개)다. 손으로 고르지 않는다.
  - **E2 seed 8413/8414는 쓰지 않는다.** 웹 대국 probe(P92/P93/P94)는 주 성적에 넣지 않는다(필요하면 sanity로 따로 본다).
- **screen.** 둔 수를 전체 깊이 0–2 클래스(공격자의 모든 조용한 수)로 판정한다. 예산은 노드 50k다.
  - 결과는 PROVEN_LOSS / VCT2_CLEAR / UNKNOWN 중 하나다.
  - 표본 16개로 잰 비용은 수당 약 45초였고 절반은 UNKNOWN이었다. 전체 약 50 CPU시간이라 데스크톱 14 workers로 3~4시간이다.
  - **한계:** 50k 안에 증명되지 않는 패배는 suite에 들어오지 않는다. screen UNKNOWN 수를 보고한다.
- **선택.**
  - PROVEN_LOSS 행만 쓴다.
  - 같은 대국에서 연속된 V8 수(ply 차이 2)는 한 전술 에피소드로 보고 첫 수만 남긴다.
  - D4 정규화 board hash로 회전·반사 중복을 없앤다.
  - control: VCT2_CLEAR 행에서 seed 2611로 40개를 뽑는다. 대국당 1개, hash 중복은 뺀다.
- **pool (policy 필요, 데스크톱).**
  - 각 국면에서 동결 설정으로, VCT2 검사만 끄고 엔진을 seed 3개로 돌린다(seed는 국면 key에서 결정적으로 만든다).
  - root 자식(방문 수, 평균값)을 기록한다. 이것이 candidate pool이고, veto 순서도 여기서 나온다.
  - 패배 국면은 pool의 모든 수를 전체 클래스(P92 참값 예산 10M)로 판정한다.
  - 선택 행과 판정 결과를 manifest(`e1_manifest.json`)에 고정한다.
- **분류.**
  - **E1-P:** pool에 VCT2_CLEAR가 있음. run마다 P-K4(상위 4개 안에 있음) / P-POOL(pool에만 있음)으로 다시 나눈다.
  - **E1-N:** pool 전부 PROVEN_LOSS.
  - **E1-UNRESOLVED:** CLEAR 없음, UNKNOWN 있음. 주 지표에서 제외한다.
  - **E1-C:** control.

#### 평가 (`scripts/e1_evaluate.py`)

- manifest의 같은 seed로 S3-VCT2-v1을 실행한다. 같은 seed면 tree가 같으므로, VCT2 검사가 처음 본 수는 builder의 수와 같아야 한다(`inconsistent_runs`로 보고).
- run마다 결과 하나:

| 클래스 | 결과 |
|---|---|
| E1-P | RESCUED, TREE_AVOIDED, DETECT_MISS, POOL_MISS, K4_MISS, BUDGET_AMBIGUITY, SELECTION_ERROR, TRUTH_UNRESOLVED, ROUTE_OTHER |
| E1-N | KEPT, LOSS_TO_LOSS_SWITCH, TRUTH_UNRESOLVED |
| E1-C | NO_VETO, VETO_ON_LOSS, FALSE_VETO |

- **주 지표:** `rescue_rate` = 최종 수가 VCT2_CLEAR인 E1-P run의 비율.
- **함께 보는 지표:** veto_recall, rescue_pool_coverage, rescue_k4_coverage, conditional_escape_rate, unknown_replacement_rate, false_veto, unsound_witness(0이어야 함), 비용(VCT2 검사 시간과 착수 시간의 p50/p95/max).
- 단위는 run이다(국면 × seed 3개). 국면별 결과 목록도 같이 낸다.
- **규모:** E1-P 국면이 최소 20개, 가능하면 30개 이상이면 비율로 해석한다. 그보다 적으면 퍼센트가 아니라 개별 실패 유형으로 해석한다. 숫자를 억지로 채우지 않는다.
- **dev / holdout.**
  - 지금 네 출처에서 나온 국면은 모두 **E1-dev**다.
  - E1-holdout은 앞으로 쌓이는 로그(8413/8414 제외)에서 같은 규칙으로 캔다.
  - E3에서 바꾼 정책은 holdout에서 판정한다. dev에서 원인을 보고 고친 뒤 dev로 다시 재서 성능을 주장하지 않는다.

#### E3 방향 (E1 결과를 보기 전에 고정)

| E1에서 많이 나온 실패 | E3 후보 |
|---|---|
| DETECT_MISS | selective 탐지 범위·패턴 |
| POOL_MISS / K4_MISS | K 확대 또는 후보 순서 |
| BUDGET_AMBIGUITY | 대안에만 추가 예산, 2-pass 확인 |
| SELECTION_ERROR | replacement 순위·fallback |
| E1-N 비중이 큼 | veto만으로는 해결 불가(더 앞 수의 문제) |
| unsound_witness > 0 | **E3 전에 solver 버그 수정** |

BUDGET_AMBIGUITY와 K4_MISS를 반드시 나눠 본다. p8 하나만 보면 "예산을 늘리면 된다"고 보이지만, K4 밖의 rescue가 많다면 예산 증가는 엉뚱한 해결책이다.

#### 로드맵 (수정)

```
S3-VCT2-v1 ADOPT + FREEZE → E0 재검증 ✅ → [웹 실전 stress] → E1-dev → E2(새 seed 8413/8414, VCT2를 응징하는 상대)
  → E1 + E2 종합: E3 필요? ─ 예 → E3 → E1-holdout 재검증 → H6
                           └ 아니오 ──────────────────────→ H6
```

#### 웹 실전 stress test (External Web Stress Cases)

- `scripts/run_web_play.py`에 상대 **"S3-VCT2-v1 (동결)"**(`s3`)을 넣었다.
  - `--policy-checkpoint`(기본 `runs/h3_policy_64x4/best.pt`)가 있고 SHA-256이 S3와 같을 때만 등록된다.
  - 엔진은 `analysis.s3_vct2_v1.make_agent`다(설정 변경 없음).
- **사람 중계 방식.**
  - 외부 사이트 상대의 수를 로컬 보드에 클릭하면, 엔진이 다음 수를 계산해 "AI 착수"에 표시한다. 이 수를 사람이 외부 사이트에 직접 둔다.
  - 외부 사이트를 자동으로 조작하지 않는다(사이트 규칙 보호).
  - 보드에 좌표(열 A–O, 행 1–15 아래부터, 중앙 H8)를 표시한다.
  - "무르기"는 잘못 입력한 상대 수를 되돌린다(횟수를 기록).
  - 로컬 규칙상 흑의 첫 수는 중앙(H8)으로 고정이다.
- **기록.**
  - `game.json`의 `agent_info`: 엔진 이름, 동결 커밋, 실행 커밋, git_dirty, 전체 설정, checkpoint와 metadata 해시
  - 착수마다: board hash(착수 전), 경로, tree 선택(`v8_v7_move`), V8-C 상태, VCT2 검사 목록과 상태, 교체 여부, 노드, 시간. `moves.csv`에도 `vct2_*` 열이 들어간다.
  - **VCT2 veto가 일어난 국면**은 그 자리에서 `logs/web_play/veto_positions/<시작시각>_ply<N>.json`으로 저장한다.
  - 끝나지 않은 채 새 게임을 시작해도 `result = UNFINISHED`로 기록을 남긴다.
- **해석 규칙.**
  - 이 결과는 E1/E2 성적과 섞지 않는다. 볼 것은 정상 착수, 렌주 금수 오류 여부, veto 발동, UNKNOWN replacement 발생, 실전 시간이다.
  - 흑 3판 + 백 3판 정도면 충분하다.
  - 웹에서 발견한 국면은 offline으로 참값을 확정한 뒤 **E1-dev 후보로만** 등록한다.
  - 웹 결과를 보고 설정을 바꾸거나, 그 사례로 성능을 주장하지 않는다.

### 12.27 웹 실전 stress test 결과 (8판)와 stage 경로 coverage gap (2026-10-09)

입력: `docs/mcts-v8-results/web_stress_20261009/`(원본 로그 8판과 veto 국면 1개). 재검증 결과는 `web_stress_20261009_recheck.json`, 재검증 도구는 `scripts/web_recheck.py`. 좌표는 1-indexed (행, 열)이다.

**provenance.**
- 8판 모두 엔진은 S3-VCT2-v1이고 설정과 checkpoint 해시가 동결값과 같다. git_dirty = False.
- 실행 커밋은 `dde01ea`다. `b06ecd6..dde01ea` 사이 `src/` 변경은 새 파일 `analysis/s3_vct2_v1.py`(설정 상수와 checkpoint 확인)뿐이다. 그래서 탐색과 solver 동작은 같다.
- `b25aa3e..b06ecd6` 사이에는 `src/` 변경이 없다.

#### 결과 요약 (External Web Stress Cases — E1/E2 성적에 넣지 않음)

- **성적:** AI 7승 1패. AI 백 3승 1패, AI 흑 4승.
  - 67수 대국은 무르기 2회라 정식 집계에서 빼면 6승 1패다.
  - 모든 판이 같은 seed와 같은 상대였고, 4판은 18~25수에 끝났다.
  - **승률로 해석하지 않는다.**
- **경로:** AI 124수 중 tree 53, stage4 20, stage2 20, own_vcf 11, stage3 7, stage1 7, own_vct 5, stage5 1.
  - tree 53수는 모두 VCT2 검사를 받았다. 첫 검사 결과는 SAFE 37, UNKNOWN 15, UNSAFE 1이다.
  - **UNKNOWN 비율은 28%다.** S3 벤치마크에서도 1,910개 중 336개(17.6%)였다. "UNKNOWN이면 그대로 둔다"는 규칙은 드문 경우가 아니라 tree 수의 1/5~1/4에 적용된다.
- **시간:** AI 착수 중앙값 1.63초, p95 82.9초, 최대 124.4초. VCT2 검사는 중앙값 0.19초, p95 4.9초, 최대 9.5초다.
  - 긴 수는 모두 S3 밖이다. V8-C root 검사 최대 118초, V8-A stage4 VCT 최대 85초, V8-B 자기 공격 최대 82초.
  - 지연 문제는 별도 트랙(L)으로 둔다. S3 예산을 줄이는 것은 우선순위가 아니다.

#### veto 사례 (AI 백 20수, 대국 `161802`)

- tree의 root 자식은 **3개뿐**이었다: (9,6) 95회 방문, (8,5) 4회, (9,8) 1회.
- VCT2 검사 결과: (9,6) UNSAFE, (8,5) UNSAFE, (9,8) UNKNOWN. 그래서 (9,8)로 교체했고, 이 대국은 AI가 38수에 이겼다.
- **전체 solver 재검증(10M):** 세 수 모두 PROVEN_LOSS다.
  - (9,6): 깊이 2
  - (8,5): 깊이 1
  - (9,8): 깊이 2, 207k 노드, 149초
- **판정:** 8411 p8과 같은 유형이다. UNKNOWN 대안으로 빠져나왔지만 그 수도 졌고, 상대가 응징하지 못해 이겼다. **rescue가 아니다.**
- 후보 pool이 3개뿐이었으므로 POOL_MISS 위험도 보여 준다(E1에서 측정).

#### 1패 (대국 `155734`, AI 백, 33수 패배)

| 수 | 경로 | 선택 | 엔진 판정 | 재검증 |
|---|---|---|---|---|
| 백 12 | tree | (7,10) | VCT1 SAFE, VCT2 **UNKNOWN**(10k) | selective 1M: 패배 못 찾음. 전체 10M: **UNKNOWN**(예산 소진). 미확정 |
| 백 14 | **stage4** | (10,8) | V7은 (7,11)을 골랐는데 VCT1 UNKNOWN이었다. V8-A가 VCT1 SAFE인 (10,8)로 교체했다. **VCT2 검사는 실행되지 않았다** | 전체 클래스: **깊이 2 패배**(9.3k 노드) |
| 백 16, 18, 22 | stage4 | — | 확장 후보 22~24개가 모두 VCT1 UNSAFE | — |

**P14(백 14 차례)의 합법수 212개 전부를 전체 클래스(1M)로 판정했다.**
- 209개는 깊이 0, 1개는 깊이 1, 실제로 둔 (10,8)은 깊이 2 패배다.
- **패배가 증명되지 않은 수는 (7,11) 하나뿐**이다(1M에서 UNKNOWN, 797초). 바로 V7이 원래 골랐던 수다.
- **엔진의 10k VCT2 검사를 stage4에서 돌렸다면:** (10,8)은 UNSAFE(7,779 노드, 7초)로 잡히고, (7,11)은 UNKNOWN이다. 그러면 S3 규칙상 유일한 미반박 수 (7,11)로 교체된다.

**해석.**
- 이 패배는 S3 탐지기의 오류가 아니다. 잘못된 증명은 0건이다.
- 이 패배는 **ROUTE_COVERAGE_MISS의 강한 후보**다. stage 경로에는 VCT2 검사가 없다.
- 같은 사례에서 V8-A의 규칙 문제도 드러났다. V8-A는 "VCT1 SAFE를 VCT1 UNKNOWN보다 우선"하는데, 여기서는 VCT2 패배인 SAFE 수를 미반박 UNKNOWN 수보다 앞에 두었다.
- **확정 조건:** (7,11)의 10M 판정(데스크톱).
  - VCT2_CLEAR이면 rescue 가능한 수를 놓친 것으로 확정한다.
  - PROVEN_LOSS이면 P14는 이미 진 국면이다. 원인은 그 앞(백 12, 10M에서도 UNKNOWN)으로 넘어간다.

#### 설계 방향 (S3-VCT2-v1 동결 유지)

1. **E1에 route 축을 추가한다.**
   - manifest에 `source_route`, `vct2_eligible`(tree만 true), `vct2_checked`를 기록한다.
   - 주 지표(E1-P/N/C)는 지금처럼 tree 경로만 쓴다(S3-v1의 적용 범위).
   - 별도 층 **E1-S(stage 경로)**: 같은 로그의 stage4/5 착수를 screen한다. 둔 수가 VCT2 패배이고 방어 집합(`_stage4_order`과 확장 후보)에 VCT2_CLEAR나 미반박 수가 있으면 `ROUTE_COVERAGE_MISS`로 센다. 그 빈도와 rescue 가능성이 S3를 stage 경로로 넓힐 근거가 된다.
   - **구현은 사용자 확인 뒤에 한다.** 이미 시작한 E1 실행과 파일이 섞이지 않게 별도 단계와 별도 파일로 만든다.
2. **E3 후보 표에 추가한다**(E1 결과를 보기 전에 고정):

| E1 결과 | E3 후보 |
|---|---|
| ROUTE_COVERAGE_MISS가 많음 | **S3-stage**: stage4/5 최종 수에도 같은 selective VCT2 veto. 대안은 방어 집합 안에서만. V8-A의 "VCT1 SAFE 우선"보다 VCT2 증명 패배 회피를 앞에 둔다 |
| BUDGET_AMBIGUITY가 많음 (p8, 웹 veto 유형) | 대안에만 추가 예산, 2-pass 확인 |

3. **UNKNOWN 비율(17.6%~28%) 자체를 추적한다.** E1의 DETECT_MISS와 TRUTH_UNRESOLVED가 이 구간에서 나온다. 웹 백 12도 10M에서조차 UNKNOWN이었다. 일부는 depth-2 solver로는 판정이 불가능한 국면으로 남는다.
4. **지연(L 트랙)은 분리한다.** 제한시간이 있는 웹 대국이 목표가 될 때 V8-C, V8-A, V8-B의 solver 꼬리를 다룬다. E1/E2와 섞지 않는다.

**순서:** (7,11) 10M 판정 → E1-dev(tree 주 지표, 지금 실행분) → E1-S(stage 층, 승인 후 구현) → E2 → 필요하면 E3(S3-stage 포함) → E1-holdout → H6.

**E1 이후 Track B 방향**(주기 단위 엔진 동결, 채택 gate, 새 패배 처리 순서, H6 라벨 보완)은 [track-b-post-e1.md](track-b-post-e1.md)에 정리했다.
Track B의 계열 분리(B1-Champion / B1-Matched / B2)와 비교 실험 설계는 [track-b-b1-b2.md](track-b-b1-b2.md)다.

### 12.28 E1-S와 E2 상대 구현 (2026-10-09)

**동결 유지.** `analysis/mcts_v8.py`, `V8_DEFAULTS`, `S3_VCT2_V1`은 한 줄도 바꾸지 않았다. 새 파일과 runner·비교 도구의 옵션만 추가했다.
그래서 진행 중인 E1-dev 실행과 그 파일(`runs/e1/`)에는 영향이 없다. 좌표는 0-indexed다.

#### E1-S: stage 경로 층 (`scripts/e1s_build_suite.py`, §12.27 설계 1)

- **입력과 예산.** E1-dev와 같은 로그 네 개(`h5_policy_8401/8402`, `s3_base_8411/8412`)와 같은 예산(`SCREEN_BUDGET` 50k, `TRUTH_BUDGET` 10M)을 쓴다. 출력은 `runs/e1s/`로 분리한다.
  policy는 필요 없다(stage 경로는 tree를 쓰지 않는다). 두 baseline arm에서 stage 경로의 동작은 S3-VCT2-v1과 같다(VCT2 검사는 tree에만 있다).
- **screen.** stage4/stage5 V8 착수 566개 중 V8-A가 이미 VCT1 패배로 증명한 26개를 뺀 540개가 대상이다. 둔 수를 전체 깊이 0–2 클래스로 판정한다.
- **defense.** PROVEN_LOSS 행만 쓴다. 한 전술 에피소드에서는 첫 수만 남기고, D4 중복을 없앤다(E1-dev의 `select`와 같은 규칙). 각 국면에서 V8-A가 둘 수 있었던 방어 집합을 다시 만든다.
  - `forced`: Stage 4 순서(`_stage4_order`, V7 수가 처음) 또는 Stage 5의 한 점
  - `widened`: V8-A가 넓힐 때 쓰는 V6 root 후보(`_root_candidates_v6`)
  - 국면을 재생한 결과가 로그의 stage와 다르거나, 둔 수가 방어 집합에 없으면 오류로 멈춘다.
  - 방어 집합의 모든 수를 `TRUTH_BUDGET`으로 판정한다.
- **클래스.**

| 클래스 | 뜻 |
|---|---|
| `ROUTE_COVERAGE_MISS` | 방어 집합에 VCT2_CLEAR 수가 있다. stage 경로에서 검사했다면 rescue할 수 있었다 |
| `ROUTE_COVERAGE_UNRESOLVED` | CLEAR는 없고 UNKNOWN(미반박) 수가 있다 |
| `ROUTE_NO_RESCUE` | 방어 집합 전부 PROVEN_LOSS다. 원인은 그 앞 수에 있다 |

- **반사실(S3 규칙을 stage 경로에 적용).**
  - 예산은 S3-VCT2-v1의 selective 10k다. 순서는 forced → widened이고, V8-A가 UNSAFE로 본 수와 즉시 지는 수는 건너뛴다.
  - `k4`는 둔 수를 포함해 최대 4개를 검사하고(tree 경로와 같음), `all`은 집합 전체를 검사한다.
  - 결과: `DETECT_MISS`, `RESCUED`, `KEPT`, `SWITCH_TO_LOSS`, `SWITCH_TO_UNRESOLVED`.
  - 이것은 §12.27 E3 후보(S3-stage)의 사전 측정이다. 엔진에는 넣지 않는다.
- **route 축.** manifest 행마다 `source_route`, `vct2_eligible = false`, `vct2_checked = false`를 기록한다.
  E1-dev manifest(실행 중)는 바꾸지 않는다. 그 행은 모두 tree 경로이므로 `vct2_eligible = true`, `vct2_checked = true`로 읽는다.
- **확인.** 웹 대국 `155734` P14에서 방어 집합의 forced 층은 (6,10), (9,7), (11,5)다.
  반사실(k4)은 (9,7)을 PROVEN_LOSS, (6,10)을 UNKNOWN으로 판정해 (6,10)으로 바꾼다. §12.27의 기록과 같다(`tests/test_e1s_suite.py`).

#### E2 상대: selective VCT2 공격 (`analysis/vct2_attack.py`, `--opponent v8:full+vct2atk`)

- **정의.** 둘 차례의 수 m 뒤에 상대의 **모든 합법 응수**가 깊이 1 안에서 지면(VCF, 또는 조용한 수 1개 + VCF) WIN이다.
  - 후보는 V8-B와 같다(4나 열린 3을 만드는 수).
  - 깊이 1의 공격 쪽 조용한 수는 위협 수로만 제한한다(`SelectiveSolver`). 방어 쪽 응수는 줄이지 않는다(§12.18 원칙 3). 그래서 WIN은 증명이고, "못 찾음"은 불완전할 뿐이다.
  - **증명된 WIN만 둔다.**
  - 예산은 노드 200k, VCF 호출 20k, VCF 1회 20k 노드다. V8-B처럼 후보 사이에 공정하게 나눈다(`_first_proven`).
- **연결.** `VCT2AttackAgent`는 V8 agent를 감싸기만 하고 바꾸지 않는다.
  - base(`v8:full`)가 먼저 수를 고른다.
  - 경로가 stage4·stage5·tree일 때만 깊이 2 공격을 찾고, WIN이면 그 수로 바꾼다(경로 `own_vct2`).
  - Stage 1–3, 자기 VCF, V8-B(깊이 1 공격)는 base가 고른 그대로 둔다. 그래서 base보다 약해지지 않는다.
- **확인.**
  - P93(백 차례, 증명된 백 VCT2 승리)에서 witness인 (11,12)를 찾는다(13.6초, 16k 노드). V8-B(깊이 1)는 아무 수도 찾지 못한다.
  - 찾은 수는 전체 `ThreatSolver` 깊이 1로도 UNSAFE다(`tests/test_vct2_attack.py`).
- **기록.**
  - 게임마다 `opponent_vct2_attack`: 상대 수마다 base 경로, 결과, 노드, 시간
  - 요약 `opponent_vct2_attack`: 실행 수, WIN 수, base 경로별 WIN 수, 예산 소진, 시간
  - 요약 `v8_played_vct2_lost`: 우리 VCT2 검사가 패배를 증명했는데도 둔 수(탈출 수 없음) 다음에 상대가 `own_vct2`로 응징했는지
  - 상대 key가 `...@v8:full+vct2atk/...`라 기존 JSONL과 섞이지 않는다.
- **비교.** `scripts/s3_compare.py --opponent v8:full+vct2atk`. E2에는 채택 gate가 없다(S3-VCT2-v1은 이미 채택). 실행 전에 해석을 고정한다.

| reading | 조건 |
|---|---|
| `EFFICACY` | 안전 위반 0, 쌍 점수 차(vct2 − baseline) 95% 하한 > 0 |
| `NOT_ESTABLISHED` | 안전 위반 0, 구간이 0을 포함하고 차 ≥ −0.05 |
| `HARM` | 차 < −0.05 또는 상한 < 0 |
| `SAFETY_VIOLATION` | 어느 arm이든 안전 위반 |

  - 비용 비율(end-to-end, 상대 대비 시간 비율)과 `punishment`(상대 `own_vct2` 횟수, 그 판의 패배 수)를 함께 보고한다.
  - 상대가 깊이 2 공격을 계산하므로 상대 착수 시간이 늘어난다. 그래서 "상대 대비 시간 비율"은 S3 결과와 비교하지 않고, 두 arm 사이에서만 비교한다.

#### E2 실행 프로토콜 (실행 전 고정)

- arm: `puct_policy`, `puct_policy_vct2`. `--puct-c 1.5`, H3 checkpoint(SHA-256이 S3와 같아야 함)
- 상대: `v8:full+vct2atk`. seed 8413, 8414, 각 25쌍(arm당 100판)
- 실행은 한 번에 하나씩 순서대로 한다(시간 비율 오염 방지): 8413 baseline → 8413 vct2 → 8414 baseline → 8414 vct2
- 해석은 위 표대로 한다. E1-dev·E1-S 결과와 함께 E3 필요 여부를 정한다(§12.26 표, §12.27 표).

### 12.29 P14 (7,11) 10M 결과, 웹 1패 해석 정정, E2 운영 규칙 (2026-10-09)

#### P14 (7,11) 결과 (데스크톱, `a85e311`)

| 검사 | 결과 | 비용 |
|---|---|---|
| selective(1M) | `NO_TARGETED_VCT2_FOUND` | 653초 |
| 전체 클래스 10M | `UNKNOWN`(10,000,000 노드 소진) | 7,219초 |

- 결과는 `docs/mcts-v8-results/web_stress_20261009_recheck.json`(`key_moves`, `p14_root_truth`)에 넣었다.
- 두 결과 모두 안전하다는 뜻이 아니다. (7,11)은 **VCT2_CLEAR도 PROVEN_LOSS도 아닌 UNKNOWN**이다.
- **P14 root 참값은 `UNRESOLVED`다.** 합법수 212개 중 211개는 깊이 0–2 패배가 증명됐고 (7,11) 하나만 UNKNOWN이다. P92(129 패배, 4 UNKNOWN)와 같은 구조다. 거의 모든 수가 진다고 해서 국면이 패배로 증명된 것은 아니다.
- **더 깊게 계산하지 않는다.** 10M에 2시간이 걸렸고, 20M·50M으로 올려 한 국면을 푸는 것보다 E1-S로 같은 유형이 얼마나 자주 나오는지 재는 편이 E3 판단에 중요하다. P14는 위 상태로 봉인한다.

#### 웹 1패(`155734`) 해석 정정 (§12.27의 "확정 조건"에 대한 결론)

| 구분 | 내용 |
|---|---|
| 확정 | 실제로 둔 (10,8)은 전체 클래스 깊이 2 패배다 |
| 확정 | 그 수는 stage4 경로라 S3-VCT2 검사가 실행되지 않았다 |
| 확정 | stage4에 같은 10k 검사를 적용하면 (10,8)은 PROVEN_LOSS, (7,11)은 UNKNOWN이라 S3 규칙상 (7,11)로 **바뀐다**(재현, §12.28) |
| 확정 | V8-A의 "VCT1 SAFE를 VCT1 UNKNOWN보다 우선" 규칙이 고른 SAFE 수가 깊이 2에서 증명된 패배였다 |
| 미확정 | (7,11)이 실제로 버티는 수인지. 그래서 바뀐 수가 대국을 살렸는지 |

- 그래서 이 사례는 **`ROUTE_COVERAGE_MISS` 확정이 아니다.** E1-S 분류로는 `ROUTE_COVERAGE_UNRESOLVED`(CLEAR 없음, UNKNOWN 있음)에 해당하고, 반사실 결과는 `SWITCH_TO_UNRESOLVED`다.
- 웹 대국은 E1-S 입력이 아니므로 성적에는 넣지 않는다. 확정되지 않은 사례를 결과에 맞춰 MISS로 올리지 않는다.
- stage 경로 검사가 필요하다는 근거는 여전히 남는다. VCT1 기준 선택이 깊이 2 패배를 골랐다는 사실은 확정이다. 빈도는 E1-S가 잰다.

#### E2 운영 규칙 (실행 전 고정)

- **공격 예산은 200k를 유지한다**(§12.28 고정값). E2 상대는 배포용이 아니라 **stress opponent**다. 목표는 속도가 아니라 VCT2 공격 recall이다.
  - P93은 16k 노드면 충분했지만 대표 probe 하나일 뿐이다. smoke 1판에서도 공격 검사 28회 중 3회가 200k를 소진했다. 100k로 낮추면 실제 강제승을 UNKNOWN으로 놓치는 경우가 늘어 E2의 목적과 반대가 된다.
  - 100k는 공식 실험에 쓰지 않는다. 필요하면 나중에 별도 비용 ablation으로 잰다.
  - smoke 비용(4코어 컨테이너, 전체 테스트와 동시 실행): 공격 검사 중앙값 3초, 최대 167초, 한 판 합계 960초. 이 실행 시간은 배포 목표가 아니다. 지연은 L 트랙에서 다룬다.
- **mechanism exposure**(승률 gate가 아니라 "이번 E2가 기능을 시험할 기회를 충분히 만들었나"를 보는 진단):
  - 지표는 두 arm 합계의 상대 `own_vct2` 착수 수(증명된 깊이 2 공격 개입)다.
  - **10회 미만이면 `low_power`로 표시**한다. 그때 "점수 차 없음"은 효능이 없다는 근거가 되지 않는다. 20회 이상이면 더 좋다.
  - low_power면 **E2b**(`--opponent v8:full+vct2atk400k`, 같은 seed 8413/8414, 같은 arm)를 별도로 돌릴 수 있다. key가 달라 E2 파일과 섞이지 않는다. **E2와 E2b는 합산하지 않는다.**
  - 400k 이상은 E2 결과를 본 뒤 이 조건일 때만 쓴다.
- **시간은 쪽별로 보고한다.**
  - 상대가 느려서 전체 대국 시간 비율은 S3 overhead를 희석한다. 그래서 E2의 game-time ratio로 "비용 증가가 거의 없다"고 결론 내리지 않는다.
  - `s3_compare --opponent ...`는 시험 엔진의 착수 시간(`cost.move_seconds`, end-to-end 비율)과 상대의 착수·공격 시간(`e2.opponent_time`)을 따로 낸다.
  - S3의 비용 gate는 기존 S3 벤치마크 결과(§12.25)를 유지한다. E2에서 주로 보는 것은 효능과 응징 여부다.

**순서:** P14 봉인(완료) → E1-dev(실행 중) → E1-S → E2(200k, 200판) → exposure 확인(10회 이상이면 그대로 해석, 미만이면 low_power 표시 후 필요할 때 E2b 400k) → E1 + E1-S + E2 종합 → E3 여부.

### 12.30 E1-dev 결과와 dev 진단 (2026-10-10)

결과 파일은 `docs/mcts-v8-results/e1/`에 있다.
- `e1_manifest.json`, `e1_eval.json`: 데스크톱, `a85e311`, git_dirty = False, H3 checkpoint SHA-256이 S3와 같다.
- `screen_summary.json`: 11.4 MB인 screen 원본의 요약과 SHA-256이다. manifest에 기록된 screen 해시와 일치한다.
- `e1_dev_budget_diagnostic.json`: 아래 dev 진단 결과다.

#### 결과 (run = 국면 × seed 3)

| 항목 | 값 |
|---|---|
| screen 3,991개 | VCT2_CLEAR 1,763 / UNKNOWN 2,222(55.7%) / PROVEN_LOSS 6 |
| suite | E1-P 5개(같은 에피소드 1개 제외), E1-N 0, E1-UNRESOLVED 0, E1-C 40 |
| E1-P 15 run | TREE_AVOIDED 9, RESCUED 2, DETECT_MISS 3, TRUTH_UNRESOLVED 1 |
| E1-C 120 run | NO_VETO 120 |
| 지표 | rescue_rate 11/15, veto_recall 2/5, pool coverage 15/15, K4 coverage 14/15, conditional escape 2/2, unknown replacement 0/2, false veto 0, unsound witness 0, inconsistent 0 |
| 비용 | VCT2 검사 p50 0.04초 / p95 6.2초 / 최대 9.7초. 착수 p50 2.2초 / p95 8.4초 / 최대 16.5초 |

#### 읽는 법

- **rescue_rate 11/15는 탐지기의 성과가 아니다.**
  - 9 run은 tree가 처음부터 CLEAR 수를 골랐다(TREE_AVOIDED).
  - 1 run은 tree가 참값이 UNKNOWN인 수를 골랐다(TRUTH_UNRESOLVED, 8401-p4 (3,5)).
  - VCT2 검사가 실제로 패배수를 바꾼 것은 2 run이다.
- **run은 독립이 아니다.**
  - tree가 증명된 패배수를 고른 5 run은 국면 3개에서 나왔다.
  - 국면 단위로 보면 탐지 성공 1개(8412-p4, 2 run), 실패 2개(8401-p16 1 run, 8411-p8 2 run)다.
  - veto_recall 2/5의 Wilson 구간 [0.12, 0.77]도 run을 독립으로 본 값이라 실제보다 좁다.
- **E1-C는 판별력 시험이 아니라 구현 건전성 확인이다.**
  - control 120 run의 첫 검사는 모두 `SAFE`(NO_TARGETED)로 바로 끝났다. 50k screen에서 CLEAR로 끝난 쉬운 국면들이기 때문이다.
  - selective PROVEN_LOSS는 구성상 증명이다(§12.18 원칙 3). 그래서 false veto 0은 "구현에 버그가 없다"는 확인이다. 어려운 국면에서의 오탐률 추정이 아니다.
  - 국면 단위로 0/40이면 95% 상한은 8.8%다. run 단위 0/120이면 3.1%다.
- **K4 coverage 14/15에서 빠진 1 run이 바로 패배수를 고른 run이다**(8411-p8 run 1). 패배수를 고른 5 run만 보면 K4 coverage는 4/5다.
- **suite가 한쪽에 몰려 있다.**
  - screen의 PROVEN_LOSS 6개와 E1-P 5개는 **모두 백 11–13수 개국 국면**이다.
  - 그중 3개(8411-p8, 8411-p22, 8412-p4)는 이미 §12.25에서 본 S3 veto 국면이다.
  - 50k screen이 중후반 패배를 거의 증명하지 못하기 때문이다(UNKNOWN 55.7%).
  - 그래서 E1-dev 결과는 개국 직후 백의 방어 문제에 대한 메커니즘 진단이다. 실전 VCT2 실패율 추정으로 쓰지 않는다.

#### dev 진단: DETECT_MISS는 예산 문제인가 (`e1_dev_budget_diagnostic.json`)

**selective 검사 하나의 예산을 바꿔 가며 판정한 결과**(node_limit 20k, call_limit 20k)

| run | 10k | 20k | 50k 이상 | 증명에 든 selective 노드 |
|---|---|---|---|---:|
| 8401-p16 (8,11) | UNKNOWN | UNKNOWN | PROVEN_LOSS | 39,378 |
| 8411-p8 (7,8) | UNKNOWN | PROVEN_LOSS | PROVEN_LOSS | 11,320 |
| 8411-p8 (5,7) | UNKNOWN | PROVEN_LOSS | PROVEN_LOSS | 10,057 |
| 8412-p4 (5,7) / (5,6) | PROVEN_LOSS | 같음 | 같음 | 7,770 / 5,574 |

- 세 DETECT_MISS는 모두 **selective 클래스 안의 패배**다. 공격 쪽 가지치기 때문에 원리적으로 못 찾는 경우가 아니라, 10k 안에 끝나지 않은 예산 문제다. 두 건은 10k를 아주 조금 넘었다.
- 전체 클래스 참값의 노드 수(11,606 / 24,258 / 188,881)는 selective 노드 수와 다르다. 비교 기준으로 쓰지 않는다.

**S3 교체 규칙을 패배수 5 run에 다시 적용한 결과**(기록된 root 자식 순서 사용. V8-C UNSAFE 건너뛰기는 eval 로그에 없어서 생략했다)

| 예산 | RESCUED | 패배→패배 교체 | DETECT_MISS |
|---|---|---|---|
| 10k(실제) | 2 | 0 | 3 |
| 20k | 3 | 1 | 1 |
| 50k | 4 | 1 | 0 |

- 20k와 50k 모두에서 8411-p8 run 1은 (7,8)을 잡은 뒤 다음 자식 (6,6)을 판정하지 못해(UNKNOWN) 그 수로 바꾼다. (6,6)은 참값이 PROVEN_LOSS다.
  - 이 run에서 유일한 CLEAR 수 (8,8)은 K4 밖이다.
  - 그래서 §12.25 8411 p8과 같은 유형(UNKNOWN 대안으로 교체했는데 그 수도 진다)이 다시 나온다.
- 20k에서 8411-p8 run 2의 rescue는 UNKNOWN으로 판정된 대안 (8,8)이 우연히 CLEAR였기 때문이다. 50k에서는 (8,8)이 SAFE(NO_TARGETED)로 판정된다.
- **해석.**
  - 탐지 예산을 늘리면 detection recall은 오른다.
  - 그러면 그다음 병목인 교체 단계(UNKNOWN 대안 허용, K4)가 드러난다.
  - 예산을 늘리는 arm은 recall만으로 판정하지 않는다. `unknown_replacement_rate`, 패배→패배 교체, 비용을 함께 판정해야 한다.

#### E3에 주는 근거 (결정은 E1-S·E2 뒤에 한다)

- **1순위 후보는 E3-A: 탐지 예산 증가, K4 유지.**
  - 20k는 dev의 놓친 3건 중 2건, 50k는 3건을 잡는다.
  - 예산 값은 E1-S·E2를 본 뒤 holdout 판정 **전에** 고정한다. dev에서 3/3을 맞춘 50k를 골라 dev 성능으로 주장하지 않는다(§12.26 dev/holdout 규칙).
- **비용.**
  - UNKNOWN으로 끝나는 검사는 예산을 전부 쓴다. tree 수의 17.6–28%가 10k에서 UNKNOWN이었다(§12.27).
  - 그래서 예산을 늘리면 그만큼의 착수에서 VCT2 비용이 거의 비례해서 는다. dev 재적용에서도 20k는 교체 포함 최대 28초, 50k는 51초였다.
  - E3-A는 S3와 같은 end-to-end 비용 gate(시간 비율 ≤ 1.5)를 다시 통과해야 한다.
- **E3-B(K6)와 교체 규칙**(검사한 대안이 모두 UNKNOWN일 때 무엇을 둘지)은 E3-A와 따로 측정한다. dev 재적용상 E3-A 다음 병목은 이쪽이다.
- E1-holdout은 같은 규칙으로 캐면 다시 개국 국면에 몰린다. 그래서 holdout 보고에는 국면의 수순 위치 분포를 같이 낸다.

**순서:** E1-dev(완료) → E1-S → E2(200k) → 종합 → E3(필요하면 E3-A부터) → E1-holdout.
