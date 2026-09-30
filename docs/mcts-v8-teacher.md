# MCTS-v8 Teacher 설계서 (초안, 검토 반영)

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

¹ 기존 `check_v8_branch_points.py`는 `_forced_v5_move` 뒤 root 후보를 만들 때 새 `_RootContext`를 생성해
Stage-5의 `context.injected`를 버리는 재현 오차가 있었다. 실제 V6/V7은 같은 context를 이어 쓴다. 스크립트는 이를 고쳤고
`--structure-only`로 실제 root 후보 구조를 먼저 재확인한다. 후보 목록이 기존 20개와 같으면 1,452초 분류 결과를 그대로 쓸 수
있고, 다르면 `--root-vct`를 다시 실행한다. (10,8) 자체의 VCT1-UNSAFE 증명은 probe fixture에 독립적으로 남아 있다.

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
4. **`164723`은 MCTS 경로다.** 이 판만은 stage 4 수정으로 고쳐지지 않는다. v7 수 (10,8)이 VCT1-UNSAFE라는 사실은 독립
   probe 증명으로 확정이다. 다만 “V6 root 20개 중 12 SAFE / 8 UNSAFE, 순서 1위 (6,7)이 SAFE”는 위 context 재현 버그를
   고친 뒤 root 구조가 같음을 확인해야 실제 V7 root에 대한 사실로 확정할 수 있다. 따라서 V8-C의 직접 게이트는 우선
   **(10,8)을 제거하고 VCT1-SAFE 수를 고르는지**로 둔다.

### 2.3 VCT1 비용

- stage 4 방어 4개: 10~48초(VCF 호출 881~2,012회). 사람 패배 3판 모두 모든 후보를 끝까지 판정했다(UNKNOWN 0).
- `164723`에서 기존 스크립트가 만든 root 후보 20개 전부: **1,452초(약 24분).** 한 후보 평균 73초다. 정확한 V7 root
  후보 목록과 동일한지는 §2.2의 context 수정 후 구조 재검증 대상이다. 검토서 §6.1에서도 "VCT가 없음을 증명"하는 쪽이
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
- 검사는 V5 key 순이 아니라 **기존 V7 stage 4 정렬 순**으로 하고 첫 SAFE에서 멈춘다. 첫 후보가 SAFE면 추가 비용은 한 번뿐이다.
- stage 4 집합 전체가 UNSAFE이면 **root 후보 전체로 넓혀** VCT1 검사를 계속한다(§2.2에서 `163810`·`163903`은 stage 4 밖에도
  SAFE가 하나씩 있었다: (4,7), (11,8)). 예산을 다 쓰면 원래 V7 선택으로 돌아간다.
- stage 5도 같다. 단일 쌍위협 차단점이 VCT1-UNSAFE면 root 후보로 넓힌다.
- 게이트: `vct_probes_v1`의 `must_defend_vct` 24개(D4 포함)에서 V8의 수가 `correct_moves` 안, `avoid_moves` 밖.

### 4.2 V8-B — 자기 VCT1 공격 (2순위)

- 사람 패배 4판에서 사람의 결정타는 모두 VCT1 첫 수였다(`vct_attack` probe). V8이 같은 수를 스스로 찾으면 teacher의
  공격 label(`vct_attack`)을 V8 대국에서 직접 얻을 수 있다.
- 후보 생성은 우선 "4 또는 열린 3을 만드는 수"로 좁히는 **속도 휴리스틱**을 쓴다. 선택된 후보의 증명은 상대의 모든 응수가
  VCF로 지는지 실제로 검사한다. 이는 `ThreatSolver` depth 1의 **후속 판정 의미**와 같지만, depth 1 solver가 모든 합법
  quiet move를 열거하는 것과 달리 후보 열거 자체는 불완전할 수 있다. UNKNOWN이면 공격하지 않는다(건전성 우선).
- 예산은 V8-A/C와 별도로 둔다. 실패하는 후보는 살아남는 응수 하나를 찾으면 끝나지만, 성공 증명은 상대의 모든 응수를 검사해야 한다.
  비용은 V8-4에서 잰다.
- 게이트: `vct_attack` 32개에서 V8 첫 수가 증명된 승리 위협 집합 안.

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
  브랜치 `feat/mcts-v8-teacher`.

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
  먼저 수정된 `check_v8_branch_points.py --structure-only`로 실제 V7 context를 보존한 root 후보가 기존 측정 목록과 같은지
  확인한다. 같으면 기존 12 SAFE / 8 UNSAFE 결과를 재사용하고, 다르면 그 국면만 `--root-vct`를 다시 측정한다.

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
| V8-1 | 기존 격리 규칙 유지, `analysis/mcts_v8.py` 골격(V7 위임), root record | V7과 착수 동일한 V8(모듈 모두 끔) | — |
| V8-2 | V8-A stage 4/5 VCT1 | `must_defend_vct` 24/24 | V8-1 |
| V8-3 | V8-B 자기 VCT1 | `vct_attack` 게이트 | V8-1 |
| V8-4 | 비용 측정 → V8-C 범위 결정 → V8-C | 착수 시간 표, `164723` 게이트 | V8-2 |
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
