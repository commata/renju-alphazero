# 3단계 순수 MCTS

이 단계는 신경망 없이 **UCT 기반 Monte Carlo Tree Search**의 구조와 렌주 엔진 연동을 검증한다. AlphaZero의 PUCT, 정책망, 가치망, 자기대국 학습은 아직 포함하지 않는다.

## V2: 현재 검증 기준본

2차 수정본은 다음 설정을 고정한다.

- simulations: **10**
- candidate limit: **8**
- 지역성 점수 기반 shortlist
- 루트 즉시 승리 / 단일 강제 방어
- 전술 rollout
- private state + undo

seed 42 기준 5판씩 측정에서는 Random/Tactical 양쪽 색 네 조건 모두 MCTS-v2가 5승 0패였다. 표본과 seed가 제한되어 있으므로 일반적인 기력 우위를 확정하는 결과는 아니다.

## V3: 25 simulations / 후보 16개

3차 수정본은 V2를 덮어쓰지 않고 별도 `MCTSV3Agent`로 유지한다.

기본 설정:

- simulations: **25**
- candidate pool: **16**
- progressive widening initial width: **6**
- neighborhood radius: **2**
- priority top-k: **5**

### 왜 25 / 16 / 6인가

V2의 10회 탐색과 후보 8개는 작은 CPU 예산에서 UCT 재탐색을 만들기 위한 설정이었다. V3에서는 탐색 예산을 25회로 2.5배 늘렸으므로 후보 다양성도 함께 늘린다.

단순히 후보를 16개로 늘려 처음부터 모두 확장하면 25회 중 상당수가 첫 방문에 소비된다. 따라서 V3는 progressive widening을 유지한다.

허용 자식 수:

```text
initial_width + floor(sqrt(node visits))
```

기본 initial width 6일 때:

```text
visits 0  -> 최대 6개
visits 9  -> 최대 9개
visits 25 -> 최대 11개
```

즉 후보 pool은 최대 16개를 보유하지만 25번 안에 전부 얕게 훑는 대신, 약 6~11개를 단계적으로 노출하면서 나머지 simulation을 UCT 재탐색에 사용할 수 있다.

### 후보 생성 최적화

V3는 전체 225칸을 매번 점수화하지 않는다.

```text
기존 돌 주변 Chebyshev 거리 2 빈칸
→ local pool만 move_score
→ 합법성 확인
→ 최대 16개 후보
```

빈 보드에서는 중앙 5×5 영역에서 시작한다. local pool이 부족하거나 흑 금수 때문에 후보가 부족할 때만 전체 `Game.legal_moves()`로 fallback한다.

루트의 즉시 승리와 단일 즉시 패배 방어는 전체 합법수 기준으로 유지한다.

### V3.1 우선순위 기반 확장 정책

25 simulations / 16 candidates V3를 V2와 10판 직접 대결했을 때 V2 6승, V3 4승이 나왔다. 표본이 작아 우열을 단정할 수는 없지만, 기존 V3가 정렬된 16개 후보를 만들어 놓고도 expansion에서 전체 미확장 후보 중 균등 랜덤으로 하나를 뽑는 문제가 확인됐다.

V3.1은 후보 pool과 progressive widening은 그대로 두고 새 후보를 열 때 정책만 바꾼다.

```text
정렬된 후보 최대 16개
→ 아직 보지 않은 후보 중 상위 5개만 선택 창으로 사용
→ 순위 가중치 5, 4, 3, 2, 1
→ 가중 랜덤으로 1개 확장
```

즉 1위 후보를 항상 강제하지는 않지만 6~16위 후보가 초반에 무작위로 먼저 확장되는 일도 막는다. 전술 수가 없는 rollout의 랜덤 수 역시 같은 상위 5개 순위 가중 정책을 사용한다.

## V2 vs V3 비교

두 revision은 의도적으로 다른 기본 탐색 예산을 유지한다.

- V2: 10 simulations / 8 candidates
- V3: 25 simulations / 16 candidates / initial width 6

실행:

```bash
python scripts/run_mcts_versions.py --games 5 --v2-simulations 10 --v3-simulations 25 --v2-candidate-limit 8 --v3-candidate-limit 16 --v3-initial-width 6 --v3-radius 2 --v3-priority-top-k 5 --seed 42
```

이 비교는 "동일 계산량에서 알고리즘만 비교"하는 실험이 아니라, **검증된 V2 기준본과 더 큰 탐색 예산을 사용하는 V3 후보 모델을 비교하는 승급 테스트**다.

알고리즘 자체만 공정하게 비교하고 싶다면 두 revision의 simulation 수를 같게 지정해 별도 실험할 수 있다.

예:

```bash
python scripts/run_mcts_versions.py --games 5 --v2-simulations 25 --v3-simulations 25 --v2-candidate-limit 8 --v3-candidate-limit 16 --v3-initial-width 6 --v3-radius 2 --v3-priority-top-k 5 --seed 42
```

## 검증

```bash
python -m unittest discover -s tests -v
```

V3의 다음 판단 기준은 V2 직접 대결 결과, 판당 실행시간, 평균 수, 그리고 후보 생성/rollout profile이다.


## V3.1 검증 결과

V3.1의 top-5 순위 가중 정책을 V2 기준본과 50판씩 두 seed에서 검증했다.

| Seed | V2 승 | V3.1 승 | 무승부 |
| ---: | ---: | ---: | ---: |
| 42 | 18 | 32 | 0 |
| 123 | 9 | 41 | 0 |
| 합계 | 27 | 73 | 0 |

두 seed 합계 100판에서 V3.1이 73승, V2가 27승이었다. 이 결과를 바탕으로 V3.1의 25 simulations / 16 candidates / top-5 weighted expansion은 유지하고, 다음 실험은 후보 순위를 더 정교하게 만드는 V3.2로 분리한다.

## V3.2 색상별 렌주 전술 우선순위

V3.2는 V3.1을 덮어쓰지 않는다. `MCTSV3Agent`는 V3.1 기준본으로 유지하고 `MCTSV32Agent`를 별도로 추가한다.

탐색 파라미터는 동일하다.

```text
simulations = 25
candidate_limit = 16
initial_width = 6
neighborhood_radius = 2
priority_top_k = 5
```

변경점은 **후보 16개를 정렬하는 정책**이다. 먼저 V3.1의 저비용 locality로 최대 32개를 prefilter한 뒤, 그 bounded set에만 4/3 패턴 분석을 적용하고 상위 16개를 선택한다. rollout은 속도 폭증을 막기 위해 상세 33/44/43 탐지 대신 색상별 locality 가중치만 사용한다.

### 흑: 공격 성향

흑은 규칙 엔진에서 금수가 먼저 제거된다. 따라서 33, 44, 장목은 점수 대상이 아니라 후보에서 제외된다.

우선순위는 다음과 같다.

```text
1. 정확히 5목            -> 즉시 선택
2. 백 즉시 승리 방어     -> 즉시 방어
3. 43 생성               -> +1800
4. 합법적인 4 생성       -> +1300
5. 백의 4 생성점 차단    -> +1100
6. 열린 3 생성           -> +600
7. 일반 공격 연결        -> attack locality × 14
8. 일반 방어 연결        -> defense locality × 9
```

43은 한 방향의 4 위협과 다른 방향의 열린 3 위협이 동시에 존재하는 경우로 탐지한다.

### 백: 방어 성향 + 금수 없는 공격 활용

백은 33, 44, 5목 이상이 모두 합법이므로 렌주 비대칭을 후보 점수에 직접 반영한다.

```text
1. 5목 이상              -> 즉시 선택
   - 여러 승리 수가 있으면 더 긴 run(6목 이상 포함)을 tie-break로 우선
2. 흑 즉시 승리 방어     -> 즉시 방어
3. 백 44                 -> +1800
4. 흑 43 생성점 차단     -> +1500
5. 흑 4 생성점 차단      -> +1400
6. 백 단일 4             -> +1200
7. 백 33                 -> +900
8. 흑 열린 3 생성점 차단 -> +700
9. 백 열린 3             -> +500
10. 일반 방어 연결       -> defense locality × 14
11. 일반 공격 연결       -> attack locality × 10
```

백 6목 이상은 별도의 승패 규칙이 아니라 기존 엔진과 동일하게 `>=5` 즉시 승리다. 다만 여러 즉시 승리 수가 동시에 있을 경우 V3.2는 더 긴 연속수를 tie-break로 선호한다.

### V3.1 vs V3.2 비교

```bash
python scripts/run_mcts_v3_policies.py --games 25 --simulations 25 --candidate-limit 16 --initial-width 6 --radius 2 --priority-top-k 5 --seed 42
```

이 명령은 V3.1 흑 vs V3.2 백 25판, V3.2 흑 vs V3.1 백 25판으로 총 50판을 실행한다.


## 경기 로그 자동 저장

MCTS 비교 runner는 실행할 때 경기 로그를 자동으로 저장한다.

기본 경로:

```text
logs/mcts_v3_policies/<timestamp>_seedN/
logs/mcts_versions/<timestamp>_seedN/
```

생성 파일:

```text
games.csv   경기별 승자, 에이전트, 수 수, 경기 시간
moves.csv   모든 착수를 한 행씩 기록
games.json  실행 설정 + matchup 요약 + 전체 착수 기록
```

CSV는 Excel에서 한글이 깨지지 않도록 UTF-8 BOM으로 저장한다. `moves.csv`와 JSON에는 내부 0-based 좌표(`row0`, `col0`)와 사람이 보는 1-based 좌표(`row`, `col`)를 모두 기록한다.

원하는 경로를 직접 지정할 수도 있다.

```bash
python scripts/run_mcts_v3_policies.py --games 25 --seed 42 --log-dir logs/my_experiment
```

첫 번째 색 배치가 끝난 직후 한 번 checkpoint 저장하고, 두 번째 색 배치가 끝나면 두 matchup을 합쳐 다시 저장한다.

## V3.2 50판 결과와 성능 메모

사용자 로컬 V3.1 vs V3.2 50판 결과:

```text
V3.1 wins: 16
V3.2 wins: 34
Draws: 0
```

확인 가능한 두 번째 색 배치(V3.2 흑 vs V3.1 백)에서는 V3.2가 19승, V3.1이 6승이었고 평균 13.80수, 25판 실행 시간이 1203.907초였다.

V3.2가 오래 걸리는 주된 원인은 현재 코드 구조상 상세 전술 패턴 평가 비용으로 추정된다.

- 후보 한 수의 `_v32_priority_score()`가 자기/상대 양쪽에 대해 `_pattern_features_for_move()`를 호출한다.
- 각 pattern feature는 4방향을 검사하면서 여러 extension 후보를 다시 시험한다.
- 열린 3 검사는 extension을 놓아본 뒤 winning extension을 다시 찾기 때문에 중첩된 후보 검사가 발생한다.
- 흑 합법성 확인은 `forbidden_reason()`을 호출하며, 이 함수 자체가 재귀적 삼삼/사사 판정을 포함한다.
- V3.2는 최대 32개를 prefilter한 뒤 상세 점수화를 하고, MCTS expansion 때 생성되는 child 후보에서도 이 과정을 반복한다.
- 따라서 25 simulations × 여러 실제 착수 × 최대 32개 상세 후보 분석이 누적되어 V3.1보다 큰 CPU 비용이 발생한다.

이 평가는 코드 경로를 기준으로 한 병목 추정이며, 정확한 함수별 비중은 별도의 cProfile로 확인해야 한다. 현재 로그 기능은 판별 실행 시간까지 남기므로 느린 경기와 긴 경기의 상관관계를 먼저 확인할 수 있다.


## V3.2.1 성능 최적화

cProfile 2경기에서 V3.2의 병목이 전술 패턴 평가 중 흑 금수 판정 재귀로 확인됐다.

```text
전체: 326.928 s
forbidden_reason: 1,610,077회 / 299.878 s 누적
_forbidden_after_black_move: 약 197만회 / 298.623 s
_pattern_features_for_move: 28,420회 / 268.153 s
_search_candidates_v32: 425회 / 259.587 s
_open_three: 약 777만회 / 226.442 s
_straight_four_after_extension: 47,281,480회 / 178.082 s
```

V3.2.1은 V3.2의 점수와 탐색 파라미터를 유지하되, **후보 우선순위용 가상 extension에서는 재귀 `forbidden_reason()` 호출을 제거**한다.

핵심 원칙:

```text
실제 후보 수의 합법성
→ 기존 Game/rules 엔진으로 정확히 검사

후보의 43/44/33 휴리스틱 평가
→ 주변 방향의 구조만 빠르게 검사
→ 가상 비승리 extension에서는 재귀 33/44 판정 생략
→ 흑 장목은 run_length로 즉시 거부

흑 상대 위협의 top-level 후보
→ forbidden_reason을 후보당 최대 한 번만 호출
```

흑의 exact-five winning extension은 규칙 엔진과 동일하게 다른 방향의 장목 여부만 확인한다. exact five가 성립하면 삼삼/사사보다 승리가 우선되는 기존 규칙 순서를 그대로 이용한다.

V3.2.1은 별도 `MCTSV321Agent`로 보존한다. 따라서 느린 V3.2와 직접 비교하여 속도와 기력 회귀를 따로 측정할 수 있다.

직접 비교:

```bash
python scripts/run_mcts_v32_optimization.py --games 1 --simulations 25 --candidate-limit 16 --initial-width 6 --radius 2 --priority-top-k 5 --seed 42
```

프로파일:

```bash
python -m cProfile -o v321_profile.prof scripts/run_mcts_v32_optimization.py --games 1 --simulations 25 --candidate-limit 16 --initial-width 6 --radius 2 --priority-top-k 5 --seed 42
python -m pstats v321_profile.prof
```

pstats에서:

```text
sort cumulative
stats 30
```

V3.2.1의 첫 성능 목표는 `forbidden_reason`과 `_straight_four_after_extension` 호출 수를 V3.2 대비 크게 줄이는 것이다. 상세 전술 스캐너는 규칙 판정기가 아니라 후보 정렬 휴리스틱이므로, 실제 착수 합법성은 계속 `Game.play()`와 기존 rules 엔진이 최종 보장한다.

## V4 기준 정책과 V5

V4.1/V4.2는 기존 구현 그대로 유지한다. V4는 내 즉시 승리, 상대 즉시 승리 차단,
상대 열린4 생성점 차단 순으로 강제수를 선택한 뒤 V3.2.1을 호출한다.
V4.1은 50 simulations / 후보 20 / 초기 폭 8 / top-k 8,
V4.2는 50 / 24 / 10 / 10이다. 두 버전 모두 radius 2, exploration sqrt(2)다.

### V5의 목적과 unstoppable four

V4의 열린4 방어만으로는 백의 사사, 흑 금수점에 완성점이 있는 백 4,
한 수로 생성되는 복수 위협을 충분히 예방하지 못한다. V5는 이 위협을 먼저
만들거나 상대가 만들기 전에 방어하는 별도 root 정책이다.
`src/renju/` 및 V2~V4.2의 정책과 rollout은 수정하지 않는다.

수 하나를 놓은 뒤 다음 조건을 모두 만족하면 unstoppable four로 정의한다.

1. 상대의 합법적인 즉시 승리가 없다.
2. 방금 놓은 돌을 포함하는 라인에 다음 수 승리 완성점 S가 있다.
3. 상대가 S의 합법적인 칸을 하나씩 막아 보아도 내 승리 완성점이 남는다.
   S 전부가 상대 흑에게 금수여도 해당한다.

흑은 정확히 5목이며, 다른 방향에 장목을 만드는 완성점은 제외한다.
백은 5목 이상이다. 생성수와 흑 방어수의 합법성은 기존 규칙 엔진으로 확인한다.
임시 board 변경은 `try/finally`로 복구한다. 후보는 4방향의 길이 5 window 중
내 돌 3개 이상, 상대 돌 0개인 window의 빈칸으로 한정한다.

### V5 강제 정책: Stage 1~6

1. **Own immediate win**: 내 즉시 승리. 백 승리점 동률은 긴 run 우선.
2. **Immediate block**: 상대 승리점 중 내게 합법인 수를 방어한다.
   모두 흑 금수이면 불법수를 반환하지 않고 다음 단계로 진행한다.
3. **Own unstoppable four**: 내 unstoppable creator를 선택한다.
4. **Prevent opponent unstoppable four**: 상대 creator와 관련 5칸 window의
   합법 빈칸을 방어 후보로 검사한다. 착수 후 상대 creator가 가장 적게 남는 수를 택한다.
5. **Double-threat prevention**: 빠른 패턴 검사로 열린3 2방향 이상, 4 2방향 이상,
   또는 43을 만드는 상대 수를 찾는다. 흑 creator는 실제 합법이어야 한다.
   정확히 1개이고 내게 합법이면 점유한다. 2개 이상이면 합법적인 모든 creator를
   MCTS root 후보 앞에 주입하고 candidate limit 초과를 허용한다.
6. **MCTS**: V3.2.1의 progressive widening, top-k 순위 가중 확장,
   종국까지 rollout, 최대 visits → mean value 동률 해소를 유지한다.

전술 동률은 `_v321_move_key`를 사용한다. Stage 5는 후보 순위용 빠른 구조 휴리스틱을
사용하므로 가상 열린3 extension까지 재귀 규칙 검사를 하는 증명 탐색은 아니다.
실제 흑 착수의 합법성은 정확히 확인한다.

### Adaptive simulations와 프리셋

root 후보 정렬 때 계산한 전술 점수를 재사용한다. 최댓값이 threshold 이상이면
`tactical_simulations`, 아니면 `simulations`를 적용한다. `None`은 적응형 전환 OFF다.

| 설정 | V5a | V5b | V5c (기본) |
| --- | ---: | ---: | ---: |
| simulations | 50 | 80 | 80 |
| tactical_simulations | 50 | 150 | 150 |
| tactical_score_threshold | None | 600 | 600 |
| exploration | sqrt(2) | 1.0 | 1.0 |
| candidate_limit | 20 | 20 | 14 |
| initial_width | 8 | 8 | 6 |
| neighborhood_radius | 2 | 2 | 2 |
| priority_top_k | 8 | 8 | 6 |

V5a는 V4.1과 동일한 탐색 파라미터로 정책 효과를 분리한다.

```python
from agents import MCTSV5Agent

agent = MCTSV5Agent(stage="c", seed=42, tactical_score_threshold=None)
move = agent.select_move(game)
print(agent.diagnostics)
```

각 생성자 파라미터를 override할 수 있다. `tactical_score_threshold=None`도 명시적으로
override된다. `mcts_search_v5(..., diagnostics=SearchDiagnostics())`로도 진단을 수집한다.
강제수는 `forced_policy_stage=1..5`, `selected_simulations=0`, tactical score는 `None`이다.
일반 탐색은 forced stage가 `None`이며 `simulation_mode`로 normal/tactical을 구분한다.
진단은 호출자/에이전트 소유이며 전역 random이나 전역 진단 상태를 사용하지 않는다.

### 2판 smoke와 profile

`--games`는 **색상별 판수**이고 기본값은 1이다. 다음 명령 하나는 총 2판이다.
V5a 흑 vs V4.1 백 1판, V4.1 흑 vs V5a 백 1판을 실행한다.

```bash
python -m unittest discover -s tests -v
python scripts/run_mcts_v5_vs_v41.py --stage a --games 1 --seed 42
python -m cProfile -o v5_profile_before.prof scripts/run_mcts_v5_vs_v41.py --stage a --games 1 --seed 42
python -c "import pstats; pstats.Stats('v5_profile_before.prof').strip_dirs().sort_stats('cumtime').print_stats(50)"
```

최적화 후에도 동일 조건으로 측정한다.

```bash
python scripts/run_mcts_v5_vs_v41.py --stage a --games 1 --seed 42
python -m cProfile -o v5_profile_after.prof scripts/run_mcts_v5_vs_v41.py --stage a --games 1 --seed 42
python -c "import pstats; pstats.Stats('v5_profile_after.prof').strip_dirs().sort_stats('cumtime').print_stats(50)"
```

일반 smoke와 cProfile은 각각 실제 2판을 실행한다. 따라서 전후 smoke/profile을 모두
실행하면 2판 × 4회 = 8판이다. unit test의 simulations=1 RandomAgent 회귀는 별도다.
2판은 불법 착수, crash, 로그, 정책·진단 동작 확인용이며 기력 결론을 내리지 않는다.

기존 `games.csv`, `moves.csv`, `games.json` 스키마를 그대로 사용한다.
추가 `summary.json`에는 승수, 색상별 승수, 수·시간, 에이전트별 초/수,
강제 stage 횟수, 복수 주입, normal/tactical decision 횟수, root score min/max/avg,
착수별 진단, 마지막 위협 유형을 기록한다.
마지막 위협은 승리 착수 직전의 승자 이전 착수로 정의하고, 패자의 방어 전 국면에서
열린4 → 사사 → 금수점 4 → 기타 순서로 분류한다. 무승부는 기타로 기록한다.
이는 마지막 구조 분류이며 전체 승리 원인의 증명은 아니다.

### 선택적 Early Draw

기본 **OFF**. `--early-draw`를 명시하면 100수 이상에서 양쪽 모두 상대 돌 0개,
내 돌 3개 이상인 5칸 window가 없는 상태가 10수 연속될 때 스크립트가 무승부로
종료한다. 규칙 엔진의 `done`/`winner`를 변경하지 않는다. 사유는 summary에만 기록한다.
이것은 비교 시간 단축용 휴리스틱이며 공식 무승부 판정이나 게임 이론적 증명이 아니다.

### 사용자 장기 벤치마크

아래 명령은 사용자가 별도로 실행할 때 총 100판을 수행한다.
이번 구현 작업에서는 실행하지 않는다.

```bash
python scripts/run_mcts_v5_vs_v41.py --stage a --games 50 --seed 42
```

V5b/V5c 실전 비교와 early draw ON 비교도 사용자가 별도로 실행한다.
CLI에서 `--simulations`, `--tactical-simulations`, `--tactical-score-threshold`,
`--exploration`, `--candidate-limit`, `--initial-width`, `--neighborhood-radius`,
`--priority-top-k`는 V5 설정을 override한다. V4.1은 기준 프리셋을 유지한다.
threshold를 끄려면 `--tactical-score-threshold none`을 사용한다.


### V5 ?? ?? ??? ??

? 2? profile?? V5 ?? ?? ??? ?? ?? ?? ??? window ????
???? ?? ??? ????.

- Stage 5 ?? ?? ??? ?? 2???? ?? ?? ? 2? ??? ?? ????
  ????. ?? ?? ??? 33/44/43 ??? ??? ? ??.
- ?? ??? unstoppable creator?? ??? ? ?? ?? ?? window? ? ??
  ????. ?? ?? ? ? ??? ?? ??? ??? ???? ?? ????.
- root?? ?? ??? ? ??? ??? ? ?? ??? unstoppable ??? ?????.
- root ??? ?? ??? adaptive threshold ?? ??? ?????.

?? ?? ???? ??? ???, ??? ?? ???? ??, ?? seed ?? ???
?? ?? ??? ????. ?? V3.2.1 tree/rollout? ?? ??? ???? ???.


### V5 ?? ?? ?? ?? (seed 42)

?? `python -m unittest discover -s tests -v`: **96? PASS**.

?? A/C/E, V4 ?? ??, ???, ???global random ??, ?? ??,
RandomAgent ?? ?? 2??(simulations=1), root ??, ?? ??? ??? ????.

?? ??? V5a ?? 1?? smoke ?/?? cProfile ?/? ?? ? ?? ????.
?? early draw OFF??, ? ??? 167? ?? ??? ??? ??? ????.

| Smoke | V5a ? vs V4.1 ? | V4.1 ? vs V5a ? | ? ?? | ?/? |
| --- | --- | --- | ---: | ---: |
| ??? ? | ? ?, 110? | ? ?, 57? | 229.492 | 1.374 |
| ??? ? | ? ?, 110? | ? ?, 57? | 211.258 | 1.265 |

IllegalMove/crash ??. V5a 0?, V4.1 2??? ?? ??? ?? ???.
V5 Stage 3/4/5-single: **0/11/8**, Stage 5 multi ??: **2**.
normal/tactical ?? decision: **41/0**. root score min/max/avg: **0/2344/1588.683**.
??? ?? ??: ?? 1, ??4 1.

?? ??? cProfile ?? ?????. ?? ??? ?? ????? ?? ??? ???.

| ?? | ? | ? |
| --- | ---: | ---: |
| ?? ?? ? | 927,635,340 | 926,549,751 |
| ?? cProfile ??(s) | 1278.537 | 519.323 |
| ?? cProfile ?? / 167?(s) | 7.656 | 3.110 |
| `mcts_search_v5` (?? / ?? s) | 83 / 1008.855 | 83 / 213.042 |
| `_forced_v5_move` (?? / ?? s) | 83 / 1.018 | 83 / 0.495 |
| `_unstoppable_four_moves` (?? / ?? s) | 165 / 0.469 | 165 / 0.321 |
| `_is_unstoppable_four` (?? / ?? s) | 713 / 0.416 | 709 / 0.218 |
| `_four_completions` (?? / ?? s) | 711 / 0.043 | 711 / 0.045 |
| `forbidden_reason` (?? / ?? s) | 1,388,496 / 1060.856 | 1,388,044 / 272.197 |
| `legal_moves` (?? / ?? s) | 585 / 833.488 | 585 / 11.202 |
| `_double_threat_moves` (?? / ?? s) | 49 / 0.457 | 49 / 0.070 |
| `_window_candidates` (?? / ?? s) | 1,083 / 0.319 | 450 / 0.164 |
| `deepcopy` (?? / ?? s) | 97,400 / 0.214 | 97,400 / 0.252 |

**?? ???:** ??? ? profile? 2?? ?? 2?(V5a ?) ? ???
828.772?? ????. ??? ???? ????, ?? ?? ??? ????
?? ??? ???? ???. ?? ??? ??? ?? ?? ???? ?? ???.

?? ??? 1,085,589? ????. Stage 5 ?? ?? ???
1,440 ? 158?, window ?? ??? 1,083 ? 450?, ?? ??? 452? ????.
V5 ?? ?? ?? ??? 1.018 ? 0.495??.
?? V3.2.1 tree/rollout? ????? ??? ??? ?? ????.

?? smoke?? V5a/V4.1 ?? ?? ??? ? 1.135, ? 0.714??.
???? +25% ?? ????, ? ??? ????????? ?? ??? ???
??? ?? ??? ???? ?? ?? ????? ??? ????? ???.

?? profile: `v5_profile_before.prof`, `v5_profile_after.prof`.
?? ??: `logs/v5_smoke_before`, `logs/v5_profile_before`,
`logs/v5_smoke_after`, `logs/v5_profile_after`.
`logs/`? ?? gitignore? ?? ??? ????.

10? ?? ?? ??, 50? ??, 100? ??, V5a/b/c ?? ??? ???.
??? ?? ???? ??? ?? ????.
