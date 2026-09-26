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

흑의 exact-five winning extension은 규칙 엔진과 동일하게 **다른 방향의 장목이 동시에 생겨도 승리수로 인정**한다. RIF 9.2는 흑의 금수를 "동시에 five in a row를 만들지 않았을 때" 적용하므로, exact five가 성립하면 장목/삼삼/사사보다 승리가 우선한다.

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


### V5 강제 정책 성능 최적화

2판 cProfile을 기준으로 V5 강제 정책에서 반복되던 상세 패턴 검사와 window 후보 생성을 줄였다.

- Stage 5 이중 위협 후보는 서로 다른 두 방향에서 위협이 될 가능성이 있는 수만 상세 검사한다. 상세 패턴 검사는 **1,440회 → 158회**로 감소했다.
- 상대 즉시 승리 후보와 unstoppable creator 검사에서 사용할 구조적 window 후보를 재사용한다.
- root에서 계산한 현재 플레이어 합법수 목록을 재사용하여 동일 상태의 불필요한 합법성 검사를 줄였다.
- window 후보 생성은 **1,083회 → 450회**로 감소했다.
- 전체 함수 호출은 **1,085,589회 감소**했다.
- V5 강제 정책 누적 시간은 **1.018초 → 0.495초**로 감소했다.
- 최적화는 V5 강제 정책 내부의 중복 계산 제거에 한정했으며 V3.2.1 tree/rollout과 정책 우선순위는 변경하지 않았다.

최적화 전후 동일 seed에서 **167수 전체 기보와 착수별 진단값이 모두 일치**했다. 따라서 이번 변경에서 관측된 범위에서는 정책 의미와 결정성이 보존됐다.

### V5 구현 및 검증 결과 (seed 42)

최종 `python -m unittest discover -s tests -v`: **96개 PASS**.

국면 A/C/E, V4 회귀, 결정성, Game 상태·global random 보존, 설정 검증,
RandomAgent 양쪽 색상 2판씩(`simulations=1`), adaptive budget, root 주입,
early draw 경계 검증을 통과했다. `compileall`, `git diff --check`도 통과했다.

일반 smoke와 cProfile은 V5a 흑 1판 / 백 1판으로 각각 총 2판씩 실행했다.
모든 실행은 early draw OFF이며 이 결과만으로 기력 우열을 판단하지 않는다.

| Smoke | V5a 흑 vs V4.1 백 | V4.1 흑 vs V5a 백 | 총 시간(s) | 초/수 |
| --- | --- | --- | ---: | ---: |
| 최적화 전 | 백 승, 110수 | 흑 승, 57수 | 229.492 | 1.374 |
| 최적화 후 | 백 승, 110수 | 흑 승, 57수 | 211.258 | 1.265 |

최적화 후 smoke 상세:

- V5a 흑 vs V4.1 백: **백 승, 110수, 133.432초**
- V4.1 흑 vs V5a 백: **흑 승, 57수, 77.826초**
- 총 **211.258초**, **1.265초/수**
- IllegalMove·crash 없음
- V5 Stage 3/4/5-single: **0 / 11 / 8회**
- Stage 5 multi root injection: **2회**
- normal/tactical 탐색 결정: **41 / 0회**
- root score min/max/avg: **0 / 2344 / 1588.683**

V5a는 `tactical_score_threshold=None`이므로 adaptive tactical simulation은 비활성화되어 있다. 따라서 위 실행에서 tactical decision이 0회인 것은 의도된 동작이다.

최적화 전후 cProfile 결과는 다음과 같다. 함수 시간은 누적 초다.

| 지표 | 최적화 전 | 최적화 후 |
| --- | ---: | ---: |
| 전체 함수 호출 수 | 927,635,340 | 926,549,751 |
| 전체 cProfile 시간(s) | 1,278.537 | 519.323 |
| cProfile 시간 / 167수(s) | 7.656 | 3.110 |
| `mcts_search_v5` (호출 / 누적 s) | 83 / 1,008.855 | 83 / 213.042 |
| `_forced_v5_move` (호출 / 누적 s) | 83 / 1.018 | 83 / 0.495 |
| `_unstoppable_four_moves` (호출 / 누적 s) | 165 / 0.469 | 165 / 0.321 |
| `_is_unstoppable_four` (호출 / 누적 s) | 713 / 0.416 | 709 / 0.218 |
| `_four_completions` (호출 / 누적 s) | 711 / 0.043 | 711 / 0.045 |
| `forbidden_reason` (호출 / 누적 s) | 1,388,496 / 1,060.856 | 1,388,044 / 272.197 |
| `legal_moves` (호출 / 누적 s) | 585 / 833.488 | 585 / 11.202 |
| `_double_threat_moves` (호출 / 누적 s) | 49 / 0.457 | 49 / 0.070 |
| `_window_candidates` (호출 / 누적 s) | 1,083 / 0.319 | 450 / 0.164 |
| `deepcopy` (호출 / 누적 s) | 97,400 / 0.214 | 97,400 / 0.252 |

**측정 한계:** 최적화 전 profile에는 V5a 착수 한 번이 약 **828.772초**로 기록된 원인 미확정 이상치가 있다. `forbidden_reason`과 `legal_moves` 호출 수는 전후 거의 같은데 누적 시간이 크게 달라졌으므로, 전체 cProfile 시간 감소를 순수한 코드 최적화 효과로 해석하면 안 된다.

코드 변경과 직접 대응해 신뢰할 수 있는 개선은 다음과 같다.

- Stage 5 상세 패턴 검사: **1,440 → 158회**
- window 후보 생성: **1,083 → 450회**
- 전체 함수 호출: **1,085,589회 감소**
- `_forced_v5_move`: **1.018 → 0.495초**
- 전후 동일 seed의 전체 기보·진단 결과 보존

일반 smoke에서 관측된 V5a/V4.1 수당 시간 비율은 **1.135 → 0.714**였다.
관측값 자체는 +25% 성능 목표 이내지만 서로 다른 플레이어가 서로 다른 국면을 계산한 작은 표본이므로 엄밀한 V5 정책 오버헤드 상한 검증으로 보지는 않는다.

프로파일 파일:

- `v5_profile_before.prof`
- `v5_profile_after.prof`

측정 로그:

- `logs/v5_smoke_before`
- `logs/v5_profile_before`
- `logs/v5_smoke_after`
- `logs/v5_profile_after`

`logs/`는 `.gitignore` 대상이므로 실행 결과는 로컬에 남는다.

### 미실행 장기 검증

이번 구현·최적화 단계에서는 다음 대규모 실전 비교를 실행하지 않았다.

- 10판 이상 실전 비교: 미실행
- 50판 비교: 미실행
- 100판 비교: 미실행
- V5a/b/c 승률 비교: 미실행
- early draw ON 대국: 미실행

대규모 실전 테스트는 별도로 실행한다.

## V5 FINAL과 V6 Threat Planning

```text
V5 = reactive forced tactical policy
V6 = proactive threat construction
   + proactive opponent threat prevention
```

### 비교 기준: V5 FINAL

V6 비교 runner는 기존 V5a/b/c 기본값과 별개로 양쪽에 아래 값을 명시적으로 전달한다.
기존 V5 생성자의 기본값과 공개 API는 바꾸지 않는다. 이번 작업에서 파라미터 튜닝은 하지 않았다.

| 설정 | V5 FINAL / V6 공통 |
| --- | ---: |
| simulations | 50 |
| tactical_simulations | 100 |
| tactical_score_threshold | 1800 |
| exploration | sqrt(2) |
| candidate_limit | 20 |
| initial_width | 8 |
| neighborhood_radius | 2 |
| priority_top_k | 8 |

사용자가 제공한 V5 FINAL 기준 기록이며, 아래 100판을 이번 V6 작업에서 다시 실행한 것은 아니다.

| 검증 | V5 W/L/D | 흑 W/L/D | 백 W/L/D |
| --- | --- | --- | --- |
| seed 42, 100판, 상대 V4.1 | 65/19/16 | 28/11/11 | 37/8/5 |
| seed 43, 20판, 상대 V4.1 | 15/2/3 | 5/2/3 | 10/0/0 |

seed 42 score rate는 전체 73%, 흑 67%, 백 79%다. normal/tactical 결정은
1665/785회(약 32%), V5/V4.1 착수당 시간은 각각 약 1.499/1.291초/수다.
seed 43 tactical 비중은 약 26%다. V6의 직접 비교 상대는 **V5 FINAL**이다.

### V6.0: 독립 pattern detector

`src/search/threat_patterns.py`는 AI 선택과 독립적으로 사용할 수 있다.

```python
from search.threat_patterns import (
    black_legal_43_moves, black_immediate_43_creators,
    white_43_moves, white_44_moves, white_33_moves,
)
```

- four는 creator를 포함한 네 돌과 실제 승리 completion으로 표현한다.
  동일한 네 돌을 표현하는 여러 5-cell window는 `(방향, 돌 집합)` 하나로 합친다.
- three는 creator를 포함한 세 돌을 합법적으로 연장해 양끝에 합법 승리점이 있는
  곧은 four를 만들 수 있는 구조다. 연장점뿐 아니라 양끝 방어점도 보관한다.
- 독립성은 **서로 포함되지 않는 돌 집합이며 방어점 집합이 겹치지 않는 위협**으로 정의한다.
  방향이 같아도 서로 다른 승리점으로 이어지는 독립 44는 탐지한다.
  같은 돌 집합의 sliding window, 더 강한 패턴의 부분집합, 공통 방어점이 있는 표현을
  독립 위협 두 개로 세지 않는다.
- 43은 독립적인 four와 three의 조합이다. four의 각 합법 방어를 실제로 놓은 뒤
  같은 three의 합법 연장이 남는지 다시 검사한다. 이는 상대의 모든 counter-attack까지
  검증한 강제승 증명과는 구분한다.
- 흑 creator, three 연장, 승리 completion, 흑 방어 모두 기존 `forbidden_reason()`에
  기반한 합법성 helper를 사용한다. 흑 금수 규칙을 새 detector에 복제하지 않는다.
  현재 엔진은 정확한 5목 승리를 장목/33/44보다 먼저 판정한다.
  따라서 교차 장목과 동시에 exact five를 만드는 completion은 합법적인 승리수이며,
  금수 반례는 exact five가 아닌 비승리 creator·three 연장 수로 검증한다.
- four 방어가 다른 교차선을 막아서 three 연장의 금수를 해소할 수 있으므로,
  흑 43의 three는 실제 four 방어 이후의 보드에서도 검사한다.
- 백은 독립 four 두 개를 44, three 두 개를 33, four+three를 43으로 인식한다.
  즉시 승리수는 compound creator에서 제외하고 기존 Stage 1에 맡긴다.

캐시하는 것은 불변 보드 geometry뿐이다. 상태별 결과를 영구 보관하지 않으므로
테스트나 호출자가 `game.board`를 직접 바꾸어도 이전 결과가 재사용되지 않는다.
임시 돌은 context manager의 `try/finally`로 복구하며 turn/history/terminal flags를 바꾸지 않는다.

### V6.1: 공격 후보와 백의 흑 43 예방

`src/search/mcts_v6.py`는 우선 `_forced_v5_move()`를 그대로 호출한다.

1. 내 즉시 승리
2. 상대 즉시 승리 차단
3. 내 unstoppable four
4. 상대 unstoppable four 예방
5. 상대 double threat: 기존 single forced block / multi-root injection
6. 위에서 반환하지 않았을 때만 V6 planner와 Adaptive MCTS

흑의 legal 43, 백의 44/43/33 creator를 root에 주입한다.
백은 상대 흑의 legal 43 creator와 관련 four completion / three 연장·끝점을
방어 후보로 추가한다. 새 방어는 **forced return이 아니다**.
백 자신의 즉시 승리나 기존 unstoppable four가 항상 먼저 처리된다.

기존 V5 top candidates와 Stage 5 주입을 먼저 얻고, 그 전체 집합에 V6 후보를 합친다.
후보는 실제 root 합법수와 교차한다. 일반 candidate limit는 20 그대로이며,
전술 주입 때문에 root 크기가 20을 넘는 것은 허용한다.

V6 root 정렬은 큰 점수 덧셈 대신 ordinal tier를 사용한다.
백 44 → legal 43 / 흑 43 방어 / 금수 방어 유도 → 백 33 → future setup 순이며,
Stage 5 주입은 43 방어와 같은 tier를 유지한다. 같은 tier에서는 관측한 방어 수와
기존 `_v321_move_key`를 사용한다. future 공격은 검사한 합법 response 수가 적은 것을 우선한다.
V6 후보가 없으면 기존 V5 root 순서를 그대로 반환한다.

adaptive budget의 전술 점수에는 planner tier를 더하지 않는다. 후보들의 원래 V3.2.1 점수와
고정 threshold 1800을 사용한다. 탐색 후보 변화로 normal/tactical 비중은 달라질 수 있다.
공유 `_search_v5_tree()`는 기존 V5 tree loop를 함수로 추출한 것으로,
progressive widening·UCT·V3.2.1 rollout·seeded tie-break는 그대로다.

### V6.2: 2-ply Threat Planning

`src/search/threat_planning.py`는 `X → 상대 response → Y compound`를 검사한다.

- 내 돌 2개 이상인 unblocked window의 빈칸에서 구조적으로 유망한 X를 선택한다.
- X 자체가 이미 compound 또는 즉시 승리면 future setup으로 중복 계산하지 않는다.
- X 이후 Y가 만드는 compound에는 **X가 해당 compound의 구성 돌로 참여**해야 한다.
  보드 다른 곳에 이미 있던 공격을 X의 효과로 세지 않는다.
- X가 즉시 four를 만들면 상대의 승리 또는 completion 차단을 response로 검사한다.
  그 외에는 관련 compound creator·critical window 방어점과 전역 counter-four를 검사한다.
- 각 합법 response 뒤에 합법 Y와 completion을 다시 검사한다.
  상대가 먼저 즉시 승리하거나, Y 이후 상대의 즉시 승리가 남으면 제외한다.
- 검사한 모든 response에서 유지되는 종류만 future 43/44/33으로 기록한다.
  흑은 legal 43만 기록한다. 이는 제한된 threat-space 평가이며 전체 minimax 증명이 아니다.
- 백은 흑 future legal-43 setup과 그 연결·방어점을 root에 주입한다.
  즉시 흑 43이 없는 synthetic position에서도 이를 검증했다.

초기 계산 상한은 setup 6개, 후속 creator 8개, relevant defense 24개다.
방어가 24개를 넘으면 해당 setup을 제외한다. 방어 일부를 버린 뒤 성공으로 판정하지 않는다.
setup/continuation 상한은 후보 누락을 허용하는 성능 제약이며 최적값이라고 주장하지 않는다.
score 기반 새 activation threshold를 튜닝하지 않고 2-stone window 존재를 구조적 gate로 사용한다.
각 상한 도달 횟수와 검사량을 diagnostics에 노출한다.

### V6.3: WHITE forbidden-defense induction

백의 four 공격 X 이후 completion 방어점이 흑에게 금수인지 기존 엔진으로 검사한다.
`white_defense_profile()`은 실제로 모든 백 즉시 승리를 없애는 합법 방어 수,
금수 방어점 수, 상대의 가능한 completion 방어 후 최소 잔여 승리점 수를 제공한다.
흑의 즉시 counter-win은 합법 방어로 포함한다.

초기 구현의 induction 계수는 **four completion 방어**에 한정된다.
가상 three 끝점을 곧바로 강제 방어로 간주하지 않는다. 모든 방어가 금수인 명백한 four는
이미 V5 Stage 3에서 처리되므로, 그 수에서는 V6 planner 계수가 0일 수 있다.
2-ply 탐색 중에도 흑 response·creator의 합법성을 재검사하지만,
future induction을 별도 승리 증명으로 반환하지 않는다.

### Diagnostics와 비교 runner

```powershell
python -m unittest discover -s tests -v
python -m compileall src tests scripts
git diff --check
python scripts/run_mcts_v6_vs_v5.py --games 1 --seed 42
python scripts/run_mcts_v6_vs_v5.py --games 5 --seed 42
python scripts/run_mcts_v6_vs_v5.py --audit-log-dir logs/v6_smoke_10
```

`--games`는 색상별 판수다. 기본 1은 총 2판이다. runner에는 파라미터 튜닝 옵션이나
early draw를 추가하지 않았다. V6/V5 FINAL에 같은 설정을 전달하고 기존 CSV/JSON exporter를 재사용한다.

`SearchDiagnostics`는 기존 V5 diagnostics를 상속하며 다음을 추가한다.

- `black_43_candidates`, `white_43_candidates`, `white_44_candidates`, `white_33_candidates`
- `black_43_defense_candidates`, `black_43_defense_injections`
- `future_black_43_setups`, `future_white_43_setups`, `future_white_44_setups`, `future_white_33_setups`
- `future_black_43_defense_candidates`, `forbidden_defense_induction_count`
- `legal_defense_count`, `forbidden_defense_count`, `remaining_winning_continuations`
- `v6_root_injection_count`, `v6_selected_threat_type`, `v6_selected_reasons`
- `v6_threat_planner_seconds`, `planner_*` 검사량과 상한 도달 계수

숫자는 후보 수를 합산한 값이다. 같은 수가 여러 위협을 만들면 종류별 계수는 중복될 수 있고,
root 주입은 좌표별로 한 번만 센다. 기존 후보에 이미 있던 전술 좌표도 주입 계수에 포함한다.
강제 Stage 1~5에서 반환한 수에서는 V6 planner를 실행하지 않으므로 새 계수는 0이다.

`summary.json`은 V6 흑/백 W/L/D, 색상별 detector·주입·선택 횟수,
에이전트별 초/수, planner 시간, forced stage와 simulation mode를 집계한다.
선택된 백의 흑 43 방어는 기보를 별도로 replay하여 방어 직후 남은 흑 43 creator 수와
바로 다음 흑 착수가 실제 43이었는지도 기록한다. 이 계측은 착수 시간 측정 밖에서 수행한다.
후속 흑 착수가 없으면 결과를 `null`로 남겨 성공 방어로 잘못 집계하지 않는다.

`--audit-log-dir`는 새 대국 없이 기보를 replay하여 강제 반환수, 순서가 있는 root 후보,
전술 점수·simulation budget, detector·주입·선택 계수를 기존 로그와 대조한다.
결과는 `root_audit.json`에 저장한다. tree/rollout과 설정이 같은 상태에서 모든 root 입력이 같으면
seeded 탐색 경로도 유지되지만, 이 검사는 end-to-end 실행 시간을 재측정하지 않는다.

### 성능과 검증 범위

5-cell window와 cell별 window geometry를 사전 계산하고, 서로 다른 2-stone 이상 구조가 있는
creator만 상세 분석한다. planner는 root에만 연결되어 rollout에서 재귀 탐색하지 않는다.
각 호출의 결과와 root tactical score는 호출 안에서만 재사용한다.

새 테스트는 4방향·edge, 33/44/장목 creator, 후속 33/44 금수와 교차 장목 completion,
canonical window 중복, full-board detector와 structural shortlist 비교,
실제 규칙 엔진의 completion 승리, 흑/백 future setup, 금수 방어 유도,
백의 즉시·future 흑 43 예방, forced stage 보존, 예외 복원, seeded determinism,
직접 board 수정, global random·history·turn·winner 보존, runner 색상 집계를 포함한다.

관측 결과는 아래 검증 기록에 정리한다. 기력에 맞춰 점수나 threshold를 조정하지 않는다.

### V6 검증 기록 (2026-09-25)

- 전체 `python -m unittest discover -s tests -v`: **136 tests PASS**.
  기존 96개를 그대로 유지하고 V6 관련 40개를 추가했다.
- `python -m compileall src tests scripts`: PASS.
- `git diff --check`: PASS.
- V5 변경은 tree loop를 공통 함수로 추출한 9줄뿐이다. 기존 V5의 정책·설정·diagnostics,
  V3.2.1/V4·규칙 엔진·기존 runner의 코드는 변경하지 않았다.
- seed 42 사전 smoke, 1판/색: V6 흑 0W/1L/0D, 백 1W/0L/0D.
  각각 78수/82수, 총 226.789초. V6 1.473초/수, V5 FINAL 1.361초/수, 관측 비율 +8.24%.
  이 사전 실행 뒤에는 setup 연결 조건과 잔여 승리점 집계를 보강했고,
  아래 10판 실행은 보강된 구현을 사용한다.
- 사전 smoke의 백은 future 43/44/33 후보 7/4/1개, 중복 제거 root 주입 11개,
  future 44 선택 1회였다. 78수의 future 44 선택 뒤 80수 Stage 3, 82수 Stage 1로 승리했다.
  인과적 기력 개선의 증명은 아니며 기보·진단에서 확인한 순서다.
  replay에서 0-based `(3,5) → 흑 (2,4) → 백 (1,5)`의 실제 44 생성을 확인했다.

#### 10판 smoke: seed 42 / 색상당 5판 / early draw OFF

| 배치 | V6 W/L/D | 대국 수 |
| --- | --- | ---: |
| V6 Black vs V5 FINAL White | 1/2/2 | 5 |
| V5 FINAL Black vs V6 White | 2/2/1 | 5 |
| 합계 | 3/4/3 | 10 |

총 1066수, 1611.504초. IllegalMove·crash 없이 모두 정상 종료했다.
무승부 3판은 모두 225수로 보드를 채웠으며 조기 종료가 아니다.

| Detector / 정책 | 후보 탐지 수 | 실제 선택 수 |
| --- | ---: | ---: |
| 흑 own legal 43 | 4 | 1 |
| 흑 own future legal 43 | 4 | 1 |
| 백 own 43 | 6 | 0 |
| 백 own 44 | 0 | 0 |
| 백 own 33 | 1 | 1 |
| 백 future 43 | 17 | 3 |
| 백 future 44 | 9 | 1 |
| 백 future 33 | 2 | 0 |
| 백 forbidden-defense induction | 0 | 0 |
| 백의 흑 immediate 43 예방 | 흑 creator 2 / 방어 후보 11 | 1 |
| 백의 흑 future 43 예방 | 흑 setup 11 / 방어 후보 20 | 2 |

중복 제거 root 주입은 흑 8개, 백 64개로 총 72개다. 전술 이유가 있는 수의 실제 선택은
흑 2회, 백 8회다. 백의 immediate 43 방어 후보 11개는 모두 root에 주입됐다.
선택된 immediate/future 방어 3회 모두 바로 다음 흑 착수에서 43은 발생하지 않았다.
그러나 game 10의 immediate 방어 직후에는 흑 43 creator가 여전히 1개 남았다.
따라서 다음 수의 43 부재를 완전한 방어 성공이나 강제패 회피 증명으로 해석하지 않는다.

| 10판 실행 시간 지표 | 값 |
| --- | ---: |
| V6 seconds/move | 1.478502 |
| V5 FINAL seconds/move | 1.544740 |
| 관측 시간 비율 V6/V5 - 1 | -4.29% |
| V6 planner seconds 합계 | 3.778649 |
| V6 normal / tactical 결정 | 247 / 99 |
| V6 forced 결정 | 187 |

같은 방향의 독립 44 지원은 smoke 실행 중 최종 보강했다. 보강 후 전체 기보를 재생해
**533개 V6 결정(강제수 187개 + MCTS 346개)의 반환수/root 순서/점수/budget 불일치 0개**,
detector·주입·선택 계수 불일치 0개를 확인했다. 기존 shared tree는 변경하지 않았으므로
이 로그의 seeded 선택 경로를 유지한다. 최종 보강 후 planner 재생 시간은 8.809초였다.
위 end-to-end 초/수는 10판 실행 당시 관측이며, 보강 후 전체 대국 시간의 재측정값은 아니다.
서로 다른 국면과 일부 초기 회귀 테스트 동시 실행의 영향도 있으므로 순수 오버헤드 상한으로 주장하지 않는다.

이 결과는 10판 정상 동작과 전술 전달의 증거다. V6 우월성을 입증하지 않으며,
3승 4패 3무에 맞춰 score/threshold를 재조정하지 않았다.

로그는 `logs/v6_unit_tests.txt`, `logs/v6_smoke_2/`, `logs/v6_smoke_10/`에 저장한다.
각 smoke 폴더는 `games.csv`, `moves.csv`, `games.json`, `summary.json`을 포함한다.
`logs/`는 기존 `.gitignore` 대상이므로 로컬 기록이고, 요약은 이 문서에 보존한다.

### 알려진 범위 제한

- 2-ply는 setup/continuation 상한과 relevant-defense 집합을 사용하는 휴리스틱이다.
  전체 합법 응수에 대한 강제승 증명이 아니므로 forced return으로 승격하지 않는다.
- 선택된 root 후보가 모두 실제로 tree expansion을 받는 것은 아니다.
  기존 progressive widening과 50/100 simulations를 유지하기 때문이다.
- future setup 이후의 계획 수를 자동 실행하지 않는다. 10판 중 game 5의 흑 55수
  `(2,9)`는 백 `(1,10)` 응수 뒤 legal 43 `(1,9)`를 남겼지만,
  실제 57수에서는 기존 V5 Stage 5가 `(4,13)` 방어를 선택했다(좌표는 0-based).
  planner는 compound 가능성을 확인하지만 이후 V5 강제 정책과의 계획 일관성까지 증명하지 않는다.
  이 사례에서 기존 Stage 5를 덮어쓰는 조정은 하지 않았다.
- game 10의 백 28수 `(4,9)`는 흑 43 creator `(4,10)`을 방해했지만 `(7,5)`는 남겼다.
  흑은 29·31·33수에 단일 승리점을 계속 만들었고 백은 매번 Stage 2 방어를 선택했다.
  흑 35수 Stage 3, 37수 Stage 1로 대국이 끝났다. 새 방어 후보의 주입·선택은 확인했으나
  모든 상대 위협을 제거하는 방어수라는 보장은 없음을 보여주는 실제 패배 사례다.
- forced Stage 1~5가 먼저 처리한 수는 V6 detector 계수에 포함되지 않는다.
  따라서 실전에서 백 44나 금수 방어 유도 계수가 0이어도 해당 전술 자체의 부재를 뜻하지 않는다.
- 서로 다른 대국 국면의 초/수 비율은 관측 비용이며 순수 planner 오버헤드 상한 증명은 아니다.
  100판 및 seed 교차검증은 이번 작업에서 실행하지 않는다.



### V6 후속 정책 보강: 계획 지속성과 흑 43 방어 coverage

10판 smoke의 3승 4패 3무는 V6 우월성을 보여주지 못했지만, 두 실제 기보가 다음 수정 지점을 분리해 주었다.

1. **future setup 실행 가능성**: game 5에서는 흑 55수의 future legal 43 계획이 존재했지만 다음 흑 차례에 기존 V5 Stage 5가 다른 수를 강제해 계획이 이어지지 않았다. 이제 각 상대 응수 뒤 실제 다음 턴의 V5 Stage 1~5를 재평가한다. Stage 2/4/5 등이 계획된 continuation과 다른 수를 강제하면 해당 future setup을 제외한다. Stage 1 즉시승리나 Stage 3 unstoppable four처럼 계획보다 강한 강제 공격은 유효한 setup으로 유지한다. `planner_forced_plan_conflicts` / `planner_forced_plan_preserved`로 이 필터의 작동을 기록한다.
2. **백의 흑 43 방어 coverage**: game 10에서는 백의 선택이 한 creator를 막았지만 다른 creator를 남겨 연속 강제방어 뒤 패배했다. 이제 흑 43 방어 후보를 실제로 임시 착수한 뒤 남는 immediate legal 43 수를 먼저 계산하고, immediate 43을 모두 제거한 후보에 대해서만 future legal 43 수까지 계산한다. 같은 V6 priority 안에서는 남는 immediate/future 43이 적은 방어를 먼저 정렬한다. `black_43_defense_complete_candidates`, 최소 잔여 immediate/future 43 diagnostics를 추가한다.
3. **후속 계측**: 비교 runner의 offline replay가 선택된 백 방어 직후 immediate/future 흑 43 잔여 수와 complete defense 수를 함께 집계한다. 이 계측은 착수 시간 밖에서 수행한다.

V5 FINAL의 50/100 simulations, threshold 1800, exploration/candidate/width/radius/top-k와 Stage 1~5 우선순위는 변경하지 않았다. 점수나 threshold를 3승 4패 3무 결과에 맞춰 조정하지 않았으며, 이 보강 뒤 100판 장기 비교도 아직 실행하지 않는다. 먼저 전체 unit/compile 검증 후 동일 seed의 10판 smoke에서 계획 충돌 감소와 완전 방어 선택 여부를 확인한다.

## Stage 3 Final Baseline

Stage 3의 최종 기준 에이전트는 **MCTS-v6**로 확정한다. 이 버전은 이후 정책·가치 신경망 및 AlphaZero 탐색을 평가하기 위한 고정 benchmark baseline으로 사용하며, Stage 4 이후의 학습 경로에서는 V5/V6의 강제수·전술 planner 계층을 직접 재사용하지 않는다.

### 최종 검증 결과

> 아래 Stage 3 수치와 seed 777 SHA는 **RIF exact-five 우선순위 교정 이전**에 측정한 역사적 baseline이다.
> V5/V6가 공유 규칙 엔진과 V3.2.1 fast scanner를 사용하므로 현재 규칙 계약과 동일 조건의 수치로
> 직접 비교하지 않는다.

> RIF 교정 후 2026-09-26 동일 seed 777로 10판(V6 흑 5, 백 5) smoke를 재검증했다.
> V6는 흑 **1승 2패 2무**, 백 **5승 0패 0무**, 합계 **6승 2패 2무(score 70.0%)**였다.
> 이 값은 10판 smoke 관측치이며, 과거 100판 55.0%와 직접 비교해 기력 향상으로 해석하지 않는다.
> `run_mcts_v6_vs_v5.py`가 timing/diagnostics를 제외한 `(winner, history)` 목록의 SHA256을 직접 출력하도록 한 뒤,
> 동일 seed 777 10판을 연속 두 번 실행했다. 두 실행의 최종 `Outcome/history SHA256`은 모두
> `fd3f8ee61cb954c7c91249eeb79f420b5829c43f361f50e1055ebfbafa851d0f`로 일치했다.
> 따라서 이 값을 **post-RIF determinism baseline**으로 고정한다.

> 재현 명령: `python scripts/run_mcts_v6_vs_v5.py --games 5 --seed 777`
> 검증 로그: `logs/mcts_v6_vs_v5/20260926-015229_seed777`, `logs/mcts_v6_vs_v5/20260926-021654_seed777`

```text
Stage 3 Final Baseline
======================

Final agent:
MCTS-v6

Regression:
147 / 147 PASS

V6 vs Random:
Black : 20-0-0
White : 20-0-0
Total : 40-0-0
Score : 100%

V6 vs Tactical:
Black : 50-0-0
White : 50-0-0
Total : 100-0-0
Score : 100%

V6 vs V5 Final:
48W / 38L / 14D
Score : 55.0%

Determinism:
seed = 777

SHA256:
924adffa7e7ef4d81167d98fc8e2688905e6d335ec0cc0f838d8ad5c3f4fa9b2

Result:
Stage 3 PASS
MCTS-v6 frozen as benchmark baseline.
```

### 최종 결과 해설

전체 회귀 테스트는 **147개 모두 통과**했다. 여기에는 렌주 규칙, 금수 판정, 상태 복원, terminal 처리, MCTS V5/V6 탐색, 위협 패턴 및 planner, runner와 웹 대국 관련 검증이 포함된다. 따라서 Stage 3 종료 시점의 엔진과 탐색 구현은 현재 테스트 범위에서 회귀 없이 동작하는 것으로 판단한다.

기준선 대결에서는 V6가 Random 상대 양쪽 색에서 각각 20전 전승하여 총 **40승 0패 0무**, Tactical 상대에서도 양쪽 색 각각 50전 전승하여 총 **100승 0패 0무**를 기록했다. 이 결과는 Random/Tactical 기준선에 대해 탐색 파이프라인이 안정적으로 동작하며 선후공 한쪽에서 구조적으로 붕괴하지 않는다는 sanity check로 사용한다. 다만 두 기준선 상대의 100% 승률만으로 일반적인 렌주 기력을 확정하지는 않는다.

보다 강한 고정 기준인 V5 FINAL과의 100판 비교에서는 **48승 38패 14무**, 무승부를 0.5점으로 계산한 score **55.0%**를 기록했다. 이 결과를 바탕으로 V6를 Stage 3의 최종 MCTS 기준본으로 동결하고, 이후 신경망 체크포인트의 성장 정도를 측정하는 benchmark opponent로 사용한다.

seed 777로 동일한 10판 묶음(V6 흑 5판, V6 백 5판)을 두 번 실행한 결과 SHA256이 두 실행 모두
`924adffa7e7ef4d81167d98fc8e2688905e6d335ec0cc0f838d8ad5c3f4fa9b2`
로 일치했다. 따라서 해당 조건에서 경기 결과와 history의 재현성을 최종 확인했다.

RIF 교정 후 seed 777 10판의 결과/history 결정성은 두 번의 독립 실행에서 동일 SHA로 확인됐다.
착수 시간은 실행별로 달랐으므로 결정성 hash에서 제외한다. 첫 검증 실행에서는 V6/V5 FINAL이
각각 약 **1.441 / 1.490 s/move**, 두 번째 실행에서는 약 **1.190 / 1.232 s/move**였다.
따라서 timing은 환경 부하의 영향을 받는 관측치로만 남기며 성능 우위 증명으로 사용하지 않는다.

Stage 3 이후에는 MCTS-v6의 handcrafted 전술 계층을 더 확장하지 않는다. Stage 4에서는 기존 `Game`/렌주 규칙 엔진과 합법수 판정을 유지하되, 정책·가치 신경망을 별도 경로로 구현한다. V6는 학습 MCTS의 직접적인 부모 구현이 아니라 **비교 평가용 고정 baseline**으로 보존한다.

