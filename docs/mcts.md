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
