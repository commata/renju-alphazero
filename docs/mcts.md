# 3단계 순수 MCTS

이 단계는 신경망 없이 **UCT 기반 Monte Carlo Tree Search**의 구조와 렌주 엔진 연동을 검증한다. AlphaZero의 PUCT, 정책망, 가치망, 자기대국 학습은 아직 포함하지 않는다.

## 현재 설정

- 기본 탐색 예산: 착수당 **10 simulations**
- 기본 탐색 후보: 노드당 **최대 8개**
- Selection: UCT
- Expansion: 합법수 중 주변 돌과의 연관도가 높은 후보를 우선하는 shortlist
- Simulation: 즉시 승리 → 가까운 즉시 패배 방어 → 지역 후보 중 seeded random rollout
- Backpropagation: 승리 +1, 패배 -1, 무승부 0
- 가치 관점: 각 노드에서 **그 노드로 들어오는 수를 둔 플레이어(player_just_moved)** 관점
- 난수: 에이전트별 `random.Random(seed)`
- 원본 `Game`: 탐색 중 변경하지 않음

노드의 값이 직전 착수자 관점으로 저장되므로, 턴이 바뀔 때 상대가 협조적으로 최대화하는 문제가 생기지 않는다.

## 1차 구현 측정

10 simulations, seed 42, 각 조건 1판의 최초 구현 측정:

| 대결 | 결과 | 평균 수 | 시간 |
| --- | --- | ---: | ---: |
| MCTS(흑) vs Random(백) | 백 승 | 140 | 130.693 s |
| Random(흑) vs MCTS(백) | 백 승 | 84 | 117.690 s |
| MCTS(흑) vs Tactical(백) | 백 승 | 42 | 92.526 s |
| Tactical(흑) vs MCTS(백) | 흑 승 | 51 | 93.034 s |

각 조건이 1판뿐이므로 기력 비교 결론으로 사용하지 않는다. 다만 초기 구현은 한 판에 약 90~130초가 걸렸고, 225개에 가까운 루트 합법수 중 10개만 한 번씩 확장하므로 10회 예산에서 UCT 재방문이 거의 일어나지 않는 문제가 확인됐다.

## 2차 수정

10회라는 작은 예산에서도 실제 Selection 단계가 발생하도록 다음을 추가했다.

1. **후보 제한**: 각 노드에서 지역성 점수 기준 최대 8개를 탐색한다. 규칙 판정은 바꾸지 않으며 반환 후보는 모두 합법수다.
2. **루트 전술 선처리**: 현재 플레이어의 즉시 승리는 바로 선택하고, 상대의 즉시 승리점이 하나뿐이면 그 수를 강제로 막는다.
3. **전술 rollout**: shortlist 안에서 즉시 승리, 상대 즉시 승리 방어, seeded random 순으로 진행한다.
4. **빠른 흑 후보 생성**: rollout과 내부 노드에서는 225개 전체의 흑 합법수를 매번 만들지 않고 지역성 순으로 빈칸을 검사해 필요한 합법 후보 수만 확보한다.
5. **상태 재사용**: simulation마다 `deepcopy(Game)`하지 않고 private state 하나를 사용한 뒤 `undo()`로 root까지 복원한다.

후보 제한은 **MCTS 탐색 휴리스틱**이며 `Game.legal_moves()`나 렌주 규칙 자체를 변경하지 않는다. AlphaZero 단계에서는 정책망이 이 후보 우선순위 역할을 대체하게 된다.

## 실행

먼저 전체 테스트:

```bash
python -m unittest discover -s tests -v
```

MCTS 10회와 Random을 양쪽 색으로 각 1판:

```bash
python scripts/run_mcts.py --games 1 --simulations 10 --candidate-limit 8 --seed 42
```

Tactical 비교까지 포함:

```bash
python scripts/run_mcts.py --games 1 --simulations 10 --candidate-limit 8 --seed 42 --include-tactical
```

프로파일이 필요하면 Python의 cProfile을 그대로 사용할 수 있다.

```bash
python -m cProfile -s cumulative scripts/run_mcts.py --games 1 --simulations 10 --candidate-limit 8 --seed 42
```

## 다음 측정

2차 수정 후에는 먼저 10 simulations를 유지한 채 1차 측정과 실행 시간을 비교한다. 그 뒤 10 → 25 → 50 → 100 simulations 순서로 늘리면서 다음을 기록한다.

- Random/Tactical 상대 양쪽 색 승·무·패
- 평균 착수 시간 및 전체 대국 시간
- 동일 seed 재현성
- cProfile 병목
- 탐색 수 증가 대비 성능 변화

후보 shortlist 때문에 이 단계의 Pure MCTS는 전체 225개 행동을 매번 동등하게 확장하는 교과서적 full-width UCT와는 다르다. 현재 목적은 15×15 렌주에서 작은 CPU 예산으로 탐색 구조와 가치 전달을 검증하는 것이다.
