# 3단계 순수 MCTS

이 단계는 신경망 없이 **UCT 기반 Monte Carlo Tree Search**의 구조와 렌주 엔진 연동을 검증한다. AlphaZero의 PUCT, 정책망, 가치망, 자기대국 학습은 아직 포함하지 않는다.

## 현재 설정

- 기본 탐색 예산: 착수당 **10 simulations**
- Selection: UCT
- Expansion: 현재 엔진의 `Game.legal_moves()` 중 아직 확장하지 않은 수 하나
- Simulation: 종료 상태까지 균등 무작위 합법수 rollout
- Backpropagation: 승리 +1, 패배 -1, 무승부 0
- 가치 관점: 각 노드에서 **그 노드로 들어오는 수를 둔 플레이어(player_just_moved)** 관점
- 난수: 에이전트별 `random.Random(seed)`
- 원본 `Game`: 탐색 중 변경하지 않음

노드의 값이 직전 착수자 관점으로 저장되므로, 턴이 바뀔 때 별도의 협조적 최대화가 일어나지 않는다. 각 노드에서는 그 상태에서 수를 둘 플레이어가 자신의 자식 값이 큰 가지를 선택한다.

## 실행

먼저 테스트:

```bash
python -m unittest discover -s tests -v
```

MCTS 10회와 Random을 양쪽 색으로 각 1판:

```bash
python scripts/run_mcts.py --games 1 --simulations 10 --seed 42
```

Tactical 비교까지 포함:

```bash
python scripts/run_mcts.py --games 1 --simulations 10 --seed 42 --include-tactical
```

10회 탐색은 15×15의 큰 분기 수에 비해 매우 작은 예산이다. 이 단계의 목적은 강한 기력을 주장하는 것이 아니라 **Selection → Expansion → Rollout → Backpropagation이 정확하게 연결되는지**, 금수를 포함한 합법수만 탐색하는지, 재현성과 실행 속도가 어떤지를 확인하는 것이다.

## 다음 측정

구조 검증 후 10 → 25 → 50 → 100 simulations 순서로 늘리면서 다음을 기록한다.

- Random/Tactical 상대 양쪽 색 승·무·패
- 평균 착수 시간 및 전체 대국 시간
- 동일 seed 재현성
- cProfile 병목
- 탐색 수 증가 대비 성능 변화

실제 병목을 확인한 뒤에만 트리 재사용, 더 빠른 rollout, 상태 복사 최적화 등을 검토한다.
