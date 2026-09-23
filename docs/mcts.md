# 3단계 순수 MCTS

이 단계는 신경망 없이 **UCT 기반 Monte Carlo Tree Search**의 구조와 렌주 엔진 연동을 검증한다. AlphaZero의 PUCT, 정책망, 가치망, 자기대국 학습은 아직 포함하지 않는다.

## V2: 현재 검증 기준본

2차 수정본은 다음 설정을 유지한다.

- 착수당 10 simulations
- 노드당 후보 최대 8개
- 지역성 점수 기반 shortlist
- 루트 즉시 승리 / 단일 강제 방어
- 전술 rollout
- simulation private state 재사용 + undo

### V2 5판씩 측정

seed 42, 10 simulations, candidate limit 8:

| 대결 | 결과 | 평균 수 | 시간 |
| --- | --- | ---: | ---: |
| MCTS-v2(흑) vs Random(백) | 5승 0패 | 18.60 | 14.377 s |
| Random(흑) vs MCTS-v2(백) | 0승 5패 | 18.80 | 10.920 s |
| MCTS-v2(흑) vs Tactical(백) | 5승 0패 | 21.40 | 20.410 s |
| Tactical(흑) vs MCTS-v2(백) | 0승 5패 | 21.60 | 13.111 s |

즉 이 고정 seed/설정의 20개 측정에서는 MCTS-v2가 모두 승리했다. 다만 표본 수가 아직 작고 특정 seed에 고정되어 있으므로 일반적인 기력 우위를 확정하는 결과로 사용하지 않는다.

프로파일에서는 후보 생성 쪽의 비용이 새 병목으로 확인됐다. 특히 매 rollout 수마다 전체 빈칸에 대해 `_move_score()`를 계산하고 정렬하는 비용이 크다.

## V3: 후보 12개 + progressive widening

3차 수정본은 V2를 덮어쓰지 않고 별도 `MCTSV3Agent`로 추가한다. V2는 `MCTSV2Agent`로 고정해 직접 대결할 수 있다.

기본값:

- simulations: 10
- candidate pool: **12**
- initial width: **4**
- neighborhood radius: **2**

후보 수를 12개로 늘리면 10 simulations만으로는 다시 모든 simulation이 신규 후보 확장에 소비될 수 있다. 그래서 V3는 **progressive widening**을 사용한다.

허용 자식 수는 대략:

```text
initial_width + floor(sqrt(node visits))
```

로 증가한다. candidate pool 전체는 12개지만 처음부터 12개를 전부 열지 않는다. 노드 방문이 쌓일수록 후보를 조금씩 추가해, 10회 예산에서도 UCT 재탐색과 후보 다양성을 함께 유지한다.

### V3 후보 생성 최적화

V2는 rollout마다 전체 빈칸을 점수화한 뒤 정렬했다.

V3는:

```text
기존 돌 주변 Chebyshev 거리 2 이내 빈칸 수집
→ 이 local pool만 move_score 계산
→ 합법성 확인
→ 최대 12개 사용
```

순서로 후보를 만든다.

빈 보드에서는 중앙 주변 5×5 영역을 시작 pool로 사용한다. local pool이 부족하거나 흑 금수 때문에 합법 후보를 충분히 만들지 못한 경우에만 전체 `Game.legal_moves()`로 fallback한다.

루트의 즉시 승리/단일 강제 방어는 여전히 전체 합법수를 대상으로 검사해 전술 안전성을 유지한다.

## 실행

전체 테스트:

```bash
python -m unittest discover -s tests -v
```

V2 기준선:

```bash
python scripts/run_mcts.py --games 5 --simulations 10 --candidate-limit 8 --seed 42 --include-tactical
```

V2와 V3 직접 대결:

```bash
python scripts/run_mcts_versions.py --games 5 --simulations 10 --v2-candidate-limit 8 --v3-candidate-limit 12 --v3-initial-width 4 --v3-radius 2 --seed 42
```

V3만 다른 후보 수로 실험할 때는 `MCTSV3Agent` 생성자 값을 바꾸거나 direct comparison runner의 인자를 변경한다.

## 평가 원칙

V3의 목표는 단순히 후보 수를 8→12로 늘리는 것이 아니다.

1. 더 넓은 후보 pool을 유지하고,
2. progressive widening으로 작은 탐색 예산에서도 재탐색을 보장하며,
3. 전체 빈칸 정렬을 local pool 정렬로 줄여 CPU 비용을 낮추고,
4. V2와 동일 seed/동일 simulation 예산으로 직접 비교한다.

V2와 V3는 같은 브랜치에 동시에 존재하므로 결과가 나쁘면 V2 기준본을 잃지 않는다.
