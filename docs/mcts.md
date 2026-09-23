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

## V2 vs V3 비교

두 revision은 의도적으로 다른 기본 탐색 예산을 유지한다.

- V2: 10 simulations / 8 candidates
- V3: 25 simulations / 16 candidates / initial width 6

실행:

```bash
python scripts/run_mcts_versions.py --games 5 --v2-simulations 10 --v3-simulations 25 --v2-candidate-limit 8 --v3-candidate-limit 16 --v3-initial-width 6 --v3-radius 2 --seed 42
```

이 비교는 "동일 계산량에서 알고리즘만 비교"하는 실험이 아니라, **검증된 V2 기준본과 더 큰 탐색 예산을 사용하는 V3 후보 모델을 비교하는 승급 테스트**다.

알고리즘 자체만 공정하게 비교하고 싶다면 두 revision의 simulation 수를 같게 지정해 별도 실험할 수 있다.

예:

```bash
python scripts/run_mcts_versions.py --games 5 --v2-simulations 25 --v3-simulations 25 --v2-candidate-limit 8 --v3-candidate-limit 16 --v3-initial-width 6 --v3-radius 2 --seed 42
```

## 검증

```bash
python -m unittest discover -s tests -v
```

V3의 다음 판단 기준은 V2 직접 대결 결과, 판당 실행시간, 평균 수, 그리고 후보 생성/rollout profile이다.
