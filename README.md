# Renju AlphaZero

렌주 규칙 엔진, Random/Tactical 기준선과 **신경망 없는 순수 MCTS**를 개발 중인 프로젝트입니다. 현재 MCTS는 검증된 V2와 실험용 V3를 함께 유지해 직접 비교할 수 있습니다. 전체 계획은 [ROADMAP.md](ROADMAP.md)를 참고하세요.

## 실행

Python 3.10 이상이 필요합니다. 저장소 루트에서:

```bash
python -m pip install -e .
python -m renju
```

`8 8`처럼 **행 열**을 1부터 15까지 입력합니다. `u`는 한 수 되돌리기, `q`는 종료입니다.

## 테스트

```bash
python -m unittest discover -s tests -v
```

## MCTS V2

V2는 10 simulations / 후보 8개 기준본입니다.

```bash
python scripts/run_mcts.py --games 5 --simulations 10 --candidate-limit 8 --seed 42 --include-tactical
```

현재 V2는 지역성 shortlist, 즉시 승리/단일 방어, 전술 rollout, private state 재사용을 사용합니다.

## MCTS V3

V3는 V2를 덮어쓰지 않고 별도 `MCTSV3Agent`로 구현합니다.

기본 설정:

- simulations: 10
- candidate pool: 12
- progressive widening initial width: 4
- neighborhood radius: 2

후보 pool을 12개로 넓히되 처음부터 모두 확장하지 않고, 노드 방문 수가 늘어날 때 후보를 점진적으로 추가합니다. 또한 rollout 후보는 전체 15×15 빈칸을 매번 정렬하지 않고 기존 돌 주변 거리 2의 local pool에서 우선 생성합니다.

V2와 V3 직접 대결:

```bash
python scripts/run_mcts_versions.py --games 5 --simulations 10 --v2-candidate-limit 8 --v3-candidate-limit 12 --v3-initial-width 4 --v3-radius 2 --seed 42
```

Python API:

```python
from agents import MCTSV2Agent, MCTSV3Agent

v2 = MCTSV2Agent(seed=42, simulations=10, candidate_limit=8)
v3 = MCTSV3Agent(
    seed=42,
    simulations=10,
    candidate_limit=12,
    initial_width=4,
    neighborhood_radius=2,
)
```

자세한 설계와 측정 기록은 [docs/mcts.md](docs/mcts.md)를 참고하세요.

## 규칙 엔진

- 15×15, 흑 선공, 자유 착수
- 흑은 정확히 5목 승리, 장목·사사·삼삼 금지
- 백은 5목 이상 승리
- 재귀적 삼삼 판정
- `Game.play()`, `Game.undo()`, `Game.legal_moves()`, `Game.has_legal_move()` 제공

규칙 세부 사항은 [docs/rules.md](docs/rules.md)를 참고하세요.

## 커밋 규칙

`유형: 한글 메시지` 형식으로 작성합니다.
