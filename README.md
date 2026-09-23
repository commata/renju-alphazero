# Renju AlphaZero

렌주 규칙 엔진, Random/Tactical 기준선과 **신경망 없는 순수 MCTS**를 개발 중인 프로젝트입니다. 현재 MCTS는 검증 기준 V2와 실험용 V3를 함께 유지해 직접 비교할 수 있습니다. 전체 계획은 [ROADMAP.md](ROADMAP.md)를 참고하세요.

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

V2는 기존 검증 기준본입니다.

- simulations: 10
- candidate limit: 8

```bash
python scripts/run_mcts.py --games 5 --simulations 10 --candidate-limit 8 --seed 42 --include-tactical
```

## MCTS V3

V3는 V2를 덮어쓰지 않고 별도 `MCTSV3Agent`로 구현합니다.

기본 설정:

- simulations: **25**
- candidate pool: **16**
- progressive widening initial width: **6**
- neighborhood radius: **2**

탐색 횟수를 10→25로 늘리면서 후보 pool도 8→16으로 넓혔습니다. 다만 16개를 처음부터 전부 확장하지 않고 progressive widening으로 방문 수가 증가할 때 후보를 단계적으로 추가해 UCT 재탐색을 유지합니다.

V2와 V3 직접 대결:

```bash
python scripts/run_mcts_versions.py --games 5 --v2-simulations 10 --v3-simulations 25 --v2-candidate-limit 8 --v3-candidate-limit 16 --v3-initial-width 6 --v3-radius 2 --seed 42
```

Python API:

```python
from agents import MCTSV2Agent, MCTSV3Agent

v2 = MCTSV2Agent(seed=42, simulations=10, candidate_limit=8)
v3 = MCTSV3Agent(
    seed=42,
    simulations=25,
    candidate_limit=16,
    initial_width=6,
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
