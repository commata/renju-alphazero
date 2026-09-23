# Renju AlphaZero

렌주 규칙 엔진, Random/Tactical 기준선과 **신경망 없는 순수 MCTS**를 개발 중인 프로젝트입니다. 현재 MCTS는 V2, 검증된 V3.1, 색상별 전술 우선순위를 추가한 V3.2를 함께 유지해 직접 비교할 수 있습니다. 전체 계획은 [ROADMAP.md](ROADMAP.md)를 참고하세요.

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
- priority top-k: **5**

탐색 횟수를 10→25로 늘리면서 후보 pool도 8→16으로 넓혔습니다. 16개를 처음부터 전부 확장하지 않고 progressive widening으로 후보를 단계적으로 추가합니다. 새 후보를 열 때는 아직 보지 않은 후보 중 상위 5개만 대상으로 5:4:3:2:1 순위 가중 랜덤을 적용해, 우선순위를 유지하면서도 탐색 다양성을 남깁니다.

V2와 V3 직접 대결:

```bash
python scripts/run_mcts_versions.py --games 5 --v2-simulations 10 --v3-simulations 25 --v2-candidate-limit 8 --v3-candidate-limit 16 --v3-initial-width 6 --v3-radius 2 --v3-priority-top-k 5 --seed 42
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
    priority_top_k=5,
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


## MCTS V3.2

V3.2는 V3.1의 25 simulations / 후보 16개 / progressive widening / top-5 가중 선택을 그대로 유지하면서 후보 순위에 렌주 색상 비대칭을 반영합니다.

- 흑: 공격형. 합법적인 43을 가장 강한 비승리 공격 패턴으로 우선하며, 4와 열린 3을 뒤따르게 합니다. 33/44/장목은 규칙 엔진이 후보에서 제거합니다.
- 백: 방어형. 흑의 43/4/3 차단에 높은 점수를 주고, 동시에 백에게 합법인 44와 33을 적극적으로 우선합니다.
- 백 5목 이상은 즉시 승리이며, 승리 수가 여러 개면 6목 이상처럼 더 긴 run을 tie-break로 우선합니다.
- 상세 43/44/33 분석은 저비용 prefilter 이후 bounded candidate set에만 적용해 rollout 전체 비용 폭증을 피합니다.

V3.1과 V3.2 직접 대결:

```bash
python scripts/run_mcts_v3_policies.py --games 25 --simulations 25 --candidate-limit 16 --initial-width 6 --radius 2 --priority-top-k 5 --seed 42
```

Python API:

```python
from agents import MCTSV3Agent, MCTSV32Agent

v31 = MCTSV3Agent(seed=42)
v32 = MCTSV32Agent(seed=42)
```
