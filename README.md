# Renju AlphaZero

렌주 규칙 엔진, Random/Tactical 기준선과 **신경망 없는 순수 MCTS**까지 구현한 개발 버전입니다. 현재 MCTS의 기본 탐색 예산은 착수당 10회이며, 정책·가치 신경망과 강화학습은 아직 구현 전입니다. 전체 계획은 [ROADMAP.md](ROADMAP.md)를 참고하세요.

## 실행

Python 3.10 이상이 필요합니다. 저장소 루트에서:

```bash
python -m pip install -e .
python -m renju
```

`8 8`처럼 **행 열**을 1부터 15까지 입력합니다. `u`는 한 수 되돌리기, `q`는 종료입니다. 두 명이 번갈아 입력합니다. 테스트는 `python -m unittest discover -s tests -v`로 실행합니다.

## 현재 규칙과 엔진 인터페이스

- 15×15, 흑 선공, 자유 착수. 공식 대회 개국 절차와 시간 제한은 적용하지 않습니다.
- 흑은 정확히 5목으로 승리하며 장목·사사·삼삼 착수를 금지합니다. 백은 5목 이상으로 승리합니다. 흑의 장목 판정은 5목보다 우선하며, 장목이 없는 정확히 5목은 다른 금수보다 우선합니다.
- 삼삼 판정은 열린 3의 연장 수가 다시 금지된 삼삼을 만드는 경우까지 재귀적으로 검사합니다. 금수인 연장으로만 완성되는 겉보기 3은 유효한 열린 3으로 세지 않습니다.
- 돌을 전부 두거나 현 차례에 합법수가 없으면 무승부로 종료합니다. 금수를 시도하면 상태를 바꾸지 않고 다시 입력받습니다.
- `Game.play(row, col)`, `Game.undo()`, `Game.legal_moves()`를 제공합니다. Python 좌표는 0부터 시작하고 `board[row][col]`의 돌 값은 흑 `1`, 백 `-1`, 빈칸 `0`입니다.

규칙의 세부 판정과 현재 구현의 범위는 [docs/rules.md](docs/rules.md)에 기록했습니다.

## 2단계 기준선 및 벤치마크

```bash
python scripts/run_baseline.py --games 20 --seed 42
python scripts/benchmark_engine.py --iterations 100 --games 10 --seed 42 --profile
```

Random은 합법수 중 균등 선택하고 Tactical은 자기 즉시 승리 → 상대 즉시 승리 방어 → seeded random 순으로 선택합니다.

## 3단계 순수 MCTS

`MCTSAgent`는 신경망 없이 UCT Selection → Expansion → Rollout → Backpropagation을 수행합니다. 기본값은 **착수당 10 simulations**, **노드당 최대 8개 탐색 후보**입니다.

2차 구현에서는 작은 10회 예산에서도 UCT 재방문이 발생하도록 지역성 기반 shortlist를 사용하고, 루트의 즉시 승리/단일 즉시 위협 방어와 rollout의 간단한 승리/방어 정책을 추가했습니다. 이 shortlist는 탐색 휴리스틱일 뿐 렌주 합법수 규칙을 변경하지 않습니다.

```bash
python scripts/run_mcts.py --games 1 --simulations 10 --candidate-limit 8 --seed 42
python scripts/run_mcts.py --games 1 --simulations 10 --candidate-limit 8 --seed 42 --include-tactical
```

Python에서 직접 사용할 수도 있습니다.

```python
from agents import MCTSAgent

agent = MCTSAgent(seed=42, simulations=10, candidate_limit=8)
move = agent.select_move(game)
```

현재 10회 탐색은 구조 및 CPU 비용 검증을 위한 작은 예산입니다. 자세한 설계와 1차 측정 결과는 [3단계 MCTS 문서](docs/mcts.md)를 참고하세요.

Python API의 공통 에이전트 인터페이스는 `Agent.select_move(game) -> tuple[int, int]`입니다. 자동 대국은 `play_game()`, `run_match()`를 사용합니다.

## 커밋 규칙

`유형: 한글 메시지` 형식으로 작성합니다. 예: `feat: 렌주 대국 엔진과 콘솔 화면 추가`

| 유형 | 용도 |
| --- | --- |
| `feat` | 새로운 기능 추가 |
| `fix` | 버그 수정 |
| `docs` | 문서 수정 |
| `style` | 코드 형식, 세미콜론 등 동작에 영향 없는 수정 |
| `refactor` | 코드 리팩토링 (구조 개선) |
| `test` | 테스트 코드 추가 및 수정 |
| `chore` | 빌드 업무 수정, 패키지 매니저 설정 등 자잘한 수정 |

유형은 소문자 영문으로 쓰고 콜론 뒤의 메시지는 한글로 작성합니다.
