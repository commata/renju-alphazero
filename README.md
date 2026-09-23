# Renju AlphaZero

렌주 규칙 엔진과 두 사람이 터미널에서 대국할 수 있는 초기 버전입니다. Random/Tactical 기준선 에이전트를 제공하며 강화학습 모델과 MCTS는 아직 구현 전입니다. 전체 계획은 [ROADMAP.md](ROADMAP.md)를 참고하세요.

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
- 돌을 전부 두거나 현 차례에 합법수가 없으면 무승부로 종료합니다. 금수를 시도하면 상태를 바꾸지 않고 다시 입력받습니다.
- `Game.play(row, col)`, `Game.undo()`, `Game.legal_moves()`를 제공합니다. Python 좌표는 0부터 시작하고 `board[row][col]`의 돌 값은 흑 `1`, 백 `-1`, 빈칸 `0`입니다. 추후 탐색기의 행동 번호는 `row * 15 + col`로 변환할 수 있습니다.

규칙의 세부 판정과 현재 구현의 범위는 [docs/rules.md](docs/rules.md)에 기록했습니다.

## 2단계 기준선 및 벤치마크

위의 editable 설치 후 저장소 루트에서 실행합니다. 외부 런타임 의존성은 없습니다.

```bash
python scripts/run_baseline.py --games 100 --seed 42
python scripts/benchmark_engine.py --iterations 1000 --games 50 --seed 42
```

- 기준선은 **대결별** 지정 판수로 Random(흑) vs Random(백), Random(흑) vs Tactical(백), Tactical(흑) vs Random(백)을 실행합니다. 승/무/패, 총 수, 평균 수, 경과 시간, 초당 대국 수와 수순·결과 SHA256을 출력합니다. 기본값은 `--games 100 --seed 42`입니다.
- Random은 `Game.legal_moves()` 중 균등 선택합니다. Tactical은 자기 즉시 승리 → 합법적으로 막을 수 있는 상대 즉시 승리 → 무작위 순입니다. 여러 전술 후보가 있으면 행·열 순으로 첫 수를 선택하며, 여러 위협을 모두 방어하는 탐색은 하지 않습니다. 복사한 상태에서 양쪽 합법수와 기존 `run_length()` 판정을 사용하므로 원본 대국을 변경하지 않습니다.
- 에이전트는 각자 `random.Random(seed)`를 소유합니다. runner는 master seed에서 각 판·색별 seed를 생성해 매 판 새 에이전트와 `Game()`을 만듭니다. 같은 코드·Python 환경·seed·판수에서 승패와 전체 수순이 재현되며 시간 값은 비교 대상이 아닙니다. 전역 난수 상태를 변경하지 않습니다.
- 벤치마크는 기존 엔진 그대로 `play()+undo()` 한 쌍, 빈 보드의 흑 합법수, 1수 후 백 합법수, 정상 대국 60/61수 후 흑/백 합법수, 개별 금수 판정, Random 전체 대국 처리량을 측정합니다. 상태 준비와 에이전트 생성은 측정에서 제외합니다. 중반 준비 중 종료되면 seed를 증가시켜 재시도하고 실제 seed를 출력합니다.
- 벤치마크 기본값은 노트북용 `--iterations 20 --games 2 --seed 42`입니다. 위의 1000회/50판 명령은 큰 측정이므로 먼저 아래 작은 값으로 실행하세요. `--profile`을 붙이면 별도 한 판의 `cProfile` 상위 누적 시간을 출력합니다. 프로파일 시간에는 계측 오버헤드가 있고 호출 관계가 중첩되므로 일반 벤치마크 시간과 합산하지 않습니다.

```bash
python -m unittest discover -s tests -v
python scripts/run_baseline.py --games 20 --seed 42
python scripts/benchmark_engine.py --iterations 100 --games 10 --seed 42
```

Python API는 `Agent.select_move(game) -> tuple[int, int]`, `play_game(black_agent, white_agent)`, `run_match(black_factory, white_factory, games=100, seed=42)`입니다. factory에는 seed를 위치 인자로 전달합니다. 예:

```python
from agents import RandomAgent, TacticalAgent
from evaluation import run_match

result = run_match(RandomAgent, TacticalAgent, games=2, seed=42)
print(result.black_wins, result.white_wins, result.draws)
print(result.results[0].history)  # 0부터 시작하는 (행, 열) 수순
```

합법수가 없는 상태에서 에이전트를 호출하거나 에이전트가 불법 수를 반환하면 `IllegalMove` 예외를 발생시킵니다. 결과의 `winner=None`은 무승부입니다. 기존 규칙의 재귀적 삼삼 판정 한계는 그대로이며 [규칙 문서](docs/rules.md)를 참고하세요.

실제 실행 환경, 기준선 성적과 엔진 측정값은 [2단계 측정 기록](docs/baseline.md)에 정리합니다.

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
