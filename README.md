# Renju AlphaZero

렌주 규칙 엔진, Random/Tactical 기준선, 동결된 **순수 MCTS-v6 benchmark**, 정책·가치 신경망을 기반으로 AlphaZero PUCT/self-play/학습 파이프라인을 단계적으로 구현하는 프로젝트입니다. 전체 계획은 [ROADMAP.md](ROADMAP.md)를 참고하세요.

## 현재 단계

- **Stage 0~3 완료** (태그 `v0.2-mcts`): 저장소·규칙 명세, 렌주 엔진, 기준선·벤치마크, 순수 MCTS
- **Stage 3 최종 baseline: MCTS-v6** — RIF exact-five 우선순위 교정 전 기준으로 회귀 테스트 147/147 PASS, V5 FINAL 상대 100판 48승 38패 14무(score 55.0%). 에이전트 버전은 고정 비교 상대로 보존합니다. 교정 후 seed 777 10판 smoke에서는 6승 2패 2무(score 70.0%)를 관측했으며, 이는 새 장기 승률이 아니라 현재 규칙 기준 smoke baseline입니다.
- **Stage 4 정책·가치 신경망 완료** (`feat/policy-value-network`): 버전 고정 6-plane 입력, residual policy/value, legal mask/loss, checkpoint, D4, eval tiny overfit 및 CPU benchmark. [계약·검증 결과](docs/policy-value-network.md)를 참고하세요. V5/V6 전술 계층은 학습 경로에서 재사용하지 않습니다.
- **Stage 5 AlphaZero 탐색 결합 구현** (`feat/stage5-alphazero-search`): MCTS-v6를 확장하지 않고 `Game`/규칙 엔진과 Stage 4 모델 계약만 공유하는 **별도 AlphaZero PUCT search**(`search.alphazero`), torch-free evaluator 경계(`search.evaluator`), 정책·가치망 evaluator(`model.evaluator`), self-play `Sample`/`GameRecord`·replay·canonical hash(`training.self_play`)를 구현했습니다. terminal value/backup은 player identity 기준이며, 계약과 구현 결정·검증 결과는 [Stage 5 AlphaZero Search Integration](docs/stage5-alphazero.md)에 있습니다.

과거 버전(V2~V5)은 비교 재현을 위해 덮어쓰지 않고 별도 Agent로 보존합니다. 설계와 측정 기록은 [docs/mcts.md](docs/mcts.md)에 있습니다.

## 실행

Python 3.10 이상이 필요합니다. 저장소 루트에서:

```bash
python -m pip install -e .
python -m renju
```

`8 8`처럼 **행 열**을 1부터 15까지 입력합니다. `u`는 한 수 되돌리기, `q`는 종료입니다.

## 테스트

기본 회귀 테스트는 Python 표준 라이브러리 `unittest`를 사용하므로 별도 `test` extra가 필요하지 않습니다. Torch가 없는 환경에서는 neural 전용 테스트가 skip되고 engine/search 회귀는 계속 실행됩니다.

```bash
python -m unittest discover -s tests -v
```

## CI

GitHub Actions는 `.github/workflows/ci.yml`에서 다음 4개 고정 job을 실행합니다. Required Check는
job 이름을 그대로 사용할 수 있도록 matrix를 쓰지 않습니다.

- `core-no-torch`: Python 3.10, torch가 설치되지 않았음을 먼저 확인한 뒤 전체 회귀를 실행합니다.
  skip은 사유가 정확히 `requires torch`인 neural 테스트만 허용합니다.
- `full-tests`: Python 3.13 + 고정 CPU torch 2.14.0에서 전체 회귀를 실행하며 **skip 0**을 강제합니다.
- `fake-smoke`: uniform evaluator, seed 42, 8 simulations의 Stage 5 game SHA256을 golden 값과 비교합니다.
- `frozen-baseline`: V5/V6/threat 구현 파일 SHA-256 잠금과 seed 777, 1 game per color,
  reduced simulation budget의 V6-vs-V5 outcome/history fingerprint
  `91bb91e55a8e73a3fa87dce6a839623170746a712abbad0669c63008b509bc44`를 검증합니다. 기존 post-RIF 10게임 `fd3f8ee6...851d0f`는 장기 검증 기록으로
  문서에 계속 보존합니다.

CI 러너는 테스트 개수를 하드코딩하지 않습니다.

```bash
python scripts/ci_run_tests.py --skip-policy none
python scripts/check_frozen_baseline.py
python scripts/ci_v6_behavior.py --expect-sha256 91bb91e55a8e73a3fa87dce6a839623170746a712abbad0669c63008b509bc44
python scripts/run_stage5_self_play.py --seed 42 --simulations 8 \
  --expect-sha256 7e1b04a0fedc3e8c11a22069222ecbf4dbcfb74f0efe3907c9dcfead124b4510
```

`tests/frozen_baseline.sha256` 또는 golden fingerprint가 달라졌다고 해서 CI를 통과시키기 위해
값만 갱신하면 안 됩니다. 의도적인 benchmark 변경인지 원인을 먼저 확인하고, 변경 이유와 새 검증 기록을
같은 PR에 남긴 뒤 잠금값을 갱신합니다.

## Stage 5 self-play smoke

`uniform` evaluator 모드는 torch 없이 동작합니다. neural 모드는 checkpoint **파일**을 사용하고 그 SHA256을 기록합니다.
checkpoint는 `checkpoints/`, 결과 JSON은 `logs/stage5/`에 저장되며 둘 다 커밋하지 않습니다.

```bash
python scripts/run_stage5_self_play.py --seed 42 --simulations 8
python scripts/run_stage5_self_play.py --evaluator neural   --create-random-checkpoint checkpoints/stage5_random_init_seed0.pt --model-seed 0   --seed 42 --simulations 64 --threads 1
python scripts/run_stage5_self_play.py --evaluator neural   --checkpoint checkpoints/stage5_random_init_seed0.pt --seed 42 --simulations 64 --threads 1
```

같은 commit/checkpoint/config/seed/device/thread 수/batch size에서 두 실행의 `game_sha256`이 같아야 합니다.

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

- 15×15, 흑 선공. **흑 첫 수는 정중앙 (8, 8)으로 고정**, 이후 자유 착수
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


## 경기 로그

MCTS 비교 runner는 별도 옵션 없이 CSV와 JSON 로그를 자동 저장합니다.

```text
logs/
├─ mcts_versions/<timestamp>_seedN/
│  ├─ games.csv
│  ├─ moves.csv
│  └─ games.json
└─ mcts_v3_policies/<timestamp>_seedN/
   ├─ games.csv
   ├─ moves.csv
   └─ games.json
```

- `games.csv`: 경기별 요약과 실행 시간
- `moves.csv`: 모든 착수를 ply 단위로 저장
- `games.json`: 실험 파라미터와 전체 경기/착수 로그
- CSV 좌표는 0-based(`row0/col0`)와 1-based(`row/col`)를 함께 제공합니다.

필요하면 `--log-dir`로 저장 경로를 지정할 수 있습니다.


## MCTS V3.2.1

V3.2.1은 V3.2의 흑 공격형 / 백 방어형·33·44 정책을 유지하면서 cProfile에서 확인된 금수 판정 재귀 병목을 최적화한 버전입니다.

- 실제 MCTS 후보 합법성: 기존 Renju rules로 정확히 검사
- 43/44/33 후보 순위용 가상 extension: 빠른 구조 스캐너 사용
- 가상 비승리 extension에서 재귀 `forbidden_reason()` 제거
- 흑 장목은 빠른 run-length 검사로 제거
- V3.2는 비교 기준으로 보존
- V3.1 로그 이름도 `MCTS-v3.1`로 명확하게 수정

V3.2와 V3.2.1 속도/기력 비교:

```bash
python scripts/run_mcts_v32_optimization.py --games 1 --simulations 25 --candidate-limit 16 --initial-width 6 --radius 2 --priority-top-k 5 --seed 42
```


## MCTS V4~V6

V4 이후 버전도 이전 버전을 덮어쓰지 않고 별도 Agent로 유지합니다.

- **V4.1 / V4.2**: 내 즉시 승리 → 상대 즉시 승리 차단 → 상대 열린4 생성점 차단 순의 강제수 뒤 V3.2.1 탐색
- **V5 FINAL**: 5단계 강제 정책(unstoppable four, 이중 위협 예방)과 adaptive simulations.
  비교 기준 설정은 50/100 simulations, threshold 1800, 후보 20, 초기 폭 8, top-k 8
- **V6**: V5 강제 정책 뒤에 43/44/33 위협 탐지, 2-ply threat planning,
  백의 흑 43 방어 coverage를 root 후보에 주입

V6 vs V5 FINAL 비교:

```bash
python scripts/run_mcts_v6_vs_v5.py --games 5 --seed 42
```

## Stage 3 최종 baseline: MCTS V6

| 검증 | 결과 |
| --- | --- |
| 회귀 테스트 | 147 / 147 PASS |
| V6 vs Random | 40승 0패 0무 (흑·백 각 20판) |
| V6 vs Tactical | 100승 0패 0무 (흑·백 각 50판) |
| V6 vs V5 FINAL (pre-RIF) | 48승 38패 14무, score 55.0% (100판) |
| 결정성 (pre-RIF) | seed 777 10판 2회 실행 SHA256 일치 |
| V6 vs V5 FINAL (post-RIF smoke) | seed 777, 10판: 6승 2패 2무, score 70.0% |
| 결정성 (post-RIF) | seed 777 10판 2회 실행 SHA256 `fd3f8ee6...851d0f` 일치 |

MCTS-v6는 이후 신경망 체크포인트의 성장 정도를 측정하는 **고정 benchmark opponent**로 동결합니다.
기존 100판 승패와 기존 seed 777 SHA는 RIF exact-five 교정 이전 측정값입니다. 교정 후 seed 777 10판을 두 번 실행해 `(winner, history)` SHA256이 동일함을 확인했으며, `fd3f8ee61cb954c7c91249eeb79f420b5829c43f361f50e1055ebfbafa851d0f`를 post-RIF 결정성 baseline으로 사용합니다.
세부 기록은 [docs/mcts.md](docs/mcts.md)의 "Stage 3 Final Baseline"을 참고하세요.

## 로컬 웹 대국

브라우저에서 V3.2.1 / V4.1 / V4.2 / V5 FINAL / V6와 직접 대국할 수 있습니다.
외부 패키지 없이 Python 표준 라이브러리 서버를 사용합니다.

```bash
python scripts/run_web_play.py
```

`http://127.0.0.1:8000`에 접속합니다. 완료된 대국은 `logs/web_play/`에 JSON/CSV로 자동 저장됩니다.
자세한 내용은 [web/README.md](web/README.md)를 참고하세요.

## 규칙 엔진 성능 검증

합법수 집합과 행 우선 순서를 유지하면서 금수 판정의 불필요한 호출을 줄였습니다.
동결 reference oracle, 단계별 측정값과 정확성 근거는
[최적화 검증 기록](docs/rule-optimization.md)을 참고하세요.

```powershell
python -m unittest discover -s tests -v
python scripts/validate_rule_optimization.py --seed 42 --positions 10000
python scripts/benchmark_rule_optimization.py --seed 42 --iterations 20 --repeats 5 --games 2
python scripts/benchmark_rule_optimization.py --seed 42 --smoke
python scripts/benchmark_engine.py --iterations 100 --games 10 --seed 42 --profile
```

별도 benchmark는 같은 프로세스에서 reference/optimized를 교대로 실행하며,
준비·warm-up을 제외한 `perf_counter()` 측정의 mean/median/min/max와 ops/sec를 출력합니다.
`--smoke`는 V6의 흑·백 각 한 판을 simulations/tactical_simulations=1로 실행하여
reference와 optimized의 승자 및 전체 기보를 비교합니다.
