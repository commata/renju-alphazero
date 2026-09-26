# Stage 5 AlphaZero Search Integration

Stage 5의 목표는 **학습 가능한 self-play 데이터를 생성하는 새로운 AlphaZero 탐색 경로**를 완성하는 것이다.
이 단계의 성공 기준은 기력이 아니라 탐색/데이터 계약의 정확성, 재현성, 규칙 준수다.

Stage 3의 **MCTS-v6는 고정 benchmark opponent로 동결**한다. Stage 5는 V6를 확장하거나
V5/V6의 강제수·전술 planner를 상속하지 않는다. 공유하는 것은 `Game` 엔진, 렌주 규칙,
합법수 판정과 Stage 4의 정책·가치망 계약뿐이다.

## 1. 범위와 비범위

Stage 5에 포함한다.

- 별도 AlphaZero MCTS와 PUCT
- torch-free `Evaluator` 추상화와 fake evaluator
- Stage 4 `PolicyValueNet` 연결
- legal move, terminal value, backup 관점 계약
- root visit distribution `pi`
- self-play용 temperature와 root Dirichlet noise
- batch evaluation 인터페이스
- 한 판의 `Sample` / `GameRecord` 생성, 저장, replay 검증
- 탐색 시간 구성요소 계측

Stage 5에 포함하지 않는다.

- V5/V6 Stage 1~5 강제수, 43/44/33 planner를 self-play 경로에 삽입
- replay buffer와 optimizer 학습 루프
- checkpoint 승급 arena
- virtual loss 기반 병렬 tree search
- 수 사이 tree reuse
- 기력 승률을 Stage 5 PASS gate로 사용

V6는 Stage 6 이후에도 **비교 평가용 baseline**으로만 사용한다.

## 2. 기존 Stage 4 계약

Stage 5는 `docs/policy-value-network.md`와 현재 코드를 그대로 따른다.

- encoder: `renju-relative-6p-v1`
- action index: `row-major-15x15-v1`
- policy output: raw logits `[B,225]`
- value output: `[B,1]`, **현재 `game.to_play` 관점**
- plane 5: 현재 상태의 legal action mask
- leaf 평가 시 `Game.legal_moves()`를 한 번 계산하고 그 목록으로 만든 동일 mask를
  encoder plane 5와 policy masking에 재사용
- terminal/no-legal 상태에서는 policy softmax를 호출하지 않음
- neural inference는 `model.eval()` + `torch.inference_mode()` 사용

## 3. Stage 3 코드 및 torch 의존성과의 경계

Stage 5 search는 `src/search/`에 **새 구현**으로 추가한다. 기존 `mcts_v6.py`를 수정해
학습 MCTS로 바꾸지 않는다.

core search와 fake evaluator는 torch가 설치되지 않은 engine-only 환경에서도 import/실행돼야 한다.
Stage 4의 `src/model/__init__.py`도 같은 이유로 torch를 직접 import하지 않는다.

권장 경로:

```text
src/
├─ search/
│  ├─ mcts_v6.py              # 동결 benchmark
│  ├─ alphazero.py            # 신규 PUCT, torch import 금지
│  └─ evaluator.py            # Protocol + Uniform/Scripted, torch import 금지
├─ model/
│  ├─ ...                     # Stage 4 계약
│  └─ evaluator.py            # PolicyValueEvaluator, 여기서만 torch/model 사용
└─ training/
   └─ self_play.py            # record 생성; evaluator interface에만 의존

tests/
├─ test_alphazero_search.py
├─ test_alphazero_evaluator.py
└─ test_self_play_record.py
```

공식 회귀 명령은 저장소의 기존 방식과 동일하게 유지한다.

```bash
python -m unittest discover -s tests -v
```

기본 회귀는 표준 라이브러리 `unittest`를 사용하므로 `pytest`는 필수 의존성이 아니다.
neural extra가 없을 때도 core PUCT/fake evaluator 테스트는 skip 없이 실행되어야 한다.

## 4. Evaluator 추상화

PUCT의 정확성을 PyTorch와 분리해 검사할 수 있도록 먼저 evaluator 경계를 둔다.
core interface는 torch tensor를 노출하지 않는다.

```python
class Evaluator(Protocol):
    def evaluate(self, game, legal_moves):
        """Return priors[225] and value from game.to_play perspective."""

    def evaluate_batch(self, items):
        """Same contract for a batch; Stage 5 search may still use batch size 1."""
```

테스트용 evaluator:

- `UniformEvaluator`: 합법수에 균등 prior, value `0`
- `ScriptedEvaluator`: 지정 action에 prior/value를 고정
- 실제 `PolicyValueEvaluator`: Stage 4 encoder/network/masking 사용

`PolicyValueEvaluator`는 search가 넘긴 `legal_moves`로 torch legal mask를 한 번 만들고,
그 **같은 mask**를 `encode_game(..., legal_mask)`와 `masked_softmax`에 재사용한다.
따라서 `Game.legal_moves()`를 evaluator 내부에서 다시 호출하지 않는다.

fake evaluator로 PUCT, terminal, legal mask, `pi`, tie-break를 먼저 검증하고
torch 통합 테스트는 그 다음에 둔다.

## 5. Node / edge 통계와 PUCT 규약

### 5.1 관점과 terminal cache

Stage 5에서는 child node에 저장된 통계를 **그 child로 들어오는 action을 둔 플레이어,
즉 parent의 `to_play` 관점**으로 정의한다.

```text
child.prior
child.visit_count
child.value_sum        # player_who_moved 관점
child.player_who_moved
child.to_play          # 이 node 상태의 Game.to_play
child.is_expanded
child.is_terminal
child.terminal_value   # terminal이면 child.to_play 관점, 아니면 None
child.q = value_sum / visit_count
```

`is_expanded`를 `children이 비었는가`로 대체하지 않는다. terminal node는 정상적으로
expanded 상태이면서 children이 0개일 수 있기 때문이다.

root는 들어오는 action이 없으므로 `root.value_sum`의 의미를 두지 않는다.
root의 `visit_count`는 완료된 descent simulation 수를 세는 용도로만 사용한다.

이 규약을 사용하면 parent selection에서 child `Q`를 그대로 최대화할 수 있다.

### 5.2 PUCT

프로젝트의 첫 구현은 다음 규약으로 고정한다.

```text
score(parent, child)
  = Q(child)
  + c_puct * P(child)
    * sqrt(max(1, N(parent)))
    / (1 + N(child))
```

- FPU(first-play urgency): `0`
- `N(parent)==0`: `sqrt(max(1, N(parent)))` 사용
- PUCT 동점: score 큰 순 → prior 큰 순 → action index 작은 순
- 모든 child는 `Game.legal_moves()`에서 나온 action만 생성
- Stage 5에서는 progressive widening과 V6 후보 주입을 사용하지 않음

위 `N=0`, FPU, tie-break는 **이 프로젝트의 재현성 규약**이며 AlphaZero 계열 구현 전체의
유일한 정답이라고 주장하지 않는다.

### 5.3 Root 초기화와 simulation 예산

`num_simulations`의 의미가 구현마다 달라지지 않도록 다음 순서를 고정한다.

일반적인 non-terminal root에서 합법수가 2개 이상이면:

```text
1. root legal_moves 계산
2. evaluator로 root를 1회 확장
   - 이 호출은 simulation에 포함하지 않음
   - root evaluator의 value는 이 사전 확장 자체로 backup하지 않음
3. self-play면 합법 root prior에 Dirichlet noise 적용
4. 정확히 num_simulations회 selection/descent/leaf evaluation/backup
5. root.visit_count == num_simulations
6. sum(root child visit_count) == num_simulations
```

따라서 저장할 root visit counts의 합은 searched root에서 항상 `num_simulations`이다.

### 5.4 합법수가 하나뿐인 root fast path

현재 규칙은 초기 흑 착수를 중앙 `(7,7)` 하나로 강제한다. 이후에도 드물게 합법수가
하나만 남는 상태가 생길 수 있다. 이런 root에는 다음 fast path를 적용한다.

- PUCT simulation을 실행하지 않음
- evaluator를 호출하지 않음
- Dirichlet noise와 temperature sampling을 적용하지 않음
- 유일한 합법 action을 바로 선택
- 학습 sample은 **그대로 저장**
- 저장 visit counts는 유일한 action에 synthetic count `1`, 나머지 `0`
- 따라서 derived `pi`는 one-hot

이 sample의 policy 정보는 자명하지만 이후 실제 결과로 채워지는 value target `z`에는 의미가 있으므로
버리지 않는다.

searched root의 `sum(visits)==num_simulations` invariant와 단일 합법수 fast path의
`sum(visits)==1`은 서로 다른 명시적 계약이다.

## 6. Terminal value와 backup

### 6.1 현재 Game의 중요한 의미

현재 `Game.play()`는 승리 착수에서 `to_play`를 상대에게 넘기지 않는다.

```text
승리 terminal:
game.winner == 방금 둔 플레이어
game.to_play == 방금 둔 플레이어
```

반면 비승리 착수 뒤 합법수가 없어 무승부가 되면 `to_play`는 이미 다음 플레이어로
전환된 상태다.

따라서 terminal value를 "직전 수를 둔 쪽이 이겼으니 현재 차례 관점 -1"처럼
ply 홀짝만으로 계산하면 현재 엔진에서 부호 오류가 난다.

terminal value는 player identity로 정의한다.

```python
def terminal_value(game, perspective):
    if not game.done:
        raise ValueError
    if game.winner is None:
        return 0.0
    return 1.0 if game.winner == perspective else -1.0
```

leaf가 terminal이면 NN을 호출하지 않는다. 현재 leaf의 value 기준 플레이어는
`leaf_game.to_play`로 두고 위 함수로 값을 얻는다.

### 6.2 Terminal node 재방문

terminal child를 처음 만들 때 `is_terminal=True`, `to_play`, `terminal_value`를 캐시한다.
selection이 이후 같은 terminal node에 다시 도달하면 그 자리에서 descent를 끝낸다.

- 빈 children에 대해 `max()`나 추가 expansion을 시도하지 않음
- `Game.legal_moves()`와 evaluator/NN을 다시 호출하지 않음
- 캐시된 terminal value와 player identity로 바로 backup

즉 terminal node 방문 횟수가 증가해도 evaluator 호출 수는 증가하지 않아야 한다.

### 6.3 Backup

단순히 매 ply마다 무조건 `v = -v`만 수행하지 않는다. terminal 승리에서 현재 엔진은
`to_play`를 유지하므로, selected edge의 실제 `player_who_moved`와 leaf value의
`perspective`를 비교해 부호를 정한다.

개념 규약:

```python
leaf_player = leaf_node.to_play
leaf_value = (
    leaf_node.terminal_value
    if leaf_node.is_terminal
    else evaluator_value  # leaf_player 관점
)

root.visit_count += 1
for child in reversed(selected_children):  # root 제외
    child.visit_count += 1
    signed = leaf_value if child.player_who_moved == leaf_player else -leaf_value
    child.value_sum += signed
```

두 플레이어 zero-sum 게임이라는 전제에서 이 방식은 non-terminal leaf와 현재 엔진의
terminal 승리 의미를 모두 명시적으로 처리한다.

## 7. Leaf expansion과 legal move invariant

한 non-terminal leaf에서 다음 순서를 고정한다.

```text
1. node.is_terminal 확인
2. terminal이면 cache value로 backup, evaluator 호출 금지
3. legal_moves = game.legal_moves() 한 번 계산
4. legal_moves가 비어 있으면 RuntimeError/AssertionError: Game invariant 위반
5. evaluator(game, legal_moves)
6. neural evaluator라면 legal_moves로 mask 생성
7. 같은 mask로 encode + prior masking/softmax
8. 합법 child만 확장
```

현재 `Game.play()`는 비승리 착수 뒤 `has_legal_move()==False`이면 즉시 `done=True`로 만든다.
초기 상태도 중앙 한 수가 합법이다. 따라서 정상적인 production `Game`에서
`game.done == False`인데 `game.legal_moves() == []`인 상태는 존재하면 안 된다.

Stage 5 search는 이 경우를 조용히 무승부로 보정하지 않는다. 엔진/search 상태 오염을 숨기지 않도록
명시적으로 예외를 발생시킨다.

항상 유지할 invariant:

- illegal prior = 정확히 `0`
- legal prior 합 = `1` (허용오차 `1e-5`)
- 금수 action은 child로 생성되지 않음
- search가 반환하는 action은 해당 root의 합법수
- simulation state 이동은 보드 deepcopy 반복 대신 기존 `Game.play()` / `Game.undo()`를 사용

## 8. Search policy target과 실제 착수 분리

학습 target은 temperature와 분리한다. canonical 저장값은 정수 visit counts다.

```text
pi_target(a) = visit_counts(a) / sum_b visit_counts(b)
```

즉 `pi`는 저장된 정수 counts에서 파생한다. `pi`는 불법 위치 `0`, 전체 합 `1`이어야 한다.
searched root에서는 counts 합이 `num_simulations`, 단일 합법수 fast path에서는 합이 `1`이다.

실제 self-play 착수 선택에만 temperature를 적용한다.

```text
selection_prob(a) ∝ visit_counts(a)^(1 / tau)
```

초기 설정 후보:

- `tau = 1.0`
- `temperature_moves = 10`
- 이후 deterministic `argmax(visit_counts)`

`temperature_moves`는 **0-based ply**, 즉 착수 직전 `len(game.history)`를 기준으로 한다.
`ply < temperature_moves`인 착수에 temperature 정책을 적용한다.

따라서 `temperature_moves=10`이면 강제 중앙 첫 수(ply 0)도 window에는 포함되지만
합법수가 하나뿐이라 실제 난수 선택은 발생하지 않는다. 정상적으로 여러 합법수가 있다면
초기 10 plies 안에서 실제 stochastic decision은 최대 9번이다.

temperature 종료 뒤 `argmax` 동점은 다음으로 고정한다.

```text
visit_count 큰 순 → prior 큰 순 → action index 작은 순
```

첫 수는 모든 기보에서 동일하므로 self-play 기보 다양성/개국 entropy는 **2수째(ply 1)부터**
측정하고, 강제 중앙 착수를 다양성 지표에 포함하지 않는다.

`temperature_moves=10`은 프로젝트 초기값일 뿐 Stage 5 완료 기준이 아니다.
실제 대국 길이와 다양성을 측정해 Stage 6에서 조정한다.

## 9. Root Dirichlet noise와 RNG

noise는 self-play root에만 적용하고 eval/search 검증 모드에서는 끈다.

```text
P'(a) = (1 - epsilon) * P(a) + epsilon * eta(a)
eta ~ Dirichlet(alpha)
```

초기 실험값:

```text
epsilon = 0.25
alpha = 0.05
```

`epsilon=0.25`, `alpha=0.05`는 Renju용 **출발 설정**으로 취급한다.
특히 `alpha=0.05`를 보편적인 AlphaZero 상수로 간주하지 않는다.

적용 순서:

1. root를 evaluator로 확장
2. 합법 child 집합에 대해서만 Dirichlet sample 생성
3. 합법 prior에만 noise 혼합
4. 재정규화
5. illegal action은 계속 `0`

프로젝트는 NumPy를 필수 의존성으로 두지 않는다. Dirichlet은 search/self-play가 소유한
`random.Random(seed)` 인스턴스의 `gammavariate(alpha, 1.0)`를 합법 action 수만큼 호출하고
양수 샘플을 정규화해 생성한다.

temperature sampling도 **같은 caller-owned/local RNG 계열**을 사용하며 module-level
`random.seed()`, `random.random()` 등 전역 난수 상태를 변경하지 않는다. 이는 기존
Random/MCTS Agent가 `Random(seed)`를 소유해 search에 전달하는 패턴과 맞춘다.

Stage 5에서는 수 사이 tree reuse를 기본 OFF로 두어 root noise 재적용과 결정성 검증을 단순화한다.

## 10. Batch inference 범위

한 게임의 순차 MCTS는 자연스럽게 leaf 하나씩 발생하므로 Stage 5에서 억지로
virtual loss 병렬화를 넣지 않는다.

Stage 5 요구사항:

- core `Evaluator.evaluate_batch(...)` 인터페이스 제공
- `PolicyValueEvaluator`의 batch size 1과 N 결과가 같은 입력에서 허용오차 내 일치
- 기본 search는 batch size 1이어도 PASS
- determinism 검증에서는 **evaluator batch size도 config에 기록하고 동일하게 고정**

virtual loss, 여러 게임의 inference queue, tree-parallel search는 Stage 6 이후 성능 작업으로 남긴다.

## 11. Self-play 저장 포맷

인코딩 tensor 자체와 각 sample의 전체 prefix 기보를 중복 저장하지 않는다.
원시 전체 기보와 sample의 `ply` index, 정수 visit counts, version metadata를 저장하고
Stage 6 학습 시 해당 encoder version으로 상태와 `pi`를 다시 구성한다.

권장 논리 스키마:

```python
Sample:
    ply: int                         # 이 상태 이전의 착수 수
    to_play: int                     # BLACK / WHITE
    visit_counts: tuple[int, ...]    # len=225, illegal=0; pi의 canonical source
    action: int                      # 실제 self-play 착수
    z: float                         # 이 sample의 to_play 관점 {-1,0,+1}

GameRecord:
    seed
    model_seed
    search_config
    config_hash
    checkpoint_hash
    git_commit
    encoder_version
    action_index_version
    result
    moves: tuple[int, ...]
    samples
```

sample 상태는 `GameRecord.moves[:sample.ply]`를 새 `Game`에 재생해 복원한다.
따라서 매 sample마다 `moves_before` 전체를 복사하는 O(game_length²) 저장을 피한다.

`pi`는 저장하지 않아도 다음처럼 정확히 다시 만든다.

```python
total = sum(sample.visit_counts)
pi = [count / total for count in sample.visit_counts]
```

`total`은 searched root에서 `num_simulations`, 단일 합법수 fast path에서 `1`이어야 한다.
hash/replay의 canonical 데이터는 정수 counts를 사용하고 derived float `pi`는 결정성 hash 입력에서 제외한다.

게임 종료 후 각 sample의 `z`는 ply 홀짝이 아니라 저장한 `to_play`와 winner를 비교한다.

```python
if winner is None:
    z = 0.0
else:
    z = 1.0 if winner == sample.to_play else -1.0
```

Replay 검증은 `moves`를 `action_to_coordinate()`로 변환해 새 `Game`에 순서대로
`Game.play()`하고, 각 sample ply에서 `to_play`/합법 action/count mask를 검사한 뒤
최종 board / winner / done / history가 원 기록과 같은지 확인한다.

### 11.1 학습 전 random-init checkpoint

첫 Stage 5 self-play smoke도 in-memory random model이 아니라 **파일로 고정된 checkpoint**를 사용한다.

```text
1. torch/manual model seed 고정
2. PolicyValueNet 생성
3. model.eval()
4. save_checkpoint()로 파일 1회 저장
5. 저장된 파일 bytes의 SHA256 계산
6. 그 checkpoint를 다시 load해 self-play에 사용
7. model_seed + checkpoint_hash를 GameRecord에 기록
```

같은 seed로 checkpoint 파일 bytes를 매번 새로 저장했을 때 hash까지 같아야 한다고 요구하지 않는다.
한 번 만든 checkpoint 파일을 immutable input으로 취급하고 **실제로 사용한 파일의 hash**를 기록한다.

## 12. 재현성 범위

bit-exact 재현성은 다음 조건 안에서만 요구한다.

- 동일 Git commit
- 동일 checkpoint bytes/hash
- 동일 search config/hash
- 동일 seed와 RNG 알고리즘/호출 순서
- 동일 evaluator batch size
- 동일 device와 PyTorch/backend
- `model.eval()`
- `torch.inference_mode()`
- torch thread 수 고정
- 사용한 Python / torch / device 정보 기록
- module-level global random state를 변경하지 않음

같은 조건에서 최소한 `(winner, moves)`의 SHA256이 일치해야 한다.
필요하면 timing과 derived float `pi`를 제외하고 integer visit counts까지 포함한 별도
record fingerprint를 추가할 수 있다.

CPU ↔ Intel GPU ↔ RX 6600 등 장치가 달라지면 floating-point 차이로 visit count와
기보가 갈라질 수 있으므로 장치 간 bit-exact 일치는 Stage 5 요구사항이 아니다.

## 13. 성능 계측

Stage 4의 현재 CPU 측정에서는 `legal_moves`가 환경에 따라 대략 **2.6~3.5 ms**,
forward B1이 약 **2.9~3.6 ms**, full pipeline B1이 약 **6.2~8.5 ms** 범위로 관측됐다.
과거 Stage 3 profile의 약 19 ms/call 수치는 현재 최적화된 규칙 엔진의 Stage 5 기준값으로
사용하지 않는다.

Stage 5 self-play smoke에서 수당 시간을 최소 다음으로 나눠 기록한다.

```text
legal_moves_ms
inference_ms
tree_ms
total_move_ms
```

단일 합법수 fast path는 simulation/inference를 생략하므로 searched move와 별도로 집계한다.
성능은 PASS/FAIL의 1차 gate가 아니라 Stage 6 병목 결정을 위한 관측값이다.

## 14. 작업 분할

```text
5-0  torch-free Evaluator + Uniform/Scripted fake evaluator
5-A  AZ Node/edge stats + terminal cache + PUCT + root-expansion/simulation 규약
5-B  single-legal fast path + legal invariant + player-identity terminal/backup
5-C  PolicyValueEvaluator 연결 + evaluate_batch 동등성
5-D  integer root visits/derived pi + 착수 temperature + deterministic argmax
5-E  local-RNG legal-root Dirichlet + self-play/eval mode 분리
5-F  Sample(ply)/GameRecord + checkpoint hash + replay + determinism hash + timing
```

각 작업은 기존 V6 search를 수정하지 않고 신규 AlphaZero 경로에 추가한다.

## 15. 필수 테스트

최소 다음을 `unittest`로 고정한다.

- torch 없는 환경에서 `search.alphazero`와 fake evaluator import 가능
- Uniform evaluator에서 legal prior만 생성되고 합이 1
- Scripted evaluator의 높은 prior가 PUCT 첫 선택/tie-break 규약에 반영됨
- non-forced root 사전 확장은 simulation에 포함되지 않음
- `sum(root child visits) == num_simulations`
- 합법수가 1개면 evaluator/noise/simulation 없이 synthetic count 1의 one-hot target 생성
- 초기 `Game()`에서 강제 중앙 action만 저장되고 sample 자체는 유지됨
- terminal leaf에서는 evaluator가 호출되지 않음
- terminal node를 여러 번 다시 방문해도 evaluator 호출 수가 증가하지 않음
- 현재 `Game`의 승리 terminal에서 `game.to_play == game.winner`인 경우 value/backup 부호가 맞음
- no-legal draw의 value는 0
- `done=False`인데 legal move가 0개인 fixture는 명시적 invariant 예외
- root `pi = visits/sum(visits)`, illegal mass 0, 합 1
- temperature는 저장 counts/`pi`를 바꾸지 않고 action sampling에만 영향
- `temperature_moves`가 `len(history)` 기준임
- deterministic argmax 동점은 visit → prior → 작은 action index 순
- Dirichlet noise가 합법 root child에만 적용됨
- seeded local RNG 사용 전후 module-level global random state가 변하지 않음
- `evaluate_batch([x])`와 동일 입력의 batch N 결과가 허용오차 내 일치
- scripted/bounded fixture에서 terminal 승리와 상대 승리 회피가 올바른 Q로 전달됨
- 저장된 모든 `action`이 해당 `GameRecord.moves[:ply]` 상태에서 합법
- visit counts의 illegal 위치가 0이고 searched/forced 합계 규약이 맞음
- `z`가 `winner`와 sample `to_play` 비교로 계산됨
- GameRecord replay 결과가 원래 final board/result/history와 일치
- 동일 장치/백엔드/batch size/seed/config/checkpoint에서 `(winner,moves)` SHA256 일치
- Stage 5 search 경로가 V5/V6 forced/planner 함수를 호출하지 않음
- 기존 전체 회귀 + 신규 Stage 5 테스트 모두 PASS

현재 Stage 4 완료 기준의 전체 회귀는 torch 설치 환경에서 **174 PASS**다.
Stage 5 완료 시에는 고정 숫자 147을 gate로 쓰지 않고, 이 기존 회귀와 새 테스트가 모두
통과하는지를 확인한다.

## 16. Stage 5 완료 기준

다음을 모두 만족하면 Stage 6으로 넘어간다.

1. V6와 독립된 torch-free AlphaZero PUCT core가 동작한다.
2. fake evaluator만으로 PUCT/legal/terminal/backup 핵심 규약이 결정적으로 검증된다.
3. 일반 root 사전 확장은 simulation 예산에 포함되지 않고 child visits 합이 정확히 N이다.
4. 단일 합법수 root는 search를 생략해 one-hot target을 저장하며 sample은 유지한다.
5. terminal node 재방문이 evaluator를 재호출하지 않는다.
6. 실제 `PolicyValueNet`이 별도 neural evaluator를 통해 leaf evaluation에 연결된다.
7. 저장 counts에서 파생한 모든 `pi`의 illegal mass가 0이고 합이 1이다.
8. 저장된 모든 action이 그 상태에서 합법이다.
9. terminal과 NN value의 관점이 player identity 기준으로 일관된다.
10. self-play 한 판 이상에서 `Sample(ply, visits, action, z)` 전체가 생성된다.
11. GameRecord를 replay해 동일 final board/result/history를 복원한다.
12. 동일 장치/백엔드/batch size 조건에서 결과/기보 hash 재현성을 확인한다.
13. 수당 legal/inference/tree 시간을 기록한다.
14. 기존 회귀와 신규 Stage 5 테스트가 모두 PASS한다.
15. V5/V6 강제수·전술 planner는 self-play 탐색 경로에서 호출되지 않는다.

**기력, V6 상대 승률, loss 감소는 Stage 5 PASS 조건이 아니다.**
Stage 6부터 replay buffer와 학습 루프를 연결한 뒤 고정 baseline을 상대로 성장 여부를 평가한다.
