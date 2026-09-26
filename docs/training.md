# Stage 6 Self-Play Training Loop

Stage 6은 Stage 1~5에서 이미 검증한 요소를 연결해 **AlphaZero 학습 파이프라인이 end-to-end로
동작하고, 기록되고, 중단 후 재개될 수 있음**을 확인하는 단계다. 새 알고리즘이나 기력 향상은 목표가 아니다.

```text
현재 모델 → self-play → (state, π, z) → ReplayBuffer → mini-batch + D4 augmentation
→ policy/value 학습 → evaluation → checkpoint → 다음 self-play
```

## 1. 목적과 범위

포함: FIFO replay buffer, 정책/가치 손실, 주기적 평가(monitoring), generation 경계 resume,
metrics/metadata, 원자적 checkpoint.

제외(Stage 7 이후): 병렬 self-play, 분산 학습, prioritized replay, Elo, best-model gating/rollback,
대량 평가, hyperparameter 탐색, lr scheduler, GPU 최적화, dynamic inference batching,
opening book, 신규 전술 MCTS, mid-generation recovery.

모델 갱신 정책은 **always-latest**다. 평가에서 져도 rollback하지 않는다.

## 2. 재사용한 Stage 4/5 인터페이스

| 용도 | 실제 코드 |
|---|---|
| 규칙·합법수·종료 | `renju.Game` (`legal_moves`, `play`, `done`, `winner`) — `src/renju/game.py` |
| 네트워크 | `model.network.PolicyValueNet`, `model.config.ModelConfig` |
| 입력 인코딩 | `model.encoding.encode_game` (`[6,15,15]` float32, plane 5 = legal mask) |
| mask·loss | `model.masking.legal_moves_to_mask`, `policy_loss`, `value_loss` |
| D4 대칭 | `model.symmetry.transform_spatial / transform_policy / transform_mask` |
| 초기 모델 파일 | `model.checkpoint.load_checkpoint` (`training.init_checkpoint`) |
| NN evaluator | `model.evaluator.PolicyValueEvaluator`, `file_sha256` |
| PUCT | `search.alphazero.SearchConfig`, `run_search`, `select_action`, `visit_policy` |
| self-play 기록 | `training.self_play.play_self_play_game`, `replay_record`, `GameRecord`, `record_hash`, `game_hash`, `summarize_timing` |
| provenance | `training.provenance.git_provenance`, `base_runtime_env` |
| 평가 상대 | `agents.RandomAgent`, `agents.TacticalAgent`, `agents.MCTSV6Agent` (V5_FINAL frozen) |

Stage 1~5 소스는 수정하지 않았다(`src/renju`, `src/search`, `src/model`, `src/agents`,
`training/self_play.py`, `training/provenance.py` 모두 불변). frozen baseline SHA-256 잠금도 그대로다.

신규 모듈:

```text
src/training/config.py              # YAML 로딩, 기본값, 검증, training-critical hash
src/training/replay_buffer.py       # TrainingSample, Batch, ReplayBuffer
src/training/dataset.py             # record→sample, validate_samples, augmentation, build_batch
src/training/trainer.py             # build_optimizer, train_step, inference_mode_for
src/training/training_state.py      # TrainingState, RNG 직렬화, 초기화
src/training/training_checkpoint.py # 원자적 저장, load/restore, prune
src/training/evaluation.py          # PUCTAgent, 개시 쌍, 상대별 평가·집계
src/training/metrics.py             # metrics.jsonl, metadata segment, resume 정리
src/training/loop.py                # run_training, run_generation, verify_checkpoint
scripts/run_stage6_training.py      # CLI
scripts/compare_stage6_runs.py      # 두 run 정확 일치 비교
configs/stage6_mvp.yaml, configs/stage6_test.yaml
```

## 3. TrainingSample 계약과 z 관점

Stage 5 `Sample(ply, to_play, visit_counts, action, z)`는 상태 tensor를 저장하지 않는다.
`training.dataset.samples_from_record()`가 먼저 Stage 5 `replay_record()`로 record 계약을 검증하고,
`moves[:ply]`를 새 `Game`에 재생하며 각 sample을 만든다.

```python
TrainingSample(
    state,        # encode_game(game, mask)  [6,15,15] float32
    policy,       # visit_policy(visit_counts) = N/sum(N)  [225] float32
    value,        # Stage 5 Sample.z 그대로 (부호 변경 없음)
    legal_mask,   # legal_moves_to_mask(game.legal_moves())  [225] bool
    generation_id, game_id, ply,   # provenance 전용, 학습 feature 아님
)
```

`z`는 Stage 5 `compute_z(winner, to_play)`로 이미 **sample의 side-to-move 관점**이다. Stage 6은 부호를
다시 뒤집지 않는다. `test_training_dataset`이 흑 승리 대국에서 흑 차례 sample z=+1, 백 차례 z=-1,
백 승리 대국에서 그 반대임을 plane 3(`current_is_black`)으로 확인한다.

`validate_samples()`는 buffer에 넣기 전에 다음을 검사하고, 위반 시 `SampleValidationError`를 던진다
(조용히 버리거나 보정하지 않음): state/policy/mask shape·dtype, finite, `legal_mask.any()`
(terminal sample 없음), encoder plane 5 == legal_mask, π ≥ 0, illegal π == 0, |Σπ − 1| ≤ 1e-5,
z ∈ {−1, 0, +1}.

## 4. ReplayBuffer

- **sample-count FIFO** 하나: `ReplayBuffer(capacity)`; 초과 시 가장 오래된 sample을 덮어쓴다.
- 고정 크기 텐서 링버퍼: state `uint8`(encoder plane은 정확히 0/1이므로 무손실, 비이진 값은 거부),
  policy `float32`, value `float32`, mask `bool`, provenance `int64[3]`.
- 추출: `sample_indices(batch_size, rng=sample_rng)` — **복원추출(with replacement)**, 항목당
  `rng.randrange(len)` 1회. 전역 `random`은 사용하지 않으며 `random.Random`이 아니면 `TypeError`.
- index는 **논리 index**(0 = 가장 오래된 sample)다. `state_dict()`는 채워진 sample을 논리 순서로 저장하므로
  reload 후 물리 offset이 달라도 같은 RNG 상태가 같은 sample을 고른다.
- `equals()`는 내용과 순서를 비교한다. buffer는 training checkpoint 안에 포함된다(별도 파일 없음).

## 5. Symmetry augmentation

buffer에는 원본만 저장하고, batch 구성 시 row마다 `augment_rng.randrange(8)`로 D4 변환 하나를 골라
Stage 4 `transform_spatial`(state 6 plane 전체), `transform_policy`(π), `transform_mask`(mask)에
**같은 id**를 적용한다. value는 불변. `augmentation.enabled: false`이면 `augment_rng`를 소비하지 않는다.
sample index는 `sample_rng`만, 대칭 선택은 `augment_rng`만 사용한다.

**금수 D4 불변성 검증**: `test_training_dataset.AugmentationTest`는 흑 금수가 존재하는 국면
(Stage 5 neural 테스트와 같은 fixture)과 그 직후 백 차례 국면에서 8개 대칭 모두에 대해,
원래 기보를 좌표 변환해 새 `Game`에 다시 두고(엔진이 금수를 독립적으로 재판정) 그 `legal_moves()`
mask가 변환된 mask와 같고, 변환된 state 전체가 `encode_game(변환 게임)`과 같음을 확인한다.
가정이 아니라 엔진 재계산으로 검증했다.

## 6. Loss와 optimizer

```text
total_loss = policy_loss + value_weight * value_loss (+ l2_coeff * Σ‖θ‖²)
```

- `policy_loss`, `value_loss`는 Stage 4 `model.masking`의 함수를 그대로 쓴다. Stage 4 policy loss는
  illegal logit을 `-inf`로 마스킹한 뒤 log-softmax하고, **곱셈 전에 illegal log-prob를 0으로 채운다**.
  따라서 `0 * -inf = NaN`이 생기지 않으며, 지시문의 "큰 유한 음수" 방식과 같은 목적을 달성한다.
  또한 target의 illegal mass가 0이 아니면 거부한다. illegal logit을 바꿔도 loss와 gradient가
  변하지 않음을 테스트한다.
- value head 출력은 이미 `tanh`, shape `[B,1]`; target도 `[B,1]`로 맞춘다.
- optimizer: Stage 4에 공식 학습 optimizer 정책은 없고 tiny overfit이 Adam(lr 1e-3)을 사용했으므로
  기본값을 `adam, lr 0.001, weight_decay 0.0001`로 둔다. `sgd`(momentum)도 선택 가능하다.
- `optimizer.weight_decay`는 optimizer 설정이다(torch `Adam`의 coupled L2 방식). loss에 `‖θ‖²`를 직접
  더하는 것은 `loss.l2_coeff`뿐이며, **둘 다 0이 아니면 config 오류**다.
- lr scheduler 없음(상수 lr). checkpoint의 `scheduler_state_dict`는 `null`.
- gradient clipping은 `training.grad_clip`(기본 `null` = OFF). metrics의 `grad_norm`은 clip 전 norm.
- model output, loss, grad norm이 non-finite면 `TrainingDivergenceError`를 **`optimizer.step()` 전에** 던진다.

### train/eval 모드

```text
Training   → model.train()                  (train_step 내부)
Self-play  → inference_mode_for(model)      eval + no_grad, 종료 후 원래 모드 복원
Evaluation → inference_mode_for(model/prev) 동일
```

`PolicyValueEvaluator` 생성자는 공유 모델을 `.eval()`로 바꾸므로 복원은 context manager가 담당한다.
BatchNorm running stat이 eval context에서 변하지 않음을 테스트한다. PolicyValueEvaluator 자체는
추론 시 `torch.inference_mode()`를 사용한다.

## 7. Checkpoint 형식

`format_version: stage6-training-checkpoint-v1` 사전 하나에 다음을 저장한다.

```text
model_state_dict, optimizer_state_dict, scheduler_state_dict (null)
generation (다음에 실행할 generation), global_step
replay_buffer (state_dict)
global_rng {python, torch, numpy}, component_rng {self_play_rng, sample_rng, augment_rng}
config (full resolved), critical_config_hash
git_commit, git_dirty, device, model_config, model_contract (encoder/action/plane/Stage 4 format)
python_version, torch_version
```

- 파일 번호 = checkpoint 안의 `generation`. `checkpoint_init.pt`(generation 0, 학습 전),
  gen 0 완료 후 `checkpoint_gen001.pt`(generation 1), … `latest.pt`는 가장 최근 generation 파일의 복사본.
- **원자적 저장**: 같은 디렉터리 임시 파일(`.<name>.tmp-<pid>`)에 `torch.save` → flush → `os.fsync`
  → 파일 핸들을 닫은 뒤 `os.replace`. `latest.pt`도 같은 방식으로 복사·교체한다. 실패하면 임시 파일을
  지우고 기존 final/latest는 그대로 남는다(예외 주입 테스트).
- `keep_checkpoints: K` — 최근 K개의 `checkpoint_genNNN.pt`만 유지. `checkpoint_init.pt`, `latest.pt`는 삭제하지 않는다.
- **`torch.load` 정책**: 모든 값이 텐서 또는 int/float/str/bool/None/list/dict이다. Python `Random` 상태는
  `{version, internal(list), gauss_next}`, NumPy 상태는 list로 변환한다. 로드는 항상
  `torch.load(weights_only=True)`이며 `weights_only=False`는 쓰지 않는다.
- 호환성 검사: format_version, model contract, 모델 구조(`model_config`), training-critical config hash가
  다르면 `CheckpointCompatibilityError`/`ConfigError`로 이유(다른 key 목록)를 알려준다.

## 8. RNG 관리

| RNG | 용도 | 저장 |
|---|---|---|
| `self_play_rng` | 게임마다 `getrandbits(63)`로 Stage 5 게임 seed 파생 → 그 게임의 `Random(seed)`가 temperature와 Dirichlet을 담당 | checkpoint |
| `sample_rng` | replay index | checkpoint |
| `augment_rng` | 대칭 선택 | checkpoint |
| evaluation | stateless: `derive_seed(seed, generation, opponent, 'opening'/'agent', index)` | 저장 안 함 |
| Python/NumPy/torch 전역 | 초기 seed 고정, 코드가 직접 쓰지는 않음 | checkpoint |

Stage 5 self-play는 이미 게임별 RNG를 주입받으므로 Stage 5 코드 변경이 필요 없었다.
`derive_seed`는 SHA-256 기반이라 `PYTHONHASHSEED`와 무관하다. 평가는 학습 RNG를 건드리지 않고,
resume 여부와 무관하게 같은 개시·같은 agent seed를 사용한다(테스트로 확인).

## 9. Resume semantics

- **generation boundary resume만 지원**. 순서는
  `self-play → buffer → training → evaluation → generation += 1 → checkpoint`.
  checkpoint가 있으면 그 generation의 평가도 끝난 것이다. generation G 도중 crash나면
  checkpoint G(`latest.pt`)에서 G를 처음부터 다시 실행한다.
- resume은 checkpoint가 속한 run 디렉터리(`<run>/checkpoints/latest.pt`의 두 단계 위)에서 이어간다.
- 시작 시 `metrics.jsonl`을 `metrics.jsonl.bak-<timestamp>`로 복사하고 `generation < G` 이벤트만 남긴다.
  `self_play/genNNN.json`, `evaluation/genNNN.json` 중 NNN ≥ G인 파일은 `.bak-<timestamp>`로 이름을 바꾼다.
- `metadata.json`의 `segments`에 실행마다 시작/종료 시각, duration, start/end generation, resumed,
  status(`completed`/`stopped`/`failed`+error), 정리 결과, git commit/dirty를 누적한다.
- config 우선순위: training-critical 값(모델 구조, optimizer, loss, replay/batch/steps/games, augmentation,
  self-play·evaluation 탐색과 판수, seed)은 **checkpoint 값이 기준**이며 `--config`가 다르면 오류로 중단한다.
  실행 제어 값(`training.generations`, `keep_checkpoints`, `init_checkpoint`, `device`, `torch_threads`,
  `output`)과 CLI `--generations`, `--stop-after-generation`은 바꿀 수 있다.

### Resume 등가성 테스트 (`tests/test_training_resume.py`)

stage6_test 설정을 3 generations로 늘려(평가는 gen 0과 최종 gen 2, Tactical 제외 — 테스트 시간 단축용)
세 run을 만든다: A 연속, B `stop_after=2` 후 `latest.pt` resume, C gen 1 두 번째 train step에서 예외 주입 후
resume. B, C의 최종 `latest.pt`를 A와 비교한다: model/optimizer state, replay buffer 내용·순서,
generation/global_step, component RNG, 전역 RNG, 전 generation self-play record hash·수순,
evaluation 기보·결과, train loss 순서. 모두 **`torch.equal`/`==` 정확 일치**다(CPU, single process,
DataLoader 미사용, threads 1, `use_deterministic_algorithms(True)`). checkpoint 파일 byte 일치는 요구하지 않는다.
C는 crash 후 `latest.pt.generation == 1`, 첫 segment `failed`, metrics에 gen 1 이벤트 중복 없음도 확인한다.

## 10. Evaluation protocol

- 학습 모델과 previous는 **같은** evaluation PUCT 설정: `puct_simulations`, `c_puct`, noise OFF,
  temperature 0 → Stage 5 `argmax_action` (visit → prior → 작은 action index tie-break).
- **개시 쌍**: 강제 중앙 첫 수 + `opening_random_plies`(기본 2)개의 무작위 합법수(중앙에서 Chebyshev
  거리 ≤ `opening_radius`=2, 후보가 없으면 전체 합법수). 개시 수는 `Game.play()`가 검증한다.
  같은 개시를 학습 모델 흑/백으로 한 번씩 둔다. 따라서 기록의 `opening_plies`는 3이고, 착수 시간 통계는
  개시 이후 수만 포함한다. 결과에 `unique_games`, `duplicate_games`를 기록한다.
- **previous 정의**: generation g의 previous = g 학습 시작 직전 모델 = checkpoint g(gen 0은 `checkpoint_init.pt`).
- 상대: Random, Tactical, previous(각 흑 2/백 2), MCTS-v6(마지막 generation만 흑 1/백 1, frozen
  `V5_FINAL`, `MCTSV6Agent(seed=…)`를 override 없이 생성). agent seed는 대국별로 다르게 파생한다.
- 평가는 학습 직후·checkpoint 저장 전에 수행한다. 불법 착수는 `EvaluationIllegalMoveError`로 즉시 중단하므로
  기록된 `illegal_moves`는 항상 0이다.
- 기록(상대별): games, unique/duplicate, W/L/D, 흑·백별 W/L/D, illegal_moves, 평균 착수 시간(학습 모델/상대),
  평균 대국 길이, opening_plies, 상대 설정, 학습 모델 탐색 설정, 소요 시간. 전체 기보는 `evaluation/genNNN.json`.
- 2~4판 결과로 기력 향상을 주장하지 않는다. MCTS-v6 2판은 **benchmark integration smoke**다.

## 11. Config

모든 설정은 YAML에서 읽는다(`training.config.load_config`). 누락 key는 `DEFAULTS`(self-play 탐색 기본값은
Stage 5 `SearchConfig()`에서 가져옴)로 채우고, 모르는 key는 오류다. resolved config는 run의 `config.yaml`과
checkpoint에 기록된다. PyYAML(`PyYAML>=6.0`)을 `pyproject.toml` core 의존성에 추가했다(순수 Python,
torch 불필요 — torch 없는 CI job에서도 config 테스트가 실행된다).

- `stage6_mvp.yaml`: Stage 4 기본 구조(64ch×4 blocks), 3 generations, generation당 4판, 25 simulations,
  batch 32, 50 steps, capacity 10000, Adam 1e-3 + weight_decay 1e-4, 평가 25 simulations.
- `stage6_test.yaml`: 테스트 전용 초소형(8ch×1 block, 2 generations, 1판, 3 simulations, 2 steps, batch 8,
  평가 상대별 흑1/백1, MCTS-v6 제외). 실험 설정이 아니다.
- `self_play.max_moves`는 225만 허용한다. Stage 5 self-play에는 대국 절단 기능이 없고 `replay_record()`가
  종료된 대국을 요구하므로, 절단(새 z 의미)을 Stage 6에서 만들지 않았다. 테스트 전용 수순 제한도 두지 않았다.

## 12. 실행 방법

bash / PowerShell 공통(한 줄 명령):

```bash
python scripts/run_stage6_training.py --config configs/stage6_mvp.yaml
python scripts/run_stage6_training.py --config configs/stage6_mvp.yaml --stop-after-generation 2
python scripts/run_stage6_training.py --config configs/stage6_mvp.yaml --resume runs/<run>/checkpoints/latest.pt
python scripts/run_stage6_training.py --verify-checkpoint runs/<run>/checkpoints/latest.pt
python scripts/compare_stage6_runs.py runs/<runA> runs/<runB>
python -m unittest discover -s tests -v
```

- 새 run은 `runs/<YYYYMMDD-HHMMSS>_<output.run_name>/`에 만든다(`--run-dir`로 지정 가능, 비어 있지 않으면 거부).
- `--stop-after-generation N`은 checkpoint generation이 N이 되면 멈춘다(프로세스 강제 종료 없이 중단 재현).
- `--resume`만 주면 checkpoint의 config를 그대로 쓴다. `--generations`로 총 목표를 늘릴 수 있다.
- `--verify-checkpoint`는 checkpoint를 reload해 `(seed, 'verify', generation)` 파생 seed로 self-play 1판을
  두고 replay/sample 검증을 수행한다(저장된 RNG는 사용하지 않음).
- CLI는 `torch.use_deterministic_algorithms(True)`와 config의 `torch_threads`를 적용한다. 경로는 `pathlib`로 처리한다.

run 디렉터리:

```text
runs/<run>/
├── config.yaml        # resolved config
├── metadata.json      # git branch/commit/dirty, seed, device, 버전, segments
├── metrics.jsonl      # type = train | generation | evaluation | checkpoint
├── checkpoints/       # checkpoint_init.pt, checkpoint_genNNN.pt, latest.pt
├── self_play/genNNN.json   # GameRecord + game/record SHA256
└── evaluation/genNNN.json  # 상대별 summary + 전체 기보
```

`sample_reuse_ratio = steps_per_generation × batch_size / 이번 generation 신규 sample 수`
(분모 0이면 `null`). augmentation 전 추출 횟수 기준이다.

## 13. MVP 실행 결과

실행 환경: 2026-09-26, Windows 11 Pro 10.0.26200, Python 3.13.14, torch 2.14.0+cpu, NumPy 2.5.3,
CPU, `torch_threads: 1`, deterministic algorithms ON. branch `feat/stage6-training`,
commit `50beeb0` (`git_dirty: false`, 두 run의 모든 segment 동일). seed 42, 설정 `configs/stage6_mvp.yaml`
(resolved 값은 각 run의 `config.yaml`). run 디렉터리와 checkpoint는 커밋하지 않는다(`runs/`는 ignore 대상).

실행한 명령:

```bash
python scripts/run_stage6_training.py --config configs/stage6_mvp.yaml --run-dir runs/stage6_mvp_B --stop-after-generation 2
python scripts/run_stage6_training.py --config configs/stage6_mvp.yaml --resume runs/stage6_mvp_B/checkpoints/latest.pt
python scripts/run_stage6_training.py --config configs/stage6_mvp.yaml --run-dir runs/stage6_mvp_A
python scripts/compare_stage6_runs.py runs/stage6_mvp_A runs/stage6_mvp_B
python scripts/run_stage6_training.py --verify-checkpoint runs/stage6_mvp_B/checkpoints/latest.pt
```

소요 시간(wall): run B segment 1(gen 0~1, stopped) 6 m 38 s, segment 2(resume, gen 2 + MCTS-v6) 3 m 31 s,
run A 연속 9 m 47 s. A와 B는 동시에 실행했다(각 1 thread).

### 13.1 Generation별 결과 (run B = resume run; run A 값과 모두 동일)

| gen | self-play 판 (흑/백/무) | 평균 수 | 신규 sample | buffer | total loss 첫→끝 step | policy 첫→끝 | value 첫→끝 | reuse ratio | self-play / training s | checkpoint |
|---|---|---:|---:|---:|---|---|---|---:|---|---|
| 0 | 4 (2/2/0) | 112.5 | 450 | 450 | 6.215 → 4.568 | 5.209 → 4.502 | 1.006 → 0.066 | 3.556 | 88.8 / 14.9 | `checkpoint_gen001.pt` |
| 1 | 4 (3/1/0) | 60.3 | 241 | 691 | 5.408 → 4.947 | 4.750 → 4.644 | 0.657 → 0.303 | 6.639 | 31.2 / 11.2 | `checkpoint_gen002.pt` |
| 2 | 4 (4/0/0) | 86.0 | 344 | 1035 | 5.257 → 4.752 | 4.643 → 4.698 | 0.614 → 0.054 | 4.651 | 51.7 / 12.3 | `checkpoint_gen003.pt` |

- 50 steps × batch 32 = generation당 1600 sample 추출. global_step 50/100/150.
- loss는 모든 step에서 finite. NaN/Inf loss·model output 0건(발생 시 즉시 예외로 중단되는 구조).
- self-play searched move 평균: gen 0 198 ms(inference 132 ms, legal_moves 56 ms), gen 1 130 ms, gen 2 151 ms.
- self-play illegal move 0(모든 record가 Stage 5 `replay_record` 통과), illegal policy mass 0(`validate_samples`).
- checkpoint 파일: `checkpoint_init.pt`, `checkpoint_gen001~003.pt`, `latest.pt`(keep 3이므로 삭제 없음).

### 13.2 평가 (학습 모델 기준 W/L/D, 흑·백 괄호는 W/L/D)

| gen | Random | Tactical | previous | MCTS-v6 |
|---|---|---|---|---|
| 0 | 0/4/0 (흑 0/2/0, 백 0/2/0) | 0/4/0 (0/2/0, 0/2/0) | 0/4/0 (0/2/0, 0/2/0) | — |
| 1 | 3/1/0 (1/1/0, 2/0/0) | 0/4/0 (0/2/0, 0/2/0) | 3/1/0 (1/1/0, 2/0/0) | — |
| 2 | 3/1/0 (2/0/0, 1/1/0) | 0/4/0 (0/2/0, 0/2/0) | 2/2/0 (1/1/0, 1/1/0) | 0/2/0 (0/1/0, 0/1/0) |

- 모든 상대에서 고유 기보 = 판 수(Random/Tactical/previous 4/4, MCTS-v6 2/2), 중복 기보 0, illegal 0.
  opening_plies 3(강제 중앙 + 무작위 2수).
- 평균 착수 시간(학습 모델 / 상대, gen 2): Random 0.151 / 0.002 s, Tactical 0.159 / 0.004 s,
  previous 0.168 / 0.127 s, MCTS-v6 0.120 / 0.553 s. 학습 모델·previous는 PUCT 25 simulations, c_puct 1.5,
  noise OFF, temperature 0.
- **MCTS-v6 smoke**(frozen `V5_FINAL`: simulations 50, tactical_simulations 100, threshold 1800,
  exploration √2, candidate_limit 20, initial_width 8, radius 2, top-k 8): 흑 12수 패, 백 9수 패,
  평균 10.5수. benchmark 연결이 오류 없이 동작함을 확인한 것일 뿐 기력 결론은 없다.
- 판 수가 4판 이하이므로 위 승패는 기력 변화의 근거가 아니다. 3 generation × 4판 규모의 학습은
  기력 향상을 기대할 수 있는 규모가 아니다.

### 13.3 Resume 검증 (run A 연속 vs run B 2 generation 후 중단·재개)

`scripts/compare_stage6_runs.py` 결과 `EQUAL`(정확 일치, `torch.equal`/`==`):

| 항목 | 결과 |
|---|---|
| generation / global_step | 3 / 150 양쪽 동일 |
| model state_dict (BN buffer 포함) | equal |
| optimizer state (Adam moments, step) | equal |
| replay buffer 내용·순서 (1035 samples) | equal |
| self_play / sample / augment RNG | equal |
| critical config hash | equal |
| gen 0/1/2 self-play record SHA256·수순 | equal (gen 2: `f90bd8b6…`, `50b31d96…`, `5108fbb8…`, `21a81fff…`) |
| gen 0/1/2 evaluation 기보·결과 (MCTS-v6 포함) | equal |

resume segment는 checkpoint `generation=2, global_step=100`에서 시작했고 정리 단계에서 dropped event 0건
(정상 중단이므로 재실행 generation 기록이 없음). `checkpoint_gen001/002.pt`는 파일 byte까지 같았고
`checkpoint_gen003.pt`는 byte가 달랐지만 내용 비교는 위와 같이 모두 일치했다(byte 일치는 요구사항이 아님).

최종 checkpoint reload 후 self-play 1판(`--verify-checkpoint`): generation 3, 백 승 60수, sample 60개,
replay·`validate_samples` 통과, illegal 0, checkpoint SHA256 `d9e62e0e…`.

## 14. 알려진 제한과 Stage 5 carry-over

- **batch inference (Stage 5 carry-over)**: `PolicyValueEvaluator.evaluate_batch`는 batch N을 지원하지만
  Stage 5 PUCT는 `evaluator_batch_size=1`만 허용한다(virtual loss 없음). Stage 6은 이를 확장하지 않았다.
- self-play 비용은 `Game.legal_moves()`(흑 금수 판정)가 가장 크다. 테스트 설정 profile에서 전체 시간의
  절반 이상이 `legal_moves`였다. 규칙 엔진은 Stage 6 수정 범위가 아니다.
- 대국 절단(`max_moves` < 225) 미지원(11장).
- 평가 판수(상대별 2~4판)는 표본이 매우 작다. 승률은 monitoring 값이며 기력 판단 근거가 아니다.
- `git_dirty`는 Stage 4/5와 같이 `--untracked-files=no` 기준이다. 실험 전 새 소스 파일은 commit해야 한다.
- 장치 간(CPU↔GPU) bit-exact resume은 요구하지 않는다. 등가성은 같은 장치·thread 수·torch 버전 조건에서만 검증했다.
- 테스트 시간: Stage 6 테스트 추가로 전체 회귀가 기준 약 21 s에서 약 86 s(torch 설치, 339 tests)로 늘었다
  (대부분 `test_training_resume`의 3-run 비교와 `legal_moves` 비용).
