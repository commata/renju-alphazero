# Stage 8 Plan — 데스크톱 이관, GPU 우선 처리량 확장

Stage 8은 Stage 7에서 검증한 D16 학습 설정을 데스크톱으로 옮기고, **학습 의미를 가능한 한 유지한 채
GPU를 포함한 장치 처리량을 최대화하여 장기 학습 파이프라인을 확립하는 단계**다.

가장 강한 최종 모델을 확정하는 것은 Stage 9의 역할이다. Stage 8에서는 먼저 같은 64×4 network와
16 games/generation, PUCT v2 50 simulations 설정으로 CPU와 GPU 실행 경로를 비교하고,
가장 빠르고 안정적인 경로를 선택한다. 그 뒤에만 simulation 수, network 크기, batch 등
training-critical 규모 확장을 별도 실험으로 검토한다.

핵심 원칙:

1. **처리량 개선(execution/backend)과 학습 알고리즘 변경(training-critical)을 같은 실험에 섞지 않는다.**
2. Stage 8의 GPU 목표는 B=1 추론을 단순히 GPU로 옮기는 것이 아니라,
   **여러 독립 self-play game의 NN 요청을 모아 batched inference로 GPU를 계속 사용하게 하는 것**이다.
3. `games_per_generation=16`은 초기 Stage 8 동안 유지한다. worker/concurrency는 같은 16판을 더 빨리 생성하기 위한 실행 방식이다.
4. CPU serial은 항상 correctness/reference 경로로 보존한다.
5. GPU backend가 바뀌면 float kernel 차이 때문에 game hash exact equality를 장치 간 계약으로 요구하지 않는다.

## 1. 기준선

| 항목 | 값 |
|---|---|
| 코드 | main `52fbf91` (PR #17 병합), 태그 `v0.6-stage7` |
| config | `configs/stage7d_b16.yaml`, critical hash `25ab9c5a…` |
| 시작 checkpoint | `runs/stage7d_b16/checkpoints/checkpoint_gen160.pt`, SHA-256 `58aea679…` (generation=160) |
| network | 64 channels × 4 blocks |
| search | PUCT v2(`tactical_rules`), 50 simulations, c_puct 1.5, FPU 0 |
| self-play | 16 games/generation, temperature 10수, Dirichlet α 0.05 · ε 0.25 |
| training | batch 32, 50 steps/generation, replay 10,000, Adam lr 1e-3, 8-way augmentation |
| 루프 내 평가 | 매 generation random·tactical·previous 각 4판(v2 25 sims), 마지막 generation에 MCTS-v6 2판 |
| Stage 7 장치 | CPU, `torch_threads: 1` |
| Stage 7 외부 평가(gen 160) | Tactical 49/50, MCTS-v2 43/50(흑 20·백 23), v321 7/10, v7 0/10 |

**worker/concurrency를 늘려도 `games_per_generation=16`은 유지한다.**
Stage 7-D의 D32는 한 색 붕괴가 관측됐으므로 32판/generation으로 늘리는 것과 병렬화를 같은 변경으로 취급하지 않는다.

---

## 2. 현재 main에서 GPU를 바로 쓰기 전에 필요한 변경

main `52fbf91`을 기준으로 확인한 결과, `device: cuda`만 바꾸는 것으로는 Stage 8의 GPU 목표를 달성할 수 없다.

### 2.1 search는 현재 강제로 B=1이다

`SearchConfig.evaluator_batch_size`는 존재하지만 `__post_init__`에서 1이 아니면 오류를 낸다.

또한 `search_with_tree()`는 leaf를 찾을 때마다 `_expand() -> evaluate_validated(evaluator, [snapshot])`를
동기적으로 호출한다. 한 tree 안에서는 이전 simulation의 value/visit 결과가 다음 selection에 영향을 주므로
현재 알고리즘을 그대로 유지하면서 한 tree에서 임의로 여러 leaf를 동시에 뽑을 수 없다.

반면 `PolicyValueEvaluator.evaluate_batch()` 자체는 이미 여러 snapshot을 한 번에 처리할 수 있다.

따라서 Stage 8 GPU batching의 기본 단위는 **한 tree 안의 여러 leaf가 아니라 여러 독립 game/search**로 잡는다.

### 2.2 현재 evaluator는 GPU에서 작은 연산을 snapshot별로 만들 수 있다

현재 `PolicyValueEvaluator.evaluate_batch()`는 각 snapshot마다:

1. `legal_moves_to_mask(..., device=self.device)`
2. `encode_game(..., mask)`
3. 이후 `torch.stack`

순서로 처리한다.

GPU에서는 15×15의 매우 작은 상태마다 mask/plane tensor를 장치에서 개별 생성하는 것보다,
**CPU에서 batch 전체를 인코딩하고 contiguous tensor로 만든 뒤 한 번에 device transfer**하는 경로를 별도로
벤치마크한다.

CPU reference evaluator의 결과와 tolerance 내에서 같아야 한다.

### 2.3 training batch는 현재 CPU tensor다

`ReplayBuffer.get()`와 `build_batch()`는 CPU tensor를 반환하고,
`train_step()`은 그 batch를 model device로 옮기지 않는다.

따라서 현재 구조에서 model만 GPU로 옮기면 training 단계에서 device mismatch가 발생할 수 있다.

Stage 8 GPU enablement의 선행 변경으로 다음을 넣는다.

~~~text
ReplayBuffer / augmentation: CPU
        ↓
build_batch()
        ↓
batch_to_device(batch, target_device)
        ↓
train_step()
~~~

provenance는 loss에 필요하지 않으므로 CPU에 남겨도 되고, states/policies/values/legal_masks만 옮기는 방식을 우선한다.

### 2.4 기존 CPU/frozen 경로는 바꾸지 않는다

새 GPU 최적화 evaluator와 batched scheduler를 추가하더라도 기존:

~~~text
run_search()
PolicyValueEvaluator.evaluate_batch([snapshot])
CPU B=1
~~~

경로를 reference wrapper로 유지한다.

Stage 5 golden hash, V6/V7 frozen behavior, Stage 7 rule differential test는 GPU 변경 PR에서도 그대로 통과해야 한다.

---

## 3. Stage 8-A — 데스크톱 이관과 CPU 기준선

먼저 GPU 최적화를 넣지 않은 상태로 D16 gen160을 데스크톱에서 복원한다.

### 3.1 run 복사

`run_training(resume=...)`는 checkpoint의 상위 run directory를 그대로 사용하고 `metadata.json`을 요구한다.
Stage 7 증거 run을 직접 이어 쓰지 않는다.

~~~text
runs/stage7d_b16/  # 원본 보존
runs/stage8_cpu/   # Stage 8 복사본
~~~

복사본의 `latest.pt`에서 resume한다.

### 3.2 이관 검증

1. git 기준점과 tag 기록
2. checkpoint file SHA-256 = `58aea679…`
3. critical config hash = `25ab9c5a…`
4. full tests / frozen baseline / rule differential
5. `--verify-checkpoint` 반복 실행
6. 3~5 generation 정도의 CPU serial throughput 측정

CPU 기준선에서 기록:

- self-play s/gen
- training s/gen
- in-loop evaluation s/gen
- checkpoint s/gen
- total s/gen
- games/h
- samples/s
- searched move ms
- NN calls/move

**512판 전체를 CPU serial로 먼저 돌리는 것을 Stage 8 선행 조건으로 두지 않는다.**
GPU 경로를 최대한 빨리 검증하기 위해 correctness와 throughput 기준선에 필요한 짧은 run만 수행한다.

### 3.3 재현성 범위

같은 기계/같은 backend/B=1 CPU에서는 같은 checkpoint·seed의 game/record hash가 같아야 한다.

그램 CPU ↔ 데스크톱 CPU 또는 CPU ↔ GPU처럼 backend/kernel이 달라지는 경우에는
float 연산 차이로 탐색 tie가 달라질 수 있으므로 game hash exact equality를 요구하지 않는다.

장치 간 공통 계약은 다음이다.

- file SHA / critical hash
- 합법수와 rule 결과
- visit sum
- 불법수 0
- evaluator output tolerance
- 동일 seed 소비 규칙
- checkpoint/resume 구조

---

## 4. Stage 8-B — 흑/백 편향 감시

### 4.1 `color_imbalance` — 정보성 경고

최근 10 generation(160판)의:

- black win rate
- white win rate
- draw rate
- average game length
- 같은 방향의 극단 window 연속 횟수

를 기록한다.

흑 승률 ≥ 0.90 또는 ≤ 0.10이면 `health_warning / color_imbalance`를 남기되
이 경고 자체로 stop/rollback하지 않는다.

성공한 D16도 극단적 색 우세 구간을 길게 통과했기 때문이다.

### 4.2 `color_regression` — 외부 평가 기반 조치 신호

직전 checkpoint만 기준으로 Fisher 검정을 반복하면 이미 붕괴한 뒤
`17/25 -> 3/25 -> 1/25`처럼 유지되는 경우 두 번째 비교가 유의하지 않아 지속 붕괴를 놓칠 수 있다.

따라서 각 색에 대해 **마지막 healthy reference 평가**를 유지한다.

light evaluation마다:

1. 같은 fixed openings에서 MCTS-v2 25판/색 결과를 얻는다.
2. 현재 결과를 그 색의 healthy reference와 Fisher 단측 검정한다.
3. p < 0.05로 하락하면 `color_regression_candidate`를 기록하고 reference는 유지한다.
4. 다음 light 평가에서도 **같은 reference 대비** p < 0.05 하락이면 `color_regression_confirmed`로 올린다.
5. 하락이 해소되면 현재 checkpoint를 새 healthy reference 후보로 갱신한다.

confirmed 신호에서도 자동 rollback은 하지 않는다.
사람이 마지막 healthy checkpoint, seed 분기, 계속 학습 중 하나를 선택한다.

---

## 5. Stage 8-C — RX 6600 GPU backend 검증

### 5.1 2026-09-28 기준 지원 상태

RX 6600은 RDNA2 `gfx1032`다.

released ROCm 7.14.1의 공식 Radeon 호환 목록은 RX 7000/9000과 일부 PRO RDNA2를 포함하지만
RX 6600은 포함하지 않는다.

다만 AMD의 TheRock 개발 배포는 별도다.

2026-09-28 기준 TheRock:

- `gfx1032`를 Linux와 Windows 모두 Build Passing / Sanity Tested / Release Ready로 표시
- multi-arch release table에 `AMD Radeon RX 6600 XT / 6600 -> device-gfx1032` 명시
- Linux/Windows용 ROCm/PyTorch multi-arch package 제공
- PyTorch compatibility table에 torch 2.14 조합 명시

따라서 Stage 8의 RX 6600 우선순위는 다음으로 바꾼다.

~~~text
1순위: TheRock multi-arch PyTorch + gfx1032
2순위: torch-directml (Windows fallback)
3순위: CPU
~~~

기존 `HSA_OVERRIDE_GFX_VERSION=10.3.0` 방식은 TheRock gfx1032 native package가 실패했을 때만
별도 Linux 실험으로 둔다.

### 5.2 TheRock는 별도 환경에서 검증

TheRock 문서 자체가 개발 중이며 production-stable이 아니라고 명시하므로 Stage 7 CPU 환경을 덮어쓰지 않는다.

~~~text
.venv-cpu     # 기존 재현 환경
.venv-gfx1032 # TheRock GPU 실험 환경
~~~

설치 버전과 wheel hash, driver version, OS, `torch.__version__`,
`torch.version.hip`, device name을 metadata에 기록한다.

TheRock의 multi-arch 패키지는 `device-gfx1032` target을 사용한다.
실제 설치 명령은 실행 시점의 TheRock release 문서를 그대로 따른다.

### 5.3 DirectML은 fallback

Microsoft의 `torch-directml`은 DirectX 12 GPU에서 training/inference를 지원하지만 Public Preview다.
PyPI 최신 공개 패키지의 갱신 주기도 느리므로, 현재 torch 2.14 코드베이스와 바로 맞는다고 가정하지 않는다.

TheRock가 작동하지 않을 때 별도 환경에서 다음만 검증한다.

- Conv2d / BatchNorm / Linear / Tanh forward
- backward
- Adam step
- checkpoint load
- batch inference

DirectML을 사용하려면 현재 `torch.device(config['device'])` 경로와 다른 device adapter가 필요할 수 있으므로
TheRock보다 구현 우선순위를 낮춘다.

### 5.4 GPU smoke gate

backend가 무엇이든 다음을 통과해야 GPU 경로 구현을 계속한다.

1. GPU enumerate
2. D16 gen160 checkpoint load
3. B=1 forward
4. B=16 forward
5. B=32 train forward/backward/Adam step
6. 100 training steps에서 NaN/Inf 0
7. legal mask 결과 정상
8. checkpoint save/reload
9. 같은 GPU/backend에서 반복 inference tolerance 확인
10. `torch.use_deterministic_algorithms(True)` 실행 가능 여부 기록

---

## 6. Stage 8-D — GPU용 evaluator data path

GPU 이득을 최대화하려면 NN forward만 장치에 올리는 것이 아니라 작은 tensor 생성/전송 횟수를 줄여야 한다.

### 6.1 Batched CPU encoding

GPU optimized evaluator 후보:

~~~text
EvaluationSnapshot x B
        ↓
CPU에서 board / mask / planes batch 생성
        ↓
states[B,6,15,15] + masks[B,225]
        ↓
한 번의 device transfer
        ↓
GPU model forward
        ↓
GPU masked_softmax
        ↓
한 번의 CPU result transfer
~~~

reference `PolicyValueEvaluator`는 그대로 둔다.

새 경로는 예를 들어:

~~~text
BatchedDeviceEvaluator
encode_snapshots_cpu()
~~~

처럼 분리한다.

### 6.2 benchmark batch

raw evaluator benchmark는:

~~~text
B = 1 / 2 / 4 / 8 / 16 / 32 / 64
~~~

까지 측정한다.

단 실제 **의미보존 self-play**에서 한 generation에 동시에 존재할 수 있는 독립 game은 16개이므로,
within-tree batching을 사용하지 않는 한 실전 inference batch의 상한은 대체로 16이다.

B=32/64는 training과 향후 within-tree batching의 가능성을 보기 위한 장치 benchmark다.

기록:

- total batch latency
- ms/position
- positions/s
- encode CPU time
- H2D
- forward
- softmax
- D2H
- GPU utilization
- VRAM
- CPU utilization
- batch output vs B=1 max prior/value difference

ROCm에서 유효할 경우 pinned memory / non_blocking transfer도 별도 측정하되
이득을 가정하지 않는다.

---

## 7. Stage 8-E — multi-game batched PUCT

이 단계가 Stage 8의 **GPU 핵심 구현**이다.

### 7.1 원칙

한 game의 PUCT simulation은 순차성을 유지한다.

~~~text
Game A: sim 1 결과 적용 -> sim 2 selection
Game B: sim 1 결과 적용 -> sim 2 selection
...
~~~

대신 서로 독립인 여러 game에서 현재 필요한 leaf evaluation을 모은다.

~~~text
Game A ─ leaf request ┐
Game B ─ leaf request ├─> evaluate_batch(B) ─> 각 game으로 결과 반환
Game C ─ leaf request ┤
...                   │
Game P ─ leaf request ┘
~~~

한 game/tree에는 동시에 **최대 1개의 미완료 NN request만** 허용한다.

따라서 virtual loss, provisional visits, inherited visits 같은 새 search 의미가 들어가지 않는다.

### 7.2 SearchSession / resumable search

현재 blocking `search_with_tree()`의 내용을 state machine으로 분해한다.

예상 인터페이스:

~~~text
session = SearchSession(game, config, rng)

request = session.advance_until_evaluation()
session.accept_evaluation(result)
...
SearchResult = session.result
~~~

`advance_until_evaluation()`은:

- selection
- play
- legal_moves
- tactical filter
- terminal/proven 처리

를 진행하다 NN leaf가 필요하면 immutable `EvaluationSnapshot`을 반환한다.

결과를 받은 뒤 해당 node expand + backup + undo를 수행하고 다음 simulation으로 진행한다.

### 7.3 기존 `run_search()` 보존

기존 API는 SearchSession 위의 B=1 wrapper로 유지한다.

~~~text
run_search()
    ↓
SearchSession
    ↓
한 request 즉시 evaluator.evaluate_batch([snapshot])
~~~

Uniform/ScriptedEvaluator 기준으로 기존 frozen search 결과가 exact equality여야 한다.

이렇게 해야 GPU scheduler 도입이 Stage 5/7 기준선을 깨지 않는다.

### 7.4 MultiGameScheduler

16개의 self-play game을 동시에 소유한다.

각 game은:

- 자기 `Random(seed)`
- 자기 `Game`
- 현재 ply
- 현재 `SearchSession`
- pending sample list

를 가진다.

scheduler는 active game을 round-robin으로 진행하고 pending NN request를 모아서
최대 `gpu.max_batch_size`까지 한 번에 평가한다.

초기 기본값:

~~~yaml
parallel:
  mode: gpu_batch
  self_play_concurrency: 16

gpu:
  max_batch_size: 16
~~~

batch를 채우기 위해 인위적으로 긴 sleep을 넣지 않는다.
16개 active game에서 즉시 준비된 request를 모아 실행하고 실제 batch-size histogram을 기록한다.

### 7.5 의미보존 계약

CPU B=1 SearchSession wrapper는 기존 search와 exact equality여야 한다.

GPU batched path는 float kernel이 다르므로 game hash exact equality를 요구하지 않는다.
대신:

- game seed 소비 순서 동일
- game마다 RNG 호출 순서 동일
- root visit 합 = simulations
- illegal visit = 0
- illegal move = 0
- evaluator output B=1 대비 tolerance
- search result drift rate 기록
- 같은 GPU/backend/config에서 repeatability 확인

을 계약으로 둔다.

---

## 8. Stage 8-F — GPU가 실제 generation 시간을 줄이는지 측정

raw NN benchmark만 보고 GPU를 채택하지 않는다.

같은 D16 checkpoint와 고정 seed set에서 다음 실행 모드를 비교한다.

| mode | self-play | training | 목적 |
|---|---|---|---|
| CPU-serial | CPU B=1 | CPU | Stage 7 reference |
| CPU-workers | 여러 CPU process, 각 B=1 | CPU | CPU fallback ceiling |
| GPU-B1 | GPU B=1 | GPU | 단순 device 이동의 효과 |
| GPU-batched | 16-game scheduler, B≤16 | GPU | **주 후보** |
| hybrid | CPU producer 병렬화 + shared GPU inference | GPU | GPU starvation 시 후보 |

기록:

- self-play s/gen
- training s/gen
- in-loop eval s/gen
- checkpoint s/gen
- total s/gen
- games/h
- samples/s
- evaluator positions/s
- batch size p50/p95/max
- GPU busy %
- GPU memory
- CPU utilization
- per-move legal/rule/tree/NN 비중

**최종 선택 기준은 positions/s가 아니라 total generation wall time이다.**

### 8.1 in-loop evaluation도 GPU batching 재사용

self-play만 빨라지고 매 generation의 12판 평가가 serial이면 전체 속도 향상이 제한된다.

GPU scheduler가 안정화된 뒤 deterministic evaluation에도 같은 multi-game batching을 재사용한다.

- random/tactical 상대: model 쪽 NN 요청만 batch
- previous 상대: current model 요청과 previous model 요청을 **모델별 queue**로 분리
- MCTS-v6: classical opponent는 CPU, model 요청만 GPU batch

평가 결과의 opening/seed와 색 교환 계약은 그대로 유지한다.

---

## 9. GPU가 굶는 경우의 확장 순서

16-game cross-game batching 후에도 GPU utilization이 낮으면 아래 순서로 병목을 해결한다.

### 9.1 먼저 CPU feeder를 늘린다

tree selection, `legal_moves`, tactical filter는 CPU Python 코드다.

single-process scheduler가 GPU request를 충분히 만들지 못하면 2/4개의 CPU producer process가
각자 game subset을 소유하고 **하나의 GPU inference service**로 요청을 보내는 hybrid를 실험한다.

GPU model copy는 하나만 유지하는 것을 우선한다.

측정 후 IPC/serialization 비용이 큰 경우에만 shared-memory queue를 검토한다.

### 9.2 그 다음에만 within-tree batching을 검토한다

16 concurrent games보다 큰 batch가 반드시 필요하다면 한 tree에서 여러 leaf를 동시에 만드는 방법이 필요하다.

이 단계는 virtual loss/provisional visit 등으로 PUCT 의미가 바뀌므로 **execution-only가 아니다.**
별도 training-critical 실험으로 분리한다.

### 9.3 GPU가 빨라진 뒤 training-critical scale 실험

GPU pipeline이 안정화된 뒤에만 다음을 하나씩 비교한다.

1. self-play simulations 50 vs 100
2. 더 큰 network
3. training batch/step 조합
4. mixed precision 가능성

Stage 7 `9.1에서 100 simulations의 search-only 이득은 제한적이었으므로
단순히 GPU를 더 사용하기 위해 simulations를 늘리지 않는다.
teacher 품질 또는 최종 실력이 좋아지는지 별도 run으로 확인한다.

network 확대 역시 기존 64×4 checkpoint와 architecture가 달라지므로 새 run/초기화 전략을 먼저 설계한다.

---

## 10. Config 확장

기존 checkpoint와 critical hash를 깨지 않도록 실행 설정은 optional/non-critical로 둔다.

~~~yaml
parallel:
  mode: serial                 # serial | cpu_workers | gpu_batch | hybrid
  self_play_workers: 1
  self_play_concurrency: 16

accelerator:
  backend: cpu                 # cpu | rocm | directml
  max_batch_size: 16
  encode_on_cpu: true

health:
  color_imbalance:
    enabled: true
    window: 10
    lower: 0.10
    upper: 0.90
~~~

`parallel`, `accelerator`, `health`는 `NON_CRITICAL`에 둔다.

단 Stage 7 checkpoint에 키가 없으므로 `validate_config`와 resume 경로는
`config.get(...)`으로 이전 config를 허용해야 한다.

regression:

- Stage 7 D16 critical hash = `25ab9c5a…`
- Stage 7 checkpoint가 Stage 8 코드에서 resume 가능
- worker/backend/batch scheduler execution 설정 변경으로 critical hash가 바뀌지 않음
- self-play simulations/network/games_per_generation 변경은 여전히 critical

GPU backend 자체는 execution setting이지만, backend가 바뀌면 bit-exact continuation은 보장하지 않는다는 사실을 metadata에 남긴다.

---

## 11. 외부 평가 주기화

외부 평가는 학습 중 GPU 처리량 측정을 오염시키지 않도록 checkpoint 경계에서 실행한다.

~~~text
resume
→ run_training(stop_after=next_eval_generation)
→ checkpoint
→ external eval + probes
→ resume
~~~

| 종류 | 주기 | 내용 |
|---|---|---|
| light | 320판 = 20 gen | v1 25 sims, Tactical 50판, MCTS-v2 50판 + tactical/defense probes |
| heavy | 1,280판 = 80 gen | light + MCTS-v3.2.1 10판 + MCTS-v7 10판 |

시작 gen160 기준:

- light: 180, 200, 220, 240, …
- heavy: 240, 320, 400, 480, …

`keep_every: 20`과 정확히 맞는다.

외부 평가도 GPU multi-game scheduler가 검증된 뒤 선택적으로 같은 evaluator batching을 쓸 수 있지만,
결과 비교를 위해 search config/opening/seed는 바꾸지 않는다.

---

## 12. 장기 학습 pilot

GPU/CPU 실행 경로를 먼저 선택한 뒤 pilot을 시작한다.

| gate | Stage 8 추가 self-play | generation | 목적 |
|---|---:|---:|---|
| pre-pilot | 3~5 gen | 160 → 163~165 | CPU/GPU correctness와 throughput |
| Gate 1 | 512판 | 160 → 192 | 선택한 최종 execution path 안정성 |
| Gate 2 | 2,048판 | 160 → 288 | 외부 평가/색별 성능/probe 추세 |
| Gate 3 | 5,120판 선택 | 160 → 480 | Gate 2에서 퇴행 없을 때 |

Gate 1은 더 이상 CPU worker=1로 고정하지 않는다.
GPU batched path가 smoke/재현성/처리량 gate를 통과하면 **Gate 1부터 GPU 경로를 사용**한다.

판단은 루프 내 previous 단기 승률이 아니라:

- MCTS-v2/Tactical 색별 외부 평가
- v321/v7
- policy/value probe
- defense probe
- color health
- game length
- crash/NaN/illegal

을 함께 본다.

---

## 13. PR 분리

| PR | 내용 |
|---|---|
| 8-A | Stage 8 문서 + 데스크톱 CPU 기준선 |
| 8-B | health monitoring + backward-compatible config |
| 8-C | GPU device/training batch transfer + TheRock gfx1032 smoke/benchmark |
| 8-D | batched CPU encoding + device evaluator |
| 8-E | SearchSession + 16-game GPU scheduler, 기존 B=1 exact regression |
| 8-F | CPU-worker/hybrid feeder + in-loop evaluation batching(필요한 경우) |
| 8-G | external evaluation orchestration |
| 8-H | 512/2,048/5,120 pilot 결과와 최종 config |
| 별도 실험 | simulations/network/within-tree batching/mixed precision/tree reuse |

8-C에서 RX 6600 GPU backend가 실패하면 8-D/8-E를 억지로 진행하지 않고 CPU-workers를 주 경로로 전환한다.
반대로 GPU backend가 정상이고 B=8~16에서 CPU보다 유의하게 빠르면 CPU multiprocessing을 먼저 완성하는 대신
8-D/8-E를 우선한다.

---

## 14. GPU 경로 완료 기준

Stage 8 전체 완료 기준과 별도로 GPU path는 다음을 만족해야 한다.

1. RX 6600 backend가 checkpoint forward/backward/Adam을 실행한다.
2. 100-step smoke에서 NaN/Inf가 없다.
3. CPU B=1 대비 evaluator output 차이가 문서화된 tolerance 안이다.
4. batched encoder가 snapshot별 GPU encode보다 빠르거나 최소한 병목을 악화시키지 않는다.
5. SearchSession B=1 wrapper가 기존 Uniform/Scripted search와 exact equality다.
6. 16-game batched scheduler에서 illegal move/visit 0건이다.
7. 실제 batch-size 분포를 기록한다.
8. GPU-B1보다 GPU-batched가 빠르다.
9. best CPU path와 GPU path의 total generation wall time을 비교한다.
10. 선택한 backend/driver/wheel 버전을 run metadata에 남긴다.
11. 중단/resume 후 정상 동작한다.
12. GPU path가 실패해도 CPU reference path로 즉시 돌아갈 수 있다.

---

## 15. Stage 8 완료 기준

1. D16 gen160 checkpoint가 데스크톱에서 복원되고 file SHA-256과 critical hash가 같다.
2. rule/frozen/golden/training regression이 모두 통과한다.
3. CPU serial 기준선이 기록된다.
4. RX 6600 TheRock gfx1032 경로를 실제 검증한다.
5. 실패 시 DirectML 또는 CPU fallback 결과를 기록한다.
6. training batch device transfer가 올바르게 동작한다.
7. GPU evaluator의 batch data path를 profile했다.
8. multi-game batched PUCT가 기존 per-game search 의미를 보존한다.
9. CPU/GPU/hybrid의 generation wall time을 비교해 지속 가능한 실행 방식을 하나 선택한다.
10. `color_imbalance`와 healthy-reference 기반 `color_regression`이 동작한다.
11. 320판/1,280판 외부 평가와 probe orchestration이 동작한다.
12. Gate 1 512판이 crash·NaN·illegal 없이 끝난다.
13. Gate 2 2,048판까지 진행할 수 있는 장기 설정을 확정한다.
14. GPU utilization을 더 높이기 위한 training-critical 변경은 별도 실험으로 분리되어 있다.
15. 최종 config/환경/성능 결과를 문서화하고 `v0.7-stage8` 태그를 만든다.

Stage 8의 핵심 산출물은 단순한 multiprocessing 코드가 아니다.

~~~text
검증된 D16 학습 의미
+
RX 6600에서 실제 동작하는 accelerator backend
+
여러 게임을 묶는 batched PUCT inference
+
CPU tree/rule 처리와 GPU NN 처리의 균형
+
색 붕괴 감시
+
자동 외부 평가
+
장기 학습 가능한 checkpoint
~~~

이 구조가 완성된 뒤 Stage 9에서 고정 예산으로 최종 기력과 사람 대국 성능을 평가한다.
