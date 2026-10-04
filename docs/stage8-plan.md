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

### 2.4 GPU 한 장으로 얻을 수 있는 상한 (Amdahl)

GPU는 NN 부분만 줄인다. selection·`legal_moves`·규칙 필터·play/undo·encode는 CPU Python에 남는다.
사용자 PC의 D16 gen 160 profile(v2 50 sims, probe 국면, 최적화 후)에서 착수당 72 ms 중
NN forward·softmax·전송은 약 40 ms, CPU에 남는 부분(흑/백 `legal_moves` 13.2, 규칙 필터 6.0, tree 6.4,
snapshot 2.1, encode 3.3, play/undo 0.9 ms)은 약 32 ms였다.

~~~text
단일 프로세스 GPU batching 상한 ≈ 72 / 32 ≈ 2.2배 (NN 시간이 0이 되어도)
CPU worker k개                   ≈ k배에 가까움 (코어가 남는 한)
hybrid (producer k개 + GPU)      ≈ k × (1 / CPU 잔여 비중)
~~~

probe 국면은 즉시 승리·필수 방어가 많아 착수당 NN 호출이 적다(12.2회). 실제 self-play 값은 기존 D16
`metrics.jsonl`의 `self_play_timing.searched`로 계산했다(새 실행 없음).

**실측(D16 마지막 20 generation, 그램, D32 동시 실행 중, `legal_moves` 최적화 전):**

| 항목 | 값 |
|---|---|
| 착수당 전체 | 246.8 ms |
| `inference_ms` (snapshot + encode + NN + 검증) | 200.9 ms (81%) |
| 나머지 (`legal_moves` + tree) | 45.9 ms |

~~~text
낙관 상한 = 246.8 / 45.9 ≈ 5.4배        (inference 전체가 0이 된다고 가정)
보수 상한 ≈ 246.8 / (45.9 + 약 26) ≈ 3.4배 (inference 안의 CPU 부분 약 13% = snapshot·encode는 남음, probe 비율 적용)
~~~

실제 self-play에서는 **NN이 착수 시간의 약 80%**이므로 GPU 단일 프로세스만으로도 약 3.4~5.4배까지 가능하다.
그 뒤 `legal_moves`가 2.6배 빨라져 CPU 잔여가 줄었으므로 실제 상한은 이 범위의 위쪽일 가능성이 있다. GPU 우선 방향을 유지한다.
단, 이 상한은 GPU batch가 NN 시간을 거의 없앨 만큼 빠르고 batch가 충분히 찰 때의 값이다. §6.2와 §7.4의 실측으로 확인한다.

결론: **GPU는 CPU 병렬화를 대체하지 않고 곱해진다.** GPU-batched 단일 프로세스(상한 약 3.4~5.4배)가
주 경로다. 데스크톱 CPU 코어 수가 이 배수보다 많으면 CPU worker가 더 빠를 수 있으므로 §8에서 함께 비교한다.
GPU가 굶는 것이 확인되면 hybrid(§9.1)로 곱한다.

### 2.5 기존 CPU/frozen 경로는 바꾸지 않는다

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
runs/stage8_d16/   # Stage 8 복사본 (configs/stage8_d16.yaml, 실행 방식과 무관하게 같은 run)
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

추가로 두 가지를 잰다(코드 변경 없음).

- 기존 D16 `metrics.jsonl`의 `self_play_timing`으로 실제 self-play의 NN 비중과 GPU 단일 프로세스 상한(§2.4)
- **CPU 확장성 프록시:** `--verify-checkpoint` 같은 1-thread self-play 프로세스를 k개(1/2/4/물리 코어 − 1)
  동시에 띄워 프로세스당 속도 저하를 잰다. multiprocessing을 구현하지 않고도 CPU-workers 상한을 알 수 있다.

**Gate 1 전체를 CPU serial로 먼저 돌리는 것을 Stage 8 선행 조건으로 두지 않는다.**
GPU 경로를 최대한 빨리 검증하기 위해 correctness와 throughput 기준선에 필요한 짧은 run만 수행한다.

### 3.2.1 데스크톱 CPU 기준선 실측 (i5-12600K, Windows, Python 3.13, torch 2.14 CPU, 1 thread)

`configs/stage8_d16.yaml`로 `runs/stage8_d16`(D16 복사본)을 gen 160 → 163까지 이어서 실행했다. 비교 대상은 그램 D16의 마지막 10세대(gen 150~159)다.

| 항목 (generation당) | 그램 gen 150~159 | 데스크톱 gen 160~162 | 배율 |
|---|---|---|---|
| self-play | 36.2 s | 22.0 s | 1.65 |
| 학습 50 step | 10.0 s | 6.4 s | 1.56 |
| 루프 내 평가 12판 | — | 4.7 s | |
| 착수당 시간 (searched) | 184 ms | 118.7 ms | 1.55 |
| └ `inference_ms` | 146.7 ms (80%) | 93.8 ms (79%) | |
| └ tree | 34.4 ms | 23.9 ms | |
| 착수당 NN 호출 | 37.6 | 37.2 | |

- 정상 동작: 불법수 0, checkpoint 161~163 저장. `color_imbalance` 경고는 gen 160에 한 번 나왔다(백 우세, 흑 4.4%,
  극단 구간 37세대 연속). gen 161~162에서는 중복 억제가 동작했다.
- generation당 약 34 s(self-play 64%, 학습 19%, 평가 14%, 나머지 약 3%)이면 **CPU 직렬만으로 약 1,700판/시간**이다.
  gate 1(640판)은 약 25분, gate 2(2,560판)는 약 1.5시간, gate 3(5,120판)은 약 3시간이다.
- **generation 수준의 Amdahl:** self-play만 무한히 빨라져도 학습과 평가(약 11 s)가 남아 전체는 최대 약 3배다.
  GPU로 세대 시간을 크게 줄이려면 학습(GPU), 루프 내 평가 batching(§8.1)도 함께 줄여야 한다.
- 결론: 현재 64×4 / 50 sims / 16판 설정에서는 **처리량이 Stage 8 gate의 병목이 아니다.** gate 1~2는 CPU 직렬로
  바로 진행할 수 있다(§12). GPU 경로는 Stage 9 이후의 규모 확대(더 큰 network, 더 많은 simulation, 더 긴 학습)를
  위한 투자로 계속 개발한다.

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

따라서 각 색에 대해 **healthy reference 평가**를 유지한다. reference는 지금까지 healthy였던 결과 중 **가장 좋은 것**이다
(마지막 healthy 결과를 쓰면 17 → 14 → 11 → 8처럼 조금씩 떨어질 때 기준도 같이 내려가 붕괴를 놓친다).

light evaluation마다:

1. 같은 fixed openings에서 MCTS-v2 25판/색 결과를 얻는다.
2. 현재 결과를 그 색의 healthy reference와 Fisher 단측 검정한다.
3. p < 0.05로 하락하면 `color_regression_candidate`를 기록하고 reference는 유지한다.
4. 다음 light 평가에서도 **같은 reference 대비** p < 0.05 하락이면 `color_regression_confirmed`로 올린다.
5. 하락이 아니면 healthy로 돌아가고, 현재 결과가 reference보다 좋으면 reference를 갱신한다.

구현: `training.health.update_color_regression`(8-B). 외부 평가 orchestration(8-G)이 light 평가마다 호출한다.

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

확인한 원문(2026-09-28, `ROCm/TheRock` `SUPPORTED_GPUS.md`, `RELEASES.md`):

- gfx1032는 Linux·Windows 모두 Build Passing ✅ / Sanity Tested ✅ / Release Ready ✅다.
- 같은 문서에 "still under active development and is not yet stable for production use"라고 적혀 있다.
  Build Passing은 "런타임이 대상 하드웨어에서 동작한다는 뜻이 아니다"라는 주석도 있다.
- PyTorch 설치: `pip install --index-url https://nightly.repo.amd.com/rocm/whl-next/ "torch[device-gfx1032]" …`.
  이 주소는 **nightly 저장소**이므로 설치 시점의 정확한 버전과 wheel hash를 기록해 고정한다.
- 확인 절차가 `torch.cuda.is_available()`, `torch.cuda.get_device_name(0)`이다. ROCm PyTorch는 **`cuda` device API를
  그대로 쓰므로** 코드에서는 `device: cuda`로 지정한다(§10).

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

**구현(8-C):**

- `training.trainer.batch_to_device`: `train_step`이 batch를 모델이 있는 장치로 옮긴다(loss 텐서 4개만, provenance는 CPU에 둔다).
  batch가 이미 그 장치에 있으면 같은 객체를 그대로 돌려주므로 CPU 학습은 비트 단위로 이전과 같다.
  자가대국과 루프 내 평가는 이미 `config['device']`를 evaluator에 넘기고 있었다.
- `scripts/benchmark_stage7_batch.py`: `benchmark(..., device=)`. forward 시간은 동기화한 뒤 잰다.
- `scripts/check_stage8_gpu.py`: 위 게이트를 실제 checkpoint(D16 gen 160)로 한 번에 점검하고 JSON으로 남긴다.
  1. backend 정보
  2. B=1·B=16 추론을 CPU와 비교(허용 오차 1e-3)
  3. 같은 batch 반복 추론 비교
  4. checkpoint의 replay·optimizer로 실제 학습 100 step(유한성, step당 ms를 CPU와 비교)
  5. 장치에서 저장한 checkpoint를 CPU에서 다시 읽어 가중치 동일 확인
  6. `use_deterministic_algorithms(True)`에서 forward+backward
  7. B=1…64 벤치마크(장치/CPU)

  run 디렉터리에는 아무것도 쓰지 않는다. `--device cpu`는 스크립트 자체를 검증하는 모의 실행이다(`tests/test_stage8_gpu.py`).
- 주의: 학습 CLI(`run_stage6_training.py`, `run_stage8_training.py`)는 `torch.use_deterministic_algorithms(True)`를 켠다.
  6번이 실패하면 GPU 학습 전에 이 설정을 장치별로 다루는 변경이 필요하다.

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

**꼬리 효과:** 16판은 동시에 시작하지만 길이가 다르다(D16 후반 평균 약 12수). 먼저 끝난 게임이 빠지면서
generation 후반에는 batch가 줄어든다. 다음 generation의 게임은 새 가중치가 필요하므로 앞당겨 시작할 수 없다.
그래서 active game 수와 batch 크기의 시간 분포를 함께 기록하고, 평균 유효 batch로 §6.2 벤치마크를 해석한다.

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

판단 규칙:

- GPU-batched(단일 프로세스)가 CPU-workers보다 느리거나 비슷하면 **hybrid로 바로 간다**(§9.1). §2.4 상한상
  코어가 많을수록 이 경우가 흔하다.
- hybrid도 CPU-workers 대비 1.3배 미만이면 복잡도를 감수할 가치가 없으므로 CPU-workers를 채택한다.
- 벤치마크 중에는 학습이나 다른 측정을 동시에 돌리지 않는다. 전원은 최고 성능 모드로 둔다(stage7-plan §4).

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

Stage 7 §9.1에서 100 simulations의 search-only 이득은 제한적이었으므로
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
  backend: torch               # torch | directml
  max_batch_size: 16
  encode_on_cpu: true

health:
  color_imbalance:
    enabled: true
    window: 10
    lower: 0.10
    upper: 0.90
~~~

장치는 이미 있는 `device` 키(non-critical)로 지정한다. ROCm PyTorch는 `device: cuda`, CPU는 `device: cpu`다.
`accelerator.backend`는 DirectML처럼 문자열 device로 표현할 수 없는 경우에만 `directml`로 둔다.
backend 종류(`torch.version.hip`, driver, wheel)는 run metadata에 기록한다.

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

**구현(8-G):** `scripts/run_stage8_training.py` + `training.schedule`.

- `<run>/checkpoints/latest.pt`에서 이어가며, 다음 평가 지점(`anchor + k × light_every`, 목표 generation으로 제한)에서
  `run_training(stop_after=…)`로 멈춘다.
- 이미 지난 평가 지점을 먼저 따라잡는다. 예: 데스크톱 기준선이 gen 163에서 멈췄으면 gen 160 light+heavy부터 평가한다.
- 출력:
  - `external_eval/genNNN_light.json`: Tactical·MCTS-v2 각 25쌍, 규칙 끈 v1, seed 7007(Stage 7과 같은 오프닝)
  - `external_eval/genNNN_heavy.json`: v321·v7 각 5쌍
  - `probes/genNNN.json`, `probes/genNNN_defense.json`
  - `external_eval/color_regression.json`: 모든 light 결과를 generation 순으로 다시 계산
- 파일이 있으면 건너뛰고 원자적으로 쓴다. 중단 후 다시 실행해도 안전하고, `--eval-only`는 평가만 따라잡는다.
- `color_regression`이 `confirmed`이면 로그에 `*** CONFIRMED`를 남긴다. 학습은 멈추지 않는다(§4.2, 사람이 판단).
- 시작할 때 `keep_every`가 `light_every`와 `anchor`를 나누는지 검사한다.
- `tests/test_stage8_schedule.py`:
  - 평가 지점 계산
  - D32형 붕괴는 후보 → 확정으로 잡히고, D16형은 경보 없음
  - 작은 run으로 지점별 파일 생성, 재실행 시 파일 불변
  - **구간으로 끊어 학습한 모델 = 한 번에 학습한 모델**(가중치 동일)

---

## 12. 장기 학습 pilot

GPU/CPU 실행 경로를 먼저 선택한 뒤 pilot을 시작한다.

| gate | Stage 8 추가 self-play | generation | 목적 |
|---|---:|---:|---|
| pre-pilot | 3~5 gen | 160 → 163~165 | CPU/GPU correctness와 throughput |
| Gate 1 | 640판 | 160 → 200 | 선택한 최종 execution path 안정성 |
| Gate 2 | 2,560판 | 160 → 320 | 외부 평가/색별 성능/probe 추세 (heavy 평가 지점) |
| Gate 3 | 5,120판 선택 | 160 → 480 | Gate 2에서 퇴행 없을 때 (heavy 평가 지점) |

gate 끝을 평가 지점(light 20세대, heavy 80세대 간격, gen 160 기준)에 맞췄다. 원래 목표였던 512 / 2,048판(gen 192 / 288)은
평가 지점이 아니어서, gate가 끝나는 checkpoint에 외부 평가가 돌지 않기 때문이다. 데스크톱 CPU 기준선(§3.2.1)으로
Gate 1은 약 25분, Gate 2는 약 1.5시간이다.

Gate 1은 더 이상 CPU worker=1로 고정하지 않는다.
GPU batched path가 smoke/재현성/처리량 gate를 통과하면 **Gate 1부터 GPU 경로를 사용**한다.

선택: GPU 경로 개발이 길어지면 그동안 데스크톱을 CPU serial로 Gate 1에 쓸 수 있다. 실행 방식은 critical이
아니므로 CPU로 만든 checkpoint에서 나중에 GPU 경로로 이어가도 학습 계약이 유지된다(기계·backend 간 bit 재현만
포기). 벤치마크를 잴 때는 학습을 멈춘다.

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

### 12.1 Gate 1~2 결과 (gen 160 → 320, 2,560판, 데스크톱 CPU)

**운영: 통과.**

- 불법수 0, generation·global step 연속, checkpoint SHA가 metrics·probe·외부 평가에서 모두 일치했다.
- gen 160 외부 평가가 그램의 Stage 7 결과와 한 판까지 같았다.
- 자동 평가·경고·resume이 모두 정상 동작했다.

**실력: 정체.**

- MCTS-v2: 43, 42, 37, 44, 38, 35, 43, 42, 40 / 50(평균 80.9%, 추세 없음)
- v321: 7 → 4 → 6/10, v7: 계속 0/10
- probe: Gate 1(첫 640판)에서 오른 뒤 유지만 했다. 즉시 승리 배율 5 → 11~12, 방어 top-3 40 → 약 52%.

**색 편향.** gen 220~319의 자가대국 1,600판 중 흑이 1,564승(97.75%)을 거뒀다. 같은 기간:

- value loss가 0.42 → 0.086으로 떨어졌다. 결과를 예측하기 쉬워진 것이지, 판을 더 잘 이해한 것이 아니다.
- **value probe가 색으로 갈렸다.** 백 즉시 승리 국면의 value 평균이 gen 200의 +0.48(정답률 80%)에서 gen 320의 −0.31(20%)로 떨어졌다.
  흑 패배 국면은 gen 220~300에 +0.19~+0.45였다.
- 네트워크가 "흑 차례 = 좋음, 백 차례 = 나쁨"이라는 색 지름길을 배운 것으로 본다. 전체 separation(+0.40)만으로는 보이지 않았다.
- 외부 평가에서도 gen 160 → 320에 흑 20 → 24, 백 23 → 16으로 능력이 색 사이에서 옮겨갔다.
  백 `color_regression` 후보가 두 번(gen 260, 320) 나왔지만 연속이 아니라 확정되지는 않았다.

**학습 속도.** 세대당 학습 시간이 6.4 → 22.5 s로 늘었다. 원인은 subnormal(denormal) 값이다.
gen 320에서 가중치의 0.8%, Adam 상태의 0.5%가 subnormal이었다. `torch.set_flush_denormal(True)`로 50 step이 20.6 → 6.3 s가 됐다.
로그 파일 추가 가설은 기각했다(5MB 파일과 새 파일의 차이 < 1 ms).

**수정한 방향:**

1. `flush_denormal`(execution-only, 기본 false, `stage8_d16.yaml`은 true)
2. 색 분리 계측:
   - probe의 `value_overall.by_color`·`color_bias`
   - 평가 지점마다 `external_eval/genNNN_replay.json`(replay buffer의 차례 색 × value target 수)
3. **`runs/stage8_d16`은 gen 320에서 동결**한다(Gate 1~2 증거물). 같은 설정으로 gen 480까지 늘리지 않는다.
   색 편향을 더 강화할 위험이 있기 때문이다.
4. **Gate 3을 "조건 A/B 실험"으로 다시 정의한다.** 모두 새 run이고 한 번에 한 변수만 바꾼다.
   - (a) **평가 전용 탐색 예산**: gen 320(과 gen 200)을 25/50/100 sims로 MCTS-v2·v321·v7과 붙인다.
     병목이 network인지 search인지 가른다.
   - (b) 색 편향 대응(학습 조건 변경): 흑·백 승리 게임 표본 균형 샘플링, replay 창 확대 등.
     색 계측 결과를 본 뒤 설계한다. 출발점은 value가 두 색 모두 정상이던 gen 200을 우선 검토한다.
   - (c) (a)에서 100 sims가 확실히 나을 때만 self-play 50 vs 100 sims A/B
   - network 확대(96×6 등)는 색 편향 문제를 푼 뒤로 미룬다. 편향된 데이터를 더 잘 외울 수 있기 때문이다.

### 12.2 GPU 경로의 우선순위 (Gate 2 이후 조정)

Gate 1~2에서 데스크톱 CPU만으로 세대당 약 35 s(denormal 수정 후), 시간당 약 1,700판이 나왔다. 지금 병목은
처리량이 아니라 **학습 품질**(색 편향, 실력 정체)이다. 그래서:

- **8-D(GPU용 batch 인코딩), 8-E(여러 게임 묶음 탐색)는 보류한다.** 8-C의 `batch_to_device`와 확인 스크립트는 이미 들어가 있다.
- **`check_stage8_gpu.py` 1회 실행(약 10분)은 권장한다.** 컴퓨터가 쉬는 동안 RX 6600이 TheRock으로 동작하는지만 확인해 둔다.
- **재개 조건:** 채택한 학습 조건 변경 때문에 세대 시간이 약 2분을 넘거나, 계획한 run이 하룻밤(약 8시간)을 넘으면 재개한다.
  예: self-play 100 sims는 자가대국이 약 2배, network 96×6은 NN이 약 2~3배 늘어난다.
- **재개할 때의 순서:** 먼저 구현이 단순한 CPU worker 병렬화(P코어 6개, 약 4~5배)를 검토한다.
  더 필요하면 GPU 경로(8-D/8-E, hybrid)로 간다.

### 12.3 Gate 3 준비 측정 결과 (gen 160~320, 학습 없음)

**replay buffer의 value target (차례 색별):**

| gen | 흑 차례 +1 / −1 | 백 차례 +1 / −1 | 흑승 게임에서 온 표본 |
|---|---|---|---|
| 160 | 447 / 4,588 | 4,589 / 376 | 약 8% |
| 200 | 580 / 4,465 | 4,465 / 490 | 약 11% |
| 240 | 3,406 / 1,887 | 1,887 / 2,820 | 약 62% |
| 280 | 5,244 / 225 | 225 / 4,306 | 약 96% |
| 320 | 5,271 / 200 | 200 / 4,329 | 약 96% |

- 흑 차례의 −1 수와 백 차례의 +1 수는 매번 같다(백승 게임의 양쪽 표본). value 부호 변환은 정상이다.
  buffer는 최근 약 60세대 self-play 결과를 그대로 반영한다.
- buffer는 **백승 약 92%(gen 160)에서 흑승 약 96%(gen 280~320)로 뒤집혔다.** 어느 시점에도 균형이었던 적은 거의 없다.

**value 색 편향(`color_bias`, 같은 정답 조건에서 흑 value − 백 value):**

| gen | 160 | 200 | 240 | 280 | 320 |
|---|---|---|---|---|---|
| `color_bias` | −0.23 | **−0.03** | +0.31 | **+0.45** | +0.26 |

buffer 구성을 약간 늦게 따라간다. **두 색이 모두 정상인 것은 gen 200뿐**이다(두 색 모두 승리 국면 정답률 85%).

**탐색 예산 비교(평가 전용, 상대별 20판):**

| | MCTS-v2 25/50/100 sims (흑+백) | v321 | v7 |
|---|---|---|---|
| gen 200 | 17 / 19 / 18 (백 9 / 10 / 9) | 11 / 9 / 10 | 0 / 0 / 0 |
| gen 320 | 14 / 16 / 16 (백 **4 / 6 / 6**) | 13 / 13 / 12 | 0 / 1 / 1 |

- 탐색량 효과가 없다. 두 checkpoint를 합쳐 25 → 50 sims는 31 → 35/40으로 p = 0.19이고, 50 → 100은 차이가 없다.
  **병목은 search가 아니라 network(value의 색 편향 포함)다.** self-play 100 sims 실험은 하지 않는다.
- MCTS-v2 상대, 세 가지 sims를 합친 결과:
  - gen 200 54/60 대 gen 320 46/60(p = 0.04)
  - 백만 보면 **28/30 대 16/30(p = 0.0005)**
- gen 320은 v321에서 38/60 대 30/60(p = 0.10)으로 약간 앞서지만 유의하지 않다.
- 기본 light 평가에서 gen 200의 흑 16/25는 잡음으로 보인다. 같은 오프닝 10쌍, 세 가지 sims에서 흑 26/30이었다.

**GPU 확인(RX 6600, TheRock torch 2.14.0+rocm10.2.0a20260927, HIP 7.17): PASS.**

- 추론 차이: CPU 대비 prior 3.6e-7, value 1.2e-6. 반복 추론 결과 동일, 결정적 연산 모드 가능.
- 학습 step: **12.5 대 134.7 ms(10.7배)**
- 추론 전체 경로(ms/국면): B16 0.75 대 CPU 1.44(1.9배), B64 0.62 대 1.39(2.2배)
- forward만 보면 B64 0.035 대 1.28 ms로 36배다. 따라서 GPU 전체 경로 시간의 약 94%가 인코딩·전송 오버헤드이고,
  §6의 CPU batch 인코딩이 효과를 낼 부분이 여기다.
- CPU B=1 행(9.4 ms/국면)은 믿기 어렵다. B=2(1.8 ms)보다 5배 느리고, 실제 self-play에서 잰 호출당 약 2.5 ms와도 맞지 않는다.
  **GPU B=1이 CPU보다 빠르다고 결론 내리지 않는다.**

### 12.4 Gate 3 설계 — 색 균형 A/B

**가설:** self-play 한쪽 색 독주 → replay buffer 색 쏠림 → value가 색 지름길을 학습 → 약한 쪽 색의 수 선택이 나빠지고
실력이 정체된다. gen 200 → 320에서 buffer 흑승 11% → 96%, `color_bias` −0.03 → +0.45, MCTS-v2 상대 백 28/30 → 16/30이
같은 방향으로 움직였다.

**출발점:** gen 200 가중치(두 색 value가 모두 정상인 유일한 지점)를 export해 **새 run 세 개를 동시에** 시작한다.

- 학습 조건을 바꾸면 기존 run을 resume할 수 없어서 새 run으로 시작한다(optimizer·replay 초기화).
- 세 arm 모두 같은 조건에서 출발하므로 공정하다.
- 각 1-thread라 동시에 돌려도 결과는 같고 속도만 느려진다.

| arm | 바꾸는 것 (한 가지) | 목적 |
|---|---|---|
| C (대조군) | 없음(stage8_d16과 같은 학습 조건, `flush_denormal`) | 새 run에서도 색 쏠림이 재현되는지 |
| B (균형 샘플링) | 학습 batch를 흑승 게임과 백승 게임 표본에서 반씩 뽑는다. 한쪽이 없으면 균등 샘플링 | buffer 쏠림이 value에 전달되는 경로를 끊는다 |
| W (긴 replay) | `replay_capacity` 10,000 → 40,000(약 240세대 분량) | 흑백 우세 교대를 평균낸다. 새 데이터 비중이 줄어드는 대가 |

- 길이는 각 **80세대(1,280판)**이고, 20세대마다 light 평가, 80세대에 heavy 평가를 한다. 세 arm 동시 실행으로 약 1.5~2시간이다.
- **판정(gen 280 기준, C와 비교):**
  1. `|color_bias|` ≤ 0.15 유지
  2. MCTS-v2 백 ≥ 20/25
  3. MCTS-v2 전체가 C 이상
  4. v321 heavy가 C 이상
  5. 흑 방어 probe 유지
- B는 소수 색 게임이 적을 때(예: 100세대에 36판) **같은 게임을 과하게 반복 학습**할 위험이 있다.
  소수 색 표본의 재사용 배율도 기록한다.
- network 확대(96×6)와 self-play sims 변경은 이 결과가 나온 뒤로 미룬다.

**함께 만들 도구:**

- `training.balanced_sampling`: 기본값 끔. 이 키가 없는 기존 설정·checkpoint는 critical hash가 그대로다.
  게임 승자는 표본의 차례 색(인코더 평면 3)과 value target으로 알 수 있어서 buffer 형식을 바꾸지 않는다.
- checkpoint 대 checkpoint 직접 대국(색 교대, 100판 이상): gen 200 / 280 / 320과 A/B 결과 모델의 순위를 정한다.
- GPU 학습은 선택 사항이다. self-play도 같은 `device`를 따라가서 B=1 GPU가 되고, 이득이 확인되지 않았다. 당분간 CPU로 한다.

### 12.5 Gate 3 구현과 실행 계획 (약 8시간)

**구현:**

- `training.balanced_sampling`:
  - `ReplayBuffer.winner_groups()`는 표본의 차례 색과 value target으로 흑승/백승 게임 표본을 나눈다(무승부는 제외).
  - `sample_indices_balanced()`는 batch의 짝수 자리를 흑승, 홀수 자리를 백승 표본에서 균등하게 뽑는다.
    한쪽이 비어 있으면 균등 샘플링과 완전히 같다.
  - 기본값 false는 critical config에서 빠지므로 기존 hash가 모두 그대로다.
  - 켜져 있으면 generation 이벤트에 `balanced_sampling`(두 그룹 표본 수, 소수 색 표본당 이번 세대 예상 반복 수)을 남긴다.
- `configs/stage8_g3_{c,b,w}.yaml`:
  - C는 `25ab9c5a…`(D16과 같음), B는 `balanced_sampling`, W는 `replay_capacity` 40,000만 다르다.
  - 공통: 초기 가중치 = stage8_d16 gen 200 export, 480 generation(7,680판), `keep_every` 20, `keep_checkpoints` 3,
    `flush_denormal`, milestones 끔.
- `run_stage8_training.py --new-run --anchor 0`: 새 run을 만들고 gen 0(초기 가중치)부터 평가 일정을 따른다.
  같은 명령을 다시 실행하면 이어간다.
- `scripts/run_stage8_head_to_head.py`: checkpoint 라운드로빈.
  - 오프닝 쌍마다 색을 바꿔 두 판씩 둔다(v1 25 sims, seed 8008).
  - 결과는 A의 점수, A의 색별 성적, 양측 이항 p, Elo 추정이다.
- `tests/test_stage8_gate3.py`

**실행:** 세 arm을 동시에 돌린다(각 1 thread, 데스크톱 CPU 코어 10개).

- 단독 실행 기준 세대당 약 35 s + 평가(light 24회, heavy 6회)로 arm당 약 5.5~6시간이다.
  동시 실행에 따른 경쟁을 넣으면 **약 7~8시간**이다.
- 20 generation 뒤의 실측 속도로 목표를 조정한다. `--target-generation`으로 줄이거나 늘릴 수 있다.
- 디스크: checkpoint 약 80 MB(C, B), 약 320 MB(W) × 약 27개 보존 → **여유 공간 약 15 GB 필요.**

**판정(gen 160·320·480, C와 비교):**

1. `color_bias` 절댓값 ≤ 0.15 유지
2. replay 흑승 표본 비율이 극단(>90% 또는 <10%)에 머무는 기간
3. MCTS-v2 백 ≥ 20/25, 전체 ≥ C
4. v321 heavy(10쌍) ≥ C
5. 방어 probe 유지
6. 마지막에 C·B·W와 d16 gen 200·320의 라운드로빈(쌍당 50오프닝 = 100판)

### 12.6 Gate 3 결과 (arm당 480 generation, 7,680판, 세 arm 동시 실행)

| | C 대조군 | **B 균형 샘플링** | W replay 40k |
|---|---|---|---|
| self-play 흑승(전체 / 마지막 80세대) | 97.1% / 99.7% | **54.6% / 29.2%** | 98.5% / 98.8% |
| replay 흑승 표본이 90% 초과 또는 10% 미만인 평가 지점 | 19/24 | **1/24** | 24/24 |
| MCTS-v2 gen 480 (흑 / 백) | 36 (22 / 14) | **48 (25 / 23)** | 36 (21 / 15) |
| MCTS-v2 백 ≥ 20/25인 평가 지점 | 6/24 | **23/24** | 0/24 |
| `color_regression` 경보 | 9 (확정 5) | **0** | 13 (확정 9) |
| v321 heavy(gen 80~480, 20판씩) | 11·6·6·8·6·8 | **17·15·14·14·15·11** | 11·8·7·6·7·6 |
| v7 | 0 | 0(gen 320만 1/20) | 0 |
| 라운드로빈(400판) | 32.5% | **80.5%** | 39.75% |

- **B만 효과가 있었다.** 균형 샘플링은 "한 색 승리만 학습 → 그 색이 더 강해짐 → 자가대국에서 그 색만 이김 → replay가 더 쏠림"
  이라는 되먹임 고리를 끊었다. replay 비율은 13~95%로 계속 움직이지만(완전히 50:50은 아님) 한쪽에 갇히지 않았다.
  소수 색 표본의 세대당 반복 수는 중앙값 0.3, 최대 10.4로, 같은 게임을 과하게 외울 위험은 제한적이었다.
- **라운드로빈:** B480은 d16_200에 68-32(p = 0.0004, 약 +130 Elo), d16_320에 78-22, C480에 89-11, W480에 87-13으로 이겼다.
  B의 색별 성적은 흑 83.5%, 백 77.5%였다. C480은 백일 때 11.5%에 그쳤고, B를 상대로는 백 0/50이었다.
- **W는 효과가 없다.** C와 W는 gen 60까지 결과가 완전히 같았다(같은 seed, replay가 1만 개를 넘기 전까지 조건이 같음). 갈라진 뒤에도
  흑 편향이 오히려 오래 남았다. replay를 키우는 방향은 종료한다.
- **d16_200 대 d16_320:** 42-58, p = 0.13으로 차이가 유의하지 않다. Gate 2의 "gen 320이 더 강하지 않다"와 일치한다.
- **학습 시간:** 세대당 6.3 s로 끝까지 일정했다. `flush_denormal` 수정이 효과를 냈다.

**정정: probe `color_bias`는 판정 기준으로 쓰지 않는다.** replay가 95~99% 흑승인 C도 24개 지점 중 13개에서 |bias| ≤ 0.15였고,
값이 지점마다 −0.18~+0.61로 크게 흔들렸다. 색마다 20개 안팎의 value probe로는 잡음이 커서 편향을 가를 수 없다.
참고 지표로만 남기고, 판정은 **색별 외부 평가(MCTS-v2·v321), replay 흑승 비율, 라운드로빈 색별 성적**으로 한다.
같은 이유로 "차례 색 × target 부호 4그룹 균형"은 하지 않는다. 한 게임 안에서 두 색의 표본 수는 거의 같아서
4그룹 균형은 지금의 2그룹(승자 색) 균형과 사실상 같다.

**판정: Gate 3 통과. 균형 샘플링을 기본 학습 조건으로 채택한다.**

### 12.7 다음 단계

1. **B 계열 후보 선정(라운드로빈 2차).** 참가: B160, B240, B320, B400, B480, 기준점 d16_200. 쌍당 100판, 15쌍이다.
   - B480이 최신이라고 최고로 보지 않는다. v321이 gen 80에서 17/20, gen 480에서 11/20이었다(20판이라 잡음이 크다).
2. **평가 사다리 보강.** MCTS-v2가 48~50/50으로 포화됐다.
   - light 평가: v321(25쌍)을 추가한다.
   - heavy 평가: v321·v5·v6·v7(각 10쌍)로 바꾼다. v321(약 70%)과 v7(0%) 사이를 채운다.
   - B*(1번에서 뽑힌 checkpoint)로 평가 전용 25/50/100 sims도 다시 잰다. Gate 2의 "탐색량 효과 없음"은 편향된 모델에서 잰 결과였다.
3. **B* 장기 학습(약 8시간).** 균형 샘플링을 유지한 채 이어간다. 정체되는 시점과, 그때 병목이 network 크기인지를 판단한다.
4. **Stage 8 마무리 → Stage 9.** 3번에서 64×4가 정체되면 Stage 9에서 network 확대(96×6 등, 새 초기화)를 한다.
   이때 GPU 경로(학습 10.7배, self-play는 8-D/8-E 필요)를 재개한다.

### 12.8 라운드로빈 2차와 B400 장기 학습

**2차 결과(쌍당 100판, 총 1,500판):**

| | B400 | B480 | B320 | B160 | B240 | d16_200 |
|---|---|---|---|---|---|---|
| 점수 / 500 | **345** | 301 | 279 | 245 | 233 | 97 |

- B400은 B160에 70-30, B240에 76-24, B320에 67-33, d16_200에 79-21로 이겼다(모두 p < 0.001).
- **B400 대 B480은 53-47(p = 0.62)로 차이가 없다.** B 계열끼리만 봐도 B400이 66.5%로 1위다.
- **B* = B400.** B160 → B400(240세대)에서는 실력이 올랐고, 400 → 480에서는 오른 증거가 없다. 첫 정체 신호다.

**분기 도구(`scripts/branch_stage8_run.py`):**

- 문제: run을 복사한 뒤 과거 checkpoint에서 이어가면, resume 정리가 `metrics.jsonl`·`self_play`·`evaluation`만 자른다.
  gen > N의 `external_eval`·`probes`는 남고, 오케스트레이터가 있는 파일을 건너뛰기 때문에 **새 궤적에 옛 결과가 붙는다.**
- 해결: 분기 도구는 gen ≤ N의 결과와 gen N checkpoint(`latest.pt`)만 새 디렉터리로 복사하고 `BRANCH.json`을 남긴다.
  원본은 수정하지 않는다(`tests/test_stage8_branch.py`).

**정체 판정(`--h2h-anchor`):** heavy 지점(80세대)마다 기준 checkpoint(B400)와 100판을 둔다(`genNNN_h2h.json`).

- 판정: 세 번 연속(240세대) 기준 대비 유의한 향상이 없으면 64×4의 정체로 본다. 점수 ≤ 55% 또는 p ≥ 0.05이면 향상 없음이다.
- 그때 Stage 8을 닫고 Stage 9(network 확대)로 간다.

**평가 사다리:**

- light(20세대): Tactical·MCTS-v2·**v321** 각 25쌍
- heavy(80세대): v321·**v5·v6**·v7 각 10쌍 + B400 기준 대국

**GPU:** 지금 설정으로 `device: cuda`만 바꾸면 self-play도 B=1 GPU가 된다. 이득이 증명되지 않았으므로 장기 학습은 CPU로 한다.

- 세대 시간 중앙값: self-play 17.7 s, 학습 6.3 s. 학습만 GPU로 옮겨도 전체 이득은 약 1.3배다.
- 처리량을 올릴 때는 구현이 단순하고 효과가 큰 **CPU 병렬 self-play**를 먼저 한다. P코어 6개면 self-play가 약 4~5배가 되어
  세대 시간이 약 2.4배 줄 것으로 본다. 학습 GPU는 그다음이다.
- GPU 묶음 self-play(8-D/8-E)는 network를 키워 NN 비중이 커지는 Stage 9에서 한다.

### 12.9 v7 인간 대국과 teacher arm

사람 대 v7 6판(사람 4승)을 SAFE/UNSAFE/UNKNOWN 3상태 VCF 분석기로 다시 확인했다(unknown 0).
강제 방어(stage 2)는 이미 패배가 증명된 뒤에 나왔다. 마지막으로 SAFE 수가 남아 있던 착수는 stage 4(탐색 0회) 3판,
MCTS 100회 1판이었다. 원인은 삼을 섞은 forcing 공격(VCT)이라는 가설이 가장 유력하지만, 아직 증명하지 않았다.

v7은 동결 상대로 두고 NN을 연결하지 않는다. B400 정체가 학습 신호 문제인지 용량 문제인지 가르기 위해 두 arm을
같은 판수로 비교한다.

- 대조군: B400 continuation(B480·B560·B640 anchor 판정)
- teacher arm: **증명된 전술 국면**(stage 1, 승리점 1개인 stage 2, stage 3, own VCF. 패배 증명 국면은 value만)으로
  B400을 fine-tune한 뒤 self-play

VCT probe, 웹 대국 NN 에이전트, Stage 9 진입 기준은 [v7-nn-integration-review.md](v7-nn-integration-review.md)에 있다.
도구(`analysis.threats`, `build_vct_probes.py`, `build_tactical_dataset.py`, `make_teacher_branch.py`,
`compare_teacher_arms.py`, `run_web_play.py --az-checkpoint`)는 구현했고, 데스크톱 실행 절차는 그 문서 §6.5에 있다.
light 지점 probe에 VCT set(`probes/genNNN_vct.json`)이 추가된다.

### 12.10 B400 장기 학습 결과 (gen 400 → 880, 7,680판)

`runs/stage8_b400_long`: stage8_g3_b gen 400에서 분기, 설정 불변, heavy 지점마다 B400과 100판.

| gen | B400 대비 | v321 heavy | v5 / v6 / v7 (각 20판) | raw must_block top-1 | value 균형 정답률 | open3 top-3 |
|---|---|---|---|---|---|---|
| 400 | 0.50 | 15/20 | — / — / 0 | 0.23 | 0.51 | 0.50 |
| 480 | 0.47 (p 0.62) | 11/20 | 1 / 0 / 0 | 0.28 | 0.52 | 0.70 |
| 560 | 0.48 (p 0.76) | 16/20 | 0 / 0 / 0 | 0.23 | 0.55 | 0.60 |
| 640 | **0.34 (p 0.002)** | 14/20 | 0 / 1 / 0 | 0.10 | 0.56 | 0.63 |
| 720 | 0.50 (p 1.0) | 13/20 | 1 / 1 / 0 | 0.18 | 0.60 | 0.68 |
| 800 | 0.52 (p 0.76) | 13/20 | 1 / 0 / 1 | 0.18 | 0.57 | 0.60 |
| 880 | 0.53 (p 0.62) | 18/20 | 0 / 0 / 1 | 0.15 | 0.58 | 0.60 |

- **정체 확정.** heavy 6개 지점 모두 B400 대비 향상이 없다(기준은 3회 연속). 640은 유의하게 **약했고**, 이후 원래 수준으로 돌아왔다.
- policy loss는 2.32 → 2.12로 내려갔지만 value loss는 0.33 → 0.45로 올랐다. self-play 분포에 맞춰질 뿐 기력은 오르지 않는다.
- **탐색 예산은 해법이 아니다.** B400을 25/50/100 simulations로 평가해도 v321 0.75/0.75/0.80, v5·v6·v7은 모두 0~1/20이다(`stage8_b400_sims`).
- **self-play가 퇴화했다.** B400 이후 self-play 7,680판 중 **40.8%가 9수, 43.6%가 10수**에 끝났다. 9수는 흑이, 10수는 백이 방어 없이
  5목을 만드는 가장 빠른 길이다. `temperature_moves: 10`, `tau: 1`이라 이 게임들은 **처음부터 끝까지** 방문 비율 샘플링으로 두어졌다.
  열린 3을 막는 수의 방문 비율이 절반이어도 절반의 확률로 막지 않는다. 그래서 학습 데이터는 대부분 "양쪽이 방어하지 않는 경주"이고,
  샘플 재사용이 약 10배다. gen 0~400(Gate 3)도 평균 약 10수였다.
- raw 전술 probe가 낮다(must_block top-1 0.10~0.28, value 균형 정답률 0.51~0.60). 같은 64×4를 증명 label 1,443개로만 학습시키면
  held-out must_block 0.72, value 0.91이 나온다(검토 문서 §6.3). 따라서 이 정체는 **용량 한계라는 증거가 아니다.**
  `compare_teacher_arms.py`의 판정도 `signal`(학습 신호 문제)이다.

**결정:** Stage 9(network 확대)로 바로 가지 않는다. 같은 gen 400 상태에서 다음 arm을 짝지어 비교한다
(대조군은 이미 있는 `stage8_b400_long`):

1. **S4(레시피):** `temperature_moves` 10 → 4. 다른 것은 동일(`configs/stage8_b400_temp4.yaml`, `scripts/make_recipe_branch.py`).
2. **T1(teacher):** 증명 label fine-tune(검토 문서 §6.4).
3. (선택) **S4+T1.**

실행 명령은 [v7-nn-integration-review.md §10](v7-nn-integration-review.md)에 있다.

### 12.11 S4 / T1 결과 (gen 400 → 640, arm당 3,840판)

세 arm 모두 stage8_g3_b gen 400 상태(가중치·replay·optimizer·RNG)에서 분기했다. 대조군은 `stage8_b400_long`이다.

**self-play 게임 길이**(구간별, arm당 1,280판):

| 구간 | 대조군 ≤10수 / 평균 | S4 ≤10수 / 평균 | T1 ≤10수 / 평균 |
|---|---|---|---|
| 400~479 | 90.4% / 10.2 | 72.5% / 12.1 | 91.2% / 10.1 |
| 480~559 | 86.3% / 10.1 | 71.2% / 13.2 | 85.4% / 10.1 |
| 560~639 | 86.9% / 10.0 | **54.3% / 17.5** | 85.0% / 10.3 |

**직접 대국(각 100판, arm 점수) / B400 anchor 대국:**

| gen | S4 대 대조군 | T1 대 대조군 | S4 대 B400 | T1 대 B400 | 대조군 대 B400 |
|---|---|---|---|---|---|
| 400 | — | — | 0.50 | **0.66** (fine-tune 직후) | 0.50 |
| 480 | **0.79** | 0.64 | 0.69 | 0.51 | 0.47 |
| 560 | **0.70** | 0.61 | 0.68 | 0.54 | 0.48 |
| 640 | **0.93** (p≈3e-20) | 0.69 | **0.84** (p≈3e-12) | 0.47 | 0.34 |

**외부 평가(heavy 각 20판) / raw probe:**

| gen 640 | v321 | v5 | v6 | v7 | must_block top-1 | vcf top-1 | open3 top-3 | value 균형 정답률 |
|---|---|---|---|---|---|---|---|---|
| 대조군 | 14 | 0 | 1 | 0 | 0.10 | 0.17 | 0.62 | 0.56 |
| **S4** | **19** | **5** | **6** | **6** | **0.72** | **0.47** | **0.88** | 0.49 |
| T1 | 15 | 0 | 1 | 0 | 0.12 | 0.17 | 0.53 | 0.52 |

- **S4는 성공으로 판정한다.** 480부터 대조군과 B400을 계속 유의하게 이겼고, 640에서 다시 크게 올랐다(560까지 단조 증가는 아니었다).
  opening pair(50개) 단위로 다시 계산해도 S640 대 B400 p≈2e-8, 대 control640 p≈2e-13이다(외부 검토 계산).
- **"온도가 퇴화의 원인"이라는 가설을 강하게 지지한다.** 증명은 아니다(branch 실험 1회). 게임이 길어졌고(≤10수: 대조군 400~479 구간
  90.4%, `stage8_b400_long` metrics → S4 560~639 구간 54.3%, 평균 17.5수), 샘플 재사용도 줄었다(9.8 → 6.15).
- **raw 전술:** S4 자체의 must_block top-1은 0.225(gen 400) → 0.300 → 0.275 → **0.725(gen 640)**다. 같은 stage7 probe에서 control640은 0.10,
  fine-tune 직후 T1(gen 400)은 0.475였다. 즉 S4는 같은 기준에서 T1 fine-tune 직후보다도 높다. (이전에 쓴 "지도학습 수준과 같다"는
  서로 다른 평가셋을 비교한 표현이라 철회한다. `TEACHER.json`의 0.713은 teacher dataset 자체 평가이고, 검토 문서 §6.3의 0.72는
  stage7 probe의 held-out 부분집합이다.)
- **v7 6/20은 첫 의미 있는 승리 신호다**(95% 구간 약 12~54%, 이전 0/20 대비 한쪽 Fisher p≈0.01). v7과 경쟁 가능한 수준이라고
  단정할 단계는 아니다. 흑 1/10, 백 5/10으로 흑번이 특히 약하다.
- **loss:** S4의 value loss가 0.42(480~559) → 0.55(560~639)로 올랐다. self-play 분포가 길고 다양해진 것과 함께 나타났고, 외부 대국
  성능이 동시에 좋아졌으므로 현재로서는 퇴화 신호로 보지 않는다(원인은 해석이다). value 균형 정답률은 0.49로 아직 낮다.
- **T1: 한 번의 fine-tune은 지속되지 않는다(확실).** B400 대비 0.66 → 0.51 → 0.54 → 0.47이고, raw must_block은 gen 400 0.475에서
  **gen 420에 이미 0.175**, 이후 0.150 → 0.125로 떨어졌다. 원인 후보는 여럿이고 아직 가르지 못했다.
  (a) 퇴화한 self-play 분포, (b) teacher 행을 계속 섞지 않음, (c) fine-tune 뒤 **이전 가중치의 Adam 상태**로 학습을 재개함,
  (d) lr이 fine-tune 2e-4에서 run의 1e-3로 5배 뛴 것. `make_teacher_branch.py --optimizer-state reset`을 추가해 (c)를 없앨 수 있게 했다.
- T1 dataset은 원시 개수가 쉬운 kind에 치우쳤다(immediate_win 3,936 / unstoppable_four 3,987 / forced_loss 3,848 / must_block 1,249 /
  vcf 323). 하지만 fine-tune은 `--balance-kinds`로 kind당 20%씩 뽑았으므로 개수 불균형 자체가 원인은 아니다. 문제가 있다면
  퇴화한 경주 게임에서 나온 **국면 다양성**이다.
- **판정 역할 분리:** 고정 anchor는 각 arm의 절대 진행을, 같은 세대 직접 대국은 두 레시피의 비교를 본다. T1은 control640에 69%로
  이겼지만 B400에는 47%였다. T1이 강해진 것이 아니라 control이 더 약해진 것이다. `compare_teacher_arms.py`에 이 경우를
  `relative_only`로 넣었다(직접 대국은 이기지만 마지막 3개 anchor 지점에서 향상 없음). 실제 데이터로 판정하면 S4는 `teacher_better`,
  T1은 `relative_only`다.
- **판정:** 지금까지 확인된 주요 병목은 네트워크 크기보다 self-play 레시피였다. S4가 아직 오르는 중이므로 Stage 9는 서두르지 않는다.

**다음(§12.12):**

1. **S4를 기준 레시피로 채택**하고 gen 880까지 이어서 학습한다. 이미 B400을 크게 넘었으므로, 향상 여부는 **S4 gen 640을 새 anchor로** 판정한다.
2. **S2 arm:** S4 gen 640에서 `temperature_moves` 4 → 2로 분기한다(`configs/stage8_s640_temp2.yaml`). 같은 세대 S4 continuation과
   직접 대국한다. 아직 46%가 10수 이하로 끝나므로 온도를 더 낮출 여지가 있는지 본다.
3. **teacher는 한 번의 fine-tune으로는 쓰지 않는다.** 다시 한다면 S4 self-play 기보(더 다양한 국면)에서 dataset을 새로 만들고,
   매 세대 섞는 지속 혼합(T4)으로 한다. 이때 optimizer 상태·lr 교란을 없앤다(`--optimizer-state reset` 또는 지속 혼합 자체로 해소).
   S4 레시피가 정체한 뒤의 후보다.

### 12.12 S2(temperature 2) 대 S4 continuation (gen 640 → 880, arm당 3,840판)

두 run 모두 S4 gen 640 상태에서 분기했다(S2는 `temperature_moves` 4 → 2만 변경, `RECIPE.json` 확인). anchor는 S640(S4 gen 640)이다.

| gen | S4c 대 S640 | S2 대 S640 | S2 대 S4c(직접, S2 점수) |
|---|---|---|---|
| 720 | 0.42 (p 0.13) | 0.55 (p 0.37) | 0.54 (p 0.48) |
| 800 | 0.56 (p 0.27) | **0.64 (p 0.007)** | 0.50 (p 1.0) |
| 880 | **0.68 (p 0.0004)** | **0.66 (p 0.002)** | 0.55 (p 0.37) |

| self-play 구간 | S4c ≤10수 / 평균 / 재사용 | S2 ≤10수 / 평균 / 재사용 |
|---|---|---|
| 640~719 | 72.2% / 13.1 / 7.98 | 44.1% / 18.4 / 6.36 |
| 720~799 | 62.2% / 17.6 / 6.36 | 24.8% / 26.6 / 4.33 |
| 800~879 | 28.1% / 28.7 / 3.61 | 23.4% / 20.6 / 5.41 |

| heavy 720~880 합계(상대당 60판) | v321 | v5 | v6 | v7 | 합(v5~v7) |
|---|---|---|---|---|---|
| S4c | 57 | 11 | 16 | 17 | 44/180 |
| S2 | 59 | 22 | 14 | 9 | 45/180 |

raw probe(gen 880): S4c must_block 0.70 · vcf 0.50 · open3 top-3 0.95 · VCT 방어 0.96 · VCT 공격 0.88 · value 0.64 /
S2 must_block 0.80 · vcf 0.57 · open3 0.97 · VCT 방어 0.83 · VCT 공격 0.72 · value 0.62.

- **두 레시피 모두 S640보다 강해졌다**(880에서 0.68 / 0.66). S640을 기준으로 아직 정체가 아니다. 240세대에 약 +120 Elo다.
- **temperature 2와 4의 차이는 검출되지 않았다.** 직접 대국 합계는 S2 159/300(0.53, p≈0.3)이고, heavy 합계도 44 대 45로 같다.
  `compare_teacher_arms.py` 판정은 `undecided`다.
- 차이가 보인 것은 **과정**이다. S2는 짧은 게임이 바로 줄었다(44% → 25%). S4c는 720에서 짧은 게임이 다시 늘었고(72%), 그때 anchor
  점수도 0.42로 내려갔다. 이후 길어지면서 회복했다. 자기대국 게임 길이와 기력이 함께 움직인다.
- heavy 평가는 상대당 20판이라 세대마다 크게 흔들린다(S4c의 v7: 2 → 11 → 4/20). v5~v7 판정에는 표본이 부족하다.
- raw 전술 probe는 두 arm 모두 must_block ≥ 0.6을 넘었다. 이제 이 레시피가 정체하면 `capacity` 판정(Stage 9 후보)이 가능하다.
- (참고) S2 branch의 `metrics_events_kept: 0`은 원본 S4 run의 `metrics.jsonl`에 gen 640 미만 이벤트가 없었다는 뜻이다. S2 run의
  학습에는 영향이 없고, S2의 metrics는 640부터 시작한다.

**결정과 다음(§12.13):**

1. **본선은 S2(temperature 2)로 둔다.** 근거는 동률에서의 선택 기준이다. anchor 진행이 단조로웠고 짧은 게임이 빨리 줄었다.
   S2가 더 강하다는 증거는 없다. S4c는 여기서 멈춰 코어를 아낀다.
2. **새 anchor S880 = S2 gen 880.**
3. **다음 단일 변수 실험: self-play simulations 50 → 100**(`configs/stage8_s880_sims100.yaml`, S2 gen 880에서 분기).
   예전에 "탐색 예산 효과 없음"(Gate 2, B400 25/50/100)은 퇴화한 self-play에서 잰 것이라 다시 잰다. self-play 시간이 약 2배라 가장 느린 arm이다.
4. **heavy 평가 표본을 늘린다:** `--heavy-pairs 25`(상대당 50판). v5~v7 차이를 보려면 20판으로는 부족하다.
5. 본선(S2 continuation)이 S880을 3지점 연속 못 넘고 raw 전술이 기준 이상이면 판정은 `capacity`다. 그때 Stage 9(network 확대)로 간다.

### 12.13 본선 S2(880 → 1120)와 sims100 결과

두 run 모두 S2 gen 880 상태에서 분기했다(sims100은 `self_play.simulations` 50 → 100만 변경). anchor는 S880, heavy는 상대당 50판이다.

| gen | 본선 대 S880 | sims100 대 S880 | 본선 대 sims100(직접, 본선 점수) |
|---|---|---|---|
| 960 | **0.67 (p 0.0009)** | 0.50 (p 1.0) | 0.59 (p 0.09) |
| 1040 | **0.65 (p 0.004)** | 0.51 (p 0.92) | 0.53 (p 0.62) |
| 1120 | 0.56 (p 0.27) | 0.49 (p 0.92) | 0.59 (p 0.09) |

| self-play 구간 | 본선 ≤10수 / 평균 / 흑승 | sims100 ≤10수 / 평균 / 흑승 / 세대당 self-play |
|---|---|---|
| 880~959 | 19.8% / 16.9 / 43% | 19.7% / 20.1 / 39% / 175 s |
| 960~1039 | 37.3% / 14.9 / 61% | 43.8% / 13.3 / 53% / 114 s |
| 1040~1119 | 22.5% / 17.0 / 66% | **63.9% / 13.0** / 50% / 48 s |

| heavy 960~1120 합계(상대당 150판) | v321 | v5 | v6 | v7 | v5~v7 |
|---|---|---|---|---|---|
| 본선 | 145 | 52 | 72 | 43 | **167/450 (37%)** |
| sims100 | 143 | 50 | 49 | 36 | 135/450 (30%) |

- **sims100은 채택하지 않는다.** S880보다 나아지지 않았고(판정 `undecided`, 자기 anchor 기준 plateau), 직접 대국 합계는 본선 171/300
  (0.57, p≈0.018)이다. self-play가 다시 짧아졌다(≤10수 20% → 64%). 퇴화가 기력 정체와 함께 나타나는 패턴이 세 번째로 관찰됐다.
  왜 탐색을 늘리면 짧아지는지는 확인하지 않았다(가설: 더 날카로운 visit target이 공격 위주 분포를 강화). simulations는 50을 유지한다.
- **본선은 960·1040에서 S880을 유의하게 이겼지만, 1120은 유의하지 않다(0.56).** S880 대비 점수가 0.67 → 0.65 → 0.56으로 내려갔으므로
  960 이후 정체나 소폭 후퇴일 수 있다. 정체 판정 기준(3지점 연속 향상 없음)은 아직 충족되지 않았다(`s2_main_status`: `undecided`).
- 본선 heavy: v6 27/50(54%, 1040·1120)으로 v6과 비슷해졌다. v7은 11~18/50(22~36%), v5는 13~20/50이다.
- raw probe는 거의 포화다(gen 1120: must_block 0.75, open3 top-3 1.00, VCT 방어 0.71, VCT 공격 0.75, VCT value 0.88).
- 본선 960~1119에서 self-play 흑승이 61~66%로 올랐다(이전 43~52%). 균형 샘플링이 켜져 있지만 지켜볼 신호다.

**다음(§12.14):**

1. **본선 내부 라운드로빈**(S880, S960, S1040, S1120, 쌍당 100판): 960 이후에도 오르는지 직접 가린다. S880 하나만 기준으로 하면
   점수 하락이 "S880이 상대하기 어려운 스타일"인지 "실제 후퇴"인지 구분하기 어렵다.
2. **본선 continuation 1120 → 1360**, anchor는 라운드로빈 1위 checkpoint. heavy 3지점(1200/1280/1360)에서 향상이 없고 raw probe가
   포화 상태면 판정은 `capacity`다. 이 경우 Stage 9(network 확대)를 준비한다.
3. 레시피 단일 변수 실험은 쉬어 간다. temperature 4/2는 동률이었고, simulations 100은 손해였다.

### 12.14 외부 검토 반영: "정체 판정"에서 "안정적 지속 향상" 루프로

외부 검토의 수치를 업로드 데이터로 다시 계산했다. 모두 일치했다.

| 항목 | 재계산 |
|---|---|
| 본선 대 sims100 직접 대국, opening pair 단위 | 본선 우세(2승) 45쌍, 열세(2패) 24쌍, 1승 1패 81쌍, sign test p = 0.015 |
| sims100 정확히 9수 / 10수 (구간 끝 960 / 1040 / 1120) | 13.9 / 43.2 / 32.6% · 5.8 / 0.5 / **31.3%** |
| 본선 정확히 9수 (같은 구간) | 10.0 / **36.4** / 22.3% |
| 세대당 새 국면 / 재사용 | sims100 322 → 212 → 209 / 5.36 → 8.28 → 8.24 · 본선 270 → 239 → 272 / 6.41 → 7.20 → 6.55 |
| 10세대 이동 흑 승률(최소 ~ 최대) | 본선 0.03 ~ 0.99, sims100 0.01 ~ 1.00 |
| 본선 v5~v7 합계 | S960 51/150, S1040 58/150, S1120 58/150 (v5 26→40%, v7 36→22%) |
| S1120 raw probe | must_block 0.75 · immediate_win 0.45 · vcf 0.50 · VCT 공격 0.41 · **VCT 방어 0.25** · open3 0.80 · **forced_loss value 0.53** |

**§12.13에서 고치는 점:**

1. "S1120 후퇴"는 근거가 없다. S880 대비 점수는 내려갔지만 v5~v7 합계는 S1040과 같고 상대별 구성만 바뀌었다
   (전략 이동, 비추이적 변화와도 맞는다).
2. "raw probe 거의 포화"는 틀렸다. 쉬운 지표(must_block, open3 top-3)만 높고 VCT 방어 top-1 0.25, forced_loss value 0.53이다.
   `compare_teacher_arms.py`의 포화 판정을 **다섯 지표 모두**(must_block, vcf, forced_loss value, VCT 방어, VCT 공격)로 바꿨다.
3. "1360에서도 정체면 capacity"는 이르다. 데이터·탐색·학습 안정성 문제를 먼저 배제해야 한다.
4. 진단 오류: §12.13의 명령은 anchor만 바꾸고 학습은 최신 S1120에서 계속했다. champion이 S1120이 아니면 **champion
   checkpoint에서 branch**해야 한다.
5. "짧은 self-play가 원인"은 강한 징후일 뿐 인과는 미확정이다. 짧은 게임 → 세대당 새 국면 감소 → **고정 50 step**이라 재사용 증가
   → 정책 편향 강화 → 더 짧은 게임이라는 **양의 피드백 루프**를 가장 유력한 가설로 둔다. 본선도 960~1040에 한 번 붕괴했다가
   회복했다(진동형). 흑 승률이 0.03~0.99로 오가는 것도 같은 진동으로 본다.

**현재 상태 해석:** S2는 용량 한계에 닿은 모델이 아니다. S880을 넘은 뒤 self-play 분포가 진동하는 모델이고, S960~S1120에는 실제
향상과 전략 이동이 섞여 있다. sims100은 그 불안정성을 키웠다.

**새 도구:**

| 도구 | 역할 |
|---|---|
| `training.adaptive_steps`(설정) | 세대당 SGD step = round(target_reuse × 새 국면 / batch), [min, max]로 제한. 기본 null(기존 설정 해시 불변) |
| `configs/stage8_s2_adaptive.yaml` | S2 + `adaptive_steps {target_reuse 6.4, min 20, max 100}` (6.4 = 본선 880~959 재사용) |
| `analyze_self_play_health.py` | 구간별 게임 길이(≤10, 9, 10수), 흑 승률과 10세대 이동 범위, 새 국면·재사용·step, **opening 다양성**(2~4/6/8수 prefix 수와 엔트로피, 최다 prefix 비율), 게이트 |
| `forensic_short_games.py` | 9~10수 게임의 opening 군집, 패자의 마지막 VCF-safe 결정, 그 국면의 prior·value·탐색 50/400·noise 반복으로 `noise` / `search_budget` / `prior_blind` / `value_blind` 분류 |
| `segment_gate.py` | 40세대 구간마다 건강 게이트 + champion 대국 → PROMOTE / HOLD / STOP(종료 코드 0/1/2) |
| `run_stage8_head_to_head.py` | opening pair 단위 통계(pair 승/패/분할, sign-test p) 추가. 판정 도구는 pair p를 우선한다 |

**게이트 기준(초기값, 데이터가 쌓이면 조정):** ≤10수 비율 30% 이상이면 WARN, 2구간 연속 40% 이상이면 STOP. 재사용이 기준(6.4)의
1.2배 이상이면 WARN, 2구간 연속 1.3배 이상이면 STOP. 6수 prefix 엔트로피가 기준 구간의 0.8배 미만이면 WARN. 흑 승률은
절대 50%가 아니라 기준 구간 대비 편차를 본다(현재는 보고만 한다).

**새 루프:** RR → champion 확정(pair 통계 + 상위 2개 재확인) → 9~10수 원인 분석 → (adaptive 대 고정) A/B를 40세대 구간 +
게이트로 → 이긴 레시피로 champion 승격 루프 → 반복.

**Stage 9 진입 조건(수정):** 다음이 **모두** 성립할 때만 network 확대를 원인 후보로 올린다.
(1) 건강 게이트가 3구간 이상 연속 OK(재사용 정상, 짧은 게임 30% 미만),
(2) champion이 3구간(120세대) 연속 승격되지 않음,
(3) 다섯 probe 게이트가 모두 충족되거나, forensic에서 깊은 탐색(400)으로도 못 고치는 `prior_blind` / `value_blind`가 대부분임,
(4) adaptive 대 고정 A/B에서 차이가 없음.

**뒤로 미룬 것(근거가 생기면):** 어려운 국면만 높은 탐색으로 다시 target을 만드는 reanalyse, root prior temperature(KataGo식),
opening prefix pool 혼합. forensic에서 `noise`가 많으면 root noise/temperature를, `search_budget`가 많으면 reanalyse를,
opening 군집 집중과 엔트로피 하락이 보이면 root prior temperature나 opening pool을 다음 단일 변수로 한다.

### 12.15 B단계 결과: adaptive steps(ADA) 대 고정 50 steps(CTL), C1120에서 분기 (2026-10-02)

두 arm 모두 S2 gen1120(C1120)에서 분기했다. 달라진 것은 학습 steps 규칙뿐이다. 1320세대까지 누적 SGD step은 ADA 67,049, CTL 66,000으로 1.6%만 다르다.
따라서 차이는 학습량이 아니라 **배분**에서 나왔다.

| 세대 | ADA 대 C1120 (pair, p_pairs) | CTL 대 C1120 (pair, p_pairs) | ADA v5+v6+v7 /150 | CTL v5+v6+v7 /150 |
|---:|---|---|---:|---:|
| 1120 | — | — | 58 (38.7%) | 58 (38.7%) |
| 1160 | 52% (9-7, 0.80) HOLD | 56% (15-9, 0.31) HOLD | 48 | 54 |
| 1200 | 47% (9-12, 0.66) HOLD | **36%** (6-20, 0.009) HOLD | 41 | 39 |
| 1240 | **67%** (25-8, 0.005) **PROMOTE** | 42% (6-14, 0.12) HOLD | 54 | 39 |
| 1280 | 56% (15-9, 0.31) HOLD | **35%** (3-18, 0.001) HOLD | 61 | 54 |
| 1320 | 61% (22-11, 0.080) HOLD | **28%** (2-24, 1e-5) **STOP** | **77 (51.3%)** | 57 (38.0%) |
| 1360 | **64%** (20-6, 0.009) **PROMOTE** | — | 63 (42.0%) | — |

Self-play 창(40세대):

| arm | 창 | 평균 길이 | <=10수 | 흑 승률 | fresh/세대 | reuse | steps/세대 |
|---|---|---:|---:|---:|---:|---:|---:|
| ADA | 1160-1199 | 16.5 | 42% | 62% | 265 | 6.40 | 53.0 |
| ADA | 1240-1279 | 15.9 | 43% | 72% | 255 | 6.38 | 50.5 |
| ADA | 1280-1319 | 19.1 | 27% | 21% | 306 | 6.33 | 59.9 |
| ADA | 1320-1359 | 20.9 | 28% | 71% | 334 | 6.36 | 66.1 |
| CTL | 1160-1199 | 13.4 | 46% | 68% | 215 | 7.84 | 50 |
| CTL | 1200-1239 | 17.6 | 10% | 14% | 282 | 5.95 | 50 |
| CTL | 1240-1279 | 12.0 | 60% | 92% | 192 | 8.90 | 50 |
| CTL | 1280-1319 | 11.8 | 69% | 9% | 189 | 8.82 | 50 |

**판정:**

- **ADA 채택.** ADA는 C1120을 두 번 유의하게 넘었다(1240 p=0.005, 1360 p=0.009). 6번 본 것을 보정한 Bonferroni 0.05/6=0.0083에서도 1240은 통과하고 1360은 경계다.
  CTL은 1200 이후 네 번 중 세 번 C1120보다 유의하게 약했고, 1320에 게이트 STOP이 났다. 계산량은 같으므로 단순성 비용 외의 손해가 없다.
- **인과는 "reuse가 짧은 게임을 만든다"가 아니다.** 두 arm 모두 짧은 게임 폭발(40% 이상)이 먼저 왔고, 그 직전 창의 reuse는 정상(6.4)이었다.
  짧은 게임 폭발은 **색 진동**(한 색의 정책이 상대 색의 구멍을 찾아 연속 승리, CTL 흑 92% → 9%, ADA 72% → 21% → 71%)과 같이 온다.
  adaptive는 폭발을 막지 못했다. 대신 폭발 구간에서 학습을 줄여 그 데이터를 과하게 외우지 않게 했고, 매번 다음 창에서 회복했다(42% → 23%, 43% → 27%).
  CTL도 한 번은 회복했다(46% → 10%). 그러나 두 번째 폭발(60%, 69%)에서는 회복하지 못했다. 근거는 seed당 1회 실행이다.
- **외부 기준(v5/v6/v7)은 약한 근거다.** 1120 기준 38.7%와 같은 seed로 비교하면 ADA 1320만 높다(51.3%, z=2.2, p≈0.03, 보정 전).
  ADA 1360(42.0%, p≈0.56)과 CTL 1320(38.0%)은 기준과 구분되지 않는다. 즉 CTL의 붕괴는 **C1120 상대 직접 대결에서만** 보였다.
  ADA의 범용 기력 향상은 1320 한 점에만 기대고 있다.
- **probe는 판정 근거로 쓰지 않는다.** 기본 probe는 종류당 40국면이다. 같은 ADA run 안에서도 checkpoint마다 VCF top1이 20/27/23/17로 움직인다.
  VCT probe는 기본 국면 3~4개 × 대칭 8이라 실질 표본이 3~4개다. 예: vct_attack top3 31/32는 사실상 4국면 결과다.
- **후보 순위는 아직 모른다.** 1240(67%)·1320(61%)·1360(64%)의 C1120 상대 점수는 오차(±9%p) 안에서 같다.
  게다가 이 셋은 seed 8008 개국으로 고른 것이라 그 개국에서는 낙관적으로 편향돼 있다. 새 seed로 다시 잰다(`docs/v7-nn-integration-review.md` §15).

**다음 단일 변수 후보:** 짧은 게임 폭발의 원인인 색 진동. 먼저 ADA의 폭발 창(1240-1279)을 `forensic_short_games.py`로 분류한다.
`noise`가 많으면 root noise/temperature를, `prior_blind`가 많으면 표본 보강(VCT 방어 등)을 다음 단일 변수로 한다.

### 12.16 C0 결과: champion 선정(새 seed), arm 직접 대결, forensic (2026-10-02)

**라운드로빈(seed 9009, 100쌍=200판, 행 기준 점수):**

| | 대 C1120 (pair, p) | 대 ADA1240 | 대 ADA1320 | 대 ADA1360 | 총점 /600 |
|---|---|---|---|---|---:|
| C1120 | — | 83 | 94 | 79 | 256 |
| ADA1240 | **117** (36-19, 0.030) | — | 73 (10-37, 0.0001) | 97 (27-30, 0.79) | 287 |
| ADA1320 | 106 (26-20, 0.46) | 127 | — | 83 (18-35, 0.027) | 316 |
| ADA1360 | **121** (36-15, 0.0046) | 103 | 117 | — | 341 |

seed 8008에서 67/61/64%였던 C1120 상대 점수가 58.5/53.0/60.5%로 내려왔다. 개국 선택 편향이 실제로 있었다. 1320은 자격을 잃었다.
비추이성도 크다: 1320은 1240을 127:73으로 이기지만 C1120과는 비슷하고, 1360에는 진다.

**heavy(seed 7107, 상대당 50쌍=100판, v5+v6+v7):** C1120 107/300(35.7%), ADA1240 133(44.3%), ADA1320 142(47.3%), ADA1360 129(43.0%).
같은 300개 대국 단위 대응 비교(이 checkpoint만 이긴 판 대 C1120만 이긴 판): 1240 75:49(p=0.024), 1320 90:55(p=0.005), 1360 83:61(p=0.080).
1240 대 1360은 69:65(p=0.80)로 구분되지 않는다. C1120도 seed 7007 38.7%에서 7107 35.7%로 움직였다. heavy 150~300판도 seed에 따라 ±3%p 흔들린다.

**사전 규칙 적용:**

- 자격: ADA1240, ADA1360. 자격자 1·2위는 1360(341)과 1240(287)이다.
- 둘의 직접 대결은 103:97(p=0.79)이라 동률이다. heavy 동률 깨기는 133 대 129로 1240이 4승 앞선다.
  "3승 이하면 늦은 세대" 조건을 넘으므로 **규칙을 글자 그대로 적용하면 champion은 ADA1240**이다.
- **규칙 결함 수정 → ADA1360.** 300판 대응 비교에서 4승 차이는 p=0.80이다. 차이의 표준오차가 약 ±12승이므로 "3승" 문턱은 통계적으로 의미가 없는 값이었다(규칙 설계 오류).
  동률 깨기 단계도 다른 단계처럼 유의성으로 판정하도록 고친다: 대응 비교 p < 0.05가 아니면 늦은 세대.
  그러면 champion은 **ADA1360**이다. 이 수정은 결과를 본 뒤의 변경이라는 점을 기록해 둔다.
  다만 세 측정(직접 대결, heavy, C1120 상대) 모두 1240 ≈ 1360이므로 어느 쪽을 골라도 근거상 손해가 없다.
  1360이면 분기 없이 이어 학습할 수 있다. 규칙 밖 근거로는 1360이 1320을 이기고(117:83) 1240은 1320에 진다(73:127)는 점이 있다.

**arm 직접 대결(seed 8008, 50쌍, CTL 점수):** 1200 45%(pair 10-15, p=0.42), 1280 42%(7-15, p=0.13), **1320 21%(2-31, p=1.3e-7; 흑 10/50, 백 11/50)**.
ADA 채택 판정(§12.15)을 강화한다. 근거는 여전히 arm당 run 1개다.

**forensic(짧은 게임 중 표본 200개, 깊은 탐색 400회):**

| 창 | 짧은 게임 | 서로 다른 6수 개국 / 상위 10 | 지는 쪽, 수 | prior_safe | 50회 결정적 탐색 safe | 400회 safe (방문 비중) | 분류 |
|---|---:|---|---|---:|---:|---:|---|
| S2 960-1040 | 477/1280 | 464 / 23 | 백 6수째 196 | 0.139 | 136 | 175 (0.17) | noise 84, base_safe 52, search_budget 41, prior_blind 22 |
| ADA 1240-1280 | 273/640 | 273 / 10 | 백 6수째 199 | 0.045 | 39 | 199 (0.20) | search_budget 160, noise 31 |
| CTL 1280-1320 | 439/640 | 216 / 68 | 흑 7수째 187 | 0.043 | 21 | 108 (0.04) | prior_blind 92, search_budget 87, noise 21 |

- 맞는 해석: CTL은 같은 개국 군집이 반복되고(상위 10개가 15%), 400회 탐색으로도 절반이 못 고쳐진다. 고착의 흔적이다. ADA는 실패 개국이 모두 다르고, 400회로 거의 다 고쳐진다.
- **주의 1: 색이 다르다.** ADA 실패는 흑 우세 창(흑 72%)에서 백이 3번째 수에 열린 3을 막지 못한 것이다. CTL 실패는 백 우세 창(흑 9%)에서 흑이 4번째 수에 진 것이다. 두 창은 같은 현상을 비교하는 게 아니다.
- **주의 2: "search_budget"은 탐색이 건강하다는 뜻이 아니다.** ADA에서도 prior는 막는 수(평균 2개)에 4.5%만 준다. S2 시절(0.139)보다 낮다. value는 -0.93으로 이미 졌다고 본다.
  400회에서도 막는 수의 방문 비중은 20%뿐이다. 막지 않는 수가 곧바로 지는 말단이라 겨우 argmax가 될 뿐이다.
  즉 "모든 수가 진다"고 보는 value 포화 때문에 탐색이 수를 가리지 못하는 상태에 가깝다. 탐색 예산만의 문제가 아니다.
- **주의 3: forensic은 실패한 게임만 본다.** 짧은 게임 비율이 다른 창끼리의 분류 비율은 실패율이 아니다.
- **self-play 탐색은 이미 50회다**(25회는 평가용). 50 → 100은 고정 steps에서 이미 시험했다(§12.13): 본선에 직접 대결 p=0.015로 졌고, 짧은 게임과 reuse를 키웠다.
  adaptive 아래에서 다시 시험할 가치는 있다. 다만 self-play 비용이 두 배이므로, 먼저 forensic을 100/200회로 돌려 몇 회에서 고쳐지는지 곡선을 본다.

**결정:** champion = ADA1360. C1은 `$X = 1360`으로 그대로 진행한다. C1 동안 레시피는 바꾸지 않는다.

### 12.17 C1 결과: ADA1360 상대 3구간 HOLD = 정체, 색 우세 상태가 반복해서 뒤바뀜 (2026-10-03, 외부 검토 반영)

| 구간 끝 | 대 ADA1360 (pair, p) | heavy v5+v6+v7 /150 (seed 7007) | self-play 창 | 평균 길이 | <=10수 | 흑 승 | 독식 세대* | 색 margin** |
|---:|---|---:|---|---:|---:|---:|---:|---:|
| 1360 | (champion) | 63 | 1320-1359 | 20.9 | 28% | 71% | 40% | 0.622 |
| 1400 | **38%** (5-17, 0.017) | 67 | 1360-1399 | **11.1** | **70%** | **99%** | **100%** | **0.984** |
| 1440 | 45% (8-13, 0.38) | 66 | 1400-1439 | 14.8 | 26% | 33% | 70% | 0.816 |
| 1480 | 47% (9-12, 0.66) | 72 | 1440-1479 | 20.6 | 20% | 57% | 52.5% | 0.713 |

\* 한 세대 16판 중 한 색이 15판 이상 이긴 세대의 비율(`one_sided_share`).
\*\* 세대별 |흑 승 - 백 승| / 16의 평균(`mean_abs_color_margin`). 14:2나 13:3처럼 15판 문턱 아래의 쏠림도 반영한다.
예를 들어 1440-1479는 흑 승률 57%로 정상처럼 보이지만, 세대 안에서는 평균 0.71만큼 한쪽으로 쏠려 있다.

- **1400에서 일시적인 유의한 퇴행이 있었다. 1440~1480에서 회복했고, 지금까지 지속적인 퇴행의 증거는 없다.**
  - 1400은 champion에 pair 5-17(p=0.017)로 졌다.
  - heavy는 63 → 67 → 66 → 72로 내려가지 않았다(72 대 63은 유의하지 않다).
  - probe는 1400~1480 사이 대응 비교에서 유의한 변화가 없다.
- **ADA1360은 가장 긴 흑 독식 국면 안에 있었다.** 1357~1402의 46세대 연속으로 흑이 15판 이상을 이겼다(평균 길이 11수).
  그 데이터로 학습한 1400이 champion에 졌다.
  - 그 뒤에도 백 독식(1417~1427 등)과 흑 독식이 번갈아 나왔지만 간격은 일정하지 않다.
  - 따라서 이것은 일정한 주기의 진동이 아니다. **색 우세 상태가 반복해서 뒤바뀌는 현상**(color-mode switching)이다.
  - 1481~1487 백 우세, 1489~1491 13:3, 1492 16:0이 가장 최근 예다.
- **높은 독식률은 만성이다.** 880 이후 대부분의 40세대 창에서 독식 세대가 37.5~72.5%였고, 색 margin은 0.62~0.86이었다. 1360-1399에서는 100%와 0.984까지 악화됐다.
  - 40세대 창의 흑 승률은 평균이라 이 쏠림을 가린다.
  - 건강 게이트도 짧은 게임 40% 이상이 두 창 연속이어야 STOP이라, 한 창짜리 붕괴(70% → 26%)에는 반응하지 않았다(설계대로).
- **reuse의 변화는 이번 현상을 설명하지 않는다.** reuse는 모든 창에서 6.39~6.40으로 일정했다. 다만 reuse 6.4라는 수준 자체가 만성 쏠림에 기여하지 않는다는 증거는 아니다. 학습률 1e-3도 똑같이 내내 일정했다.

**가설(C2에서 검증):** 고정 학습률 1e-3이 세대당 정책 변화를 너무 크게 만들어 색 우세 상태 전환을 일으킨다.
근거는 승패가 4세대(약 200 SGD step) 만에 16:0에서 0:16으로 뒤집히는 경우가 있다는 점이다. 0세대부터 학습률을 낮춘 적이 없다.
진행이 멈추면 학습률을 낮추는 것이 AlphaZero 계열의 표준 단계다. 비용은 같고, 바뀌는 변수는 하나다. 현재 데이터만으로 원인이라고 확정하지는 않는다.

**C2: 학습률 1e-3 → 3e-4 (`configs/stage8_ada_lr3e4.yaml`, ADA gen 1480에서 분기)**

- 1480에서 시작하는 이유: 1480 ≈ ADA1360(47%, p=0.66)이고 heavy는 더 높다. 대조군(ADA, 1e-3)이 이미 1480 이후로 이어지고 있어서 같은 지점에서 짝 비교가 된다.
  분기는 optimizer 상태(Adam moments)까지 그대로 두고 학습률만 바꾼다.
- 버그 수정: checkpoint에서 optimizer 상태를 복원하면 저장된 학습률이 config 값을 덮어썼다. 그래서 학습률만 바꾼 recipe 분기는 1e-3으로 돌아갔을 것이다.
  이제 config 값이 이긴다(`restore_training_state`). 일반 재개에서는 두 값이 같아 동작이 바뀌지 않는다. 지금까지의 분기는 학습률을 바꾼 적이 없다.
- 두 arm 모두 anchor를 ADA1360으로 고정한다(`run_champion_loop.py --fixed-anchor`).

**사전 판정 규칙** — `scripts/compare_recipe_arms.py`가 그대로 계산한다:

1. **직접 대결:** 1520·1560·1600에서 LR 대 CTL을 100쌍(200판)씩 둔다. seed는 **9109·9110·9111로 각각 다르게** 한다.
   같은 seed면 세 세대가 같은 개국을 반복하므로, 합친 pair를 독립 표본으로 셀 수 없다. 스크립트는 seed가 겹치면 거부한다.
   한 경기 안에서는 두 arm이 같은 개국을 공유한다. 세 경기의 결정적 pair(2:0, 0:2)를 합쳐 정확 sign test를 한다.
   LR 쪽으로 p < 0.05이면 `adopt_stronger`, CTL 쪽으로 p < 0.05이면 `reject`이다.
2. 1번이 결정하지 못하면, 다음을 **모두** 만족할 때만 `adopt_stability`이고 아니면 `keep_control`이다.
   - **안정성(1480-1599 self-play):** LR의 독식 세대 비율이 CTL보다 15%p 이상 낮고, 색 margin도 낮다.
   - **heavy 비열등:** 1520·1560·1600 heavy(seed 7007이라 두 arm의 개국과 색이 같다. 스크립트가 확인한다) 450판을 대국 단위로 짝짓는다.
     LR − CTL 승률 차이의 단측 95% 하한이 −5%p보다 커야 한다.
     "유의한 차이 없음"은 같다는 뜻이 아니므로, 허용 퇴행폭을 미리 정해 둔다.
   - **probe 퇴행 없음:** 1600 probe 전 행을 id로 짝지은 top-1 McNemar 검정에서 CTL이 유의하게(p < 0.05) 낫지 않다.
3. `adopt_stability`는 **앞으로의 학습 레시피로 채택**한다는 뜻이지, 더 강한 모델이라는 뜻이 아니다. champion은 따로 정한다.
   어느 arm에서든 ADA1360 상대 PROMOTE가 나오면 그 checkpoint는 champion 후보이고, C0처럼 새 seed 라운드로빈으로 확정한다.

### 12.18 C2 결과: 학습률 3e-4는 색 쏠림을 줄이지 못했다 → `keep_control` (2026-10-04)

두 arm 모두 1560 게이트에서 STOP이 났다(짧은 게임 40% 이상 두 창 연속). 그래서 판정은 사전 규칙대로 두 arm이 모두 도달한 1520·1560으로 했다.
LR arm의 `train` 기록은 1480 이후 `lr 0.0003`이다. 학습률 수정이 실제로 적용됐다.

| | CTL (1e-3) | LR (3e-4) |
|---|---|---|
| 대 ADA1360 1520 / 1560 | 48% (5-7) / 49% (13-14) HOLD / STOP | 46% (7-11) / 43% (11-18) HOLD / STOP |
| heavy 1520 / 1560 (/150, seed 7007) | 61 / 65 | 72 / 75 |
| self-play 1480-1559: 평균 길이, <=10수, 흑 승 | 14.0, 50%, 72% | 12.2, 68%, 85% |
| 독식 세대 / 색 margin | 74% / 0.853 | 86% / 0.914 |

세대별 우세 색(대문자 = 15판 이상 독식, 1480 → 1559):

```text
CTL WWWWwWwWwbbbBBBBBBBBBBBBBBBBBBBBBbBBBBBBBBBBBbbBBBbBBBBBBBBbBbbBbwbwWbWwWWwWWbWW
LR  WWWWWWW=wWbwBbBBbBBBBBBBBBBBBBBBBBBBbBBBBBBBBBBBBBBBbBBBBBBBBBBBBBbBBBBBBBBBBbBb
```

**판정(`compare_recipe_arms.py`): `keep_control`.**

- 직접 대결: LR 기준 pair 28-16(1520), 19-24(1560). 합계 47-40, p=0.52로 결론이 나지 않았다.
- 안정성 조건이 둘 다 실패했다: LR의 독식 세대가 오히려 12%p 많고, 색 margin도 더 높다.
- heavy 비열등과 probe 퇴행 없음은 통과했다.

**해석(외부 검토 반영):**

- **3e-4는 안정화 수단으로 기각한다.**
  - 두 arm 모두 1490 전후로 같은 흑 우세 상태에 들어갔다. 평균 9~11수, 흑이 일찍 끝내는 짧은 게임이다.
  - CTL은 1489~1544의 56세대 동안 흑 우세였다. 그 뒤 균형으로 돌아온 게 아니라 **백 우세로 넘어갔다**(1550~1559 흑 승률 14%).
  - LR은 1492~1559의 68세대 동안 흑 우세에 고착됐다(1550~1559 흑 승률 96%).
  - 즉 학습률을 낮춰도 색 쏠림의 진폭이나 독식 빈도는 줄지 않았고, 한 방향의 쏠림에 더 오래 고착됐다.
  - 다만 "높은 학습률이 색 우세 방향의 전환을 더 쉽게 만든다"는 좁은 가설은 이번 실험으로 판정하지 않는다. CTL이 흑에서 백으로 넘어가고 LR이 고착된 것은 오히려 그 가설과 양립한다.
- **직접 대결은 1520(LR 28-16)과 1560(LR 19-24) 사이에 우위 방향이 뒤집혔다.**
  두 모델 모두 불안정한 궤적 위에 있어서, 특정 checkpoint끼리의 대결 결과가 크게 흔들린다고 본다. 어느 한쪽이 더 강하다는 근거로 쓰지 않는다.
- **heavy는 LR이 높았다**(300판 대응 비교: LR만 이김 77, CTL만 이김 56, +7%p, 단측 95% 하한 +0.7%p, 양측 sign p=0.08).
  그러나 LR1560은 v5 60%, v6 60%, v7 30%로 상대별 편차가 크고, ADA1360 상대로는 43%다.
  비추이성의 증거로 기록만 하고 champion 선정에는 쓰지 않는다.
- **`no_probe_regression`은 정책(top-1) 검사만 본 것이다. 가치망에는 백 차례 국면의 큰 낙관 편향이 생겼다.**
  1560 probe 중 value 라벨이 있는 행(forced_loss, vcf_loss = 진 국면; immediate_win, vcf, vct_attack = 이긴 국면)을 둘 차례인 색별로 나눈 결과다.

  | 둘 차례 | | CTL1560 | LR1560 |
  |---|---|---:|---:|
  | 백 | 이긴 국면 평균 value / 진 국면 평균 value | +0.25 / −0.40 | **+0.84 / +0.31** |
  | 백 | 분리도(이긴 − 진) | 0.66 | 0.53 |
  | 백 | forced_loss 부호 정확도 (20개) | 65% | **20%** |
  | 흑 | 이긴 / 진 평균 value | −0.07 / −0.73 | +0.32 / −0.83 |
  | 흑 | 분리도 | 0.66 | **1.15** |

  - 백 forced_loss에서 같은 probe끼리 짝지으면 CTL만 맞힌 것이 10개, LR만 맞힌 것이 1개다(p=0.012). 검토 수치가 맞다.
  - 하지만 LR은 백의 이긴 국면에서는 더 많이 맞힌다(immediate_win 12 → 18, vcf 12 → 19). 그래서 value 부호 정확도를 모두 합치면 LR이 오히려 40 대 25로 앞선다.
  - 즉 LR 백 쪽 가치망은 판별력을 크게 잃은 것(분리도 0.66 → 0.53)이 아니라, **백 차례 국면 전체를 +0.5 안팎 위로 올려 보는 보정(calibration) 편향**이다.
    같은 노드의 자식끼리 비교하는 탐색에는 일정한 편향이 분리도 손실보다 덜 해롭다. 하지만 학습 target·탐색 백업이 −1/+1 말단값과 섞일 때는 왜곡이 생긴다.
  - replay의 백 차례 표본 target 평균은 −0.87인데 probe(돌이 많은 전술 국면)에서는 +0.3~+0.8이다. 9~11수 짧은 게임만 학습한 결과가 분포 밖 국면에서 이렇게 나타난 것으로 본다.
  - 판정 스크립트의 probe 검사에 가치망을 추가했다: 둘 차례인 색별 분리도가 0.2 이상 떨어지면 퇴행이다. 부호 정확도를 합친 값은 이런 편향을 가리므로 쓰지 않는다.
    C2 데이터로는 백 분리도 하락이 0.13이라 퇴행으로 잡히지 않는다. 판정(`keep_control`)은 바뀌지 않는다.
- **replay buffer가 한 색으로 쏠렸다.** 1560 시점(10,000 표본) 승자 기준 구성:
  CTL은 흑 승 5,960 / 백 승 4,040, LR은 **흑 승 9,416 / 백 승 584**(94%)다.
  balanced sampling은 소수인 백 승 표본을 반복해서 뽑게 된다.
- **연결된 가설:** 흑 단기승 증가 → 백 승리·방어 궤적 감소 → replay 다양성 붕괴 → 같은 소수 백 표본 반복 → 가치·정책 학습 신호 악화 → 백 방어 실패 증가 → 흑 단기승 증가.
  - §12.16 forensic에서 본 "지는 쪽 value −0.9 포화"는 self-play 분포(짧은 게임)의 국면이었다. probe의 전술 국면에서는 LR 백 쪽이 반대로 위로 치우쳤다. 색별 보정이 분포에 따라 크게 흔들린다는 뜻이다.
  - 그래서 하나의 원인보다 다음 세 가지가 맞물린 것으로 본다: 방어 국면에서 탐색이 만드는 정답(target)의 질 저하, 가치망의 색별 보정(calibration) 붕괴, replay 다양성 붕괴.
- champion은 여전히 ADA1360이다. 1480 이후 같은 레시피(학습률과 무관)로 두 arm 모두 붕괴했다.

**다음 단계(진단, 학습 없음):** STOP 창(1520-1560)에서 forensic을 100·200·400회 깊은 탐색으로 반복한다.
이번에 forensic에 두 가지를 추가했다.

- 진 쪽 색(`loser`)과 색별 요약(`by_loser`).
- 막는 수들의 root P / N / Q, 그리고 선택된 수의 Q. Q는 둘 차례인 쪽 관점이다.

**판정표:**

| forensic 결과 (색별) | 해석 | 다음 실험 |
|---|---|---|
| 100~200회에서 **백·흑 모두** 대부분 복구 | 50회 탐색 예산 부족 | self-play 50 → 100/200, 또는 일부 수만 깊게(playout cap randomization) |
| 200~400회에서만 복구 | 탐색 부족 + prior/value 약함 | 일부 수 깊은 탐색 우선 |
| **흑 방어는 복구되는데 백 방어만 실패** | 색별 가치망·보정 문제 | 백 방어 표본 / 가치망 진단 |
| P(막는 수) 낮고, 깊어질수록 Q·N 회복 | prior / policy target 문제 | 깊은 탐색으로 만든 policy target |
| P는 괜찮은데 Q(막는 수)가 계속 낮음(≈ −1) | 가치망 문제 | VCF 증명 기반 value·방어 표본 보강 |
| 400회에서도 P·Q·N 모두 이상 | 학습 데이터 / label 문제 | tactical supervision 재검토 |

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
| 8-H | 640/2,560/5,120 pilot 결과와 최종 config |
| 별도 실험 | simulations/network/within-tree batching/mixed precision/tree reuse |

**8-C 직후 결정 지점:** §2.4 상한(실제 self-play NN 비중), CPU 확장성 프록시(§3.2), GPU B=16 raw 처리량을 나란히 놓고
GPU 주 경로를 유지할지 정한다. 8-E(SearchSession)는 hybrid에서도 그대로 쓰이므로 GPU를 채택하면 낭비가 없다.

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
12. Gate 1 640판(gen 200)이 crash·NaN·illegal 없이 끝난다.
13. Gate 2 2,560판(gen 320)까지 진행할 수 있는 장기 설정을 확정한다.
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
