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
