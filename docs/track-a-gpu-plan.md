# Track A GPU/CUDA 최적화 설계

> **v3 (2026-10-11) — 이 문서의 위치를 바로잡는다.**
> Track A 본선은 `feat/stage8-plan`(`b55cbee`)이다. 그 브랜치에는 이미 G0(§12.20–12.22)가 있다:
> `search_steps` 제너레이터, lock-step 스케줄러(`search.batched.run_lockstep`), CPU 배치 인코딩 + pinned 전송,
> 학습 GPU, IEEE float32 고정(`model.precision`), `self_play_parallel_games`, `benchmark_self_play.py`.
> 이 브랜치(`claude/renju-gpu-inference-optimization-vpbl1r`)는 Track B 계열(`ccr-993dfc5f`) 위에서 같은 G0를 다시 만들었다.
> **그대로 본선에 병합하지 않는다**(Track B 커밋 63개가 함께 들어가고, `search_steps`·`self_play`·`config`가 충돌한다).
> 본선에 없는 부분만 옮긴다(§0b).

## 0a. Track A 전체 단계 (본선 기록 기준)

| 구간 | 단계 | 결과 | 상태 |
|---|---|---|---|
| Stage 1–5 | 규칙, MCTS 기준선, policy/value 네트워크, AlphaZero PUCT | — | 완료 |
| Stage 6–7 | 학습 MVP, 7-A~7-D(PUCT v2 `tactical_rules`, sample reuse) | D16 | 완료 |
| Stage 8 Gate 1–3 | 데스크톱 CPU 이관, 색 쏠림 → 균형 샘플링(B) 채택 | B400 | 완료 |
| B400 장기 | gen 400→880 | 정체(9–10수 경주) | 완료 |
| S4 / T1 | temperature 10→4 / teacher(→Track B) | S4가 정체를 깸(대조군 상대 0.93) | 완료 |
| S2, sims100 | temperature 2, 탐색 100 | sims100 기각 | 완료 |
| 안정 루프 | adaptive steps(ADA), champion 게이트, C0 | **ADA1360 champion** | 완료 |
| C1, C2 | 정체(HOLD), lr 3e-4 | `keep_control` | 완료 |
| forensic | 지는 쪽 탐색이 220회 이하에서 "1회씩 방문"으로 퇴화 | 탐색 > 합법수 필요 | 완료 |
| **G0** | GPU lock-step 엔진 + 배치 인코딩 + 학습 GPU, RTX 5070 점검 | 1프로세스 약 2.1배, CPU 병목 | **완료** |
| **G1** | self-play 탐색 50→400, ADA1480에서 120세대 | **`adopt_stronger`**, v7 상대 처음 50% 이상 | **완료** |
| **C3** | S400 champion 선정(seed 9301 라운드로빈 + seed 7107 heavy) | — | **← 여기서 멈춤**(설계·명령 완료, 결과 미커밋) |
| 동결 | `freeze_champion.py` → `runs/anchors/S400C.pt` | — | 대기 |
| **MP** | `self_play_workers` W, 프로세스마다 GPU lock-step | 목표 self-play 3배 | 미구현 |
| G2a / G2b / G2c | S400 레시피로 누적 120 / 320 / 640세대 | — | 대기 |
| Stage 9 | 최종 라운드로빈, v5/v6/v7, probe, 사람 대국 | — | 대기 |

(§12.20의 H3/H4 "증명 라벨 지속 혼합"은 solver supervision이라 ROADMAP의 Track A 정의 밖이다. Track A 단계에서 뺐다.)

## 0b. GPU 최적화 단계 (본선 G0 이후)

| 단계 | 내용 | 기준 | 상태 |
|---|---|---|---|
| A1 | RTX 5070 점검, B1/B16/B64 | B16 0.23 ms/pos(CPU 1.49), B64 0.12 | **완료(G0)** |
| A2 | 학습 GPU | 100 step 17–21배 | **완료(G0)** |
| A3 | GPU lock-step 16, 1프로세스 | S400 self-play 중앙값 약 304 s/세대, 전체 약 366 s/세대, 평가 약 452–517회/s | **완료(G1 실측)** |
| A4 | 비동기 겹치기(`eager`) | 상한 약 1.2배(GPU가 평가 시간의 약 15–20%) | 이 브랜치에 구현, 본선 미이식 |
| A5 | checkpoint 대국 배치(refill, `--max-active 64`) | C3 라운드로빈·G2 champion 게이트 시간 | 이 브랜치에 구현, 본선 미이식 |
| **A6** | **MP: `self_play_workers` W=2/3/4/6** | **≤120 s/세대**의 주 수단 | **미구현, 다음 엔지니어링 단계** |
| A7 | CUDA Graph | A6 뒤에도 GPU 쪽이 의미 있을 때만 | 보류 |

**목표 게이트(수정).**

- 기준선은 **G1의 366 s/세대(GPU lock-step 16, S400)**다. CPU 기준선이 아니다. CPU 순차 S400은 같은 16판 벤치마크에서 약 2배 느리다(292.9 대 140.6 s).
- 최종 목표: **전체 ≤120 s/세대**(3.05배). self-play가 약 304 s에서 약 100 s로 줄어야 한다.
  1프로세스는 CPU 병목으로 약 2배가 상한이고 이미 그 위치에 있으므로, **A6(MP)이 선택이 아니라 필수**다.
- batch 크기는 판정 게이트로 쓰지 않는다. 1프로세스 평균 batch는 5–8이었고(꼬리 + CPU 병목), GPU가 병목이 아니라서 batch를 키워도 sec/gen이 줄지 않는다.
  W개로 나누면 프로세스당 batch 상한은 16/W가 되지만 같은 이유로 문제가 아니다. batch·evals/s는 진단용으로만 기록한다.
- A6 판정: W=2 → 3 → 4 → 6 순서로 `benchmark_self_play.py --workers`를 같은 16판·S400으로 재고, sec/gen이 가장 짧은 W를 고른다.
  ≤120이면 PASS, 120–150이면 채택하고 추가 최적화를 검토, >150이면 프로세스별 프로파일 후 A4/A7 검토.
- 정확성: CPU에서 W=1과 W>1의 기록이 같음, GPU에서는 게임 길이·색·짧은 게임 분포가 같은 범위, resume, 불법수 0.

---

# (v2) Track A GPU/CUDA 최적화 설계 (2026-10-10)

**범위 결정(사용자, 2026-10-10):** GPU/CUDA 최적화는 Track A에서만 한다. Track B와 E1/E2/S 계열(V8 대국)은
GPU 작업을 하지 않고 현재 구현·결과를 그대로 보존한다(근거: [gpu-throughput-review.md](gpu-throughput-review.md) §1).
과거 측정과의 비교 정의가 얽혀 있어서 런타임 구조를 바꾸지 않는다.

**v2 변경 요약(외부 검토 반영):**

- 등급을 E0 / E1 / C 세 개로 나눴다(§1). v1의 "E = 학습 결과에 영향 없음"은 GPU 수치 차이를 감안하면 너무 강했다.
- 순서를 바꿨다: smoke 수정 → 장치 분리·계측 → 학습 GPU → **공용 BatchedEvaluator → checkpoint 대국 검증** → seed 선추출 → SearchSession → 16판 dynamic batching → CPU 병렬과 비교 → CUDA Graph → (필요 시) multiprocess.
- 고정 8+8 double-buffer 대신 **ready queue + 비동기 GPU 실행**으로 바꿨다.
- CUDA Graph는 batching 뒤로 내리고, graph가 없는 크기는 eager fallback으로 처리한다.
- checkpoint 대국은 진행 중인 게임 수를 일정하게 유지하는 **refill scheduler**로 바꿨다.
- 검토에서 사실과 다른 부분 세 가지를 바로잡았다(§5).

## 0. Track A 현재 상태 (저장소 기준)

| 항목 | 값 |
|---|---|
| 계보 | Stage 8 B400(균형 샘플링) → 장기 학습 gen 880에서 정체 확정(stage8-plan §12.10) |
| 진행 중 arm | **S4**: `temperature_moves` 10 → 4(`configs/stage8_b400_temp4.yaml`). **코드 변경 없이 끝까지 진행한다.** 결과는 아직 저장소에 없다 |
| 다음 후보 | S4 판정 → GPU 인프라 완성 → **Stage 9(network 확대)**. 인프라 없이 큰 network로 가면 self-play가 크게 느려진다 |
| 설정 | 64×4, PUCT 50 sims, `tactical_rules: true`, **세대당 16판**, batch 32 × 50 step, `device: cpu`, `torch_threads: 1` |
| 세대 시간(데스크톱 CPU) | 약 34 s = self-play 64%(그 안의 NN 79%, 세대 전체의 약 50%) + 학습 19% + 루프 내 평가 12판 14% |
| 외부 평가 | light 20세대마다(tactical·MCTS-v2·v321 25쌍, probe 3종), heavy 80세대마다(v321·v5·v6·v7 10쌍 + B400 100판) |
| GPU 코드 | `batch_to_device`(학습)와 `check_stage8_gpu.py`뿐이다. `SearchConfig.evaluator_batch_size`는 1만 허용한다. **SearchSession·scheduler는 저장소에 없다** |
| 하드웨어 기록 | ROADMAP·stage8-plan은 RX 6600(ROCm)이다. 지금 GPU는 RTX 5070(CUDA)이다 |

**하나의 `device` 키가 self-play·학습·평가를 모두 정한다**(`loop.py`의 `PolicyValueEvaluator(state.model, device=config['device'])`).
그래서 지금 `device: cuda`로 바꾸면 self-play가 B=1 GPU가 되고 이득이 없다(stage8-plan §12.8).

**사용자 측정(RTX 5070, 데스크톱, 저장소 밖):** evaluator 전체 경로 B1 약 488 / B64 약 8,151 positions/s, forward만 B64 약 58,250/s.
self-play parallel 16에서 평균 batch 5.4–8.2(50–400 sims). 이 측정에 쓴 scheduler 코드는 저장소에 없다(§5-1).

## 1. 등급

| 등급 | 의미 | 검증 | 예 |
|---|---|---|---|
| **E0** | 결과까지 같은 실행 변경. 기보·checkpoint가 비트 단위로 같다 | 기존 경로와 exact equality | telemetry, 장치 키 분리(값이 같을 때), seed 선추출, SearchSession(fake evaluator 기준), scheduler 로직 |
| **E1** | 알고리즘은 같고 float 연산 순서만 다르다. checkpoint·기보가 비트 단위로 같지 않을 수 있다 | CPU 대비 prior/value tolerance, 학습 loss 곡선 비교, 같은 seed의 대국 결과 비교. critical hash는 그대로 | CUDA 학습, GPU batched 추론, batch 크기에 따른 cuDNN 알고리즘 차이, CUDA Graph, pinned/non_blocking 전송 |
| **C** | 학습 알고리즘이나 데이터 분포가 바뀐다 | 별도 arm(같은 상태에서 분기, 직접 대국) | BF16/TF32, 세대당 게임 수, virtual loss, 학습 batch 크기 |

- E1은 stage8-plan §7.5의 "GPU 경로는 기보 해시 동일성을 요구하지 않는다"와 같은 범위다.
- **한 run 안에서 E1 경로를 섞지 않는다.** 한 run은 시작부터 끝까지 같은 장치 조합으로 돌린다. 장치 조합은 run metadata에 남긴다(stage8-plan §14-10).
- E1 경로는 언제든 CPU reference 경로로 되돌릴 수 있어야 한다(stage8-plan §14-12).

## 2. 작업 목록 (실행 순서)

### 단계 1 — 기반

| # | 작업 | 등급 | 내용과 완료 조건 |
|---|---|---|---|
| 1 | S4 완료 | — | 코드 변경 없이 끝낸다. 아래 작업은 별도 브랜치에서 하고 S4 실행 중인 checkout에는 넣지 않는다 |
| 2 | RTX 5070 smoke 수정·재실행 | E0 | `check_stage8_gpu.py`의 training 단계가 데스크톱에서 `self_play_parallel_games` KeyError로 실패했다(사용자 보고). 이 키는 저장소에 없으므로 **데스크톱 로컬 코드로 만든 checkpoint의 config와 저장소 config 검증이 맞지 않는 것**으로 본다. 원인 확인 → inference parity → training smoke → checkpoint roundtrip → batch benchmark 순서로 통과시킨다. ROADMAP·stage8-plan 하드웨어 행을 RTX 5070으로 갱신한다 |
| 3 | 장치 키 분리 | E0 | `device` → `training_device` / `self_play_device` / `evaluation_device`(NON_CRITICAL). 기존 `device` 키만 있으면 셋 다 그 값으로 읽어서 기존 설정·hash가 그대로다 |
| 4 | 계측 | E0 | 세대별 phase 시간(self-play / 학습 / 루프 내 평가 / checkpoint), GPU 시간(CUDA event), batch 크기 histogram(p50/p95/max, 시간 가중), active game 수의 시간 분포. **판정 지표는 seconds/generation**이고 GPU 사용률은 참고만 한다 |

### 단계 2 — 학습 GPU

| # | 작업 | 등급 | 내용과 완료 조건 |
|---|---|---|---|
| 5 | 학습만 CUDA | E1 | `training_device: cuda`, self-play·평가는 CPU. self-play용 CPU 복사본은 세대마다 동기화한다(64×4 가중치 약 1.6 MB). **RTX 5070에서 CPU 대비 새로 잰다.** 학습이 세대의 19%라 무한히 빨라져도 세대 전체는 약 1/(0.81 + 0.19/10) ≈ 1.2배다 |
| 6 | step 동기화 줄이기 | E1 | `train_step`의 non-finite 검사·loss 기록이 step마다 `.item()`으로 기다린다. 50 step 동안 GPU tensor로 모아 세대 끝에 한 번 읽는다. 위반이 나오면 그 step 번호와 함께 지금과 같은 오류를 낸다. 단, 위반 뒤의 step이 이미 실행됐을 수 있으므로 **위반 시에는 세대 시작 checkpoint에서 다시 시작**하는 규칙으로 바꾼다 |
| 7 | D4 증강 GPU 적용 | E1 | 변환 번호(0–7)는 지금처럼 `augment_rng`(CPU)에서 같은 순서로 뽑고, rot90/flip 적용만 GPU batch tensor에서 한다. 같은 입력에서 CPU 적용 결과와 tensor가 같아야 한다 |

### 단계 3 — 공용 BatchedEvaluator와 평가 검증

| # | 작업 | 등급 | 내용과 완료 조건 |
|---|---|---|---|
| 8 | BatchedEvaluator (8-D) | E1 | 지금 GPU 경로는 position마다 GPU 위에서 encode한다(작은 kernel 약 10개 + 작은 H2D). CPU numpy로 `[B,6,15,15]`·`[B,225]`를 **미리 잡은 pinned buffer**에 채우고 H2D 1회(`non_blocking`) → forward → masked softmax → D2H 1회. 검증은 batch 단위 numpy로 한다(계약 불변: 합법수 밖 0, 합 1, value 범위). `Evaluator` 프로토콜(`evaluate_batch`)을 그대로 구현해 기존 호출부에서 바로 쓸 수 있게 한다. 비동기용 `submit(batch) -> handle`, `collect(handle)`도 둔다(단계 4의 ready queue용). 완료 조건: CPU `PolicyValueEvaluator` 대비 tolerance, B64 전체 경로가 forward에 가까워짐 |
| 9 | checkpoint 대국 batching | E1 | `run_stage8_head_to_head.play_match`(지금 한 판씩 직렬)를 여러 판 동시 진행으로 바꾼다. 양쪽이 모두 NN이고 게임이 독립이라 batch를 16보다 크게 모을 수 있다. 수 선택이 temperature 0·noise off라 RNG가 없다 |
| 9a | refill scheduler | E0(로직) | 게임이 끝나면 아직 시작하지 않은 게임을 넣어 **active game을 목표 수(예: 64)로 유지**한다. 라운드로빈(쌍당 100판, 15쌍)은 쌍을 넘어 한 queue로 돌린다. 모델별로 batch를 따로 만든다 |
| 9b | 검증 | — | (1) Scripted/Uniform evaluator로 직렬 `play_match`와 **기보 exact equality**(E0, scheduler 로직 검증). (2) 실제 NN에서는 GPU-batched 대 CPU 직렬의 **기보 일치율과 점수**를 보고한다. visit 동률이 float 차이로 뒤집힐 수 있어 100% 일치를 요구하지 않는다(E1). (3) 처리량: 같은 100판의 wall time |

### 단계 4 — self-play batching

| # | 작업 | 등급 | 내용과 완료 조건 |
|---|---|---|---|
| 10 | seed 선추출 | E0 | 게임별 RNG는 **이미 있다**: `play_self_play_game`은 게임마다 `Random(seed)`를 만들고 Dirichlet noise와 temperature 샘플링은 그 RNG만 쓴다(`search.alphazero`). 남은 것은 `generate_self_play`가 seed를 게임 시작 직전에 `self_play_rng`에서 뽑는 순서뿐이다. 세대의 16개 seed를 **시작 전에 같은 순서로 모두 뽑으면** 순차와 같은 seed가 되고 scheduling 순서와 무관해진다. 직렬 경로에서도 기록이 비트 단위로 같아야 한다 |
| 11 | SearchSession (8-E §7.2) | E0 | `run_search`를 "평가할 snapshot을 내보내고(`NeedEvaluation`), 결과를 받아 이어가는(`resume`)" 형태로 바꾼다. 기존 `run_search`는 이 session을 B=1로 즉시 돌리는 wrapper로 남긴다. Uniform/Scripted evaluator에서 기존 구현과 **exact equality**, `tactical_rules`·fast path·terminal 처리 포함 |
| 12 | 16판 dynamic batching (8-E §7.4) | E1 | 16개 session을 돌면서 snapshot이 생기는 대로 **ready queue**에 넣는다. GPU가 비어 있고 queue가 차 있으면 그때까지 모인 것을 `submit`하고, 그동안 CPU는 다른 session을 진행한다. 결과가 준비되면(CUDA event) `collect`해 해당 session에 돌려준다. 인위적 대기는 넣지 않는다. 상한: **세대당 16판 × 게임당 미완료 요청 1개 × virtual loss 없음 → batch ≤ 16**. 세대 끝에는 남은 게임 수만큼 줄어든다(다음 세대 게임은 새 가중치가 필요해 앞당길 수 없다). 고정 8+8 묶음은 느린 게임 하나가 묶음 전체를 붙잡으므로 쓰지 않는다 |
| 13 | CPU 병렬 기준선과 비교 | E0/E1 | GPU 없이 16판을 CPU 프로세스(P코어 수)에 나누는 경로를 같은 계측으로 잰다. 비교는 seconds/generation과 phase 분해다. 64×4에서는 CPU 병렬이 비슷하거나 빠를 수 있다. 둘 중 빠른 쪽을 Gate 경로로 고르고, 다른 쪽은 reference로 남긴다 |

### 단계 5 — 측정 뒤에만

| # | 작업 | 등급 | 내용과 조건 |
|---|---|---|---|
| 14 | CUDA Graph | E1 | 12의 batch 분포와 profiler에서 kernel launch가 GPU 시간을 지배할 때만. 자주 나오는 크기(histogram 상위)만 capture하고 **나머지는 eager fallback**으로 시작한다. padding(2의 거듭제곱으로 올림)은 그다음 선택지다. Windows에서는 `torch.compile`(Triton)보다 설정이 단순하다 |
| 15 | multiprocess + GPU 서버 (8-F) | E1 | 12·13 뒤에도 CPU tree 쪽이 병목일 때만. CPU actor 프로세스 k개가 shared memory 버퍼에 encode한 입력을 쓰고, queue에는 (actor, slot, 개수) 같은 작은 descriptor만 보낸다. Windows에서 Python 객체를 pickle해 queue로 보내는 방식은 쓰지 않는다. batch 상한은 여전히 16 |
| 16 | 루프 내 평가 12판 batching (8-F §8.1) | E1 | 11·12를 재사용한다. random·tactical 상대는 모델 쪽만, previous 상대는 두 모델이 각자 batch를 만든다 |
| 17 | probe batch | E1 | probe 국면 raw policy/value를 한 번의 batch forward로 |

### 단계 6 — Stage 9 (별도 arm)

| # | 작업 | 등급 | 내용 |
|---|---|---|---|
| 18 | network 확대 | C | 단계 1–4가 끝나고 13의 비교가 나온 뒤 시작한다 |
| 19 | BF16/TF32 추론·학습, fused Adam, 큰 학습 batch | C | 속도 → policy/value parity → probe → 직접 대국 순서로 채택한다. 작은 policy 차이가 MCTS 선택으로 커질 수 있다 |
| 20 | batch 16 초과 | C | 세대당 게임 수 증가 또는 virtual loss(tree 내 여러 leaf). 학습 데이터 분포가 바뀐다 |

## 2a. 구현 상태 (2026-10-10, 브랜치 `claude/renju-gpu-inference-optimization-vpbl1r`)

데스크톱 GPU에서 아직 재지 않았다. 아래 "검증"은 이 컨테이너(CPU)의 테스트 결과다.

| # | 상태 | 구현 | 검증 |
|---|---|---|---|
| 2 | 부분 | `check_stage8_gpu.py`의 batch benchmark에 `device_batched`(BatchedEvaluator) 행 추가 | KeyError는 저장소에 없는 키라 재현 불가(§5-1). 데스크톱에서 재실행 필요 |
| 3 | 완료 | `devices: {training, self_play, evaluation}`(null = `device`), `training.config.role_device` | NON_CRITICAL: critical hash 불변(`test_track_a_gpu`) |
| 4 | 완료 | generation 이벤트에 `devices`, `self_play_batching`(batch histogram, p50/p95/max, request 가중 평균, device/wait 시간), 평가 파일에 `seconds`, 평가 summary에 `batching`, run metadata에 `devices`·`parallel` | 테스트 |
| 5 | 완료 | 모델은 학습 장치에 두고, self-play·평가는 `model_for_device` 복사본(장치가 같으면 같은 객체). checkpoint의 `device`는 학습 장치 | 기존 학습·resume 테스트 전부 통과 |
| 6 | **보류** | step별 `.item()` 동기화는 그대로다 | 5를 데스크톱에서 잰 뒤 학습 시간이 여전히 의미 있을 때만 |
| 7 | 완료 | `build_batch(device=...)`: GPU면 먼저 옮기고 `augment_batch_grouped`(symmetry별 묶음, 같은 RNG 소비) | 기존 `augment_batch`와 tensor 동일 |
| 8 | 완료 | `model/batched_evaluator.py`: CPU 일괄 encode(`encode_snapshots`), CUDA면 재사용 pinned buffer + `non_blocking` H2D/D2H + CUDA event, `submit`/`collect` | encode가 `encode_game`과 동일, CPU 출력이 같은 batch의 `PolicyValueEvaluator`와 비트 동일 |
| 9, 9a | 완료 | `run_stage8_head_to_head.py --batched --device cuda --max-active 64 [--max-batch N] [--eager]`: 모든 match-up의 게임을 한 queue로, active 게임 수 유지(refill) | Scripted가 아닌 실제 신경망으로도 직렬 기보와 동일(CPU) |
| 10 | 완료 | `generate_self_play`가 세대의 seed를 먼저 다 뽑는다 | 기존 기록 해시 테스트 통과 |
| 11 | 완료 | `search.alphazero.search_steps`(generator). `search_with_tree`는 `drive_steps`(batch 1) wrapper | 기존 search 테스트 73개 전부 통과 |
| 12 | 완료 | `search/scheduler.py`(`run_tasks`), `parallel: {self_play, evaluation: serial/batched, max_batch, eager}` | 직렬과 기록 해시 동일(Scripted, 실제 신경망 CPU), 비동기 evaluator stub에서도 동일, tiny run 2세대의 기보·가중치 동일 |
| 13 | 미착수 | CPU 병렬 기준선 | 데스크톱 측정 단계 |
| 14–16 | 일부 | 16(루프 내 평가 batching)은 12와 함께 들어갔다(상대별로 묶음). 14·15는 측정 뒤 | — |

**등급 보충.** CPU에서도 batch 구성에 따라 float가 미세하게 달라진다(64×4, B16 대 B1 prior 차 1.1e-6). 테스트의 기록 동일성은
visit count가 그 차이에 흔들리지 않았다는 관측이지 보장이 아니다. 그래서 batched 경로는 CPU에서도 E1로 다룬다.
scheduler 로직 자체(순서·RNG·refill)는 Scripted evaluator로 E0 검증했다.

### 사용법

S4가 끝난 뒤 별도 run이나 분기에서 쓴다. 키는 모두 NON_CRITICAL이라 기존 checkpoint에서 이어갈 수 있지만,
한 run 안에서 장치 조합을 섞지 않는다(§1).

```yaml
devices:
  training: cuda          # null이면 device
  self_play: cuda         # batched와 함께 써야 의미가 있다
  evaluation: cpu
parallel:
  self_play: batched      # serial | batched
  evaluation: serial      # 루프 내 12판; batched면 상대별로 묶음
  max_batch: null         # null = 진행 중인 게임 수(세대당 16)
  eager: false            # true: 장치가 비면 작은 batch라도 바로 보냄(비동기 장치에서만 의미)
```

```text
# 데스크톱 측정 순서 (각 10세대 이상, 같은 checkpoint·seed, 다른 run 없이)
1. python scripts/check_stage8_gpu.py --checkpoint <ckpt> --device cuda --output runs/gpu5070_smoke.json
2. 학습만 GPU:      devices.training: cuda
3. self-play batched: devices.self_play: cuda, parallel.self_play: batched
4. eager 비교:      parallel.eager: true
5. checkpoint 대국:  python scripts/run_stage8_head_to_head.py --checkpoint A=... --checkpoint B=... \
                       --pairs 50 --batched --device cuda --max-active 64
   (같은 명령을 --batched 없이 CPU로 한 번 돌려 기보 일치율과 시간을 비교)
```

`metrics.jsonl`의 generation 이벤트에서 `self_play_seconds`, `training_seconds`, `self_play_batching.request_weighted_batch`,
`device_seconds`, `wait_seconds`를 §4 표에 옮긴다.

## 3. 하지 않는 것

- self-play 추론 batch 64/128/2048 목표(단계 6의 C arm 없이는 불가능)
- 학습·추론을 CUDA stream 두 개로 동시에(세대 순서가 재현성 계약)
- 렌주 규칙·tree를 CUDA로 옮기기
- 매 iteration `torch.cuda.synchronize()`(측정 코드에서만)
- GPU 사용률을 목표로 삼기(목표는 같은 학습 정의를 더 짧은 시간에)

## 4. 결과 보고 형식

| 경로 | sec/gen | self-play | 학습 | 루프 내 평가 | batch p50/p95 | 등급 |
|---|---|---|---|---|---|---|
| CPU 직렬(기준) | | | | | 1/1 | — |
| 학습 GPU | | | | | 1/1 | E1 |
| CPU 병렬 | | | | | 1/1 | E0 |
| GPU 16판 batching | | | | | | E1 |

같은 checkpoint, 같은 seed, 학습 중단 없이 각 경로를 최소 10세대 잰다. 벤치마크 중에는 다른 run을 돌리지 않는다.

## 5. 외부 검토에서 바로잡은 사실

1. **parallel 16 측정 코드와 `self_play_parallel_games` 키는 저장소에 없다.** 데스크톱에 로컬 구현이 있다면 먼저 커밋해야 한다.
   단계 4는 그 코드를 기준으로 이어 갈 수 있다. 그 전까지 smoke의 KeyError는 로컬 config와 저장소 검증 코드의 불일치로 본다.
2. **"RX 6600에서는 CUDA를 쓸 수 없었다"는 절반만 맞다.** NVIDIA CUDA는 없지만 TheRock ROCm PyTorch가 같은 `cuda` 장치 API로 동작했고
   smoke를 통과했다(stage8-plan §12.3: 학습 step 12.5 vs 134.7 ms, 10.7배). ROADMAP 98행의 "CUDA를 사용할 수 없으므로"는 그 이전에 쓴 문장이다.
   다만 기준 장치가 바뀌었으므로 RTX 5070 수치는 새로 잰다(작업 2, 5).
3. **게임별 RNG stream은 이미 있다.** 바꿀 것은 seed를 뽑는 시점뿐이다(작업 10).
4. Track A self-play에는 VCT·VCF 탐색이 없다. 게임 사이 속도 차이는 `tactical_rules`(즉승·즉방 필터), 금수 판정, 게임 길이에서 온다.
   ready queue를 쓰는 이유는 같다(느린 게임이 다른 게임을 붙잡지 않게).
