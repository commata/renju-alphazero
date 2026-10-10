# Track A GPU/CUDA 최적화 목록 (2026-10-10)

**범위 결정(사용자, 2026-10-10):** GPU/CUDA 최적화는 Track A에서만 한다. Track B와 E1/E2/S 계열(V8 대국)은
GPU 작업을 하지 않고 현재 CPU 경로를 그대로 유지한다(근거: [gpu-throughput-review.md](gpu-throughput-review.md) §1).

## 0. Track A 현재 상태 (저장소 기준)

| 항목 | 값 |
|---|---|
| 계보 | Stage 8 B400(균형 샘플링) → 장기 학습 gen 880에서 정체 확정(stage8-plan §12.10) |
| 진행 중 arm | **S4**: `temperature_moves` 10 → 4(`configs/stage8_b400_temp4.yaml`). 결과는 아직 저장소에 없다 |
| 다음 후보 | S4 판정 뒤 64×4 계속 또는 **Stage 9(network 확대, 96×6 등)**. stage8-plan §12.8은 GPU 묶음 self-play(8-D/8-E)를 Stage 9에서 하기로 했다 |
| 설정 | 64×4, PUCT 50 sims, `tactical_rules: true`, **세대당 16판**, batch 32 × 50 step, `device: cpu`, `torch_threads: 1` |
| 세대 시간(데스크톱 CPU) | 약 34 s = self-play 64%(착수당 NN 79%) + 학습 19% + 루프 내 평가 12판 14% |
| 외부 평가 | light 20세대마다(tactical·MCTS-v2·v321 25쌍, probe 3종), heavy 80세대마다(v321·v5·v6·v7 10쌍 + B400 100판) |
| GPU 코드 | `batch_to_device`(학습)와 `check_stage8_gpu.py`뿐이다. `SearchConfig.evaluator_batch_size`는 1만 허용한다. **SearchSession·scheduler는 없다** |
| 하드웨어 기록 | ROADMAP·stage8-plan은 아직 RX 6600(ROCm)이다. 지금 GPU는 RTX 5070(CUDA)이다 |

**하나의 `device` 키가 self-play·학습·평가를 모두 정한다**(`loop.py`의 `PolicyValueEvaluator(state.model, device=config['device'])`).
그래서 지금 `device: cuda`로 바꾸면 self-play가 B=1 GPU가 되고 이득이 없다(stage8-plan §12.8).

## 1. 목록

표기: **E** = execution-only(학습 결과 계약 불변, critical hash 불변), **C** = training-critical(별도 arm).
GPU float는 CPU와 비트 단위로 같지 않다. stage8-plan §7.5대로 GPU 경로에는 기보 해시 동일성을 요구하지 않고, tolerance와 대국 결과로 확인한다.

### A. 기반 (먼저)

| # | 항목 | 위치 | 구분 | 내용 |
|---|---|---|---|---|
| A1 | RTX 5070 smoke 재실행 | `scripts/check_stage8_gpu.py` | E | CUDA wheel·드라이버 버전, CPU 대비 prior/value 차이, 학습 step 시간을 다시 기록한다. ROADMAP 하드웨어 행을 갱신한다 |
| A2 | 장치 키 분리 | `training/config.py`, `loop.py`, `evaluation.py` | E | `device` 하나를 `train_device` / `self_play_device` / `eval_device`로 나눈다(NON_CRITICAL). 기존 키는 셋 다 같은 값으로 읽는다 |
| A3 | 측정 항목 | `loop.py` metrics | E | 세대 wall time, self-play/학습/평가 분해, batch 크기 histogram(p50/p95/max), GPU 시간. **판정 기준은 세대 wall time**(stage8-plan §8) |

### B. 학습 (세대 시간의 19%, 가장 쉬움)

| # | 항목 | 구분 | 내용 |
|---|---|---|---|
| B1 | 학습만 GPU | E | A2 뒤 `train_device: cuda`. RX 6600에서 학습 step 10.7배였다. 학습 6.3 s → 1 s 미만, 세대 시간 약 1.2배. self-play용으로는 CPU 복사본을 세대마다 동기화한다(64×4 가중치 약 1.6 MB) |
| B2 | step 내 동기화 줄이기 | E | `train_step`의 non-finite 검사·loss 기록이 step마다 `.item()`으로 CPU를 기다린다. 50 step을 GPU에서 이어 돌리고, 검사는 GPU tensor로 모아서 끝에 한 번 읽는다(검사 의미는 유지: 위반 시 그 step에서 멈춘 것과 같은 오류) |
| B3 | augmentation | E | `augment_batch`의 D4 변환 선택은 `augment_rng`(CPU) 그대로 두고, 적용(rot90/flip)만 GPU batch tensor에서 한다. 결과 tensor가 같아야 한다 |
| B4 | BF16 AMP, fused Adam | **C** | batch 32의 64×4에서는 이득이 작다. Stage 9(큰 network·큰 batch)에서 arm으로 |

### C. Self-play (세대 시간의 64%, 핵심)

| # | 항목 | 구분 | 내용 |
|---|---|---|---|
| C1 | evaluator data path (8-D) | E | 지금은 position마다 GPU 위에서 encode한다(작은 kernel 약 10개 + 작은 H2D). CPU numpy로 `[B,6,15,15]`·`[B,225]`를 **미리 잡은 pinned buffer**에 채우고 H2D 1회 → forward → masked softmax → D2H 1회. 검증은 batch 단위. 사용자 측정 B64 전체 7.85 ms 중 forward 밖 약 6.7 ms를 줄이는 항목 |
| C2 | SearchSession (8-E §7.2) | E | `run_search`를 "leaf snapshot을 내보내고 결과를 받아 이어가는" resumable 형태로 바꾼다. 기존 `run_search`는 B=1 wrapper로 남기고 Uniform/Scripted evaluator에서 **exact equality** 테스트 |
| C3 | MultiGameScheduler (8-E §7.4) | E | 세대의 16판을 한 프로세스에서 같이 진행하고, 준비된 leaf를 모아 `evaluate_batch(≤16)`. 게임마다 자기 `Random(seed)`를 그대로 쓰므로 evaluator 출력이 같으면 게임 결과도 같다. **batch 상한 = 16**, 세대 끝에는 남은 게임 수만큼 줄어든다. 인위적 대기는 넣지 않는다 |
| C4 | CPU/GPU 겹치기 | E | 단일 프로세스에서는 CPU(선택·`legal_moves`·규칙 필터·encode)와 GPU가 번갈아 돈다. 16판을 두 묶음으로 나눠 한 묶음의 batch가 GPU에 있는 동안(`non_blocking` + CUDA event) 다른 묶음의 tree를 진행한다. batch는 8로 줄지만 GPU 시간이 가려진다. C3 측정 뒤 결정 |
| C5 | CUDA Graph | E | 64×4 B≤16 forward는 kernel launch가 지배한다. batch를 1/2/4/8/16 bucket으로 padding하고 bucket마다 graph를 capture한다(eval 모드 BatchNorm이라 padding 행은 다른 행에 영향 없음). Windows에서는 `torch.compile`(Triton)보다 설정이 단순하다 |
| C6 | hybrid (8-F) | E | C3·C4 뒤에도 CPU 잔여가 크면(stage8-plan §2.4 상한 약 3.4–5.4배) CPU actor 프로세스 k개 + GPU 서버 1개(shared memory 큐). 16판을 actor에 나눠 CPU 부분을 병렬화한다. batch 상한은 여전히 16 |
| C7 | 비교 대상 | — | **CPU worker 병렬**(GPU 없이 P코어 6개에 게임 분산, stage8-plan §12.8 예상 4–5배)과 세대 wall time으로 비교한다. 64×4에서는 이쪽이 비슷하거나 빠를 수 있다 |
| C8 | TF32/BF16 추론 | **C 취급** | float가 바뀌면 visit count가 바뀐다. 사용자 측정에서 TF32 이득은 B64 8,151 → 8,599/s로 작다. Stage 9에서 parity 확인 뒤 |
| C9 | batch 16 초과 | **C** | 세대당 게임 수 증가나 tree 내 virtual loss가 필요하다. 둘 다 학습 데이터 분포를 바꾸므로 별도 arm(stage8-plan §9.2·§9.3) |

### D. 평가 (batch 상한이 16에 묶이지 않는 곳)

| # | 항목 | 구분 | 내용 |
|---|---|---|---|
| D1 | 루프 내 평가 12판 (8-F §8.1) | E | C2·C3 재사용. random·tactical 상대는 모델 쪽 요청만 batch로 묶고, previous 상대는 두 모델이 각자 batch를 만든다 |
| D2 | **checkpoint 대 checkpoint 대국** | E | heavy 지점의 B400 기준 100판, 라운드로빈(쌍당 100판, 2차 1,500판)은 **양쪽이 모두 NN**이다. 게임이 독립이라 batch를 50–100까지 모을 수 있다. `run_stage8_head_to_head.play_match`가 지금은 한 판씩 직렬이다. GPU 이득이 가장 확실한 곳이다 |
| D3 | probe | E | probe 국면(stage7·defense·VCT set)의 raw policy/value를 한 번의 batch forward로 처리한다. 작지만 구현이 쉽다 |
| D4 | 고전 상대 대국 (MCTS-v2, v321, v5, v6, v7) | — | 시간 대부분이 상대(CPU)다. 모델 쪽만 D1처럼 묶고, 게임 단위 프로세스 병렬로 처리한다. GPU 효과는 작다 |

### E. Stage 9 (network 확대)에서

| # | 항목 | 구분 | 내용 |
|---|---|---|---|
| E1 | C1–C6 전체가 전제 | E | 96×6이면 NN 비용이 약 2–3배라 self-play의 NN 비중이 더 커진다. B=1 CPU로는 세대 시간이 2분을 넘을 가능성이 크다(stage8-plan §12.2 재개 조건) |
| E2 | 세대당 게임 수 / virtual loss | C | C9를 Stage 9 설계 안에서 arm으로 정한다 |
| E3 | BF16 AMP 학습·추론, fused Adam, 큰 학습 batch | C | B4·C8을 이 시점에 parity → 대국 비교로 채택 |

## 2. 하지 않는 것

- self-play 추론 batch 64/128/2048 목표(세대당 16판이라 불가능, C9 없이는)
- 학습·추론을 CUDA stream 두 개로 동시에(세대 순서가 재현성 계약)
- 렌주 규칙·tree를 CUDA로 옮기기
- 매 iteration `torch.cuda.synchronize()`(측정 코드에서만)

## 3. 권장 순서

```text
A1 smoke → A2 장치 분리 + A3 측정
  → B1 학습 GPU (+B2, B3)          세대 약 1.2배, 위험 낮음
  → C1 evaluator data path
  → D2 checkpoint 대국 batching    평가 시간 큰 폭 감소, C2/C3의 첫 사용처
  → C2 SearchSession → C3 16판 scheduler → C5 CUDA Graph
  → C7 CPU worker와 세대 wall time 비교 → 필요하면 C4 / C6
  → Stage 9: E1–E3
```

D2를 C3보다 먼저 두는 이유: checkpoint 대국은 temperature 0·noise off라 결과 검증이 단순하고(같은 기보가 나와야 함),
batch를 16보다 크게 모을 수 있어서 scheduler의 효과를 가장 분명하게 잴 수 있다.

각 E 항목의 검증: B=1 wrapper exact equality(C2), illegal move/visit 0건, CPU 경로 대비 prior/value tolerance,
같은 seed에서 checkpoint 대국 기보 비교, 중단/resume 정상, CPU 경로로 즉시 되돌릴 수 있음(stage8-plan §14).
