# i5-12600K 처리량 최적화 설계 (검토 반영본)

i5-12600K(6P + 4E, 16 logical CPU) 기준으로 구상한 최적화 설계와 그에 대한 외부 검토 의견을
**현재 저장소의 실측 기록**(`docs/rule-optimization.md`, `docs/stage8-plan.md`, `docs/mcts-v8-teacher.md`)과
대조해 고친 문서다. 결론부터 적는다.

1. **"엔진 → 병렬화 → 코어 배치"라는 큰 방향은 맞다.** 다만 검토 의견의 출발 수치 일부가 최적화 전(2026-09-25 이전)
   값이고, 그 사이 이미 반영된 항목이 있다(§1).
2. **병목은 작업 종류마다 다르다(§2).** AlphaZero self-play는 CPU B=1 NN 추론이 착수 시간의 약 80%다.
   고전 탐색(V7/V8 teacher·평가 상대)은 규칙 판정기보다 **탐색 쪽 패턴 스캔**이 더 비싸다.
   "`forbidden_reason` 호출 수 감소가 1순위"는 두 작업 어느 쪽에도 그대로 맞지 않는다.
3. **지금 프로젝트의 병목은 처리량이 아니라 학습 신호다**(stage8-plan §12.10). 처리량 작업은
   §12.2의 재개 조건(세대 시간 약 2분 초과 또는 계획한 run이 하룻밤 초과)을 만족할 때 시작한다.
   예외는 V8 teacher 데이터 생성처럼 이미 CPU에 묶인 작업이다(§4).
4. 재개 순서: **측정 → CPU 프로세스 병렬(게임 단위) → 학습 GPU·평가 병렬 → 고전 탐색 hot path → GPU batched self-play**.
   P/E 코어 배치(affinity)는 마지막이다.

---

## 1. 검토 의견 사실 확인

| 검토 의견 | 저장소 기준 사실 | 판정·조치 |
|---|---|---|
| 중반 흑 `legal_moves()` 9.851 ms, Random 약 1.91 games/s가 "최신" | 9.851 ms는 **최적화 전** 사용자 측정값이다. 최적화 후 흑 중반 `legal_moves` **2.4 ms**, Random 대국 약 **20 games/s**(교대 측정, Core Ultra 7 155H, rule-optimization.md) | 수치 갱신. 아래 계획은 최적화 후 값 기준 |
| `play()`에서 전체 `legal_moves()` 제거, `has_legal_move()`처럼 조기 종료 | **이미 구현됨.** `Game.play()`는 `has_legal_move()`를 호출하고, 흑은 첫 합법점에서 종료한다(`src/renju/game.py`) | 할 일 없음 |
| 금수 사전 필터를 "±4 안 흑돌이 적으면 건너뛰기"로 하면 장목을 놓칠 수 있다 | **이미 필요조건 기반 safe prefilter가 있다**(`_quiet_counts`). 장목은 착수점을 지나는 연속 6목이라 ±4 안에 다른 흑돌이 반드시 4개 이상 있다(한쪽 끝이어도 거리 1~4). 그래서 ±4 계수로 놓치지 않는다. 사(四)는 한 5칸 창에 3개, 삼은 ±3에 2개가 필요조건이다. False는 항상 정확 판정기로 간다. 10,000 + 1,000 국면, 178만 칸 differential 불일치 0 | 원칙(false negative 금지, 애매하면 정확 판정)에 동의. 구현은 이미 그 형태 |
| "MCTS에서 `forbidden_reason` 100만 회 이상, 가장 큰 비용" | 그 기록(docs/mcts.md, 1.39M~1.61M회)은 V3.2/V5 시절이다. **현재 HEAD V7 실측(§2.2)에서 `forbidden_reason` 누적은 전체의 약 11%** | 1순위에서 내림. 실제 hot path는 탐색 쪽 패턴 스캔 |
| `forbidden_reason → _open_three → _straight_four_after_extension` 순으로 비용 집중 | 최적화 전 profile이다. 최적화 후 `_straight_four_after_extension` 호출 97.9% 감소 | 최신 profile로 대체 |
| 정책 top 후보만 정확 검사하는 lazy masking | AlphaZero 탐색은 자식 = `Game.legal_moves()` 전체이고 prior를 합법수 위에서 정규화한다. lazy masking은 prior 정규화·root noise·visit 분포를 바꾸므로 **execution-only가 아니다**(stage8-plan 원칙 1) | 기본 경로에서는 하지 않는다. 필요하면 training-critical 실험으로 분리 |
| 9칸 LUT(3^9)는 최종 판정기 말고 특징 추출에 | 동의. 재귀 삼삼은 연장점의 다른 축 금수에 의존한다 | LUT/line code는 **고전 탐색의 패턴 특징**(§2.2 hot path)에 먼저 쓴다 |
| Numba는 단순 배열 hot function부터, 호출 수 감소가 먼저 | 동의. 현재 의존성은 순수 Python + torch이고 보드는 `list[list[int]]`다 | 순수 Python 구조 개선을 먼저 재고, 그 뒤에만 검토 |
| 단일 GPU inference 프로세스, batch 64~256 | **semantic 상한이 16이다.** 세대당 16판이고, 한 tree에는 미완료 NN 요청을 1개만 둔다(stage8-plan §7.1). 64~256은 세대당 판수 증가 또는 tree 내 virtual loss가 필요하고, 둘 다 training-critical이다 | batch 목표를 "≤16, cross-game"으로 고친다(§5) |
| 데스크톱 GPU를 5060 Ti 16GB로 가정 | 저장소 기록상 데스크톱 GPU는 **RX 6600**이고 TheRock ROCm 경로가 PASS했다(stage8-plan §12.3). ROCm PyTorch도 `device: cuda`를 쓴다 | 설계는 단일 GPU 공통으로 쓴다. 5060 Ti로 바꾸면 `check_stage8_gpu.py`를 다시 돌리고 ROADMAP 하드웨어 행을 갱신한다 |
| P-core 0~11 / E-core 12~15 하드코딩 금지, 처음엔 스케줄러에 맡김 | 동의 | §3 |
| actor에 `OMP/MKL/OPENBLAS_NUM_THREADS=1` | 동의. `run_stage8_training.py`는 이미 `torch.set_num_threads(1)` | 환경 변수는 worker 시작 전에 설정 |
| WSL2 필수 아님, worker 재사용이면 spawn 비용 무시 | 동의. 지금까지 데스크톱 실측은 모두 Windows다. OS를 바꾸면 기준선부터 다시 잰다 | Windows 유지 |

---

## 2. 작업별 병목

### 2.1 AlphaZero self-play (64×4, PUCT 50 sims, CPU 1 thread)

데스크톱 실측(stage8-plan §3.2.1, §2.4):

| 항목 | 값 |
|---|---|
| 착수당 시간 (searched) | 118.7 ms |
| └ `inference_ms` (snapshot·encode·NN·검증) | 93.8 ms (**79%**) |
| └ tree (`legal_moves` 포함) | 23.9 ms |
| 착수당 NN 호출 | 37.2 |
| 세대 시간 중앙값 | self-play 17.7 s + 학습 6.3 s + 루프 내 평가 약 4.7 s |

- probe 국면 profile에서도 규칙 관련(흑/백 `legal_moves` 13.2 + 규칙 필터 6.0 ms)은 착수 72 ms 중 약 27%였다.
  **규칙 판정이 0이 되어도 self-play는 약 1.2~1.35배**가 상한이다.
- 착수당 NN 37회 × 약 2.5 ms가 대부분이다. CPU 한 코어에서 이걸 줄이는 방법은 **프로세스 병렬**(코어 수만큼)과
  **GPU batch**(§5)뿐이다.
- 세대 단위 Amdahl: self-play가 무한히 빨라져도 학습 + 평가 약 11 s가 남는다.

### 2.2 고전 탐색 (V7, V8 teacher 기반) — 현재 HEAD 실측

이 문서를 쓰면서 `eb88b46`에서 MCTS-v7 자기대국 40수를 cProfile로 쟀다(이 컨테이너 CPU, profiler 오버헤드 포함,
비율만 의미 있음). 착수당 4.56 s(profiler 포함).

| 함수 (누적) | 호출 | 누적 s | 비중 |
|---|---:|---:|---:|
| 전체 `mcts_search_v7` | 40 | 182.3 | 100% |
| `_fast_pattern_features_for_move` (후보 정렬 특징, v321) | 134,278 | 85.2 | **47%** |
| └ `_fast_open_three_in_direction` | 536,056 | 70.7 | 39% |
| `_rollout_v321` | 1,650 | 58.9 | 32% |
| `run_length` | 26.1M | 38.3 | 21% |
| `_local_legal_candidates` | 33,321 | 37.4 | 21% |
| VCF (`_find_vcf_recursive`) | 11,461 | 31.8 | 17% |
| `_threat_windows` | 48,515 | 28.7 | 16% |
| `inside` | **105.7M** | 20.7 (self) | 11% |
| `forbidden_reason` | 552,563 | 20.2 | **11%** |

(누적 시간은 겹치므로 더하지 않는다.)

해석:

- 고전 탐색의 비용은 **규칙 판정기보다 탐색 쪽 패턴 스캔**(후보 정렬 특징, rollout, 위협 창)에 있다.
- `inside()` 1억 회, `run_length()` 2,600만 회처럼 **칸 하나씩 Python 함수 호출**하는 구조 자체가 비싸다.
  호출당 비용이 작은 함수라 profiler가 과장하는 부분이 있지만, 순위는 바뀌지 않는다.
- 따라서 LUT·line code·Numba의 첫 적용 대상은 `rules.py`가 아니라 `search/mcts_v321.py`, `search/mcts_v5.py`,
  `search/mcts_v3.py`의 패턴 함수다.

### 2.3 V8 teacher

- pilot(데스크톱, 4 프로세스): 판당 약 3분, **시간당 약 80판**, 착수 시간 중앙값 0.95 s / p95 23.9 s / 최대 168.5 s.
  V8 사고 시간의 51%가 V8-B(VCT 공격), 36%가 V8-A(VCT 안전)이다(mcts-v8-teacher.md §11.12).
- **heavy-tail**이 핵심이다. 처리량 설계는 평균이 아니라 꼬리를 기준으로 해야 한다(§3.3).

- V8 자체의 cProfile은 이 컨테이너에서 40수가 25분 안에 끝나지 않아 얻지 못했다. V8 = V7 탐색 + VCT 모듈이므로
  V7 hot path(§2.2)는 공유하고, VCT 쪽 분해는 단계 0에서 데스크톱으로 잰다.

---

## 3. 12600K 프로세스 병렬 설계

### 3.1 단위: 게임 단위 프로세스, worker 재사용

- worker 하나 = Python 프로세스 하나 = 1 thread. 시작 시 `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
  `OPENBLAS_NUM_THREADS`를 1로 두고 `torch.set_num_threads(1)`, `flush_denormal`을 설정한다.
- Windows spawn이므로 worker는 run 동안 유지한다. 세대마다 바뀌는 것은 가중치뿐이다. 64×4 state_dict는 작으므로
  세대 시작 시 checkpoint 경로(또는 bytes)를 보내 다시 읽는다.
- 게임마다 seed는 지금처럼 master seed에서 **게임 index 순서로** 미리 뽑아 전달한다. 어느 worker가 어떤 순서로
  끝내든 같은 seed → 같은 기보가 나와야 한다(CPU B=1 exact equality 계약 유지, stage8-plan §3.3).
- 결과 기록은 **게임 index 순서로 정렬**해서 replay에 넣는다. 완료 순서대로 넣으면 replay 순서가 실행마다 달라져
  재현성이 깨진다.

### 3.2 AlphaZero self-play: 16판 장벽과 worker 수

세대당 16판을 모두 끝내야 학습으로 넘어간다(장벽). 그래서 worker 수는 코어 수보다 **16판을 몇 라운드로 나누는지**가
중요하다.

| worker 수 | 라운드 | 비고 |
|---|---|---|
| 6 | 3 (6·6·4) | P코어 6개. 마지막 라운드 2개 비어 있음 |
| 8 | 2 (8·8) | 6P + 2E. 낭비 없음 |
| 10 | 2 (10·6) | 4개 비어 있음 → 8과 비슷할 가능성 |
| 16 | 1 | HT 형제와 E코어까지 전부. 가장 느린 1판이 세대 시간을 정함 |

- 후보는 **8과 16**이다. 6·10은 라운드 양자화 때문에 이득이 작을 것으로 예상한다(실측으로 확인).
- 판 길이 편차(9~10수 경주 vs 긴 판)가 크면 장벽 대기가 늘어난다. `per-game wall` p50/p95와 worker 유휴 시간을 기록한다.
- 루프 내 평가 12판(약 4.7 s)도 같은 pool로 돌린다.
- 세대를 겹쳐 실행(다음 세대 self-play를 학습 중에 시작)하는 것은 이전 가중치로 둔 판이 섞이므로 **training-critical**이다.
  하지 않는다.

### 3.3 고전 탐색(V8 teacher, 외부 평가): 동적 큐와 꼬리

- 장벽이 없는 작업이다. `ProcessPoolExecutor`에 **판 단위로 하나씩**(chunksize 1) 넣고, 판 수를 worker 수보다
  훨씬 많이(예: 10배 이상) 준다. 긴 판이 E코어나 HT 형제에 걸려도 다른 worker가 나머지를 처리한다.
- `run_mcts_v8_benchmark.py --workers`가 이미 이 구조다(판마다 submit, 결과는 task key 순서로 정리). 데이터 생성 스크립트도 같은 방식으로 맞춘다.
- 꼬리 자체(착수 최대 168 s)는 병렬로 안 줄어든다. V8-5 예산 조정의 몫이다.
- 남는 코어가 있어도 **평가 측정과 학습을 같은 시간에 돌리지 않는다**(stage8-plan §8). 시간 기반 예산이 있는 탐색은
  다른 프로세스 부하에 따라 결과가 바뀐다.

### 3.4 코어 배치

- 1단계: affinity 없이 worker 수만 바꿔 잰다: **1 / 2 / 4 / 6 / 8 / 10 / 12 / 14 / 16**.
  판정 지표는 CPU 사용률이 아니라 **games/h(고전 탐색) 또는 세대 wall time(AlphaZero)**.
- 1-thread Python worker 두 개가 HT 형제로 한 P코어를 나누면 각자 단독보다 느리다. E코어 1개도 P코어보다 느리다.
  그래서 12600K의 실효 병렬도는 16이 아니라 대략 P코어 6개 + α이고, 정확한 값은 위 sweep으로 정한다.
  참고: Gate 3에서 1-thread 프로세스 3개를 동시에 돌렸을 때 arm당 예상 5.5~6시간이 7~8시간이 됐다(약 1.3배 감속).
  코어가 남는데도 감속이 있었으므로 **선형 확장을 가정하지 않는다.**
- 2단계(선택): sweep에서 같은 worker 수 대비 편차가 크면 그때 affinity를 실험한다.
  - logical CPU 번호를 하드코딩하지 않는다. Windows는 `GetLogicalProcessorInformationEx`의 EfficiencyClass,
    Linux는 `/sys/devices/system/cpu/cpu*/cpu_capacity` 또는 `lscpu -e`로 P/E를 읽는다.
  - 우선권은 CPU를 실제로 쓰는 self-play/탐색 worker에 준다. 학습 프로세스는 GPU로 옮긴 뒤에는 CPU를 거의 안 쓴다.
- 전원 모드는 최고 성능. 측정 중 다른 작업 금지.

### 3.5 기대치 (가설, 실측으로 대체)

- AlphaZero self-play: stage8-plan §12.2의 추정 그대로 self-play 약 4~5배. 세대 시간 28.7 s 중 self-play 17.7 s가
  약 4 s가 되면 세대는 약 15 s(약 1.9배)이고, 이때 **학습 6.3 s가 최대 항목**이 된다 → §4의 다음 단계가 학습 GPU인 이유.
- V8 teacher: 4 프로세스 80판/h에서 8~10 프로세스면 대략 140~180판/h를 기대한다(HT·E코어 효율 미확인).

---

## 4. 재개 순서와 각 단계의 통과 조건

모든 단계는 **execution-only**여야 한다. CPU B=1 경로의 game/record hash, Stage 5 golden hash,
V6/V7 frozen behavior(`ci_v6_behavior.py`, `ci_v7_behavior.py`), rule differential이 그대로 통과해야 한다.

| 단계 | 내용 | 통과 조건 | 시작 조건 |
|---|---|---|---|
| 0. 측정 | AlphaZero: 세대 timing, `self_play_timing` 분해, NN calls/move. 고전: V7/V8 cProfile(데스크톱), games/h, 착수 시간 p50/p95/max. 둘 다 worker sweep(§3.4) | 측정 스크립트와 결과 JSON 커밋 | 즉시 가능(코드 변경 없음) |
| 1. CPU 프로세스 병렬 | §3. AlphaZero는 세대 내 16판 + 평가 12판 pool, 고전은 동적 큐 | 같은 seed에서 serial과 기보 hash 동일(순서 무관), 세대 wall time 또는 games/h 개선 | AlphaZero: §12.2 재개 조건. V8 teacher: 대량 생성 시작 시 |
| 2. 학습 GPU + 평가 병렬 | `batch_to_device` 경로는 이미 있음(학습 step 10.7배). 세대 timing에서 학습 비중 확인 | 학습 loss가 CPU 대비 tolerance 안, resume 정상 | 1단계 후 학습이 최대 항목일 때 |
| 3. 고전 탐색 hot path | §2.2 순위대로: `inside()` 제거(사전 계산 좌표 tuple), `run_length`·`_run_info`를 line 단위 표/코드로, `_fast_open_three_in_direction`·`_threat_windows`를 line code + LUT로. 규칙 판정기는 그 뒤 | 출력 동일(후보 순서까지). V7 frozen behavior, V8 테스트, rule differential 통과 | V8 teacher 처리량이 실제 병목일 때 |
| 4. GPU batched self-play | stage8-plan 8-D(CPU batch 인코딩, GPU 경로 시간의 94%가 인코딩·전송) → 8-E(SearchSession, 16판 cross-game, B≤16) → 필요 시 hybrid(producer 프로세스 k개 + GPU 1개) | §8의 기준: total generation wall time. hybrid는 CPU-workers 대비 1.3배 미만이면 버림 | network 확대(Stage 9)로 NN 비중이 커질 때 |
| 5. affinity | §3.4 2단계 | sweep 대비 유의한 개선 | 마지막 |

### 4.1 3단계 세부 원칙

- **정확성이 필요한 곳과 아닌 곳을 나눈다.** 후보 정렬 특징은 휴리스틱이지만 바뀌면 frozen behavior가 깨진다.
  최적화도 **같은 값**을 내야 한다. 근사로 바꾸는 것은 새 엔진 버전(V9 등)의 일이다.
- line code: 각 칸에서 4방향 9칸(±4)을 3진수로 묶은 정수(3^9 = 19,683)를 보드 갱신 시 점진 갱신하고,
  "이 방향에 열린 삼 후보가 있나", "five/four 완성점" 같은 1차원 판정은 표 조회로 바꾼다.
  1차원으로 결정되지 않는 것(재귀 삼삼, 연장점의 다른 축 금수)은 표에 "애매"를 두고 기존 함수로 보낸다.
- 표는 import 시 생성하거나 생성 스크립트 + 해시로 고정한다. 기존 함수와 모든 3^9 패턴에서 같은지 테스트한다.
- 규칙 판정기 추가 개선은 측정으로 비중이 다시 커졌을 때만: 같은 보드 hash + 좌표 단위 금수 cache
  (Zobrist hash 기준이면 exact), `_fours` window 사전 계산(rule-optimization.md에서 보류한 항목).
- **하지 않는 것:** 부모→자식 국면의 점진적 합법수 갱신. 재귀 삼삼은 연장점의 다른 축까지 보므로 새 돌 하나가
  금수 판정을 바꾸는 범위의 상한이 증명돼 있지 않다. 증명과 differential 없이 넣지 않는다.
- Numba/Cython: 위 순수 Python 구조 개선 뒤에도 고전 탐색이 병목일 때만. Python 3.13 지원 버전과 Windows wheel을
  먼저 확인하고, `int8[15,15]` 배열 함수부터 옮긴다. 보드 표현을 바꾸면 변환 비용도 같이 잰다.

---

## 5. GPU 관련 정정

- **batch 상한 16.** 의미 보존 self-play에서 동시에 존재하는 독립 게임은 세대당 16판이다. B=64~256은
  (a) 세대당 판수 증가 또는 (b) tree 내 virtual loss가 필요하고 둘 다 학습 의미를 바꾼다. D32에서 한 색 붕괴가 관측됐고, 그 뒤 균형 샘플링을
  채택했지만 32판/세대는 다시 검증하지 않았다. (a)도 별도 실험으로 한다.
- **inference 프로세스 1개 + actor k개(hybrid)는 마지막 수단.** 단일 프로세스 16판 cross-game scheduler(8-E)로 GPU가
  굶는 것이 확인된 뒤에만 간다(stage8-plan §9.1). 처음부터 IPC 구조로 가지 않는다.
- **actor마다 GPU 모델 복사는 하지 않는다.** 검토 의견에 동의한다(CUDA/HIP context가 프로세스마다 생긴다).
- **동기식 단계 유지.** self-play → 학습 → 평가. 비동기 learner는 replay에 이전 가중치 데이터가 섞이므로 training-critical이다.
- **현재 GPU 수치(RX 6600):** 학습 step 12.5 vs 134.7 ms(10.7배). 추론 전체 경로 B16 0.75 vs CPU 1.44 ms/국면(1.9배),
  forward만은 B64 36배 → 인코딩·전송이 GPU 경로 시간의 94%. 8-D가 먼저인 이유다.
- **5060 Ti 16GB로 바꾸는 경우:** CUDA 경로라 backend 확인이 단순해진다. `check_stage8_gpu.py`로 같은 항목(추론 차이,
  학습 step, B1~B64 경로 시간)을 다시 재고, ROADMAP의 데스크톱 사양과 8단계 행을 갱신한다. 위 설계(B≤16, 8-D 우선)는
  GPU 종류와 무관하게 그대로다.

---

## 6. 측정 항목 (단계 0 및 각 단계 후 공통)

| 작업 | 지표 |
|---|---|
| AlphaZero 세대 | self-play / 학습 / 평가 / checkpoint s/gen, total s/gen, games/h, samples/s, 착수당 NN 호출, `inference_ms`·`legal_moves_ms`·tree ms, per-game wall p50/p95, worker 유휴 비율 |
| 고전 탐색 | games/h, 착수 시간 p50/p95/max, 예산 소진률, cProfile 상위 20(호출 수 포함), `forbidden_reason`·`run_length`·`inside` 호출 수/착수 |
| 공통 | worker 수, logical CPU 수, OS, Python/torch 버전, 전원 모드, 커밋 |

결과는 기존 관례대로 `docs/*-results.json` 또는 run 디렉터리의 JSON으로 남기고, 문서에는 비율로 비교한다
(그램·컨테이너·데스크톱의 절대 시간은 다르다).
