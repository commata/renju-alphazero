# Web_SER5: TB-RT30 — 미니PC(SER5 MAX, 5800H/16GB) 웹대국 30초 착수 제한

브랜치 `feat/Web_SER5`. 목표는 웹대국에서 **사용자가 보는 착수 시간 < 30초(위반 0)** 이다. TB-Engine-v1은 동결 그대로 두고,
실시간 제한이 들어간 파생 엔진 **TB-RT30**을 `src/rt30/`에 만든다. 동결 파일(`CLAUDE.md` 목록, `search/mcts_v5.py`·`mcts_v7.py`의
탐색 코드 포함)은 수정하지 않는다.

## 1. 출발점: E3-A 기록 실측

E3-A(= TB-Engine-v1) 2,938수(`docs/mcts-v8-results/e3/puct_policy_vct2_50k_841{5,6}.jsonl`, 데스크톱, 14 workers)를 단계별로 집계했다.

| 항목 | 값 |
|---|---|
| 착수 시간 | 중앙값 2.56초, p95 123.8초, 최대 410.7초 |
| 30초 초과 / 27.5초 초과 | **512수(17.4%) / 527수(17.9%)**. 드문 tail이 아니다 |
| 단계별 합계 | V8-C 427분, V8-B 206분, **VCT2 171분**, V8-A 109분, PUCT 58분 |
| 30초 초과 수에서 가장 오래 걸린 단계 | V8-C 184, VCT2 148, V8-B 115, V8-A 65 |
| PUCT | 중앙값 1.61초, 최대 4.9초 (데스크톱 측정값이며 상한 보장은 아님) |
| 전술 탐색 속도 | 느린 수 기준 약 1,600 nodes/s. 컨테이너 단일 프로세스는 약 1,200 nodes/s (부하 때문이 아님) |
| tree route 최악 예산 | V8-B 200k + V8-C 400k + VCT2 4×50k ≈ 800k 노드 ≈ 8분 |

V8-B 최악 국면 profile 결과 **`search.mcts_v5._threat_windows`가 77%** 였다(VCF 노드마다 572개 window를 Python으로 순회).
`_board_key`·`ordered_moves`는 이번 profile에서 병목이 아니었다.

현재 속도로 27.5초에 쓸 수 있는 노드는 약 45k로 TB1의 1/10이다. 따라서 **노드 속도 개선이 1순위**이고, deadline은 그 위의 안전망이다.
속도 향상별로 기록상 27.5초를 넘었을 수의 비율은 이렇다(같은 수의 시간이 k배 줄어든다고 가정).

| 속도 향상 | ×1 | ×2.2 | ×4 | ×8 |
|---|---|---|---|---|
| 27.5초 초과 | 17.9% | 9.9% | 6.0% | 1.3% |

## 2. 원칙

1. **동작이 같은 변경과 동작이 바뀌는 변경을 분리한다.**
   - 동작 동일(성능만): `deadline=None` + TB1 예산에서 수·diagnostics(`*_seconds` 제외)·nodes·calls가 TB1과 완전히 같아야 한다.
     통과하면 이름만 "TB1-fast"이고 기력 gate가 필요 없다.
   - 동작 변경: 예산 축소, deadline, cache 공유, fallback 변경, V8-C 모드. 결과 공간이 바뀌므로 별도 gate가 필요하다.
2. **두 가지 모드.** `deadline=None`은 재현성·기력 평가용, `deadline=27.5`는 실서비스 지연 시간 평가용이다.
   노드 예산이 1차 장치이고 wall-clock deadline은 드물게만 걸리는 2차 안전망이다(§12.9 "시간은 비교에 쓰지 않는다").
3. **시간 초과는 UNKNOWN이다.** 증명을 잃을 수는 있어도 만들어내지는 않는다(SAFE/WIN/UNSAFE는 끝까지 돈 탐색만 낸다).

## 3. 설계

### 3.1 `_threat_windows` 정확 대체 (구현됨: `rt30/fast_windows.py`)

- numpy 벡터화 버전(약 3.7배)과, numpy가 없을 때 쓰는 unrolled pure-Python 버전(약 1.8배)을 둔다.
  출력은 동결 함수와 **순서까지 같은 list**다(`WINDOWS` 순서). 후속 코드가 첫 원소나 순서를 tie-break에 쓸 수 있기 때문이다.
- 원본은 lazy generator이고 대체는 eager다. 호출부(`mcts_v5`, `mcts_v7`, `mcts_v8`, `threat_patterns`, 스크립트)를 모두 확인했는데,
  순회 중에 판을 바꾸는 곳이 없으므로 정확하다. `next(...)`로 일찍 멈추는 호출은 early exit만 잃는다.
- `install()`은 동결 함수를 import한 **모든 로드된 모듈**의 이름을 바꿔 묶는다. 영향은 그 프로세스 하나뿐이다.
  web worker는 install하고, gate·벤치마크는 명시적으로 선택한다.
- 검증:
  - 무작위 합법 국면 3,000개 × 흑/백 × minimum 0~5에서 `list(old) == list(new)`
  - 빈 판·꽉 찬 판
  - V8-B 기록 국면에서 수·checked·nodes·calls 일치(`tests/test_rt30_fast_windows.py`)
  - 기록 국면 replay에서 TB1과 동일한지(§5)
- incremental threat table(수를 놓을 때마다 window 계수를 갱신)은 더 빠를 수 있지만 정확성 위험이 커서 **별도 실험**으로 둔다.

### 3.2 Deadline (핵심 구현됨: `rt30/deadline.py`, `vcf.py`, `solver.py`)

**절대 시각으로 넘긴다.**

```text
request_received = monotonic()
engine_deadline  = request_received + 27.5   # worker에는 절대 시각을 넘긴다(대기열 시간 포함)
watchdog         = request_received + 29.0
```

**예외 경로: `SearchTimeout(_BudgetExhausted)`.** `_BudgetedSolver._bounded`가 노드 예산 cut과 똑같이 UNKNOWN으로 바꾼다.

- **캐시 오염 없음.** 예외는 `vcf`의 캐시 기록 전에, `ThreatSolver.after_move`의 캐시 기록 전에 빠져나간다.
- **스케줄러 정지.** timeout 시 `exhausted=True`로 만들어 `_proven`의 공정 배분 루프를 끝낸다.
  이걸 빼면 pending 수가 매번 즉시 timeout되면서 루프가 끝나지 않는다.
- **상태 복구.**
  - solver 안의 모든 `play`는 `finally: undo`로 되돌린다.
  - VCF probe는 `placed()` context manager로 돌을 되돌린다.
  - 테스트는 매 cut 지점마다 `deepcopy(vars(game))`(board, history, to_play, done, winner)가 그대로인지 확인한다.

**검사 위치.**

- VCF 노드마다: 동결 재귀 `_find_vcf_recursive`가 노드마다 부르는 `take_node`를 subclass로 바꿔 넣는다. 재귀 코드를 복사하지 않는다.
  - 이게 필요한 이유: VCF 한 번이 최대 20k 노드이고, 현재 속도로 수 초~15초가 걸린다.
- `vcf` 호출마다: 캐시 hit 포함. 캐시된 국면을 decision으로 열거하는 시간은 노드 예산에 잡히지 않는다.
  - 기록에 "calls 222, nodes 0, 0.34초" 같은 수가 있다.
- `after_move` 진입마다.

**아직 deadline이 닿지 않는 곳.** V7 동결 모듈(own VCF 5k, safety 8k 노드)은 deadline 없이 끝까지 돈다.
작은 예산이므로 reserve로 흡수하고, 최대 시간을 실측한다.

**검증(`tests/test_rt30_deadline.py`).**

- `deadline=None`과 먼 deadline에서 `_BudgetedSolver`·`SelectiveSolver`와 status·nodes·calls·두 캐시가 같다.
- `CheckLimit(k)`로 검사 k번째에서 강제로 cut한다(기록 국면 SAFE/UNSAFE/WIN 3개, k를 0~전체에서 로그 간격으로).
  - 결과는 UNKNOWN이거나 끝까지 돈 결과와 같다.
  - 캐시 항목은 전체 실행 캐시의 부분집합이다.
  - 판 상태가 그대로다.
  - 멈춘 solver는 다시 탐색하지 않는다.
- VCF 단독: k = 0…노드 수 전체에서 raise 또는 동결 결과와 같고, 판 상태가 그대로다.

### 3.3 Route별 시간 배분 (다음 단계: `rt30/engine.py`)

V8-A와 V8-C는 한 수에서 같이 돌지 않는다. 하나의 절대 deadline을 모든 단계가 공유하고, 앞 단계가 남긴 시간은 뒤로 넘어간다.
독립 stopwatch는 두지 않는다.

```text
absolute deadline = request + 27.5s
emergency move: 탐색 전에 만든다(§3.4)
stage1-3, own VCF (V7, deadline 없음, reserve 안)
V8-B           min(deadline, 시작 + 4초)        # 공격 증명만 둔다. UNKNOWN은 두지 않는다
stage4/5:  V8-A   deadline - FINAL_MARGIN
tree:      [guard] remaining < PUCT_RESERVE + FINAL_MARGIN 이면 PUCT를 건너뛰고 best-so-far
           PUCT   내부 deadline 없음(결정적, 끝까지)
           V8-C   deadline - (VCT2_RESERVE + FINAL_MARGIN)
           VCT2   deadline - FINAL_MARGIN. 남은 시간이 VCT2_MIN 미만이면 건너뜀
response
```

- **PUCT 진입 guard.** 4.9초는 데스크톱 측정값이지 상한이 아니다.
  - `PUCT_RESERVE`는 미니PC에서 **모든 worker를 가동한 상태의 PUCT p99**(열·전력 제한 포함)에 여유를 더해 정한다.
  - 진입하면 끝까지 결정적으로 돈다.
- **V8-C 첫 수 몫(`node_budget // 2` = 200k)이 핵심 지점이다.** 결정적 예산 보정(§4 7단계) 때 먼저 볼 값이다.
- 기존 단계별 fallback 규칙을 그대로 쓴다(V8-A `_stage_vct_move`, V8-C `_verify_root_choice`, VCT2 `_vct2_veto`).
  - timeout은 그 규칙 안에서 "UNKNOWN"과 같게 처리된다.
  - 예: V7 수가 UNSAFE가 아니면 그 수, UNSAFE면 아직 반박되지 않았고 즉시 지지 않는 첫 자식.
- **구현 방식.** 동결 `mcts_search_v8`은 solver를 내부에서 만들므로, RT 엔진은 흐름을 복사한 별도 모듈로 만든다.
  `deadline=None`에서 TB1과 완전히 같은지 replay로 확인한다.

### 3.4 Best-so-far와 비상수

- **비상수는 탐색 전에 만든다.**
  - stage1-3 강제수가 있으면 그것을 둔다.
  - stage4/5면 V7 강제수다.
  - 그 밖에는 `_not_immediately_lost`를 통과하는 policy 1위 수다.
- 비상수는 worker가 **첫 진행 메시지**로 보내고, 이후 PUCT 결과와 V8-C/VCT2의 현재 fallback 수로 갱신한다.
- 부모(supervisor)는 탐색 계산을 하지 않는다. 포화 상태에서 부모까지 CPU를 쓰면 과부하가 커진다.
  - 진행 메시지를 하나도 받지 못한 극단적인 경우에만 ms 단위 규칙으로 고른다(즉시 승리·즉시 방어, 그 외 마지막 돌 근처 합법수).

### 3.5 VCF cache 공유 (동작 변경, 이후 gate)

- 캐시 key에는 proof horizon을 정하는 **모든 의미 파라미터**를 넣는다: board, attacker, max_fours, 금수 규칙 semantics, solver mode(VCF/VCT depth, selective 여부).
- 공유 가능 여부:
  - **WIN:** 공유한다(증명).
  - **NONE:** 같은 horizon에서 끝까지 돈 경우만 공유한다. `max_fours=10`인 V7 own VCF·safety의 NONE은 ThreatSolver 계열(max_fours 무제한)과 공유하지 않는다.
  - **UNKNOWN·cut·timeout:** 공유하지 않는다.
- 공유하면 노드 사용량이 바뀌어 결과도 바뀐다. 따라서 gate가 필요하다.

### 3.6 서버 구조 (이후 단계)

**현재 구조.** `run_web_play.py`는 `ThreadingHTTPServer`지만 `PlaySession.lock` 안에서 `agent.select_move()`를 **동기로** 호출한다.
게임은 하나다. policy는 `RootPolicy(..., threads=None)`이라 torch 기본 thread 수(전체 코어)를 쓴다.

**바꿀 구조.**

```text
Browser ─HTTPS→ Nginx → Web/API (게임 상태, 대기열, supervisor)
                           │ 절대 deadline, 국면
                           ▼
                     Engine worker × N (프로세스당 1게임, torch/BLAS thread 1)
                           │ 진행 메시지: 비상수 → PUCT 수 → 현재 fallback 수
                           ▼
                     결과 (수 + diagnostics + 시간 기록)
```

- **시간 제어.**
  - 정상 경로는 cooperative deadline이다: `SearchTimeout` → UNKNOWN → fallback.
  - **29초 watchdog**은 hang이나 deadline 코드 결함에 대비한 마지막 안전장치다. 걸리면 worker를 강제 종료하고 best-so-far를 둔다.
- **worker 관리.**
  - `ProcessPoolExecutor`는 작업 하나만 kill하면 pool 전체가 broken 상태가 될 수 있다. 그래서 **커스텀 multiprocessing worker + IPC supervisor**로 만든다.
  - kill 뒤에는 예열된 예비 worker를 바로 투입하고, 새 예비를 백그라운드에서 로드한다(policy 로드에 수 초 걸림).
- **thread 수.** worker마다 `torch.set_num_threads(1)`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`로 oversubscription을 막는다.
- **대기열.** 대기 시간도 deadline에 포함된다.
  - worker 수를 넘는 요청은 대기열에 넣는다.
  - 남은 시간이 PUCT guard 미만이면 짧은 탐색(비상수 + 가능한 단계만)으로 응답한다.
- **worker 수는 실측으로 정한다.** 6은 시작값일 뿐이다(§5).
- **세션.** 멀티 사용자 세션(게임별 상태, 기록)은 PlaySession을 대체하는 별도 작업이다.

### 3.7 병렬 V8-C 자식 검사 (선택)

동시 대국과 코어를 두고 경쟁하므로 사용자가 적을 때만 의미가 있다. 맨 마지막에, 측정 결과를 보고 결정한다.

## 4. 개발 순서와 상태

| # | 단계 | 상태 |
|---|---|---|
| 1 | `_threat_windows` 정확 대체 + 동일성 테스트 | **완료** (`rt30/fast_windows.py`, 테스트) |
| 2 | Deadline 기반: clock, `SearchTimeout`, VCF 내부 검사, solver | **완료** (`rt30/deadline.py`·`vcf.py`·`solver.py`, fault-injection 테스트) |
| 3 | 미니PC baseline: TB1 그대로, 기록 국면 replay | 사용자 실행 (§5) |
| 4 | TB1-fast 동일성 확인(체크포인트 포함 전체 TB1) | 사용자 실행 (§5). 컨테이너는 uniform prior로 확인 |
| 5 | profile 상위 항목 추가 정확 고속화 | 대기 |
| 6 | RT 엔진: route별 soft cap, PUCT guard, 비상수, 기존 fallback | 다음 |
| 7 | worker supervisor, 29초 watchdog, 대기열·절대 deadline | 대기 |
| 8 | 미니PC 포화 측정: workers 1/2/4/6/8 | 대기 |
| 9 | 필요하면 결정적 예산 재보정(V8-C 첫 수 몫, VCT2 예산) → `deadline=None` 짝 대국 gate | 대기 |
| 10 | (선택) V8-C 자식 병렬 검사, incremental threat table | 대기 |

PUCT 시뮬레이션 수와 H3 policy는 이 단계에서 바꾸지 않는다(PUCT 최대 4.9초, NN 착수당 3ms).

## 5. 실행 방법

**동일성 확인과 baseline 측정.** `scripts/run_rt30_replay.py`가 기록 국면을 fresh agent(고정 seed)로 다시 탐색한다.
arm 둘이면 동일성 확인, 하나면 시간 측정이다. 수 하나에 수 분이 걸리는 국면이 있으므로 진행 로그가 1분마다 나온다.

```powershell
# TB1 vs TB1-fast 동일성: 기록상 30초 초과 국면 전부
python scripts/run_rt30_replay.py --checkpoint runs\h3_policy_64x4\best.pt --select slow --workers 4 `
  --output runs\rt30\equivalence_slow.json 2>&1 | Tee-Object runs\rt30\equivalence_slow.log
# 미니PC baseline(TB1 그대로, 한 번에 하나): 무작위 300국면
python scripts/run_rt30_replay.py --checkpoint runs\h3_policy_64x4\best.pt --arms frozen --select sample --sample 300 `
  --workers 1 --output runs\rt30\baseline_frozen_w1.json 2>&1 | Tee-Object runs\rt30\baseline_frozen_w1.log
# 같은 300국면, TB1-fast
python scripts/run_rt30_replay.py --checkpoint runs\h3_policy_64x4\best.pt --arms fast --select sample --sample 300 `
  --workers 1 --output runs\rt30\baseline_fast_w1.json 2>&1 | Tee-Object runs\rt30\baseline_fast_w1.log
```

**포화 측정.** `--workers 1/2/4/6/8`로 바꿔 가며 국면당 시간(median/p95/p99/최대)과 총 처리량을 비교한다.
같은 시간대에 CPU 클럭·온도(HWiNFO 등)도 기록한다. 한 번에 하나씩 실행하고, 다른 작업은 같이 돌리지 않는다.

## 6. 미니PC 목표 Gate

| 항목 | 목표 |
|---|---|
| hard max(사용자 체감) | < 30초, 위반 0 |
| 엔진 soft max | ≤ 27.5초 |
| median / p95 / p99 | 3~5초 / ≤ 20초 / 측정 후 결정 |
| deadline에 걸리는 비율 | 고속화 결과를 보고 정함(위 추정상 약 4배 이상 빨라져야 ≤ 5%) |
| 불법수 | 0 |
| timeout이 만든 오판 | 0 (fault-injection 테스트 + 운영 중 timeout 수의 증명 상태 기록) |
| E1 / Holdout 전술 회귀 | `deadline=None` 모드에서 확인 |
| TB1과의 착수 일치율 | 진단 지표 |
| TB1 짝 대국 기력 | 동작이 바뀌는 변경을 넣었을 때 최종 gate |
