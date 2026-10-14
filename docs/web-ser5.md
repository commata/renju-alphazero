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

### 3.3 Route별 시간 배분 (구현됨: `rt30/engine.py`)

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
- **구현 방식.** 동결 `mcts_search_v8`은 solver를 내부에서 만들므로, RT 엔진(`rt30_search`, `RT30Agent`)은 TB1 흐름
  (PUCT tree, H4 root-policy 옵션 없음)만 복사한 별도 모듈이다. solver를 만드는 세 helper(V8-B, V8-C, VCT2)만 복사해
  solver를 주입했고, 나머지는 동결 함수를 그대로 부른다.
  - deadline 정보는 동결 diagnostics에 섞지 않고 `RTInfo`(timeouts, skipped, emergency, best)에 따로 둔다.
  - `deadline=None`에서 TB1과 같은지 replay(`rt` arm)와 `tests/test_rt30_engine.py`로 확인했다(§5.1).
- **시작값**(`RT30_TIMING`): final_margin 1.5초, V8-B cap 4초, PUCT reserve 6초, VCT2 reserve 4초, VCT2 최소 2초.
  미니PC 측정으로 다시 정한다.

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

### 3.6.1 구현 (`rt30/supervisor.py`)

- `EngineSupervisor.select(game_key, seed, history, received_at)`:
  - deadline은 `received + 27.5`, watchdog은 `received + 29`다. 둘 다 `time.monotonic()` 절대 시각이고,
    Linux와 Windows에서 이 시계는 프로세스 간에 공유된다.
  - worker는 `started` → `progress(stage, move)`(비상수, PUCT, V8-C, VCT2 순) → `result` 순으로 메시지를 보낸다.
  - 결과가 오지 않은 채 watchdog이 되면, **마지막 progress 수를 먼저 반환하고** 그 뒤에 worker를 정리한다.
    정리는 별도 thread에서 terminate → kill 순으로 하고, 예열된 spare가 바로 교체된다.
    progress가 전혀 없으면 ms 단위 규칙으로 고른다(`fallback_move`: 즉시 승리 → 즉시 방어 → 마지막 돌 근처).
  - worker 오류는 best-so-far를 두고 worker를 유지한다. worker가 죽으면 교체한다.
  - 동시 탐색은 1개다(lock). lock을 기다린 시간도 같은 deadline 안에 들어간다.
- `ProcessPoolExecutor`가 아닌 직접 관리하는 `spawn` 프로세스와 `Pipe`를 쓴다.
  - worker 안에서 torch와 BLAS thread를 1로 고정한다.
  - 게임마다 엔진 agent를 따로 둔다(난수 흐름 유지, 최대 8게임).
  - policy checkpoint는 worker에서만 로드한다.
- 웹: `python scripts/run_web_play.py --rt30 [--rt30-uniform-prior] --db data/renju_web.db`로 "rt30" 상대를 띄운다.
  DB `moves`에 `queue_seconds`, `engine_seconds`, `wall_seconds`, `deadline_seconds`, `timed_out`, `emergency_used`가 기록된다.
  `rt_json`에는 timeouts, skipped, source, watchdog, best, error가 들어간다.
- 측정: `run_rt30_replay.py --arms sup --deadline 27.5`는 같은 국면을 웹 서버와 같은 경로로 돈다.
  `seconds`가 요청부터의 wall time이고, `sup_wall`에 source·watchdog·30초 초과·IPC overhead가 나온다.

### 3.6.2 timeout 아래의 V8-C fallback (40국면에서 발견)

tree 수가 패배로 증명된 뒤 자식 검사가 deadline에 걸리면, 동결 규칙대로 "첫 UNKNOWN 자식"을 둔다.
timeout 아래의 UNKNOWN은 대부분 "검사하지 못함"이라, 증명된 패배 수를 둘 수 있다
(`docs/rt30-ser5-40pos-results.md` §5.1의 `14/15`). V8-B 승리 손실과 함께 보정해야 할 항목이다.

후보 대책:

- tree 수가 UNSAFE로 판정되면 자식 검사에 남은 시간을 우선 배정한다(VCT2 reserve를 양보).
- timeout fallback에서는 V8-C가 SAFE로 증명한 자식, 아니면 VCT2/`_not_immediately_lost`를 통과한 자식을 우선한다.

둘 다 동작이 바뀌므로 gate가 필요하다.

### 3.8 V8-B 승리 손실: 벤치마크와 후보 순서·스케줄러 실험 (`scripts/run_v8b_bench.py`)

V8-B는 tree보다 먼저 돌고 난수와 policy를 쓰지 않는다. 그래서 E3-A 기록이 그대로 TB1 V8-B의 정답이다.

- **positive 67개:** 기록상 own_vct. TB1이 VCT1 승리를 증명했고 승리 수가 기록에 있다.
- **hard negative 202개:** V8-B가 5초 넘게 돌았지만 승리가 없던 수.

**관찰 (컨테이너, fast windows, TB1 V8-B 예산).**

| 항목 | 결과 |
|---|---|
| 기록상 승리 수의 후보 순위 | 1위는 15/67, 중앙값 6위(후보 10~15개) |
| 열린 3을 만드는 수 우선(`threes_first`) | 1위 33/67, 3위 안 49/67, 중앙값 2위 |
| **상한**: 승리 수를 처음부터 검사할 때 증명 시간 | 1초 23/67, 2초 37/67, **4초 44/67, 8초 56/67**, 12초 60/67 |

**cap별 recall (positive 67개).**

| 순서 / 스케줄러 | cap 4초 | cap 8초 |
|---|---|---|
| frozen / fair (현재) | 34 | 43 |
| frozen / widen | 37 | 42 |
| frozen / widen3 | 32 | 37 |
| threes_first / fair | 31 | 41 |
| threes_first / widen | 36 | 42 |
| threes_first / widen3 | 35 | 42 |

- `widen`은 상위 1, 2, 4, …개에 남은 **시간**의 절반씩을 주는 진행형 확장이다.
  node slice로 하면 cap보다 길어서 첫 단계가 cap을 다 쓴다.
- `widen3`은 상위 1개에 40%, 상위 3개에 남은 시간의 60%, 나머지에 남은 시간을 준다.

**결론.**

- 싼 순서·스케줄러 변경은 recall을 0~+3/67밖에 바꾸지 못했다(잡음 수준). 상한과의 차이는 4초에서 약 10개, 8초에서 약 13개로
  그대로 남는다. 승리 수가 앞에 와도, 그보다 앞선 후보와 상위 단계가 시간을 먼저 쓰기 때문이다.
- **cap이 실제 지렛대다**(4초 → 8초: 34 → 43). 40국면에서 놓친 3개 승리는 모든 변형에서 8초로도 찾지 못했다.
- 변형은 opt-in으로만 둔다(`RT30Agent(attack_order=..., attack_scheduler=...)`). 기본값은 V8 그대로다.
- hard negative에서는 cap 아래라면 어떤 변형이든 시간이 ≈ cap이다(찾을 게 없으면 cap까지 돈다). 그래서 비교 지표는 cap 안의 recall이다.

**다음 지렛대(우선순서).**

1. **정확 고속화 2차.** fast windows를 적용한 뒤에도 V8-B 시간의 39%가 `threat_windows`(numpy 호출 overhead),
   25%가 `_fast_winning_extensions_in_direction`이다. 속도가 1.5배 오르면 cap이 한 단계 오르는 효과가 있고, 기력 변화도 없다.
2. **V8-B cap 재배분.** RT 전체(40국면)에서 cap 4/8/12초를 비교해, 놓친 승리 수와 V8-C·VCT2 timeout 증가를 함께 본다.
   cap은 뒤 단계의 시간을 가져온다.
3. **policy 순서.** H3 policy로 후보를 정렬한다(체크포인트 필요). 열린 3 규칙보다 승리 수를 더 앞에 둘 가능성이 크다.
4. **V8-C timeout fallback 보정**(§3.6.2).

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
| 6 | RT 엔진: route별 soft cap, PUCT guard, 비상수, 기존 fallback | **완료** (`rt30/engine.py`, 테스트). 시간 값은 미니PC 측정 후 조정 |
| 7 | worker supervisor, 29초 watchdog, 대기열·절대 deadline | **완료** (`rt30/supervisor.py`, 웹 `--rt30`, replay `--arms sup`). 미니PC 측정 대기 |
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

### 5.2 고정 평가셋과 H3 baseline (2026-10-14)

**왜 다시 만드는가.** `--select slow --limit 40`은 파일 순서대로 앞의 40개를 고른다. 그래서 seed 8415 한 파일의 13판에서
연속된 수만 뽑혔다. 앞으로 비교는 저장소에 동결한 key manifest로만 한다. 비교할 때마다 다시 뽑지 않는다.

| manifest | 내용 | 용도 |
|---|---|---|
| `docs/rt30-eval/slow-stratified-v1.json` | 기록상 30초 초과 수에서 파일(seed)당 30개, 판당 최대 2개, `Random(42)`. 실제 결과는 60판에서 1개씩, route는 tree 47·own_vct 7·stage4 5·stage5 1 | 일반 성능·기력 gate |
| `docs/rt30-eval/critical-regressions-v1.json` | 이미 발견한 실패 6국면(`9/26`, `10/20`, `15/32` 승리 손실, `14/15` 지는 수, `17/23`, `22/37` tree 수 유지)과 각 기대 동작 | 회귀 gate. 통계에 섞지 않는다 |

- 두 셋은 `10/20` 하나가 겹친다. 무작위 추출에서 우연히 겹친 것이고 일부러 넣은 것이 아니다.
- `scripts/build_rt30_eval_set.py`는 같은 인자로 같은 key를 만든다. 테스트가 커밋된 manifest와 다시 만든 결과가 같은지 확인한다.
  이미 있는 manifest와 key가 다르면 덮어쓰지 않는다(새 version으로 만든다).
- replay는 `--keys-file`로 manifest의 국면을 manifest 순서대로 돈다. 출력 `manifest`에 path와 version이 남는다.

**수정 전 baseline (현재 코드, H3, SER5).** fast는 deadline이 없어 수가 시간에 의존하지 않으므로 workers 4로 돌려도 된다.
rt와 sup은 시간을 재므로 workers 1로 돌린다. frozen = fast = rt(deadline 없음)는 H3 8국면에서 이미 확인했으므로 frozen은 생략한다.

```bash
CK=runs/h3_policy_64x4/best.pt; SET=docs/rt30-eval/slow-stratified-v1.json
python scripts/run_rt30_replay.py --checkpoint $CK --arms fast --keys-file $SET --workers 4 \
  --output runs/rt30/fast_h3_s60.json 2>&1 | tee runs/rt30/fast_h3_s60.log
python scripts/run_rt30_replay.py --checkpoint $CK --arms rt --deadline 27.5 --keys-file $SET --workers 1 \
  --baseline runs/rt30/fast_h3_s60.json --output runs/rt30/rt27_h3_s60.json 2>&1 | tee runs/rt30/rt27_h3_s60.log
python scripts/run_rt30_replay.py --checkpoint $CK --arms sup --deadline 27.5 --keys-file $SET \
  --baseline runs/rt30/fast_h3_s60.json --output runs/rt30/sup27_h3_s60.json 2>&1 | tee runs/rt30/sup27_h3_s60.log
# 회귀셋도 같은 방식으로: SET=docs/rt30-eval/critical-regressions-v1.json, 출력 이름 *_h3_crit
```

**이후 변경의 gate 정의.**

- **정확 고속화.**
  - correctness: frozen = fast. 수·diagnostics·nodes·calls가 같아야 한다.
  - RT 영향: 고정셋에서 승리 손실·지는 수·timeout·지연 시간을 다시 잰다.
  - 같은 4초 안에 더 많이 탐색하므로, deadline이 있는 RT에서는 착수가 달라질 수 있다. "기력 변화 없음"은 deadline이 없을 때만 맞다.
- **V8-C rescue**(다음 코드 단계, §3.6.2).
  - tree 수가 UNSAFE로 증명되면 VCT2를 포기한다(`skipped += vct2_root_rescue`).
  - V8-C는 VCT2 reserve를 뺀 deadline 대신 `global − final_margin`까지 **기존 VCT1 solver로** SAFE를 계속 찾는다.
  - 못 찾으면 증명된 UNSAFE는 제외하고, UNKNOWN 중 `_not_immediately_lost`를 통과한 수를 둔다.
  - 새 휴리스틱(VCF만 확인하는 "빠른 안전 검사")은 넣지 않는다. VCF가 없다는 것은 VCT1 SAFE가 아니다.
  - 구현하려면 normal/rescue 두 deadline을 V8-C에 함께 넘겨야 한다.
  - gate:
    - 바뀐 수(RT ≠ fast)를 full-budget VCT1로 다시 검사한 결과 `changed_move_proven_unsafe = 0`
      (baseline이 검사하지 않은 수도 잡기 위해 `played_baseline_unsafe`보다 강한 기준을 쓴다)
    - 회귀셋 `14/15`에서 증명된 UNSAFE 수를 두지 않을 것
    - 불변식 테스트: RT가 스스로 UNSAFE로 증명한 수를 두지 않을 것
    - wall 최대 26초 이하
- **V8-B cap.** V8-C rescue 이후에 4/6/8초를 전체 RT로 비교한다.
  지표: lost_own_vct, changed_move_proven_unsafe, root·vct2 timeout, vct2 skip, wall max, 30초 초과.
- **동시성.** EngineSupervisor 큐의 동시 요청 테스트는 지금 할 수 있다.
  실제 다중 사용자 HTTP 동시성은 멀티 세션을 구현한 뒤에 한다.

### 5.1 컨테이너 확인 결과 (2026-10-14, 4 vCPU, H3 체크포인트 없음 → `--uniform-prior`)

체크포인트가 없어 tree의 PUCT prior만 uniform이다. 전술 모듈(V8-A/B/C, VCT2)과 흐름은 TB1과 같다.
미니PC에서 `--checkpoint`로 다시 확인해야 한다.

| run | 결과 |
|---|---|
| frozen vs fast, 무작위 24국면 | **24/24 동일**, 전체 1.36배(쉬운 국면은 PUCT 비중이 큼) |
| frozen vs fast, 기록상 30초 초과 8국면 | **8/8 동일**, 전체 **2.20배**(V8-B 2.1~2.6배, V8-C 2.3~2.5배, VCT2 1.8~2.2배) |
| frozen vs fast vs rt(deadline 없음), 무작위 40국면 | **40/40 동일**, fast 2.05배, rt 2.05배 |
| rt `--deadline 27.5`, 기록상 30초 초과 40국면(4 workers) | **최대 26.02초, 27.5초 초과 0**, 중앙값 21.0초. timeout 있는 수 18/40(V8-B 13, V8-C 8, VCT2 7) |

`--deadline` run에서 확인한 점:

- **V8-B 4초 cap은 승리를 놓친다.** 이 40국면 중 기록상 own_vct(VCT1 승리 증명)였던 3국면이 모두 cap에 걸려 tree route로 갔다.
  fast 기준으로도 증명에 24~32초가 드는 국면이다. 첫 번째 보정 항목이다(cap 상향, 또는 남은 시간에 따라 V8-B 몫 조정).
  SER5 smoke(`docs/rt30-ser5-smoke-results.md` §4.1)에서도 8국면 중 2개 승리가 끊겼고, `8415/10/20`은 실제로 승리가 아닌 수를 뒀다.
  그 승리 한 수의 증명은 7초인데 V8-B 전체는 32~43초다. 공정 배분이 예산을 후보 전체에 나누기 때문이라,
  cap 상향과 함께 후보 순서와 배분 방식도 본다.
- **deadline이 닿지 않는 V7 모듈의 실제 몫은 최대 2.3초였다.** `v7_module_seconds`는 own VCF 직전부터 V7 root 준비가
  끝날 때까지 재는 타이머라 **V8-B 시간이 안에 들어 있다.** 실제 V7 몫은 `v7_modules − attack`이다(replay 출력의 `v7_only`).
  이 문서의 첫 판에 쓴 "최대 4.7초"는 V8-B가 섞인 값이라 틀렸다. 미니PC 포화 상태에서 다시 측정해야 한다.
- 시간이 남은 수가 끝에서 26.0초 근처에 몰린다. 마지막 단계(VCT2)가 deadline − final_margin까지 쓰기 때문이다(정상 동작).

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

## 7. 웹대국 로그 DB (SQLite, `src/weblog/store.py`)

**구조.** 다른 PC는 DB 파일을 직접 열지 않고 미니PC의 API로 조회한다(API는 다음 단계). raw 로그
(`logs/web_play/<game>/game.json`, `moves.csv`, `veto_positions/*.json`)는 형식 그대로 대국 종료 시 쓴다.
DB는 **착수마다** 기록하므로 서버가 죽어도 진행 중인 대국이 남는다. `games.raw_log_dir`로 둘을 연결한다.

**쓰기 경로.**

- 쓰는 쪽은 웹 서버 프로세스 하나다. engine worker는 DB를 건드리지 않고 결과를 서버에 넘긴다.
- 이벤트 메서드(`start_game`, `record_move`, `mark_undone`, `record_veto`, `finish_game`)는 queue에 넣기만 한다.
  전용 writer thread가 순서대로 반영한다.
- 착수 응답이 디스크를 기다리지 않는다. DB 오류가 나도 대국은 계속되고, 오류는 `failures`/`last_error`로 집계해
  `/api/health`에 보인다.
- PRAGMA: `journal_mode=WAL`, `synchronous=NORMAL`, `foreign_keys=ON`, `busy_timeout=5000`. 스키마 버전은 `PRAGMA user_version`이다.
  버전이 다른 DB는 열지 않는다.

**스키마(v1).**

| 테이블 | 내용 |
|---|---|
| `games` | `game_id`(서버가 만든 hex), 시작·종료 시각(UTC ISO), `status`(in_progress/finished/unfinished/abandoned), 결과, 엔진·seed·사람 색, `undo_count`, provenance(engine, config hash, commit, git_dirty, policy·metadata sha256), `agent_info_json`, `final_board_json`, `raw_log_dir` |
| `moves` | `seq`(undo된 수 포함 기록 순서), `ply`, player, actor, `row0/col0`(0 기준), `board_hash`, `engine_seconds`, `queue_seconds`, **`wall_seconds`(요청 수신 → 수 준비, 30초 판정 기준)**, `deadline_seconds`, `route`, `timed_out`, `emergency_used`, `diagnostics_json`, `rt_json`, `undone_at` |
| `veto_positions` | `(game_id, ply)`, `reason`(VCT2_PROVEN_LOSS), tree 수, 둔 수, raw 파일 내용 전체(`payload_json`), `raw_path` |

- **undo.** 행을 지우지 않고 `undone_at`을 표시한다. 살아 있는 수끼리만 `(game_id, ply)`가 unique다(partial index).
- **엔진별 세부 정보.** 컬럼으로 늘리지 않고 JSON에서 읽는다. 예: `json_extract(diagnostics_json, '$.v8_root_nodes')`.
- **대국 상태 전환.**
  - 끝까지 둔 대국: `finished`
  - 동결 엔진(s3/tb1)에서 AI 수가 있는데 새 대국으로 넘어간 경우: `unfinished`(기존 규칙과 같다)
  - 그 밖에 새 대국으로 넘어간 경우: `abandoned`
  - 서버가 종료되거나 죽은 경우: `in_progress`로 남는다

**실행.** `--db`가 없으면 기존 동작과 같다.

```bash
python scripts/run_web_play.py --db data/renju_web.db --policy-checkpoint runs/h3_policy_64x4/best.pt
```

**백업.** WAL 모드에서는 `.db` 파일만 복사하면 안 된다. 아직 반영되지 않은 데이터가 `-wal` 파일에 있다.

```bash
sqlite3 data/renju_web.db ".backup data/backup/renju_web_$(date +%F).db"
```

**다음 단계.**

1. 기존 `logs/web_play/*/game.json`을 DB로 옮기는 import 스크립트
2. 인증이 붙은 읽기 전용 로그 API: `/api/logs/games`, `/api/logs/games/{id}/moves`, `export.csv`, latency 통계.
   읽기는 `open_readonly`(`mode=ro`)로 한다.
3. RT30 supervisor가 `deadline_seconds`, `timed_out`, `emergency_used`, `rt_json`을 채운다.

외부에 공개하기 전에 알아둘 점이 있다. 현재 대국 API는 서버 전체에 게임이 하나뿐(`SESSION`)이고 인증이 없다.
멀티 세션으로 바꾸기 전까지는 대국 화면도 인증(Nginx basic auth 등) 뒤에 둔다.
