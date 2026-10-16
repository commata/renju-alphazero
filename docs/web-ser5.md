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

### 3.6.3 V8-C root rescue 구현과 측정 (2026-10-14)

**구현** (`rt30.engine._verify_root_choice(..., rescue_deadline=)`, `RT30Agent(root_rescue=True)` 기본):

- tree 수가 V8-C에서 **UNSAFE로 증명**되면 자식 검사의 deadline을 `deadline − vct2_reserve − final_margin`에서
  `deadline − final_margin`으로 늘린다.
- VCT2는 그 수에서 건너뛴다(`skipped += vct2_root_rescue`, `RTInfo.rescued`).
- 검사와 status는 V8-C 그대로다. 새 휴리스틱은 없다. 잘린 자식은 UNKNOWN이고 fallback 규칙도 동결 그대로다.
- deadline이 없으면 rescue는 꺼진다. 그래서 `deadline=None`은 계속 TB1과 같다(기존 동일성 테스트 통과).
  `--no-root-rescue`로 이전 동작을 재현할 수 있다.

**불변식 테스트**(`tests/test_rt30_engine.py::RootRescueTest`, scripted solver):

- 자식 4개의 status 81가지 조합 × cut 지점 6가지를 돌렸다.
  RT가 스스로 UNSAFE로 증명한 수를 두는 것은, 고를 수 있는 자식이 모두 UNSAFE일 때뿐이다.
- 버그 경로도 재현했다. rescue가 없으면 첫 미검사 자식을 두고, rescue가 있으면 증명된 SAFE 자식을 둔다.

**실제 국면 `14/15`(uniform).**

- 시간 제한 없이 rescue로 끝까지 검사하면(CheckLimit) fast와 같다. `[5,6]`, `[5,9]` UNSAFE, `[5,4]` SAFE로 `[5,4]`를 둔다.
- 하지만 **27.5초 안에서는 부족하다.**
  - 컨테이너 RT에서 rescue가 걸렸지만 V8-C가 21초(94k 노드) 뒤 다시 잘렸고, 여전히 `[5,6]`을 뒀다.
  - fast(SER5)는 이 국면 V8-C에 51초, 221k 노드를 썼다.
  - 자식 하나의 증명에도 각각 50~65k 노드(컨테이너 10~14초)가 든다. `[5,4]` SAFE가 65k, `[5,6]` UNSAFE가 65k, `[5,9]` UNSAFE가 50k다.
  - 그래서 순서를 바꿔도 창 안에 증명 1~2개밖에 들어가지 않는다.
- 결론: rescue는 **안전 불변식과 시간 재배정을 위한 수정**이다. 이 국면의 실패를 직접 풀려면 속도(2차 정확 고속화)가 필요하다.
- uniform critical 결과(컨테이너, workers 2): 최대 26.016초, rescue 1회(`14/15`). 나머지 5국면은 이전과 같은 분류다.

### 3.9 2차 정확 고속화 (`rt30/fast.py`, `fast_windows.threat_windows_codes`, `fast_extensions.py`)

**출발점.** 1차 고속화 뒤의 profile(V8-B `8415/10/20`, V8-C `14/15`의 `[5,4]` SAFE 증명)이 두 단계 모두 같았다.

- `threat_windows`가 38~39%였다. 대부분 numpy 호출 overhead다.
- `_fast_winning_extensions_in_direction`(4 연장, `_run_info`·`inside` 포함)이 25~26%였다.

**변경(둘 다 결과가 같다).**

- **windows:** 각 5칸 window를 base-3 코드(빈칸 0, 흑 1, 백 2)로 바꾸고, (player, minimum)마다 243칸 표를 한 번 조회한다.
  호출당 41μs에서 20μs가 됐다. 순서까지 같은 list를 돌려준다(기존 테스트에 이 구현을 추가했다).
- **4 연장:** anchor를 지나는 player 돌 연속(run)의 양 끝 칸만 후보가 될 수 있다. 더 먼 빈칸은 그 사이에 끝 칸(빈칸)이 끼어
  run이 끊긴다. 그래서 8칸에 돌을 두고 매번 run을 재던 것을, 양쪽 스캔 한 번으로 바꿨다.
  흑은 정확히 5, 백은 5 이상이다. anchor가 player 돌이 아니면 빈 집합이다. 함수 단독으로 3.9배다.
  - 검증: 무작위 국면 250개 × 225칸 × 4방향 × 2색, 손으로 만든 줄(4, 끊긴 4, 장목, 막힌 끝, 판 가장자리, 전치)에서 같은 집합이다. 판 상태도 바뀌지 않는다.
- `rt30.fast.install_all()`이 모든 정확 고속화를 한 번에 설치한다. replay `fast` arm, supervisor worker, 재검사·벤치 스크립트가 쓴다.
  `fast_windows.install()`은 이전 버전에서 다른 버전으로도 바꿔 끼운다.

**전체 효과(컨테이너, 1차 대비, 결과 동일).**

| 측정 | 1차 | 1차 + 2차 | 배율 |
|---|---|---|---|
| V8-C `14/15` `[5,4]` SAFE 증명(65k 노드) | 12.1초 | 7.8초 | 1.55배 |
| V8-B `9/26`(60k 노드 예산) | 9.5초 | 6.0초 | 1.58배 |

- status·nodes·calls·checked가 모두 같았다.
- frozen 대비로는 약 3.7배다(2.4 × 1.55).
- replay 동일성(uniform, critical 6국면, 컨테이너 workers 4): frozen = fast = rt(deadline 없음) **6/6**, fast는 frozen 대비 **4.12배**다
  (1차만 썼을 때 같은 국면들에서 2.2~2.5배).
- 4 연장만 바꾸면 1.05배에 그쳤다. profile의 비중은 측정 overhead가 부풀린 값이었고, 이득 대부분은 windows에서 나온다.

**gate.**

1. 동일성: frozen = fast = rt(deadline 없음), H3 8국면(SER5).
2. RT 영향: H3 60셋과 회귀셋을 다시 재고(같은 시간에 더 탐색하므로 착수가 달라질 수 있다), V8-B 67 벤치 recall을 본다.

### 3.10 fast2 SER5 검증(`6eb46c7`)과 V8-B cap 실험 설계 (2026-10-15)

**검증 결과.**

- 동일성: H3 8국면 frozen = fast = rt **8/8**, fast는 frozen 대비 3.94배(같은 실행 안의 비교, workers 4).
- H3 60, cap 4초: median 14.14초에서 9.06초, 최대 26.004초.
  - timeout이 난 수가 31에서 27로 줄었다(stage VCT는 2에서 0).
  - 착수 분류는 rescue 때와 같다. same 50, 승리 손실 7(다른 수 6), tree 수 유지 4.
  - 재검사 gate 통과(UNSAFE 0).
- V8-B 67개 벤치: cap 4초에서 36 → **46**, cap 8초에서 48 → **57**.

**cap별 recall은 cap 8초 run 하나에서 바로 구한다.**

- V8-B는 결정적 탐색을 시간으로만 자른 것이다. 그래서 cap 8초 run에서 승리를 증명한 시각을 보면,
  더 작은 cap에서 무엇을 찾았을지 알 수 있다(시간 측정 잡음 범위 안에서).
- fast2 기준 cap별 recall: 4초 46, 5초 48, **6초 51**, 7초 53, **8초 57** (67개 중).
- **H3 60셋의 승리 손실 7개는 fast2 cap 8초 벤치에서 4개가 5.9~7.2초에 증명된다.**
  - 예측: cap 6초면 1개, 7초면 3개, 8초면 4개를 되찾는다. 회귀셋 3개는 8초로도 못 찾는다.
  - 그 대가로 V8-C·VCT2 시간이 줄어든다(V8-B는 tree보다 먼저 돈다).
  - 그래서 결정은 전체 RT run으로 한다.
- **uniform `14/15` + fast2**(컨테이너): rescue는 걸렸지만 여전히 `[5,6]`이다.
  다만 이번에는 자식 3개(`[5,6]`, `[5,9]`, `[5,4]`) 모두 검사가 시작됐다가 미판정으로 끝났다. SER5 결과로 확인한다.

**실험(SER5, H3).** 셋은 `slow-stratified-v1`과 `critical-regressions-v1`, baseline은 `fast_h3_s60` / `fast_h3_crit`다.
`--timing attack_seconds=6`과 `=8`로 rt run과 재검사를 돌린 뒤, `scripts/compare_rt30_runs.py`로 cap 4(fast2 결과)와 나란히 비교한다.

**결정 기준(우선순위 순).**

1. 재검사 UNSAFE 0, 27.5초 초과 0, 최대 26초대.
2. 지는 수(baseline_unsafe) 증가 없음. V8-C·VCT2 timeout 증가는 tree 수 유지 증가와 함께 본다.
3. 그 조건 아래에서 승리 손실(다른 수)을 가장 많이 줄이는 cap.

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
| A1 | V8-B cap 4/6/8초 결정 | **완료: 8초** (§9.1) |
| A2 | TB-RT30-v1 동결(설정 hash + 파일 lock) | **완료** (`rt30/tb_rt30_v1.py`, §9.2) |
| A3 | E3-A 100판 짝 대국 gate | **HARM** (−0.06, §9.4) → A3.1 진단, TB-RT30-v2 설계 |
| A3.1 | 34 분기 수를 게임 상태로 재생: tb1/v1/v2 × workers 4/1 | **완료**: CPU 경쟁 가설 기각, A3b 불필요 (§9.6) |
| v2-A | TB-RT30-v2 후보: V8-B 남은 시간 재시도 (`rt30/engine_v2.py`) | A1 gate 통과, A3 복원 +1.0점 < 기준 2.0 → A3-v2A 생략 (§9.6) |
| R0 | 예측기 정정: v1 시간 배분 / 순차 / X2 / X2+X3, A3 점수 | **완료** (§9.8) |
| X1-A | native threat-window scan (`rt30/fast3*.py`, `rt30/native/`) | **구현**: 컨테이너 2.15~2.31배, 동일 (§9.8) |
| X2 | V8-B 상주 helper 병렬 (`rt30/engine_v3.py`, `attack_helper.py`) | **구현**: 컨테이너 V8-B 6/6 (§9.8) → SER5 v3_check/v3_dev/v3_a1/v3_a3 |
| A3-v3 | SER5 gate + E3-A 100판 (`ed3d8cf`) | **PASS**: −0.01, CI95 [−0.025, 0.0] (§9.10). `rt30_no_deadline` 1건은 기준 기록의 환경 차이(`76cce57`) |
| A4 | TB-RT30-v3 동결 (`tb_rt30_v3.py` `STATUS='frozen'`, `tests/frozen_tb_rt30_v3.sha256`) | **완료** (§9.10) |
| X1-B | 금수 판정 가속 | 필요할 때만: A3-v3가 시간 부족으로 HARM일 때 v4 후보 (§9.9) |
| W | 웹 B1/B2(멀티 세션, AI job queue, search slot) | **v3·native의 production 연결보다 먼저** (§8, §9.9) |
| V3-web | v3를 웹 search slot(main + V8-B helper 형제 프로세스)으로 연결 | W 완료 + v3 동결 뒤 (§9.9) |
| T1 (v3T) | V8-C 시간 분할 50% | fallback: A3-v3가 HARM일 때만 |

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

### 5.3 H3 60국면 baseline 검토 (`7eba94c`, 2026-10-14)

결과 파일: `docs/rt30-results/{fast,rt27,sup27}_h3_s60.json`, `recheck_rt27_h3_s60.json`.
조건은 실제 H3, `slow-stratified-v1`, 코드 `1215558`이다. fast는 workers 4라 시간을 rt와 직접 비교하지 않는다.

**시간과 SLA.**

- RT와 sup 모두 최대 26.003초, 27.5초 초과 0이다.
- sup은 source가 60/60 모두 engine이고, watchdog 0, IPC 최대 1ms다.
- **rt와 sup의 착수는 60/60 같다.** worker 경로가 결과를 바꾸지 않는다.

**승리 손실 7/7(다른 수 6).** 그런데 cap 4 → 6/8초로는 이 손실을 되찾을 수 없다.

- RT의 V8-B는 fast와 같은 결정적 탐색을 시간으로만 자른 것이다.
- fast(workers 4)가 이 7개 승리를 증명하는 데 10.1~22.3초가 걸렸다(desktop 기록은 32~68초).
- SER5 V8-B 벤치마크(workers 1)에서도 이 7개는 cap 8초까지 **하나도** 찾지 못했다.
- 이 셋은 "30초 넘은 수"에서 뽑았기 때문에 **어려운 V8-B 승리에 몰려 있다.**
  cap 효과는 전체 기록 승리 67개 벤치(4초 36 → 8초 48)로 보고, 이 셋은 부작용(V8-C·VCT2 timeout)을 보는 데 쓴다.
- 기록 전체로 환산하면 E3-A 100판에 V8-B 승리가 67개 있었고, cap 4초면 그중 31개, 8초면 19개를 증명하지 못한다(SER5 벤치 기준).

**V8-C.** 이 60셋에서 tree 수가 UNSAFE로 증명된 국면(rescue가 동작할 상황)은 1개(`8415/38/11`)뿐이다.
나머지 4개 tree 수 유지는 다른 메커니즘이다. tree 수가 UNKNOWN일 때 aggressive 교체(증명된 SAFE 자식으로 교체)가 시간에 잘린 경우다.

**바뀐 5수의 full-budget 재검사** (`check_rt30_changed_moves.py`, VCT1 800k / VCT2 200k 노드).

| 국면 | RT 수 | VCT1 | 노드 | fast 수 VCT1 노드 |
|---|---|---|---|---|
| `8415/38/11` | `[9, 3]` | SAFE | 53,546 | 896 |
| `8415/46/49` | `[12, 3]` | UNKNOWN | 800,000 | 5,699 |
| `8416/5/10` | `[9, 5]` | SAFE | 415,647 | 197 |
| `8416/7/25` | `[8, 3]` | SAFE | 763,052 | 359 |
| `8416/48/83` | `[9, 11]` | SAFE | 296,793 | 12,905 |

- `changed_move_proven_unsafe = 0`이다. gate는 통과했다. 다만 `46/49`는 800k 노드로도 판정되지 않았다.
- 패턴이 분명하다. fast는 **싸게 증명되는 안전한 수**로 바꾸고(수백~수천 노드), RT가 유지한 tree 수는 안전하더라도
  증명에 수십만 노드가 든다. H1(§12.9)의 관찰과 같다.
- 회귀셋의 `14/15` `[5, 6]`(uniform)은 다시 검사해도 **VCT1 UNSAFE(65k 노드에 증명)**다.

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

## 8. B1/B2 설계: 멀티 세션 + polling (A3 이후 구현, 2026-10-15 확정 방향)

**기본 구조(제안 채택).**

- 세션마다 `PlaySession` 하나, 엔진은 전역 하나다.
  - `SessionManager`가 쿠키 `renju_sid`로 세션을 찾는다. 세션 ID는 `secrets.token_urlsafe(32)`이고,
    쿠키 속성은 `HttpOnly`, `SameSite=Lax`, `Path=/`, HTTPS에서는 `Secure`다.
  - 전역 `AIJobQueue` 하나와 dispatcher가 `EngineSupervisor`를 쓴다.
- 착수는 `202`로 즉시 응답하고, 클라이언트가 `/api/state`를 polling한다.
  세션 상태는 `READY / QUEUED / THINKING / FINISHED`이고, 오류는 `last_error`에 담는다.
- `QUEUED`·`THINKING` 중에는 move·undo·new가 `409`다. 서버가 막고, 클라이언트도 보드를 잠근다.
  대기열이 차면 `503` + `Retry-After`다. 세션 정리는 lazy cleanup이고, 진행 중이던 게임은 DB에 `abandoned`로 남긴다.
- 파일 구조: `src/webplay/sessions.py`, `src/webplay/jobs.py`. `PlaySession`은 우선 현재 파일에 둔다.

**수정·보완 사항.**

1. **엔진 탐색은 세션 lock 밖에서 한다(필수).**
   - 지금 `play_human()`은 lock을 잡은 채 `_play_ai_locked()`에서 탐색까지 끝낸다. 그런데 `state()`도 같은 lock을 잡는다.
   - 그래서 lock을 쥔 채 탐색하면 그 세션의 polling이 최대 29초 막힌다.
   - 바꿀 흐름:
     - `play_human()`은 사람 수를 반영하고, 착수 기록(history)의 사본과 `epoch`(게임 ID와 수 번호)를 job에 담아 `QUEUED`로 바꾼 뒤 lock을 놓는다.
     - dispatcher가 lock 없이 탐색한다.
     - 끝나면 lock을 잡고 epoch가 그대로인지 확인한 뒤 AI 수를 반영한다. epoch가 다르면 결과를 버린다.
   - 사람이 백일 때 `/api/new` 직후 두는 AI 첫 수도 같은 경로로 처리한다.
2. **대기열 자리는 사람 수를 반영하기 전에 확보한다.** 대기열이 차서 `503`을 줄 때 사람 수가 이미 반영돼 있으면 안 된다.
   세션당 대기 중인 job은 최대 1개다(409 규칙과 같은 조건).
3. **deadline 정책을 명시한다.** 엔진 하나로는 동시 요청 4개 모두에 "요청부터 30초"를 보장할 수 없다. 그래서 둘로 나눈다.
   - 엔진 deadline: **탐색 시작 + 27.5초**(watchdog은 시작 + 29초). 탐색 품질은 대기열 길이와 관계없이 일정하다.
     - 구현은 dispatcher가 job을 꺼낼 때 `supervisor.select(received_at=꺼낸 시각)`을 부르는 것이다(인자의 의미가 "탐색 시작"이 된다).
   - 사용자 체감 시간: 대기 + 탐색. 대기열 길이로 상한을 정한다.
     - `MAX_AI_QUEUE = 4`이면 최악 대기는 약 4 × 26초 ≈ 105초 + 탐색이다.
     - 공개 SLA를 "탐색 30초 이내 + 대기 n명 표시"로 할지, 대기열을 줄일지는 B3 측정 뒤에 정한다.
   - DB: `queue_seconds`는 웹 대기열에서, `engine_seconds`와 `deadline_seconds`는 supervisor에서 받는다.
     `wall_seconds`는 사람 착수 요청부터 AI 수 반영까지다.
4. **나간 세션의 job은 탐색하지 않는다.** dispatcher는 job을 꺼낼 때 세션이 살아 있는지, epoch가 맞는지부터 확인한다.
   `QUEUED` 중 new는 job을 취소하고 바로 진행해도 된다. `THINKING` 중에는 409다.
5. **worker의 agent 캐시 크기를 세션 수에 맞춘다.** supervisor worker는 게임별 agent(난수 흐름)를 최대 8개 둔다(`len(agents) >= 8`).
   `MAX_SESSIONS = 16`이면 캐시가 밀려 같은 게임의 agent가 다시 만들어진다(동작은 안전하지만 난수 흐름이 다시 시작된다).
   캐시를 `MAX_SESSIONS` 이상으로 두거나, 세션이 끝날 때 그 game_key를 정리한다.
6. **DB 스키마 v2: `games.session_id`.** v1에는 이 컬럼이 없다.
   원문 세션 ID가 아니라 **hash**(예: SHA-256 앞 16자)를 저장한다. DB나 로그 API로 세션이 노출돼 탈취되는 일을 막기 위해서다.
   `user_version`을 2로 올려 migration한다.
7. **CSRF와 요청 검증.** POST는 `Content-Type: application/json`만 받는다. 쿠키는 `SameSite=Lax`다. 베타 동안에는 Nginx Basic Auth 뒤에 둔다.
8. **동시 worker 수는 설정값으로 둔다.** dispatcher가 thread N개, supervisor가 worker N개를 쓰는 구조로 두되, 처음 값은 N=1이다.
   B3에서 N=2가 탐색 품질(nodes/s)을 얼마나 떨어뜨리는지 보고 정한다.
9. **서버 재시작.** 메모리의 세션은 사라진다. DB에는 `in_progress`로 남는다. 베타에서는 이대로 받아들이고, 재접속 시 복원은 나중에 한다.
10. **테스트.**
    - fake 엔진(`tests/rt30_fake_agents.py` 방식)으로 확인할 것: 세션 격리, 202 → polling → READY, 409/503, epoch 불일치 시 결과 버림,
      대기 중 취소, polling이 탐색 중에도 막히지 않는 것, TTL 정리.
    - 그 뒤 B3의 HTTP 부하 클라이언트(세션 1/2/4개)로 측정한다.

**구현 순서:** SessionManager와 API 세션 분리 → AIJobQueue와 dispatcher(탐색은 lock 밖, epoch 확인) → `/api/state` polling 필드 →
UI(생각 중·대기 표시, 보드 잠금) → DB v2(`session_id` hash) → 테스트 → B3 부하 측정.

## 9. 엔진 확정: A1 cap 결정, A2 TB-RT30-v1 동결, A3 대국 gate (2026-10-15)

### 9.1 A1: V8-B cap 8초 채택 (`1ec0435`)

| cap | median | max | >27.5 | timeout 수 | same | 승리 손실(다른 수) | tree 수 유지 | 재검사 UNSAFE | phase timeout (attack/stage/root/vct2) |
|---|---|---|---|---|---|---|---|---|---|
| 4초 | 9.06 | 26.004 | 0 | 27 | 50 | 7 (6) | 4 | 0 | 17/0/11/10 |
| 6초 | 9.48 | 26.004 | 0 | 25 | 51 | 6 (5) | 4 | 0 | 15/0/11/10 |
| **8초** | 9.80 | 26.004 | 0 | **21** | **53** | **3 (3)** | 4 | 0 | 11/0/10/9 |

H3 `slow-stratified-v1` 60국면, baseline `fast_h3_s60`. 결정 기준(§3.10) 순서대로 본다.

1. **안전·시간**: 세 cap 모두 재검사 UNSAFE 0, 27.5초 초과 0, 최대 26.004초다.
2. **지는 수 증가 없음**: baseline_unsafe는 세 cap 모두 0이다.
   - 예상했던 대가(V8-B가 길어지면 V8-C·VCT2 시간이 준다)는 나타나지 않았다.
   - root·VCT2 timeout은 11/10 → 10/9로 오히려 줄었다. V8-B가 이기는 수를 찾으면 뒤 단계가 아예 돌지 않기 때문이다.
3. **승리 손실**: 7 → 3. 그래서 **cap 8초**다. 초안의 결론과 같다.

**초안에 보탠 확인.**

- cap 4초 → 8초에서 착수가 바뀐 국면은 **3개뿐**이고, 모두 승리를 되찾은 경우다(4/16, 26/15, 34/15).
  37/17은 착수는 같고 증명만 되찾았다. 나머지 56국면은 수가 같다.
- 이 4개는 §3.10의 벤치 예측과 맞는다. "fast2 cap 8 벤치에서 5.9~7.2초에 증명되는 4개를 되찾는다"는 예측이었다.
- critical 6국면은 세 cap에서 결과가 같다. 회귀 3개는 8초로도 못 찾는다(§3.10 예측대로). median만 9.0 → 11.0초로 늘었다.
- V8-B 67개 벤치(fast2): cap 4초 46, 6초 51, 8초 57.

**남은 비용(베타에서 볼 것).**

- V8-B가 cap까지 돌고도 승리를 못 찾는 국면에서는, 착수가 최대 4초 늦어진다.
  60국면 합계는 740.9초 → 760.0초로 약 +3%다.
- 실제 대국에서 이런 국면이 얼마나 자주 나오는지는 A3 대국 기록과 E2 베타 로그(`route`, `timeouts.attack`)로 확인한다.
- uniform `14/15`는 fast2 + rescue로도 여전히 `[5,6]`이다(root 22.4초, 160k 노드). A4 연구 항목이다.

### 9.2 A2: TB-RT30-v1 동결

- **정의**: `src/rt30/tb_rt30_v1.py`.
  - TB-Engine-v1 설정 전체를 그대로 적어 넣었다(다른 곳의 기본값이 바뀌어도 이 엔진은 바뀌지 않는다).
  - 그 위에 deadline 27.5초, watchdog 29초, timing(`final_margin` 1.5, `attack_seconds` **8.0**,
    `puct_reserve` 6, `vct2_reserve` 4, `vct2_min` 2), attack order `frozen`, scheduler `fair`, root rescue on,
    정확 고속화 2종을 둔다.
- **설정 hash**: `4bea6459d55eef979894de924f79fd08c6a4bdbb317dbcd2f84cc9aac7a266cc`.
  웹 서버는 대국마다 이 hash를 DB(`games.engine_config_hash`)와 `/api/state`에 남긴다.
- **파일 lock**: `tests/frozen_tb_rt30_v1.sha256`가 착수를 정하는 파일 8개를 잠근다.
  - 대상: `engine.py`, `solver.py`, `deadline.py`, `vcf.py`, `fast.py`, `fast_windows.py`, `fast_extensions.py`, `tb_rt30_v1.py`.
  - CI(`frozen-baseline` job)가 검사한다. 하나라도 바꾸면 TB-RT30-v2다(CLAUDE.md).
- **`tests/test_tb_rt30_v1.py`**가 확인하는 것:
  - 기반 설정 == `TB_ENGINE_V1`, 고정된 hash.
  - timing·deadline·rescue 값이 A1 cap 8초 run 파일에 기록된 값과 같다.
  - A1 gate 수치(UNSAFE 0, 27.5초 초과 0)가 기록과 같다.
  - agent 속성, 고속화 설치, supervisor factory.
- **웹**: `run_web_play.py --rt30`은 이제 `make_tb_rt30_v1_agent`로 이 엔진을 띄운다.
  - `--rt30-deadline`·`--rt30-watchdog`은 선택 override다.
  - 쓰면 엔진 이름에 ` (deadline override)`가 붙어 기록이 섞이지 않는다.

### 9.3 A3: E3-A 100판 짝 대국 gate (`scripts/run_rt30_match.py`)

**설계.** TB-Engine-v1을 채택한 E3-A run과 같은 대국을 TB-RT30-v1로 다시 둔다.

- 같은 것: 오프닝 50쌍(seed 8415·8416, 각 25쌍, 색 교환), 판별 seed, 상대 `v8:full+vct2atk`.
- E3-A 기록: 76승 10무 14패, score 0.81 (Wilson 0.722~0.875).
- 두 엔진 모두 결정적이다. 그리고 deadline 사건이 없으면 RT30 = TB-Engine-v1이다.
  - 그래서 각 판은 **deadline이 바꾼 첫 RT30 수까지 E3-A 기록과 똑같이** 진행돼야 한다.
  - `--reference`로 판마다 기록과 짝지어 갈라진 지점을 분류한다.
    - `same`: 같은 판.
    - `deadline`: timeout·skip이 있던 RT30 수에서 갈라졌다. 예상되는 경우다.
    - `rt30_no_deadline`, `opponent`: deadline 사건 없이 갈라졌다. 원래 없어야 한다.
      policy 평가 차이(Windows 데스크톱 ↔ SER5 torch)나 설정 오류를 뜻한다.
- 컨테이너 smoke(uniform prior, 8415 쌍 0): 상대 수가 기록과 같았고(고속화 설치 상태), 오프닝·seed 검사를 통과했다.
  `tests/test_rt30_match.py`는 100판의 오프닝·seed가 E3-A 기록과 같음을 확인한다.

**gate(실행 전에 고정).**

1. **시간·합법성**: 불법수 0(나오면 스크립트가 멈춘다). RT30 착수 27.5초 초과 0, 30초 이상 0.
2. **재현성**: `rt30_no_deadline` + `opponent` 분기가 0이어야 정상이다.
   - 5판을 넘으면 결과를 읽기 전에 원인부터 본다. 그 판들의 첫 분기 국면을 TB1(`deadline=None`)로 SER5에서 다시 둬 본다.
3. **기력**(E3 판정 규칙과 같다): 짝 차이 = RT30 − E3-A, opening 쌍 bootstrap 95% 구간.
   - **HARM**: 차이 < −0.05, 또는 구간 상한 < 0. 이 경우 TB-RT30-v1로 배포하지 않고 A4(엔진 연구)로 간다.
   - 그 밖에는 **PASS**다. B1 구현으로 넘어간다.
4. **보고(gate 아님)**:
   - 상대 대비 score와 색별 score.
   - RT30 착수 시간 분포(median/p95/p99/max), phase별 timeout, rescue.
   - `result_changed` 목록. 특히 E3-A가 이겼는데 RT30이 진 판의 분기 수는 A4의 입력이다.

**실행(SER5).** 다른 작업 없이, workers 4로 돌린다. 예상 시간은 3~5시간이다.
E3-A 판 길이가 평균 63수이고, 상대 수는 고속화로 약 4배 빨라진다. 진행은 1분마다 출력되고, 중단돼도 같은 명령으로 이어진다.

```bash
cd ~/renju-alphazero && git pull && source .venv-ser5/bin/activate
mkdir -p runs/rt30
python scripts/run_rt30_match.py --checkpoint runs/h3_policy_64x4/best.pt --workers 4 \
    --reference docs/mcts-v8-results/e3/puct_policy_vct2_50k_8415.jsonl \
                docs/mcts-v8-results/e3/puct_policy_vct2_50k_8416.jsonl \
    --games-jsonl runs/rt30/a3_e3.jsonl --output runs/rt30/a3_e3.json 2>&1 | tee runs/rt30/a3_e3.log
```

- 결과 파일(`a3_e3.json`·`.jsonl`·`.log`)은 RenjuNet 파생 데이터가 아니다(자체 대국 기록). 그래서 `docs/rt30-results/`로 옮겨 커밋해도 된다.
- 먼저 짧게 확인하고 싶으면 `--seeds 8415 --pairs 2 --games-jsonl runs/rt30/a3_e3.jsonl`로 4판만 둔다.
  같은 jsonl을 쓰면 본 run이 그 4판을 이어받는다.

### 9.4 A3 결과 (`66a3441`): 시간·재현성 통과, 기력 **HARM**

**gate별 결과.**

| gate | 결과 |
|---|---|
| 불법수 / 27.5초 초과 / 30초 이상 | 0 / 0 / 0 (2,878수, 최대 26.08초) |
| 재현성 | `same` 66, `deadline` 34, `rt30_no_deadline` 0, `opponent` 0 |
| 기력 | 71승 8무 21패, 0.75. E3-A 0.81 대비 **−0.06**, 쌍 bootstrap **[−0.11, −0.015]** |

- 사전에 고정한 규칙(차이 < −0.05 또는 상한 < 0)의 두 조건에 모두 걸린다. 그래서 **TB-RT30-v1은 HARM**이다.
- 이 판정은 뒤의 진단으로 바꾸지 않는다. workers=1에서 다시 돌려 통과하면 그것을 판정으로 삼는 식의 재판정은 하지 않는다.
  배포 조건이 바뀌거나(동시 탐색 수) 엔진이 바뀌면(v2), 그 조건으로 A3를 새로 돌린다.
- 통과선과의 거리: 같은 66판은 차이가 0이라 구간이 좁다. 그래서 1점(−0.05)으로는 부족하고, **약 2~3점을 되찾아야** 두 조건을 함께 넘는다.

**34판 분기의 원인 (`docs/rt30-eval/a3-divergence-v1.json`).**

- 34개의 첫 분기는 모두 **TB1의 전술 override(V8-B 증명된 WIN, V8-C SAFE 자식 선택)가 RT30의 시간 제한 때문에 재현되지 않은 수**다. 6개는 tree보다 앞에서 도는 V8-B의 WIN 누락이고, 28개는 V8-C의 SAFE 자식 선택 누락이다. 평범한 PUCT 수에서 deadline 사건 없이 갈라진 판은 0이다.
- 원인은 두 가지뿐이다.

| 원인 | 판 | 결과 바뀐 판 | 점수 |
|---|---|---|---|
| **V8-B cap**: TB1은 V8-B로 VCT 승리를 찾았는데, RT30은 8초 cap에서 잘렸다 | 6 | 3 (모두 흑: 승→패 2, 승→무 1) | **−2.5** |
| **V8-C timeout**: TB1은 tree 수가 미판정(UNKNOWN)이라 아래 자식 중 SAFE로 증명된 수로 바꿨다(aggressive 규칙). RT30은 tree 수 검사 중에 시간이 다 돼 tree 수를 유지했다 | 28 | 10 (손실 7, 이득 3) | **−3.5** |

- 색별 차이도 이것으로 설명된다. 흑 −0.09(E3-A 0.77 → 0.68)는 V8-B 손실 −2.5가 모두 흑이고 V8-C −2.0이다. 백은 −0.03이다.
- V8-C 28판 중 1판(8415/2 흑)은 tree 수가 UNSAFE라 rescue가 돌았지만, 시간 안에 SAFE 자식을 못 찾았다. 결과는 패 → 패다.
- 분기 전 구간의 root timeout 66수를 TB1 기록과 대조했다.
  - TB1도 tree 수를 유지한 경우: 늦게 SAFE 7, UNKNOWN인데 SAFE 자식 없음 28. RT30도 tree 수를 유지하므로 같은 수다.
  - TB1이 바꾼 경우 27: 모두 분기했다.
  - 즉 지금의 "timeout이면 tree 수 유지"는 TB1이 tree 수를 유지할 때만 맞다.

**TB1이 그 수에서 실제로 쓰는 시간 (컨테이너 측정, 고속화 설치, deadline 없음, 4 병렬).**

- 컨테이너는 E3-A 데스크톱보다 약 3.5배 빠르다. SER5는 약 5.2배다(§3.10 동일성 run). 그래서 SER5 ≈ 컨테이너 ÷ 1.45로 환산한다.
  SER5 실측은 A3.1에서 한다.
- **V8-B**: TB1이 승리를 증명하는 시간은 SER5 환산 9~24초다. cap은 8초다.
  - 손실 3판에서 RT30이 그 수에 쓴 시간은 10~17초였다. cap 이후 9~16초가 남았다.
  - 그 남은 시간을 V8-B에 더 주었다면 2~3판은 시간 안에 들어온다(13.3초 필요 / 17.2초 가능, 18.4 / 18.1, 24.3 / 24.0).
- **V8-C**: TB1의 기록 판정을 25/28국면에서 그대로 재현했다.
  나머지 3개는 측정 스크립트가 head 자식 수를 기록된 자식만으로 줄여서 생긴 차이다. 엔진 차이가 아니다(frozen과 fast 모두 같은 UNKNOWN).
  - tree 수 검사(노드 예산의 절반 200k를 다 쓰고 UNKNOWN): 컨테이너 median 32초 → **SER5 약 22초**.
  - 이어지는 자식 검사(첫 SAFE까지): median 11초 → **SER5 약 7.5초**. 합계 약 30초다.
  - 반면 RT30의 root 창은 약 19초다. 27.5초에서 final margin 1.5초, VCT2 reserve 4초, 앞 단계 약 3초를 뺀 값이다.
  - 그래서 tree 수 검사가 끝나기도 전에 창이 닫힌다.
  - **workers=1의 이득(약 10%로 추정)으로는 이 차이(22초 + 7.5초 대 19초)를 메울 수 없다.**

A3.1 진단과 TB-RT30-v2 계획은 §9.5로 옮겼다. 처음 안의 "fresh seed 재생 3회"는 tree 수를 재현하지 못해 폐기했다(§9.5).

### 9.5 A3 이후 계획, A3.1 진단, v2-A 구현 (2026-10-15)

**흐름(확정).** 한 번에 한 변경만 넣고, 단계마다 gate를 본다.

| 단계 | 내용 | 다음 조건 |
|---|---|---|
| A3 | TB-RT30-v1, workers 4 | **HARM 확정**(§9.4). 뒤의 결과로 번복하지 않는다 |
| A3.1 | 34 분기 수, tb1/v1/v2 × workers 4/1 | 원인 진단만 한다 |
| A3b (선택) | E3-A 100판, workers 1 = "엔진 동시 탐색 1개" 배포 조건 | 아래 A3.1 기준을 넘을 때만 돌린다. PASS면 v1 + worker 1로 배포 |
| v2-A | **V8-B 남은 시간 재시도만** | 개발 세트 + A1 gate |
| A3-v2A | E3-A 100판(같은 규칙) | PASS면 여기서 끝난다(v2-C 불필요) |
| v2-C | 필요할 때만: V8-C 시간 분할 50% | 개발 세트 + control 세트 + A1 gate |
| A3-v2 | E3-A 100판(같은 규칙) | PASS 필요 |
| A2-v2 | v2 설정 hash + `tests/frozen_tb_rt30_v2.sha256` | 이후 변경 금지 |
| B1/B2 | 멀티 세션 + polling (§8) | 시작 |

이렇게 하면 결과를 단계별로 나눠 설명할 수 있다. 정확 고속화, cap 8, V8-B 재시도, V8-C 스케줄러가 각각 얼마를 바꿨는지가 따로 남는다.

**버전 관리.**

- TB-RT30-v1 파일(`tests/frozen_tb_rt30_v1.sha256`)은 **영구히 그대로 둔다.** v1 = A3에서 HARM이었던 바로 그 엔진이다.
- v2는 새 파일이다.
  - `src/rt30/engine_v2.py`: v1의 `rt30_search`를 복사하고 V8-B 재시도만 바꿨다. 나머지 helper는 v1 모듈(lock 대상)에서 import한다.
  - `src/rt30/tb_rt30_v2.py`: v1 설정 + `attack_retry`, `timing.attack_retry_min` 1초. `STATUS = 'candidate'`, 설정 hash `9bccd7c2…`.
- solver는 v1 것을 그대로 쓴다. 재시도에는 solver의 상태 필드를 되돌리는 것만 필요하다.
- v2가 최종 gate를 통과한 뒤에만 `tests/frozen_tb_rt30_v2.sha256`을 만든다. `engine_v2.py`와 `tb_rt30_v2.py`에 더해, v2가 쓰는 v1 파일도 넣는다.

**A3.1 설계 수정: fresh seed 재생은 tree 수를 재현하지 못한다.**

- `run_rt30_replay.py`는 국면마다 새 agent(seed 42)로 탐색한다. 그런데 PUCT rollout(`_rollout_v321`)은 agent의 난수를 쓰고, 대국에서는 그 난수가 앞의 수들을 거치며 진행돼 있었다.
- 그래서 V8-C 28국면에서는 tree 수 자체가 대국 때와 달라질 수 있다. V8-B 6국면은 tree 전에 돌아서 상관없다.
- 새 도구 `scripts/run_rt30_divergence.py`:
  - 대국을 오프닝부터 다시 둔다. 엔진은 A3/E3-A와 같은 seed를 받고, 자기 차례마다 탐색한다(대국과 같은 난수 흐름).
  - 착수는 기록대로 둔다. 상대 엔진은 돌리지 않는다.
  - 분기 수에서 엔진의 수가 TB1의 수와 같으면 `recovered`다.
  - `prefix_mismatches`는 앞선 자기 수가 기록과 달랐던 수다. 달랐다면 난수 흐름이 달라졌을 수 있다.
  - `tree_move_as_recorded`는 tree 수가 기록과 같은 V8-C 국면 수다.
- 실행은 엔진 3개(tb1 = TB1 + 고속화, deadline 없음 / v1 / v2) × workers(4 = A3 부하, 1 = 엔진 하나)다. 처음 안의 fast w1을 포함한다.
  - tb1 w4와 w1의 차이: CPU 경쟁이 TB1의 증명 시간을 몇 % 늘리는지.
  - v1 w4와 w1의 차이: 같은 경쟁이 RT30의 결정을 몇 개 바꾸는지.
  - v2 w4: v2-A 개발 세트 결과. v2 w1: 엔진 하나 조건에서의 v2.
- 요약 집계:
  - 원인별(`attack` 6 / `root` 28) 복원 수.
  - 결과가 바뀐 13판, 그중 손해를 만든 10판(`loss_making`), 이득 3판(`gain_making`)의 복원 수.
  - `points_if_recovered`: 복원된 판이 E3-A 결과로 돌아간다고 볼 때의 점수. 상한이다(복원된 판도 나중에 다시 갈라질 수 있다). 이득 판의 복원은 음수로 센다.

**A3.1 판단 기준(실행 전에 고정).**

- **A3b(100판 workers 1)는 v1 w1의 순 `points_if_recovered`(loss + gain)가 +3.0 이상일 때만 돌린다.**
  -6점에서 1점만 되찾으면 −0.05가 돼도 구간 상한이 0을 넘지 못하기 때문이다.
- 그렇지 않으면 바로 v2-A의 A1 gate로 간다.
- 어느 경우든 v1 workers 4의 A3 HARM은 그대로다.

**v2-A: V8-B 남은 시간 재시도 (`engine_v2.rt30_search_v2`).**

- **soft cap과 hard deadline을 구분한다.** V8-B는 v1처럼 `min(시작 + 8초, deadline − final margin)`에서 멈춘다.
  - 앞쪽(8초 cap)이 먼저 걸렸으면 `attack_cap_hit`다. 이때 잘린 탐색의 solver(VCF cache), statuses, 후보 순서를 그대로 보존한다.
- **재시도 조건.**
  - 시간으로 잘린 탐색만 재시도한다(`solver.timed_out`). WIN을 찾았거나 node/call 예산을 다 써서 끝난 탐색은 다시 돌리지 않는다.
  - move deadline 자체가 V8-B를 자른 경우도 재시도하지 않는다.
- **재시도 시점.** v1과 똑같이 나머지 단계(stage VCT / PUCT / V8-C / VCT2)가 끝나 착수가 정해진 뒤다.
  - 그때 `deadline − final margin`까지 `attack_retry_min`(1초) 이상 남았으면 같은 solver로 이어서 돈다.
  - solver의 `deadline`을 바꾸고, `timed_out`과 `exhausted`를 해제한다. 해제는 시간 cut일 때만 한다.
  - 예산(200k 노드 / 10k 호출)은 처음 시도와 합쳐서 센다.
- **결과.**
  - WIN이 증명되면 그 수를 둔다(`own_vct`, best `own_vct_retry`).
  - 아니면 v1의 수를 그대로 둔다. 그래서 착수가 바뀌는 경우는 증명된 승리뿐이다.
- **TB1과의 차이.** 재시도는 남은 예산으로 공정 분배 라운드를 처음부터 다시 시작한다(`_first_proven`, 미판정·미검사 후보 대상, 끝난 하위 결과는 cache에서 가져온다).
  그래서 찾은 승리는 증명된 것이지만, TB1의 단일 run이 먼저 찾는 승리와 다른 승리일 수 있다.
- **로그.** soft cap은 실패가 아니라 스케줄링 사건이라 따로 남긴다(`RTInfoV2`).
  - `attack_cap_hit`, `attack_retried`, `attack_retry_seconds`, `attack_retry_win`.
  - `timeouts`의 'attack'은 착수 시점에 V8-B가 여전히 미완이었을 때만 남는다.
  - `run_rt30_replay.py --arms rt2`, `run_rt30_match.py --engine v2`, `run_rt30_divergence.py --engine v2`가 이 값을 기록한다.
- **테스트** (`tests/test_rt30_engine_v2.py`, 8개):
  - deadline 없음 = TB1(모든 route, 진단 동일).
  - 재시도 off = v1(같은 clock에서 착수·진단 동일).
  - soft cap에 잘린 승리를 남은 시간에 되찾는다(새 frozen solver로 WIN 재증명, 게임 상태 원상).
  - hard deadline cut, 남은 시간 부족, 예산 소진 시에는 재시도하지 않는다.
  - 설정은 v1 + 재시도만이다.
- **컨테이너 확인** (V8-B 6국면, deadline 27.5초, workers 4, uniform prior):
  - 조건: 컨테이너는 SER5보다 약 1.45배 느리고, tree가 H3가 아니라 뒤 단계 시간이 다르다. V8-B 자체는 tree와 무관하다.
  - v1: TB1 수 1/6(8416/4 백, tree가 우연히 같은 수).
  - v2: 3/6. 재시도로 되찾은 것은 **8415/20 흑**(A3에서 승→패)과 **8415/1 백**이다(재시도 11.7초, 7.2초).
  - 8416/22 흑은 재시도 5.9초로 모자랐다(TB1은 컨테이너에서 26.7초 필요).
  - 27.5초 초과 0, 재시도 착수의 최대 26.0초.
  - SER5에서는 남는 시간이 더 길어(손실 3판의 A3 착수가 10~17초에 끝났다) 1~2판을 더 기대한다. 확인은 A3.1의 v2 w4로 한다.
- **v2-A만으로 통과할 수 있는 상한.** V8-B가 걸린 점수는 −2.5점이다. 모두 되찾아도 차이는 −0.035다.
  그래서 A3-v2A PASS는 거의 다 되찾고 새 손실이 없을 때만 가능하다. 그래도 낮은 위험의 변경을 먼저 단독으로 재는 것이 순서다.
- **A3-v2A 실행 조건.** A3.1 v2 w4의 `loss_making.points_if_recovered` ≥ 2.0이고, A1 gate(재검사 UNSAFE 0, 27.5초 초과 0, 지는 수 증가 없음)를 통과해야 한다.
  아니면 A3-v2A를 건너뛰고 v2-C로 간다.

**v2-C 설계(보류, v2-A 다음에 필요할 때만).**

- **비율 고정.** `root_tree_time_share = 0.50`으로 고정하고 조정하지 않는다.
  근거: TB1이 tree 수에 node·call 예산의 정확히 절반을 먼저 주므로, 시간 slice도 절반이다. 개발 세트를 보고 0.55, 0.60으로 맞추지 않는다.
- **soft cut.**
  - 처음 단계: root 창의 50%(v1과 같은 VCT2 reserve 적용) 동안 tree 수만 검사한다.
    - SAFE면 tree 수 유지.
    - UNSAFE면 기존 rescue.
    - soft cut으로 미판정이면 solver·cache·statuses를 보존하고 soft 상태를 해제한다.
  - 다음 단계: 남은 창에서 frozen `_search_root_children`를 돈다. **tree 수도 미판정 후보로 포함한다**(TB1 구조 그대로).
- **로그.** `root_tree_soft_cut`, `root_tree_first_status`, `root_child_search_started`, `root_child_safe_found`.
- **control 세트 `docs/rt30-eval/a3-root-control-v1.json`(26국면).**
  - 기준: E3-A tree 수 중 V8-C가 tree 수를 데스크톱 60초 이상(SER5 약 12~26초) 걸려 SAFE로 증명한 수 전부.
  - TB1은 여기서 tree 수를 유지한다. 시간 분할이 이 수를 괜히 바꾸는지 본다.
  - 개발 세트(바꿔야 할 수를 몇 개 복원하는가)와 함께 본다.

**SER5 실행(백그라운드).**

- `scripts/ser5/start.sh`는 job을 새 session(`setsid nohup`, 터미널 없음)으로 띄운다. **SSH가 끊겨도 계속 돈다.**
- job 여러 개를 주면 순서대로 돈다(앞 job이 실패하거나 중지되면 멈춘다). 측정이 섞이지 않도록 job은 한 번에 하나(체인)만 돌린다.
- 로그·상태 파일은 `runs/rt30/bg/<job>.out`, `.pid`(실행 중), `.done`(끝남)이다.
- 각 단계는 출력 파일이 있으면 건너뛴다. 대국·분기 러너는 jsonl로 단계 안에서도 이어간다. 그래서 재부팅이나 중지 후 **같은 명령으로 다시 시작하면 이어진다.**

```bash
cd ~/renju-alphazero && git pull
scripts/ser5/start.sh a31 v2a_a1     # A3.1 (약 4.5시간) -> v2-A A1 gate (약 1시간), 연달아
scripts/ser5/status.sh               # 상태와 마지막 로그 (SSH 다시 접속한 뒤)
tail -f runs/rt30/bg/a31.out         # 진행 보기 (Ctrl-C는 보기만 멈춘다)
scripts/ser5/stop.sh a31             # 중지 (체인 전체); 다시 start.sh a31 v2a_a1 이면 이어진다
# A3.1·A1 결과가 위 기준을 넘으면:
scripts/ser5/start.sh v2a_a3         # A3-v2A 100판 (약 2.5~3시간)
```

- 결과 커밋 대상:
  - `runs/rt30/a31_*.json`·`.jsonl`·`.log`
  - `v2a_h3_*.json`·`.log`, `recheck_v2a_*.json`, `v2a_compare_*.md`
  - `a3_v2a_e3.*`
- 위 파일을 `docs/rt30-results/`로 복사해 커밋한다. 모두 자체 엔진 기록이라 RenjuNet 파생 데이터가 아니다.

### 9.6 A3.1 / v2-A 결과 (`d3dc9c3`): CPU 경쟁 가설 기각, v2-A는 A3 실행 기준 미달

**결론.** A3의 HARM은 workers 4의 CPU 경쟁 때문이 아니다. **27.5초 안에서 지금의 단계별 시간 배분으로는 TB1의 전술적 수 교체에 도달하지 못하는 구조 문제**다.

v2-A(V8-B 재시도)는 정상 동작하고 A1 gate도 통과했다. 그러나 A3 분기 복원은 2/34, 점수로는 +1.0점이다. 사전 기준(+2.0점)에 못 미치므로 **A3-v2A 100판은 돌리지 않는다.** v2-A는 버리지 않고 유지한다. 안전하게 남길 가치가 있는 부분 개선이다.

**A3.1 (34 분기 수, 게임 상태 재생).**

| 엔진 | workers 4 복원 | workers 1 복원 | 착수 median / max | 27.5초 초과 |
|---|---|---|---|---|
| TB1 + 고속화 (deadline 없음) | **34/34** | **34/34** | 32.9 / 72.6초 (w1 33.1 / 73.1) | 25 (w1 26) / 34 |
| TB-RT30-v1 | 0/34 | 0/34 | 26.00 / 26.02초 | 0 |
| TB-RT30-v2 후보 (v2-A) | 2/34 | 2/34 | 26.00 / 26.02초 | 0 |

- **재생은 신뢰할 수 있다.** TB1이 34/34를 재현했고, `prefix_mismatch_cases` 0, V8-C 28국면의 tree 수도 28/28 기록과 같다.
  fresh seed 재생 대신 게임 상태 재생으로 바꾼 것(§9.5)이 맞았다.
- **CPU 경쟁은 없었다.** TB1의 시간은 w4와 w1이 같다(V8-B median 16.98 / 17.07초, V8-C 30.19 / 30.43초). w1이 오히려 근소하게 느렸다.
  이전의 "w1이면 약 10% 빠를 것" 추정은 SER5(물리 8코어)에서 성립하지 않았다.
  - **A3b는 돌리지 않는다**(v1 w1 복원 +0.0점, 기준 +3.0점).
  - 기력 때문에 엔진 worker를 1개로 제한할 필요도 없다. worker 수는 B3 부하 시험에서 응답성으로 정한다.
- **필요한 시간과 주어진 시간 (SER5 실측).**

| phase | TB1이 쓰는 시간 (median) | v1이 주는 시간 |
|---|---|---|
| V8-B (attack 6국면) | 17.0초 (10.5~25.0) | 8.0초 (cap) |
| V8-C (root 28국면) | 30.2초 (21.9~41.1) | 19.7초 |

**v2-A 동작.**

- A3.1 w4에서 `attack_cap_hit` 8, `attack_retried` 5, `attack_retry_win` 2. w1에서는 8 / 4 / 2다.
  - cap hit 8은 V8-B 국면 6개에 V8-C 국면 2개(8416/13 흑, 8416/1 흑)를 더한 수다. 뒤의 2개는 V8-B가 승리 없이 8초를 쓴 국면이다.
- 되찾은 2개:
  - 8415/20 흑(A3 승→패, 재시도 7.8초): 실제 손실 1판 회복.
  - 8415/1 백(A3 승→승, 재시도 4.3초): 결과에는 영향이 없다.
- **못 찾은 원인은 재시도의 위치다.** 재시도는 VCT2까지 끝난 뒤에 돈다. 그런데 그 국면들에서 v1의 VCT2가 약 6초를 썼다(8416/22 흑 6.2초, 8416/23 흑 5.9초). 8416/23 흑은 V8-C도 8.8초를 썼다.
  - 그래서 남은 시간이 1.5~10초였고, TB1이 필요로 한 V8-B 시간(20.8초, 21.8초, 25.0초)에 못 미쳤다.
  - 8416/4 백은 V8-B·V8-C·VCT2가 모두 잘려 재시도할 시간이 없었다.

**v2-A의 A1 gate.**

| 세트 | 지표 | v1 (cap 8) | v2-A |
|---|---|---|---|
| slow60 | same / 승리 손실 / tree 수 유지 | 53 / 3 / 4 | **56 / 0** / 4 |
| | attack timeout / timeout 수 | 11 / 21 | **5 / 15** |
| | 재검사 UNSAFE / 27.5초 초과 | 0 / 0 | **0 / 0** (바뀐 tree 수 4개: SAFE 3, UNKNOWN 1) |
| | median / 합계 | 9.80 / 760.0초 | 9.77 / 846.9초 (+11.4%) |
| critical6 | same / 승리 손실 | 3 / 3 | **4 / 1** (남은 것: 8415/15/32, TB1 [5,11] 대 [6,12]) |
| | 재검사 UNSAFE / 27.5초 초과 | 0 / 0 | **0 / 0** (바뀐 수 8415/22/37 [2,7]은 VCT1 SAFE) |
| | median / 합계 | 11.0 / 90.1초 | 20.8 / 116.7초 |

- 비용은 모든 수에 퍼지지 않는다. median은 그대로이고, cap에 걸린 어려운 수에서만 남는 시간을 쓴다. 최대 26.01초로 SLA는 그대로다.
- **slow60에서는 승리 손실이 0이 됐는데 A3에서는 +1점인 이유.** 두 세트의 역할이 다르다.
  - slow60의 v1 승리 손실 3개는 재시도로 닿는 범위에 있었다.
  - A3의 손실은 V8-B −2.5점, V8-C −3.5점이다. v2-A는 V8-C를 건드리지 않는다. V8-B 6개 중에서도 시간 안에 닿은 것은 2개다.

| 항목 | 판정 |
|---|---|
| A3(v1) HARM | 유지 |
| workers 4 CPU 경쟁 가설 | 기각 (TB1 w4 = w1) |
| A3b (workers 1, 100판) | 실행하지 않음 |
| v2-A 구현 / SLA / 안전 | 정상 / 27.5초 초과 0 / UNSAFE 0 |
| v2-A 공격력 (A1) | 크게 개선 (slow60 승리 손실 3 → 0, critical 3 → 1) |
| v2-A A3 복원 | 2/34, +1.0점 < 기준 2.0 → A3-v2A 생략 |

### 9.7 시간 제한 아래 기력 복구: 가능성 판단과 설계 초안

> **§9.8에서 정정됨.** 이 절의 표는 첫 예측기 기준이고, 세 가지가 과장돼 있다.
> 1. "병렬"이 V8-B 병렬(X2)에 VCT2 투기 실행(X3)을 섞어 셌다.
> 2. "34/34 복원 = A3 차이 0"은 성립하지 않는다. 34개는 각 판의 *첫* 분기라, 복원된 판도 뒤에서 다시 갈라질 수 있다.
> 3. "C/Cython이면 3배 이상"은 근거가 부족했다.
>
> 순서도 바뀌었다(T1은 fallback). 정정된 수치와 확정 순서는 §9.8에 있다.

**가능성: 있다. 시간 제한 자체가 TB1의 기력을 막지는 않는다.**

- E3-A 2,938수의 TB1 phase 시간을 SER5로 환산했다(데스크톱 ÷ 4.69, A3.1 34수의 실측 비율 median; `scripts/predict_rt30_recovery.py`).
- 26초(27.5초 − final margin)를 넘는 수는 **150수(5.1%)**뿐이고, 거의 모두 26초의 2.5~3배 안에 있다.
- **두 가지 정확한(착수가 바뀌지 않는) 변경**이 있다.
  - TB1 자체를 빠르게 하는 것.
  - V8-B를 나머지와 동시에 돌리는 것.
- 이 둘로 거의 모든 수를 시간 안에 TB1과 **같은 수**로 둘 수 있다. 모든 수가 TB1과 같으면 A3의 100판은 E3-A의 100판과 같아진다. 그러면 차이는 정의상 0이다.

| 정확 고속화 배수 | E3-A 26초 초과 (순차 / V8-B 병렬) | A3 V8-B 6국면 (병렬) | A3 V8-C 28국면 (순차 / 병렬) | 그중 결과가 바뀐 10판 (병렬) |
|---|---|---|---|---|
| 1.0 (지금) | 150 / 121 | **6/6** | 1 / 1 | 0/10 |
| 1.5 | 63 / 43 | 6/6 | 21 / 22 | 8/10 |
| 2.0 | 29 / **13 (0.4%)** | 6/6 | 27 / **28** | **10/10** |
| 2.5 | 6 / **0** | 6/6 | 27 / 28 | 10/10 |
| 3.0 | 3 / 0 | 6/6 | 28 / 28 | 10/10 |

- **V8-B 병렬은 지금 속도로 A3 V8-B 6국면을 전부 되찾는다.** TB1의 V8-B가 10.5~25.0초로 26초 안에 끝나기 때문이다. −2.5점에 해당한다.
- **V8-C는 정확 고속화 2배면 28/28이다**(−3.5점).
- 이 표는 시간 측면의 상한이다. 정확 변경은 "시간 안에 끝나면 TB1과 같은 수"이므로, 위 숫자가 곧 TB1 수 재현 수다.

**3배를 어디서 얻나 (V8-C tree 수 검사 프로파일, 8416/9 백, cProfile, 200k 노드).**

- `threat_windows_codes`(+ `numpy.array` 변환): 누적 **약 42%**.
  - 노드마다 판 전체를 numpy 배열로 바꿔 모든 window를 다시 센다(노드당 2.6회 호출).
  - VCF는 한 노드에서 돌 몇 개만 바뀐다. 그래서 window 집계를 **증분 갱신**하면(돌을 둘 때 그 점을 지나는 window 20개만) 대부분 없앨 수 있다.
- 흑 금수 판정(`forbidden_reason` → `_classify_black_stone`, `run_length`, `_fours`): 누적 **약 26%**.
- VCF 재귀 자체(`_find_vcf_recursive`, 후보 정렬, `inside`, generator): 나머지.
- 판단:
  - 순수 Python의 증분화·메모화로 약 1.5~2배를 기대한다(위 두 항목을 대부분 줄였을 때 1/(1 − 0.5)).
  - 핵심 루프(window 집계, 금수 판정, VCF 재귀)를 C 확장/Cython으로 옮기면 3배 이상이 현실적이다.
  - cProfile은 작은 함수 비용을 부풀리므로, 각 변경은 fast1·fast2처럼 microbenchmark와 동일성 테스트로 잰다.

**설계: 정확 경로(X)를 본선으로, 시간 정책(T)을 잔여 수의 안전망으로.**

| 변경 | 종류 | 무엇을 | 기대 효과 | 위험·비용 |
|---|---|---|---|---|
| **X1. 3차 정확 고속화** (`rt30/fast3_*`) | 정확(착수·노드 수 동일) | 증분 threat window, 금수 판정 가속, 필요하면 C/Cython 핵심 루프 | 2배: V8-C 27~28/28, E3-A 초과 29수 | 구현량이 크다. C 확장은 SER5 빌드 환경(gcc) 필요. 동일성 테스트로 검증한다 |
| **X2. V8-B 병렬 실행** | 정확(V8-B가 끝나면 TB1과 같은 수) | V8-B를 엔진 worker의 보조 프로세스에서 26초 전체 동안 돌리고, 본 프로세스는 stage/tree/V8-C/VCT2를 돈다. V8-B가 WIN이면 그 수, 아니면 본선의 수 | V8-B 6/6(지금 속도), 본선은 V8-B cap(최대 8초)을 쓰지 않으니 V8-C 창이 늘어난다 | 탐색 1개가 코어 2개를 쓴다. 동시 탐색 수는 B3에서 정한다. v2-A 재시도는 X2가 대체한다 |
| X3. VCT2 투기 실행 (선택) | 정확 | VCT2를 tree 수에 대해 V8-C와 동시에 돌린다. V8-C가 수를 바꾸면 버리고 다시 돈다 | VCT2 reserve 4초를 V8-C에 돌려준다 | 코어 1개 더 |
| **T1. v2-C: V8-C 50% 시간 분할** | 시간 정책 | §9.5 설계 그대로(비율 0.50 고정, tree 수 포함 soft cut) | X가 끝나기 전에도 V8-C 국면을 되찾는다. 컨테이너 측정으로 자식 검사가 9.85초 창에 들어오는 국면 22/25, 결과가 바뀐 판 10/10 | control 세트(26국면)에서 늦게 SAFE인 tree 수를 바꿀 위험 |
| T0. v2-A 재시도 | 시간 정책 | 유지 | X2가 들어오면 거의 쓰이지 않는다 | — |

- **왜 X를 본선으로 보나.**
  - X는 "TB1과 같은 수"라 기력 손실 상한이 0이다. A3도 사실상 동일성 확인이 된다.
  - T는 TB1이 더 오래 생각해서 내렸을 결정을 추정할 뿐이다. 그래서 개발 세트에 맞추는 위험과 control 세트의 대가가 따른다.
  - 다만 X1은 시간이 걸린다. 그래서 T1을 먼저 넣어 배포 가능한 v2를 빨리 만들고, X로 그 위를 대체해 간다.
- **버전.**
  - TB-RT30-v2 = v1 + T0 + T1. 지금 계획대로이고 A3-v2로 판정한다.
  - TB-RT30-v3 = v2 + X2 (+ X1, X3). 정확 변경은 착수를 바꾸지 않지만, 설정(`exact_speedups`, 병렬 여부)과 lock 파일이 바뀌므로 새 버전이다.
  - v1·v2 lock은 그대로 둔다.

**순서와 gate.**

1. **T1 구현(v2-C)** → SER5에서 `a31`과 같은 방식으로 v2 w4를 개발 세트 34국면 + control 세트 26국면에 돌리고, A1 gate를 본다.
   - 사전 기준:
     - 개발 세트에서 결과가 바뀐 판의 순 `points_if_recovered` ≥ +3.0.
     - control 세트에서 tree 수 유지 ≥ 23/26(바꾸는 수 ≤ 3, 그것도 증명된 SAFE).
     - A1: 재검사 UNSAFE 0, 27.5초 초과 0.
   - 통과하면 **A3-v2 100판**을 돌린다.
2. **X1 타당성(컨테이너에서 가능, SER5 불필요)**: 증분 threat window 시제품.
   - 동일성: 같은 window 목록·순서, V8-B/V8-C의 같은 statuses·노드 수.
   - microbenchmark로 배수를 잰다. 1.3배 미만이면 C/Cython으로 간다.
3. **X2 구현**: 엔진 worker 안의 보조 프로세스(상주, 같은 monotonic deadline). 테스트는 다음을 확인한다.
   - deadline 없음 = TB1.
   - V8-B가 끝나면 TB1과 같은 수.
   - 보조 프로세스가 죽거나 늦으면 본선 결과.
4. **v3 평가**: `run_rt30_divergence.py --engine v3`(복원 수, 정확 경로라 34에 가까워야 한다), `predict_rt30_recovery.py`(실측 배수로 E3-A 초과 수), A1 gate, A3-v3.
5. 배포 엔진은 A3를 통과한 가장 높은 버전이다(v2 PASS면 v2로 B1을 시작하고, v3는 그 뒤에 교체한다).

**그 밖에 검토했지만 지금은 넣지 않는 것.**

- **상대 시간 활용(pondering).** 사람이 생각하는 동안 예상 응수에 대해 미리 탐색한다. 효과는 클 수 있지만, 예측이 맞을 때만 이득이다. 동시 사용자 수와 CPU를 다투고, A3(봇 상대)로는 평가할 수 없다. B 단계 이후 과제로 둔다.
- **phase 간 VCF cache 공유**(§3.5). 노드 예산이 새 노드만 세기 때문에, cache가 따뜻해지면 같은 예산으로 더 멀리 탐색한다. 그래서 TB1과 다른 판정이 나온다(정확 변경이 아니다). X1이 끝난 뒤에도 필요하면 별도 gate로 본다.

### 9.8 확정 방향: R0 → X2 → X1 → 분기·A1 → A3-v3 (T1은 fallback), 구현과 실행 (2026-10-16)

**사실 검토(사용자 검토안 대조).**

| 검토 항목 | 판정 | 근거 |
|---|---|---|
| A3 손실은 시간 문제다 | 맞다 | TB1 34/34, v1 0/34(w4·w1 모두). V8-B 17초 대 8초, V8-C 30.2초 대 19.7초 |
| V8-B 병렬은 지금 속도로 6/6 가능 | 맞다, **컨테이너로 확인** | TB1 V8-B 10.5~25.0초. 컨테이너 v3 6/6(아래) |
| X2는 동시 탐색 수를 늘린다(4판 × 2프로세스 = 8) | 맞다 | gate에 동시 탐색 1/2/4를 넣었다(`v3_dev`) |
| 예측기 "parallel"에 X3(VCT2 투기)가 섞였다 | 맞다 | `side = other + tree + vct2`. R0에서 분리했다 |
| A3 V8-C 28/28(2배)은 X3 덕분이 아니다 | 맞다 | 28국면 모두 TB1이 수를 바꿔(`switched`) VCT2가 본선에 있다. 분리 후에도 X2 2배에서 28/28 |
| 34/34 복원이 A3 차이 0을 보장하지 않는다 | 맞다 | 첫 분기만 본 것이다. E3-A 전체에서 남는 초과 수(아래 `e3a_over_budget`)가 다시 갈라질 수 있는 곳이다 |
| 3배는 근거 부족, 목표는 2배 | 맞다(목표 조정) | cProfile 비율로 Amdahl 상한은 1/0.32 = 3.1배다 |
| 같은 단서: 실측은 프로파일 추정보다 좋았다 | 추가 | window scan 하나로 2.15~2.31배(아래). Amdahl 추정 1.6배보다 높다. cProfile이 작은 함수 호출 비용을 부풀려 window 몫을 낮게 보였다 |
| 금수 판정을 함께 바꾸지 말 것 | 따랐다 | X1-A는 window scan만 바꾼다. 다음 병목이 금수 판정(fast3 기준 누적 약 40%)임은 확인만 했다 |
| T1은 fallback으로 | 따랐다 | 정확 경로만으로 A3가 HARM일 때만 v3T로 넣는다 |

**R0: 예측기 정정 (`scripts/predict_rt30_recovery.py`, `docs/rt30-results/predict_recovery_tb1_w4.json`).**

일정(schedule)은 네 가지다.

- `v1_timing`: v1의 시간 배분. V8-B cap 8초, V8-C는 22초 시점까지, VCT2 reserve. **X1 단독**이다.
- `sequential`: TB1 순서에서 cap 없이 합만 본다. 참고용이다.
- `x2`: V8-B 병렬.
- `x2_x3`: X2에 VCT2 투기 실행을 더한다.

각 일정에 대해 A3 첫 분기 34국면이 시간 안에 들어오는 수와 그 판들이 가진 점수(A3 −6.0점 중), E3-A 2,938수 중 26초 초과 수를 낸다.

| 정확 고속화 | v1_timing (X1만) | x2 (X2) | x2_x3 | x2: V8-B / V8-C / 결과 바뀐 판 |
|---|---|---|---|---|
| 1.0 | 0점, 초과 225 | **2.5점**, 초과 125 | 2.5점, 121 | 6/6 / 1/28 / 3/13 |
| 1.5 | 1.5점, 148 | 4.5점, 53 | 4.5점, 43 | 6/6 / 22/28 / 11/13 |
| **2.0** | 4.5점, 92 | **6.0점, 초과 24 (0.8%)** | 6.0점, 13 | 6/6 / **28/28** / **13/13** |
| 2.5 | 4.5점, 58 | 6.0점, 0 | 6.0점, 0 | 6/6 / 28/28 / 13/13 |

- **X2 + 실측 약 2배가 목표다.** 2배에서 A3의 첫 분기 손실 −6.0점이 모두 시간 안에 들어오고, E3-A에서 26초를 넘는 수는 24수(0.8%)다.
- X1 단독(v1 시간 배분)은 2배여도 4.5점이고 초과 92수다. 8초 cap이 V8-B 6국면 중 3개를 계속 막는다. 그래서 X2가 먼저다.
- `v1_timing`의 1.0배 초과 225수는 A3에서 deadline 사건이 있던 261수와 비슷하다. 모델이 실측과 맞는다는 근거다.
- x2가 1.5배에서 sequential보다 0.5점 낮은 것은 이득이었던 판(8415/22 백, 무→승)을 하나 더 TB1 수로 되돌리기 때문이다. 오류가 아니다.

**X1-A: native threat-window scan (구현, `rt30/native/threat_windows.c`, `rt30/fast3_windows.py`, `rt30/fast3.py`).**

- 판(파이썬 list of lists)을 그 자리에서 읽어 572개 window를 C로 훑고, frozen과 같은 순서의 window 목록을 돌려준다. 상태도 캐시도 없고 판에 쓰지 않는다.
  frozen 보조 함수들이 판에 직접 쓰기 때문에 증분 상태는 두지 않았다. 그래서 정확성 검증이 단순하다.
- 빌드는 `python -m rt30.native.build`다(C 컴파일러 + Python 헤더 필요). `.so`는 커밋하지 않는다.
  CI의 두 테스트 job이 빌드 후 테스트한다. 빌드가 없으면 v3 `make_agent`가 거부한다(fast2로 돌면 평가 대상 엔진이 아니다).
- **동일성.**
  - 무작위 3,000국면 × 양측 × minimum 0~5에서 frozen과 같은 목록·순서다.
  - V8-B 결과·statuses·노드·호출 수가 같다.
  - replay frozen = fast3 = rt3(deadline 없음)이 같다(컨테이너 2국면, SER5 H3 8국면은 `v3_check`).
- **속도(컨테이너, 단일 프로세스, 동일 결과·노드).**
  - 호출당: fast2 39.5µs → **4.1µs**(frozen 247µs).
  - V8-C tree 수 검사(8416/9 백, 200k 노드 UNKNOWN): 40.1초 → 18.6초, **2.15배**.
  - V8-B(8415/20 흑, WIN, 101k 노드): 21.5초 → 9.3초, **2.31배**.
  - SER5 배수는 `v3_check`가 TB1 자신의 시간으로 잰다(`a31_tb1f3_w4` 대 `a31_tb1_w4`, `v3_speedup.md`).
- 다음 병목은 흑 금수 판정이다(`forbidden_reason` 누적 약 40%). **X1-B는 SER5 실측이 목표에 못 미칠 때만** 한다.

**X2: V8-B 병렬 (구현, `rt30/attack_helper.py`, `rt30/engine_v3.py`).**

- 엔진 프로세스마다 **상주 helper 프로세스** 하나를 둔다(spawn, 첫 사용 때 시작해 계속 쓴다).
- 착수마다 수순, V8-B 예산, 절대 deadline을 보낸다(같은 기계의 `time.monotonic`은 프로세스 간 같은 시계다).
  helper는 v1의 deadline V8-B(같은 후보·예산·공정 분배, fast3 설치)를 **cap 없이 deadline − final margin까지** 돈다.
- 본 프로세스는 그동안 v1과 같은 나머지(stage VCT / PUCT / V8-C / VCT2)를 돈다.
- 판정 순서(고정).
  1. helper가 WIN을 증명했으면 그 수를 둔다. 이때 진단은 V8-B 이전 상태에 V8-B 것을 더한 상태로 되돌리고, 에이전트 난수(PUCT rollout이 썼다)도 V8-B 시작 시점으로 되돌린다.
     그래서 **다음 수부터의 난수 흐름이 TB1과 같다**(테스트: own_vct → tree → own_vct → tree 연속에서 수와 난수 상태가 같다).
  2. WIN 없이 끝났으면 본선의 수를 둔다(= TB1의 수).
  3. deadline까지 미완이면 본선의 수를 두고 `timeouts`에 'attack'을 남긴다.
- 본선이 진행 중에 WIN이 오면 다음 phase 경계에서 끝낸다(`attack_win_early`).
- helper를 못 쓰는 경우에는 V8-B를 v1처럼 inline(cap 8초)으로 돈다(`attack_helper_failed`에 이유를 남긴다).
  - 데몬 프로세스 안이라 자식 프로세스를 못 만드는 경우.
  - helper가 죽은 경우.
  - deadline이 테스트 시계인 경우.
- **웹 연결 전 할 일.** `rt30.supervisor`의 worker가 `daemon=True`라 지금은 helper를 만들 수 없다.
  그래서 `tb_rt30_v3.make_agent`는 helper를 못 쓰는 프로세스에서 **거부한다**(`allow_inline_attack=True`일 때만 inline V8-B로 만든다).
  v1 시간 배분으로 도는 엔진이 `TB-RT30-v3` 이름을 달지 않게 하기 위해서다. 웹 연결 구조는 §9.9의 search slot이다.
  평가 스크립트의 ProcessPoolExecutor worker는 non-daemon이라 문제없다.
- **테스트** (`tests/test_rt30_engine_v3.py`, 11개):
  - native scan 동일성·설치/해제.
  - deadline 없음 = TB1(모든 route, 진단 동일).
  - V8-B 승리 후 난수 흐름 = TB1.
  - 느린 V8-B가 deadline 안에 끝나거나 'attack' 미완으로 남는다.
  - 빠른 승리.
  - 테스트 시계 = v1, off = v1.
  - 설정 = v1 + X1 + X2.
- **컨테이너 확인** (A3 V8-B 6국면, uniform prior: V8-B는 tree와 무관, workers 2, deadline 27.5초):
  - v3 **6/6**(+2.5점; v1 0/6, v2-A 2/6).
  - 착수 median 16.1초, 최대 20.0초, 27.5초 초과 0.

**버전.**

- TB-RT30-v1은 동결이다(A3 HARM, 영구 보존).
- TB-RT30-v2A(`tb_rt30_v2.py`, V8-B 재시도)는 실험 기록으로 남기고 동결하지 않는다.
- **TB-RT30-v3 후보**(`tb_rt30_v3.py`, `STATUS = 'candidate'`, hash `f80d2348…`) = v1 + X2 + X1-A. 정확 복구 본선이다.
- TB-RT30-v3T = v3 + T1(V8-C 50% 분할)은 A3-v3가 HARM일 때만 만든다.
- v3가 gate를 통과하면 `tests/frozen_tb_rt30_v3.sha256`을 새로 만든다.
  `engine_v3.py`, `attack_helper.py`, `fast3*.py`, `native/threat_windows.c`, `tb_rt30_v3.py`와 v3가 쓰는 v1 파일이 대상이다.

**gate (실행 전에 고정, `scripts/check_v3_gate.py gate`).** A3-v3는 아래가 모두 PASS일 때만 돈다(`v3_a3.sh`가 먼저 확인하고, FAIL이나 REVIEW면 100판을 돌리지 않는다).
종료 코드는 PASS 0, FAIL 1, REVIEW 2다. control 기준은 SER5 실행 전에 23/26에서 26/26으로 고쳤다(§9.9).

| 항목 | 기준 |
|---|---|
| 동일성 (H3 8국면, deadline 없음) | frozen = fast3 = v3, 8/8 |
| 분기 34국면, 동시 탐색 4 | 순 `points_if_recovered` ≥ +3.0, 27.5초 초과 0 |
| X2 사용 | 분기·control·A1의 모든 수에서 helper가 쓰였다(`attack_helper_failed` 0). inline V8-B는 v1의 시간 배분이라 평가 대상이 아니다 |
| control 26국면, 동시 탐색 4 | TB1의 tree 수 유지 **26/26**. 바뀐 수가 있으면: timeout·skip 없이 바뀐 수는 정확성 위반(FAIL), deadline에 잘려 바뀐 수는 수동 원인 분석(REVIEW, §9.9) |
| A1 (slow60, critical6) | 재검사 UNSAFE 0, 27.5초 초과 0 |

- 동시 탐색 2와 1(`a31_v3_w2`, `w1`)은 기록만 하고 gate로 쓰지 않는다. X2의 CPU 몫을 보는 진단이다.
- A3-v3 자체의 판정 규칙은 A3와 같다(차이 < −0.05 또는 구간 상한 < 0이면 HARM).
- A3-v3가 HARM이면 다음을 본다.
  - X1-B(금수 판정 가속)로 배수를 올리는 쪽: 예측기에서 남은 손실이 시간 초과 수에 있을 때.
  - T1(v3T)을 넣는 쪽: 시간 안에 들어오는데도 다른 경우.

**SER5 실행 (백그라운드, SSH가 끊겨도 계속).**

```bash
cd ~/renju-alphazero && git pull
# 한 번만: native scan 빌드 도구. 헤더는 .venv-ser5 Python(3.14.4)의 것이어야 한다(§9.9).
.venv-ser5/bin/python -c 'import sysconfig, pathlib; p = pathlib.Path(sysconfig.get_paths()["include"], "Python.h"); print(p, p.exists())'
scripts/ser5/start.sh v3_check v3_dev v3_a1 v3_a3   # 약 6.5시간, 순서대로; A3-v3는 gate PASS일 때만
scripts/ser5/status.sh                               # 상태 (다시 접속한 뒤)
tail -f runs/rt30/bg/v3_check.out                    # 진행 (v3_dev.out, v3_a1.out, v3_a3.out)
cat runs/rt30/v3_speedup.md runs/rt30/v3_gate.md     # SER5 배수, gate 결과
scripts/ser5/stop.sh v3_check                        # 중지 (체인 전체); 같은 start 명령이면 이어진다
```

| job | 내용 | 예상 시간 |
|---|---|---|
| `v3_check` | native 빌드(소스·Python이 바뀌었을 때만 다시, `native_build_info.json` 기록), H3 8국면 동일성, TB1 + fast3의 34국면 시간(배수), 예측 | 약 40분 |
| `v3_dev` | v3의 34 분기 국면 × 동시 탐색 4/2/1, control 26국면 | 약 2.5시간 |
| `v3_a1` | slow60 + critical6, 재검사, v1·v2A와 비교표 | 약 1시간 |
| `v3_a3` | gate 확인 → E3-A 100판 | 약 2.5시간 |

결과 파일은 `runs/rt30/`의 다음 파일이다.

- `native_build_info.json`, `v3_equivalence_h3_8.*`, `a31_tb1f3_w4.*`, `predict_recovery_tb1f3_w4.json`, `v3_speedup.md`
- `a31_v3_w{4,2,1}.*`, `ctrl_v3_w4.*`
- `v3_h3_*`, `recheck_v3_*`, `v3_compare_*.md`
- `v3_gate.md`, `a3_v3_e3.*`

이 파일들을 `docs/rt30-results/`로 복사해 커밋한다.

### 9.9 v3 검토 반영: 용어, control gate, 웹 통합 순서, native 범위 (2026-10-16)

`4e6e9d3` 검토 결과를 반영한다. **v3 후보의 탐색 코드는 바꾸지 않는다.** 바꾼 것은 gate, `make_agent` 거부 조건, 빌드 기록이다.

**용어.** X1-A와 X2는 종류가 다르다. 앞 절의 "정확"은 이 구분으로 읽는다.

| 변경 | 종류 | deadline 없음 | deadline 아래 |
|---|---|---|---|
| X1-A native scan | **정확 계산 고속화**(exact computational speed-up) | TB1과 같다 | 같은 window·순서·노드, 더 빠를 뿐 |
| X2 V8-B helper | **TB1 순서 보존 병렬 스케줄링**(TB1-order-preserving parallel scheduling) | TB1과 같다 | 시간 배분이 v1과 다르다(V8-B cap 8초 → 수 전체). 그래서 v1과 일부러 다른 수를 둘 수 있고, 그것이 A3 복원의 핵심이다 |

공통 성질은 "끝까지 돈 phase의 결정은 TB1의 결정"이다. 그래서 v3의 기력 손실 상한은 시간 안에 못 끝난 수에만 있다.

**control gate: 23/26 → 26/26.**

- 23/26은 v2-C(T1)처럼 시간 분할로 수가 바뀔 수 있는 엔진의 허용치였다. v3에는 맞지 않는다.
- v3가 control 수(TB1이 늦게 SAFE로 증명한 tree 수)를 바꾸는 길은 둘뿐이다.
  - deadline이 어떤 phase를 잘랐다(`timeouts`·`skipped` 기록). 원인을 국면마다 본다 → **REVIEW**.
  - 아무것도 잘리지 않았는데 바뀌었다. 정확성 위반이다 → **FAIL**.
- REVIEW일 때 `v3_a3.sh`는 멈춘다. 바뀐 수를 모두 확인한 뒤 `V3_REVIEWED=1 scripts/ser5/start.sh v3_a3`로 다시 시작한다.
- **X2 사용 확인을 gate에 추가했다.** 분기·control·A1의 어느 수라도 `attack_helper_failed`가 있으면 FAIL이다(그 수는 v1 시간 배분이다).

**웹 통합 순서.** 멀티 웹대국(B1/B2, §8)을 먼저 끝낸다. native core의 추가 확장과 v3의 production 연결은 그 뒤다.

- X1-A·X2 코드는 그대로 둔다. 지금은 benchmark·replay·A1·A3용 **후보 엔진**이다. 웹은 계속 v1만 등록한다.
- `tb_rt30_v3.make_agent`는 helper를 만들 수 없는 프로세스(데몬 worker)에서 거부한다. 지금 웹 worker에 v3를 등록하면 바로 오류가 난다.
- 웹 연결 구조는 nested process가 아니라 **형제(sibling) 구조**로 한다.

```text
Web → AIJobQueue → dispatcher
                    ├─ Search slot 1: main worker + V8-B helper worker
                    └─ Search slot 2: main worker + V8-B helper worker
```

- supervisor가 slot의 두 프로세스를 함께 만들고, watchdog 때 함께 끝낸다. 데몬 제한도 생기지 않는다.
- 처음 값은 `MAX_AI_SEARCHES = 2`다(CPU를 많이 쓰는 프로세스 최대 4개). B3에서 3, 4로 올려 보고 deadline이 지켜지면 올린다.
- **엔진 deadline은 탐색 시작부터다(§8의 3번에서 이미 정했다).** dispatcher가 job을 꺼낸 시각을 `received_at`으로 넘긴다.
  대기열에서 기다린 시간은 27.5초에 들어가지 않는다. 지금 `select()`의 `received + 27.5`는 단일 worker 웹에서만 맞다.
  대기열 대기가 탐색 시간을 깎으면 v3의 복원 효과가 다시 사라진다.

**native 범위.** 이름도 "C++ native core"가 아니라 "native core"다.

- 작고 상태 없는 kernel은 C로 쓴다(X1-A). C++로 다시 쓰지 않는다.
- VCF 재귀 전체를 native로 옮기게 되면 그때만 C++17(Board, VCFState, 후보 vector, RAII placed/undo)을 쓴다.
- 순서는 아래로 고정한다. 앞 단계로 충분하면 뒤 단계는 하지 않는다.

| 순서 | 단계 | 조건 |
|---|---|---|
| 1 | v3_check: SER5 native 동일성 + 실측 배수 | 다음 |
| 2 | v3_dev: 34 분기 + 26 control, 동시 탐색 4/2/1 | 다음 |
| 3 | v3_a1: slow60 + critical6 + UNSAFE 재검사 | PASS 필요 |
| 4 | A3-v3: E3-A 같은 100판 | 최종 기력 판정 |
| 5 | v3 동결: lock 파일 + native provenance | A3 PASS일 때 |
| 6 | 웹 연결: search slot 형제 구조 | 웹 B1/B2 완료 뒤 |
| 7 | X1-B 금수 판정 native(v4 후보) | A3-v3가 **시간 부족**으로 HARM일 때만 |
| 8 | VCF 전체 native(C++17) | X1-B까지 해도 부족할 때만 |
| 9 | T1(v3T/v4-T1) | 시간은 충분한데 결정 손실이 남을 때만 |

**버전.**

```text
TB-RT30-v1            동결 baseline (A3 HARM, 영구 보존)
TB-RT30-v2A           V8-B 재시도 실험, 기록만
TB-RT30-v3 candidate  v1 + X1-A + X2 (4e6e9d3, 탐색 코드 그대로)
  A3 PASS → 그대로 v3 동결
  A3 HARM ├ 시간 부족           → TB-RT30-v4 candidate = v3 + X1-B
          └ 시간 충분, 결정 손실 → v4-T1 검토
```

v3의 의미는 동결 뒤에 바꾸지 않는다. 가속을 더하면 새 버전이다.

**Python 3.14.4와 빌드 기록.**

- SER5 production은 CPython 3.14.4(`.venv-ser5`)다. `apt install python3-dev`는 시스템 Python 버전의 헤더를 설치하므로, 3.14 헤더라는 보장이 없다.
  기준은 **`.venv-ser5` Python의 `sysconfig` include 경로에 `Python.h`가 있는 것**이다. `rt30.native.build`가 이것을 확인하고, 없으면 설치할 패키지 이름(예: `python3.14-dev`)을 알려 준다.
- `python -m rt30.native.build`는 `src/rt30/native/build_info.json`을 남긴다: Python 버전, SOABI, 컴파일러와 버전, 플래그(`-O2`), C 소스와 `.so`의 SHA256.
  `--if-stale`은 소스·Python·플래그가 바뀌었을 때만 다시 빌드한다(`git pull` 뒤 낡은 `.so`를 쓰지 않는다). `ensure_native`가 이것을 쓰고 기록을 `runs/rt30/native_build_info.json`으로 복사한다.
- v3를 동결할 때 이 기록을 lock과 함께 커밋한다.

### 9.10 SER5 v3 결과 (`ed3d8cf`): gate PASS, A3-v3 **PASS**, 동결 전 확인 1건 (2026-10-16)

**판정.** 사전에 고정한 gate(§9.8, §9.9)를 모두 통과했고, A3-v3는 HARM 기준(차이 < −0.05 또는 쌍 CI95 상한 < 0)에 걸리지 않는다.
"TB1과 통계적으로 동등함을 증명했다"는 뜻은 아니다. **사전 HARM 기준을 통과했고, 관측된 기력 차이가 −0.06에서 −0.01로 줄었다**는 뜻이다.

**속도 (X1-A, SER5, `v3_speedup.md`).** fast2 → fast3, TB1 자신의 시간, 34국면 모두 같은 수(34/34).

| 구간 | 중앙 배수 (범위, n) |
|---|---|
| V8-B attack | 2.14 (1.94~2.53, 10) |
| V8-C root | 2.17 (1.76~2.71, 28) |
| 전체 | **2.07** (1.76~2.53, 34) |

컨테이너 측정(2.15~2.31배)이 SER5에서도 재현됐다. 목표(약 2배)를 달성했다.

**gate (`v3_gate.md`).** 동일성 8/8, inline V8-B 0, 분기 34/34(+6.0점, 결과가 바뀐 13판 13/13), 27.5초 초과 0, control **26/26**, A1 재검사 UNSAFE 0.

- 분기 34국면은 동시 탐색 4/2/1 모두 34/34다. 착수 시간 중앙값은 15.55 / 15.72 / 15.75초, 최대는 22.1~22.4초다.
  4판을 동시에 돌려도 복원이 무너지지 않는다. A3 손실이 CPU 경쟁이 아니라 시간 배분과 Python hot path 때문이라는 §9.6 진단과 맞는다.
- control 26/26 유지와 inline V8-B 0은 복원이 "다른 수를 골라 운 좋게 이긴 것"이 아니라 **TB1의 결정을 시간 안에 재현한 것**이라는 근거다.

**A1 (`v3_compare_*.md`).**

| 세트 | 엔진 | median | p95 | max | timeout 수 | baseline과 같은 수 | lost win |
|---|---|---|---|---|---|---|---|
| slow60 | v1 | 9.80 | 26.00 | 26.00 | 21 | 53/60 | 3 |
| slow60 | v2-A | 9.77 | 26.01 | 26.01 | 15 | 56/60 | 0 |
| slow60 | **v3** | **5.37** | **24.49** | **25.38** | **4** | **60/60** | **0** |
| critical6 | v1 | 11.03 | 26.0 | 26.0 | 5 | 3/6 | 3 |
| critical6 | v2-A | 20.82 | 26.01 | 26.01 | 3 | 4/6 | 1 |
| critical6 | **v3** | **8.58** | **17.54** | **17.54** | **0** | **6/6** | **0** |

slow60의 phase timeout(attack/stage/root/vct2)은 v1 11/0/10/9에서 v3 0/0/3/0이다. 26초 근처에 몰려 있던 꼬리가 줄었다.

**A3-v3 (`a3_v3_e3.*`, workers 4, 27.5초).**

| | TB-RT30-v1 (A3) | **TB-RT30-v3** | E3-A (기준) |
|---|---|---|---|
| 점수 | 0.75 | **0.80** (74승 12무 14패) | 0.81 |
| 흑 / 백 | 0.68 / 0.82 | **0.77 / 0.83** | 0.77 / 0.85 |
| 차이, 쌍 CI95 | −0.06, [−0.11, −0.015] | **−0.01, [−0.025, 0.0]** | |
| 기준과 같은 판 | 66 | **97** | |
| 착수 p95 / p99 / max (초) | 26.00 / 26.00 / 26.08 | **12.51 / 25.00 / 25.89** | |
| 27.5초 초과 / helper 실패 | 0 / — | **0 / 0** | |
| 대국 시간 합 | 568.1분 | **321.6분** (−43%) | |

- **흑은 기준과 같다**(0.77 = 0.77). A3.1에서 V8-B 손실이 주로 흑에서 났다는 진단과 맞는다.
- **남은 −0.01은 백 2판이 전부다.** 둘 다 승→무이고, 둘 다 tree 수의 V8-C가 deadline에 잘린 경우다(`root` timeout, 25.0초·25.2초).
  - `8416/2/white` ply 93: TB1은 이 수에 142.5초(데스크톱)를 썼다.
  - `8415/8/white` ply 47: TB1은 226.9초(데스크톱)를 썼다.
  - 2배로도 닿지 않는 긴 V8-C 증명이다. 예측기(§9.8)가 2배 + X2에서도 남는다고 본 E3-A 초과 수(24수, 0.8%)에 속하는 종류다.
- 100판 착수 3,013수 중 deadline 사건이 있는 수는 67수(root 54, 그 밖에 skip)다. attack·stage·VCT2 timeout은 0이다.

**확인할 1건: `rt30_no_deadline` (`8416/23/white`, ply 21).**

- v3는 [4, 6](1.5초), TB1 기록은 [7, 9](2.6초)다. 두 판 모두 승이라 결과는 같다. 그 수에는 timeout도 skip도 없다(route `tree`, PUCT 수 그대로).
- 같은 판의 v1은 ply 11에서 deadline으로 갈라져 ply 21까지 오지 않았다. 그래서 v1 기록과 비교할 수는 없다.
- 이 판의 ply 11은 TB1이 187.5초(데스크톱)를 쓰고 tree 수를 바꾼(`changed`) 수다. v3는 21.5초에 deadline 사건 없이 같은 수를 뒀다.
  2배 가속만으로는 설명하기 어려운 차이라서, 이 수도 같이 본다.
- 판의 수와 앞선 착수가 모두 같으므로, ply 21에서 다른 수를 둘 원인은 셋 중 하나다.

| 가설 | 판별 (`v3_nd`) | 결론 |
|---|---|---|
| H1. SER5의 policy 평가 차이(기계·torch 부동소수점). PUCT 방문 수가 근소한 국면에서 수가 바뀐다 | TB1 + fast3(deadline 없음)도 [7, 9]가 아니다 | v3 문제가 아니다. 그대로 동결하고 기록만 남긴다 |
| H2. v3의 상태 버그(난수 복원, helper 응답 처리 등) | TB1은 [7, 9], v3를 시간 압박 없이(deadline 100000초) 돌려도 다르다 | **동결 보류**. 원인 수정, 해당 판 재생과 회귀 테스트, 필요하면 A3-v3 재실행 |
| H3. 앞선 수(예: ply 11)에서 기록되지 않은 deadline 효과가 상태를 바꿨다 | TB1·v3(시간 압박 없음)는 [7, 9], v3(27.5초)만 다르다. 난수 digest가 처음 갈라지는 ply가 그 수다 | 분류 누락이다. 그 phase의 cut을 `timeouts`에 기록하도록 고친다(착수 의미는 그대로) |

- 세 실행 모두 [7, 9]이면 재현되지 않은 일시적 경우다. 기록으로 남기고 동결한다.
- `run_rt30_divergence.py --trace`가 착수마다 수, route, deadline 사건, 탐색 전후 난수 digest를 남기고, 사례 ply의 root 방문 수를 남긴다.
  `scripts/compare_rt30_traces.py`가 세 실행을 나란히 놓고, 수와 난수 상태가 처음 갈라지는 ply를 보여 준다.
  manifest는 `docs/rt30-eval/a3v3-no-deadline-v1.json`, job은 `scripts/ser5/v3_nd.sh`(약 15~30분)다.

**native 빌드 기록 (`native_build_info.json`).** Python 3.14.4, SOABI `cpython-314-x86_64-linux-gnu`, GCC 15.2.0, `-O2 -shared -fPIC`, C 소스와 `.so`의 SHA256. 동결 lock에 넣을 준비가 됐다.

**다음 단계.**

1. `v3_nd`로 위 1건을 판별한다(H1 또는 재현 안 됨 → 2로, H2 → 수정 후 재검증, H3 → 기록 수정 후 2로).
2. **v3 동결**: `STATUS = 'frozen'`, `FREEZE_EVIDENCE`(gate·A3 결과 파일), `tests/frozen_tb_rt30_v3.sha256`
   (`engine_v3.py`, `attack_helper.py`, `fast3*.py`, `native/threat_windows.c`, `native/build.py`, `tb_rt30_v3.py`와 v3가 쓰는 v1 파일),
   native 빌드 기록. v1 lock은 그대로 둔다.
3. **추가 가속(X1-B, VCF 전체 native)과 T1은 보류한다.** 남은 손실 2판은 2배로도 닿지 않는 긴 증명이라, 이를 좇는 비용이 남은 0.01점보다 크다.
4. **웹 B1/B2(§8)로 넘어간다.** 그 뒤 §9.9의 search slot(main + V8-B helper 형제 프로세스, 탐색 시작부터 deadline)으로 v3를 production에 연결하고,
   B3에서 `MAX_AI_SEARCHES` 2 → 3 → 4를 잰다(A3 workers 4가 착수 시간에 영향을 주지 않았다는 점은 좋은 출발점이지만, 웹 부하는 따로 잰다).

**`rt30_no_deadline` 1건 판정 (`76cce57`, `nd_compare.md`).** 같은 SER5에서 같은 판 수순과 H3 policy로 세 번 재생했다.

| 실행 | 조건 | ply 21 |
|---|---|---|
| `nd_tb1` | TB-Engine-v1 + fast3, deadline 없음 | [4, 6] |
| `nd_v3_free` | v3, deadline 100000초 | [4, 6] |
| `nd_v3_rt` | v3, 27.5초 | [4, 6] |

- ply 3~21의 자체 착수 10수가 세 실행에서 모두 같다. 탐색 직전 난수 digest도 한 번도 갈라지지 않는다(`first different move: None; first different random state: None`).
- ply 21의 root 방문 수도 같다: [4, 6] 15회(Q 0.2), [4, 7] 10회, [8, 5] 6회, [4, 9] 4회. E3-A 기록의 [7, 9]는 상위 4개에 없다.
- ply 11도 세 실행이 [6, 7]로 같고 20.8~21.8초다. 기록의 187.5초는 E3-A 데스크톱 실행(14 workers)의 시간이다. SER5의 TB1 자신도 21초 안팎이므로, v3만 다른 경로를 탔다는 근거는 없다.

| 가설 | 판정 | 근거 |
|---|---|---|
| H1. SER5의 TB1이 E3-A 기록의 수를 재현하지 못한다 | **채택** | `nd_tb1`도 [4, 6] |
| H2. v3 상태 버그(난수 복원, helper) | 기각 | TB1 = v3(시간 압박 없음) = v3(27.5초): 수·난수·방문 수 모두 같다 |
| H3. 앞선 수의 기록되지 않은 deadline 효과 | 기각 | v3 시간 압박 없음 = v3 27.5초 |

- 결론: v3 회귀가 아니다. **기준 기록(E3-A, 데스크톱)과 지금 SER5 사이의 환경 의존 재현 차이**다.
  세부 원인(policy 추론 수치, 라이브러리 버전 등)은 이 실험에서 나누지 않았다. 그래서 "부동소수점 차이"로 확정하지는 않는다.
- 빈도는 낮다. A3.1의 34개 분기 국면에서 SER5 TB1은 기록 수를 34/34 재현했고, A3-v3 착수 3,013수 중 이런 경우는 이 1수다. 이 판은 양쪽 모두 승이라 점수에 영향이 없다.
- A3 분기의 의미는 다음과 같다: 같은 판 97, deadline으로 갈라진 판 2(백 승→무 2판, 남은 −0.01의 전부), 기준 환경 차이 1.

**최종 판정.** A3-v3에서 유일한 `rt30_no_deadline` 분기를 SER5에서 TB-Engine-v1, 시간 압박 없는 v3, 27.5초 v3로 재생했다.
세 실행은 모든 자체 착수와 탐색 전 난수 상태가 같았고, ply 21에서 모두 [4, 6]을 골랐으며 root 방문 분포까지 같았다.
E3-A 기록에는 [7, 9]가 있다. 따라서 이 차이는 v3의 deadline·X2 상태 버그가 아니라, 기준 기록을 만든 환경과 SER5 사이의 재현 차이로 판정한다.
이 판은 양쪽 모두 승이라 A3 점수에도 영향이 없다. **TB-RT30-v3는 사전 정의한 A3 HARM 기준과 안전성·동일성 gate를 모두 만족했으므로 동결한다.**
A3 100판은 다시 돌리지 않는다. 추가 native 가속(X1-B, VCF 전체)과 T1은 보류하고, 웹 B1/B2로 넘어간다.

**동결 (A4).**

- `rt30/tb_rt30_v3.py`: `STATUS = 'frozen'`, `FREEZE_EVIDENCE`(속도, 동일성, 분기 w4/w2/w1, control, A1과 재검사, gate, A3, no-deadline 재생, native 빌드 기록). 설정 hash는 그대로 `f80d2348…`다.
- `tests/frozen_tb_rt30_v3.sha256`: v3가 쓰는 v1 파일 8개 + `engine_v3.py`, `attack_helper.py`, `fast3.py`, `fast3_windows.py`, `native/__init__.py`, `native/threat_windows.c`, `native/build.py`, `tb_rt30_v3.py`.
- `tests/test_tb_rt30_v3.py`가 다음을 확인한다.
  - 설정 hash, v1과의 차이(X1-A·X2뿐).
  - 결과 파일이 이 설정으로 돌았다는 것: 엔진 이름, hash, native scan.
  - gate와 A3 기준.
  - no-deadline 재생 세 실행의 동일성.
  - lock 파일 hash. v1의 lock 파일은 모두 포함된다.
  - SER5 빌드의 C 소스 hash = 저장소의 C 소스 hash.
- `.so`는 커밋하지 않는다(기계마다 빌드). SER5 빌드의 Python 3.14.4·GCC 15.2.0·`.so` SHA256은 `native_build_info.json`에 있다. 다른 기계에서 빌드한 `.so`는 hash가 달라도 같은 C 소스면 같은 엔진이다(동일성 테스트로 확인).
