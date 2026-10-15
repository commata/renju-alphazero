# Native core (C++) 개발 방향 초안

상태: **초안(검토용)**, 2026-10-15. 기준 코드는 `origin/feat/Web_SER5` `ac7c12e`.
사전 분석 두 편(“C/C++로 옮겼을 때 효과가 큰 곳”, “전체 C++화는 비추천”)을 코드와 기존 측정으로 검토한 결과다.

## 0. 요약

- **결론은 두 분석과 같다.** 전체 C++화는 하지 않는다. 측정된 CPU hot path만 native core로 만들고, 나머지는 Python으로 둔다.
- **단, 첫 단계는 바꾼다.** 두 분석은 `threat_windows` → 금수 → (필요하면) VCF 순서를 권한다.
  이 코드에서는 **`_find_vcf_with_stats` 호출 하나를 통째로 native로 넘기는 것**이 첫 배포 단위로 더 낫다.
  - 함수 단위 native(`threat_windows`만, `forbidden_reason`만)는 매 호출 경계 비용(15×15 list 변환, 결과 tuple 생성)을 낸다.
    지금 fast2의 `threat_windows_codes`는 이미 호출당 약 20μs이고, 그 대부분이 변환 비용이다. 그래서 효과가 작다.
  - 증분 window 갱신은 board 상태가 native 쪽에 계속 있어야만 가능하다. Python이 `game.board[r][c]`를 직접 바꾸는 지금 구조에서는
    VCF 호출 안에서만 할 수 있다.
  - `_find_vcf_with_stats`는 **board만 읽는 순수 함수**다. cache·history·난수가 없고, 후보 순서는 전부 `sorted`다.
    그래서 의미 동일성(결과·노드 수·cut 여부)을 함수 하나의 차분 테스트로 판정할 수 있다. 두 분석이 생각한 것보다 위험이 작다.
- **내부 개발은 bottom-up, 배포 경계는 VCF 호출 하나.** 금수·window·4 연장을 C++ 안에서 먼저 만들고 하나씩 차분 테스트한 뒤,
  Python에는 `find_vcf_with_stats` 하나(와 Track A용 `legal_black_points`)만 내보낸다.
- **설치 방식은 fast1/fast2와 같다.** frozen 파일은 고치지 않고, `rt30.fast`의 `install()`처럼 모듈 이름을 바꿔 끼운다.
  native가 없거나 시작 시 자체 검사에 실패하면 Python(fast2)으로 돌아간다.
- 목표: SER5에서 V8-C 증명 **전체 ≥ 2배(fast2 대비)**. 기존 예측표(§9.7)에서 2배면 A3 V8-C 28/28, 결과가 바뀐 10판 10/10이다.

## 1. 두 분석의 사실 확인

| 주장 | 확인 결과 |
|---|---|
| 직접 작성한 C/C++ 코드 없음 | 맞다. `pyproject.toml`은 setuptools만 쓰고 ext_modules가 없다. numpy도 `server` extra의 선택 의존성이다 |
| V8-C에서 `threat_windows_codes` 약 42%, 흑 금수 약 26% | 맞다(`docs/web-ser5.md` §9.7, 8416/9 백, cProfile, 200k 노드). **단 cProfile 값이다.** fast2 때 profile 25~26%였던 4 연장을 3.9배 빠르게 해도 전체는 1.05배였다(§3.9). 작은 함수 비중은 부풀려져 있다 |
| 나머지 약 32%는 VCF 재귀 등 | 맞다. `_find_vcf_recursive`, 후보 정렬, `inside`, generator. 함수 단위 native로는 이 부분이 그대로 남아 상한이 약 3배다 |
| 웹 서버·NN을 C++로 바꿀 이유 없음 | 맞다. 한 수의 시간은 탐색(최대 약 26초)이 정한다. NN은 이미 PyTorch native/CUDA다 |
| Track A도 금수 판정 이득 | 맞다. Stage 7 실측에서 흑 `legal_moves`가 착수당 17~33%(사용자 PC, `legal_black_points` 최적화 뒤), NN forward 47~66%다(`docs/stage7-plan.md` §4, §6) |
| SER5 Python 3.14 | **문서에서 확인하지 못했다.** 빌드 전에 SER5에서 `python3 --version`으로 확인한다(개발 컨테이너는 3.13) |
| “X1에서 C 확장 고려” | 이미 계획에 있다(§9.7 X1: “필요하면 C/Cython 핵심 루프”, “1.3배 미만이면 C/Cython으로 간다”) |

두 분석의 위험 지적(후보 순서가 바뀌면 node budget·UNKNOWN 위치가 달라져 “새 엔진”이 된다)은 정확하다. 이 초안의 gate는 그 기준을 그대로 쓴다.

## 2. 왜 VCF 호출 단위가 최소 위험·최대 이득인가

### 2.1 경계 비용

| 경계 위치 | 경계 통과 횟수 | 경계 비용이 차지하는 몫 | Python에 남는 부분 |
|---|---|---|---|
| `threat_windows` 함수 | VCF 노드당 약 2.6회 | 크다(board 변환 + 결과 tuple list) | 금수, 재귀, 정렬 전부 |
| `forbidden_reason` 함수 | 후보마다 | quiet 경로는 Python 판정 자체와 비슷한 크기 | window, 재귀 |
| **`_find_vcf_with_stats` 호출** | VCF 호출당 1회(호출당 수백~수만 노드) | 무시할 수준 | VCT/ThreatSolver 조직, cache, stage 로직 |

### 2.2 기대 배율(추정, Phase 0에서 실측으로 바꾼다)

- 함수 단위(window + 금수 native): profile 비중을 그대로 믿으면 약 2배지만, §3.9의 교훈대로 실제는 **1.3~1.8배로 예상**한다.
  나머지 Python 32%가 상한을 3배로 막는다.
- VCF 호출 단위: VCF 안의 코드는 작은 정수 연산과 분기뿐이라 C++에서 수십 배가 흔하다.
  V8-C 시간 중 VCF 호출 안의 비율을 *p*, VCF 안 가속을 *k*라 하면 전체는 1/((1−p) + p/k)다.
  - *p* = 0.9, *k* = 30이면 약 7.7배. *p* = 0.7이면 약 3배. *p*가 작아도 함수 단위보다 낮아지지 않는다.
  - 그래서 **Phase 0에서 *p*를 먼저 잰다.** 이 숫자 하나로 투자 여부가 정해진다.

### 2.3 의미 동일성을 판정하기 쉽다

`search.mcts_v7._find_vcf_recursive`의 의존성은 닫혀 있다.

```text
_find_vcf_with_stats(board, attacker, max_fours, node_limit)
 ├─ _winning_moves        → _window_candidates(4) → _threat_windows, _wins_for_player(run_length)
 ├─ _window_candidates(3) → sorted(set(...))      (row-major 순서)
 ├─ _is_legal_for_player  → rules.forbidden_reason (흑만)
 ├─ _legal_completions    → sorted(_placed_completions) → 4 연장(fast_extensions와 같은 결과)
 ├─ placed / undo         → board[r][c] 직접 대입
 └─ _VCFState.take_node   → node 수, node_limit cut (rt30.vcf는 여기서 deadline 검사)
→ (VCFResult | None, nodes, exhausted)
```

- 모든 후보 순서가 `sorted`라 set 순회 순서에 기대는 곳이 없다. C++에서는 row-major index 정렬과 같다.
- 출력에 `nodes`와 `exhausted`가 있다. 후보 순서가 하나라도 다르면 node 수가 달라지므로, 순서 동일성이 자동으로 검사된다.
- 재귀 깊이는 `max_fours`(≤ 빈칸/2 + 1)로 묶인다.

## 3. 범위

### 3.1 넣는다 (native core v1)

하나의 C++17 모듈 `renju_native`(소스는 `native/`). 내부 구성:

- `Board`: `int8[225]`, 흑 1·백 −1(Python과 같은 값).
- 규칙: `run_length`, `_fours`, `_straight_four_after_extension`, `_open_three`(재귀), `_classify_black_stone`, `forbidden_reason`,
  `legal_black_points`. `src/renju/rules.py`를 줄 단위로 옮긴다. quiet prefilter도 그대로 옮긴다(결과는 같고 빠르다).
- window: 572개 window index 표, base-3 코드 표(fast2와 같은 방식), 필요하면 VCF 안에서 증분 갱신.
- 4 연장: `fast_extensions.winning_extensions_in_direction`과 같은 알고리즘.
- VCF: `_find_vcf_recursive`와 같은 순서·같은 node 계산. deadline 검사 포함.

Python에 내보내는 API(이것만 배포 경계):

```python
find_vcf_with_stats(board, attacker, max_fours, node_limit, *, deadline_at=None, check_limit=None)
    -> (attack_moves, completion_points, fours, nodes, exhausted, timed_out)
legal_black_points(board) -> list[tuple[int, int]]      # Track A용, Phase 3에서 설치
forbidden_reason(board, row, col) -> str | None          # 테스트·Game.play용
```

- `deadline_at`은 `time.monotonic()` 값이다. Linux에서는 둘 다 CLOCK_MONOTONIC이라 같은 시계다. 매 노드마다 검사한다(frozen과 같은 지점).
- `check_limit`은 테스트용 `rt30.deadline.CheckLimit`(검사 횟수로 끊는 결정적 deadline)를 그대로 재현한다. 시간 의존 없이 cut 경로를 비교할 수 있다.
- `timed_out`이면 Python 래퍼가 `SearchTimeout`을 던진다. board는 native 사본이라 복원할 것이 없다.
- 내부 함수(window, 4 연장, `_fours` 등)는 테스트 전용으로만 내보낸다.

### 3.2 조건부 (측정 후 결정)

| 항목 | 조건 |
|---|---|
| Track A/B self-play의 `Game.legal_moves()`를 native `legal_black_points`로 설치 | v1 gate 통과 뒤. 같은 코드라 추가 비용이 작다. 기대: 흑 `legal_moves` 17~33% 중 대부분 → self-play 1.2~1.4배(추정). Windows(사용자 PC) 빌드가 필요하므로 그때 정한다 |
| VCT 조직(`ThreatSolver.after_move`, `SelectiveSolver`)의 native화 | v1 뒤 profile에서 VCF 바깥 Python이 다시 주 병목일 때만. VCF cache·budget 의미가 얽혀 위험이 크다 |
| stage4·`_double_threat_moves` 등 VCF 밖의 `threat_windows` | fast2로 둔다. v1 뒤 profile에서 남으면 그때 본다 |

### 3.3 하지 않는다

PUCT tree(GIL·NN 대기가 지배, 재현성 영향 큼), PolicyValueNet·학습 루프·replay·checkpoint, 웹 서버·세션·SQLite·queue,
평가·실험 스크립트, VCF cache를 phase 간에 공유하는 것(정확 변경이 아니다, §9.7).

## 4. 정확성 계약과 gate

“정확 고속화”의 정의는 fast1/fast2와 같다: deadline이 없으면 착수·diagnostics·nodes·calls가 frozen과 같고, deadline이 있으면 같은 시간에 더 멀리 갈 뿐이다.

| Gate | 내용 | 통과 기준 |
|---|---|---|
| G0 규칙 | C++ `forbidden_reason`, `legal_black_points`, `run_length` vs `rules.py`. 무작위 국면 ≥ 100k × 모든 빈칸, `tests/reference_rules.py`·`rule_validation.py`·`test_rule_optimization.py`의 fixture(33·44·장목·exact five·금수 예외·재귀 삼) | 불일치 0 |
| G1 부품 | window(목록·순서), 4 연장(집합), `_winning_moves`, `_window_candidates` vs Python | 불일치 0 |
| G2 VCF | `find_vcf_with_stats` vs frozen `_find_vcf_with_stats`. 실제 국면(E3-A jsonl, `docs/rt30-eval/*`, A3 국면) + 무작위, 양쪽 attacker, node_limit 여러 값(작은 값으로 cut 경로 포함), `CheckLimit` 여러 값 | 결과·`nodes`·`exhausted`·timeout 위치 모두 동일 |
| G3 엔진 | replay 동일성: frozen = fast2 = fast3(deadline 없음). H3 8국면, critical 6국면, A3 divergence 국면 | 착수·diagnostics 동일 |
| G4 속도 | SER5 microbenchmark: V8-C `14/15` `[5,4]` SAFE 증명(65k 노드), V8-B `9/26` | fast2 대비 ≥ 2배(목표), < 1.5배면 중단 검토 |
| G5 운영 | A1 gate(재검사 UNSAFE 0, 27.5초 초과 0) → A3-v3 | 기존 기준 |

추가 안전장치:

- **시작 시 자체 검사.** engine worker가 native를 설치하기 전에 고정 fixture(수십 국면)를 Python과 비교한다. 하나라도 다르면 설치하지 않고 fast2로 돌며 로그를 남긴다.
- **provenance.** TB-RT30-v3 lock에 native 소스 SHA256, 빌드한 컴파일러·플래그, `.so` SHA256, 모듈 버전을 넣는다. 결과 JSON의 `exact_speedups`에 `native` 버전을 기록한다.
- **frozen 파일은 고치지 않는다.** `threats.py`, `mcts_v8.py` 등이 import한 `_find_vcf_with_stats` 이름을 `rt30/fast3.install()`이 바꿔 끼운다(`fast_windows.install()`과 같은 방식). v1·v2 lock은 그대로 둔다.
- CI: native 빌드가 되는 환경에서 G0~G2를 돌리고, 안 되는 환경에서는 fallback 경로 테스트만 돈다.

## 5. 단계

| Phase | 내용 | 산출물 | 진행 조건 |
|---|---|---|---|
| **0. 측정** (반나절) | V8-C/V8-B/VCT2에서 `_find_vcf_with_stats` 안의 wall-clock 비율 *p*를 잰다(cProfile이 아니라 래퍼 타이머). SER5 Python 버전·gcc 확인 | *p*, 호출 수, 호출당 노드 분포 | *p* ≥ 0.6이면 Phase 1. 작으면 VCT 조직 쪽을 먼저 profile |
| **1. 규칙 core** | `Board`, `run_length`, 금수 판정, `legal_black_points` C++ + 빌드 뼈대(`native/`, pybind11, Python fallback) | G0 통과 | 불일치 0 |
| **2. VCF core** | window·4 연장·`_winning_moves`·VCF 재귀·deadline/CheckLimit, `rt30/fast3.py` install | G1·G2·G3 통과, G4 측정 | G4 ≥ 1.5배. 2배 이상이면 v3 후보 |
| **3. 배포** | TB-RT30-v3(= v2 + X2 + native, §9.7), 시작 시 자체 검사, provenance | G5 | A3 PASS |
| 4. (조건부) Track A | `Game.legal_moves()` 흑 경로에 native 설치 옵션, Windows 빌드 | Stage 7 방식 차분·golden hash | self-play가 CPU 병목일 때 |

멈추는 기준: Phase 2에서 G2 불일치를 원인까지 설명하지 못하면 배포하지 않는다. G4가 1.5배 미만이면 Python 증분 window(X1 원안)로 돌아간다.
Phase 3에서 A3를 통과하면 native 범위 확장은 거기서 멈춘다(두 번째 분석의 결론과 같다).

## 6. 빌드·배포 선택

- **pybind11 + C++17**, setuptools `ext_modules`로 빌드한다(CMake는 필요할 때만). 현재 setuptools 구성과 가장 가깝다.
  CPython C API 직접 작성도 가능하지만 경계 API가 3개뿐이라 차이는 작다. 대상 Python(3.13, SER5 버전)을 pybind11이 지원하는지 Phase 0에서 확인한다.
- Cython은 대안이다. 다만 결국 타입을 붙여 다시 쓰게 되므로 동일성 검증 부담이 같고, 증분 window 같은 자료구조는 C++이 더 명확하다. numba는 list-of-lists board와 JIT warmup 때문에 맞지 않는다.
- native는 **선택 의존성**이다(`server` extra처럼). 없으면 지금과 같은 fast2로 돈다.
- SER5(Linux, gcc)가 1차 대상이다. Windows(MSVC) 빌드는 Phase 4에서만 필요하다.

## 7. 위험과 대응

| 위험 | 대응 |
|---|---|
| 렌주 규칙 포팅 오류(재귀 삼, exact five, 장목) | G0을 가장 엄격하게. `rules.py`와 1:1 구조로 옮기고 최적화는 동일성 확인 뒤에만 |
| 후보 순서 차이로 node 수가 달라짐 | G2가 `nodes`까지 비교하므로 바로 드러난다 |
| deadline 의미 차이 | 노드마다 같은 지점에서 검사, `CheckLimit`으로 결정적 비교 |
| 빌드 환경 차이(Python ABI, 컴파일러) | fallback 필수, 시작 시 자체 검사, provenance 기록 |
| 예상보다 작은 배율 | Phase 0의 *p*로 사전 판단, G4 중단 기준 |
| 유지보수 이중화(Python·C++ 규칙 두 벌) | Python이 기준(oracle)이고 C++은 그것을 따르는 구현이다. 규칙을 바꾸면 엔진 버전 단계로 다룬다 |

## 8. 열린 질문

1. SER5의 Python 버전과 빌드 도구(gcc, pybind11 설치 가능 여부).
2. Phase 0의 *p* 값. V8-C와 VCT2가 다르면 어느 쪽을 기준으로 할지.
3. X2(V8-B 병렬)와 native를 v3에 함께 넣을지, native를 v4로 분리할지. 분리하면 A3 원인 분석이 쉬워진다.
4. Track A 쪽 Windows 빌드를 언제 필요로 할지(B1 시작 시점과의 관계).
