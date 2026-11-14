# R2-S400-TRAIN: S400 목표 후보의 학습·평가 계약 설계 (초안, 결정 전)

- 상태: seed 예약 v2·새 개발 suite 준비(§8.1), 학습 도구·테스트·계약 준비(§8.2). **모든 단계 미승인**. prepare·gate 데스크톱 블록 준비(§9). search·train·Arena 블록과 v2 Arena 실행·판독은 아직 없다.
- 채택된 결정(검토 2026-11-13): 재탐색 방식 B, 행 집합 고정, 새 개발 opening으로 평가, 보조 arm 생략, T 재현 gate를 재탐색 앞으로. 근거와 세부는 §2–§6.
- 근거
  - R2-P1-POLICY-REPLAY-A1(`faeb72b0f625`): T(정책 보존, S200 목표)는 champion 대비 NOT_ESTABLISHED, C 대비 BETTER_THAN_CONTROL
  - R2-S400-PRE(`e8ca3049a4ee`, 기록 `3608bb8`): 표본 2,421국면에서 S200 → S400 학습 목표가 바뀐다(top-1 16.9%, TV 0.141). 바뀐 목표가 더 좋은 수를 가르치는지는 측정하지 않았다. 전체 재탐색은 약 4.4시간, train 범위만이면 약 4.0시간으로 추정된다.
- 이 계약이 답할 질문: **A1 T와 다른 모든 조건을 같게 두고 train의 방문 목표만 S400으로 바꾸면, S50으로 두는 정책이 더 강해지는가.**
- 전체 재탐색 비용은 학습과 Arena에서 무엇을 비교할지 고정한 뒤에 쓴다.

## 1. 고정할 비교

| 항목 | 대조군 | 후보 |
|---|---|---|
| 정책 | A1의 T(seed마다) | S400 목표 정책(seed마다) |
| train 방문 목표 | c2 S200 | S400 재탐색 |
| 그 밖의 train 행 | c2 그대로 | c2 그대로 (증명 one-hot 행, 마스크) |
| val 행 (checkpoint 선택) | c2 그대로 | c2 그대로 (S200 목표) |
| 정책 보존 | μ 0.5, replay batch 512, 생성기 seed + 1000003 | 같음 |
| 학습량·선택 규칙 | c2 설정, guard + cycle val objective | 같음 |
| 수치 설정·진입점 | A1의 결정론 설정, `execute` | 같음 |
| 학습 seed | 20261024 / 20261111 / 20261112 | 같음 (seed마다 짝) |

- val 행과 선택 지표를 대조군과 같게 두는 이유: checkpoint 선택 기준이 달라지면, 점수 차이가 목표 변경에서 왔는지 선택에서 왔는지 나눌 수 없다.
- 같은 seed의 두 정책은 cycle batch 순서, D4 대칭, 학습률, replay 국면이 같다. 행 집합과 순서도 같게 유지해야 한다(§2).

## 2. 후보의 학습 자료 만드는 규칙

**재탐색 범위** (권장: 방식 B)

| | 방식 A: 전체 재탐색 후 label 다시 생성 | 방식 B: train 범위 재탐색 후 목표만 교체 |
|---|---|---|
| 재탐색 | 20,910국면, 약 4.4시간 | 18,965국면, 약 4.0시간 |
| 만드는 방법 | `h7c_labels --retarget`으로 행 파일 전체를 새로 만든다 | c2 행 파일에서 train의 방문 목표 행만 S400 목표로 바꾼다 |
| val 행 | S400으로 바뀐다 (선택 지표가 달라짐) | c2 그대로 |
| 기존 도구 | `h7c_retarget run`은 c1 결정에서 factor 4를 읽어 S400을 만들 수 없다. 별도 진입점이 필요하다 | 별도 진입점과 행 교체 도구가 필요하다 |

- **방식 B로 정했다.** §1의 "val과 선택 지표를 같게" 조건을 그대로 지킨다.

**행 집합 고정** (방식 B)
- 후보의 train 행 집합과 순서는 c2 train 행과 **같다**. 바뀌는 것은 방문 목표 행의 `target` 값뿐이다.
- 목표 규칙은 c2와 같다: `h7c_retarget.retarget_target`. S400 방문 수에서 actor와 S400의 증명된 패배수 합집합을 뺀다.
- 한쪽에서만 목표가 남는 경우(표본에서는 0건이었다. 전체 자료에서는 생길 수 있다)
  - c2에 행이 있는데 S400 목표에 질량이 없으면: 그 행은 S200 목표를 그대로 둔다. 건수를 기록한다. 행을 빼면 행 수가 달라져 batch 순서가 대조군과 어긋나기 때문이다.
  - c2에 행이 없는데 S400 목표가 생기면: 행을 추가하지 않는다. 건수를 기록한다.
- 마스크는 c2 행의 마스크를 그대로 따른다.
- S400 재탐색 기록에는 S400-PRE와 같은 검사를 적용한다: 적격 국면 coverage, 행 검사, 유한값, 이어받은 기록 재검사.
- S400 재탐색 앞에, S400-PRE 표본의 처음 5게임(74국면)으로 S200 재현을 다시 확인한다(G0 규칙). S400-PRE에서 2,437국면이 정확히 재현됐으므로 짧은 확인만 둔다.

**출처 기록과 학습 전 검사** (검토 반영)
- A1의 `execute`는 넘겨받은 행으로 학습하지만, 기록에는 c2 행 경로를 적는다. 후보 행만 바꿔 넘기면 기록에 원본 경로가 남을 수 있다. 그래서 새 진입점을 둔다.
- 새 진입점은 학습 기록과 checkpoint 메타데이터에 세 SHA를 함께 남긴다: 원본 c2 행 파일, 후보 행 파일, S400 재탐색 기록 파일.
- 학습 전에 후보 행 파일을 c2 행 파일과 대조한다. 다음이 모두 같아야 한다.
  - 행 수와 순서, `game_id`·`ply`·`split`·`kind`·마스크·플래그
  - val 행 전체(목표 포함)
  - train의 증명 one-hot 행 전체
  - train 방문 목표 행에서 바뀌는 것은 `target`과 그 출처 필드뿐이다.
  - 바뀐 행의 목표는 S400 재탐색 기록에서 `retarget_target`으로 다시 계산한 값과 같아야 한다.
- 교체 통계를 보고한다: train 방문 목표 행 수, S400 목표로 교체한 행 수와 비율, S200 유지(fallback) 건수와 비율, 추가하지 않은 S400 전용 목표 건수.

## 3. 학습 단계의 gate (순서: S400 재탐색보다 먼저)

- 대조군 T는 A1의 checkpoint와 기록을 SHA로 그대로 쓴다. 다시 학습하지 않는다.
- 다만 현재 환경이 A1과 같은 계산을 하는지 먼저 확인한다: **T(20261024)를 같은 진입점으로 다시 학습해 A1 기록과 정확히 같아야 한다**(지문, 학습 기록, 선택 state 해시, 고정 국면 log-확률 해시. 허용 오차 0).
  - 이 확인에는 S400 자료가 필요 없으므로, 약 4시간의 재탐색보다 먼저 한다. 실패하면 재탐색을 하지 않는다.
  - 통과하면 S200 재현 확인(5게임) → S400 재탐색 → 후보 행 생성과 검사 → 후보 3 seed 학습 순서로 간다.
  - A1 gate에서 이 계산이 비트 단위로 재현됐다(`b3eb145`).
- 후보 학습은 A1과 같은 `execute`, 같은 결정론 설정, 같은 gate 수치 설정 요구를 쓴다. 입력 행 파일만 바뀐다.

## 4. 평가 (Arena)

- 비교 표본: **새 개발 opening suite 두 묶음, 각 300쌍** (§5)
  - A1은 O0·O1의 opening을 300개씩 모두 썼다. 같은 suite에서 seed만 바꾸면 opening 집합은 같고 순서와 대국 난수만 달라진다.
  - O0 규칙은 가능한 수순이 최대 552개라 이미 쓴 300개를 빼면 새 300개를 확보할 수 없다. 그래서 두 묶음 모두 O1 계열(5수) 규칙으로 만든다.
- arm: 각 묶음에서 같은 opening·seed·pairing으로
  - 후보(20261024) 대 champion
  - A1 T(20261024) 대 champion (A1의 checkpoint를 새 표본에서 다시 둔다. 그래서 짝 비교가 유지된다)
- 판독
  - **D′ = 후보 − T** (쌍 대응): 목표 변경의 효과
  - **P′ = 후보 대 champion**: 다음 단계(독립 확인)로 넘어가는 기준. 같은 S50 예산이다.
  - 판독 규칙은 A1과 같은 형식으로 한다(suite 층화, 50:50, 쌍 대응, 단측 α 0.025, class 단위 민감도).
- 검출력은 새 표본 구성에 맞춰 다시 추정한다. 두 묶음이 모두 O1 계열(거의 opening마다 다른 동치류)이므로, A1 O1 arm의 쌍 점수 표준편차(T 0.256, C 0.264)를 쓰면 P′의 표준오차는 약 0.5 × √(2 × 0.256²/300) ≈ 1.04%p다. 이때 우위 판독에 필요한 관측 점수차는 약 +2.05%p, 검출력 80%에 필요한 실제 효과는 약 +2.93%p다. suite가 고정되면 다시 계산한다.
  - 이 값들은 **계획값이다. 판정 문턱으로 쓰지 않는다.** 판정은 계약에 고정한 bootstrap 하한으로만 한다.
- **보조 arm은 두지 않는다.** 첫 Arena는 사전 지정 seed 20261024의 본 비교에 집중한다.

## 5. 새 개발 suite와 seed 예약

**새 개발 suite** (O1 계열, 300개씩 두 묶음)
- 규칙: O1과 같다(5수: 중앙, 백·흑 반경 2, 백·흑 반경 3. 백 차례).
- 겹침 검사(동치류 단위): 기존 O0·O1·O2, 확인용 `O1-confirm-main`·`O1-confirm-anchor`, 지금까지의 Arena 기록이 쓴 opening과 같은 동치류는 뽑지 않는다. 두 묶음 사이에도 같은 동치류가 없어야 한다.
  - 지금 `h7r_openings.build_suite`는 수순(history) 단위로만 제외한다. 동치류 단위 제외를 기본값이 꺼진 선택 인자로 더하고, 기존 suite(O0/O1/O2와 확인 suite)가 같은 seed에서 똑같이 다시 만들어지는지 테스트한다.
  - 3수 opening과 5수 opening은 돌 수가 달라 동치류가 겹칠 수 없다. 그래서 5수 opening이 기존 3수 opening의 연장인 경우(prefix 겹침)는 건수만 기록한다.
- 생성 seed는 대국 seed와 따로 관리한다. 기존 suite처럼 날짜형 값(예: 202611131, 202611132)을 쓰고, 다른 suite의 생성 seed와 겹치지 않음을 확인한다.

**Arena 대국 seed 예약 v2** (권장안. 결정 필요)
- 사정: H7-R 블록 8781–8799는 모두 사용 중이거나 예약되어 있다. 새 블록을 쓰려면 `h7r_eval.SEED_BLOCK`과 `hybrid.h6_labels.RESERVED_SEEDS`가 그 값을 받아들여야 한다.
  - 그런데 두 파일은 A1, P1, D1–D4, S400-PRE 계약이 SHA로 고정한 source다. 고치면 이 계약들의 source 검증이 지금 checkout에서 실패한다. 기록은 그대로 남지만, 검증을 다시 할 수 없게 된다.
- 권장안(B): 두 파일은 고치지 않고 버전을 분리한다.
  - 새 모듈 `hybrid/h7r_seeds.py`에 v2 예약을 둔다. 대국 seed 블록 9101–9140(후보), opening 생성 seed 목록, 등록 날짜와 용도
  - v2 블록은 다음과 겹치지 않음을 테스트로 확인한다: `RESERVED_SEEDS` 전체, self-play 학습 run seed 8801–8999와 측정용 8800, S1 probe 기본 seed 9201부터, 커밋된 모든 Arena 계약의 seed
  - 새 평가 진입점 `scripts/h7r_eval_v2.py`는 v2 계약의 seed를 v2 예약으로 검사하고, 그 밖의 계약 검사·대국·arm 기록 검증은 기존 `h7r_eval`을 그대로 쓴다. 기존 `check_contract`가 v1 블록을 직접 확인하므로, v2 진입점 안에서만 seed 규칙을 v2로 바꿔 적용하고 끝나면 되돌린다.
  - 테스트: 커밋된 기존 계약이 모두 기존 경로로 계속 통과, v2 계약은 기존 경로에서 거부, 고정 source를 쓰는 계약들의 source 검증 계속 통과, v2 seed가 학습 run seed로 쓰이면 self-play가 거부(8801–8999 밖)
- 다른 안(A): 두 파일에 새 블록을 직접 더한다. 가장 단순하지만 위 계약들의 source 검증이 깨진다.

## 6. 결정 (2026-11-13)

1. seed 예약 방식: **B(버전 분리)**. 새 seed 정책은 v2 계약을 검사하는 경로에만 적용한다. 기존 검사기의 범위를 바꾸거나 다시 묶지 않는다.
2. v2 대국 seed 블록 9101–9140, opening 생성 seed 202611131·202611132
3. 새 suite는 내부에서도 동치류당 opening 하나로 구성한다(300쌍 = 서로 다른 300개 동치류).

## 7. 순서와 비용 (결정 뒤)

1. seed 예약 v2와 새 개발 suite 도구·테스트 → suite 두 묶음 생성(CPU, 클라우드에서 가능)과 겹침 검사 기록 → 커밋 (별도 승인)
2. R2-S400-TRAIN 계약 고정(미승인), 도구·테스트
3. prepare (입력 대조)
4. T(20261024) 재현 gate (S400 자료 없이)
5. S200 재현 확인(5게임) → S400 재탐색: train 범위 18,965국면, 약 4.0시간(비율법 추정)
6. 후보 행 파일 생성과 학습 전 검사(출처 SHA, 동일성, 교체 통계)
7. 후보 3 seed 학습
8. Arena 계약 생성(새 suite, v2 seed), Arena(본 arm 4개, 각 300쌍), judge

- 단계마다 따로 승인한다. 4단계를 통과하기 전에는 5단계의 재탐색을 시작하지 않는다.

## 8. 진행 기록

### 8.1 seed 예약 v2와 새 개발 suite (2026-11-13)

**seed 예약 v2** (`src/hybrid/h7r_seeds.py`)
- Arena 대국 seed 9101–9140, opening 생성 seed 202611131(묶음 a)·202611132(묶음 b). 등록일 2026-11-13
- 충돌 없음을 테스트로 확인했다: 기존 예약(`RESERVED_SEEDS`) 전체, 학습 run seed 8801–8999, 측정 seed 8800, S1 probe 기본 seed 9201, 테스트 fixture seed(8795·8796), 커밋된 모든 기록의 seed 값
- `h7r_eval.py`와 `h6_labels.py`는 바꾸지 않았다. 이 파일들을 고정한 P1, A1, S400-PRE 계약은 지금 checkout에서도 source 검증을 통과한다(테스트).

**v2 계약 검사** (`src/hybrid/h7r_eval_v2.py`)
- 형식 `h7r-eval-contract-v2`, `seed_policy: h7r-seeds-v2`, 개발 계약(purpose pilot)만 받는다.
- 모델 신원, arm(이름·모델·budget·쌍 수·suite SHA), 공유 seed의 pairing 규칙, 통계 블록은 v1과 같은 기준으로 검사하고, seed만 v2 블록으로 검사한다.
- v1 검사기는 v2 계약을, v2 검사기는 v1 계약을 거부한다. v2 검사 뒤에도 v1의 `SEED_BLOCK`은 그대로다(테스트).
- seed와 계약 id는 v1(`docs/h7r/contracts`)과 v2(`docs/h7r/contracts-v2`) 계약 전체에서 겹치지 않아야 한다.
- 같은 pairing의 arm은 하나의 표본을 뽑아야 한다: seed, 쌍 수, suite 경로와 SHA가 모두 같아야 한다(D′가 opening마다 짝지어진다). 쌍 수는 suite의 opening 수 이내여야 한다(검토 반영, 회귀 테스트).
- v2 대국 실행 진입점은 Arena 단계에서 만든다. 기존 `scripts/h7r_eval.py play`는 내부에서 v1 계약 검사기를 부르고, A1 계약이 그 파일을 SHA로 고정했기 때문이다. v2 진입점은 v2 검사를 거친 계약으로 기존 대국·arm 검증 함수(`build_tasks`, `arm_identity`, `check_games`, `load_arm`의 검사)를 재사용한다.

**학습 유입 차단** (seed 범위와는 별개의 검사)
- v2 seed는 학습 run seed 범위 밖이라 self-play가 run seed로 거부하고, holdout run seed로도 거부한다(테스트).
- v2 Arena의 대국 기록(journal 행)과 arm 기록은 label 단계(`h7c_labels.check_games`)에서 거부된다. label 단계는 깨끗한 commit의 생성기가 만든 self-play 게임만 받는다(테스트).

**새 개발 suite** (`scripts/h7r_dev_suites.py`, 생성 commit `43106d6`, clean)
- `docs/h7r/suites/O1-dev-a.json`(seed 202611131), `O1-dev-b.json`(seed 202611132), 기록 `O1-dev-record.json`
- 규칙 O1(5수), 각 300개 opening이 서로 다른 300개 동치류다.
- 제외한 동치류 1,949개(SHA `8837e7ae0de0…`). 출처와 SHA:
  - suite: O0, O1, O2, O1-confirm-main, O1-confirm-anchor
  - 계약 arm 표본: R1-PILOT-1(1,020개), R1-SMOKE-1(3개), R2-P1-A1-ARENA-1(1,300개). arm 표본이 없는 R1-HDEV-1, R1-SMOKE-HOLDOUT은 기록만 했다.
- 겹침: 두 묶음 모두 모든 출처와 공유하는 동치류가 0이고, 두 묶음 사이에도 동치류·수순 겹침이 0이다.
- prefix 겹침(처음 3수가 이전에 쓴 3수 opening과 같은 경우): 묶음 a 201개, b 207개. O0 규칙의 3수 수순이 최대 552가지라 생기는 일이며, 5수 opening의 동치류는 겹치지 않는다. 계획대로 건수만 기록한다.
- `build_suite`의 새 인자는 기본값이 꺼져 있다. 기존 O0·O1·O2를 같은 seed로 다시 만들면 생성 함수의 출력이 커밋된 파일과 같다(테스트. 파일에만 있는 `git`, `overlap_with_earlier` 필드 제외).
- `scripts/h7r_dev_suites.py check`가 커밋된 파일에서 제외 목록, 두 suite의 재생성 결과, 겹침을 모두 다시 계산해 기록과 대조한다.
  - 출처는 생성 당시 기록된 경로와 SHA로 다시 읽는다. 생성 뒤에 추가된 계약(이 suite를 쓰는 Arena 계약 등)은 생성 출처가 아니다. 기록된 출처 파일이 바뀌면 멈춘다(검토 반영).
  - 새 suite를 쓰는 v2 Arena 계약이 함께 있는 상태에서도 suite 검사와 계약 검사가 함께 통과함을 테스트로 확인했다.

### 8.2 학습 도구·테스트·계약 (미승인)

- 도구 `scripts/h7r_s400_train.py`, 테스트 `tests/test_h7r_s400_train.py`
- 계약 `docs/h7r/r2/R2-S400-TRAIN/s400-train-contract.json`(초안 `s400-train-contract.draft.json`), 실험 SHA `67c6ba3c8081…`(검토 반영 뒤 다시 고정. 이전 고정본 `6c579b5ba682…`, `831d74d33cc6…`은 실행되지 않았다). 단계 prepare → gate → search → train → arena, 모두 false
- 그대로 가져온 것(테스트가 A1 계약과 같음을 확인): 입력, arm T 정의와 replay, 학습량, 결정론 설정, step 구조, 학습 seed 3개. 대조군 T는 A1 seed 기록이 이름 붙인 checkpoint를 SHA로 쓴다.

| 명령 | 단계 | 하는 일 |
|---|---|---|
| `prepare` | prepare | A1 입력(`h7r_p1.check_inputs`), c2 self-play·S200 기록·탐색 모델, A1 T checkpoint 대조 |
| `gate` | gate | T(20261024)를 A1 gate의 수치 설정으로 A1과 같은 진입점(`h7r_p1a1.execute`)에서 다시 학습해, 커밋된 A1 seed 기록과 정확히 같아야 한다(지문, 학습 기록, P1 기록, 설정, state·log-확률 해시). S400 탐색보다 먼저 |
| `search` | search | S400-PRE 표본의 처음 5게임으로 S200 재현(G0 규칙) → 통과하면 train 범위(936게임, 18,965국면) S400 탐색. 기록 purpose `r2-s400-train`(label 단계가 거부), 이어받은 기록도 같은 검사 |
| `rows` | search | 후보 행 파일 생성(§2 규칙)과 검사. 교체 통계(교체 수·비율, S200 유지 수·비율, 추가하지 않은 S400 전용 목표 수) |
| `train --seed S` | train | 커밋된 gate와 rows를 다시 판정하고, 후보 파일 SHA와 행 검사를 다시 확인한 뒤 학습. 기록과 checkpoint 메타데이터에 c2 행·후보 행·S400 기록 SHA |
| `arena-contract` | arena | v2 Arena 계약(O1-dev-a·b, 300쌍, seed 9101·9102, suite마다 S와 T 한 pairing) |

- 후보 행 파일은 gzip mtime을 0으로 고정해, 같은 행이면 같은 SHA가 나온다.
- 검토 반영(2026-11-13)
  - `train`은 변경 감시를 먼저 시작하고, `verified_rows`가 SHA를 확인한 바로 그 바이트에서 읽은 행 객체를 그대로 학습에 넘긴다(다시 읽지 않는다). 감시 시작 시점의 후보 파일 SHA가 rows 기록과 다르면 멈추고, 실행 중에 바뀌면 아무것도 쓰지 않는다.
  - `arena-contract`는 seed·계약 id 중복 검사까지 마친 뒤에 파일을 쓴다(`h7r_eval_v2.check_unique`). 충돌하면 파일이 남지 않는다.
  - S400 기록 로더는 범위의 모든 게임이 정확히 한 번씩 있고 범위 밖 게임이 없어야 받는다. `verified_rows`는 S400 기록 SHA를 rows 기록과 search 기록 양쪽에 대조한다.
- 테스트(11개)
  - 행 규칙: train 방문 목표만 바뀜, S400 질량이 없으면 S200 유지, 행 추가 없음, 다른 변경은 모두 거부(행 수·순서·마스크·val·증명 행·목표 값·fallback 표시)
  - 실제 파일로 rows → 학습 전 재검사 → 후보 파일 변조 거부, 빈 S400 기록 거부
  - gate: 정확 일치만 통과(손실 1e-12 차이, 스레드 수 차이도 거부), 통과 기록이라도 내용이 A1과 다르면 재판정에서 거부, A1 gate 수치 설정 요구
  - search: S200 재현이 다르면 S400을 시작하지 않고 기록
  - train: 후보 행으로 학습하고 세 SHA를 기록
  - Arena 계약: v2 검사기 통과, 새 suite 검사와 함께 통과
- 학습과 엔진 탐색은 테스트에서 대체했다. 실제 학습 경로는 A1 gate(`b3eb145`)에서, 실제 탐색 경로는 S400-PRE(`3608bb8`)와 그 엔진 테스트에서 이미 확인한 함수를 그대로 쓴다.

**남은 준비** (해당 단계 승인 전에)
- 단계별 데스크톱 블록(prepare, gate, search+rows, train, Arena)
- v2 Arena 실행 진입점과 판독(D′, P′): v2 검사를 거친 계약으로 기존 대국·arm 검증 함수를 재사용

### 8.3 prepare: PASS (`bf25f74`, 2026-11-14)

- 실행 HEAD `b4fb0ca`(clean, prepare 승인 커밋), 실험 `67c6ba3c8081`, 계약 파일 SHA `d3f54b5ae740`. 로그는 manifest SHA와 같고 `__EXIT__ 0`, 결과에 로그 이름이 남아 있다.
- 입력 파일 18개의 SHA: A1 prepare의 입력 8개(c2 행, H3·c5 checkpoint·metadata, RenjuNet cache)와 같고, 탐색 입력 6개(c2 self-play, c2 S200 기록, 탐색 모델)와 A1 T checkpoint·metadata 6개는 계약과 같다.
- c2 행: 마스크되지 않은 train 24,077행, val 1,423행
- 대조군: A1 T 3개(20261024, 20261111, 20261112)의 checkpoint·metadata SHA가 각 A1 seed 기록과 같다.
- 재탐색 범위: 936게임, 적격 18,965국면(S400-PRE prepare의 학습 범위와 같다). S200 초기 재현 5게임은 S400-PRE 표본의 처음 5게임과 같다.

### 8.4 gate: PASS (`2e894cf`, 2026-11-14)

- 실행 HEAD `5686713`(clean, gate 승인 커밋), 실험 `67c6ba3c8081`(변하지 않음), 계약 파일 SHA `4ff14ab1f695`. 로그 `gate-2HbG6QKO.log`는 manifest SHA(`463bc99ba309`)와 같고 `__EXIT__ 0`, 결과의 `log_name`과 같다.
- 판정: `passed: true`, `differences: []`. cloud에서 커밋된 A1 T-20261024 seed 기록과 다시 비교해도 차이가 0이다(`a1.differences`). A1 seed 기록 SHA도 계약과 같다.
- 재현된 값: 초기 상태 `093393…`, 선택 상태 `58429752f6d9`(A1과 같음), 고정 국면 5,519개 log-prob SHA `6540c228a72b`, 96 step 중 step 96 선택(`CANDIDATE`), val objective 1.9812, guard top-1 0.5178(H3 0.5241, 하락 0.0063), 학습 24,077행 / val 1,423행.
- 실행 설정: A1 gate 설정과 같다(torch 2.14.0+cu130, CUDA 13.0, cuDNN 92400, RTX 5070, deterministic algorithms, cuDNN deterministic, benchmark off, TF32 matmul off, threads 8, `CUBLAS_WORKSPACE_CONFIG=:4096:8`). 시간은 108.0초(A1 91.1초); 시간은 비교 항목이 아니다.
- cloud에는 `runs/`의 A1 T checkpoint가 없어서 `verified_gate`의 checkpoint 파일 SHA 검사만 계약 SHA로 대신했다. 데스크톱의 search·train은 시작할 때 이 검사를 실제 파일로 다시 한다.

### 8.5 search·rows 블록 준비 (search 미승인, 2026-11-14)

- 로그 연결 방식: sidecar 실행 기록(§9.1, `scripts/h7r_attempts.py`). 학습 도구와 계약은 바꾸지 않았다. 실험 SHA는 `67c6ba3c8081`로 그대로이고, 기존 prepare·gate 기록이 그대로 이어진다. 승인 플래그는 prepare·gate만 true이다.
- 블록 테스트(`tests/test_h7r_s400_train.py`의 `BlockTest`, 실제 bash와 임시 저장소, 도구는 대역)가 다루는 경우:
  - 정상 완료(search 뒤 rows)
  - S200 재현 실패(exit 2 기록, rows는 아무것도 쓰지 않고 멈춤)
  - 스테이징 뒤 중단(다시 돌리지 않고 커밋)
  - 결과를 쓴 뒤 종료 줄 전에 중단(열린 실행 기록에서 복구)
  - 결과 전에 중단(다음 실행이 닫고 함께 기록)
  - 후보 gzip만 남은 중단(프로세스 종료, Guard 실패 exit 1 두 경우: 파일 유지, 안내 후 멈춤, 틀린 SHA 거부, 확인 뒤 옮기고 다시 생성, 같은 파일 확인)
  - 쓰다 만 `.tmp`
  - 같은 단계 블록의 동시 실행
  - 결과가 SHA로 가리키는 출력 파일 대조(검토 P2 반영, 2026-11-14): 정상 완료 뒤 출력이 바뀐 경우(`finish`가 성공을 기록하지 않고 멈춤, 다음 실행의 복구도 거부), 커밋 전 복구 때 출력이 없어진 경우(`recover`가 멈추고 커밋·푸시하지 않음). search와 rows 각각.

## 9. 데스크톱 실행 (승인 뒤에만)

- s5-1은 `approved.prepare` 뒤, s5-G는 `approved.gate` 뒤, s5-S와 s5-R은 `approved.search` 뒤에 실행한다(rows 명령도 search 승인으로 돈다). train, Arena 블록은 해당 단계 승인 전에 추가한다.
- 블록은 R2-S400-PRE의 블록과 같은 방식이다.
  - 결과(`prepare.json`, `gate.json`)에 실행 로그 이름(`log_name`)을 남긴다.
  - 다시 실행할 때: 결과가 HEAD에 커밋되어 있으면 멈춘다. 결과가 커밋 전이면(스테이징만 된 경우 포함) 다시 돌리지 않고, 결과가 뜻하는 종료 코드와 로그의 `__EXIT__`이 같은지 확인한 뒤 함께 커밋한다. 로그가 없거나 두 종료 코드가 다르면 멈춘다.
  - 이 블록이 기록할 파일이 아닌 것이 스테이징되어 있으면 멈춘다.
- s5-G는 프로세스 시작 전에 `CUBLAS_WORKSPACE_CONFIG=:4096:8`을 설정한다. gate가 FAIL(종료 코드 2)이면 결과와 로그를 커밋하고 멈춘다. 재탐색으로 넘어가지 않는다.
- gate 예상 시간: A1 gate의 T 실행(약 100초)과 같은 규모다.

s5-1. prepare

```bash
(
  set -euo pipefail
  cd ~/renju-h6
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src:$PWD/scripts"
  [ "$(git branch --show-current)" = feat/h7-c1 ] || { echo "STOP: wrong branch"; exit 1; }
  git diff --quiet || { echo "STOP: unstaged tracked changes"; exit 1; }
  if git diff --cached --quiet; then git pull --ff-only origin feat/h7-c1; fi
  N=prepare
  D=docs/h7r/r2/R2-S400-TRAIN
  L=runs/h7r/R2-S400-TRAIN/logs
  O="$D/$N.json"
  tracked(){ git cat-file -e "HEAD:$1" 2>/dev/null; }
  if tracked "$O"; then echo "STOP: $N is already recorded"; exit 1; fi
  mkdir -p "$L"
  rc=0
  if [ ! -e "$O" ]; then
    git diff --cached --quiet || { echo "STOP: resolve staged changes before a new run"; exit 1; }
    python scripts/check_track_b_frozen.py
    log="$(mktemp "$L/$N-XXXXXXXX.log")"
    set +e
    python -u scripts/h7r_s400_train.py "$N" --device cuda --log-name "$(basename "$log")" 2>&1 | tee "$log"
    codes=("${PIPESTATUS[@]}")
    set -e
    rc="${codes[0]}"
    [ "${codes[1]}" -eq 0 ] || { echo "STOP: tee failed; log: $log"; exit 1; }
    echo "__EXIT__ $rc" >> "$log"
  else
    echo "an uncommitted $O exists: recording it (no new run)"
  fi
  if [ -f "$O" ]; then
    result="$(python - "$O" "$N" <<'PY'
import json, re, sys
from pathlib import Path
r = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
name = r.get('log_name', '')
if r.get('format') != 'h7r-s400-train-prepare-v1' or not re.fullmatch(sys.argv[2] + r'-[A-Za-z0-9]+\.log', name):
    sys.exit('STOP: invalid result or missing attempt log name')
print(name, 0 if r.get('passed') is True else 1)
PY
)"
    read -r name rc <<< "$result"
    log="$L/$name"
    [ -f "$log" ] || { echo "STOP: the result's log is missing: $log"; exit 1; }
    last="$(tail -n 1 "$log")"
    case "$last" in
      "__EXIT__ $rc") ;;
      __EXIT__*) echo "STOP: the result and its log disagree on the exit code"; exit 1 ;;
      *) printf '\n%s\n__EXIT__ %s\n' "Recovered exit code from $O" "$rc" >> "$log" ;;
    esac
  elif [ "$rc" -eq 0 ] || [ "$rc" -eq 2 ]; then
    echo "STOP: $N did not produce a result"; exit 1
  fi
  paths=()
  shopt -s nullglob
  for f in "$L/$N"-*.log; do
    [[ "$(tail -n 1 "$f")" =~ ^__EXIT__\ [0-9]+$ ]] || continue
    n="$(basename "$f" .log)"
    tracked "$D/stage-logs/$n.log" && continue
    python scripts/h7r_d4_records.py copy-logs --resume --src-dir "$L" --pattern "$n.log" \
      --dest-dir "$D/stage-logs" --manifest "$D/stage-logs/$n.manifest.json"
    paths+=("$D/stage-logs/$n.log" "$D/stage-logs/$n.manifest.json")
  done
  if [ -f "$O" ]; then paths+=("$O"); fi
  [ "${#paths[@]}" -gt 0 ] || { echo "STOP: nothing to record"; exit 1; }
  while IFS= read -r -d '' staged; do
    allowed=false
    for f in "${paths[@]}"; do if [ "$f" = "$staged" ]; then allowed=true; break; fi; done
    "$allowed" || { echo "STOP: unrelated staged path: $staged"; exit 1; }
  done < <(git diff --cached --name-only -z)
  expect=()
  for f in "${paths[@]}"; do expect+=(--expect "$f"); done
  git add -- "${paths[@]}"
  python scripts/h7r_d4_records.py verify-staged "${expect[@]}"
  git commit -m "H7-R S400-TRAIN: $N exit $rc (records only)"
  git push origin feat/h7-c1
  echo "__RECORDED__ $rc"
  exit "$rc"
)
```

s5-G. T 재현 gate

```bash
(
  set -euo pipefail
  cd ~/renju-h6
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src:$PWD/scripts"
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
  [ "$(git branch --show-current)" = feat/h7-c1 ] || { echo "STOP: wrong branch"; exit 1; }
  git diff --quiet || { echo "STOP: unstaged tracked changes"; exit 1; }
  if git diff --cached --quiet; then git pull --ff-only origin feat/h7-c1; fi
  N=gate
  D=docs/h7r/r2/R2-S400-TRAIN
  L=runs/h7r/R2-S400-TRAIN/logs
  O="$D/$N.json"
  tracked(){ git cat-file -e "HEAD:$1" 2>/dev/null; }
  if tracked "$O"; then echo "STOP: $N is already recorded"; exit 1; fi
  tracked "$D/prepare.json" || { echo "STOP: prepare is not recorded (run s5-1 first)"; exit 1; }
  mkdir -p "$L"
  rc=0
  if [ ! -e "$O" ]; then
    git diff --cached --quiet || { echo "STOP: resolve staged changes before a new run"; exit 1; }
    python scripts/check_track_b_frozen.py
    log="$(mktemp "$L/$N-XXXXXXXX.log")"
    set +e
    python -u scripts/h7r_s400_train.py "$N" --device cuda --log-name "$(basename "$log")" 2>&1 | tee "$log"
    codes=("${PIPESTATUS[@]}")
    set -e
    rc="${codes[0]}"
    [ "${codes[1]}" -eq 0 ] || { echo "STOP: tee failed; log: $log"; exit 1; }
    echo "__EXIT__ $rc" >> "$log"
  else
    echo "an uncommitted $O exists: recording it (no new run)"
  fi
  if [ -f "$O" ]; then
    result="$(python - "$O" "$N" <<'PY'
import json, re, sys
from pathlib import Path
r = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
name = r.get('log_name', '')
if r.get('format') != 'h7r-s400-train-gate-v1' or not re.fullmatch(sys.argv[2] + r'-[A-Za-z0-9]+\.log', name):
    sys.exit('STOP: invalid result or missing attempt log name')
print(name, 0 if r.get('passed') is True else 2)
PY
)"
    read -r name rc <<< "$result"
    log="$L/$name"
    [ -f "$log" ] || { echo "STOP: the result's log is missing: $log"; exit 1; }
    last="$(tail -n 1 "$log")"
    case "$last" in
      "__EXIT__ $rc") ;;
      __EXIT__*) echo "STOP: the result and its log disagree on the exit code"; exit 1 ;;
      *) printf '\n%s\n__EXIT__ %s\n' "Recovered exit code from $O" "$rc" >> "$log" ;;
    esac
  elif [ "$rc" -eq 0 ] || [ "$rc" -eq 2 ]; then
    echo "STOP: $N did not produce a result"; exit 1
  fi
  paths=()
  shopt -s nullglob
  for f in "$L/$N"-*.log; do
    [[ "$(tail -n 1 "$f")" =~ ^__EXIT__\ [0-9]+$ ]] || continue
    n="$(basename "$f" .log)"
    tracked "$D/stage-logs/$n.log" && continue
    python scripts/h7r_d4_records.py copy-logs --resume --src-dir "$L" --pattern "$n.log" \
      --dest-dir "$D/stage-logs" --manifest "$D/stage-logs/$n.manifest.json"
    paths+=("$D/stage-logs/$n.log" "$D/stage-logs/$n.manifest.json")
  done
  if [ -f "$O" ]; then paths+=("$O"); fi
  [ "${#paths[@]}" -gt 0 ] || { echo "STOP: nothing to record"; exit 1; }
  while IFS= read -r -d '' staged; do
    allowed=false
    for f in "${paths[@]}"; do if [ "$f" = "$staged" ]; then allowed=true; break; fi; done
    "$allowed" || { echo "STOP: unrelated staged path: $staged"; exit 1; }
  done < <(git diff --cached --name-only -z)
  expect=()
  for f in "${paths[@]}"; do expect+=(--expect "$f"); done
  git add -- "${paths[@]}"
  python scripts/h7r_d4_records.py verify-staged "${expect[@]}"
  git commit -m "H7-R S400-TRAIN: $N exit $rc (records only)"
  git push origin feat/h7-c1
  echo "__RECORDED__ $rc"
  exit "$rc"
)
```


### 9.1 search와 rows: 실행 기록(sidecar)

- search와 rows의 결과(`search.json`, `rows.json`)에는 로그 이름이 없다. 학습 도구의 소스는 계약에 SHA로 고정되어 있어서(실험 SHA `67c6ba3c8081`), 도구를 바꾸지 않고 블록이 로그 옆에 실행 기록을 따로 남긴다(`scripts/h7r_attempts.py`; 계약의 sources가 아니므로 실험 SHA와 기존 prepare·gate 기록은 그대로다).
  - 실행 전(`start`): 단계, 계약의 실험 SHA와 파일 SHA, HEAD, 로그 경로, 결과 경로, 해시할 출력 파일을 `runs/h7r/R2-S400-TRAIN/logs/<단계>-XXXXXXXX.attempt.json`에 쓴다.
  - 실행 뒤(`finish`): 로그에 `__EXIT__ <코드>`를 쓴 다음 종료 코드, 로그 SHA, 결과 SHA, 출력 파일 SHA를 더한다. 결과는 그 기록의 실험 SHA·HEAD와 같아야 하고, 결과가 뜻하는 종료 코드(search: `SEARCHED` 0, `S200_NOT_REPRODUCED` 2; rows: 0)가 실제 종료 코드와 같아야 한다. 종료 코드 0이나 2인데 결과가 없으면 멈춘다.
  - 결과가 SHA로 가리키는 출력 파일은 실제 파일과 대조한다. `search.json`(`SEARCHED`)의 `s400.records_sha256`은 `runs/h7r/R2-S400-TRAIN/s400.jsonl`, `rows.json`의 `candidate_rows_sha256`은 `runs/h7r/R2-S400-TRAIN/policy_rows_s400.jsonl.gz`의 SHA여야 한다. 그 파일은 실행 기록의 출력 파일이어야 하고, 없거나 SHA가 다르면 성공을 기록하지 않고 멈춘다(실행 기록은 열린 채 남는다). `S200_NOT_REPRODUCED`는 가리키는 출력이 없다.
  - 커밋: 실행 기록은 로그 사본, manifest와 함께 `stage-logs/`에 들어간다. 사본을 만들 때 로그 사본 SHA가 실행 기록의 로그 SHA와 같은지 확인한다.
- 다시 실행할 때:
  - 결과가 HEAD에 커밋되어 있으면 멈춘다.
  - 결과가 커밋 전이면(스테이징만 된 경우 포함) 다시 돌리지 않는다(`recover`). 결과 SHA를 기록한 실행이 있으면 결과가 가리키는 출력 파일을 실제 파일과 다시 대조한 뒤 그 기록으로 커밋한다(그 사이 출력이 바뀌거나 없어졌으면 멈춘다). 없으면 결과와 실험 SHA·HEAD가 같은, 끝나지 않은 실행 기록이 정확히 하나여야 한다. 그 로그에 종료 줄이 없으면 결과가 뜻하는 코드를 `Recovered exit code`와 함께 붙이고, 있으면 두 코드가 같아야 한다. 아니면 멈춘다.
  - 결과가 없으면 새로 실행한다. 이전의 끝나지 않은 실행(중단)은 먼저 닫는다. 로그에 종료 줄이 없으면 `__INTERRUPTED__` 줄을 붙이고, 그때 남은 출력 파일 SHA를 남긴다. 닫힌 실행의 로그와 기록도 이번 실행과 함께 커밋한다(실패 기록 보존).
  - search 재탐색은 `runs/h7r/R2-S400-TRAIN/s200-check.jsonl`, `s400.jsonl`에서 이어서 한다(끝난 게임은 다시 돌리지 않고 읽어서 다시 검사한다).
  - 같은 단계의 블록이 이미 돌고 있으면 `flock`으로 멈춘다.
- rows 후보 파일만 남은 중단: rows는 후보 행(`runs/h7r/R2-S400-TRAIN/policy_rows_s400.jsonl.gz`)을 쓴 뒤 `rows.json`을 쓴다. 그 사이에 중단되면(프로세스 종료, 또는 Guard가 파일 변경을 잡아 exit 1), 도구는 이미 있는 후보 파일을 덮어쓰지 않으므로 다시 돌지 않는다.
  - s5-R은 이 상태를 보면 아무것도 바꾸지 않고 멈춘다. 후보 파일의 SHA 앞 12자리와 복구 절차를 출력한다.
  - 복구: `export S5_SET_ASIDE=<SHA 앞 12자리>` 뒤 s5-R을 다시 실행하고, 끝나면 `unset S5_SET_ASIDE`. 값이 그 파일의 SHA와 다르면 다시 멈춘다.
  - 블록은 그 파일을 `runs/h7r/R2-S400-TRAIN/interrupted/<새 실행 이름>.policy_rows_s400.jsonl.gz`로 옮긴다(삭제·덮어쓰기 없음). 새 실행 기록에 옮긴 경로, SHA, 그 파일을 남긴 실행 이름을 적고, 커밋된 S400 기록에서 후보 행을 다시 만든다.
  - gzip mtime이 0이라 같은 행이면 같은 파일이 된다. 새 실행 기록의 `same_as_set_aside`가 옮긴 파일과 같은지 남긴다. 판정은 바꾸지 않고 기록만 한다.
  - 쓰다 만 `policy_rows_s400.jsonl.gz.tmp`는 결과가 아니므로 확인 없이 같은 폴더로 옮겨 기록한다.
- s5-R은 search가 커밋되어 있고 그 상태가 `SEARCHED`일 때만 실행한다. `S200_NOT_REPRODUCED`이면 아무것도 쓰지 않고 멈춘다.
- search 예상 시간: S200 재현 5게임 뒤 S400 936게임(약 4.0시간, R2-S400-PRE 측정). tmux에서 실행한다. search는 R2-S400-PRE measure처럼 `CUBLAS_WORKSPACE_CONFIG`을 정하지 않는다(그 측정의 S200 재현과 같은 조건).

s5-S. S400 재탐색 (tmux 권장)

```bash
(
  set -euo pipefail
  cd ~/renju-h6
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src:$PWD/scripts"
  [ "$(git branch --show-current)" = feat/h7-c1 ] || { echo "STOP: wrong branch"; exit 1; }
  git diff --quiet || { echo "STOP: unstaged tracked changes"; exit 1; }
  if git diff --cached --quiet; then git pull --ff-only origin feat/h7-c1; fi
  N=search
  D=docs/h7r/r2/R2-S400-TRAIN
  R=runs/h7r/R2-S400-TRAIN
  L="$R/logs"
  O="$D/$N.json"
  A=scripts/h7r_attempts.py
  tracked(){ git cat-file -e "HEAD:$1" 2>/dev/null; }
  if tracked "$O"; then echo "STOP: $N is already recorded"; exit 1; fi
  for f in prepare gate; do tracked "$D/$f.json" || { echo "STOP: $f is not recorded"; exit 1; }; done
  mkdir -p "$L"
  command -v flock > /dev/null || { echo "STOP: flock is missing (util-linux)"; exit 1; }
  exec 9> "$L/$N.lock"
  flock -n 9 || { echo "STOP: another $N block is running"; exit 1; }
  if [ ! -e "$O" ]; then
    git diff --cached --quiet || { echo "STOP: resolve staged changes before a new run"; exit 1; }
    python scripts/check_track_b_frozen.py
    log="$(mktemp "$L/$N-XXXXXXXX.log")"
    python "$A" start --stage "$N" --log "$log" --result "$O" --output "$R/s200-check.jsonl" --output "$R/s400.jsonl"
    set +e
    python -u scripts/h7r_s400_train.py "$N" --device cuda 2>&1 | tee "$log"
    codes=("${PIPESTATUS[@]}")
    set -e
    rc="${codes[0]}"
    [ "${codes[1]}" -eq 0 ] || { echo "STOP: tee failed; log: $log"; exit 1; }
    echo "__EXIT__ $rc" >> "$log"
    python "$A" finish --attempt "${log%.log}.attempt.json" --rc "$rc"
  else
    echo "an uncommitted $O exists: recording it (no new run)"
    result="$(python "$A" recover --stage "$N" --log-dir "$L" --result "$O")"
    read -r name rc <<< "$result"
  fi
  paths=()
  shopt -s nullglob
  for a in "$L/$N"-*.attempt.json; do
    n="$(basename "$a" .attempt.json)"
    tracked "$D/stage-logs/$n.attempt.json" && continue
    python scripts/h7r_d4_records.py copy-logs --resume --src-dir "$L" --pattern "$n.log" \
      --dest-dir "$D/stage-logs" --manifest "$D/stage-logs/$n.manifest.json"
    python "$A" copy --attempt "$a" --dest-dir "$D/stage-logs"
    paths+=("$D/stage-logs/$n.log" "$D/stage-logs/$n.manifest.json" "$D/stage-logs/$n.attempt.json")
  done
  if [ -f "$O" ]; then paths+=("$O"); fi
  [ "${#paths[@]}" -gt 0 ] || { echo "STOP: nothing to record"; exit 1; }
  while IFS= read -r -d '' staged; do
    allowed=false
    for f in "${paths[@]}"; do if [ "$f" = "$staged" ]; then allowed=true; break; fi; done
    "$allowed" || { echo "STOP: unrelated staged path: $staged"; exit 1; }
  done < <(git diff --cached --name-only -z)
  expect=()
  for f in "${paths[@]}"; do expect+=(--expect "$f"); done
  git add -- "${paths[@]}"
  python scripts/h7r_d4_records.py verify-staged "${expect[@]}"
  git commit -m "H7-R S400-TRAIN: $N exit $rc (records only)"
  git push origin feat/h7-c1
  echo "__RECORDED__ $rc"
  exit "$rc"
)
```

s5-R. 후보 행

```bash
(
  set -euo pipefail
  cd ~/renju-h6
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src:$PWD/scripts"
  [ "$(git branch --show-current)" = feat/h7-c1 ] || { echo "STOP: wrong branch"; exit 1; }
  git diff --quiet || { echo "STOP: unstaged tracked changes"; exit 1; }
  if git diff --cached --quiet; then git pull --ff-only origin feat/h7-c1; fi
  N=rows
  D=docs/h7r/r2/R2-S400-TRAIN
  R=runs/h7r/R2-S400-TRAIN
  L="$R/logs"
  O="$D/$N.json"
  C="$R/policy_rows_s400.jsonl.gz"
  A=scripts/h7r_attempts.py
  tracked(){ git cat-file -e "HEAD:$1" 2>/dev/null; }
  if tracked "$O"; then echo "STOP: $N is already recorded"; exit 1; fi
  tracked "$D/search.json" || { echo "STOP: search is not recorded (run s5-S first)"; exit 1; }
  python - "$D/search.json" <<'PY'
import json, sys
from pathlib import Path
status = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')).get('status')
if status != 'SEARCHED':
    sys.exit(f'STOP: the search ended {status}: no candidate rows')
PY
  mkdir -p "$L"
  command -v flock > /dev/null || { echo "STOP: flock is missing (util-linux)"; exit 1; }
  exec 9> "$L/$N.lock"
  flock -n 9 || { echo "STOP: another $N block is running"; exit 1; }
  if [ ! -e "$O" ]; then
    git diff --cached --quiet || { echo "STOP: resolve staged changes before a new run"; exit 1; }
    aside=()
    if [ -e "$C" ]; then
      sha="$(sha256sum "$C" | cut -c1-12)"
      if [ "${S5_SET_ASIDE:-}" != "$sha" ]; then
        echo "STOP: an interrupted rows run left $C (SHA-256 $sha...) without $O."
        echo "  The file is kept as it is: the rows tool never writes over it, and nothing was changed."
        echo "  To move it to $R/interrupted/ and build the rows again from the committed S400 records:"
        echo "    export S5_SET_ASIDE=$sha   # then run this block again, and afterwards: unset S5_SET_ASIDE"
        echo "  The new attempt records the moved file, the attempt that left it, and whether the new file equals it."
        exit 1
      fi
      aside+=(--set-aside "$C")
    fi
    if [ -e "$C.tmp" ]; then aside+=(--set-aside "$C.tmp"); fi
    python scripts/check_track_b_frozen.py
    log="$(mktemp "$L/$N-XXXXXXXX.log")"
    python "$A" start --stage "$N" --log "$log" --result "$O" --output "$C" ${aside[@]+"${aside[@]}"}
    set +e
    python -u scripts/h7r_s400_train.py "$N" --device cuda 2>&1 | tee "$log"
    codes=("${PIPESTATUS[@]}")
    set -e
    rc="${codes[0]}"
    [ "${codes[1]}" -eq 0 ] || { echo "STOP: tee failed; log: $log"; exit 1; }
    echo "__EXIT__ $rc" >> "$log"
    python "$A" finish --attempt "${log%.log}.attempt.json" --rc "$rc"
  else
    echo "an uncommitted $O exists: recording it (no new run)"
    result="$(python "$A" recover --stage "$N" --log-dir "$L" --result "$O")"
    read -r name rc <<< "$result"
  fi
  paths=()
  shopt -s nullglob
  for a in "$L/$N"-*.attempt.json; do
    n="$(basename "$a" .attempt.json)"
    tracked "$D/stage-logs/$n.attempt.json" && continue
    python scripts/h7r_d4_records.py copy-logs --resume --src-dir "$L" --pattern "$n.log" \
      --dest-dir "$D/stage-logs" --manifest "$D/stage-logs/$n.manifest.json"
    python "$A" copy --attempt "$a" --dest-dir "$D/stage-logs"
    paths+=("$D/stage-logs/$n.log" "$D/stage-logs/$n.manifest.json" "$D/stage-logs/$n.attempt.json")
  done
  if [ -f "$O" ]; then paths+=("$O"); fi
  [ "${#paths[@]}" -gt 0 ] || { echo "STOP: nothing to record"; exit 1; }
  while IFS= read -r -d '' staged; do
    allowed=false
    for f in "${paths[@]}"; do if [ "$f" = "$staged" ]; then allowed=true; break; fi; done
    "$allowed" || { echo "STOP: unrelated staged path: $staged"; exit 1; }
  done < <(git diff --cached --name-only -z)
  expect=()
  for f in "${paths[@]}"; do expect+=(--expect "$f"); done
  git add -- "${paths[@]}"
  python scripts/h7r_d4_records.py verify-staged "${expect[@]}"
  git commit -m "H7-R S400-TRAIN: $N exit $rc (records only)"
  git push origin feat/h7-c1
  echo "__RECORDED__ $rc"
  exit "$rc"
)
```
