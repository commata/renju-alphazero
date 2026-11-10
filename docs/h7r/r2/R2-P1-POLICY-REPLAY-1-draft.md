# R2-P1-POLICY-REPLAY-1: 정책 학습에 H3 정책 보존 replay를 더하는 대조 실험 (초안)

- 상태: **draft**. 계약 초안 `R2-P1-POLICY-REPLAY-1/p1-contract.draft.json`의 승인 플래그는 모두 false다.
- 이번 범위는 문서와 계약 초안까지다. 구현, 학습, Arena는 실행하지 않았다.
- 목표: Track B 최강 기력 (사용자 결정, 2026-11-10)

## 1. 질문과 근거

- **질문**
  - 정책 fine-tune이 기존 정책의 판단을 잃는 것을 막는 항을 더하면, 같은 S50 예산에서 정책 후보가 연구 champion(H3 정책 + c5 value)보다 강해지는가.
  - 그리고 기존 학습법보다 나아지는가.
- **근거** (커밋된 기록. 원본 checkpoint·기보를 다시 실행한 검증은 아니다)

| 기록 | 사실 |
|---|---|
| c1 P-only (`docs/h7c/c1/arena_verdict.json`) | 0.36 [0.24, 0.49], 향상 [−0.26, −0.01], 25쌍 |
| c2 P-only (`docs/h7c/c2/arena_verdict.json`) | 0.36 [0.27, 0.45], 향상 [−0.23, −0.05], 25쌍. teacher는 S200 |
| c2 정책 학습 (`docs/h7c/c2/policy_record.json`) | 24,077행, 96 step(2 epoch). RenjuNet guard 하락 1.93%p(허용 2%p) |
| 정책 trainer (`src/hybrid/h7c_train.py`) | KL(H3 ‖ student)은 self-play 행에만 있다. RenjuNet은 val top-1 guard로만 쓴다. trunk를 포함해 학습하고, BN은 parent 통계를 유지한다. |
| c5 S400 대 S50 (`docs/h7c/e2/sb_c5_saturation.json`) | 150쌍 0.5767, 향상 [+2.5, +12.67]%p, 수당 시간 ×4.94. 탐색으로 얻을 기력은 있다. |

- **가설** (인과는 입증되지 않았다)
  - 탐색에서 배울 것이 있지만, 적은 self-play 자료에 맞추는 동안 기존 정책의 다른 판단까지 변한다. 그 손실이 이득을 상쇄하거나 넘는다.
  - 작은 확률 변화도 S50의 방문 배분과 선택을 크게 바꿀 수 있다. 그래서 KL 수치만으로 변화가 작았다고 단정하지 않는다.
  - RenjuNet 정확도 하락이 기력 저하의 원인이라는 보장도 없다.

## 2. 두 조건

| | 대조군 C | 처방군 T |
|---|---|---|
| cycle 학습 | c2 정책 학습법 그대로 | C와 같음 |
| 추가 항 | 없음 | RenjuNet **train** 국면에서 KL(H3 ‖ student), 온도 1 |

- **cycle 쪽은 두 군이 완전히 같다**
  - 입력: c2 `policy_rows.jsonl.gz`(`ba2e1bc9…`, 24,077 / 1,423행)
  - 출발점: H3(`efee4832…`), teacher: H3
  - 설정: batch 512, 2 epoch(96 step), lr 2e-4, weight decay 1e-4, min lr 비율 0.05, warm-up, KL λ 0.5(self-play 행), D4 증강, BN eval, evaluation 10회, guard 하락 0.02
  - cycle batch의 순서와 난수 생성기도 같다.
- **replay 항** (처방군만)
  - 매 step 같은 cycle batch에 더해, RenjuNet train split에서 512개 국면을 따로 뽑는다(복원 추출, 균등).
  - replay 전용 난수 생성기: seed = 학습 seed + 1,000,003. cycle 생성기와 섞이지 않는다.
  - D4 증강을 같은 방식으로 하되, replay 생성기를 쓴다.
  - 손실: L = L_cycle(기존 정의 그대로) + μ · mean(KL(H3 ‖ student)), **μ = 0.5**
    - μ는 기존 self-play KL 계수와 같은 크기로 미리 고정한다. 결과를 보고 바꾸지 않는다.
  - cycle batch를 replay로 바꾸지 않는다. 그래서 cycle 학습 노출은 두 군이 같다.
  - 사람 착수 CE replay는 넣지 않는다. 다음 변수로 남긴다.
- **checkpoint 선택** (두 군 동일)
  - 기존 규칙을 그대로 쓴다: guard(RenjuNet **val** top-1)가 H3 대비 0.02 이내인 평가 중, cycle val objective가 가장 낮은 것
  - replay 항은 선택 지표에 들어가지 않는다.
- **value**: c5(`4e28a62a…`)로 고정한다. 새 head 확대나 value 재학습은 섞지 않는다.

## 3. seed

| 용도 | 값 |
|---|---|
| 학습 seed (두 군 공통) | **20261024**(c2와 같음), 20261111, 20261112 |
| Arena 후보 seed (결과 전 고정) | **20261024** |
| Arena 대국 seed | 20261121 (기존 계약과 겹치지 않음. 도구가 다시 확인한다) |
| bootstrap | 20,000회, seed 20261131, opening 쌍 단위 percentile |

- Arena 후보 seed를 20261024로 둔 이유: 이 seed의 대조군은 실제 c2 정책을 재현한 것이다. 그래서 과거 기록과 직접 이어진다(§5 재현 gate).

## 4. 입력 검증 (prepare, 학습 전)

- **SHA 전체 대조**
  - c2 `policy_rows.jsonl.gz` 파일과 행 내용
  - H3 정책 checkpoint와 metadata(`best.json`)
  - c5 value checkpoint와 metadata
- **RenjuNet cache** (`data/external/renjunet/h3_cache`)
  - manifest 형식과 code identity(규칙·encoder)
  - train과 val split 텐서의 내부 해시(`tensor_sha256`)
  - split별 국면 수와 게임 수
  - train과 val의 `game_id`가 서로 겹치지 않는지
  - 이 값들은 처음 기록된다. 기록 뒤 바뀌면 거부한다.
- **c2 재현 대상**: c2 `policy_record.json`과 `policy_best.json`의 SHA
- 모든 원본은 `data/`와 `runs/`에만 둔다. 커밋하는 것은 해시와 수치뿐이다(RenjuNet 파생 자료).

## 5. 단계와 gate

| 단계 | 하는 일 | 통과 조건 | 실패 시 |
|---|---|---|---|
| prepare | §4 검증 | 모두 일치, split 비중첩 | 멈춤, 기록 |
| reproduce | 새 trainer, replay off, seed 20261024 | c2 `policy_record`의 평가 기록(val objective, guard top-1)이 1e-6 이내로 같다. checkpoint SHA 일치는 기록하되 조건은 아니다. | 멈춤. 환경이나 코드 차이로 보고 Arena 없음 |
| train | 3 seed × 2 군 = 6회 | 기록 완성 | 실패한 실행은 기록. Arena 후보 seed가 실패하면 멈춤 |
| link | Arena 후보 = seed 20261024의 T와 C 선택 checkpoint | 둘 다 `CANDIDATE` | 한쪽이 `NO_CANDIDATE`면 멈춤. 다른 seed로 바꾸지 않는다 |
| arena | 아래 §6 | 모든 대국 완료, 상대 불법 수 0 | 멈춤, 보존. 재실행·표본 보충 없음 |
| judge | 아래 §7 | — | — |

- **재현 gate의 전제**
  - c2 학습 커밋(`c7e4123`) 뒤 `h7c_train.py`는 세 번 바뀌었다(`f0b0022`, `0ced742`, `76e8a49`). 바뀐 것은 모두 value 쪽이고, 정책 학습 함수는 그대로다.
  - 그래도 torch·CUDA 버전이나 비결정성 때문에 재현이 안 될 수 있다. 그때는 위 표대로 멈추고 보고한다. 허용 오차를 사후에 넓히지 않는다.

## 6. Arena (주 판독)

- 상대: 연구 champion = H3 정책 + c5 value, S50
- 시험: T(seed 20261024 정책 + c5 value), C(같은 seed 정책 + c5 value), 둘 다 S50
- factory: `analysis.tb_v3_net`(v1). 정책은 frozen architecture이고, value는 c5다.
- **opening**: 개발 suite O0(300)와 O1(300) 전부, 600개. opening마다 색을 바꾼 한 쌍을 둔다.
  - T와 C는 같은 opening 목록과 같은 대국 seed로 둔다.
  - 확인용 suite(`O1-confirm-main`, `O1-confirm-anchor`)는 쓰지 않는다.
  - 이전 pilot·Arena와 일부 수순이 겹친다. 개발 판독이라 허용하고, 기록에 적는다.
- **규모 근거** (계획값이며 보장이 아니다)
  - 측정된 작은 차이 arm의 pair SD 상한이 0.246–0.252다. 3%p를 검출력 0.8로 보려면 527–553쌍이 필요하다.
  - 600쌍은 그보다 조금 크다. T − C 차이의 SD는 두 arm의 상관에 달려 있어 미리 알 수 없다.
- **비용**
  - R1 pilot 실측(S50 작은 차이 arm, 6 worker, 쌍당): O1 5.7–7.3초, O0 11.0–12.2초
  - 그대로 쓰면 arm 하나에 약 1.4–1.6시간, 두 arm에 약 3시간이다.
  - 새 정책 후보의 비용은 smoke(각 arm 처음 10쌍)로 다시 추정해 기록한다. 추정 결과로 규모를 바꾸지 않는다.
- **보조** (판독·안정성 확정·승급 근거로 쓰지 않는다)
  - seed 20261111과 20261112의 T와 C를 각각 champion과 25쌍(O2 앞 25개)
  - 학습 기록의 교사 일치율, RenjuNet val top-1, cycle val과 RenjuNet val에서의 KL(H3 ‖ student)

## 7. 판독 (정책 후보용으로 새로 정한다)

- 통계: opening 쌍 단위 bootstrap, 단측 α = 0.025
  - 주 목표 P와 짝 차이 D는 각각 이 수준에서 읽는다.
  - 결정이 둘이 아니라 하나(P)이므로 α를 나누지 않는다. D는 해석에만 쓴다.
- **P: T 대 champion** (`h7r_eval.arm_summary`)
  - `STRONGER`: (점수 − 0.5)의 단측 하한 > 0
  - `WEAKER`: 단측 상한 < 0
  - 그 밖은 `NOT_ESTABLISHED`
- **D: T − C, 같은 opening 쌍의 점수 차** (`h7r_eval.anchor_difference`)
  - `BETTER_THAN_CONTROL`: 단측 하한 > 0
  - `WORSE_THAN_CONTROL`: 단측 상한 < 0
  - 그 밖은 `NOT_ESTABLISHED`
  - 한 군만 champion 대비 유의하다는 이유로 두 군이 다르다고 판정하지 않는다. 두 군의 차이는 D로만 본다.
- **해석** (기전은 정책 보존 가설과의 일치까지만 말한다)

| P | D | 기록할 문장 |
|---|---|---|
| STRONGER | 무엇이든 | 개발 후보다. 독립 확인 단계의 대상이고, champion은 바꾸지 않는다. |
| 그 밖 | BETTER_THAN_CONTROL | 정책 보존 처방이 기존 학습법보다 도움이 됐고, 지식 손실 가설과 일치한다. champion보다 강하다는 근거는 아니다(회귀 완화까지). |
| 그 밖 | NOT_ESTABLISHED | 이번 처방의 효과는 입증되지 않았다. 지식 손실 가설을 배제하지 않는다(가중치·표본·정밀도 때문에 놓쳤을 수 있다). |
| 그 밖 | WORSE_THAN_CONTROL | 이번 처방이 기존 학습법보다 나빴다. 가설의 배제가 아니라 이 처방의 결과다. |

- 기존 value ε, R2 H-dev 판독, D4 판독은 바꾸지도 쓰지도 않는다.
- **종료 조건**
  - P가 STRONGER가 아니면 이 계약은 끝난다.
  - 같은 자료로 가중치를 바꿔 다시 판정하지 않는다. 다른 μ나 사람 착수 CE는 새 계약이다.

## 8. 별도 단계로 남기는 것

- 독립 확인 평가: 확인 suite, 새 seed, H-conf 선택 품질
- anchor 비열등(원본 H3 / H6), 전술 v2, 수당 시간 gate(평균 +10%, p95 +20%)
- champion 변경, canonical·웹 변경
- S400 등 탐색 예산을 늘리는 엔진 후보. 학습 처방과 분리해 따로 검토한다.
- 강화학습 루프 연결: 이 처방이 위 단계를 통과한 뒤에만

## 9. 구현 범위 (승인 뒤, 기존 trainer·기록 보존)

- **고치지 않는 것**
  - `src/hybrid/h7c_train.py`, `scripts/h7c_*`
  - `src/hybrid/h7r_eval.py`, `scripts/h7r_eval.py`
  - frozen engine·runtime 파일, `analysis.tb_v3_net`
  - 기존 기록
- **새로 만드는 것**
  - `src/hybrid/h7r_p1_train.py`
    - `h7c_train`의 함수(`planes_from_states`, `_policy_terms`, `policy_objective`, 일정·평가 step)를 그대로 가져다 쓴다.
    - 학습 loop만 replay 항을 더한 판으로 둔다.
    - replay off일 때 `h7c_train.train_policy`와 기록·state가 같음을 테스트로 보인다.
  - `scripts/h7r_p1.py`
    - `prepare`(§4), `reproduce`, `train --arm C|T --seed`, `link`
    - `arena-contract`: h7r_eval 형식의 Arena 계약을 모델 SHA로 만든다. 학습 뒤, 결과 전
    - `judge`(§7)
    - 대국은 기존 `scripts/h7r_eval.py play`로 둔다.
  - `tests/test_h7r_p1.py`
    - replay off 동등성, replay 생성기 분리(cycle 순서 불변), μ 고정
    - split 비중첩 검사, 입력 SHA 거부, 재현 gate, `NO_CANDIDATE` 처리, 판독표
- 데스크톱 블록은 이전 방식대로 단계마다 승인 뒤 실행한다(자동 실행기 없음).

## 10. 승인 단계

1. 이 초안 검토
2. 계약 확정(`status: fixed`)과 구현 → `approved.prepare`
3. `approved.train` (재현 gate 포함)
4. `approved.arena` (모델 연결 뒤)
