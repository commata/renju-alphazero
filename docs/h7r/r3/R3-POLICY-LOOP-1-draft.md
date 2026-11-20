# R3-POLICY-LOOP-1: 갱신한 정책이 다음 자가대국을 만들 때 기력이 회복되거나 오르는가 (초안)

- 작성 2026-11-20. 상태: **계약 고정(fixed), 모든 단계 미승인**(prepare·gate·loop 모두 false). 도구·테스트·데스크톱 블록 준비(§11, §13).
- 결정(2026-11-20 검토): 최대 3사이클, 중단 폭 δ = 2%p, policy만 학습(value는 c5 고정), replay·KL teacher·guard 기준은 원본 H3 고정(μ 0.5). 자가대국·S400 재탐색·학습 초기화에는 직전 사이클에서 선택한 정책을 쓴다.
- 결정(2026-11-20 두 번째 검토): val 목표도 S400(대국 단위 split, S400 질량이 없으면 S50 유지하고 split별로 센다), 실행 승인은 prepare → gate → loop 세 번(loop 한 번으로 최대 3사이클), 자가대국은 사이클당 1,000판. 사이클 사이 변화는 opening ID로 짝짓는다.
- 계획상 위치: R3 실험이다. recipe 하나를 여러 사이클 이어서 시험하는 것이며, 오프라인으로는 잴 수 없는 성질(반복 효과)을 본다. champion 변경, 독립 확인, 정식 R4 진입은 하지 않는다. R4 진입 조건(`docs/h7-rl-plan.md` §5: R3 recipe의 세 고리 통과)은 그대로다.

## 0. 근거

- 고정된 champion 자료로 policy를 한 번 학습한 후보는 champion보다 낮았다.
  - R2-S400-TRAIN S(μ 0.5): P′ −2.00%p, 단측 [−3.92, −0.08], WEAKER
  - R2-S400-PRESERVE-1 S2(μ 2.0): P″ −3.75%p, [−5.71, −1.83], WEAKER. 보존을 강화한 효과(D″)는 정해지지 않았다.
- 두 실험 모두 actor가 고정된 자료(c2 자가대국)로 한 번만 학습했다. 학습한 정책이 다음 자가대국을 만들 때 여러 사이클에 걸쳐 회복되거나 오르는지는 검증한 적이 없다.
- B2(c4–c8)는 반복 루프였지만 이 질문에 답하지 않는다.
  - B2는 VALUE_ONLY였다. value만 바꾼 후보는 수의 약 1%만 다르게 뒀다(같은 대국인 쌍 25–35 / 50).
  - 승급하지 못하면 actor를 champion으로 유지했다. 그래서 c6–c8에서는 갱신된 모델이 자가대국을 만들지 않았다.
  - 반면 위 policy 후보들은 300쌍 중 173–209쌍에서 champion과 수가 갈렸다. 학습이 결정을 바꾸는 쪽은 policy다.
- 위 두 결과는 새 루프의 1사이클 성적 예측이 아니라, §7 중단 규칙이 어떻게 작동하는지 보여 주는 사례로 쓴다. 예를 들어 두 후보 모두 상한이 −2%p보다 높아(−0.08, −1.83) 이 규칙에서는 다음 사이클로 간다.

## 1. 질문과 범위

- 질문: H3 policy + c5 value에서 출발해, policy만 갱신하고 갱신된 policy로 다음 자가대국을 만드는 루프를 최대 3사이클 돌리면, 고정 기준(champion) 대비 개발 Arena 점수가 회복되거나 champion을 넘는가.
- 범위
  - 개발 판독까지다. champion을 넘은 후보는 별도 독립 확인 계약으로만 다룬다.
  - 사이클마다 학습 seed는 하나다. 학습 seed 간 변동은 구간에 들어가지 않는다.
  - 예산은 S50 하나, μ는 0.5 하나다.

## 2. 고정할 것

| 항목 | 값 |
|---|---|
| 출발점 | H3 policy(`efee483284df`) + c5 value(`4e28a62a4a31`) |
| 평가 상대 | 같은 champion(H3 + c5), S50. 모든 사이클을 같은 기준으로 잰다 |
| 학습 대상 | policy만. value는 모든 단계에서 c5 |
| 사이클 k의 actor | 사이클 k−1에서 선택한 policy(k = 1은 H3) + c5 value |
| 사이클 k의 S400 재탐색 prior | actor와 같은 policy + c5 value |
| 사이클 k의 학습 초기 모델 | 사이클 k−1에서 선택한 policy(k = 1은 H3) |
| KL teacher, replay teacher, guard 기준 | 원본 H3, 모든 사이클 고정 |
| replay | R2-S400-TRAIN T 설정 그대로: RenjuNet train split, batch 512, μ 0.5, 생성기 seed = 학습 seed + 1000003 |
| 학습 설정 | R2-S400-TRAIN `training` 그대로(batch 512, 2 epoch, lr 2e-4, wd 1e-4, min lr 0.05, kl_lambda 0.5, D4, 평가 10회, guard_drop 0.02, BN eval) |
| 중단 폭 δ | 2%p (§7) |
| 최대 사이클 | 3 |

- 두 checkpoint의 역할을 나눈다. 학습 초기 모델은 직전 후보이고, teacher는 원본 H3이다. 지금의 `h7r_p1a1.execute`는 둘 다 H3에서 불러오므로 그대로 쓰지 않는다(§11).
- guard 기준이 H3이므로 사이클이 이어져도 RenjuNet val top-1이 H3보다 2%p 넘게 떨어진 상태는 선택될 수 없다. 표류는 누적으로 제한된다. 그 결과로 후보가 없으면 루프가 끝난다(§7).

## 3. 사이클 정의

사이클 k(1–3)는 다음 순서다. 각 단계는 자기 입력 모델의 SHA를 기록하고, 다음 단계는 그 SHA를 확인한 뒤 시작한다.

| 단계 | 하는 일 | 쓰는 policy | 기록 |
|---|---|---|---|
| selfplay-k | `h7c_selfplay.run`, 1,000판, worker 6, run seed §8 | actor(k−1 후보) | actor·value 파일 SHA, 대국 JSONL SHA, self-play identity |
| search-k | 먼저 S50 재현 확인(앞 5게임의 모든 방문 국면이 자가대국 기록과 정확히 일치: `h7c_retarget.compare_to_actor`), 그다음 모든 방문 국면을 S400(factor 8)으로 재탐색. 둘 다 `h7c_retarget._execute`, 모델은 자가대국 identity와 대조(`check_engine`) | actor와 같은 policy | 재현 결과, S50·S400 기록 SHA |
| rows-k | `h7c_labels.build`(c2 label 설정, 대국 단위 split·val 5%·split 간 중복 마스크)로 S50 행, 그다음 모든 방문 행(train·val)의 목표를 S400으로. S400 질량이 없으면 S50 목표 유지(split별 집계). 증명 one-hot 행·마스크·행 수·순서는 그대로, 규칙대로인지 다시 검사 | — | S50·S400 행 파일 SHA, split별 교체 수 |
| train-k | 초기 모델 = k−1 후보, teacher = H3, §4 | 초기 모델 k−1 후보 | 초기·선택 state SHA, 실제 step 수, 후보 checkpoint SHA |
| arena-contract-k | v2 개발 Arena 계약 `R3-POLICY-LOOP-ARENA-k`(seed·id 중복 검사 뒤 기록) | k 후보 | 계약 SHA |
| arena-k | 대국 전 모델 파일 확인, 두 arm(이어하기), judge와 진행 결정 | k 후보 | arm 기록 SHA, 판독, 진행 결정 |

- 2사이클 진입 검사: judge-1이 커밋된 CONTINUE이고, train-1 후보 파일(checkpoint·메타데이터)이 기록의 SHA와 같아야 selfplay-2가 시작한다. selfplay-2·search-2(prior)·train-2(초기 모델)가 기록한 policy SHA는 모두 train-1 후보 SHA이고, train-2의 초기 state SHA는 train-1의 선택 state SHA와 같아야 한다. 3사이클도 같다. 도구가 검사하고 테스트로도 확인한다.
- S50 재현이 실패하면(STOP_S50_NOT_REPRODUCED) S400 재탐색을 하지 않고 루프를 멈춘다(무결성 중단).
- 사이클 k의 자가대국 자료는 그 사이클 학습에만 쓴다. 이전 사이클 자료를 합치지 않는다(반복 효과를 자료량 효과와 섞지 않기 위해).

## 4. 학습

- 진입점은 `hybrid.h7r_p1_train.train_policy`(R2-S400-TRAIN T와 같은 함수)다. 새 도구가 초기 모델과 teacher를 따로 넘긴다.
- 학습량: 기존 2 epoch 규칙을 고정한다. step 수 = round(2 × ceil(train 행 / 512))이고, 평가 시점 10회와 warmup은 함수가 그 step 수에서 정한다. 사이클마다 실제 step 수를 기록한다(자료마다 행 수가 달라 step 수는 미리 정하지 않는다).
- 선택: c2 규칙 그대로. guard(RenjuNet val top-1이 H3보다 0.02 이내)를 통과한 평가 중 cycle val objective가 가장 낮은 것. 없으면 NO_CANDIDATE.
- val 목표: train과 같이 S400으로 바꾼다(2026-11-20 결정, R2-S400-TRAIN과 다른 점). split은 대국 단위라 같은 대국의 행이 train과 val에 함께 들어가지 않는다. S400 질량이 없는 방문 행은 S50 목표를 유지하고 그 건수를 split별로 기록한다. R2-S400-TRAIN은 대조군과 val을 같게 두려고 S200 val을 유지했지만, 이 루프에는 대조군이 없고 자가대국 S50 방문 val로 고르면 actor 자신의 탐색에 가까운 checkpoint를 고르게 된다.
- 학습 seed는 사이클마다 하나(§8)다. 결정론 설정(CUBLAS workspace, cuDNN deterministic, threads)은 R2-S400-TRAIN gate와 같아야 시작한다.

## 5. 학습 경로 gate (사이클 전에 한 번)

- 새 도구의 학습 경로가 기존 경로와 같은지 확인한다. 초기 모델 파일 = H3, teacher = H3로 R2-S400-TRAIN S(seed 20261024)를 다시 학습해, 커밋된 기록과 지문·평가곡선·선택·selected state·log-확률·설정·입력을 정확히 비교한다(PRESERVE gate와 같은 방식, 약 2분).
- 하나라도 다르면 이 계약은 사이클을 시작하지 않는다.
- 이 gate는 초기 모델과 teacher를 따로 불러오는 코드가 같은 파일일 때 기존 결과를 그대로 내는지 보는 것이다. 다른 파일일 때의 동작은 테스트(초기 모델 SHA와 teacher SHA가 기록과 일치하는지)로 확인한다.

## 6. 평가

- 사이클마다 v2 개발 Arena 계약 하나(`R3-POLICY-LOOP-ARENA-k`)다. arm은 suite마다 하나: k 후보 + c5 대 champion, S50, 300쌍.
- suite: O1-dev-c / d(개발용 재사용). 사이클끼리 같은 opening이라 바로 비교할 수 있다. 이미 결과를 본 suite이므로 확인에는 쓰지 않는다.
- 통계: 두 suite 50:50 고정 가중, opening 쌍 단위 층화 bootstrap(20,000회, percentile), 단측 α 0.025. class 단위 민감도를 함께 보고한다.
- 판독: P_k = k 후보의 대 champion 점수 − 0.5. 하한 > 0이면 STRONGER, 상한 < 0이면 WEAKER, 그 밖은 NOT_ESTABLISHED. 진행 결정은 §7이다.
- 서술(판독 아님): 사이클 사이 변화(P_k − P_{k−1})를 opening ID로 짝지어 보고한다. 사이클마다 대국 seed가 달라 `pick()`이 같은 300개를 다른 순서로 돌려주므로, 배열 순서로 짝짓지 않는다. 두 사이클의 opening 집합이 다르면 계산을 거부한다.

## 7. 진행 규칙 (사전 고정)

judge-k가 기록한 구간으로 도구가 기계적으로 정한다.

| 조건 (사이클 k) | 결정 | 다음 |
|---|---|---|
| search-k의 S50 재현 실패 | STOP_S50_NOT_REPRODUCED | 무결성 중단(S400 재탐색 없음) |
| train-k가 NO_CANDIDATE | END_NO_CANDIDATE | 넘길 모델이 없으므로 종료 |
| P_k 하한 > 0 | END_STRONGER | 루프 종료. k 후보를 독립 확인 후보로 기록(별도 계약) |
| P_k 상한 < −0.02 | STOP_REGRESSION | 그 모델로 자가대국을 만들지 않고 종료 |
| 그 밖, k < 3 | CONTINUE | 사이클 k+1 |
| 그 밖, k = 3 | END_BUDGET | 3사이클 안에 champion 대비 우위 미확인으로 종료 |

- 우위가 불확실한 사이클(NOT_ESTABLISHED, 상한 ≥ −2%p)은 예산 안에서 다음 사이클로 간다. −2%p는 이 실험에서 감수할 퇴보 폭이다.
- 사용자가 블록을 언제든 멈출 수 있다. 커밋되지 않은 단계는 판독 없이 남고, 같은 블록을 다시 실행하면 그 단계 기록에서 이어간다.
- 여러 번 본다는 점: STRONGER 판독을 최대 3번 한다. 각 검정이 명목 단측 2.5%를 지킨다는 가정에서, 효과가 없을 때 세 번 중 한 번이라도 STRONGER가 나올 확률의 상한은 약 7.5%(3 × 2.5%)다. 이미 사용한 개발 suite에서 나온 END_STRONGER는 독립 확인 후보를 고르는 데만 쓴다. 독립 확인(새 suite·seed, 전술·착수 시간 gate)을 통과해야 champion 교체 대상이 된다.

## 8. seed (사전 고정, 2026-11-20 미사용 확인)

| 용도 | 사이클 1 | 사이클 2 | 사이클 3 | 확인 |
|---|---|---|---|---|
| 자가대국 run seed (`RUN_SEEDS` 8801–8999) | 8809 | 8810 | 8811 | 기록에서 쓰인 run seed는 8800–8808 |
| 학습 seed | 20261201 | 20261202 | 20261203 | 저장소에 없음 |
| Arena 대국 seed (v2 `ARENA_SEEDS` 9101–9140) O1-dev-c / d | 9105 / 9106 | 9107 / 9108 | 9109 / 9110 | 쓰인 v2 seed는 9101–9104 |
| bootstrap seed | 20261211 (모든 사이클) | | | 저장소에 없음 |

- 쓰지 않은 사이클의 seed는 다른 계약에 다시 배정하지 않는다.
- 독립 확인용 suite·seed는 이 계약이 정하지 않는다(별도 계약).

## 9. 단계와 비용 (모두 미승인)

| 단계 | 내용 | 예상 시간 |
|---|---|---|
| prepare | H3·c5 파일 SHA, R2-S400-TRAIN 계약·기록(gate 비교 대상), suite 재계산 검사, seed 충돌 검사 | 수 분 |
| gate | §5 | 약 2분 |
| loop | 사이클 1–3, 사이클마다 §3의 다섯 단계 | 사이클당 약 7.5시간 |

- 사이클당: 자가대국 약 1–1.2시간(1,000판, B2 실측 62–67분), S400 재탐색 약 5시간(c2 자가대국 기준 방문 목표 국면 20,910개, R2-S400-TRAIN 실측 18,965국면 4.5시간에서 비례), labels 수십 분, 학습 약 2분, Arena 약 1시간(300쌍 arm 두 개, PRESERVE 실측 arm당 29–32분).
- 3사이클 모두 돌면 약 23시간이다. 재탐색 국면 수는 자가대국마다 달라지므로 실제 시간은 기록으로 대체한다.
- 승인 단위(결정): prepare, gate, loop 세 번. loop 승인 하나로 최대 3사이클을 §7 규칙대로 진행한다(사이클 사이에 사람이 규칙을 바꾸지 않도록). 단계마다 기록을 따로 커밋한다. 다음 자가대국은 직전 사이클의 커밋된 CONTINUE와 후보 파일 SHA를 확인한 뒤에만 시작한다. 중단 후에는 같은 단계 기록에서 이어간다.
- 약 23시간(7.5시간 × 3)은 계획값이다. 실제 시간은 단계 기록의 시간으로 대체한다.

## 10. 결과의 쓰임과 해석의 한계

| 종료 | 기록과 다음 |
|---|---|
| END_STRONGER | 그 사이클 후보를 독립 확인 후보로 기록. 확인 계약(새 시작 국면, 전술·착수 시간 gate)은 따로 연다 |
| END_BUDGET | 3사이클 안에 우위를 확인하지 못했다고 기록. 사이클별 추정과 구간, 사이클 사이 변화를 서술 |
| STOP_REGRESSION | 그 사이클에서 −2%p를 넘는 퇴보가 확인됐다고 기록 |
| STOP_S50_NOT_REPRODUCED | 재탐색이 자가대국을 재현하지 못했다고 기록(무결성 중단, 판독 없음) |
| END_NO_CANDIDATE | 그 사이클에서 guard를 통과한 checkpoint가 없었다고 기록 |

- 세 점으로 추세를 단정하지 않는다. "회복"이나 "향상"은 사이클별 구간과 사이클 사이 쌍 차이를 그대로 보고하는 것으로 갈음한다.
- 학습 seed는 사이클마다 하나다. 사이클 간 차이에는 학습 seed 변동이 섞여 있다.
- 결과를 보고 δ, 사이클 수, seed, 학습 설정, 판독 설정을 바꾸지 않는다. 바꾸려면 새 계약이다.

## 11. 구현 (2026-11-20, 미승인)

- 계약 `docs/h7r/r3/R3-POLICY-LOOP-1/loop-contract.json`(fixed, 모든 승인 false, 실험 SHA `d464e0447642`).
  - 고정하는 것: H3·c5 파일 SHA, R2-S400-TRAIN 계약(파일 SHA `7f4a5c00a126`, 실험 SHA `67c6ba3c8081`)과 gate·rows 기록, gate 비교 대상인 S(20261024) 기록, 두 suite와 생성 기록, §8 seed, 규칙·통계·재탐색·label 설정.
  - 도구와 재사용 파일 17개를 sources로 고정한다.
- 도구 `scripts/h7r_r3_loop.py`. 고정된 기존 도구(`h7r_p1a1`, `h7r_s400_train`, `h7r_preserve`, `h7c_*`)는 바꾸지 않고 불러 쓴다.

| 명령 | 단계 | 하는 일 |
|---|---|---|
| `prepare` | prepare | H3(고정 TB-V3 정책)·c5 파일, R2-S400-TRAIN 계약·gate·rows·S(20261024) 기록, RenjuNet cache, 커밋된 suite 재계산, 계약 seed가 다른 기록에 없음, 빈 runs 폴더 |
| `gate` | gate | 초기 모델 파일과 teacher 파일을 따로 불러오는 새 학습 경로에 둘 다 H3를 넣고 S(20261024)를 다시 학습해 커밋된 기록과 정확히 비교. FAIL은 종료 코드 2, 사이클을 시작하지 않는다 |
| `next` | loop | 다음 단계를 JSON으로: run(중단 흔적이 있으면 `partial`), record(커밋 전 결과), done(결정) |
| `step --cycle K --name S` | loop | `next`가 가리키는 바로 그 단계만 실행(§3 표). 사이클 2·3은 직전 judge의 커밋된 CONTINUE와 후보 파일 SHA를 확인한 뒤에만 시작 |
| `verify --cycle K --name S` | loop | 단계 기록을 지금 파일과 대조(스테이징 전, 다시 볼 때). 행은 규칙 재검사, 후보는 checkpoint SHA, judge는 arm 기록으로 재계산 |
| `files --cycle K --name S` | loop | 그 단계 커밋에 넣을 파일 |

- 사이클 1 흐름은 클라우드 테스트에서 한 번에 돌린다. 자가대국, S50 재현, S400 목표, 학습 행이 텐서로 로딩되는 것, 학습, Arena, judge 순서다.
  - 대국 기록 형식, labels(`h7c_labels.build`), 행 교체와 검사, `policy_tensors`, 판독은 실제 코드다.
  - 엔진(자가대국 수, 탐색, Arena 대국)과 학습기만 대역이다.
  - 사이클 2는 1사이클 후보에서만 시작하는지 확인한다. 자가대국 policy, 재탐색 prior, 학습 초기 모델과 초기 state가 그 SHA를 가리켜야 한다.
- 기존 `h7c_retarget.py run`은 H7-C 이전 사이클 결정을 요구하고 `cycle < 2`를 거부한다. 그래서 쓰지 않는다. 같은 파일의 탐색 실행 함수(`_execute`, factor 1 / 8)를 직접 부른다.
- 실제 엔진·CUDA 학습·대국은 데스크톱 실행에서 처음 돈다.

## 12. 결정 (2026-11-20)

1. val 목표도 S400(§4). 대국 단위 split, S400 질량이 없으면 S50 유지(split별 집계).
2. 승인 단위는 prepare → gate → loop 세 번(§9).
3. 자가대국은 사이클당 1,000판. 약 23시간은 계획값.

## 13. 데스크톱 실행 (승인 뒤에만)

- r3-P는 `approved.prepare` 뒤, r3-G는 `approved.gate` 뒤, r3-L은 `approved.loop` 뒤에 실행한다. 도구가 승인과 앞 단계 기록을 확인한다.
- r3-P·r3-G는 PRESERVE의 pv-P·pv-G와 같은 방식이다: 로그 이름, 결과가 커밋 전이면 다시 돌리지 않고 기록, `Recovered exit code`.
- r3-L(tmux, 최대 약 23시간)
  - `next`가 가리키는 단계를 하나씩 실행한다. 단계마다 기록을 `verify`로 대조한 뒤 스테이징해 커밋·push한다.
  - 결정(done)이 나오면 `__DONE__ <결정>`을 출력하고 끝난다.
  - `git pull`은 시작할 때 한 번이고, 중단된 단계가 없을 때만 한다. 중단된 단계는 시작한 commit에서 이어간다.
  - `CUBLAS_WORKSPACE_CONFIG`는 train 단계에만 설정한다. 자가대국·탐색·Arena는 지금까지의 엔진 실행과 같은 조건이고, 도구가 확인한다.
  - 단계가 결과 없이 실패하면 로그만 커밋하고 멈춘다. 기록이 파일과 맞지 않으면 아무것도 커밋하지 않고 멈춘다.
  - 같은 블록을 다시 실행하면 이어간다.

r3-P. prepare

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
  D=docs/h7r/r3/R3-POLICY-LOOP-1
  L=runs/h7r/R3-POLICY-LOOP-1/logs
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
    python -u scripts/h7r_r3_loop.py "$N" --log-name "$(basename "$log")" 2>&1 | tee "$log"
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
if r.get('format') != 'h7r-r3-loop-prepare-v1' or not re.fullmatch(sys.argv[2] + r'-[A-Za-z0-9]+\.log', name):
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
  git commit -m "H7-R R3 LOOP: $N exit $rc (records only)"
  git push origin feat/h7-c1
  echo "__RECORDED__ $rc"
  exit "$rc"
)
```

r3-G. 학습 경로 gate

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
  D=docs/h7r/r3/R3-POLICY-LOOP-1
  L=runs/h7r/R3-POLICY-LOOP-1/logs
  O="$D/$N.json"
  tracked(){ git cat-file -e "HEAD:$1" 2>/dev/null; }
  if tracked "$O"; then echo "STOP: $N is already recorded"; exit 1; fi
  tracked "$D/prepare.json" || { echo "STOP: prepare is not recorded (run r3-P first)"; exit 1; }
  mkdir -p "$L"
  rc=0
  if [ ! -e "$O" ]; then
    git diff --cached --quiet || { echo "STOP: resolve staged changes before a new run"; exit 1; }
    python scripts/check_track_b_frozen.py
    log="$(mktemp "$L/$N-XXXXXXXX.log")"
    set +e
    python -u scripts/h7r_r3_loop.py "$N" --device cuda --log-name "$(basename "$log")" 2>&1 | tee "$log"
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
if r.get('format') != 'h7r-r3-loop-gate-v1' or not re.fullmatch(sys.argv[2] + r'-[A-Za-z0-9]+\.log', name):
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
  git commit -m "H7-R R3 LOOP: $N exit $rc (records only)"
  git push origin feat/h7-c1
  echo "__RECORDED__ $rc"
  exit "$rc"
)
```

r3-L. 루프 (tmux)

```bash
(
  set -euo pipefail
  cd ~/renju-h6
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src:$PWD/scripts"
  unset CUBLAS_WORKSPACE_CONFIG  # only the train step sets it (below)
  [ "$(git branch --show-current)" = feat/h7-c1 ] || { echo "STOP: wrong branch"; exit 1; }
  git diff --quiet || { echo "STOP: unstaged tracked changes"; exit 1; }
  D=docs/h7r/r3/R3-POLICY-LOOP-1
  R=runs/h7r/R3-POLICY-LOOP-1
  L="$R/logs"
  T=scripts/h7r_r3_loop.py
  tracked(){ git cat-file -e "HEAD:$1" 2>/dev/null; }
  field(){ python -c 'import json, sys; v = json.loads(sys.argv[1]).get(sys.argv[2]); print("" if v is None else str(v).lower() if isinstance(v, bool) else v)' "$1" "$2"; }
  mkdir -p "$L"
  command -v flock > /dev/null || { echo "STOP: flock is missing (util-linux)"; exit 1; }
  exec 9> "$L/loop.lock"
  flock -n 9 || { echo "STOP: another loop block is running"; exit 1; }
  tracked "$D/gate.json" || { echo "STOP: the gate is not recorded (run r3-G first)"; exit 1; }
  nxt="$(python "$T" next)"
  # pull only between steps: an interrupted step resumes at the commit it started on
  if git diff --cached --quiet && { [ "$(field "$nxt" action)" != run ] || [ "$(field "$nxt" partial)" = false ]; }; then
    git pull --ff-only origin feat/h7-c1
  fi
  frozen=false
  shopt -s nullglob
  while :; do
    nxt="$(python "$T" next)"
    action="$(field "$nxt" action)"
    k="$(field "$nxt" cycle)"
    s="$(field "$nxt" step)"
    if [ "$action" = done ]; then echo "__DONE__ $(field "$nxt" decision) (cycle $k, $s)"; exit 0; fi
    O="$(field "$nxt" record)"
    rc=0
    if [ "$action" = run ]; then
      git diff --cached --quiet || { echo "STOP: resolve staged changes before a new run"; exit 1; }
      if ! "$frozen"; then python scripts/check_track_b_frozen.py; frozen=true; fi
      log="$(mktemp "$L/c$k-$s-XXXXXXXX.log")"
      set +e
      if [ "$s" = train ]; then
        CUBLAS_WORKSPACE_CONFIG=:4096:8 python -u "$T" step --cycle "$k" --name "$s" --device cuda \
          --log-name "$(basename "$log")" 2>&1 | tee "$log"
      else
        python -u "$T" step --cycle "$k" --name "$s" --device cuda --log-name "$(basename "$log")" 2>&1 | tee "$log"
      fi
      codes=("${PIPESTATUS[@]}")
      set -e
      rc="${codes[0]}"
      [ "${codes[1]}" -eq 0 ] || { echo "STOP: tee failed; log: $log"; exit 1; }
      echo "__EXIT__ $rc" >> "$log"
    else
      echo "an uncommitted $O exists: recording it (no new run)"
    fi
    paths=()
    if [ -f "$O" ]; then  # the record checks against its files before it is staged
      v="$(python "$T" verify --cycle "$k" --name "$s")" || {
        echo "STOP: c$k $s: the record does not check against its files; nothing committed"; exit 1; }
      name="$(field "$v" log_name)"
      rc="$(field "$v" exit_code)"
      [[ "$name" =~ ^c$k-$s-[A-Za-z0-9]+\.log$ ]] || { echo "STOP: the record names no attempt log"; exit 1; }
      log="$L/$name"
      [ -f "$log" ] || { echo "STOP: the record's log is missing: $log"; exit 1; }
      case "$(tail -n 1 "$log")" in
        "__EXIT__ $rc") ;;
        __EXIT__*) echo "STOP: the record and its log disagree on the exit code"; exit 1 ;;
        *) printf '\n%s\n__EXIT__ %s\n' "Recovered exit code from $O" "$rc" >> "$log" ;;
      esac
      while IFS= read -r f; do paths+=("$f"); done < <(python "$T" files --cycle "$k" --name "$s")
    elif [ "$rc" -eq 0 ] || [ "$rc" -eq 2 ]; then
      echo "STOP: c$k $s did not produce its record"; exit 1
    fi
    for f in "$L"/c"$k"-"$s"-*.log; do
      [[ "$(tail -n 1 "$f")" =~ ^__EXIT__\ [0-9]+$ ]] || continue
      n="$(basename "$f" .log)"
      tracked "$D/stage-logs/$n.log" && continue
      python scripts/h7r_d4_records.py copy-logs --resume --src-dir "$L" --pattern "$n.log" \
        --dest-dir "$D/stage-logs" --manifest "$D/stage-logs/$n.manifest.json"
      paths+=("$D/stage-logs/$n.log" "$D/stage-logs/$n.manifest.json")
    done
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
    git commit -m "H7-R R3 LOOP: c$k $s exit $rc (records only)"
    git push origin feat/h7-c1
    if [ ! -f "$O" ]; then echo "__RECORDED__ $rc"; echo "STOP: c$k $s failed (exit $rc)"; exit "$rc"; fi
  done
)
```
