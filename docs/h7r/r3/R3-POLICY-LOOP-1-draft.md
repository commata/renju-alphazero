# R3-POLICY-LOOP-1: 갱신한 정책이 다음 자가대국을 만들 때 기력이 회복되거나 오르는가 (초안)

- 작성 2026-11-20. 상태: **초안, 모든 단계 미승인.** 도구·테스트·계약 JSON은 아직 없다(§11).
- 결정(2026-11-20 검토): 최대 3사이클, 중단 폭 δ = 2%p, policy만 학습(value는 c5 고정), replay·KL teacher·guard 기준은 원본 H3 고정(μ 0.5). 자가대국·S400 재탐색·학습 초기화에는 직전 사이클에서 선택한 정책을 쓴다.
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
| selfplay-k | `scripts/h7c_selfplay.py`, 1,000판, worker 6, run seed §8 | actor(k−1 후보) | policy·value 파일 SHA, 대국 JSONL SHA |
| retarget-k | 먼저 S50 재현 확인(5게임, 자가대국 방문과 정확히 일치), 그다음 방문 목표 국면을 S400으로 재탐색 | actor와 같은 policy | 재현 결과, S400 기록 SHA |
| rows-k | `h7c_labels`(v2 policy 행) 뒤 방문 목표만 S400으로 바꾼다. 증명 one-hot 행·마스크·행 수·순서는 그대로 | — | 행 파일 SHA, 바뀐 행 수 |
| train-k | 초기 모델 = k−1 후보, teacher = H3, §4 | 초기 모델 k−1 후보 | 초기·선택 state SHA, 실제 step 수, 후보 checkpoint SHA |
| arena-k | 후보 + c5 대 champion, O1-dev-c / d 300 + 300쌍, judge-k | k 후보 | Arena 계약, arm 기록 SHA, 판독, 진행 결정 |

- 2사이클 진입 검사: selfplay-2, retarget-2, train-2가 기록한 policy SHA가 모두 train-1 후보 checkpoint SHA와 같아야 시작한다. 3사이클도 같다. 도구가 검사하고 테스트로도 확인한다.
- 사이클 k의 자가대국 자료는 그 사이클 학습에만 쓴다. 이전 사이클 자료를 합치지 않는다(반복 효과를 자료량 효과와 섞지 않기 위해).

## 4. 학습

- 진입점은 `hybrid.h7r_p1_train.train_policy`(R2-S400-TRAIN T와 같은 함수)다. 새 도구가 초기 모델과 teacher를 따로 넘긴다.
- 학습량: 기존 2 epoch 규칙을 고정한다. step 수 = round(2 × ceil(train 행 / 512))이고, 평가 시점 10회와 warmup은 함수가 그 step 수에서 정한다. 사이클마다 실제 step 수를 기록한다(자료마다 행 수가 달라 step 수는 미리 정하지 않는다).
- 선택: c2 규칙 그대로. guard(RenjuNet val top-1이 H3보다 0.02 이내)를 통과한 평가 중 cycle val objective가 가장 낮은 것. 없으면 NO_CANDIDATE.
- val 목표: train과 같이 S400으로 바꾼다(R2-S400-TRAIN과 다른 점, §12-1). 그 실험은 대조군과 val을 같게 두려고 S200 val을 유지했지만, 이 루프에는 대조군이 없고 자가대국 S50 방문 val로 고르면 actor 자신의 탐색에 가까운 checkpoint를 고르게 된다.
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
- 서술(판독 아님): 같은 opening에서 사이클 사이 변화(P_k − P_{k−1}, 쌍 대응)를 보고한다.

## 7. 진행 규칙 (사전 고정)

judge-k가 기록한 구간으로 도구가 기계적으로 정한다.

| 조건 (사이클 k) | 결정 | 다음 |
|---|---|---|
| train-k가 NO_CANDIDATE | END_NO_CANDIDATE | 넘길 모델이 없으므로 종료 |
| P_k 하한 > 0 | END_STRONGER | 루프 종료. k 후보를 독립 확인 후보로 기록(별도 계약) |
| P_k 상한 < −0.02 | STOP_REGRESSION | 그 모델로 자가대국을 만들지 않고 종료 |
| 그 밖, k < 3 | CONTINUE | 사이클 k+1 |
| 그 밖, k = 3 | END_BUDGET | 3사이클 안에 champion 대비 우위 미확인으로 종료 |

- 우위가 불확실한 사이클(NOT_ESTABLISHED, 상한 ≥ −2%p)은 예산 안에서 다음 사이클로 간다. −2%p는 이 실험에서 감수할 퇴보 폭이다.
- 사용자가 언제든 멈출 수 있다. 그 경우 MANUAL_STOP으로 기록하고 판독은 하지 않는다.
- 여러 번 본다는 점: STRONGER 판독을 최대 3번 하므로, 효과가 없을 때 한 번이라도 STRONGER가 나올 확률은 최대 약 7.5%(3 × 2.5%)다. 그래서 STRONGER는 개발 신호이고, 독립 확인(새 suite·seed, 전술·착수 시간 gate)을 통과해야 champion 교체 대상이 된다.

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
- 승인 단위 제안: prepare, gate, loop. loop 승인 하나로 최대 3사이클을 §7 규칙대로 진행한다(사이클 사이에 사람이 규칙을 바꾸지 않도록). 사이클마다 기록을 커밋한다.

## 10. 결과의 쓰임과 해석의 한계

| 종료 | 기록과 다음 |
|---|---|
| END_STRONGER | 그 사이클 후보를 독립 확인 후보로 기록. 확인 계약(새 시작 국면, 전술·착수 시간 gate)은 따로 연다 |
| END_BUDGET | 3사이클 안에 우위를 확인하지 못했다고 기록. 사이클별 추정과 구간, 사이클 사이 변화를 서술 |
| STOP_REGRESSION | 그 사이클에서 −2%p를 넘는 퇴보가 확인됐다고 기록 |
| END_NO_CANDIDATE | 그 사이클에서 guard를 통과한 checkpoint가 없었다고 기록 |

- 세 점으로 추세를 단정하지 않는다. "회복"이나 "향상"은 사이클별 구간과 사이클 사이 쌍 차이를 그대로 보고하는 것으로 갈음한다.
- 학습 seed는 사이클마다 하나다. 사이클 간 차이에는 학습 seed 변동이 섞여 있다.
- 결과를 보고 δ, 사이클 수, seed, 학습 설정, 판독 설정을 바꾸지 않는다. 바꾸려면 새 계약이다.

## 11. 구현에 필요한 것 (승인 뒤, 아직 하지 않음)

- 새 도구 `scripts/h7r_r3_loop.py`(이름은 구현 때 확정)
  - 학습: `h7r_p1_train.train_policy`를 직접 부르고 초기 모델과 teacher를 각각 파일·SHA로 받는다. `scripts/h7r_p1a1.py`, `h7r_s400_train.py`, `h7r_preserve.py`는 여러 계약이 SHA로 고정한 파일이라 바꾸지 않는다.
  - 자가대국·재탐색·labels는 기존 도구를 policy 파일 인자로 부른다(`h7c_selfplay.py --policy-checkpoint`, `h7c_retarget` 함수, `h7c_labels.py`). 재탐색은 R2-S400-TRAIN search 경로를 재사용하되 prior를 k−1 후보로 바꾼다.
  - Arena는 v2 검사기와 기존 대국 루프를 재사용한다(PRESERVE와 같은 방식).
  - 단계마다 결과 기록, 로그 이름, 이어하기, 덮어쓰기 금지, 입력 SHA 확인(PRESERVE와 같은 방식).
- 확인할 것(구현 때)
  - `h7c_selfplay`·`h7c_labels`가 후보 policy 파일과 run seed 8809–8811, cycle 번호를 그대로 받는지
  - 재탐색 S50 재현 확인이 후보 prior에서도 자가대국 방문과 정확히 일치하는지
- 테스트: 진행 규칙 표, 2사이클 진입 SHA 사슬(자가대국·재탐색·학습 초기화가 1사이클 후보를 가리킴), teacher가 항상 H3인지, 2 epoch step 계산과 기록, gate 정확 비교, 덮어쓰기·이어하기, 데스크톱 블록(실제 bash, 도구는 대역).

## 12. 검토할 점

1. val 목표를 S400으로 바꾸는 것(§4). 유지하면 자가대국 S50 방문으로 checkpoint를 고르게 된다. 바꾸면 재탐색이 train 범위만일 때보다 약 10% 길어진다.
2. 승인 단위(§9). loop 승인 하나로 3사이클을 진행할지, 사이클마다 승인할지.
3. 자가대국 1,000판(c2와 같은 규모). 늘리면 사이클당 시간이 비례해 는다.
