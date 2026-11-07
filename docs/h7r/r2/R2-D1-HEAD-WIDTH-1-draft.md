# R2-D1-HEAD-WIDTH-1 — value head 폭 진단 계약 초안

- 작성일 2026-11-06, 개정 2026-11-07. 상태: **v1 prepare FAIL(§11) / v2 계약 고정, 단계 승인 전(prepare·train·evaluate 모두 false)**
- 계약 JSON: v2 `R2-D1-HEAD-WIDTH-1/d1-1k-v2-contract.json`(유효), v1 `d1-1k-contract.json`(prepare FAIL로 종료, 기록 보존)
- 도구: `scripts/h7r_d1.py`, `src/hybrid/h7r_d1.py`(v2 계약만 읽는다)
- 상위 설계: `R2-DIAG-1-draft.md` §7(판독 규칙, ε, `DEVELOPMENT_SIGNAL`, 보류)과 §8(D1 행)
- 직전 진단: `R2-D3-NO-OUTCOME-1-draft.md` §9(D3 `NOT_ESTABLISHED`)
- §1–8의 "제안값"은 §10에서 1,000행 단계 값으로 확정했다. §9는 확정 전 목록으로 남긴다.

## 0. 성격과 바꾸지 않는 것

- **진단이다.** 첫 질문은 하나로 좁힌다.

  > 같은 자료와 같은 학습 예산에서 value head의 폭(`value_channels`)만 늘리면 train 오차가 줄어드는가?

- **동기와 증거의 구분**: D3에서 outcome 학습 기여를 빼자 proof 손실은 줄고 outcome 손실은 늘었다. 이 절충이 이 질문을 시험할 동기다. 다만 용량 부족이나 목표 간 간섭을 확인한 결과는 아니다. 이 계약의 판정 기준과 문턱은 D3 결과에 맞추지 않고 결과 전에 고정한다.
- **한 번에 하나만 바꾼다.** `value_hidden`, 학습률, BN 동작 방식, replay, loss는 두 군이 같다. lr·BN 진단은 후속 단계다(§7).
- **바꾸지 않는 것**
  - D3 판독(`NOT_ESTABLISHED`), D4 판독
  - c5 유지, V9 보류
  - R2의 주 지표, ε, `DEVELOPMENT_SIGNAL` 규칙
- 이 계약의 1,000행과 5,000행 단계는 진단이다. 그 단계의 모델은 D4 후보가 되지 않는다. D4 진입은 전체 규모 단계에서 별도 계약으로만 검토한다(§7).

## 1. 비교 설계

| 항목 | 직접 control | 실험군 |
|---|---|---|
| value head | 기존 head(`value_channels = 1`, `value_hidden = 64`) | `value_channels = 4`(제안값)로만 넓힌 head, `value_hidden = 64` |
| 출발점 | c5 value head | c5 value head를 **예측을 보존하며 확장**(§2) |
| 자료 | c8 T-train(cycle train split)에서 게임 단위로 사전 고정한 약 1,000행(§3) | 같음 |
| replay·loss | c8 recipe: 매 epoch cycle 행 수만큼 H6 replay 비복원 추출(50:50), 가중 MSE(proof 1.0, outcome 0.1) | 같음 |
| 학습 설정 | c8 config(batch 1024, lr 2e-4, warmup·cosine, D4 augmentation, 평가 10회, H6 guard 1.05). epoch 수만 아래 step 예산에 맞춘다 | 같음 |
| step 예산 | 210 step(제안값; c8 실행과 같은 step 수) | 같음 |
| seed | 31001 / 31002 / 31003 | 같음 |
| BN | 기존 trainer 그대로: trunk·policy는 eval(동결), value head는 train 모드 | 같음 |

- **직접 control은 새로 학습한다.** R2-0B 실행은 전체 c8 자료로 학습했으므로 이 단계의 control이 될 수 없다. 같은 부분집합·같은 seed·같은 예산의 기존 head가 control이다(R2 §8).
- **짝맞춤 확인**: 같은 seed의 두 군은 cycle 행 수, replay 모집단, batch, step이 같으므로 자료 배정 지문(`h7r_r2.data_assignment`)이 같아야 한다. 도구가 확인하고 기록한다. 폭 확장은 trainer의 난수 흐름을 쓰지 않는다(§2).
- **비용 기록**: 폭이 커지면 step당 비용이 달라질 수 있다. 두 군 모두 실제 step 수와 GPU 시간(학습 전체, step당 평균)을 따로 기록한다.
  - value head의 첫 Linear는 225×64에서 900×64로 커진다.
  - 학습 시간의 대부분은 동결 trunk의 forward이므로 차이는 작을 것으로 예상하지만, 측정값으로만 보고한다.

## 2. 예측을 보존하는 확장과 검증

**확장 방법** (c5 → `value_channels = 4`)

1. 1×1 conv(`C → 4`, bias 없음)
   - 출력 채널 0은 c5의 가중치를 그대로 복사한다.
   - 채널 1–3은 네트워크 기본 초기화(Kaiming)로 새로 만든다. 별도의 고정 생성기를 쓴다(seed는 제안값, §9). trainer의 전역 난수는 쓰지 않는다.
2. BN(4채널)
   - 채널 0은 c5의 weight, bias, running_mean, running_var, num_batches_tracked를 복사한다.
   - 채널 1–3은 PyTorch 기본값(1, 0, 0, 1)이다.
3. 첫 Linear(`4×225 → 64`)
   - Flatten 순서상 채널 0이 차지하는 입력 열 0–224는 c5 가중치를 복사한다.
   - 새 채널의 입력 열 225–899는 **정확히 0**이다. bias는 c5 값을 복사한다.
4. 둘째 Linear(`64 → 1`)와 Tanh는 c5 값을 그대로 복사한다.

- 이 방법이면 새 채널의 기여가 0이므로, eval 모드와 train 모드 모두 첫 forward가 c5와 같다.
- 학습이 시작되면 새 열이 먼저 기울기를 받고, 그 뒤 새 conv 채널도 기울기를 받는다.

**검증 기준** (학습 전에, 하나라도 어긋나면 멈춤)

- trunk·policy의 `frozen_state_sha256`가 c5와 같다. 학습 후에도 같다(기존 trainer 검사).
- 복사한 텐서(conv 채널 0, BN 채널 0의 값과 통계, Linear 열 0–224와 bias, 둘째 Linear)가 c5의 해당 부분과 바이트 단위로 같다. 새 Linear 열은 모두 0이다.
- eval 모드 예측
  - 부분집합 train 행, cycle val, H6 val 표본, H-dev 행에서 max |v_wide − v_c5| ≤ 1e-6(제안값; float32 누산 순서 차이만 허용)
  - 확장 parent의 T-val objective와 c5의 T-val objective 차이 ≤ 1e-6
- 같은 seed의 첫 batch에서 두 군의 학습 손실 차이 ≤ 1e-6
- 기록: 확장 parent의 state 해시, 새 채널 초기화 seed, 위 검사값
- **실제로 구현한 검사는 이 계획과 다르다**(§10.1, v2는 §11.2).
  - 예측 검사는 검사한 행(부분집합, cycle val, H6 val 표본; H-dev는 쓰지 않음)에서의 초기 예측 보존 확인이다.
  - T-val objective와 첫 batch 학습 손실은 직접 비교하지 않는다. 예측 차이의 상한은 제곱손실 차이의 상한과 다르고, H6 val은 표본만 본다.
  - 첫 batch 항목은 train 모드 예측 보존 확인으로 바뀌었다: 부분집합 앞 1,024행을 한 batch로 추론한다. 실제 학습 첫 batch의 replay 혼합과 augmentation은 적용하지 않는다.

## 3. 자료 부분집합

- 출처: c8 cycle cache의 train split(35,816행). A1 기준 기록의 labels·split SHA로 확인한다.
- 선택 규칙(결과와 무관하게 사전 고정)
  1. train 게임 ID를 정렬한다.
  2. 고정 seed(제안값, §9)로 섞는다.
  3. 앞에서부터 게임을 누적해 행 수가 처음 1,000 이상이 되는 게임에서 멈춘다. 게임을 쪼개지 않는다.
- 같은 순서로 이어서 5,000행 부분집합도 지금 정한다. 1,000행 집합은 5,000행 집합에 포함된다.
- cycle val(전체), H6 replay와 H6 val은 그대로 쓴다. H-dev는 학습, 선택, 문턱 결정에 쓰지 않는다.
- 기록: 부분집합의 게임 ID 목록 SHA, 행 수, 구성(outcome / proof 행과 가중치, ply 구간)

## 4. train 적합 (주 질문)

**측정**: 부분집합의 train 행에서, 학습과 같은 가중치(proof 1.0, outcome 0.1)로 augmentation 없이 계산한 가중 MSE(이하 `W`). outcome과 proof 성분도 따로 보고한다.

- 측정 시점: 최종 step의 state, 그리고 T-val 선택 checkpoint. 둘 다 기록한다.
- 현재 trainer는 선택된 state만 돌려준다. 최종 state 측정 방법은 도구 단계에서 정한다(§9).

**바닥값(줄일 수 없는 오차)**
- 같은 입력에 서로 다른 target이 붙은 행은 어떤 모델도 동시에 맞출 수 없다. 예를 들어 같은 opening 국면이 결과가 다른 여러 게임에 나오는 경우다.
- 부분집합에서 입력이 같은 행끼리 묶어, 묶음마다 가중 평균 target으로 예측했을 때의 가중 MSE를 바닥값 F로 계산한다.
  - 묶는 기준 두 가지를 모두 보고한다: 정확히 같은 국면, D4 대칭까지 같은 국면.
  - 학습이 D4 augmentation을 쓰므로, 문턱에는 D4 기준(F_D4)을 쓴다.
- 학습 전에 계산해 기록한다. target 충돌이 큰 부분집합이면 문턱 해석에 그 크기를 함께 적는다.

**문턱과 판정** (제안값, 결과 전에 고정)

1. **적합 성공(군·seed별)**: 최종 state의 `W ≤ F_D4 + 0.01`
   - 근거: 0.01은 squared value error 단위로, 행 평균 |v − t|가 대략 0.1 수준인 근접 적합에 해당한다.
   - parent의 시작 `W`(c5 head, 부분집합 기준)와 줄인 비율 (W₀ − W) / (W₀ − F_D4)도 함께 보고한다.
2. **폭 효과(주 판정)**: 세 seed 모두 `W_wide − W_narrow < 0`(최종 state 기준)이고, 그 크기가 control 군의 seed 범위(세 seed `W_narrow`의 max − min)보다 클 때 "train에서 폭 효과 있음"
   - seed 일치 조건: 세 seed의 방향이 모두 같아야 한다. 하나라도 다르면 "seed 불일치"로 기록한다.
   - 이 판정에 H-dev의 ε를 쓰지 않는다. ε는 H-dev 주 지표용이다.

| train 결과 | 기록 | 다음 |
|---|---|---|
| 폭 효과 있음 | 1,000행에서 폭이 train 적합을 개선 | 5,000행 단계(§7) |
| 두 head 모두 적합 성공, 폭 효과 없음 | 1,000행에서는 폭이 병목이 아님 | 5,000행 단계(기존 head가 맞추지 못하는 규모를 찾는다) |
| 두 head 모두 적합 실패, 폭 효과 없음 | 폭만으로는 설명되지 않음 | 규모를 키우지 않는다. 최적화(lr)와 target 충돌(바닥값 대비 잔여)을 후속 가설로 남긴다 |
| seed 불일치 | 그대로 기록 | 자동 반복이나 seed 추가를 하지 않는다. 다음 행동을 따로 정한다 |

## 5. H-dev 판독 (train과 분리)

- 대상: 각 군의 T-val 선택 checkpoint(기존 objective와 H6 guard, NO_CANDIDATE 보존)
- 판정은 R2 §7 그대로다.
  - I_s = L(control_s) − L(wide_s), L은 H-dev 주 지표(D4 예측, network 행, game macro)
  - 세 seed 모두 I_s > ε(0.0019461168)이고, paired 95% 상한 < 0이면 `DEVELOPMENT_SIGNAL`
  - control의 seed 범위가 ε보다 크면 `HOLD`. 작은 자료의 실행은 seed 변동이 클 수 있으므로 이 보류 규칙을 그대로 적용한다.
  - 그 밖은 `NOT_ESTABLISHED`
- outcome / proof 성분 macro, 종류·종류×ply 가산 분해, c5 대비는 따로 보고한다. ε는 쓰지 않는다.
- c5, control, 실험군을 같은 H-dev 입력과 같은 실행 환경에서 함께 추론한다(D3 판독과 같은 방식).
- 1,000행·5,000행 단계의 H-dev 판독은 진단 기록이다. `DEVELOPMENT_SIGNAL`이어도 그 단계의 모델은 D4 후보가 아니다.
- **train과 H-dev의 관계**
  - train에서만 개선되면 "train 적합 개선, H-dev 미입증"까지만 기록한다.
  - 두 head가 모두 train을 못 맞추면 최적화와 target 충돌을 후속 가설로 남긴다.
  - H-dev 결과로 train 판정을 바꾸지 않는다. 그 반대도 같다.

## 6. 보고

- 군·seed별 출력
  - train `W`(최종·선택), 성분별 값, 바닥값, 줄인 비율, 적합 성공 여부
  - H-dev 판독, 성분, 분해
- 확장 검증값, 자료 배정 지문 비교, 부분집합 SHA와 구성
- step 수, GPU 시간(전체, step당)
- 학습 곡선(T-val objective, cycle·H6 val), 선택 step, 출처 SHA

## 7. 단계와 넘어가는 조건

1. **1,000행 (이 계약의 첫 실행)**: §4 표의 결과에 따라 다음을 정한다.
2. **5,000행**: 1,000행에서 "폭 효과 있음" 또는 "둘 다 적합 성공"일 때 넘어간다.
   - 부분집합(§3), 판정 규칙, step 예산 결정 방식을 그대로 쓴다.
   - step 예산을 규모에 맞춰 바꿀지는 1,000행 실행 전에 함께 정한다(§9).
3. **전체 규모(35,816행)**: 5,000행에서도 세 seed 모두 train 폭 효과가 유지될 때만 검토한다.
   - 이때 control은 전체 c8 자료의 기존 head다. R2-0B 실행을 재사용할지, R0처럼 재현 점검으로 연결할지는 그 단계의 계약에서 정한다.
   - D4 진입은 이 단계의 H-dev `DEVELOPMENT_SIGNAL`과 별도 계약으로만 검토한다.
4. **lr·BN 진단(별도 계약)**: 1,000행에서 두 head가 모두 적합에 실패했을 때, 또는 seed 불일치일 때 검토한다.
   - 한 번에 하나만 바꾼다: 학습률만, 또는 value head BN 동작(train 모드 / 고정 통계)만
   - 같은 부분집합·같은 head의 원래 설정을 control로 한다(R2 §8).
- 어느 단계든 결과를 보고 문턱이나 규칙을 바꾸지 않는다. 자동으로 다음 단계를 시작하지 않는다. 단계마다 따로 승인받는다.

## 8. 예상 비용 (계획값, 측정 아님)

- 1,000행 단계
  - 학습 6회(2군 × 3 seed), 각 210 step
  - batch 크기는 c8과 같은 1024다. epoch마다 cycle 약 1,000행과 같은 수의 replay 행이므로 epoch당 2 step, 105 epoch다.
  - 참고로 전체 c8 자료의 R2-0B·D3 학습은 seed당 약 4분이었다(epoch당 70 step).
  - 이 단계의 실제 시간은 측정값으로 보고한다.
- H-dev 판독: c5와 6개 모델을 함께 추론한다. D3 판독(7개 모델, 0.36분)과 비슷한 규모로 예상한다.

## 9. 확정 전 정할 것 (§10에서 확정)

1. `value_channels` 값: 제안 4. 대안 2 / 8은 후속으로 둔다.
2. 새 채널 초기화의 고정 seed와 분포(제안: 네트워크 기본 Kaiming), 예측 보존 허용 오차(제안 1e-6)
3. 부분집합 선택 seed와 "처음 1,000행 이상" 규칙, 5,000행 집합의 동시 고정. seed 충돌은 저장소 기록에서 확인한다.
4. step 예산: 제안 210(c8과 같은 step 수). 적합을 보기에 부족할 수 있으므로, 더 큰 예산을 쓸지 실행 전에 정한다. 두 군은 반드시 같다.
5. train 적합 문턱: 제안 `W ≤ F_D4 + 0.01`. 바닥값의 묶음 기준(정확 / D4)과 보고 방식
6. 최종 state 측정 방법
   - 현재 trainer는 선택 state만 반환한다.
   - (가) trainer에 최종 state 반환 옵션 추가: 옵션을 생략한 경로가 비트 단위로 같은지 확인해야 한다. trainer는 R0가 재현한 소스이므로, 이 경우 D1은 R0와의 연결 대신 자체 control로만 비교한다.
   - (나) 선택 checkpoint만으로 적합을 판정
   - 중 하나를 정한다.
7. 넓은 head의 checkpoint를 기존 평가 도구가 읽을 수 있는지(`model_config`의 `value_channels`) 확인하고, 필요한 도구 범위를 정한다.
8. 계약 JSON(조건, 부분집합 SHA, 출처 SHA, 판정, 출력 경로)과 단계별 승인 구조(D3와 같은 실험 식별값 방식)

도구 구현, GPU 학습, H-dev 평가, D4, 새 self-play는 이 초안의 범위가 아니며 각각 별도 승인을 받는다.

## 10. 1,000행 단계 확정과 실행 절차

### 10.1 §9 항목의 확정값

모두 결과를 보기 전에 계약 JSON에 고정했다. 바꾸려면 새 계약이 필요하다.

1. **폭**: `value_channels` 1(control) 대 4. 2 / 8은 후속이다.
2. **새 채널 초기화와 허용 오차**
   - 분포: `kaiming_normal_(fan_out, relu)`(네트워크 기본과 같음)
   - seed별 전용 생성기 seed: 31001 → 2026110710, 31002 → 2026110711, 31003 → 2026110712
   - 새 모델을 만들 때의 기본 초기화도 전역 난수를 소비하지 않는다(`fork_rng`). 테스트로 확인한다.
   - 예측 보존 검사(v2, §11): 바이트 검사 + CPU float64 예측 차이 ≤ 1e-12. GPU float32 차이는 기록만 한다. v1의 float32 허용 오차 1e-6은 실패했다(§11.1).
3. **부분집합**: 선택 seed 2026110701, 규칙은 §3 그대로("처음 1,000행 이상"에서 멈춤, 게임을 쪼개지 않음). 같은 순서의 5,000행 집합도 prepare 기록에 함께 남긴다(게임 수, 행 수, 게임 목록 SHA).
4. **step 예산: 210, 두 군 같음.** epoch 수는 210 / ceil(2n / 1024)로 정하고, 도구가 실제 step 수가 210인지 확인한다. 더 큰 예산은 쓰지 않는다. 210 step에서 적합이 부족하면 그것도 결과로 기록한다(§4의 `NEITHER_FIT` / `MIXED_FIT`).
5. **train 적합 문턱과 판정 순서**
   - 적합 성공: 최종 step의 `W ≤ F_D4 + 0.01`. 바닥값은 정확 / D4 두 기준을 모두 기록하고, 문턱에는 D4 기준을 쓴다.
   - 판정은 아래 순서에서 처음 맞는 것이다(`h7r_d1.fit_reading`).

| 순서 | 분류 | 조건 | §4 표 |
|---|---|---|---|
| 1 | `WIDTH_EFFECT` | 세 seed 모두 `W_wide − W_narrow < 0`, 각 크기가 narrow seed 범위보다 큼 | 폭 효과 있음 |
| 2 | `BOTH_FIT` | 6개 실행 모두 적합 성공. 방향은 판독하지 않는다 | 두 head 모두 적합 성공 |
| 3 | `SEED_DISAGREEMENT` | 세 seed의 차이 부호가 다름(0 포함) | seed 불일치 |
| 4 | `NEITHER_FIT` | 6개 실행 모두 적합 실패 | 두 head 모두 적합 실패 |
| 5 | `MIXED_FIT` | 일부만 적합 성공, 방향은 같으나 폭 효과 조건 미달 | 표에 없던 경우. 다음 행동을 따로 정한다 |

6. **최종 state 측정: 방법 (가)의 변형.** trainer는 고치지 않는다.
   - trainer 자신의 평가 호출을 관찰한다(`fit_probe`). 평가는 parent와 평가 step마다 일어나고, 마지막 평가는 최종 step이다(`evaluation_steps`). 매번 같은 모델의 부분집합 `W`를 재고, 마지막 평가 때의 state를 복사한다.
   - trainer 소스, 난수 흐름, 선택 결과는 바뀌지 않는다. 관찰이 있을 때와 없을 때 학습 결과가 비트 단위로 같은지 테스트로 확인한다.
   - 그래서 R0가 재현한 trainer 소스를 그대로 쓴다.
   - 기록은 둘로 나눈다.
     - `last_step`: 최종 step(210)의 state와 `W`. train 판정에 쓴다.
     - `selected`: T-val 선택 checkpoint의 step, objective와 `W`. H-dev 판독에 쓴다.
   - 두 기록을 섞지 않는다. `NO_CANDIDATE`면 `selected`는 비우고 그대로 둔다. 그 경우 H-dev 판독은 멈춘다.
7. **넓은 head checkpoint 로딩**
   - checkpoint metadata의 `model_config`에 `value_channels`를 기록한다.
   - 기존 평가 로더(`h7r_r2.load_value_model`, `check_model`)가 그대로 읽는다. 1채널 설정으로 4채널 state를 읽으면 실패한다. 둘 다 테스트로 확인한다.
   - prepare는 확장 parent를 저장한 뒤 이 로더로 다시 읽어, 예측 차이가 정확히 0인지 확인한다.
8. **계약과 승인**
   - D3처럼 승인과 무관한 실험 식별값(`experiment_sha256`)으로 단계를 잇는다. 계약 파일 SHA는 출처로만 기록한다.
   - 승인은 `prepare → train → evaluate` 순서다. 앞 단계가 승인되지 않으면 다음 단계는 거부된다.
   - `fit`은 train 승인으로 돈다. `evaluate`는 커밋된 `fit.json`이 실행 기록에서 다시 계산한 판정과 같을 때만 돈다. train 판정을 H-dev 예측 전에 고정하기 위해서다.

**§2 검증 항목의 구현** (v2 기준, 자세한 내용은 §11.2)
- 검사한 행에서 초기 예측 보존을 확인한다: 부분집합 전체, cycle val 전체, H6 val 고정 표본 4,096행. H-dev는 prepare에서 쓰지 않는다(evaluate 전까지 손대지 않음).
- T-val objective의 보존은 따로 주장하지 않는다. 예측 차이의 상한은 제곱손실 차이의 상한과 다르고, H6 val은 표본만 검사한다.
- train 모드 예측 보존을 확인한다: 부분집합 앞 1,024행을 한 batch로 value head BN batch 통계로 추론한다. 실제 학습 첫 batch(replay 혼합, augmentation)의 손실을 비교하는 검사는 아니다.
- 두 군의 parent `W`(학습 시작 전)도 run 기록에 남는다.

**prepare 출력** (`R2-D1-HEAD-WIDTH-1/1k-v2/prepare.json`, 커밋)
- 부분집합: 게임 수, 행 수, 게임 SHA, 5,000행 집합
- 구성: outcome / proof 행과 가중치, ply 구간
- 바닥값: 정확 / D4 각각 F, 묶음 수, 충돌 묶음 수
- seed별 확장 parent
  - 판정 4개(`gates`): 바이트, float64, 출력 유한성, 재로딩
  - float64 / GPU 비교: 최대 차이, 그 행, 두 값, 차이 난 행 수
  - 층별 진단, 재로딩 결과, checkpoint SHA
- 공통: GPU 연산 설정, c5 batch 분할 비교, H6 val 표본 SHA
- `passed`가 false여도 기록은 남긴다. 그 경우 도구는 실패로 끝나고 학습은 거부된다.

### 10.2 실행 절차 (데스크톱)

- 각 단계는 승인 커밋을 받은 뒤 `git pull --ff-only`로 같은 HEAD에서 실행한다.
- 실패하면 다음 단계로 넘어가지 않는다. 로그와 이미 쓴 기록은 지우지 않고 보고한다.
- 어떤 출력도 덮어쓰지 않는다. 같은 경로가 있으면 도구가 멈춘다.

1. prepare와 기록 커밋 (`approved.prepare` 커밋 후; 학습 없음. c5와 확장 parent 3개의 CPU float64 추론과 GPU 추론만)
   - prepare가 실패해도 기록 단계까지 진행한다. 실행 결과(종료 코드)는 저장했다가 마지막에 그대로 반환한다.
   - 이번 시도의 로그는 `1k-v2/prepare-logs/`에 시도별 manifest와 함께 복사한다. `prepare.json`은 쓰였으면(PASS든 FAIL이든) 같이 커밋한다.

```bash
(
  set -euo pipefail
  cd ~/renju-h6
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src:$PWD/scripts"
  git pull --ff-only origin feat/h7-c1
  D=docs/h7r/r2/R2-D1-HEAD-WIDTH-1
  X="$D/1k-v2"
  L=runs/h7r/R2-D1-HEAD-WIDTH-1/1k-v2/logs
  test -z "$(git status --porcelain --untracked-files=no)"
  python scripts/check_track_b_frozen.py
  test ! -e "$X/prepare.json"
  mkdir -p "$L"
  LOG="$L/prepare-$(date +%Y%m%d-%H%M%S).log"
  B="$(basename "$LOG" .log)"
  ts(){ while IFS= read -r line; do printf '%s %s\n' "$(date +%s.%N)" "$line"; done; }
  run_rc=0
  { echo __START__; rc=0; python -u scripts/h7r_d1.py prepare --d1-contract "$D/d1-1k-v2-contract.json" \
      --cycle-cache runs/h7c/c8/value_cache --cycle-rows runs/h7c/c8/value_rows.jsonl.gz \
      --replay-cache data/tactical/h6/value_cache_v3 --c5 runs/h7c/c5/value/best.pt --device cuda 2>&1 || rc=$?
    echo "__EXIT__ $rc"; exit "$rc"; } | ts | tee "$LOG" || run_rc=$?
  python scripts/h7r_d4_records.py copy-logs --src-dir "$L" --pattern "$B.log" \
    --dest-dir "$X/prepare-logs" --manifest "$X/prepare-logs/$B.manifest.json"
  expect=(--expect "$X/prepare-logs/$B.log" --expect "$X/prepare-logs/$B.manifest.json")
  git add "$X/prepare-logs/$B.log" "$X/prepare-logs/$B.manifest.json"
  if [ -f "$X/prepare.json" ]; then
    git add "$X/prepare.json"
    expect+=(--expect "$X/prepare.json")
  fi
  python scripts/h7r_d4_records.py verify-staged "${expect[@]}"
  if [ "$run_rc" -eq 0 ]; then verdict=PASS; else verdict="FAIL (exit $run_rc)"; fi
  git commit -m "H7-R D1 v2: 1,000-row prepare $verdict: subset, floors and widened parents (records only)"
  git push origin feat/h7-c1
  echo "__RECORDED__ $run_rc"
  exit "$run_rc"
)
```

1-F. 실패 기록 보존 (1번 블록이 기록 커밋 전에 멈췄을 때만; 학습·추론 없음)
   - 예: 로그 복사, `verify-staged`, commit, push가 실패했거나 터미널이 닫혔을 때
   - 가장 최근 prepare 로그와, 있으면 아직 커밋되지 않은 `prepare.json`을 커밋한다. 이미 복사된 로그는 다시 복사하지 않는다.

```bash
(
  set -euo pipefail
  cd ~/renju-h6
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src:$PWD/scripts"
  D=docs/h7r/r2/R2-D1-HEAD-WIDTH-1
  X="$D/1k-v2"
  L=runs/h7r/R2-D1-HEAD-WIDTH-1/1k-v2/logs
  echo __START__
  test -z "$(git status --porcelain --untracked-files=no)"
  LOG="$(ls -1t "$L"/prepare-*.log | head -n 1)"
  B="$(basename "$LOG" .log)"
  echo "log: $LOG"
  tail -n 5 "$LOG"
  expect=()
  if [ ! -e "$X/prepare-logs/$B.manifest.json" ]; then
    python scripts/h7r_d4_records.py copy-logs --src-dir "$L" --pattern "$B.log" \
      --dest-dir "$X/prepare-logs" --manifest "$X/prepare-logs/$B.manifest.json"
  fi
  for f in "$X/prepare-logs/$B.log" "$X/prepare-logs/$B.manifest.json" "$X/prepare.json"; do
    if [ -f "$f" ] && ! git ls-files --error-unmatch "$f" >/dev/null 2>&1; then
      git add "$f"
      expect+=(--expect "$f")
    fi
  done
  if [ "${#expect[@]}" -eq 0 ]; then echo "nothing left to record"; echo "__EXIT__ 0"; exit 0; fi
  python scripts/h7r_d4_records.py verify-staged "${expect[@]}"
  git commit -m "H7-R D1 v2: 1,000-row prepare attempt record $B (records only)"
  git push origin feat/h7-c1
  echo "__EXIT__ 0"
)
```

2. 학습 6회 (`approved.train` 커밋 후; seed마다 narrow, wide 순서)

```bash
(
  set -euo pipefail
  cd ~/renju-h6
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src:$PWD/scripts"
  git pull --ff-only origin feat/h7-c1
  D=docs/h7r/r2/R2-D1-HEAD-WIDTH-1
  R=runs/h7r/R2-D1-HEAD-WIDTH-1/1k-v2
  L="$R/logs"
  test -z "$(git status --porcelain --untracked-files=no)"
  python scripts/check_track_b_frozen.py
  head="$(git rev-parse HEAD)"
  ts(){ while IFS= read -r line; do printf '%s %s\n' "$(date +%s.%N)" "$line"; done; }
  for s in 31001 31002 31003; do
    for arm in narrow wide; do
      test "$(git rev-parse HEAD)" = "$head"
      test ! -e "$R/$arm-$s"
      LOG="$L/train-$arm-$s-$(date +%Y%m%d-%H%M%S).log"
      { echo __START__; rc=0; python -u scripts/h7r_d1.py train --d1-contract "$D/d1-1k-v2-contract.json" \
          --arm "$arm" --seed "$s" \
          --cycle-cache runs/h7c/c8/value_cache --cycle-rows runs/h7c/c8/value_rows.jsonl.gz \
          --replay-cache data/tactical/h6/value_cache_v3 --c5 runs/h7c/c5/value/best.pt --device cuda 2>&1 || rc=$?
        echo "__EXIT__ $rc"; exit "$rc"; } | ts | tee "$LOG"
    done
  done
)
```

3. train 적합 판독과 기록 커밋 (학습과 같은 HEAD에서; checkpoint는 복사하지 않는다. run 기록은 `.gitignore`의 `runs/` 규칙을 피해 `run-records/`에 둔다)

```bash
(
  set -euo pipefail
  cd ~/renju-h6
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src:$PWD/scripts"
  D=docs/h7r/r2/R2-D1-HEAD-WIDTH-1
  X="$D/1k-v2"
  R=runs/h7r/R2-D1-HEAD-WIDTH-1/1k-v2
  L="$R/logs"
  test -z "$(git status --porcelain --untracked-files=no)"
  test ! -e "$X/fit.json"
  LOG="$L/fit-$(date +%Y%m%d-%H%M%S).log"
  ts(){ while IFS= read -r line; do printf '%s %s\n' "$(date +%s.%N)" "$line"; done; }
  { echo __START__; rc=0; python -u scripts/h7r_d1.py fit --d1-contract "$D/d1-1k-v2-contract.json" 2>&1 || rc=$?
    echo "__EXIT__ $rc"; exit "$rc"; } | ts | tee "$LOG"
  expect=(--expect "$X/fit.json")
  for s in 31001 31002 31003; do
    for arm in narrow wide; do
      mkdir -p "$X/run-records/$arm-$s"
      for f in d1_train.json record.json; do
        cp --no-clobber "$R/$arm-$s/$f" "$X/run-records/$arm-$s/$f"
        cmp "$R/$arm-$s/$f" "$X/run-records/$arm-$s/$f"
        expect+=(--expect "$X/run-records/$arm-$s/$f")
      done
    done
  done
  git add "$X/fit.json" "$X/run-records"
  python scripts/h7r_d4_records.py verify-staged "${expect[@]}"
  git commit -m "H7-R D1 v2: 1,000-row runs and the train-fit reading (records only)"
  git push origin feat/h7-c1
)
```

4. H-dev 판독과 기록 커밋 (`approved.evaluate` 커밋 후)

```bash
(
  set -euo pipefail
  cd ~/renju-h6
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src:$PWD/scripts"
  git pull --ff-only origin feat/h7-c1
  D=docs/h7r/r2/R2-D1-HEAD-WIDTH-1
  X="$D/1k-v2"
  R=runs/h7r/R2-D1-HEAD-WIDTH-1/1k-v2
  L="$R/logs"
  test -z "$(git status --porcelain --untracked-files=no)"
  python scripts/check_track_b_frozen.py
  test ! -e "$X/d1_eval.json"
  LOG="$L/evaluate-$(date +%Y%m%d-%H%M%S).log"
  ts(){ while IFS= read -r line; do printf '%s %s\n' "$(date +%s.%N)" "$line"; done; }
  { echo __START__; rc=0; python -u scripts/h7r_d1.py evaluate --d1-contract "$D/d1-1k-v2-contract.json" \
      --cache runs/h7r/R1-HDEV-1/cache --c5 runs/h7c/c5/value/best.pt --device cuda 2>&1 || rc=$?
    echo "__EXIT__ $rc"; exit "$rc"; } | ts | tee "$LOG"
  python scripts/h7r_d4_records.py copy-logs --src-dir "$L" --pattern '*.log' \
    --dest-dir "$X/run-logs" --manifest "$X/log_manifest.json"
  python scripts/h7r_d4_records.py manifest --dir "$X" --output "$X/records_manifest.json"
  expect=(--expect "$X/d1_eval.json" --expect "$X/log_manifest.json" --expect "$X/records_manifest.json")
  for f in "$X"/run-logs/*.log; do expect+=(--expect "$f"); done
  git add "$X/d1_eval.json" "$X/log_manifest.json" "$X/records_manifest.json" "$X"/run-logs/*.log
  python scripts/h7r_d4_records.py verify-staged "${expect[@]}"
  git commit -m "H7-R D1 v2: 1,000-row H-dev reading (records only)"
  git push origin feat/h7-c1
)
```

- 3번은 train 승인 범위다. 결과(`fit.json`)를 확인한 뒤 evaluate 승인을 따로 받는다.
- H-dev 판독이 어떻게 나와도 이 단계의 모델은 D4 후보가 아니다. 5,000행 단계는 §7에 따라 별도 승인으로 정한다.

## 11. v1 prepare 실패와 v2 수정 계약

### 11.1 v1 결과 (기록 `R2-D1-HEAD-WIDTH-1/1k/prepare.json`, 커밋 `37e8c45`)

- 실행: `e66cc38`의 clean 상태, v1 계약 SHA `e28e17b8…`. 약 11초 만에 학습 시작 전에 멈췄다.
- 판정: **prepare FAIL**. 미리 정한 float32 허용 오차 1e-6을 넘었다. 이 판정은 바꾸지 않는다. v1 계약은 여기서 끝난다.
- 부분집합: 20게임 1,024행(5,000행 집합 111게임 5,174행). outcome 939행, proof 85행
- 바닥값: 정확 기준 F = 0(충돌 0), D4 기준 F_D4 = 0.001118(충돌 묶음 1)

| 검사 (세 seed 모두 같은 값) | 결과 | 기준 1e-6 |
|---|---:|---|
| 동결 state·복사 텐서·새 Linear 열 0 | 전부 통과 | 통과 |
| 부분집합 예측 최대 차이 | 1.00583e-6 | 초과 |
| cycle val 예측 최대 차이 | 1.62236e-6 | 초과 |
| H6 replay val 예측 최대 차이 | 0 | 통과 |
| train 모드 예측 차이 | 9.53674e-7 | 통과 |
| 저장·재로딩 예측 차이 | 0 | 통과 |

- **원인은 확정되지 않았다.**
  - 텐서 복사 오류를 보여 주는 항목은 없다.
  - GPU float32 연산 순서 차이(첫 Linear 입력이 225에서 900으로 커지면서 바뀐 커널)는 **가설**이다.
  - 세 seed의 값이 같다는 사실로는 원인을 정할 수 없다. 새 채널에서 나가는 가중치가 0이면, conv 초기값이 seed마다 달라도 출력이 같기 때문이다.
- **검사 설계의 한계**: 1e-6은 float32 누산 순서 차이만 허용하려던 값이었다. GPU에서 그 차이가 얼마인지 측정하지 않고 정했다.

### 11.2 v2에서 바꾼 것 (`d1-1k-v2-contract.json`)

- **v1에서 그대로 가져오는 것** (도구가 v1 계약과 같은지 확인)
  - 부분집합, step 예산, 두 군, seed, 새 채널 초기화와 그 seed
  - 적합 판정과 H-dev 판독 규칙
- v1 계약과 v1 실패 기록은 SHA로 고정해 `supersedes`에 남긴다. 실패 기록의 `passed`가 false인지도 확인한다.
- 새 실험 식별값을 쓴다. 출력은 `1k-v2/`와 `runs/.../1k-v2/`에 쓴다. v1 출력은 건드리지 않는다.

**예측 보존 판정 (하나라도 실패하면 학습 거부)**
1. 바이트 검사: v1과 같다.
2. CPU float64 검사: 최대 차이 ≤ 1e-12. 1e-12는 float64 오차(약 1e-16) 기준으로 정한 값이다. 이번 v1 관측값에 맞춘 값이 아니다.
   - 방법: c5와 확장 parent를 각각 복사해 모델과 입력을 double로 바꾼다. 출력도 float64로 유지한다(`exact_values`). 원래 모델의 상태는 바꾸지 않는다.
   - eval 모드 대상: 부분집합 전체, cycle val 전체, H6 val의 고정 seed 표본 4,096행(seed 2026110702)
     - H6 val 전체는 106,223행이라 CPU float64로는 오래 걸린다.
     - 표본의 행 목록 SHA를 기록한다.
   - train 모드: 부분집합 앞 1,024행을 한 batch로, eval 검사와 별도 복사본에서 한다.
3. 출력 유한성: GPU 출력(eval·train 모드)에 NaN·Inf가 있으면 실패한다.
4. 재로딩: 저장한 checkpoint를 평가 로더로 다시 읽은 state SHA가 확장 parent와 같아야 한다. GPU 예측 차이는 정확히 0이어야 한다.

**기록만 하는 값 (판정에 쓰지 않음)**
- GPU float32 차이(부분집합, cycle val, H6 val 전체, train 모드)
  - 각 비교마다: 최대 차이, 그 행, 두 값, 차이 난 행 수
  - batch 크기 4,096
- 층별 진단(`layer_differences`): 같은 입력에서 기존 head와 확장 head의 단계별 출력을 비교한다.
  - 단계: trunk, conv, BN, ReLU(복사 채널), 첫 Linear, ReLU, 둘째 Linear, tanh
  - 첫 Linear는 두 가지로 본다. 확장 head 전체 곱과, 복사한 225열만으로 다시 계산한 값이다. 차이가 복사된 연산에서 생기는지, 넓어진 곱에서 생기는지를 구분하기 위해서다.
- 기준 잡음: c5 자신을 batch 4,096과 1,024로 나눠 예측한 차이
- 연산 설정
  - torch / CUDA / cuDNN 버전, GPU 이름
  - TF32 허용 여부(cuDNN, matmul), `float32_matmul_precision`
  - cuDNN benchmark·deterministic, CPU thread 수

**테스트로 확인한 것** (`tests/test_h7r_d1.py`)
- 복사 오류 각각이 바이트 검사와 float64 예측 검사에서 모두 잡힌다: conv 채널 0, BN 채널 0 통계, 복사 Linear 열 전체, 새 Linear 열, 둘째 Linear
- 표본 행이 닿지 않는 국소 오류는 float64 검사를 통과하고 바이트 검사에서만 잡힌다. 그래서 float64 표본 검사는 바이트 검사와 함께만 해석한다.
- 새 채널 초기값만 다른 확장은 오류가 아니다(통과).
- float64 출력의 dtype이 float64이고, 원래 모델의 state·dtype·모드가 바뀌지 않는다.
- 비정상 값: NaN·Inf가 있으면 비교가 무효가 되고, prepare가 학습을 거부한다.
- 층별 진단: 정상 확장에서 trunk와 복사 열 계산의 차이가 0이다. 새 열을 오염시키면 차이가 첫 Linear에서 처음 나타난다.
- prepare 전체: 정상이면 PASS와 진단 기록, 깨진 확장이면 FAIL과 학습 거부

**예상 비용**: 확장 parent당 CPU float64 약 7,900행 추론(c5는 한 번). 이 컨테이너의 4 thread 기준 약 350행/초로, 모델당 20초 남짓이다. 데스크톱 실제 시간은 로그로 보고한다.
