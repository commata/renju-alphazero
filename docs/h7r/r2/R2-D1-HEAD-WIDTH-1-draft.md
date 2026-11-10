# R2-D1-HEAD-WIDTH-1 — value head 폭 진단 계약 초안

- 작성일 2026-11-06, 개정 2026-11-09. 상태: **1,000행·5,000행 완료: 두 단계 모두 train `WIDTH_EFFECT`, H-dev `DEVELOPMENT_SIGNAL`(§12, §13) / 전체 규모 계약 고정·도구 구현, 단계 승인 전(§14)**
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
  [ "$(git branch --show-current)" = feat/h7-c1 ] || { echo "STOP: wrong branch"; exit 1; }
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
   - 중단된 commit이 남긴 staged 상태는 기록 폴더(`$X`)에 한해서만 먼저 해제한다(파일은 그대로 둔다).

```bash
(
  set -euo pipefail
  cd ~/renju-h6
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src:$PWD/scripts"
  [ "$(git branch --show-current)" = feat/h7-c1 ] || { echo "STOP: wrong branch"; exit 1; }
  D=docs/h7r/r2/R2-D1-HEAD-WIDTH-1
  X="$D/1k-v2"
  L=runs/h7r/R2-D1-HEAD-WIDTH-1/1k-v2/logs
  echo __START__
  git reset -q -- "$X"
  test -z "$(git status --porcelain --untracked-files=no)"
  shopt -s nullglob
  logs=("$L"/prepare-*.log)
  [ "${#logs[@]}" -gt 0 ] || { echo "STOP: no prepare log"; exit 1; }
  LOG="${logs[0]}"
  for f in "${logs[@]}"; do if [ "$f" -nt "$LOG" ]; then LOG="$f"; fi; done
  B="$(basename "$LOG" .log)"
  echo "log: $LOG"
  tail -n 5 "$LOG"
  expect=()
  python scripts/h7r_d4_records.py copy-logs --resume --src-dir "$L" --pattern "$B.log" \
    --dest-dir "$X/prepare-logs" --manifest "$X/prepare-logs/$B.manifest.json"
  for f in "$X/prepare-logs/$B.log" "$X/prepare-logs/$B.manifest.json" "$X/prepare.json"; do
    if [ -f "$f" ] && ! git ls-files --error-unmatch "$f" >/dev/null 2>&1; then
      git add "$f"
      expect+=(--expect "$f")
    fi
  done
  if [ "${#expect[@]}" -eq 0 ]; then
    git push origin HEAD:feat/h7-c1
    echo "nothing left to record; push checked"; echo "__EXIT__ 0"; exit 0
  fi
  python scripts/h7r_d4_records.py verify-staged "${expect[@]}"
  git commit -m "H7-R D1 v2: 1,000-row prepare attempt record $B (records only)"
  git push origin feat/h7-c1
  echo "__EXIT__ 0"
)
```

2. 학습 6회, train 적합 판독, 기록 커밋 (`approved.train` 커밋 후)
   - **실행 순서**: 같은 승인 HEAD에서 seed마다 narrow → wide 순서로 돈다. 한 실행이 실패하면(또는 HEAD가 바뀌면) 뒤 실행은 시작하지 않는다.
   - **fit**: 6회가 모두 성공했을 때만 돈다. 판독은 각 실행의 마지막 step(210) state로 하고, 검증 기준으로 선택된 checkpoint는 따로 기록된다(§10.1-6).
   - **실패해도 기록 단계까지 간다**
     - 종료 코드와 실패 위치를 커밋 메시지와 `__RECORDED__` 줄에 남긴다.
     - 이번 시도의 로그는 모두 `train-logs/`에 manifest와 함께 복사한다.
     - 실행 폴더가 실제로 생긴 실행(완료된 실행, 그리고 실패한 실행 중 폴더가 생긴 경우)의 JSON 파일만 `run-records/`에 복사한다. checkpoint(`.pt`)는 복사하지 않는다.
   - **기록 검증**: 복사본은 원본과 바이트 비교(`cmp`)한다. `run-records/`의 파일 목록과 SHA는 manifest로 남긴다(`.pt`가 섞이면 거부). 커밋 직전에 staged 파일 목록이 기대 목록과 정확히 같은지 확인한다(`verify-staged`).
   - **다시 실행하지 않는다**: 실행 폴더나 이전 기록이 있으면 도구와 블록이 멈춘다. 실패하면 실패 폴더를 지우거나 블록 전체를 다시 돌리지 않는다. 기록된 중단 위치를 보고 다음 처리를 따로 정한다.

```bash
(
  set -euo pipefail
  cd ~/renju-h6
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src:$PWD/scripts"
  [ "$(git branch --show-current)" = feat/h7-c1 ] || { echo "STOP: wrong branch"; exit 1; }
  git pull --ff-only origin feat/h7-c1
  D=docs/h7r/r2/R2-D1-HEAD-WIDTH-1
  X="$D/1k-v2"
  R=runs/h7r/R2-D1-HEAD-WIDTH-1/1k-v2
  L="$R/logs"
  test -z "$(git status --porcelain --untracked-files=no)"
  python scripts/check_track_b_frozen.py
  test ! -e "$X/fit.json"
  test ! -e "$X/run-records"
  test ! -e "$X/train-logs"
  for s in 31001 31002 31003; do for arm in narrow wide; do test ! -e "$R/$arm-$s"; done; done
  head="$(git rev-parse HEAD)"
  STAMP="$(date +%Y%m%d-%H%M%S)"
  mkdir -p "$L"
  ts(){ while IFS= read -r line; do printf '%s %s\n' "$(date +%s.%N)" "$line"; done; }
  run_rc=0
  where=none
  done_runs=0
  for s in 31001 31002 31003; do
    for arm in narrow wide; do
      if [ "$(git rev-parse HEAD)" != "$head" ]; then run_rc=97; where="HEAD changed before $arm-$s"; break 2; fi
      LOG="$L/train-$arm-$s-$STAMP.log"
      { echo __START__; rc=0; python -u scripts/h7r_d1.py train --d1-contract "$D/d1-1k-v2-contract.json" \
          --arm "$arm" --seed "$s" \
          --cycle-cache runs/h7c/c8/value_cache --cycle-rows runs/h7c/c8/value_rows.jsonl.gz \
          --replay-cache data/tactical/h6/value_cache_v3 --c5 runs/h7c/c5/value/best.pt --device cuda 2>&1 || rc=$?
        echo "__EXIT__ $rc"; exit "$rc"; } | ts | tee "$LOG" || run_rc=$?
      if [ "$run_rc" -ne 0 ]; then where="train $arm-$s"; break 2; fi
      done_runs=$((done_runs + 1))
    done
  done
  if [ "$run_rc" -eq 0 ] && [ "$(git rev-parse HEAD)" != "$head" ]; then
    run_rc=97; where="HEAD changed before fit"
  fi
  if [ "$run_rc" -eq 0 ]; then
    LOG="$L/fit-$STAMP.log"
    { echo __START__; rc=0; python -u scripts/h7r_d1.py fit --d1-contract "$D/d1-1k-v2-contract.json" 2>&1 || rc=$?
      echo "__EXIT__ $rc"; exit "$rc"; } | ts | tee "$LOG" || run_rc=$?
    if [ "$run_rc" -ne 0 ]; then where=fit; fi
  fi
  echo "[d1-train-block] runs completed $done_runs/6, exit $run_rc, stopped at: $where"
  python scripts/h7r_d4_records.py copy-logs --src-dir "$L" --pattern "*-$STAMP.log" \
    --dest-dir "$X/train-logs" --manifest "$X/train-logs/$STAMP.manifest.json"
  expect=(--expect "$X/train-logs/$STAMP.manifest.json")
  for f in "$X"/train-logs/*-"$STAMP".log; do expect+=(--expect "$f"); done
  copied=0
  for s in 31001 31002 31003; do
    for arm in narrow wide; do
      [ -d "$R/$arm-$s" ] || continue
      for src in "$R/$arm-$s"/*.json; do
        [ -f "$src" ] || continue
        dst="$X/run-records/$arm-$s/$(basename "$src")"
        mkdir -p "$(dirname "$dst")"
        cp --no-clobber "$src" "$dst"
        cmp "$src" "$dst"
        expect+=(--expect "$dst")
        copied=$((copied + 1))
      done
    done
  done
  if [ "$copied" -gt 0 ]; then
    python scripts/h7r_d4_records.py manifest --dir "$X/run-records" --output "$X/run-records/records_manifest-$STAMP.json"
    expect+=(--expect "$X/run-records/records_manifest-$STAMP.json")
  fi
  if [ -f "$X/fit.json" ]; then expect+=(--expect "$X/fit.json"); fi
  for e in "${expect[@]}"; do [ "$e" = --expect ] || git add "$e"; done
  python scripts/h7r_d4_records.py verify-staged "${expect[@]}"
  if [ "$run_rc" -eq 0 ]; then verdict="6/6 runs and the train-fit reading"; else verdict="FAIL at $where (exit $run_rc), $done_runs/6 runs"; fi
  git commit -m "H7-R D1 v2: 1,000-row training $verdict (records only)"
  git push origin feat/h7-c1
  echo "__RECORDED__ $run_rc"
  exit "$run_rc"
)
```

2-F. 학습 기록 보존 (2번 블록이 기록 커밋 전에 멈췄을 때만; 학습·판독 없음)
   - 예: 로그 복사, `cmp`, manifest, `verify-staged`, commit, push가 실패했거나 터미널이 닫혔을 때
   - 가장 최근 시도의 로그, 실제로 생긴 실행 폴더의 JSON, `fit.json` 가운데 아직 커밋되지 않은 것만 커밋한다.
   - 이미 복사된 파일은 원본과 다시 비교만 한다.
   - 로그 복사가 중간에 끊긴 경우에도 같은 SHA의 사본을 재사용하고, 기존 manifest를 다시 검증한다. 내용이 다르면 덮어쓰지 않고 멈춘다.
   - commit은 성공하고 push만 실패한 경우에는 새 커밋 없이 push를 재시도한다. 원격이 앞서 있으면 강제로 반영하지 않고 git 오류를 보고한다.
   - 기록 대상은 이번 시도의 로그·manifest와 원본에서 대조한 실행 JSON, 있으면 fit.json이다. 다른 파일을 폴더 전체 검색으로 추가하지 않는다.
   - 중단된 commit이 남긴 staged 상태는 기록 폴더(`$X`)에 한해서만 먼저 해제한다(파일은 그대로 둔다).

```bash
(
  set -euo pipefail
  cd ~/renju-h6
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src:$PWD/scripts"
  [ "$(git branch --show-current)" = feat/h7-c1 ] || { echo "STOP: wrong branch"; exit 1; }
  D=docs/h7r/r2/R2-D1-HEAD-WIDTH-1
  X="$D/1k-v2"
  R=runs/h7r/R2-D1-HEAD-WIDTH-1/1k-v2
  L="$R/logs"
  echo __START__
  git reset -q -- "$X"
  test -z "$(git status --porcelain --untracked-files=no)"
  shopt -s nullglob
  logs=("$L"/train-*.log)
  [ "${#logs[@]}" -gt 0 ] || { echo "STOP: no train log"; exit 1; }
  last="${logs[0]}"
  for f in "${logs[@]}"; do if [ "$f" -nt "$last" ]; then last="$f"; fi; done
  STAMP="$(basename "$last" .log | grep -oE '[0-9]{8}-[0-9]{6}$')"
  echo "attempt $STAMP"
  for f in "$L"/*-"$STAMP".log; do echo "== $f"; tail -n 3 "$f"; done
  python scripts/h7r_d4_records.py copy-logs --resume --src-dir "$L" --pattern "*-$STAMP.log" \
    --dest-dir "$X/train-logs" --manifest "$X/train-logs/$STAMP.manifest.json"
  candidates=("$X/train-logs/$STAMP.manifest.json" "$X"/train-logs/*-"$STAMP".log)
  for s in 31001 31002 31003; do
    for arm in narrow wide; do
      [ -d "$R/$arm-$s" ] || continue
      for src in "$R/$arm-$s"/*.json; do
        [ -f "$src" ] || continue
        dst="$X/run-records/$arm-$s/$(basename "$src")"
        mkdir -p "$(dirname "$dst")"
        cp --no-clobber "$src" "$dst"
        cmp "$src" "$dst"
        candidates+=("$dst")
      done
    done
  done
  if [ -d "$X/run-records" ]; then
    python scripts/h7r_d4_records.py manifest --resume --dir "$X/run-records" --output "$X/run-records/records_manifest-$STAMP.json"
    candidates+=("$X/run-records/records_manifest-$STAMP.json")
  fi
  if [ -f "$X/fit.json" ]; then candidates+=("$X/fit.json"); fi
  expect=()
  for f in "${candidates[@]}"; do
    if ! git ls-files --error-unmatch "$f" >/dev/null 2>&1; then git add "$f"; expect+=(--expect "$f"); fi
  done
  if [ "${#expect[@]}" -eq 0 ]; then
    git push origin HEAD:feat/h7-c1
    echo "nothing left to record; push checked"; echo "__EXIT__ 0"; exit 0
  fi
  python scripts/h7r_d4_records.py verify-staged "${expect[@]}"
  git commit -m "H7-R D1 v2: 1,000-row training attempt record $STAMP (records only)"
  git push origin feat/h7-c1
  echo "__EXIT__ 0"
)
```

3. H-dev 판독과 기록 커밋 (`approved.evaluate`와 평가 개정 A1의 `approved` 커밋 후)
   - **평가 도구**: `scripts/h7r_d1_evaluate.py`(평가 개정 A1, §12.6). `scripts/h7r_d1.py evaluate`는 쓰지 않는다.
   - **평가 대상**: c5와 T-val로 선택된 checkpoint 6개(`best.pt`). 마지막 step의 state(`final.pt`)는 평가하지 않는다.
   - **판독**: R2 §7 그대로. 같은 seed의 narrow가 control이다.
   - **사전 검사**
     - 도구가 확인: 승인 순서, v2 계약의 소스 SHA, 커밋·승인된 A1과 그것이 고정한 도구 SHA, 커밋된 prepare PASS 기록, 6개 실행 기록의 짝맞춤
     - 커밋된 `fit.json`이 실행 기록에서 다시 계산한 판정과 같은지 확인
     - 각 checkpoint와 metadata의 SHA, 동결 state
     - H-dev 검사값이 R2-0B 기록과 같은지 확인
     - 블록이 확인: 이전 시도의 예측·예측 기록·`.partial`이 없을 것
   - **저장 순서**
     1. 예측을 `predictions.pt.partial`에 쓴다.
     2. 예측 기록 `evaluate-predicted.json`을 쓴다(예측 SHA, checkpoint 식별값, H-dev 검사값, git 상태).
     3. `.partial`을 `predictions.pt`로 연결한다.
     4. `d1_eval.json`을 쓴다.
   - **실패해도 기록**
     - 종료 코드를 보존한다.
     - 이번 시도의 로그, (있으면) 예측 기록 사본, (쓰였으면) `d1_eval.json`을 커밋한다.
     - 예측 파일은 `runs/`에 남고 커밋하지 않는다.
   - **2–4 사이에서 멈춘 경우**: 다시 실행하면 거부된다. 3-R로 완성한다.

```bash
(
  set -euo pipefail
  cd ~/renju-h6
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src:$PWD/scripts"
  [ "$(git branch --show-current)" = feat/h7-c1 ] || { echo "STOP: wrong branch"; exit 1; }
  git pull --ff-only origin feat/h7-c1
  D=docs/h7r/r2/R2-D1-HEAD-WIDTH-1
  X="$D/1k-v2"
  R=runs/h7r/R2-D1-HEAD-WIDTH-1/1k-v2
  L="$R/logs"
  test -z "$(git status --porcelain --untracked-files=no)"
  python scripts/check_track_b_frozen.py
  git ls-files --error-unmatch "$X/prepare.json" "$X/fit.json" "$D/d1-1k-v2-evaluate-a1.json" >/dev/null
  test ! -e "$X/d1_eval.json"
  test ! -e "$R/predictions.pt"
  test ! -e "$R/predictions.pt.partial"
  test ! -e "$R/evaluate-predicted.json"
  for s in 31001 31002 31003; do for arm in narrow wide; do test -f "$R/$arm-$s/best.pt"; test -f "$R/$arm-$s/best.json"; done; done
  mkdir -p "$L"
  LOG="$L/evaluate-$(date +%Y%m%d-%H%M%S).log"
  B="$(basename "$LOG" .log)"
  ts(){ while IFS= read -r line; do printf '%s %s\n' "$(date +%s.%N)" "$line"; done; }
  run_rc=0
  { echo __START__; rc=0; python -u scripts/h7r_d1_evaluate.py --d1-contract "$D/d1-1k-v2-contract.json" \
      --cache runs/h7r/R1-HDEV-1/cache --c5 runs/h7c/c5/value/best.pt --device cuda 2>&1 || rc=$?
    echo "__EXIT__ $rc"; exit "$rc"; } | ts | tee "$LOG" || run_rc=$?
  python scripts/h7r_d4_records.py copy-logs --src-dir "$L" --pattern "$B.log" \
    --dest-dir "$X/evaluate-logs" --manifest "$X/evaluate-logs/$B.manifest.json"
  if [ -f "$R/evaluate-predicted.json" ]; then
    mkdir -p "$X/evaluate-records"
    cp --no-clobber "$R/evaluate-predicted.json" "$X/evaluate-records/evaluate-predicted.json"
    cmp "$R/evaluate-predicted.json" "$X/evaluate-records/evaluate-predicted.json"
  fi
  expect=(--expect "$X/evaluate-logs/$B.log" --expect "$X/evaluate-logs/$B.manifest.json")
  for f in "$X/evaluate-records/evaluate-predicted.json" "$X/d1_eval.json"; do
    if [ -f "$f" ] && ! git ls-files --error-unmatch "$f" >/dev/null 2>&1; then expect+=(--expect "$f"); fi
  done
  for e in "${expect[@]}"; do [ "$e" = --expect ] || git add "$e"; done
  python scripts/h7r_d4_records.py verify-staged "${expect[@]}"
  if [ "$run_rc" -eq 0 ]; then verdict=reading; else verdict="FAIL (exit $run_rc)"; fi
  git commit -m "H7-R D1 v2: 1,000-row H-dev $verdict (records only)"
  git push origin feat/h7-c1
  echo "__RECORDED__ $run_rc"
  exit "$run_rc"
)
```

3-R. 중단된 평가의 완성 (3번이 예측 기록 뒤에 멈췄을 때만; 새 추론 없음)
   - **언제**: `evaluate-predicted.json`이 있고 `d1_eval.json`이 없을 때
   - **검증**: 도구가 기록된 예측을 검증한다.
     - 예측 파일 SHA, 모델 이름 7개, 행 수, 유한값
     - H-dev 행·검사값
     - 6개 checkpoint와 c5의 식별값(파일 SHA, 동결 state)을 다시 계산해 기록과 대조
   - **완성**: 통과하면 저장된 예측으로 판독을 마친다. 결과에는 `resumed`(예측 기록 SHA, 예측 시점과 완성 시점의 git 상태)가 남는다.
   - **기록 없는 `.partial`만 있는 경우**: 1–2 사이에서 멈춘 것이다. 도구가 쓰지 않고 멈춘다. 지우지 말고 기록을 보고한 뒤 다음 처리를 따로 정한다.

```bash
(
  set -euo pipefail
  cd ~/renju-h6
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src:$PWD/scripts"
  [ "$(git branch --show-current)" = feat/h7-c1 ] || { echo "STOP: wrong branch"; exit 1; }
  D=docs/h7r/r2/R2-D1-HEAD-WIDTH-1
  X="$D/1k-v2"
  R=runs/h7r/R2-D1-HEAD-WIDTH-1/1k-v2
  L="$R/logs"
  test -z "$(git status --porcelain --untracked-files=no)"
  python scripts/check_track_b_frozen.py
  git ls-files --error-unmatch "$X/prepare.json" "$X/fit.json" "$D/d1-1k-v2-evaluate-a1.json" >/dev/null
  test ! -e "$X/d1_eval.json"
  test -f "$R/evaluate-predicted.json"
  for s in 31001 31002 31003; do for arm in narrow wide; do test -f "$R/$arm-$s/best.pt"; test -f "$R/$arm-$s/best.json"; done; done
  mkdir -p "$L"
  LOG="$L/evaluate-resume-$(date +%Y%m%d-%H%M%S).log"
  B="$(basename "$LOG" .log)"
  ts(){ while IFS= read -r line; do printf '%s %s\n' "$(date +%s.%N)" "$line"; done; }
  run_rc=0
  { echo __START__; rc=0; python -u scripts/h7r_d1_evaluate.py --d1-contract "$D/d1-1k-v2-contract.json" \
      --cache runs/h7r/R1-HDEV-1/cache --c5 runs/h7c/c5/value/best.pt --device cuda --resume 2>&1 || rc=$?
    echo "__EXIT__ $rc"; exit "$rc"; } | ts | tee "$LOG" || run_rc=$?
  python scripts/h7r_d4_records.py copy-logs --src-dir "$L" --pattern "$B.log" \
    --dest-dir "$X/evaluate-logs" --manifest "$X/evaluate-logs/$B.manifest.json"
  if [ -f "$R/evaluate-predicted.json" ]; then
    mkdir -p "$X/evaluate-records"
    cp --no-clobber "$R/evaluate-predicted.json" "$X/evaluate-records/evaluate-predicted.json"
    cmp "$R/evaluate-predicted.json" "$X/evaluate-records/evaluate-predicted.json"
  fi
  expect=(--expect "$X/evaluate-logs/$B.log" --expect "$X/evaluate-logs/$B.manifest.json")
  for f in "$X/evaluate-records/evaluate-predicted.json" "$X/d1_eval.json"; do
    if [ -f "$f" ] && ! git ls-files --error-unmatch "$f" >/dev/null 2>&1; then expect+=(--expect "$f"); fi
  done
  for e in "${expect[@]}"; do [ "$e" = --expect ] || git add "$e"; done
  python scripts/h7r_d4_records.py verify-staged "${expect[@]}"
  if [ "$run_rc" -eq 0 ]; then verdict=reading; else verdict="FAIL (exit $run_rc)"; fi
  git commit -m "H7-R D1 v2: 1,000-row H-dev resumed $verdict (records only)"
  git push origin feat/h7-c1
  echo "__RECORDED__ $run_rc"
  exit "$run_rc"
)
```

3-F. 평가 기록 보존 (3번 또는 3-R 블록이 기록 커밋 전에 멈췄을 때만; 추론 없음)
   - 가장 최근 evaluate 로그(3-R의 `evaluate-resume-*` 포함)와, 있으면 아직 커밋되지 않은 예측 기록 사본과 `d1_eval.json`을 커밋한다.
   - 이미 복사된 로그는 원본과 대조만 한다(`--resume`).
   - 중단된 commit이 남긴 staged 상태는 기록 폴더(`$X`)에 한해서만 먼저 해제한다. commit 뒤 push만 실패했으면 push를 다시 시도한다.

```bash
(
  set -euo pipefail
  cd ~/renju-h6
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src:$PWD/scripts"
  [ "$(git branch --show-current)" = feat/h7-c1 ] || { echo "STOP: wrong branch"; exit 1; }
  D=docs/h7r/r2/R2-D1-HEAD-WIDTH-1
  X="$D/1k-v2"
  L=runs/h7r/R2-D1-HEAD-WIDTH-1/1k-v2/logs
  echo __START__
  git reset -q -- "$X"
  test -z "$(git status --porcelain --untracked-files=no)"
  shopt -s nullglob
  logs=("$L"/evaluate-*.log)
  [ "${#logs[@]}" -gt 0 ] || { echo "STOP: no evaluate log"; exit 1; }
  LOG="${logs[0]}"
  for f in "${logs[@]}"; do if [ "$f" -nt "$LOG" ]; then LOG="$f"; fi; done
  B="$(basename "$LOG" .log)"
  echo "log: $LOG"
  tail -n 5 "$LOG"
  expect=()
  python scripts/h7r_d4_records.py copy-logs --resume --src-dir "$L" --pattern "$B.log" \
    --dest-dir "$X/evaluate-logs" --manifest "$X/evaluate-logs/$B.manifest.json"
  R=runs/h7r/R2-D1-HEAD-WIDTH-1/1k-v2
  if [ -f "$R/evaluate-predicted.json" ]; then
    mkdir -p "$X/evaluate-records"
    cp --no-clobber "$R/evaluate-predicted.json" "$X/evaluate-records/evaluate-predicted.json"
    cmp "$R/evaluate-predicted.json" "$X/evaluate-records/evaluate-predicted.json"
  fi
  for f in "$X/evaluate-logs/$B.log" "$X/evaluate-logs/$B.manifest.json" "$X/evaluate-records/evaluate-predicted.json" "$X/d1_eval.json"; do
    if [ -f "$f" ] && ! git ls-files --error-unmatch "$f" >/dev/null 2>&1; then
      git add "$f"
      expect+=(--expect "$f")
    fi
  done
  if [ "${#expect[@]}" -eq 0 ]; then
    git push origin HEAD:feat/h7-c1
    echo "nothing left to record; push checked"; echo "__EXIT__ 0"; exit 0
  fi
  python scripts/h7r_d4_records.py verify-staged "${expect[@]}"
  git commit -m "H7-R D1 v2: 1,000-row H-dev attempt record $B (records only)"
  git push origin feat/h7-c1
  echo "__EXIT__ 0"
)
```

- 2번(학습 6회, train 적합 판독, 기록 보존)까지가 train 승인 범위였다. 3번·3-R·3-F는 evaluate 승인(§12.5)과 평가 개정 A1 승인(§12.6) 범위다.
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

### 11.3 train 기록 복구 보완 (2026-11-07)

- 로그 복사가 중간에 끊기면 일부 사본만 남고 manifest는 없을 수 있다. 기록 도구의 `copy-logs --resume`은 원본과 SHA가 같은 사본을 재사용하고 빠진 사본을 복사한다. 기존 manifest가 있으면 현재 원본·사본과 다시 대조한다.
- `manifest --resume`은 기존 실행 기록 manifest를 현재 파일 목록·크기·SHA와 대조한다. 기존 manifest와 다른 내용이나 모델·tensor 파일이 있으면 거부한다. 기존 명령의 기본 동작은 출력이 있으면 거부하는 방식이다.
- 1-F와 2-F는 commit 뒤 push만 실패한 경우에도 push를 재시도한다. 부분 복사·staging·commit 단계에서 끊긴 경우는 기존 파일을 대조한 뒤 남은 기록을 커밋한다.
- 모든 블록은 `feat/h7-c1`인지 먼저 확인한다. 2번 블록은 마지막 학습 뒤에도 HEAD를 확인해, HEAD가 바뀌었으면 fit을 시작하지 않는다.
- 2-F는 이번 시도의 로그·manifest와 원본에서 대조한 실행 JSON, fit.json만 스테이징한다. 이미 있는 manifest도 다시 검증한다.
- 검증은 `tests.test_h7r_d1_records`에서 원자적 기록 복사 중단, 사본·manifest 변경, 모델 혼입, 실제 셸의 6회 성공·4번째 실행 실패·fit 직전 HEAD 변경, commit 후 push 실패와 복구를 재현한다. 학습·checker는 대역 도구이며 git·기록 도구·문서의 셸 블록은 실제로 실행한다.

## 12. 1,000행 학습 결과 (v2, 기록 `71cbd60`)

### 12.1 판독: `WIDTH_EFFECT` (1,024행 train 적합 개선에 한정)

- **실행**
  - 같은 20게임·1,024행 부분집합, seed별 같은 자료 배정, 두 군 모두 210 step
  - seed마다 두 군의 자료 배정 지문, 부분집합, step 수, 평가 step이 같다.
  - 6개 실행의 시작 W가 모두 0.37418이다. 확장 parent의 예측 보존이 학습 시작점에서도 유지됐다.
- **판정**: 마지막 step(210) state의 train W(학습과 같은 가중치, augmentation 없음; 낮을수록 좋음)로 판정했다.

| seed | narrow | wide | wide − narrow | 감소율 |
|---|---:|---:|---:|---:|
| 31001 | 0.34283 | 0.28374 | −0.05908 | 17.2% |
| 31002 | 0.34196 | 0.27911 | −0.06284 | 18.4% |
| 31003 | 0.34391 | 0.27996 | −0.06396 | 18.6% |

- 세 seed 모두 차이가 음수이고, 각 크기가 narrow의 seed 범위 0.001956보다 크다. 그래서 사전 규칙상 `WIDTH_EFFECT`다.
- **이 판정의 범위**
  - 부분집합의 각 행은 105 epoch 동안 D4 변환과 함께 반복 학습된다. 따라서 이 결과는 고정된 1,024행을 얼마나 맞추는지(train 적합)를 보여 준다.
  - 일반화나 기력의 증거가 아니다.
  - 결론은 "같은 자료와 같은 210 step에서, 넓은 head가 이 부분집합의 train 손실을 더 낮췄다"까지다.

### 12.2 절대 적합 문턱: 두 head 모두 미달

- 사전 문턱 W ≤ F_D4 + 0.01 = 0.011118. 6개 실행 모두 도달하지 못했다(wide도 약 0.28).
- 줄일 수 있는 오차(시작 W − F_D4) 가운데 줄인 비율: narrow 8.1–8.6%, wide 24.2–25.5%
- **문턱 설계의 한계**: 이 부분집합에는 같은 국면 중복이 거의 없다(D4 충돌 묶음 1개). 그래서 이 문턱은 사실상 거의 외우기 수준이다. 동결 trunk와 210 step에서 도달 가능한지 미리 추정하지 않았다. 규칙은 바꾸지 않는다. 이 예산에서 적합 성공 여부는 정보가 적었다고 기록한다.
- **용량 부족을 단독 원인으로 정하지 않는다.** 아래가 함께 남아 있다.
  - 학습 목적함수가 replay와 50:50으로 섞여 있다.
  - 최적화: 학습률 2e-4, cosine 감쇠로 끝에서 5%
  - 210 step 예산

### 12.3 성분, 학습 곡선, 비용 (측정 조건·seed별)

- **성분별 손실** (마지막 step, 성분별 가중치 없는 MSE)
  - narrow: outcome 0.574–0.578, proof 0.083–0.086
  - wide: outcome 0.481–0.491, proof 0.054–0.058
  - 이 부분집합에서는 두 성분이 함께 낮아졌다. 다만 proof는 85행뿐이다.
  - D3와 자료 규모·조건이 다르므로, 이것으로 D3 절충의 원인을 정하지 않는다.
- **학습 곡선** (부분집합 W, 평가 step 21–210)
  - narrow 31001: step 126 이후 거의 평평하다(0.3425 → 0.3428).
  - narrow 31002, 31003: 이후에도 내려간다(31002 0.3459 → 0.3420, 31003 0.3458 → 0.3439).
  - wide 세 seed: 210까지 내려가지만, 마지막 21 step의 감소는 0.0001–0.0013이다(31001 0.2838 → 0.2837, 31002 0.2799 → 0.2791, 31003 0.2813 → 0.2800).
  - 학습률이 끝으로 갈수록 줄어들기 때문에, 감소가 줄어드는 원인을 용량과 학습률 중 어느 쪽으로도 정할 수 없다.
- **T-val 선택 checkpoint** (H-dev 평가 대상)
  - 선택 step: narrow 21 / 105 / 105, wide 147 / 63 / 168
  - objective: narrow 0.45179 / 0.45007 / 0.45121, wide 0.44842 / 0.44391 / 0.44867(parent 0.45122)
  - wide가 세 seed 모두 낮다. 다만 10번의 평가 가운데 최선을 고른 값이라 낙관적으로 치우친다.
  - `CANDIDATE`는 H6 guard를 통과한 최선 평가점이라는 뜻이고, parent보다 나아졌다는 뜻이 아니다. narrow 31001의 선택 objective는 parent보다 높다.
  - 선택된 부분집합 W: narrow 0.3574 / 0.3484 / 0.3487, wide 0.2883 / 0.3092 / 0.2846
- **실행 비용** (RTX 5070, batch 1024, 평가 10회 포함, torch 2.14 / CUDA 13.0)
  - 실행당 237–240초, step당 1.13–1.14초. 6회 합계 1,429초(23분 49초)
  - 이 장비와 이 batch·평가 조건에서 두 head의 실행 시간은 비슷했다. 다른 조건의 비용은 측정하지 않았다.
- **평가로 확인할 질문** (추정하지 않는다)
  - T-val objective가 c5와 비슷한 narrow가 H-dev에서도 c5와 비슷한가
  - objective가 비슷해도 예측과 H-dev 손실은 다를 수 있으므로, c5·narrow·wide를 함께 평가해 확인한다(`narrow_vs_c5`, `wide_vs_c5` 참고값).

### 12.4 5,000행 step 예산: 총 210 step (A안)

- **결정 시점**: 1,000행 train 결과(`71cbd60`)를 확인한 뒤, 1,000행 H-dev 평가 전에 내렸다. H-dev 결과는 이 결정에 쓰지 않았다.
- **내용**
  - 이미 고정된 5,174행 부분집합(111게임, 게임 목록 SHA `5a3a05ea…`, prepare 기록에 있음)
  - 두 군 모두 총 210 step, epoch당 ceil(2 × 5,174 / 1,024) = 11 step으로 약 19 epoch
- **질문**: 같은 계산 예산에서 폭의 이점이 유지되는가
- **해석의 한계**: 1,000행(행당 약 105회 노출)과 5,000행(약 19회 노출)은 행당 노출량도 다르다. 두 규모의 차이를 자료량 하나의 효과로 단정하지 않는다.
- **후속 선택지**: 같은 epoch 수(105 epoch, 1,155 step, 약 2.2시간 예상)를 쓰는 실험은 별도 진단으로 남긴다.
- 5,000행 단계의 계약(부분집합 prepare, 바닥값, 판정 규칙, 출력 경로)은 별도로 작성하고 승인받는다.

### 12.5 다음: 1,000행 H-dev 평가

- v2 계약은 `approved.evaluate`만 `true`로 바꿨다. 실험 식별값(`fc309887e4d7`)과 소스 검사는 그대로다.
- 실행은 §10.2의 3번 블록이다. 평가 도구는 §12.6의 개정 A1로 바뀌었다. 예측 기록 뒤에 멈추면 3-R, 기록 커밋 전에 멈추면 3-F를 쓴다.
- 결과가 어떻게 나와도 이 단계의 모델은 D4 후보가 아니다. 기력 향상이나 강화학습 재개의 근거로 쓰지 않는다.

### 12.6 평가 개정 A1: 중단 뒤 완성할 수 있는 평가 도구

- **문제** (코드 검토에서 재현)
  - `scripts/h7r_d1.py evaluate`는 `predictions.pt`를 먼저 쓰고 `d1_eval.json`을 나중에 쓴다.
  - 그 사이에 멈추면 예측 파일만 남는다. 그러면 재실행은 "출력이 이미 있다"며 거부되고, 그 예측을 완성할 방법도 없다.
- **이 파일을 고치지 않는 이유**
  - 이 파일은 v2 계약이 SHA로 고정한 소스다.
  - 고치면 계약의 소스 검사와 실험 식별값이 바뀐다. 그러면 이미 커밋된 prepare·학습·fit 기록과의 연결이 끊긴다.
- **개정 내용** (`R2-D1-HEAD-WIDTH-1/d1-1k-v2-evaluate-a1.json`)
  - 새 평가 도구 `scripts/h7r_d1_evaluate.py`를 SHA로 고정한다. v2 실험 식별값과 예측 기록 경로도 함께 적고, 승인 여부를 둔다.
  - 도구는 v2 계약의 검사를 `scripts/h7r_d1.py`에서 그대로 불러 쓴다: `load_d1`의 evaluate 승인과 소스 검사, prepare PASS, 실행 기록 짝맞춤, `fit.json` 재계산
  - 승인되지 않았거나, 고정한 도구 SHA가 checkout과 다르면 도구는 실행하지 않는다.
- **바꾸지 않는 것**
  - v2 계약, 실험 식별값, 고정 소스, 커밋된 기록
  - 평가 대상(c5와 T-val 선택 checkpoint 6개), H-dev 입력과 검사, R2 §7 판독, 참고값, 결과·예측 경로
- **바뀌는 것**
  - 저장 순서: 예측 `.partial` → 예측 기록 → 예측 파일 연결 → 결과(§10.2의 3번)
  - `--resume`(3-R)
    - 기록된 예측과 checkpoint 식별값을 다시 검증한 뒤, 새 추론 없이 결과를 완성한다.
    - 기록 없는 `.partial`은 쓰지 않는다. 임의로 지우고 다시 평가하지 않는다.
- **검증** (`tests/test_h7r_d1_evaluate.py`, 대역 모델·대역 H-dev; 실제 GPU·H-dev는 데스크톱 실행에서만 확인된다)
  - 새 도구의 판독(H-dev 판정, seed별 짝비교, 참고값, train 판정)이 고정된 `scripts/h7r_d1.py evaluate`와 같다. 이 테스트가 고정된 evaluate도 처음부터 끝까지 실행한다.
  - 결과 저장 중단: 예측과 기록만 남는다. 일반 재실행은 거부되고, `--resume`은 추론 없이 완성하며 판독은 중단 없는 실행과 같다.
  - 연결 중단(`.partial`만 있고 기록은 있음): `--resume`이 완성한다.
  - 기록 없는 `.partial`: 일반 실행과 `--resume` 모두 거부하고, 파일은 그대로 남긴다.
  - 예측 파일 변조, checkpoint 파일 SHA 변경, checkpoint 식별값 변경: `--resume`이 거부한다.
  - 개정 기록: 미승인, 도구 SHA 불일치, 다른 실험, 예측 기록이 `runs/` 밖, 키 불일치, 미커밋을 모두 거부한다.
  - 커밋된 개정이 이 도구와 v2 실험을 고정하고, v2 계약의 고정 소스는 바뀌지 않았다.

### 12.7 1,000행 H-dev 결과: `DEVELOPMENT_SIGNAL` (기록 `3795e26`)

- **실행**
  - 개정 A1 승인 커밋 `68eea7c`, 깨끗한 상태, 새 평가 도구(`scripts/h7r_d1_evaluate.py`). 중단 없이 실행됐다(`resumed: null`). 0.37분
  - 승인 전 실행 시도(`b8180dd`)는 개정 미승인으로 예측 전에 멈췄다. 로그만 FAIL로 따로 남아 있다.
- **출처**
  - v2 계약 SHA `5c534f12…`, 실험 식별값 `fc309887e4d7`
  - 개정 A1 `dc95853b…`, 도구 `fa864d5c…`
  - 예측 파일 SHA `0f0548d2…`가 예측 기록과 같다.
  - 모델 7개의 식별값이 c5 계약과 각 실행의 선택 checkpoint SHA와 같다. 동결 state도 모두 같다.
  - H-dev는 15,985행·396게임이고, 검사값 6개가 R2-0B 기록과 같다.
- **판독** (R2 §7 그대로; 주 지표는 D4 예측, network 행, 게임 macro 오차이며 낮을수록 좋다)

| seed | narrow (control) | wide | 개선 I_s | wide − narrow 95% 구간 |
|---|---:|---:|---:|---|
| 31001 | 0.405559 | 0.396009 | 0.009550 | [−0.013477, −0.005520] |
| 31002 | 0.404800 | 0.396720 | 0.008079 | [−0.010648, −0.005386] |
| 31003 | 0.406113 | 0.395622 | 0.010492 | [−0.013075, −0.007911] |

  - 세 seed 모두 I_s > ε(0.0019461)이고, 95% 상한이 0보다 낮다.
  - control seed 범위는 0.001314로 ε보다 작다.
  - 따라서 `DEVELOPMENT_SIGNAL`이다. 기록값으로 다시 계산해도 같다.
- **c5 대비** (참고값, 판정에 쓰지 않음)
  - wide − c5: −0.008034 [−0.013573, −0.002424], −0.007323 [−0.011000, −0.003583], −0.008422 [−0.012547, −0.004330]. 세 seed 모두 구간이 0보다 낮다.
  - narrow − c5: 점추정만 있다(구간은 기록하지 않았다). +0.001516, +0.000756, +0.002070으로, 주 지표 점추정이 c5보다 0.00076–0.00207 높다.
  - 이 점추정만으로 narrow와 c5가 같다거나 다르다고 판단하지 않는다.
  - wide는 c5와 직접 비교한 구간도 모두 0보다 낮다. 따라서 이번 이득이 control의 악화만으로 설명되지는 않는다.
- **성분과 분해** (wide − narrow; 참고값, ε 없이)

| seed | outcome macro | proof macro | pooled | outcome 기여 [95%] | proof 기여 [95%] |
|---|---:|---:|---:|---|---|
| 31001 | −0.032372 | +0.005367 | −0.009317 | −0.01279 [−0.01505, −0.01053] | +0.00324 [+0.00022, +0.00638] |
| 31002 | −0.009753 | −0.009062 | −0.005573 | −0.00258 [−0.00392, −0.00120] | −0.00550 [−0.00765, −0.00320] |
| 31003 | +0.000001 | −0.014770 | −0.009950 | −0.00121 [−0.00311, +0.00070] | −0.00928 [−0.01103, −0.00752] |

  - 두 기여의 합은 주 지표 차이와 정확히 같다(가산 분해).
  - seed마다 개선의 구성이 다르다.
    - 31001: outcome이 개선을 이끌고, proof는 악화했다.
    - 31002: 두 성분이 함께 개선됐다.
    - 31003: proof가 개선을 이끌고, outcome 기여의 구간은 0을 포함한다.
  - 세 seed에 공통인 것은 주 지표의 순개선뿐이다. 어느 성분이 이득을 이끄는지는 정하지 않는다.
- **train 판독과 함께 읽기**
  - train: 1,024행 적합 개선(`WIDTH_EFFECT`, 두 head 모두 절대 문턱 미달)
  - H-dev: 학습에 쓰지 않은 개발용 자료에서 wide의 예측 오차가 같은 seed의 narrow보다 낮았다.
  - 서로의 판정을 바꾸지 않는다(§5).
- **범위**
  - 1,000행 진단의 개발용 판독이다.
  - 이 모델들은 D4 후보가 아니다. 기력, 승급, c5 변경, V9의 근거로 쓰지 않는다.
  - 확인용 평가(H-conf)가 아니다.
  - 선택 checkpoint는 T-val로 골랐고, 그 선택에 H-dev는 쓰이지 않았다.

## 13. 5,000행 단계 계약 (`R2-D1-HEAD-WIDTH-1/d1-5k-contract.json`, 승인 전; 초안 `d1-5k-contract.draft.json`)

### 13.1 질문과 고정값

- **질문**: 같은 계산 예산(총 210 step)에서 폭의 이점이 5,174행에서도 유지되는가
- **진입 근거**: §7.2. 1,000행 train 판독 `WIDTH_EFFECT`가 이 단계를 연다. H-dev 결과(§12.7)는 이 결정에 쓰지 않는다.
- **1,000행과 같은 것**
  - 두 군(`value_channels` 1 / 4), seed 31001–31003
  - 확장 방식과 초기화 seed, 예측 보존 판정(바이트, CPU float64 ≤ 1e-12, 출력 유한성, 재로드)
  - c8 config, 적합 판정 규칙(마지막 step의 `W ≤ F_D4 + 0.01`, `fit_reading`의 순서), H-dev 판독(R2 §7, ε)
- **부분집합**
  - 1,000행 선택과 같은 seed·순서의 중첩 5,000행 집합: 111게임, 5,174행, 게임 목록 SHA `5a3a05ea…`
  - 이 값은 1,000행 prepare 기록에 이미 고정돼 있다.
- **예산**: 두 군 총 210 step, epoch당 11 step, 약 19 epoch
  - 1,000행 train 결과(`71cbd60`) 뒤, 그 H-dev 평가(`3795e26`) 전에 정했다(§12.4).
  - 같은 epoch 수(1,155 step)는 후속 선택지로 둔다.
- **해석 조건**
  - 1,000행은 행당 약 105회, 5,000행은 약 19회 노출된다. 두 규모의 차이를 자료량 하나의 효과로 읽지 않는다.
  - 절대 적합 문턱은 바꾸지 않는다. 줄인 비율은 참고로 기록한다.

### 13.2 1,000행 기록과의 연결 (prepare가 검사)

- **선행 기록 고정**: 계약은 다음을 SHA로 고정한다(`predecessor`).
  - 1,000행 v2 계약, 실험 식별값 `fc309887e4d7`
  - prepare / fit / H-dev 결과
  - 평가 개정 A1
- **부분집합**: 다시 선택한 5,000행 집합의 게임 수, 행 수, 게임 목록 SHA가 1,000행 prepare 기록의 `subset.nested`와 같아야 한다. 다르면 멈춘다.
- **확장 parent**: 같은 c5, 같은 방식, 같은 초기화 seed로 다시 만든다. seed별 state SHA가 1,000행 prepare 기록의 `state_sha256`과 같아야 한다.
- **새로 기록하는 것**
  - 5,174행의 구성(outcome / proof 행과 가중치, ply 구간)과 바닥값(정확 / D4)
  - 새 부분집합에서의 float64·GPU 보존 검사와 층별 진단
- **자료 배정**: train에서 seed마다 두 군의 배정 지문(`data_assignment`, n = 5,174)을 기록한다. `fit`이 두 군의 배정, step 수, 부분집합이 같은지 다시 대조한다.

### 13.3 단계와 판정

- 단계는 prepare → train(6회, seed마다 narrow → wide) → fit → evaluate이다. 각각 따로 승인받는다.
- 단계는 승인과 무관한 실험 식별값으로 잇는다.
- 판정 규칙은 1,000행과 같다.
  - train 판독은 마지막 210 step state로 하고, T-val 선택 checkpoint는 따로 기록한다.
  - H-dev는 선택 checkpoint 6개와 c5를 함께 평가한다.
- evaluate는 처음부터 개정 A1의 저장 순서(예측 `.partial` → 예측 기록 → 연결 → 결과)와 `--resume`을 쓴다.
- **다음 단계 조건** (§7.3): 세 seed 모두 train 폭 효과가 유지될 때만 전체 규모를 검토한다. D4는 전체 규모의 별도 계약으로만 검토한다.

### 13.4 도구 변경 범위

**현재 도구가 5,000행을 실행할 수 없는 이유**
- `scripts/h7r_d1.py`
  - `load_d1`이 다음을 요구한다: 형식 `h7r-d1-contract-v2`, stage `1k-v2`, 부분집합 1,000행과 중첩 5,000행, v1 계약과 같은 부분집합·예산·두 군·seed·적합 규칙, `supersedes`
  - prepare 기록 검사(`checked_prepare`)는 `h7r-d1-prepare-v2` 형식만 받는다.
  - 각 단계 함수(prepare / train / fit / evaluate)가 내부에서 `load_d1`을 직접 부른다.
- `scripts/h7r_d1_evaluate.py`
  - 개정 기록 경로(`AMENDMENT`)가 1,000행 A1로 고정돼 있다.
  - A1은 1,000행 계약과 그 실험 식별값만 받는다. 결과의 범위 문구도 1,000행이다.
- 두 파일 모두 SHA로 고정돼 있다.
  - `scripts/h7r_d1.py`는 v2 계약의 소스이고, 실험 식별값에 들어간다.
  - `scripts/h7r_d1_evaluate.py`는 A1이 고정한다.
  - 고치면 1,000행 계약·기록을 다시 검사할 수 없다.

**방침: 기존 파일은 바꾸지 않고 새 진입점을 추가한다**
- 새 파일 `scripts/h7r_d1_5k.py` 하나: 로더 `load_d1_5k`와 prepare / train / fit / evaluate[`--resume`]
- **그대로 불러 쓰는 것**
  - `hybrid.h7r_d1`의 순수 함수: 확장, 보존 검사, float64 값, 비교, 층별 진단, 부분집합 선택, 바닥값, `fit_probe`, 두 판독
  - `scripts/h7r_d1.py`의 단계 무관 함수: `git_state`, `experiment_sha256`, `pinned`, `reference_inputs`, `checked_caches`, `composition`, `load_c5`, `run_records`, `checked_fit`
  - `scripts/h7r_d1_evaluate.py`의 `analyse`, `load_models`
- **새로 쓰는 것**
  - 5,000행 계약 로더
  - 중첩 부분집합 선택과 1,000행 기록 대조
  - 확장 parent의 state SHA 대조
  - prepare / train / fit / evaluate의 단계 함수: 1,000행 단계 함수와 같은 절차에, 경로·형식·대조만 다르다.
- **계약이 고정할 소스**: v2의 고정 소스 13개 + `scripts/h7r_d1_evaluate.py` + `scripts/h7r_d1_5k.py` + 기록 도구(`h7r_ledger`, `h7r_d4`)
  - 도구를 작성해 테스트한 뒤 SHA를 채우고 `status`를 `fixed`로 바꾼다. 지금 초안의 소스 값은 `null`이다.
- **1,000행 재현**: 기존 파일을 바꾸지 않는다. 따라서 1,000행 v2 계약(세 단계)과 A1은 지금처럼 소스 검사를 통과해 다시 읽을 수 있다.

### 13.5 테스트 계획 (`tests/test_h7r_d1_5k.py`)

1. **로더**
   - 고정값(stage, 예산, 두 군, seed, 적합·판독 규칙)과 선행 기록 SHA가 하나라도 다르면 거부한다.
   - 단계 승인 순서를 확인한다.
   - 출력은 `runs/`·`docs/.../5k/` 아래만 허용한다.
   - 소스 SHA를 검사하고, 초안(`status: draft`)이나 `null` 소스는 거부한다.
2. **부분집합**
   - 대역 자료에서 5,000행 선택이 1,000행 도구의 `nested` 기록과 같다.
   - 기록과 다른 집합(seed·순서 변경)은 거부한다.
3. **prepare**
   - 확장 parent의 state SHA가 1,000행 기록과 같아야 한다. 다르면 거부한다.
   - 보존 gate 4개는 v2와 같다: 정상 확장은 통과하고, 깨진 확장과 비정상 출력은 실패로 기록한 뒤 학습을 거부한다.
4. **train·fit**
   - 대역 자료로 6회 실행 후 fit까지 연결한다.
   - 같은 입력·예산이면 1,000행 도구의 `train`과 같은 실행 기록을 만든다(경로·형식 문자열 제외).
   - 두 군의 배정을 대조하고, 마지막 step과 선택 checkpoint를 분리해 기록한다.
5. **evaluate**
   - 같은 예측에서 `scripts/h7r_d1_evaluate.py`의 판독과 같다.
   - 결과 저장·연결 단계의 중단을 `--resume`으로 완성한다.
   - 거부하는 경우: 변조된 예측, 바뀐 checkpoint, 기록 없는 `.partial`
6. **1,000행 회귀**
   - 기존 `tests.test_h7r_d1`, `tests.test_h7r_d1_evaluate`, `tests.test_h7r_d1_records`가 그대로 통과한다.
   - 커밋된 1,000행 v2 계약 세 단계와 A1이 현재 checkout에서 읽힌다.
7. **절차**
   - 문서의 5,000행 실행 블록이 문법 검사를 통과한다.
   - 블록마다 실패 기록과 복구 블록이 있다.
   - 기록 단계는 실제 git과 기록 도구로, 학습·평가는 대역 도구로 실행해 본다.

### 13.6 실행 절차 (데스크톱)
- 블록은 §10.2의 1,000행 블록과 같다. 기계적으로 바꾼 것은 계약·도구·경로·커밋 문구뿐이다.
  - 바꾼 결과에 `1k`, `1,000`, 1,000행 도구 이름이 남지 않았는지 테스트로 확인한다.
  - 기록 단계는 임시 저장소에서 실제 git과 기록 도구로, 도구는 대역으로 실행해 확인했다.
- 각 단계는 승인 커밋을 받은 뒤 `git pull --ff-only`로 같은 HEAD에서 실행한다.
- 실패하면 다음 단계로 넘어가지 않는다. 실패 폴더를 지우거나 블록을 다시 돌리지 않는다. 중단 위치를 보고 따로 정한다.

5k-1. prepare와 기록 커밋 (`approved.prepare` 커밋 후; 학습 없음)
   - 1,000행 1번 블록과 같은 절차다. 도구는 `scripts/h7r_d1_5k.py prepare`이고, 출력은 `5k/`에 쓴다.
   - 도구가 확인하는 것: 다시 고른 5,174행 집합이 1,000행 기록의 중첩 집합과 같은지, 확장 parent의 state SHA가 1,000행 기록과 같은지, 그리고 보존 gate 4개(바이트, float64, 출력 유한성, 재로드)
   - PASS든 FAIL이든 `prepare.json`과 로그를 커밋한다.

```bash
(
  set -euo pipefail
  cd ~/renju-h6
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src:$PWD/scripts"
  [ "$(git branch --show-current)" = feat/h7-c1 ] || { echo "STOP: wrong branch"; exit 1; }
  git pull --ff-only origin feat/h7-c1
  D=docs/h7r/r2/R2-D1-HEAD-WIDTH-1
  X="$D/5k"
  L=runs/h7r/R2-D1-HEAD-WIDTH-1/5k/logs
  test -z "$(git status --porcelain --untracked-files=no)"
  python scripts/check_track_b_frozen.py
  test ! -e "$X/prepare.json"
  mkdir -p "$L"
  LOG="$L/prepare-$(date +%Y%m%d-%H%M%S).log"
  B="$(basename "$LOG" .log)"
  ts(){ while IFS= read -r line; do printf '%s %s\n' "$(date +%s.%N)" "$line"; done; }
  run_rc=0
  { echo __START__; rc=0; python -u scripts/h7r_d1_5k.py prepare --d1-contract "$D/d1-5k-contract.json" \
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
  git commit -m "H7-R D1 5k: 5,000-row prepare $verdict: subset, floors and widened parents (records only)"
  git push origin feat/h7-c1
  echo "__RECORDED__ $run_rc"
  exit "$run_rc"
)
```

5k-1-F. prepare 기록 보존 (5k-1이 기록 커밋 전에 멈췄을 때만)

```bash
(
  set -euo pipefail
  cd ~/renju-h6
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src:$PWD/scripts"
  [ "$(git branch --show-current)" = feat/h7-c1 ] || { echo "STOP: wrong branch"; exit 1; }
  D=docs/h7r/r2/R2-D1-HEAD-WIDTH-1
  X="$D/5k"
  L=runs/h7r/R2-D1-HEAD-WIDTH-1/5k/logs
  echo __START__
  git reset -q -- "$X"
  test -z "$(git status --porcelain --untracked-files=no)"
  shopt -s nullglob
  logs=("$L"/prepare-*.log)
  [ "${#logs[@]}" -gt 0 ] || { echo "STOP: no prepare log"; exit 1; }
  LOG="${logs[0]}"
  for f in "${logs[@]}"; do if [ "$f" -nt "$LOG" ]; then LOG="$f"; fi; done
  B="$(basename "$LOG" .log)"
  echo "log: $LOG"
  tail -n 5 "$LOG"
  expect=()
  python scripts/h7r_d4_records.py copy-logs --resume --src-dir "$L" --pattern "$B.log" \
    --dest-dir "$X/prepare-logs" --manifest "$X/prepare-logs/$B.manifest.json"
  for f in "$X/prepare-logs/$B.log" "$X/prepare-logs/$B.manifest.json" "$X/prepare.json"; do
    if [ -f "$f" ] && ! git ls-files --error-unmatch "$f" >/dev/null 2>&1; then
      git add "$f"
      expect+=(--expect "$f")
    fi
  done
  if [ "${#expect[@]}" -eq 0 ]; then
    git push origin HEAD:feat/h7-c1
    echo "nothing left to record; push checked"; echo "__EXIT__ 0"; exit 0
  fi
  python scripts/h7r_d4_records.py verify-staged "${expect[@]}"
  git commit -m "H7-R D1 5k: 5,000-row prepare attempt record $B (records only)"
  git push origin feat/h7-c1
  echo "__EXIT__ 0"
)
```

5k-2. 학습 6회, train 적합 판독, 기록 커밋 (`approved.train` 커밋 후)
   - seed마다 narrow → wide 순서로 돌린다. 실패하거나 HEAD가 바뀌면 뒤 실행은 시작하지 않는다. fit은 6회가 모두 끝났을 때만 돈다.
   - 실행당 210 step(epoch당 11 step)이다. 예상 약 24분이며, 1,000행 실행 시간 기준의 추정이다.

```bash
(
  set -euo pipefail
  cd ~/renju-h6
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src:$PWD/scripts"
  [ "$(git branch --show-current)" = feat/h7-c1 ] || { echo "STOP: wrong branch"; exit 1; }
  git pull --ff-only origin feat/h7-c1
  D=docs/h7r/r2/R2-D1-HEAD-WIDTH-1
  X="$D/5k"
  R=runs/h7r/R2-D1-HEAD-WIDTH-1/5k
  L="$R/logs"
  test -z "$(git status --porcelain --untracked-files=no)"
  python scripts/check_track_b_frozen.py
  test ! -e "$X/fit.json"
  test ! -e "$X/run-records"
  test ! -e "$X/train-logs"
  for s in 31001 31002 31003; do for arm in narrow wide; do test ! -e "$R/$arm-$s"; done; done
  head="$(git rev-parse HEAD)"
  STAMP="$(date +%Y%m%d-%H%M%S)"
  mkdir -p "$L"
  ts(){ while IFS= read -r line; do printf '%s %s\n' "$(date +%s.%N)" "$line"; done; }
  run_rc=0
  where=none
  done_runs=0
  for s in 31001 31002 31003; do
    for arm in narrow wide; do
      if [ "$(git rev-parse HEAD)" != "$head" ]; then run_rc=97; where="HEAD changed before $arm-$s"; break 2; fi
      LOG="$L/train-$arm-$s-$STAMP.log"
      { echo __START__; rc=0; python -u scripts/h7r_d1_5k.py train --d1-contract "$D/d1-5k-contract.json" \
          --arm "$arm" --seed "$s" \
          --cycle-cache runs/h7c/c8/value_cache --cycle-rows runs/h7c/c8/value_rows.jsonl.gz \
          --replay-cache data/tactical/h6/value_cache_v3 --c5 runs/h7c/c5/value/best.pt --device cuda 2>&1 || rc=$?
        echo "__EXIT__ $rc"; exit "$rc"; } | ts | tee "$LOG" || run_rc=$?
      if [ "$run_rc" -ne 0 ]; then where="train $arm-$s"; break 2; fi
      done_runs=$((done_runs + 1))
    done
  done
  if [ "$run_rc" -eq 0 ] && [ "$(git rev-parse HEAD)" != "$head" ]; then
    run_rc=97; where="HEAD changed before fit"
  fi
  if [ "$run_rc" -eq 0 ]; then
    LOG="$L/fit-$STAMP.log"
    { echo __START__; rc=0; python -u scripts/h7r_d1_5k.py fit --d1-contract "$D/d1-5k-contract.json" 2>&1 || rc=$?
      echo "__EXIT__ $rc"; exit "$rc"; } | ts | tee "$LOG" || run_rc=$?
    if [ "$run_rc" -ne 0 ]; then where=fit; fi
  fi
  echo "[d1-train-block] runs completed $done_runs/6, exit $run_rc, stopped at: $where"
  python scripts/h7r_d4_records.py copy-logs --src-dir "$L" --pattern "*-$STAMP.log" \
    --dest-dir "$X/train-logs" --manifest "$X/train-logs/$STAMP.manifest.json"
  expect=(--expect "$X/train-logs/$STAMP.manifest.json")
  for f in "$X"/train-logs/*-"$STAMP".log; do expect+=(--expect "$f"); done
  copied=0
  for s in 31001 31002 31003; do
    for arm in narrow wide; do
      [ -d "$R/$arm-$s" ] || continue
      for src in "$R/$arm-$s"/*.json; do
        [ -f "$src" ] || continue
        dst="$X/run-records/$arm-$s/$(basename "$src")"
        mkdir -p "$(dirname "$dst")"
        cp --no-clobber "$src" "$dst"
        cmp "$src" "$dst"
        expect+=(--expect "$dst")
        copied=$((copied + 1))
      done
    done
  done
  if [ "$copied" -gt 0 ]; then
    python scripts/h7r_d4_records.py manifest --dir "$X/run-records" --output "$X/run-records/records_manifest-$STAMP.json"
    expect+=(--expect "$X/run-records/records_manifest-$STAMP.json")
  fi
  if [ -f "$X/fit.json" ]; then expect+=(--expect "$X/fit.json"); fi
  for e in "${expect[@]}"; do [ "$e" = --expect ] || git add "$e"; done
  python scripts/h7r_d4_records.py verify-staged "${expect[@]}"
  if [ "$run_rc" -eq 0 ]; then verdict="6/6 runs and the train-fit reading"; else verdict="FAIL at $where (exit $run_rc), $done_runs/6 runs"; fi
  git commit -m "H7-R D1 5k: 5,000-row training $verdict (records only)"
  git push origin feat/h7-c1
  echo "__RECORDED__ $run_rc"
  exit "$run_rc"
)
```

5k-2-F. 학습 기록 보존 (5k-2가 기록 커밋 전에 멈췄을 때만)

```bash
(
  set -euo pipefail
  cd ~/renju-h6
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src:$PWD/scripts"
  [ "$(git branch --show-current)" = feat/h7-c1 ] || { echo "STOP: wrong branch"; exit 1; }
  D=docs/h7r/r2/R2-D1-HEAD-WIDTH-1
  X="$D/5k"
  R=runs/h7r/R2-D1-HEAD-WIDTH-1/5k
  L="$R/logs"
  echo __START__
  git reset -q -- "$X"
  test -z "$(git status --porcelain --untracked-files=no)"
  shopt -s nullglob
  logs=("$L"/train-*.log)
  [ "${#logs[@]}" -gt 0 ] || { echo "STOP: no train log"; exit 1; }
  last="${logs[0]}"
  for f in "${logs[@]}"; do if [ "$f" -nt "$last" ]; then last="$f"; fi; done
  STAMP="$(basename "$last" .log | grep -oE '[0-9]{8}-[0-9]{6}$')"
  echo "attempt $STAMP"
  for f in "$L"/*-"$STAMP".log; do echo "== $f"; tail -n 3 "$f"; done
  python scripts/h7r_d4_records.py copy-logs --resume --src-dir "$L" --pattern "*-$STAMP.log" \
    --dest-dir "$X/train-logs" --manifest "$X/train-logs/$STAMP.manifest.json"
  candidates=("$X/train-logs/$STAMP.manifest.json" "$X"/train-logs/*-"$STAMP".log)
  for s in 31001 31002 31003; do
    for arm in narrow wide; do
      [ -d "$R/$arm-$s" ] || continue
      for src in "$R/$arm-$s"/*.json; do
        [ -f "$src" ] || continue
        dst="$X/run-records/$arm-$s/$(basename "$src")"
        mkdir -p "$(dirname "$dst")"
        cp --no-clobber "$src" "$dst"
        cmp "$src" "$dst"
        candidates+=("$dst")
      done
    done
  done
  if [ -d "$X/run-records" ]; then
    python scripts/h7r_d4_records.py manifest --resume --dir "$X/run-records" --output "$X/run-records/records_manifest-$STAMP.json"
    candidates+=("$X/run-records/records_manifest-$STAMP.json")
  fi
  if [ -f "$X/fit.json" ]; then candidates+=("$X/fit.json"); fi
  expect=()
  for f in "${candidates[@]}"; do
    if ! git ls-files --error-unmatch "$f" >/dev/null 2>&1; then git add "$f"; expect+=(--expect "$f"); fi
  done
  if [ "${#expect[@]}" -eq 0 ]; then
    git push origin HEAD:feat/h7-c1
    echo "nothing left to record; push checked"; echo "__EXIT__ 0"; exit 0
  fi
  python scripts/h7r_d4_records.py verify-staged "${expect[@]}"
  git commit -m "H7-R D1 5k: 5,000-row training attempt record $STAMP (records only)"
  git push origin feat/h7-c1
  echo "__EXIT__ 0"
)
```

5k-3. H-dev 판독과 기록 커밋 (`approved.evaluate` 커밋 후)
   - 평가 대상은 c5와 T-val 선택 checkpoint 6개, 판독은 R2 §7 그대로다.
   - 저장 순서는 A1과 같다: 예측 `.partial` → 예측 기록 → 연결 → 결과

```bash
(
  set -euo pipefail
  cd ~/renju-h6
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src:$PWD/scripts"
  [ "$(git branch --show-current)" = feat/h7-c1 ] || { echo "STOP: wrong branch"; exit 1; }
  git pull --ff-only origin feat/h7-c1
  D=docs/h7r/r2/R2-D1-HEAD-WIDTH-1
  X="$D/5k"
  R=runs/h7r/R2-D1-HEAD-WIDTH-1/5k
  L="$R/logs"
  test -z "$(git status --porcelain --untracked-files=no)"
  python scripts/check_track_b_frozen.py
  git ls-files --error-unmatch "$X/prepare.json" "$X/fit.json" "$D/d1-5k-contract.json" >/dev/null
  test ! -e "$X/d1_eval.json"
  test ! -e "$R/predictions.pt"
  test ! -e "$R/predictions.pt.partial"
  test ! -e "$R/evaluate-predicted.json"
  for s in 31001 31002 31003; do for arm in narrow wide; do test -f "$R/$arm-$s/best.pt"; test -f "$R/$arm-$s/best.json"; done; done
  mkdir -p "$L"
  LOG="$L/evaluate-$(date +%Y%m%d-%H%M%S).log"
  B="$(basename "$LOG" .log)"
  ts(){ while IFS= read -r line; do printf '%s %s\n' "$(date +%s.%N)" "$line"; done; }
  run_rc=0
  { echo __START__; rc=0; python -u scripts/h7r_d1_5k.py evaluate --d1-contract "$D/d1-5k-contract.json" \
      --cache runs/h7r/R1-HDEV-1/cache --c5 runs/h7c/c5/value/best.pt --device cuda 2>&1 || rc=$?
    echo "__EXIT__ $rc"; exit "$rc"; } | ts | tee "$LOG" || run_rc=$?
  python scripts/h7r_d4_records.py copy-logs --src-dir "$L" --pattern "$B.log" \
    --dest-dir "$X/evaluate-logs" --manifest "$X/evaluate-logs/$B.manifest.json"
  if [ -f "$R/evaluate-predicted.json" ]; then
    mkdir -p "$X/evaluate-records"
    cp --no-clobber "$R/evaluate-predicted.json" "$X/evaluate-records/evaluate-predicted.json"
    cmp "$R/evaluate-predicted.json" "$X/evaluate-records/evaluate-predicted.json"
  fi
  expect=(--expect "$X/evaluate-logs/$B.log" --expect "$X/evaluate-logs/$B.manifest.json")
  for f in "$X/evaluate-records/evaluate-predicted.json" "$X/d1_eval.json"; do
    if [ -f "$f" ] && ! git ls-files --error-unmatch "$f" >/dev/null 2>&1; then expect+=(--expect "$f"); fi
  done
  for e in "${expect[@]}"; do [ "$e" = --expect ] || git add "$e"; done
  python scripts/h7r_d4_records.py verify-staged "${expect[@]}"
  if [ "$run_rc" -eq 0 ]; then verdict=reading; else verdict="FAIL (exit $run_rc)"; fi
  git commit -m "H7-R D1 5k: 5,000-row H-dev $verdict (records only)"
  git push origin feat/h7-c1
  echo "__RECORDED__ $run_rc"
  exit "$run_rc"
)
```

5k-3-R. 중단된 평가의 완성 (예측 기록이 있고 결과가 없을 때만; 새 추론 없음)

```bash
(
  set -euo pipefail
  cd ~/renju-h6
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src:$PWD/scripts"
  [ "$(git branch --show-current)" = feat/h7-c1 ] || { echo "STOP: wrong branch"; exit 1; }
  D=docs/h7r/r2/R2-D1-HEAD-WIDTH-1
  X="$D/5k"
  R=runs/h7r/R2-D1-HEAD-WIDTH-1/5k
  L="$R/logs"
  test -z "$(git status --porcelain --untracked-files=no)"
  python scripts/check_track_b_frozen.py
  git ls-files --error-unmatch "$X/prepare.json" "$X/fit.json" "$D/d1-5k-contract.json" >/dev/null
  test ! -e "$X/d1_eval.json"
  test -f "$R/evaluate-predicted.json"
  for s in 31001 31002 31003; do for arm in narrow wide; do test -f "$R/$arm-$s/best.pt"; test -f "$R/$arm-$s/best.json"; done; done
  mkdir -p "$L"
  LOG="$L/evaluate-resume-$(date +%Y%m%d-%H%M%S).log"
  B="$(basename "$LOG" .log)"
  ts(){ while IFS= read -r line; do printf '%s %s\n' "$(date +%s.%N)" "$line"; done; }
  run_rc=0
  { echo __START__; rc=0; python -u scripts/h7r_d1_5k.py evaluate --d1-contract "$D/d1-5k-contract.json" \
      --cache runs/h7r/R1-HDEV-1/cache --c5 runs/h7c/c5/value/best.pt --device cuda --resume 2>&1 || rc=$?
    echo "__EXIT__ $rc"; exit "$rc"; } | ts | tee "$LOG" || run_rc=$?
  python scripts/h7r_d4_records.py copy-logs --src-dir "$L" --pattern "$B.log" \
    --dest-dir "$X/evaluate-logs" --manifest "$X/evaluate-logs/$B.manifest.json"
  if [ -f "$R/evaluate-predicted.json" ]; then
    mkdir -p "$X/evaluate-records"
    cp --no-clobber "$R/evaluate-predicted.json" "$X/evaluate-records/evaluate-predicted.json"
    cmp "$R/evaluate-predicted.json" "$X/evaluate-records/evaluate-predicted.json"
  fi
  expect=(--expect "$X/evaluate-logs/$B.log" --expect "$X/evaluate-logs/$B.manifest.json")
  for f in "$X/evaluate-records/evaluate-predicted.json" "$X/d1_eval.json"; do
    if [ -f "$f" ] && ! git ls-files --error-unmatch "$f" >/dev/null 2>&1; then expect+=(--expect "$f"); fi
  done
  for e in "${expect[@]}"; do [ "$e" = --expect ] || git add "$e"; done
  python scripts/h7r_d4_records.py verify-staged "${expect[@]}"
  if [ "$run_rc" -eq 0 ]; then verdict=reading; else verdict="FAIL (exit $run_rc)"; fi
  git commit -m "H7-R D1 5k: 5,000-row H-dev resumed $verdict (records only)"
  git push origin feat/h7-c1
  echo "__RECORDED__ $run_rc"
  exit "$run_rc"
)
```

5k-3-F. 평가 기록 보존 (5k-3 또는 5k-3-R이 기록 커밋 전에 멈췄을 때만)

```bash
(
  set -euo pipefail
  cd ~/renju-h6
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src:$PWD/scripts"
  [ "$(git branch --show-current)" = feat/h7-c1 ] || { echo "STOP: wrong branch"; exit 1; }
  D=docs/h7r/r2/R2-D1-HEAD-WIDTH-1
  X="$D/5k"
  L=runs/h7r/R2-D1-HEAD-WIDTH-1/5k/logs
  echo __START__
  git reset -q -- "$X"
  test -z "$(git status --porcelain --untracked-files=no)"
  shopt -s nullglob
  logs=("$L"/evaluate-*.log)
  [ "${#logs[@]}" -gt 0 ] || { echo "STOP: no evaluate log"; exit 1; }
  LOG="${logs[0]}"
  for f in "${logs[@]}"; do if [ "$f" -nt "$LOG" ]; then LOG="$f"; fi; done
  B="$(basename "$LOG" .log)"
  echo "log: $LOG"
  tail -n 5 "$LOG"
  expect=()
  python scripts/h7r_d4_records.py copy-logs --resume --src-dir "$L" --pattern "$B.log" \
    --dest-dir "$X/evaluate-logs" --manifest "$X/evaluate-logs/$B.manifest.json"
  R=runs/h7r/R2-D1-HEAD-WIDTH-1/5k
  if [ -f "$R/evaluate-predicted.json" ]; then
    mkdir -p "$X/evaluate-records"
    cp --no-clobber "$R/evaluate-predicted.json" "$X/evaluate-records/evaluate-predicted.json"
    cmp "$R/evaluate-predicted.json" "$X/evaluate-records/evaluate-predicted.json"
  fi
  for f in "$X/evaluate-logs/$B.log" "$X/evaluate-logs/$B.manifest.json" "$X/evaluate-records/evaluate-predicted.json" "$X/d1_eval.json"; do
    if [ -f "$f" ] && ! git ls-files --error-unmatch "$f" >/dev/null 2>&1; then
      git add "$f"
      expect+=(--expect "$f")
    fi
  done
  if [ "${#expect[@]}" -eq 0 ]; then
    git push origin HEAD:feat/h7-c1
    echo "nothing left to record; push checked"; echo "__EXIT__ 0"; exit 0
  fi
  python scripts/h7r_d4_records.py verify-staged "${expect[@]}"
  git commit -m "H7-R D1 5k: 5,000-row H-dev attempt record $B (records only)"
  git push origin feat/h7-c1
  echo "__EXIT__ 0"
)
```

### 13.7 진행 상태와 승인 단계

- **구현 완료** (이 커밋)
  - `scripts/h7r_d1_5k.py`와 `tests/test_h7r_d1_5k.py`
  - 계약 `d1-5k-contract.json`
    - 초안(`d1-5k-contract.draft.json`)의 값을 그대로 두고, 소스 17개의 SHA를 채워 `status: fixed`로 고정했다.
    - 초안은 설계 기록으로 남긴다. 테스트가 고정 계약과 초안이 같은지 확인한다(`status`, `approved`, `sources`, `note`만 다르다).
    - 실험 식별값 `84ed0100d792`, 모든 `approved` false
  - §13.6의 실행 블록 7개
- **테스트로 확인한 것** (대역 입력; 실제 c8·H-dev·GPU에서는 데스크톱 실행으로만 확인된다)
  - 로더: 커밋된 계약은 세 단계 모두 미승인으로 거부한다. 승인한 사본은 세 단계 모두 읽힌다. 고정값·선행 기록·출력 경로가 틀리면 거부한다.
  - 부분집합: 5,000행 선택이 1,000행 선택의 중첩 집합과 같다. 다른 seed는 거부한다.
  - prepare: 같은 입력에서 1,000행 prepare와 기록이 같다(검사값, 진단, 확장 parent의 checkpoint SHA). parent state가 1,000행 기록과 다르면 그 seed가 FAIL이고 학습을 거부한다.
  - train·fit: 같은 입력·예산에서 1,000행 train과 실행 기록이 같다(자료 배정, 적합 곡선, 마지막 step, 선택 checkpoint). fit은 그 기록으로 판독한다.
  - evaluate: A1 도구와 판독이 같다. 결과 저장 중단은 `--resume`이 추론 없이 완성한다. 변조된 예측과 기록 없는 `.partial`은 거부한다.
  - 판독 중 변경 (검토 보완 P2): 결과 저장 직전에 HEAD와 tracked tree를 다시 검사한다.
    - 정상 실행과 `--resume` 모두, 판독 계산 중 HEAD나 dirty 상태가 바뀌면 결과를 쓰지 않는다.
    - 예측과 예측 기록은 그대로 남고, 안정된 상태에서 `--resume`으로 완성된다.
    - 이 수정으로 도구 SHA가 바뀌었다. 그래서 계약의 소스를 다시 고정했고, 실험 식별값이 `84ed0100d792`로 바뀌었다. 아직 5,000행 기록은 없다.
  - 1,000행 회귀: v2 계약 세 단계, A1, prepare 기록이 그대로 읽힌다. 5,000행 계약이 고정한 1,000행 파일의 SHA는 v2 계약의 값과 같다.
- **승인과 실행** (단계마다 승인 커밋 → `git pull --ff-only` → 실행)
  1. 도구·계약 검토
  2. prepare 승인 → 5k-1: 부분집합 대조, 구성·바닥값, parent 대조, 보존 gate
  3. train 승인 → 5k-2: 6회 학습과 fit. 예상 약 24분(추정)
  4. evaluate 승인 → 5k-3: H-dev 판독(예상 1분 안팎)
- 5,000행 결과는 위 판정 규칙으로만 읽는다. 결과를 보고 문턱이나 규칙을 바꾸지 않는다.

### 13.8 5,000행 prepare 결과: PASS (기록 `837ddab`)

- **실행과 기록의 일치**
  - 승인 커밋 `e432194`에서 깨끗한 상태로 실행했다. 약 100초 걸렸다.
  - 계약 SHA `bfa1ca6e…`, 실험 식별값 `84ed0100d792`, 로그 SHA는 manifest와 같다.
- **1,000행 기록과의 연결**
  - 부분집합: 111게임 · 5,174행 · 게임 목록 SHA `5a3a05ea…`. 1,000행 prepare의 `subset.nested`와 같다.
  - 1,000행 prepare와 같은 값: c5 식별값, H6 val 표본 SHA, GPU 연산 설정
  - 확장 parent: 세 seed 모두 state SHA가 1,000행과 같다(`same_parent_as_1k`).
  - checkpoint **파일** SHA는 1,000행과 다르다.
    - 저장 함수(`model.checkpoint.save_checkpoint`)가 저장 시점의 `git_commit`을 파일에 넣기 때문이다. 다른 커밋에서 저장하면 state가 같아도 파일 바이트가 달라진다.
    - 단계 간 초기 모델 비교에는 state SHA를 쓴다. 이 단계 안의 무결성은 이번 prepare가 기록한 파일 SHA로 검사한다(train 진입 검사).
- **보존 gate**: 세 seed 모두 5개 통과(바이트, 1,000행 parent와 동일, CPU float64 최대 6.1e-16, 출력 유한성, 재로드)
- **새 부분집합의 구성값** (학습 성능 변화로 해석하지 않는다)
  - outcome 4,598행(가중치 459.8), proof 576행
  - proof 행 비중: 1,000행 약 8.3% → 5,000행 약 11.1%
  - 바닥값
    - 정확 기준 F = 0.0000483(충돌 묶음 1)
    - D4 기준 F_D4 = 0.0050846367(충돌 묶음 31)
  - 적합 문턱: F_D4 + 0.01 = **0.0150846367**(규칙은 계약 그대로; 1,000행 문턱 0.0111179)
- **GPU float32 차이** (기록만)
  - 부분집합 1.15e-6, cycle val 1.62e-6, H6 val 0
  - 층별 진단은 1,000행과 같다: 차이는 넓어진 첫 Linear의 곱에서 처음 생긴다.
- **다음**: train 승인(`approved.train`). 6회 학습과 fit은 §13.6의 5k-2, 기록 커밋 전에 멈추면 5k-2-F

### 13.9 5,000행 학습 결과: `WIDTH_EFFECT` (기록 `67cf457`)

- **실행**
  - 6회 모두 승인 커밋 `e390f6e`, 깨끗한 상태에서 210 step을 실행했다. fit은 6회 완료 뒤 같은 HEAD에서 돌았다.
  - 6회 모두 `CANDIDATE`이고, H6 guard는 평가 10회 모두 통과했다.
- **기록의 일치**
  - 실행 기록 6개의 실험 식별값(`84ed0100d792`)과 부분집합이 prepare 기록과 같다.
  - trainer 기록, `best.json`, `final.json`의 SHA가 각 실행 기록과 같다.
  - 기록 사본 24개와 로그 7개의 SHA가 manifest와 같다. 커밋된 `.pt`는 없다.
- **짝맞춤**: seed마다 두 군의 자료 배정과 평가 step이 같다. 시작 W는 6개 실행 모두 0.33657이다.
- **판독**: 마지막 step(210) state의 train W로 판정한다(학습과 같은 가중치, augmentation 없음, 낮을수록 좋음).

| seed | narrow | wide | wide − narrow | 감소율 |
|---|---:|---:|---:|---:|
| 31001 | 0.32545 | 0.29327 | −0.03218 | 9.9% |
| 31002 | 0.32387 | 0.28551 | −0.03836 | 11.8% |
| 31003 | 0.32419 | 0.29210 | −0.03209 | 9.9% |

  - 세 seed 모두 차이가 음수이고, 각 크기가 narrow seed 범위 0.00158보다 크다. 그래서 `WIDTH_EFFECT`다. 기록값으로 다시 계산해도 같다.
  - 범위는 5,174행 train 적합 개선이다. 일반화나 기력의 증거가 아니다.
- **절대 적합 문턱**(0.0150846): 두 head 모두 미달이다.
  - 줄일 수 있는 오차(시작 W − F_D4) 가운데 줄인 비율: narrow 3.4–3.8%, wide 13.1–15.4%
  - 용량 부족을 단독 원인으로 정하지 않는다(§12.2와 같은 이유).
- **성분별 손실** (마지막 step, 가중치 없는 성분별 MSE)
  - narrow: outcome 0.578–0.581, proof 0.119–0.124
  - wide: outcome 0.530–0.547, proof 0.085–0.103
  - 두 성분이 함께 낮다.
- **마지막 학습 구간** (step 189 → 210)
  - wide: 거의 평탄해졌다. 31001은 +0.000076으로 조금 늘었고, 31002는 −0.000296, 31003은 −0.000410으로 줄었다.
  - narrow: +0.000117 / −0.000022 / −0.000016
  - step 126 이후 변화: wide 0.2949 → 0.2933, 0.2891 → 0.2855, 0.2980 → 0.2921 / narrow 0.3256 → 0.3255, 0.3258 → 0.3239, 0.3269 → 0.3242
  - 학습률이 끝으로 갈수록 줄어들기 때문에, 평탄화의 원인을 용량과 학습률 중 어느 쪽으로도 정하지 않는다.
- **T-val 선택 checkpoint** (H-dev 평가 대상)
  - 선택 step: narrow 21 / 126 / 105, wide 189 / 126 / 189
  - objective (parent 0.45121): narrow 0.45305 / 0.44951 / 0.45066(31001은 parent보다 높다), wide 0.44329 / 0.43592 / 0.44088
  - 10번의 평가 가운데 최선을 고른 값이라 낙관적으로 치우친다.
  - 선택 checkpoint의 부분집합 W: narrow 0.3330 / 0.3258 / 0.3244, wide 0.2932 / 0.2891 / 0.2925
- **실행 비용** (RTX 5070, batch 1024, 평가 10회 포함)
  - 6회 합계 27.5분, 실행당 229–298초, step당 1.09–1.42초
  - 같은 군 안에서도 실행마다 다르다(narrow 31001 1.09초, narrow 31002 1.42초). 그래서 폭에 따른 비용 차이로 읽지 않는다.
  - 1,000행 실행 시간으로 추정한 약 24분보다 길었다.
- **1,000행과 함께 읽기**
  - wide − narrow: 1,000행 −0.059 – −0.064, 5,000행 −0.032 – −0.038
  - 시작 W, 부분집합 구성, 행당 노출(약 105회 vs 약 19회)이 모두 다르다. 그래서 이 차이를 자료량 하나의 효과로 읽지 않는다(§13.1).
- **§7.3 조건**: 5,000행에서도 세 seed의 train 폭 효과가 유지됐다. 전체 규모의 계약은 5,000행 H-dev 결과까지 본 뒤 구체화한다. D4는 그 계약에서만 검토한다.
- **다음**: evaluate 승인. 실행 블록은 §13.6의 5k-3이다. 중단되면 5k-3-R, 기록 커밋 전에 멈추면 5k-3-F를 쓴다.

### 13.10 5,000행 H-dev 결과: `DEVELOPMENT_SIGNAL` (기록 `42425b7`)

- **실행**: 승인 커밋 `ff4f0bf`에서 깨끗한 상태로 중단 없이 실행했다(`resumed: null`, 0.48분). 예측 시점과 결과 시점의 git 상태가 같다.
- **출처**
  - 계약 SHA `4dc76d9a…`, 실험 식별값 `84ed0100d792`, 도구 SHA `11eb8b6d…`(계약에 고정된 값과 같다)
  - 예측 파일 SHA `49ac8c2c…`가 예측 기록과 같다. 예측 기록 사본과 로그의 SHA도 일치한다.
  - 모델 7개의 식별값이 c5 계약과 각 실행의 선택 checkpoint와 같다. 동결 state도 같다.
  - H-dev는 15,985행이고, 검사값이 R2-0B 기록과 같다.
- **판독** (R2 §7 그대로, 주 지표는 낮을수록 좋다)

| seed | narrow (control) | wide | 개선 I_s | wide − narrow 95% 구간 |
|---|---:|---:|---:|---|
| 31001 | 0.402379 | 0.391212 | 0.011167 | [−0.015306, −0.006957] |
| 31002 | 0.402262 | 0.393807 | 0.008455 | [−0.011286, −0.005583] |
| 31003 | 0.401844 | 0.390864 | 0.010980 | [−0.013528, −0.008414] |

  - 세 seed 모두 I_s > ε(0.0019461)이고, 95% 상한이 0보다 낮다.
  - control seed 범위 0.000535 < ε. 따라서 `DEVELOPMENT_SIGNAL`이다. 기록값으로 다시 계산해도 같다.
- **c5 대비** (참고값)
  - wide − c5: −0.012831 [−0.017555, −0.007975], −0.010236 [−0.013502, −0.006894], −0.013180 [−0.016786, −0.009472]. 세 seed 모두 구간이 0보다 낮다.
  - narrow − c5: 점추정만 있다(−0.001664, −0.001782, −0.002199). 구간이 없으므로 narrow가 c5보다 낫다거나 같다고 판단하지 않는다.
- **성분 기여** (wide − narrow, 가산 분해; 참고값, ε 없이)

| seed | outcome 기여 [95%] | proof 기여 [95%] |
|---|---|---|
| 31001 | −0.00162 [−0.00446, +0.00130] | −0.00954 [−0.01185, −0.00725] |
| 31002 | −0.00185 [−0.00399, +0.00030] | −0.00660 [−0.00819, −0.00500] |
| 31003 | −0.00980 [−0.01189, −0.00769] | −0.00118 [−0.00273, +0.00038] |

  - 31001·31002는 proof가 이득을 이끌고, 31003은 outcome이 이끈다. 이끌지 않는 쪽의 구간은 0을 포함한다.
  - 공통으로 확정된 것은 주 지표의 순개선뿐이다. 어느 성분이 이득을 이끄는지는 정하지 않는다.
- **1,000행과 함께 보는 기술값** (짝지은 비교가 아니다: 학습 부분집합과 행당 노출이 다르다)
  - 개선 I_s: 1,000행 0.0081–0.0105, 5,000행 0.0085–0.0112
  - 주 지표: narrow는 1,000행 0.405–0.406, 5,000행 0.402. wide는 1,000행 0.396–0.397, 5,000행 0.391–0.394
  - narrow − c5 점추정: 1,000행 +0.0008 – +0.0021, 5,000행 −0.0017 – −0.0022
- **범위**
  - 5,000행 진단의 개발용 판독이다. 확인용 평가(H-conf)가 아니다.
  - 이 모델들은 D4 후보가 아니다. 기력, 승급, c5 변경, V9와 무관하다.
  - 선택 checkpoint는 T-val로 골랐고, 그 선택에 H-dev는 쓰이지 않았다.
- **다음**: §7.3의 조건(5,000행에서도 세 seed 모두 train 폭 효과 유지)을 충족했다. 전체 규모 계약 초안은 §14에 있다.

## 14. 전체 규모 단계 계약 (`R2-D1-HEAD-WIDTH-1/d1-full-contract.json`, 승인 전; 초안 `d1-full-contract.draft.json`)

### 14.1 질문과 고정값

- **질문**: c8 T-train 전체(35,816행)와 같은 210 step 예산에서도, value head 폭의 이점이 train 적합과 H-dev에서 유지되는가
- **진입 근거**: §7.3. 5,000행에서 세 seed 모두 train 폭 효과가 유지됐다(§13.9). 5,000행 H-dev 결과(§13.10)는 이 진입 결정에 쓰지 않는다.
- **이전 단계와 같은 것**
  - 두 군(`value_channels` 1 / 4), seed 31001–31003
  - 확장 방식과 초기화 seed
  - 예측 보존 판정(바이트, CPU float64 ≤ 1e-12, 출력 유한성, 재로드)
  - c8 config, 적합 판정 규칙(마지막 step `W ≤ F_D4 + 0.01`과 `fit_reading`의 순서), R2 §7 판독과 ε
- **자료**: cycle cache의 train split 전체
  - 35,816행, split SHA `487880b9…`, labels SHA `dd6dd35a…`
  - outcome 30,792행, proof 5,024행(D3 mask 기록)
  - 게임 수와 게임 목록 SHA, 구성, 바닥값(정확 / D4)은 prepare가 기록한다.
- **예산**: 두 군 총 210 step
  - ceil(2 × 35,816 / 1,024) = 70 step per epoch, 정확히 3.0 epoch. c8 config 그대로이고, R2-0B 학습과 같다.
- **해석 조건**: 행당 노출은 이 단계 약 3회, 5,000행 약 19회, 1,000행 약 105회다. 규모 간 차이를 자료량 하나의 효과로 읽지 않는다.

### 14.2 control: 새로 학습한다

- **결정**: narrow 3회와 wide 3회를 모두 새로 학습한다.
- **이유**: R2-0B 실행(seed 31001–31003)에는 T-val 선택 checkpoint와 검증 곡선만 남아 있다(선택 step 126 / 84 / 84). D1의 train 판독에 필요한 마지막 210 step의 state와 train 적합도는 없다. 재현 점검 한 번으로 세 seed의 누락 기록을 채울 수 없다.
- **recipe 확인**: R2-0B의 학습은 D1과 같은 `train_value` 호출이다(A1: "c8 (unmasked)"). `--recipe-decision`은 입력 검사에만 쓰였다. 따라서 새 narrow는 R2-0B와 같은 recipe다.
- **R2-0B의 역할: 참고 비교만 한다(판정에 쓰지 않음)**
  - seed마다 새 narrow와 R2-0B 실행을 비교해 보고한다: 평가 곡선(T-val objective, cycle·H6 val), 선택 step과 objective, H-dev 주 지표(R2-0B 기록 `epsilon.json`의 seed 값)
  - 비교 대상 파일: R2-0B 실행 기록(`record.json`, `r2_train.json`)과 `epsilon.json`. 계약이 SHA로 고정한다.

### 14.3 짝맞춤과 실행 환경

- **seed마다 두 군이 같아야 하는 것**
  - 자료 배정 지문(`data_assignment`: cycle 행 순서, replay 비복원 추출, D4 augmentation)
  - 평가 step(`evaluation_steps`)
- 6회는 같은 승인 HEAD와 같은 고정 소스에서 실행한다. 블록이 실행마다 HEAD를 확인하고, 바뀌면 뒤 실행을 시작하지 않는다.
- **실행 환경 기록**: 실행마다 `hybrid.h7r_d1.numeric_settings`의 12개 필드를 기록한다.
  - 필드: torch, cuda, cudnn, device_name, cudnn_enabled, cudnn_benchmark, cudnn_deterministic, cudnn_allow_tf32, matmul_allow_tf32, float32_matmul_precision, deterministic_algorithms, threads
  - 동일성 기준(계약 `environment`): 각 필드가 6회 실행 사이와 prepare 기록에서 정확히 같아야 한다.
- **비교가능성 검사** (fit이 판독 전에 한다)
  - 기준: 환경 동일성, seed마다 두 군의 자료·자료 배정·step 수·평가 step 동일, 6회 모두 하나의 깨끗한 commit
  - 하나라도 어긋나면 비교가능성이 없다. fit은 판독(`reading`) 없이 `comparable: false`와 사유를 `fit.json`에 기록하고 실패로 끝난다(블록이 그 기록을 커밋한다).
  - evaluate는 커밋된 fit 기록이 `comparable: true`이고 같은 검사를 다시 통과해야만 실행된다.
- **parent**: 확장 parent의 state SHA가 1,000행·5,000행 prepare 기록의 값과 같아야 한다.

### 14.4 단계와 판정

- 단계는 prepare → train(6회, seed마다 narrow → wide) → fit → evaluate이다. 각각 따로 승인받는다. 단계는 승인과 무관한 실험 식별값으로 잇는다.
- **기록**: 각 실행에서 두 가지를 따로 남긴다.
  - 마지막 210 step의 state와 train 적합도: train 판독에 쓴다.
  - T-val 선택 checkpoint의 step, objective, 적합도: H-dev 판독에 쓴다.
- **evaluate**
  - A1의 저장 순서(예측 `.partial` → 예측 기록 → 연결 → 결과)와 `--resume`을 쓴다.
  - 결과 저장 직전에 HEAD와 tracked tree를 다시 검사한다.
- **D4**
  - 전체 규모 H-dev가 `DEVELOPMENT_SIGNAL`일 때만 별도 D4 계약(착수 선택 품질)을 검토한다.
  - 이전 D4·X1 평가는 예측 오차 개선이 착수 선택 개선으로 이어진다는 연결을 입증하지 못했다(`NOT_ESTABLISHED`). 연결이 없다고 판정한 것은 아니다. 그래서 D4는 별도로 확인해야 하는 관문이다.
  - 이 계약 자체는 D4 후보, c5 변경, V9를 정하지 않는다.

### 14.5 도구 변경 범위

- **현재 도구가 이 단계를 실행할 수 없는 이유**
  - `scripts/h7r_d1_5k.py`: 로더가 형식 `h7r-d1-contract-5k-v1`, stage `5k`, 5,000행 중첩 부분집합, 1,000행 predecessor를 요구한다.
  - 각 단계 함수가 내부에서 그 로더를 직접 부른다.
  - 이 파일은 5,000행 계약이 SHA로 고정한다. 고치면 5,000행 계약·기록을 다시 검사할 수 없다.
- **방침**: 기존 파일(1,000행 3개, 5,000행 1개)은 바꾸지 않는다. 새 진입점 `scripts/h7r_d1_full.py`를 만든다.
  - **그대로 불러 쓰는 것**
    - `hybrid.h7r_d1`의 순수 함수(확장, 보존 검사, float64 값, 비교, 층별 진단, 바닥값, fit probe, 두 판독)
    - `scripts/h7r_d1.py`의 단계 무관 함수(캐시·모델 검사, 실행 기록 짝맞춤, fit 기록 검사)
    - `scripts/h7r_d1_evaluate.py`의 `analyse`, `load_models`
  - **새로 쓰는 것**
    - 전체 규모 계약 로더: 5,000행 계약·기록을 predecessor로 고정한다.
    - 전체 split 사용: 부분집합 선택 없이 cycle train 전체를 쓴다. 행 수와 split SHA를 대조하고, 게임 수와 게임 목록 SHA를 기록한다.
    - parent state SHA를 1,000행·5,000행 기록과 대조한다.
    - 실행 환경 기록과 그 대조
    - R2-0B 참고 비교(fit과 evaluate 출력에 추가, 판정 밖)
    - prepare / train / fit / evaluate[`--resume`]: 5,000행 도구와 같은 절차
  - **계약이 고정할 소스**: 5,000행의 17개 + `scripts/h7r_d1_full.py`. 도구를 작성·테스트한 뒤 SHA를 채우고 `status: fixed`로 바꾼다. 지금은 `null`이다.
- **비용 추정** (측정값 아님)
  - prepare: CPU float64로 약 4만 2천 행 × 모델 4개. 수 분 예상
  - train: 6회 × 210 step. R2-0B와 5,000행 실행 시간 기준으로 약 25–30분
  - evaluate: 1분 안팎

### 14.6 테스트 계획 (`tests/test_h7r_d1_full.py`)

1. **로더**
   - 고정값(stage, 예산, 두 군, seed, 규칙)과 predecessor SHA·판독이 하나라도 다르면 거부한다.
   - 단계 승인 순서를 확인하고, 출력은 `docs/.../full/`·`runs/.../full` 아래만 허용한다.
   - 소스 SHA를 검사하고, 초안이나 `null` 소스는 거부한다.
2. **자료**
   - 전체 split이 행 수·split SHA와 같아야 하고, 다르면 거부한다.
   - 대역 자료에서 부분집합 선택 없이 split 전체를 쓰는지 확인한다.
3. **prepare**
   - 같은 입력에서 5,000행 prepare와 기록이 같다(자료만 다름).
   - parent state가 이전 기록과 다르면 FAIL이고 학습을 거부한다.
4. **train·fit**
   - 같은 입력·예산에서 5,000행 train과 실행 기록이 같다.
   - 실행 환경이 기록된다. 계약이 정한 필드가 6회 사이 또는 prepare 기록과 하나라도 다르면, fit이 판독 없이 사유를 기록하고 중단한다. evaluate도 거부한다.
   - R2-0B 참고 비교가 판정과 분리돼 기록된다.
5. **evaluate**
   - 5,000행 도구와 판독이 같다.
   - 중단 후 `--resume`이 완성한다.
   - 판독 중 HEAD나 dirty 상태가 바뀌면 결과를 쓰지 않는다.
   - 변조된 예측과 기록 없는 `.partial`은 거부한다.
6. **회귀**
   - 1,000행 v2 계약과 A1, 5,000행 계약의 세 단계와 기록이 그대로 읽힌다.
   - 전체 규모 계약이 고정한 기존 파일의 SHA가 5,000행 계약의 값과 같다.
7. **절차**
   - 5,000행 블록에서 기계적으로 바꾼 전체 규모 블록에 `5k`·`5,000`·이전 도구 이름이 남지 않는다.
   - 기록 단계를 임시 저장소에서 실제 git과 기록 도구로 실행한다.

### 14.7 남은 항목과 승인 단계 (초안 시점; 구현 뒤 상태는 §14.9)

- **구현** (이 초안의 범위 밖, 별도 진행)
  1. `scripts/h7r_d1_full.py`와 그 테스트
  2. 계약 고정: 소스 SHA를 채우고 `status: fixed`, 이름은 `d1-full-contract.json`
  3. 실행 블록
- **승인과 실행**
  1. 도구·계약 검토
  2. prepare 승인
  3. train 승인(6회 + fit)
  4. evaluate 승인
- 결과는 위 판정 규칙으로만 읽는다. 결과를 보고 문턱이나 규칙을 바꾸지 않는다. 자동으로 다음 단계를 시작하지 않는다.

### 14.8 실행 절차 (데스크톱)
- 블록은 §13.6의 5,000행 블록과 같다. 계약·도구·경로·커밋 문구만 기계적으로 바꿨다.
  - 바꾼 결과에 이전 단계의 이름이 남지 않았는지 테스트로 확인한다.
  - 기록 단계는 임시 저장소에서 실제 git과 기록 도구로, 도구는 대역으로 실행해 확인했다.
- 각 단계는 승인 커밋을 받은 뒤 `git pull --ff-only`로 같은 HEAD에서 실행한다.
- 실패하면 다음 단계로 넘어가지 않는다. 실패 폴더를 지우거나 블록을 다시 돌리지 않는다. 중단 위치를 보고 따로 정한다.

full-1. prepare와 기록 커밋 (`approved.prepare` 커밋 후; 학습 없음)
   - 단계별 경과 시간을 출력하고 `prepare.json`의 `timings_seconds`에 기록한다. 자료, 바닥값, c5 float64, seed별 검사가 각각 한 단계다.
   - 도구가 확인하는 것: 전체 split의 행 수·split·labels SHA·구성, 확장 parent의 state SHA(5,000행·1,000행 기록과 같아야 함), 보존 gate

```bash
(
  set -euo pipefail
  cd ~/renju-h6
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src:$PWD/scripts"
  [ "$(git branch --show-current)" = feat/h7-c1 ] || { echo "STOP: wrong branch"; exit 1; }
  git pull --ff-only origin feat/h7-c1
  D=docs/h7r/r2/R2-D1-HEAD-WIDTH-1
  X="$D/full"
  L=runs/h7r/R2-D1-HEAD-WIDTH-1/full/logs
  test -z "$(git status --porcelain --untracked-files=no)"
  python scripts/check_track_b_frozen.py
  test ! -e "$X/prepare.json"
  mkdir -p "$L"
  LOG="$L/prepare-$(date +%Y%m%d-%H%M%S).log"
  B="$(basename "$LOG" .log)"
  ts(){ while IFS= read -r line; do printf '%s %s\n' "$(date +%s.%N)" "$line"; done; }
  run_rc=0
  { echo __START__; rc=0; python -u scripts/h7r_d1_full.py prepare --d1-contract "$D/d1-full-contract.json" \
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
  git commit -m "H7-R D1 full: full-scale prepare $verdict: subset, floors and widened parents (records only)"
  git push origin feat/h7-c1
  echo "__RECORDED__ $run_rc"
  exit "$run_rc"
)
```

full-1-F. prepare 기록 보존 (full-1이 기록 커밋 전에 멈췄을 때만)

```bash
(
  set -euo pipefail
  cd ~/renju-h6
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src:$PWD/scripts"
  [ "$(git branch --show-current)" = feat/h7-c1 ] || { echo "STOP: wrong branch"; exit 1; }
  D=docs/h7r/r2/R2-D1-HEAD-WIDTH-1
  X="$D/full"
  L=runs/h7r/R2-D1-HEAD-WIDTH-1/full/logs
  echo __START__
  git reset -q -- "$X"
  test -z "$(git status --porcelain --untracked-files=no)"
  shopt -s nullglob
  logs=("$L"/prepare-*.log)
  [ "${#logs[@]}" -gt 0 ] || { echo "STOP: no prepare log"; exit 1; }
  LOG="${logs[0]}"
  for f in "${logs[@]}"; do if [ "$f" -nt "$LOG" ]; then LOG="$f"; fi; done
  B="$(basename "$LOG" .log)"
  echo "log: $LOG"
  tail -n 5 "$LOG"
  expect=()
  python scripts/h7r_d4_records.py copy-logs --resume --src-dir "$L" --pattern "$B.log" \
    --dest-dir "$X/prepare-logs" --manifest "$X/prepare-logs/$B.manifest.json"
  for f in "$X/prepare-logs/$B.log" "$X/prepare-logs/$B.manifest.json" "$X/prepare.json"; do
    if [ -f "$f" ] && ! git ls-files --error-unmatch "$f" >/dev/null 2>&1; then
      git add "$f"
      expect+=(--expect "$f")
    fi
  done
  if [ "${#expect[@]}" -eq 0 ]; then
    git push origin HEAD:feat/h7-c1
    echo "nothing left to record; push checked"; echo "__EXIT__ 0"; exit 0
  fi
  python scripts/h7r_d4_records.py verify-staged "${expect[@]}"
  git commit -m "H7-R D1 full: full-scale prepare attempt record $B (records only)"
  git push origin feat/h7-c1
  echo "__EXIT__ 0"
)
```

full-2. 학습 6회, train 적합 판독, 기록 커밋 (`approved.train` 커밋 후)
   - seed마다 narrow → wide 순서로 돌린다. 각 실행은 210 step(epoch당 70 step, 3.0 epoch)이고 실행 환경을 기록한다.
   - fit은 6회가 모두 끝났을 때만 돈다. 비교가능성 검사에 실패하면 사유를 기록하고 판독 없이 실패로 끝난다.

```bash
(
  set -euo pipefail
  cd ~/renju-h6
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src:$PWD/scripts"
  [ "$(git branch --show-current)" = feat/h7-c1 ] || { echo "STOP: wrong branch"; exit 1; }
  git pull --ff-only origin feat/h7-c1
  D=docs/h7r/r2/R2-D1-HEAD-WIDTH-1
  X="$D/full"
  R=runs/h7r/R2-D1-HEAD-WIDTH-1/full
  L="$R/logs"
  test -z "$(git status --porcelain --untracked-files=no)"
  python scripts/check_track_b_frozen.py
  test ! -e "$X/fit.json"
  test ! -e "$X/run-records"
  test ! -e "$X/train-logs"
  for s in 31001 31002 31003; do for arm in narrow wide; do test ! -e "$R/$arm-$s"; done; done
  head="$(git rev-parse HEAD)"
  STAMP="$(date +%Y%m%d-%H%M%S)"
  mkdir -p "$L"
  ts(){ while IFS= read -r line; do printf '%s %s\n' "$(date +%s.%N)" "$line"; done; }
  run_rc=0
  where=none
  done_runs=0
  for s in 31001 31002 31003; do
    for arm in narrow wide; do
      if [ "$(git rev-parse HEAD)" != "$head" ]; then run_rc=97; where="HEAD changed before $arm-$s"; break 2; fi
      LOG="$L/train-$arm-$s-$STAMP.log"
      { echo __START__; rc=0; python -u scripts/h7r_d1_full.py train --d1-contract "$D/d1-full-contract.json" \
          --arm "$arm" --seed "$s" \
          --cycle-cache runs/h7c/c8/value_cache --cycle-rows runs/h7c/c8/value_rows.jsonl.gz \
          --replay-cache data/tactical/h6/value_cache_v3 --c5 runs/h7c/c5/value/best.pt --device cuda 2>&1 || rc=$?
        echo "__EXIT__ $rc"; exit "$rc"; } | ts | tee "$LOG" || run_rc=$?
      if [ "$run_rc" -ne 0 ]; then where="train $arm-$s"; break 2; fi
      done_runs=$((done_runs + 1))
    done
  done
  if [ "$run_rc" -eq 0 ] && [ "$(git rev-parse HEAD)" != "$head" ]; then
    run_rc=97; where="HEAD changed before fit"
  fi
  if [ "$run_rc" -eq 0 ]; then
    LOG="$L/fit-$STAMP.log"
    { echo __START__; rc=0; python -u scripts/h7r_d1_full.py fit --d1-contract "$D/d1-full-contract.json" 2>&1 || rc=$?
      echo "__EXIT__ $rc"; exit "$rc"; } | ts | tee "$LOG" || run_rc=$?
    if [ "$run_rc" -ne 0 ]; then where=fit; fi
  fi
  echo "[d1-train-block] runs completed $done_runs/6, exit $run_rc, stopped at: $where"
  python scripts/h7r_d4_records.py copy-logs --src-dir "$L" --pattern "*-$STAMP.log" \
    --dest-dir "$X/train-logs" --manifest "$X/train-logs/$STAMP.manifest.json"
  expect=(--expect "$X/train-logs/$STAMP.manifest.json")
  for f in "$X"/train-logs/*-"$STAMP".log; do expect+=(--expect "$f"); done
  copied=0
  for s in 31001 31002 31003; do
    for arm in narrow wide; do
      [ -d "$R/$arm-$s" ] || continue
      for src in "$R/$arm-$s"/*.json; do
        [ -f "$src" ] || continue
        dst="$X/run-records/$arm-$s/$(basename "$src")"
        mkdir -p "$(dirname "$dst")"
        cp --no-clobber "$src" "$dst"
        cmp "$src" "$dst"
        expect+=(--expect "$dst")
        copied=$((copied + 1))
      done
    done
  done
  if [ "$copied" -gt 0 ]; then
    python scripts/h7r_d4_records.py manifest --dir "$X/run-records" --output "$X/run-records/records_manifest-$STAMP.json"
    expect+=(--expect "$X/run-records/records_manifest-$STAMP.json")
  fi
  if [ -f "$X/fit.json" ]; then expect+=(--expect "$X/fit.json"); fi
  for e in "${expect[@]}"; do [ "$e" = --expect ] || git add "$e"; done
  python scripts/h7r_d4_records.py verify-staged "${expect[@]}"
  if [ "$run_rc" -eq 0 ]; then verdict="6/6 runs and the train-fit reading"; else verdict="FAIL at $where (exit $run_rc), $done_runs/6 runs"; fi
  git commit -m "H7-R D1 full: full-scale training $verdict (records only)"
  git push origin feat/h7-c1
  echo "__RECORDED__ $run_rc"
  exit "$run_rc"
)
```

full-2-F. 학습 기록 보존 (full-2가 기록 커밋 전에 멈췄을 때만)

```bash
(
  set -euo pipefail
  cd ~/renju-h6
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src:$PWD/scripts"
  [ "$(git branch --show-current)" = feat/h7-c1 ] || { echo "STOP: wrong branch"; exit 1; }
  D=docs/h7r/r2/R2-D1-HEAD-WIDTH-1
  X="$D/full"
  R=runs/h7r/R2-D1-HEAD-WIDTH-1/full
  L="$R/logs"
  echo __START__
  git reset -q -- "$X"
  test -z "$(git status --porcelain --untracked-files=no)"
  shopt -s nullglob
  logs=("$L"/train-*.log)
  [ "${#logs[@]}" -gt 0 ] || { echo "STOP: no train log"; exit 1; }
  last="${logs[0]}"
  for f in "${logs[@]}"; do if [ "$f" -nt "$last" ]; then last="$f"; fi; done
  STAMP="$(basename "$last" .log | grep -oE '[0-9]{8}-[0-9]{6}$')"
  echo "attempt $STAMP"
  for f in "$L"/*-"$STAMP".log; do echo "== $f"; tail -n 3 "$f"; done
  python scripts/h7r_d4_records.py copy-logs --resume --src-dir "$L" --pattern "*-$STAMP.log" \
    --dest-dir "$X/train-logs" --manifest "$X/train-logs/$STAMP.manifest.json"
  candidates=("$X/train-logs/$STAMP.manifest.json" "$X"/train-logs/*-"$STAMP".log)
  for s in 31001 31002 31003; do
    for arm in narrow wide; do
      [ -d "$R/$arm-$s" ] || continue
      for src in "$R/$arm-$s"/*.json; do
        [ -f "$src" ] || continue
        dst="$X/run-records/$arm-$s/$(basename "$src")"
        mkdir -p "$(dirname "$dst")"
        cp --no-clobber "$src" "$dst"
        cmp "$src" "$dst"
        candidates+=("$dst")
      done
    done
  done
  if [ -d "$X/run-records" ]; then
    python scripts/h7r_d4_records.py manifest --resume --dir "$X/run-records" --output "$X/run-records/records_manifest-$STAMP.json"
    candidates+=("$X/run-records/records_manifest-$STAMP.json")
  fi
  if [ -f "$X/fit.json" ]; then candidates+=("$X/fit.json"); fi
  expect=()
  for f in "${candidates[@]}"; do
    if ! git ls-files --error-unmatch "$f" >/dev/null 2>&1; then git add "$f"; expect+=(--expect "$f"); fi
  done
  if [ "${#expect[@]}" -eq 0 ]; then
    git push origin HEAD:feat/h7-c1
    echo "nothing left to record; push checked"; echo "__EXIT__ 0"; exit 0
  fi
  python scripts/h7r_d4_records.py verify-staged "${expect[@]}"
  git commit -m "H7-R D1 full: full-scale training attempt record $STAMP (records only)"
  git push origin feat/h7-c1
  echo "__EXIT__ 0"
)
```

full-3. H-dev 판독과 기록 커밋 (`approved.evaluate` 커밋 후)
   - 평가 대상은 c5와 T-val 선택 checkpoint 6개, 판독은 R2 §7이다. R2-0B의 H-dev 주 지표는 참고값으로만 함께 기록된다.

```bash
(
  set -euo pipefail
  cd ~/renju-h6
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src:$PWD/scripts"
  [ "$(git branch --show-current)" = feat/h7-c1 ] || { echo "STOP: wrong branch"; exit 1; }
  git pull --ff-only origin feat/h7-c1
  D=docs/h7r/r2/R2-D1-HEAD-WIDTH-1
  X="$D/full"
  R=runs/h7r/R2-D1-HEAD-WIDTH-1/full
  L="$R/logs"
  test -z "$(git status --porcelain --untracked-files=no)"
  python scripts/check_track_b_frozen.py
  git ls-files --error-unmatch "$X/prepare.json" "$X/fit.json" "$D/d1-full-contract.json" >/dev/null
  test ! -e "$X/d1_eval.json"
  test ! -e "$R/predictions.pt"
  test ! -e "$R/predictions.pt.partial"
  test ! -e "$R/evaluate-predicted.json"
  for s in 31001 31002 31003; do for arm in narrow wide; do test -f "$R/$arm-$s/best.pt"; test -f "$R/$arm-$s/best.json"; done; done
  mkdir -p "$L"
  LOG="$L/evaluate-$(date +%Y%m%d-%H%M%S).log"
  B="$(basename "$LOG" .log)"
  ts(){ while IFS= read -r line; do printf '%s %s\n' "$(date +%s.%N)" "$line"; done; }
  run_rc=0
  { echo __START__; rc=0; python -u scripts/h7r_d1_full.py evaluate --d1-contract "$D/d1-full-contract.json" \
      --cache runs/h7r/R1-HDEV-1/cache --c5 runs/h7c/c5/value/best.pt --device cuda 2>&1 || rc=$?
    echo "__EXIT__ $rc"; exit "$rc"; } | ts | tee "$LOG" || run_rc=$?
  python scripts/h7r_d4_records.py copy-logs --src-dir "$L" --pattern "$B.log" \
    --dest-dir "$X/evaluate-logs" --manifest "$X/evaluate-logs/$B.manifest.json"
  if [ -f "$R/evaluate-predicted.json" ]; then
    mkdir -p "$X/evaluate-records"
    cp --no-clobber "$R/evaluate-predicted.json" "$X/evaluate-records/evaluate-predicted.json"
    cmp "$R/evaluate-predicted.json" "$X/evaluate-records/evaluate-predicted.json"
  fi
  expect=(--expect "$X/evaluate-logs/$B.log" --expect "$X/evaluate-logs/$B.manifest.json")
  for f in "$X/evaluate-records/evaluate-predicted.json" "$X/d1_eval.json"; do
    if [ -f "$f" ] && ! git ls-files --error-unmatch "$f" >/dev/null 2>&1; then expect+=(--expect "$f"); fi
  done
  for e in "${expect[@]}"; do [ "$e" = --expect ] || git add "$e"; done
  python scripts/h7r_d4_records.py verify-staged "${expect[@]}"
  if [ "$run_rc" -eq 0 ]; then verdict=reading; else verdict="FAIL (exit $run_rc)"; fi
  git commit -m "H7-R D1 full: full-scale H-dev $verdict (records only)"
  git push origin feat/h7-c1
  echo "__RECORDED__ $run_rc"
  exit "$run_rc"
)
```

full-3-R. 중단된 평가의 완성 (예측 기록이 있고 결과가 없을 때만; 새 추론 없음)

```bash
(
  set -euo pipefail
  cd ~/renju-h6
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src:$PWD/scripts"
  [ "$(git branch --show-current)" = feat/h7-c1 ] || { echo "STOP: wrong branch"; exit 1; }
  D=docs/h7r/r2/R2-D1-HEAD-WIDTH-1
  X="$D/full"
  R=runs/h7r/R2-D1-HEAD-WIDTH-1/full
  L="$R/logs"
  test -z "$(git status --porcelain --untracked-files=no)"
  python scripts/check_track_b_frozen.py
  git ls-files --error-unmatch "$X/prepare.json" "$X/fit.json" "$D/d1-full-contract.json" >/dev/null
  test ! -e "$X/d1_eval.json"
  test -f "$R/evaluate-predicted.json"
  for s in 31001 31002 31003; do for arm in narrow wide; do test -f "$R/$arm-$s/best.pt"; test -f "$R/$arm-$s/best.json"; done; done
  mkdir -p "$L"
  LOG="$L/evaluate-resume-$(date +%Y%m%d-%H%M%S).log"
  B="$(basename "$LOG" .log)"
  ts(){ while IFS= read -r line; do printf '%s %s\n' "$(date +%s.%N)" "$line"; done; }
  run_rc=0
  { echo __START__; rc=0; python -u scripts/h7r_d1_full.py evaluate --d1-contract "$D/d1-full-contract.json" \
      --cache runs/h7r/R1-HDEV-1/cache --c5 runs/h7c/c5/value/best.pt --device cuda --resume 2>&1 || rc=$?
    echo "__EXIT__ $rc"; exit "$rc"; } | ts | tee "$LOG" || run_rc=$?
  python scripts/h7r_d4_records.py copy-logs --src-dir "$L" --pattern "$B.log" \
    --dest-dir "$X/evaluate-logs" --manifest "$X/evaluate-logs/$B.manifest.json"
  if [ -f "$R/evaluate-predicted.json" ]; then
    mkdir -p "$X/evaluate-records"
    cp --no-clobber "$R/evaluate-predicted.json" "$X/evaluate-records/evaluate-predicted.json"
    cmp "$R/evaluate-predicted.json" "$X/evaluate-records/evaluate-predicted.json"
  fi
  expect=(--expect "$X/evaluate-logs/$B.log" --expect "$X/evaluate-logs/$B.manifest.json")
  for f in "$X/evaluate-records/evaluate-predicted.json" "$X/d1_eval.json"; do
    if [ -f "$f" ] && ! git ls-files --error-unmatch "$f" >/dev/null 2>&1; then expect+=(--expect "$f"); fi
  done
  for e in "${expect[@]}"; do [ "$e" = --expect ] || git add "$e"; done
  python scripts/h7r_d4_records.py verify-staged "${expect[@]}"
  if [ "$run_rc" -eq 0 ]; then verdict=reading; else verdict="FAIL (exit $run_rc)"; fi
  git commit -m "H7-R D1 full: full-scale H-dev resumed $verdict (records only)"
  git push origin feat/h7-c1
  echo "__RECORDED__ $run_rc"
  exit "$run_rc"
)
```

full-3-F. 평가 기록 보존 (full-3 또는 full-3-R이 기록 커밋 전에 멈췄을 때만)

```bash
(
  set -euo pipefail
  cd ~/renju-h6
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src:$PWD/scripts"
  [ "$(git branch --show-current)" = feat/h7-c1 ] || { echo "STOP: wrong branch"; exit 1; }
  D=docs/h7r/r2/R2-D1-HEAD-WIDTH-1
  X="$D/full"
  L=runs/h7r/R2-D1-HEAD-WIDTH-1/full/logs
  echo __START__
  git reset -q -- "$X"
  test -z "$(git status --porcelain --untracked-files=no)"
  shopt -s nullglob
  logs=("$L"/evaluate-*.log)
  [ "${#logs[@]}" -gt 0 ] || { echo "STOP: no evaluate log"; exit 1; }
  LOG="${logs[0]}"
  for f in "${logs[@]}"; do if [ "$f" -nt "$LOG" ]; then LOG="$f"; fi; done
  B="$(basename "$LOG" .log)"
  echo "log: $LOG"
  tail -n 5 "$LOG"
  expect=()
  python scripts/h7r_d4_records.py copy-logs --resume --src-dir "$L" --pattern "$B.log" \
    --dest-dir "$X/evaluate-logs" --manifest "$X/evaluate-logs/$B.manifest.json"
  R=runs/h7r/R2-D1-HEAD-WIDTH-1/full
  if [ -f "$R/evaluate-predicted.json" ]; then
    mkdir -p "$X/evaluate-records"
    cp --no-clobber "$R/evaluate-predicted.json" "$X/evaluate-records/evaluate-predicted.json"
    cmp "$R/evaluate-predicted.json" "$X/evaluate-records/evaluate-predicted.json"
  fi
  for f in "$X/evaluate-logs/$B.log" "$X/evaluate-logs/$B.manifest.json" "$X/evaluate-records/evaluate-predicted.json" "$X/d1_eval.json"; do
    if [ -f "$f" ] && ! git ls-files --error-unmatch "$f" >/dev/null 2>&1; then
      git add "$f"
      expect+=(--expect "$f")
    fi
  done
  if [ "${#expect[@]}" -eq 0 ]; then
    git push origin HEAD:feat/h7-c1
    echo "nothing left to record; push checked"; echo "__EXIT__ 0"; exit 0
  fi
  python scripts/h7r_d4_records.py verify-staged "${expect[@]}"
  git commit -m "H7-R D1 full: full-scale H-dev attempt record $B (records only)"
  git push origin feat/h7-c1
  echo "__EXIT__ 0"
)
```

### 14.9 진행 상태

- **구현 완료** (이 커밋)
  - `scripts/h7r_d1_full.py`와 `tests/test_h7r_d1_full.py`
  - 계약 `d1-full-contract.json`
    - 초안 값에 검토 보완 세 가지를 더했다: 환경 필드와 동일성 기준, 짝맞춤 기준, D4 문구
    - 소스 18개의 SHA를 채워 `status: fixed`로 고정했다.
    - 실험 식별값 `25c49e408aff`, 모든 `approved` false
    - 초안은 설계 기록으로 남긴다. 테스트가 고정 계약과 초안이 위 항목 말고는 같은지 확인한다.
  - §14.8의 실행 블록 7개
- **테스트로 확인한 것** (대역 입력; 실제 c8·H-dev·GPU는 데스크톱 실행에서만 확인된다)
  - 로더: 커밋된 계약은 세 단계 모두 미승인으로 거부한다. 승인한 사본은 세 단계 모두 읽힌다. 고정값, predecessor, 자료, 예산, control, 환경, 출력, 이전 도구 SHA가 틀리면 거부한다.
  - 자료: split 전체를 선택 없이 쓴다. 행 수, 구성, split·labels SHA가 다르면 거부한다.
  - prepare: 같은 입력에서 5,000행 prepare와 기록이 같다. parent state가 이전 기록과 다르면 FAIL이고 학습을 거부한다. 단계별 경과 시간을 기록한다.
  - train·fit
    - 같은 입력·예산에서 5,000행 train과 실행 기록이 같다. 실행 환경이 기록된다.
    - fit은 R2-0B 참고 비교를 함께 기록한다.
    - 환경 필드 하나 또는 짝맞춤(평가 step, 자료 배정, commit, 자료)이 다르면 판독 없이 사유를 기록하고 멈춘다. evaluate 진입 검사도 거부한다.
  - evaluate
    - 5,000행 도구와 판독이 같고, R2-0B H-dev 참고값을 기록한다.
    - 결과 저장 중단은 `--resume`이 추론 없이 완성한다.
    - 재개 중 HEAD가 바뀌거나 정상 실행 중 dirty가 되면 결과를 쓰지 않는다. 예측과 기록은 보존된다.
    - 변조된 예측과 기록 없는 `.partial`은 거부한다.
  - 회귀: 1,000행 v2와 5,000행 계약의 세 단계, 그리고 그 기록이 그대로 읽힌다.
- **실제 데이터에서 남은 확인 항목** (prepare 이후)
  - c8 cycle cache의 train split이 35,816행이고, 구성이 outcome 30,792 / proof 5,024인지
  - 게임 수와 게임 목록 SHA. 처음 기록된다.
  - 전체 split의 바닥값과 적합 문턱
  - 확장 parent가 실제 GPU에서도 이전 기록과 같은 state인지
  - float64 검사의 실제 소요 시간. 단계별 경과 시간으로 예상 시간을 갱신한다.
  - 학습 6회의 실제 시간과 환경 동일성
  - 새 narrow와 R2-0B의 같은 seed 실행이 얼마나 비슷한지(참고값)
- **승인과 실행**
  1. 구현 검토
  2. prepare 승인 → full-1
  3. train 승인 → full-2
  4. evaluate 승인 → full-3

## 15. 자율 진행 위임 R2-D1-AUTOPILOT-1 (2026-11-09)

- **위임 기록**: `R2-D1-HEAD-WIDTH-1/autopilot/delegation-1.json`
  - 담은 것: 범위, 결정값, 진행 조건, 상한, 중단 조건, 최종 승인으로 남긴 항목
  - 상태 기록은 `autopilot/state.md`에 둔다.
- **우선순위**: 사용자의 명시적 위임을 먼저 적용하고, 그 범위 안에서 CLAUDE.md를 따른다.
- **범위**: 전체 규모 학습·fit·H-dev → 넓은 head용 새 후보 factory(`tb_v3_net`는 그대로) → D4 1회 → 확인 평가 1세트(H-conf, A-conf) → R4 프로그램 준비
  - 범위 안의 하위 승인 플래그는 진행 조건을 확인한 뒤 직접 커밋한다.
  - 커밋 메시지에 위임 기록과 근거 기록을 남긴다.
- **결정값**
  - D4 주 비교: 같은 seed의 wide:narrow 3쌍. wide:c5는 참고만 한다.
  - D4 다음 단계 진입: `ALL_THREE_BETTER`일 때만 확인 평가로 간다. 그 밖은 보존·보고 후 종료한다.
  - 확인 Arena 후보: wide seed 31001. 결과와 무관하게 고정한다.
  - H-conf 예측 검사: 세 seed 모두 평가한다.
  - 시간 gate: 같은 측정 조건에서 평균 +10%, p95 +20% 이내
- **상한**: 총 벽시계 24시간(추정 시간이 아니라 위임 상한), D4 본 평가 1회, 확인 평가 1세트
- **사전 고정 순서**
  - factory와 D4의 방법·표본·예산·판독 규칙은 전체 규모 H-dev evaluate 승인 전에 고정·커밋한다.
  - 모델 SHA는 학습 뒤, D4와 확인 결과를 열기 전에 연결한다.
  - 확인 평가의 미정 값은 확인 자료를 열기 전에 근거와 함께 계약에 고정한다.
- **재시도**: 판독과 무관한 인프라 오류에 한해, 원인을 확인한 뒤 1회만 한다. 실패 기록은 보존한다.
- **중단**
  - SHA·모델 연결·비교가능성 오류는 중단 사유로 보고한다.
  - 계약 기준 미충족, `HOLD`, 진입 조건 불충족, 상한 도달, 범위 밖 판단이 필요하면 결과를 보존하고 끝낸다.
- **최종 승인 대상**: R4 루프 시작, champion·canonical·웹 배포 변경
- **실행 분담**
  - 클라우드 세션: 코드, 계약, 테스트, 승인 커밋, 기록 분석
  - 데스크톱: GPU·Rapfi 실행. 긴 작업은 tmux, 실행 중 HEAD 고정
- **사용자 승인문**(요약 인용): "현재 단계부터 R4 재개 준비까지의 자율 진행을 승인합니다. … 루프 시작과 실제 champion·canonical·웹 배포 변경은 최종 승인 대상으로 남깁니다."

## 16. P2 후보 factory v2와 P3 D4 계약 (전체 규모 H-dev evaluate 승인 전에 고정)

### 16.1 P2: 후보 factory v2 (`src/analysis/tb_v3_net_v2.py`)

- **왜 필요한가**: `analysis.tb_v3_net`(factory v1)은 frozen architecture(`value_channels` 1)만 받는다. 넓힌 head(`value_channels` 4)는 v1이 거부하므로, D4와 그 뒤의 Arena에 쓸 새 factory가 필요하다. v1은 그대로 둔다.
- **바뀌는 것은 허용 value architecture 목록 하나다** (`VALUE_ARCHITECTURES`: frozen 구조, 그리고 `value_channels` 4)
  - engine·runtime 검사, fast5 소스·binding 검사, 탐색 설정(v1의 `_build`)은 v1과 같다.
  - 모델 정체성(SHA 네 개)과 agent 이름도 v1의 함수를 쓴다.
- **trunk·policy head 고정 증명**
  - 불러온 value 모델에서 `frozen_state`를 다시 계산해 `FROZEN_STATE_SHA256`(`c2fbaac3…`)과 같아야 한다.
  - 이 값은 R2-DIAG-1과 R2-D4-BASELINE-1 계약이 고정하고 다시 계산한 값이다. 테스트가 두 계약과 같은지 확인한다.
  - metadata가 H3 parent를 적었다면 frozen H3 checkpoint여야 하고, 선언한 frozen state는 다시 계산한 값과 같아야 한다.
- **범위**: 연구 도구 전용이다. `CANONICAL = False`이고, `analysis.canonical`의 어떤 factory도 v2가 아니다. 테스트가 `scripts/h7[cr]_*` 밖에서 v2를 쓰지 않는지 확인한다.
- **테스트** (`tests/test_tb_v3_net_v2.py`)
  - 1채널 모델 쌍은 v1과 같은 정체성을 가지고, v2로 두어도 v1과 같은 수·진단 기록을 낸다.
  - 넓힌 head는 v2만 받고, v2로 실제로 둔다.
  - 거부하는 경우: 다른 architecture(`value_channels` 8), 바뀐 trunk, 다른 H3 parent, 틀린 frozen state 선언, H6-2a가 아닌 value, 넓힌 policy 네트워크, 경로·형식 오류
- **시간 gate** (`scripts/h7r_factory_v2_timing.py`, 기록 형식 `h7r-factory-v2-timing-v1`)
  - A: narrow seed 31001. factory v2의 1채널 경로로 만든다. 이 경로는 factory v1의 build와 같다(개정 A1, §16.6).
  - B: factory v2와 wide seed 31001
  - 위치: 고정 O2 suite의 처음 24개 opening에서 A의 수를 따라간 6수씩. A와 B는 같은 위치를 각자 사본으로 탐색한다.
  - 순서는 위치마다 번갈아 바꾼다. agent마다 warm-up 1수는 세지 않는다.
  - 통과 조건: mean(B) ≤ 1.10 × mean(A), p95(B) ≤ 1.20 × p95(A). 위임 결정값 그대로다.
  - 기록은 한 번만 쓴다. 실행 설정, 모델 정체성, 모든 측정값이 들어간다.
  - 같은 수 비율도 기록하지만 판정에는 쓰지 않는다.

### 16.2 P3: D1 D4 계약 (`R2-D1-HEAD-WIDTH-1/d4/d1-d4-contract.json`, 도구 `scripts/h7r_d1_d4.py`)

- **질문**: 전체 규모에서 넓힌 head의 H-dev 예측 이득이 수 선택으로 이어지는가. 직접 대조군(같은 seed의 narrow)과 비교한다.
- **방법은 baseline D4(R2-D4-BASELINE-1)와 같다.** 계약 로더가 baseline 계약과 하나씩 대조한다.
  - 같은 것: sealed H-dev, frozen state, runtime 정체성, Rapfi 프로그램·정체성, 위치 선택, 예산, device, 통계(`d4.STATS`), INVALID 정책
  - 표본: baseline과 같은 eligible manifest 규칙(c5가 actor). prepare가 계약의 `output.sample`로 고정한다(개정 A2부터 `d4/eligible-a2.json`).
- **비교**
  - 주 비교: `wide31001:narrow31001`, `wide31002:narrow31002`, `wide31003:narrow31003`
  - 참고: 각 wide와 c5. 구간만 기록하고 판독 label은 지운다.
- **모델 연결**
  - 실행할 때 커밋된 전체 규모 실행 기록(`full/run-records/{arm}-{seed}/d1_train.json`)에서 읽는다.
  - 조건: 실험 `25c49e408aff`, verdict `CANDIDATE`, T-val로 고른 checkpoint(`selected.state`)
  - D4나 H-dev 결과로 모델을 고르지 않는다.
  - 실행 시 각 파일의 SHA·metadata·frozen state를 다시 확인한다(baseline의 `checked_models`). 넓힌 checkpoint가 이 검사를 통과하는지 테스트로 확인했다.
- **진입 조건** (실행할 때 도구가 커밋된 기록으로 확인)
  - 전체 규모 fit이 비교가능(`comparable`)하다.
  - 전체 규모 H-dev 판독이 `DEVELOPMENT_SIGNAL`이다.
  - 시간 gate가 통과했고, 연결된 seed 31001 쌍에서 잰 것이다.
  - full은 추가로 커밋된 smoke 기록이 필요하다. 같은 실험이고, c5가 actor의 수를 모두 재현했어야 한다.
- **재생**
  - 일곱 모델 모두 factory v2로 만든다. c5도 v2로 만들어 actor의 수를 재현해야 하므로, 이 재생이 v2의 1채널 경로를 실제 자료로 다시 검사한다.
- **판독** (주 비교 3쌍만)
  - 쌍마다 baseline label을 쓴다: `label95`(95%), `label_individual`(98.333%)
  - `ALL_THREE_BETTER`: 세 쌍 모두 `label95`가 `BETTER`
  - `ALL_THREE_WORSE`: 세 쌍 모두 `label95`가 `WORSE`
  - `MIXED_DIRECTION`: 개별 label에 `BETTER`와 `WORSE`가 함께 있다.
  - `PARTIAL_SIGNAL`: `BETTER`가 있고 `WORSE`는 없다.
  - `NOT_ESTABLISHED`: 그 밖
  - 참고 비교는 판독에 들어가지 않는다.
- **진행**: `ALL_THREE_BETTER`일 때만 확인 평가(P5)로 간다. 그 밖의 label은 보존·보고하고 위임을 끝낸다.
- **정밀도 주의**
  - baseline D4의 95% 반폭은 평균 continuation delta로 약 0.04였다.
  - 같은 표본·예산이면 `NOT_ESTABLISHED`가 나올 가능성이 크다. 이것을 "차이 없음"으로 읽지 않는다.
  - 결과를 본 뒤 표본이나 예산을 늘리지 않는다.
- **승인 단계**: `approved.smoke` → smoke 실행 → `approved.full` → full 실행 1회(위임 상한)
- **테스트** (`tests/test_h7r_d1_d4.py`)
  - 계약: baseline 방법과 같다. 틀린 고정값(비교, 통계, factory, smoke 크기, H-dev, 승인 순서, 실험 고정, 소스, 키)은 거부한다.
  - 모델 연결: 커밋된 실행 기록에서 읽는다. 다른 실험·`NO_CANDIDATE`·미커밋 기록은 거부한다.
  - 진입 조건: 비교불가, `NOT_ESTABLISHED`, `HOLD`, 시간 gate 실패, 다른 쌍에서 잰 시간 gate는 거부한다.
  - 판독 규칙, factory v2 재생, sensitivity가 모든 비교를 덮는지 확인한다.
- **예상 시간**
  - baseline full은 모델 5개, 비교 4개로 158분이었다.
  - 이번에는 모델 7개, 비교 6개라 약 3.5–5시간으로 추정한다.
  - smoke는 baseline에서 1.2분이었다.

### 16.3 실행 절차 (데스크톱)

- 실행 순서
  1. 클라우드: P2·P3 고정 커밋, 그 뒤 evaluate 승인 커밋
  2. 데스크톱: full-3(H-dev) → v2-1(시간 gate) → d4-1(eligible manifest). 한 tmux에서 순서대로 돈다.
  3. 클라우드: 진입 조건 확인, 승인 커밋
  4. 데스크톱: d4-run smoke, 그 뒤 d4-run full
- d4-run은 실패하면 기록을 커밋하지 않는다.
  - 이유: 도구가 실행 정체성에 commit을 넣는다. 기록 커밋으로 HEAD가 바뀌면 같은 폴더에서 재개할 수 없다.
  - 실패 기록은 `runs/`에 보존된다.
  - 다음 중 하나를 정한다: 인프라 오류면 같은 블록으로 1회 재개(위임 재시도 규칙), 아니면 d4-F로 기록하고 멈춤.
  - 성공하면 그때까지의 모든 시도 로그를 함께 커밋한다.
- v2-1과 d4-1은 결과와 무관하게 기록을 커밋한다. 재개가 필요 없는 짧은 단계다.

v2-1. factory v2 시간 gate와 기록 커밋 (전체 규모 학습 기록 커밋 뒤)
   - 먼저 두 checkpoint가 커밋된 실행 기록의 선택 checkpoint인지 SHA로 확인한다.
   - 도구의 종료 코드: 0 PASS, 2 FAIL(기록됨), 그 밖 STOP

```bash
(
  set -euo pipefail
  cd ~/renju-h6
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src:$PWD/scripts"
  [ "$(git branch --show-current)" = feat/h7-c1 ] || { echo "STOP: wrong branch"; exit 1; }
  git pull --ff-only origin feat/h7-c1
  D=docs/h7r/r2/R2-D1-HEAD-WIDTH-1
  X="$D/factory-v2"
  F=runs/h7r/R2-D1-HEAD-WIDTH-1/full
  L=runs/h7r/R2-D1-HEAD-WIDTH-1/factory-v2/logs
  test -z "$(git status --porcelain --untracked-files=no)"
  python scripts/check_track_b_frozen.py
  test ! -e "$X/timing.json"
  for arm in narrow wide; do
    rec="$D/full/run-records/$arm-31001/d1_train.json"
    git ls-files --error-unmatch "$rec" >/dev/null
    python - "$rec" "$F/$arm-31001/best.pt" <<'PY'
import hashlib, json, sys
rec, pt = sys.argv[1:]
want = json.load(open(rec))['selected']['state']['checkpoint_sha256']
got = hashlib.sha256(open(pt, 'rb').read()).hexdigest()
assert got == want, f'STOP: {pt} {got[:12]} is not the selected checkpoint {want[:12]}'
print('linked', pt, got[:12])
PY
  done
  mkdir -p "$L"
  LOG="$L/timing-$(date +%Y%m%d-%H%M%S).log"
  [ ! -e "$LOG" ] || { echo "STOP: $LOG exists"; exit 1; }
  B="$(basename "$LOG" .log)"
  ts(){ while IFS= read -r line; do printf '%s %s\n' "$(date +%s.%N)" "$line"; done; }
  run_rc=0
  { echo __START__; rc=0; python -u scripts/h7r_factory_v2_timing.py --policy runs/h3_policy_64x4/best.pt \
      --narrow-value "$F/narrow-31001/best.pt" --wide-value "$F/wide-31001/best.pt" --device cuda \
      --output "$X/timing.json" 2>&1 || rc=$?
    echo "__EXIT__ $rc"; exit "$rc"; } | ts | tee "$LOG" || run_rc=$?
  expect=()
  shopt -s nullglob
  for f in "$L"/timing-*.log; do
    b="$(basename "$f" .log)"
    if git ls-files --error-unmatch "$X/timing-logs/$b.log" >/dev/null 2>&1; then continue; fi
    python scripts/h7r_d4_records.py copy-logs --resume --src-dir "$L" --pattern "$b.log" \
      --dest-dir "$X/timing-logs" --manifest "$X/timing-logs/$b.manifest.json"
    expect+=(--expect "$X/timing-logs/$b.log" --expect "$X/timing-logs/$b.manifest.json")
  done
  if [ -f "$X/timing.json" ]; then expect+=(--expect "$X/timing.json"); fi
  for e in "${expect[@]}"; do [ "$e" = --expect ] || git add "$e"; done
  python scripts/h7r_d4_records.py verify-staged "${expect[@]}"
  case "$run_rc" in 0) verdict=PASS ;; 2) verdict=FAIL ;; *) verdict="STOP (exit $run_rc)" ;; esac
  git commit -m "H7-R P2: factory v2 time gate $verdict on the linked seed-31001 pair (records only)"
  git push origin feat/h7-c1
  echo "__RECORDED__ $run_rc"
  exit "$run_rc"
)
```

d4-1. D4 eligible manifest와 기록 커밋 (P3 계약 커밋 뒤; 승인 불필요, 결과를 열지 않음)

```bash
(
  set -euo pipefail
  cd ~/renju-h6
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src:$PWD/scripts"
  [ "$(git branch --show-current)" = feat/h7-c1 ] || { echo "STOP: wrong branch"; exit 1; }
  git pull --ff-only origin feat/h7-c1
  D=docs/h7r/r2/R2-D1-HEAD-WIDTH-1
  X="$D/d4"
  L=runs/h7r/R2-D1-D4-1/logs
  test -z "$(git status --porcelain --untracked-files=no)"
  python scripts/check_track_b_frozen.py
  git ls-files --error-unmatch "$X/d1-d4-contract.json" >/dev/null
  SAMPLE="$(python -c "import json, sys; print(json.load(open(sys.argv[1]))['output']['sample'])" "$X/d1-d4-contract.json")"
  test ! -e "$SAMPLE"
  mkdir -p "$L"
  LOG="$L/prepare-$(date +%Y%m%d-%H%M%S).log"
  [ ! -e "$LOG" ] || { echo "STOP: $LOG exists"; exit 1; }
  B="$(basename "$LOG" .log)"
  ts(){ while IFS= read -r line; do printf '%s %s\n' "$(date +%s.%N)" "$line"; done; }
  run_rc=0
  { echo __START__; rc=0; python -u scripts/h7r_d1_d4.py prepare --cache runs/h7r/R1-HDEV-1/cache 2>&1 || rc=$?
    echo "__EXIT__ $rc"; exit "$rc"; } | ts | tee "$LOG" || run_rc=$?
  python scripts/h7r_d4_records.py copy-logs --src-dir "$L" --pattern "$B.log" \
    --dest-dir "$X/prepare-logs" --manifest "$X/prepare-logs/$B.manifest.json"
  expect=(--expect "$X/prepare-logs/$B.log" --expect "$X/prepare-logs/$B.manifest.json")
  if [ -f "$SAMPLE" ]; then expect+=(--expect "$SAMPLE"); fi
  for e in "${expect[@]}"; do [ "$e" = --expect ] || git add "$e"; done
  python scripts/h7r_d4_records.py verify-staged "${expect[@]}"
  if [ "$run_rc" -eq 0 ]; then verdict=PASS; else verdict="FAIL (exit $run_rc)"; fi
  git commit -m "H7-R D1 D4: eligible manifest $verdict (records only)"
  git push origin feat/h7-c1
  echo "__RECORDED__ $run_rc"
  exit "$run_rc"
)
```

d4-run. D4 실행과 기록 커밋 (`bash ~/d1-d4-run.sh smoke` 또는 `full`; 해당 `approved` 커밋 뒤)
   - 같은 명령을 다시 실행하면 같은 HEAD에서 이어서 한다(재개). 실패했을 때는 커밋하지 않는다.
   - 성공하면 `identity.json`, `summary.json`, 아직 커밋되지 않은 그 단계의 모든 로그를 커밋한다.

```bash
(
  set -euo pipefail
  MODE="${1:-}"
  case "$MODE" in smoke|full) ;; *) echo "STOP: mode must be smoke or full"; exit 1 ;; esac
  cd ~/renju-h6
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src:$PWD/scripts"
  [ "$(git branch --show-current)" = feat/h7-c1 ] || { echo "STOP: wrong branch"; exit 1; }
  git pull --ff-only origin feat/h7-c1
  D=docs/h7r/r2/R2-D1-HEAD-WIDTH-1
  X="$D/d4/$MODE"
  R="runs/h7r/R2-D1-D4-1/$MODE"
  L=runs/h7r/R2-D1-D4-1/logs
  F=runs/h7r/R2-D1-HEAD-WIDTH-1/full
  P=runs/h3_policy_64x4/best.pt
  test -z "$(git status --porcelain --untracked-files=no)"
  python scripts/check_track_b_frozen.py
  python - "$D/d4/d1-d4-contract.json" "$MODE" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
assert c['approved'][sys.argv[2]] is True, f'STOP: {sys.argv[2]} is not approved'
print('approved:', sys.argv[2])
PY
  test ! -e "$X/summary.json"
  test ! -e "$R/summary.json"
  models=(--model "c5=$P,runs/h7c/c5/value/best.pt")
  for s in 31001 31002 31003; do for arm in narrow wide; do models+=(--model "$arm$s=$P,$F/$arm-$s/best.pt"); done; done
  mkdir -p "$L"
  LOG="$L/$MODE-$(date +%Y%m%d-%H%M%S).log"
  [ ! -e "$LOG" ] || { echo "STOP: $LOG exists"; exit 1; }
  ts(){ while IFS= read -r line; do printf '%s %s\n' "$(date +%s.%N)" "$line"; done; }
  run_rc=0
  { echo __START__; rc=0; python -u scripts/h7r_d1_d4.py run --mode "$MODE" --cache runs/h7r/R1-HDEV-1/cache \
      "${models[@]}" --device cuda --workers 6 --rapfi-dir /mnt/c/engines/rapfi --out-dir "$R" 2>&1 || rc=$?
    echo "__EXIT__ $rc"; exit "$rc"; } | ts | tee "$LOG" || run_rc=$?
  if [ "$run_rc" -ne 0 ] || [ ! -f "$R/summary.json" ]; then
    echo "__NOT_RECORDED__ $run_rc: $R and $LOG are kept; resume with the same command or record with d4-F"
    exit 1
  fi
  expect=()
  mkdir -p "$X"
  shopt -s nullglob
  for f in "$L/$MODE"-*.log; do
    b="$(basename "$f" .log)"
    if git ls-files --error-unmatch "$X/run-logs/$b.log" >/dev/null 2>&1; then continue; fi
    python scripts/h7r_d4_records.py copy-logs --resume --src-dir "$L" --pattern "$b.log" \
      --dest-dir "$X/run-logs" --manifest "$X/run-logs/$b.manifest.json"
    expect+=(--expect "$X/run-logs/$b.log" --expect "$X/run-logs/$b.manifest.json")
  done
  for n in identity.json summary.json; do
    if git ls-files --error-unmatch "$X/$n" >/dev/null 2>&1; then cmp "$R/$n" "$X/$n"; continue; fi
    cp --no-clobber "$R/$n" "$X/$n"
    cmp "$R/$n" "$X/$n"
    expect+=(--expect "$X/$n")
  done
  for e in "${expect[@]}"; do [ "$e" = --expect ] || git add "$e"; done
  python scripts/h7r_d4_records.py verify-staged "${expect[@]}"
  reading="$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['reading'])" "$X/summary.json")"
  git commit -m "H7-R D1 D4: $MODE $reading (records only)"
  git push origin feat/h7-c1
  echo "__RECORDED__ $MODE $reading"
)
```

d4-F. D4 실패 기록 보존 (`bash ~/d1-d4-F.sh smoke` 또는 `full`; 재개하지 않기로 정했을 때만)

```bash
(
  set -euo pipefail
  MODE="${1:-}"
  case "$MODE" in smoke|full) ;; *) echo "STOP: mode must be smoke or full"; exit 1 ;; esac
  cd ~/renju-h6
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src:$PWD/scripts"
  [ "$(git branch --show-current)" = feat/h7-c1 ] || { echo "STOP: wrong branch"; exit 1; }
  D=docs/h7r/r2/R2-D1-HEAD-WIDTH-1
  X="$D/d4/$MODE"
  R="runs/h7r/R2-D1-D4-1/$MODE"
  L=runs/h7r/R2-D1-D4-1/logs
  echo __START__
  if pgrep -af 'scripts/h7r_d1_d4.py run'; then echo "STOP: a D4 runner is still alive"; exit 1; fi
  git reset -q -- "$D/d4"
  test -z "$(git status --porcelain --untracked-files=no)"
  expect=()
  shopt -s nullglob
  for f in "$L/$MODE"-*.log; do
    b="$(basename "$f" .log)"
    echo "== $f"; tail -n 3 "$f"
    if git ls-files --error-unmatch "$X/run-logs/$b.log" >/dev/null 2>&1; then continue; fi
    python scripts/h7r_d4_records.py copy-logs --resume --src-dir "$L" --pattern "$b.log" \
      --dest-dir "$X/run-logs" --manifest "$X/run-logs/$b.manifest.json"
    expect+=(--expect "$X/run-logs/$b.log" --expect "$X/run-logs/$b.manifest.json")
  done
  for n in identity.json summary.json; do
    [ -f "$R/$n" ] || continue
    if git ls-files --error-unmatch "$X/$n" >/dev/null 2>&1; then cmp "$R/$n" "$X/$n"; continue; fi
    mkdir -p "$X"
    cp --no-clobber "$R/$n" "$X/$n"
    cmp "$R/$n" "$X/$n"
    expect+=(--expect "$X/$n")
  done
  if [ "${#expect[@]}" -eq 0 ]; then
    git push origin HEAD:feat/h7-c1
    echo "nothing left to record; push checked"; echo "__EXIT__ 0"; exit 0
  fi
  for e in "${expect[@]}"; do [ "$e" = --expect ] || git add "$e"; done
  python scripts/h7r_d4_records.py verify-staged "${expect[@]}"
  git commit -m "H7-R D1 D4: $MODE attempt records, not completed (records only)"
  git push origin feat/h7-c1
  echo "__EXIT__ 0"
)
```

### 16.4 진행 상태

- P2: factory v2, 시간 gate 도구, 테스트 구현
- P3: D4 도구, 계약(`approved` 모두 false), 테스트, §16.3 실행 블록 4개
- 모두 전체 규모 H-dev evaluate 승인 전에 커밋한다. 모델 SHA는 학습 기록 커밋 뒤 실행할 때 연결한다.

### 16.5 데스크톱 자동 진행 (`scripts/h7r_d1_autopilot.py`, 지시문 `autopilot/desktop-agent-prompt.md`)

- §16.3의 진행 순서와 위임의 진행·중단·재시도 규칙을 코드로 고정했다. 데스크톱 에이전트는 이 도구가 말한 것만 한다.
- `next`: 커밋된 기록과 launch 기록으로 다음 action을 정한다.
  - action 종류: `wait` / `run` / `recover` / `approve` / `stop`
- 단계 순서: H-dev(full-3) → 시간 gate(v2-1) → eligible(d4-1) → smoke 승인 → smoke → full 승인 → full
- 멈추는 경우
  - H-dev가 `DEVELOPMENT_SIGNAL`이 아님
  - 시간 gate FAIL
  - D4 진입 조건 불충족
  - smoke가 재현되지 않음
  - full 승인 시점에 위임 상한까지 6시간 미만
  - 상한(2026-11-10 15:24:27 KST) 경과
  - full 결과가 나옴. `ALL_THREE_BETTER`면 P5 준비가 다음 단계다.
- 실패 처리
  - 재개: D4 실행이 도구의 `STOP:` 거부 없이 죽은 경우(충돌·강제 종료)만, 같은 명령·같은 HEAD로 1회
  - 기록 보존: 기록 커밋 전에 멈춘 단계는 보존 블록을 1회 돌린 뒤 멈춘다(H-dev는 full-3-F, D4는 d4-F).
  - 그 밖의 실패는 바로 멈춘다.
- `approve`: `next`가 그 승인을 가리킬 때만 한다.
  - 플래그 하나만 바꾸고, 실험 식별값이 그대로인지 확인한다.
  - 커밋한 계약을 로더가 다시 읽는다. 거부되면 push 전에 그 커밋을 되돌린다.
  - 상태 기록에 근거를 한 줄 남기고 push한다.
- 확인
  - `tests/test_h7r_d1_autopilot.py`: 단계 순서, 중단, 재개, 보존, 상한
  - 실제 저장소 사본(임시 origin)에서 `approve smoke` 커밋·push, 중복 승인 거부, tmux `launch`·`wait`를 실행했다.
  - 같은 사본에서, frozen 검사 실패로 도구를 실행하기 전에 멈춘 블록은 재개하지 않고 d4-F로 가는 것을 확인했다.
- 자동 진행 중에는 클라우드 세션이 push하지 않는다. D4 재개는 같은 HEAD가 필요하기 때문이다.

### 16.6 개정 A1: 시간 gate 기준 A의 factory (`d4/amendment-a1.json`, 2026-11-09)

- **발생**: 자동 진행의 두 번째 launch(시간 gate)가 첫 위치를 재기 전에 멈췄다.
  - 오류: `narrow-31001/best.pt: not a value head trained on the frozen H3 trunk`
  - 측정값은 없다. `timing.json`도, 측정 행도 없다.
- **원인 1: 도구 결함**
  - 기준 A를 factory v1(`tb_v3_net`)로 만들었다.
  - v1은 value metadata에 `parent_h3_checkpoint_sha256`가 있어야 받는다. D1 학습기는 이 필드를 쓰지 않는다.
  - 커밋된 `best.json` 6개로 확인했다: v1은 6개 모두 거부하고, v2의 metadata 검사는 6개 모두 통과한다.
  - 테스트는 parent 필드가 있는 대역 metadata만 썼기 때문에 이 결함을 놓쳤다.
- **원인 2: 기록 경로**
  - 실패 기록 경로 `factory-v2/logs/`가 `.gitignore`의 `logs/` 규칙에 걸려 커밋되지 않았다.
  - 블록 시뮬레이션 저장소의 `.gitignore`가 실제와 달라서 놓쳤다.
  - 다른 기록 경로는 실제 저장소에서 `git check-ignore`로 모두 확인했다.
- **수정**
  - 시간 gate 도구: A를 factory v2의 1채널 경로로 만든다. 이 경로는 v1의 `_build`이고, 같은 수와 진단 기록을 낸다(`tests/test_tb_v3_net_v2.py`).
  - 그대로인 것: suite, 위치 수, seed, 순서, warm-up, 문턱(평균 ×1.10, p95 ×1.20). A와 B는 value head 폭만 다르다.
  - v2-1 블록: 기록을 `factory-v2/timing-logs`에 둔다. 아직 커밋되지 않은 시간 gate 로그를 모두 함께 커밋하므로, 첫 시도의 실패 로그도 다음 실행에서 보존된다.
  - 첫 시도의 로그 사본은 `factory-v2/logs`(무시됨)에 커밋되지 않은 채 남는다. 지우지 않는다.
- **D4 계약**
  - 바뀐 것은 시간 gate 도구의 SHA 고정값 하나다. 실험 식별값이 `b6369b8e0fc9`에서 `032ede7b88af`로 바뀌었다.
  - 방법, 표본, 예산, 통계, 비교, 모델 연결, 진입 조건, 판독은 그대로다. 개정 기록이 전후 SHA를 남긴다.
- **판독과의 독립성**
  - 발견 시점에 전체 규모 H-dev 판독(`DEVELOPMENT_SIGNAL`, `9a419af`)이 열려 있었다.
  - 그러나 이 수정은 H-dev 판독이 무엇이든 같다. D4 값과 시간 gate 값은 아직 하나도 없었다.
- **재시도**
  - 위임 재시도 규칙에 따른다: 판독과 무관한 오류, 원인 확인 뒤, 1회.
  - `autopilot/retries.json`이 실패한 launch 2를 지정한다. 자동 진행 도구는 이 기록이 있을 때만 그 단계를 한 번 더 돈다.

### 16.7 개정 A2: eligible manifest가 묶는 계약 SHA (`d4/amendment-a2.json`, 2026-11-10)

- **발생**: 첫 D4 smoke(launch 5)가 preflight에서 멈췄다. replay도 branch도 시작하지 않았다.
  - 오류: `STOP: the committed eligible manifest differs from the sealed H-dev`
- **원인: 도구 설계 결함**
  - eligible manifest는 prepare 당시 계약 파일의 SHA를 기록한다(`5be35de05548`, 승인 없음).
  - 승인 플래그는 같은 파일 안에 있다. 그래서 `approved.smoke` 커밋(`2897a44`)이 파일 SHA를 바꿨다.
  - 승인된 실행은 바뀐 SHA로 manifest를 다시 만들어 비교하므로, 승인된 계약에서는 항상 불일치가 났다.
  - baseline D4는 승인을 별도 파일에 두었기 때문에 이 문제가 없었다.
  - 테스트도 블록 시뮬레이션도 "prepare → 승인 → run" 순서를 실제 도구로 거치지 않아서 놓쳤다.
- **수정**
  - 도구: prepare와 run이 `contract_base_sha256`을 쓴다. 승인을 모두 false로 둔 계약의 SHA다.
    - 아무것도 승인되지 않았을 때는 파일 SHA와 같다.
    - 그 뒤의 승인으로는 바뀌지 않는다.
  - 새 테스트: 실제 `prepare`, 승인, `run`을 차례로 돌린다. 수정 전 코드로는 데스크톱과 같은 오류로 실패하고, 수정 뒤에는 통과한다.
- **D4 계약**
  - 도구 SHA 고정값이 바뀌었다. 실험 식별값은 `032ede7b88af`에서 `ee06faf80d9b`로 바뀌었다.
  - 승인은 둘 다 false로 되돌렸다. 앞의 smoke 승인은 A1 실험에 대한 것이었다.
  - 표본 경로는 `d4/eligible-a2.json`이다. A1의 `d4/eligible.json`은 그대로 둔다.
- **그대로인 것**: 방법, 표본 규칙, 예산, 통계, 비교, 모델 연결, 진입 조건, 판독
  - 새 manifest는 A1 manifest와 `contract_sha256` 말고는 같아야 한다.
  - 자동 진행 도구가 이것을 확인하고, 다르면 멈춘다.
- **판독과의 독립성**
  - H-dev(`DEVELOPMENT_SIGNAL`)와 시간 gate(PASS)는 열려 있었다.
  - 이 수정은 그 결과와 무관하게 같다. D4 값은 하나도 없었다.
- **재시도**: `autopilot/retries.json`이 두 launch를 지정한다.
  - eligible(launch 4): 실패가 아니라 A1 계약에 묶인 manifest라서 다시 만든다.
  - smoke(launch 5): preflight에서 멈췄고 실행 폴더가 생기지 않았다.
  - smoke는 승인이 되돌려졌으므로, 자동 진행 도구가 gate를 확인해 다시 승인한 뒤 재시도한다.

### 16.8 자동 진행 도구 보완 (검토 지적 4건, 2026-11-10)

`389b88e` 검토에서 자동 진행을 다시 멈출 수 있는 실행·복구 관리 문제 4건이 재현됐다. 측정 결과나 계약에는 영향이 없다.

1. **실행 중 pull**
   - 문제: `next`가 실행 여부보다 pull을 먼저 했다. 원격에 커밋이 생기면 도는 블록의 HEAD가 바뀌었다. D4는 끝에서 HEAD 변경 검사에 걸리고, 같은 identity로 재개할 수도 없게 된다.
   - 수정: 블록이 도는 동안 `next`는 pull도 트리 검사도 하지 않고 `wait`를 낸다(`pulled: false`). pull은 블록이 없고 트리가 깨끗할 때만 한다.
2. **staged 기록과 복구**
   - 문제: 블록이 `git add` 뒤 커밋 전에 멈추면, clean 검사 때문에 복구 판단에 도달하지 못했다.
   - 수정: 기록 폴더(`full/`, `factory-v2/`, `d4/`) 아래에 새로 staged된 파일만 있으면 `tree: records`로 본다.
     - 이 상태에서는 `recover` / `stop` / `wait` / `interrupt`만 허용하고, pull하지 않는다.
     - 보존 블록은 먼저 그 폴더를 unstage한 뒤 다시 커밋한다.
     - 그 밖의 변경(추적 파일 수정, 다른 경로)은 계속 거부한다.
3. **상한**
   - 문제 1: 이미 승인된 full은 상한 직전에도 실행을 허용했다.
   - 문제 2: 실행 중 상한을 넘으면 계속 `wait`를 냈다.
   - 수정 1: 모든 실행·재개·재시도·승인에 단계별 최소 남은 시간을 요구한다(full 6시간, smoke 0.5시간 등).
   - 수정 2: 상한을 넘은 채 블록이 돌면 `interrupt`를 낸다. `interrupt` 명령이 tmux에 Ctrl-C를 보내고 원장에 표시한다.
   - 중단된 실행은 재개하지 않는다. 기록 보존만 하고 위임을 끝낸다(`final: true`).
4. **tmux 시작 실패**
   - 문제: 원장을 먼저 쓰기 때문에, tmux가 실패하면 출력·종료 표시 없는 원장만 남아 다음 호출이 멈췄다.
   - 수정: launch 뒤에 `started` 또는 `start_failed`를 표시한다. `start_failed`(또는 표시도 출력 파일도 없는 launch)는 시도로 세지 않는다.
- **추가로 고친 것**
  - `BLOCK EXIT` 없이 끝난 블록(강제 종료, 세션 소실)은 이제 실패로 처리해 보존 블록으로 간다. 전에는 바로 멈췄다.
  - 보존 블록이 실패하면 "보존됨"이 아니라 "보존 실패"로 멈춘다.
- **확인**: `tests/test_h7r_d1_autopilot.py` 14개
  - 실제 임시 git 원격과 실제 tmux로 확인한 것
    - 도는 블록이 있을 때 원격 커밋이 들어와도 HEAD가 그대로이고, 블록이 끝난 뒤에야 pull한다.
    - staged 기록만 있으면 `recover`이고 pull하지 않는다. 다른 staged 파일이나 계약 수정은 거부한다.
    - tmux 시작 실패는 시도로 세지 않고, 다음 launch가 정상으로 돈다.
  - 시각을 고정해 확인한 것: 상한 직전 full 거부, 실행 중 상한 도달 시 `interrupt`, 중단 뒤 재개 없이 보존하고 최종 중단, 보존 실패 보고
- **검증 범위**: 실제 CUDA·Rapfi 실행은 데스크톱에서만 확인된다.

## 17. 전체 규모 결과와 위임 종료 (2026-11-10)

### 17.1 결과

| 단계 | 기록 | 결과 |
|---|---|---|
| 학습·fit | `d8c0d10` | 6회 모두 `CANDIDATE`, 비교가능, `WIDTH_EFFECT` (wide − narrow: 31001 −0.0231, 31002 −0.0278, 31003 −0.0210) |
| H-dev | `9a419af` | `DEVELOPMENT_SIGNAL`. 개선폭 0.00884 / 0.00777 / 0.01073이 모두 ε 0.00195를 넘고 95% 상한이 0 아래다. HOLD 없음 |
| 시간 gate | `b939f2c` | PASS. 144위치, 평균 ×1.0026, p95 ×1.0038, 같은 수 96.5% (개정 A1) |
| D4 smoke | `1b4a274` | `SMOKE`, c5가 actor의 수를 재현 (개정 A2) |
| D4 full | `c859f35` | **`NOT_ESTABLISHED`**. 279.8분, branch 1,988개(resolved 1,965 / structural 23 / unresolved 0), c5 actor 재현 |

D4 주 비교 (실험 `ee06faf80d9b`, 연결 모델은 전체 규모 실행 기록의 T-val 선택 checkpoint)

| 주 비교 | 관측 | 평균 delta | 95% 구간 | better / worse / equal | label95 | 개별 label |
|---|---|---|---|---|---|---|
| wide31001 : narrow31001 | 379 | −0.0026 | [−0.0552, +0.0475] | 55 / 56 / 268 | INCONCLUSIVE | INCONCLUSIVE |
| wide31002 : narrow31002 | 520 | +0.0067 | [−0.0321, +0.0444] | 72 / 70 / 378 | INCONCLUSIVE | INCONCLUSIVE |
| wide31003 : narrow31003 | 459 | −0.0065 | [−0.0485, +0.0340] | 53 / 56 / 350 | INCONCLUSIVE | INCONCLUSIVE |

- 참고 비교(wide : c5)는 판독에 들어가지 않는다.
  - 평균 delta는 −0.0160 / −0.0142 / −0.0137이고, 95% 구간은 모두 0을 포함한다.
  - 보조 통계 game macro에서 wide31001 : c5만 구간이 0을 포함하지 않는다([−0.147, −0.005]).
  - narrow : c5는 이 D4의 비교가 아니다. 그래서 이 음의 방향이 head 폭 때문인지 학습 자료·절차 때문인지 구분할 수 없다.
- 판독 규칙을 커밋된 비교값에 다시 적용해도 `NOT_ESTABLISHED`가 나온다. 연결된 모델 SHA와 eligible manifest(A1과 결속 SHA만 다름)도 기록과 같다.

### 17.2 읽기

- 진단 계획(R2-DIAG-1 §8, D4 행)에 따라 D1은 **"예측-선택 연결 미입증"**으로 기록한다.
  - 넓은 head의 H-dev 예측 개선은 세 seed 모두 ε을 넘었다.
  - 그러나 같은 seed의 narrow 대비 수 선택 개선은 입증되지 않았다.
  - 이것은 "차이 없음"이나 동등성 판정이 아니다. 목표 불일치를 확정한 것도 아니다.
- 기술 통계 (판독이 아니라 다음 설계를 위한 참고)
  - wide와 narrow가 다른 수를 고른 위치는 eligible 15,985 ply 가운데 2.4–3.3%다(379 / 520 / 459). 모두 tree route에서 나왔고, tree ply 7,252 기준으로는 5.2–7.2%다.
  - 수가 달라진 위치에서 better와 worse의 비율은 거의 같다(better 비율 0.486–0.507).
  - (정정 2026-11-10) 처음에는 "수가 갈린 위치의 95% 상한을 전체 위치에 펼치면 위치당 0.001 남짓"이라고 적었다.
    - 이 값은 불일치율에 조건부 구간 상한을 곱한 기술적 환산값일 뿐이다.
    - 전체 국면의 신뢰상한이 아니고, 승률이나 기력 향상의 상한도 아니다.
  - 낮은 불일치율만으로 검출력이 부족하다고 결론 내리지 않는다. 정밀도는 불일치 관측 수, 결과가 갈린 관측 수, 게임 내 상관에 달려 있다.
    - 이번에는 구간이 넓으면서 점추정도 0 근처다. 검출력을 높이면 개선이 나올 것이라는 근거는 없다.
    - 정밀도와 비용 검토는 `R2-D4-PRECISION-1-draft.md`에서 한다.
- baseline D4(c8 recipe)에 이어 두 번째 `NOT_ESTABLISHED`다. 두 경우 모두 H-dev 신호는 있었고 수 선택 신호는 없었다.

### 17.3 위임 R2-D1-AUTOPILOT-1 종료

- 진행 조건(`ALL_THREE_BETTER`)을 충족하지 못했다. 위임 규칙대로 P5(확인 평가)와 P6(R4 준비)는 진행하지 않고 위임을 끝낸다.
- 상한: 시작 2026-11-09 15:24:27 KST, D4 full 기록 커밋 2026-11-10 05:14 KST로 상한(15:24:27) 안에 끝났다.
- 쓴 횟수: D4 본 평가 1회(상한 1회), 확인 평가 0세트
- 재시도와 개정: 모두 판독 전에 멈춘 단계에서, 원인을 확인한 뒤 1회씩 했다(`autopilot/retries.json`).
  - 개정 A1: 시간 gate 기준 factory (§16.6)
  - 개정 A2: eligible manifest가 묶는 계약 SHA (§16.7)
  - 자동 진행 도구 보완 (§16.8)
  - 세 번 모두 결과(H-dev, 시간 gate)가 열린 뒤 계약 SHA를 다시 고정했다. 방법·표본·예산·판독은 바뀌지 않았고, 각 개정 기록에 판독과의 독립성 근거를 남겼다.
- 바뀌지 않은 것: champion(c5), canonical, 웹, `tb_v3_net`(factory v1). factory v2는 연구 도구로만 남는다.
- 다음 진단의 선택은 이 위임의 범위 밖이다. 사용자가 정한다.
