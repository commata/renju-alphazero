# R2-D1-HEAD-WIDTH-1 — value head 폭 진단 계약 초안

- 작성일 2026-11-06. 상태: **draft / 미승인 / 실행 불가**
- 상위 설계: `R2-DIAG-1-draft.md` §7(판독 규칙, ε, `DEVELOPMENT_SIGNAL`, 보류)과 §8(D1 행)
- 직전 진단: `R2-D3-NO-OUTCOME-1-draft.md` §9(D3 `NOT_ESTABLISHED`)
- 이 문서의 수치 중 "제안값"은 확정 전에 정할 값이다(§9).

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

## 9. 확정 전 정할 것

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
