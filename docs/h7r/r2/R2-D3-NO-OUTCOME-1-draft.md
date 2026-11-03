# R2-D3-NO-OUTCOME-1 — cycle outcome 기여 제거 대조 계약 초안

- 작성일 2026-11-03. 상태: **draft / 미승인 / 실행 불가**
- 상위 설계: `R2-DIAG-1-draft.md` §7(판독 규칙, ε, `DEVELOPMENT_SIGNAL`)과 §8(D3 행)
- 사전 구성 확인 도구: `scripts/h7r_d3_prep.py`(읽기 전용, CPU), 계산 `src/hybrid/h7r_d3.py`, 테스트 `tests/test_h7r_d3.py`
- 데스크톱 명령: `docs/h7-rl-plan.md` §9.26

## 0. 성격과 바꾸지 않는 것

- **진단이다.** cycle outcome 행의 학습 기여를 없앴을 때 무엇을 얻고 무엇을 잃는지 측정한다.
- label 잡음을 원인으로 확정하는 실험이 아니다. outcome이 해롭다는 근거도 아니고, D1·D2를 배제하지도 않는다.
- 기존 모델 비교(R2-0A/0B, D4, X1)로 이 ablation의 결과를 미리 확정하지 않는다. outcome 성분이 나빠질 수 있다는 것은 가능한 예상일 뿐이다. 학습 간섭이 줄어 다른 결과가 나올 수도 있다.
- 바꾸지 않는 것
  - D4 판독(`NOT_ESTABLISHED`), c5 유지, V9 보류
  - R2의 주 지표, ε, `DEVELOPMENT_SIGNAL` 규칙, D4 진입 조건
- proof 전용 판독과 D4 진입 규칙 변경은 이 계약에 넣지 않는다. 이 진단에서 무엇을 배웠는지 본 뒤 별도 연구 질문으로 정한다.

## 1. 질문

같은 parent·자료·학습 예산·seed에서 cycle outcome 행의 손실 기여만 0으로 하면, H-dev의 기존 주 지표와 그 구성(outcome / proof, ply, 차례)이 같은 seed의 기준 recipe 대비 어떻게 달라지는가?

## 2. 조건과 직접 control

| 항목 | 기준 recipe(직접 control) | D3 조건 |
|---|---|---|
| parent | c5 | 같음 |
| 자료 | c8 cycle 자료(self-play seed 8808)와 split, H6 replay | 같음 |
| config | c8 config(batch 1024, 3 epoch, lr 2e-4, D4 augmentation, 평가 10회, H6 guard 1.05) | 같음 |
| 학습 seed | 31001 / 31002 / 31003 | 같음 |
| cycle outcome 행 | 가중치 0.1로 학습 | **손실 기여 0** |
| cycle proof 행, H6 replay | 학습 | 같음 |
| checkpoint 선택 | 기존 T-val objective와 H6 guard | **같음**(아래 2.2) |

- 직접 control은 R2-0B의 같은 seed 실행 세 개다(`docs/h7r/r2/R2-DIAG-1/r2_0b/runs/seed-3100x/`). 이미 학습돼 있으므로 다시 학습하지 않는다. 기존 기록과의 호환성(아래 2.3)이 확인될 때만 control로 쓴다.
- 새 self-play는 만들지 않는다.

### 2.1 손실

    batch loss = Σ_i k_i w_i (v_i − t_i)² / Σ_i w_i
    k_i = 0 (cycle outcome 행), 1 (cycle proof 행, replay 행)

- 분모는 batch의 **원래 가중치 합**이다. 제거된 기여를 proof나 replay 행에 재분배하지 않는다.
- 이 손실 형태는 기존 H7C-NMASK-1 경로(`h7c_train.masked_wmse`)와 같다.
- cycle 쪽 kept weight가 0인 batch는 cycle 항 0, replay 항 유지로 처리한다. 그 수를 기록한다(기존 `zero_cycle_weight_batches`).
- 실제 gradient 기여량이 함께 줄어드는 처방이다. label 잡음만 분리한 효과라고 부르지 않는다.
- 행 순서·replay 추출·augmentation은 k와 무관하다. 그래서 같은 seed의 자료 배정 지문(`h7r_r2.data_assignment`)이 control과 같아야 한다. 도구로 확인한다.

### 2.2 checkpoint 선택

- control과 **같은** 선택 규칙을 쓴다.
  - objective = (cycle val 가중 MSE, outcome 0.1·proof 1.0 그대로) 와 (H6 val 가중 MSE)의 평균
  - eligible = H6 val 가중 MSE ≤ 1.05 × parent
  - best = eligible 평가 중 objective가 가장 낮은 것
- 기존 NMASK 경로는 선택 지표도 masked cycle val로 바꾼다. D3은 그렇게 하지 않는다. 학습 마스킹으로 선택 기준까지 바꾸면 두 변수가 함께 움직이기 때문이다.
- masked cycle val 값은 측정으로만 함께 기록한다. 선택에는 쓰지 않는다.
- 결과: D3 모델은 학습하지 않은 outcome 행의 val 손실을 일부 포함한 objective로 선택된다. 이 점을 결과 해석에 적는다.

### 2.3 구현과 호환성 (확정 전 할 일)

- 현재 trainer에는 "마스킹된 손실 + 원래 선택 지표" 조합이 없다. `h7c_train.train_value`에 선택 지표 옵션을 하나 추가하는 안을 확정 단계에서 구현한다.
  - 옵션을 쓰지 않으면 코드 경로와 모든 수가 기존 recipe와 같아야 한다.
  - 같은 seed의 control 실행을 다시 학습하지 않으므로, **변경 후 코드로 기준 recipe를 재현하는 점검**(R2-0B 재현 seed 20261025 방식)이 필요한지 확정 단계에서 정한다.
- control 호환성 검사
  - parent·cycle·replay labels와 cache split SHA, config(학습 seed 외 동일), 학습 행 수와 step 수
  - 자료 배정 지문이 같은 seed의 control과 같은지
  - control의 기록 SHA(`r2_train.json`, `record.json`, `config.json`)는 이미 D4 계약 sources에 고정돼 있다.
- 실행 시간 한도, 출력 경로, 실패 처리(NO_CANDIDATE 보존)를 계약 JSON에 적는다.

## 3. 판정

R2 설계 §7 규칙을 **그대로** 쓴다. 조건 C = D3, control K = 같은 seed의 기준 recipe(R2-0B 실행).

    I_s = L(K_s) − L(C_s)        (L = H-dev 주 지표: D4 예측, network 행, game macro; 양수가 개선)

- `DEVELOPMENT_SIGNAL`: 세 seed 모두 I_s > ε(0.0019461168062766864), 그리고 세 seed 모두 paired Δ(C_s:K_s)의 95% 상한 < 0
- 그 밖의 결과는 모두 `NOT_ESTABLISHED`다.
- control의 seed 범위(세 값의 max − min, R2-0B 0.00067)도 보고한다. 그것이 ε보다 크면 판독을 보류한다(§7).
- c5 대비 결과는 참고값으로 함께 보고한다(§7). 판독에는 쓰지 않는다.
- 구성 지표(outcome / proof 성분, ply, 차례)에는 주 지표의 ε를 적용하지 않는다. 같은 bootstrap 구간과 함께 기술한다.
- H-dev는 학습, early stopping, checkpoint 선택에 쓰지 않는다.

## 4. 사전 구성 확인 (계약 확정 전, 읽기 전용)

`scripts/h7r_d3_prep.py`가 기존 기록만으로 다음을 낸다. 결과는 계약을 확정하기 전에 이 문서에 기록한다. 그 숫자를 보고 판정 규칙을 바꾸지 않는다.

1. **H-dev 구성**: ply 구간 × label 종류(outcome, proof:vcf / unstoppable_four / forced_loss) × route(network / exact_leaf)의 행·게임·가중치
2. **주 지표의 게임별 가중치 몫**: 게임 g의 network 행 분모 den_g 중 각 종류(그리고 종류 × ply)가 차지하는 몫의 게임 평균. 전체 행을 합친 가중치 몫과는 다른 값이다.
3. **기존 주 지표 차이의 가산 분해**: seed 모델 대 c5·c8, c8 대 c5
   - 게임별 원래 분모를 유지한다: Δmacro = Σ_p mean_g [Δnum_g^p / den_g]
   - 부분의 합은 전체 차이와 정확히 같다(도구가 확인). bootstrap 회차마다도 같다(공통 게임 표본·공통 분모).
   - 이 기여도는 그 종류만의 macro(`network/outcome` 성분)와 다르다. 둘을 함께 보고한다.
   - 재계산한 전체와 성분이 커밋된 R2-0B paired 값과 같아야 한다(c8:c5는 R2-0A와 1e-5 이내). 다르면 멈춘다.
   - 구간은 기술용이다. ε를 부분에 적용하지 않는다. 기존 모델의 분해로 D3의 결과를 예측하지 않는다.
- 입력: 고정 R2 계약, R2-0B epsilon 기록(SHA 고정), 그 기록이 가리키는 `seed_predictions.pt`, R2-0A 기록, H-dev cache. 하나라도 없거나 다르면 멈춘다.

## 5. 보고 (실행 시)

- seed별 I_s와 paired 구간, `DEVELOPMENT_SIGNAL` 판정, c5 대비 참고값
- 주 지표의 종류별·종류 × ply 가산 기여도(4절과 같은 분해), outcome / proof 성분 macro, 차례별, pooled
- 학습 기록: 선택 step, objective 곡선, masked cycle val(측정만), H6 guard, cycle weight / kept weight, cycle 항이 0인 batch 수
- control과의 자료 배정 지문 비교, 출처 SHA, 소요 시간

## 6. 후속 행동

- `DEVELOPMENT_SIGNAL`이 아니면 D4로 자동 진행하지 않는다.
- proof 성분만 좋아졌다는 이유로 결과를 본 뒤 D4 진입 기준을 바꾸지 않는다.
- `DEVELOPMENT_SIGNAL`이어도 D4는 별도 계약·승인으로 정한다.
- NO_CANDIDATE나 실패 seed는 버리거나 다른 seed로 바꾸지 않는다. 기록을 보존하고 원인을 보고한다.
- 결과가 무엇이든 이 진단에서 배운 것을 기록한 뒤 다음 질문(proof 전용 판독, D1 등)을 따로 정한다.

## 7. 확정 전 정할 것

1. 4절 결과의 기록과 검토
2. trainer 선택 지표 옵션의 구현과 동치성 점검 방식(2.3), 재현 점검 필요 여부
3. 계약 JSON: 조건, control 경로와 SHA, 출처 SHA, 실행 시간 한도, 출력 경로
4. 실행 승인 단계: 학습 3회 → H-dev 평가와 판독. 각 단계는 따로 승인받는다.

## 8. 예상 비용 (계획값, 측정 아님)

- 학습: R2-0B seed 실행 4.2–4.5분 × 3회 정도
- H-dev 평가: R2-0B epsilon 단계와 비슷한 규모
- 4절 사전 확인: CPU, 수 분 이내 예상
