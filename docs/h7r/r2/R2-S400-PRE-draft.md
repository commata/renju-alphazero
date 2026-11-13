# R2-S400-PRE: S400 정책 목표 사전 측정 계획 (초안, 승인 전)

- 상태: 도구·테스트·계약(**미승인**, `approved` 모두 false) 준비 완료. 측정은 실행하지 않았다. 실행 여부는 이 계획서를 검토한 뒤 결정한다.
- 도구 `scripts/h7r_s400_pre.py`, 테스트 `tests/test_h7r_s400_pre.py`, 계약 `docs/h7r/r2/R2-S400-PRE/s400-pre-contract.json`(초안 `s400-pre-contract.draft.json`), 실험 SHA `e8ca3049a4ee…`(도구 보완 뒤 다시 고정. 처음 고정본 `3e1a863799f8…`은 실행되지 않았다)
- 이번 측정에서 하지 않는 것: 학습, Arena, champion 변경, 기존 기록 수정
- 배경
  - R2-P1-POLICY-REPLAY-A1(`faeb72b0f625`)은 P NOT_ESTABLISHED, D BETTER_THAN_CONTROL로 끝났다.
  - 다음 질문은, 정책 보존을 유지한 채 학습 목표의 탐색 품질을 S200에서 S400으로 올렸을 때 S50으로 두는 모델이 그 이득을 배울 수 있는가이다.
- 이 측정이 답하는 것은 그 앞 단계다: **S400 목표가 S200 목표와 실제로 얼마나 다르고, 전체 재탐색에 얼마나 드는가.**

## 1. 근거 기록

| 항목 | 값 | 출처 |
|---|---|---|
| c2 학습 목표 | S50 self-play 국면 20,910개를 S200(일반 200 / 전술 400)으로 다시 탐색, mode `restored` | `docs/h7c/c2/retarget_s200.json` |
| S200 재탐색 비용 | 8,857초(약 2.5시간), 국면당 평균 탐색 2.51초 | 같음 |
| S50 → S200 최선수 변경 | 24.4% | 같음 |
| 수당 탐색 시간 (budget sweep) | S200 2.64초, S400 4.46초 (약 1.69배) | `docs/h7c/e2/sb_c5_saturation.json` |
| S400 대 S50 대국 | 50쌍 sweep: +9.5%p, 95% 구간 [0.0, +18.5], NOT_DETECTED. 150쌍 포화 확인: +7.7%p, [+2.5, +12.7], STRONGER | 같음 |
| S200 목표로 학습한 정책 | c2 방식(A1의 C)은 champion 대비 −4.5%p, 정책 보존(A1의 T)은 −1.3%p | A1 `judge.json` |

- 탐색 자체는 S400이 S50보다 강했다(150쌍 기준). 그러나 S200 목표를 증류한 정책도 champion을 넘지 못했다. 그래서 목표 품질을 올리면 정책이 그 이득을 흡수하는지는 열려 있는 질문이다.

## 2. 고정하는 것

**입력** (c2 재탐색과 같음)
- c2 self-play: `selfplay_jsonl_sha256` `48b65dd7ed30…`(cycle 2, run seed 8802)
- 탐색 모델: c2 self-play를 둔 parent 쌍. H3 policy `efee483284df…`와 당시 parent value `a0fcccc401a8…`
  - c5 value가 아니다. `h7c_retarget.check_engine`은 self-play의 모델과 다른 checkpoint를 거부한다.
- 엔진: 고정된 TB-V3(engine hash `d0ef0da6b8b9…`), runtime fast5 cuda(`7c727150922d…`), mode `restored`
- 비교 기준: c2의 S200 재탐색 기록. 데스크톱 `runs/`의 jsonl이며, SHA는 `retarget_s200.json`의 `jsonl_sha256`과 같아야 한다.

**예산**
- S200 = factor 4(일반 200 / 전술 400), S400 = factor 8(일반 400 / 전술 800). `h7c_retarget.budget(factor)` 규칙을 쓴다.
- 이 두 예산은 **측정 계약에 직접 고정한다.**
  - 기존 `h7c_retarget.py run`은 이전 cycle의 결정(`approved_budget`)에서 factor를 읽는다. c2에 그대로 쓰면 승인된 S200만 나온다.
  - 기존 `check`(G0)는 앞쪽 N개 게임을 actor(S50)와 비교하는 검사다.
  - 그래서 측정은 별도 도구가 하고, 기존 `h7c_retarget.py`, `hybrid/h7c_retarget.py`, `hybrid/h7c_targets.py`는 바꾸지 않는다. 측정 도구는 그 순수 규칙(적격 국면, coverage key, 정책 목표, 증명된 패배수의 합집합)을 그대로 가져다 쓰고, 측정 계약이 그 파일들의 SHA를 고정한다.

**표본**
- 단위: c2 **train split에서 label 생성에 남긴 게임**. split은 c2와 같은 규칙(`split_for`, salt `h7c-v1`, val 5%)으로 정한다. c2 label 보고서와 당시 probe 입력을 고정해 게임 제외와 행 마스크를 재현한다.
- 크기: 100게임. 약 2,090국면은 전체 재탐색 자료의 게임당 평균으로 얻은 계획값이다. 실제 표본 국면 수는 고정할 coverage 목록에서 센다.
- 선택 규칙: train split 게임 id마다 `sha256("r2-s400-pre-v1:" + game_id)`를 계산해 오름차순 앞 100개를 고른다. 무작위 seed가 없는 결정적 규칙이고, 계약에 적는다. 선택된 게임 id 목록과 coverage key 목록의 SHA를 계약에 고정한다.
- 게임 단위로 뽑는 이유: `restored` 모드가 게임 단위로 탐색을 복원하고, 같은 게임의 국면끼리는 서로 상관되기 때문이다. 구간도 게임 단위로 resample한다.
- 검색은 각 표본 게임의 actor `target.kind == 'visits'` 국면을 모두 포함한다. 실제 학습 목표 집계에는 c2의 행 마스크를 적용하며, 마스크 행은 건수를 따로 기록한다.

## 3. 실행 내용 (승인 뒤)

1. **입력 대조**: self-play jsonl, 모델, 엔진·runtime, c2 S200 기록의 SHA를 대조한다. 표본 게임 목록이 계약과 같은지도 확인한다.
2. **S200 재현 검사 (두 단계)**
   - 먼저 고정된 100게임 가운데 표본 순서상 처음 5게임을 S200으로 다시 탐색해 c2 기록과 대조한다. 하나라도 다르면 나머지를 돌리지 않고 멈춘다(`S200_EARLY_NOT_REPRODUCED`).
   - 통과하면 같은 표본의 나머지 95게임을 이어서 탐색하고, 표본 전체를 다시 대조한다. 다르면 S400을 돌리지 않고 멈춘다(`S200_NOT_REPRODUCED`).
   - 대조 규칙은 기존 G0 규칙이다: ply, 색, seed, 국면 해시, route, 엔진 수, 방문 수, 증명된 패배수, E4-v2 단계, tree budget, cooldown 복원은 정확히 같아야 한다. 평균값(mean value)만 `h7c_retarget.Q_TOLERANCE`(1e-5) 이내를 요구한다. 기록이 평균값을 소수 6자리로 반올림해 저장하기 때문이며, 새로 정한 허용 오차가 아니다.
   - 멈춰도 결과 기록(`measure.json`)과 로그는 남긴다. 같은 국면에서 지금 환경이 c2 재탐색을 재현하지 못한다는 뜻이므로, 기준을 낮추지 않는다.
3. **S400 실행**: S200 전체 재현을 통과한 뒤에만, 같은 표본을 같은 모델·복원 방식에서 S400으로 탐색한다.
   - route와 E4-v2 단계는 예산과 무관하므로 S200과 같아야 한다. 다르면 오류로 기록한다(`h7c_retarget`의 규칙과 같음).
4. **참고 기준**: 같은 국면의 actor(S50) 방문 분포를 self-play 기록에서 읽는다. 이를 이용해 S50 → S200 변화량도 같은 지표로 낸다. c2가 실제로 학습한 변화의 크기와 비교하기 위해서다.

## 4. 지표 (국면마다 계산하고 게임 단위로 집계)

**원시 방문 분포** (근 방문 수를 합으로 정규화. 아무것도 빼지 않음)
- 최다 방문 수가 바뀐 비율
- total variation 거리, Jensen–Shannon 발산(밑 2)
- 엔트로피, 최다 방문 수의 비중, 방문한 자식 수
- 증명된 패배수 집합의 비교 건수(같음 / S200에만 추가 / S400에만 추가 / 양쪽에 각각 고유한 수가 있음)

**실제 학습 목표** (`h7c_retarget.retarget_target`과 c2 label 규칙)
- 두 예산에서 모두 학습 목표가 있는 마스크되지 않은 행에서 분포를 짝지어 비교한다. 증명된 패배수를 뺀 뒤 질량이 모두 사라지면 c2처럼 행을 제외한다. 두 예산 각각의 제외 건수와 S200에서만 또는 S400에서만 목표가 남은 건수를 별도로 기록한다. 참고 S50 → S200 비교도 같은 방식으로 분모를 밝힌다.
- 목표 top-1이 바뀐 비율, total variation 거리, Jensen–Shannon 발산
- 목표 엔트로피
- H3 정책과의 거리: KL(목표 ‖ H3). 학습과 같은 입력 인코딩·합법수 마스크를 써서 목표가 H3 정책과 얼마나 떨어져 있는지 잰다.
- 원시 방문 분포와 목표를 따로 내는 이유: 증명된 패배수를 빼면 두 분포의 차이가 커지거나 작아질 수 있다. 학습에 들어가는 것은 목표다.

**분모와 공통 국면**
- 표는 두 가지로 낸다.
  - 짝별 표: 각 짝(S50 → S200, S200 → S400)의 두 예산 모두에서 목표가 남는 마스크되지 않은 행
  - **공통 국면 표**: S50, S200, S400 세 예산 모두에서 목표가 남는 마스크되지 않은 행. 두 짝을 같은 국면에서 읽을 수 있으므로, 분모 차이와 분포 변화의 크기를 구분할 수 있다.
- 함께 적는 건수: 표본 국면 수, 마스크된 행, c2 행이 없는 국면(S200 목표가 없어 c2가 행을 만들지 않음. 마스크 여부를 알 수 없음), 예산별 목표 없음, S200에서만 목표가 남은 국면, S400에서만 목표가 남은 국면(마스크되지 않은 c2 행과 마스크 미확인을 나눠 센다), 짝별·공통 국면 수
  - 마스크 미확인 국면은 c2가 행을 만들지 않은 곳이다. 학습 자료에 추가될 행 수로 읽지 않는다.
- 원시 방문 분포 표는 표본의 모든 적격 국면에서 낸다.

**집계**
- 국면을 같은 비중으로 센 평균, 중앙값, 90번째 백분위. top-1이 동률이면 행·열 사전순에서 첫 수를 쓴다.
- 수순 구간별(1–10, 11–20, 21–40, 41수 이후)
- 게임 단위 bootstrap 95% 구간(resample 10,000회, 계약에 고정한 seed). 뽑힌 게임의 국면을 함께 가져와 같은 국면 가중 집계를 다시 계산한다.
- 같은 표에 S50 → S200(참고)과 S200 → S400(측정)을 나란히 둔다.

**판정하지 않는다**
- 이 측정에는 통과 기준이 없다. 전체 재탐색과 학습 계약으로 갈지는 결과를 보고 사용자가 정한다.
- 판단에 쓸 비교는 미리 적어 둔다: S200 → S400의 목표 변화가 S50 → S200 변화(c2가 학습한 것)에 비해 어느 정도인가.

## 5. 시간 추정 방법

- 국면마다 worker의 탐색 초(예산별)를 기록하고, 게임마다 wall time을 기록한다. 표본 전체의 wall time과 worker 수도 남긴다.
- **대상 범위를 둘로 나눠 각각 추정한다.**
  - 전체 재탐색: c2 게임 1,000개의 적격 국면 20,910개. 지금의 `h7c_retarget run`이 탐색하고 `h7c_labels --retarget`이 요구하는 범위다. 8,857초도 이 범위의 실측이다.
  - 실제 학습 자료: label 단계가 남긴 train split 게임의 적격 국면. prepare가 c2 policy 행에서 센다.
  - 표본은 probe로 제외된 게임을 뺀 train 게임에서 뽑았으므로, 두 범위 모두 그 표본의 국면당 비용으로 추정한다.
- 추정 방법 (모두 보고한다)
  1. **비율법**: 8,857초 × (표본의 S400 탐색 초 합 / 표본의 S200 탐색 초 합) × (범위 국면 수 / 20,910)
  2. **외삽법**: 표본 S400 wall time × (범위 국면 수 / 표본 국면 수). 같은 worker 수(6)를 가정한다. 탐색 단계가 앞선 실행에서 이어받은 게임이 있으면 wall time이 불완전하므로 내지 않는다.
  3. **탐색 시간만**: 표본 S400의 국면당 탐색 초 × 범위 국면 수 ÷ worker 수
- 검증: 표본의 S200 wall time을 20,910국면으로 외삽한 값을 실제 8,857초와 비교해 외삽의 오차를 적는다. 표본 S200은 두 번(5게임, 95게임)에 나눠 돌리므로 worker 기동 비용이 한 번 더 들어간다.
- prepare는 전체 적격 국면 수가 20,910과 같은지 확인한다(다르면 멈춘다).
- 사전 예상(계획용, 측정이 아님. 측정 결과로 바꾼다)
  - 표본 실행: 약 2,090국면 × (2.5 + 4.2초) ÷ 6 workers ≈ 40분 안팎
  - 전체 S400 재탐색: 8,857초 × 1.69 ≈ 4.2시간 안팎

## 6. 이후 Arena의 검출력 (600쌍, 계획용)

- 기준: A1과 같은 판독(O0·O1 각 300쌍, suite 층화 50:50, 쌍 대응, 단측 α 0.025)
- 쌍 점수 표준편차는 A1의 T arm 값을 쓴다(O0 0.283, O1 0.256). 이때 P의 표준오차는 0.5 × √(0.283²/300 + 0.256²/300) ≈ **1.10%p**이다. A1 P 구간의 폭에서 역산한 값(1.11%p)과도 맞는다.

| 비교 | 표준오차 | 우위 판독에 필요한 **관측** 점수차 | 검출력 80%에 필요한 **실제** 효과 | 검출력 90% |
|---|---:|---:|---:|---:|
| 새 후보 대 champion (P) | 1.10%p | +2.16%p 이상 | +3.08%p | +3.57%p |
| 새 후보 − A1 T (쌍 대응) | 1.06%p | +2.08%p 이상 | +2.98%p | +3.45%p |

- 두 값의 뜻
  - **관측 점수차**: 결과가 STRONGER로 나오려면 실제로 관측되어야 하는 크기다. 단측 하한이 0을 넘는 경계로, 정규 근사 1.96 × SE이다. 실제 판정은 percentile bootstrap이므로 근사다.
  - **실제 효과**: 그 판정을 미리 정한 확률(80%, 90%)로 얻으려면 필요한 참 효과 크기다. (1.96 + z_검출력) × SE이다.
- 새 후보 − A1 T의 표준오차는 A1 D 구간의 폭에서 역산했다. 새 후보와 T의 상관이 A1의 T와 C 사이 상관과 비슷하다고 가정한 값이다.
- A1 T의 champion 대비 추정은 −1.29%p였다(이것도 SE 약 1.1%p). 그 추정이 참이라면, 새 후보가 champion 대비 +3.08%p에 이르려면 T보다 약 +4.4%p 나아야 한다. 이번 D(+3.17%p)보다 큰 개선이다.
- 이 표에 들어 있지 않은 것: 학습 seed 간 변동, 같은 개발 suite(O0/O1)의 반복 사용, 실험 간 다중 비교. 그래서 개발 판독에서 우위가 나온 후보는 독립 확인으로 넘긴다.

## 7. Arena seed (측정 자체에는 필요 없음)

- H7-R 예약 블록 8781–8799는 모두 사용되었거나 예약되어 있다.

| seed | 사용처 |
|---|---|
| 8781–8786 | R1-PILOT-1 |
| 8787 | R1-HDEV-1 |
| 8788–8790 | R2-P1-A1-ARENA-1 |
| 8791–8794 | 확인 단계용으로 예약 |
| 8795–8796 | 테스트 fixture |
| 8797 | R1-SMOKE-HOLDOUT |
| 8798–8799 | R1-SMOKE-1 |

- 새 개발 Arena를 두려면 새 예약 블록이 필요하다(`hybrid.h6_labels.RESERVED_SEEDS`, `h7r_eval.SEED_BLOCK`).
  - `h7r_eval.py`는 이전 계약들(A1 포함)이 SHA로 고정한 source다. 그래서 별도의 버전 단계로 결정해야 한다.
  - S400 측정에는 대국이 없으므로 이 결정 전에도 측정은 할 수 있다.

## 8. 구현 (완료, 미승인)

- 도구 `scripts/h7r_s400_pre.py`
  - `prepare`(`approved.prepare`): 입력 7개의 SHA, self-play 신원, 엔진·runtime·장치, c2 S200 기록의 신원과 coverage를 대조한다. 표본(게임 id 목록, coverage key SHA)과 두 시간 범위의 국면 수를 `prepare.json`에 남긴다.
  - `measure`(`approved.measure`): 커밋된 표본을 다시 계산해 대조한 뒤 §3의 순서로 탐색한다. 탐색은 `h7c_retarget._execute`를 그대로 쓴다(같은 agent, 예산 규칙, 행 검사, 재개, 진행률 출력). 결과는 `measure.json` 하나이고, 재현 검사에서 멈출 때도 남긴다.
  - 탐색 기록 파일에서 이어받은 게임도 새로 탐색한 게임과 같은 검사를 거친다: 적격 국면 목록이 정확히 같아야 하고(빈 행·누락·추가 거부), 행마다 `h7c_retarget.check_row`, 방문 수는 음이 아닌 정수, 평균값은 유한해야 한다.
  - S200 재현 검사는 비교한 국면 수가 기대한 적격 국면 수와 같을 때만 통과한다. 평균값이 유한하지 않으면 차이로 센다.
  - 결과(`prepare.json`, `measure.json`)에 데스크톱 실행 로그 이름(`log_name`)을 남긴다. 복구할 때 그 로그와 종료 코드를 함께 확인한다.
  - 탐색 기록의 신원은 purpose `r2-s400-pre`다. `h7c_labels --retarget`은 purpose가 `retarget`이 아닌 파일을 거부하므로, 이 기록은 학습 자료로 쓰일 수 없다.
  - 기존 `h7c_retarget.py`, `hybrid/h7c_retarget.py`, `hybrid/h7c_targets.py`, `h7c_labels.py`는 바꾸지 않았다. 계약이 이 파일들의 SHA를 고정한다.
- 계약: 표본 규칙, 예산(S200 = factor 4, S400 = factor 8), 재현 규칙, 지표, bootstrap(10,000회, seed 20261113), 시간 기준, 입력·근거·source SHA를 고정한다. 단계는 prepare → measure이고 둘 다 미승인이다.
- 테스트
  - 지표, 표본 순서, bootstrap을 손으로 계산한 값과 대조
  - 임시 저장소에서 stand-in 게임·policy 행·c2 기록으로 prepare → measure 전체: 초기 5게임 불일치 시 즉시 멈춤, 나중 게임 불일치 시 S400 전에 멈춤, G0 허용 범위 안의 평균값 차이는 통과, 커밋된 표본 재대조, 입력 변경 거부, 덮어쓰기 거부
  - 이어받은 기록의 빈 행·NaN 평균값 거부, 비교 건수 부족 시 재현 실패 (이전 도구에서는 실패하는 회귀 테스트)
  - 데스크톱 블록: 스테이징 직후 중단된 결과 기록, 로그 없는 결과·종료 코드 불일치 시 멈춤, 종료 코드가 잘린 로그 복구, 관계없는 스테이징 파일 거부
  - 실제 TB-V3 agent(CPU, stand-in 네트워크): `h7c_retarget`의 실제 S200 실행을 c2 기록으로 삼아, 도구의 S200이 그것을 정확히 재현하고 S400이 행 검사를 통과하며 재개가 동작하는지 확인
- 결과 처리
  - 탐색 기록(`s200.jsonl`, `s400.jsonl`)과 국면별 지표(`positions.jsonl.gz`)는 `runs/h7r/R2-S400-PRE/`에 둔다.
  - 커밋하는 것은 `prepare.json`, `measure.json`(집계, 분모, 시간, 해시)과 로그다.

## 9. 데스크톱 실행 (승인 뒤에만)

- s4-1은 `approved.prepare` 뒤, s4-2는 `approved.measure` 뒤에 실행한다.
- 다시 실행했을 때
  - 결과가 HEAD에 이미 커밋되어 있으면 멈춘다.
  - 결과 파일이 있지만 커밋 전이면(스테이징만 된 경우 포함) 다시 돌리지 않는다. 결과에 남긴 로그 이름(`log_name`)으로 해당 실행 로그를 찾아, 결과가 뜻하는 종료 코드와 로그의 `__EXIT__`이 같은지 확인한 뒤 함께 커밋한다.
    - 로그가 없으면 멈춘다. 로그에 종료 코드가 없으면(출력 직후 중단) 결과에서 복구한 값임을 로그에 적고 기록한다. 두 종료 코드가 다르면 멈춘다.
  - 이 블록이 기록할 파일이 아닌 것이 스테이징되어 있으면 멈춘다. 새 실행은 스테이징이 비어 있을 때만 시작한다.
  - 커밋은 됐고 push만 실패했다면 `git push origin feat/h7-c1`로 다시 보낸다.
  - measure가 결과를 쓰기 전에 끊겼으면 블록을 다시 실행한다. 끝난 게임은 탐색 기록에서 이어받고, 이어받은 기록도 새로 탐색한 기록과 같은 검사를 거친다.
- measure는 표본만으로 약 40분 안팎으로 예상한다(계획값). tmux를 권한다.

s4-1. prepare

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
  D=docs/h7r/r2/R2-S400-PRE
  L=runs/h7r/R2-S400-PRE/logs
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
    python -u scripts/h7r_s400_pre.py "$N" --device cuda --log-name "$(basename "$log")" 2>&1 | tee "$log"
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
if r.get('format') != 'h7r-s400-pre-prepare-v1' or not re.fullmatch(sys.argv[2] + r'-[A-Za-z0-9]+\.log', name):
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
  git commit -m "H7-R S400-PRE: $N exit $rc (records only)"
  git push origin feat/h7-c1
  echo "__RECORDED__ $rc"
  exit "$rc"
)
```

s4-2. measure (tmux 권장)

```bash
(
  set -euo pipefail
  cd ~/renju-h6
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src:$PWD/scripts"
  [ "$(git branch --show-current)" = feat/h7-c1 ] || { echo "STOP: wrong branch"; exit 1; }
  git diff --quiet || { echo "STOP: unstaged tracked changes"; exit 1; }
  if git diff --cached --quiet; then git pull --ff-only origin feat/h7-c1; fi
  N=measure
  D=docs/h7r/r2/R2-S400-PRE
  L=runs/h7r/R2-S400-PRE/logs
  O="$D/$N.json"
  tracked(){ git cat-file -e "HEAD:$1" 2>/dev/null; }
  if tracked "$O"; then echo "STOP: $N is already recorded"; exit 1; fi
  tracked "$D/prepare.json" || { echo "STOP: prepare is not recorded (run s4-1 first)"; exit 1; }
  mkdir -p "$L"
  rc=0
  if [ ! -e "$O" ]; then
    git diff --cached --quiet || { echo "STOP: resolve staged changes before a new run"; exit 1; }
    python scripts/check_track_b_frozen.py
    log="$(mktemp "$L/$N-XXXXXXXX.log")"
    set +e
    python -u scripts/h7r_s400_pre.py "$N" --device cuda --log-name "$(basename "$log")" 2>&1 | tee "$log"
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
if r.get('format') != 'h7r-s400-pre-measure-v1' or not re.fullmatch(sys.argv[2] + r'-[A-Za-z0-9]+\.log', name):
    sys.exit('STOP: invalid result or missing attempt log name')
print(name, 0 if r.get('status') == 'MEASURED' else 2)
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
  git commit -m "H7-R S400-PRE: $N exit $rc (records only)"
  git push origin feat/h7-c1
  echo "__RECORDED__ $rc"
  exit "$rc"
)
```
