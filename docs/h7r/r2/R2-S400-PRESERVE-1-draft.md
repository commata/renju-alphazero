# R2-S400-PRESERVE-1: 같은 S400 학습 자료에서 H3 보존을 강하게 하면 후보가 좋아지는가 (초안)

- 작성 2026-11-19. 상태: **초안, 결정 전. 모든 단계 미승인. 도구·계약 JSON·suite 없음.**
- 결정(2026-11-19 검토): 보존 계수 `μ = 0.5` 대 `μ = 2.0`, 새 O1 계열 개발 suite 두 묶음, 독립 확인은 별도 계약.
- 이 계약은 개발 판독까지다. champion 변경, 독립 확인, 강화학습 루프 재개는 하지 않는다.

## 0. 근거

| 기록 | 내용 |
|---|---|
| R2-P1-POLICY-REPLAY-A1 (`389d09a`) | S200 목표, 주 비교 학습 seed 20261024 하나. H3 보존 항을 더한 T가 대조군 C보다 +3.17%p [+1.08, +5.25] (D BETTER_THAN_CONTROL). T의 champion 대비 우위는 입증되지 않았다(P −1.29%p, NOT_ESTABLISHED) |
| R2-S400-TRAIN (`e1aa371`) | S400 목표 후보 S(μ 0.5)는 champion 대비 −2.00%p [−3.92, −0.08] WEAKER. S − T는 +0.46%p [−0.92, +1.83] NOT_ESTABLISHED |
| R2-S400-VT-MEASURE-1 (`0b2622e`) | S400 값 목표 경로 종료(Q1·Q2 INCONCLUSIVE) |

- 지금까지 같은 S50 예산의 짝 Arena에서 개선이 확인된 변경은 H3 보존 항을 더한 것 하나다. 보존을 더 강하게 했을 때의 효과는 검증된 적이 없다.
- 현재 자료와 학습법에서는 S400 목표의 우위가 입증되지 않았다. 자료 부족, 시작 국면 분포, checkpoint 선택은 각각 따로 검증할 가설이다. 이 계약은 그중 어느 것도 다루지 않고 보존 계수 하나만 바꾼다.
- 기존 S400 학습 자료를 재사용한다. 한 번의 S400 재탐색에 약 4.5시간이 들었다(R2-S400-TRAIN §8.6).

## 1. 질문과 범위

- **질문**: R2-S400-TRAIN과 같은 S400 학습 행, 같은 seed, 같은 학습량, 같은 선택 규칙에서 보존 계수만 0.5에서 2.0으로 올리면 (1) 같은 opening에서 기존 S400 후보보다 좋아지는가, (2) champion을 넘는가.
- `μ = 2.0`은 사전에 고른 실험값이다. 코드에서 μ는 replay KL의 계수다(`loss = cycle batch mean + μ × replay KL mean`, `src/hybrid/h7r_p1_train.py`). KL과 cycle 손실의 숫자 크기만으로는 μ가 실제 모델 갱신에 미치는 영향을 비교할 수 없다. 이 계약은 **기존 계수의 4배에서 더 강한 보존이 대국 성능에 도움이 되는지를 한 번 검증한다.**
- 결론의 범위는 사전 지정 학습 seed 20261024의 두 후보다. 학습 seed 간 변동은 판독 구간에 들어 있지 않다.
- 하지 않는 것: 자료 규모·시작 국면 분포·학습량·선택 규칙 변경, 다른 μ 값 탐색, 결과를 본 뒤의 표본 보충.

## 2. 고정할 비교

| 항목 | S(μ 0.5) — 기존 | S(μ 2.0) — 이번 |
|---|---|---|
| 학습 행 | R2-S400-TRAIN 후보 행 `runs/h7r/R2-S400-TRAIN/policy_rows_s400.jsonl.gz` (SHA `add6022ca8d4…`, 26,000행) | 같음 |
| cycle 항 | c2 정책 방법(CE + λ_kl 0.5 · KL(H3‖student)), batch 512, 2 epoch = 96 step, warmup 5, lr 2e-4, weight decay 1e-4, min lr ratio 0.05, D4 증강 | 같음 |
| replay 항 | RenjuNet train 512국면, 균등 복원 추출, 생성기 seed = 학습 seed + 1000003(D4 대칭 포함) | **같은 생성기, 같은 표본**. 계수만 2.0 |
| 선택 규칙 | RenjuNet val top-1 guard가 H3에서 0.02 이내인 평가 중 cycle val objective 최저(replay 항은 선택 지표가 아님) | 같음 |
| 평가 step | 10, 19, 29, 38, 48, 58, 67, 77, 86, 96 | 같음 |
| 결정론 설정 | R2-S400-TRAIN과 같음(`CUBLAS_WORKSPACE_CONFIG=:4096:8`, deterministic algorithms, TF32 matmul off 등) | 같음 |
| 학습 seed | 20261024 / 20261111 / 20261112 | 같음 |
| 실행 | 새로 학습하지 않는다. R2-S400-TRAIN의 기록과 checkpoint를 SHA로 쓴다(S 20261024: checkpoint `68928bc6ef8c…`, metadata `ed75576736ca…`) | 이 계약에서 세 seed를 학습한다 |

- 같은 seed의 두 실행은 cycle batch 순서, D4 대칭, 학습률, replay 국면이 같다. 다른 것은 replay 항의 계수와 그 결과뿐이다(지문의 batch·sym·lr SHA가 같아야 하고 loss만 달라야 한다).
- 해석용 보고(판정 아님): 평가마다의 RenjuNet val 전체 H3 KL과 guard top-1, 선택 step, cycle val objective.
  - 참고로 S(μ 0.5, 20261024)는 step 96 선택, RenjuNet val H3 KL 0.0437, guard top-1 0.5168(H3 0.5241)이었다.

## 3. 재현 gate (학습 전)

- **질문**: 지금 환경과 도구가 S(μ 0.5, 20261024)를 R2-S400-TRAIN과 정확히 같게 다시 만드는가. Arena와는 다른 질문이다.
- 방법: S(μ 0.5, 20261024)를 같은 진입점(`h7r_p1a1.execute`, kind T, replay μ 0.5)과 같은 수치 설정으로 다시 학습하고, 커밋된 R2-S400-TRAIN seed 기록과 비교한다. 허용 오차 0.
  - 비교 항목: 학습 지문(step 수, batch·sym·lr SHA, loss 열), 평가곡선 전체(평가마다의 val·guard·eligible·replay KL 기록), 선택 step과 선택 state SHA(`106e0ca5cdb6…`), 고정 국면 5,519개 log-확률 SHA(`0ccf8161f969…`), 수치 설정, 입력 SHA(후보 행, c2 행, S400 기록).
- 다시 만든 checkpoint는 Arena에 쓰지 않는다. **Arena에 쓸 S(μ 0.5) checkpoint 파일은 따로 SHA를 확인한다**(R2-S400-TRAIN seed 기록의 candidate SHA와 같아야 한다).
- gate가 실패하면 학습하지 않는다(R2-S400-TRAIN gate가 A1 T를 비트 단위로 재현한 `2e894cf`와 같은 규칙).

## 4. 학습

- `μ = 2.0`으로 세 seed를 학습한다. 실행마다 따로 커밋한다(R2-S400-TRAIN s5-T와 같은 방식: 실행 기록 sidecar, 기록 사본, checkpoint 대조, 재개 시 커밋된 seed의 파일 재대조).
- 학습 전에 후보 행 파일을 R2-S400-TRAIN rows 기록의 SHA와 행 검사로 다시 확인하고(`verified_rows`), 커밋된 gate 기록을 다시 판정한다.
- 기록과 checkpoint 메타데이터에 남길 것: μ, 후보 행·c2 행·S400 기록 SHA, gate 기록 SHA, R2-S400-TRAIN 계약과 실험 SHA, 같은 seed의 S(μ 0.5) 기록 SHA.
- Arena 진입: 사전 지정 seed 20261024의 S(μ 2.0)가 `CANDIDATE`여야 한다. 아니면 다른 seed로 대신하지 않고 계약을 끝낸다.

## 5. 새 개발 suite (O1 계열, 300개씩 두 묶음)

- 규칙: O1(5수: 중앙, 백·흑 반경 2, 백·흑 반경 3. 백 차례), 묶음마다 서로 다른 300개 동치류(opening마다 다른 동치류).
- 제외(5수 opening 동치류 단위): 기존 suite O0·O1·O2·O1-confirm-main·O1-confirm-anchor·**O1-dev-a·O1-dev-b**, 커밋된 모든 v1·v2 평가 계약의 arm 표본(R2-S400-TRAIN-ARENA-1 포함). 두 묶음 사이에도 같은 동치류가 없어야 한다. 3수 opening과의 prefix 겹침은 건수만 기록한다.
- 생성 seed: **202611191**(묶음 c), **202611192**(묶음 d). 기존 예약(`RESERVED_SEEDS`, v2 Arena·suite seed, 학습 run seed 8801–8999, 측정 seed 8800, S1 probe 9201)과 겹치지 않음을 확인했다(2026-11-19, `a97a9c1`).
- 고정 파일 보존: `src/hybrid/h7r_seeds.py`(R2-S400-TRAIN 계약의 sources)와 `scripts/h7r_dev_suites.py`(O1-dev-a·b 검사)는 바꾸지 않는다.
  - 새 생성 seed는 별도 등록(새 모듈 또는 등록 파일)에 고정한다. 새 suite 생성 경로(새 도구)가 그 등록, 제외 목록과 출처 SHA를 기록한다.
  - 기존 `h7r_dev_suites check`는 계속 통과해야 한다(새 suite와 새 Arena 계약이 함께 있어도).
- 파일: `docs/h7r/suites/O1-dev-c.json`, `O1-dev-d.json`, 생성 기록. 생성은 CPU 작업이라 클라우드에서도 할 수 있다(별도 승인 단계).

## 6. 개발 Arena

- v2 Arena 계약 `R2-S400-PRESERVE-ARENA-1`(`docs/h7r/contracts-v2/`, `h7r-eval-contract-v2`, seed policy `h7r-seeds-v2`).
  - 대국 seed: **9103**(O1-dev-c), **9104**(O1-dev-d). 기존 v2 범위(9101–9140) 안이고, `a97a9c1`의 기존 계약은 쓰지 않는다. 계약 생성 때 v1·v2 전체 계약과의 중복을 다시 검사한다(`h7r_eval_v2.check_unique`).
  - 새 opening 생성 seed 때문에 대국 seed 검사를 올리지 않는다. v2 Arena 검사기는 suite의 SHA와 형식을 확인하고 대국 seed를 따로 검사한다.
- 모델: champion(H3 정책 + c5 value), S(μ 2.0, 20261024), S(μ 0.5, 20261024). 모두 c5 value, S50. 대국 전에 세 모델의 네 파일 SHA를 계약과 대조한다(R2-S400-TRAIN의 `check-models`와 같다).
- arm(suite마다 같은 opening·seed·pairing, 각 300쌍)
  - S(μ 2.0) 대 champion
  - S(μ 0.5) 대 champion (같은 suite에서 기존 후보를 다시 둔다. 그래서 두 후보의 짝 비교가 된다)
- 판독(형식은 R2-S400-TRAIN과 같다: suite 층화, 50:50 고정 가중, opening 단위 쌍 대응, 단측 α 0.025, bootstrap 20,000회, seed **20261161**, class 단위 민감도는 보고만)
  - **D″ = S(μ 2.0) − S(μ 0.5)**: 같은 opening에서 두 계수의 성능 차이. BETTER(하한 > 0) / WORSE(상한 < 0) / NOT_ESTABLISHED
  - **P″ = S(μ 2.0) 대 champion 점수 − 0.5**: μ 2.0 후보의 champion 대비 우위. STRONGER / WEAKER / NOT_ESTABLISHED
  - 판독은 두 suite에서 S(μ 2.0)와 S(μ 0.5)의 opening 짝이 모두 일치할 때만 계산한다. 네 arm은 같은 깨끗한 commit에서 두고 judge도 그 commit에서 한다. 300쌍은 줄이지 않는다.
- 비용: arm 4개, R2-S400-TRAIN Arena 기준 약 2.4시간.

## 7. 결과의 쓰임 (사전 고정)

| P″ | D″ | 기록과 다음 |
|---|---|---|
| STRONGER | BETTER | 독립 확인 계약을 연다(새 표본·새 seed). 보존 강화가 기존 후보 대비 개선으로 기록된다 |
| STRONGER | 그 밖 | 독립 확인 계약을 열 수 있다. 보존 강도를 높인 효과는 미확정으로 기록한다 |
| 그 밖 | BETTER | 기존 후보 대비 개선으로 기록하고 이 계약을 끝낸다. champion을 넘지 못했으므로 확인으로 가지 않는다 |
| 그 밖 | 그 밖 | 이 계약을 끝낸다 |

- WORSE는 따로 적는다. 진입 여부는 위 표대로다.
- 개발 Arena에서 champion을 넘은 후보만 독립 확인으로 보낸다. 독립 확인의 통과 여부는 이후 강화학습 재개를 검토할 때 쓴다.
- 결과를 보고 μ, suite, seed, 판독 설정을 바꾸지 않는다. 바꾸려면 새 계약이다.

## 8. 단계와 비용 (모두 미승인)

| 단계 | 하는 일 | 위치 | 예상 시간 |
|---|---|---|---|
| prepare | 입력 대조: R2-S400-TRAIN 계약·rows·search·seed 기록, 후보 행 SHA와 행 검사, S(μ 0.5) checkpoint SHA, A1 입력 | 데스크톱 | 수 분 |
| suites | 새 개발 suite 두 묶음 생성과 겹침 검사 기록 | 클라우드 가능 | 수 분 |
| gate | S(μ 0.5, 20261024) 재현(허용 오차 0) | 데스크톱 | 약 2분 |
| train | S(μ 2.0) 세 seed | 데스크톱 | seed마다 약 2분 |
| arena | Arena 계약, 모델 대조, 대국 4 arm, 판독 | 데스크톱(tmux) | 약 2.5시간 |

- 단계마다 따로 승인한다. gate가 통과해야 train, 20261024가 `CANDIDATE`여야 arena다.
- 원자료(checkpoint, arm 기록, journal)는 `runs/` 아래에 두고 커밋하지 않는다. 기록·요약·로그·SHA를 커밋한다.

## 9. 남은 준비 (결정 뒤, 미승인)

- 도구(새 파일): R2-S400-TRAIN의 고정 도구(`scripts/h7r_s400_train.py`, 계약 sources)를 바꾸지 않고 재사용한다(`verified_rows`, `a1.execute`, 실행 기록 sidecar, Arena 도구의 대국·판독 경로). 새로 만들 부분은 μ 2.0 학습과 gate 비교, 새 suite 생성 경로와 별도 seed 등록, μ 0.5 checkpoint의 SHA 연결, 판독 정의(D″, P″).
- 테스트: gate의 정확 일치(지문·평가곡선·state·log-확률 하나라도 다르면 거부), μ만 다르고 replay 표본과 지문이 같음, 기존 고정 파일과 기존 suite 검사 유지, 새 suite의 제외·겹침·재생성, Arena 계약의 seed 중복 검사, 판독 표, 데스크톱 블록(실제 bash, 도구는 대역).
- 계약 JSON 고정(fixed, 모든 승인 false)과 데스크톱 블록.
