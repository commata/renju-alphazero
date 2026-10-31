# R2-D4-BASELINE-1 — 예측 개선에서 선택 품질로의 연결 진단

작성일: 2026-10-31. 상태: **fixed / 구현·테스트 승인 / 실제 smoke·본 실행 미승인**.

근거 snapshot은 `4180beb6ac9421af5d4dce5cf594ec3c5cc4cdcf`다.

- 사용자의 “구현 방향과 판단할 부분 결정해서 D4 구현과 fixed계약 진행” 요청으로 §11의 여섯 선택을 확정했다(§13).
- 기계 판독 계약은 `R2-D4-BASELINE-1.contract.json`이다. 파일명 `-draft.md`는 기존 참조를 보존한다.
- 실제 H-dev 탐색, Rapfi, 학습은 실행하지 않았다. 합성 입력·stand-in 모델 테스트와 실제 연구 실행을 구분한다.
- 계약은 원본 해시로 396판·15,985행 전체를 고정한다. 데스크톱 `prepare`로 명단을 재구성해 커밋한 뒤, smoke와 본 실행의 별도 승인이 필요하다.
- 검토 반영은 §12, 확정 내용과 구현은 §13에 있다. 과거 제안 문구보다 §13과 JSON 계약이 우선한다.

## 1. 질문과 기존 근거의 정정

질문: "c8 비 mask recipe에서 세 seed 모두 c5 대비 H-dev 예측 손실이 줄었다. 이 감소가 고정된 c5 궤적 위의 수 선택과, 고정 Rapfi continuation 점수에서도 유리하게 나타나는가?"

- 하지 않는 것: 새 학습, self-play, 후보 간 자유 대국
- 하는 것: 기존 H-dev에서 탐색 재생, 그리고 Rapfi로 두 분기를 끝까지 두는 continuation
- 판정하지 않는 것: 실제 기력, minimax 정답, 승급, V9 진입

**c8:c5의 선택 품질은 이미 두 번 평가됐다.** "평가한 적이 없다"는 앞 제안은 사실과 다르다.

| 기존 평가 | 불일치 관측 / 결정 관측 (게임) | c8 우세율 [판독 구간] | mean_delta [판독 구간] | 판독 |
|---|---:|---:|---:|---|
| E2 개발 표본 200판 | 210 / 59 (41) | 62.71% [47.62%, 75.93%] (97.5%) | +0.0561 [−0.0243, +0.1322] | INCONCLUSIVE |
| NMASK 확인 표본 800판 | 855 / 222 (165) | 49.10% [41.36%, 57.27%] (98.33%) | −0.0084 [−0.0455, +0.0292] | INCONCLUSIVE |

- 두 표본은 모두 **c8 cycle의 self-play(seed 8808)**에서 나왔다.
  - NMASK 확인 표본은 그 1,000판 가운데 E2 200판을 뺀 나머지 전부다.
  - 이 self-play는 c8과 이번 세 seed 모델이 **학습(train split)과 checkpoint 선택(val split)에 쓴 자료**다. 그래서 기존 두 평가는 학습 자료 국면 위의 선택 평가다.
- 이번 H-dev(seed 8787)는 이 모델들의 학습에 쓰이지 않은 c5 actor 대국이다. **학습 표본 밖에서 c8 recipe의 선택을 보는 첫 평가**다. 다만 H-dev는 이미 R2-0A, X1, R2-0B에서 본 개발 자료다.
- 기존 기록은 수정하지 않고, 합산하거나 다시 판독하지도 않는다.
- 기존에 두 번 불확실했으므로, "같은 평가를 더 하면 결론이 난다"고 가정하지 않는다. 예상 정밀도는 §8.1에 있다.

## 2. 실행 자격 — 기존 DEVELOPMENT_SIGNAL과 별도

- R2-0B에서 epsilon = **0.0019461168062766864**로 정해졌다.
- seed 31001 / 31002 / 31003의 c5 대비 주 지표 차이는 −0.00475 / −0.00542 / −0.00491이다. 모두 epsilon보다 크고, paired 구간도 음수다.

그러나 이것은 기존 하위 실험의 `DEVELOPMENT_SIGNAL`이 아니다.
- c5는 같은 seed로 학습한 직접 control이 아니다.
- 계획 §3.2의 D4는 원래 D1–D3에서 개선된 변형을 대상으로 한다.

따라서 이 계약은 **D1–D3 전에 baseline 연결 진단을 먼저 하는 명시적 범위 확장안**이다.
- fixed 전 승인문에 이 점을 적는다.
- 기존 DEVELOPMENT_SIGNAL 정의는 바꾸지 않는다.
- D1–D3를 완료로 처리하거나, 생략의 근거로 쓰지 않는다.

실행 전에 확인하는 것: R2-0B 재현 통과, 세 seed, ε 기록, manifest SHA 일치.
- seed나 모델을 H-dev 손실 순위로 고르지 않는다. 셋 모두 포함한다.

## 3. 자료 고정 — 396판 전체 사용 제안

**표본: 검증된 H-dev cache에 기여하는 396판 전체.** 새로 뽑지 않는다.
- 이유: 새 대국 없이 쓸 수 있고, 과거 200판에서는 결정 관측이 부족해 결론이 나지 않았다.
- 396판도 충분한 검출력을 보장하지 않는다(§8.1).

| 항목 | 값 |
|---|---|
| H-dev 계약 | `docs/h7r/contracts/R1-HDEV-1.json` |
| 계약 SHA | `348fecda98c4122dd45abc918c6203683873837cdd5f5c8219f7ac9c187a68b0` |
| binding SHA | `9724d878eed46c9b2a803d212671bc1bebb2c024ed7ba44398468dfd243d4886` |
| 원본 400판 SHA | `93b05a037f20c60104268bc11fa572dc999496ec966bbede482394a3c4bab3f6` |
| 평가 rows SHA | `ed0e4be2a212d7bcce72c501e4674d4376c1c8293fd1f079c7e17b4d99897131` |
| test tensor SHA | `c3b389dad321b82ea98670d7effb1a717d3642abc95718d53b2692dfa5e556fd` |

- 400판 전체의 계약 연결과 합법성을 확인한 뒤, 기존 probe 제외 4판을 그대로 제외한다. 4판을 보충하거나 결과가 좋은 대국을 골라 넣지 않는다.

**평가 대상 국면**
- fixed 준비 단계에서 cache의 `test_games.json`과 tensor의 `game` index로 396개 source game ID를 얻고, 목록과 SHA를 커밋한다.
- 평가 대상은 이 cache의 **15,985개 (game ID, ply)**와 정확히 같은 집합이다.
- canonical 동치류나 반복 국면을 임의로 지우지 않는다.
- eligible 국면의 ply는 5 이상이어야 한다.
- 각 eligible 행은 원본 self-play의 한 ply 기록(engine 탐색이 있었던 수)과 1:1로 맞아야 한다. 맞지 않거나 탐색 기록이 없는 행이 하나라도 있으면 실행 오류다.

**actor 재현**
- 원본 self-play의 `plies` 전체를 원래 순서로 탐색한다(`h7c_sensitivity.replay_game`과 같은 의미).
- 평가 대상보다 앞선 engine ply(opening 직후 ply 3–4 등)는 agent 상태를 재현하기 위한 warm-up으로 탐색하되, 평가 관측에는 넣지 않는다. warm-up 수와 eligible 수를 따로 보고한다.
- cache 국면과 원본 기보의 돌, 차례, 합법 수가 맞는지 확인한다.

**사용 규칙**
- H-dev는 이미 R2-0A와 X1에서 본 개발 자료다. 새 독립 확인 표본이라고 부르지 않는다.
- 이 계약에서도 학습, early stopping, checkpoint 선택에 넣지 않는다.

## 4. 모델과 비교

모든 모델은 기존 H3 policy를 쓴다.
- policy checkpoint SHA: `efee483284dfdd87eb6caa59d0ea3fd015c9dd5862e5e26925b9a23095d0ce61`
- policy metadata SHA: `5648b7fbe71dfa4ab92f6c9bc505ee9ec0af68414f86b97ea22744c9fc022609`

| 이름 | 역할 | value checkpoint SHA | value metadata SHA |
|---|---|---|---|
| c5 | actor·공통 parent | `4e28a62a4a31e462ea116a28d625a80bf2833fdcb30dccbd308b7271c2803024` | `1fcb49ead057881e964c4fac81f99968ee86f7494701343c6083eb82516008f1` |
| seed31001 | 주 비교 A1 | `88d1704ce8cc34b680a45551af23b0b411b3f4c7e28dd41e886c1a349eb52ab0` | `5bdb50fc60b25435e2dd0f60c4bb00cca54238964628eaafd15d8525bbaec836` |
| seed31002 | 주 비교 A2 | `87b690a07decd6d1c6b8476ded9dc97b152887944df242c97b82eb47ee55021b` | `b9416725ceaba6fbef622931d1151aebefc2f8df2398ad3adfa3b110d7638b4c` |
| seed31003 | 주 비교 A3 | `88a6cc6681a320da2a77a659cd659ed992b323256dd6b8c77e5f008f7bf59d29` | `eeacd09699cc0bcc6f0abd27c2cc2d89fd5f4557aba260323a5426171ee7d2c7` |
| c8 | 기존 실행의 탐색적 참고 | `7f665411889aee1e0a3438928c79c24a3e743b2298777d13d2295559c0d74923` | `d1345fce42561d41780431e4418902e44473f4bb69d715b3684fad64150c5aa2` |

- 주 비교는 `seed31001:c5`, `seed31002:c5`, `seed31003:c5` 세 개다.
- `c8:c5`는 같은 H-dev에서 참고값을 내지만, 세 비교의 정식 개발 판독에는 넣지 않는다.
- 가장 좋은 seed 대신 c8을 쓰거나, seed 하나를 골라 대표로 삼지 않는다.

**identity 검사**
- 모든 checkpoint와 명시적 metadata의 실제 SHA를 대조한다.
- 로드한 모델에서 frozen state를 다시 계산해 `c2fbaac3597185a6ab751f08819ddc9e44644e873f68361036f7cbdb22e1b41d`와 같은지 확인한다.
- engine, runtime, source SHA도 fixed 계약에 넣는다.

## 5. 탐색 조건과 actor 재현

- **탐색 설정**: TB-V3 / fast5, 기존 S50과 같은 예산, D4 추론, solver, root 설정. S200/S400, retarget, 새 factory는 쓰지 않는다.
- **장치**: 원본 actor와 같은 CUDA 계열
- **seed**: 모든 모델, 색, 대국에 같은 기존 파생식을 적용한다.
  - `derive_seed(game_seed, 'black'|'white')`, 색별 agent 생성, 원본 actor 수순 진행은 기존 `h7c_sensitivity.replay_game`과 같은 의미를 유지한다.
  - 바뀐 후보 수를 이어 두며 다음 평가 국면을 만들지 않는다.
- **c5 재현 조건**: 모든 replay ply에서 원본의 engine move, route, root visits/Q를 재현해야 한다. E2에서는 7,798 ply 모두 수, route, visits가 같았다.
  - mismatch, 누락, 중복은 해당 행을 빼는 것이 아니라 실행 오류다.
- **engine 수와 실제 착수 구분**: self-play에서 확률적으로 둔 수(sampled 1,531 ply)는 engine이 제안한 수와 다를 수 있다. 원본의 실제 수순으로 replay하되, 선택 비교에는 같은 seed의 engine 선택 수를 쓴다.
- actor와 나머지 모델을 같은 조건에서 탐색한다. 실행 순서와 worker 수는 기록하지만, 결과를 보고 예산을 바꾸지 않는다.

**민감도 보고 항목**
- eligible 15,985국면 전체의 차이율
- 양쪽 모두 tree인 국면의 차이율
- route 변화
- 대국별 차이 발생 비율과 첫 차이 ply
- 방문 분포 JS와 선택 Q의 차이

root tree와 value leaf의 network / exact_leaf 구분을 혼동하지 않는다. 차이율은 행동 진단이며, 기력이나 미래 recipe의 상한이 아니다.

## 6. Rapfi 관측과 분기

**관측 정의**
- 비교마다, eligible 국면에서 두 모델의 engine 선택 수가 다르면 관측 하나를 만든다.
- 관측은 (comparison, source game ID, ply, candidate move, c5 move)다.
- 같은 기보나 국면이 반복돼도 관측을 지우거나 가중치를 바꾸지 않는다.

**분기 공유**
- 분기 key는 전체 국면 history, 고정 수, 차례를 포함한다.
- 같은 분기는 모델과 비교를 넘어 캐시해 한 번만 실행할 수 있다. 특히 c5 수 분기는 세 seed 비교가 공유한다.
- identity에 모델·비교 이름을 넣어 같은 분기를 중복 실행하지 않는다. 반대로 수순이나 상태가 다른 분기를 임의로 합치지도 않는다.
- 공유한 분기, 국면, 관측 수를 따로 보고한다.
- 세 seed 결과는 상관돼 있으므로 독립 증거처럼 합산하지 않는다.

**continuation 규칙**
- 각 분기는 새 Rapfi 프로세스로 시작한다.
- 고정 수 뒤부터 양쪽 색 모두 Rapfi가 종료까지 둔다.
- 점수는 원래 고정 수를 둔 쪽 기준이다: 승 1 / 정상 무승부 0.5 / 패 0
- 고정 수 자체로 대국이 끝나면 continuation 없이 Game 결과를 쓴다.

**Rapfi 설정**
- 기존 X2에 등록된 executable, 6개 모델·설정 SHA, INFO, rule 4와 같게 한다.
- fixed 준비 때 실제 `engine_spec`의 identity 전체를 계약에 넣는다.
- `max_node=300`, thread 1, 수 응답 상한 600초를 유지한다.
- 파일이나 INFO가 다르면 시작 전에 거부한다.
- 다른 탐색자의 Q나 수치를 Rapfi 점수와 섞지 않는다.

**검증**: 모든 continuation을 Game으로 재생하고 금수점을 교차 확인한다. branch identity, 수순, 차례, 종료 상태, 승자, 결과가 모두 맞아야 한다. 재개와 집계 때도 같은 검증을 적용한다.

## 7. INVALID 정책 제안 — 기존 결과의 사후 재판독 아님

새 D4 계약에서 아래 규칙을 실행 전에 고정하는 안이다. 과거 v1/v2의 INVALID 판정과 결과는 바꾸지 않는다.

**정의는 새로 만들지 않는다.** R1의 `h7r-invalid-v1` 판정과 분기 검증기를 다시 쓴다.
- `hybrid.h7r_eval.check_invalid_policy`, `EXECUTION_FAILURES`
- `hybrid.h7r_evidence.branch_outcome`, `structural_failures`

R1과 달라지는 것은 미해결 분기를 다루는 방식 하나다(아래 양방향 bound).

1. **구조적 abnormal_end**: 기존 H7-R 재생 검사로 다음을 모두 확인한 경우만 **평가상 무승부 0.5**로 정규화한다.
   - 승자 없음, 흑 차례, 빈 점 정확히 1개, 합법 수 없음, 그 점이 흑 금수
   - 원본 INVALID 표시와 정규화 건수를 보존한다.
   - 조건이 하나라도 다르면 새 정책으로 임의 처리하지 않고 데이터 오류로 멈춘다.
2. **timeout / engine_error / engine_illegal / forbid_mismatch**: 기존 검사로 사건과 prefix를 검증하고 **미해결 분기**로 남긴다. 표본에서 빼지 않는다.
3. **누락, 중복, SHA 불일치, 불법 기보, 잘못된 고정 수, truncated, 정의되지 않은 사유**: 분석용 INVALID로 낮추지 않고 실행·데이터 오류로 멈춘다.
4. **미해결 비율 상한**: 비교별 미해결 관측이 5%를 넘으면 INCONCLUSIVE다.
   - 분모는 전체 불일치 관측 수이고, 경계는 정확한 분수로 비교한다.
   - 구조적 정규화는 미해결에서 빼고 따로 센다.

**미해결 분기가 있는 관측은 점수 차이의 하한과 상한을 둘 다 만든다.**
- 하한 delta: 미해결 후보 분기 0, 미해결 c5 분기 1
- 상한 delta: 미해결 후보 분기 1, 미해결 c5 분기 0
- 정상 분기와 정규화 분기의 점수는 하한·상한에서 같다.

보수적인 하한만 써서 WORSE를 선언하지 않는다. 정상 관측만으로 낸 값은 참고로 표시할 뿐, 공식 개발 판독을 대신하지 않는다.

## 8. 주 지표와 통계

**주 지표 제안: 불일치 관측 전체의 평균 continuation 점수 차이**

    delta_i = score(candidate move) − score(c5 move)
    mean_delta = Σ_i delta_i / N_disagreement

- 승·무·패가 같은 관측은 delta=0으로 포함한다.
- 단위는 continuation 점수다. 전체 대국 score 향상량이나 WMSE 단위가 아니다.
- R2의 epsilon=0.001946을 이 지표에 적용하지 않는다.
- `mean_delta`는 v1/v2 판정 기록에도 **보조 지표로 이미 있다**(같은 정의, source-game bootstrap 구간). 새로운 점은 이것을 주 지표로 올리는 것이다. v1/v2는 better_share로 판독했다.
- 따라서 **새 D4 형식/버전**으로 구현한다. v1/v2 기록의 뜻은 바꾸지 않는다.
- 이 지표를 고른 이유: 점수 차이의 크기(반점 차이 포함)와 동일 결과를 모두 반영한다.

**bootstrap**
- source-game cluster bootstrap 20,000회, percentile
- 396개 게임의 같은 multiplicities를 세 seed, c8 참고, 모든 지표에 공통으로 적용한다.
- 불일치 관측이 없는 게임은 빈 기여다. 분모가 0인 회차는 세고, 그 구간은 판독하지 않는다.
- 관측이 많은 대국이 이 주 지표에서 더 큰 비중을 갖는다는 점을 명시한다. 불일치가 있는 게임의 게임 균등 평균을 보조로 보고한다.

**구간과 다중 비교**: 두 종류의 주장을 구분한다.
- **seed 하나에 대한 판정**(seed별 label): family 0.05의 Bonferroni **98.333…% 양측 구간**을 쓴다. 의존성이 있어도 쓸 수 있는 보정이다.
- **"세 seed 모두"에 대한 판정**(`ALL_THREE_*`): 세 개가 모두 성립해야 하는 교집합 주장이다. intersection–union 원리에 따라 각 seed의 **95% 구간**으로 판정해도 그 주장의 오류율은 0.05 이하다. 여기에 Bonferroni를 겹치면 이미 낮은 검출력(§8.1)이 불필요하게 더 낮아진다.
- 95%와 98.333% 구간을 모두 보고한다.
- 개발 자료를 이미 본 설계이므로, 최종 확인 수준의 오류 보장이라고 홍보하지 않는다.

**비교별 최소 조건**
- 전체 불일치 20개 이상
- 두 분기가 모두 해결된 관측 20개 이상, 그 source game 10개 이상
- 미해결 관측 5% 이하
- 완료, identity, Game 재생, actor 재현 통과
- bootstrap 분모 0 회차 없음

미해결이 없으면 일반 mean_delta 구간을 쓴다. 미해결이 있으면 같은 multiplicities로 하한 delta와 상한 delta의 구간을 각각 계산한다. 수준 q는 98.333%(seed별 label) 또는 95%(`ALL_THREE_*`)다.
- **BETTER(q)**: 하한 delta의 q 구간 하한 > 0
- **WORSE(q)**: 상한 delta의 q 구간 상한 < 0
- **INCONCLUSIVE**: 그 밖이거나 최소 조건 미달

**보조 지표**
- 후보 수가 나은 비율 `Pr(delta>0 | delta!=0)`
- 우세 / 열세 / 동일, 둘 다 승·무·패
- 결정 관측 수와 그 게임 수
- 관측당 평균과 게임당 평균
- 미해결 비율과 정규화 비율
- 결정 관측이 없으면 better_share를 0이나 0.5로 채우지 않고 null로 둔다.

### 8.1 예상 정밀도 — 실행 전에 기록하는 기대

과거 c8:c5 기록으로 예상한 규모다. 계획용 근사이며 보장이 아니다.

- **불일치 관측 수**: E2는 200판에서 210개, NMASK는 800판에서 855개로 대국당 약 1.05개다. 세 seed가 c8과 비슷하게 c5와 다르다면 396판에서 **비교당 약 420개**(결정 관측 약 110개)를 예상한다.
- **구간 반폭**: NMASK의 855개에서 mean_delta 98.33% 구간 반폭은 약 0.037이었다. 관측 수의 제곱근에 비례한다고 보면, 420개에서는 반폭이 약 **0.053**(98.33%) / **0.045**(95%)다.
- **해석 범위**: 위 수치는 예상 구간 반폭이다. 특정 효과에서의 검출력이나 최소 검출 효과를 계산한 값은 아니다. mean_delta는 동일 결과 비율·점수 차이 크기에도 의존하므로, 이를 우세율 60%로 변환하지 않는다.
- **기존 추정치**: 학습 자료 국면에서도 E2 +0.056, NMASK −0.008이었다. 규모가 더 큰 NMASK 쪽은 0 근처다.
- **불확실한 결과의 가능성**: 기존 자료는 `NOT_ESTABLISHED`가 충분히 가능한 시나리오임을 보여 준다. 어느 결과가 가장 가능성 높은지 추정할 확률 모형은 없다. 그런 결과가 나오면 "연결 미입증"으로 기록하고, 표본이나 예산을 늘려 다시 판정하지 않는다.
- **효과 크기 상한(기술적 보고)**: 결과와 무관하게, 비교마다 상한 delta의 95% 구간 상한을 "이 표본이 배제하지 못한 가장 큰 개선"으로 함께 보고한다. 동등성 판정이 아니며, INCONCLUSIVE를 "차이 없음"으로 읽지 않는다.
- **승인할 때 판단할 점**: 이 기대를 보고도 실행할 가치가 있는지. 탐색 비용은 §11에 있다. 민감도 부분만으로도 행동 변화의 크기와 위치는 결정론적으로 얻는다.

## 9. 세 seed 종합과 다음 행동

| 종합 label | 조건 | 의미 |
|---|---|---|
| `ALL_THREE_BETTER` | 세 seed 모두 BETTER(95%) | 이 H-dev와 Rapfi 설정에서 seed에 걸친 연결 신호. 기력·배포 승인이 아니다 |
| `ALL_THREE_WORSE` | 세 seed 모두 WORSE(95%) | 현재 예측 지표와 이 선택 평가의 방향이 어긋난다는 신호. 학습 목표나 특정 원인을 확정하지 않는다 |
| `MIXED_DIRECTION` | seed별 label(98.333%)에 BETTER와 WORSE가 모두 있음 | — |
| `PARTIAL_SIGNAL` | seed별 label(98.333%)에 BETTER가 있고 WORSE가 없으며, 위 둘이 아님 | seed 전반의 연결을 주장하지 않는다 |
| `NOT_ESTABLISHED` | 그 밖 | INCONCLUSIVE가 섞이면 차이 없음, 동등, 병목 확정으로 읽지 않는다 |

**seed 결과 해석 규칙**
- 세 seed를 한 표본으로 합쳐 하나의 작은 p값을 만들지 않는다.
- seed별 H-dev 예측 손실·구성 지표와 행동·선택 결과를 같은 표에 나란히 둔다.

**결과가 정하지 않는 것**
- 이 결과로 D1이나 D3을 자동 실행하지 않고, 주 지표도 바꾸지 않는다.
- D4가 불확실해도 예측 개선이 기력과 연결되지 않는다고 단정할 수 없다.
- 다음 하위 계약은 관측, 불확실성, 비용을 보고 따로 작성한다.
- 결과를 본 뒤 국면 수, 예산, Rapfi 설정을 늘려 같은 계약을 다시 판정하지 않는다.

## 10. seed·출처·재개와 구현 필요 항목

**새 난수 제안값**
- 분기 순서 2026103101, bootstrap 2026103102
- 학습 seed와 역할이 다르다. fixed 전에 실제 예약·사용 기록과 충돌하지 않는지 확인한다. 저장소 기록의 사용 seed(20261028 / 20261029 / 20261030, 8764)와는 겹치지 않는다.
- 탐색 seed는 원본 actor에서 파생하며, 새 난수로 바꾸지 않는다.

**계약 JSON과 명단에 넣을 것**
- 부모 R2 계약, R1-HDEV 계약과 manifest
- R2-0A 기록, R2-0B의 epsilon·reproduction·manifest, 세 모델의 학습 기록, c8 기록
- 코드, 엔진, runtime, Rapfi 전체 SHA
- 기대값이 null인 채로 fixed 실행하지 않는다.

**identity와 재개**
- identity: clean code commit, 계약 SHA, 입력, 모델, 명단, 예산, seed, 엔진 identity
- 완료한 game과 branch를 바로 저장하고, 같은 identity일 때만 재개한다.
- OS 잠금과 순서 없는 수집(`run_chunks`)을 쓴다. 약 60초마다 stdout에 진행 상황(done / total / % / 경과 / ETA)을 출력한다. CLAUDE.md의 장시간 작업 규칙이다.
- 다른 HEAD, 중복, 중간 손상은 거부하고 원본을 보존한다.

**도구 재사용 범위**
- 기존 `h7c_sensitivity` CLI는 c8 evidence 형식에 묶여 있다. 그대로 H-dev에 적용할 수 있다고 쓰지 않는다.
- 내부 replay·check 함수와 H7-R holdout 검증기(`h7r_holdout.checked_cache`)를 다시 쓰는 **개발용 wrapper**가 필요하다.
- 기존 `h7r_eval selection-source`는 H-conf 확인 계약 전용이다. purpose를 conf로 위장하거나 확인 ledger를 소비하지 않는다.

**구현 필요 항목**: H-dev 입력과 eligible·warm-up 연결, 다중 모델 계약, 원본 branch 재생, 양쪽 미해결 bound, 위 주 지표와 seed별·종합 판독, provenance, 재개, 잠금
- fixture/fake 엔진 테스트와 실제 actor smoke를 구분한다.
- smoke는 별도 `runs/` 경로와 SMOKE 표시로 끝내고 공식 판독을 내지 않는다.

**산출물 제안**

| 산출물 | 경로 |
|---|---|
| 계약 | `docs/h7r/r2/R2-D4-BASELINE-1.contract.json` (아직 만들지 않음) |
| sample/eligible manifest, 결과·SHA 요약 | `docs/h7r/r2/R2-D4-BASELINE-1/` |
| 원본 replay, branches, 로그, tensor | `runs/h7r/R2-D4-BASELINE-1/` |
| 기록 사본 | docs 아래 **`records/`**(docs 아래 `runs/`는 `.gitignore`의 `runs/` 규칙에 걸린다) |

## 11. 비용과 확정 전 선택

신경망 학습과 새 self-play 비용은 없다.

**탐색 작업량**: 396판의 replay ply 전체(warm-up 포함, 약 16,990 ply) × 5모델
- H-dev self-play의 실측 처리량(수당 평균 0.80초, 6 workers)으로 환산하면 **약 2시간**이다(16,990 × 5 × 0.8초 / 6).
- 계획용 추정일 뿐 완료 시간을 보장하지 않는다.
- 사전에 고정한 소규모 smoke에서 처리량을 기록해 ETA만 갱신한다. 처리량을 보고 표본, 예산, 후보를 바꾸지 않는다.

**Rapfi 작업량**
- 관측은 비교당 약 420개 × 4 비교로 예상한다. c5 분기를 공유하면 고유 분기는 약 2,000개 이하다.
- 과거 v1의 분기당 평균 1.46초와 4 workers로 환산하면 수십 분이다. 다른 표본의 참고값이며, 이번 국면 길이에서의 보장은 아니다.
- 최대 관측 수와 분기 수는 비용 상한일 뿐 실제 실행량이 아니다.
- 합계 계획용 추정은 **약 2–3시간**이다.

**fixed 전에 승인해야 할 설계 선택**
1. D1–D3 전에 baseline D4를 수행한다는 범위 확장
2. 200판 부분표본 대신 기존 396판 전체와 15,985 eligible 국면 사용
3. 세 seed:c5를 주 비교로, c8:c5는 기술적 참고로 포함
4. 평균 continuation delta를 주 지표로 하고, seed별 label은 98.333%, `ALL_THREE_*`는 95%(intersection–union)로 판독
5. 구조적 abnormal_end의 평가상 무승부(`h7r-invalid-v1` 재사용)와 미해결 양방향 bound 정책
6. §8.1의 예상 정밀도와 `NOT_ESTABLISHED` 가능성을 알고도 진행할지

승인은 세 단계로 나눠 받는다: 문서 검토 → 구현·테스트와 fixed 계약 → 실제 smoke와 본 실행. 이 초안을 작성한 것 자체는 어느 단계의 실행 승인도 아니다.

## 12. 검토 반영 (2026-10-31)

| # | 원 초안 | 수정 | 이유 |
|---|---|---|---|
| 1 | 기존 두 평가는 "c8 self-play(seed 8808)"에서 나왔다 | 그 self-play가 c8과 세 seed 모델의 **학습 자료**라는 점, 이번 H-dev가 학습 표본 밖의 첫 선택 평가라는 점 추가 | `docs/h7c/nmask/confirm_sample.json`: E2 200판을 뺀 c8 self-play 전부. c8 기록의 `selfplay_run_seed` 8808 |
| 2 | 평균 점수 차이를 새 주 지표로 제시 | v1/v2에도 `mean_delta`가 보조로 있다는 점과 기존 추정치·구간을 §1 표에 추가 | 주 지표 선택이 기존 결과를 본 뒤의 선택이라는 맥락을 남긴다. 정밀도 계획의 근거도 된다 |
| 3 | 모든 판독에 98.333% | seed별 label은 98.333%, `ALL_THREE_*`는 각 seed 95%(intersection–union) | 교집합 주장에는 Bonferroni가 필요 없다. 이미 낮은 검출력을 더 깎지 않는다 |
| 4 | 정밀도 서술 없음 | §8.1: 비교당 약 420 관측, 반폭 약 0.05, 가장 가능성이 높은 결과 `NOT_ESTABLISHED`, 효과 크기 상한 보고 | 결과를 본 뒤 "예상 밖으로 불확실했다"며 표본을 늘리는 것을 막는다. 실행할 가치는 승인 단계에서 판단한다 |
| 5 | INVALID 규칙을 새로 서술 | R1 `h7r-invalid-v1` 판정과 분기 검증기 재사용을 명시. 다른 점은 미해결 양방향 bound 하나 | 같은 사건을 두 정의로 판정하지 않는다 |
| 6 | eligible = cache 행 | 각 행이 원본 self-play의 engine 탐색 ply와 1:1로 맞아야 한다는 조건 추가 | cache 행과 결정 국면이 어긋나면 관측이 정의되지 않는다 |
| 7 | 비용 "수 시간" | H-dev self-play 실측 처리량으로 환산한 계획용 추정(약 2–3시간) | 승인 판단의 근거 |
| 8 | — | 장시간 작업 진행 출력, `run_chunks` 명시 | CLAUDE.md 규칙 |

## 13. 구현·fixed 결정 (2026-10-31)

§11의 선택 1–6을 채택한다. 학습 자료 밖 H-dev에서 행동 변화와 continuation 차이를 함께 얻는 가치가 있어 진행한다. 결과가 불확실할 수 있다는 이유로 중간에 표본을 늘리거나 예산을 바꾸지 않는다. 실제 실행은 별도 승인 범위다.

- **범위**: baseline 연결 진단을 D1–D3 앞에 둔다. 세 seed 전부와 c8 참고를 포함하며 H-dev 전체를 쓴다. 기존 `DEVELOPMENT_SIGNAL`, ε, 승급·수렴·V9·champion 규칙은 바꾸지 않는다.
- **명단**: 원본 파일 여섯 해시와 전체 cache 규칙이 대상을 이미 결정한다. 로컬 원본이 없는 이 환경에서 ID 목록을 추측하지 않는다. `prepare`는 검증한 원본에서 모든 ID와 eligible ply를 정렬해 `eligible.json`을 만들고, 실행기는 커밋된 이 명단을 원본에서 다시 생성한 값과 정확히 대조한다. 표본 선택 권한은 없다.
- **검증**: 각 eligible cache 행의 흑·백 돌, last, 합법 수를 원본 수순과 대조한다. 전체 원본 engine ply를 재생해 c5의 move/route/visits/Q를 확인한 뒤 eligible 행만 관측에 쓴다. hash·행 연결·actor 재현 오류는 즉시 중단한다.
- **모델**: checkpoint와 실제 factory가 읽는 인접 `best.json`을 확인한다. frozen state는 실제 로드한 값에서 다시 계산한다. CUDA 원본 engine/runtime hash와 Rapfi X2의 exe·가중치·설정·INFO도 일치해야 한다. 계약에 source SHA를 고정했다.
- **판독**: 주 지표는 전체 불일치 관측의 평균 delta다. 미해결은 양방향 bound로 남기고, 5% 경계는 정수 분수로 검사한다. 최소 관측·해결 게임 조건과 bootstrap 분모 0 검사를 통과해야 판독한다. 세 seed 공통 방향은 95%, 개별 seed는 98.333…%다. c8은 공식 label 없이 참고 구간만 낸다.
- **추정**: `game_index` 오름차순인 공통 396게임 multiplicities 20,000회, seed 2026103102. percentile 구간은 `torch.quantile`의 선형 보간으로 고정한다. 분기 순서 seed는 2026103101이다. 기존 실험의 난수나 판독을 바꾸지 않는다.
- **상관**: source history+고정 수가 같은 분기만 공유한다. 비교별 관측은 유지하며 공유 분기 행렬을 기록한다. seed 결과를 합쳐 독립 반복처럼 취급하지 않는다.
- **재개**: OS 잠금, 완료 즉시 fsync, identity와 모든 재생 기록 재검증. 손상된 중간 줄·중복·알 수 없는 기록은 거부한다. 개행 없이 끊긴 마지막 JSON만 `.partial-*`에 보존한 뒤 재실행한다. 완료 summary는 덮어쓰지 않는다. HEAD·입력·source·모델·엔진·장치·실행 환경이 같아야 재개한다.
- **실행 승인**: 계약의 `execution`은 prepare만 true다. smoke/full은 각각 계약 SHA와 mode를 지정한 별도의 커밋된 `h7r-d4-execution-v1` 승인 파일이 있어야 한다. 이번 커밋에는 이 실행 승인 파일을 만들지 않는다.
- **smoke 격리**: 정렬된 첫 2게임의 전체 ply와 고정 순서 첫 8분기까지만 실행한다. `runs/.../smoke/`의 `SMOKE` 기록으로 끝내고 정식 label을 내지 않는다. 본 실행의 입력·journal로 재사용하지 않는다.
- **출력**: 원본 재생·분기 JSONL과 로그는 `runs/h7r/R2-D4-BASELINE-1/{smoke,full}/`에 둔다. 요약에는 이 파일 SHA, 입력·모델·코드 출처, actor 재현, warm-up/eligible 수, 민감도, 양방향 구간, branch 정규화 건수, 개별·종합 판독을 담는다. source rows·tensor·checkpoint는 커밋하지 않는다.

도구: `scripts/h7r_d4.py prepare|run`, `src/hybrid/h7r_d4.py`. 데스크톱 준비 명령은 계획 문서 §9.19에 있다. 실제 c5/seed checkpoint·H-dev·CUDA·Rapfi를 쓰는 smoke와 본 실행은 아직 수행하지 않았다.

검증: 관련 8개 모듈 묶음 156 tests PASS. 최종 D4 suite는 18개이며 합성 입력의 전체 실행·중단·재개·집계, fake protocol engine, 실제 fast5/stand-in 모델의 CPU actor 재생을 구분해 다룬다. frozen 검사 `--no-models` PASS, 명령 블록 2개 `bash -n` PASS. 실제 H-dev·생산 모델·CUDA·등록 Rapfi는 이 checkout에 없어 검증하지 못했다.

## 근거

- 계획: `docs/h7-rl-plan.md` §3.2, §9.17
- R2-0B 결과: `docs/h7r/r2/R2-DIAG-1/r2_0b/epsilon.json`
- 기존 c8:c5 개발 판정: `docs/h7c/e2_adjudicate/adjudication.json`
- 기존 c8:c5 확인 표본 판정: `docs/h7c/nmask/adjudication.json`, 표본 `docs/h7c/nmask/confirm_sample.json`
- 탐색 재생: `scripts/h7c_sensitivity.py` (`replay_game`), E2 actor 재현 `docs/h7c/e2/sensitivity_c8.json`
