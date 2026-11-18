# R2-S400-VT-MEASURE-1: S50 탐색 경합 국면에서 S400 탐색 결과가 value 목표로 쓸 만한가 (측정 계약)

- 작성 2026-11-18, 개정 2026-11-18(검토 반영). 상태: **계약 고정(fixed), 모든 단계 미승인**(`approved` prepare·smoke·full 모두 false). 도구·테스트·데스크톱 블록 준비(§12, §13).
- 계약 `docs/h7r/r2/R2-S400-VT-MEASURE-1/vt-measure-contract.json`, 도구 `scripts/h7r_vt_measure.py`, 테스트 `tests/test_h7r_vt_measure.py`.
- 이 계약은 측정만 한다. 학습, 자유 대국 Arena, champion 변경, MCTS V9, 강화학습 루프 재개는 하지 않는다.
- 결과가 사전 기준(§8)을 충족할 때만 별도의 학습 계약(value 목표만 바꾸는 비교) 초안을 연다.

## 0. 근거

| 기록 | 내용 |
|---|---|
| `docs/h7c/e2/sb_c5_saturation.json` | 같은 H3 + c5로 S400이 S50보다 강했다. 150쌍 확인에서 점수 0.5767, 개선 95% 구간 [+2.5, +12.7]%p, STRONGER. 수당 시간은 50쌍 예산 arm에서 4.94배(S400 arm 단독은 +9.5%p, 하한 0.0, NOT_DETECTED) |
| R2-D1-HEAD-WIDTH-1 (D4 full) | value 예측 손실은 줄었지만(H-dev `DEVELOPMENT_SIGNAL`) H-dev 선택 품질 판독은 `NOT_ESTABLISHED` |
| R2-S400-TRAIN (`4ff2818`) | S400 정책 목표로 학습한 후보: P′ WEAKER(−2.00%p), D′ NOT_ESTABLISHED(+0.46%p) |
| R2-D4 (D1 D4 full 기록) | H-dev 15,985국면 중 S50이 tree로 탐색한 국면 7,252개(약 45%) |

- 깊은 탐색은 대국에서 더 강하다. 그러나 그 결과를 학습 목표로 옮긴 두 시도(정책 목표, 넓힌 value head)는 착수 개선으로 이어지지 않았다.
- 그래서 학습 전에 잰다: 깊게 탐색해서 바뀐 수가 독립 판정에서도 더 좋은가(Q1), 그리고 S400의 값이 c5 value 망의 값보다 독립 판정 결과에 대한 오차가 작은가(Q2). 두 질문 모두 S50의 탐색이 경합한 국면에서 본다.

## 1. 질문

- **Q1 (수)**: S50 탐색 경합 국면(V, §4)에서 S400과 S50의 최종 착수가 갈릴 때, S400의 수가 Rapfi continuation 판정으로 더 좋은가.
- **Q2 (값)**: V의 모든 국면에서, 그 국면을 Rapfi가 처음부터 끝까지 둔 고정 결과 r에 대해 S400 루트 값의 제곱오차가 c5 value 망 값의 제곱오차보다 작은가.
  - 결론은 **이 고정된 Rapfi 결과에 대한 제곱오차**로 한정한다. r은 국면당 한 번의 continuation이고, Rapfi의 국면별 기대 점수를 따로 추정하지 않는다. 두 값은 같은 r을 기준으로 짝지어 비교된다.
- 답하지 않는 것: 실제 기력, minimax 정답, 학습한 모델의 성능, 승급, V9 진입. 두 결과가 함께 좋아도 그 값을 학습한 S50 모델의 기력은 이후 대국에서 확인해야 한다.

## 2. 표본: H-dev 396판 전체

- R2-D4와 같은 H-dev를 쓴다. 새로 뽑지 않는다.
  - H-dev 계약 `docs/h7r/contracts/R1-HDEV-1.json`(SHA `348fecda98c4…`), binding `9724d878eed4…`, 원본 400판 `93b05a037f20…`, test tensor `c3b389dad321…`(모두 R2-D4 계약의 값과 같아야 한다)
  - 평가 국면은 R2-D4가 고정한 목록 `docs/h7r/r2/R2-D4-BASELINE-1/eligible.json`(`28f0d48`)의 396판·15,985개 (game, ply)다. 도구는 H-dev cache에서 이 목록을 다시 만들어 커밋된 파일과 같아야 진행한다. warm-up ply 792개는 탐색하되 관측에 넣지 않는다.
- 사용 규칙
  - H-dev는 R2-0A, X1, R2-0B, D4, D1 D4에서 이미 본 개발 자료다. 이 계약의 결과는 개발 판독이다. 독립 확인이라고 부르지 않는다.
  - 학습, early stopping, checkpoint 선택에는 쓰지 않는다. 뒤따르는 학습 계약도 H-dev를 학습 자료로 쓰지 않는다.
- R2-S400-TRAIN의 S400 탐색 기록(c2 train 범위)은 학습 자료여서 쓰지 않는다.

## 3. 탐색 조건

| 이름 | 설정 | 역할 |
|---|---|---|
| S50 | frozen TB-V3 / fast5, H3 정책 + c5 value, 기존 S50(`h7c_budget.make_budget_agent(1, …)` = `h7c_arena.make_side`) | actor 재현과 기준 선택 |
| S400 | 같은 모델과 검사, tree 예산만 8배(`h7c_budget.make_budget_agent(8, …)`) | 깊은 탐색의 선택과 루트 값 |

- 모델 SHA(H3 정책 `efee483284df…`/`5648b7fbe71d…`, c5 value `4e28a62a4a31…`/`1fcb49ead057…`), frozen value state, engine·runtime SHA, Rapfi identity를 실행 전과 발행 직전에 대조한다. 모두 R2-D4 계약의 값과 같아야 한다.
- actor 재현은 R2-D4 §5와 같다.
  - 원본 self-play의 ply 전체를 원래 순서로 탐색한다. 색별 agent seed는 `derive_seed(game_seed, color)`로 같다.
  - S50은 모든 ply에서 원본 actor의 engine 수, route, root visits/Q를 재현해야 한다. 하나라도 다르면 분기 전에 멈춘다(행을 빼지 않는다).
  - S400은 같은 국면, 같은 seed 규칙으로 탐색한다. 다음 국면은 원본 수순이다.
- 국면마다 기록(두 탐색 각각): route, 최종 착수(루트 검사 뒤), 트리 자체의 선택(`v8_v7_move`, 루트 검사 전), 루트 prior 1순위(`v8_puct_prior_top`), root visits/Q(`h7c_targets.raw_visits` 형식). 그리고 c5 value 망의 원값(두 탐색 뒤에 계산).

## 4. 국면 분류 (Rapfi 결과를 보기 전에 고정)

| 집합 | 정의 |
|---|---|
| E | 평가 국면 15,985개 |
| T | E 중 S50과 S400이 모두 tree 경로이고 루트 방문이 있는 국면 |
| V (S50 탐색 경합 국면) | T 중 V-A 또는 V-B인 국면 |
| — V-A | S50의 최종 착수가 루트 prior 1순위와 다르다 |
| — V-B | S50 루트에서 방문한 자식이 둘 이상이고, 방문 1위와 2위의 차이가 루트 자식 방문 합의 20% 이하다(방문 수만으로 계산, 경계 포함) |

- **V는 "S50의 탐색이 경합한 국면"이다. value가 수를 바꿨다는 뜻이 아니다.**
  - 트리의 선택은 루트 전술 검사(VCT 안전, VCT2 거부)에서 바뀔 수 있다. 트리 선택의 동률 처리에도 정책 prior가 들어간다(방문, Q, prior 순).
  - 그래서 트리 선택과 최종 착수가 달라진 건수를 탐색마다, 집합마다(E, T, V, V-A, V-B) 따로 보고한다.
- 방문한 자식이 둘 미만이면 V-B가 아니다. prior 1순위가 없으면(tree 밖) V-A가 아니다.
- 각 집합의 국면 수, 대국 수, 착수 불일치 수와 비율을 보고한다.

## 5. Q1: 수

- **분모**: V 안에서 S400과 S50의 최종 착수가 다른 국면(관측 하나 = (game, ply, S400 수, S50 수)). T와 E의 불일치도 같은 방법으로 보고하지만 판독하지 않는다.
- **판정**: R2-D4 §6의 Rapfi continuation을 그대로 쓴다.
  - 고정 수 뒤부터 양쪽 모두 Rapfi가 끝까지 둔다. 분기마다 새 프로세스. 점수는 고정 수를 둔 쪽 기준(승 1, 무 0.5, 패 0).
  - Rapfi executable·모델·설정 SHA, INFO, rule 4, `max_node=300`, thread 1, 응답 상한 600초는 R2-D4 계약의 identity와 같아야 한다.
  - 같은 분기(같은 history와 고정 수)는 한 번만 실행한다. 모든 continuation을 Game으로 재생해 검증한다(`h7r_evidence.branch_outcome`).
- **INVALID**: R2-D4 §7(`h7r-invalid-v1`). 구조적 abnormal_end만 0.5로 정규화하고, 실행 실패는 미해결 분기로 남긴다. 데이터 오류는 멈춘다.
- **지표**: `delta = score(S400 수) − score(S50 수)`, 평균(동일 결과는 0으로 포함). 미해결 분기는 하한(S400 쪽 0, S50 쪽 1)과 상한(반대)으로 계산한다(`h7r_d4.selection_stats`).

## 6. Q2: 값

- **분모**: V의 모든 국면(수가 갈렸는지와 무관). T 전체도 보고하지만 판독하지 않는다.
- **기준 결과 r**: 국면에서 고정 수 없이 Rapfi가 첫 수부터 양쪽을 끝까지 둔 결과, 그 국면의 차례 쪽 점수(승 1, 무 0.5, 패 0).
  - 위치 분기는 Q1의 수 분기와 키가 다르다(`sha({'position_branch': history})`, 수 분기는 `sha([history, move])`).
  - 설정·재생 검증·INVALID 규칙은 Q1과 같다. Rapfi의 첫 수가 있으면 그 수를 고정 수 자리에 두고 R2-D4 검증기로 재생한다(차례 쪽은 그 수 전의 차례). 첫 수 전에 실패하면 국면이 끝나지 않았고 사유가 실행 실패인지, 그 세부가 판과 맞는지를 따로 확인한다.
- **값**(모두 차례 쪽 점수 척도 [0, 1])
  - `v_c5 = (c5 원값 + 1) / 2`: c5 value 망은 차례 쪽 관점의 [−1, 1] 값을 낸다.
  - `v_S400 = (Q + 1) / 2`: Q는 S400 루트에서 방문이 있는 자식들의 Q를 방문 수로 가중 평균한 값이다. 루트 자식 Q는 루트의 차례 쪽(루트에서 수를 둔 쪽) 관점이다(`puct_value_v1.backup`). 뒤따를 학습 계약의 value 목표 후보와 같은 정의다.
  - 참고: `v_S50`(같은 정의), H-dev 기존 value label(차례 쪽 +1/0/−1을 같은 척도로). label은 c5 actor 대국에서 나와 독립 판정이 아니다.
- **지표**: 국면마다 `d = (v_c5 − r)² − (v_S400 − r)²`의 평균. 양수면 S400 값의 제곱오차가 작다.
  - d는 r에 대해 선형이다. 미해결 r은 d(0)과 d(1) 가운데 작은 값을 하한, 큰 값을 상한으로 둔다. 구조적 정규화는 r = 0.5.
- **함께 보고**: 해결된 국면의 c5·S400·S50 제곱오차 평균, `v_S50` 대비 같은 지표, `|v_S400 − v_c5|` 평균, H-dev label 대비 제곱오차(참고).

## 7. 통계

- source-game cluster bootstrap 20,000회, percentile, seed **20261151**. 396판의 같은 multiplicity를 모든 지표에 쓴다.
- 판독(Q1, Q2 각각, 단측 α 0.025 = 양측 95% 구간)
  - **BETTER**: 하한 지표(미해결을 S400에 불리하게 둔 값)의 구간 하한 > 0
  - **WORSE**: 상한 지표의 구간 상한 < 0
  - **INCONCLUSIVE**: 그 밖이거나 최소 조건 미달
- full 최소 조건(지표마다): 분모 100개 이상, 그 분모가 나온 대국 30판 이상, 미해결 5% 이하(정확한 분수 비교), bootstrap 분모 0 회차 없음. actor 재현·identity·Game 재생은 통과가 전제다.
- 지표마다 전체·해결·미해결·정규화 건수와 대국 수를 기록한다.
- Q1과 Q2는 §8에서 둘 다 성립해야 하는 교집합 주장이라 각각 95% 구간으로 판정한다(intersection–union). 보조 지표는 보고만 한다.
- 계획값(판정 문턱 아님): smoke 뒤 각 집합의 비율로 full의 국면 수를 추정해 기록한다(`projection`). 기준과 표본은 바꾸지 않는다.

## 8. 학습 계약 진입 기준 (사전 고정)

| Q1 (V, 착수 불일치) | Q2 (V 전체) | 다음 |
|---|---|---|
| BETTER | BETTER | value 목표만 바꾸는 학습 계약 초안을 연다: 학습 행·정책·모델 구조·S50 예산을 c5와 같게 두고 value 목표만 S400 루트 값(정의 §6)으로 바꾼다. 기력은 독립 확인 Arena에서 판단한다 |
| BETTER | 그 밖 | 열지 않는다. 깊은 탐색은 그 국면에서 더 좋은 수를 고르지만, 그 값이 더 나은 목표라는 근거는 없다 |
| 그 밖 | BETTER | 열지 않는다. 값은 고정 결과에 더 가깝지만, 깊은 탐색이 그 국면에서 더 좋은 수를 고른다는 근거가 없다 |
| 그 밖 | 그 밖 | S400 값 목표 방향을 멈춘다 |

- WORSE가 나오면 따로 적는다. 진입 여부는 위 표대로다.
- 결과를 보고 V의 정의, 문턱, 표본, 판정 설정을 바꾸지 않는다. 바꾸려면 새 계약이다.

## 9. 하지 않는 것

- 학습, self-play, 자유 대국 Arena, 승급, champion·canonical·웹 변경, V9 진입
- S200 비교, 결과에 따른 표본 보충, seed 교체, 문턱 조정

## 10. 단계와 비용

| 단계 | 하는 일 | 앞 단계 | 예상 시간 |
|---|---|---|---|
| prepare | H-dev 입력, D4 목록 재구성·대조, 모델(SHA, frozen state), runtime, Rapfi identity | `approved.prepare` | 수 분 |
| smoke | 실제 CUDA에서 재생·분류·두 종류의 분기·재생 검증·기록이 이어지는지 확인 | `approved.smoke`, 커밋된 prepare 기록 | 수십 분 |
| full | 396판 재생, 분기, 통계, 진입 판정 | `approved.full`, 커밋된 smoke(재현, 두 분기 종류 모두 실행) | 약 4–5시간 |

- **smoke 표본 선정 규칙**(Rapfi 결과를 열기 전에 고정): game index 오름차순으로 2판을 재생한다. 그 판들에서 V의 착수 불일치가 하나도 없으면 다음 판을 하나씩 더 재생한다(최대 10판, 넘으면 멈춤). 분기는 V 착수 불일치 가운데 (game, ply) 순서로 처음 4개의 두 수 분기, V 국면 가운데 처음 4개의 위치 분기다. 재생 결과만으로 정해진다. smoke는 판독하지 않는다(`SMOKE`).
- full 시간 추정(worker 6, D1 D4 full 기록 기준): S50 재생 약 40분, S400 재생 그 약 4.94배(약 3.2시간), Rapfi 분기(E 불일치마다 최대 2개, T 국면마다 1개, D1 D4는 1,988개에 9.5분) 30–60분.
- 원자료(재생 journal, 분기 기록)는 `runs/h7r/R2-S400-VT-MEASURE-1/<단계>/`에 두고 커밋하지 않는다. prepare 기록, smoke·full 요약, 로그, manifest를 커밋한다.

## 11. 결정 (2026-11-18 검토)

1. V-B 문턱 **20% 유지**(탐색 경합을 고르는 사전 기준). 방문 수만으로 계산, 방문한 자식 둘 미만은 V-B 아님.
2. Q2 기준 **고정 수 없는 Rapfi continuation**. H-dev label과 S50 루트 값은 참고.
3. full 최소 조건 **지표별 100개·30판, 미해결 5% 이하**. smoke 뒤 예상 국면 수를 기록.
4. 계약 id **`R2-S400-VT-MEASURE-1`**, bootstrap seed **20261151**.
5. 두 주 지표가 모두 BETTER일 때만 학습 계약 초안을 연다.

## 12. 구현 (2026-11-18, 미승인)

- 도구 `scripts/h7r_vt_measure.py`(R2-D4 도구의 입력 대조·journal·분기 실행·검증·통계 함수를 재사용)
  - `prepare --rapfi-dir … --log-name …`: 입력 전체를 대조하고 `prepare.json`을 쓴다.
  - `run --mode smoke|full …`: 재생 → 분류 → 분기(수, 위치) → 통계(full) → 요약. 재생과 분기는 journal로 이어서 한다(같은 실행 identity만, HEAD 포함). 요약 발행 직전에 HEAD, 계약, 앞 단계 기록, 입력 전체(H-dev, 목록, 모델, runtime, Rapfi)를 다시 대조하고, 하나라도 다르면 아무것도 쓰지 않는다.
- 고정 계약 `vt-measure-contract.json`: R2-D4 계약(SHA 고정)의 H-dev·frozen state·runtime·Rapfi 값, D4 목록 SHA, 분류·값·통계·smoke·INVALID 정책 값, 도구와 재사용 파일의 SHA(sources). 실험 SHA는 승인 플래그를 뺀 계약의 SHA다.
- 테스트 `tests/test_h7r_vt_measure.py`
  - 분류: V-A, V-B 경계(정확히 20%는 포함), 방문 자식 하나는 V-B 아님, tree 밖 제외, 트리 선택과 최종 착수 차이 집계
  - 값: 실제 `search_tree_puct_value`에서 차례 쪽이 이기는 국면의 S400 루트 값 > 0.9(흑 차례, 백 차례 각각), 척도와 방문 없는 자식 제외
  - 위치 분기: 수 분기와 다른 키, 가짜 Piskvork 엔진으로 흑 차례·백 차례 실행과 재생 검증(점수, 차례 쪽, 수순 변조 거부), 첫 수 전 실패(엔진 종료, 금수 불일치)와 그 세부 변조 거부, 수 분기는 R2-D4 검증 유지
  - 통계: Q1 보수적 하한 판독, 대국 수·미해결 비율 조건, Q2 미해결 r의 d(0)/d(1) 범위와 구조적 0.5, 진입 표
  - 실행(임시 저장소, 탐색·Rapfi는 대역): prepare → smoke(2판에 V 불일치가 없어 3판째 추가) → full, 단계 순서와 승인 거부, 재생·분기 중단 뒤 재개(끝난 것은 다시 하지 않음), 다른 HEAD에서 재개 거부, 실행 중 HEAD 변경·입력 변경(목록, 모델, Rapfi, H-dev)·앞 단계 기록 변경 시 발행 없음
  - 블록(실제 bash, 도구는 대역): §13

## 13. 데스크톱 실행 (승인 뒤에만)

- vt-P는 `approved.prepare` 뒤, vt-S는 `approved.smoke` 뒤, vt-F는 `approved.full` 뒤에 실행한다. 도구가 승인과 앞 단계 기록을 확인한다.
- 실행 중단: 결과 전에 멈추면 같은 블록을 다시 실행한다. 실행 identity에 HEAD가 들어가므로, 이어서 할 때는 pull하지 않고 HEAD가 그 실행의 HEAD와 같아야 한다.
- 실패(종료 코드 0이 아님): smoke·full은 아무것도 커밋하지 않고 멈춘다(HEAD 유지). 원인을 확인한 뒤 다시 실행하면 이어서 한다. 실패 로그는 요약과 함께 커밋된다. prepare는 실패 로그를 커밋하고 멈춘다.
- 결과가 커밋 전에 남아 있으면(결과 뒤 중단, 스테이징 뒤 중단) 다시 실행하지 않는다. 결과에 적힌 로그 이름의 로그가 있어야 한다. 로그에 종료 줄이 있으면 마지막 종료 줄이 `__EXIT__ 0`이어야 하고(아니면 멈춤), 없으면 도구가 결과를 마지막에 쓰고 0으로 끝나므로 `Recovered exit code`와 `__EXIT__ 0`을 붙여 함께 커밋한다.
- full은 tmux에서 실행한다(약 4–5시간).

vt-P. prepare

```bash
(
  set -euo pipefail
  cd ~/renju-h6
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src:$PWD/scripts"
  [ "$(git branch --show-current)" = feat/h7-c1 ] || { echo "STOP: wrong branch"; exit 1; }
  git diff --quiet || { echo "STOP: unstaged tracked changes"; exit 1; }
  if git diff --cached --quiet; then git pull --ff-only origin feat/h7-c1; fi
  D=docs/h7r/r2/R2-S400-VT-MEASURE-1
  R=runs/h7r/R2-S400-VT-MEASURE-1
  L="$R/logs"
  O="$D/prepare.json"
  T=scripts/h7r_vt_measure.py
  tracked(){ git cat-file -e "HEAD:$1" 2>/dev/null; }
  if tracked "$O"; then echo "STOP: prepare is already recorded"; exit 1; fi
  mkdir -p "$L"
  command -v flock > /dev/null || { echo "STOP: flock is missing (util-linux)"; exit 1; }
  exec 9> "$L/prepare.lock"
  flock -n 9 || { echo "STOP: another prepare block is running"; exit 1; }
  if [ ! -e "$O" ]; then
    git diff --cached --quiet || { echo "STOP: resolve staged changes before a new run"; exit 1; }
    python scripts/check_track_b_frozen.py
    log="$(mktemp "$L/prepare-XXXXXXXX.log")"
    set +e
    python -u "$T" prepare --rapfi-dir /mnt/c/engines/rapfi --log-name "$(basename "$log")" 2>&1 | tee "$log"
    codes=("${PIPESTATUS[@]}")
    set -e
    rc="${codes[0]}"
    [ "${codes[1]}" -eq 0 ] || { echo "STOP: tee failed; log: $log"; exit 1; }
    echo "__EXIT__ $rc" >> "$log"
  else
    echo "an uncommitted $O exists: recording it (no new run)"
    name="$(python -c "import json, sys; print(json.load(open(sys.argv[1]))['log_name'])" "$O")"
    [ -f "$L/$name" ] || { echo "STOP: the log $name of $O is missing: report it"; exit 1; }
    if grep -q '^__EXIT__ ' "$L/$name"; then
      [ "$(grep '^__EXIT__ ' "$L/$name" | tail -n 1)" = "__EXIT__ 0" ] || { echo "STOP: $O and its log disagree on the exit code"; exit 1; }
    else  # the tool publishes its record last and then exits 0: the record gives the exit code
      printf '\nRecovered exit code from %s\n__EXIT__ 0\n' "$O" >> "$L/$name"
    fi
    rc=0
  fi
  paths=()
  shopt -s nullglob
  for f in "$L"/prepare-*.log; do
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
  git commit -m "H7-R VT-MEASURE: prepare exit $rc (records only)"
  git push origin feat/h7-c1
  echo "__RECORDED__ $rc"
  exit "$rc"
)
```

vt-S. smoke

```bash
(
  set -euo pipefail
  cd ~/renju-h6
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src:$PWD/scripts"
  [ "$(git branch --show-current)" = feat/h7-c1 ] || { echo "STOP: wrong branch"; exit 1; }
  MODE=smoke
  D=docs/h7r/r2/R2-S400-VT-MEASURE-1
  R=runs/h7r/R2-S400-VT-MEASURE-1
  L="$R/logs"
  RAW="$R/$MODE"
  O="$D/$MODE/summary.json"
  T=scripts/h7r_vt_measure.py
  tracked(){ git cat-file -e "HEAD:$1" 2>/dev/null; }
  if tracked "$O"; then echo "STOP: $MODE is already recorded"; exit 1; fi
  mkdir -p "$L"
  command -v flock > /dev/null || { echo "STOP: flock is missing (util-linux)"; exit 1; }
  exec 9> "$L/$MODE.lock"
  flock -n 9 || { echo "STOP: another $MODE block is running"; exit 1; }
  git diff --quiet || { echo "STOP: unstaged tracked changes"; exit 1; }
  if [ ! -e "$O" ] && [ -f "$RAW/identity.json" ]; then
    head="$(python -c "import json, sys; print(json.load(open(sys.argv[1]))['git']['commit'])" "$RAW/identity.json")"
    [ "$(git rev-parse HEAD)" = "$head" ] || { echo "STOP: HEAD moved since this $MODE run started ($head); restore it or report"; exit 1; }
    echo "resuming the $MODE run at $head (no pull)"
  elif [ ! -e "$O" ] && git diff --cached --quiet; then
    git pull --ff-only origin feat/h7-c1
  fi
  if [ ! -e "$O" ]; then
    git diff --cached --quiet || { echo "STOP: resolve staged changes before a new run"; exit 1; }
    python scripts/check_track_b_frozen.py
    log="$(mktemp "$L/$MODE-XXXXXXXX.log")"
    set +e
    python -u "$T" run --mode "$MODE" --device cuda --workers 6 --rapfi-dir /mnt/c/engines/rapfi --out-dir "$RAW" \
      --log-name "$(basename "$log")" 2>&1 | tee "$log"
    codes=("${PIPESTATUS[@]}")
    set -e
    rc="${codes[0]}"
    [ "${codes[1]}" -eq 0 ] || { echo "STOP: tee failed; log: $log"; exit 1; }
    echo "__EXIT__ $rc" >> "$log"
    if [ "$rc" -ne 0 ]; then
      echo "STOP: the $MODE run failed (exit $rc); nothing committed, HEAD kept: report it, or rerun this block to resume"
      exit "$rc"
    fi
  else
    echo "an uncommitted $O exists: recording it (no new run)"
    name="$(python -c "import json, sys; print(json.load(open(sys.argv[1]))['log_name'])" "$O")"
    [ -f "$L/$name" ] || { echo "STOP: the log $name of $O is missing: report it"; exit 1; }
    if grep -q '^__EXIT__ ' "$L/$name"; then
      [ "$(grep '^__EXIT__ ' "$L/$name" | tail -n 1)" = "__EXIT__ 0" ] || { echo "STOP: $O and its log disagree on the exit code"; exit 1; }
    else  # the tool publishes its record last and then exits 0: the record gives the exit code
      printf '\nRecovered exit code from %s\n__EXIT__ 0\n' "$O" >> "$L/$name"
    fi
    rc=0
  fi
  paths=()
  shopt -s nullglob
  for f in "$L/$MODE"-*.log; do
    n="$(basename "$f" .log)"
    tracked "$D/stage-logs/$n.log" && continue
    python scripts/h7r_d4_records.py copy-logs --resume --src-dir "$L" --pattern "$n.log" \
      --dest-dir "$D/stage-logs" --manifest "$D/stage-logs/$n.manifest.json"
    paths+=("$D/stage-logs/$n.log" "$D/stage-logs/$n.manifest.json")
  done
  paths+=("$O")
  while IFS= read -r -d '' staged; do
    allowed=false
    for f in "${paths[@]}"; do if [ "$f" = "$staged" ]; then allowed=true; break; fi; done
    "$allowed" || { echo "STOP: unrelated staged path: $staged"; exit 1; }
  done < <(git diff --cached --name-only -z)
  expect=()
  for f in "${paths[@]}"; do expect+=(--expect "$f"); done
  git add -- "${paths[@]}"
  python scripts/h7r_d4_records.py verify-staged "${expect[@]}"
  git commit -m "H7-R VT-MEASURE: $MODE exit $rc (records only)"
  git push origin feat/h7-c1
  echo "__RECORDED__ $rc"
  exit "$rc"
)
```

vt-F. full (tmux)

```bash
(
  set -euo pipefail
  cd ~/renju-h6
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src:$PWD/scripts"
  [ "$(git branch --show-current)" = feat/h7-c1 ] || { echo "STOP: wrong branch"; exit 1; }
  MODE=full
  D=docs/h7r/r2/R2-S400-VT-MEASURE-1
  R=runs/h7r/R2-S400-VT-MEASURE-1
  L="$R/logs"
  RAW="$R/$MODE"
  O="$D/$MODE/summary.json"
  T=scripts/h7r_vt_measure.py
  tracked(){ git cat-file -e "HEAD:$1" 2>/dev/null; }
  if tracked "$O"; then echo "STOP: $MODE is already recorded"; exit 1; fi
  mkdir -p "$L"
  command -v flock > /dev/null || { echo "STOP: flock is missing (util-linux)"; exit 1; }
  exec 9> "$L/$MODE.lock"
  flock -n 9 || { echo "STOP: another $MODE block is running"; exit 1; }
  git diff --quiet || { echo "STOP: unstaged tracked changes"; exit 1; }
  if [ ! -e "$O" ] && [ -f "$RAW/identity.json" ]; then
    head="$(python -c "import json, sys; print(json.load(open(sys.argv[1]))['git']['commit'])" "$RAW/identity.json")"
    [ "$(git rev-parse HEAD)" = "$head" ] || { echo "STOP: HEAD moved since this $MODE run started ($head); restore it or report"; exit 1; }
    echo "resuming the $MODE run at $head (no pull)"
  elif [ ! -e "$O" ] && git diff --cached --quiet; then
    git pull --ff-only origin feat/h7-c1
  fi
  if [ ! -e "$O" ]; then
    git diff --cached --quiet || { echo "STOP: resolve staged changes before a new run"; exit 1; }
    python scripts/check_track_b_frozen.py
    log="$(mktemp "$L/$MODE-XXXXXXXX.log")"
    set +e
    python -u "$T" run --mode "$MODE" --device cuda --workers 6 --rapfi-dir /mnt/c/engines/rapfi --out-dir "$RAW" \
      --log-name "$(basename "$log")" 2>&1 | tee "$log"
    codes=("${PIPESTATUS[@]}")
    set -e
    rc="${codes[0]}"
    [ "${codes[1]}" -eq 0 ] || { echo "STOP: tee failed; log: $log"; exit 1; }
    echo "__EXIT__ $rc" >> "$log"
    if [ "$rc" -ne 0 ]; then
      echo "STOP: the $MODE run failed (exit $rc); nothing committed, HEAD kept: report it, or rerun this block to resume"
      exit "$rc"
    fi
  else
    echo "an uncommitted $O exists: recording it (no new run)"
    name="$(python -c "import json, sys; print(json.load(open(sys.argv[1]))['log_name'])" "$O")"
    [ -f "$L/$name" ] || { echo "STOP: the log $name of $O is missing: report it"; exit 1; }
    if grep -q '^__EXIT__ ' "$L/$name"; then
      [ "$(grep '^__EXIT__ ' "$L/$name" | tail -n 1)" = "__EXIT__ 0" ] || { echo "STOP: $O and its log disagree on the exit code"; exit 1; }
    else  # the tool publishes its record last and then exits 0: the record gives the exit code
      printf '\nRecovered exit code from %s\n__EXIT__ 0\n' "$O" >> "$L/$name"
    fi
    rc=0
  fi
  paths=()
  shopt -s nullglob
  for f in "$L/$MODE"-*.log; do
    n="$(basename "$f" .log)"
    tracked "$D/stage-logs/$n.log" && continue
    python scripts/h7r_d4_records.py copy-logs --resume --src-dir "$L" --pattern "$n.log" \
      --dest-dir "$D/stage-logs" --manifest "$D/stage-logs/$n.manifest.json"
    paths+=("$D/stage-logs/$n.log" "$D/stage-logs/$n.manifest.json")
  done
  paths+=("$O")
  while IFS= read -r -d '' staged; do
    allowed=false
    for f in "${paths[@]}"; do if [ "$f" = "$staged" ]; then allowed=true; break; fi; done
    "$allowed" || { echo "STOP: unrelated staged path: $staged"; exit 1; }
  done < <(git diff --cached --name-only -z)
  expect=()
  for f in "${paths[@]}"; do expect+=(--expect "$f"); done
  git add -- "${paths[@]}"
  python scripts/h7r_d4_records.py verify-staged "${expect[@]}"
  git commit -m "H7-R VT-MEASURE: $MODE exit $rc (records only)"
  git push origin feat/h7-c1
  echo "__RECORDED__ $rc"
  exit "$rc"
)
```
