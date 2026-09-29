# v7 인간 대국 분석과 "v7 + 신경망" 설계 검토

작성일 2026-09-29. 입력: 사람 대 MCTS-v7 웹 대국 6판(`logs/web_play/2026092916*`, 커밋하지 않음)과
"v7을 교사·전술 안전망으로 쓰고 정책·가치망 MCTS로 넘어간다"는 설계 제안.
분석 도구: `scripts/analyze_web_play_losses.py`(읽기 전용, VCF 증명 기반).

## 0. 결론

| 제안의 주장 | 판정 | 근거 |
|---|---|---|
| 29수 패배는 24·26·28수 `forced_policy_stage=2`(탐색 0회) 때문이다 | **틀림** | §1.2. stage 2는 상대 4를 막는 유일한 수다. 22수 시점에 이미 모든 합법수가 VCF 패배였다 |
| 강제 규칙이 MCTS를 우회해 약점이 생긴다 | **부분적으로 맞음** | 대상은 stage 2가 아니라 stage 4/5. 패배 4판 중 3판의 마지막 분기점이 stage 4(탐색 0회) 착수였다 |
| 근본 원인 | **제안에 없음** | 사람의 공격은 삼을 섞은 연속 위협(VCT)이었다. v7은 VCF(4만)만 본다. 1판은 MCTS 100회(tactical) 착수 뒤에 졌다 |
| 지금 정책·가치망을 도입해야 한다 | **이미 되어 있음** | Stage 4~8: 정책·가치망, 독립 PUCT, self-play, 학습, 외부 평가가 완료·운영 중이다(B400) |
| 즉승·즉방은 hard rule, 금수는 엔진 | **이미 되어 있음** | PUCT v2 `search.tactics.tactical_filter`가 모든 노드에서 적용하고 root는 그래도 N회 탐색한다 |
| 확정 VCF도 hard rule로 | **보류(실험 arm)** | Stage 7-B에서 VCF/VCT 규칙을 학습 탐색에 넣지 않기로 결정. 대국(배포) 모드 전용은 허용 가능 |
| v7에 NN을 연결 | **하지 않음** | v7은 CI SHA 잠금 동결 benchmark다. 연결하면 측정 기준이 사라진다 |
| v7 기보로 NN을 사전학습(증류) | **타당, 새 실험 arm으로** | Stage 7 지도학습 sanity가 학습 가능성을 보였다. 단 v7 착수의 59%가 탐색 0회라 target 설계가 필요하다(§3.2) |
| `pi = visit 분포`, `z = ±1/0` | **이미 되어 있음** | Stage 5 계약(`pi=N/sum(N)`), Stage 6 target |
| NN이 v7을 추월하는 경로 | **아직 멀다** | 최신 B 계열은 v7 상대 heavy 평가 0/20(gen 320만 1/20). ★ 등급은 근거가 없다 |

수정된 방향: **v7은 연결 대상이 아니라 (1) 동결된 평가 상대, (2) 선택적 교사 데이터 공급원, (3) 사람 대국용
선택적 안전망이다.** 학습 경로는 기존 PUCT v2 + self-play를 유지하고, v7 증류는 B400 계열과 같은 판수로 비교하는
별도 arm으로 넣는다.

## 1. 대국 로그 사실 확인

### 1.1 요약

6판 중 사람 4승, v7 2승. v7 착수 75수 중 44수(59%)가 탐색 0회였다.

| 경로 | normal(50회) | tactical(100회) | stage 1 | stage 2 | stage 3 | stage 4 | stage 5 | own VCF |
|---|---|---|---|---|---|---|---|---|
| 착수 수 | 28 | 3 | 2 | 17 | 2 | 18 | 3 | 2 |

v7이 이긴 판(`164255`, 36수)의 30수 기록은 사실이다: `future_black_43_defense`, 안전성 탐색 2,243노드,
후보 20개 검사 중 16개 제거.

`_forced_v5_move`의 stage 의미(`src/search/mcts_v5.py`):

| stage | 조건 | 선택 | 탐색 |
|---|---|---|---|
| 1 | 내가 바로 5목 | 승리점 | 0 |
| 2 | 상대 승리점(4)이 있음 | 합법 방어점 | 0 |
| 3 | 내가 막을 수 없는 4(열린 4 등)를 만들 수 있음 | 그 수 | 0 |
| 4 | 상대가 막을 수 없는 4를 만들 수 있음(열린 3 등) | 위협 창의 빈 점 중 남는 위협 수 최소, v7 M4 동률 규칙 | 0 |
| 5 | 상대 쌍위협 생성점이 1개 | 그 점 | 0 |

v7은 stage 1~3을 먼저 반환하고, own VCF(M1)를 확인한 뒤 stage 4/5를 반환한다(`mcts_search_v7`).

### 1.2 `164723` (사람 흑 29수 승리) — 제안이 인용한 판

각 v7 착수 시점의 **VCF-safe 수**(두면 상대 VCF가 없는 합법수, 자기 4 반격 포함) 수:

| 수 | 경로 | 착수 | VCF-safe 수 | 판정 |
|---|---|---|---|---|
| 14 | stage 4 | (10,7) | 2 | 선택 수가 safe |
| 16 | stage 4 | (9,10) | 5 | 선택 수가 safe |
| 18 | stage 5 | (10,12) | 10 | 선택 수가 safe |
| 20 | tactical 100회 | (11,9) | 206 | 선택 수가 safe |
| 22 | stage 4 | (9,7) | **0** | 이미 패배 |
| 24 / 26 / 28 | stage 2 | 방어 | 0 | 상대 4를 막는 유일한 수, 28수 직전 흑 승리점 2개 |

- 24·26·28수는 다른 어떤 수를 둬도 다음 수에 지는 자리다. MCTS가 몇 번을 탐색해도 결과가 같다.
- 20수에서는 VCF 기준으로 206수가 모두 안전했는데, 사람의 21수 뒤 22수에서는 0수가 되었다.
  즉 21수는 **삼을 섞은 위협**(VCT)이었고, 이것을 놓친 착수는 **MCTS 100회로 둔 20수(또는 그 이전)**다.

### 1.3 나머지 사람 승리 3판

| 판 | 사람 색 | 마지막 분기점(safe > 0) | 그 착수의 경로 | 다음 v7 착수 |
|---|---|---|---|---|
| `163810` | 백 | 15수, safe 4수 중 선택 | stage 4 | 17수 stage 4, safe 0 |
| `163903` | 백 | 13수, safe 4수 중 선택 | stage 4 | 15수 stage 4, safe 0 |
| `164445` | 흑 | 14수, safe 2수 중 선택 | stage 4 | 16수 stage 4, safe 0 |

세 판 모두 마지막 분기점의 v7 착수 자체는 VCF-safe였다. 그 뒤 사람이 삼을 포함한 위협을 만들었고, 다음 착수
시점에는 모든 수가 VCF 패배였다. stage 4의 방어 후보(위협 창 안의 빈 점) 전체가 VCF를 허용했고, safe였던
수는 자기 4 반격 2수뿐이었다(시간만 끄는 수).

따라서 이 로그만으로 "stage 4가 더 나은 방어를 두고 틀린 수를 골랐다"고 단정할 수 없다. 분기점의 safe 후보 각각에 대해
사람의 모든 응수 뒤에도 살아남는 수가 있었는지(3수 앞 확인)는 탐색 비용 때문에 이번에 확정하지 못했다.

### 1.4 해석

- "강제 규칙이 탐색을 우회한다"는 문제의식은 맞지만, 원인으로 지목한 stage 2는 틀렸다. stage 1~3은 규칙상 사실(1수
  앞 확정)이라 탐색해도 결과가 같다.
- 실제로 탐색 없이 **판단**하는 곳은 stage 4/5다. 2수 앞 방어를 "남는 위협 수 최소" 휴리스틱으로 고른다.
- 그러나 1판(`164723`)은 MCTS 100회로 둔 수 뒤에 졌다. v7의 MCTS(50~100회, 휴리스틱 rollout)도 VCT를 보지 못한다.
  **stage 4를 MCTS로 돌려도 이 약점이 사라진다는 근거는 없다.**
- 6판은 작은 표본이고, 같은 사람이 같은 수법(삼+사 연속 위협)으로 이긴 기록이다. "v7의 약점 = VCT 방어"까지만
  말할 수 있다.

## 2. 제안과 저장소 현황 대조

| 제안 구성요소 | 현재 저장소 | 위치 |
|---|---|---|
| 렌주 규칙 hard constraint | 엔진 `legal_moves`/`forbidden_reason`, 망 입력은 합법 mask | `src/renju/`, `src/model/masking.py` |
| Policy-Value Network | 6-plane 입력, residual 64×4 | Stage 4, `docs/policy-value-network.md` |
| PUCT MCTS | V6/V7과 독립된 `search.alphazero` | Stage 5, `docs/stage5-alphazero.md` |
| 즉승·즉방 hard rule + 그래도 탐색 | PUCT v2 `tactical_filter`: 즉승 제한·+1 증명, 유일 방어점 제한, 방어 불가 −1 증명. root는 정확히 N회 탐색 | `src/search/tactics.py`, Stage 7-B |
| VCF/VCT hard rule | **없음.** 7-B 결정: 2수 이상의 전술은 규칙으로 넣지 않고 학습으로 해결되는지 본다 | `docs/stage7-plan.md` §8.4 |
| `pi` = root visit 분포 | raw visit counts 저장, `pi=N/sum(N)` | Stage 5 §8 |
| `z` = 승 +1 / 무 0 / 패 −1 | player identity 기준 | Stage 6 §3 |
| self-play 반복 | 운영 중, 균형 샘플링 채택 | Stage 8 §12.6 |
| 세대별 평가 | 루프 내 + 외부 light/heavy(v321·v5·v6·v7), B400 anchor h2h, 라운드로빈 | Stage 8 §11·§12.8 |
| 사람 대국 로그의 root visits | **없음.** 웹 대국은 V2~V7만 지원하고 visit 분포를 남기지 않음 | `scripts/run_web_play.py` |

현재 위치: B400이 B 계열 1위, 400 → 480 향상 증거 없음(첫 정체 신호). v7 상대는 사실상 0승이다.
따라서 제안의 출발점("이제 NN을 붙이자")은 이미 지났고, 실제 질문은 **"정체를 v7 지식으로 풀 수 있는가"**다.

### 2.1 v7에 NN을 붙이지 않는 이유

- v7 구현 파일은 `tests/frozen_baseline.sha256`과 behavior fingerprint로 CI에 잠겨 있다. 고치면 과거 모든 비교가
  무효가 된다.
- NN의 향상을 v7 상대로 재는데, v7 안에 NN이 들어가면 상대와 피측정 대상이 섞인다.
- 제안이 원하는 구조(규칙 → PUCT → 사실 기반 안전망)는 이미 `search.alphazero` + PUCT v2다. v7의 stage 4/5·
  future 4-3·tactical score 같은 휴리스틱은 7-B 원칙대로 학습 탐색에 넣지 않는다.

## 3. 수정된 설계

### 3.1 기본 경로 유지

B400 + PUCT v2 + 균형 샘플링 장기 학습(§12.8)과 정체 판정(B400 anchor, 3회 연속 무향상)을 그대로 둔다.
정체가 확정되면 Stage 9 network 확대가 기본 선택지다. 아래 3.2는 그와 **나란히 비교하는** arm이다.

### 3.2 새 arm: v7 증류 warm start (Stage 7 "선택지 c")

근거: `run_stage7_supervised_sanity.py`에서 V7 대국의 전술 국면 1,098개만으로 held-out 필수 방어 top-1 0.67~0.78을
얻었다(self-play gen 30은 0.00). 학습 가능성은 확인되어 있다.

데이터:

1. **수집기는 frozen v7 파일을 수정하지 않는다.** 새 모듈(예: `training/teacher_v7.py`)이 v7 함수를 호출하고
   결과 착수만 기록한다. 수집기를 거친 대국의 착수가 frozen v7과 같다는 fingerprint 테스트를 먼저 둔다.
2. 대국: v7 대 v7, v7 대 v5/v6, 색 교환, 기존 오프닝 세트(50쌍)로 다양화한다. seed만 바꾼 v7 대 v7은 수순이
   거의 같을 수 있으므로 **고유 국면 수**를 기록하고 판단 근거로 쓴다.
3. policy target:
   - stage 1~3, own VCF: one-hot(규칙상 사실).
   - normal/tactical: root visit 분포가 좋지만 v7은 최대 20후보·50~100회라 해상도가 낮다. 수집기에서 root visit을
     얻을 수 없으면 one-hot + label smoothing으로 시작한다. **"visit 분포가 one-hot보다 훨씬 좋다"는 이 조건에서는
     검증되지 않았다.** 두 target을 probe로 비교한다.
   - stage 4/5: 이번 분석이 보여준 약점 지점이다. one-hot으로 넣으면 그 휴리스틱을 그대로 배운다. 기본은
     **가중치를 낮추거나 제외**하고, 포함/제외를 probe(`must_defend_open3`)로 비교한다.
4. value target: 최종 결과 z. 흑/백 승 비율이 치우치면 Gate 3의 균형 샘플링을 똑같이 적용한다.

학습과 비교:

1. 새 network(64×4, 같은 config)를 지도학습 → `training.init_checkpoint`로 PUCT v2 self-play **새 run**.
2. 대조: B400 계열 continuation. 같은 self-play 판수 지점끼리 라운드로빈 + heavy 사다리.
3. **판정에서 v7 승률을 주 지표로 쓰지 않는다.** v7 기보로 배운 모델은 v7 상대 성적이 부풀 수 있다. v5·v6·v321,
   라운드로빈, 사람 대국을 주 지표로, v7은 보조로 본다.
4. 비용 추정(실측 전): v7 착수당 약 0.02~0.9 s, 한 판 20~40수 → 1만 판에 단일 코어 수십 시간. 6코어 병렬로
   수집하고, 먼저 1,000판으로 probe 효과를 본 뒤 늘린다.

### 3.3 전술 안전망은 "대국 모드"에만

- 학습 탐색(self-play·in-loop 평가)은 PUCT v2 그대로 둔다(7-B 결정 유지).
- 사람 대국/최종 배포에서만 선택적으로 root 앞에 둔다:
  - own VCF가 **증명**되면 그 첫 수(증명된 승리이므로 hard rule 가능).
  - 상대 VCF 안전 필터는 bounded 탐색이라 inconclusive가 있다. hard 제거가 아니라 후보 순위 조정으로만 쓴다.
  - VCT는 증명 비용이 크고 불완전하므로 hard rule로 쓰지 않는다.
- 대국 모드 on/off는 외부 평가에서 별도 항목으로 기록한다(network 비교와 system 비교 분리, 7-B 방식).

### 3.4 휴리스틱 prior 보정은 보류

`P = (1−λ)·P_NN + λ·P_v7` 같은 혼합은 visit target이 휴리스틱을 따라가서 사실상 증류가 된다. 3.2가 같은 효과를
더 통제된 방식으로 준다. 3.2 결과가 나온 뒤에만 λ→0 annealing arm으로 검토한다.

### 3.5 v7 자체 개선은 새 버전으로만

stage 4/5를 "후보 제한 + 탐색 검증"으로 바꾸거나 VCT를 추가하는 것은 v8(새 파일)로만 가능하다. v7은 그대로 동결한다.
§1.4대로 탐색만으로는 VCT가 해결되지 않으므로 효과가 불확실하고, 프로젝트 목표(AlphaZero)와 거리가 있어 우선순위는 낮다.

## 4. 작업 순서와 완료 기준

| # | 작업 | 완료 기준 |
|---|---|---|
| 1 | 웹 대국에 AlphaZero checkpoint 에이전트 추가(PUCT v2, 대국 모드 옵션). 착수마다 root visits·value·prior top-k 기록 | B400과 사람 대국 가능, 로그로 `pi` 재구성 가능. 기존 V2~V7 대국 동작 불변 |
| 2 | 사람 대국 분석 도구를 NN 로그에도 적용 | `analyze_web_play_losses.py`가 NN 대국의 분기점을 출력 |
| 3 | v7 교사 데이터 수집기(frozen 파일 불변) | 수집 대국 fingerprint가 frozen v7과 일치, frozen-baseline CI PASS |
| 4 | 1,000판 수집 → 지도학습 → probe | target 변형(visit/one-hot, stage 4/5 포함/제외)별 probe 표 |
| 5 | 증류 init self-play run vs B400 continuation | 같은 판수 라운드로빈·heavy 사다리. v7 외 지표로 판정 |
| 6 | 결과에 따라 Stage 9(network 확대)에 증류 init을 쓸지 결정 | 결정과 근거를 이 문서와 Stage 8 계획에 기록 |

## 5. 한계

- 6판, 한 사람, 비슷한 수법. 일반화하지 않는다.
- "safe"는 VCF 탐색(max 20 fours, 20,000 노드) 기준이다. 탐색 한도를 넘은 경우는 VCF 없음으로 센다.
- 삼을 포함한 위협(VCT)은 직접 증명하지 않았다. "safe 여러 개 → 다음 착수 safe 0"으로 간접 확인했다.
