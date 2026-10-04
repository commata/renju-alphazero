# v7 인간 대국 분석과 "v7 + 신경망" 설계 검토 (2차)

작성 2026-09-29. 입력: 사람 대 MCTS-v7 웹 대국 6판(`logs/web_play/2026092916*`, 커밋하지 않음)과
"v7을 교사·전술 안전망으로 쓰고 정책·가치망 MCTS로 넘어간다"는 설계 제안, 그리고 1차 검토에 대한 반론.
분석 도구: `scripts/analyze_web_play_losses.py`(읽기 전용, SAFE/UNSAFE/UNKNOWN 3상태).

2차에서 바뀐 점:

1. 분석기를 3상태로 바꾸고 다시 실행했다. 1차의 "safe"는 탐색 한도에 걸려도 safe로 세었다.
2. "근본 원인은 VCT"를 **검증할 가설**로 낮췄다.
3. stage 2 정의를 코드대로 고쳤다(상대 승리점이 여러 개여도 stage 2).
4. "v7 증류"를 **해법이 증명된 전술 국면만 쓰는 bootstrap**으로 좁혔다. 다음 작업 순서를 진단 → 측정 → 실험 순으로 다시 짰다.

## 0. 결론

| 주장 | 판정 | 근거 |
|---|---|---|
| 29수 패배는 24·26·28수 stage 2(탐색 0회) 때문이다 | **틀림** | §1.3. 22수 시점에 모든 합법수가 VCF로 패배함이 증명된다(safe 0, unknown 0) |
| 강제 규칙이 MCTS를 우회해 약점이 생긴다 | **부분적으로 맞음** | 대상은 stage 4/5다. 패배 4판 중 3판에서 마지막으로 safe 수가 남아 있던 착수가 stage 4였다 |
| 근본 원인은 VCT | **깊이 1에서 확인** | §6.1: 네 판 모두 v7의 분기 착수가 "위협 1수 + VCF"로 지는 수로 증명됐고, 사람의 실제 다음 수가 증명된 승리 위협이었다. stage 4 3판에는 VCT1로도 지지 않는 대안이 있었다(가지치기 없는 solver로 재검증 완료, §8) |
| 이제 정책·가치망을 도입해야 한다 | **이미 되어 있음** | Stage 4~8(정책·가치망, 독립 PUCT, self-play, 학습, 외부 평가) 운영 중, 현재 B400 |
| 즉승·즉방 hard rule + 그래도 탐색 | **이미 되어 있음** | PUCT v2 `tactical_filter`가 모든 노드에 적용되고, root는 정확히 N회 탐색한다 |
| v7에 NN을 연결 | **하지 않음** | `tests/frozen_baseline.sha256`에 `mcts_v7.py`·`mcts_v7_agent.py`가 잠겨 있는 동결 benchmark다 |
| v7 기보 전체를 모방(증류) | **하지 않음** | v7의 약점(stage 4/5, VCT를 못 봄)까지 복사한다 |
| 해법이 증명된 전술 국면으로 bootstrap | **실험 arm으로 채택** | Stage 7 sanity: V7 대국의 전술 국면 1,098개로 held-out must_block top-1 0.78(1,000 step)·0.67(1,500 step), self-play gen 30은 0.00 |
| 64×4가 정체했다 | **첫 신호뿐** | B400 → B480 한 번. 판정 기준(B400 anchor, heavy 3회 연속 무향상)은 아직 충족되지 않았다 |

**v7의 역할:** 동결된 평가 상대이자, 학습용 전술 국면을 **뽑아 올 원천**이다. 정답(target)은 v7의 착수가 아니라
규칙과 증명에서 얻는다(§3.2).

## 1. 대국 로그 사실 확인

### 1.1 집계

6판 중 사람 4승, v7 2승이다. v7 착수 75수 중 44수(59%)가 탐색 0회였다.

| 경로 | normal(50회) | tactical(100회) | stage 1 | stage 2 | stage 3 | stage 4 | stage 5 | own VCF |
|---|---|---|---|---|---|---|---|---|
| 착수 수 | 28 | 3 | 2 | 17 | 2 | 18 | 3 | 2 |

v7이 이긴 판(`164255`, 백 36수)의 30수 기록: `future_black_43_defense`, 안전성 탐색 2,243노드, 후보 20개 중 16개 제거.

### 1.2 stage 정의(코드 기준)

`_forced_v5_move`(`src/search/mcts_v5.py`):

| stage | 조건 | 선택 | 성격 |
|---|---|---|---|
| 1 | 내가 바로 5목 | 승리점 | 사실 |
| 2 | 상대 즉시 승리점이 **1개 이상**이고 그중 합법인 점이 있음 | 그중 key가 가장 작은 점 | 승리점 1개면 유일한 비패배수(사실). 2개 이상이면 이미 진 국면(stage 1이 없었으므로)이고 선택은 무의미 |
| 3 | 내가 막을 수 없는 4를 만들 수 있음 | 그 수 | 사실(`_is_unstoppable_four`가 상대의 모든 합법 방어와 상대 즉승을 확인) |
| 4 | 상대가 막을 수 없는 4를 만들 수 있음 | 위협 창의 빈 점 중 남는 위협 수가 최소인 수 + v7 M4 동률 규칙 | 휴리스틱 |
| 5 | 상대의 쌍위협 생성점이 1개 | 그 점 | 휴리스틱 |

v7 순서: stage 1~3이면 바로 반환 → own VCF(M1)가 있으면 반환 → stage 4/5 반환 → 그 외 50/100회 MCTS.

### 1.3 3상태 분석 결과

분석 조건: 각 v7 착수 시점의 모든 합법수를 분류했다(VCF 노드 한도 100,000, 내 4 연쇄 한도 6).
v7 VCF solver는 `max_fours` 한도에 걸려도 "소진"으로 보고하지 않는다. 그래서 `max_fours`를 빈 칸 수/2 + 1로
두어 이 한도에 도달할 수 없게 했다. 따라서 남은 한도는 노드 한도 하나이고, 이 한도는 solver가 보고한다.

- **UNSAFE**: 상대에게 VCF(증명 수순)나 즉시 5목이 있다.
- **SAFE**: VCF 탐색이 끝까지 돌았고 VCF가 없다.
- **UNKNOWN**: 노드 한도 또는 연쇄 한도에 걸렸다.

**전 구간 unknown = 0.** 1차 수치는 모두 그대로 유지되고, 이제 "탐색이 끝까지 돈 결과"로 읽을 수 있다.
"패배 증명"은 safe = 0이고 unknown = 0인 경우에만 말한다.

| 판(사람 색) | 착수 | 경로 | 상대 승리점 | safe / unknown / unsafe | 판정 |
|---|---|---|---|---|---|
| `163810`(백) | 13 | stage 5 | 0 | 213 / 0 / 0 | 선택 수 SAFE |
| | **15** | **stage 4** | 0 | 4 / 0 / 207 | 선택 수 SAFE(safe는 4개뿐) |
| | 17 | stage 4 | 0 | 0 / 0 / 209 | 패배 증명 |
| | 19 | stage 2 | 2 | 0 / 0 / 206 | 패배 증명 |
| `163903`(백) | **13** | **stage 4** | 0 | 4 / 0 / 209 | 선택 수 SAFE |
| | 15 | stage 4 | 0 | 0 / 0 / 211 | 패배 증명 |
| | 17·19 / 21 | stage 2 | 1 / 2 | 0 / 0 / * | 패배 증명 |
| `164445`(흑) | 12 | stage 4 | 0 | 2 / 0 / 212 | 선택 수 SAFE |
| | **14** | **stage 4** | 0 | 2 / 0 / 210 | 선택 수 SAFE |
| | 16 | stage 4 | 0 | 0 / 0 / 210 | 패배 증명 |
| | 18·20 / 22 | stage 2 | 1 / 2 | 0 / 0 / * | 패배 증명 |
| `164723`(흑) | 18 | stage 5 | 0 | 10 / 0 / 198 | 선택 수 SAFE |
| | **20** | **tactical 100회** | 0 | 206 / 0 / 0 | 선택 수 SAFE |
| | 22 | stage 4 | 0 | 0 / 0 / 204 | 패배 증명 |
| | 24·26 / 28 | stage 2 | 1 / 2 | 0 / 0 / * | 패배 증명 |

- 굵게 표시한 수가 마지막으로 SAFE 수가 남아 있던 착수다. 3판은 stage 4, 1판은 MCTS 100회였다.
- 패배가 증명된 뒤에도 stage 2(승리점 1개)가 나온다(`163903` 17·19수 등). 착수 자체는 맞지만 이미 진 국면이다.
- 마지막 SAFE 착수에서 v7이 고른 수는 모두 SAFE였다. 그 뒤 사람이 한 수 두자 모든 후보가 UNSAFE가 됐다.
  따라서 v7의 실수는 "VCF를 허용하는 수를 골랐다"가 아니라 **"다음 수에 VCF 위협을 만드는 상대의 수를 미리 막지 못했다"**이다.

### 1.4 해석과 한계

- **SAFE의 범위:** 동결된 v7 solver 기준이다. 이 solver는 방어자의 강제 방어가 4를 만드는 수순을 건너뛴다.
  그래서 SAFE는 "그 범위의 VCF가 없다"는 뜻이지 "강제승이 없다"는 뜻이 아니다. VCT는 아예 탐색하지 않는다.
- **VCT 가설:** "VCF 없음 → 사람 한 수 → 모든 후보 VCF 패배"라는 흐름은 삼을 섞은 forcing 공격과 맞는다.
  이후 VCT solver certificate로 깊이 1에서 확인했다(§6.1).
- **stage 4:** 마지막 SAFE 착수에서 SAFE 후보가 2~4개뿐이었다(`164723` 20수 제외). 이 표만으로는 판단할 수 없었지만,
  §6.1에서 stage 4 3판 모두 VCT1로도 지지 않는 대안이 있음을 확인했다(가지치기 없는 solver로 재검증, §8).
- **`164723` 20수는 MCTS 100회로 둔 수다.** 탐색을 거쳐도 같은 종류의 실수가 나왔다. stage 4를 MCTS로 바꾸는 것만으로
  해결된다는 근거는 없다.
- 6판, 사람 한 명, 비슷한 수법이다. 일반화하지 않는다.

## 2. 제안과 저장소 현황 대조

| 제안 구성요소 | 현재 | 위치 |
|---|---|---|
| 렌주 규칙 hard constraint | 엔진 합법수, 망 입력은 전체 합법 mask | `src/renju/`, `src/model/masking.py` |
| Policy-Value Network | 6-plane, residual 64×4 | Stage 4 |
| PUCT MCTS | V6/V7과 독립된 `search.alphazero` | Stage 5 |
| 즉승·유일 방어 hard rule + 탐색 | PUCT v2 `tactical_filter`, 모든 노드, root N회 | `src/search/tactics.py`, Stage 7-B |
| VCF/VCT를 탐색 규칙으로 | 없음(7-B 결정) | `docs/stage7-plan.md` §8.4 |
| `pi` = visits, `z` = ±1/0 | 구현됨 | Stage 5 §8, Stage 6 §3 |
| 세대별 평가 | light/heavy 사다리, B400 anchor h2h, 라운드로빈 | Stage 8 §11, §12.8 |
| 사람 대국 NN 로그 | 없음(웹 대국은 V2~V7만 지원) | `scripts/run_web_play.py` |

## 3. 수정된 설계

### 3.1 지금 풀어야 할 질문

"신경망을 넣을까"가 아니다. **B400의 정체가 학습 신호(전술 표현) 부족 때문인지, 64×4 용량 부족 때문인지**를 구분하는
것이다. 대조군(B400 continuation)과 teacher arm(아래 tactical bootstrap)을 같은 self-play 판수로 비교해 가른다.

```
                 B400 (64×4, PUCT v2, 균형 샘플링)
                   │
        ┌──────────┴───────────┐
   Control arm             Teacher arm
   B400 continuation       B400 + 증명된 전술 국면 fine-tune
        │                   → PUCT v2 self-play
        └──────────┬───────────┘
            같은 판수, 같은 평가 세트
```

### 3.2 Teacher arm: 증명된 전술 국면 bootstrap

**원칙: 정답은 v7의 착수가 아니라 증명에서 얻는다.** v7 대국은 국면을 공급할 뿐이고, 각 국면의 target은
규칙이나 solver가 증명한 것만 쓴다. 그러면 v7의 휴리스틱 선택이 target에 섞이지 않는다.

| 국면 종류 | policy target | value target | 사용 |
|---|---|---|---|
| stage 1(즉승) | 승리점들(균등) | +1(증명) | 사용 |
| stage 2, 상대 승리점 1개 | 그 방어점 one-hot | 증명된 경우만(아래) | 사용 |
| stage 2, 상대 승리점 2개 이상 | **제외** | −1(증명) | value만 사용 |
| stage 3(막을 수 없는 4) | 그런 수들(균등) | +1(증명) | 사용 |
| own VCF 발견 | VCF 첫 수 one-hot | +1(증명) | 사용 |
| 분석기 LOST(VCF) 국면 | 제외 | −1(증명) | value만 사용 |
| stage 4 / 5 | 제외 | 제외 | 기본 제외 |
| normal / tactical MCTS | 제외 | 제외 | 1차 실험에서 제외 |

1차 제안과 사용자 수정안에서 더 바꾼 점:

1. **value는 대국 결과 z가 아니라 증명값을 쓴다.** 전술 국면은 승패가 증명되므로 v7 대 v7 결과(잡음)보다 정확하다.
   stage 2의 승리점 2개 이상 국면은 policy에서는 빼지만 value −1 target으로는 가치가 크다. Stage 7 sanity의
   `forced_loss`와 같은 종류다.
2. **normal/tactical MCTS 착수도 1차 실험에서 뺀다.** 동결 v7은 root visit을 밖으로 내보내지 않는다
   (`SearchDiagnostics`에는 `root_candidates`뿐이다). visit을 얻으려면 v7 코드를 복제해야 하고, 50~100회·최대 20후보라
   해상도도 낮다. one-hot으로 넣으면 v7 스타일 모방이 된다.
3. **국면 공급원을 v7에 한정하지 않는다.** v7/v5/v6 benchmark 기보, B 계열 self-play, 사람 대국에서 같은 규칙으로
   추출한다. 공급원별 개수를 기록한다.
4. **예상 효과를 낮게 잡는다.** stage 1과 유일한 stage 2는 PUCT v2가 탐색 중에 이미 100% 처리한다. 그래서 새 정보는
   주로 (a) raw policy/value가 이 국면을 알아 두어 그 **이전** 국면의 value가 좋아지는 효과, (b) stage 3·VCF 같은
   2수 이상 전술이다. 인간전에서 드러난 약점(열린 3·VCT 방어)은 이 set에 **들어 있지 않다**. 그것은 §3.3의 probe로 측정만 한다.
5. **초기화는 새 network가 아니라 B400이다.** 새로 초기화하면 400 generation을 다시 돌려야 한다. B400 가중치를
   낮은 lr로 fine-tune한다(B400 replay와 섞어 망각을 막는다). 그다음 균형 샘플링 PUCT v2 self-play를 **새 run**으로
   분기한다(`branch_stage8_run.py` 규칙). 대조군은 같은 분기점에서의 B400 continuation이다.

구현(§6): v7을 다시 두지 않는다. 이미 있는 기보(benchmark 결과 파일, 웹 대국, self-play 기록)에서 국면만 꺼내고
label은 `analysis.tactical_labels`가 규칙과 VCF 증명으로 붙인다. 그래서 v7 fingerprint 테스트가 필요 없고, frozen 파일은
읽기만 한다(`_winning_moves`, `_unstoppable_four_moves`, VCF solver). probe fixture 국면을 지나는 게임은 통째로 빼서
probe를 held-out으로 유지한다.

### 3.3 VCT/open-three probe와 offline solver

- 사람 승리 4판의 마지막 SAFE 착수 국면(§1.3 굵은 행)과 그 직후 국면을 추출한다. D4 대칭 8배로 `must_defend_vct` probe를 만든다.
- **offline bounded VCT solver**(`analysis.threats`, 분석 전용, 탐색에 넣지 않음)로 각 국면의 정답을 certificate로 검증한다.
  검증되지 않은(UNKNOWN) 국면은 probe에서 뺀다. VCT 깊이 d는 "조용한 수 d번 뒤 VCF"다. 모든 조용한 수를 두고, 방어자의
  응수도 실제로 두므로 금수점 효과를 엔진이 처리한다. 흑 방어자가 자기 돌로 막을 자리를 금수로 만드는 경우와, 백 돌이 흑
  금수를 풀어 주는 경우가 모두 들어간다. 처음 구현은 null-move 가지치기를 썼는데, 위 두 경우 때문에 **두 색 모두에서
  SAFE가 건전하지 않았다**(§8). 지금은 분석용 옵션 `prune_quiet`로만 남겼다. VCF 단계는 동결 solver를 그대로 쓰므로
  §1.4의 범위 한계(방어 수가 4가 되는 수순을 건너뜀)는 **그대로 남는다.**
- 이 solver의 label을 **학습 target**으로 쓸지는 별도 결정이다. 7-B 원칙(규칙이 network의 몫을 대신하지 않는다)과
  충돌할 수 있으므로 teacher arm 1차 결과를 본 뒤 정한다.

### 3.4 웹 대국에 AlphaZero 에이전트

B400(및 이후 checkpoint)과 사람이 둘 수 있게 한다. 착수마다 다음을 기록한다.

- chosen action, 탐색 횟수
- root visits(전체), root prior top-k, root NN value
- 방문 상위 top-k 자식의 Q
- `tactical_filter` 적용 여부와 결과(허용 수 제한, 증명값)

이 기록으로 사람이 이긴 순간을 **policy가 못 봄 / value가 틀림 / PUCT가 방문하지 못함** 중 어느 것인지 가른다.
`search_with_tree`가 트리를 돌려주므로 search 계약을 바꾸지 않고 기록할 수 있다.

### 3.5 전술 안전망(대국 모드)은 뒤로

own VCF가 증명되면 그 수를 두는 root 전처리는 배포용으로 허용할 수 있다. 다만 우선순위가 낮다. 학습 탐색은 PUCT v2를
유지하고, 켜는 경우 외부 평가에서 별도 항목으로 기록한다. 휴리스틱 prior 혼합(`P = (1−λ)P_NN + λP_v7`)은 하지 않는다.

### 3.6 판정

우선순위:

1. B400 직접 대국(anchor h2h)
2. v321 / v5 / v6
3. VCT·open-three probe, 기존 tactical/value probe(raw network)
4. 색별 성적
5. 사람 대국
6. v7(보조 지표. teacher arm은 v7 기보 국면을 봤으므로 주 지표로 쓰지 않는다)

### 3.7 Stage 9(network 확대) 진입 기준

| 결과 | 해석 | 행동 |
|---|---|---|
| teacher arm이 대조군보다 유의하게 강함 | 학습 신호가 병목이었다 | 64×4에서 teacher 방식을 기본으로 하고 계속 |
| 두 arm 차이 없음, 둘 다 plateau(anchor 3회 연속), probe도 포화 | 용량 병목 근거가 강하다 | Stage 9: 96×6 또는 128×6, 새 초기화, GPU 경로 재개 |
| 두 arm 차이 없음, probe는 아직 오름 | 판단 보류 | 판수를 늘려 다음 heavy 지점까지 진행 |

plateau 판정은 기존 §12.8 기준을 따른다: heavy 지점(80 generation)마다 B400과 100판을 두고, 점수 ≤ 55% 또는
p ≥ 0.05이면 향상 없음으로 본다. 이것이 3회 연속이면 plateau다.

## 4. 작업 순서

| # | 작업 | 완료 기준 | 상태 |
|---|---|---|---|
| 1 | 분석기 3상태화, 6판 재실행 | SAFE/UNSAFE/UNKNOWN, `max_fours` 한도 제거, 재실행 표(§1.3), 단위 테스트 | **완료** |
| 2 | VCT/open-three probe + offline bounded VCT solver(분석 전용) | 4판에서 추출한 국면이 certificate로 검증되고, D4 확장 probe fixture가 생성됨. solver는 search/학습 경로에서 import되지 않음(격리 테스트) | **완료**(§6.1) |
| 3 | 웹 대국 AlphaZero 에이전트 + 착수별 root 로그(§3.4) | B400과 대국 가능, 로그로 `pi`·top-k Q 재구성 가능, 기존 V2~V7 대국 동작 불변 | **완료**(§6.2) |
| 4 | B400 continuation 유지 | heavy 지점 B480·B560·B640에서 B400 anchor h2h, plateau 판정 기록 | 데스크톱 실행(§6.5) |
| 5 | 증명된 전술 국면 추출기 + teacher dataset | frozen 파일 불변, 공급원·종류별 개수와 증명 방식 기록, probe 국면 제외 | **도구 완료**, benchmark 기보로 검증(§6.3). self-play 기보 추출은 데스크톱 |
| 6 | Teacher arm: B400 fine-tune → probe → 분기 self-play | fine-tune 전후 지표 기록, 가중치 외 상태 동일 | **도구 완료**(§6.4). 실행은 데스크톱 |
| 7 | Stage 9 진입 판단 | §3.7 표에 따라 결정 | **판정 도구 완료**(§6.5). 판정은 B640 이후 |

B400 checkpoint와 run 디렉터리는 데스크톱에만 있다(`runs/`는 커밋하지 않음). 그래서 4·6·7의 **실행**은 데스크톱 몫이고,
이 저장소에는 도구, 테스트, 실행 절차를 넣었다.

## 5. 재현

```bash
python scripts/analyze_web_play_losses.py logs/web_play --losses-only --tail 6          # VCF
python scripts/analyze_web_play_losses.py logs/web_play --losses-only --tail 2 --vct-depth 1
python scripts/build_vct_probes.py                                                    # 약 1~2시간
python -m unittest tests.test_analysis_threats tests.test_alphazero_web_agent \
    tests.test_teacher_branch tests.test_compare_teacher_arms
```

## 6. 구현과 결과

### 6.1 VCT solver와 probe (작업 2)

`src/analysis/threats.py`(`ThreatSolver`)는 SAFE/UNSAFE/UNKNOWN과 증거 수순을 돌려준다.
`search`·`training`·`model`은 `analysis`를 import하지 않는다(`tests/test_analysis_threats.py` 격리 테스트).

`scripts/build_vct_probes.py`(VCT 깊이 1, 노드 한도 100,000, 후보 상한 40). 결과 fixture는 `tests/fixtures/vct_probes_v1.json`이고,
기본 probe 11개를 D4로 8배 늘려 88개다. 노드 한도에 걸린 탐색은 0회였다(VCF 호출 27,375회).

| 판 | 분기 착수(경로) | VCF-safe 후보 | VCT1 결과 | v7 착수 | 대안(가지치기 판정) | 사람 실제 다음 수 |
|---|---|---|---|---|---|---|
| `163810` | 15(stage 4) | 4 | SAFE 2 / UNSAFE 2 | (12,7) **UNSAFE** | (5,8), (8,11) | (11,10) = 유일한 승리 위협 |
| `163903` | 13(stage 4) | 4 | SAFE 2 / UNSAFE 2 | (7,9) **UNSAFE** | (7,5), (12,9) | (9,6) = 유일한 승리 위협 |
| `164445` | 14(stage 4) | 2 | SAFE 1 / UNSAFE 1 | (8,4) **UNSAFE** | (4,8) | (6,7) ∈ 승리 위협 {(6,7), (6,8)} |
| `164723` | 20(MCTS 100회) | 206 | v7 착수만 증명 | (11,9) **UNSAFE** | 미증명(후보 206개) | (6,10) ∈ 승리 위협 {(6,10), (4,10)} |

(좌표 1-based. UNSAFE = 사람의 위협 1수 뒤 모든 응수가 VCF로 진다는 증명이 있음. SAFE = 위협 1수 + VCF로는 지지 않음.)

- **§1.4의 VCT 가설은 깊이 1에서 확인됐다.** 네 판 모두 v7의 분기 착수가 "위협 1수 + VCF"로 지는 수로 증명됐고,
  사람이 실제로 둔 다음 수가 증명된 승리 위협이었다.
- **stage 4 3판에서는 VCT1로도 지지 않는 대안이 있었다.** 그러니 "stage 4가 VCT를 못 보고 틀린 방어를 골랐다"는 깊이 1 기준에서
  증명된 사실이다. 이 SAFE 판정과 "승리 위협 전부" 집합은 처음에 가지치기 solver로 만들었지만, 가지치기 없는 solver로 다시 만든
  결과가 fixture와 완전히 같았다(§8). 대안이 더 깊은 VCT(위협 2수 이상)로 질 수 있는지는 확인하지 않았다.
- `164723`은 MCTS 100회로 둔 수도 같은 종류의 실수였다. 탐색만 늘리는 방식으로는 해결되지 않는다는 §1.4 판단과 맞는다.
- 후보가 40개를 넘는 국면(`164723` 20수)은 정답 집합을 완전하게 만들 수 없어서 `must_defend_vct`를 만들지 않았다.
  top-k만 검사한 정답 집합으로는 검사하지 않은 안전한 수를 오답으로 채점하게 된다.

| probe kind | 기본 개수 | D4 후 | 채점 |
|---|---|---|---|
| `must_defend_vct` | 3 | 24 | policy top-1/top-3이 VCT1-SAFE 집합 안인가, `avoid_moves`(증명된 UNSAFE)의 확률 |
| `vct_attack` | 4 | 32 | 증명된 승리 위협수를 찾는가 + value +1 |
| `vcf_loss` | 4 | 32 | value −1(VCF 패배 증명) |

비용: 후보가 적은 판은 한 판에 1~5분이 걸렸다. 가장 비싼 단계는 "승리 위협수 전부 찾기"였다(`164723` 21수 29분).
VCT가 **없음**을 증명하는 쪽이 있음을 증명하는 쪽보다 훨씬 비싸다.

### 6.2 웹 대국 AlphaZero 에이전트 (작업 3)

```bash
python scripts/run_web_play.py --az-checkpoint runs/stage8_g3_b/checkpoints/checkpoint_gen400.pt
# 옵션: --az-simulations 100, --az-tactical-rules on|off|auto(기본: checkpoint의 평가 설정)
```

- 상대 목록에 `az`가 추가된다. 탐색은 외부 평가와 같다(noise 끔, temperature 0).
- `game.json`에는 착수마다 `root_visits`(전체), `top_visits`(visits·prior·Q), `top_priors`, `root_value`, `root_q`,
  `tactical_allowed`·`tactical_proven`이 저장된다. 대국 단위로는 `agent_info`(checkpoint SHA-256, generation, search 설정)가 저장된다.
- `moves.csv`에는 `az_*` 요약 열이 추가된다(`az` 대국에만). V2~V7 대국의 CSV 열과 동작은 그대로다.
- 사람이 이긴 판은 `analyze_web_play_losses.py`로 분기점을 찾고, 그 착수의 `top_visits`/`top_priors`를 본다.
  정답 수의 prior가 낮으면 policy 문제, 방문은 됐는데 Q가 틀리면 value 문제, prior는 있는데 방문이 적으면 탐색 예산 문제다.

### 6.3 증명된 전술 dataset (작업 5)

`src/analysis/tactical_labels.py` + `scripts/build_tactical_dataset.py`. 국면은 D4 동치로 중복을 제거하고, probe fixture
국면을 지나는 게임은 통째로 뺀다(§7-1; 아래 표는 그 수정 전, 국면 단위 제외로 만든 것이다). VCF 탐색이 노드 한도에 걸린 국면은 label을 붙이지 않는다(건전성 우선, 완전성 포기).

benchmark 기보(V7 대 V5/V6, 200판)와 사람 대국 6판 기준, 워커 3개로 15분:

| 전체 국면 | 중복(D4) | probe 국면 제외 | 노드 한도로 label 없음 | must_block | vcf | unstoppable_four | forced_loss | immediate_win |
|---|---|---|---|---|---|---|---|---|
| 11,186 | 796 | 160 | 86 | 1,869 | 609 | 202 | 162 | 162 |

독립 검증:

- 규칙 label 3,004개 전부 PUCT v2 `search.tactics.tactical_filter`(별도 구현)와 일치했다(즉승 집합, 유일 방어점,
  −1 증명).
- `vcf` 609개 전부에서 동결 VCF solver가 찾은 첫 수가 label 집합 안에 있었다.

**held-out sanity(새 64×4 network, 이 dataset만으로 2,000 step):** Stage 7 sanity와 같은 분리 방식을 썼다.
학습은 V7 대 V5 기보(label 1,443개)만 썼고, 평가는 V7 대 V6 기보와 VCF 회귀 국면에서 나온 probe로 했다.

| probe(held-out) | 학습 전 | 학습 후 | 참고: Stage 7 sanity |
|---|---|---|---|
| must_block top-1 | 0.00 | **0.72** | 0.67~0.78 |
| vcf top-1 / top-3 | 0.00 / 0.05 | **0.38 / 0.63** | 학습 안 함 |
| immediate_win top-1 / top-3 | 0.00 / 0.05 | 0.11 / 0.42 | ≤ 0.11 |
| forced_loss value 부호 | 0.00 | **0.85** | 0.75~0.85 |
| value 균형 정답률 / separation | 0.42 / −0.03 | **0.91 / +1.66** | — |
| must_defend_open3 top-1 / top-3 (학습 kind 아님) | 0.00 / 0.06 | **0.88 / 1.00** | — |

- 증명 label만으로 raw network에 전술을 넣을 수 있고, 학습하지 않은 열린 3 방어까지 일반화된다.
- 이것은 "tactical bootstrap이 가능하다"는 근거일 뿐 **기력 향상 근거가 아니다.** 기력은 §6.5의 두 arm 비교로만 판단한다.
- Stage 7-C의 B gen 110 raw network는 must_block top-1 0~7%, open3 top-3 25~38%였다. B400의 값은 데스크톱 probe
  결과(`probes/gen400*.json`)로 확인해야 한다. B400이 이미 높다면 teacher arm이 얻을 몫도 작다.

### 6.4 Teacher branch (작업 6)

`scripts/make_teacher_branch.py`:

1. `branch_stage8_run.branch`로 gen N을 새 디렉터리에 복사한다.
2. gen N 자체의 external_eval/probe 결과는 지운다(옛 가중치 결과이므로 orchestrator가 다시 계산한다).
3. dataset과 run 자신의 replay(균형 샘플링 설정을 따름)를 절반씩 섞어 fine-tune한다. 기본값: 1,000 step,
   batch 64, lr 2e-4, Adam은 fine-tune 전용으로 새로 만든다.
4. `checkpoint_genNNN.pt`와 `latest.pt`의 `model_state_dict`만 바꾼다. config, optimizer 상태, replay, RNG,
   generation은 그대로다.

그래서 대조군(같은 run의 gen N continuation)과 teacher arm은 **가중치만 다르고 self-play seed까지 같다**(짝지은 비교).
`TEACHER.json`에는 dataset 해시, 설정, 전후 지표(전술 set top-1/value 부호, 고정 replay 표본의 policy/value loss)가 남는다.
replay loss가 크게 오르면 fine-tune이 self-play 지식을 덮은 것이므로 step이나 lr을 줄인다.

### 6.5 데스크톱 실행 절차 (작업 4~7)

오래 걸리는 증명·라벨링·학습은 모두 데스크톱에서 실행한다. 시간은 실측 전 추정치다.

```bash
# 0) 대조군: §12.8의 B400 continuation (이미 진행 중이면 그대로). heavy 지점마다 B400과 100판.
python scripts/run_stage8_training.py --run-dir runs/stage8_b400_long --config configs/stage8_g3_b.yaml \
    --anchor 400 --target-generation 640 \
    --light-opponents tactical mcts_v2 mcts_v321 --heavy-opponents mcts_v321 mcts_v5 mcts_v6 mcts_v7 \
    --heavy-pairs 10 --h2h-anchor B400=runs/stage8_g3_b/checkpoints/checkpoint_gen400.pt

# 1) T1 dataset: VCF 수준 증명 label (워커 6개, 1~2시간 추정). probe가 나온 게임은 통째로 빠진다.
python scripts/build_tactical_dataset.py --benchmark "docs/mcts-v7-results/*.json" \
    --web-games tests/fixtures/web_play_v7_human_games_v1.json \
    --self-play-run runs/stage8_g3_b --generations 160 400 --workers 6 \
    --output runs/teacher/tactical_t1.json

# 2) T1 branch (B400 = stage8_g3_b gen 400, 대조군과 같은 분기점)
python scripts/make_teacher_branch.py --source runs/stage8_g3_b --generation 400 \
    --dataset runs/teacher/tactical_t1.json --dest runs/stage8_b400_t1 --balance-kinds

# 3) T1 학습: 0)과 같은 옵션, run-dir만 다르게
python scripts/run_stage8_training.py --run-dir runs/stage8_b400_t1 --config configs/stage8_g3_b.yaml \
    --anchor 400 --target-generation 640 \
    --light-opponents tactical mcts_v2 mcts_v321 --heavy-opponents mcts_v321 mcts_v5 mcts_v6 mcts_v7 \
    --heavy-pairs 10 --h2h-anchor B400=runs/stage8_g3_b/checkpoints/checkpoint_gen400.pt

# 4) heavy 지점마다 직접 대국 (480, 560, 640)
python scripts/run_stage8_head_to_head.py \
    --checkpoint control480=runs/stage8_b400_long/checkpoints/checkpoint_gen480.pt \
    --checkpoint T1_480=runs/stage8_b400_t1/checkpoints/checkpoint_gen480.pt \
    --pairs 50 --output runs/teacher_h2h/t1_gen480.json

# 5) 판정 (§3.7)
python scripts/compare_teacher_arms.py --control runs/stage8_b400_long --teacher runs/stage8_b400_t1 \
    --direct "runs/teacher_h2h/t1_*.json" --teacher-label-prefix T1 --output runs/teacher_h2h/t1_verdict.json
```

T2(§7)는 1)~5)에서 dataset, run-dir, label만 바꾼다.

```bash
# T2 pilot: 처리량부터 잰다 (20세대 분량). 로그의 "VCT phase: N threat candidates"와 걸린 시간을 확인한다.
python scripts/build_tactical_dataset.py --self-play-run runs/stage8_g3_b --generations 380 400 \
    --vct-depth 1 --workers 6 --output runs/teacher/tactical_t2_pilot.json
# T2 본 실행: T1과 같은 소스 + VCT label
python scripts/build_tactical_dataset.py --benchmark "docs/mcts-v7-results/*.json" \
    --web-games tests/fixtures/web_play_v7_human_games_v1.json \
    --self-play-run runs/stage8_g3_b --generations 160 400 --vct-depth 1 --workers 6 \
    --output runs/teacher/tactical_t2.json
python scripts/make_teacher_branch.py --source runs/stage8_g3_b --generation 400 \
    --dataset runs/teacher/tactical_t2.json --dest runs/stage8_b400_t2 \
    --balance-kinds --kind-weights vct_attack=0.4,must_defend_vct=0.4,vct_loss=0.2
# 이후 3)~5)와 같다. run-dir runs/stage8_b400_t2, label T2_480 .... 직접 대국은 arm마다 control과 1:1 파일로 따로 만든다
python scripts/run_stage8_head_to_head.py \
    --checkpoint control480=runs/stage8_b400_long/checkpoints/checkpoint_gen480.pt \
    --checkpoint T2_480=runs/stage8_b400_t2/checkpoints/checkpoint_gen480.pt \
    --pairs 50 --output runs/teacher_h2h/t2_gen480.json
python scripts/compare_teacher_arms.py --control runs/stage8_b400_long --teacher runs/stage8_b400_t2 \
    --direct "runs/teacher_h2h/t2_*.json" --teacher-label-prefix T2 --control-label-prefix control \
    --output runs/teacher_h2h/t2_verdict.json
```

- 0)의 run이 이미 다른 옵션으로 돌고 있다면, 3)을 **그 옵션에 맞춘다**. 모든 arm의 평가 설정이 같아야 한다.
- teacher arm의 gen 400 지표(anchor h2h 제외)는 fine-tune 직후 값이다. B400(0세대 차)과의 차이가 fine-tune 효과다.
- light probe 지점마다 `probes/genNNN_vct.json`이 새로 생긴다. 대조군의 과거 지점도 catch-up에서 계산된다(raw network라 몇 초).
- `compare_teacher_arms.py` 판정: `teacher_better` / `capacity` / `undecided`. 기준 수치(55%, p 0.05, probe +0.05)는 스크립트 상단에 있다.
- 결과를 받으면 `TEACHER.json`, `verdict.json`, 각 run의 `external_eval/`·`probes/`를 공유해 달라. 그걸로 이 문서의 판정을 갱신한다.

## 7. 데이터 규모 단계(T1/T2/T3) 검토

제안: T1(현재 3,004) → T2(약 1만) → T3(3만 이상)으로 늘리고, 개수보다 VCT 공격·방어 데이터를 우선한다.
방향에 동의한다. 다음을 고치거나 덧붙였다.

1. **누수를 코드에서 막았다(버그 수정).** 이전 추출기는 probe와 **같은 국면**(D4 포함)만 뺐다. 그래서 probe가 나온 게임의
   앞뒤 수는 학습에 들어갔다(stage7 probe 94판, 열린 3 probe 35판, VCT probe 4판). 이제 probe 국면을 지나는 게임은
   **통째로** 뺀다(`excluded_probe_games`). §6.3의 benchmark 표(3,004개)는 수정 전 방식이었다. 다만 held-out sanity는 학습
   게임과 평가 게임을 따로 나눠서 영향이 없다.
2. **T1은 3,004개가 아니다.** 3,004개는 이 컨테이너에서 benchmark 200판만 돌린 표본이다. §6.5의 T1은 B 계열 self-play
   gen 160~400(약 3,840판)을 포함해서 훨씬 크다. 게임 단위 제외 때문에 benchmark 쪽은 오히려 줄어든다.
   T1은 "개수"가 아니라 **"VCF 수준 증명 label 전체"**로 정의하고, 실제 개수는 dataset `stats`에 기록한다.
3. **T2는 개수가 아니라 내용으로 정의한다.** T2 = T1 + VCT label(`--vct-depth 1`). 추가되는 kind:
   - `vct_attack`: 게임에서 실제로 둔 위협수 h. h 뒤 모든 응수가 VCF로 진다는 증명이 있고 value는 +1이다.
     다른 승리 위협수가 더 있을 수 있어서 **정답 집합이 완전하지 않다(one-hot)**. 전부 찾는 데 한 국면에 30분까지 걸린 적이 있다.
   - `vcf_loss`: h 뒤 방어자 국면. value −1이다.
   - `must_defend_vct` / `vct_loss`: h 직전 방어자 국면. VCF-safe 후보가 12개 이하일 때만 전부 분류해서, 정답 집합이 완전하다.
     후보가 적은 **날카로운 국면에 치우친다**는 편향이 있다.
   - 후보는 게임 진행으로 고른다. "t에는 VCF 수준 label이 없고 t+2에 같은 편의 VCF 승리가 있음"인 곳만 증명한다. 그래서 모든
     국면을 VCT 탐색하지 않는다. 대신 게임에서 실제로 나오지 않은 VCT는 빠진다.
   - T3(개수 확대)는 T2가 T1보다 나을 때만 한다. 소스는 더 많은 self-play 세대나 새 benchmark 기보다. 같은 국면 복제는 의미가 없다.
4. **투입량(dose)을 고정해야 데이터 효과만 본다.** fine-tune이 보는 teacher 행 수는 dataset 크기가 아니라
   `steps × batch × teacher_fraction`(기본 1,000 × 64 × 0.5 = 32,000행, `TEACHER.json`의 `teacher_rows_drawn`)이다.
   T1·T2·T3는 같은 step·lr·비율로 돌린다. 그러면 3천 개 set은 행마다 약 10번, 3만 개 set은 약 1번 본다.
   **다만 이것은 "레시피 비교"이지 "VCT 내용만의 효과"가 아니다.** 총 행 수가 같으니 VCT kind를 넣으면 기존 kind의 몫이
   줄어든다. 예를 들어 `--balance-kinds`에서 kind가 5종 → 9종이 되면 기존 kind는 각 20% → 약 11%가 된다. 그래서 T2의 좋고
   나쁨을 VCT 데이터만의 효과로 돌리지 않는다. 희석을 줄이려면 `--kind-weights`로 VCT kind 묶음의 합을 기존 kind 하나 정도로
   둔다. 예: `--balance-kinds --kind-weights vct_attack=0.4,must_defend_vct=0.4,vct_loss=0.2` → 기존 5종이 각 약 16.7%.
   실제 비중은 `TEACHER.json`의 `settings.kind_shares`에 기록된다. VCT의 순수 효과를 따로 보고 싶으면, 투입량을 고정하지
   않고 T1 행 수를 그대로 둔 채 VCT 행만 더하는 arm을 추가해야 한다. 이때는 총 학습량이 달라지는 것이 교란 요인이 된다.
5. **통계 해상도.** anchor h2h 100판의 표준오차는 약 5%p다. 그래서 "+2% 대 +3%" 같은 차이는 구분할 수 없고,
   약 10~14%p 이상 차이가 나야 유의하다. 판정은 arm끼리 **직접 대국**(heavy 지점마다 100판)으로 한다.
   세 지점을 모두 보면 흔들림이 줄어든다.
6. **한 번의 fine-tune은 씻겨 나갈 수 있다.** 240세대 self-play 동안 전술 지식이 다시 흐려질 수 있다. gen 400~480에는 이득이
   보이다가 640에서 사라지면, 매 세대 학습 batch에 전술 행을 섞는 방식(T4)을 검토한다. 이 방식은 학습 루프 변경이라
   training-critical 설정이 바뀐다. 그러니 T1/T2 결과를 먼저 본다.
7. **실행 순서.** 대조군과 T1을 먼저 시작한다. T2 dataset은 그동안 만든다(pilot으로 처리량부터 확인). 모든 arm이 같은 gen 400
   상태에서 분기하므로, T2를 늦게 시작해도 짝지은 비교는 유지된다. Gate 3처럼 세 arm을 동시에 돌려도 되지만,
   그때는 시간 지표를 비교하지 않는다.

| 결과 | 해석 |
|---|---|
| T1 ≫ 대조군 | 학습 신호 부족이 병목이다. T2로 VCT 내용을 더한다 |
| T1 ≈ 대조군, T2 ≫ 대조군 | VCF 수준 지식은 이미 있다. 부족했던 건 VCT 계열이다 |
| T1 ≈ T2 ≈ 대조군, probe도 포화 | 데이터보다 64×4 용량이나 탐색 구조를 의심한다. Stage 9로 간다 |
| T1·T2 초반 이득 → 640에서 소멸 | fine-tune이 씻겨 나간다. T4(지속 혼합)를 검토한다 |

## 8. 외부 검토 반영 (`4f620b3` 이후)

| 지적 | 판단 | 조치 |
|---|---|---|
| `defense_label`이 흑 공격자에서 비-exact | **맞음, 범위는 더 넓음.** null-move 가지치기는 백 공격자에서도 건전하지 않다(흑 방어자가 자기 돌로 막을 자리를 금수로 만들 수 있음). UNSAFE 증명은 실제 수순이라 영향이 없고, SAFE 쪽(방어 정답 집합, 승리 위협 집합의 완전성)이 영향을 받는다 | 가지치기를 없앴다. 모든 조용한 수를 두고 첫 SAFE 응수에서 멈춘다. 같은 국면(`164445` 14수)에서 비용 48.0 s 대 46.6 s, 결과 동일. `prune_quiet`는 분석용 옵션으로만 남김. 색을 제한할 필요가 없어졌다 |
| probe 게임 검사에서 마지막 국면 누락 | **맞음** | 최종 국면(`len(moves)` ply)까지 검사한다. 라벨은 여전히 비종료 국면에만 붙인다. 테스트 추가 |
| `--balance-kinds`로 T1→T2가 "내용만의 차이"가 아님 | **맞음** | §7-4 문구를 "레시피 비교"로 고쳤다. `--kind-weights`를 추가했고, `TEACHER.json`에 `kind_shares`를 기록한다 |
| round robin 파일을 넣으면 comparator가 T1 vs T2를 teacher 결과로 읽음 | **맞음(코드 버그)** | `direct_result`가 teacher 접두사 대 control 접두사(`--control-label-prefix`, 기본 `control`) 경기만 쓴다. 테스트 추가. 그래도 절차는 arm별 1:1 파일로 적었다 |
| GitHub에 CI 기록이 없음 | 사실 | `.github/workflows/ci.yml`은 `pull_request`와 `main` push에서만 돈다. 이 브랜치의 결과는 로컬 실행(`ci_run_tests.py --skip-policy none`, torch 차단 `--skip-policy torch-only`, `check_frozen_baseline.py`)이다. PR을 열면 CI가 돈다 |

**fixture 재검증 완료:** `tests/fixtures/vct_probes_v1.json`은 처음에 가지치기 solver로 만들었다. 데스크톱에서 가지치기 없는 solver로
다시 만든 결과가 `check: SAME`(모든 probe의 id, 정답 수, 피할 수, value 부호가 동일)이었다. 그래서 fixture를 그대로 쓴다. §10의 0단계는
이제 선택 사항이다.

## 9. B400 장기 학습 결과 반영 (gen 400 → 880)

결과 표와 수치는 [Stage 8 계획 §12.10](stage8-plan.md)에 있다. 설계에 반영한 점:

1. **정체는 확정됐지만 원인은 용량이 아니다.** B400 대비 6개 heavy 지점에서 향상이 없었고(640은 유의하게 약함), 탐색 예산 25/50/100도
   효과가 없었다. 그러나 raw 전술 probe(must_block top-1 0.10~0.28)는 같은 64×4가 증명 label만으로 도달하는 수준(0.72)보다 훨씬 낮다.
   `compare_teacher_arms.py`에 이 기준(`MUST_BLOCK_CEILING = 0.6`)을 넣었다. 두 arm이 정체해도 raw 전술이 이 아래면 판정은
   `capacity`가 아니라 `signal`이다. 이번 run의 판정은 `signal`이다.
2. **새 원인 후보: self-play 퇴화.** 게임의 84%가 9~10수(방어 없는 최단 5목)에 끝나고, `temperature_moves: 10`이라 그 게임 전체가
   샘플링으로 두어졌다. teacher보다 싸고 영향이 클 수 있어서 **레시피 arm S4(temperature_moves 4)**를 T1과 함께 1순위로 올린다.
   다만 온도 때문이라는 것은 **가설**이다. 방문 분포 자체가 방어를 선호하지 않을 수도 있다. S4에서 게임 길이 분포가 바뀌는지가
   첫 확인 지표다(`metrics.jsonl`의 `game_lengths`).
3. **S4는 짝지은 branch로 만든다.** 새 run(가중치만 export)은 replay·optimizer·RNG가 초기화되어 대조군과 조건이 달라진다.
   `make_recipe_branch.py`는 gen 400 상태를 그대로 두고 checkpoint 설정과 critical hash만 바꾼다. `--in-place`로 T1 branch에도
   적용할 수 있다(S4+T1).
4. **대조군은 다시 돌리지 않는다.** `runs/stage8_b400_long`이 이미 gen 480~880까지 있다. 각 arm은 gen 640까지(heavy 3지점) 돌리고,
   같은 세대의 대조군 checkpoint와 직접 대국한다.
5. **판정 순서:** (a) S4 게임 길이 분포 → (b) arm 대 대조군 직접 대국(480/560/640) → (c) raw probe(기존 + VCT) → (d) v321/v5/v6/v7.
   S4나 T1이 대조군보다 유의하게 강하면 그 레시피로 64×4를 계속한다. 둘 다 효과가 없고 raw 전술이 여전히 낮으면, 다음 후보는
   self-play 탐색 설정(Dirichlet ε, 방문 수)과 지속 혼합(T4)이다. Stage 9는 raw 전술이 기준선 가까이 올라온 뒤에 정체할 때 간다.

| 결과 | 해석 | 다음 |
|---|---|---|
| S4 ≫ 대조군 | self-play 레시피가 병목이었다 | S4를 기본 설정으로 채택하고, 그 위에서 T1/T2 |
| T1 ≫ 대조군, S4 ≈ 대조군 | 전술 학습 신호 부족 | T2(VCT label)로 확장 |
| S4, T1 모두 ≫ | 둘 다 기여 | S4+T1 조합을 확인 |
| 모두 ≈ 대조군, raw 전술 낮음 | 판정 `signal` 유지 | 탐색 설정·지속 혼합(T4) 검토 |
| 모두 ≈ 대조군, raw 전술 높음(≥ 0.6) | 판정 `capacity` | Stage 9(network 확대) |

## 10. 데스크톱 실행 명령 총정리 (PowerShell)

저장소 루트(`C:\오목 강화학습\renju-alphazero`)에서 `.venv-cpu`를 활성화한 상태를 기준으로 한다. 시간은 실측 전 추정치다.
CPU 코어: S4·T1 학습은 각 1코어(`torch_threads 1`)이므로 동시에 돌려도 된다. dataset 생성(워커 5~6개)은 학습과 코어를 나눈다.
권장 순서: 0 → 1 → 2를 창 하나에서 시작 → 다른 창에서 3 → 4 → 둘 다 gen 640까지 끝나면 5 → 6.
긴 학습 명령은 `Tee-Object`로 화면과 로그 파일에 같이 남긴다(Windows PowerShell 5에서 torch 경고가 빨간 오류처럼 보여도 무시해도 된다).

```powershell
# ---------------------------------------------------------------------------
# 0. 최신 코드 + 로컬 테스트 (VCT probe fixture 재검증은 SAME으로 완료, 마지막 두 줄은 생략 가능)
# ---------------------------------------------------------------------------
git fetch origin feat/stage8-plan
git checkout feat/stage8-plan
git pull origin feat/stage8-plan
python scripts/ci_run_tests.py --skip-policy none                       # 로컬 회귀 (수 분)
python scripts/build_vct_probes.py --output runs/vct_probes_v1_recheck.json `
    --check-against tests/fixtures/vct_probes_v1.json

# ---------------------------------------------------------------------------
# 1. 대조군(stage8_b400_long)에 VCT probe만 추가 계산 (학습 없음, 수 분)
# ---------------------------------------------------------------------------
python scripts/run_stage8_training.py --run-dir runs/stage8_b400_long --eval-only `
    --anchor 400 --light-opponents tactical mcts_v2 mcts_v321 `
    --heavy-opponents mcts_v321 mcts_v5 mcts_v6 mcts_v7 --heavy-pairs 10 `
    --h2h-anchor B400=runs/stage8_g3_b/checkpoints/checkpoint_gen400.pt

# ---------------------------------------------------------------------------
# 2. S4 arm: temperature_moves 10 -> 4, gen 400 상태 그대로 분기 → gen 640까지 (약 2~3시간)
# ---------------------------------------------------------------------------
python scripts/make_recipe_branch.py --source runs/stage8_g3_b --generation 400 `
    --config configs/stage8_b400_temp4.yaml --dest runs/stage8_b400_s4
python scripts/run_stage8_training.py --run-dir runs/stage8_b400_s4 `
    --config configs/stage8_b400_temp4.yaml --anchor 400 --target-generation 640 `
    --light-opponents tactical mcts_v2 mcts_v321 `
    --heavy-opponents mcts_v321 mcts_v5 mcts_v6 mcts_v7 --heavy-pairs 10 `
    --h2h-anchor B400=runs/stage8_g3_b/checkpoints/checkpoint_gen400.pt `
    2>&1 | Tee-Object -FilePath runs/stage8_b400_s4.log

# ---------------------------------------------------------------------------
# 3. T1 dataset: VCF 수준 증명 label, probe 게임 전체 제외 (워커 5개, 1~2시간 추정)
# ---------------------------------------------------------------------------
python scripts/build_tactical_dataset.py --benchmark "docs/mcts-v7-results/*.json" `
    --web-games tests/fixtures/web_play_v7_human_games_v1.json `
    --self-play-run runs/stage8_g3_b --generations 160 400 --workers 5 `
    --output runs/teacher/tactical_t1.json

# ---------------------------------------------------------------------------
# 4. T1 arm: 가중치만 fine-tune (수 분) → gen 640까지 (약 2~3시간)
# ---------------------------------------------------------------------------
python scripts/make_teacher_branch.py --source runs/stage8_g3_b --generation 400 `
    --dataset runs/teacher/tactical_t1.json --dest runs/stage8_b400_t1 --balance-kinds
python scripts/run_stage8_training.py --run-dir runs/stage8_b400_t1 `
    --config configs/stage8_g3_b.yaml --anchor 400 --target-generation 640 `
    --light-opponents tactical mcts_v2 mcts_v321 `
    --heavy-opponents mcts_v321 mcts_v5 mcts_v6 mcts_v7 --heavy-pairs 10 `
    --h2h-anchor B400=runs/stage8_g3_b/checkpoints/checkpoint_gen400.pt `
    2>&1 | Tee-Object -FilePath runs/stage8_b400_t1.log

# ---------------------------------------------------------------------------
# 5. 직접 대국: arm마다 대조군과 1:1, heavy 지점 480 / 560 / 640 (지점당 수 분)
# ---------------------------------------------------------------------------
foreach ($g in 480, 560, 640) {
  python scripts/run_stage8_head_to_head.py `
      --checkpoint "control$($g)=runs/stage8_b400_long/checkpoints/checkpoint_gen$($g).pt" `
      --checkpoint "S4_$($g)=runs/stage8_b400_s4/checkpoints/checkpoint_gen$($g).pt" `
      --pairs 50 --output "runs/arm_h2h/s4_gen$($g).json"
  python scripts/run_stage8_head_to_head.py `
      --checkpoint "control$($g)=runs/stage8_b400_long/checkpoints/checkpoint_gen$($g).pt" `
      --checkpoint "T1_$($g)=runs/stage8_b400_t1/checkpoints/checkpoint_gen$($g).pt" `
      --pairs 50 --output "runs/arm_h2h/t1_gen$($g).json"
}

# ---------------------------------------------------------------------------
# 6. 판정 (파일만 읽음, 즉시)
# ---------------------------------------------------------------------------
python scripts/compare_teacher_arms.py --control runs/stage8_b400_long --teacher runs/stage8_b400_s4 `
    --direct "runs/arm_h2h/s4_*.json" --teacher-label-prefix S4 --output runs/arm_h2h/s4_verdict.json
python scripts/compare_teacher_arms.py --control runs/stage8_b400_long --teacher runs/stage8_b400_t1 `
    --direct "runs/arm_h2h/t1_*.json" --teacher-label-prefix T1 --output runs/arm_h2h/t1_verdict.json
```

**선택 단계 (6의 결과를 본 뒤):**

```powershell
# S4+T1 조합: T1 branch를 하나 더 만들고 같은 자리에서 temperature만 바꾼다
python scripts/make_teacher_branch.py --source runs/stage8_g3_b --generation 400 `
    --dataset runs/teacher/tactical_t1.json --dest runs/stage8_b400_s4t1 --balance-kinds
python scripts/make_recipe_branch.py --in-place --dest runs/stage8_b400_s4t1 --generation 400 `
    --config configs/stage8_b400_temp4.yaml
# 학습은 2와 같은 명령에서 run-dir만 runs/stage8_b400_s4t1

# T2 pilot: VCT label 처리량 측정 (20세대 분량). 로그의 "VCT phase: N threat candidates"와 걸린 시간을 알려 주기
python scripts/build_tactical_dataset.py --self-play-run runs/stage8_g3_b --generations 380 400 `
    --vct-depth 1 --workers 5 --output runs/teacher/tactical_t2_pilot.json

# 사람 대국: B400과 S4/T1 결과물을 직접 두어 보기 (착수마다 root 로그 저장)
python scripts/run_web_play.py --az-checkpoint runs/stage8_g3_b/checkpoints/checkpoint_gen400.pt
```

**공유해 줄 것:** 0단계 `check:` 줄, `runs/stage8_b400_s4`·`runs/stage8_b400_t1`의 `metrics.jsonl`·`external_eval/`·`probes/`·로그,
`runs/stage8_b400_t1/TEACHER.json`, `runs/stage8_b400_s4/RECIPE.json`, `runs/arm_h2h/` 전체.

## 11. S4 / T1 결과와 다음 실행 (2026-09-30)

결과 표와 외부 검토를 반영한 해석은 [Stage 8 계획 §12.11](stage8-plan.md)에 있다. 요약:

- **S4(temperature 4)는 성공으로 판정한다.** 480부터 대조군과 B400을 계속 이겼고, 640에서 다시 크게 올랐다(대조군 93%, B400 84%).
  v7에 처음으로 의미 있는 승리 신호(6/20)가 나왔다. "온도가 원인"이라는 가설을 강하게 지지하지만 증명은 아니다.
- **T1(한 번의 fine-tune)은 지속되지 않았다.** 20세대 만에 raw 전술이 원래대로 돌아갔다. 원인 후보(분포, 지속 혼합 부재,
  optimizer 상태 불일치, lr 5배 상승)는 아직 가르지 못했다.
- **판정 역할을 나눴다.** 고정 anchor = 각 arm의 절대 진행, 같은 세대 직접 대국 = 레시피 비교다. 직접 대국은 이기는데 anchor
  진행이 없으면 `relative_only`로 판정한다(T1의 경우). 실제 데이터로 판정하면 S4 `teacher_better`, T1 `relative_only`다.
- 지금까지 확인된 주요 병목은 네트워크 크기보다 self-play 레시피였다. S4 레시피를 채택하고, S640을 새 anchor로 temperature 2(S2)를
  비교한다. S2 설정은 S4 설정과 `self_play.temperature_moves`만 다르다(`tests/test_teacher_branch.py`가 검사).

### 11.1 다음 실행 명령 (PowerShell)

S4 continuation과 S2 arm은 각각 1코어라 두 창에서 동시에 돌린다. 각 4~5시간(추정, S4 self-play는 세대당 약 35~45초).

```powershell
# 0. 최신 코드
git pull origin feat/stage8-plan

# 1. 새 anchor: S4 gen 640을 run 밖으로 복사 (학습 중 가지치기와 무관하게 고정)
New-Item -ItemType Directory -Force runs/anchors | Out-Null
Copy-Item runs/stage8_b400_s4/checkpoints/checkpoint_gen640.pt runs/anchors/S640.pt

# 2. S2 arm: S4 gen 640 상태에서 temperature 4 -> 2로 분기 (가중치/replay/optimizer/RNG 동일)
python scripts/make_recipe_branch.py --source runs/stage8_b400_s4 --generation 640 `
    --config configs/stage8_s640_temp2.yaml --dest runs/stage8_s640_s2

# 3. (창 1) S4 continuation: 640 -> 880. 720/800/880 heavy 지점의 anchor는 S640
python scripts/run_stage8_training.py --run-dir runs/stage8_b400_s4 `
    --config configs/stage8_b400_temp4.yaml --anchor 400 --target-generation 880 `
    --light-opponents tactical mcts_v2 mcts_v321 `
    --heavy-opponents mcts_v321 mcts_v5 mcts_v6 mcts_v7 --heavy-pairs 10 `
    --h2h-anchor S640=runs/anchors/S640.pt `
    2>&1 | Tee-Object -FilePath runs/stage8_b400_s4_cont.log

# 4. (창 2) S2 arm: 640 -> 880, 같은 평가 설정과 anchor
python scripts/run_stage8_training.py --run-dir runs/stage8_s640_s2 `
    --config configs/stage8_s640_temp2.yaml --anchor 400 --target-generation 880 `
    --light-opponents tactical mcts_v2 mcts_v321 `
    --heavy-opponents mcts_v321 mcts_v5 mcts_v6 mcts_v7 --heavy-pairs 10 `
    --h2h-anchor S640=runs/anchors/S640.pt `
    2>&1 | Tee-Object -FilePath runs/stage8_s640_s2.log

# 5. 직접 대국: S4 continuation 대 S2, 720 / 800 / 880
foreach ($g in 720, 800, 880) {
  python scripts/run_stage8_head_to_head.py `
      --checkpoint "S4c$($g)=runs/stage8_b400_s4/checkpoints/checkpoint_gen$($g).pt" `
      --checkpoint "S2_$($g)=runs/stage8_s640_s2/checkpoints/checkpoint_gen$($g).pt" `
      --pairs 50 --output "runs/arm_h2h/s2_gen$($g).json"
}

# 6. 판정: S4 continuation(대조) 대 S2. S4 run의 400~640 anchor 파일은 B400, 720 이후는 S640이다.
#    정체 판정은 마지막 3지점(720/800/880, 모두 S640 기준)만 쓴다.
python scripts/compare_teacher_arms.py --control runs/stage8_b400_s4 --teacher runs/stage8_s640_s2 `
    --direct "runs/arm_h2h/s2_*.json" --teacher-label-prefix S2 --control-label-prefix S4c `
    --output runs/arm_h2h/s2_verdict.json

# (선택) 사람 대국: S4 gen 640과 직접 두어 보기. 착수마다 root 로그 저장
python scripts/run_web_play.py --az-checkpoint runs/anchors/S640.pt
```

**해석 기준:**

- S640 anchor 대국은 각 run의 **절대 진행**이다(S4 continuation과 S2 모두). 720/800/880 직접 대국은 **temp 4 대 temp 2 비교**다.
- S2가 직접 대국에서 이기더라도 S640 anchor를 넘지 못하면 `relative_only`다. S4 continuation이 약해진 것일 수 있으니 채택하지 않는다.
- S4 continuation이 S640 anchor를 3지점 연속 못 넘으면 S4 레시피도 정체한 것이다.
  그때 raw 전술이 기준(must_block 0.6) 이상이면 `capacity`, 즉 Stage 9 후보가 된다.
- S2 ≫ S4 continuation이면 온도를 더 낮춘 쪽을 채택한다.
  S2 ≪ S4이면 탐험 부족(비슷한 게임 반복)을 의심한다. 이때 로그의 unique game 수와 흑/백 균형을 본다.

**공유해 줄 것:** 두 run의 `metrics.jsonl`, `external_eval/`, `probes/`, 로그, `runs/stage8_s640_s2/RECIPE.json`, `runs/arm_h2h/s2_*`.
사람 대국을 했다면 `logs/web_play/` 폴더도 함께 보내 줘.

## 12. S2 결과와 다음 실행 (2026-10-01)

결과는 [Stage 8 계획 §12.12](stage8-plan.md)에 있다.
- S4c와 S2 모두 S640보다 강해졌다(880에서 0.68 / 0.66).
- temperature 2와 4의 차이는 검출되지 않았다(직접 대국 합계 0.53, p≈0.3).
- 동률에서의 선택 기준에 따라 본선을 S2로 두고, 다음 단일 변수로 self-play simulations 50 → 100을 비교한다.
- heavy 평가는 상대당 50판으로 늘린다.

### 12.1 실행 명령 (PowerShell)

S2 continuation과 sims100 arm을 두 창에서 동시에 돌린다. S2 continuation은 4~5시간, sims100은 self-play가 약 2배라 8~10시간으로 추정한다.
S4c는 더 돌리지 않는다.

```powershell
# 0. 최신 코드
git pull origin feat/stage8-plan

# 1. 새 anchor 고정: S2 gen 880
Copy-Item runs/stage8_s640_s2/checkpoints/checkpoint_gen880.pt runs/anchors/S880.pt

# 2. sims100 arm: S2 gen 880 상태 그대로, self-play simulations 50 -> 100만 변경
python scripts/make_recipe_branch.py --source runs/stage8_s640_s2 --generation 880 `
    --config configs/stage8_s880_sims100.yaml --dest runs/stage8_s880_sims100

# 3. (창 1) 본선: S2 continuation 880 -> 1120, heavy 상대당 50판, anchor S880
python scripts/run_stage8_training.py --run-dir runs/stage8_s640_s2 `
    --config configs/stage8_s640_temp2.yaml --anchor 400 --target-generation 1120 `
    --light-opponents tactical mcts_v2 mcts_v321 `
    --heavy-opponents mcts_v321 mcts_v5 mcts_v6 mcts_v7 --heavy-pairs 25 `
    --h2h-anchor S880=runs/anchors/S880.pt `
    2>&1 | Tee-Object -FilePath runs/stage8_s640_s2_cont.log

# 4. (창 2) sims100 arm 880 -> 1120, 같은 평가 설정과 anchor
python scripts/run_stage8_training.py --run-dir runs/stage8_s880_sims100 `
    --config configs/stage8_s880_sims100.yaml --anchor 400 --target-generation 1120 `
    --light-opponents tactical mcts_v2 mcts_v321 `
    --heavy-opponents mcts_v321 mcts_v5 mcts_v6 mcts_v7 --heavy-pairs 25 `
    --h2h-anchor S880=runs/anchors/S880.pt `
    2>&1 | Tee-Object -FilePath runs/stage8_s880_sims100.log

# 5. 둘 다 끝나면: 같은 세대 직접 대국 960 / 1040 / 1120
foreach ($g in 960, 1040, 1120) {
  python scripts/run_stage8_head_to_head.py `
      --checkpoint "S2c$($g)=runs/stage8_s640_s2/checkpoints/checkpoint_gen$($g).pt" `
      --checkpoint "N100_$($g)=runs/stage8_s880_sims100/checkpoints/checkpoint_gen$($g).pt" `
      --pairs 50 --output "runs/arm_h2h/n100_gen$($g).json"
}

# 6. 판정
python scripts/compare_teacher_arms.py --control runs/stage8_s640_s2 --teacher runs/stage8_s880_sims100 `
    --direct "runs/arm_h2h/n100_*.json" --teacher-label-prefix N100 --control-label-prefix S2c `
    --output runs/arm_h2h/n100_verdict.json
python scripts/compare_teacher_arms.py --control runs/stage8_s640_s2 --teacher runs/stage8_s640_s2 `
    --output runs/arm_h2h/s2_main_status.json     # 본선 단독 정체/용량 판정(같은 run을 양쪽에 넣음)
```

**해석 기준:**

- 본선이 960/1040/1120에서 S880을 한 번도 넘지 못하면 정체다. raw must_block ≥ 0.6이면 판정은 `capacity`가 되고, Stage 9로 간다.
- sims100이 직접 대국에서 이기고 S880도 넘으면 탐색 예산을 100으로 올린 레시피를 채택한다. 직접 대국만 이기면 `relative_only`다.
- 둘 다 오르면서 차이가 없으면 시간이 덜 드는 50 simulations를 유지한다.

**공유해 줄 것:** 두 run의 `metrics.jsonl`, `external_eval/`, `probes/`, 로그, `runs/stage8_s880_sims100/RECIPE.json`, `runs/arm_h2h/n100_*`,
`runs/arm_h2h/s2_main_status.json`.

## 13. 본선 1120 / sims100 결과와 다음 실행 (2026-10-01)

결과는 [Stage 8 계획 §12.13](stage8-plan.md)에 있다.
- sims100은 기각한다. 정체했고, 직접 대국에서 본선에 0.43으로 졌으며, self-play가 다시 짧아졌다.
- 본선은 960·1040에서 S880을 이겼지만 1120은 유의하지 않다.
- 960 이후에도 오르는지 라운드로빈으로 가린 뒤, 1위 checkpoint를 anchor로 1360까지 이어 간다.

### 13.1 실행 명령 (PowerShell)

```powershell
# 0. 최신 코드
git pull origin feat/stage8-plan

# 1. 본선 내부 라운드로빈: S880 / S960 / S1040 / S1120, 쌍당 100판 (6쌍, 수십 분 추정)
python scripts/run_stage8_head_to_head.py `
    --checkpoint S880=runs/anchors/S880.pt `
    --checkpoint S960=runs/stage8_s640_s2/checkpoints/checkpoint_gen960.pt `
    --checkpoint S1040=runs/stage8_s640_s2/checkpoints/checkpoint_gen1040.pt `
    --checkpoint S1120=runs/stage8_s640_s2/checkpoints/checkpoint_gen1120.pt `
    --pairs 50 --output runs/arm_h2h/s2_roundrobin_880_1120.json
#    마지막에 출력되는 "Sxxx: 점수/게임 (비율)" 줄이 총점 순위다. 1위의 세대를 아래 $best에 넣는다.

# 2. 1위 checkpoint를 새 anchor로 고정 (예: 1위가 S1040이면 $best = 1040)
$best = 1040
Copy-Item "runs/stage8_s640_s2/checkpoints/checkpoint_gen$($best).pt" "runs/anchors/S$($best).pt"
#    1위가 S880이면 복사하지 말고 아래 anchor를 runs/anchors/S880.pt로 쓴다.

# 3. 본선 continuation 1120 -> 1360 (heavy 1200 / 1280 / 1360)
python scripts/run_stage8_training.py --run-dir runs/stage8_s640_s2 `
    --config configs/stage8_s640_temp2.yaml --anchor 400 --target-generation 1360 `
    --light-opponents tactical mcts_v2 mcts_v321 `
    --heavy-opponents mcts_v321 mcts_v5 mcts_v6 mcts_v7 --heavy-pairs 25 `
    --h2h-anchor "S$($best)=runs/anchors/S$($best).pt" `
    2>&1 | Tee-Object -FilePath runs/stage8_s640_s2_cont2.log

# 4. 정체/용량 판정 (같은 run을 양쪽에 넣으면 본선 단독 상태를 본다)
python scripts/compare_teacher_arms.py --control runs/stage8_s640_s2 --teacher runs/stage8_s640_s2 `
    --output runs/arm_h2h/s2_main_status_1360.json

# (선택, 3과 동시에) 사람 대국: 라운드로빈 1위와 직접 두어 보기
python scripts/run_web_play.py --az-checkpoint "runs/anchors/S$($best).pt"
```

**해석 기준:**

- 라운드로빈에서 S1040/S1120이 S960을 유의하게 넘지 못하면 960 이후 정체 신호다. 이때 3단계는 정체를 확정하는 용도가 된다.
- 3단계에서 1200/1280/1360 모두 anchor 대비 향상이 없고(점수 ≤ 0.55 또는 p ≥ 0.05) raw must_block ≥ 0.6이면 `capacity`다.
  이 경우 다음은 Stage 9 설계(96×6 등 network 확대, 처음부터 학습할지 지식을 옮겨 시작할지, GPU self-play 경로)다.
- self-play 흑승이 계속 65% 이상이거나 짧은 게임(≤10수)이 40% 이상으로 다시 늘면 그 구간을 알려 줘. 지금까지는 퇴화가 기력 정체보다 먼저 보였다.

**공유해 줄 것:** `runs/arm_h2h/s2_roundrobin_880_1120.json`, 본선의 `metrics.jsonl`·`external_eval/`·`probes/`·로그,
`s2_main_status_1360.json`, 사람 대국을 했다면 `logs/web_play/`.

## 14. 안정적 지속 향상 루프 (2026-10-01, 외부 검토 반영)

사실 확인과 수정 내용은 [Stage 8 계획 §12.14](stage8-plan.md)에 있다.

- 목표는 "언제 멈출지 판정"이 아니라 **안정적으로 계속 오르는 루프**다.
- 학습 전에 champion을 확정하고 짧은 게임의 원인을 진단한다(A).
- 그다음 adaptive steps와 고정 steps를 40세대 구간 + 게이트로 비교한다(B).
- Stage 9는 §12.14의 네 조건이 모두 성립할 때만 검토한다.

### 14.1 A단계: 진단 (학습 없음)

```powershell
git pull origin feat/stage8-plan

# A1. 본선 내부 라운드로빈 (pair 단위 통계가 함께 출력됨: "pairs W-L p_pairs=")
python scripts/run_stage8_head_to_head.py `
    --checkpoint S880=runs/anchors/S880.pt `
    --checkpoint S960=runs/stage8_s640_s2/checkpoints/checkpoint_gen960.pt `
    --checkpoint S1040=runs/stage8_s640_s2/checkpoints/checkpoint_gen1040.pt `
    --checkpoint S1120=runs/stage8_s640_s2/checkpoints/checkpoint_gen1120.pt `
    --pairs 50 --output runs/arm_h2h/s2_roundrobin_880_1120.json

# A2. 상위 2개 재확인 (예: S1040과 S1120이 상위면). opening seed를 바꿔 새 100쌍
python scripts/run_stage8_head_to_head.py `
    --checkpoint S1040=runs/stage8_s640_s2/checkpoints/checkpoint_gen1040.pt `
    --checkpoint S1120=runs/stage8_s640_s2/checkpoints/checkpoint_gen1120.pt `
    --pairs 100 --seed 9009 --output runs/arm_h2h/s2_top2_confirm.json
#    champion 규칙: 재확인에서 pair p < 0.05로 이긴 쪽. 유의하지 않으면 더 최신 checkpoint를 champion으로 한다.

# A3. self-play 건강 진단 (본선 640~1120, sims100 880~1120), 40세대 구간
python scripts/analyze_self_play_health.py runs/stage8_s640_s2 --from 640 --window 40 `
    --reference-from 880 --reference-reuse 6.4 --output runs/health/s2_640_1120.json
python scripts/analyze_self_play_health.py runs/stage8_s880_sims100 --from 880 --window 40 `
    --reference-from 880 --reference-reuse 6.4 --output runs/health/n100_880_1120.json

# A4. 짧은 게임 원인 분석 (각 30~60분 추정, 게임당 400-sim 탐색 포함)
python scripts/forensic_short_games.py runs/stage8_s640_s2 --from 960 --to 1040 `
    --checkpoint runs/stage8_s640_s2/checkpoints/checkpoint_gen1040.pt --limit 200 `
    --output runs/forensics/s2_960_1040.json
python scripts/forensic_short_games.py runs/stage8_s880_sims100 --from 1040 --to 1120 `
    --checkpoint runs/stage8_s880_sims100/checkpoints/checkpoint_gen1120.pt --limit 200 `
    --output runs/forensics/n100_1040_1120.json
```

A3의 엔트로피 열(H4/H6/H8)과 top6, A4의 `categories`와 opening 군집 상위 5개를 보면 원인을 가를 수 있다.

| 관찰 | 해석 | 다음 단일 변수 |
|---|---|---|
| 짧은 게임이 몇 개 opening 군집에 몰림 + 엔트로피 급락 | opening 탐험 붕괴 | root prior temperature / opening pool |
| `noise`가 대부분 | root noise·temperature가 패착을 만듦 | Dirichlet ε를 낮추거나 noise를 초반에만 |
| `search_budget`가 대부분 | 50 sim target이 부족함 | 어려운 국면만 높은 탐색으로 reanalyse |
| `prior_blind` / `value_blind`가 대부분 | network가 그 전술을 모름 | 지속 혼합 teacher(T4), 최후에 network 확대 |

### 14.2 B단계: adaptive 대 고정, 40세대 구간 + 게이트 (각 arm 6~7시간 추정, 두 창에서 동시)

```powershell
# champion 세대와 anchor (A2 결과로 바꾼다)
$C = 1120
Copy-Item "runs/stage8_s640_s2/checkpoints/checkpoint_gen$($C).pt" "runs/anchors/C$($C).pt"

# 두 arm 모두 champion 상태에서 분기 (학습도 champion에서 시작한다)
python scripts/branch_stage8_run.py --source runs/stage8_s640_s2 --generation $C --dest runs/stage8_ctl_c$C
python scripts/make_recipe_branch.py --source runs/stage8_s640_s2 --generation $C `
    --config configs/stage8_s2_adaptive.yaml --dest runs/stage8_ada_c$C
```

```powershell
# (창 1) 고정 steps 대조군. 창 2는 $run, $cfg만 바꿔 같은 루프를 돌린다:
#   $run = "runs/stage8_ada_c$C"; $cfg = "configs/stage8_s2_adaptive.yaml"
$C = 1120
$run = "runs/stage8_ctl_c$C"; $cfg = "configs/stage8_s640_temp2.yaml"
New-Item -ItemType Directory -Force runs/gates | Out-Null
for ($g = $C + 40; $g -le $C + 240; $g += 40) {
  python scripts/run_stage8_training.py --run-dir $run --config $cfg --anchor 400 `
      --target-generation $g --light-opponents tactical mcts_v2 `
      --heavy-every 40 --heavy-opponents mcts_v5 mcts_v6 mcts_v7 --heavy-pairs 25 `
      --h2h-anchor "C$($C)=runs/anchors/C$($C).pt" `
      2>&1 | Tee-Object -Append -FilePath "$run.log"
  if ($LASTEXITCODE -ne 0) { Write-Host "TRAINING FAILED at $g (exit $LASTEXITCODE)"; break }
  python scripts/segment_gate.py $run --from ($g - 40) --to $g --reference-reuse 6.4 `
      --output "runs/gates/$(Split-Path $run -Leaf)_$($g).json"
  if ($LASTEXITCODE -eq 12) { Write-Host "GATE STOP at $g"; break }
  if ($LASTEXITCODE -ne 10 -and $LASTEXITCODE -ne 11) { Write-Host "GATE FAILED at $g (exit $LASTEXITCODE)"; break }
}
```

```powershell
# B 종료 후: 같은 세대 직접 대국 (C+80, C+160, C+240) → 판정
$C = 1120
foreach ($g in ($C + 80), ($C + 160), ($C + 240)) {
  python scripts/run_stage8_head_to_head.py `
      --checkpoint "ctl$($g)=runs/stage8_ctl_c$($C)/checkpoints/checkpoint_gen$($g).pt" `
      --checkpoint "ADA_$($g)=runs/stage8_ada_c$($C)/checkpoints/checkpoint_gen$($g).pt" `
      --pairs 50 --output "runs/arm_h2h/ada_gen$($g).json"
}
python scripts/compare_teacher_arms.py --control "runs/stage8_ctl_c$C" --teacher "runs/stage8_ada_c$C" `
    --direct "runs/arm_h2h/ada_*.json" --teacher-label-prefix ADA --control-label-prefix ctl `
    --output runs/arm_h2h/ada_verdict.json
```

**B단계 해석:**

- 게이트 종료 코드: 10 PROMOTE, 11 HOLD, 12 STOP. 1·2는 python 오류(예외, 파일 없음)라 게이트 판정이 아니다.
- 게이트 STOP이 난 arm은 그 구간에서 멈춘다. 그 구간에 forensic을 돌린다.
- adaptive가 직접 대국에서 이기고 C도 넘으면(`teacher_better`) adaptive를 기본 레시피로 채택한다.
- 둘 다 C를 넘는데 차이가 없으면 단순한 고정 steps를 유지한다. 대신 게이트는 계속 쓴다.
- 이긴 레시피로 다음 단계(C단계)에 들어간다. 40세대마다 `segment_gate.py`가 PROMOTE를 내면 그 checkpoint를 새 champion으로 복사하고,
  다음 구간의 `--h2h-anchor`로 쓴다.

**공유해 줄 것:** A단계 파일 전부(`runs/arm_h2h/s2_*`, `runs/health/`, `runs/forensics/`), B단계 두 run의 `metrics.jsonl`·`external_eval/`·`probes/`·로그,
`runs/gates/`, `runs/arm_h2h/ada_*`.

## 15. B단계 결과와 C단계: champion 선정 → adaptive로 이어서 학습 (2026-10-02)

결과 표와 해석은 `docs/stage8-plan.md` §12.15에 있다. 요약은 다음과 같다.

- ADA(adaptive steps)를 기본 레시피로 채택한다. 계산량은 같다. ADA는 C1120을 두 번 유의하게 넘었고(1240, 1360), CTL은 1320에 게이트 STOP이 났다.
- 짧은 게임 폭발은 두 arm 모두에서 났다. adaptive가 막은 것은 폭발 자체가 아니라 그 뒤의 고착이다. 폭발의 동반 현상은 색 진동이다.
- 외부 기준(v5/v6/v7)으로 C1120보다 높은 것은 ADA 1320 한 점뿐이다. probe는 표본이 작아 판정에 쓰지 않는다.
- 후보 1240/1320/1360은 seed 8008 개국으로 고른 것이다. champion은 **새 seed**로 고른다.

### 15.1 C0: champion 선정 (학습 없음, 미리 정한 규칙)

규칙은 결과를 보기 전에 고정한다.

1. 새 seed 9009, 100쌍(200판) 라운드로빈: C1120, ADA1240, ADA1320, ADA1360.
2. **자격:** C1120 상대 점수 > 0.55이고 `p_pairs_two_sided` < 0.05.
3. 자격자 중 라운드로빈 총점 1위가 champion이다. 1·2위의 직접 대결 `p_pairs_two_sided` ≥ 0.05이면 동률로 본다.
   동률이면 새 seed 7107 heavy(v5+v6+v7, 50쌍) 총승수가 높은 쪽을 고른다. 그래도 차이가 3승 이하이면 늦은 세대를 고른다.
4. **거부권:** heavy 총승수가 같은 seed의 C1120보다 유의하게 낮으면(두 비율 검정 p < 0.05) 자격을 잃는다.
5. 자격자가 없으면 champion은 C1120으로 둔다. 이 경우 ADA 1360에서 이어 학습하되 anchor는 C1120을 유지한다.

**적용 결과(2026-10-02, `docs/stage8-plan.md` §12.16):**

- 글자 그대로 적용하면 ADA1240이다. heavy 133 대 129로 4승 차이가 "3승 이하" 문턱을 넘는다.
- 3단계의 "3승" 문턱은 300판 대응 비교에서 의미가 없는 값이었다. 차이 4승은 p=0.80이다. 그래서 다음과 같이 고친다.
  **동률 깨기도 heavy 대응 비교 p < 0.05일 때만 적용하고, 아니면 늦은 세대를 고른다.**
- 이 수정으로 champion은 **ADA1360**이다. 결과를 본 뒤의 수정이라는 점을 기록해 둔다.
- 앞으로의 선정 규칙에서 모든 동률 깨기는 유의성으로 정의한다.

```powershell
# 창 1: worktree 최신화 후 라운드로빈 (NN끼리라 빠르다)
cd "C:\오목 강화학습\renju-stage8"
& "C:\오목 강화학습\renju-alphazero\.venv-cpu\Scripts\Activate.ps1"
git fetch origin feat/stage8-plan
git merge --ff-only origin/feat/stage8-plan
$R = "C:\오목 강화학습\renju-alphazero\runs"
$A = "$R\stage8_ada_c1120\checkpoints"
New-Item -ItemType Directory -Force "$R\champion" | Out-Null
python scripts/run_stage8_head_to_head.py `
    --checkpoint "C1120=$R\anchors\C1120.pt" `
    --checkpoint "ADA1240=$A\checkpoint_gen1240.pt" `
    --checkpoint "ADA1320=$A\checkpoint_gen1320.pt" `
    --checkpoint "ADA1360=$A\checkpoint_gen1360.pt" `
    --pairs 100 --seed 9009 --output "$R\champion\rr_seed9009.json"
```

```powershell
# 창 2: 새 seed heavy (C1120 기준점도 같은 개국으로 다시 잰다)
cd "C:\오목 강화학습\renju-stage8"
& "C:\오목 강화학습\renju-alphazero\.venv-cpu\Scripts\Activate.ps1"
$R = "C:\오목 강화학습\renju-alphazero\runs"
$A = "$R\stage8_ada_c1120\checkpoints"
New-Item -ItemType Directory -Force "$R\champion" | Out-Null
$cands = [ordered]@{ "C1120" = "$R\anchors\C1120.pt"; "ADA1240" = "$A\checkpoint_gen1240.pt";
                     "ADA1320" = "$A\checkpoint_gen1320.pt"; "ADA1360" = "$A\checkpoint_gen1360.pt" }
foreach ($k in $cands.Keys) {
  python scripts/run_stage7_checkpoint_eval.py --checkpoint $cands[$k] `
      --opponents mcts_v5 mcts_v6 mcts_v7 --pairs 50 --seed 7107 --tactical-rules off `
      --output "$R\champion\heavy_$($k)_seed7107.json"
  if ($LASTEXITCODE -ne 0) { Write-Host "HEAVY FAILED at $k"; break }
}
```

```powershell
# 창 1, 라운드로빈 뒤: 사전 등록한 arm 직접 대결 (CTL은 1320에서 멈췄으므로 1360 대신 1320)
foreach ($g in 1200, 1280, 1320) {
  python scripts/run_stage8_head_to_head.py `
      --checkpoint "ctl$($g)=$R\stage8_ctl_c1120\checkpoints\checkpoint_gen$($g).pt" `
      --checkpoint "ADA_$($g)=$A\checkpoint_gen$($g).pt" `
      --pairs 50 --output "$R\arm_h2h\ada_gen$($g).json"
}
# 짧은 게임 폭발 창 forensic (깊은 탐색 400회라 오래 걸린다; 창 하나에서 순서대로)
python scripts/forensic_short_games.py "$R\stage8_ada_c1120" --from 1240 --to 1280 `
    --checkpoint "$A\checkpoint_gen1260.pt" --limit 200 --output "$R\forensics\ada_1240_1280.json"
python scripts/forensic_short_games.py "$R\stage8_ctl_c1120" --from 1280 --to 1320 `
    --checkpoint "$R\stage8_ctl_c1120\checkpoints\checkpoint_gen1300.pt" --limit 200 `
    --output "$R\forensics\ctl_1280_1320.json"
```

- forensic의 checkpoint는 창 중간 세대다. 그 창의 게임을 실제로 둔 정책에 가깝다.
- numpy 경고는 무해하다. 정리하려면 `.venv-cpu`를 쓰는 작업이 하나도 없을 때 `pip install numpy`를 실행한다.

### 15.2 C1: champion에서 ADA로 이어 학습, PROMOTE마다 anchor 교체

`$X = 1360`으로 확정했다(15.1 적용 결과). 기존 run(`stage8_ada_c1120`)에서 분기 없이 이어 간다.

루프는 `scripts/run_champion_loop.py` 하나다. 상태(현재 champion, 연속 HOLD, 다음 구간)를 게이트 파일에서 다시 만들기 때문에,
**처음 시작과 중단 뒤 재시작이 같은 명령**이다. 40세대 구간마다 다음을 한다.

1. 학습을 구간 끝까지 진행한다. `latest.pt`에서 재개하고, 이미 있는 평가 파일은 건너뛴다.
2. `segment_gate.py`로 판정한다.
3. PROMOTE이면 그 checkpoint를 `anchors\ADA<세대>.pt`로 복사하고 다음 구간의 anchor로 쓴다.

멈추는 조건과 종료 코드: 1600 도달(0), STOP(12), 연속 3구간 HOLD = 정체(13), 하위 단계 실패(1).

```powershell
# 창 1 (처음 시작과 재시작 모두 이 블록)
cd "C:\오목 강화학습\renju-stage8"
& "C:\오목 강화학습\renju-alphazero\.venv-cpu\Scripts\Activate.ps1"
$R   = "C:\오목 강화학습\renju-alphazero\runs"
$run = "$R\stage8_ada_c1120"
if (-not (Test-Path "$R\anchors\ADA1360.pt")) {
  Copy-Item "$run\checkpoints\checkpoint_gen1360.pt" "$R\anchors\ADA1360.pt"
}
python scripts/run_champion_loop.py --run-dir $run --config configs/stage8_s2_adaptive.yaml `
    --start 1360 --end 1600 --champion "ADA1360=$R\anchors\ADA1360.pt" `
    --anchors-dir "$R\anchors" --gates-dir "$R\gates" --log "$run.c1.log"
Write-Host "loop exit $LASTEXITCODE"
```

**중단 뒤 재시작(정전, 재부팅, Ctrl+C, 창 닫힘):** 위 블록을 그대로 다시 실행한다.

- 학습 도중에 꺼졌으면 마지막 `latest.pt`(완료된 세대)부터 이어 간다. 진행 중이던 세대는 버리고 다시 둔다.
- 구간 끝 평가(heavy, champion 대결, probe) 도중에 꺼졌으면 없는 파일만 다시 만든다. 파일은 원자적으로 쓰므로 반쯤 쓴 파일이 남지 않는다.
- 게이트 직전에 꺼졌으면 그 구간은 학습 없이 게이트만 돈다.
- PROMOTE 뒤 anchor 복사 전에 꺼졌으면 복사를 다시 한다.
- 시작할 때 `champion ADA…, consecutive HOLDs …` 줄로 복원된 상태를 보여 준다.

**재시작 전에 하지 말 것:**

- `latest.pt`, `checkpoints\`, `runs\gates\stage8_ada_c1120_14*.json` 같은 파일을 지우거나 옮기지 않는다. 상태가 이 파일들에 있다.
- 원본 폴더에서 `git checkout`이나 `git reset`을 하지 않는다. worktree(`renju-stage8`)는 괜찮다.

**확인만 할 때(학습 없음):**

```powershell
Get-ChildItem "$R\gates\stage8_ada_c1120_1[4-6]*.json" | ForEach-Object {
  $j = Get-Content $_ -Raw | ConvertFrom-Json; "$($_.Name)  $($j.decision)  $($j.reason)" }
python -c "import sys; sys.path.insert(0, 'src'); from training.training_checkpoint import load_checkpoint_payload as load; print(load(r'$run\checkpoints\latest.pt')['generation'])"
```

**C1과 함께 돌릴 수 있는 진단(선택):** forensic의 깊은 탐색을 100/200회로 바꿔 같은 ADA 창을 다시 본다.
그래서 몇 회부터 막는 수를 고르는지 본다. 이 곡선이 다음 단일 변수(self-play 탐색 50 → 100을 adaptive 아래에서 재시험할지)의 근거다.
C1과 CPU를 나눠 쓰므로 C1이 느려진다.

```powershell
foreach ($d in 100, 200) {
  python scripts/forensic_short_games.py "$R\stage8_ada_c1120" --from 1240 --to 1280 `
      --checkpoint "$R\stage8_ada_c1120\checkpoints\checkpoint_gen1260.pt" --limit 100 `
      --deep-simulations $d --output "$R\forensics\ada_1240_1280_deep$($d).json"
}
```

**C1 해석:**

- 연속 3구간(120세대) 동안 PROMOTE가 없으면 이 레시피는 정체한 것이다. 그때 forensic 결과로 다음 단일 변수를 고른다(§12.14 끝, §12.15 끝).
- STOP이 나면 그 창을 forensic으로 분류한다. ADA에서 STOP이 나는 경우는 짧은 게임 40% 이상이 두 창 연속일 때뿐이다(reuse는 고정이다).
- anchor가 바뀌면 이후 h2h 점수는 새 champion 기준이다. 그래서 구간 사이 점수를 그대로 비교하지 않는다.
  장기 추세는 heavy(v5/v6/v7, seed 7007 고정)로 본다.

**공유해 줄 것:** `runs/champion/`, `runs/arm_h2h/ada_gen*.json`, `runs/forensics/`. C1을 돌렸다면
run의 `metrics.jsonl`·`external_eval/`·`probes/`·로그와 `runs/gates/`도 함께 보낸다.

## 16. C2: 학습률 A/B (1e-3 대 3e-4), ADA gen 1480에서 (2026-10-03, 외부 검토 반영)

배경과 사전 판정 규칙은 `docs/stage8-plan.md` §12.17에 있다.
요약: C1은 3구간 HOLD로 정체했고, 색 우세 상태가 반복해서 뒤바뀐다. 고정 학습률 1e-3이 원인이라는 가설을 검증한다.
두 arm 모두 anchor를 ADA1360으로 고정한다(`--fixed-anchor`). 같은 명령으로 시작과 재시작을 모두 한다.

```powershell
# 공통 준비 (한 번만): 코드 최신화 + LR arm 분기
cd "C:\오목 강화학습\renju-stage8"
& "C:\오목 강화학습\renju-alphazero\.venv-cpu\Scripts\Activate.ps1"
git fetch origin feat/stage8-plan
git merge --ff-only origin/feat/stage8-plan
$R = "C:\오목 강화학습\renju-alphazero\runs"
if (-not (Test-Path "$R\stage8_lr3_c1480")) {
  python scripts/make_recipe_branch.py --source "$R\stage8_ada_c1120" --generation 1480 `
      --config configs/stage8_ada_lr3e4.yaml --dest "$R\stage8_lr3_c1480"
}
```

```powershell
# 창 1: CTL (학습률 1e-3, 기존 ADA run을 1493에서 이어 감). 재시작도 이 블록.
cd "C:\오목 강화학습\renju-stage8"
& "C:\오목 강화학습\renju-alphazero\.venv-cpu\Scripts\Activate.ps1"
$R = "C:\오목 강화학습\renju-alphazero\runs"
python scripts/run_champion_loop.py --run-dir "$R\stage8_ada_c1120" `
    --config configs/stage8_s2_adaptive.yaml --start 1480 --end 1600 --fixed-anchor `
    --champion "ADA1360=$R\anchors\ADA1360.pt" --anchors-dir "$R\anchors" `
    --gates-dir "$R\gates" --log "$R\stage8_ada_c1120.c2.log"
Write-Host "loop exit $LASTEXITCODE"
```

```powershell
# 창 2: LR (학습률 3e-4, 1480 분기). 재시작도 이 블록.
cd "C:\오목 강화학습\renju-stage8"
& "C:\오목 강화학습\renju-alphazero\.venv-cpu\Scripts\Activate.ps1"
$R = "C:\오목 강화학습\renju-alphazero\runs"
python scripts/run_champion_loop.py --run-dir "$R\stage8_lr3_c1480" `
    --config configs/stage8_ada_lr3e4.yaml --start 1480 --end 1600 --fixed-anchor --prefix LR `
    --champion "ADA1360=$R\anchors\ADA1360.pt" --anchors-dir "$R\anchors" `
    --gates-dir "$R\gates" --log "$R\stage8_lr3_c1480.c2.log"
Write-Host "loop exit $LASTEXITCODE"
```

```powershell
# 두 창 모두 exit 0 이후: 직접 대결(세대마다 다른 seed) → 판정 → 건강 표
$R = "C:\오목 강화학습\renju-alphazero\runs"
$seeds = @{ 1520 = 9109; 1560 = 9110; 1600 = 9111 }
foreach ($g in 1520, 1560, 1600) {
  python scripts/run_stage8_head_to_head.py `
      --checkpoint "CTL$($g)=$R\stage8_ada_c1120\checkpoints\checkpoint_gen$($g).pt" `
      --checkpoint "LR$($g)=$R\stage8_lr3_c1480\checkpoints\checkpoint_gen$($g).pt" `
      --pairs 100 --seed $seeds[$g] --output "$R\arm_h2h\lr_gen$($g).json"
}
python scripts/compare_recipe_arms.py --control "$R\stage8_ada_c1120" --treatment "$R\stage8_lr3_c1480" `
    --from 1480 --to 1600 --points 1520 1560 1600 --direct "$R\arm_h2h\lr_gen*.json" `
    --control-prefix CTL --treatment-prefix LR --output "$R\arm_h2h\lr_verdict.json"
python scripts/analyze_self_play_health.py "$R\stage8_ada_c1120" --from 1440 --to 1600 --window 40 `
    --reference-from 1440 --reference-reuse 6.4 --output "$R\health\ctl_c2_1440_1600.json"
python scripts/analyze_self_play_health.py "$R\stage8_lr3_c1480" --from 1440 --to 1600 --window 40 `
    --reference-from 1440 --reference-reuse 6.4 --output "$R\health\lr_c2_1440_1600.json"
```

- 공통 준비의 분기 출력에 `"optimizer.lr": {"from": 0.001, "to": 0.0003}`가 보여야 한다.
  학습 중에는 `stage8_lr3_c1480\metrics.jsonl`의 `train` 줄에서 `"lr": 0.0003`을 확인할 수 있다.
- 종료 코드: 0 완료, 12 STOP(그 arm만 멈춘다. 다른 창은 계속), 1 하위 단계 오류(같은 블록을 재실행). `--fixed-anchor`에서는 정체(13)로 멈추지 않는다.
- 한 arm이라도 게이트 STOP(exit 12)으로 1600 전에 멈추면, 직접 대결과 판정은 **두 arm이 모두 도달한 구간 끝**까지만 한다.
  예: 둘 다 1560에서 STOP이면 `--to 1560 --points 1520 1560`으로 하고 1600 대결은 하지 않는다. 판정 스크립트는 빠진 heavy 파일을 먼저 확인한다.
- `compare_recipe_arms.py`의 마지막 줄 `verdict:`가 사전 규칙의 판정이다(`adopt_stronger` / `reject` / `adopt_stability` / `keep_control`).

**공유해 줄 것:** `runs/arm_h2h/lr_gen*.json`, `lr_verdict.json`, `runs/health/*_c2_*.json`,
`runs/gates/stage8_ada_c1120_15*.json`·`_1600.json`, `runs/gates/stage8_lr3_c1480_*.json`,
두 run의 `metrics.jsonl`·`external_eval/gen15*·gen1600*`·`probes/gen15*·gen1600*`, 두 `.c2.log`.

## 17. C2 이후: STOP 창 forensic 탐색 곡선, 색별 + P/Q/N (학습 없음, 2026-10-04, 외부 검토 반영)

결과와 판정(`keep_control`)과 판정표는 `docs/stage8-plan.md` §12.18에 있다.
두 arm의 STOP 창(1520-1560)에서 checkpoint 1540을 고정하고, 깊은 탐색 횟수만 100/200/400으로 바꾼다.
같은 `--seed`와 `--limit`이라 세 번 모두 같은 표본이다. 기본 탐색(50회, self-play 설정)과 noise 시행은 매번 같이 계산된다.

forensic에 새로 추가된 것:

- 진 쪽 색(`loser`), 색별 요약(`by_loser`, 화면에는 `loser BLACK` / `loser WHITE` 줄로 출력).
- 막는 수들의 N(`*_n_safe`), 가장 좋은 막는 수의 Q(`*_q_safe_best`), 선택된 수의 Q(`*_q_chosen`), 가장 큰 prior(`prior_safe_max`).

```powershell
cd "C:\오목 강화학습\renju-stage8"
& "C:\오목 강화학습\renju-alphazero\.venv-cpu\Scripts\Activate.ps1"
git fetch origin feat/stage8-plan
git merge --ff-only origin/feat/stage8-plan
$R = "C:\오목 강화학습\renju-alphazero\runs"
$arms = [ordered]@{ "ctl" = "$R\stage8_ada_c1120"; "lr" = "$R\stage8_lr3_c1480" }
foreach ($k in $arms.Keys) {
  foreach ($d in 100, 200, 400) {
    python scripts/forensic_short_games.py $arms[$k] --from 1520 --to 1560 `
        --checkpoint "$($arms[$k])\checkpoints\checkpoint_gen1540.pt" --limit 100 `
        --deep-simulations $d --output "$R\forensics\c2_$($k)_1520_1560_deep$($d).json"
    if ($LASTEXITCODE -ne 0) { Write-Host "FORENSIC FAILED $k $d"; break }
  }
}
```

- 두 창으로 나눠 돌려도 된다(창 1은 `ctl`, 창 2는 `lr`만 남긴다). 400회가 가장 오래 걸린다.
- CTL 창은 흑 우세(1520~1544)와 백 우세(1545~)가 섞여 있어서 양쪽 색의 실패가 모두 나온다. LR 창은 거의 백이 진 게임이다.
- 판정 스크립트(`compare_recipe_arms.py`)의 probe 검사에 가치망 분리도를 추가했다. C2를 다시 계산해도 판정은 `keep_control`이다.

**공유해 줄 것:** `runs/forensics/c2_*.json` 6개.
