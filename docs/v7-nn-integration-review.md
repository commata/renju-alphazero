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
| 근본 원인은 VCT | **가설(가장 유력)** | VCF가 없던 국면에서 사람이 한 수 둔 뒤 모든 후보가 VCF 패배가 됐다. 삼을 섞은 forcing 수순과 맞지만 VCT solver로 증명하지 않았다 |
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
  하지만 증명은 VCT solver certificate가 있어야 한다(§4 작업 2).
- **stage 4는 원인 후보일 뿐이다.** 마지막 SAFE 착수에서 SAFE 후보가 2~4개뿐이었다(`164723` 20수 제외).
  그 후보들이 사람의 응수 뒤에도 살아남는지는 확인하지 못했다(3수 앞 확인은 비용이 컸다).
  "stage 4가 이길 수 있는 방어를 두고 틀린 수를 골랐다"는 아직 말할 수 없다.
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

수집기와 추출기는 frozen 파일을 수정하지 않는다. 새 모듈에서 v7 함수를 호출하고, 재생한 착수가 frozen v7과
같다는 fingerprint 테스트를 둔다.

### 3.3 VCT/open-three probe와 offline solver

- 사람 승리 4판의 마지막 SAFE 착수 국면(§1.3 굵은 행)과 그 직후 국면을 추출한다. D4 대칭 8배로 `must_defend_vct` probe를 만든다.
- **offline bounded VCT solver**(분석 전용, 탐색에 넣지 않음)로 각 국면의 정답을 certificate로 검증한다. 검증되지 않은
  국면은 probe에서 뺀다. solver는 방어자 4 반격 수순도 다뤄서 §1.4의 SAFE 범위 한계를 함께 해소한다.
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
| 2 | VCT/open-three probe + offline bounded VCT solver(분석 전용) | 4판에서 추출한 국면이 certificate로 검증되고, D4 확장 probe fixture가 생성됨. solver는 search/학습 경로에서 import되지 않음(격리 테스트) | |
| 3 | 웹 대국 AlphaZero 에이전트 + 착수별 root 로그(§3.4) | B400과 대국 가능, 로그로 `pi`·top-k Q 재구성 가능, 기존 V2~V7 대국 동작 불변 | |
| 4 | B400 continuation 유지 | heavy 지점 B480·B560·B640에서 B400 anchor h2h, plateau 판정 기록 | 진행 |
| 5 | 증명된 전술 국면 추출기 + teacher dataset | frozen 파일 불변, fingerprint 일치, 공급원·종류별 개수와 증명 방식 기록 | |
| 6 | Teacher arm: B400 fine-tune → probe → 분기 self-play | fine-tune 전후 probe 표(기존 + VCT). 같은 판수 지점에서 대조군과 §3.6 순서로 비교 | |
| 7 | Stage 9 진입 판단 | §3.7 표에 따라 결정하고 근거를 이 문서와 Stage 8 계획에 기록 | |

작업 2와 3은 서로 독립이라 병행할 수 있다. 작업 4는 계속 돌아간다. 작업 6은 5가 끝나야 시작한다.

## 5. 재현

```bash
python scripts/analyze_web_play_losses.py logs/web_play --losses-only --tail 6
python -m unittest tests.test_analyze_web_play_losses
```

기본값은 노드 한도 100,000, 4 연쇄 한도 6이다. 한 판에 수 분이 걸리고, 대부분의 시간은 패배가 증명되는 국면에서 쓴다.
