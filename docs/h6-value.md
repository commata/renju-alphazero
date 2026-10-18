# H6 value 학습: H6-0 인프라 (2026-10-11)

브랜치 `feat/h6-value`(`feat/mcts-v8-teacher` `80390d8`에서 분기). 설계는 [mcts-v8-teacher.md §12.18–12.19](mcts-v8-teacher.md),
[track-b-post-e1.md §4](track-b-post-e1.md), [track-b-b1-b2.md](track-b-b1-b2.md)를 따른다.
E1-holdout이 데스크톱에서 `feat/mcts-v8-teacher`로 도는 동안 병렬로 만든다(§12.34 조건).

## 병렬로 진행해도 되는 이유와 경계

- 엔진 코드(`analysis/mcts_v8.py`, `e3_arms.py`, `s3_vct2_v1.py`)와 E1 도구는 건드리지 않는다. 새 파일을 추가하고, runner에는 옵션 하나만 추가했다.
  그래서 E1-holdout 결과의 provenance(git_commit, git_dirty, 설정 해시)에는 영향이 없다. 데스크톱의 E1-holdout 작업 폴더에서 이 브랜치를 checkout하거나 pull하지 않는다.
- 하는 것: 라벨 행 스키마, 라벨 규칙, probe 제외, 분할·누출 검사, provenance 검사, label builder CLI, 테스트, H6-5용 runner 옵션.
- 하지 않는 것: H6-1 라벨 세트 실제 생성(TB-Engine-v1 동결 뒤), value 학습, value leaf 통합(H6-4).

## 구현

| 파일 | 내용 |
|---|---|
| `src/hybrid/h6_labels.py` | 행 스키마 `h6-value-row-v1`, 라벨 규칙, probe·split·누출·provenance 검사, 보고 |
| `scripts/build_h6_labels.py` | H2 RenjuNet 게임 → value 라벨 행(builder만). 누출 검사를 통과해야 출력을 쓴다 |
| `scripts/run_mcts_v8_benchmark.py` | `--opponent-puct-c`: PUCT 상대(H6-5의 주 상대 `v8:puct_policy`)를 허용. key에 `[puct_c=…]`가 붙는다 |
| `tests/test_h6_labels.py` | 아래 계약 전부 |

### 행 (`h6-value-row-v1`, 국면 하나당 한 행, 그 국면은 대국의 `ply`수 뒤)

`game_id, source, split, ply, moves, side_to_move, value_target, value_mask, value_perspective = "side_to_move", label_kind, weight,
canonical_hash(D4, 둘 차례 포함), solver_class, solver_depth, solver_budget, engine_config_hash, model_hash, git_commit, flags`

### 라벨 규칙 (§12.18 계약 + track-b-post-e1.md §4)

- **value는 항상 둘 차례 관점이다.** 증명된 승리 +1, 증명된 패배 −1, 0은 실제로 끝난 무승부(판이 다 참)일 때만 준다. 합의 무승부와 결과를 알 수 없는 국면은 mask한다.
- **증명만 라벨이다.**
  - 증명 라벨은 `analysis.tactical_labels`의 VCF 계열 판정(`immediate_win`, `unstoppable_four`, `vcf` +1 / `forced_loss` −1, 옵션 `vcf_loss` −1)을 쓴다.
  - 옵션으로 V8-B의 깊이 1 공격 증명(`vct1_win` +1)을 더한다.
  - SAFE, NOT_FOUND, UNKNOWN, VCT2_CLEAR는 라벨이 아니다.
- **증명이 대국 결과보다 우선이다.**
  - 증명이 있는 국면은 증명 행으로 바꾼다.
  - 대국 결과와 어긋나는 첫 증명 지점부터는 결과 행을 mask한다(`outcome_conflicts_proof`). 예: 증명된 패배를 상대가 응징하지 못한 대국.
- RenjuNet 결과는 작은 weight(기본 0.1)를 준다. 종국의 81%가 기권·시간·합의(§12.6)이고, 이런 판에는 `end_unknown`을 표시한다.

### 누출 방지

- probe와 평가 국면에 한 번이라도 닿은 대국은 **통째로** 뺀다. 기본 목록은 다음과 같다.
  - `tests/fixtures/*probes*.json`
  - `docs/mcts-v8-results/probe_*.json`(P92–P94)
  - E1 계열 manifest(`e1*/*manifest.json`: E1-dev, E1-S, 그리고 앞으로 E1-holdout)
- 분할은 대국 단위로 한다. RenjuNet은 H2의 대회 단위 분할을 그대로 쓰고, 그 밖의 출처는 `split_for(game_id)`(salt 고정, 5%/5%)를 쓴다.
- 앞선 분할에 이미 나온 국면은 뒤 분할에서 mask한다(train < val < test, H2의 `masked_plies`도 반영). 그 뒤 `check_leakage`가 mask되지 않은 교차 국면이 남아 있으면 **출력을 거부**한다.
- 한 대국 안에서 엔진 설정이나 모델이 바뀌면 거부한다(`check_game_provenance`, H7용).
- 벤치마크·평가 seed(8401–8405, 8411–8418)의 대국은 학습 데이터로 쓰지 않는다(`check_reserved_seed`). 8403–8405는 H6-5 평가용으로 예약돼 있다.
- RenjuNet에서 나온 행은 `data/` 또는 `runs/`(git 제외)에만 쓴다. 다른 경로로 쓰려 하면 CLI가 거부한다.

### 미룬 것

- `--value-checkpoint`(§12.19 H6-0 목록)는 **H6-4로 미룬다.** value leaf가 아직 없어서 지금 넣으면 아무 효과 없는 옵션이 된다. H6-4에서 leaf 통합(leaf 노드 값 = −v, 부호 테스트)과 함께 넣는다.

## H6-1 결과 (2026-10-13, `docs/h6/h6_1_report.json`)

실행: 커밋 `b47bdd6`, RenjuNet H2 전체(train/val/test), VCF 증명 + 결과, `from_ply 5`, 결과 weight 0.1, 증명 weight 1.0.
라벨 코드(`h6_labels.py`, `build_h6_labels.py`의 라벨 로직)는 지금 HEAD와 같다. 그 뒤 바뀐 것은 진행률 출력뿐이다.

| 항목 | 값 |
|---|---|
| 입력 대국 | 108,011 |
| probe로 제외된 대국 | **34,067 (31.5%)** |
| 남은 대국 | 73,944 |
| 행 / 라벨 있는 행 | 2,409,192 / 1,986,865 (82.5%) |
| 분할(행) | train 2,174,614 (90.3%) · val 120,454 (5.0%) · test 114,124 (4.7%) |
| mask 422,327 | 합의 무승부 등 결과 없음 ≈ 335,263 · H2 중복 58,590 · 증명과 결과 충돌 28,474 · 교차 분할 추가 0 |
| 결과 라벨 | 1,872,867 (94.3%). 이 중 85.6%(1,603,953)가 `end_unknown`(기권·시간·합의) |
| 증명 라벨 | 113,998 (5.7%): vcf 68,781 · unstoppable_four 19,509 · immediate_win 14,408 · forced_loss 11,300 |
| 증명 부호 | +1 102,698 (90.1%) · −1 11,300 (9.9%) |
| 결과 부호 | +1 911,207 (48.7%) · −1 961,660 (51.3%) |
| 전체 부호 | +1 51.0% · 흑 차례 +1 50.7% · 백 차례 +1 51.4% |
| 학습 weight 비중 | 증명 37.8% · 결과 62.2% (113,998 : 187,287) |
| 깊이 | 0만 있음(`--vct1` 끔) |

판독:
- **라벨 계약은 정상이다.** 부호가 흑·백 양쪽에서 균형이고, 누출 검사를 통과했고, provenance가 맞다. 교차 분할 추가 mask가 0인 것은 H2의 `masked_plies`(58,590)가 같은 D4 규칙으로 이미 처리했기 때문이다.
- **증명 −1이 적다(9.9%).** 예상한 대로다. VCF 증명은 둘 차례의 승리이고, −1은 `forced_loss`뿐이다. H6-3에서 −1 쪽을 따로 본다. 늘려야 하면 `--prove-losses`(별도 실행)를 쓴다.
- **증명이 행으로는 5.7%지만 weight로는 37.8%다.** 10:1 weight가 의도대로 작동한다.
- **문제: probe 제외가 과하다.**
  - 원인: E1·E1-holdout manifest의 E1-C(대조군) 국면 11개가 **3수 국면이다. 모두 중앙 5×5 안이라 RIF 표준 opening 그 자체다.**
  - "probe에 닿은 대국은 통째로 제외" 규칙 때문에, 그 opening으로 둔 RenjuNet 대국이 전부 빠졌다. 26개 표준 opening 중 최대 11개 계열이다.
  - 그 결과 데이터 31.5%를 잃었고, opening 분포도 한쪽으로 치우쳤다. H3 policy는 이 대국들로 학습했으므로 value만 이 opening들을 모르게 된다.
  - 통째 제외의 목적은 중반 전술 probe(P92–P94, E1-P/N, VCT probe)의 답이 대국 진행으로 새는 것을 막는 데 있다. 수가 적은 opening 국면은 수많은 대국이 거치는 공통 국면이라 이 목적에 맞지 않는다.

조치(`build_h6_labels.py --probe-game-min-ply N`, 기본 0 = v1 동작). 검토 기록: [`h6/h6_1_review.md`](h6/h6_1_review.md).
- 돌이 N개 이상인 probe에 닿은 대국은 통째로 뺀다. 그보다 이른 probe는 그 국면 행만 mask한다(`probe_position`). 보고서에 `dropped.probe_game_by_ply`를 남긴다.
- **N = 4로 확정했다(처음 제안한 9는 철회).**
  - 처음 분석에서 "전술 probe는 모두 12수 이상"이라고 한 것은 **틀렸다.**
  - `stage7_probes_defense_v1.json`에는 돌 5–8개짜리 `must_defend_open3` 전술 probe가 8개 있다.
  - N = 9로 하면 이 probe의 국면만 mask되고 이후 진행은 학습에 들어간다. 그러면 답이 샌다.
- 행 단위 예외는 3돌 opening probe만 허용한다(`MAX_ROW_ONLY_PROBE_PLY = 3`). 4돌 이상 probe가 행 단위로 처리되게 만드는 N은 builder가 `ValueError`로 거부한다. probe 정보가 ply 없는 hash 집합으로만 주어졌을 때 N > 0을 쓰는 경우도 거부한다.
- 남는 위험: 돌 4–8개 probe는 39개다(E1-C 대조군 포함). 4–5돌 국면도 흔한 opening 진행이면 N = 4에서도 대국이 많이 빠질 수 있다. 이 규모는 audit의 `games_dropped_by_min_ply["4"]`로 확인한다.
- 학습 weight로 본 부호 비중은 +1 64.3% · −1 35.7%다(증명 +1 102,698 + 결과 +1 911,207×0.1 대 11,300 + 961,660×0.1). gate 실패 사유는 아니지만, H6-3에서 −1 쪽 지표를 따로 보고한 뒤 `--prove-losses`가 필요한지 결정한다.
- `scripts/h6_probe_audit.py`: 증명 없이 몇 분 만에, probe별로 닿는 대국 수와 N별 제외 대국 수를 센다. 개수만 들어 있으므로 커밋해도 된다.

**probe audit 결과** (`docs/h6/h6_probe_audit.json`, 커밋 `91ba84b`, 108,011판, probe 국면 327개 중 RenjuNet에 나타난 것 22개):

| N | 통째로 빠지는 대국 |
|---|---|
| 0 (v1) | 34,067 (31.5%) |
| **4 (v2)** | **888 (0.82%)** |
| 5 / 6 / 7 / 8 | 440 / 339 / 300 / 300 |
| 9 이상 | 0 (돌 9개 이상 probe는 RenjuNet 대국에 한 번도 나타나지 않음) |

- **v1에서 빠진 대국의 97.4%(33,179판)는 3돌 opening probe 7개 때문이었다.** 가장 큰 것은 `8401-p19-white/3`(11,711판)이고, 다음은 e1h `8418-p12`(5,408)·`p8`(4,015), e1 `8411-p20`(3,948), e1h `8418-p13`(3,447)·`p2`(3,218), e1 `8402-p8`(1,705)이다.
- N = 4에서도 빠지는 888판에 관련된 probe(probe별 대국 수는 서로 겹치므로 합하면 888보다 크다. 888은 대국 단위 합집합이다):
  - 4돌 E1-C 2개: 각각 317판, 150판
  - 8돌 `8402-p2-black/8`: 296판
  - 5–8돌 probe들(stage7 `must_defend_open3` 39판·29판 포함)
- N = 9 대신 N = 4로 정해서 더 잃는 대국은 888판뿐이다. 엄격한 정책의 비용이 작다는 것이 확인됐다.
- 분할별로 probe에 닿는 비율은 train 31.4%, val 31.4%, test 34.7%로 비슷하다. 이 값은 아무 probe에나 닿은 대국의 비율, 즉 N = 0(v1) 기준이다. 따라서 v1 제외가 특정 분할에 몰리지 않았다는 것까지만 말할 수 있다. N = 4의 888판이 분할별로 어떻게 빠졌는지는 audit에 기록되지 않았으므로, v2 보고서의 `by_split`으로 확인한다.
- v2 예상: 대국 약 107,123판(v1의 1.45배), 행 약 350만. 실행 시간도 v1의 약 1.45배다.

결정: **H6-2a는 v1 라벨로 시작하지 않는다.** audit에서 N = 4의 효과를 확인하고 v2 라벨(`--probe-game-min-ply 4`, 다른 설정은 v1과 같게)을 만든 뒤, v2 보고서로 H6-1 gate를 다시 판정한다. H6-2a 설정(`h6_value_head.yaml`)은 v2 경로를 가리킨다.

## H6-1 v2 결과와 gate (2026-10-14, `docs/h6/h6_1_report_v2.json`)

실행 조건:
- 커밋 `e68f441`(라벨 코드는 `57c7eab`와 같다. 그 뒤 커밋은 문서만 바꿨다).
- v1과 다른 설정은 `--probe-game-min-ply 4` 하나다. `restricted false`.

| 항목 | v1 | v2 | v2/v1 |
|---|---|---|---|
| 남은 대국 | 73,944 | **107,123** | 1.449 |
| probe로 제외 | 34,067 | **888** (4돌 467 · 5돌 82 · 6돌 39 · 8돌 300) | — |
| 행 / 라벨 있는 행 | 2,409,192 / 1,986,865 | 3,499,207 / **2,893,475** | 1.452 / 1.456 |
| 라벨 비율 | 82.5% | 82.7% | |
| 분할(행) train · val · test | 90.3 · 5.0 · 4.7% | 90.1 · 5.0 · 4.9% | test 1.51, train 1.45, val 1.44 |
| 증명 | 113,998 (5.74%) | 167,757 (5.80%) | 1.472 |
| ┗ vcf · 막을 수 없는 4 · 즉시 승리 · forced_loss | 68,781 · 19,509 · 14,408 · 11,300 | 99,783 · 28,852 · 22,064 · 17,058 | 1.45 · 1.48 · 1.53 · 1.51 |
| 증명 +1 / −1 | 90.1 / 9.9% | 89.8 / 10.2% | |
| 결과 +1 / −1 | 48.7 / 51.3% | 48.7 / 51.3% | |
| 행 수 기준 +1 / −1 | 51.0 / 49.0% | 51.0 / 49.0% | |
| 흑 차례 +1 · 백 차례 +1 | 50.7 · 51.4% | 50.1 · 52.0% | |
| weight 기준 +1 / −1 | 64.3 / 35.7% | **64.3 / 35.7%** (283,333 : 156,996) | |
| 증명 weight 비중 | 37.8% | 38.1% | |
| `end_unknown`(결과 라벨 중) | 85.6% | 84.9% | |
| mask: 결과 없음 · H2 중복 · 증명 충돌 | 335,263 · 58,590 · 28,474 | 478,045 · 86,065 · 41,622 | |
| 교차 분할 추가 mask | 0 | 0 | |

판독:
- **33,179판을 되살린 뒤에도 전체 데이터의 주요 분포가 v1과 거의 같게 유지됐다.** 모든 개수가 대국 비율(1.449) 근처에서 1.44–1.53배로 늘었다. 보고서는 되살린 대국만 따로 집계하지 않는다. 그래서 확인된 것은 전체 집합 수준의 큰 분포 변화가 없다는 것까지다. 되살린 부분집합의 분포가 기존과 같다고 직접 증명한 것은 아니다.
  - 부호 비율, 증명 비율, weight 비중은 소수점 첫째 자리까지 v1과 같다.
  - 증명 중 `immediate_win`과 `forced_loss`가 조금 더 늘었다(1.51–1.53배). 되살린 opening 계열의 대국이 약간 더 전술적이라는 정도이고 무시할 수준이다.
- **test가 가장 많이 늘었다(1.51배).** audit에서 v1 test 대국의 34.7%가 probe에 닿았던 것(train·val 31.4%)과 맞는다.
- **N = 4를 적용한 뒤의 최종 분할 분포가 H2 원본과 거의 같다.** 행 비율은 90.1 / 5.0 / 4.9%, H2 대국 분할은 89.9 / 5.0 / 5.1%다. 따라서 분할 왜곡 문제는 없다. 888판 자체가 분할별로 어떻게 빠졌는지는 보고서에 기록되지 않는다.
- **흑 차례 +1이 50.7%에서 50.1%로** 더 균형에 가까워졌다.
- `probe_position` mask가 0인 것은 정상이다. 행 단위로 처리되는 3돌 probe는 라벨 시작(5수째)보다 앞에 있다.
- `dropped.probe_game_by_ply`의 합은 888로 audit과 같다(467 + 82 + 39 + 300).
- **보고서 내부 합계가 모두 맞는다.**
  - 대국: 107,123 + 888 = 108,011
  - 행: train 3,152,862 + val 173,955 + test 172,390 = 3,499,207
  - 라벨 종류: 결과 2,725,718 + 증명 167,757 = 2,893,475
  - 라벨 부호: +1 1,477,042 + −1 1,416,433 = 2,893,475
  - mask된 행: 3,499,207 − 2,893,475 = 605,732 = 결과 없음 478,045 + H2 중복 86,065 + 증명 충돌 41,622
- `end_unknown`(2,314,848)은 mask가 아니다. 기권·시간·합의로 끝난 대국의 결과 라벨이고, weight 0.1로 학습에 들어간다.

**H6-1 gate: PASS (v2). H6-1 재생성은 더 필요 없다.**
- 라벨 정확성, 관점, 흑·백 균형, 누출, provenance가 모두 통과다. v1에서 실패했던 분포 항목도 해결됐다.
- **증명 −1 부족은 H6-3에서 관찰한다(MONITOR).**
  - 현황: 증명 +1 150,699 대 −1 17,058(10.2%), weight 기준 +1 64.3% 대 −1 35.7%.
  - 설계상 예상된 결과다. 싸게 얻는 증명은 대부분 둘 차례의 강제승이고, 확실한 −1을 얻으려면 모든 방어를 반박해야 한다.
  - 지금 `--prove-losses`로 v3를 다시 만들지 않는다. 먼저 v2로 value head를 학습한다. 그다음 H6-3에서 다음 지표를 따로 본다.
    - 증명 +1 / −1 부호 정확도
    - value > 0 / < 0 calibration
    - 흑·백별, ply 구간별 지표
    - P94(증명된 패배 국면) < 0
  - −1 쪽이 실제로 약하게 나오면 그때 `--prove-losses`를 명확한 실험 가설로 세운다.
- 다음 단계는 H6-2a다. value 캐시(v2) → smoke → 본 학습. 설정(`h6_value_head.yaml`)은 이미 v2 경로를 가리킨다.
  - 캐시 manifest가 분할별 라벨 국면 수와 증명·결과 weight 비중을 다시 계산한다. 이것을 본 학습 직전의 마지막 데이터 점검으로 쓴다.

## H6-2a value 캐시 v2 (2026-10-14)

- 소요 81.8초(12 worker).
- `labels_sha256` `1e8a37b5…`, `h6_report_sha256` `bd730267…`(v2 보고서). 보고서의 `labelled`와 개수가 일치해야 캐시가 써지므로, 대조 검사를 통과했다.
- 합계가 v2 보고서와 모두 맞는다.
  - 국면: 2,682,580 + 106,223 + 104,672 = 2,893,475
  - 버린 mask 행: 470,282 + 67,732 + 67,718 = 605,732
  - 분할별 행(국면 + mask): train 3,152,862, val 173,955, test 172,390
  - 증명: +1 150,699, −1 17,058
  - 결과: 2,725,718
- 캐시의 `games`(89,044 + 4,953 + 4,996 = 98,993)는 **라벨 있는 행이 하나 이상인 대국 수**다. 보고서의 107,123보다 8,130판 적다. 행이 전부 mask된 대국(합의 무승부 등)은 캐시에 들어가지 않기 때문이다.

| 분할 | 국면 | mask 비율 | 증명 weight 비중 | 증명 −1 비율 | 결과 +1 비율 |
|---|---|---|---|---|---|
| train | 2,682,580 | 14.9% | 37.1% | 10.0% | 48.7% |
| val | 106,223 | 38.9% | 50.0% | 12.3% | 48.1% |
| test | 104,672 | 39.3% | 47.5% | 10.6% | 48.3% |

- **val·test는 mask 비율이 높고 증명 비중이 크다.**
  - 원인은 H2의 `masked_plies`(86,065행)다. 앞 분할에 이미 나온 국면을 val·test에서 mask하는데, 이런 국면은 주로 흔한 초반 국면이고 대부분 결과 라벨 행이다.
  - 그래서 val·test는 상대적으로 후반 국면과 증명 행 쪽으로 치우쳐 있다. 누출 방지의 결과이며 정상이다.
- 영향:
  - val로 best를 고르는 weighted MSE는 증명에 약 50% 비중을 준다. 학습 목표의 37%보다 높다.
  - H6-3의 test 지표를 train 분포의 지표로 읽으면 안 된다. 그래서 H6-3은 kind별, ply 구간별 지표로 판정한다.
- 학습량: train 2,682,580국면. batch 1024 기준 epoch당 2,619 step, 3 epoch에 7,857 step이다.

## H6-2a smoke (2026-10-14, RTX 5070, 300 step, 커밋 `ac2876c`, `git_dirty false`)

| step | train loss | val wmse | 증명 부호 정확도 | 결과 MSE |
|---|---|---|---|---|
| 100 | 0.925 | 0.860 | 0.883 | 1.119 |
| 200 | 0.896 | 0.841 | 0.881 | 1.123 |
| 300 | 0.881 | 0.813 | 0.895 | 1.139 |

- **인프라는 정상이다.**
  - CUDA, 깨끗한 커밋, 캐시·보고서 해시 확인, 숫자 일치(train 2,682,580, 증명 weight 0.371, epoch당 2,619 step).
  - 속도는 안정 구간에서 약 19,000 samples/s다. 본 학습 7,857 step(약 805만 샘플)의 순수 학습 시간은 약 7분이다. 본 설정은 val 전체(106,223)를 평가하고(`eval_max_states: 0`) checkpoint도 저장하므로, 전체는 10분대 초중반까지 걸릴 수 있다.
- **학습 내용은 아직 상수 편향 단계다.** 300 step이고 warmup(500) 중이라 실패는 아니다. 다만 숫자를 그대로 읽으면 안 된다.
  - 증명 부호 정확도 0.895는 겉보기만 좋다. 이 표본에서 증명 target의 평균은 0.766, 즉 +1이 88.3%다. 항상 +를 출력해도 0.883이 나온다.
  - 결과 국면에서 value 평균이 +0.31인데 target 평균은 −0.04다. 결과 부호 정확도는 0.489이고, 결과 MSE 1.14는 모든 국면에 0을 출력하는 경우(1.000)보다 나쁘다. 게다가 step마다 조금씩 나빠졌다.
  - 해석: head는 먼저 weight를 건 target 평균(train 기준 약 +0.29, 증명 +1 편중 때문)을 상수로 배운다. 국면을 구별하는 학습은 그 뒤에 와야 한다.
- 조치: 평가에 **target 부호별 지표**(`by_kind_sign`: proof+1, proof−1, outcome+1, outcome−1)를 추가했다. 로그에도 네 값을 출력한다. 이 지표가 없으면 상수 편향과 실제 판별을 구분할 수 없다.
- **본 학습을 읽는 기준**(진단용, H6-3 gate 아님. checkpoint 선택은 계속 val weighted MSE로만 한다).
  - 보는 순서:
    1. val weighted MSE가 계속 줄어드는가
    2. 결과 MSE < 1.000(모든 국면에 0을 출력하는 경우보다 좋음). 크기까지 맞는지 보는 1차 기준이다
    3. 증명 ±1 부호 정확도
    4. 결과 ±1 부호 정확도
    5. value 평균이 상수 +0.29에서 벗어났는가, 포화율
  - 부호 판별은 **두 부호 정확도의 평균**(balanced accuracy)이 0.5를 넘는지로 본다. 증명 ±1과 결과 ±1에 각각 적용한다.
    - 두 부호 모두 0.5를 넘는 것은 편향까지 맞춰졌다는 뜻이라 더 강한 조건이다. 판별은 배웠지만 아직 양수 쪽으로 치우친 모델은 예를 들어 +1 0.75 / −1 0.40으로 나온다.
    - 결과 라벨은 기권·시간·합의가 많은 약한 지도라서, value가 잘 학습돼도 결과 부호 정확도가 아주 높아지지는 않는다.
  - 패턴별 해석:
    - 상수 편향이 남은 경우: +1 정확도 ≈ 1, −1 정확도 ≈ 0, 결과 MSE > 1.
    - 실제로 판별하는 경우: 두 부호의 정확도가 함께 오르고, 결과 MSE < 1.
    - 증명 ±1은 둘 다 좋은데 결과는 0.5 근처이고 결과 MSE ≈ 1인 경우: 고정된 H3 trunk가 전술 증명은 구별하지만 장기 승패 정보는 충분히 담지 못한다는 뜻일 가능성이 크다. 이것이 H6-2b(trunk 미세조정)를 검토할 가장 분명한 근거다.
  - 기준을 못 넘으면 원인을 따로 본다. 후보는 고정된 trunk 표현의 한계, 증명 weight 1.0의 편향, 결과 라벨의 잡음이다. 그다음 증명 weight 조정이나 H6-2b를 검토한다.
- `ff4d93b`는 평가 지표만 바꿨다. loss, optimizer, sampling, weight, D4, LR, checkpoint 선택 기준은 그대로다. 그래서 smoke를 다시 돌릴 필요는 없다.

## H6-2a 본 학습 결과 (2026-10-14, `docs/h6/h6_2a_best.json`, `h6_2a_metrics.jsonl`, `h6_2a_cache_manifest.json`)

provenance:
- 학습 코드 `ff4d93b`, `git_dirty false`, RTX 5070, 3 epoch 7,857 step.
- 부모 H3 checkpoint SHA `efee4832…`(H3 step 31,030, `value_trained false`).
- 라벨 SHA `1e8a37b5…`, v2 보고서 SHA `bd730267…`, 캐시 분할 개수가 manifest와 일치한다.
- test는 로드하지 않았다.
- 결과 커밋 `7502368`은 `f1cca21`(문서) 위에 rebase됐다. 학습 코드와는 무관하다.
- 파일 이름에 `h6_2a_` 접두사를 붙였다.

| step | val wmse | 증명 +1 | 증명 −1 | 결과 +1 | 결과 −1 | 결과 MSE | 결과 −1 평균 v |
|---|---|---|---|---|---|---|---|
| 2,000 | 0.5973 | 0.949 | 0.915 | 0.655 | 0.551 | 0.984 | +0.050 |
| 4,000 | 0.5727 | 0.963 | 0.919 | 0.765 | 0.467 | 0.974 | +0.074 |
| 6,000 | 0.5648 | 0.952 | 0.933 | 0.677 | 0.574 | 0.945 | −0.003 |
| **7,857** | **0.5620** | **0.951** | **0.933** | **0.698** | **0.560** | **0.939** | −0.001 |

(표의 증명 ±1, 결과 ±1 값은 부호 정확도다.)

- **진단 기준을 모두 통과했다.**
  - val wmse가 한 번도 오르지 않았다.
  - 결과 MSE 0.939로, 모든 국면에 0을 출력하는 경우(1.000)보다 낮다.
  - 두 부호 정확도의 평균(balanced accuracy)은 증명 0.942, 결과 0.629다.
  - 네 부호 정확도가 모두 0.5를 넘는다.
  - smoke에서 보인 상수 편향 상태는 벗어났다. 4,000 step에서 잠시 양수 쪽으로 기울었다가(결과 −1 0.467) 회복했다.
- **고정된 H3 trunk에도 승패 정보가 있다.** 결과 판별이 0.5보다 확실히 높으므로, "trunk가 장기 승패를 표현하지 못한다"는 H6-2b 근거는 지금 없다.
- 마지막 step이 best인 것은 학습률을 cosine으로 최솟값까지 줄인 schedule에서 자연스럽다. step 사이 감소 폭이 0.0245 → 0.0079 → 0.0028로 줄었으므로 수렴에 가깝다고 본다. 수렴을 증명한 것은 아니다.
- 흑·백 부호 정확도는 0.651 · 0.660, ply 구간별 부호 정확도는 0.60(5–9수)에서 0.68(50수 이상)으로 오른다. 모든 구간의 MSE가 1보다 낮고, |v| > 0.95 포화율은 3.4%다(증명 21.8%, 결과 1.6%).

**MONITOR (H6-3에서 판정할 것)**
1. **양수 편향이 남아 있다.** 결과 국면의 value 평균은 +0.124인데 target 평균은 −0.038이다(편향 +0.16). 결과 −1 국면의 평균 value는 −0.001로, 패배 국면을 거의 0으로 본다. 원인은 증명 +1에 큰 weight가 실린 것이다.
   - 같은 깊이의 형제 노드에는 같은 편향이 더해지므로 MCTS의 순서에는 영향이 작다.
   - 그러나 증명된 ±1 terminal, 다른 깊이의 leaf, rollout과 섞일 때 왜곡이 생긴다.
   - 결과 calibration(ECE)에도 직접 나타난다. ECE는 §12.19의 λ 규칙(ECE ≤ 0.10이면 λ = 1.0, 아니면 0.5)에 쓰인다.
2. **흑 차례 쪽 편향이 더 크다.** 평균 value와 평균 target의 차이가 흑 차례 +0.20(0.194 대 −0.009), 백 차례 +0.06(0.134 대 0.078)이다. 부호 정확도는 흑·백이 비슷하지만 calibration은 다르다. **P94는 흑 차례의 패배 국면(< 0이어야 함)이라 이 편향의 영향을 직접 받는다.**
3. **ply가 늘수록 편향이 커진다.** 평균 value가 5–9수 0.02에서 50수 이상 0.27로 오르는데, 평균 target은 0.01–0.04다. 후반일수록 증명(+1 위주) 비중이 커지는 영향이다. 표본에 증명과 결과가 섞여 있어서 순수한 결과 편향과 분리되지는 않는다.
4. **증명 −1 정확도 0.933이 깊은 패배까지 맞힌다는 뜻은 아니다.** val의 증명 −1은 모두 `forced_loss`(깊이 0, 상대의 막을 수 없는 위협) 패턴이다. P94(모든 수가 VCT1 패배)는 더 깊은 패배다. 그래서 `--prove-losses`/H6-2b 결정은 H6-3의 P94 결과를 보고 내린다.

**H6-2a 판정: PASS** (진단 기준 기준). 추가 학습, `--prove-losses`, H6-2b는 지금 하지 않는다. 다음은 H6-3 오프라인 gate다.

## H6-3A 결과 (2026-10-15, `docs/h6/h6_3a_gate.json`): **FAIL (G3, G4)**

실행 조건:
- 커밋 `78af28c`, `git_dirty false`, checkpoint step 7,857
- part A: 2,795판, 53,531국면. part B(2,637판)는 열지 않았다.

| gate | 결과 | 판정 |
|---|---|---|
| G1 증명 | balanced accuracy 0.945, +1 0.953, −1 0.938(469국면), MSE 0.179 | PASS |
| G2 결과 | balanced accuracy 0.640(+1 0.705, −1 0.576), MSE 0.918 | PASS |
| G3 D4 | 표준편차 평균 **0.124**, 95번째 백분위 **0.276**, 최대 0.631, 확신 국면(1,723개) 부호 일치 **0.835**, 전체 0.724 | **FAIL** |
| G4 probe | P93(백 차례, +여야 함): 원래 −0.361, 평균 −0.390(범위 −0.97 ~ +0.65, 표준편차 0.50). P94(흑 차례, −여야 함): 원래 **+0.980**, 평균 +0.894(8개 모두 +0.69 이상) | **FAIL** |
| G5 흑·백 | 0.639 · 0.643, 차이 0.004 | PASS |

보고 항목:
- 결과 ECE 0.076 → 규칙상 λ 1.0. 다만 전체 판정이 FAIL이므로 기록만 한다. 수정 모델은 B에서 ECE와 λ를 다시 계산한다.
- 편향(평균 v − 평균 t):
  - 결과 +0.153(흑 차례 +0.195, 백 차례 +0.112)
  - 결과의 ply 구간별 +0.016 → +0.282
  - val에서 본 편향이 test에서도 그대로 나타났다.
- 편향을 뺀 결과 balanced accuracy는 0.634로 원래(0.640)보다 낮다. 전역 상수 보정으로는 해결되지 않는다.
- P92: 실제 진 수 (3,4)가 133수 중 6위다. 1위와 value 차이는 작다(0.409 대 0.360).

**검토로 확인하거나 보탠 점**
1. **D4 변환 코드의 버그는 원인이 아니다.**
   - `test_h6_value`가 이미 이를 검증한다. 증강한 plane과, 변환한 대국을 `encode_game`으로 인코딩한 결과가 8개 대칭 모두에서 같다.
   - P92·P93·P94 원 국면으로도 8개 대칭을 다시 확인했고 모두 일치했다.
   - 따라서 G3는 **모델의 방향 민감성**이다. value head가 `Flatten → Linear(225→64)`라서 칸마다 다른 weight를 쓰고, 구조상 D4 불변이 보장되지 않는다. 학습 때는 샘플마다 무작위 대칭 하나만 보여 줬고 3 epoch뿐이었다.
2. **P93과 P94는 같은 방향으로 틀렸다. 두 국면 모두 "흑이 좋다"고 본다.**
   - P93은 백 차례 −0.36(흑이 좋다는 뜻), P94는 흑 차례 +0.98이다. 이 대국에서 이긴 쪽은 백이다.
   - 흑 차례 결과 편향(+0.195)과 같은 방향이다.
3. **P93·P94는 깊은 전술일 뿐 아니라 RenjuNet 분포 밖의 국면이다.**
   - 이 국면은 웹 대국(사람 백 대 MCTS-v8)에서 나왔다. 94수 시점에 흑은 왼쪽 위에 큰 덩어리를 이루고, 백 돌 30여 개는 판 전체에 격자처럼 흩어져 있다. 프로 기보에는 거의 없는 형태다.
   - 깊이 0 증명 탐색으로는 P93·P94 모두 라벨이 나오지 않는다(`--prove-losses`를 켜도 없음). 두 국면의 승패는 VCF보다 깊다(P93은 VCT2 승리, P94는 모든 수가 VCT1 패배).
   - 그래서 G4 실패에는 두 원인이 섞여 있다. 하나는 **깊이**(학습 증명은 깊이 0뿐), 다른 하나는 **분포 밖 국면**이다. 흑의 큰 연결 덩어리를 우세로 읽는 휴리스틱으로 보인다.
   - 이 국면은 엔진이 실제 대국에서 진 국면이다. 엔진이 마주치는 국면에서 value가 이렇게 틀린다는 뜻이므로 무시할 수 없다.
4. **RenjuNet 위에서 더 깊은 증명 라벨만 늘리는 것으로는 부족할 수 있다.** RenjuNet에는 P93·P94 같은 국면이 거의 없다. 엔진 대국 국면에서 나온 깊은 라벨이 필요할 가능성이 크다. 엔진 대국 데이터는 H7(자기대국·hybrid 데이터)의 역할과도 겹친다.
5. **D4 평균 추론(8개 대칭 평균을 value로 쓰기)은 구조상 G3를 해결한다.** 다만 G3 정의를 결과를 본 뒤에 바꾸는 것은 허용되지 않는다. 이것은 "모델 변경"이므로 B에서 검증해야 한다. 또 P94는 8개 평균도 +0.89라서 G4는 해결되지 않는다.

**다음 단계 (dev만 사용, B와 Final 세트는 열지 않음): H6-3D 진단** — 원인을 숫자로 나눈 뒤에 수정 방향을 정한다.
- **D1 D4 분해(A):** 종류·ply·흑백·|v| 구간별 D4 표준편차. 모든 대칭에 고르게 흔들리는지, 특정 반사·회전에서만 흔들리는지 본다. H3 policy 자체의 D4 일관성(대칭 간 top-1 일치율)도 함께 재서 trunk 쪽 원인인지 가린다.
- **D2 엔진 국면 깊은 패배 세트:** E1·E1-S manifest의 10M truth를 쓴다. `PROVEN_LOSS` 157수(e1 74, e1s 83)의 수 뒤 국면은 "상대 차례에서 상대 승리"(value > 0)여야 한다. 부호 정확도와 평균 value를 잰다.
  - 이 국면들은 probe라서 학습에서 빠졌다. 그래서 dev 진단에는 쓸 수 있다.
  - `VCT2_CLEAR`는 value 라벨이 아니므로 쓰지 않는다.
  - 깊은 패배가 P94 한 국면만의 문제인지, 엔진 국면 전반의 문제인지 가린다.
- **D3 D4 평균 추론의 효과(A, D2):** G1·G2·G5 지표와 D2 정확도가 어떻게 바뀌는지 본다.
- 진단 결과에 따른 갈림:
  - D2가 나쁘면 엔진 국면 데이터가 필요하다. 깊은 증명 라벨은 TB-Engine-v1 자기대국에서 새 seed로 만든다(예약 seed 제외).
  - D2가 괜찮고 D4만 문제면 학습 방식을 고친다. 후보는 8개 대칭 일관성 학습, D4 평균 추론, pooling head(구조 변경)이다.
  - H6-2b(trunk 미세조정)는 G2가 통과했으므로 아직 근거가 없다. D1에서 trunk 쪽 원인이 보이면 다시 검토한다.
- 바뀐 모델의 최종 gate는 **B와 새 깊은 승리·패배 probe**로 한다. A와 P93·P94는 이제 dev다.

## H6-2a′ 계획 (확정 2026-10-15): 부모 단위 패배 라벨 하나만 바꾸는 재학습

목표: G4(부모 단위 깊은 패배)를 고친다. **바꾸는 변수는 라벨 하나**(v2 → v3, `vcf_loss` 추가)다. 추론은 D4 평균으로 고정한다(G3 해결, 학습은 바꾸지 않음). D4 일관성 학습은 이번 실험에 넣지 않는다. H6-4에서 D4 평균이 실제로 비쌀 때 별도 arm으로 한다.

**순서 (앞 단계가 끝나야 다음으로 간다)**
1. **fresh-final probe의 seed와 규칙을 동결한다(이 커밋).**
   - seed 8601–8610을 `RESERVED_SEEDS`에 미리 추가했다(`FRESH_PROBE_SEEDS`). 이 seed의 대국은 H6, H7, 자기대국 어떤 학습 데이터에도 넣지 않는다.
2. **fresh probe를 생성하고 봉인한다.**
   - TB-Engine-v1(`puct_policy_vct2_50k`, c 1.5)과 `v8:full`의 대국을 seed 8601부터 돌린다. 그 대국에서 `scripts/make_h6_fresh_probes.py`로 probe를 뽑는다. 모델은 쓰지 않고 solver truth만 쓴다.
   - 증명이 없는 깊이 0 패턴(즉시 승리, 막을 수 없는 4, VCF, `forced_loss`, `vcf_loss`)은 제외한다.
   - +1: 상대의 즉시 위협이 없고, 둘 차례에게 증명된 깊이 1 VCT 공격(`vct1_win`)이 있는 국면.
   - −1: `defense_label`이 `vct_loss`를 증명한 국면. VCF를 버티는 모든 수가 깊이 1에서 진다는 뜻이며, P94와 같은 종류다.
   - **한 대국에서 probe는 최대 1개**다. 대국마다 가장 이른 후보를 고르고, 대국 순서대로 개수가 적은 부호에 배정한다(동률이면 −1).
   - 기존 probe·평가 국면과 D4로 같은 국면은 제외한다.
   - **멈추는 규칙(truth 개수만 본다):** seed를 차례로 추가해 +1과 −1이 **각각 서로 다른 대국 10개 이상**이 되면 멈춘다. 목표는 각 20대국이다. 8610까지 가도 10개에 못 미치면 멈추고 다시 설계한다.
   - **8601 결과(2026-10-15, truth 개수만):**
     - 20판 중 승리 후보가 있는 대국 17판, 패배 후보가 있는 대국 16판.
     - 한 대국에 1개 규칙을 적용해 뽑은 probe는 +1 8개, −1 9개(17대국). 나머지 3판은 후보가 없거나 기존 probe와 D4로 겹쳤다.
     - 깊은 국면은 대부분의 엔진 대국에 있다. 개수를 막는 것은 후보 부족이 아니라 "대국당 1개" 규칙이다. 따라서 대국 1판당 probe는 약 0.85개다.
   - **추가 결정(모델을 보기 전, 개수만 보고 정함):**
     - 부호당 10개면 오답 하나가 정확도 10%p다. 그래서 목표치(부호별 약 20대국)에 맞춰 **seed 8602와 8603을 모두 돌리고 거기서 멈춘다**(최대 60판, 예상 probe는 부호별 약 25개).
     - 그래도 어느 부호든 10개에 못 미치면 8604부터 하나씩 추가한다.
   - **8601–8603 결과(2026-10-16, 60판, truth 개수만): +1 28개, −1 28개(서로 다른 56대국). 기준(부호별 10대국 이상)과 목표(약 20)를 모두 넘었으므로 여기서 멈춘다.**
     - 후보가 있는 대국: 승리 52판, 패배 45판. probe가 없는 대국 4판.
     - 상한에 걸려 건너뛴 국면 37개(17판). 대부분 길고 위협이 많은 후반 국면이다. 다른 국면에서 후보가 나와 probe 수에는 거의 영향이 없었다.
     - 탐색 버전: v2 52판, v3 8판. 처음 v3로 시도했다가 멈춘 판과 마지막 1판이 v3다. 두 버전은 노드 300,000을 넘는 국면에서만 결과가 다를 수 있고, 두 버전 모두 모델 없이 solver 판정만 쓴다.
     - 최종 G4의 판정 범위: 부호당 28개이므로 정확도 0.80 기준이면 각 부호에서 오답 5개까지 허용된다(6개부터 실패).
   - 파일 `docs/h6/h6_fresh_final_probes_v1.json`을 커밋한다. 이 파일은 기본 probe 파일 목록에 들어가므로, 이후 라벨 생성에서 이 국면을 지나는 RenjuNet 대국이 빠진다.
   - **최종 gate 전에는 이 파일로 어떤 모델도 평가하지 않는다.** 개발 중에는 A, D2, P93·P94만 본다.
3. **`--prove-losses-last-n`의 N을 train 표본 pilot으로 정한다.**
   - `--splits train --sample-every 50 --prove-losses`(모든 ply). 결과에 따라 고르지 않는 해시 표본, 약 1,900판이다.
   - 보고서의 `proof_by_distance_to_end.vcf_loss`에서 **`vcf_loss`의 95% 이상을 포함하는 가장 작은 N**을 고른다.
   - **결과: N = 12** (2026-10-17, 아래 "3단계 pilot 결과").
   - N을 정할 때 val·test와 모델은 보지 않는다.
4. **v3 라벨과 캐시를 만든다.** v2 설정에 `--prove-losses --prove-losses-last-n N`만 더한다(N = 4 probe 정책 그대로).
5. **H6-2a′를 학습한다.**
   - 같은 H3 checkpoint에서 value head를 처음부터 다시 학습한다. H6-2a에서 이어 학습하지 않는다.
   - seed, optimizer, epoch, 학습률 schedule은 H6-2a와 같다. 바뀌는 것은 캐시와 out_dir뿐이다.
6. **dev 비교:** H6-2a + D4 평균 대 H6-2a′ + D4 평균을 A, D2, P93·P94에서 비교한다(`h6_diagnose.py`).
7. **최종 gate(한 번):** 후보 하나로 **part B + fresh probe + D4 평균 추론**을 실행한다.
   - G1·G2·G3·G5는 H6-3 기준 그대로다. G3는 D4 평균이라 구조상 0에 가깝다.
   - **G4는 fresh probe로 판정한다.** +1과 −1 각각 서로 다른 대국 10개 이상, 각 부호 정확도 ≥ 0.80. 실행 스크립트(`h6_offline_gate.py`)는 part B를 `--fresh-probes`와 `--d4-average` 없이는 실행하지 않는다.
   - ECE와 λ는 B에서 다시 계산한다. A의 λ = 1.0을 이어받지 않는다.
   - P93·P94는 dev이므로 최종 판정에서 뺀다.

**실행 명령 (2단계: fresh probe 대국과 추출)**
```powershell
cd "C:\오목 강화학습\renju-h6"
..\renju-alphazero\.venv-cuda\Scripts\Activate.ps1
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = "$PWD\src"
git status --short
git pull --ff-only
New-Item -ItemType Directory -Force runs\h6_fresh | Out-Null
python -u scripts\run_mcts_v8_benchmark.py --arm puct_policy_vct2_50k --puct-c 1.5 --opponent v8:full --seed 8601 --pairs 10 `
  --workers 6 --policy-checkpoint ..\renju-alphazero\runs\h3_policy_64x4\best.pt `
  --games-jsonl runs\h6_fresh\fresh_8601.jsonl --output runs\h6_fresh\fresh_8601.json 2>&1 | Tee-Object runs\h6_fresh\fresh_8601.log
python -u scripts\make_h6_fresh_probes.py --games runs\h6_fresh\fresh_8601.json --workers 12 `
  --jsonl runs\h6_fresh\scan.jsonl --output runs\h6_fresh\count_8601.json
```
- `--jsonl`에 대국별 탐색 결과가 끝나는 대로 저장된다. 중단돼도 같은 명령을 다시 실행하면 끝난 대국은 건너뛴다(2026-10-15 추가).
  - 첫 실행은 저장 기능이 없어서, 20판 중 19판을 끝낸 뒤 중단되어 결과를 모두 잃었다.
  - 같은 날 탐색도 빠르게 했다. VCF를 버티는 수가 8개를 넘으면 그 자리에서 멈추고, `vcf_loss` 확인과 `vct_loss` 확인이 이 한 번의 탐색을 같이 쓴다. 판정은 이전과 같고 약 3배 빠르다.
- **국면당 VCF 호출 상한 5,000(2026-10-15, scan v2):**
  - 상황: seed 8601–8603 추출에서 60판 중 58판은 78분 안에 끝났지만, 마지막 2판이 2시간 넘게 끝나지 않았다. CPU는 계속 쓰고 있었으므로 멈춘 것이 아니라 계산이 폭증한 것이다.
  - 원인: 깊이 1 분류(`classify`, vct_depth = 1)에는 국면 전체의 상한이 없다. 위협수가 많은 후반 국면에서는 후보 × 응수 × VCF 검색이 끝없이 늘어난다.
  - 조치: 한 국면의 모든 검사가 공유하는 **캐시되지 않은 VCF 호출 상한 5,000회**를 둔다. P94는 약 550회가 든다. 상한을 넘은 국면은 probe 후보에서 빠진다.
  - 실행 시간이 아니라 호출 횟수로 정하므로, 어느 컴퓨터에서 돌려도 결과가 같다. 상한을 넘은 국면 수는 보고서에 남긴다.
  - v1(상한 없음) 탐색 결과는 다시 쓰지 않는다. 이어서 실행할 때 `scan_version`이 다른 행은 버리고 다시 탐색한다. 8601의 개수(8 / 9)도 v2로 다시 센다.
- **국면당 VCF 노드 상한 300,000 추가(2026-10-16, scan v3):**
  - 상황: v2에서도 마지막 1판이 100분 넘게 끝나지 않았다.
  - 원인: 호출 횟수 상한만으로는 시간이 묶이지 않는다. 호출 하나가 최대 20,000노드를 쓰는데, 이 solver는 초당 1–2천 노드 정도만 처리한다(P94 측정: 552회 호출, 19,799노드, 10.5초). 그래서 호출 5,000회도 최악이면 수 시간이다.
  - 조치: 노드 상한(300,000, P94의 15배)을 더했다. 호출 상한과 노드 상한 중 하나라도 넘으면 그 국면은 건너뛴다. 이것으로 국면 하나가 최악의 경우에도 수 분 안에 끝난다.
  - 60초 넘게 걸린 국면은 `[slow position] 대국 ply 시간 -> 결과` 줄로 출력해, 상한이 실제로 작동하는지 보이게 했다.
  - **v2로 끝난 59판은 그대로 쓰고, 남은 1판만 v3로 탐색한다.** v3와 v2는 노드 300,000을 넘는 국면에서만 결과가 다를 수 있다. 탐색은 모델 없이 solver 판정만 쓰므로 버전이 섞여도 평가 독립성은 그대로다. 어느 대국을 어느 버전으로 탐색했는지는 출력의 `games_by_scan_version`에 남긴다.
- `count_8601.json`은 개수 확인용이다(`counts`).
  - +1과 −1이 각각 10대국에 못 미치면 seed 8602, 8603… 대국을 추가한다. 그다음 **모든 seed 파일을 함께** 넣어 다시 추출한다.
  - 기준을 채우면 최종 추출 결과를 `docs/h6/h6_fresh_final_probes_v1.json`으로 출력해 커밋한다.

**3단계 pilot 첫 실행 중단과 수정 (2026-10-16)**
- 첫 실행: 3시간 반 동안 묶음이 하나도 끝나지 않았다.
- 원인: `label_position(prove_losses=True)`가 국면마다 **모든 합법수**를 끝까지 검사했고, 국면 전체에 대한 상한이 없었다.
- 수정(`6321d4d` 이후):
  - 버티는 수가 하나 나오면 바로 멈춘다. 판정은 이전과 같다(합성 761행, P92–P94 대국 전 국면에서 일치).
  - 국면당 상한: VCF 호출 5,000회, 노드 100,000. 이 solver는 초당 1–2천 노드를 처리하므로, 국면 하나가 최악이어도 약 1분이다. 상한에 걸린 국면은 라벨 없이 건너뛰고 `prove_losses_budget_cuts`로 보고한다. **v3 라벨도 같은 상한으로 만든다.**
  - 진행률 묶음은 4판 단위로 줄였다.

**실행 명령 (3단계: pilot, CPU)**
```powershell
..\renju-alphazero\.venv-cpu\Scripts\Activate.ps1
python -u scripts\build_h6_labels.py --games $GAMES --workers 12 --splits train --sample-every 50 --prove-losses `
  --probe-game-min-ply 4 --output data\tactical\h6\pilot_prove_losses.jsonl.gz --report runs\h6\h6_pilot_prove_losses.json 2>&1 | Tee-Object runs\h6\h6_pilot.log
```
- 보고서는 개수만 담는다. `docs/h6/`에 복사해 커밋한다. 이것을 보고 N을 정한다.

**solver 가속: `--fast off|fast2|fast3` (2026-10-16)**

- 방향: native 확대 순서는 `web-ser5.md` §9.9를 따른다. 이미 검증된 정확 가속(fast2, X1-A)을 먼저 재사용한다.
  그래도 부족하면 X1-B(금수 판정 native, C)로 가고, VCF 전체 C++17은 마지막이다. C++ Game/PUCT, Windows MSVC 빌드는 지금 하지 않는다.
- `feat/Web_SER5`에서 정확 가속만 가져왔다(`src/rt30/fast*.py`, `src/rt30/native/`, 파일 내용 그대로). deadline, X2 helper, engine_v3, 웹 코드는 가져오지 않았다.
- `hybrid.h6_fast`가 모드를 설치한다. frozen 파일은 고치지 않고, 이 프로세스의 이름만 바꿔 끼운다.
  - `off`: frozen Python solver. 기본값이고, 지금 pilot과 같다.
  - `fast2`: `rt30.fast`(numpy window scan + 4 연장 규칙). 컴파일러가 필요 없다.
  - `fast3`: fast2 + C window scan. **엄격하다.** native가 빌드되지 않았거나 낡았으면(`build.stale()`) 시작 전에 오류로 멈춘다. fast2로 조용히 내려가지 않는다.
- worker마다 설치한다(`ProcessPoolExecutor(initializer=h6_fast.install)`). Windows spawn과 Python 3.14 Linux forkserver worker는 부모의 바꿔 끼움을 물려받지 않기 때문이다.
  묶음마다 worker가 실제로 쓴 함수를 함께 돌려보낸다. 하나라도 요청한 모드와 다르면 builder가 실패한다.
- 보고서에 `acceleration`을 기록한다: 요청 모드, 실제 함수 이름, fast3면 native 빌드 기록(Python, SOABI, 컴파일러, 플래그, 소스·모듈 SHA256).
  라벨 의미와 row 형식은 바뀌지 않으므로 라벨 버전도 바꾸지 않는다.
- 동일성 테스트(`tests/test_h6_fast.py`):
  - `_find_vcf_with_stats`의 결과·노드 수·exhausted 비교. 노드 상한 50과 3,000에서 잰다.
  - `vcf_loss_status`의 loss/budget/None 비교. 국면 예산 300과 100k에서 잰다.
  - builder 전체(작은 예산으로 budget cut이 실제로 나는 설정 포함)의 row와 summary 비교. summary에는 `prove_losses_budget_cuts`와 `proof_by_distance_to_end`가 들어 있다.
  - pool worker와 spawn worker 설치, 다른 모드로 돈 worker의 거부.
  - 이 모든 비교에서 세 모드가 같다.
- 컨테이너 실측(E3-A 엔진 대국 12판, 마지막 8수 prove-losses, 4 workers, `scripts/check_h6_fast.py`):

| 모드 | 시간 | 국면/초 | off 대비 | 결과 |
|---|---|---|---|---|
| off | 75.0초 | 3.7 | 1.0배 | 기준 |
| fast2 | 17.4초 | 15.8 | 4.3배 | 같음 |
| fast3 | 8.8초 | 31.4 | 8.6배 | 같음 |

- 엔진 대국은 RenjuNet보다 전술 국면이 많다. 그래서 위 배수는 v3 시간 추정에 그대로 쓰지 않는다. 실제 RenjuNet 표본으로 다시 잰다(아래).

**실측 명령 (pilot이 끝난 뒤. 같은 기계에서 pilot과 동시에 돌리지 않는다)**
```powershell
# Desktop (Windows): off, fast2
python -u scripts\check_h6_fast.py --games $GAMES --modes off fast2 --sample-every 500 --workers 12 `
  --report runs\h6\h6_fast_check_desktop.json 2>&1 | Tee-Object runs\h6\h6_fast_check_desktop.log
```
```bash
# SER5 (Linux): native를 먼저 빌드한다 (.venv-ser5 Python의 헤더 필요, web-ser5.md §9.9)
cd src && ../.venv-ser5/bin/python -m rt30.native.build --if-stale && cd ..
.venv-ser5/bin/python -u scripts/check_h6_fast.py --games $GAMES --modes off fast2 fast3 --sample-every 500 \
  --workers 8 --report runs/h6/h6_fast_check_ser5.json 2>&1 | tee runs/h6/h6_fast_check_ser5.log
```
- 판정:
  - `same: true`가 아니면 그 모드는 쓰지 않는다(종료 코드 1).
  - 기계 비교는 배수가 아니라 같은 표본의 **절대 국면/초**로 한다.
  - pilot의 N이 정해지면 `--prove-losses-last-n N`을 붙여 다시 잰다. 그 국면/초로 v3 전체 시간을 다시 계산한다.
- v3 생성: 위에서 고른 기계·모드로 기존 명령에 `--fast fast2` 또는 `--fast fast3`만 더한다.

**3단계 pilot 결과 (2026-10-17, `docs/h6/h6_pilot_prove_losses.json`, 실행 코드 `fa4de82`, 결과 커밋 `ed5a2a8`): N = 12**

- 표본 구성:
  - 입력 108,011판 중 train·표본·probe 제외를 거쳐 1,961판, 64,353국면을 검사했다.
  - `prove_losses_last_n = 0`이므로 표본 대국의 대상 ply를 모두 검사했다.
  - 손실 증명 예산은 VCF 호출당 20,000 노드, 국면당 100,000 노드, 5,000 호출이다.
- 라벨:
  - 64,353행 중 54,757행이 value label을 가진다. 부호는 +1 27,897 · −1 26,860이다.
  - proof는 `vcf_loss` 587, `vcf` 1,835, `unstoppable_four` 512, `immediate_win` 409, `forced_loss` 321개다.
  - budget cut은 988국면이다(전체 국면의 1.54%).
- `vcf_loss`의 종료까지 거리(`scripts/h6_last_n.py`로 계산):

| N | 2 | 4 | 6 | 8 | 10 | **12** | 14 | 16 |
|---|---|---|---|---|---|---|---|---|
| N 이내 | 150 | 298 | 446 | 516 | 547 | **560** | 574 | 580 |
| 비율 | 25.6% | 50.8% | 76.0% | 87.9% | 93.2% | **95.4%** | 97.8% | 98.8% |

- **사전 규칙("관측 `vcf_loss`의 95% 이상을 포함하는 가장 작은 N")에 따라 N = 12로 정한다.**
  `vcf_loss`는 종국에 강하게 몰려 있다. 따라서 모든 ply에 prove-losses를 돌릴 필요가 없다는 가설이 지지된다.
- **budget cut은 N 결정을 바꾸지 않는다.**
  - 규칙이 세는 것은 "100k 예산의 builder가 만드는 `vcf_loss` 라벨" 가운데 마지막 N수가 남기는 몫이다.
  - v3도 같은 예산을 쓴다. cut 국면은 모든 ply를 돌려도, 마지막 N수만 돌려도 라벨이 없다. 그래서 N 밖의 cut이 사실은 패배였더라도, last-N이 v3에서 잃는 라벨 비율은 그대로다.
  - cut이 답하는 질문은 따로 있다. **"예산 때문에 깊고 먼 패배가 학습 데이터에서 체계적으로 빠지는가"**이다. 이는 라벨 범위(coverage) 문제이고 N과는 무관하다.
  - 더 높은 예산(300k, 1M)의 tail 감사로 N을 다시 고르면, v3 builder가 만들지도 않을 라벨에 N을 맞추는 셈이다. 그래서 감사는 N의 gate가 아니라 H6-3에서 −1 쪽을 해석할 때 쓰는 진단으로 둔다.
- **N = 12의 진짜 불확실성은 표본 크기다.**
  - 587개에서 95.4%다. 이항 표준오차가 약 0.9%p이고, 대국 단위로 묶여 있어 실제로는 더 크다. 그래서 여유 0.4%p는 표본 잡음 안에 있다.
  - N = 14(97.8%)로 올리면 대국당 prove-losses 국면이 2개 늘어난다. 얻는 라벨은 `vcf_loss`의 약 2.4%다.
  - 규칙은 결과를 보기 전에 정했으므로 그대로 12를 쓴다. 이 단서만 기록한다.
- 참고: 이 보고서에는 cut의 종료 거리가 없었다. 지금 builder는 다음을 남긴다.
  - 보고서: `prove_losses_budget_cuts_by_distance_to_end`
  - row 파일 옆: `<output>.budget_cuts.jsonl` manifest(game_id, ply, distance_to_end). RenjuNet 파생이므로 `data/`·`runs/`에만 둔다.
  - 각 증명 row의 `solver_budget`: `prove_losses` 예산과 `prove_losses_last_n`. 라벨 의미는 바뀌지 않는다.

**다음 순서**

1. 실제 RenjuNet 표본 속도 측정(위 `check_h6_fast.py`, Desktop off·fast2 / SER5 off·fast2·fast3). 실행 기계와 모드를 고른다.
2. **같은 pilot을 fast로 재실행해 대조한다.** 실패한 실행의 재실행이 아니라, 64k 국면 규모의 정확성 검사다. cut 거리도 함께 얻는다.
   `--expect`가 공통 summary 키(587 / 988 / 거리 분포 포함)와 설정이 모두 같다고(SAME) 해야 v3로 간다.

```powershell
python -u scripts\build_h6_labels.py --games $GAMES --workers 12 --splits train --sample-every 50 --prove-losses `
  --probe-game-min-ply 4 --fast fast2 --output data\tactical\h6\pilot_prove_losses_fast.jsonl.gz `
  --report runs\h6\h6_pilot_prove_losses_fast.json 2>&1 | Tee-Object runs\h6\h6_pilot_fast.log
python scripts\h6_last_n.py runs\h6\h6_pilot_prove_losses_fast.json --expect docs\h6\h6_pilot_prove_losses.json
```

   - 출력에는 N별 `cuts>N`과 최악의 경우 비율(N 밖 cut이 모두 패배일 때)도 나온다. 이 값은 기록만 한다(위 이유).
3. N = 12로 last-N 속도를 잰다: `check_h6_fast.py --prove-losses-last-n 12 --modes <고른 모드>`. 이 값으로 v3 전체 시간을 계산한다.
4. v3 라벨과 캐시를 만든다. v2 설정에 `--prove-losses --prove-losses-last-n 12 --fast <모드>`만 더한다.
5. H6-2a′ 학습 → dev 비교 → 최종 gate(위 5–7단계).

**v3 라벨 전체 생성 결과 (2026-10-17, `docs/h6/h6_1_report_v3.json`, 실행 코드 `4ec548e`, 결과 커밋 `f16292b`)**

- 설정: v2와 같은 전체(train·val·test, `sample_every 0`)에 `--prove-losses --prove-losses-last-n 12 --fast fast3`만 더했다.
  손실 증명 예산은 pilot과 같다(VCF 호출당 20,000 노드, 국면당 100,000 노드, 5,000 호출).
- 실행: Desktop WSL2(Ubuntu 24.04, i5-12600K, Python 3.14.8, gcc 13.3, workers 16), **416.8분**.
  - native 빌드의 C 소스 SHA256 `2a1e309c…`는 SER5와 같다. `.so` hash는 컴파일러가 달라 다르다.
  - 같은 작업을 SER5(5800H, fast3)에서도 돌리고 있다. 631.3분에 85%이므로 끝나면 약 740분으로 본다. Desktop WSL2가 **약 1.7–1.8배** 빠르다.

| | v2 | v3 | 차이 |
|---|---|---|---|
| 국면(rows) | 3,499,207 | 3,499,207 | 0 |
| 라벨 있는 국면 | 2,893,475 | 2,894,077 | +602 |
| outcome 라벨 | 2,725,718 | 2,696,327 | −29,391 |
| proof `vcf` / `unstoppable_four` / `immediate_win` / `forced_loss` | 99,783 / 28,852 / 22,064 / 17,058 | 같음 | 0 |
| **proof `vcf_loss`** | 0 | **29,993** | +29,993 |
| 부호 +1 / −1 | 1,477,042 / 1,416,433 | 1,476,725 / 1,417,352 | −317 / +919 |
| flag `end_unknown` / `outcome_conflicts_proof` / `h2_masked` | 2,314,848 / 41,622 / 86,065 | 2,297,449 / 41,055 / 86,065 | −17,399 / −567 / 0 |
| budget cut | — | 33,342 | |

- **v3는 의도한 것만 바꿨다.**
  - 국면, split별 개수, 제외된 대국(probe 888판), 기존 proof 4종의 개수가 v2와 같다.
  - 새로 생긴 것은 `vcf_loss` 29,993개뿐이다.
  - 그중 29,391개는 원래 outcome 라벨(가중치 0.1)이 있던 국면이고, 증명 라벨(가중치 1.0)로 바뀌었다. 602개는 라벨이 없던 국면에 새로 생겼다.
  - outcome이 증명과 부딪혀 가려지던 국면(`outcome_conflicts_proof`)도 567개 줄었다.
- **pilot 예측과 맞는다.** pilot은 1,961판에서 모든 ply로 `vcf_loss` 587개였다.
  전체 107,123판이면 54.6배로 약 32,100개이고, N = 12가 그중 95.4%를 담으므로 약 30,600개를 예상했다. 실제는 29,993개(98%)다.
- `vcf_loss`는 종료까지 거리가 짝수인 곳에 몰려 있다(2, 4, 6수가 각각 7–8천). 패배하는 쪽이 둘 차례인 국면이다.
  budget cut도 같은 패턴이다(짝수 거리 26,099 / 33,342). cut은 패배 증명이 예산을 넘은 국면에 생기기 때문이다.
- `proof_by_distance_to_end`의 합(예: `vcf_loss` 30,116)은 `by_kind`(29,993)보다 조금 크다. 이 분포는 `from_ply` 5 미만을 포함한 모든 증명을 센다. 그래서 다섯 종류 모두 같은 방식으로 조금씩 크다. 오류가 아니다.
- **남은 확인: 기계가 달라도 라벨이 같은가.** v3 builder는 모델 없이 규칙과 solver만 쓴다(`model_hash` None). 그래서 SER5와 Desktop WSL2의 결과는 같아야 한다.
  SER5가 끝나면 아래로 확인한다. 같으면 앞으로 CPU 라벨 작업은 Desktop WSL2에서 돌린다.

```bash
# 두 기계 각각: 행 내용 hash (gzip 헤더의 시각·파일명은 빼고 압축을 푼 내용으로 잰다)
zcat <v3 labels>.jsonl.gz | sha256sum
zcat <v3 labels>.jsonl.gz.budget_cuts.jsonl 2>/dev/null | sha256sum || sha256sum <v3 labels>.jsonl.gz.budget_cuts.jsonl
# summary 비교 (SER5 보고서를 Desktop으로 가져온 뒤)
python - <<'PY'
import json
a = json.load(open('docs/h6/h6_1_report_v3.json'))['summary']
b = json.load(open('runs/h6/h6_1_report_v3_ser5.json'))['summary']
diff = [k for k in set(a) | set(b) if a.get(k) != b.get(k)]
print('SAME' if not diff else f'DIFFERENT: {diff}')
PY
```

- 다음: v3 value 캐시 → H6-2a′ 학습 → dev 비교 → 최종 gate(위 5–7단계). 캐시는 v2와 같은 명령에 v3 라벨과 `docs/h6/h6_1_report_v3.json`을 주고, 출력 폴더만 `value_cache_v3`로 바꾼다.

**v3 value 캐시 (2026-10-17, Desktop Windows `.venv-cpu`, 12 worker, 88.1초, `data/tactical/h6/value_cache_v3`)**

- v2 캐시와 같은 환경·명령으로 만들었다. 라벨 파일과 보고서만 v3다(`labels_sha256` `b80f617f…`, `h6_report_sha256` `fc81b8ff…` = 커밋된 `docs/h6/h6_1_report_v3.json`).
- 합계가 v3 보고서와 모두 맞는다.

| | v2 캐시 | v3 캐시 | 차이 |
|---|---|---|---|
| 국면 train / val / test | 2,682,580 / 106,223 / 104,672 | 2,683,098 / 106,274 / 104,705 | +518 / +51 / +33 (합 +602 = 보고서 `labelled` 차이) |
| 버린 mask 행 | 470,282 / 67,732 / 67,718 | 469,764 / 67,681 / 67,685 | 같은 수만큼 감소. 분할별 행(국면 + mask)은 그대로 |
| 증명 +1 / −1 (합) | 150,699 / 17,058 | 150,699 / **47,051** | −1만 +29,993 (= `vcf_loss`) |
| 결과 라벨 (합) | 2,725,718 | 2,696,327 | −29,391 |
| 대국 수 | 98,993 | 98,996 | +3: 행이 모두 mask였던 대국에 `vcf_loss`가 생겼다 |

| 분할 | 증명 weight 비중 v2 → v3 | 증명 중 −1 비율 v2 → v3 | 결과 +1 비율 v2 → v3 |
|---|---|---|---|
| train | 37.1% → 41.3% | 10.0% → 23.6% | 48.7% → 49.2% |
| val | 50.0% → 54.3% | 12.3% → 25.1% | 48.1% → 48.9% |
| test | 47.5% → 52.3% | 10.6% → 24.9% | 48.3% → 49.1% |

- **바뀐 것은 계획한 변수 하나다.** 분할, 국면 집합, 기존 증명 라벨은 같다. `vcf_loss`가 결과 라벨(가중치 0.1)을 증명 라벨(1.0)로 바꿨다.
  - 그래서 증명 −1이 2.8배가 되고, 증명 weight 비중이 4%p 안팎 오른다.
  - 결과 +1 비율이 조금 오르는 것은, 바뀐 결과 라벨 대부분이 진 쪽(−1)의 국면이었기 때문이다.
- **해석 주의: v2와 v3 모델을 val 손실로 비교하지 않는다.** val의 라벨 자체가 바뀌었다(증명 비중 50 → 54%, −1 증명 2.0배).
  그래서 val weighted MSE는 두 실험 사이에 같은 잣대가 아니다. 각 실험 안에서 best를 고르는 데만 쓴다.
  v2와 v3의 비교는 H6-3 gate(고정 probe, kind별·ply별 지표)로 한다.
- 학습량: train 2,683,098국면, batch 1024면 epoch당 약 2,620 step, 3 epoch에 약 7,860 step이다(v2 7,857).

H6-2a′ 실행(Desktop Windows, RTX 5070). 계획대로 캐시, 보고서, 출력 폴더만 바꾼다. 나머지는 `configs/hybrid/h6_value_head.yaml`(H6-2a와 같은 설정)이다.

```powershell
$H3 = "..\renju-alphazero\runs\h3_policy_64x4\best.pt"
..\renju-alphazero\.venv-cuda\Scripts\Activate.ps1
# smoke (GPU, 300 step, 결과는 쓰지 않음)
python -u scripts\train_h6_value.py --config configs\hybrid\h6_value_head.yaml --h3-checkpoint $H3 `
  --cache-dir data\tactical\h6\value_cache_v3 --h6-report docs\h6\h6_1_report_v3.json `
  --out-dir runs\h6\value_v3_smoke --max-steps 300 --eval-every 100 --eval-max-states 50000 2>&1 | Tee-Object runs\h6\value_v3_smoke.log
# 본 실행 (중단되면 같은 인자에 --resume)
python -u scripts\train_h6_value.py --config configs\hybrid\h6_value_head.yaml --h3-checkpoint $H3 `
  --cache-dir data\tactical\h6\value_cache_v3 --h6-report docs\h6\h6_1_report_v3.json `
  --out-dir runs\h6_value_head_v3 2>&1 | Tee-Object runs\h6\value_v3.log
```

## H6-2a′ 학습 결과 (2026-10-17, `runs/h6_value_head_v3/best.json`, 커밋 `a5a3ea7`, `git_dirty false`)

- 실행: RTX 5070, 7,860 step(2,620 × 3 epoch), 22,179 samples/s, 6.0분. best는 마지막 step이다(val weighted MSE 0.6227 → 0.5878 → 0.5766 → **0.5736**).
- 통제: H6-2a(v2)와 같은 H3 부모(`efee4832…`), 같은 `frozen_state_sha256`(`c2fbaac3…`, trunk·policy head 고정), 같은 seed·optimizer·epoch·schedule이다. 바뀐 것은 라벨(`b80f617f…`)과 캐시뿐이다.

**val 비교: 같은 잣대가 아닌 지표를 가려 읽는다.** v3는 val의 outcome −1 1,584행과 +1 18행을 `vcf_loss` 증명으로 바꿨다. 그래서 지표마다 비교 가능한 정도가 다르다.

| 지표 | v2 | v3 | 비교 가능성 |
|---|---|---|---|
| 전체 sign acc / MSE (가중치 없음) | 65.56% / 0.8705 | 66.24% / 0.8593 | **거의 같은 국면·같은 목표값**: 바뀐 행도 목표 부호(−1)는 같다. 새 국면은 51개 |
| 전체 평균 예측 − 평균 목표 | +0.130 | +0.082 | 같음. 낙관 편향이 줄었다 |
| outcome −1 sign acc | 56.02% (50,106) | 61.22% (48,522) | **부분집합이 다르다.** 빠진 1,584행에서 v2가 모두 맞혔거나 모두 틀렸다고 해도, 남은 국면에서 v2는 54.6–57.9%다. 그래서 v3의 개선은 **+3.4–6.6%p**로 남는다 |
| outcome +1 sign acc | 69.77% | 65.22% | 18행만 빠져 사실상 같다. **−4.5%p는 실제 손실이다** |
| outcome 전체 | 62.63% / 0.9390 | 63.18% / 0.9288 | 위 두 가지가 섞인 값 |
| proof −1 sign acc | 93.32% (1,182) | 87.20% (2,835) | **비교 불가.** v3에는 새 `vcf_loss` 1,653행이 더해졌다 |
| proof +1 sign acc | 95.08% | 93.56% | 같은 8,462행. −1.5%p |
| weighted MSE | 0.5620 | 0.5736 | **비교 불가.** 가중치(증명 비중 50 → 54%)와 목표가 다르다 |

- 해석:
  - value head의 균형점이 음수 쪽으로 옮겨 갔다. 결과 패배 쪽(+3.4–6.6%p)과 전체 편향은 좋아졌다. 결과 승리 쪽(−4.5%p)과 증명 승리 쪽(−1.5%p)은 대가를 치렀다.
  - 전체 가중치 없는 지표는 소폭 좋아졌다.
- `vcf_loss` 1,653행에서의 정확도는 best.json에 따로 없다.
  forced_loss 1,182행에서 v2만큼(93%) 맞혔다고 가정하면, 남은 `vcf_loss`는 약 82%다. 추정일 뿐이다.
  **v2가 같은 국면에서 몇 %였는지는 모른다.** 이것이 G4가 묻는 것이다.
- ply별: 5–9수만 −1.47%p다. 2,797국면이라 표준오차가 약 0.9%p이므로 잡음 범위다. 10수 이후는 모든 구간에서 +0.5–0.85%p다.
- side별: 백은 평균 예측이 목표와 거의 같아졌다(0.0767 대 0.0766). 흑은 낙관 편향이 남아 있다(+0.155 대 −0.010, v2 +0.194).
- 포화(|v| > 0.95)는 3.39% → 3.48%로 같다.
- **판정: 학습은 정상이고, 의도한 방향(패배 쪽 민감도)으로 움직였다. 우열은 아직 정하지 않는다.** epoch을 늘리는 등 설정은 바꾸지 않는다(H6-2a와 같은 조건 유지).

**6단계 dev 비교 (같은 국면에서, part B·fresh probe는 열지 않는다)**

1. `scripts/h6_cross_eval.py`: 두 모델을 **v3 캐시의 같은 국면**(val + test part A)에서 raw와 D4 평균으로 잰다.
   - v3가 바꾼 것으로 나눈다: `new_loss`(v3가 더한 `vcf_loss`), `old_loss`(forced_loss), `proof_win`, `outcome_win`, `outcome_loss`, `all`.
   - `new_loss`에서 v2와 v3를 직접 비교할 수 있다. G4와 같은 종류의 국면이다.
2. `scripts/h6_diagnose.py`(v3 모델 + v3 캐시)를 돌린다. 기존 v2 결과 `docs/h6/h6_3d_diag.json`과 D2(엔진 대국의 깊은 패배)·P93·P94의 D4 평균을 비교한다. 이 셋은 라벨과 무관하므로 그대로 비교된다.

```powershell
..\renju-alphazero\.venv-cuda\Scripts\Activate.ps1
$env:PYTHONPATH = "$PWD\src"; $env:PYTHONIOENCODING = "utf-8"
git pull --ff-only; git status --short
python -u scripts\h6_cross_eval.py --model v2=runs\h6_value_head_v1\best.pt --model v3=runs\h6_value_head_v3\best.pt `
  --v2-cache data\tactical\h6\value_cache_v2 --v3-cache data\tactical\h6\value_cache_v3 --device cuda `
  --output docs\h6\h6_2a_prime_cross_eval.json 2>&1 | Tee-Object runs\h6\h6_cross_eval.log
python -u scripts\h6_diagnose.py --checkpoint runs\h6_value_head_v3\best.pt `
  --cache-dir data\tactical\h6\value_cache_v3 --device cuda --output docs\h6\h6_3d_diag_v3.json 2>&1 | Tee-Object runs\h6\h6_3d_diag_v3.log
```

- 7단계(최종 gate, 한 번)는 이 두 결과를 보고 후보 하나를 정한 뒤에만 한다. 후보는 part B + fresh probe + D4 평균으로 돈다.

## H6-2a′ dev 비교 결과 (2026-10-17, `docs/h6/h6_2a_prime_cross_eval.json`, `docs/h6/h6_3d_diag_v3.json`, 커밋 `2d18616`)

두 모델을 v3 캐시의 **같은 국면**(val 106,274, test part A 53,556)에서 raw와 D4 평균으로 쟀다(`h6_cross_eval.py`).
H6-3D 진단은 v3 모델로 다시 돌려 v2 결과(`h6_3d_diag.json`)와 비교했다. part B와 fresh probe는 열지 않았다.

**1. 같은 국면 비교 (D4 평균, 최종 gate와 같은 추론). sign 정확도**

| subset | val v2 → v3 | testA v2 → v3 | 비고 |
|---|---|---|---|
| **new_loss** (v3가 더한 `vcf_loss`, 1,653 / 835) | 73.3% → **85.2%** (+11.9) | 72.7% → **85.1%** (+12.5) | MSE 0.850 → 0.553, 0.856 → 0.548 |
| old_loss (forced_loss, 1,182 / 469) | 95.7% → 95.6% (−0.1) | 95.1% → 94.2% (−0.9) | raw에서는 −1.0 / −1.5 |
| proof_win (8,462 / 4,022) | 96.5% → 95.2% (−1.3) | 96.8% → 94.9% (−1.9) | |
| outcome_win (46,455 / 23,653) | 70.4% → 65.7% (−4.6) | 70.5% → 66.2% (−4.3) | |
| outcome_loss (48,522 / 24,577) | 57.3% → 62.3% (+5.0) | 59.1% → 64.3% (+5.1) | |
| outcome balanced (승·패 평균) | 63.8% → 64.0% (+0.2) | 64.8% → 65.2% (+0.4) | |
| all | 66.8% → 67.1% (+0.3) | 67.5% → 68.0% (+0.5) | MSE 0.848 → 0.835, 0.835 → 0.820 |

- **목표 부분집합은 크게 좋아졌다.** `new_loss`가 두 집합에서 모두 +12%p 안팎이다. 표준오차는 1–1.5%p라 잡음이 아니다.
- **목표에 맞춘 이동이다. 전체 편향만 옮긴 것이 아니다.** `new_loss`의 평균 value는 −0.15 → −0.35(−0.20)로 움직였다. outcome 국면은 −0.03~−0.06, proof_win은 −0.03만 움직였다.
- **outcome의 승·패 맞교환은 대부분 결정 경계 이동이다.**
  - outcome 승 −4.5%p, 패 +5%p인데, 균형 정확도는 +0.2–0.4%p로 거의 같다.
  - H6-3D에서도 outcome balanced accuracy가 0.6503 → 0.6523(D4)이다.
  - 대신 calibration이 좋아졌다: outcome 편향 0.153 → 0.092, ECE 0.076 → 0.046(D4).
  - 즉 v2의 낙관 편향을 줄인 것이고, 승패를 더 잘 가리게 된 것은 아니다.
- D4 평균에서는 old_loss의 손실이 val에서 사라진다(−0.1%p). all의 이득은 raw(+0.7%p)보다 작은 +0.3–0.5%p다.

**2. H6-3D: 깊은 전술은 좋아지지 않았다**

| 진단 | v2 | v3 | 해석 |
|---|---|---|---|
| D2 깊은 승리 child 인식, D4 (157수, root 9개) | 96.8% (152) | **92.4%** (145) | **−4.5%p.** root 단위 97.1% → 91.4%, 패배 깊이 2: 96.9% → 92.2% |
| D2 raw | 92.4% | 89.2% | 같은 방향 |
| **P94** (흑 차례, 증명된 패배, **< 0이어야 함**) | 원래 +0.980, D4 +0.894 | 원래 +0.974, D4 **+0.983**(+0.94 ~ +1.00) | **여전히 FAIL, 오히려 더 확신한다** |
| P93 (백 차례, 깊은 승리, > 0이어야 함) | 원래 −0.361, D4 −0.390 | 원래 +0.000, D4 −0.011 | 0 쪽으로 왔지만 여전히 FAIL |
| D1 8개 부호 모두 같음 / 표준편차 평균 | 56.5% / 0.136 | 60.6% / 0.140 | 거의 같다 |

- D2는 진 수 뒤의 국면(상대 차례, +여야 함)을 맞히는지 보는 child 단위 진단이다. v3가 패배 쪽으로 옮겨 가면서 깊은 **승리** 인식이 7수 줄었다. 표본이 157수(9 root)라 크기는 불확실하지만, 방향은 outcome 승·proof 승의 손실과 같다.
- **P94는 이 실험이 고치려던 G4의 바로 그 종류(부모 단위 깊은 패배, 모든 수가 VCT1에서 진다)인데 개선되지 않았다.**
  - `vcf_loss`는 깊이 0 증명(VCF로 지는 국면)이다. P94의 패배는 그보다 깊다(§H6-3A의 3번).
  - 이번 결과는 그 우려와 맞는다: v3는 RenjuNet 분포의 VCF 패배를 배웠지만, 깊은 패배 표현으로는 옮겨 가지 않았다.
  - 단서: P93·P94는 한 대국의 두 국면이다. 이것만으로 일반화할 수는 없다.

**3. 최종 gate 기준에 비춘 dev 추정 (test part A, D4)**

| gate | 기준 | v3 dev 값 | 판단 |
|---|---|---|---|
| G1 | proof BA ≥ 0.90, proof −1 ≥ 0.85 | 0.916 / 0.884 | 통과 예상(여유 1.6 / 3.4%p) |
| G2 | outcome MSE < 1.0, BA ≥ 0.60 | 0.883 / 0.652 | 통과 예상 |
| G3 | D4 흔들림 | D4 평균이라 구조상 0 | 통과 |
| G5 | 흑·백 BA ≥ 0.55, 차이 ≤ 0.05 | 0.641 / 0.665, 차이 0.024 | 통과 예상 |
| **G4** | fresh probe 부호별 정확도 ≥ 0.80(28개 중 오답 5개까지) | dev 근거: P94 FAIL(더 확신), D2 승리 쪽 −4.5%p | **불확실. 가장 큰 위험** |

- fresh probe의 −1은 `defense_label`이 `vct_loss`를 증명한 국면이다. P94와 같은 종류다. +1은 깊이 1 VCT 승리라 D2·P93 쪽과 가깝다.
- 최종 gate는 한 번뿐이다. 실패하면 part B를 다시 쓸 수 없다(`FAILURE_PATHS['G4_fresh']`: "no B re-run").

**4. D4 평균의 비용.** D4는 처리량이 12.9배 줄어든다(125.6k → 9.7k positions/s). 그런데 국면 하나의 지연은 오히려 같거나 짧다(raw 1.19ms, D4 8개 batch 0.88ms).
PUCT처럼 leaf를 하나씩 평가하면 GPU가 놀고 있으므로 D4 비용이 거의 없다. H6-4에서 batch 크기별로 잰다.

**판정 (dev).**

- v3는 학습 목표였던 `vcf_loss` 국면을 크게 개선했다(+12%p, 같은 국면). 낙관 편향과 calibration도 좋아졌다.
- 그러나 outcome 판별력은 거의 그대로이고, 승리 쪽(outcome 승 −4.5%p, proof 승 −1.3~1.9%p, D2 깊은 승리 −4.5%p)을 내줬다.
- **H6-2a′의 목적인 G4(부모 단위 깊은 패배)는 dev에서 개선된 증거가 없다**(P94 여전히 +0.98).
- 그래서 "v3를 그대로 최종 gate에 올린다"는 초안의 결론은 보류한다. G1·G2·G3·G5는 통과가 예상되지만, G4가 실패할 가능성이 높아 보인다. 실패하면 최종 집합(part B·fresh probe)을 써 버리게 된다.
- 다음 결정은 아래 "다음 단계 선택지"로 한다. 결정 규칙은 dev deep probe를 보기 전에 정한다.

**다음 단계 선택지**

- **A. 계획대로 최종 gate(v3, part B + fresh probe + D4)를 지금 한 번 돌린다.**
  - 사전 프로토콜에 가장 충실하다. dev 결과로 후보를 바꾸지 않는다.
  - 대신 G4 실패 위험을 안고 최종 집합을 쓴다.
- **B. (권장) 최종 gate 전에 dev용 깊은 probe 집합을 따로 만든다.**
  - 같은 생성기(`make_h6_fresh_probes.py`)에 새 seed(예: 8611–8613, 학습 데이터에서 영구 제외)를 쓴다.
  - v2+D4와 v3+D4를 부호별로 잰다.
  - 규칙은 지금 정한다: **v3의 dev deep-win·deep-loss 정확도가 둘 다 0.80 이상이면 최종 gate로 간다.** 아니면 최종 집합을 쓰지 않고, 엔진 대국에서 나온 깊은 라벨(H6-2b/H7, §H6-3A의 4번)로 간다.
  - 비용: 엔진 대국 60판(8601–8603과 같은 규모)과 probe 추출. 모델 학습은 없다.
- **결정(2026-10-17): B안.** 프로토콜과 판정 규칙은 아래 "H6-2a′ B안" 절에 결과 보기 전에 고정했다.

## H6-2a′ B안: dev deep-probe 사전 점검 (2026-10-17, 결과 보기 전에 동결)

사용자가 B안을 골랐다. 아래 프로토콜과 판정 규칙은 **dev probe 대국을 하나도 두기 전에** 이 커밋으로 고정한다. 결과를 본 뒤에는 바꾸지 않는다.

**동결하는 것**

- **모델:** v3 checkpoint를 지금 것으로 동결한다(`runs\h6_value_head_v3\best.pt`, sha256 `a0fcccc401a833a08994e21cd7e963d5e7adc7a7cff715de14259bbed733ceca`). 평가 스크립트가 sha256을 확인하고, 다르면 실행하지 않는다.
- **재조정 금지:** 결과가 나빠도 LR, epoch, loss weight 등을 바꿔 같은 dev probe에 다시 맞추지 않는다. 통과하면 추가 튜닝 없이 바로 최종 gate로 간다.
- **seed:** 기본 8611–8613, 확장 8614–8616(아래 확장 규칙으로만 사용). 여섯 개 모두 `RESERVED_SEEDS`에 넣었다(`DEV_PROBE_SEEDS`, `DEV_PROBE_EXTENSION_SEEDS`). 쓰든 안 쓰든 이 seed의 대국은 H6, H7, 자기대국 어떤 학습 데이터에도 **영구히** 넣지 않는다.
- **대국:** 8601–8603과 같은 설정이다. TB-Engine-v1(`puct_policy_vct2_50k`, c 1.5) 대 `v8:full`, seed당 20판(`--pairs 10`).
- **probe 생성:** `make_h6_fresh_probes.py --dev`. 탐색과 truth는 최종 fresh probe와 같다(scan v3, 같은 상한). from-ply 10.
  - +1 = 깊이 0 라벨이 없고 `vct1_win`이 증명된 국면. −1 = `defense_label`이 `vct_loss`를 증명한 국면(P94 종류).
  - 최종 probe와 다른 점은 하나다. 표본을 늘리려고 **대국마다 부호별로 하나씩**(가장 이른 깊은 승리, 가장 이른 깊은 패배) 뽑는다. 부호 안에서는 여전히 대국당 하나이므로, 부호별 판정의 독립성은 최종 probe와 같다.
  - 최종 fresh probe와 D4로 같은 국면, 기존 probe·평가 국면은 뺀다.
  - 출력 `docs/h6/h6_fresh_dev_probes_v1.json`은 `purpose: dev`, `sealed: false`로 표시된다. 최종 gate(`h6_offline_gate.py`)는 이 파일을 `--fresh-probes`로 받지 않는다. 기본 probe 파일 목록에는 들어가므로, 이후 라벨 생성에서 이 국면을 지나는 RenjuNet 대국이 빠진다.

**확장 규칙 (개수만 보고 정한다. 모델은 아직 돌리지 않는다)**

1. 8611–8613(60판)으로 추출해 `counts`의 win·loss 개수만 확인한다.
2. 두 부호가 모두 50개 이상이면 그대로 평가한다.
3. 어느 한 부호라도 50개 미만이면 8614–8616(60판)을 추가하고, **여섯 seed 파일을 함께** 넣어 다시 추출한다. 확장은 이 한 번뿐이다.
4. 확장 뒤에도 어느 부호가 20개 미만이면 판정할 표본이 없는 것으로 보고 **FAIL**로 처리한다(최종 집합은 열지 않는다).
- 참고: 최종 probe 추출(60판)에서 깊은 승리가 있는 대국은 52판, 깊은 패배가 있는 대국은 45판이었다. 그래서 기본 60판으로는 loss 쪽이 50개에 못 미쳐 확장될 가능성이 높다.

**판정 규칙 (D4 평균 추론만, v3만 판정)**

| 판정 | 조건 | 다음 단계 |
|---|---|---|
| **PASS** | deep-win 정확도 ≥ 0.80 **그리고** deep-loss 정확도 ≥ 0.80, 각 부호 20개 이상 | 추가 튜닝 없이 최종 gate(v3, part B + sealed fresh probe + D4) 한 번 |
| **FAIL** | 어느 한 부호라도 0.80 미만, 또는 20개 미만 | 최종 집합(part B·fresh probe)은 열지 않는다. H6-2b/H7(더 좋은 깊은 라벨)로 간다 |

- 정확도는 value와 기대 부호의 곱이 양수인 비율이다(`h6_gate.fresh_probes`와 같다. value 0은 오답).
- 결과는 반드시 표본 수와 함께 적는다. 예: `48 / 57 = 84.2%`. 스크립트가 이 형식으로 출력한다.
- v2+D4도 같은 probe로 재지만 **참고용**이다. 판정에는 쓰지 않는다.
- 이 점검은 한 번만 돈다. 같은 출력 파일이 있으면 실행하지 않는다.

**실행 명령 (Windows, 8601–8603 때와 같은 환경)**

```powershell
cd "C:\오목 강화학습\renju-h6"
..\renju-alphazero\.venv-cuda\Scripts\Activate.ps1
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = "$PWD\src"
git pull --ff-only
git status --short
New-Item -ItemType Directory -Force runs\h6_dev | Out-Null

# 1) 엔진 대국: 기본 seed 8611-8613 (seed당 20판)
foreach ($seed in 8611, 8612, 8613) {
  python -u scripts\run_mcts_v8_benchmark.py --arm puct_policy_vct2_50k --puct-c 1.5 --opponent v8:full --seed $seed --pairs 10 `
    --workers 6 --policy-checkpoint ..\renju-alphazero\runs\h3_policy_64x4\best.pt `
    --games-jsonl runs\h6_dev\dev_$seed.jsonl --output runs\h6_dev\dev_$seed.json 2>&1 | Tee-Object runs\h6_dev\dev_$seed.log
}

# 2) 개수만 확인 (모델 없음). runs\ 아래 임시 출력
python -u scripts\make_h6_fresh_probes.py --dev --games runs\h6_dev\dev_*.json --workers 12 `
  --jsonl runs\h6_dev\scan.jsonl --output runs\h6_dev\count_base.json
```

- `count_base.json`의 `counts.win`, `counts.loss`가 둘 다 50 이상이면 3)으로 간다. 하나라도 50 미만이면 1)을 seed 8614, 8615, 8616으로 돌린 뒤 2)를 `--output runs\h6_dev\count_ext.json`으로 다시 돌려 개수를 확인한다(`dev_*.json`이 여섯 파일을 모두 잡고, 끝난 대국은 `scan.jsonl`에서 건너뛴다).

```powershell
# 3) 확정 probe 파일 (개수 확인과 같은 입력) -> 커밋
python -u scripts\make_h6_fresh_probes.py --dev --games runs\h6_dev\dev_*.json --workers 12 `
  --jsonl runs\h6_dev\scan.jsonl --output docs\h6\h6_fresh_dev_probes_v1.json
git add docs\h6\h6_fresh_dev_probes_v1.json
git commit -m "H6-2a' option B: dev deep probes (truth only)"

# 4) 평가 (깨끗한 worktree에서 한 번)
python -u scripts\h6_dev_probe_eval.py --model v3=runs\h6_value_head_v3\best.pt --model v2=runs\h6_value_head_v1\best.pt `
  --probes docs\h6\h6_fresh_dev_probes_v1.json --device cuda `
  --output docs\h6\h6_dev_probe_eval.json 2>&1 | Tee-Object runs\h6_dev\dev_eval.log
```

- 마지막 줄 `decision: PASS` 또는 `decision: FAIL`이 판정이다. PASS면 다음은 최종 gate 명령 하나다. FAIL이면 최종 집합을 열지 않고 H6-2b/H7 설계로 간다.

## H6-2a′ B안 중간 결과와 프로토콜 이탈 (2026-10-17, `docs/h6/h6_fresh_dev_probes_v1.json`, 커밋 `4e824a4`)

**개수 (seed 8611–8613, 60판, scan v3, 모델 없음)**
- probe: win 50, loss 41. 깊은 승리가 있는 대국 50판, 깊은 패배가 있는 대국 41판이다.
- 상한을 넘어 빠진 국면은 120개(30판)다. 최종 probe 추출(37개, 17판)보다 많다. 이런 국면은 probe 후보에서 빠지므로, probe는 "상한 안에서 증명되는 국면" 쪽으로 치우친다. 최종 probe도 같은 규칙이라 둘은 비교할 수 있다.

**프로토콜 이탈**
- 확장 규칙 3번에 따르면 loss 41 < 50이므로 8614–8616을 추가한 뒤 평가했어야 한다. 실제로는 확장 없이 확정 파일을 만들고 평가까지 실행했다.
- 그래서 아래 결과는 **중간 결과(interim)**로 기록한다. 공식 판정은 사전에 정한 평가 집합(여섯 seed)으로 한 번 더 한다.
- 이렇게 해도 사후 조작은 아니다. 확장 여부는 결과가 아니라 개수(41 < 50)로 이미 정해져 있었고, 확장 seed도 미리 고정되어 있었다. 공식 판정 결과가 FAIL이면 그대로 따른다.

**중간 결과 (D4 평균, `docs/h6/h6_dev_probe_eval.json`)**

| 모델 | deep win | deep loss | 규칙 |
|---|---|---|---|
| **v3 (판정)** | **50 / 50 = 100.0%** (Wilson 95% 92.9–100%) | **39 / 41 = 95.1%** (83.9–98.7%) | PASS |
| v2 (참고) | 50 / 50 = 100.0% | 35 / 41 = 85.4% (71.6–93.1%) | PASS |

- 두 부호 모두 95% 구간의 하한이 0.80을 넘는다. 표본이 41개라도 우연히 통과했을 가능성은 낮다.
- 확장으로 loss가 약 40개 늘어도, 그중 26개 이상(약 65%)만 맞히면 합계가 0.80 이상이다. 그래서 공식 판정도 PASS일 가능성이 높다.
- v3는 deep loss에서 v2보다 4개를 더 맞혔다(+9.8%p). H6-2a′의 목적과 같은 방향이다. 다만 41개에서 4개 차이라 우열을 단정하지는 않는다.
- P94(v3 +0.98, 실패)는 이 probe 종류의 대표가 아니었던 것으로 보인다. 같은 종류의 국면 41개 중 v3는 39개, v2는 35개를 맞혔다. P93·P94는 대국 하나에서 나온 국면이라는 기존 해석(D2)과 맞는다.

**공식 판정 (사전 규칙대로)**
1. seed 8614–8616 대국을 추가한다.
2. 여섯 seed로 `docs/h6/h6_fresh_dev_probes_v2.json`을 만든다. dev 파일끼리는 서로 제외하지 않으므로(`--dev`는 이전 dev 파일을 제외 목록에서 뺀다), v1의 probe가 그대로 포함된다.
3. 같은 규칙으로 한 번 평가한다(`docs/h6/h6_dev_probe_eval_v2.json`). 이 결과가 판정이다. PASS면 튜닝 없이 최종 gate로 가고, FAIL이면 최종 집합을 열지 않는다.

## H6-2a′ B안 공식 판정: PASS (2026-10-18, `docs/h6/h6_fresh_dev_probes_v2.json`, 커밋 `008589b`, `docs/h6/h6_dev_probe_eval_v2.json`)

사전 규칙대로 seed 8614–8616을 더해 여섯 seed(120판)로 한 번 평가했다. 이 결과가 B안의 판정이다.

**개수 (모델 없음)**
- probe: win 93, loss 81(깊은 승리가 있는 대국 93판, 깊은 패배가 있는 대국 81판). 두 부호 모두 최소 20을 넘고, loss는 목표 50도 넘었다.
- 상한을 넘어 빠진 국면 254개(58판). 3 seed 때와 비율이 같다(대국의 약 절반에서 한 국면 이상).

**결과 (D4 평균)**

| 모델 | deep win | deep loss | 판정 |
|---|---|---|---|
| **v3 (판정)** | **93 / 93 = 100.0%** (Wilson 95% 96.0–100%) | **78 / 81 = 96.3%** (89.7–98.7%) | **PASS** |
| v2 (참고) | 93 / 93 = 100.0% | 70 / 81 = 86.4% (77.3–92.2%) | PASS |

- 확장분만 따로 보면(3 seed 결과를 뺀 값): v3 deep win 43 / 43, deep loss 39 / 40. v2 deep loss 35 / 40. 중간 결과와 같은 경향이 새 대국에서도 재현됐다.
- **판정: PASS.** 규칙대로 튜닝 없이 v3로 최종 gate를 한 번 돈다.

**해석과 주의점**
- deep loss: v3가 v2보다 8개 더 맞혔다(+9.9%p). 같은 probe에 대한 쌍대 비교이고, 3 seed와 확장분에서 모두 같은 방향이다. `vcf_loss` 라벨 추가가 부모 단위의 깊은 패배(`vct_loss`) 인식에도 옮겨 간 것으로 보인다. 다만 v2도 0.80을 넘으므로, 이 probe에서 v2가 "실패"했던 것은 아니다.
- deep win은 두 모델 모두 100%다. 이 종류(깊이 0 위협이 없는 상태의 `vct1_win`)는 이 망에게 쉬워서 모델을 구별하지 못한다. D2 깊은 승리(엔진 국면, v3 D4 92.4%)와 종류가 다르다는 점은 남는다.
- P94 실패(v3 +0.98)는 같은 종류 81개 중 3개 오답에 해당하는 드문 경우로 보인다. G4를 대국 하나(P93·P94)로 판정하지 않고 fresh probe로 바꾼 결정이 맞았다.
- probe는 계산 상한 안에서 증명되는 국면만 담는다(약 절반의 대국에서 한 국면 이상이 상한으로 빠짐). 최종 fresh probe도 같은 생성기라서 dev 결과는 최종 G4를 잘 예측할 것으로 보지만, 아주 복잡한 국면에 대해서는 말하지 않는다.
- 최종 gate의 G4 기준은 dev보다 표본이 작다(부호별 28개). 28개 중 오답 5개까지 허용된다. dev 오답률(loss 3.7%)이 그대로라면 기대 오답은 약 1개다.

**다음: 최종 gate (한 번, 후보 v3)**
- part B + 봉인된 fresh probe(`h6_fresh_final_probes_v1.json`) + D4 평균. 결과에 따라 모델·설정을 바꾸지 않는다.
- PASS면 H6-4(Value → PUCT)로 간다. FAIL이면 실패한 gate의 `next_steps_if_failed`를 따르고, part B는 다시 쓰지 않는다.

## H6-3 최종 gate: PASS (2026-10-18, `docs/h6/h6_final_gate.json`, v3 + D4 평균, part B 1회)

H6-2a′ B안의 공식 판정(PASS) 뒤, 튜닝 없이 봉인된 최종 gate를 한 번 실행했다. 대상은 v3 checkpoint(`a0fcccc4…`) + D4 평균 추론이다. part B(test split 중 개발에 쓰지 않은 절반)와 봉인된 fresh probe를 처음 열었다.

| 항목 | 결과 |
|---|---|
| 판정 | **PASS** (G1–G5 모두 통과) |
| part B | 2,637대국, 51,149국면 (part A 2,795대국과 겹치지 않음) |
| G1 proof (5,015국면) | balanced 91.94%, +1 94.77%, −1 89.10% (1,266국면), MSE 0.248 |
| G2 outcome (46,134국면) | balanced 64.28%, +1 65.66%, −1 62.89%, MSE 0.906 |
| G3 D4 (2,000국면) | 평균 std 4.2e−9, p95 3.0e−8, 확신 국면 부호 일치 100% |
| G4 fresh probe | deep win **28 / 28 = 100%**(평균 +0.807), deep loss **26 / 28 = 92.9%**(평균 −0.566, 오답 F29 +0.04, F38 +0.54) |
| G5 색별 outcome | 흑 64.08%, 백 64.84% (차이 0.77%p) |
| ECE / λ | 0.0495 → **λ = 1.0** (ECE ≤ 0.10, 사전 규칙: leaf는 value만, rollout 혼합 없음) |

**1. held-out에서도 재현된다.**
- dev(part A, D4)와 비교하면 proof balanced 91.6% → 91.9%, outcome balanced 65.2% → 64.3%, ECE 0.046 → 0.049다. 차이는 1%p 안팎이다. 개발 partition에 맞춰진 모델이라는 근거는 없다.
- proof −1 89.1%는 **깊이 0 패배**(`forced_loss`, `vcf_loss`)의 정확도다. v3가 더한 `vcf_loss` 학습이 새 대국에도 일반화됐다는 뜻이다.
  - 깊은 패배(부모 단위 `vct_loss`)는 학습 라벨에 없다. 이 능력의 근거는 G4(fresh probe)와 dev deep probe(78 / 81)뿐이다. proof −1 수치를 깊은 패배의 근거로 쓰지 않는다.

**2. outcome 판별력은 중간 수준이다.**
- 증명 국면에서는 강하지만(약 92%), 일반 국면의 승패 방향은 약 64%만 맞힌다.
- 따라서 value는 solver나 전술 veto를 대신하지 않는다. 증명된 결과(V8 Stage, VCF/VCT 판정, PROVEN_LOSS veto)가 항상 우선이고, value는 tree leaf 평가만 맡는다.

**3. G3는 구성상 통과한다. 망이 대칭에 불변해진 것은 아니다.**
- D4 평균은 8개 대칭의 평균이므로 수학적으로 정확히 불변이다. std 1e−8은 부동소수점 오차다. 이 수치는 구현이 맞다는 확인이지, 망 자체가 개선됐다는 증거가 아니다.
- 망의 raw 출력은 여전히 방향에 따라 흔들린다(v2 dev 기준 국면별 std 평균 0.136).
- 그래서 검증된 것은 **"v3 + D4 평균" evaluator**다. raw 단일 추론은 다른 evaluator다. part B를 이미 썼으므로 raw로 바꾸면 offline gate로 다시 검증할 수 없고, 대국 평가로만 판단해야 한다. H6-4·H6-5는 D4 평균으로 고정한다.

**4. P93·P94 실패는 G4를 뒤집지 않는다.**
- P93 −0.010(기대 +, 사실상 0), P94 +0.983(기대 −, 확신 오답). dev 값과 같다.
- 사전에 G4를 대국 하나(P93·P94)가 아니라 fresh probe로 판정하기로 바꿨고, 같은 종류 81개 dev probe에서 v3는 78개를 맞혔다. P93·P94는 회귀 관찰용 진단 국면으로 남긴다.

**5. P92는 policy 순위가 아니라 value 순위다. 그리고 크게 좋아졌다.**
- `h6_gate.probes`의 P92는 흑의 모든 합법수 뒤 국면을 value로 매긴 순위다(점수 = −v(child), 1위가 value의 최선). policy head와는 무관하다.
- 실제 진 수 (3,4)의 순위는 v2에서 **6 / 133**(value가 거의 최선으로 봄)이었고, v3에서는 **126 / 133**이다. value는 이제 이 수를 최하위권으로 본다.
- 단, (3,4) 뒤 국면이 P93이므로 그 점수(+0.010)는 P93 value(−0.010)의 부호만 바꾼 값이다. 순위가 낮은 이유는 "(3,4)가 진다"고 확신해서가 아니라, 다른 수 대부분을 흑에게 더 좋게 보기 때문이다. 또 P92의 참값은 아직 UNRESOLVED다(§12.20). 그래서 "개선 신호"로만 기록하고, 판정은 H6-4/5에서 search 단위로 한다.

**6. 색별 bias.**
- outcome bias(평균 v − 평균 t)는 흑 차례 +0.177, 백 차례 +0.013이다. 흑 차례 국면을 실제보다 낙관한다. balanced accuracy는 흑·백이 같아 G5는 통과했다.
- PUCT에서 같은 부모의 자식들이 모두 같은 색 차례 leaf로 끝나면 일정한 offset은 형제 비교에서 상쇄된다. 하지만 leaf 깊이가 섞이면 상쇄되지 않고, Q와 탐색항(U)의 균형도 바뀐다. H6-4에서 leaf의 차례 색별 value 평균을 기록하고, H6-5의 색별 점수 guard(흑·백 각 ≥ 0.45)로 본다.

**결론.**
- H6-3은 PASS다. **"offline evaluator 검증 완료"이지 "value 모델 완성"이 아니다.** 다음은 search 단위의 효용 검증(H6-4, H6-5)이다.
- H6 value head 자체의 추가 튜닝은 하지 않는다. H6-2b(trunk 미세조정)는 로드맵대로 H6-5 ADOPT 뒤에만 고려한다.
- λ = 1.0로 고정한다(사전 규칙). 대국 후보는 하나(v3 + D4, value-only leaf)다.

## H6-4 설계 방향 (2026-10-18, 초안)

목표: TB-Engine-v1의 tree route에서 **leaf 평가 하나만** rollout → value로 바꾼다. 나머지(V8 Stage 1–5, own VCF, V8-A/B/C, VCT2 veto, H3 policy prior, c 1.5, 시뮬레이션 50/100)는 그대로 둔다.

**1. 엔진 버전 단계로 진행한다(사용자 확인 필요).**
- `mcts_v8.py`가 `puct_v8.search_tree_puct`를 직접 부르므로, 동결 파일을 건드리지 않고는 leaf를 바꿀 수 없다. 모듈 전역 함수를 바꿔치기하는 방식은 다중 slot·테스트에서 위험하므로 쓰지 않는다.
- 제안:
  - 새 파일 `src/analysis/puct_value_v1.py`: `search_tree_puct`를 복사하고 leaf만 바꾼 함수. `puct_v8.py`는 그대로 둔다.
  - `mcts_v8.py`에 **추가만** 한다: `tree_mode="puct_value"` 분기와 `leaf_value` 인자. 기존 `tree_mode` 경로의 코드는 바꾸지 않는다.
  - 새 설정 `TB_V1_VALUE`(TB-Engine-v1 설정 + `tree_mode: puct_value` + value checkpoint sha)로 별도 config hash를 갖는다. TB-Engine-v1 자체는 바뀌지 않는다.
  - 회귀 증명: 기존 `tests/test_tb_engine_v1.py`와 고정 국면에서 TB-Engine-v1의 착수가 비트 단위로 같아야 한다.

**2. leaf 계약(§12.18).**
- 종국이면 기존과 같다(승자 기준 ±1, 무승부 0).
- 아니면 leaf 국면 s에서 v = V_D4(s)(둘 차례 관점, [−1, 1])를 구하고, leaf 노드 값(그 노드로 둔 쪽 관점) = **−v**. 부호 변환은 한 함수(`leaf_value`)에서만 한다.
- backup은 실수 값을 받는 별도 함수로 한다(기존 `_backpropagate`는 승자만 받는다). 부모 방향으로 부호를 바꿔 가며 더한다.
- solver 우선: leaf에서 둘 차례가 즉시 5목을 만들 수 있으면 value 대신 확정값을 쓴다. 이 검사는 싸므로 넣는다. 그보다 깊은 증명은 leaf에서 하지 않는다(root 모듈이 이미 한다).
- λ = 1.0이므로 rollout은 돌리지 않는다. 그러면 tree route는 seed와 무관하게 결정적이 된다. 다양성은 상대(rollout 사용)와 opening에서 나온다. H6-5에서 같은 대국이 반복되는 비율을 기록한다.

**3. evaluator.**
- `ValueLeaf`: H6-2a checkpoint 로더(`h4_policy.RootPolicy`는 value checkpoint를 거부하므로 따로 만든다). 기대 sha256을 확인하고, D4 평균, eval 모드, `torch.no_grad`로 동작한다.
- policy prior는 기존 H3 `RootPolicy`를 그대로 쓴다. value 망과 trunk·policy head가 같으므로 한 번의 forward로 둘 다 얻을 수 있지만, 이것은 속도 최적화다. 정확성 검증 뒤 별도 단계로 한다(batch 구성이 바뀌면 policy 출력이 비트 단위로 같다는 보장이 깨질 수 있다).
- 캐시: 같은 국면(정규화 hash)의 value를 수 단위로 재사용한다. 결과가 바뀌지 않으므로 넣어도 된다.

**4. 부호 테스트(H6-4 통과 조건, 로드맵 그대로).**
- 즉시 5목, 상대 5목 허용, 1·2수 강제승, 부모·자식 부호, 무승부.
- 추가: `leaf_value`의 부호를 뒤집은 가짜 evaluator로는 1수 강제승을 놓치는지 확인한다(부호 버그가 테스트에 잡히는지 확인).
- 추가: 값이 상수인 evaluator(v ≡ 0)에서는 탐색이 prior만 따르는지 확인한다.

**5. 대국 전 오프라인 점검(H6-4b, 대국 없이).**
- 고정 국면 집합(P92, e1 tree-route 국면, 웹 패배 국면)에서 TB-Engine-v1과 후보를 같은 seed로 돌려 다음을 기록한다.
  - tree route를 타는 수의 비율. value는 tree route에서만 작용하므로, 이 비율이 효과의 상한이다.
  - 착수 변경률, root visit 분포, C4 지표(root Q range, std, |Q| > 0.95 비율), leaf value의 차례 색별 평균.
  - 수당 시간(value 호출 수 × 호출당 시간 vs rollout 시간). CPU와 CUDA 둘 다 잰다. SER5는 CPU뿐이다.
  - P92: 10 seed에서 (3,4)를 고른 횟수(기준선 `puct_policy`의 S1 결과와 비교).
- 이 점검은 판정이 아니라 고장 확인이다. 여기서 설정을 바꾸지 않는다.

**6. H6-5 전에 고칠 사전 규칙.**
- 로드맵의 H6-5 guard "P93·P94 부호 유지"는 그대로 적용하면 v3가 무조건 실패한다(offline 값이 이미 틀림). G4가 fresh probe로 바뀐 것과 맞춰, 대국 전에 다음으로 바꾸는 것을 제안한다: "P92에서 (3,4)를 고른 seed 수가 기준선 이하".
- 주 상대: §12.19의 `v8:puct_policy`는 TB-Engine-v1 이전 기준이다. value 효과만 분리하려면 **TB-Engine-v1 자체**(rollout leaf)가 주 상대여야 한다(0.5 = 동등). 보조 상대는 `v8:full`. seed는 8403, 8404, 예비 8405로 한다(예약됨). 이 선택은 결과 전에 고정한다.
- 시간 guard(≤ 1.5배)는 GPU 기준으로 판정하고, CPU 시간은 따로 기록한다(웹 반영 판단용).

**7. 범위 밖(H6-5 ADOPT 뒤).**
- 웹(TB-RT30-v3) 반영: 시간 제한 탐색이라 시뮬레이션 수가 고정되지 않는다. CPU에서 value 비용을 보고 따로 정한다. RenjuNet으로 학습한 망은 로컬 전용 제약(§12.1)이 있으므로 공개 범위도 같이 정한다.
- raw 단일 추론, policy·value 한 번에 forward, batch leaf 평가 같은 속도 최적화.
- H6-2b(trunk 미세조정), H7.

## H6-4 확정 사항과 H6-4a 구현 (2026-10-18)

위 "H6-4 설계 방향"의 세 질문을 사용자가 모두 승인했다. 아래는 **H6-5 대국 전에** 고정한 규칙이다.

**1. H6-5 guard: P93·P94 → P92 non-regression**
- 로드맵(§12.19)의 "P93·P94 부호 유지"는 **승격 guard에서 뺀다.** 결과표에는 계속 출력하고 `diagnostic_only`로 표시한다.
- 대신 P92 guard를 쓴다: **같은 10 seed**(`run_s1_probes.py`, base seed 9201, c 1.5)에서 후보가 (3,4)를 고른 횟수 ≤ TB-Engine-v1이 고른 횟수.
  - 판정: `PASS` / `FAIL`. TB-Engine-v1이 10 / 10을 고르면 판별력이 없으므로 `NOT_DISCRIMINATING`으로 기록하고 대국 결과를 따른다(통과 증거로 쓰지 않음).
  - non-regression guard일 뿐, H6-4의 성공 증거가 아니다. P92 참값은 UNRESOLVED다.
  - 참고: S1에서 `puct_policy`는 (3,4)를 0 / 10 골랐다. TB-Engine-v1도 0이면 후보도 0이어야 통과한다.
- 나머지 guard는 로드맵 그대로: 안전 위반 0, tree not-SAFE ≤ 기준선 + 0.03, 시간 비율 ≤ 1.5, 흑·백 각 ≥ 0.45.

**2. H6-5 상대와 seed**
- 주 상대: **TB-Engine-v1**(`--opponent v8:puct_policy_vct2_50k --opponent-puct-c 1.5`). leaf만 다르므로 점수 차이는 value leaf의 효과로 읽는다(0.5 = 동등).
- 보조 상대: `v8:full`(역사적 기준선에 대한 회귀 확인). 같은 seed로 TB-Engine-v1 대 `v8:full`도 돌려 쌍 비교와 tree not-SAFE 기준선으로 쓴다.
- 각 25쌍. 공식 seed는 **8403, 8404**다.
- **8405 사용 조건:** 로드맵 표의 INCONCLUSIVE(8403+8404 점수 ≥ 0.55이고 쌍 bootstrap 하한 ≤ 0.5)일 때만, 한 번. 8403에서 점수 < 0.50이면 8404 없이 REJECT, 8403+8404가 0.50–0.55이면 REJECT다. 결과를 본 뒤 다른 이유로 seed를 추가하지 않는다.
- 시간 비율 guard는 value를 CUDA로 돌린 대국으로 판정한다. CPU 비용은 H6-4b에서 따로 기록한다(웹 반영 판단용).

**3. 엔진 버전 단계(승인됨)**
- `src/analysis/puct_v8.py`: 그대로(동결).
- `src/analysis/puct_value_v1.py`(신규): `search_tree_puct_value`. `search_tree_puct`와 행동 집합·prior·선택 규칙·착수 선택이 같고 leaf만 다르다.
  - 종국: 결과 그대로(±1, 무승부 0).
  - 정확한 지름길(value보다 우선): 둘 차례가 5목을 만들 수 있으면 둘 차례 승, 아니면 상대의 5목 자리가 2개 이상이면 둘 차례 패. 상대 5목 자리가 1개면 막을 수 있으므로 value에 맡긴다.
  - 그 외: v = V_D4(s)(둘 차례 관점), 노드 값 = **−v**. 실수 값 backup(`backup`).
  - λ = 1.0: rollout 없음.
- `src/analysis/mcts_v8.py`: `TREE_MODES`에 `"puct_value"`를 더하고 `leaf_value` 인자와 분기, 진단 필드(`v8_value_*`)만 추가했다. `V8_DEFAULTS`의 키는 바꾸지 않았다(새 값은 `tree_mode`의 허용값 하나).
- `src/analysis/mcts_v8_agent.py`: `leaf_value` 인자. `puct_value`일 때만 받고, 그때는 반드시 받는다.
- `src/analysis/tb_v1_value.py`: `TB_V1_VALUE` = TB-Engine-v1 설정에서 `tree_mode`만 `puct_value`. value checkpoint sha256, D4 평균, λ를 포함한 `config_hash`.
- `src/hybrid/h6_value_leaf.py`: `ValueLeaf`. gate를 통과한 v3 checkpoint만 받는다(sha256 확인). D4 평균, 정확한 국면 단위 캐시(결과 불변).
- 회귀: 고치기 **전에** TB-Engine-v1(`tree_mode="puct"`)의 결정값(착수, 경로, root 자식 방문·평균, NN 호출 수, V8-C·VCT2 상태)을 고정 국면 6개에서 기록했다(`tests/fixtures/tb_engine_v1_puct_regression.json`, 커밋 `60239c4`). 고친 뒤 같은지 테스트한다. 시간 값은 비교하지 않는다.

**runner와 점검 도구**
- `run_mcts_v8_benchmark.py`: arm `puct_value_vct2_50k`(c 1.5에서 `TB_V1_VALUE`와 같음). `--value-checkpoint`, `--value-device`. value arm은 상대로 쓸 수 없다. 수마다 `value` 기록(호출 수, 시간, 지름길·종국 leaf, 색별 leaf 평균)과 모든 tree 수의 `root_q`(C4: range, std, |Q| > 0.95 비율). 요약에 `root_q`, `value`, `distinct_games`(rollout이 없어 같은 대국이 반복되는지)를 넣었다.
- `run_s1_probes.py`: value arm 지원. TB-Engine-v1과 후보를 함께 돌리면 요약에 `h6_5_p92_guard`가 나온다.
- `scripts/h6_value_leaf_check.py`(H6-4b): TB-Engine-v1이 실제 대국에서 tree route로 둔 국면(E1-holdout 8417·8418 기록, 학습에 쓰지 않음)을 두 엔진으로 같은 seed에서 탐색한다. tree route 비율, 착수 변경 수, 수당 시간과 비율, root C4, 색별 leaf value 평균을 기록한다. 판정 없음(`verdict: null`).

**테스트** (`tests/test_h6_value_leaf.py`)
- 지름길: 자기 5목 → +1, 상대 5목 하나 → value에 맡김, 상대 열린 4(5목 자리 둘) → −1.
- backup 부호(부모·자식, 무승부 0).
- 탐색 부호: 상대에게 나쁜 leaf로 가는 수를 고른다. 같은 탐색에서 value 부호를 뒤집으면 다른 수를 고른다(부호 버그를 테스트가 잡는지 확인).
- 즉시 승리는 value와 무관하게 고른다. value가 상수면 prior를 따른다. 범위 밖 value는 거부한다.
- TB-Engine-v1 `puct` 경로 회귀 fixture 일치, 후보 설정은 `tree_mode`만 다름, runner arm = `TB_V1_VALUE`.
- `ValueLeaf`: 다른 checkpoint 거부, D4 불변(좌우 반사·회전에서 같은 값), 캐시, 범위.
- runner: value arm은 상대 불가, checkpoint 필수. P92 guard 판정(PASS / FAIL / NOT_DISCRIMINATING).

**H6-4b 실행 명령 (Windows, 대국 없음)**

```powershell
cd "C:\오목 강화학습\renju-h6"
..\renju-alphazero\.venv-cuda\Scripts\Activate.ps1
$env:PYTHONUNBUFFERED = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:PYTHONPATH = "$PWD\src"
git pull --ff-only; git status --short
New-Item -ItemType Directory -Force runs\h6_4 | Out-Null

# 0) 테스트
python -m unittest tests.test_h6_value_leaf tests.test_tb_engine_v1 tests.test_mcts_v8_benchmark

# 1) P92 guard (TB-Engine-v1 vs 후보, 같은 10 seed). H6-5 guard 값으로 그대로 쓴다
python -u scripts\run_s1_probes.py --arms puct_policy_vct2_50k puct_value_vct2_50k --seeds 10 --puct-c 1.5 `
  --policy-checkpoint ..\renju-alphazero\runs\h3_policy_64x4\best.pt --value-checkpoint runs\h6_value_head_v3\best.pt `
  --value-device cuda --output docs\h6\h6_4b_s1_probes.json 2>&1 | Tee-Object runs\h6_4\s1_probes.log

# 2) 고정 국면 점검: CPU(SER5와 같은 조건)와 CUDA
python -u scripts\h6_value_leaf_check.py --games docs\mcts-v8-results\e1h\puct_policy_vct2_50k_8417.json `
  docs\mcts-v8-results\e1h\puct_policy_vct2_50k_8418.json --per-game 3 --workers 6 `
  --policy-checkpoint ..\renju-alphazero\runs\h3_policy_64x4\best.pt --value-checkpoint runs\h6_value_head_v3\best.pt `
  --value-device cpu --output docs\h6\h6_4b_check_cpu.json 2>&1 | Tee-Object runs\h6_4\check_cpu.log
python -u scripts\h6_value_leaf_check.py --games docs\mcts-v8-results\e1h\puct_policy_vct2_50k_8417.json `
  docs\mcts-v8-results\e1h\puct_policy_vct2_50k_8418.json --per-game 3 --workers 6 `
  --policy-checkpoint ..\renju-alphazero\runs\h3_policy_64x4\best.pt --value-checkpoint runs\h6_value_head_v3\best.pt `
  --value-device cuda --output docs\h6\h6_4b_check_cuda.json 2>&1 | Tee-Object runs\h6_4\check_cuda.log
```

- 결과 파일에는 국면(엔진 대국 수순)과 지표만 있고 RenjuNet 데이터는 없으므로 커밋한다.
- H6-4b는 고장 확인이다. 여기서 설정을 바꾸지 않는다. 고장(부호 이상, 예외, 시간 폭증)이 보이면 고친 뒤 다시 돌리고, 그 사실을 기록한다.
- H6-5 대국 명령은 H6-4b 결과를 본 뒤 정리한다(규칙은 위에서 이미 고정).

## H6-4b 결과: 고장 없음, H6-5로 진행 (2026-10-18, 커밋 `7c56033`)

결과 파일: `docs/h6/h6_4b_s1_probes.json`, `h6_4b_check_cpu.json`, `h6_4b_check_cuda.json`. 모두 코드 `b6595d8`, `git_dirty: false`, value sha256 `a0fcccc4…`, D4 평균으로 실행했다. H6-4b는 판정 단계가 아니다(`verdict: null`).

**1. P92 guard (H6-5 공식 기준값으로 고정)**

| 엔진 | (3,4) 선택 | 최종 착수 | 방문한 root 자식 | root 포화 |
|---|---|---|---|---|
| TB-Engine-v1 | 0 / 10 | (1,5) 10 / 10 | 20 | 10 / 10 |
| TB-V1-Value | **0 / 10** | (1,5) 10 / 10 | 2 | 0 / 10 |

- `h6_5_p92_guard = PASS`(0 ≤ 0). 기준선이 10 / 10이 아니므로 `NOT_DISCRIMINATING`도 아니다. 이 값을 H6-5의 P92 guard로 쓴다.
- 내부는 크게 다르다. rollout tree는 모든 자식이 Q = −1(흑 패)로 포화되어 20개에 방문이 흩어졌다. value tree는 50회 중 49회를 (4,4)에 쓰고 Q = +0.82로 봤다. 흑이 실제로 진 국면을 value는 흑에게 크게 유리하다고 본다(P93 ≈ 0, 흑 차례 bias +0.18과 같은 방향).
- 최종 착수가 같은 것은 tree 뒤의 검증 단계가 같은 수를 골랐기 때문이다. value의 확신이 틀린 국면에서 전술 안전망이 실제로 막는 역할을 한다는 사례다. 진단으로만 기록한다.
- P93 (13,11) 10 / 10, P94 stage4 10 / 10(모든 검사 수 UNSAFE)은 두 엔진이 같다(진단).

**2. 고정 국면 298개 (E1-holdout 8417·8418, 100대국 1,890수)**

| 항목 | TB-Engine-v1 | TB-V1-Value |
|---|---|---|
| tree route 비율(원 대국) | 55.2% | — |
| 경로가 달라진 국면 | — | **0 / 298** |
| 최종 착수 변경 | — | **157 / 298 (52.7%)** (tree 착수 158) |
| root Q range / std (평균) | 1.156 / 0.454 | 0.335 / 0.125 |
| root 자식 중 \|Q\| > 0.95 비율 | 39.1% | 1.0% |
| 방문한 root 자식 수 평균 (2개 이하 국면) | 7.93 (14) | 5.39 (29) |
| tree 시간 평균 CPU / CUDA | 1.25 s / 1.18 s | 1.44 s / 0.95 s |
| 수당 시간 평균 CPU / CUDA | 9.98 s / 9.44 s | 10.56 s / 9.47 s |
| 수당 시간 p95 CPU / CUDA | 34.6 s / 31.6 s | **50.9 s / 45.7 s** |
| 전체 시간 비율 CPU / CUDA | — | 1.058 / 1.004 |
| value 호출당 CPU / CUDA | — | 11.7 ms / 2.6 ms |

- **고장 없음.** 경로 차이 0, 예외 0, 회귀 fixture 일치. value는 tree 안에서만 행동을 바꿨다.
- **지름길 계약이 실제로 작동한다.** leaf 16,050개 중 지름길 1,411개, 종국 93개(합 9.4%), 망 호출 14,546개다(초안의 10.3%는 망 호출만 분모로 잡은 값).
- **CPU와 CUDA의 최종 착수는 298개 국면 모두 같았다**(두 엔진 모두, 국면별 대조). root Q 값이 비트 단위로 같은 국면은 47개뿐이다(부동소수점 차이). 착수 수준에서는 장치 무관, 내부 값은 비트 동일이 아니다.
- **색별 leaf 평균:** 흑 차례 +0.280, 백 차례 +0.135(CPU·CUDA 같음). 둘 다 양수다(leaf에서는 둘 차례가 유리하다고 보는 경향). 차이 0.144는 정답이 없는 예측 평균의 차이이지 bias 측정이 아니다. final gate의 흑 bias(+0.177 대 +0.013)와 방향은 같다.

**3. 초안과 다르게 보는 점**
- **시간: 평균은 같지만 꼬리가 45% 늘었다.** p95가 34.6 → 50.9 s(CPU), 31.6 → 45.7 s(CUDA)다.
  - 원인은 tree가 아니다(tree는 CUDA에서 오히려 20% 빠름). **착수가 바뀐 157개 국면에서만** tree 밖 모듈 시간 p95가 32.7 → 48.5 s로 늘었다. 착수가 같은 141개 국면은 31.3 → 31.1 s로 그대로다.
  - tree가 다른 수(또는 다른 root 순서)를 내면 V8-C·VCT2 검증이 더 많은 자식을 검사하게 된다. 최대 사례는 같은 최종 착수에서 16.8 → 138.8 s다.
  - H6-5 시간 guard(평균 비율 ≤ 1.5)에는 영향이 작다. 하지만 시간 제한이 있는 웹(TB-RT30)에서는 꼬리가 중요하므로, 웹 반영은 H6-5 뒤 따로 잰다.
- **52.7%를 전부 value의 정보로 읽을 수는 없다.**
  - rollout 엔진은 seed마다 착수가 달라지는 확률적 탐색이다. 같은 TB-Engine-v1을 다른 seed로 돌렸을 때의 변경률(잡음 바닥)을 아직 모른다. value 엔진은 rollout이 없어 seed와 무관하게 결정적이다.
  - value tree는 Q 차이가 작고(std 0.125 대 0.454) 덜 흔들리므로, c 1.5(rollout 기준으로 보정)에서는 prior를 더 따르고 소수 자식에 집중한다(방문 자식 7.9 → 5.4). 변경의 일부는 "value의 판단"이 아니라 "prior 쪽으로 기운 탐색"일 수 있다.
  - 어느 쪽이든 판정은 H6-5 대국으로 한다. c와 λ는 바꾸지 않는다(사전 규칙).
- **Q 포화 감소(39% → 1%)는 목적에 맞는 변화지만, 연속값이 맞는 값이라는 뜻은 아니다.** P92처럼 확신이 틀린 경우가 있다.

**판정:** H6-4b에서 구조적 고장이나 즉시 멈출 시간 회귀는 없다. H6-5로 간다. λ = 1.0, c 1.5, D4 평균 그대로다.

## H6-5 설계 (2026-10-18, 대국 전 고정)

질문: 같은 TB-Engine-v1 tree에서 rollout leaf를 검증된 v3 + D4 value leaf로 바꾸면 대국에서 더 강한가, 그리고 전술 안전성을 지키는가.

**대국 (seed마다 세 run, 각 25쌍 = 50대국, 같은 opening)**

| run | 엔진 | 상대 | 쓰임 |
|---|---|---|---|
| A (main) | TB-V1-Value (`puct_value_vct2_50k`, c 1.5, value CUDA) | TB-Engine-v1 (`v8:puct_policy_vct2_50k`, c 1.5) | 점수 판정, 시간 비율, 색별 점수 |
| B (aux) | TB-V1-Value | `v8:full` | tree not-SAFE, 안전 위반, 쌍 비교 보고 |
| C (aux) | TB-Engine-v1 (`puct_policy_vct2_50k`, c 1.5) | `v8:full` | B의 기준선 |

- seed: 8403 → (조건부) 8404 → (INCONCLUSIVE일 때만) 8405.

**판정표 (`scripts/summarize_h6_5.py`, 로드맵 §12.19 + H6-4 확정 사항)**

| 단계 | 조건 | 판정 |
|---|---|---|
| 8403 | guard 실패 또는 A 점수 < 0.50 | REJECT |
| 8403 | A 점수 ≥ 0.50 | 8404 진행 |
| 8403+8404 | 점수 ≥ 0.55, 쌍 bootstrap 하한 > 0.5 | **ADOPT** |
| 8403+8404 | 점수 ≥ 0.55, 하한 ≤ 0.5 | INCONCLUSIVE → 8405 한 번, 세 seed에 같은 기준 |
| 8403+8404 | 점수 < 0.55 | REJECT |

- guard(모든 단계): 세 run 모두 안전 위반 0, B의 tree not-SAFE ≤ C + 0.03, A의 시간 비율(후보 / TB-Engine-v1, 같은 대국) ≤ 1.5, A의 흑·백 각 ≥ 0.45, P92 guard ≠ FAIL(H6-4b 값 PASS).
- 보조 상대 쌍 비교(B − C)는 보고만 하고 점수 판정에 쓰지 않는다.
- 보고(판정 무관): `distinct_games`(결정적 탐색이라 같은 대국이 반복되는지), 색별 leaf 평균, root 포화, 방문 자식 수, 수당 시간 p95, tree가 prior 1위를 고른 비율(B 대 C).
- 결과가 나빠도 같은 seed로 λ, c, checkpoint를 바꿔 다시 돌리지 않는다. 바꾼 후보는 새 arm이고 새 seed가 필요하다.

**H6-5 뒤**
- ADOPT: H6 종료. TB-V1-Value를 value 엔진 후보로 기록하고, 웹(TB-RT30) 반영은 CPU 시간 꼬리를 따로 재서 정한다. 엔진 기준선 교체(TB-Engine-v2)는 track-b-post-e1.md §2 규칙대로 H7 주기 경계에서 한다.
- REJECT: 원인 분석(시간 꼬리, 패배 국면에서 value의 확신, prior 쏠림, 색별 차이). 후속 후보(예: c 재보정, value와 rollout 혼합)는 새 arm과 새 seed로만 한다.

**실행 명령 (Windows, 8403 먼저)**

```powershell
cd "C:\오목 강화학습\renju-h6"
..\renju-alphazero\.venv-cuda\Scripts\Activate.ps1
$env:PYTHONUNBUFFERED = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:PYTHONPATH = "$PWD\src"
git pull --ff-only; git status --short
New-Item -ItemType Directory -Force runs\h6_5 | Out-Null
$pol = "..\renju-alphazero\runs\h3_policy_64x4\best.pt"; $val = "runs\h6_value_head_v3\best.pt"; $seed = 8403

# A: TB-V1-Value vs TB-Engine-v1
python -u scripts\run_mcts_v8_benchmark.py --arm puct_value_vct2_50k --puct-c 1.5 --opponent v8:puct_policy_vct2_50k `
  --opponent-puct-c 1.5 --seed $seed --pairs 25 --workers 6 --policy-checkpoint $pol --value-checkpoint $val --value-device cuda `
  --games-jsonl runs\h6_5\value_vs_tb1_$seed.jsonl --output runs\h6_5\value_vs_tb1_$seed.json 2>&1 | Tee-Object runs\h6_5\value_vs_tb1_$seed.log
# B: TB-V1-Value vs v8:full
python -u scripts\run_mcts_v8_benchmark.py --arm puct_value_vct2_50k --puct-c 1.5 --opponent v8:full `
  --seed $seed --pairs 25 --workers 6 --policy-checkpoint $pol --value-checkpoint $val --value-device cuda `
  --games-jsonl runs\h6_5\value_vs_full_$seed.jsonl --output runs\h6_5\value_vs_full_$seed.json 2>&1 | Tee-Object runs\h6_5\value_vs_full_$seed.log
# C: TB-Engine-v1 vs v8:full
python -u scripts\run_mcts_v8_benchmark.py --arm puct_policy_vct2_50k --puct-c 1.5 --opponent v8:full `
  --seed $seed --pairs 25 --workers 6 --policy-checkpoint $pol `
  --games-jsonl runs\h6_5\tb1_vs_full_$seed.jsonl --output runs\h6_5\tb1_vs_full_$seed.json 2>&1 | Tee-Object runs\h6_5\tb1_vs_full_$seed.log

# 판정 (8403 단계)
python scripts\summarize_h6_5.py --main runs\h6_5\value_vs_tb1_8403.json `
  --aux runs\h6_5\value_vs_full_8403.json runs\h6_5\tb1_vs_full_8403.json `
  --p92 docs\h6\h6_4b_s1_probes.json --output docs\h6\h6_5_summary_8403.json
```

- 중단되면 같은 명령을 다시 실행한다. `--games-jsonl`에서 끝난 대국을 건너뛴다.
- `RUN_8404`가 나오면 `$seed = 8404`로 A·B·C를 돌리고, 판정은 두 seed 파일을 모두 넣어 `--output docs\h6\h6_5_summary.json`으로 한다.
- 대국 기록(run 파일)은 엔진끼리의 대국이라 RenjuNet 데이터가 아니므로 `docs/h6/h6_5/`로 옮겨 커밋해도 된다. 요약 파일은 반드시 커밋한다.
- (선택, 판정 무관) 잡음 바닥: `h6_value_leaf_check.py`를 `--seed 20261019 --value-device cuda --output docs\h6\h6_4b_check_seed2.json`으로 한 번 더 돌리면, TB-Engine-v1이 seed만 바꿔 착수를 바꾸는 비율을 알 수 있다(value 엔진은 결정적이므로 같다). 52.7% 해석에 쓴다.

## H6-5 8403 결과: REJECT (2026-10-19, `docs/h6/h6_5_summary_8403.json`, 커밋 `c3ed1f2`)

사전에 고정한 `summarize_h6_5.py`의 판정은 **REJECT**다. 이유는 하나다: B(후보 대 `v8:full`)의 tree not-SAFE **6.85%**(61 / 891)가 C(TB-Engine-v1 대 `v8:full`) **3.62%**(38 / 1,050) + 3.00%p = 6.62%를 넘었다. 규칙대로 이 후보(v3, D4, λ = 1.0, c 1.5)로는 8404를 돌리지 않는다. TB-Engine-v1이 기준 엔진으로 남는다. 판정은 결과를 본 뒤 바꾸지 않는다.

**대국 결과**

| run | W/D/L | 점수 (쌍 95%) | 흑 / 백 | tree not-SAFE | 시간 비율 | 안전 위반 |
|---|---|---|---|---|---|---|
| A 후보 대 TB-Engine-v1 | 26 / 0 / 24 | **0.52** (0.40–0.64) | 0.56 / 0.48 | 9.2% (44 / 477) | **1.378** | 0 |
| B 후보 대 `v8:full` | 41 / 6 / 3 | 0.88 (0.80–0.95) | 0.92 / 0.84 | **6.85%** | 1.232 | 0 |
| C TB-Engine-v1 대 `v8:full` | 36 / 7 / 7 | 0.79 (0.70–0.88) | 0.76 / 0.82 | **3.62%** | 0.868 | 0 |

- P92 guard PASS(H6-4b 값). 시간·색 guard 통과. B − C 쌍 차이 +0.09(95% −0.05 – +0.22, 보고용).
- guard 실패가 없었다면 A 점수 0.52 ≥ 0.50이므로 `RUN_8404`였다.

**초안과 다르게 보는 점 (진단: `scripts/h6_5_diagnose.py` → `docs/h6/h6_5d_8403.json`, 기록된 대국만 사용)**

1. **"0.23%p 근소 초과"도, "후보가 확실히 더 위험"도 이 표본으로는 말할 수 없다.**
   - 한 대국 안의 수들은 독립이 아니다. 같은 opening끼리 대국 단위로 bootstrap하면 차이는 +3.23%p, **95% 구간 −0.5 – +7.3%p**다. 차이가 margin 3%p를 넘는 비율은 52%다. 이 guard는 50대국에서 사실상 동전 던지기에 가까운 판별력을 가졌다.
   - 그래도 판정은 사전 규칙대로 REJECT다. 이 사실은 다음 후보의 guard를 설계할 때 쓴다.
2. **not-SAFE는 판이 꽉 찬 무승부에 몰려 있다.**
   - 무승부는 모두 225수(판이 다 참)다. 그 대국들의 tree 수에서 not-SAFE는 후보 44 / 559(7.9%), 기준선 26 / 673(3.9%)이다.
   - 승패가 난 대국에서는 후보 17 / 332(5.1%), 기준선 12 / 377(3.2%), 차이 +1.9%p다.
   - 한 대국(8403/13/white, 무승부)이 61개 중 18개다. 이 대국을 빼면 후보는 5.4%다.
   - 이것은 사후 부분집합이라 판정을 바꾸는 근거로 쓰지 않는다. 원인 진단의 출발점이다.
3. **증가분은 대부분 UNKNOWN이고, 그 절반은 V8-C 예산 소진이다.**
   - UNKNOWN 55 대 35, UNSAFE 6 대 3.
   - V8-C 예산(400k 노드)을 다 쓴 수가 후보 30개, 기준선 13개이고, 모두 not-SAFE 수 안에 있다.
   - not-SAFE 수는 느린 수와 같은 수다(수당 시간 중앙값 141 s, tree 수 전체는 2.1 s). **H6-4b의 시간 꼬리와 이번 not-SAFE 증가는 같은 현상**(V8-C가 끝내지 못하는 국면)이다. A의 시간 비율 1.378도 같은 원인으로 1.5에 가까워졌다.
4. **안전 관점에서 중요한 수는 tree 수가 아니라 실제로 둔 수다.**
   - not-SAFE tree 수 뒤 V8-C가 고른 수의 상태: 후보 SAFE 26 / UNKNOWN 33 / UNSAFE 2, 기준선 24 / 13 / 1.
   - UNSAFE로 둔 수는 모두 다른 수도 전부 진 국면이었다(안전 위반 0).
   - 결국 차이는 **"증명되지 않은 수를 둔 횟수 33 대 13"**이다.
5. **prior 쏠림은 확인됐지만 not-SAFE의 직접 원인으로 보이지는 않는다.**
   - tree가 prior 1위를 고른 비율: 후보 75.1%, 기준선 65.0%(A 72.3%). H6-4b의 가설(value tree는 prior를 더 따른다)과 맞다. `summarize_h6_5`의 요약은 `puct_value` 수를 빼고 세어 None으로 나왔다. 보고 항목 `tree_prior_top_rate`를 고쳤다(판정 무관).
   - not-SAFE 수만 보면 prior 1위 비율이 후보 55.7%, 기준선 47.4%로 둘 다 평균보다 낮다. prior를 따른 수가 위험한 수였던 것은 아니다.
6. **색별 leaf 평균은 안정된 bias로 읽기 어렵다.** A 흑 +0.350 / 백 +0.152, B 흑 +0.265 / 백 +0.342로 상대에 따라 방향이 바뀐다.
7. **A의 not-SAFE 9.2%는 대부분 진 대국에서 나왔다**(44개 중 38개, 60수 이후 17 / 31). 이미 진 국면에서는 SAFE 수가 없는 것이 정상이다. 기준선(상대 쪽)의 같은 값은 기록되지 않아 비교하지 않는다.

**판정과 해석**
- REJECT는 그대로다. 현재 후보는 승격하지 않는다.
- 근거가 말하는 것: 기력 저하의 증거는 없다(A 0.52, B − C +0.09, 둘 다 구간이 넓다). guard 실패는 "V8-C가 증명하지 못하는 국면(주로 꽉 찬 판의 무승부 종반)에 더 자주 들어가거나 그런 수를 더 자주 고른다"는 쪽이다. 이것이 "value가 실제로 진 수를 고른다"인지 "증명이 어려운 수를 고른다"인지는 아직 모른다.

## H6-5D 설계: 실패 원인 분해 (2026-10-19, 대국 없이)

목표: not-SAFE 증가의 원인을 아래 셋 중 하나로 가른다. 결과를 보기 전에 대응을 정해 둔다.

| 원인 | 뜻 | 다음 후보 |
|---|---|---|
| A. value의 오판 | 같은 국면에서 value tree가 실제로 지는 수를 더 자주 고른다 | 깊은 패배 라벨(`vct_loss`) 보강 → 새 value(H6-2b/H7 데이터) |
| B. 탐색 균형 | 같은 국면에서 고른 수의 상태는 비슷하다. 대신 Q 폭이 좁아 prior·소수 자식에 쏠려 대국 흐름이 달라진다 | c 재보정 또는 λ = 0.5 혼합 중 **하나**를 새 arm으로 |
| C. 증명 예산 | UNKNOWN 대부분이 큰 예산에서 SAFE로 풀린다(어려운 수일 뿐 지는 수가 아님) | value 문제가 아니다. 예산·종반 처리는 두 엔진 공통의 엔진 변경으로 별도 검토하고, guard는 그 결과로 다시 정의 |

**D1 (끝남, 기록 분해):** `docs/h6/h6_5d_8403.json`(위 표의 수치).

**D2 UNKNOWN 해소 (탐색만, 대국 없음):**
- 대상은 B와 C의 not-SAFE tree 수 99개(61 + 38)다. 그 국면에서 tree 수의 상태를 V8-C 예산의 10배(4M 노드)로 다시 판정한다.
- 결과는 SAFE / UNSAFE / 여전히 UNKNOWN 중 하나다. UNSAFE로 풀리는 비율이 후보에서 뚜렷이 높으면 A, 대부분 SAFE면 C다.

**D3 같은 국면 반사실 비교 (대국 없음):**
- 국면은 B와 C의 not-SAFE 국면 99개와, 같은 run에서 고정 seed로 뽑은 SAFE 국면 99개다.
- 두 엔진을 같은 국면·같은 seed로 돌려, 각자 tree 수의 V8-C 상태를 표준 예산으로 기록한다.
- 같은 국면에서 후보의 not-SAFE가 기준선보다 많으면 수 선택 문제(A 또는 B)다. 비슷하면 국면 분포 문제(대국 흐름, B)다.

**공통 규칙**
- 진단은 판정을 바꾸지 않는다. 8403 결과는 REJECT로 남는다.
- 다음 후보는 **하나만**, 새 arm 이름과 새 config hash로 만든다.
- 다음 후보의 평가 seed는 쓰지 않은 **8404, 8405(공식)와 8406(예비)**다. 8406은 이번 커밋에서 `RESERVED_SEEDS`에 넣었다. 8403은 다시 쓰지 않는다.
- not-SAFE guard는 50대국에서 판별력이 낮다(D1). 다음 후보의 guard를 바꿀지(예: 대국 단위 구간 사용, 두 seed 합산 뒤 판정)는 D2·D3 결과로 정하고, **다음 후보의 대국 전에** 문서로 고정한다. 기준선에도 같은 정의를 적용한다.
- D2·D3 도구는 이 설계를 확정한 뒤 구현한다. D2는 국면당 수 분, D3는 국면당 두 엔진 탐색이므로 둘 다 수 시간 규모다.

## H6-3D 결과 (2026-10-15, `docs/h6/h6_3d_diag.json`, 커밋 `c8800c2`, part A 전체 53,531국면, 18초)

**D1 D4 분해**
- 처음부터 끝까지의 흔들림:
  - 국면별 표준편차 평균은 0.136이다(95번째 백분위 0.276, 최대 0.631). 8개 부호가 모두 같은 국면은 56.5%다.
  - **대칭별 흔들림은 고르다.** 원래 value와의 차이가 7개 대칭 모두 0.161–0.177이다. 특정 변환 하나만 튀는 패턴이 아니므로, 변환 버그일 가능성은 다시 한 번 낮아진다.
  - ply가 늘수록 흔들림이 커진다(5–9수 0.084 → 50수 이상 0.179). \|v\|가 0.6 이상이면 오히려 작다(0.099). tanh가 포화되면서 눌리기 때문이다.
- **policy top-1 일치율은 대칭마다 0.78, 8개 모두 같은 비율은 0.54다.** H3 망 전체(trunk + policy head)도 처음부터 끝까지 D4에 일관되지 않다. value head만의 문제가 아니다.
- **"trunk 등변성"과 "value head만의 흔들림" 지표는 진단에 쓸 수 없다(측정 설계 오류, 철회).**
  - 결과: 상대 L2 0.88, cosine 0.61, head만의 표준편차 0.54.
  - 이 두 지표는 trunk feature가 회전해도 채널이 그대로인 스칼라장이라고 가정한다. 그런데 일반 CNN의 방향 선택적 채널(예: 가로선과 세로선 검출)은 회전하면 **서로 다른 채널로 옮겨 가는 것이 정상**이다. 그래서 출력이 완벽한 망이라도 이 값들은 크게 나온다.
  - 실제로 좌우 반사(s = 4)만 값이 작다(L2 0.70, cosine 0.75). 이 해석과 맞는다.
  - 따라서 "trunk 등변성이 나쁨 → H6-2b"라는 판단표는 적용하지 않는다. head만의 흔들림 0.54도 head가 학습 때 본 적 없는 입력(채널 그대로 회전한 feature)에 대한 반응일 뿐이다.
- 결론: D4 흔들림은 고르고 체계적이다. H3 망 전체(policy 출력 포함)가 방향에 민감하다. 이것을 trunk 탓인지 head 탓인지 나누려면 측정이 아니라 실험이 필요하다. value head만 D4 일관성 학습을 했을 때 흔들림이 줄어드는지를 보면 된다.

**D2 엔진 국면의 깊은 패배** (157수, root 9개, child는 모두 서로 다름)

| | 수 단위 | root 평균 | 깊이 0 (78) | 깊이 1 (15) | 깊이 2 (64) | 평균 v |
|---|---|---|---|---|---|---|
| 원래 추론 | 0.924 | 0.907 | 0.923 | 1.000 | 0.906 | +0.54 |
| D4 평균 | 0.968 | 0.971 | 0.962 | 1.000 | 0.969 | +0.53 |

- **엔진 대국 국면에서 "둘 차례가 깊게 이긴다"는 child 단위 인식은 잘 된다.** 깊이 2에서도 0.91(D4 평균 0.97)이다. 흑·백(0.92 / 0.93), 출처(e1 0.92 / e1s 0.93)도 비슷하다.
- 표본은 root 9개뿐이다. 157개를 독립으로 보면 Wilson 95% 구간이 원래 87.1–95.6%, D4 평균 92.8–98.6%다. 실제로는 같은 root의 child끼리 상관되어 있으므로 불확실성이 이보다 크다. "엔진 국면 전반이 분포 밖이라 실패한다"는 가설(H6-2c 근거)은 **지지되지 않는다.** 다만 "엔진 domain 문제가 없다"는 데까지는 말하지 않는다.
- 그렇다면 P93의 실패(같은 종류의 child형 깊은 승리를 −0.36으로 판단)는 그 웹 대국 국면이 특이하기 때문으로 보인다(흑 덩어리, 백 격자). P93과 P94는 **같은 대국 하나**에서 나왔으므로, G4는 사실상 대국 1개에 걸린 판정이다.

**D3 D4 평균 추론** (A 기준, 원래 → 평균)
- 증명 balanced accuracy 0.945 → 0.959, 증명 −1 0.938 → 0.951
- 결과 balanced accuracy 0.640 → 0.650, 결과 MSE 0.918 → 0.896
- 흑 0.639 → 0.641, 백 0.643 → 0.660
- ECE 0.0764 → 0.0764, 결과 편향 +0.153 → +0.153
- D2 0.924 → 0.968
- 정리:
  - 모든 판별 지표가 조금씩 좋아진다. G3는 구조상 해결된다(표준편차 0).
  - **calibration과 편향은 그대로다.** 대칭 평균으로는 상수 편향이 사라지지 않는다.
  - **P93·P94는 평균해도 틀린다**(−0.39 / +0.89). G4는 D4 평균으로 해결되지 않는다.
- **비용(RTX 5070):**
  - 한 국면 지연은 1.00ms → 0.95ms로 사실상 같다(지연이 병목인 구간이라 8개를 한 batch로 돌려도 차이가 없다).
  - batch 처리량은 110,103 → 9,483국면/초로 11.6배 느려진다. 최대 VRAM은 116MB → 609MB다.
  - 이 측정은 미리 만든 plane에 대한 `value_forward`(고정된 trunk + value head)만 잰 microbenchmark다. 게임 인코딩, policy head, evaluator 전체는 포함하지 않는다. 현재 PUCT는 evaluator batch 1로 leaf를 하나씩 평가하므로 이 구조에서는 추가 지연이 관찰되지 않았다. **H6-4에서 실제 evaluator에 통합한 뒤 검색 전체 지연을 다시 잰다.** batch 처리량이 11.6배 떨어진다는 단서는 유지한다.

**`--prove-losses` 해석 정정**
- **고신뢰 증명 라벨(weight 1.0)에 부모 단위의 깊은 패배가 없다.** 결과 라벨(weight 0.1)에는 패배 국면이 −1로 많이 들어 있지만, 약한 지도다. v2 증명 167,757개 가운데 −1은 `forced_loss` 17,058개뿐이고 `vcf_loss`·`vct_loss`는 0개다. 증명만 보면 +1 대 −1이 8.8 : 1이다. 학습 데이터의 증명 −1(`forced_loss`)은 "상대가 다음 수에 오목을 두 군데 이상 만들 수 있다"(또는 막을 자리가 금수)는 가장 얕은 패턴뿐이다. val·test의 증명 −1 정확도 93–95%는 이 패턴을 맞힌 것이다.
- `--prove-losses`가 추가하는 `vcf_loss`는 `solver.decision(game, 0)`이 UNSAFE인 국면이다. 즉 **"내 모든 수 뒤에 상대가 VCF로 이긴다"는 부모 국면 단위 패배 라벨**이다. P94(모든 수 뒤 상대 VCT1 승리)와 **같은 종류이고 깊이만 한 단계 얕다.** 앞서 "`--prove-losses`는 P94형이 아니다"라고 한 해석은 고친다.
- P94 자체는 `--prove-losses`로도 라벨이 나오지 않는다(H6-3A 때 확인). 깊이 1의 부모 단위 패배는 **판정 함수가 이미 있다**(`analysis.tactical_labels.defense_label` → `vct_loss`). 다만 H6 builder에 연결되어 있지 않다. fresh probe 도구(아래)가 이 함수로 P94를 `vct_loss`로 판정하는 것을 확인했다.

**종합**

| 문제 | 근거 | 대응 후보 |
|---|---|---|
| G3 D4 흔들림 | 대칭 전반에 고르게 0.136 | D4 평균 추론(지연 비용 거의 없음). 또는 value head D4 일관성 학습 |
| G4 부모 단위 패배(P94) | 고신뢰 증명 −1이 가장 얕은 패턴뿐. child형 깊은 승리는 잘 인식(D2) | 부모 단위 패배 라벨: `--prove-losses`(VCF). 필요하면 `defense_label`(`vct_loss`)을 builder에 연결 |
| G4 P93(특이 국면) | 엔진 국면 전반은 정상(D2). 대국 1개의 특이 국면 | 새 fresh probe를 여러 대국에서 만들어 최종 판정의 표본을 넓힘 |
| 양수 편향 +0.15 | D4 평균으로 변하지 않음 | 부모 단위 −1 라벨이 늘면 일부 줄어들 수 있음. 관찰 |

## H6-3D 진단 (구현 2026-10-15, dev 데이터만)

구현: `src/hybrid/h6_diag.py`, `scripts/h6_diagnose.py`, `tests/test_h6_diag.py`.
- 쓰는 데이터는 test part A(dev)와 E1·E1-holdout·E1-S manifest뿐이다.
- 스크립트가 분할을 A로 고정하므로 B는 열 수 없다.
- 결과 파일은 지표만 담으므로 커밋한다.

**D1 D4 원인 분해** (part A 전체 53,531국면 × 8개 대칭)
- 처음부터 끝까지(end-to-end): 국면별 8개 value의 표준편차를 종류, 흑백, ply 구간, \|8개 평균\| 구간(0–0.1, 0.1–0.3, 0.3–0.6, 0.6–1)별로 본다. 대칭별 \|v_s − v_원래\| 평균과 8개 부호 일치율도 잰다.
- **trunk 등변성:** 대칭 s마다 T_s⁻¹F(T_s x)와 F(x)의 상대 L2와 cosine을 잰다(F는 trunk feature).
- **value head만의 방향 민감성:** 원래 feature F(x)를 8방향으로 변환해 value head에 넣고 국면별 표준편차를 잰다. 이때 feature는 구조상 정확히 등변이므로, 남는 흔들림은 head 몫이다.
- 보조: policy top-1이 8개 대칭에서 일치하는 비율(원래 좌표로 되돌려 비교). policy head도 D4 불변이 아니므로 trunk 판단에는 쓰지 않는다.

해석(**폐기**: 아래 H6-3D 결과의 측정 설계 오류 참조. 이 표로 판단하지 않는다):

| trunk 등변성 | head만의 표준편차 | 해석 |
|---|---|---|
| 좋음 | 큼 | value head가 주원인 |
| 나쁨 | 작음 | H3 trunk가 주원인 |
| 나쁨 | 큼 | 둘 다 |
| 좋음 | 작음, 그런데 end-to-end만 큼 | 예상 밖. 구현이나 BN을 다시 조사 |

**D2 엔진 국면의 깊은 패배**
- E1·E1-holdout·E1-S의 10M truth가 `PROVEN_LOSS`인 수를 둔 뒤의 국면을 쓴다. 상대 차례이고 상대가 이기므로 value > 0이어야 한다.
- 표본은 **157수인데 root 국면은 9개뿐이다.** child 157개는 D4로 겹치는 것이 없다.
  - 출처: e1 74, e1s 83
  - 깊이: 0이 78, 1이 15, 2가 64
  - child의 둘 차례: 흑 99, 백 58
  - 실질적인 독립 표본은 root 9개 수준이다. 그래서 수 단위 정확도와 함께 **root별 평균 정확도**와 깊이별 정확도를 본다.
- 이것은 child 단위로 깊은 승리를 알아보는지를 시험한다. P94처럼 부모 국면에서 "모든 수가 진다 → v < 0"을 판단하는 능력과는 다르다. 그래서 D2가 좋아도 P94 문제가 해결된다고 결론 내리지 않는다.
- manifest에 `PROVEN_WIN` truth가 없어서 반대 부호 비교군은 없다.

**D3 D4 평균 추론** V_D4(x) = 8개 대칭 value의 평균
- 같은 A와 D2에서 원래 추론과 나란히 비교한다. 비교 항목은 다음과 같다.
  - 증명 balanced accuracy, 증명 −1 정확도
  - 결과 balanced accuracy와 MSE, 흑·백 결과 balanced accuracy
  - **ECE**, 결과 편향
  - D2 정확도(수 단위, child 단위, root 평균)와 평균 value
- P93·P94 값은 dev 참고로만 남긴다.
- **비용:** 한 국면 지연(원래 vs 8개를 한 batch로), batch 256 처리량, 최대 VRAM을 잰다. 클라우드 CPU smoke에서 한 국면 지연은 2.8배, 처리량은 약 13배 느려졌다. GPU 값은 실행 결과로 정한다.

**진단 결과에 따른 갈림**
- ~~trunk 등변성이 나쁘면: H6-2b(trunk 미세조정) 후보.~~ (폐기, 측정이 채널 이동을 반영하지 못함)
- trunk는 괜찮고 head만의 흔들림이 크면: value head만 D4 일관성 학습을 한다(8개 대칭 모두 사용, 대칭 간 value 분산 penalty).
- D4 평균 추론만으로 지표가 충분히 좋아지고 비용을 감당할 수 있으면: D4 평균 추론이 후보가 된다. 바뀐 모델이므로 B에서 검증한다.
- D2도 나쁘면: 엔진 국면의 깊은 증명 데이터로 적응하는 **H6-2c**(H6-2b와 구분). TB-Engine-v1 자기대국에 새 seed를 쓰고, seed를 학습용 / dev probe용 / 최종 fresh probe용으로 미리 나눈다. 예약 seed는 제외한다.
- D2는 좋은데 P94만 계속 틀리면: 부모 국면의 강제 패배 라벨과 그 학습 정의를 따로 설계한다.
- 수정 우선순위: D4 평균 추론(학습 없이 상한 확인) → value head D4 일관성 학습 → pooling 등 head 구조 변경 → H6-2b.

**실행** (깨끗한 worktree, GPU, A 전체 기준 수 분):
```powershell
cd "C:\오목 강화학습\renju-h6"
..\renju-alphazero\.venv-cuda\Scripts\Activate.ps1
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = "$PWD\src"
git status --short
git pull --ff-only
python -u scripts\h6_diagnose.py --checkpoint runs\h6_value_head_v1\best.pt `
  --cache-dir data\tactical\h6\value_cache_v2 --device cuda --output docs\h6\h6_3d_diag.json 2>&1 | Tee-Object runs\h6\h6_3d_diag.log
```

## H6-3 오프라인 gate (확정 2026-10-14, test를 열기 전)

구현: `src/hybrid/h6_gate.py`(기준값 `THRESHOLDS` 상수), `scripts/h6_offline_gate.py`, `tests/test_h6_gate.py`.
**아래 기준은 test를 처음 열기 전에 확정했다. 실행 뒤에는 바꾸지 않는다.**

**test 분할 고정(H6-3A / H6-3B)**
- test 대국(4,996판)을 대국 단위로 둘로 나눈다. 규칙은 `sha256("h6-3-v1:" + game_id)`가 짝수면 A, 홀수면 B다.
- **H6-3A**는 첫 gate에 쓴다. **H6-3B**는 예비로 남겨 둔다.
- A가 통과하면 B는 쓰지 않은 채로 둔다.
- A가 실패한 뒤 모델을 바꾸면(`--prove-losses` v3, H6-2b 등) A는 그때부터 dev 집합이다. 바뀐 모델의 최종 gate는 B로 한다.
- P93·P94도 마찬가지다. 이 probe를 보고 모델을 바꿨다면 그 뒤로는 dev probe다. 바뀐 모델에는 새 깊은 패배 probe가 따로 필요하다.
- 결과 파일에는 분할의 대국 수, 국면 수, 대국 목록 SHA를 남긴다.

| gate | 기준(모두 만족해야 PASS) | 참고(val) |
|---|---|---|
| G1 증명 부호 | 증명 balanced accuracy ≥ **0.90** 그리고 증명 −1 정확도 ≥ **0.85** | 0.942 / 0.933 |
| G2 결과 판별 | 결과 MSE < **1.000** 그리고 결과 balanced accuracy ≥ **0.60** | 0.939 / 0.629 |
| G3 D4 일관성 | 결과 1,000 + 증명 1,000국면(seed 20261014)에서 다음 셋을 모두 만족: 8개 대칭 value의 국면별 표준편차 **평균 ≤ 0.05**, 그 **95번째 백분위 ≤ 0.10**, \|8개 평균\| ≥ 0.10인 국면 중 **8개 부호가 모두 같은 비율 ≥ 0.95** | 처음 측정 |
| G4 probe | **P93**(백 차례): 원래 방향 value > 0 그리고 8개 평균 > 0. **P94**(흑 차례): 원래 방향 < 0 그리고 8개 평균 < 0 | 처음 측정 |
| G5 흑·백 | 결과 국면만으로: 흑 차례 balanced accuracy ≥ **0.55**, 백 차례 ≥ **0.55**, 둘의 차이 ≤ **0.05** | — |

- balanced accuracy = (target +1 국면의 부호 정확도 + target −1 국면의 부호 정확도) / 2. 부호 정확도는 v > 0과 t > 0이 일치하는 비율이다.
- 어떤 값을 계산할 표본이 없으면(예: 증명 −1이 0개) 그 gate는 FAIL로 처리한다.
- G4는 크기 기준을 두지 않는다. 미리 정할 근거가 없기 때문이다. 대신 원래 방향 value, 8개 평균·최소·최대·표준편차를 모두 남긴다. 그래서 0에 가까운 약한 통과인지도 보인다.
- G3에서 8개 부호 일치 비율은 확신 있는 국면(\|평균\| ≥ 0.10)으로만 판정한다. 0 근처 국면은 부호가 쉽게 뒤집히기 때문이다. 전체 국면의 일치 비율은 보고만 한다.

보고만 하는 항목(gate 아님):
- **결과 ECE.** 결과 국면만, p = (v+1)/2 대 y = (t+1)/2, 같은 폭 10구간, weight 없이 계산한다. 이 값으로 λ를 정한다(§12.19 규칙 그대로: ECE ≤ 0.10이면 1.0, 아니면 0.5).
- 편향(평균 v − 평균 t): 전체, 결과, 증명, 흑·백 차례, ply 구간별, 결과의 ply 구간별.
- 편향을 뺀 결과 부호 정확도.
- P92: 흑의 각 합법수 뒤 국면의 value 순위. 실제로 진 수 (3,4)의 순위, 그리고 상위 5수.

**실패했을 때의 경로**

| 실패 | 해석 | 다음 행동 |
|---|---|---|
| G1 | 얕은 증명이나 −1 일반화 문제 | `--prove-losses`, −1 라벨 구성 검토 |
| G2 | 고정 trunk의 장기 승패 표현 부족 가능성 | H6-2b 검토 |
| G3 | D4 표현이나 추론 문제 | D4 변환·encoder·BN·증강 점검. 바로 재학습하지 않음 |
| P94만 실패(G1–G3 통과) | 깊은 패배 표현 부족 | `--prove-losses` v3 우선 |
| P94와 G1–G3 중 일부가 함께 실패 | — | 그 gate부터 해결. P94만으로 `--prove-losses`를 결정하지 않음 |
| P93 실패 | 승리 쪽 부호나 probe 특유의 문제 | 원인 조사. `--prove-losses`로 바로 가지 않음 |
| G5 | 색에 따라 일반화가 다른 문제 | 흑·백별 편향을 종류·ply별로 분석한 뒤 weight나 학습 수정 |

결과 파일의 `next_steps_if_failed`에 실패한 gate별 경로가 적힌다.

**실행(한 번만, 깨끗한 worktree)**
```powershell
cd "C:\오목 강화학습\renju-h6"
..\renju-alphazero\.venv-cuda\Scripts\Activate.ps1
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = "$PWD\src"
git status --short
git pull --ff-only
python -u scripts\h6_offline_gate.py --checkpoint runs\h6_value_head_v1\best.pt `
  --cache-dir data\tactical\h6\value_cache_v2 --part A --device cuda --output docs\h6\h6_3a_gate.json 2>&1 | Tee-Object runs\h6\h6_3a_gate.log
```
- 실행이 거부되는 경우: 같은 출력 파일이 이미 있을 때, worktree가 dirty일 때, checkpoint가 H6-2a가 아닐 때, 캐시 라벨이 학습에 쓴 라벨과 다를 때.
- 결과 파일(`docs/h6/h6_3a_gate.json`)에는 국면 없이 지표와 해시만 들어 있으므로 커밋한다.
- part B는 A가 실패하고 모델을 바꾼 뒤에만 실행한다.

## H6-2a: H3 망을 고정하고 value head만 학습 (2026-10-12)

H3 policy 망의 trunk와 policy head를 그대로 두고 value head만 H6-1 라벨로 학습한다. 엔진 파일(`tb_engine_v1.py`, `mcts_v8.py`, `e3_arms.py`, `s3_vct2_v1.py`, `puct_v8.py`, `threats.py`)은 건드리지 않았고, PUCT leaf 통합은 H6-4에서 한다.

### 주의사항 검토 결과

사용자가 준 주의사항을 코드와 대조해 확인했다. 모두 사실이었고 전부 반영했다. 보완한 점은 각 항목의 "보완" 열에 적었다.

| 주의사항 | 확인 결과 | 보완 |
|---|---|---|
| H3 `best.pt`(64×4)를 strict `load_checkpoint`로 읽는다 | 맞다. `load_checkpoint`는 설정·encoder·action index 버전이 다르거나 키가 하나라도 다르면 거부한다 | `best.json`이 `stage H3`, `policy_trained true`, `value_trained false`, 64×4인지도 검사한다 |
| trunk와 policy head의 BatchNorm 통계까지 고정한다(`model.eval()`, `value_head.train()`만, `model.train()` 금지) | 맞다. trunk의 BN 9개와 policy head의 BN 1개는 train 모드에서 forward만 해도 running 통계가 바뀐다 | value를 계산하는 함수가 trunk나 policy head가 train 모드이면 오류를 낸다. 고정 부분(파라미터와 BN 버퍼)의 해시를 시작할 때 저장하고, `best.pt`·`last.pt`를 쓰기 전에 같은지 다시 확인한다 |
| optimizer에는 value head 파라미터만 넣고, trunk는 `no_grad` | 맞다 | 나머지 파라미터는 `requires_grad False`로 둔다 |
| `value_mask False`인 행만 쓴다 | 맞다 | mask된 행은 **캐시를 만들 때 버린다**(분할별 개수는 manifest에 남긴다). 그래서 trainer는 그 행을 볼 수 없다. 남는 행은 weight > 0이고 target이 ±1 또는 0인지 검사한다 |
| loss는 Σw·l/Σw(증명 1.0, 결과 0.1) | 맞다. l은 tanh 출력의 제곱 오차다 | 행 하나의 비율은 10:1이지만 **전체 비중은 행 개수에 달려 있다.** 캐시 manifest와 학습 시작 출력에 kind별 weight 비중(`weight_share`)을 보고한다 |
| 둘 차례 관점, 부호를 뒤집지 않는다 | 맞다. encoder가 상대 좌표다(plane 0 = 둘 차례 돌, plane 1 = 상대 돌, plane 3 = 둘 차례가 흑인지). H6-1 target도 둘 차례 관점이다 | target을 그대로 복사한다. 흑·백 차례 국면 모두 행 값과 같다는 테스트가 있다 |
| 저장된 split 필드를 쓴다(train은 gradient, val은 선택, test는 손대지 않음) | 맞다. H6-1 행은 H2 대회 단위 분할을 그대로 쓰고, H3도 같은 분할로 학습했다. 그래서 val 국면은 trunk도 본 적이 없다 | trainer는 `test.pt`를 아예 열지 않는다. 테스트에서는 `test.pt`를 지운 상태로 학습한다 |
| D4 증강, value target은 바꾸지 않는다 | 맞다 | H3의 `d4_tables`를 다시 쓰고, 여섯 plane(합법수 plane 포함)을 함께 변환한다. 금수 규칙이 D4에 불변이라는 사실은 H3 `verify_d4`로 확인돼 있다. 변환한 plane이 변환한 대국의 `encode_game`과 같다는 테스트가 있다 |
| `best.pt` + `best.json` provenance | 맞다 | 해시를 남긴다: H3 checkpoint SHA는 원본 바이트로, H3 `best.json`과 H6-1 보고서 SHA는 CRLF를 LF로 바꿔서 계산한다(Windows 커밋 대비). 라벨 행 SHA는 gzip을 푼 내용으로 계산한다(gzip 헤더 시간의 영향 제거). 고정 부분 해시와 캐시 요약도 넣는다 |
| 별도 원자적 `snapshot.pt`, resume은 H3 SHA·라벨 SHA·설정·seed·batch·optimizer가 바뀌면 거부 | 맞다 | 보고서 SHA, 캐시 SHA, 고정 부분 해시도 비교한다. 학습 결과를 바꾸는 설정 12개를 비교한다(seed, 구조, batch, epochs, max_steps, lr, wd, warmup, min_lr, D4, optimizer) |
| 깨끗한 커밋 상태에서만 학습 | 맞다 | `require_clean_git: true`이면 git이 dirty이거나 확인할 수 없을 때 시작하지 않는다. 판단 기준은 `save_checkpoint`의 `git_dirty`와 같다(추적 중인 파일만 본다). `--allow-dirty`는 smoke 전용이고 `best.json`에 기록된다 |

코드를 보다가 추가로 확인한 사실:
- value head에도 BatchNorm(채널 1개)이 있다. 학습하는 부분이라 train 모드에서 통계가 갱신되는 것이 맞다. 검증과 이후 사용은 eval 모드(running 통계)로 한다.
- H3는 value head를 한 번도 실행하지 않았다. 그래서 value head는 H3 checkpoint에 저장된 초기값(H3 seed로 정해짐)에서 시작한다.
- `h4_policy.RootPolicy`는 `value_trained != false`인 checkpoint를 거부한다. 따라서 H6-2a checkpoint는 H4 root policy로 쓸 수 없다(의도된 동작이다). policy 출력은 H3와 비트 단위로 같으므로, H6-4에서 value leaf용 로더를 따로 만든다.
- **증명 라벨은 +1에 크게 치우친다.** VCF 증명은 둘 차례의 승리이고, −1은 `forced_loss`(옵션 `vcf_loss`)뿐이다. 합성 smoke에서는 val 증명 행의 비율이 +1 155 : −1 9였다. H6-1 실제 보고서에서 비율을 확인하고, H6-3에서 −1 쪽(P94 < 0 포함)을 따로 본다.
- 선택 기준은 val의 weighted MSE(학습 목표와 같음)다. 증명·결과별 MSE, 부호 정확도, 포화율, 흑·백별, ply 구간별 지표는 보고용으로만 남긴다. 오프라인 gate는 H6-3이 test로 판정한다.

### 파일

| 파일 | 내용 |
|---|---|
| `src/hybrid/h6_value_cache.py` | H6-1 행 → bit-packed value 캐시. 라벨 있는 행만 남기고 합법수 mask는 한 번만 계산한다. 대국 행이 연속인지, moves가 prefix인지, split이 하나인지, 보고서의 `labelled` 개수가 맞는지 검사한다. 스트리밍 방식이라 메모리 사용이 제한된다 |
| `scripts/build_h6_value_cache.py` | 캐시 CLI. 출력은 `data/`나 `runs/` 아래에만 쓰고, 기존 캐시가 있으면 덮어쓰지 않는다 |
| `src/hybrid/h6_value_train.py` | trainer(고정, weighted MSE, D4, val 선택, snapshot·resume, provenance, non-finite 오류) |
| `scripts/train_h6_value.py` | 학습 CLI(`--resume`, `--allow-dirty`, override) |
| `configs/hybrid/h6_value_head.yaml` | 본 실행 설정(cuda, batch 1024, 3 epoch, lr 1e-3, warmup 500). 첫 실행 계획값이며 튜닝하지 않은 값이다 |
| `tests/test_h6_value.py` | 16개 테스트(아래) |

테스트 항목:
- 불러오기 검사: H3 strict load(키 누락 거부), 부모 metadata(`value_trained true` 거부), 구조 불일치 거부
- 고정 검사: 학습 가능한 파라미터가 value head뿐인지, optimizer가 value head만 갖는지, 학습 뒤 trunk·policy의 파라미터와 BN 버퍼가 H3와 비트 단위로 같은지, value head 값은 바뀌었는지
- 출력 검사: `best.pt`의 policy logits가 H3와 비트 단위로 같고 value는 다른지
- 캐시 검사: mask된 행이 제외되는지, 10:1 weighting, 흑·백 관점 부호, plane = `encode_game`, D4 변환과 target 불변, 보고서 개수 불일치와 연속되지 않은 대국 거부
- 학습 환경 검사: `test.pt` 없이 학습되는지, provenance 필드, dirty worktree 거부, 보고서 불일치 거부
- resume 검사: 중단 후 resume한 결과가 한 번에 학습한 결과와 같은지(비트 단위), batch·seed·H3 변경 거부, NaN loss 오류

클라우드 컨테이너(CPU torch 2.14)에서 실행한 결과:
- `test_h6_value`, `test_h3`, `test_h6_labels`, `test_model_checkpoint` 40개 테스트가 통과했다.
- 합성 RenjuNet 120판으로 `build_h6_labels` → `build_h6_value_cache` → `train_h6_value`(40 step, CPU 826 samples/s)를 끝까지 실행했다. 같은 out-dir 재실행 거부, `--resume`, `git_dirty false` 기록까지 확인했다.

### 실행 명령 (데스크톱, `renju-h6` worktree)

**순서:** probe audit → H6-1 v2 라벨 → 보고서 커밋 → value 캐시 → (gate 판정) → smoke → 본 학습. 캐시와 학습은 v2 보고서 해시를 확인한다.

```powershell
cd "C:\오목 강화학습\renju-h6"
..\renju-alphazero\.venv-cpu\Scripts\Activate.ps1
$env:PYTHONUNBUFFERED = "1"; $env:PYTHONIOENCODING = "utf-8"
git status --short ; git pull --ff-only      # 깨끗한 상태에서 최신 코드
$GAMES = "..\renju-alphazero\data\external\renjunet\games.jsonl.gz"

# A) probe audit (증명 없음, 몇 분). 개수만 들어 있어 커밋해도 된다
python -u scripts\h6_probe_audit.py --games $GAMES --workers 12 --output docs\h6\h6_probe_audit.json 2>&1 | Tee-Object runs\h6\probe_audit.log
git add docs\h6\h6_probe_audit.json ; git commit -m "H6-1 probe audit" ; git push origin feat/h6-value

# B) H6-1 v2 라벨 (v1보다 대국이 ~45% 많아 시간도 그만큼 더 걸린다. 1분마다 진행률)
python -u scripts\build_h6_labels.py --games $GAMES --workers 12 --probe-game-min-ply 4 `
  --output data\tactical\h6\renjunet_labels_v2.jsonl.gz --report runs\h6\h6_1_report_v2.json 2>&1 | Tee-Object runs\h6\h6_1_v2.log

# C) 끝나면(실행 중에는 pull하지 않는다): 보고서 커밋
$REPORT = Get-Content runs\h6\h6_1_report_v2.json -Raw | ConvertFrom-Json
$REPORT.git_commit ; git rev-parse HEAD      # 같아야 함
git pull --ff-only
Copy-Item runs\h6\h6_1_report_v2.json docs\h6\
git add docs\h6\h6_1_report_v2.json ; git commit -m "H6-1 v2 label report" ; git push origin feat/h6-value
```

```powershell
$H3 = "..\renju-alphazero\runs\h3_policy_64x4\best.pt"
Test-Path $H3, ($H3 -replace 'best.pt$','best.json'), data\tactical\h6\renjunet_labels_v2.jsonl.gz, docs\h6\h6_1_report_v2.json

# 1) value 캐시 (CPU, 1회)
..\renju-alphazero\.venv-cpu\Scripts\Activate.ps1
python -u scripts\build_h6_value_cache.py --labels data\tactical\h6\renjunet_labels_v2.jsonl.gz `
  --report docs\h6\h6_1_report_v2.json --out-dir data\tactical\h6\value_cache_v2 --workers 12 2>&1 | Tee-Object runs\h6\value_cache.log

# 2) smoke (GPU, 300 step, 결과는 쓰지 않음)
..\renju-alphazero\.venv-cuda\Scripts\Activate.ps1
python -u scripts\train_h6_value.py --config configs\hybrid\h6_value_head.yaml --h3-checkpoint $H3 `
  --out-dir runs\h6\value_smoke --max-steps 300 --eval-every 100 --eval-max-states 50000 2>&1 | Tee-Object runs\h6\value_smoke.log

# 3) 본 실행
python -u scripts\train_h6_value.py --config configs\hybrid\h6_value_head.yaml --h3-checkpoint $H3 2>&1 | Tee-Object runs\h6\value_v1.log
# 중단되면 같은 인자에 --resume
```

- **진행률.** 1시간 이상 걸릴 수 있는 작업은 1분마다 진행률을 출력한다(`src/hybrid/progress.py`, stdout).
  - `build_h6_labels.py`: `[proofs] 완료/전체 positions  %  경과  ETA  chunk 수`. 이어서 행 생성, 교차 mask·누출 검사, 파일 쓰기 단계마다 한 줄씩 출력한다.
    - 16판 단위 chunk를 끝나는 순서대로 받는다(`run_chunks`). 결과는 입력 순서로 되돌려 놓으므로 출력 행은 이전과 같다(합성 120판에서 `git_commit` 외 동일 확인).
    - 1분 동안 끝난 chunk가 없어도 `... pending` heartbeat 줄을 찍는다. 처음 버전(`5ce7b77`)은 순서대로 받는 `pool.map`이라, 앞쪽 chunk가 느리면 그동안 아무것도 출력하지 않았다.
  - `build_h6_value_cache.py`: `[value cache] 완료/전체`. 전체 개수는 보고서의 `labelled` 값이다. 캐시 순서가 해시에 들어가므로 chunk는 순서대로 받고, 가장 오래된 chunk를 기다리는 동안 1분마다 heartbeat를 찍는다.
  - `train_h6_value.py`: step 로그에 %, 경과 시간, ETA가 붙는다.
- 캐시 출력에서 확인할 것: 분할별 `states`와 `masked_rows_dropped`, 그리고 `by_kind.proof.weight_share`(증명의 실제 비중)와 증명 행의 +1/−1 개수.
- 결과 전달: `runs\h6_value_head_v1\best.json`과 `metrics.jsonl`, 캐시 `manifest.json`을 `docs\h6\`에 복사해 커밋한다. 셋 다 해시와 지표만 들어 있다. `best.pt`와 캐시는 RenjuNet에서 나온 결과물이라 **로컬에만 둔다.**

## 다음 (TB-Engine-v1 동결 뒤)

H6-1 라벨 생성(RenjuNet train/val/test, VCF 증명 + 결과) → 출처·깊이·부호·색별 보고 → H6-2a(trunk 고정, value head만 학습, **구현 완료**) → H6-3 오프라인 gate → λ 고정 → H6-4 → H6-5(seed 8403/8404, 주 상대 `v8:puct_policy`).
