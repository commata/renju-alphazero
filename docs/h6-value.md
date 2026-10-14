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
