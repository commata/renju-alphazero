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

조치(`build_h6_labels.py --probe-game-min-ply N`, 기본 0 = 이번 v1 동작):
- 돌이 N개 이상인 probe에 닿은 대국만 통째로 뺀다.
- 그보다 이른 probe는 **그 국면 행만 mask한다**(`probe_position`).
- 사유별 제외 수를 보고서에 남긴다(`dropped.probe_game_by_ply`).
- **권장 N = 9.**
  - 8수 이하는 opening 단계다. 해당 probe는 E1-C 대조군 국면과 H3 policy gate의 `must_defend_open3` 일부다.
  - 전술 probe(VCT 12수 이상, stage7 12수 이상, P92–P94, E1-P/N)는 모두 통째 제외가 유지된다.
- `scripts/h6_probe_audit.py`: 증명 없이 몇 분 만에, probe별로 닿는 대국 수와 N별 제외 대국 수를 센다. 개수만 들어 있으므로 커밋해도 된다.

결정: **H6-1 gate는 v1 라벨로 통과시키지 않는다.** audit로 N을 확인하고, `--probe-game-min-ply 9`로 v2 라벨을 다시 만든 뒤 H6-2a를 진행한다. H6-2a 설정(`h6_value_head.yaml`)은 v2 경로를 가리킨다.

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
python -u scripts\build_h6_labels.py --games $GAMES --workers 12 --probe-game-min-ply 9 `
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
