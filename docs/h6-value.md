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

## 다음 (TB-Engine-v1 동결 뒤)

H6-1 라벨 생성(RenjuNet train/val/test, VCF 증명 + 결과) → 출처·깊이·부호·색별 보고 → H6-2a(trunk 고정, value head만 학습) → H6-3 오프라인 gate → λ 고정 → H6-4 → H6-5(seed 8403/8404, 주 상대 `v8:puct_policy`).
