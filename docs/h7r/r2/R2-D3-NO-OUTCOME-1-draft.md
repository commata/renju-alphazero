# R2-D3-NO-OUTCOME-1 — cycle outcome 기여 제거 대조 계약 초안

- 작성일 2026-11-03. 상태: **draft / 미승인 / 실행 불가**
- 상위 설계: `R2-DIAG-1-draft.md` §7(판독 규칙, ε, `DEVELOPMENT_SIGNAL`)과 §8(D3 행)
- 사전 구성 확인 도구: `scripts/h7r_d3_prep.py`(읽기 전용, CPU), 계산 `src/hybrid/h7r_d3.py`, 테스트 `tests/test_h7r_d3.py`
- 데스크톱 명령: `docs/h7-rl-plan.md` §9.26

## 0. 성격과 바꾸지 않는 것

- **진단이다.** cycle outcome 행의 학습 기여를 없앴을 때 무엇을 얻고 무엇을 잃는지 측정한다.
- label 잡음을 원인으로 확정하는 실험이 아니다. outcome이 해롭다는 근거도 아니고, D1·D2를 배제하지도 않는다.
- 기존 모델 비교(R2-0A/0B, D4, X1)로 이 ablation의 결과를 미리 확정하지 않는다. outcome 성분이 나빠질 수 있다는 것은 가능한 예상일 뿐이다. 학습 간섭이 줄어 다른 결과가 나올 수도 있다.
- 바꾸지 않는 것
  - D4 판독(`NOT_ESTABLISHED`), c5 유지, V9 보류
  - R2의 주 지표, ε, `DEVELOPMENT_SIGNAL` 규칙, D4 진입 조건
- proof 전용 판독과 D4 진입 규칙 변경은 이 계약에 넣지 않는다. 이 진단에서 무엇을 배웠는지 본 뒤 별도 연구 질문으로 정한다.

## 1. 질문

같은 parent·자료·학습 예산·seed에서 cycle outcome 행의 손실 기여만 0으로 하면, H-dev의 기존 주 지표와 그 구성(outcome / proof, ply, 차례)이 같은 seed의 기준 recipe 대비 어떻게 달라지는가?

## 2. 조건과 직접 control

| 항목 | 기준 recipe(직접 control) | D3 조건 |
|---|---|---|
| parent | c5 | 같음 |
| 자료 | c8 cycle 자료(self-play seed 8808)와 split, H6 replay | 같음 |
| config | c8 config(batch 1024, 3 epoch, lr 2e-4, D4 augmentation, 평가 10회, H6 guard 1.05) | 같음 |
| 학습 seed | 31001 / 31002 / 31003 | 같음 |
| cycle outcome 행 | 가중치 0.1로 학습 | **손실 기여 0** |
| cycle proof 행, H6 replay | 학습 | 같음 |
| checkpoint 선택 | 기존 T-val objective와 H6 guard | **같음**(아래 2.2) |

- 직접 control은 R2-0B의 같은 seed 실행 세 개다(`docs/h7r/r2/R2-DIAG-1/r2_0b/runs/seed-3100x/`). 이미 학습돼 있으므로 다시 학습하지 않는다. 기존 기록과의 호환성(아래 2.3)이 확인될 때만 control로 쓴다.
- 새 self-play는 만들지 않는다.

### 2.1 손실

    batch loss = Σ_i k_i w_i (v_i − t_i)² / Σ_i w_i
    k_i = 0 (cycle outcome 행), 1 (cycle proof 행, replay 행)

- 분모는 batch의 **원래 가중치 합**이다. 제거된 기여를 proof나 replay 행에 재분배하지 않는다.
- 이 손실 형태는 기존 H7C-NMASK-1 경로(`h7c_train.masked_wmse`)와 같다.
- cycle 쪽 kept weight가 0인 batch는 cycle 항 0, replay 항 유지로 처리한다. 그 수를 기록한다(기존 `zero_cycle_weight_batches`).
- 실제 gradient 기여량이 함께 줄어드는 처방이다. label 잡음만 분리한 효과라고 부르지 않는다.
- 행 순서·replay 추출·augmentation은 k와 무관하다. 그래서 같은 seed의 자료 배정 지문(`h7r_r2.data_assignment`)이 control과 같아야 한다. 도구로 확인한다.

### 2.2 checkpoint 선택

- control과 **같은** 선택 규칙을 쓴다.
  - objective = (cycle val 가중 MSE, outcome 0.1·proof 1.0 그대로) 와 (H6 val 가중 MSE)의 평균
  - eligible = H6 val 가중 MSE ≤ 1.05 × parent
  - best = eligible 평가 중 objective가 가장 낮은 것
- 기존 NMASK 경로는 선택 지표도 masked cycle val로 바꾼다. D3은 그렇게 하지 않는다. 학습 마스킹으로 선택 기준까지 바꾸면 두 변수가 함께 움직이기 때문이다.
- masked cycle val 값은 측정으로만 함께 기록한다. 선택에는 쓰지 않는다.
- 결과: D3 모델은 학습하지 않은 outcome 행의 val 손실을 일부 포함한 objective로 선택된다. 이 점을 결과 해석에 적는다.

### 2.3 구현과 호환성 (확정 전 할 일)

- trainer의 선택 지표 옵션과 기존 경로의 동치성 검증은 §2.4에 기록했다. 같은 seed의 기존 control을 사용하기 위한 기준 recipe 재현 점검은 §2.5의 D3-R0으로 수행한다.
- control 호환성 검사
  - parent·cycle·replay labels와 cache split SHA, config(학습 seed 외 동일), 학습 행 수와 step 수
  - 자료 배정 지문이 같은 seed의 control과 같은지
  - control의 기록 SHA(`r2_train.json`, `record.json`, `config.json`)는 이미 D4 계약 sources에 고정돼 있다.
- 실행 시간 한도, 출력 경로, 실패 처리(NO_CANDIDATE 보존)를 계약 JSON에 적는다.

### 2.4 trainer 선택 옵션 구현 (2026-11-05)

- `h7c_train.train_value(..., cycle_keep=..., cycle_selection='original')`
  - 학습 손실은 기존 `masked_wmse`(원래 분모)다. kept weight가 0인 batch는 cycle 항 0, replay 항 유지다.
  - 선택 objective는 원래 cycle val WMSE와 H6 val WMSE의 평균이다. H6 guard, 평가 시점, 동률(먼저 나온 평가 유지), `NO_CANDIDATE`는 기존 trainer를 따른다.
  - 기록: `cycle_wmse`(선택에 쓴 원래 지표)와 `cycle_wmse_masked`(진단값)를 parent와 각 평가에 나란히 남기고, `record['selection']`에 선택 규칙을 적는다. `mask` 기록은 NMASK와 같다.
  - 옵션은 난수, 자료 순서, replay 추출, augmentation, step, optimizer에 닿지 않는다. 학습 궤적은 같은 mask의 NMASK와 같다(테스트로 확인).
  - `cycle_keep` 없이 쓰거나 다른 값을 주면 거부한다.
- 옵션을 생략한 경로: 일반 학습과 NMASK 모두, 변경 전 trainer(`3066dba`, SHA `529cffb8…`)와 CPU fixture에서 기록과 선택 state가 비트 단위로 같다(`tests/test_h7c_train_selection.py`).
- trainer SHA는 바뀌었다. R2-0B 기록에 남은 `529cffb8…`은 그 실행의 출처 기록으로 그대로 둔다.
- outcome 마스크의 생성과 전달(실행 도구)은 아직 없다. 계약 확정 뒤 별도로 구현한다.
- 실제 자료 재현 점검은 §2.5를 따른다.

### 2.5 D3-R0 재현 점검 도구 (2026-11-05, 미실행)

- 목적: D3 실행 전에 현재 trainer와 환경이 control을 만든 R2-0B 조건을 재현하는지 확인한다. control은 다시 학습하지 않으므로 이 점검이 짝 비교의 전제다.
- 도구 `scripts/h7r_d3_r0.py`, 테스트 `tests/test_h7r_d3_r0.py`
  - `train`: 기존 `h7r_r2_seed.train`으로 c8 recipe를 seed 20261025에서 한 번 학습한다. 마스크와 선택 옵션은 쓰지 않는다.
  - `check`: 기존 `h7r_r2_seed.reproduce`로 기준 c8 실행과 비교한다. A1 판정 함수 `reproduction_verdict`와 A1 허용 오차(`max_abs_prediction ≤ 0.001`, `abs_primary ≤ 0.0002`)를 그대로 쓴다.
- **과거 기록과의 직접 대조**: 현재 코드가 양쪽을 함께 다르게 계산해 일치로 보이는 경우를 막는다.
  - 재실행의 자료 배정 지문 = 기록된 R2-0B 지문 `e142e971a29da275a6edc908baca816a66d5472061ecf46806e9c17f649f7224`
  - 현재 코드가 재구성한 기준 지문 = 같은 기록값
  - 재실행의 선택 step = 기록된 `147`
  - 고정값 자체는 R2-0B 재현 기록, seed 20261025 실행 기록, c8 원 기록(step)과 대조해 확인한다.
- 판정: A1 판정이 `BITWISE_REPRODUCED` 또는 `METRIC_ONLY`이고 직접 대조가 모두 일치할 때만 `PASS`다. 하나라도 다르면 `NOT_REPRODUCED`(종료 코드 1)로 기록하고 D3는 시작하지 않는다.
- [R0 계약](R2-D3-NO-OUTCOME-1/r0-contract.json)에 R2 계약과 A1, trainer 소스 네 파일의 SHA, 과거 기록과 그 값, A1 허용 오차, seed, 출력 경로를 고정했다. 상태는 `fixed`이고 실행 승인 플래그는 `false`다. 도구는 승인된 계약이 커밋된 경우 실행한다.
- 환경(torch, CUDA, GPU)은 R2-0B 기록과 비교해 함께 남긴다. 다르면 사실로 기록하되 판정 조건에는 넣지 않는다.

데스크톱에서는 승인된 계약과 깨끗한 checkout에서 아래 순서로 실행한다. 입력 위치는 아래 명령에 적고, 내용은 R2·A1이 고정한 c8 기록의 입력 해시와 대조한다. `train`의 기록 형식은 R2-0B를 따르므로, 같은 HEAD에서 `check`까지 이어서 수행한다.

```bash
(
  set -euo pipefail
  cd ~/renju-h6
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src:$PWD/scripts"

  D=docs/h7r/r2/R2-D3-NO-OUTCOME-1
  R=runs/h7r/R2-D3-NO-OUTCOME-1/r0
  test -z "$(git status --porcelain --untracked-files=no)"
  r0_head="$(git rev-parse HEAD)"
  python - "$D/r0-contract.json" <<'PY'
import sys
from scripts.h7r_d3_r0 import load_r0
load_r0(sys.argv[1])
PY
  python scripts/check_track_b_frozen.py
  test ! -e "$R/seed-20261025"
  test ! -e "$R/reproduce.json"
  test ! -e "$D/r0/r0.json"
  mkdir -p "$R/logs"

  train_log="$(mktemp "$R/logs/train-XXXXXXXX.log")"
  python -u scripts/h7r_d3_r0.py train --r0-contract "$D/r0-contract.json" \
    --cycle-cache runs/h7c/c8/value_cache --cycle-rows runs/h7c/c8/value_rows.jsonl.gz \
    --replay-cache data/tactical/h6/value_cache_v3 --parent runs/h7c/c5/value/best.pt \
    --device cuda 2>&1 | tee "$train_log"

  test "$(git rev-parse HEAD)" = "$r0_head"
  check_log="$(mktemp "$R/logs/check-XXXXXXXX.log")"
  python -u scripts/h7r_d3_r0.py check --r0-contract "$D/r0-contract.json" \
    --cache runs/h7r/R1-HDEV-1/cache \
    --reference-model runs/h7c/c8/value/best.pt,runs/h7c/c8/value/best.json \
    --device cuda 2>&1 | tee "$check_log"
)
```

결과 판정은 `r0/r0.json`에, A1 비교 원본과 단계별 로그는 `runs/`에 남긴다. 학습이 `NO_CANDIDATE`를 반환하거나 입력 오류가 발생하면 해당 단계의 비정상 종료 코드로 멈춘다. 결과 JSON이 생기기 전에 멈춘 경우에는 실행 기록과 로그를 보존해 원인을 확인한다.

- `check`는 실행 기록(`r2_train.json`)의 `git`이 자기 실행 commit과 같고 둘 다 clean일 때만 진행한다. 다르면 A1 비교 전에 거부하므로 `reproduce.json`도 생기지 않는다. 그 commit에 커밋된 승인 계약이 `train`의 근거가 된다. 결과에는 `train_git`과 `git`을 함께 남긴다.
- A1 비교는 기록하지 않는 모드로 실행하고, 비교가 끝난 직후 실제 HEAD와 clean 상태를 다시 확인한다. 시작 때와 같을 때만 `reproduce.json`과 `r0.json`을 쓴다. 비교 도중 HEAD가 바뀌거나 트리가 dirty가 되면 두 파일 모두 남기지 않고 멈춘다.

**기록 복사와 커밋 (records only)**: `check`가 `r0.json`을 쓴 뒤, 같은 HEAD에서 실행한다. `NOT_REPRODUCED`도 같은 방식으로 보존한다.

```bash
(
  set -euo pipefail
  cd ~/renju-h6
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src:$PWD/scripts"

  D=docs/h7r/r2/R2-D3-NO-OUTCOME-1
  R=runs/h7r/R2-D3-NO-OUTCOME-1/r0
  X="$D/r0"
  test -z "$(git status --porcelain --untracked-files=no)"
  test -f "$X/r0.json"
  extra="$(find "$X" -mindepth 1 ! -path "$X/r0.json" -print -quit)"
  if [[ -n "$extra" ]]; then
    printf 'STOP: unexpected record path %s\n' "$extra" >&2
    exit 1
  fi
  python - "$X/r0.json" "$R/reproduce.json" "$(git rev-parse HEAD)" <<'PY'
import hashlib, json, sys
r0 = json.load(open(sys.argv[1]))
raw = open(sys.argv[2], 'rb').read()
assert r0['reproduction']['sha256'] == hashlib.sha256(raw).hexdigest(), 'reproduce.json is not the one r0.json names'
assert r0['git'] == {'commit': sys.argv[3], 'dirty': False} == r0['train_git'], 'not the clean HEAD of train and check'
print('R0', r0['verdict']['verdict'], '/ A1', r0['verdict']['a1_verdict']['verdict'], r0['verdict']['reasons'])
PY
  cp --no-clobber "$R/reproduce.json" "$X/reproduce.json"
  cmp "$R/reproduce.json" "$X/reproduce.json"
  python scripts/h7r_d4_records.py copy-logs --src-dir "$R/logs" --pattern '*.log' \
    --dest-dir "$X/run-logs" --manifest "$X/log_manifest.json"
  python scripts/h7r_d4_records.py manifest --dir "$X" --output "$X/records_manifest.json"
  git add "$X/r0.json" "$X/reproduce.json" "$X/log_manifest.json" "$X/records_manifest.json" "$X"/run-logs/*.log
  expect=(--expect "$X/r0.json" --expect "$X/reproduce.json" --expect "$X/log_manifest.json"
          --expect "$X/records_manifest.json")
  for f in "$X"/run-logs/*.log; do expect+=(--expect "$f"); done
  python scripts/h7r_d4_records.py verify-staged "${expect[@]}"
  git commit -m "H7-R D3-R0: reproduction check of the c8 recipe with the current trainer (records only)"
  git push origin feat/h7-c1
)
```

- 시작할 때 기록 폴더에는 `r0.json`만 있어야 한다. `r0.json`이 가리키는 `reproduce.json`의 SHA, 같은 clean HEAD를 확인한 뒤 복사한다. 하나라도 실패하면 커밋하지 않는다.
- 결과가 `PASS`가 아니면 D3 학습으로 넘어가지 않는다. 결과와 원인을 기록한 뒤 다음 행동을 따로 정한다.

## 3. 판정

R2 설계 §7 규칙을 **그대로** 쓴다. 조건 C = D3, control K = 같은 seed의 기준 recipe(R2-0B 실행).

    I_s = L(K_s) − L(C_s)        (L = H-dev 주 지표: D4 예측, network 행, game macro; 양수가 개선)

- `DEVELOPMENT_SIGNAL`: 세 seed 모두 I_s > ε(0.0019461168062766864), 그리고 세 seed 모두 paired Δ(C_s:K_s)의 95% 상한 < 0
- 그 밖의 결과는 모두 `NOT_ESTABLISHED`다.
- control의 seed 범위(세 값의 max − min, R2-0B 0.00067)도 보고한다. 그것이 ε보다 크면 판독을 보류한다(§7).
- c5 대비 결과는 참고값으로 함께 보고한다(§7). 판독에는 쓰지 않는다.
- 구성 지표(outcome / proof 성분, ply, 차례)에는 주 지표의 ε를 적용하지 않는다. 같은 bootstrap 구간과 함께 기술한다.
- H-dev는 학습, early stopping, checkpoint 선택에 쓰지 않는다.

## 4. 사전 구성 확인 (계약 확정 전, 읽기 전용)

`scripts/h7r_d3_prep.py`는 계약 확정 전에 기존 기록에서 다음 항목을 계산한다. 판정 기준은 §3을 적용한다.

1. **H-dev 구성**: ply 구간 × label 종류(outcome, proof:vcf / unstoppable_four / forced_loss) × route(network / exact_leaf)의 행·게임·가중치
2. **주 지표의 게임별 가중치 몫**: 게임 g의 network 행 분모 den_g 중 각 종류(그리고 종류 × ply)가 차지하는 몫의 게임 평균. 전체 행을 합친 가중치 몫과는 다른 값이다.
3. **기존 주 지표 차이의 가산 분해**: seed 모델 대 c5·c8, c8 대 c5
   - 게임별 원래 분모를 유지한다: Δmacro = Σ_p mean_g [Δnum_g^p / den_g]
   - 부분의 합은 전체 차이와 정확히 같다(도구가 확인). bootstrap 회차마다도 같다(공통 게임 표본·공통 분모).
   - 이 기여도는 그 종류만의 macro(`network/outcome` 성분)와 다르다. 둘을 함께 보고한다.
   - 재계산한 전체와 성분이 커밋된 R2-0B paired 값과 같아야 한다(c8:c5는 R2-0A와 1e-5 이내). 다르면 멈춘다.
   - 구간은 기술용이다. ε를 부분에 적용하지 않는다. 기존 모델의 분해로 D3의 결과를 예측하지 않는다.
- 입력: 고정 R2 계약, R2-0B epsilon 기록(SHA 고정), 그 기록이 가리키는 `seed_predictions.pt`, R2-0A 기록, H-dev cache. 하나라도 없거나 다르면 멈춘다.

### 4.1 실행 및 기록 확인 (2026-11-05)

- 결과 커밋 `061e100`의 분석 실행은 깨끗한 `11e1bab`에서 8.3초에 완료됐다. B·C 로그의 종료 코드는 0이고 C와 D에는 각각 `VERIFY PASS`가 남았다. [결과 JSON](R2-D3-NO-OUTCOME-1/prep.json) · [로그 manifest](R2-D3-NO-OUTCOME-1/log_manifest.json)
- 기록 manifest의 5개 파일과 로그 manifest의 로그 3개를 실제 크기·SHA와 대조했다. 분석 소스 5개의 SHA는 실행 커밋 및 검토 checkout의 파일과 일치한다. 커밋된 입력 3개의 SHA를 재계산했고, 예측 tensor의 기록 SHA와 H-dev 검사값은 R2-0B 및 고정 계약의 해당 값과 대조했다. 원본 tensor와 cache 검증의 근거는 데스크톱 실행 로그다. [기록 manifest](R2-D3-NO-OUTCOME-1/records_manifest.json)
- seed 모델의 6개 비교에서 전체 차이와 outcome·proof 성분 macro가 R2-0B paired 기록과 일치했다. `c8:c5` 전체 차이는 R2-0A와 같았다. 7개 비교 모두 종류별 기여의 합과 종류 × ply별 기여의 합이 각각 전체 차이와 일치했다(절대 오차 1e-12 이내).

### 4.2 H-dev 구성과 가중치 몫

- 주 지표의 network 행은 396게임에서 15,279개다. outcome 13,805행과 proof 1,474행(VCF 1,042, unstoppable four 374, forced loss 58)으로 구성된다. [5,20)의 network proof는 141게임에 걸쳐 256행이다.
- exact_leaf 706행은 모두 proof이며 immediate win 382행, forced loss 324행이다. 주 지표의 분모에는 network 행을 사용한다.
- 게임별 원래 분모에서의 가중치 몫을 게임 균등 평균하면 outcome 43.7%, proof 56.3%다. 행을 통합해 계산한 network 가중치 몫은 각각 48.4%, 51.6%다. 게임 균등 몫은 proof [20,40) 33.3%, outcome [5,20) 25.3%, proof [5,20) 15.2% 순으로 크다.

### 4.3 기존 기준 모델의 가산 분해와 해석

각 값은 후보의 손실에서 c5의 손실을 뺀 차이다. 음수는 손실 감소를 뜻한다. 기여는 각 게임의 원래 network 분모로 계산하며 대괄호는 기술용 95% 구간이다.

| 비교 | 전체 Δ | outcome 기여 [95% 구간] | proof 기여 [95% 구간] | outcome [5,20) 기여 |
|---|---:|---:|---:|---:|
| seed31001:c5 | −0.004748 | −0.006016 [−0.007901, −0.004099] | +0.001268 [+0.000045, +0.002537] | −0.004071 |
| seed31002:c5 | −0.005421 | −0.005262 [−0.007425, −0.003084] | −0.000159 [−0.001231, +0.000916] | −0.004227 |
| seed31003:c5 | −0.004913 | −0.006097 [−0.008164, −0.004017] | +0.001183 [−0.000092, +0.002523] | −0.004314 |
| c8:c5(참고) | −0.004547 | −0.005160 [−0.006919, −0.003371] | +0.000613 [−0.000280, +0.001505] | −0.003520 |

- 세 seed의 outcome 기여 구간은 모두 0 아래다. 전체 outcome 순감소 중 [5,20) 칸의 몫은 seed 31001부터 각각 67.7%, 80.3%, 70.8%다. seed 31001·31003은 양의 proof 기여가 outcome 이득의 일부를 상쇄했고, seed 31002의 proof 기여는 0에 가깝다.
- outcome 기여는 −0.0061~−0.0053이고, outcome 성분 macro 차이는 −0.0230~−0.0208이다. 성분 macro는 outcome 자체의 분모를 사용하므로 별도의 값으로 보고한다.
- 같은 기준 recipe의 다른 학습 seed를 비교한 `seed:c8`에서도 proof 기여가 −0.000771~+0.000656 사이에서 움직였다. `seed31002:c8`의 proof 기여 구간은 [−0.001185, −0.000353], outcome [20,40) 기여는 +0.000325 [+0.000122, +0.000536]으로, 두 구간 모두 0을 포함하지 않았다.
- bootstrap은 각 비교의 학습된 모델을 고정한 채 H-dev 원본 게임을 20,000회 재표집한다. 따라서 구간은 H-dev 표본 변동을 반영하며 학습 실행 간 변동은 별도로 읽는다. D3에서는 같은 seed의 직접 control과 짝을 이루고, 종류·ply별 기여와 구간은 기술적으로 보고한다. 구성 지표의 구간만으로 처방 효과를 판정하거나 주 지표의 ε를 적용하지 않는다.

## 5. 보고 (실행 시)

- seed별 I_s와 paired 구간, `DEVELOPMENT_SIGNAL` 판정, c5 대비 참고값
- 주 지표의 종류별·종류 × ply 가산 기여도(4절과 같은 분해), outcome / proof 성분 macro, 차례별, pooled
- 학습 기록: 선택 step, objective 곡선, masked cycle val(측정만), H6 guard, cycle weight / kept weight, cycle 항이 0인 batch 수
- control과의 자료 배정 지문 비교, 출처 SHA, 소요 시간

## 6. 후속 행동

- `DEVELOPMENT_SIGNAL`이 아니면 D4로 자동 진행하지 않는다.
- proof 성분만 좋아졌다는 이유로 결과를 본 뒤 D4 진입 기준을 바꾸지 않는다.
- `DEVELOPMENT_SIGNAL`이어도 D4는 별도 계약·승인으로 정한다.
- NO_CANDIDATE나 실패 seed는 버리거나 다른 seed로 바꾸지 않는다. 기록을 보존하고 원인을 보고한다.
- 결과가 무엇이든 이 진단에서 배운 것을 기록한 뒤 다음 질문(proof 전용 판독, D1 등)을 따로 정한다.

## 7. 확정 전 정할 것

1. R0 실행 승인 및 데스크톱 재현 판정 확인(§2.5)
2. D3 outcome 마스크 생성·검증과 실행 도구 구현
3. 계약 JSON: 조건, control 경로와 SHA, 출처 SHA, 실행 시간 한도, 출력 경로
4. 실행 승인 단계: 학습 3회 → H-dev 평가와 판독. 각 단계는 따로 승인받는다.

## 8. 예상 비용 (계획값, 측정 아님)

- 학습: R2-0B seed 실행 4.2–4.5분 × 3회 정도
- H-dev 평가: R2-0B epsilon 단계와 비슷한 규모
- 4절 사전 확인: CPU, 수 분 이내 예상
