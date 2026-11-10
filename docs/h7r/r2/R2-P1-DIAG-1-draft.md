# R2-P1-DIAG-1: R2-P1 재현 gate 실패의 원인 진단

- 상태: 완료. 1차 시도는 도구 버그로 실패(`0b1dae8`, §6.1), 2차 시도 10회 모두 완료(`0fe5aa1`, §6.2). A1은 초안만 있다(`R2-P1-A1-draft.md`)
- 사용자 지시(2026-11-10): 원인 진단만 진행한다.
  - 기존 실험 `e41841eaafa8`의 FAIL 기록(`a1cd152`)은 보존한다.
  - 허용 오차 변경, P1 본학습 6회, Arena는 승인하지 않았다.
  - 진단 뒤 개정 A1은 초안으로 제안만 한다.
- 도구: `scripts/h7r_p1_diag.py`, 테스트 `tests/test_h7r_p1_diag.py` (§4 블록도 임시 git 저장소에서 bash로 시험: 실행 하나 실패, 비교 실패 뒤 재실행, 기록 없는 실행 디렉터리)
- 바꾸지 않은 것: P1 계약, 코드(`h7r_p1.py`, `h7r_p1_train.py`, `h7c_train.py` 포함), 기록. 도구는 P1 계약을 읽기만 하고, 입력도 P1의 `check_inputs`로 같은 기대값과 대조한다.

## 1. 질문

- c2 재현은 구조(step, eligibility, 선택 step 77)가 정확히 같았다. 그런데 수치는 평가 29부터 1e-6을 넘었다(§9.3, `R2-P1-POLICY-REPLAY-1-draft.md`).
- 그 차이가 어디서 오는지 묻는다.
  - 현재 환경에서 같은 trainer를 다시 돌려도 생기는가(실행 간 변동)
  - 기존 trainer와 새 trainer 사이에서만 생기는가(코드 경로 차이)
  - 새 trainer의 추가 진단 연산(RenjuNet val KL) 때문인가
  - 결정론 설정을 켜면 사라지는가
- 원인을 미리 정하지 않는다. GPU 비결정성도 가설의 하나일 뿐이다.

## 2. 실행 (각각 별도 프로세스, 같은 입력·seed 20261024·c2 설정)

| 이름 | trainer | 옵션 | 횟수 |
|---|---|---|---:|
| `old` | `h7c_train.train_policy` (c2를 만든 trainer) | 기본 설정 | 2 |
| `new` | `h7r_p1_train.train_policy`, replay off | 기본 설정 | 2 |
| `new-diag` | 위와 같음 | 재현 gate 실행처럼 RenjuNet val KL 진단 켬 | 2 |
| `old-det` | 기존 | `torch.use_deterministic_algorithms(True)`, cuDNN deterministic, `CUBLAS_WORKSPACE_CONFIG=:4096:8` | 2 |
| `new-det` | 새, replay off | 위와 같음 | 2 |

- 결정론 실행은 일부 연산이 결정론 구현을 갖지 않으면 오류로 멈출 수 있다. 그것도 결과로 기록한다.
- 실행마다 기록하는 것
  - 수치 설정: torch, CUDA, cuDNN, GPU, cuDNN enabled·benchmark·deterministic, TF32(cuDNN·matmul), float32 matmul precision, 결정론 알고리즘, 스레드 수, 관련 환경 변수
  - 입력 SHA(c2 행, H3, cache 텐서 해시)와 초기 parent·teacher state 해시
  - step마다의 지문: cycle batch index, D4 대칭, 학습률 계수, 학습 손실
    - 기존 함수를 감싸 기록만 하고, 감싼 함수의 결과를 그대로 돌려준다. 결과가 바뀌지 않는다는 것은 테스트로 확인했다.
  - 평가 곡선(val objective·CE·KL, guard top-1), eligibility, 선택 step
  - 선택 state의 해시, 그리고 고정 국면(RenjuNet val 앞 4,096개와 cycle val 1,423행)에서의 선택 정책 log-확률
    - state와 log-확률은 `runs/`에만 두고, 기록에는 해시만 남긴다.

## 3. 비교 (`compare`, 기술만 하고 gate는 만들지 않는다)

- 같은 설정의 반복끼리: 실행 간 변동
- `old` 대 `new`: 코드 경로 차이
- `new` 대 `new-diag`: 추가 진단 연산의 영향
- `old-det` 대 `new-det`, `old` 대 `old-det`: 결정론 설정의 효과
- 쌍마다 보는 것
  - 설정·입력·초기 state가 같은지
  - 지문 4종(batch, D4, 학습률, step 수)이 같은지
  - 학습 손실이 처음 달라지는 step, 손실 차이의 최댓값
  - 평가 곡선 차이의 최댓값과 처음 달라지는 평가
  - eligibility와 선택 step이 같은지
  - 선택 state 차이의 최댓값
  - 고정 국면 log-확률 차이의 최댓값과 argmax 일치율
- 모든 실행을 커밋된 c2 기록과도 비교한다.
- 이 비교에서 허용 오차를 뽑아내지 않는다. 관측 최댓값에 맞춰 기준을 자동으로 넓히지 않는다.

## 4. 데스크톱 실행 (tmux 권장, 10회 × 약 2분)

- 실행마다 고유 로그를 쓰고 끝에 `__EXIT__` 코드를 남긴다. 한 실행이 실패해도 나머지는 계속 돈다.
- 다시 실행하면 끝난 실행(`record.json` 있음)은 건너뛴다. 기록 없이 파일만 남은 실행이 있으면 멈춘다.
- 모두 끝나면 비교 결과, 실행 기록 사본, 모든 로그를 한 번에 커밋한다. 입력 원본과 state, log-확률은 커밋하지 않는다.

```bash
(
  set -euo pipefail
  cd ~/renju-h6
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src:$PWD/scripts"
  [ "$(git branch --show-current)" = feat/h7-c1 ] || { echo "STOP: wrong branch"; exit 1; }
  git pull --ff-only origin feat/h7-c1
  test -z "$(git status --porcelain --untracked-files=no)"
  python scripts/check_track_b_frozen.py
  D=docs/h7r/r2/R2-P1-DIAG-1
  R=runs/h7r/R2-P1-DIAG-1
  L="$R/logs"
  mkdir -p "$L"
  tracked(){ git ls-files --error-unmatch "$1" >/dev/null 2>&1; }
  if tracked "$D/diagnosis.json"; then echo "STOP: the diagnosis is already recorded"; exit 1; fi
  if [ -e "$D/diagnosis.json" ]; then echo "STOP: an uncommitted diagnosis.json exists"; exit 1; fi
  STAMP="$(date +%Y%m%d-%H%M%S)"
  for spec in "old-1|old|" "old-2|old|" "new-1|new|" "new-2|new|" "new-diag-1|new|--diag" "new-diag-2|new|--diag" \
              "old-det-1|old|--deterministic" "old-det-2|old|--deterministic" \
              "new-det-1|new|--deterministic" "new-det-2|new|--deterministic"; do
    IFS='|' read -r run trainer flag <<< "$spec"
    out="$R/$run"
    if [ -f "$out/record.json" ]; then echo "done: $out"; continue; fi
    if [ -d "$out" ] && [ -n "$(ls -A "$out")" ]; then echo "STOP: $out holds files without a record"; exit 1; fi
    b="run-$run-$STAMP"
    if [ -e "$L/$b.log" ]; then echo "STOP: $L/$b.log exists"; exit 1; fi
    rc=0
    if [ "$flag" = --deterministic ]; then
      CUBLAS_WORKSPACE_CONFIG=:4096:8 python -u scripts/h7r_p1_diag.py run --trainer "$trainer" --deterministic \
        --device cuda --out-dir "$out" 2>&1 | tee "$L/$b.log" || rc=$?
    else
      python -u scripts/h7r_p1_diag.py run --trainer "$trainer" ${flag:+$flag} --device cuda --out-dir "$out" \
        2>&1 | tee "$L/$b.log" || rc=$?
    fi
    echo "__EXIT__ $rc" >> "$L/$b.log"
    echo "[diag] $run exit $rc"
  done
  b="compare-$STAMP"
  rc=0
  python -u scripts/h7r_p1_diag.py compare --run-dir "$R" --output "$D/diagnosis.json" 2>&1 | tee "$L/$b.log" || rc=$?
  echo "__EXIT__ $rc" >> "$L/$b.log"
  expect=()
  shopt -s nullglob
  for f in "$L"/*.log; do
    n="$(basename "$f" .log)"
    tracked "$D/stage-logs/$n.log" && continue
    python scripts/h7r_d4_records.py copy-logs --resume --src-dir "$L" --pattern "$n.log" \
      --dest-dir "$D/stage-logs" --manifest "$D/stage-logs/$n.manifest.json"
    expect+=(--expect "$D/stage-logs/$n.log" --expect "$D/stage-logs/$n.manifest.json")
  done
  for rec in "$R"/*/record.json; do
    run="$(basename "$(dirname "$rec")")"
    dst="$D/run-records/$run/record.json"
    tracked "$dst" && continue
    mkdir -p "$(dirname "$dst")"
    cp --no-clobber "$rec" "$dst"
    cmp "$rec" "$dst"
    expect+=(--expect "$dst")
  done
  if [ -f "$D/diagnosis.json" ]; then expect+=(--expect "$D/diagnosis.json"); fi
  for e in "${expect[@]}"; do [ "$e" = --expect ] || git add "$e"; done
  python scripts/h7r_d4_records.py verify-staged "${expect[@]}"
  git commit -m "H7-R P1 diagnosis: reproduction-failure runs and comparison, compare exit $rc (records only)"
  git push origin feat/h7-c1
  echo "__RECORDED__ $rc"
  exit "$rc"
)
```

## 5. 다음

- 결과를 커밋된 기록으로 대조한 뒤, 개정 A1을 **초안으로만** 제안한다. 검토할 방안은 다음과 같다.
  - 현재 환경에서 기존·새 trainer의 동등성을 확인하는 gate
  - 과거 c2 재현을 판정이 아닌 참고 기록으로 분리하는 방안
- 허용 오차는 관측 최댓값에 맞춰 정하지 않는다.

## 6. 실행 기록

### 6.1 1차 시도 (`0b1dae8`): 학습 전 전부 실패, 원인은 진단 도구 버그

- 10회 모두 `STOP: RuntimeError: Numpy is not available`, `__EXIT__ 1`. 비교는 `no runs`, exit 1이다.
- 원인: 진단 도구가 state 해시와 log-확률 해시에 `tensor.numpy()`를 썼다. 데스크톱 `.venv`에는 NumPy가 없다(`h3_cache.tensor_bytes`가 NumPy 없이 바이트를 읽는 이유).
  - 해시는 학습 전, 초기 parent·teacher state를 기록하는 단계에서 처음 계산된다. 그래서 학습은 한 step도 돌지 않았다.
  - cloud 환경에는 NumPy가 있어 테스트가 잡지 못했다.
- 이 시도는 R2-P1 재현 실패에 대해 아무것도 말하지 않는다. 기록과 로그는 그대로 둔다.
- 수정: 두 해시를 `h3_cache.tensor_bytes`로 계산한다. 같은 바이트이므로 해시 정의는 바뀌지 않는다.
  - `Tensor.numpy`가 오류를 내도록 막은 상태에서 실행과 비교가 끝나는지 시험하는 테스트를 추가했다. 이 테스트는 고치기 전 코드에서 실패한다.
- 재실행은 §4 블록 그대로다. 실패한 실행은 디렉터리를 남기지 않았으므로 10회를 모두 다시 돌린다. 1차 로그는 이미 커밋되어 있어 건너뛰고, 새 로그는 새 이름으로 남는다.

### 6.2 2차 시도 (`0fe5aa1`): 10회 모두 완료, 비교 exit 0

- 검증: 커밋된 실행 기록 10개로 비교를 다시 계산했고 `diagnosis.json`과 같았다(state·log-확률 파일이 필요한 세 항목은 제외).
- 모든 쌍에서 같았던 것
  - 입력 SHA, 학습 설정, 초기 parent·teacher state
  - 지문 4종(step 수 96, batch 순서, D4 대칭, 학습률)
  - 평가 step, eligibility, 선택 step 77
- 수치 설정(기록값): torch 2.14.0+cu130, CUDA 13.0, cuDNN 9.24, RTX 5070, cuDNN benchmark off, cuDNN TF32 on, matmul TF32 off, float32 matmul precision `highest`, 스레드 8. 결정론 실행만 `deterministic_algorithms`·cuDNN deterministic·`CUBLAS_WORKSPACE_CONFIG=:4096:8`이 달랐다.

| 쌍 | 손실이 처음 다른 step | 평가 곡선이 처음 다른 평가 | val objective 최대 차 | guard top-1 최대 차 | 선택 state 최대 차 | 고정 국면 argmax 일치 |
|---|---:|---:|---:|---:|---:|---:|
| old 반복 | 21 | 29 | 1.0e-4 | 9.4e-5 | 7.5e-4 | 99.87% |
| new 반복 | 21 | 29 | 1.0e-4 | 1.4e-4 | 7.4e-4 | 99.91% |
| new-diag 반복 | 60 | 58 | 3.4e-5 | 7.1e-5 | 3.3e-5 | 99.96% |
| old 대 new | 53 | 58 | 2.8e-5 | 1.0e-4 | 4.6e-5 | 99.95% |
| new 대 new-diag | 7 | 10 | 1.1e-4 | 1.2e-4 | 5.7e-4 | 99.82% |
| **old-det 반복** | 없음 | 없음 | 0 | 0 | 0 | 100% |
| **new-det 반복** | 없음 | 없음 | 0 | 0 | 0 | 100% |
| **old-det 대 new-det** | 없음 | 없음 | 0 | 0 | 0 | 100% |
| old 대 old-det | 3 | 10 | 1.0e-4 | 1.3e-4 | 7.8e-4 | 99.93% |

- 결정론 실행 4회의 선택 state 해시는 모두 `4e7939153b37…`이다. 기본 설정 실행 6회의 해시는 서로 모두 다르다.
- c2 기록과의 비교(10회 모두): eligibility, 선택 step, teacher guard, parent val은 같다.
  - 기본 설정 실행은 평가 29 또는 38부터 다르다. val objective 차이는 최대 1.0e-4, guard top-1 차이는 최대 1.5e-4(127,316 국면 중 약 19개)이다.
  - 결정론 실행은 평가 10부터 다르다. 결정론 구현은 다른 kernel을 쓰므로, 기본 설정으로 만든 c2와 첫 평가부터 다르다.
- 재현 gate 실행(`a1cd152`)도 평가 29부터, 같은 크기로 달랐다.

### 6.3 해석 (관측이 지지하는 범위)

- **재현 gate 실패는 현재 환경의 기본 GPU 설정에서 생기는 실행 간 수치 변동으로 설명된다.**
  - 같은 trainer, 같은 입력·설정·초기 state·batch·D4·학습률로 별도 프로세스에서 두 번 돌리면, 기본 설정에서는 old와 new 모두 손실이 step 21부터, 평가 곡선이 평가 29부터 달라진다.
  - 그 크기(val objective 약 1e-4)와 시작 지점이 재현 gate 실행과 c2의 차이와 같다.
  - 결정론 설정을 켜면 반복이 비트 단위로 같아진다. 그래서 변동의 원천은 결정론 설정이 바꾸는 연산(비결정적 GPU kernel) 쪽에 있다. 어느 연산인지는 특정하지 않았다.
- **코드 경로 차이의 증거는 없다.** 결정론 설정에서 기존 trainer와 새 trainer(replay off, 진단 off)는 손실, 곡선, 선택 state, 고정 국면 예측까지 비트 단위로 같다.
- **추가 진단 연산(RenjuNet val KL)의 영향은 분리하지 못했다.**
  - 기본 설정에서는 어떤 두 실행도 다르므로, new 대 new-diag의 차이를 실행 간 변동과 구별할 수 없다. 처음 다른 step 7은 반복 쌍들(21, 60)보다 이르지만, 반복 쌍끼리도 21과 60으로 퍼져 있다.
  - 진단 연산은 batch·D4·학습률 지문을 바꾸지 않았다(난수 소비 없음). 결정론 설정에서 진단 연산을 켠 실행은 이번 행렬에 없었다.
- **말할 수 없는 것**
  - c2 원래 실행이 같은 변동을 겪었는지 직접 볼 수는 없다. 기록된 설정은 같다.
  - 1e-6은 현재 환경에서 c2 절차를 두 번 돌렸을 때의 차이보다 작다. 이것은 허용 오차를 넓힐 근거가 아니라, 기본 설정 실행을 1e-6으로 판정하는 gate가 이 환경에서 통과를 보장할 수 없다는 뜻이다.
