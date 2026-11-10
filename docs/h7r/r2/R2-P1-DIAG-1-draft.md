# R2-P1-DIAG-1: R2-P1 재현 gate 실패의 원인 진단

- 상태: 도구·테스트 완료, 데스크톱 실행 대기
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
