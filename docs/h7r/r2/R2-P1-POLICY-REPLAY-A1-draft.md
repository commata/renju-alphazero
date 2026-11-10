# R2-P1-POLICY-REPLAY-A1: 결정론 동등성 gate를 둔 policy replay 대조 실험

- 상태: prepare PASS(`03d3ec5`), gate 승인(`a0f2987`)·PASS(`b3eb145`). train·arena는 미승인
- 사용자 결정(2026-11-10): A1을 채택한다.
  - 과거 c2의 수치, eligibility, 선택 step은 참고 기록으로 둔다.
  - 입력, 모델 신원, 학습 설정과 step 구조는 계속 엄격히 검사한다.
  - 현재 환경 동등성 gate와 본학습 모두 같은 결정론 설정을 쓴다.
  - E2/E3는 replay off + 진단 on, E4/E5는 replay on + 진단 on으로 돌린다.
  - gate는 본학습과 같은 진입점을 쓰고, 정확 일치를 요구한다.
  - 실패하면 멈추고, 기준을 완화하지 않는다.
  - prepare, gate, 본학습, Arena는 단계별로 승인한다.
- 근거: `R2-P1-DIAG-1-draft.md` §6 (기록 `0fe5aa1`), 제안 `R2-P1-A1-draft.md`
- 보존: 기존 실험 R2-P1-POLICY-REPLAY-1(`e41841eaafa8`)의 계약과 모든 기록(재현 gate FAIL `a1cd152` 포함)은 그대로 둔다. 그 기록은 이 실험의 통과 근거로 쓰지 않는다.
- 새 식별값
  - 계약: `docs/h7r/r2/R2-P1-POLICY-REPLAY-A1/p1a1-contract.json` (`h7r-p1a1-contract-v1`, 초안 `p1a1-contract.draft.json`)
  - 실험 SHA-256: 계약에서 `approved`를 뺀 본문의 해시. §6에 적는다.
  - 기록: `docs/h7r/r2/R2-P1-POLICY-REPLAY-A1/`, 실행 출력: `runs/h7r/R2-P1-POLICY-REPLAY-A1/`
  - Arena 계약: `docs/h7r/contracts/R2-P1-A1-ARENA-1.json`
- 바꾸지 않은 것
  - 입력, arm 정의(μ 0.5, replay batch 512, 생성기 seed + 1000003), 학습량과 선택 규칙
  - 학습 seed 3개, 후보 seed 20261024, Arena arm·쌍 수·seed(8788 / 8789 / 8790)·suite, bootstrap seed
  - 판독 기준과 해석, 데이터 규칙
  - 코드: `h7c_train.py`, `h7r_p1_train.py`, `h7r_p1.py`, `h7r_p1_diag.py`. 새 도구는 이들을 고치지 않고 가져다 쓴다.
- 도구: `scripts/h7r_p1a1.py`, 테스트 `tests/test_h7r_p1a1.py`

## 1. 결정론 설정 (계약 `determinism`)

- 프로세스 시작 전에 환경 변수 `CUBLAS_WORKSPACE_CONFIG=:4096:8`를 설정해야 한다. cuBLAS는 시작할 때 한 번만 읽는다. 값이 다르거나 없으면 도구는 시작하지 않는다.
- 도구가 실행 동안 켜는 것(끝나면 원래대로 되돌린다)
  - `torch.use_deterministic_algorithms(True)`
  - `cudnn.deterministic = True`, `cudnn.benchmark = False`
- 검사만 하고 바꾸지 않는 것: cuDNN TF32 on, matmul TF32 off, float32 matmul precision `highest`. c2와 진단이 이 PyTorch 기본값으로 돌았다.
- 실행 시작 때 이 값들을 모두 확인하고, 끝날 때 바뀌지 않았는지 다시 확인한다.
- 전체 수치 설정(torch, CUDA, cuDNN, GPU, 스레드 포함)을 기록한다.
  - gate 5회는 서로 같아야 한다.
  - 본학습 6회는 각각 gate와 같아야 한다. 다르면 학습을 시작하지 않는다.
- 결정론 구현이 없는 연산을 만나면 실행이 오류로 멈춘다. 이것도 실패로 기록하고, 비결정론 설정으로 돌아가는 대체 경로는 없다.

## 2. 하나의 진입점 (`execute`)

- gate 실행(`gate-run`)과 본학습(`train`)은 모두 같은 함수 `execute`로 학습한다.
  - 진행 순서: 결정론 설정 → 입력(`h7r_p1.check_inputs`가 대조한 객체) → 학습 → 선택 state의 고정 국면 log-확률
  - 학습하는 동안 지문을 기록한다(`h7r_p1_diag.Fingerprints`). 지문은 cycle batch, D4 대칭, 학습률, step별 학습 손실이다. 감싼 함수의 결과를 그대로 돌려주므로 학습은 바뀌지 않는다(진단에서 테스트로 확인).
- 실행 종류
  - `old`: `h7c_train.train_policy`, 곧 c2 trainer
  - `C`: `h7r_p1_train.train_policy`, replay off, 진단 on
  - `T`: 같은 trainer, replay on(μ 0.5, batch 512), 진단 on
  - C와 T의 인자는 기존 `h7r_p1.run_training`과 같다.
- 기록: 수치 설정, 입력, 초기 parent·teacher state 해시, 지문, trainer 기록 전체, P1 기록(replay·진단), 선택 state 해시, 고정 국면 log-확률 해시, 소요 시간
  - state·checkpoint·log-확률 파일은 `runs/`에만 둔다.

## 3. 현재 환경 동등성 gate

| 실행 | 종류 | 설정 |
|---|---|---|
| E1 | `old` | c2 trainer |
| E2, E3 | `C` | replay off, 진단 on |
| E4, E5 | `T` | replay on, 진단 on |

- 모두 seed 20261024, 결정론 설정. 실행마다 별도 프로세스(`gate-run --run En`)다.
- 판정(`gate`)은 커밋된 gate 기록 5개만 읽는다.

**엄격 검사** (수치 계산과 무관한 항목. 하나라도 어긋나면 FAIL)
- 실행 신원: 기록 형식, 실행 이름, 종류, seed
- 입력: cycle 행 SHA, H3 parent·teacher 신원, RenjuNet cache 텐서 해시. c2 기록의 입력 SHA와도 같아야 한다.
- 학습 설정: c2 기록과 같아야 한다(device 제외). device는 계약값이어야 한다.
- 수치 설정: §1의 값. 5회 모두에서 전체 수치 설정, source SHA, 초기 state가 같아야 한다.
- step 구조: total 96, epoch당 48, warmup 5, 평가 step 목록, train·val 행 수, 지문이 기록한 step 수
- arm: C에는 replay가 없고, T는 계약의 replay 설정으로 돌았으며, E2–E5는 진단이 켜져 있어야 한다.
- c2 기록 자체가 계약의 step 구조를 보여야 한다.

**정확 일치** (허용 오차 0)
- G1 결정론: E2 = E3, E4 = E5
- G2 동등성: E1 = E2
- 비교하는 것
  - 지문(step별 손실 포함)
  - trainer 기록 전체(평가마다의 값, eligibility, 선택, guard, verdict)
  - 초기·선택 state 해시, 고정 국면 log-확률 해시
  - G1에서는 P1 기록(replay·진단)도 비교한다. G2는 E1에 없는 P1 기록만 뺀다.
- 허용 오차 0은 관측값에 맞춘 값이 아니다. "결정론 설정에서 같은 계산을 한다"는 말의 정의다.

**참고 기록** (판정하지 않는다)
- 과거 c2 기록과 각 gate 실행의 비교를 `gate.json`의 `c2_reference`에 남긴다.
  - 평가별 차이의 최댓값과 처음 다른 평가
  - eligibility와 선택 step이 같은지, guard·verdict
- 결정론 실행은 기본 설정으로 만든 c2와 평가 10부터 다르다(진단 §6.2). 그래서 이 값들은 판정에 쓰지 않는다.

**실패하면**
- 본학습 전에 멈춘다. 기록과 로그는 커밋한다.
- 하지 않는 것: 기준 완화, 통과할 때까지 반복 실행, 진단 끄기, 설정 바꾸기
- 원인 진단은 따로 승인받는다.
- gate 실행 하나가 오류로 끝난 경우도 같다. 블록은 그 로그를 커밋하고 멈추며, 다시 실행해도 거부한다. 보고 뒤 결정을 받는다.

**train 단계의 재판정**
- `train`은 학습 전에 커밋된 gate 기록 5개로 판정을 다시 계산한다.
- 그 결과가 커밋된 `gate.json`과 같고 통과일 때만 진행한다. `passed` 값만 믿지 않는다.

## 4. 본학습, link, Arena, judge

- 본학습: 3 seed × 2 arm. `execute`로 돌리고, 수치 설정은 gate와 같아야 한다. step 구조도 다시 확인한다. 실행마다 따로 커밋한다.
- link: 후보 seed 20261024의 T와 C가 모두 CANDIDATE여야 한다. 아니면 멈추고, 다른 seed로 대체하지 않는다.
- Arena 계약: `R2-P1-A1-ARENA-1`. arm 8개(본 arm 4개, 쌍 300, seed 8788·8789 / 보조 arm 4개, 쌍 25, seed 8790), budget S50
- judge
  - Arena 계약은 A1 계약과 검증된 link로 다시 만든 것과 정확히 같아야 한다.
  - arm 기록은 모두 완전해야 한다.
  - P(T 대 champion)와 D(T − C)를 suite 층화, 50:50, 쌍 대응으로 읽는다. class 단위 민감도도 함께 낸다.
  - 판독 기준과 해석은 R2-P1-POLICY-REPLAY-1과 같다. Arena는 champion 변경이 아니다.

## 5. 단계와 승인

| 단계 | 승인 | 명령 | 블록 |
|---|---|---|---|
| prepare | `approved.prepare` | `prepare` | a1-1 |
| gate | `approved.gate` | `gate-run` ×5, `gate` | a1-G |
| 본학습 | `approved.train` | `train` ×6 | a1-2 |
| link | `approved.train` | `link` (클라우드) | a1-3 |
| Arena 계약 | `approved.arena` | `arena-contract` (클라우드) | a1-3 |
| Arena·judge | `approved.arena` | `h7r_eval.py play`, `judge` | a1-4 |

- 승인은 순서대로만 켤 수 있다. 앞 단계가 승인되지 않았으면 도구가 거부한다.
- 승인을 바꿔도 실험 SHA는 바뀌지 않는다. 계약 파일 SHA는 기록마다 따로 남긴다.

## 6. 계약 고정

- 초안 `p1a1-contract.draft.json`은 기존 P1 계약에서 만들었다. 바뀌지 않은 절(goal, question, inputs, arms, seeds, arena, reading, data_rule, separate_stages)은 기존 계약과 같고, 테스트가 이를 확인한다.
- 고정본 `p1a1-contract.json`은 초안에 `status: fixed`, source SHA, 초안 SHA를 더한 것이다.
- 근거(evidence)로 고정한 기록: c1·c2 Arena 판정, c2 policy 기록, cache 보고서, 기존 P1 계약과 그 재현 FAIL 기록, 진단 결과(`diagnosis.json`)
- 실험 SHA-256: `faeb72b0f625baba213cc88504388dcd1d5eed03a16307de8da2217583c02648` (`faeb72b0f625`). 기존 실험 `e41841eaafa8`과 다르다.

## 7. 실행 절차 (데스크톱)

- 단계마다 승인 커밋을 받은 뒤에만 실행한다.
- 로그
  - 명령마다 시각이 붙은 고유 로그를 `runs/.../logs/`에 남기고, 끝에 `__EXIT__ <코드>`를 덧붙인다.
  - 성공·실패와 무관하게 로그 사본과 manifest를 `docs/.../stage-logs/`에 커밋한다.
  - 이미 있는 로그와 기록은 덮어쓰지 않는다.
- 기록 위치
  - gate 기록 사본은 `gate-records/`, 학습 기록 사본은 `seed-records/`에 둔다. `runs/`는 커밋되지 않는다.
  - checkpoint는 커밋하지 않는다.
- 중단 뒤 복구
  - 블록이 기록 커밋 전에 멈추면 **a1-F**를 실행한다. 아무것도 다시 실행하지 않고 기록만 커밋한다.
  - Arena(a1-4)가 끝나지 않은 동안에는 a1-F를 쓰지 않는다. a1-4를 다시 실행해 같은 HEAD에서 재개한다.
- 예상 시간
  - prepare: 수 분
  - gate: 5회 × 약 1–2분(진단 10회가 약 10분이었다)
  - 본학습: 6회 × 약 1–2분
  - Arena: arm 4개 약 3시간과 보조 arm

a1-1. prepare (`approved.prepare` 뒤)

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
  D=docs/h7r/r2/R2-P1-POLICY-REPLAY-A1
  L=runs/h7r/R2-P1-POLICY-REPLAY-A1/logs
  if git ls-files --error-unmatch "$D/prepare.json" >/dev/null 2>&1; then echo "STOP: prepare is already recorded"; exit 1; fi
  if [ -e "$D/prepare.json" ]; then echo "STOP: an uncommitted prepare.json exists: run a1-F"; exit 1; fi
  mkdir -p "$L"
  B="prepare-$(date +%Y%m%d-%H%M%S)"
  if [ -e "$L/$B.log" ]; then echo "STOP: $L/$B.log exists"; exit 1; fi
  rc=0
  python -u scripts/h7r_p1a1.py prepare 2>&1 | tee "$L/$B.log" || rc=$?
  echo "__EXIT__ $rc" >> "$L/$B.log"
  python scripts/h7r_d4_records.py copy-logs --src-dir "$L" --pattern "$B.log" \
    --dest-dir "$D/stage-logs" --manifest "$D/stage-logs/$B.manifest.json"
  expect=(--expect "$D/stage-logs/$B.log" --expect "$D/stage-logs/$B.manifest.json")
  if [ "$rc" -eq 0 ]; then expect+=(--expect "$D/prepare.json"); fi
  for e in "${expect[@]}"; do [ "$e" = --expect ] || git add "$e"; done
  python scripts/h7r_d4_records.py verify-staged "${expect[@]}"
  if [ "$rc" -eq 0 ]; then verdict=PASS; else verdict="FAIL (exit $rc)"; fi
  git commit -m "H7-R P1-A1: prepare $verdict (records only)"
  git push origin feat/h7-c1
  echo "__RECORDED__ $rc"
  exit "$rc"
)
```

a1-F. 기록 보존 (어느 블록이든 기록 커밋 전에 멈췄을 때만. 아무것도 다시 실행하지 않는다)

```bash
(
  set -euo pipefail
  cd ~/renju-h6
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src:$PWD/scripts"
  [ "$(git branch --show-current)" = feat/h7-c1 ] || { echo "STOP: wrong branch"; exit 1; }
  D=docs/h7r/r2/R2-P1-POLICY-REPLAY-A1
  R=runs/h7r/R2-P1-POLICY-REPLAY-A1
  L="$R/logs"
  echo __START__
  # an A1 command still running in this checkout: a python process here whose script is an A1 tool
  if python - "$PWD" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
for proc in Path('/proc').iterdir():
    try:
        if not proc.name.isdigit() or (proc / 'cwd').resolve() != root:
            continue
        args = [a.decode() for a in (proc / 'cmdline').read_bytes().split(b'\0') if a]
    except (OSError, UnicodeError):
        continue
    if args and Path(args[0]).name.startswith('python') and \
            {'scripts/h7r_p1a1.py', 'scripts/h7r_eval.py'} & set(args[1:3]):
        print('running:', proc.name, ' '.join(args))
        sys.exit(0)
sys.exit(1)
PY
  then echo "STOP: an A1 command is still running"; exit 1; fi
  A=docs/h7r/contracts/R2-P1-A1-ARENA-1.json
  if [ -f "$R/arena/HEAD" ] && [ -f "$A" ]; then
    for arm in $(python -c "import json, sys; print(' '.join(a['name'] for a in json.load(open(sys.argv[1]))['arms']))" "$A"); do
      if [ ! -e "$R/arena/$arm.json" ]; then
        echo "STOP: the Arena is unfinished ($arm): resume a1-4 first (a commit now would move HEAD and break the resume)"
        exit 1
      fi
    done
  fi
  git reset -q -- "$D"
  test -z "$(git status --porcelain --untracked-files=no)"
  tracked(){ git ls-files --error-unmatch "$1" >/dev/null 2>&1; }
  expect=()
  shopt -s nullglob
  for f in "$L"/*.log; do
    b="$(basename "$f" .log)"
    tracked "$D/stage-logs/$b.log" && continue
    echo "== $f"; tail -n 2 "$f"
    python scripts/h7r_d4_records.py copy-logs --resume --src-dir "$L" --pattern "$b.log" \
      --dest-dir "$D/stage-logs" --manifest "$D/stage-logs/$b.manifest.json"
    expect+=(--expect "$D/stage-logs/$b.log" --expect "$D/stage-logs/$b.manifest.json")
  done
  for n in prepare gate link judge; do
    if [ -f "$D/$n.json" ] && ! tracked "$D/$n.json"; then expect+=(--expect "$D/$n.json"); fi
  done
  for rec in "$R"/gate/E*/record.json; do
    run="$(basename "$(dirname "$rec")")"
    dst="$D/gate-records/$run/record.json"
    tracked "$dst" && continue
    mkdir -p "$(dirname "$dst")"
    cp --no-clobber "$rec" "$dst"
    cmp "$rec" "$dst"
    expect+=(--expect "$dst")
  done
  for rec in "$R"/[CT]-*/record.json; do
    run="$(basename "$(dirname "$rec")")"
    dst="$D/seed-records/$run/record.json"
    tracked "$dst" && continue
    mkdir -p "$(dirname "$dst")"
    cp --no-clobber "$rec" "$dst"
    cmp "$rec" "$dst"
    expect+=(--expect "$dst")
  done
  if [ "${#expect[@]}" -eq 0 ]; then
    git push origin HEAD:feat/h7-c1
    echo "nothing left to record; push checked"; exit 0
  fi
  for e in "${expect[@]}"; do [ "$e" = --expect ] || git add "$e"; done
  python scripts/h7r_d4_records.py verify-staged "${expect[@]}"
  git commit -m "H7-R P1-A1: attempt records after an interruption (records only)"
  git push origin feat/h7-c1
  echo "__RECORDED__"
)
```

a1-G. 동등성 gate (`approved.gate` 뒤)
   - E1–E5를 차례로, 각각 별도 프로세스로 돌린다. 실행마다 기록과 로그를 따로 커밋한다.
   - 다섯 기록이 모두 커밋되면 `gate`가 판정한다. 결과는 PASS든 FAIL이든 커밋한다.
   - gate 실행이 하나라도 오류로 끝나면 멈춘다. 그 로그가 커밋된 뒤에는 블록을 다시 실행해도 거부한다. 보고 뒤 결정을 받는다.
   - 기록까지 끝났지만 커밋 전에 멈춘 실행은 다시 돌리지 않고 복사만 한다.

```bash
(
  set -euo pipefail
  cd ~/renju-h6
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src:$PWD/scripts"
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
  [ "$(git branch --show-current)" = feat/h7-c1 ] || { echo "STOP: wrong branch"; exit 1; }
  git pull --ff-only origin feat/h7-c1
  test -z "$(git status --porcelain --untracked-files=no)"
  python scripts/check_track_b_frozen.py
  D=docs/h7r/r2/R2-P1-POLICY-REPLAY-A1
  R=runs/h7r/R2-P1-POLICY-REPLAY-A1
  L="$R/logs"
  mkdir -p "$L"
  STAMP="$(date +%Y%m%d-%H%M%S)"
  tracked(){ git ls-files --error-unmatch "$1" >/dev/null 2>&1; }
  run(){  # log base, then the command; sets rc; the log ends with __EXIT__
    local b="$1"; shift
    if [ -e "$L/$b.log" ]; then echo "STOP: $L/$b.log exists"; exit 1; fi
    rc=0
    "$@" 2>&1 | tee "$L/$b.log" || rc=$?
    echo "__EXIT__ $rc" >> "$L/$b.log"
  }
  record(){  # message, log base ("" for none), then the record files
    local msg="$1" b="$2" e; shift 2
    local expect=()
    if [ -n "$b" ]; then
      python scripts/h7r_d4_records.py copy-logs --src-dir "$L" --pattern "$b.log" \
        --dest-dir "$D/stage-logs" --manifest "$D/stage-logs/$b.manifest.json"
      expect+=(--expect "$D/stage-logs/$b.log" --expect "$D/stage-logs/$b.manifest.json")
    fi
    for f in "$@"; do expect+=(--expect "$f"); done
    for e in "${expect[@]}"; do [ "$e" = --expect ] || git add "$e"; done
    python scripts/h7r_d4_records.py verify-staged "${expect[@]}"
    git commit -m "$msg"
    git push origin feat/h7-c1
  }
  if tracked "$D/gate.json"; then echo "STOP: the gate is already recorded"; exit 1; fi
  if [ -e "$D/gate.json" ]; then echo "STOP: an uncommitted gate.json exists: run a1-F"; exit 1; fi
  shopt -s nullglob
  for f in "$D"/stage-logs/gate-run-*.log; do
    if [ "$(tail -n 1 "$f")" != "__EXIT__ 0" ]; then
      echo "STOP: a gate run failed before ($f): report it (a gate run is not repeated)"; exit 1
    fi
  done
  for g in E1 E2 E3 E4 E5; do
    dst="$D/gate-records/$g/record.json"
    if tracked "$dst"; then echo "done: $g"; continue; fi
    b=""
    rc=0
    if [ ! -f "$R/gate/$g/record.json" ]; then
      b="gate-run-$g-$STAMP"
      run "$b" python -u scripts/h7r_p1a1.py gate-run --run "$g" --device cuda
    fi
    files=()
    if [ -f "$R/gate/$g/record.json" ]; then
      mkdir -p "$(dirname "$dst")"
      cp --no-clobber "$R/gate/$g/record.json" "$dst"
      cmp "$R/gate/$g/record.json" "$dst"
      files+=("$dst")
    fi
    record "H7-R P1-A1: gate run $g exit $rc (records only)" "$b" ${files[@]+"${files[@]}"}
    if [ "$rc" -ne 0 ]; then echo "STOP: gate run $g failed (exit $rc)"; exit 1; fi
  done
  b="gate-$STAMP"
  run "$b" python -u scripts/h7r_p1a1.py gate
  files=()
  if [ -f "$D/gate.json" ]; then files+=("$D/gate.json"); fi
  record "H7-R P1-A1: equivalence gate exit $rc (records only)" "$b" ${files[@]+"${files[@]}"}
  if [ "$rc" -ne 0 ]; then echo "STOP: the gate did not pass (exit $rc)"; exit 1; fi
  echo "__RECORDED__"
)
```

a1-2. 본학습 6회 (`approved.train` 뒤. 실행마다 따로 커밋한다)
   - 다시 실행하면 커밋된 실행은 건너뛴다. 기록까지 끝났지만 커밋 전에 멈춘 실행은 다시 학습하지 않고 복사만 한다.
   - 실행이 실패하면 끝난 실행은 보존하고 멈춘다.

```bash
(
  set -euo pipefail
  cd ~/renju-h6
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src:$PWD/scripts"
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
  [ "$(git branch --show-current)" = feat/h7-c1 ] || { echo "STOP: wrong branch"; exit 1; }
  git pull --ff-only origin feat/h7-c1
  test -z "$(git status --porcelain --untracked-files=no)"
  python scripts/check_track_b_frozen.py
  D=docs/h7r/r2/R2-P1-POLICY-REPLAY-A1
  R=runs/h7r/R2-P1-POLICY-REPLAY-A1
  L="$R/logs"
  mkdir -p "$L"
  STAMP="$(date +%Y%m%d-%H%M%S)"
  tracked(){ git ls-files --error-unmatch "$1" >/dev/null 2>&1; }
  run(){  # log base, then the command; sets rc; the log ends with __EXIT__
    local b="$1"; shift
    if [ -e "$L/$b.log" ]; then echo "STOP: $L/$b.log exists"; exit 1; fi
    rc=0
    "$@" 2>&1 | tee "$L/$b.log" || rc=$?
    echo "__EXIT__ $rc" >> "$L/$b.log"
  }
  record(){  # message, log base ("" for none), then the record files
    local msg="$1" b="$2" e; shift 2
    local expect=()
    if [ -n "$b" ]; then
      python scripts/h7r_d4_records.py copy-logs --src-dir "$L" --pattern "$b.log" \
        --dest-dir "$D/stage-logs" --manifest "$D/stage-logs/$b.manifest.json"
      expect+=(--expect "$D/stage-logs/$b.log" --expect "$D/stage-logs/$b.manifest.json")
    fi
    for f in "$@"; do expect+=(--expect "$f"); done
    for e in "${expect[@]}"; do [ "$e" = --expect ] || git add "$e"; done
    python scripts/h7r_d4_records.py verify-staged "${expect[@]}"
    git commit -m "$msg"
    git push origin feat/h7-c1
  }
  tracked "$D/gate.json" || { echo "STOP: the gate is not recorded (run a1-G first)"; exit 1; }
  for s in 20261024 20261111 20261112; do
    for arm in C T; do
      dst="$D/seed-records/$arm-$s/record.json"
      if tracked "$dst"; then echo "done: $arm-$s"; continue; fi
      b=""
      rc=0
      if [ ! -f "$R/$arm-$s/record.json" ]; then
        b="train-$arm-$s-$STAMP"
        run "$b" python -u scripts/h7r_p1a1.py train --arm "$arm" --seed "$s" --device cuda
      fi
      files=()
      if [ -f "$R/$arm-$s/record.json" ]; then
        mkdir -p "$(dirname "$dst")"
        cp --no-clobber "$R/$arm-$s/record.json" "$dst"
        cmp "$R/$arm-$s/record.json" "$dst"
        files+=("$dst")
      fi
      record "H7-R P1-A1: run $arm $s exit $rc (records only)" "$b" ${files[@]+"${files[@]}"}
      if [ "$rc" -ne 0 ]; then echo "STOP: run $arm $s failed (exit $rc)"; exit 1; fi
    done
  done
  echo "__RECORDED__"
)
```

a1-3. link와 Arena 계약 (클라우드 세션)
   - `link`는 커밋된 학습 기록만 읽으므로 클라우드에서 실행한다.
   - `approved.arena` 커밋 뒤 `arena-contract`로 `docs/h7r/contracts/R2-P1-A1-ARENA-1.json`을 만들어 커밋한다.

a1-4. Arena와 judge (Arena 계약 커밋 뒤, tmux 권장)
   - **대국 중에는 HEAD를 고정한다.** arm 기록의 signature에 실행 HEAD가 들어가기 때문이다.
     - 처음 시작할 때 HEAD를 `runs/.../arena/HEAD`에 적는다.
     - 끝나지 않은 arm이 있으면 pull하지 않고, HEAD가 그 값과 같아야 진행한다.
     - arm이 실패하면 아무것도 커밋하지 않고 멈춘다. 같은 블록을 다시 실행하면 같은 HEAD에서 journal로 이어서 둔다.
   - 모든 arm이 끝난 뒤에야 모든 시도의 대국 로그, judge 로그, `judge.json`을 커밋한다.
   - judge 결과를 보고 표본을 보충하거나 다시 두지 않는다.

```bash
(
  set -euo pipefail
  cd ~/renju-h6
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src:$PWD/scripts"
  [ "$(git branch --show-current)" = feat/h7-c1 ] || { echo "STOP: wrong branch"; exit 1; }
  D=docs/h7r/r2/R2-P1-POLICY-REPLAY-A1
  A=docs/h7r/contracts/R2-P1-A1-ARENA-1.json
  R=runs/h7r/R2-P1-POLICY-REPLAY-A1
  L="$R/logs"
  mkdir -p "$L" "$R/arena"
  tracked(){ git ls-files --error-unmatch "$1" >/dev/null 2>&1; }
  if tracked "$D/judge.json"; then echo "STOP: the judge is already recorded"; exit 1; fi
  if [ -e "$D/judge.json" ]; then echo "STOP: an uncommitted judge.json exists: run a1-F"; exit 1; fi
  test -z "$(git status --porcelain --untracked-files=no)"
  arms(){ python -c "import json, sys; print(' '.join(a['name'] for a in json.load(open(sys.argv[1]))['arms']))" "$A"; }
  left=()
  if [ -f "$A" ]; then for arm in $(arms); do [ -e "$R/arena/$arm.json" ] || left+=("$arm"); done; fi
  if [ -f "$R/arena/HEAD" ] && [ "${#left[@]}" -gt 0 ]; then
    if [ "$(git rev-parse HEAD)" != "$(cat "$R/arena/HEAD")" ]; then
      echo "STOP: HEAD moved since the Arena started ($(cat "$R/arena/HEAD")); restore that commit or report"; exit 1
    fi
    echo "resuming at $(cat "$R/arena/HEAD") (no pull while arms are unfinished)"
  else
    git pull --ff-only origin feat/h7-c1
    left=()
    for arm in $(arms); do [ -e "$R/arena/$arm.json" ] || left+=("$arm"); done
    if [ "${#left[@]}" -gt 0 ]; then git rev-parse HEAD > "$R/arena/HEAD"; fi
  fi
  python scripts/check_track_b_frozen.py
  STAMP="$(date +%Y%m%d-%H%M%S)"
  for arm in ${left[@]+"${left[@]}"}; do
    b="play-$arm-$STAMP"
    if [ -e "$L/$b.log" ]; then echo "STOP: $L/$b.log exists"; exit 1; fi
    rc=0
    python -u scripts/h7r_eval.py play --contract "$A" --arm "$arm" --device cuda --workers 6 \
      --jsonl "$R/arena/$arm.jsonl" --output "$R/arena/$arm.json" 2>&1 | tee "$L/$b.log" || rc=$?
    echo "__EXIT__ $rc" >> "$L/$b.log"
    if [ "$rc" -ne 0 ]; then
      echo "STOP: arm $arm failed (exit $rc); nothing committed, HEAD kept: rerun this block to resume"; exit 1
    fi
  done
  b="judge-$STAMP"
  rc=0
  python -u scripts/h7r_p1a1.py judge --arms-dir "$R/arena" 2>&1 | tee "$L/$b.log" || rc=$?
  echo "__EXIT__ $rc" >> "$L/$b.log"
  expect=()
  shopt -s nullglob
  for f in "$L"/play-*.log "$L/$b.log"; do
    n="$(basename "$f" .log)"
    tracked "$D/stage-logs/$n.log" && continue
    python scripts/h7r_d4_records.py copy-logs --resume --src-dir "$L" --pattern "$n.log" \
      --dest-dir "$D/stage-logs" --manifest "$D/stage-logs/$n.manifest.json"
    expect+=(--expect "$D/stage-logs/$n.log" --expect "$D/stage-logs/$n.manifest.json")
  done
  if [ -f "$D/judge.json" ]; then expect+=(--expect "$D/judge.json"); fi
  for e in "${expect[@]}"; do [ "$e" = --expect ] || git add "$e"; done
  python scripts/h7r_d4_records.py verify-staged "${expect[@]}"
  git commit -m "H7-R P1-A1: Arena and judge exit $rc (records only)"
  git push origin feat/h7-c1
  echo "__RECORDED__ $rc"
  exit "$rc"
)
```

## 8. 진행 기록

### 8.1 prepare: PASS (`03d3ec5`, 2026-11-10)

- 실행 HEAD `35f9334`(clean, prepare 승인 커밋), 실험 `faeb72b0f625`, 계약 파일 SHA `f6875a15707c`
- 계약의 기대값과 모두 같았다.
  - cycle 행: SHA, train 24,077행·val 1,423행
  - H3 checkpoint·metadata, c5 value checkpoint·metadata
  - RenjuNet cache: train 3,178,091국면, val 127,316국면, 텐서 해시, train·val 대국 겹침 없음
  - 근거 기록 9개(기존 P1 계약·재현 FAIL 기록·진단 결과 포함)
- 입력 파일 8개의 SHA는 기존 실험의 prepare(`78e8d38`)와 같다. 입력은 그때와 바뀌지 않았다.

### 8.2 동등성 gate: PASS (`b3eb145`, 2026-11-10)

- gate 실행 5회 모두 exit 0(`b488ea7`–`41a7f84`, 실행마다 커밋), 판정 exit 0. 실험 `faeb72b0f625`, 계약 파일 SHA `49763cfe3e25`(gate 승인 커밋)
- 검증: 커밋된 gate 기록 5개로 판정을 다시 계산했고 `gate.json`과 같았다. train 단계가 쓰는 재판정(`verified_gate`)도 통과한다.
- 수치 설정(5회 동일): torch 2.14.0+cu130, CUDA 13.0, cuDNN 9.24, RTX 5070, deterministic algorithms·cuDNN deterministic on, benchmark off, cuDNN TF32 on, matmul TF32 off, precision `highest`, 스레드 8, `CUBLAS_WORKSPACE_CONFIG=:4096:8`
- 엄격 검사: 5회 모두 위반 없음. 수치 설정·source·초기 state도 5회 모두 같았다.
- 정확 일치(허용 오차 0)

| 판정 | 결과 |
|---|---|
| G1 E2 = E3 | 차이 없음 |
| G1 E4 = E5 | 차이 없음 |
| G2 E1 = E2 | 차이 없음 |

| 실행 | 종류 | 선택 step | eligibility | 선택 state | 시간(초) |
|---|---|---:|---|---|---:|
| E1 | 기존 trainer | 77 | TTFFTTFTFF | `4e7939153b37` | 51 |
| E2 | C (replay off, 진단 on) | 77 | TTFFTTFTFF | `4e7939153b37` | 98 |
| E3 | C | 77 | TTFFTTFTFF | `4e7939153b37` | 99 |
| E4 | T (replay on, 진단 on) | 96 | 10회 모두 T | `58429752f6d9` | 101 |
| E5 | T | 96 | 10회 모두 T | `58429752f6d9` | 92 |

- E1의 손실·평가 곡선·선택 state·고정 국면 log-확률은 진단 실행 `old-det-1`(`0fe5aa1`)과도 비트 단위로 같다. 같은 환경에서 다른 날 다른 프로세스로 돌려도 같은 계산이다.
- E2–E5의 학습 시작 시 진단 KL은 0이다(시작 시 student = H3).
- 진단 연산을 켜도(E2) 기존 trainer(E1)와 비트 단위로 같았다. 진단에서 분리하지 못했던 추가 진단 연산의 영향이 결정론 설정에서는 없음을 직접 확인했다.
- c2 참고 기록(판정 아님): E1–E3은 평가 10부터 c2와 다르고(val objective 최대 1.05e-4), eligibility와 선택 step 77은 c2와 같다. E4·E5는 replay가 켜져 있으므로 c2와 다른 것이 당연하다.
- E4·E5의 결과는 replay 경로가 결정론 설정에서 돌고 반복된다는 확인일 뿐이다. 학습 효과로 읽지 않고, 그 state도 쓰지 않는다.
- 다음 단계는 train 승인 뒤의 a1-2(본학습 6회)다.

