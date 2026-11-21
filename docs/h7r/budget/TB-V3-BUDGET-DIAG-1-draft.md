# TB-V3-BUDGET-DIAG-1: S400이 E1-P에서 고른 증명 불가 수의 진단 (초안)

- 작성 2026-11-21. 상태: **계약 고정(fixed), 모든 단계 미승인**(prepare·replay·deepen 모두 false). 도구·테스트·데스크톱 블록 준비(§9, §10).
- 결정(2026-11-21 검토): B1·B2·B3의 10배 설정 유지, 집계가 재현되지 않아도 이번 replay의 UNKNOWN 수로 진단을 계속한다.
- 진단만 한다. 대국은 두지 않고, 배포 결정도 하지 않는다. TB-V3-BUDGET-1의 END는 그대로 결과로 남는다(§6).

## 0. 근거 (TB-V3-BUDGET-1 §13)

- 확인 대국에서 S400은 S50보다 +7.0%p [+4.0, +10.0]였다. 시간과 반칙 기준도 통과했다.
- E1 rescue rate가 12/15 대 15/15여서 END였다.
- 데스크톱 E1 집계(SHA가 `confirm.json`과 일치)에서 S400의 실패 3회는 모두 `TRUTH_UNRESOLVED`였다.
  - tree 경로에서 첫 후보가 UNKNOWN이었고, 최종 착수도 VCT2_CLEAR가 아니었다.
  - 세 수의 실제 안전 여부는 미확정이다.
- E1-P는 5국면 × 3 seed다: `8401-p4-white/11`, `8401-p16-white/11`, `8411-p8-white/11`, `8411-p22-white/13`, `8412-p4-white/11`.
- 국면별 결과(첫 수·최종 수·검사 사용량·적용 탐색량)는 기록돼 있지 않다.

## 1. 질문

1. 같은 5국면 × 3 seed에서 S50 15/15, S400 12/15와 `TRUTH_UNRESOLVED` 3회가 재현되는가. 재현된다면 어느 국면·seed인가.
2. 그 수들의 깊이 2 진실값은 더 큰 판정 예산에서 무엇인가: PROVEN_LOSS, VCT2_CLEAR, UNKNOWN.
3. UNKNOWN의 원인이 개별 VCF 호출 한도인가, 착수 전체 예산인가.
4. 그 실행에 전술 탐색량(800)이 실제로 적용됐는가.

- VCT2_CLEAR가 나와도 확인 범위는 깊이 2까지다("깊이 2 안에서 지지 않음"). 그 수가 좋은 수라는 뜻은 아니다.

## 2. 고정할 것

| 항목 | 값 |
|---|---|
| 모델 | 원본 H3 + H6 v3(TB-V3-BUDGET-1과 같음, `tb_v3.MODELS`) |
| 엔진 | `h7c_budget.make_budget_agent`, 배수 1(S50)과 8(S400), cuda |
| 국면·seed | E1 manifest의 E1-P 5국면과 각 국면의 `e1_seeds` 3개(무작위 요소 추가 없음) |
| 실행 경로 | `e1_evaluate.evaluate_position`과 같은 착수·분류(`outcome`) |
| 기준 판정 예산 B0 | `e1_build_suite.TRUTH_BUDGET`: VCF 호출당 20,000 노드, 착수당 10만 호출·1,000만 노드 |

## 3. 단계 (모두 미승인)

| 단계 | 하는 일 |
|---|---|
| prepare | 모델 파일 SHA, 커밋된 E1 manifest SHA(TB-V3-BUDGET-1이 쓴 manifest), TB-V3-BUDGET-1 confirm 기록 SHA |
| replay | 5국면 × 3 seed를 S50과 S400으로 각각 실행해 짝지어 기록. 재현 판정 |
| deepen | replay에서 UNKNOWN인 수만, 사전에 정한 더 큰 예산으로 깊이 2 진실값을 다시 계산 |
| report | §5 표 |

### replay에서 실행마다 남길 것

- 국면 key, seed, 배수
- 경로(`v8_route`), 적용 탐색량(`simulation_mode`, `selected_simulations`), 루트 전술 점수(`best_root_tactical_score`)
- 첫 후보(`pre`), 최종 착수, 전환 여부, VCT2 점검 결과, 루트 자식(수·방문·평균값)
- E1 분류(`outcome`)
- 첫 후보와 최종 착수 각각의 B0 진실값: 상태, 패배 깊이, `nodes_used`, `vcf_calls`, `vcf_cut_calls`, `budget_exhausted`

### 라벨 규칙과 재현 판정 (사전 고정)

- 라벨(분류에 쓰는 진실값)은 기존 E1의 조회 규칙을 그대로 쓴다. `e1_evaluate.evaluate_position`을 그대로 부르므로, manifest에 판정이 있으면 그것을 쓰고 없는 수만 B0로 계산한다.
- 중단 사유를 얻으려고 첫 후보와 최종 착수를 B0로 다시 검사한 값(`b0_recheck`)은 라벨과 따로 기록한다. 이 값은 분류를 바꾸지 않는다.
- S50의 E1-P 결과가 TREE_AVOIDED 15이고, S400이 TREE_AVOIDED 12 + TRUTH_UNRESOLVED 3이면 AGGREGATE_REPRODUCED다. 다르면 NOT_REPRODUCED다.
- 어느 경우든 확인되는 것은 **집계**뿐이다. 원래 실행은 국면별 기록을 남기지 않았으므로, 같은 국면·seed·착수가 재현됐는지는 확인할 수 없다.
- NOT_REPRODUCED여도 deepen은 이번 replay에서 UNKNOWN인 수로 진행한다. 원래 3회와 같은 수라는 말은 하지 않는다.

### deepen 예산 (사전 고정, 순서대로 모두 실행)

| 예산 | VCF 호출당 노드 | 착수당 호출 | 착수당 노드 | 늘린 한도 |
|---|---:|---:|---:|---|
| B1 | 200,000 | 100,000 | 10,000,000 | 개별 호출 한도만 10배 |
| B2 | 20,000 | 1,000,000 | 100,000,000 | 착수 전체 한도만 10배 |
| B3 | 200,000 | 1,000,000 | 100,000,000 | 둘 다 10배 |

- 수마다 B1·B2·B3을 모두 실행하고, 수별로 세 결과를 모두 남긴다(상태, 패배 깊이, `nodes_used`, `vcf_calls`, `vcf_cut_calls`, `budget_exhausted`, 초).
- `(국면, 착수)`가 같은 수는 한 번만 계산한다. 순서는 S400의 TRUTH_UNRESOLVED 실행의 수(첫 후보, 최종 착수)가 먼저이고, 그다음 나머지 UNKNOWN 수다.
- 결과를 보고 예산을 더 늘리지 않는다. 늘리려면 새 계약이다.
- 시간 상한
  - 수 하나·예산 하나당 30분. 계산은 별도 프로세스에서 하고, 상한에 이르면 프로세스를 실제로 종료한 뒤 `TIMEOUT`으로 남긴다. 판정기가 돌려준 `UNKNOWN`과 구분된다.
  - 전체 2시간. 남은 시간이 30분보다 적으면 그만큼만 기다린다. 상한에 이른 뒤 남은 항목은 `NOT_RUN`으로 기록한다.
- 원인 읽기(수별)
  - B1에서만 풀림: 개별 VCF 호출 한도를 늘리는 것으로 충분했다.
  - B2에서만 풀림: 착수 전체 한도를 늘리는 것으로 충분했다.
  - B3에서만 풀림: 시험한 조합 중 두 한도를 함께 늘린 경우에만 판정이 나왔다.
  - B1과 B2 모두 풀리거나 일부가 TIMEOUT·NOT_RUN이면 원인을 하나로 단정하지 않는다.

## 4. 범위 밖

- 대국, 배포 예산 변경, TB-V3-BUDGET-1 판정의 재해석
- 일반 탐색 400·전술 탐색 100 조합의 시험. §5의 결과로 근거가 생기면 별도 계약으로 연다.

## 5. 결과의 쓰임 (사전 고정)

| S400 실패 수의 B3 결과 | 기록 |
|---|---|
| PROVEN_LOSS (하나 이상) | 깊이 2 안의 실제 패배수다. 그 실행의 `simulation_mode`가 tactical(800)이었는지 함께 적는다. tactical이었다면 일반 400·전술 100 조합의 비교 근거가 된다(별도 계약) |
| 모두 VCT2_CLEAR | 기준 검사 한도(B0) 때문에 UNKNOWN이었고, 깊이 2 안에서는 지지 않는 수였다. 어느 한도(B1 / B2)에서 풀렸는지 적는다 |
| UNKNOWN이 남음 | B3로도 판정할 수 없었다고 적는다. 실패 수의 안전 여부는 미확정으로 남는다 |

- PROVEN_LOSS와 미확정 수가 섞이면 두 건수를 모두 보고한다.
- 어느 경우든 TB-V3-BUDGET-1의 END는 바뀌지 않는다.
- 배포를 다시 검토한다면 새 계약으로 한다. 전술 기준(UNKNOWN 처리 포함)은 이번 결과와 무관한 근거로 미리 정하고, 새 전술 표본을 쓴다. E1-P는 5국면뿐이고 이번에 결과를 보았다.

## 6. 비용

| 단계 | 예상 |
|---|---|
| replay | 30회 실행. E1 45국면 전체가 S400 2.5분, S50 1.3분이었으므로 수 분 |
| deepen | UNKNOWN 수 × 예산 3개, 수 하나·예산 하나에 최대 30분, 전체 상한 2시간 |

## 7. 결정 (2026-11-21)

1. deepen 예산 B1–B3(각 한도 10배), 수·예산당 30분, 전체 2시간.
2. NOT_REPRODUCED여도 이번 replay의 UNKNOWN 수로 deepen을 진행한다.

## 8. 계약

- `docs/h7r/budget/TB-V3-BUDGET-DIAG-1/diag-contract.json`(fixed, 모든 승인 false, 실험 SHA `7b3016f0dc16`). 모델 SHA, 커밋된 E1 manifest와 TB-V3-BUDGET-1 confirm 기록의 SHA, B0–B3, 시간 상한, 기대 집계, 도구와 재사용 파일 7개(sources)를 고정한다.

## 9. 구현 (2026-11-21, 미승인)

- 도구 `scripts/h7r_budget_diag.py`

| 명령 | 단계 | 하는 일 |
|---|---|---|
| `prepare` | prepare | 모델 파일 = canonical, E1 manifest·TB-V3-BUDGET-1 confirm 기록이 고정 SHA이고 커밋돼 있음, 그 confirm이 E1 gate의 END |
| `replay` | replay | 5국면 × 3 seed × (S50, S400). `evaluate_position`을 그대로 부르고 실행마다 §3의 필드와 `b0_recheck`. 집계 재현 판정. 국면 단위로 이어하기 |
| `deepen` | deepen | UNKNOWN 수를 한 번씩 B1·B2·B3로, 별도 프로세스·시간 상한·전체 상한. 수별 결과와 §5 보고. 이어하기 |
| `verify --stage S` | — | 기록의 근거(실행 commit의 승인된 계약 SHA)와 파일 대조, replay 요약·deepen 보고 재계산 |

- 테스트 `tests/test_h7r_budget_diag.py`
  - 집계 재현 판정, 항목 순서와 중복 제거, 원인 읽기, 보고 표
  - 실제 프로세스 종료로 TIMEOUT, 전체 상한에서 NOT_RUN
  - 실제 `evaluate_position`으로 라벨 규칙과 `b0_recheck` 분리
  - prepare → replay → deepen → verify 흐름, NOT_REPRODUCED에서도 deepen
  - 데스크톱 블록(실제 bash, 도구는 대역)과 실제 도구로 d-P 실행

- 2026-11-21 검토 반영(`7789dd6` 뒤)
  - `verify deepen`은 요약이 아니라 원본 journal에서 예산별 결과, `NOT_RUN`, 사용 시간, 원인 읽기, 최종 판정, 보고를 다시 만들어 요약 전체와 대조한다. deepen도 같은 함수로 기록을 만든다.
  - journal의 마지막 줄이 쓰다 잘렸으면 잘라 내고 이어간다(그 앞 줄이 깨졌으면 거부).
  - 진행 중 작업은 30초마다 경과 시간을 journal에 남긴다. 끝나지 못한 작업은 마지막 경과 시간 + 30초(그 작업의 상한 이내)를 사용 시간에 넣고, 다시 할 때는 남은 시간으로만 기다린다.
  - 같은 `(국면, 착수)`는 한 번만 계산하되, 그 수를 고른 모든 실행(seed, 첫 후보·최종 착수 구분, 결과, `simulation_mode`, `selected_simulations`)을 남긴다. 후속 비교의 근거(`tactical_800_final_move_with_proven_loss`)는 패배가 증명된 수를 **최종 착수**로 둔 S400 실행 중 전술 800이 실제 적용된 것이 있는지로 정한다.
  - 예산끼리 확정 판정이 엇갈리면(어느 예산은 PROVEN_LOSS, 다른 예산은 VCT2_CLEAR) 최종 상태를 `CONFLICT`로 두고, 원본 결과를 읽도록 보고한다.

## 10. 데스크톱 실행 (승인 뒤에만)

- d-P는 `approved.prepare` 뒤, d-R은 `approved.replay` 뒤, d-D는 `approved.deepen` 뒤에 실행한다. 블록 방식은 TB-V3-BUDGET-1과 같다(로그 이름, `verify` 뒤 기록, 시작 HEAD 고정, 실패 시 커밋 없음).
- d-R은 수 분, d-D는 최대 약 2시간이다(tmux 권장).

d-P. prepare

```bash
(
  set -euo pipefail
  cd ~/renju-h6
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src:$PWD/scripts"
  unset CUBLAS_WORKSPACE_CONFIG
  [ "$(git branch --show-current)" = feat/h7-c1 ] || { echo "STOP: wrong branch"; exit 1; }
  git diff --quiet || { echo "STOP: unstaged tracked changes"; exit 1; }
  N=prepare
  D=docs/h7r/budget/TB-V3-BUDGET-DIAG-1
  R=runs/h7r/TB-V3-BUDGET-DIAG-1
  L="$R/logs"
  T=scripts/h7r_budget_diag.py
  O="$D/$N.json"
  PIN="$R/$N.HEAD"  # the commit the stage started on: its games resume at that commit only
  tracked(){ git cat-file -e "HEAD:$1" 2>/dev/null; }
  field(){ python -c 'import json, sys; v = json.loads(sys.argv[1]).get(sys.argv[2]); print("" if v is None else v)' "$1" "$2"; }
  mkdir -p "$L"
  command -v flock > /dev/null || { echo "STOP: flock is missing (util-linux)"; exit 1; }
  exec 9> "$L/$N.lock"
  flock -n 9 || { echo "STOP: another $N block is running"; exit 1; }
  if tracked "$O"; then rm -f "$PIN"; echo "STOP: $N is already recorded"; exit 1; fi
  if [ -f "$PIN" ]; then
    [ "$(git rev-parse HEAD)" = "$(cat "$PIN")" ] || {
      echo "STOP: HEAD moved since $N started at $(cat "$PIN"): restore that commit or report"; exit 1; }
    echo "resuming $N at $(cat "$PIN") (no pull)"
  elif git diff --cached --quiet; then
    git pull --ff-only origin feat/h7-c1
  fi
  :  # the first stage
  rc=0
  if [ ! -e "$O" ]; then
    git diff --cached --quiet || { echo "STOP: resolve staged changes before a new run"; exit 1; }
    [ -f "$PIN" ] || git rev-parse HEAD > "$PIN"
    python scripts/check_track_b_frozen.py
    log="$(mktemp "$L/$N-XXXXXXXX.log")"
    set +e
    python -u "$T" "$N" --device cuda --log-name "$(basename "$log")" 2>&1 | tee "$log"
    codes=("${PIPESTATUS[@]}")
    set -e
    rc="${codes[0]}"
    [ "${codes[1]}" -eq 0 ] || { echo "STOP: tee failed; log: $log"; exit 1; }
    echo "__EXIT__ $rc" >> "$log"
  else
    echo "an uncommitted $O exists: recording it (no new run)"
  fi
  if [ ! -f "$O" ]; then  # nothing is committed: a commit would move HEAD and the games could not resume
    echo "__FAILED__ $rc"
    echo "STOP: $N ended without its record (exit $rc); HEAD kept: fix the cause or report, then rerun this block"
    exit 1
  fi
  v="$(python "$T" verify --stage "$N")" || { echo "STOP: $N: the record does not check against its files; nothing committed"; exit 1; }
  name="$(field "$v" log_name)"
  rc="$(field "$v" exit_code)"
  [[ "$name" =~ ^$N-[A-Za-z0-9]+\.log$ ]] || { echo "STOP: the record names no attempt log"; exit 1; }
  log="$L/$name"
  [ -f "$log" ] || { echo "STOP: the record's log is missing: $log"; exit 1; }
  case "$(tail -n 1 "$log")" in
    "__EXIT__ $rc") ;;
    __EXIT__*) echo "STOP: the record and its log disagree on the exit code"; exit 1 ;;
    *) printf '\n%s\n__EXIT__ %s\n' "Recovered exit code from $O" "$rc" >> "$log" ;;
  esac
  paths=("$O")
  shopt -s nullglob
  for f in "$L/$N"-*.log; do
    [[ "$(tail -n 1 "$f")" =~ ^__EXIT__\ [0-9]+$ ]] || continue
    n="$(basename "$f" .log)"
    tracked "$D/stage-logs/$n.log" && continue
    python scripts/h7r_d4_records.py copy-logs --resume --src-dir "$L" --pattern "$n.log" \
      --dest-dir "$D/stage-logs" --manifest "$D/stage-logs/$n.manifest.json"
    paths+=("$D/stage-logs/$n.log" "$D/stage-logs/$n.manifest.json")
  done
  while IFS= read -r -d '' staged; do
    allowed=false
    for f in "${paths[@]}"; do if [ "$f" = "$staged" ]; then allowed=true; break; fi; done
    "$allowed" || { echo "STOP: unrelated staged path: $staged"; exit 1; }
  done < <(git diff --cached --name-only -z)
  expect=()
  for f in "${paths[@]}"; do expect+=(--expect "$f"); done
  git add -- "${paths[@]}"
  python scripts/h7r_d4_records.py verify-staged "${expect[@]}"
  git commit -m "H7-R BUDGET DIAG: $N exit $rc (records only)"
  rm -f "$PIN"
  git push origin feat/h7-c1
  echo "__RECORDED__ $rc"
)
```

d-R. replay

```bash
(
  set -euo pipefail
  cd ~/renju-h6
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src:$PWD/scripts"
  unset CUBLAS_WORKSPACE_CONFIG
  [ "$(git branch --show-current)" = feat/h7-c1 ] || { echo "STOP: wrong branch"; exit 1; }
  git diff --quiet || { echo "STOP: unstaged tracked changes"; exit 1; }
  N=replay
  D=docs/h7r/budget/TB-V3-BUDGET-DIAG-1
  R=runs/h7r/TB-V3-BUDGET-DIAG-1
  L="$R/logs"
  T=scripts/h7r_budget_diag.py
  O="$D/$N.json"
  PIN="$R/$N.HEAD"  # the commit the stage started on: its games resume at that commit only
  tracked(){ git cat-file -e "HEAD:$1" 2>/dev/null; }
  field(){ python -c 'import json, sys; v = json.loads(sys.argv[1]).get(sys.argv[2]); print("" if v is None else v)' "$1" "$2"; }
  mkdir -p "$L"
  command -v flock > /dev/null || { echo "STOP: flock is missing (util-linux)"; exit 1; }
  exec 9> "$L/$N.lock"
  flock -n 9 || { echo "STOP: another $N block is running"; exit 1; }
  if tracked "$O"; then rm -f "$PIN"; echo "STOP: $N is already recorded"; exit 1; fi
  if [ -f "$PIN" ]; then
    [ "$(git rev-parse HEAD)" = "$(cat "$PIN")" ] || {
      echo "STOP: HEAD moved since $N started at $(cat "$PIN"): restore that commit or report"; exit 1; }
    echo "resuming $N at $(cat "$PIN") (no pull)"
  elif git diff --cached --quiet; then
    git pull --ff-only origin feat/h7-c1
  fi
  tracked "$D/prepare.json" || { echo "STOP: prepare is not recorded (run d-P first)"; exit 1; }
  rc=0
  if [ ! -e "$O" ]; then
    git diff --cached --quiet || { echo "STOP: resolve staged changes before a new run"; exit 1; }
    [ -f "$PIN" ] || git rev-parse HEAD > "$PIN"
    python scripts/check_track_b_frozen.py
    log="$(mktemp "$L/$N-XXXXXXXX.log")"
    set +e
    python -u "$T" "$N" --device cuda --log-name "$(basename "$log")" 2>&1 | tee "$log"
    codes=("${PIPESTATUS[@]}")
    set -e
    rc="${codes[0]}"
    [ "${codes[1]}" -eq 0 ] || { echo "STOP: tee failed; log: $log"; exit 1; }
    echo "__EXIT__ $rc" >> "$log"
  else
    echo "an uncommitted $O exists: recording it (no new run)"
  fi
  if [ ! -f "$O" ]; then  # nothing is committed: a commit would move HEAD and the games could not resume
    echo "__FAILED__ $rc"
    echo "STOP: $N ended without its record (exit $rc); HEAD kept: fix the cause or report, then rerun this block"
    exit 1
  fi
  v="$(python "$T" verify --stage "$N")" || { echo "STOP: $N: the record does not check against its files; nothing committed"; exit 1; }
  name="$(field "$v" log_name)"
  rc="$(field "$v" exit_code)"
  [[ "$name" =~ ^$N-[A-Za-z0-9]+\.log$ ]] || { echo "STOP: the record names no attempt log"; exit 1; }
  log="$L/$name"
  [ -f "$log" ] || { echo "STOP: the record's log is missing: $log"; exit 1; }
  case "$(tail -n 1 "$log")" in
    "__EXIT__ $rc") ;;
    __EXIT__*) echo "STOP: the record and its log disagree on the exit code"; exit 1 ;;
    *) printf '\n%s\n__EXIT__ %s\n' "Recovered exit code from $O" "$rc" >> "$log" ;;
  esac
  paths=("$O")
  shopt -s nullglob
  for f in "$L/$N"-*.log; do
    [[ "$(tail -n 1 "$f")" =~ ^__EXIT__\ [0-9]+$ ]] || continue
    n="$(basename "$f" .log)"
    tracked "$D/stage-logs/$n.log" && continue
    python scripts/h7r_d4_records.py copy-logs --resume --src-dir "$L" --pattern "$n.log" \
      --dest-dir "$D/stage-logs" --manifest "$D/stage-logs/$n.manifest.json"
    paths+=("$D/stage-logs/$n.log" "$D/stage-logs/$n.manifest.json")
  done
  while IFS= read -r -d '' staged; do
    allowed=false
    for f in "${paths[@]}"; do if [ "$f" = "$staged" ]; then allowed=true; break; fi; done
    "$allowed" || { echo "STOP: unrelated staged path: $staged"; exit 1; }
  done < <(git diff --cached --name-only -z)
  expect=()
  for f in "${paths[@]}"; do expect+=(--expect "$f"); done
  git add -- "${paths[@]}"
  python scripts/h7r_d4_records.py verify-staged "${expect[@]}"
  git commit -m "H7-R BUDGET DIAG: $N exit $rc (records only)"
  rm -f "$PIN"
  git push origin feat/h7-c1
  echo "__RECORDED__ $rc"
)
```

d-D. deepen (tmux)

```bash
(
  set -euo pipefail
  cd ~/renju-h6
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src:$PWD/scripts"
  unset CUBLAS_WORKSPACE_CONFIG
  [ "$(git branch --show-current)" = feat/h7-c1 ] || { echo "STOP: wrong branch"; exit 1; }
  git diff --quiet || { echo "STOP: unstaged tracked changes"; exit 1; }
  N=deepen
  D=docs/h7r/budget/TB-V3-BUDGET-DIAG-1
  R=runs/h7r/TB-V3-BUDGET-DIAG-1
  L="$R/logs"
  T=scripts/h7r_budget_diag.py
  O="$D/$N.json"
  PIN="$R/$N.HEAD"  # the commit the stage started on: its games resume at that commit only
  tracked(){ git cat-file -e "HEAD:$1" 2>/dev/null; }
  field(){ python -c 'import json, sys; v = json.loads(sys.argv[1]).get(sys.argv[2]); print("" if v is None else v)' "$1" "$2"; }
  mkdir -p "$L"
  command -v flock > /dev/null || { echo "STOP: flock is missing (util-linux)"; exit 1; }
  exec 9> "$L/$N.lock"
  flock -n 9 || { echo "STOP: another $N block is running"; exit 1; }
  if tracked "$O"; then rm -f "$PIN"; echo "STOP: $N is already recorded"; exit 1; fi
  if [ -f "$PIN" ]; then
    [ "$(git rev-parse HEAD)" = "$(cat "$PIN")" ] || {
      echo "STOP: HEAD moved since $N started at $(cat "$PIN"): restore that commit or report"; exit 1; }
    echo "resuming $N at $(cat "$PIN") (no pull)"
  elif git diff --cached --quiet; then
    git pull --ff-only origin feat/h7-c1
  fi
  tracked "$D/replay.json" || { echo "STOP: replay is not recorded (run d-R first)"; exit 1; }
  rc=0
  if [ ! -e "$O" ]; then
    git diff --cached --quiet || { echo "STOP: resolve staged changes before a new run"; exit 1; }
    [ -f "$PIN" ] || git rev-parse HEAD > "$PIN"
    python scripts/check_track_b_frozen.py
    log="$(mktemp "$L/$N-XXXXXXXX.log")"
    set +e
    python -u "$T" "$N" --device cuda --log-name "$(basename "$log")" 2>&1 | tee "$log"
    codes=("${PIPESTATUS[@]}")
    set -e
    rc="${codes[0]}"
    [ "${codes[1]}" -eq 0 ] || { echo "STOP: tee failed; log: $log"; exit 1; }
    echo "__EXIT__ $rc" >> "$log"
  else
    echo "an uncommitted $O exists: recording it (no new run)"
  fi
  if [ ! -f "$O" ]; then  # nothing is committed: a commit would move HEAD and the games could not resume
    echo "__FAILED__ $rc"
    echo "STOP: $N ended without its record (exit $rc); HEAD kept: fix the cause or report, then rerun this block"
    exit 1
  fi
  v="$(python "$T" verify --stage "$N")" || { echo "STOP: $N: the record does not check against its files; nothing committed"; exit 1; }
  name="$(field "$v" log_name)"
  rc="$(field "$v" exit_code)"
  [[ "$name" =~ ^$N-[A-Za-z0-9]+\.log$ ]] || { echo "STOP: the record names no attempt log"; exit 1; }
  log="$L/$name"
  [ -f "$log" ] || { echo "STOP: the record's log is missing: $log"; exit 1; }
  case "$(tail -n 1 "$log")" in
    "__EXIT__ $rc") ;;
    __EXIT__*) echo "STOP: the record and its log disagree on the exit code"; exit 1 ;;
    *) printf '\n%s\n__EXIT__ %s\n' "Recovered exit code from $O" "$rc" >> "$log" ;;
  esac
  paths=("$O")
  shopt -s nullglob
  for f in "$L/$N"-*.log; do
    [[ "$(tail -n 1 "$f")" =~ ^__EXIT__\ [0-9]+$ ]] || continue
    n="$(basename "$f" .log)"
    tracked "$D/stage-logs/$n.log" && continue
    python scripts/h7r_d4_records.py copy-logs --resume --src-dir "$L" --pattern "$n.log" \
      --dest-dir "$D/stage-logs" --manifest "$D/stage-logs/$n.manifest.json"
    paths+=("$D/stage-logs/$n.log" "$D/stage-logs/$n.manifest.json")
  done
  while IFS= read -r -d '' staged; do
    allowed=false
    for f in "${paths[@]}"; do if [ "$f" = "$staged" ]; then allowed=true; break; fi; done
    "$allowed" || { echo "STOP: unrelated staged path: $staged"; exit 1; }
  done < <(git diff --cached --name-only -z)
  expect=()
  for f in "${paths[@]}"; do expect+=(--expect "$f"); done
  git add -- "${paths[@]}"
  python scripts/h7r_d4_records.py verify-staged "${expect[@]}"
  git commit -m "H7-R BUDGET DIAG: $N exit $rc (records only)"
  rm -f "$PIN"
  git push origin feat/h7-c1
  echo "__RECORDED__ $rc"
)
```
