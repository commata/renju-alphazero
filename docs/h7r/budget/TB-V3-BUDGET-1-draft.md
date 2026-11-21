# TB-V3-BUDGET-1: 같은 모델에 탐색 시간을 더 주면 배포 엔진이 강해지는가 (초안)

- 작성 2026-11-21. 상태: **계약 고정(fixed), 모든 단계 미승인**(prepare·timing·confirm 모두 false). 도구·테스트·데스크톱 블록 준비(§10, §12). 확인 대국은 아직 실행하지 않는다.
- 결정(2026-11-21): 학습이 아니라 탐색 예산으로 간다(제안 A). 목표 장치는 데스크톱 GPU, 수당 시간 상한은 현 canonical(S50)의 5.5배(평균과 p95 모두).
- 결정(2026-11-21 검토): 모델은 원본 H3 + H6 v3, 확인 표본은 `O1-confirm-main` 300쌍(사용한 동치류를 기록), 시간 상한은 평균·p95 모두 5.5배.
- 이 계약은 개발이 아니라 **확인**까지다. 통과한 예산을 canonical로 바꾸는 일(엔진 버전 등록, 웹 SER5 CPU 인증과 smoke)은 별도 승격 단계다. GPU 학습은 하지 않는다.

## 0. 근거

- S400 대 S50(같은 모델, 수당 시간만 다름)은 다섯 번 측정됐고 모두 같은 방향이다.

| 측정 | 모델 | 쌍 | S400 − S50 | 95% 구간 | 수당 시간 |
|---|---|---:|---:|---|---:|
| SB `docs/h7c/sb_summary.json` | 원본 H3 + H6 v3 | 50 | +9.0%p | [−1, +19] | 5.07배 |
| SB 포화 `docs/h7c/e2/sb_c5_saturation.json` | H3 + c5 | 150 | +7.67%p | [+2.5, +12.67] | 4.94배(50쌍) |
| R1 pilot O0 | H3 + c5 | 160 | +12.19%p | [+7.50, +16.88] | — |
| R1 pilot O1 | H3 + c5 | 160 | +8.44%p | [+4.06, +12.81] | — |
| R1 pilot O2 | H3 + c5 | 160 | +12.97%p | [+8.44, +17.50] | — |

- 원본 모델 SB의 S200은 +11.5%p [+2.0, +20.5](50쌍), 수당 시간 2.93배였다.
- 학습 쪽 시도(R2·R3)는 champion을 넘는 후보를 만들지 못했다(R2-S400-TRAIN, R2-S400-PRESERVE-1, R3-POLICY-LOOP-1).
- 위 수당 시간은 대국 기록의 **착수 시간 합의 비율**이다(tested / opponent, 데스크톱 GPU 6 worker). 이 계약의 비율은 **수당 평균의 비율**이라 정의가 달라 선행 참고값으로만 쓴다. p95는 기록돼 있지 않다(기존 Arena는 상대 측 착수 시간을 남기지 않는다).
- 확인되지 않은 것
  - 배포 모델 쌍(원본 H3 + H6 v3)에서 S400의 이득은 50쌍 한 번뿐이다(구간이 0을 포함).
  - 새 시작 국면과 새 seed에서의 확인, 금수·전술 gate, 수당 시간 p95는 아직 없다.

## 1. 질문과 범위

- 질문: 수당 시간 상한(5.5배) 안에서 가장 큰 탐색 예산 하나를 고르면, 같은 모델의 현 canonical(S50)보다 새 시작 국면에서 강하고 gate를 모두 통과하는가.
- 범위
  - 모델은 바꾸지 않는다. 바뀌는 것은 tree 예산(simulations, tactical_simulations)의 배수뿐이다(`h7c_budget.make_budget_agent`, `hybrid.h7c_retarget.budget`).
  - 데스크톱 GPU 기준이다. 웹 SER5 CPU의 시간과 인증은 승격 단계에서 따로 잰다.
  - 예산은 S100 / S200 / S400(배수 2 / 4 / 8) 중 하나다. S800(약 10배)은 상한 밖이라 넣지 않는다.

## 2. 고정할 것

| 항목 | 값 |
|---|---|
| 모델 쌍 | 배포 canonical의 원본 H3 policy(`efee483284df`) + H6 v3 value(`a0fcccc401a8`), 양쪽 같음(결정) |
| 엔진·runtime | frozen TB-V3 Resolve, fast5, CUDA. 배수 1은 canonical과 같은 구성 |
| 후보 예산 | S400 → S200 → S100 순서로 시간 측정, 상한을 지키는 가장 큰 예산 하나(§4) |
| 상대 | 같은 모델의 S50(현 canonical) |
| 수당 시간 상한 | 평균 비율 ≤ 5.5, p95 비율 ≤ 5.5(§4의 정의) |
| 확인 suite | `O1-confirm-main`(560개, 동치류 558개, 지금까지 어떤 계약도 쓰지 않음)에서 `pick`(seed 9111)으로 300개: 동치류 299개, O1-dev-a–d와 겹침 0, `O1-confirm-anchor` 전체와 동치류 6개 겹침(prepare가 계산해 기록. 앞으로 독립 확인을 주장하는 계약은 anchor에서 이 6개를 뺀다) |
| 확인 규모 | 300쌍(600판), worker 6 |
| 통계 | opening 쌍 bootstrap 20,000회, percentile, 단측 α 0.025, class 단위 민감도 함께 보고 |

## 3. 단계 (모두 미승인)

| 단계 | 하는 일 | 쓰는 표본 |
|---|---|---|
| prepare | 모델 파일 SHA(= `tb_v3.MODELS`), frozen checker, suite 파일 SHA와 검사, seed 충돌 없음 | — |
| timing | 후보 예산별 시간 측정 대국(S_k 대 S50, 20쌍), 시간 비율만 기록. 점수는 판독하지 않는다 | O1-dev-c(이미 결과를 본 개발 suite) |
| select | §4 규칙으로 예산 하나를 기계적으로 정한다. 어느 예산도 상한을 못 지키면 END_NO_BUDGET | timing 기록 |
| confirm | 고른 예산의 확인 Arena(S_k 대 S50, 300쌍) | O1-confirm-main |
| gates | 전술(E1 suite를 예산 agent로), 금수 오류 0, 확인 대국의 수당 시간 | 고정 probe와 확인 대국 |
| judge | §6 판독표 | — |

- timing과 confirm은 서로 다른 suite·seed를 쓴다. timing 대국의 점수는 기록만 하고 선택에 쓰지 않는다.

## 4. 예산 선택 규칙 (사전 고정)

- 시간 기록: 새 대국 함수가 **양쪽 모든 착수**의 쪽(tested / opponent), 예산 배수, 경로, 초를 남긴다. 색을 바꾼 두 판에서 시간이 그 수를 둔 예산에 귀속되는지(색으로 정한 쪽과 기록된 쪽·배수가 같은지) 대국마다 검사한다. 시간이 빠진 수가 하나라도 있으면 그 대국을 받지 않고, 예산을 고르지 않는다.
- 시간 정의(같은 대국 안에서 잼, worker 6 고정: 수당 시간은 worker 수에 따라 달라진다)
  - 평균 비율 = S_k가 둔 수의 평균 착수 시간 / 같은 대국들에서 S50이 둔 수의 평균 착수 시간
  - p95 비율 = S_k 착수 시간의 95번째 백분위 / S50 착수 시간의 95번째 백분위
- 순서: S400을 먼저 잰다. 두 비율이 모두 5.5 이하면 S400을 고르고 멈춘다. 아니면 S200, 그다음 S100.
- 어느 예산도 통과하지 못하면 END_NO_BUDGET으로 끝낸다.
- 고른 뒤에는 바꾸지 않는다. 확인 결과를 보고 다른 예산으로 다시 확인하지 않는다(새 계약이다).

## 5. 확인 Arena

- arm 하나: S_k(원본 쌍) 대 S50(원본 쌍), O1-confirm-main에서 seed 9111로 300개를 뽑아 색을 바꿔 두 판씩.
- 판독: P = S_k 점수 − 0.5. STRONGER(하한 > 0) / WEAKER(상한 < 0) / NOT_ESTABLISHED.
- 기대 크기(계획용, 보장 아님): 이전 측정의 +7.7–13%p와 pair SD 0.28–0.33이면 300쌍의 단측 하한은 대략 +4–10%p다.

## 6. gate와 판독 (사전 고정)

| 항목 | 통과 조건 |
|---|---|
| 강도 | P STRONGER |
| 금수·반칙 | 확인 대국 600판에서 시험 측 반칙 0, 상대 반칙 쌍 0 |
| 전술 | E1 dev suite를 S_k agent와 S50 agent로 각각 두어(예산 agent로 실행), S_k의 unsound witness 0, rescue rate ≥ S50, veto recall ≥ S50, false veto ≤ S50. 비율은 기록된 분자·분모로 정확히 비교한다. 양쪽 모두 분모가 없으면 회귀가 아니고, 한쪽만 없으면 비교할 수 없어 실패다. value·policy probe는 모델이 같아 재지 않는다 |
| 수당 시간 | 확인 대국에서 평균 비율 ≤ 5.5, p95 비율 ≤ 5.5(§4의 정의) |

| 결과 | 기록과 다음 |
|---|---|
| 모두 통과 | PASS. S_k를 canonical 후보로 기록한다. 승격(엔진 버전 등록, 웹 SER5 CPU 시간·인증, 웹 smoke)은 별도 계약 |
| P가 STRONGER가 아님 | END. 이 예산의 이득이 새 국면에서 확인되지 않았다고 기록. canonical 유지 |
| gate 하나라도 실패 | END(실패한 gate를 기록). canonical 유지 |
| END_NO_BUDGET | 상한 안의 예산이 없었다고 기록. canonical 유지 |

- 확인은 한 번이다. 결과를 보고 쌍 수·suite·seed·예산을 바꾸지 않는다.
- 결과에는 두 시간 비율과 함께 양쪽의 실제 초/수(평균, p95)를 적는다(체감 대기 시간).
- 기존 `h7c_arena.tactical_judge`를 쓰지 않는 이유: E1 지표가 나빠져도 보고만 하고 PASS를 낼 수 있고, 그 명령은 두 쪽 모델이 같으면 거부한다.

## 7. 해석의 한계

- 같은 모델에서 시간을 더 쓰는 비교다. "엔진이 같은 시간에 더 강해졌다"는 뜻이 아니다.
- 답하는 질문은 "5.5배 안에서 고른 예산이 S50보다 강한가"까지다. 세 예산끼리 대국하지 않으므로, PASS여도 "고른 예산이 셋 중 가장 강하다"고 쓰지 않는다.
- 상대는 같은 엔진의 S50이다. 외부 엔진(Rapfi) 대비 강도는 재지 않는다(기록용으로 따로 둘 수 있다).
- 데스크톱 GPU 기준이다. 웹 SER5 CPU에서는 배수가 다를 수 있고, 승격 단계에서 다시 잰다.

## 8. seed (2026-11-21 미사용 확인)

| 용도 | 값 | 확인 |
|---|---|---|
| 확인 Arena 대국 seed (v2 `ARENA_SEEDS` 9101–9140) | 9111 | 쓰인 v2 seed는 9101–9110 |
| timing 대국 seed (같은 범위) | 9112(S400), 9113(S200), 9114(S100) | 미사용 |
| bootstrap seed | 20261221 | 저장소에 없음 |

## 9. 비용

| 단계 | 예상 |
|---|---|
| timing | S400 20쌍 약 10분(R1 pilot S400 쌍당 22–29초, worker 6). S400이 통과하지 못하면 S200 약 5분 더 |
| confirm | S400이면 300쌍 약 1.8–2.4시간, S200이면 약 1–1.3시간 |
| gates | E1 tactical suite 두 번(S_k, S50) 수십 분 |
| 합계 | 약 3–4시간 |

## 10. 구현 (2026-11-21, 미승인)

- 계약 `docs/h7r/budget/TB-V3-BUDGET-1/budget-contract.json`(fixed, 모든 승인 false, 실험 SHA `5a25248ed4bb`). 모델 SHA, 세 suite의 파일 SHA, 예산 순서, 상한, 표본 크기, seed, 통계, 전술 gate, 판독표, 도구와 재사용 파일 8개(sources)를 고정한다.
- 도구 `scripts/h7r_budget.py`(기존 v2 검사기와 확인 계약 규칙은 쓰지 않는 이 계약 전용 검사. 대국 작업 생성·쌍 집계·bootstrap·E1 요약·예산 agent는 재사용)

| 명령 | 단계 | 하는 일 |
|---|---|---|
| `prepare` | prepare | 모델 파일 = canonical(`tb_v3.MODELS`), suite 파일 SHA·커밋·`check_suite`, 확인 표본 계산(opening ID, 동치류 수, anchor·개발 suite와 겹치는 동치류), seed 미사용, 빈 runs 폴더 |
| `timing` | timing | S400 → S200 → S100 순서로 O1-dev-c 20쌍(seed 9112 / 9113 / 9114), 양쪽 모든 착수 시간, §4 선택. 기록이 선택한 예산과 확인 표본(opening ID 300개, seed, 모델 SHA)을 첫 확인 대국 전에 고정한다. 점수는 판독하지 않는다 |
| `confirm` | confirm | 고정된 표본만(다시 계산한 `pick`이 기록과 같아야 시작), 고른 예산 대 S50 300쌍(이어하기는 같은 서명·같은 표본만), E1을 예산 agent와 S50 agent로, §6 판독 |
| `verify --stage S` | — | 단계 기록의 근거를 대조한다(스테이징 전). 모든 단계는 실험 SHA, clean 실행 commit, 그 commit의 계약 파일 SHA와 그 단계 승인. prepare는 `passed`, 모델 식별값, 다시 계산한 확인 표본. timing·confirm은 대국 파일·E1 결과로 다시 계산 |

- 2026-11-21 검토 반영(`fdb60cb` 뒤)
  - b-P가 먼저 쓰는 `prepare.HEAD`가 현재 HEAD를 가리키면 잔여 파일로 보지 않는다. 다른 commit의 pin이나 다른 파일은 그대로 거부한다.
  - confirm은 첫 대국 전에 timing 원본으로 선택을 다시 계산하고, `selected_factor`·`selected`·`confirmation.factor`가 같은 예산이어야 시작한다. `verify confirm`도 대국 배수, timing 기록 SHA, 장치를 대조한다.
  - timing·confirm은 계약 장치(`cuda`)만 받고, 기록에 장치를 적는다.
  - 상한은 반올림 전 비율(`ratio_exact`)로 판정한다. 5.5는 통과, 5.50004는 탈락이다. 표시값만 반올림한다.
  - 문서의 b-P를 실제 도구로 실행하는 테스트를 추가했다(대역은 frozen checker와 모델 파일뿐).

- 테스트 `tests/test_h7r_budget.py`
  - 선택 규칙: 순서, 경계값 5.5, p95만 넘는 경우, 모두 실패.
  - 시간 기록: 색을 바꾼 두 판의 귀속, 시간이 빠지거나 쪽이 바뀐 수의 거부.
  - 전술: 기존 함수가 PASS인데 사전 지정한 E1 지표가 나빠진 경우 이 계약은 PASS하지 않는다. 분모가 없을 때의 규칙.
  - 표본: 실제 suite로 300개 / 299개 동치류 / anchor 6개 / 개발 suite 0개.
  - 흐름: prepare, timing, confirm, verify. 표본 변조 거부, 중단 뒤 같은 표본으로만 이어하기.
  - 데스크톱 블록: 실제 bash, 도구는 대역.

## 11. 결정 (2026-11-21)

1. 모델 쌍: 원본 H3 + H6 v3. 이 쌍의 S200도 50쌍 기준 +11.5%p [+2.0, +20.5]여서, S400이 시간 상한에서 빠져도 S200을 확인할 근거가 있다.
2. 확인 표본: `O1-confirm-main` 300쌍, 사용한 동치류와 anchor와의 겹침(6개)을 기록.
3. 시간 상한: 평균·p95 모두 5.5배.

## 12. 데스크톱 실행 (승인 뒤에만)

- b-P는 `approved.prepare` 뒤, b-T는 `approved.timing` 뒤, b-C는 `approved.confirm` 뒤에 실행한다. 도구가 승인과 앞 단계 기록을 확인한다.
- 세 블록 모두 PRESERVE·R3 블록과 같은 방식이다.
  - 로그 이름을 남긴다.
  - 결과가 커밋 전이면 다시 돌리지 않고 `verify`로 대조한 뒤 기록한다(`Recovered exit code`).
  - 시작 HEAD를 고정한다(`runs/.../<단계>.HEAD`). 결과 없이 실패하면 아무것도 커밋하지 않고 멈춘다. 같은 HEAD에서 다시 실행하면 대국을 이어 둔다.
- b-T는 약 10–15분, b-C는 약 2–3시간이다(tmux 권장).
- `CUBLAS_WORKSPACE_CONFIG`는 설정하지 않는다(학습이 없다). 도구가 확인한다.

b-P. prepare

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
  D=docs/h7r/budget/TB-V3-BUDGET-1
  R=runs/h7r/TB-V3-BUDGET-1
  L="$R/logs"
  T=scripts/h7r_budget.py
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
  git commit -m "H7-R BUDGET: $N exit $rc (records only)"
  rm -f "$PIN"
  git push origin feat/h7-c1
  echo "__RECORDED__ $rc"
)
```

b-T. timing (예산 선택)

```bash
(
  set -euo pipefail
  cd ~/renju-h6
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src:$PWD/scripts"
  unset CUBLAS_WORKSPACE_CONFIG
  [ "$(git branch --show-current)" = feat/h7-c1 ] || { echo "STOP: wrong branch"; exit 1; }
  git diff --quiet || { echo "STOP: unstaged tracked changes"; exit 1; }
  N=timing
  D=docs/h7r/budget/TB-V3-BUDGET-1
  R=runs/h7r/TB-V3-BUDGET-1
  L="$R/logs"
  T=scripts/h7r_budget.py
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
  tracked "$D/prepare.json" || { echo "STOP: prepare is not recorded (run b-P first)"; exit 1; }
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
  git commit -m "H7-R BUDGET: $N exit $rc (records only)"
  rm -f "$PIN"
  git push origin feat/h7-c1
  echo "__RECORDED__ $rc"
)
```

b-C. confirm (확인 대국과 판독, tmux)

```bash
(
  set -euo pipefail
  cd ~/renju-h6
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src:$PWD/scripts"
  unset CUBLAS_WORKSPACE_CONFIG
  [ "$(git branch --show-current)" = feat/h7-c1 ] || { echo "STOP: wrong branch"; exit 1; }
  git diff --quiet || { echo "STOP: unstaged tracked changes"; exit 1; }
  N=confirm
  D=docs/h7r/budget/TB-V3-BUDGET-1
  R=runs/h7r/TB-V3-BUDGET-1
  L="$R/logs"
  T=scripts/h7r_budget.py
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
  tracked "$D/timing.json" || { echo "STOP: timing is not recorded (run b-T first)"; exit 1; }
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
  git commit -m "H7-R BUDGET: $N exit $rc (records only)"
  rm -f "$PIN"
  git push origin feat/h7-c1
  echo "__RECORDED__ $rc"
)
```
