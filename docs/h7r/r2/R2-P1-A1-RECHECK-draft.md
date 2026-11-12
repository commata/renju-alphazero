# R2-P1-A1 Arena 재검증과 O0 동치류 기술 분석 (읽기 전용)

- 상태: 도구·테스트 완료, 데스크톱 실행 대기
- 대상: R2-P1-POLICY-REPLAY-A1(실험 `faeb72b0f625`)의 judge 기록 `389d09a`
  - 판정: P NOT_ESTABLISHED, D BETTER_THAN_CONTROL. 이 판정과 계약 종료는 바뀌지 않는다.
- 도구: `scripts/h7r_p1a1_recheck.py`, 테스트 `tests/test_h7r_p1a1_recheck.py`
- 새 대국, 학습, 계약 변경은 없다. 도구는 결과 파일 하나(`arena-recheck.json`)만 새로 쓰고, 덮어쓰지 않는다.

## 1. 확인 순서

1. **신원**
   - A1 계약: arena 단계, 고정된 source와 근거 기록의 SHA를 함께 확인한다.
  - 커밋된 `judge.json`: 같은 실험과 A1 계약의 것이어야 한다.
   - 커밋된 Arena 계약: SHA가 judge 기록에 적힌 값과 같아야 한다.
2. **원본 arm 8개**
   - 파일 SHA를 `judge.json`의 `arm_records_sha256`과 **먼저** 대조한다. 하나라도 다르면 아무것도 재계산하지 않고 멈춘다.
   - 그다음 `h7r_eval.load_arm`으로 읽는다. 이 단계에서 대국, opening, 신원, 요약을 다시 검사한다.
   - arm별 요약이 judge의 요약과 같아야 한다.
   - 입력 SHA는 시작 시점에 기록하고, 기술 분석까지 끝난 뒤 결과 저장 직전에 HEAD와 입력을 다시 확인한다.
3. **bootstrap 재계산**
   - judge가 쓴 함수(`h7r_p1.stratified_bounds`)와 계약의 설정(20,000회, seed 20261131, 단측 α 0.025)을 그대로 쓴다.
   - opening 단위와 동치류 단위의 P·D 구간, 그리고 판독을 judge와 대조한다.
4. **O0 동치류 기술 분석** (사후 기술 분석. 판정·기준·결론을 만들지 않는다)
   - 동치류별 쌍 수, `T − 0.5`의 합과 평균. 참고로 `C − 0.5`와 `T − C`의 평균도 낸다.
   - 동치류 하나씩 제외: O0에서 T의 점수, 전체 P, 전체 D가 얼마나 움직이는지. O1은 그대로 두고 50:50으로 다시 계산한다.
   - 집중도
     - 평균이 음수·0·양수인 동치류 수
     - 음수 합 가운데 가장 나쁜 1·3·5·10개 동치류가 차지하는 비율
     - 제외했을 때 P를 가장 크게 움직이는 동치류 5개

## 2. 종료 코드

- 0: 모든 대조가 같음. 결과 파일을 쓴다.
- 2: 재계산 값이 judge와 다름. 결과 파일에 어느 항목이 다른지 남긴다.
- 1: 멈춤. arm 파일 SHA 불일치, 기록 누락, 결과 파일이 이미 있음 등. 결과 파일을 쓰지 않는다.

## 3. 데스크톱 실행 (a1-R)

- 블록은 작업 트리를 확인하고, staging 복구 중이 아니면 pull한 뒤 실행한다.
- 결과 파일과 로그를 커밋한다. 쌍 단위 원본(`runs/.../arena/*.json`)은 커밋하지 않는다.
- 다시 실행했을 때
  - 결과가 이미 커밋되어 있으면 멈춘다.
  - 결과 파일이 있지만 커밋 전이면, 결과가 가리키는 로그를 확인해 같은 종료 코드로 기록한다. 출력 직후에 중단돼 로그에 종료 코드가 없다면 결과에서 복구한 값임을 표시한다.
  - 커밋은 됐고 push가 실패했다면 `git push origin feat/h7-c1`로 다시 보낸다.
- 예상 시간: 1분 안팎(bootstrap 재계산 약 12초 + arm 8개 재검사)

```bash
(
  set -euo pipefail
  cd ~/renju-h6
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src:$PWD/scripts"
  [ "$(git branch --show-current)" = feat/h7-c1 ] || { echo "STOP: wrong branch"; exit 1; }
  git diff --quiet || { echo "STOP: unstaged tracked changes"; exit 1; }
  if git diff --cached --quiet; then git pull --ff-only origin feat/h7-c1; fi
  D=docs/h7r/r2/R2-P1-POLICY-REPLAY-A1
  R=runs/h7r/R2-P1-POLICY-REPLAY-A1
  L="$R/logs"
  O="$D/arena-recheck.json"
  tracked(){ git cat-file -e "HEAD:$1" 2>/dev/null; }
  if tracked "$O"; then echo "STOP: the recheck is already recorded"; exit 1; fi
  mkdir -p "$L"
  rc=0
  if [ ! -e "$O" ]; then
    git diff --cached --quiet || { echo "STOP: resolve staged changes before starting a new recheck"; exit 1; }
    log="$(mktemp "$L/arena-recheck-XXXXXXXX.log")"
    set +e
    python -u scripts/h7r_p1a1_recheck.py --arms-dir "$R/arena" --output "$O" \
      --log-name "$(basename "$log")" 2>&1 | tee "$log"
    codes=("${PIPESTATUS[@]}")
    set -e
    rc="${codes[0]}"
    [ "${codes[1]}" -eq 0 ] || { echo "STOP: tee failed; log: $log"; exit 1; }
    echo "__EXIT__ $rc" >> "$log"
  else
    echo "an uncommitted $O exists: recording it (no new recheck)"
  fi
  if [ -f "$O" ]; then
    result="$(python - "$O" <<'PY'
import json, re, sys
from pathlib import Path
r = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
name = r.get('log_name', '')
equal = r.get('recomputed_equal_judge')
if (r.get('format') != 'h7r-p1a1-arena-recheck-v1' or not isinstance(equal, bool)
        or not re.fullmatch(r'arena-recheck-[A-Za-z0-9]+\.log', name)):
    sys.exit('STOP: invalid recheck result or missing attempt log name')
print(name, 0 if equal else 2)
PY
)"
    read -r name rc <<< "$result"
    log="$L/$name"
    [ -f "$log" ] || { echo "STOP: result log is missing: $log"; exit 1; }
    last="$(tail -n 1 "$log")"
    case "$last" in
      "__EXIT__ $rc") ;;
      __EXIT__*) echo "STOP: result and log exit codes differ"; exit 1 ;;
      *) printf '\n%s\n__EXIT__ %s\n' "Recovered exit code from $O" "$rc" >> "$log" ;;
    esac
  elif [ "$rc" -eq 0 ] || [ "$rc" -eq 2 ]; then
    echo "STOP: recheck did not produce a result"; exit 1
  fi
  expect=()
  paths=()
  shopt -s nullglob
  for f in "$L"/arena-recheck-*.log; do
    [[ "$(tail -n 1 "$f")" =~ ^__EXIT__\ [0-9]+$ ]] || continue
    n="$(basename "$f" .log)"
    tracked "$D/stage-logs/$n.log" && continue
    python scripts/h7r_d4_records.py copy-logs --resume --src-dir "$L" --pattern "$n.log" \
      --dest-dir "$D/stage-logs" --manifest "$D/stage-logs/$n.manifest.json"
    paths+=("$D/stage-logs/$n.log" "$D/stage-logs/$n.manifest.json")
  done
  if [ -f "$O" ]; then paths+=("$O"); fi
  [ "${#paths[@]}" -gt 0 ] || { echo "STOP: no records to commit"; exit 1; }
  while IFS= read -r -d '' staged; do
    allowed=false
    for f in "${paths[@]}"; do if [ "$f" = "$staged" ]; then allowed=true; break; fi; done
    "$allowed" || { echo "STOP: unrelated staged path: $staged"; exit 1; }
  done < <(git diff --cached --name-only -z)
  for f in "${paths[@]}"; do expect+=(--expect "$f"); done
  git add -- "${paths[@]}"
  python scripts/h7r_d4_records.py verify-staged "${expect[@]}"
  git commit -m "H7-R P1-A1: Arena recheck exit $rc (records only)"
  git push origin feat/h7-c1
  echo "__RECORDED__ $rc"
  exit "$rc"
)
```
