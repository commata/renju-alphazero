# R2-P1-A1 Arena 재검증과 O0 동치류 기술 분석 (읽기 전용)

- 상태: 도구·테스트 완료, 데스크톱 실행 대기
- 대상: R2-P1-POLICY-REPLAY-A1(실험 `faeb72b0f625`)의 judge 기록 `389d09a`
  - 판정: P NOT_ESTABLISHED, D BETTER_THAN_CONTROL. 이 판정과 계약 종료는 바뀌지 않는다.
- 도구: `scripts/h7r_p1a1_recheck.py`, 테스트 `tests/test_h7r_p1a1_recheck.py`
- 새 대국, 학습, 계약 변경은 없다. 도구는 결과 파일 하나(`arena-recheck.json`)만 새로 쓰고, 덮어쓰지 않는다.

## 1. 확인 순서

1. **신원**
   - A1 계약: arena 단계, 고정된 source와 근거 기록의 SHA를 함께 확인한다.
   - 커밋된 `judge.json`: 같은 실험의 것이어야 한다.
   - 커밋된 Arena 계약: SHA가 judge 기록에 적힌 값과 같아야 한다.
2. **원본 arm 8개**
   - 파일 SHA를 `judge.json`의 `arm_records_sha256`과 **먼저** 대조한다. 하나라도 다르면 아무것도 재계산하지 않고 멈춘다.
   - 그다음 `h7r_eval.load_arm`으로 읽는다. 이 단계에서 대국, opening, 신원, 요약을 다시 검사한다.
   - arm별 요약이 judge의 요약과 같아야 한다.
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

- Arena가 끝났으므로 HEAD 고정은 더 필요 없다. 블록은 pull한 뒤 실행한다.
- 결과 파일과 로그를 커밋한다. 쌍 단위 원본(`runs/.../arena/*.json`)은 커밋하지 않는다.
- 다시 실행했을 때
  - 결과가 이미 커밋되어 있으면 멈춘다.
  - 결과 파일이 있지만 커밋 전이면, 다시 계산하지 않고 남은 로그와 함께 커밋한다.
- 예상 시간: 1분 안팎(bootstrap 재계산 약 12초 + arm 8개 재검사)

```bash
(
  set -euo pipefail
  cd ~/renju-h6
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src:$PWD/scripts"
  [ "$(git branch --show-current)" = feat/h7-c1 ] || { echo "STOP: wrong branch"; exit 1; }
  git pull --ff-only origin feat/h7-c1
  test -z "$(git status --porcelain --untracked-files=no)"
  D=docs/h7r/r2/R2-P1-POLICY-REPLAY-A1
  R=runs/h7r/R2-P1-POLICY-REPLAY-A1
  L="$R/logs"
  O="$D/arena-recheck.json"
  tracked(){ git ls-files --error-unmatch "$1" >/dev/null 2>&1; }
  if tracked "$O"; then echo "STOP: the recheck is already recorded"; exit 1; fi
  mkdir -p "$L"
  rc=0
  if [ -e "$O" ]; then
    echo "an uncommitted $O exists: recording it (no new recheck)"
  else
    B="arena-recheck-$(date +%Y%m%d-%H%M%S)"
    if [ -e "$L/$B.log" ]; then echo "STOP: $L/$B.log exists"; exit 1; fi
    python -u scripts/h7r_p1a1_recheck.py --arms-dir "$R/arena" --output "$O" 2>&1 | tee "$L/$B.log" || rc=$?
    echo "__EXIT__ $rc" >> "$L/$B.log"
  fi
  expect=()
  shopt -s nullglob
  for f in "$L"/arena-recheck-*.log; do
    n="$(basename "$f" .log)"
    tracked "$D/stage-logs/$n.log" && continue
    python scripts/h7r_d4_records.py copy-logs --resume --src-dir "$L" --pattern "$n.log" \
      --dest-dir "$D/stage-logs" --manifest "$D/stage-logs/$n.manifest.json"
    expect+=(--expect "$D/stage-logs/$n.log" --expect "$D/stage-logs/$n.manifest.json")
  done
  if [ -f "$O" ]; then expect+=(--expect "$O"); fi
  for e in "${expect[@]}"; do [ "$e" = --expect ] || git add "$e"; done
  python scripts/h7r_d4_records.py verify-staged "${expect[@]}"
  git commit -m "H7-R P1-A1: Arena recheck exit $rc (records only)"
  git push origin feat/h7-c1
  echo "__RECORDED__ $rc"
  exit "$rc"
)
```
