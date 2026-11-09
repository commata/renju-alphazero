# R2-D1-AUTOPILOT-1 데스크톱 에이전트 지시문

이 파일은 데스크톱(`~/renju-h6`)에서 도는 에이전트(Claude Code 또는 Codex)에게 주는 지시문이다. 에이전트는 다음 할 일을 스스로 정하지 않는다. 진행 규칙은 `scripts/h7r_d1_autopilot.py`에 코드로 고정되어 있고, 에이전트는 그 도구가 말한 것만 하고 결과를 보고한다.

## 역할

- 위임 R2-D1-AUTOPILOT-1의 데스크톱 실행 담당이다.
- 맡는 범위: 전체 규모 H-dev → factory v2 시간 gate → D4 eligible manifest → D4 smoke → D4 full
  - 두 승인(`approved.smoke`, `approved.full`)은 도구가 gate를 확인한 뒤 커밋한다.
- 먼저 읽을 것
  - `docs/h7r/r2/R2-D1-HEAD-WIDTH-1/autopilot/delegation-1.json`
  - `docs/h7r/r2/R2-D1-HEAD-WIDTH-1/autopilot/state.md`
  - D1 초안 §16(`docs/h7r/r2/R2-D1-HEAD-WIDTH-1-draft.md`)
  - 이 파일
- 우선순위: 사용자의 명시적 위임을 먼저 적용하고, 그 범위 안에서 CLAUDE.md를 따른다.

## 반복 절차

작업 폴더는 `~/renju-h6`이다. 명령은 모두 저장소 루트에서 `.venv/bin/python`으로 실행한다.
명령 앞에 `cd`, `source`, 환경변수를 붙이거나 `&&`로 잇지 않는다. 허용 규칙이 명령의 시작 부분으로 맞춰지기 때문이다. 도구가 경로를 스스로 잡는다.

1. `.venv/bin/python scripts/h7r_d1_autopilot.py next`를 실행하고, 출력 JSON의 `action`을 본다.
2. `action`에 따라 하나만 한다.
   - `run` 또는 `recover`
     - `.venv/bin/python scripts/h7r_d1_autopilot.py launch`를 실행한다.
     - 블록이 tmux 세션 `d1auto`에서 돈다.
   - `wait`
     - `.venv/bin/python scripts/h7r_d1_autopilot.py wait --minutes 9`를 실행한다.
     - 명령 시간 제한을 10분으로 둘 수 있으면 `--minutes 9`를 쓴다. 기본 제한이 2분이면 `--minutes 1.5`를 쓴다.
     - 제한에 걸려 명령이 끊겨도 블록은 계속 돈다. 그대로 1로 돌아가면 된다.
     - 종료 코드 3이면 아직 도는 중이다. 같은 명령을 다시 실행한다.
     - 종료 코드 0이면 블록이 끝났다. 1로 돌아간다.
   - `approve`
     - `.venv/bin/python scripts/h7r_d1_autopilot.py approve <mode>`를 실행한다. `<mode>`는 출력의 `mode` 값이다.
   - `stop`
     - `.venv/bin/python scripts/h7r_d1_autopilot.py report`를 실행한다.
     - 아래 "최종 보고"를 쓰고 끝낸다.
   - `error`(도구가 거부함)
     - 고치지 않는다. `report` 출력과 함께 그대로 보고하고 끝낸다.
3. 블록 하나가 끝날 때마다 사용자에게 진행 보고를 2–3줄 남긴다.
   - 내용: 어느 단계가 어떤 결과로 끝났는지, 다음 action
   - `report`에 그 단계의 값(H-dev label, 시간 gate 비율, smoke·full reading)이 있으면 함께 적는다.
4. 1로 돌아간다.

세션이 끊기거나 다시 시작해도 1부터 다시 하면 된다. 도구는 커밋된 기록과 launch 기록(`runs/h7r/R2-D1-HEAD-WIDTH-1/autopilot/launches.jsonl`)으로 상태를 다시 읽는다. 블록은 에이전트와 상관없이 tmux에서 계속 돈다.

## 하지 않는 것

- 코드, 계약, 테스트, 문서를 편집하지 않는다. 승인 플래그도 손으로 바꾸지 않는다.
- git 명령을 직접 쓰지 않는다: commit, push, pull, reset, revert, checkout, stash. 커밋과 push는 블록과 `approve`만 한다.
  - 읽기 명령(`git log`, `git status`, `git show`)은 괜찮다.
- `runs/`나 `docs/`의 파일을 지우거나 옮기거나 덮어쓰지 않는다.
- 블록(`~/d1-*.sh`)을 직접 실행하지 않는다. `launch` 말고 다른 인자로 실행하지도 않는다.
  - 특히 `d1-full-3R.sh`(H-dev 재개)는 실행하지 않는다. 재개 여부는 클라우드 세션이 정한다.
- 도구가 정한 1회 재개 말고는 다시 실행하지 않는다.
- 다른 GPU 작업을 함께 돌리지 않는다. 시간 gate가 수당 시간을 잰다.
- 긴 `sleep`으로 기다리지 않는다. 기다림은 `wait`만 쓴다. 한 번 호출이 10분을 넘지 않는다.
- 범위 밖 작업을 하지 않는다.
  - 확인 평가(P5) 계약 작성
  - R4 루프 시작
  - champion·canonical·웹 배포 변경
  - 결과를 보고 계약·표본·예산을 바꾸는 것
- 결과를 해석해 진행 여부를 바꾸지 않는다. `stop`이면 멈춘다.

## 최종 보고 (`stop` 또는 `error`일 때)

한국어로 다음을 적는다.

1. 멈춘 이유: 도구의 `reason`을 그대로 인용하고, `final`(위임 자체가 끝나는지)을 적는다.
2. 실행한 단계: `report`의 `launches`(단계, 종료 코드, 기록 커밋 여부, 시작 시각)
3. 기록값: `report`의 fit, H-dev, 시간 gate, smoke, full
   - full이 있으면 주 비교 3쌍의 label과 구간, branch 결과를 적는다.
4. 커밋: `git log --oneline -8`
5. 다음 단계: `reason`에 적힌 것만 적는다.
   - 예: `ALL_THREE_BETTER`면 "클라우드 세션이 확인 평가 계약을 고정", 그 밖은 "보고 후 위임 종료"
   - 실패면 출력 파일 경로와 마지막 줄

이 보고를 사용자가 클라우드 세션에 전달한다.
