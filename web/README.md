# Local MCTS Web Play

브라우저에서 사람과 기존 MCTS 버전이 직접 대국하는 간단한 로컬 개발 도구입니다.

## 실행

저장소 루트에서:

```powershell
python scripts/run_web_play.py
```

브라우저:

```text
http://127.0.0.1:8000
```

포트를 바꾸려면:

```powershell
python scripts/run_web_play.py --port 8080
```

외부 패키지는 필요하지 않습니다. Python 표준 라이브러리 HTTP 서버를 사용합니다.

## 지원 버전

- MCTS V3.2.1
- MCTS V4.1
- MCTS V4.2
- MCTS V5 FINAL
- MCTS V6

V5 FINAL과 V6는 현재 코드의 `V5_FINAL` 설정을 그대로 사용합니다.

## 동작

- 15×15 Renju
- 사람 흑/백 선택
- 사람이 백이면 새 게임 직후 AI가 흑 첫 수를 둠
- 사람 착수는 기존 `Game.play()`로 검증하므로 흑 33/44/장목 금수도 기존 엔진과 동일하게 처리
- AI는 기존 Agent의 `select_move()`를 그대로 호출
- V5/V6는 최근 forced stage, simulation mode, V6 threat reason 등 일부 diagnostics를 화면에 표시

이 서버는 단일 로컬 브라우저 세션용입니다. 다중 사용자/배포 서버 용도가 아닙니다.

## 경기 로그

완료된 사람 vs MCTS 대국은 경기 종료 즉시 자동 저장됩니다.

```text
logs/web_play/<timestamp>_<version>_human-<color>_seed<seed>/
├─ game.json
└─ moves.csv
```

`game.json`에는 상대 버전, seed, 사람 색상, 승자/결과, 전체 착수, AI 착수별 diagnostics, 최종 보드가 저장됩니다.

`moves.csv`에는 각 수의 좌표와 AI 계산 시간, forced stage, simulation mode, selected simulations, V6 threat reason 등 분석에 자주 쓰는 필드를 평탄화해서 저장합니다.

로그는 **대국이 정상적으로 종료된 경우에만** 저장됩니다. 진행 중에 새 게임을 누르거나 서버를 종료한 미완료 대국은 저장하지 않습니다. 저장이 완료되면 웹 화면 상태 영역에 실제 저장 경로가 표시됩니다.
