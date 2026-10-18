# 외부 엔진 앵커: Rapfi를 시험관으로 쓰기 (X0–X3)

작성 2026-10-17. 브랜치 `feat/otherAI_test`. 상태: **X0 완료**, X1–X3 프로토콜과 세 결정(§14)은 **실행 전 고정**(2026-10-17 사용자 승인).
관련 문서: [track-b-post-e1.md §2.4·§6](track-b-post-e1.md), [track-b-b1-b2.md §3·§8·§13](track-b-b1-b2.md), [mcts-v8-teacher.md §12.1](mcts-v8-teacher.md).

## 0. 결론

외부 엔진(1순위 Rapfi)은 **학습시키는 선생님이 아니라 우리 AI를 시험하는 외부 시험관**으로 쓴다.

```
                         ┌─ 내부 Arena (승급 판정)
Training ── Model ───────┼─ Final-1 / Final-2 (봉인, 손대지 않음)
                         └─ 외부 앵커 (Rapfi, 기록·진단 전용)

Rapfi에게 패배 → 패배 직전 국면(loss_candidates) → offline 참값(§2.4) → E 후보 / probe / holdout
```

- Rapfi 착수는 **어느 트랙에서도 학습 샘플이 되지 않는다.** 외부 앵커 대국의 seed는 `RESERVED_SEEDS`에 넣었다(§4).
- Rapfi 점수는 **승급 조건이 아니다.** 내부 Arena와 Final-1/2의 정의는 바꾸지 않는다.
- 얻는 것: (1) 내부 상대가 공통으로 못 보는 전술 약점 발견, (2) 고정된 외부 기준점(내부 Arena 밖의 공통 상대, 외부 기력 앵커), (3) 패배 사례 공급, (4) 렌주 금수 판정 교차검증.

### 왜 학습 데이터로 쓰지 않나

| 항목 | 내용 |
|---|---|
| 양 | 100판에서 Rapfi 착수 약 1,500~3,000개. H3는 RenjuNet 10.9만 판으로 약 3,100만 샘플을 처리했다(0.01% 수준) |
| 질 | alpha-beta 엔진이라 visit 분포가 없고 one-hot이다. value는 프로젝트 원칙상 "증명 라벨 > 탐색 분포 > 대국 결과"이고, 압도적 상대와 둔 판의 결과는 편향된다 |
| 실험 설계 | Track A는 외부 teacher 금지. B2에 넣으면 RQ2·RQ3(RenjuNet 효과, catch-up 비용)이 "self-play / solver / Rapfi teacher" 중 무엇 때문인지 구분할 수 없게 된다 |

학습에 쓰는 안은 기본 off 별도 arm(`B2-X`)으로만 남기고, 결과를 RQ 비교에 섞지 않는다.

## 1. 트랙별 사용 범위

| 트랙 | 외부 엔진 대국 | 학습·초기화·라벨 |
|---|---|---|
| Track A | 평가 전용(RenjuNet 국면을 평가에만 쓰는 것과 같은 규칙) | 금지 |
| B1-Champion / B1-Matched | 평가 전용, 기록용 Arena 상대(X2), H8 보고용 앵커(X3) | 금지 |
| B2 | 평가 전용, 기록만(track-b-b1-b2.md §3 "평가 전용") | 금지 |

## 2. 구성 요소 (X0, 이 커밋)

| 파일 | 내용 |
|---|---|
| `src/agents/external_engine.py` | Piskvork 프로토콜 어댑터. `PiskvorkEngine`(프로세스, 응답 timeout), `ExternalEngineAgent`(기존 `Agent` 프로토콜 구현), `EngineConfig`(명령·INFO 설정·provenance), 좌표 변환, `YXSHOWFORBID` 금수점 질의 |
| `src/evaluation/external.py` | seed 블록, 요약(점수, 색별, 쌍 bootstrap 95% 구간, Elo, 제외 판, loss 후보, fingerprint) |
| `scripts/run_external_match.py` | 우리 agent(`tb1`, `v7`, `v8:<arm>`, `az`, `random`) 대 외부 엔진. opening 쌍 × 색 교환, JSONL resume, 프로세스 병렬, 분당 진행 출력 |
| `scripts/external_rule_crosscheck.py` | 무작위 근접 대국의 모든 흑 차례 국면에서 우리 금수점과 엔진 금수점을 비교 |
| `scripts/fake_piskvork_engine.py` | 테스트·smoke용 가짜 엔진. 정상 착수 외에 금수 착수, 무응답, 크래시, 잘못된 응답, 금수점 거짓 보고 모드 |
| `tests/test_external_engine.py` | 좌표, BOARD 필드, 프로토콜 순서, 금수 착수 → `engine_illegal`, timeout·크래시, 금수점 질의, runner 한 판, 요약(한쪽만 깨진 쌍 포함), run signature와 resume 거부, 해시 fail-fast, seed 예약 (19개) |

동결 엔진 파일(`tb_engine_v1.py` 등)은 건드리지 않았다. `hybrid/h6_labels.py`는 `RESERVED_SEEDS`에 외부 앵커 seed 블록만 추가했다.

## 3. 프로토콜 세부

- **명령.** `START 15` → `OK`, `INFO rule 4`(렌주, 어댑터가 항상 보내고 `--engine-info rule=…`은 거부), 그다음 `--engine-info`의 INFO 줄.
  수마다 `BOARD` … `DONE`으로 **전체 국면을 다시 보낸다**(stateless). 응답은 `x,y` 한 줄이다.
- **BOARD 세 번째 값은 흑/백이 아니다.** 1 = 엔진 자신(둘 차례), 2 = 상대. 엔진은 첫 돌을 흑으로 본다(Rapfi `getPosition` 확인). 그래서 돌은 반드시 착수 순서대로 보낸다.
- **좌표.** Piskvork `x,y`는 (열, 행), 우리 좌표는 (행, 열)이다. 일관된 전치는 판 전체를 대각 반사할 뿐이라(렌주는 D4 불변) 결과가 같지만, 섞이면 깨진다. 변환은 `to_piskvork`/`from_piskvork` 한 곳에서만 하고 왕복 테스트를 둔다.
- **심판은 우리 `Game`.** 엔진 응답은 `check_engine_move`로 먼저 확인한다(범위, 빈칸, 흑 금수).
- **재현성.** 시간 제한은 기계 부하에 따라 수가 바뀐다. 앵커는 **노드 제한(`max_node`) + `thread_num 1`**을 쓰고,
  `timeout_turn`·`timeout_match`는 노드 제한이 먼저 걸리도록 크게 둔다. Rapfi의 `[database] enable_by_default = false`(기본값)를 유지한다. 켜면 엔진이 판 사이에 기억을 쌓아 같은 설정이 같은 상대가 아니게 된다.
- **진단.** 엔진의 `MESSAGE` 줄 중 마지막 줄(깊이, Eval, 노드)을 수마다 저장한다. `ERROR` 줄은 판마다 마지막 5개를 저장한다.
- **provenance.** 실행 파일 SHA-256, `--engine-file`로 지정한 config·가중치 SHA-256, `ABOUT` 응답, INFO 설정 전체, git commit/dirty, 우리 agent의 checkpoint 해시(`tb1`은 `config_hash`도).
  결과 key에 INFO 설정이 들어가므로(`rapfi[max_node=100000][thread_num=1]…`) 설정이 다른 판은 한 JSONL에서 섞이지 않는다.

## 4. seed 블록 (고정, 아직 한 판도 두지 않음)

| 블록 | seed | 용도 |
|---|---|---|
| X1 보정 | 8701–8705 | 8701 강도 사다리, 8702 확인 대국, 8703 재현성 재실행, 8704–8705 예비(§8의 규칙으로만 사용) |
| X2 모니터링 | 8711–8720 | H7 주기 c에 seed 8710 + c(c = 1…10). 기록 전용 |
| X3 최종 앵커 | 8721–8725 | H8 보고용. **H8 전에는 두지 않는다** |

- 세 블록 모두 `hybrid.h6_labels.RESERVED_SEEDS`에 있다. runner는 이 블록 밖의 seed를 거부한다(`--allow-unreserved-seed`는 가짜 엔진 smoke 전용).
- opening은 중앙 + 무작위 4수(반경 2)다. Final 세트와 같은 생성 방식이라 D4 종류가 충분하다.
- **기록(2026-10-18):** X0 resume 검사 중 클라우드에서 seed **8701 pair 0**을 한 번 썼다. `v7`(smoke, 50 simulations) 대 Rapfi `max_node=3000` 2판이며, 결과는 scratch에만 있고 커밋하지 않았다.
  `tb1`은 두지 않았고 X1 선택(§8)은 `tb1` 점수로만 하므로 판정에는 영향이 없다. 이후 smoke는 예약 블록 밖 seed(`--allow-unreserved-seed`)만 쓴다.

## 5. 사전 고정 처리 규칙

| 상황 | 처리 |
|---|---|
| 엔진이 우리 규칙상 금수(또는 점유·범위 밖)를 둠 | `engine_illegal`. 판을 거기서 멈추고 **점수에서 뺀다**. `illegal_games`에 국면과 이유를 남기고 수동 규칙 검토(§6) |
| 엔진 크래시, 응답 없음(`--move-timeout`, 기본 600초), 프로토콜 위반 | `engine_error`. 점수에서 빼고 `error_games`에 남긴다 |
| 무승부 | 0.5점. 우리 엔진은 합법수가 없을 때만 무승부다 |
| 점수 | 승 1, 무 0.5, 패 0 |
| **표본: 완전한 opening 쌍** | 색을 바꾼 두 판이 모두 점수가 있는 쌍만 쓴다. `score`, `elo`, `by_colour`, 95% 구간(쌍 bootstrap 10,000회, RNG 고정)이 **모두 같은 표본**이다. 한 판만 `engine_illegal`/`engine_error`인 쌍은 통째로 빠지고 `incomplete_pairs`에 남는다. 모든 정상 판으로 낸 `raw_score`, `raw_scored_games`는 진단용이다 |
| **resume** | 판마다 `run_signature`를 저장한다. 엔진 실행 파일과 `--engine-file`의 SHA-256, 엔진 인자·INFO, 우리 agent와 checkpoint 해시(`tb1`은 config hash 포함), opening 규칙(수, 반경), `--forbid-check`, 기록 형식의 해시다. 같은 key인데 signature가 다르면(또는 없으면) 이어 쓰지 않고 **실행을 멈춘다**. `--pairs`는 넣지 않아 같은 설정으로 쌍을 늘릴 수 있다. 경로, workers, timeout guard도 넣지 않는다 |
| **신원 fail-fast** | `--engine-file`로 지정한 파일을 해시할 수 없으면 시작하지 않는다. 앵커 seed(8701–8725)에서는 엔진 실행 파일 해시도 필수다(PATH의 명령은 실제 파일로 찾아 해시한다) |
| 제외 판 비율 | 한 실행에서 `engine_illegal + engine_error`가 10%를 넘으면 그 실행 점수는 보고만 하고 해석하지 않는다. 원인을 고친 뒤 **새 seed가 아니라 같은 seed**로 다시 돈다(엔진 쪽 고장은 우리 모델 결과를 보고 고르는 것이 아니다) |

## 6. 렌주 규칙 교차검증

- 대국 중: `--forbid-check`를 켜면 모든 흑 차례 국면에서 우리 금수점과 엔진의 `YXSHOWFORBID` 금수점을 비교한다(`forbid_mismatches`).
- 대량 검사: `scripts/external_rule_crosscheck.py`. 무작위 근접 대국(반경 1 또는 2)이라 삼·사가 자주 생긴다.
- 불일치는 **어느 엔진도 자동으로 고치지 않는다.** 국면을 꺼내 [docs/rules.md](rules.md)와 RIF 규칙으로 사람이 판정한다. 우리 쪽 오류면 규칙 엔진 수정은 별도 단계(모든 frozen fingerprint에 영향)로 다룬다.

**X0 결과:** Rapfi 0.43.02와 금수점 불일치 0(§7). 삼삼 예외까지 두 독립 구현이 같은 판정을 냈다는 뜻이며, 무작위 근접 대국에서 드문 형태는 대국 중 `--forbid-check`가 계속 본다.

## 7. X0 결과 (2026-10-17, 클라우드 Linux, 4 vCPU)

Rapfi를 소스에서 빌드해(GPL-3.0, `dhbloo/rapfi` master, g++ 13.3, SSE4.1+AVX2, 실행 파일 SHA-256 `af5388229a76…`) `rapfi-networks`(CC0)의 기본 `config.toml`과 mix9svq 가중치로 실행했다.
커밋 전 작업 트리(`c062988` + 이 문서의 새 파일)에서 돌렸다. seed 2·3 실행의 `git_dirty: true`는 `h6_labels.py`의 seed 예약 한 줄 때문이며, 어댑터·교차검사 코드는 이 커밋과 같다.
클라우드 결과는 X0(연결·규칙) 확인용이다. X1 보정은 데스크톱에서 그 기계의 Rapfi 빌드로 새로 한다.

| 확인 | 결과 |
|---|---|
| 연결 | `START`/`INFO`/`ABOUT`/`BOARD`/`YXBOARD`/`YXSHOWFORBID`/`END` 정상. `ABOUT` = `Rapfi 0.43.02` |
| 재현성 | `v7`(smoke, 50 simulations) 대 `max_node=3000, thread_num=1`, 2쌍 4판을 2회 실행 → fingerprint 동일(`fdd26b00…`) |
| 강도 감 | 같은 설정에서 Rapfi 4전 4승(16~27수). 노드 3,000에서도 축소 V7보다 훨씬 강하다 → X1 사다리는 낮은 노드부터 시작한다 |
| 금수점 교차검사 | **불일치 0**. 실행 세 번의 합계는 아래 표. `docs/external/rule_crosscheck_x0.json` |

| 실행 | 반경 | seed | 판 | 흑 차례 국면 | 우리 금수점 | 불일치 |
|---|---|---|---:|---:|---:|---:|
| 예비 검증 | 1 | 1 | 300 | 8,886 | 9,517 | 0 |
| 본 교차검사 | 1 | 2 | 3,000 | 90,612 | 92,778 | 0 |
| 추가 교차검사 | 2 | 3 | 1,500 | 74,341 | 126,160 | 0 |
| **합계** | | | **4,800** | **173,839** | **228,455** | **0** |

## 8. X1: 강도 보정 (사전 고정)

목적: 현재 B1 엔진(`tb1` = TB-Engine-v1 + H3 policy)이 **점수 0.2~0.8**에 들어오는 Rapfi 강도를 고르고, 이후 바꾸지 않는다.

- X1은 **강도 보정 단계일 뿐**이다. X1 점수로 모델 기력을 주장하지 않는다.
- 강도(L_mid, L_strong)는 **X1 결과로만** 고른다. H7 Monitoring, Final-1/2, X3 결과를 본 뒤 다시 조정하지 않는다.

- **때:** H6-5 전후, 데스크톱이 다른 측정을 하지 않을 때(같은 기계에서 동시 실행 금지, track-b-b1-b2.md §6).
- **설정:** `thread_num=1`, `timeout_turn=600000`, `timeout_match=1000000000`, Rapfi 기본 config(database off). 엔진 실행 파일·config·가중치 해시를 첫 실행에서 기록하고 이후 바꾸지 않는다.
- **사다리:** `max_node` ∈ {300, 1k, 3k, 10k, 30k, 100k, 300k}. 각 단계 `tb1` 5쌍(10판), seed **8701**(모든 단계가 같은 opening을 쓴다).
- **선택 규칙:**
  1. 점수가 0.2~0.8인 단계 중 0.5에 가장 가까운 단계를 **L_mid**로 한다. |score − 0.5|가 같으면 **노드가 적은 쪽**.
  2. 그런 단계가 없으면(이웃 단계가 0.8 초과에서 0.2 미만으로 건너뜀) 두 단계의 기하 평균 노드 하나를 seed 8701로 한 번 더 둔다. 그래도 없으면 0.5에 가장 가까운 단계를 L_mid로 하고 그 사실을 기록한다.
  3. 300 노드에서도 0.2 미만이면 `max_node` 100, 30을 추가한다. 300k에서도 0.8 초과면 1M, 3M을 추가한다.
  4. **L_strong** = 사다리에서 L_mid 바로 위 단계. 모델이 강해져 L_mid 점수가 0.8을 넘을 때를 대비해 처음부터 같이 고정한다.
- **확인:** L_mid에서 `tb1` 25쌍(50판), seed **8702**. 기준 기록이며 판정 조건은 아니다.
- **재현성:** L_mid에서 seed **8703** 2쌍을 두 번 → fingerprint가 같아야 한다. 다르면 "이 빌드는 재현 불가"로 기록하고, 이후 보고에는 기보 전체를 함께 남긴다.
- **산출물:** `docs/external/x1_calibration.json`(단계별 점수, 선택 근거, L_mid·L_strong, 엔진 해시). 기보 JSONL은 `runs/external/`(git 제외).

## 9. X2: H7 모니터링 상대 결정 (B1-Champion H7-c1 시작 전)

- track-b-b1-b2.md §13에 따라 Monitoring Arena는 H7 시작 전에 고정된다. 외부 앵커를 넣을지 이때 정하고, 이후 바꾸지 않는다.
- **결정(2026-10-17): 넣는다.** **기록용 monitoring 상대**로, 주기마다 L_mid 25쌍(50판), seed 8710 + c. B1-Champion-v1 25쌍 기록용과 같은 지위다.
  - 승급·탈락 gate에는 **절대 쓰지 않는다.** 승급(직전 checkpoint 50쌍), B1 수렴 기준(§9), B2 catch-up 기준은 바꾸지 않는다.
  - B1-Champion, B1-Matched, B2, Track A를 같은 Rapfi 설정으로 기록해 장기 곡선을 비교한다.
  - L_mid 점수가 0.8을 넘으면 그 주기부터 L_strong 25쌍을 **추가로** 기록한다(L_mid도 계속 둔다. 곡선이 끊기지 않게).
- 주기 비용: 50판(L_strong 추가 시 100판). 내부 Arena 약 250판에 더해진다.

## 10. X3: H8 보고용 외부 앵커

- Final-1/Final-2의 정의(상대 3종, 판 수, 판정 기준)는 **그대로** 둔다. 외부 앵커는 별도 표로 보고한다.
- 모델마다 L_mid 50쌍 + L_strong 50쌍 = 200판, seed 8721(L_mid), 8722(L_strong). 8723–8725는 예비.
- Final 세트를 열 때 외부 앵커 opening과 Final opening의 D4 중복을 검사하고, 겹친 쌍은 따로 표시해 보고한다(빼지 않는다. 결과를 본 뒤 고르지 않기 위해).
- 보고 예: "고정 강도 Rapfi(`max_node=…`)를 상대로 B1-Champion-v1 0.xx, B1-Matched 0.xx, B2 0.xx, Track A 0.xx".

## 11. 패배 국면 처리

- 요약의 `loss_candidates`: 진 판마다 마지막 10수(`--loss-window`) 안의 **우리 착수 국면**(수순, 둔 수, 끝까지 남은 수).
- 이 국면은 track-b-post-e1.md §2.4 순서로 간다: offline 참값 → ① 버그 ② 증명 범위 안의 선택 실패(엔진 후보) ③ 응징 실패 ④ 더 깊은 지평선(probe·holdout 등록).
- 학습 라벨로는 쓰지 않는다. probe·holdout에 들어간 국면은 기존 누출 규칙대로 모든 학습 데이터에서 빠진다.

## 12. 데스크톱 실행

Rapfi 준비(Windows): GitHub releases의 `pbrain-rapfi-windows-avx2.exe`(또는 소스 빌드)와 `rapfi-networks`의 `config-example/config.toml`, `classical/*.bin`, `mix9svq/*.lz4`를 한 폴더(예: `C:\engines\rapfi`)에 둔다.
실행 파일·가중치는 저장소에 넣지 않는다(`data/external/engines/`처럼 git 제외 경로도 가능).

```powershell
New-Item -ItemType Directory -Force runs/external/x1 | Out-Null
# 0) 연결·규칙 확인 (몇 분)
python scripts/external_rule_crosscheck.py --engine C:\engines\rapfi\pbrain-rapfi-windows-avx2.exe `
  --engine-name rapfi --games 300 --seed 1 --output runs/external/rule_crosscheck.json 2>&1 | Tee-Object runs/external/rule.log

# 1) X1 사다리 (seed 8701, 단계마다 5쌍)
$rapfi = "C:\engines\rapfi"
foreach ($n in 300, 1000, 3000, 10000, 30000, 100000, 300000) {
  python scripts/run_external_match.py --ours tb1 --engine $rapfi\pbrain-rapfi-windows-avx2.exe `
    --engine-name rapfi --engine-info max_node=$n --engine-info thread_num=1 `
    --engine-info timeout_turn=600000 --engine-info timeout_match=1000000000 `
    --engine-file $rapfi\config.toml --engine-file $rapfi\mix9svqrenju_bs15_black.bin.lz4 `
    --engine-file $rapfi\mix9svqrenju_bs15_white.bin.lz4 `
    --seed 8701 --pairs 5 --workers 4 --forbid-check `
    --games-jsonl runs/external/x1/n$n.jsonl --output runs/external/x1/n$n.json 2>&1 | Tee-Object runs/external/x1/n$n.log
}
```

- `--workers`만큼 Rapfi 프로세스가 뜬다(각 1스레드). `tb1`도 프로세스마다 policy를 읽는다.
- 중단되면 같은 명령을 다시 실행한다(JSONL resume).

## 13. 라이선스

- Rapfi: **GPL-3.0**(`Copying.txt` 확인). 현재는 프로젝트 내부 평가용으로 사용하며 Rapfi 바이너리를 저장소나 결과물에 재배포하지 않는다. 향후 바이너리를 함께 배포할 경우 GPL-3.0의 배포 조건을 별도로 준수한다.
- `rapfi-networks` 가중치: **CC0 1.0**.
- Yixin(2순위): 소스 비공개 프리웨어다. 쓰기 전에 배포 조건을 확인한다. 같은 어댑터로 붙는다(`YXSHOWFORBID`가 없으면 `--forbid-check` 없이 실행).

## 14. 결정 (2026-10-17 확정)

| # | 질문 | 결정 |
|---|---|---|
| 1 | X1 사다리와 선택 규칙(§8) | **고정.** 7단계 사다리, 0.2~0.8 중 0.5에 가장 가까운 단계, 동률이면 노드가 적은 쪽. X1 결과로만 고른다 |
| 2 | H7 외부 앵커 | **채택.** 기록용 monitoring 상대(§9). 승급·탈락 gate에는 쓰지 않는다 |
| 3 | `B2-X`(외부 엔진 teacher arm) | **OFF.** "external teacher가 self-play-only보다 도움이 되는가"를 연구 확장으로 따로 실험할 때만 별도 트랙으로 켠다 |

## 15. 진행 상태와 merge 조건

| 항목 | 상태 |
|---|---|
| X0 구현 커밋 | 완료 |
| 새 테스트(`tests/test_external_engine.py`, 19개) | PASS |
| 검토 반영(2026-10-18): resume `run_signature`, 완전한 쌍 표본, 신원 fail-fast, "외부 기력 앵커" 표현 | 완료 |
| base 동기화: `origin/feat/h6-value` `5894160` merge(문서·JSON 3개 파일, 충돌 없음) | 완료 |
| frozen baseline 해시 | PASS |
| Rapfi 실제 통합, 재현성, 금수 교차검사 | 완료(§7) |
| 전체 회귀, torch 없는 환경(`ci_run_tests.py --skip-policy torch-only`, merge 후 커밋 `f8e09e3`) | 751개 실행, 576 PASS, 173 skip(torch 필요), **2 ERROR**. 두 ERROR는 torch를 skip guard 없이 import하는 기존 테스트(`test_h6_value` `ProgressBlockingTest.test_bounded_map_heartbeats_while_oldest_runs`, `test_puct_v8` `ScriptsTest.test_calibration_rule`)이며 이 브랜치가 바꾼 파일과 관련이 없다. 같은 이유로 CI `core-no-torch`에서도 실패할 수 있다. frozen baseline PASS |
| 전체 회귀, torch 있는 환경(CI `full-tests`, skip 0) | **미실행.** 클라우드에서 torch를 받을 수 없었다. CI는 PR이나 수동 실행(`workflow_dispatch`)에서만 돈다 |
| **merge** | **보류.** CI 네 job(`core-no-torch`, `full-tests`, `fake-smoke`, `frozen-baseline`)이 PASS한 뒤 다음 단계로 합친다 |
