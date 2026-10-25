# H7-c 강화학습 자동화 계약 (Auto-v1, C2_READY 시작)

사용자 승인 2026-10-25 (설계안 → 검토 2회 반영 rev 3). 구현: `src/hybrid/h7c_auto.py`(순수 전이), `scripts/h7c_auto.py`
(evidence 로딩 · SHA 사슬 · CLI), `tests/test_h7c_auto.py`.

## 범위와 원칙

- Auto-v1은 **evidence 검증 + 순수 상태 전이 + 다음 명령 제시**만 한다. 아무것도 실행하지 않는다.
- 가장 먼저 보장하는 성질: 어떤 evidence 조합에서도 잘못 다음 단계로 진행하지 않는다. 모르면 멈춘다.
- 판정식을 재구현하지 않는다. 전이는 기록된 값만 쓴다.
  - Arena: verdict의 `promotion`(`decision`, `reason`, `improvement_lower`), `candidate_mode`, `next_champion`
  - SB: `arms[*].stronger`, `saturation.reading`
  - B 신호 · 비용: decision 파일. `verify`만 `h7c_decide.decide`로 c1 decision을 다시 계산해 커밋된 파일과 비교한다.
- 아래 정규 근사 수치(s ≥ 0.540 등)는 설명용이다. 코드 상수가 아니다.
- `runs/h7c/automation/state.json`은 cache다. `status`가 evidence에서 매번 다시 쓴다.

```bash
python scripts/h7c_auto.py status        # cycle, 상태, verdict 이력, 사슬
python scripts/h7c_auto.py next          # NEXT_ACTION과 (도구가 있으면) 명령
python scripts/h7c_auto.py explain       # 근거(보류 창, 정체, preflight 등) JSON
python scripts/h7c_auto.py verify        # frozen checker, c1 decision 재계산, verdict 사슬, recipe SHA, evidence-only
python scripts/h7c_auto.py recipe --cycle 2   # G0 / G0b evidence 커밋 뒤, clean tree에서 docs/h7c/c2/recipe.json
python scripts/h7c_auto.py manifest --cycle 2 # labels 뒤: runs/의 원본 파일 SHA → docs/h7c/c2/evidence_manifest.json
python scripts/h7c_auto.py index --cycle 2    # 검사를 통과한 evidence의 SHA만으로 docs/h7c/c2/cycle.json
```

**시작 조건 (C2_READY).** c1 decision · cost review(`applies_to_cycle: 2`, decision SHA 일치), c1 verdict, G0 / G0b 계약
전체(`hybrid.h7c_auto.g0_errors`), 그리고 커밋된 `docs/h7c/c2/recipe.json`. `check`의 PASS만으로는 계약이 아니다
(`--games-limit`, `--policy-rows` 없이, 한 mode만으로도 PASS가 나온다). 그래서 다음을 모두 본다.

| | G0 (`docs/h7c/c1/retarget_g0.json`) | G0b (`retarget_g0b.json`) |
|---|---|---|
| 형식 · 판정 | `h7c-retarget-v1-check`, purpose `g0`, PASS, failures 없음 | 같음 |
| factor · 판 수 | 1, 1,000판(coverage games = 1,000) | 4, 20판 |
| 행 | rows = coverage eligible | 같음 |
| 비교 | modes = [restored], `actor` = {restored: {}}(차이 0), v1 projection 있음 ∧ differ 0 ∧ missing 0 ∧ compared > 0 | modes = {restored, sequential}, `modes_differ` = {} |
| 출처 | c1 self-play JSONL SHA(`evidence_manifest.json`) · self-play identity, clean commit, 그 commit이 HEAD의 조상 | 같음 |

2026-10-25 현재 이 checkout에는 G0 / G0b evidence가 없어서 `NOT_C2_READY`다.

**커밋 순서.** A = 자동화 코드 · 테스트 · 이 문서 → G0 / G0b evidence 커밋 → B = `recipe.json`만(`code_base_commit` = B 직전 HEAD).
recipe는 그 뒤의 코드 변경을 막으므로, 자동화 코드 수정은 모두 recipe 전에 끝낸다.

## 고정 규칙과의 대조

| 항목 | 고정 규칙 / 저장소 사실 | 자동화 |
|---|---|---|
| Arena | 공식 = 후보 대 직전 champion(엔진 동결 TB-V3 S50), seed 8730 + c, 50쌍. 승급 = score > 0.5 ∧ bootstrap 하한 ≥ −0.05 ∧ 전술 v2 PASS. opponent illegal → `OFFICIAL_INVALID` | 그대로 읽음. c2 seed 8732 |
| champion | verdict `next_champion`, 다음 cycle의 `champion_record_sha256` = 직전 verdict SHA | 사슬 검사. `cycle.json`은 색인 |
| Arena bundle | JOINT 6 arm + tactical; FALLBACK = official + mon×3 + tactical. verdict가 commit 등 동일성 검사 | 한 HEAD에서 arm 전부 + tactical + verdict, **중간 커밋 금지** |
| no-candidate | KEEP, 하한 null, 평가 cycle 아님, 정체 연속을 끊음 | R2 |
| E2 | 연속 2 평가 cycle KEEP ∧ 하한 ≤ 0, 보류 = 최근 ≤ 3 평가 cycle에 PROMOTE. 최소 cycle 수 없음 | **c2 직후 가능**(c1 KEEP, 하한 −0.20) |
| E2 진단 | `h7c_diagnose`는 `--p-only` 필수 | 마지막 정체 cycle이 FALLBACK → `E2_FALLBACK_DIAG_TOOLING_REQUIRED` |
| recipe | cost review는 cycle 2에만 적용; `h7c_retarget run`은 직전 decision(A / A_COST_REVIEW) + review 요구 | c2부터 recipe.json. c3는 도구 확장 전까지 **self-play 전에** `C3_RECIPE_TOOLING_REQUIRED` |
| E2 SB | `h7c_budget`은 H3 / H6 v3 + seed 8741만 | `E2_SB_TOOLING_REQUIRED` |
| retarget resume | `_execute`는 남은 게임으로 `rows_total`을 다시 세고 timer도 세션마다 0 → resume 후 cap이 절대값이 아님 | partial jsonl ∧ summary 없음 → `MANUAL_COST_RESUME_REVIEW_REQUIRED` |
| pool | Arena · SB 명시적 `spawn`; self-play · retarget · labels 기본 context | 바꾸지 않음(바꾸면 G0 / G0b · throughput 재측정) |
| workers | `h7c1_throughput.json`(self-play만): 6 / 8 / 10 / 12 / 14 → 514.2 / 510.1 / 500.8 / 481.7 / 473.0 games/h, 선택 6 | self-play · retarget · Arena = 6(recipe에 기록), labels · value cache = c1과 같은 12 |
| 학습 | NO_CANDIDATE → record + exit 1 | 정상 결과. CANDIDATE인데 metadata 없음 / SHA 불일치 / parent 불일치 → `BLOCKED_TRAIN` |
| X2 | 기록 전용, `--value-device cuda`, Rapfi 파일 경로는 머신마다 다름. fallback은 `--candidate-mode VALUE_ONLY_FALLBACK --policy-record`가 필요(기본 JOINT) | JOINT / FALLBACK이면 `X2_MONITOR`, no-candidate면 NOT_APPLICABLE. `runs/h7c/automation/x2_config.json`(커밋 안 함): 없으면 `X2_CONFIG_REQUIRED`, 형식 오류면 `BLOCKED_X2_CONFIG`(engine 비지 않은 문자열, engine_name 선택, engine_info · engine_files 문자열 목록, workers 정수 ≥ 1). 값은 `shlex.quote`로 넣는다(공백 경로) |

## 기초 수치 (설명용, 쌍 점수 σ = 0.325 정규 근사)

- SE: 25쌍 0.0650, 50쌍 0.0460, 150쌍 0.0265. 95% 반폭: 50쌍 0.090, 150쌍 0.052.
- 공식 승급 ≈ s ≥ 0.540(+28 Elo). P(PROMOTE): −35 Elo 2.5%, 0 **19%**, +17 37%, +35 59%, +52 78%, +70 90%. 비열등성 규칙이라 효과 0 후보도 약 1/5 승급한다.
- 정체(KEEP ∧ 하한 ≤ 0) ≈ s ≤ 0.590. 전술 FAIL로 KEEP이어도 하한 > 0이면 정체가 아니다.
- E2 신호 확률(효과 0 반복, 전술 PASS): c2 직후 ≈ 0.81(c1이 이미 KEEP), c3 이후는 사실상 3연속 비승급 ≈ 0.53.
- SB 50쌍 STRONGER ≈ s ≥ 0.590(+63 Elo, 80% 검출 +92 Elo). 포화 150쌍: SATURATED ≈ s < 0.523, STRONGER ≈ s ≥ 0.552.

## 상태 기계

```
C2_READY
 → SELFPLAY → EVIDENCE_COMMIT_REQUIRED
 → COST_PREFLIGHT (PASS / WARN / MANUAL_COST_PREFLIGHT_REVIEW_REQUIRED)
 → RETARGET → EVIDENCE_COMMIT_REQUIRED        (partial: MANUAL_COST_RESUME_REVIEW_REQUIRED; 실행 중 gate: BLOCKED_COST_CAP)
 → LABELS → EVIDENCE_COMMIT_REQUIRED → EVIDENCE_MANIFEST (원본 파일 SHA) → EVIDENCE_COMMIT_REQUIRED
 → POLICY_TRAIN → EVIDENCE_COMMIT_REQUIRED → VALUE_TRAIN → EVIDENCE_COMMIT_REQUIRED → TRAIN_ARTIFACT_VERIFY (BLOCKED_TRAIN)
   (각 학습 명령이 record · best.json을 docs/로 복사하고, 다음 학습은 clean tree를 요구하므로 학습마다 커밋한다)
 → CANDIDATE_RESOLVE: JOINT / VALUE_ONLY_FALLBACK → ARENA_BUNDLE | 그 밖 → NO_CANDIDATE_VERDICT
 → EVIDENCE_COMMIT_REQUIRED → X2_MONITOR | X2_NOT_APPLICABLE → EVIDENCE_COMMIT_REQUIRED
 → CYCLE_INDEX (docs/h7c/c<c>/cycle.json) → NEXT_CYCLE_DECISION
```

cycle c의 evidence(`docs/h7c/c<c>/`): `recipe.json`, `selfplay_c<c>_seed<8800+c>.json`, `retarget_s200.json`,
`labels_c<c>_report.json`, `policy_record.json` / `policy_best.json`, `value_record.json` / `value_best.json`,
`value_cache_manifest.json`, `evidence_manifest.json`, `arena_verdict.json`, `docs/external/x2_c<c>.json`, `cycle.json`.
위에서부터 첫 미충족이 현재 단계다. **파일이 있어도 검사를 통과하지 못하면 그 단계는 완료가 아니라 `BLOCKED_<단계>`다.**

**단계별 검사 (SHA 사슬, `hybrid.h7c_auto`의 `*_errors`).** 각 evidence의 코드 커밋은 recipe의 `code_base_commit`과 HEAD
사이에 있어야 한다.

| 단계 | 앞 단계와 맞춰 보는 것 |
|---|---|
| recipe | c1 decision · cost review · G0 · G0b · c1 verdict · policy / value config SHA, engine / runtime hash · value device(c1 self-play identity), workers 6 |
| parent 파일 | 명령에 넣는 parent 체크포인트 · metadata 4 SHA = 직전 verdict `next_champion`(이 머신의 `runs/`, 없으면 `BLOCKED_CHAMPION_FILES`) |
| self-play | 형식, cycle, seed 8800 + c, 모델 = parent, engine / runtime / device = recipe, 1,000판(start 0), clean |
| retarget | factor · budget · mode · decision · cost review · cap = recipe, self-play identity = self-play 요약, coverage eligible = completed = self-play visits 수, missing · duplicate · extra 0, clean |
| labels | v2 policy 행, clean, cycle · engine · runtime · model hash = self-play, 1,000판 입력, retarget 파일 SHA · coverage SHA = retarget 요약 |
| 원본 manifest (`h7c-evidence-v2`, labels 뒤 `manifest`) | `runs/`의 self-play JSONL · retarget JSONL · labels report · policy rows · value rows SHA. self-play JSONL = retarget이 기록한 것, retarget JSONL = retarget 요약, labels report = 커밋된 사본. c1의 `evidence_manifest.json`과 같은 역할. runs/는 git 밖이므로, policy 학습 전에는 다섯 파일 모두, 그 뒤 value 학습 전까지는 value rows를 이 머신의 현재 파일 SHA로 다시 계산해 manifest와 비교한다(바뀌었거나 없으면 `BLOCKED_MANIFEST`) |
| 학습 | 형식, cycle, clean, parent 2 SHA, source 모델 = parent, policy의 retarget = labels의 retarget, policy rows SHA = 원본 manifest, value의 labels SHA = value cache manifest. CANDIDATE는 metadata SHA = record. 학습 · Arena · X2(no-candidate 제외)가 남아 있는 동안에는 `runs/`의 best.pt SHA = record, parent 파일 4 SHA = `next_champion`(없거나 다르면 `BLOCKED_TRAIN` / `BLOCKED_CHAMPION_FILES`) |
| verdict | 형식, cycle, seed 8730 + c, parent, champion record = 직전 verdict SHA, candidate = record들이 정한 4 SHA(fallback이면 parent policy), mode, `next_champion` = PROMOTE면 candidate 아니면 parent. no-candidate면 두 record SHA |
| X2 | 형식, seed 8710 + c, clean, cycle, 모델 = verdict candidate |
| cycle.json | 위 파일들의 SHA와 판정으로 `index`가 다시 만든 내용과 같아야 한다. 파일만 있는 cycle.json은 cycle 완료가 아니다 |

**커밋 gate (모든 단계 전에).** working tree(untracked 포함)와 `code_base_commit..HEAD`를 본다.
- 바뀐 것이 현재 cycle의 evidence뿐(직전 cycle의 index는 커밋 전까지 허용, recipe 전 c2는 G0 / G0b 파일도)이고 미커밋이면 `EVIDENCE_COMMIT_REQUIRED`(next가 `git add … && git commit`을 출력)
- 그 밖의 경로가 있으면 `MANUAL_CODE_CHANGE_REVIEW_REQUIRED`
- verdict 사슬이 깨지면 `BLOCKED_EVIDENCE`

**Cost preflight (R4, 참고 단계).** pre = visits 행 × 2.8334 / 6, cap 10,728.5초.

| 판정 | 조건 | visits 행 |
|---|---|---|
| PASS | pre ≤ 0.95 · cap (10,192.1초) | ≤ 21,582 |
| WARN | ≤ cap | 21,583 – 22,718 |
| MANUAL_COST_PREFLIGHT_REVIEW_REQUIRED | > cap | ≥ 22,719 |

c1 수준 21,956행 → 10,368.4초(cap의 96.6%) → WARN. authoritative는 retarget 실행 중 gate(20판 + 256행, 10 / 25 / 50%)다.

## 다음 cycle 결정 (우선순위 순)

```
E         = 평가된 cycle(유효한 공식 arm, 하한 존재). c1 포함
hold      = E의 마지막 3개 중 PROMOTE ≥ 1
stagnant  = 실제 마지막 2 cycle이 모두 평가 cycle ∧ KEEP ∧ 하한 ≤ 0
1. 마지막 reason == OFFICIAL_INVALID → MANUAL_REVIEW_REQUIRED (R6)
2. 연속 no-candidate ≥ 2            → MANUAL_REVIEW_REQUIRED (R2)
3. stagnant ∧ ¬hold                 → E2 (도구 없으면 E2_SB_TOOLING_REQUIRED / E2_FALLBACK_DIAG_TOOLING_REQUIRED)
4. 다음 cycle > 10                  → BLOCKED_SEED_RANGE (R5)
5. 다음 cycle의 도구 없음            → C3_RECIPE_TOOLING_REQUIRED
6. 그 밖                            → NEXT_CYCLE (hold면 V9_HOLD)
```

## E2 episode

- 시작: `docs/h7c/e2/e<N>/plan.json`(episode, seed 8741 + N, champion, trigger cycles, 마지막 정체 cycle의 mode)을 **실행 전에 커밋**한다. 그 seed만 resume 가능하다. N > 8 → BLOCKED_SEED_RANGE.
- JOINT: E2 진단(마지막 정체 cycle) → episode의 B trigger set 고정 → SB(S100 / S200 / S400 대 S50, 50쌍, champion, plan seed) → 장기적으로 `e2_decision.json`.
- SB 판독(R3, 현재 teacher factor t = 4, STRONGER 집합 S):

| S | 상태 |
|---|---|
| t ∉ S, < t와 > t 모두 있음 | `E2_BUDGET_NONMONOTONIC_REVIEW` (우선) |
| > t 있음 | f* = min{f > t} → 비용 projection → `NEW_TEACHER_REVIEW` |
| t ∈ S, > t 없음 | `E2_CURRENT_TEACHER_STILL_STRONG_REVIEW` |
| < t만 | `E2_BUDGET_NONMONOTONIC_REVIEW` |
| 없음 | S400 150쌍 포화: STRONGER → `NEW_TEACHER_REVIEW`(f* = 8), INCONCLUSIVE → `REPLICATE_REQUIRED`, SATURATED → B 검사 |

- B 처방: B-P → B-V 순서, 고정 trigger set 기준. 처방 cycle의 recipe에 `source_e2_plan_sha256`, `intervention`, `trigger`, `teacher_factor`, `parent_recipe_sha256`.
  - 소진 = 그 cycle이 평가됨 ∧ KEEP ∧ 하한 ≤ 0. PROMOTE → episode 종료. no-candidate · OFFICIAL_INVALID · 하한 > 0 → 아직 증명 아님.
  - REPLICATE cycle의 recipe fingerprint(계보 필드 제외)는 직전 recipe와 같아야 한다.
- 모두 소진(또는 trigger 없음) → `V9_RESEARCH_READY` → `docs/h7c/v9_entry.json` → `MANUAL_APPROVAL_REQUIRED`. 자동화는 V9를 구현하지 않는다.
- E2 비용(c1 SB 수 시간 합 / 6, overhead 미포함): S100 / S200 / S400 50쌍 ≈ 580 / 990 / 1,540초 → SB ≈ 52분, 포화 추가 100쌍 ≈ 51분, 포화까지 ≈ 1시간 43분. S400 teacher는 22,068 × 5.1208 / 6 = 18,834초 = ceiling 8,582.8초의 2.19배.

## verify와 evidence-only

1. `scripts/check_track_b_frozen.py` PASS(모델이 없는 머신은 `--skip-frozen`).
2. c1 decision 재계산 = 커밋 파일, verdict 사슬, G0 / G0b 계약, recipe의 source SHA.
3. 현재 상태(`state`): 위 커밋 gate와 단계별 검사를 거친 결과. 이 문서, `src/`, `scripts/`, `configs/`, `tests/`가 바뀌면 `MANUAL_CODE_CHANGE_REVIEW_REQUIRED`.
4. exit code: 1 = 위 검사 중 FAIL이 있거나 상태가 오류(`BLOCKED_*`, `MANUAL_CODE_CHANGE_REVIEW_REQUIRED`). 0 = 실행할 단계이거나 사람의 결정 · 도구 · 설정을 기다리는 정상 정지(`NOT_C2_READY`, `*_TOOLING_REQUIRED`, `X2_CONFIG_REQUIRED`, `EVIDENCE_COMMIT_REQUIRED`, `MANUAL_REVIEW_REQUIRED` 등).

## 자동으로 바꾸지 않는 것

actor 예산, PUCT · candidate_limit, policy / value 하이퍼파라미터 · guard, 증명 / 결과 가중, 승급 규칙, Arena 엔진 · 쌍 수 · seed 식, 비용 상한, B 처방 선택, workers, pool context. 바꿔야 하면 `MANUAL_RECIPE_CHANGE_REQUIRED`.

## 운영 규칙

| | 규칙 |
|---|---|
| R1 | cap 10,728.5초 carry-forward(recipe SHA 사슬). cap 변경은 새 cost review artifact로만 |
| R2 | 연속 2회 no-candidate → MANUAL_REVIEW_REQUIRED |
| R3 | SB 분기(위 표), {< t, > t} 동시 STRONGER ∧ t ∉ S → NONMONOTONIC 우선 |
| R4 | preflight PASS ≤ 21,582 / WARN ≤ 22,718 / MANUAL ≥ 22,719; 실행 중 gate 초과만 BLOCKED_COST_CAP |
| R5 | cycle 11 · E2 9회째 BLOCKED_SEED_RANGE, E2 seed는 실행 전 예약 |
| R6 | `OFFICIAL_INVALID` → MANUAL_REVIEW_REQUIRED |
| — | E2 최소 3 평가 cycle은 넣지 않는다(사전 등록된 2-cycle 규칙 유지) |

## Ubuntu 환경 최적화 (결과 · identity 불변만)

| # | 항목 | 적용 |
|---|---|---|
| U1 | `runs/` · `data/` · checkpoint를 ext4(Linux FS)에 둔다. WSL이면 `/mnt/c`(9P) 금지 | c2 전 |
| U3 | 부모 프로세스 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`(worker 모델은 이미 `threads=1`) | c2 전 |
| — | pool: Arena · SB는 명시적 `spawn`, self-play · retarget · labels는 기본 context(Linux에서 fork 계열). 바꾸지 않는다 | — |
| U7 | native core는 `python -m rt30.vcfcore.build --if-stale`만(`-O2 -shared -fPIC` 고정). `-O3` · `-march=native` 금지. 새 머신 첫 실행에 checker PASS 필수 | c2 전 |
| U8 | `ulimit -n` 상향, RAM / swap 확인(6 worker × 모델) | c2 전 |
| U4 | `systemd-inhibit --what=sleep:idle`, `tmux` / `systemd-run --user` + `tee` 로그, CPU governor `performance` | best-effort(WSL2에서는 효과 제한), gate 아님 |
| U5 | `nvidia-smi -pm 1` | best-effort(WSL2 미지원 가능), gate 아님 |
| U2a | 새 머신이면 self-play worker sweep(`h7c_throughput.py`)만 재측정. retarget worker 근거로 쓰지 않음 | 선택 |
| — | pool context 변경, X2와 GPU 학습 겹치기, retarget · SB · 포화와 다른 작업 겹치기 | 하지 않음 |

## 고정 (Auto-v1 freeze)

이 문서와 `src/hybrid/h7c_auto.py`, `scripts/h7c_auto.py`, `tests/test_h7c_auto.py`는 이 커밋에서 고정한다. 다음 순서는
G0 / G0b 실행 · 검증 → evidence 커밋 → clean tree에서 `recipe --cycle 2` → recipe만 커밋 → `verify` → `next`(= `RUN_SELFPLAY`,
seed 8802)다. recipe 뒤에 이 파일들이 바뀌면 커밋 gate가 `MANUAL_CODE_CHANGE_REVIEW_REQUIRED`로 멈춘다. 바꿀 일이 생기면 새
cycle의 recipe 전에 별도 단계로 한다.

## 남은 도구 (별도 승인 단계)

`h7c_retarget run --recipe`(c3+), `h7c_budget`의 champion 모델 · seed 8742–8749(E2), fallback cycle용 진단, retarget 누적 cost ledger(자동 resume), Auto-v2 `run`(한 단계 실행; 학습 exit 1 + 유효 record = NO_CANDIDATE 정상 완료).

## CI

`scripts/ci_run_tests.py`가 `tests/test*.py`를 모두 찾으므로 `tests/test_h7c_auto.py`는 core-no-torch · full-tests에 이미 들어간다.
full-tests는 Ubuntu 24.04에서 native VCF core(`-O2 -shared -fPIC`)를 빌드하고 `check_track_b_frozen.py --no-models`로 코드 ·
엔진 · runtime을 검사한다(체크포인트 없음). workflow는 PR, main push, 수동 실행에서만 돈다(feature branch push 제외).
