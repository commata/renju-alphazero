# R2-S400-PRESERVE-1: 같은 S400 학습 자료에서 H3 보존을 강하게 하면 후보가 좋아지는가 (초안)

- 작성 2026-11-19. 상태: **계약 고정(fixed), 모든 단계 미승인**(prepare·suites·gate·train·arena 모두 false). 도구·테스트·데스크톱 블록 준비(§10, §11). suite는 아직 만들지 않았다(suites 단계).
- 결정(2026-11-19 검토): 보존 계수 `μ = 0.5` 대 `μ = 2.0`, 새 O1 계열 개발 suite 두 묶음, 독립 확인은 별도 계약.
- 이 계약은 개발 판독까지다. champion 변경, 독립 확인, 강화학습 루프 재개는 하지 않는다.

## 0. 근거

| 기록 | 내용 |
|---|---|
| R2-P1-POLICY-REPLAY-A1 (`389d09a`) | S200 목표, 주 비교 학습 seed 20261024 하나. H3 보존 항을 더한 T가 대조군 C보다 +3.17%p [+1.08, +5.25] (D BETTER_THAN_CONTROL). T의 champion 대비 우위는 입증되지 않았다(P −1.29%p, NOT_ESTABLISHED) |
| R2-S400-TRAIN (`e1aa371`) | S400 목표 후보 S(μ 0.5)는 champion 대비 −2.00%p [−3.92, −0.08] WEAKER. S − T는 +0.46%p [−0.92, +1.83] NOT_ESTABLISHED |
| R2-S400-VT-MEASURE-1 (`0b2622e`) | S400 값 목표 경로 종료(Q1·Q2 INCONCLUSIVE) |

- 지금까지 같은 S50 예산의 짝 Arena에서 개선이 확인된 변경은 H3 보존 항을 더한 것 하나다. 보존을 더 강하게 했을 때의 효과는 검증된 적이 없다.
- 현재 자료와 학습법에서는 S400 목표의 우위가 입증되지 않았다. 자료 부족, 시작 국면 분포, checkpoint 선택은 각각 따로 검증할 가설이다. 이 계약은 그중 어느 것도 다루지 않고 보존 계수 하나만 바꾼다.
- 기존 S400 학습 자료를 재사용한다. 한 번의 S400 재탐색에 약 4.5시간이 들었다(R2-S400-TRAIN §8.6).

## 1. 질문과 범위

- **질문**: R2-S400-TRAIN과 같은 S400 학습 행, 같은 seed, 같은 학습량, 같은 선택 규칙에서 보존 계수만 0.5에서 2.0으로 올리면 (1) 같은 opening에서 기존 S400 후보보다 좋아지는가, (2) champion을 넘는가.
- `μ = 2.0`은 사전에 고른 실험값이다. 코드에서 μ는 replay KL의 계수다(`loss = cycle batch mean + μ × replay KL mean`, `src/hybrid/h7r_p1_train.py`). KL과 cycle 손실의 숫자 크기만으로는 μ가 실제 모델 갱신에 미치는 영향을 비교할 수 없다. 이 계약은 **기존 계수의 4배에서 더 강한 보존이 대국 성능에 도움이 되는지를 한 번 검증한다.**
- 결론의 범위는 사전 지정 학습 seed 20261024의 두 후보다. 학습 seed 간 변동은 판독 구간에 들어 있지 않다.
- 하지 않는 것: 자료 규모·시작 국면 분포·학습량·선택 규칙 변경, 다른 μ 값 탐색, 결과를 본 뒤의 표본 보충.

## 2. 고정할 비교

| 항목 | S(μ 0.5) — 기존 | S(μ 2.0) — 이번 |
|---|---|---|
| 학습 행 | R2-S400-TRAIN 후보 행 `runs/h7r/R2-S400-TRAIN/policy_rows_s400.jsonl.gz` (SHA `add6022ca8d4…`, 26,000행) | 같음 |
| cycle 항 | c2 정책 방법(CE + λ_kl 0.5 · KL(H3‖student)), batch 512, 2 epoch = 96 step, warmup 5, lr 2e-4, weight decay 1e-4, min lr ratio 0.05, D4 증강 | 같음 |
| replay 항 | RenjuNet train 512국면, 균등 복원 추출, 생성기 seed = 학습 seed + 1000003(D4 대칭 포함) | **같은 생성기, 같은 표본**. 계수만 2.0 |
| 선택 규칙 | RenjuNet val top-1 guard가 H3에서 0.02 이내인 평가 중 cycle val objective 최저(replay 항은 선택 지표가 아님) | 같음 |
| 평가 step | 10, 19, 29, 38, 48, 58, 67, 77, 86, 96 | 같음 |
| 결정론 설정 | R2-S400-TRAIN과 같음(`CUBLAS_WORKSPACE_CONFIG=:4096:8`, deterministic algorithms, TF32 matmul off 등) | 같음 |
| 학습 seed | 20261024 / 20261111 / 20261112 | 같음 |
| 실행 | 새로 학습하지 않는다. R2-S400-TRAIN의 기록과 checkpoint를 SHA로 쓴다(S 20261024: checkpoint `68928bc6ef8c…`, metadata `ed75576736ca…`) | 이 계약에서 세 seed를 학습한다 |

- 같은 seed의 두 실행은 cycle batch 순서, D4 대칭, 학습률, replay 국면이 같다. 다른 것은 replay 항의 계수와 그 결과뿐이다(지문의 batch·sym·lr SHA가 같아야 하고 loss만 달라야 한다).
- 해석용 보고(판정 아님): 평가마다의 RenjuNet val 전체 H3 KL과 guard top-1, 선택 step, cycle val objective.
  - 참고로 S(μ 0.5, 20261024)는 step 96 선택, RenjuNet val H3 KL 0.0437, guard top-1 0.5168(H3 0.5241)이었다.

## 3. 재현 gate (학습 전)

- **질문**: 지금 환경과 도구가 S(μ 0.5, 20261024)를 R2-S400-TRAIN과 정확히 같게 다시 만드는가. Arena와는 다른 질문이다.
- 방법: S(μ 0.5, 20261024)를 같은 진입점(`h7r_p1a1.execute`, kind T, replay μ 0.5)과 같은 수치 설정으로 다시 학습하고, 커밋된 R2-S400-TRAIN seed 기록과 비교한다. 허용 오차 0.
  - 비교 항목: 학습 지문(step 수, batch·sym·lr SHA, loss 열), 평가곡선 전체(평가마다의 val·guard·eligible·replay KL 기록), 선택 step과 선택 state SHA(`106e0ca5cdb6…`), 고정 국면 5,519개 log-확률 SHA(`0ccf8161f969…`), 수치 설정, 입력 SHA(후보 행, c2 행, S400 기록).
- 다시 만든 checkpoint는 Arena에 쓰지 않는다. **Arena에 쓸 S(μ 0.5) checkpoint 파일은 따로 SHA를 확인한다**(R2-S400-TRAIN seed 기록의 candidate SHA와 같아야 한다).
- gate가 실패하면 학습하지 않는다(R2-S400-TRAIN gate가 A1 T를 비트 단위로 재현한 `2e894cf`와 같은 규칙).

## 4. 학습

- `μ = 2.0`으로 세 seed를 학습한다. 실행마다 따로 커밋한다(R2-S400-TRAIN s5-T와 같은 방식: 실행 기록 sidecar, 기록 사본, checkpoint 대조, 재개 시 커밋된 seed의 파일 재대조).
- μ 하나의 효과만 보도록, 같은 seed의 S(μ 0.5) 기록과 step 수·batch·대칭·학습률 지문이 모두 같은 실행만 저장한다. 하나라도 다르면 후보도 기록도 남기지 않는다(손실과 선택 state는 μ에 따라 달라진다).
- 학습 전에 후보 행 파일을 R2-S400-TRAIN rows 기록의 SHA와 행 검사로 다시 확인하고(`verified_rows`), 커밋된 gate 기록을 다시 판정한다.
- 기록과 checkpoint 메타데이터에 남길 것: μ, 후보 행·c2 행·S400 기록 SHA, gate 기록 SHA, R2-S400-TRAIN 계약과 실험 SHA, 같은 seed의 S(μ 0.5) 기록 SHA.
- Arena 진입: 사전 지정 seed 20261024의 S(μ 2.0)가 `CANDIDATE`여야 한다. 아니면 다른 seed로 대신하지 않고 계약을 끝낸다.

## 5. 새 개발 suite (O1 계열, 300개씩 두 묶음)

- 규칙: O1(5수: 중앙, 백·흑 반경 2, 백·흑 반경 3. 백 차례), 묶음마다 서로 다른 300개 동치류(opening마다 다른 동치류).
- 제외(5수 opening 동치류 단위): 기존 suite O0·O1·O2·O1-confirm-main·O1-confirm-anchor·**O1-dev-a·O1-dev-b**, 커밋된 모든 v1·v2 평가 계약의 arm 표본(R2-S400-TRAIN-ARENA-1 포함). 두 묶음 사이에도 같은 동치류가 없어야 한다. 3수 opening과의 prefix 겹침은 건수만 기록한다.
- 생성 seed: **202611191**(묶음 c), **202611192**(묶음 d). 기존 예약(`RESERVED_SEEDS`, v2 Arena·suite seed, 학습 run seed 8801–8999, 측정 seed 8800, S1 probe 9201)과 겹치지 않음을 확인했다(2026-11-19, `a97a9c1`).
- 고정 파일 보존: `src/hybrid/h7r_seeds.py`(R2-S400-TRAIN 계약의 sources)와 `scripts/h7r_dev_suites.py`(O1-dev-a·b 검사)는 바꾸지 않는다.
  - 새 생성 seed는 별도 등록(새 모듈 또는 등록 파일)에 고정한다. 새 suite 생성 경로(새 도구)가 그 등록, 제외 목록과 출처 SHA를 기록한다.
  - 기존 `h7r_dev_suites check`는 계속 통과해야 한다(새 suite와 새 Arena 계약이 함께 있어도).
- 파일: `docs/h7r/suites/O1-dev-c.json`, `O1-dev-d.json`, 생성 기록. 생성은 CPU 작업이라 클라우드에서도 할 수 있다(별도 승인 단계).

## 6. 개발 Arena

- v2 Arena 계약 `R2-S400-PRESERVE-ARENA-1`(`docs/h7r/contracts-v2/`, `h7r-eval-contract-v2`, seed policy `h7r-seeds-v2`).
  - 대국 seed: **9103**(O1-dev-c), **9104**(O1-dev-d). 기존 v2 범위(9101–9140) 안이고, `a97a9c1`의 기존 계약은 쓰지 않는다. 계약 생성 때 v1·v2 전체 계약과의 중복을 다시 검사한다(`h7r_eval_v2.check_unique`).
  - 새 opening 생성 seed 때문에 대국 seed 검사를 올리지 않는다. v2 Arena 검사기는 suite의 SHA와 형식을 확인하고 대국 seed를 따로 검사한다.
- 모델: champion(H3 정책 + c5 value), S(μ 2.0, 20261024), S(μ 0.5, 20261024). 모두 c5 value, S50. 대국 전에 세 모델의 네 파일 SHA를 계약과 대조한다(R2-S400-TRAIN의 `check-models`와 같다).
- arm(suite마다 같은 opening·seed·pairing, 각 300쌍)
  - S(μ 2.0) 대 champion
  - S(μ 0.5) 대 champion (같은 suite에서 기존 후보를 다시 둔다. 그래서 두 후보의 짝 비교가 된다)
- 판독(형식은 R2-S400-TRAIN과 같다: suite 층화, 50:50 고정 가중, opening 단위 쌍 대응, 단측 α 0.025, bootstrap 20,000회, seed **20261161**, class 단위 민감도는 보고만)
  - **D″ = S(μ 2.0) − S(μ 0.5)**: 같은 opening에서 두 계수의 성능 차이. BETTER(하한 > 0) / WORSE(상한 < 0) / NOT_ESTABLISHED
  - **P″ = S(μ 2.0) 대 champion 점수 − 0.5**: μ 2.0 후보의 champion 대비 우위. STRONGER / WEAKER / NOT_ESTABLISHED
  - 판독은 두 suite에서 S(μ 2.0)와 S(μ 0.5)의 opening 짝이 모두 일치할 때만 계산한다. 네 arm은 같은 깨끗한 commit에서 두고 judge도 그 commit에서 한다. 300쌍은 줄이지 않는다.
- 비용: arm 4개, R2-S400-TRAIN Arena 기준 약 2.4시간.

## 7. 결과의 쓰임 (사전 고정)

| P″ | D″ | 기록과 다음 |
|---|---|---|
| STRONGER | BETTER | 독립 확인 계약을 연다(새 표본·새 seed). 보존 강화가 기존 후보 대비 개선으로 기록된다 |
| STRONGER | WORSE | 독립 확인 계약을 열 수 있다. 보존 강화는 기존 μ 0.5 후보 대비 악화로 기록한다 |
| STRONGER | NOT_ESTABLISHED | 독립 확인 계약을 열 수 있다. 보존 강도를 높인 효과는 미확정으로 기록한다 |
| 그 밖 | BETTER | 기존 후보 대비 개선으로 기록하고 이 계약을 끝낸다. champion을 넘지 못했으므로 확인으로 가지 않는다 |
| 그 밖 | 그 밖 | 이 계약을 끝낸다 |

- P″가 STRONGER가 아닐 때 D″ WORSE는 기존 μ 0.5 후보 대비 악화로 기록하고 계약을 끝낸다. WEAKER·WORSE는 따로 적는다. 진입 여부는 위 표대로다.
- 개발 Arena에서 champion을 넘은 후보만 독립 확인으로 보낸다. 독립 확인의 통과 여부는 이후 강화학습 재개를 검토할 때 쓴다.
- 결과를 보고 μ, suite, seed, 판독 설정을 바꾸지 않는다. 바꾸려면 새 계약이다.

## 8. 단계와 비용 (모두 미승인)

| 단계 | 하는 일 | 위치 | 예상 시간 |
|---|---|---|---|
| prepare | 입력 대조: R2-S400-TRAIN 계약·rows·search·seed 기록, 후보 행 SHA와 행 검사, S(μ 0.5) checkpoint SHA, A1 입력 | 데스크톱 | 수 분 |
| suites | 새 개발 suite 두 묶음 생성과 겹침 검사 기록 | 클라우드 가능 | 수 분 |
| gate | S(μ 0.5, 20261024) 재현(허용 오차 0) | 데스크톱 | 약 2분 |
| train | S(μ 2.0) 세 seed | 데스크톱 | seed마다 약 2분 |
| arena | Arena 계약, 모델 대조, 대국 4 arm, 판독 | 데스크톱(tmux) | 약 2.5시간 |

- 단계마다 따로 승인한다. gate가 통과해야 train, 20261024가 `CANDIDATE`여야 arena다.
- 원자료(checkpoint, arm 기록, journal)는 `runs/` 아래에 두고 커밋하지 않는다. 기록·요약·로그·SHA를 커밋한다.

## 9. 남은 준비 (결정 뒤, 미승인)

- 도구(새 파일): R2-S400-TRAIN의 고정 도구(`scripts/h7r_s400_train.py`, 계약 sources)를 바꾸지 않고 재사용한다(`verified_rows`, `a1.execute`, 실행 기록 sidecar, Arena 도구의 대국·판독 경로). 새로 만들 부분은 μ 2.0 학습과 gate 비교, 새 suite 생성 경로와 별도 seed 등록, μ 0.5 checkpoint의 SHA 연결, 판독 정의(D″, P″).
- 테스트: gate의 정확 일치(지문·평가곡선·state·log-확률 하나라도 다르면 거부), μ만 다르고 replay 표본과 지문이 같음, 기존 고정 파일과 기존 suite 검사 유지, 새 suite의 제외·겹침·재생성, Arena 계약의 seed 중복 검사, 판독 표, 데스크톱 블록(실제 bash, 도구는 대역).
- 계약 JSON 고정(fixed, 모든 승인 false)과 데스크톱 블록.

## 10. 구현 (2026-11-19, 미승인)

- 계약 `docs/h7r/r2/R2-S400-PRESERVE-1/preserve-contract.json`(fixed, 모든 승인 false, 실험 SHA `d832623f66ec`). R2-S400-TRAIN 계약의 파일 SHA(`7f4a5c00a126`)와 실험 SHA(`67c6ba3c8081`), rows 기록 SHA, 후보 행 SHA, μ 0.5 seed 기록 세 개와 20261024 checkpoint SHA를 고정한다. 도구와 재사용 파일 12개를 sources로 고정한다.
- 도구 `scripts/h7r_preserve.py`(R2-S400-TRAIN의 고정 도구·계약·기록은 바꾸지 않고 읽어서 재사용)

| 명령 | 단계 | 하는 일 |
|---|---|---|
| `prepare` | prepare | R2-S400-TRAIN 계약(고정 SHA)과 gate·rows 기록, 후보 행(SHA와 행 검사), μ 0.5 seed 기록, Arena에 쓸 20261024 checkpoint 파일, A1 입력 대조 |
| `gate` | gate | 커밋된 두 suite와 생성 기록을 다시 계산해 확인한 뒤(`h7r_dev_suites_2.check`) S(μ 0.5, 20261024)를 `h7r_p1a1.execute`로 다시 학습해 커밋된 R2-S400-TRAIN 기록과 정확히 비교(`h7r_s400_train.gate_view`의 지문·평가곡선·선택·state·log-확률·설정, 입력 SHA). 이 checkpoint는 쓰지 않는다. FAIL은 종료 코드 2 |
| `train --seed S` | train | 커밋된 gate를 다시 판정한 뒤 S(μ 2.0) 학습. R2-S400-TRAIN 계약의 사본에서 replay 계수만 2.0으로 바꿔 같은 진입점에 넘긴다(`with_mu`). 같은 seed의 μ 0.5 기록과 step 수·batch·대칭·학습률 지문이 다르면 아무것도 쓰지 않는다. 기록에 μ, 입력, gate 기록 SHA, 같은 seed의 μ 0.5 기록 SHA, 지문 일치 |
| `verify-run --seed S` | train | 커밋된 run 기록의 checkpoint 파일 대조(CANDIDATE) 또는 파일 없음(NO_CANDIDATE) |
| `arena-contract`, `check-contract` | arena | 커밋된 새 suite(재계산 검사 통과)와 S2 20261024 기록으로 v2 Arena 계약, seed·id 중복 검사 뒤 기록 |
| `check-models`, `play`, `judge`, `verify-judge` | arena | R2-S400-TRAIN Arena와 같은 방식. 판독은 D″·P″, 표는 §7 |

- suite: `src/hybrid/h7r_suite_seeds_dev2.py`(생성 seed 202611191·202611192 등록과 충돌 검사), `scripts/h7r_dev_suites_2.py`(`build` / `check`, `h7r_dev_suites.py`의 출처 읽기·동치류 함수 재사용, 기록 `docs/h7r/suites/O1-dev-cd-record.json`). 미리 계산한 결과(쓰지 않음): 제외 동치류 2,549개, 두 묶음 모두 서로 다른 300개 동치류, 출처·두 묶음 사이 겹침 0.
- suites 단계는 데이터가 필요 없어 `approved.suites` 뒤 클라우드에서 `build`로 만들고(`build`가 커밋된 계약의 suites 승인과 prepare 기록 커밋을 확인한다) 깨끗한 commit에서 커밋한다(O1-dev-a·b와 같은 방식). 데스크톱 블록은 없다.
- 테스트 `tests/test_h7r_preserve.py`: §9의 항목.

## 11. 데스크톱 실행 (승인 뒤에만)

- pv-P는 `approved.prepare` 뒤, pv-G는 `approved.gate` 뒤, pv-T는 `approved.train` 뒤, pv-AC와 pv-A는 `approved.arena` 뒤에 실행한다. suite는 그 전에 커밋돼 있어야 한다(arena-contract가 확인한다).
- pv-P·pv-G·pv-T는 결과에 로그 이름을 남긴다. 결과가 커밋 전이면 다시 돌리지 않고, 로그의 종료 줄을 확인한 뒤(없으면 `Recovered exit code`) 함께 커밋한다.
- pv-T: seed마다 따로 커밋한다. 각 seed의 기록을 스테이징하기 전에 checkpoint·메타데이터를 기록과 대조한다(`verify-run`, 정상 실행과 복구 모두). 다르면 아무것도 커밋하지 않고 멈춘다. 커밋된 seed는 파일을 다시 대조하고 건너뛴다. 실패하면 기록하고 다음 seed 전에 멈춘다. 결과 없이 파일만 남은 중단은 멈추고 보고한다(덮어쓰지 않는다).
- pv-A: R2-S400-TRAIN s5-A와 같다(대국 전 모델 확인, HEAD 고정, 실패 시 커밋 없음과 재개, judge 뒤 중단은 verify-judge로 확인 후 커밋).

pv-P. prepare

```bash
(
  set -euo pipefail
  cd ~/renju-h6
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src:$PWD/scripts"
  [ "$(git branch --show-current)" = feat/h7-c1 ] || { echo "STOP: wrong branch"; exit 1; }
  git diff --quiet || { echo "STOP: unstaged tracked changes"; exit 1; }
  if git diff --cached --quiet; then git pull --ff-only origin feat/h7-c1; fi
  N=prepare
  D=docs/h7r/r2/R2-S400-PRESERVE-1
  L=runs/h7r/R2-S400-PRESERVE-1/logs
  O="$D/$N.json"
  tracked(){ git cat-file -e "HEAD:$1" 2>/dev/null; }
  if tracked "$O"; then echo "STOP: $N is already recorded"; exit 1; fi
  mkdir -p "$L"
  rc=0
  if [ ! -e "$O" ]; then
    git diff --cached --quiet || { echo "STOP: resolve staged changes before a new run"; exit 1; }
    python scripts/check_track_b_frozen.py
    log="$(mktemp "$L/$N-XXXXXXXX.log")"
    set +e
    python -u scripts/h7r_preserve.py "$N" --log-name "$(basename "$log")" 2>&1 | tee "$log"
    codes=("${PIPESTATUS[@]}")
    set -e
    rc="${codes[0]}"
    [ "${codes[1]}" -eq 0 ] || { echo "STOP: tee failed; log: $log"; exit 1; }
    echo "__EXIT__ $rc" >> "$log"
  else
    echo "an uncommitted $O exists: recording it (no new run)"
  fi
  if [ -f "$O" ]; then
    result="$(python - "$O" "$N" <<'PY'
import json, re, sys
from pathlib import Path
r = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
name = r.get('log_name', '')
if r.get('format') != 'h7r-preserve-prepare-v1' or not re.fullmatch(sys.argv[2] + r'-[A-Za-z0-9]+\.log', name):
    sys.exit('STOP: invalid result or missing attempt log name')
print(name, 0 if r.get('passed') is True else 2)
PY
)"
    read -r name rc <<< "$result"
    log="$L/$name"
    [ -f "$log" ] || { echo "STOP: the result's log is missing: $log"; exit 1; }
    last="$(tail -n 1 "$log")"
    case "$last" in
      "__EXIT__ $rc") ;;
      __EXIT__*) echo "STOP: the result and its log disagree on the exit code"; exit 1 ;;
      *) printf '\n%s\n__EXIT__ %s\n' "Recovered exit code from $O" "$rc" >> "$log" ;;
    esac
  elif [ "$rc" -eq 0 ] || [ "$rc" -eq 2 ]; then
    echo "STOP: $N did not produce a result"; exit 1
  fi
  paths=()
  shopt -s nullglob
  for f in "$L/$N"-*.log; do
    [[ "$(tail -n 1 "$f")" =~ ^__EXIT__\ [0-9]+$ ]] || continue
    n="$(basename "$f" .log)"
    tracked "$D/stage-logs/$n.log" && continue
    python scripts/h7r_d4_records.py copy-logs --resume --src-dir "$L" --pattern "$n.log" \
      --dest-dir "$D/stage-logs" --manifest "$D/stage-logs/$n.manifest.json"
    paths+=("$D/stage-logs/$n.log" "$D/stage-logs/$n.manifest.json")
  done
  if [ -f "$O" ]; then paths+=("$O"); fi
  [ "${#paths[@]}" -gt 0 ] || { echo "STOP: nothing to record"; exit 1; }
  while IFS= read -r -d '' staged; do
    allowed=false
    for f in "${paths[@]}"; do if [ "$f" = "$staged" ]; then allowed=true; break; fi; done
    "$allowed" || { echo "STOP: unrelated staged path: $staged"; exit 1; }
  done < <(git diff --cached --name-only -z)
  expect=()
  for f in "${paths[@]}"; do expect+=(--expect "$f"); done
  git add -- "${paths[@]}"
  python scripts/h7r_d4_records.py verify-staged "${expect[@]}"
  git commit -m "H7-R PRESERVE: $N exit $rc (records only)"
  git push origin feat/h7-c1
  echo "__RECORDED__ $rc"
  exit "$rc"
)
```

pv-G. 재현 gate

```bash
(
  set -euo pipefail
  cd ~/renju-h6
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src:$PWD/scripts"
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
  [ "$(git branch --show-current)" = feat/h7-c1 ] || { echo "STOP: wrong branch"; exit 1; }
  git diff --quiet || { echo "STOP: unstaged tracked changes"; exit 1; }
  if git diff --cached --quiet; then git pull --ff-only origin feat/h7-c1; fi
  N=gate
  D=docs/h7r/r2/R2-S400-PRESERVE-1
  L=runs/h7r/R2-S400-PRESERVE-1/logs
  O="$D/$N.json"
  tracked(){ git cat-file -e "HEAD:$1" 2>/dev/null; }
  if tracked "$O"; then echo "STOP: $N is already recorded"; exit 1; fi
  tracked "$D/prepare.json" || { echo "STOP: prepare is not recorded (run pv-P first)"; exit 1; }
  mkdir -p "$L"
  rc=0
  if [ ! -e "$O" ]; then
    git diff --cached --quiet || { echo "STOP: resolve staged changes before a new run"; exit 1; }
    python scripts/check_track_b_frozen.py
    log="$(mktemp "$L/$N-XXXXXXXX.log")"
    set +e
    python -u scripts/h7r_preserve.py "$N" --device cuda --log-name "$(basename "$log")" 2>&1 | tee "$log"
    codes=("${PIPESTATUS[@]}")
    set -e
    rc="${codes[0]}"
    [ "${codes[1]}" -eq 0 ] || { echo "STOP: tee failed; log: $log"; exit 1; }
    echo "__EXIT__ $rc" >> "$log"
  else
    echo "an uncommitted $O exists: recording it (no new run)"
  fi
  if [ -f "$O" ]; then
    result="$(python - "$O" "$N" <<'PY'
import json, re, sys
from pathlib import Path
r = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
name = r.get('log_name', '')
if r.get('format') != 'h7r-preserve-gate-v1' or not re.fullmatch(sys.argv[2] + r'-[A-Za-z0-9]+\.log', name):
    sys.exit('STOP: invalid result or missing attempt log name')
print(name, 0 if r.get('passed') is True else 2)
PY
)"
    read -r name rc <<< "$result"
    log="$L/$name"
    [ -f "$log" ] || { echo "STOP: the result's log is missing: $log"; exit 1; }
    last="$(tail -n 1 "$log")"
    case "$last" in
      "__EXIT__ $rc") ;;
      __EXIT__*) echo "STOP: the result and its log disagree on the exit code"; exit 1 ;;
      *) printf '\n%s\n__EXIT__ %s\n' "Recovered exit code from $O" "$rc" >> "$log" ;;
    esac
  elif [ "$rc" -eq 0 ] || [ "$rc" -eq 2 ]; then
    echo "STOP: $N did not produce a result"; exit 1
  fi
  paths=()
  shopt -s nullglob
  for f in "$L/$N"-*.log; do
    [[ "$(tail -n 1 "$f")" =~ ^__EXIT__\ [0-9]+$ ]] || continue
    n="$(basename "$f" .log)"
    tracked "$D/stage-logs/$n.log" && continue
    python scripts/h7r_d4_records.py copy-logs --resume --src-dir "$L" --pattern "$n.log" \
      --dest-dir "$D/stage-logs" --manifest "$D/stage-logs/$n.manifest.json"
    paths+=("$D/stage-logs/$n.log" "$D/stage-logs/$n.manifest.json")
  done
  if [ -f "$O" ]; then paths+=("$O"); fi
  [ "${#paths[@]}" -gt 0 ] || { echo "STOP: nothing to record"; exit 1; }
  while IFS= read -r -d '' staged; do
    allowed=false
    for f in "${paths[@]}"; do if [ "$f" = "$staged" ]; then allowed=true; break; fi; done
    "$allowed" || { echo "STOP: unrelated staged path: $staged"; exit 1; }
  done < <(git diff --cached --name-only -z)
  expect=()
  for f in "${paths[@]}"; do expect+=(--expect "$f"); done
  git add -- "${paths[@]}"
  python scripts/h7r_d4_records.py verify-staged "${expect[@]}"
  git commit -m "H7-R PRESERVE: $N exit $rc (records only)"
  git push origin feat/h7-c1
  echo "__RECORDED__ $rc"
  exit "$rc"
)
```

pv-T. S(μ 2.0) 세 seed 학습

```bash
(
  set -euo pipefail
  cd ~/renju-h6
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src:$PWD/scripts"
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
  [ "$(git branch --show-current)" = feat/h7-c1 ] || { echo "STOP: wrong branch"; exit 1; }
  git diff --quiet || { echo "STOP: unstaged tracked changes"; exit 1; }
  if git diff --cached --quiet; then git pull --ff-only origin feat/h7-c1; fi
  D=docs/h7r/r2/R2-S400-PRESERVE-1
  R=runs/h7r/R2-S400-PRESERVE-1
  L="$R/logs"
  T=scripts/h7r_preserve.py
  tracked(){ git cat-file -e "HEAD:$1" 2>/dev/null; }
  tracked "$D/gate.json" || { echo "STOP: the gate is not recorded (run pv-G first)"; exit 1; }
  mkdir -p "$L"
  command -v flock > /dev/null || { echo "STOP: flock is missing (util-linux)"; exit 1; }
  exec 9> "$L/train.lock"
  flock -n 9 || { echo "STOP: another train block is running"; exit 1; }
  shopt -s nullglob
  frozen=false
  for s in 20261024 20261111 20261112; do
    out="$R/S2-$s"
    O="$out/record.json"
    dst="$D/seed-records/S2-$s/record.json"
    if tracked "$dst"; then  # recorded: its checkpoint files must still be the ones the record names
      python "$T" verify-run --seed "$s"
      echo "done: S2-$s"
      continue
    fi
    rc=0
    if [ ! -e "$O" ]; then
      git diff --cached --quiet || { echo "STOP: resolve staged changes before a new run"; exit 1; }
      left=("$out"/*)
      [ "${#left[@]}" -eq 0 ] || { echo "STOP: an interrupted run left files in $out without its record: report it (nothing is overwritten)"; exit 1; }
      if ! "$frozen"; then python scripts/check_track_b_frozen.py; frozen=true; fi
      log="$(mktemp "$L/train-XXXXXXXX.log")"
      set +e
      python -u "$T" train --seed "$s" --device cuda --log-name "$(basename "$log")" 2>&1 | tee "$log"
      codes=("${PIPESTATUS[@]}")
      set -e
      rc="${codes[0]}"
      [ "${codes[1]}" -eq 0 ] || { echo "STOP: tee failed; log: $log"; exit 1; }
      echo "__EXIT__ $rc" >> "$log"
    else
      echo "an uncommitted $O exists: recording it (no new run)"
    fi
    if [ -f "$O" ]; then
      name="$(python - "$O" "$s" <<'PY'
import json, re, sys
from pathlib import Path
r = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
name = r.get('log_name', '')
if r.get('format') != 'h7r-preserve-run-v1' or r.get('seed') != int(sys.argv[2]) \
        or not re.fullmatch(r'train-[A-Za-z0-9]+\.log', name):
    sys.exit('STOP: invalid run record or missing attempt log name')
print(name)
PY
)"
      log="$L/$name"
      [ -f "$log" ] || { echo "STOP: the record's log is missing: $log"; exit 1; }
      case "$(tail -n 1 "$log")" in
        "__EXIT__ 0") ;;
        __EXIT__*) echo "STOP: the record and its log disagree on the exit code"; exit 1 ;;
        *) printf '\n%s\n__EXIT__ 0\n' "Recovered exit code from $O" >> "$log" ;;
      esac
      rc=0
      mkdir -p "$(dirname "$dst")"
      [ -e "$dst" ] || cp "$O" "$dst"  # an existing copy is never overwritten, only compared
      cmp "$O" "$dst" || { echo "STOP: $dst differs from $O"; exit 1; }
      python "$T" verify-run --seed "$s" || { echo "STOP: the files of S2-$s are not the ones its record names; nothing committed"; exit 1; }
    elif [ "$rc" -eq 0 ]; then
      echo "STOP: train S2 $s did not produce its record"; exit 1
    fi
    paths=()
    for f in "$L"/train-*.log; do
      [[ "$(tail -n 1 "$f")" =~ ^__EXIT__\ [0-9]+$ ]] || continue
      n="$(basename "$f" .log)"
      tracked "$D/stage-logs/$n.log" && continue
      python scripts/h7r_d4_records.py copy-logs --resume --src-dir "$L" --pattern "$n.log" \
        --dest-dir "$D/stage-logs" --manifest "$D/stage-logs/$n.manifest.json"
      paths+=("$D/stage-logs/$n.log" "$D/stage-logs/$n.manifest.json")
    done
    if [ -f "$dst" ]; then paths+=("$dst"); fi
    [ "${#paths[@]}" -gt 0 ] || { echo "STOP: nothing to record"; exit 1; }
    while IFS= read -r -d '' staged; do
      allowed=false
      for f in "${paths[@]}"; do if [ "$f" = "$staged" ]; then allowed=true; break; fi; done
      "$allowed" || { echo "STOP: unrelated staged path: $staged"; exit 1; }
    done < <(git diff --cached --name-only -z)
    expect=()
    for f in "${paths[@]}"; do expect+=(--expect "$f"); done
    git add -- "${paths[@]}"
    python scripts/h7r_d4_records.py verify-staged "${expect[@]}"
    git commit -m "H7-R PRESERVE: train S2 $s exit $rc (records only)"
    git push origin feat/h7-c1
    if [ "$rc" -ne 0 ]; then echo "__RECORDED__ $rc"; echo "STOP: train S2 $s failed (exit $rc)"; exit "$rc"; fi
  done
  for s in 20261024 20261111 20261112; do
    python "$T" verify-run --seed "$s"
  done
  echo "__RECORDED__ 0"
)
```

pv-AC. Arena 계약

```bash
(
  set -euo pipefail
  cd ~/renju-h6
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src:$PWD/scripts"
  [ "$(git branch --show-current)" = feat/h7-c1 ] || { echo "STOP: wrong branch"; exit 1; }
  git diff --quiet || { echo "STOP: unstaged tracked changes"; exit 1; }
  if git diff --cached --quiet; then git pull --ff-only origin feat/h7-c1; fi
  D=docs/h7r/r2/R2-S400-PRESERVE-1
  R=runs/h7r/R2-S400-PRESERVE-1
  L="$R/logs"
  A=docs/h7r/contracts-v2/R2-S400-PRESERVE-ARENA-1.json
  T=scripts/h7r_preserve.py
  tracked(){ git cat-file -e "HEAD:$1" 2>/dev/null; }
  if tracked "$A"; then echo "STOP: the Arena contract is already recorded"; exit 1; fi
  tracked "$D/seed-records/S2-20261024/record.json" || { echo "STOP: train S2 20261024 is not recorded"; exit 1; }
  mkdir -p "$L"
  command -v flock > /dev/null || { echo "STOP: flock is missing (util-linux)"; exit 1; }
  exec 9> "$L/arena-contract.lock"
  flock -n 9 || { echo "STOP: another arena-contract block is running"; exit 1; }
  logged(){  # log path, then the command; sets rc; the log ends with __EXIT__
    local log="$1"; shift
    set +e
    "$@" 2>&1 | tee "$log"
    local codes=("${PIPESTATUS[@]}")
    set -e
    rc="${codes[0]}"
    [ "${codes[1]}" -eq 0 ] || { echo "STOP: tee failed; log: $log"; exit 1; }
    echo "__EXIT__ $rc" >> "$log"
  }
  rc=0
  if [ ! -e "$A" ]; then
    git diff --cached --quiet || { echo "STOP: resolve staged changes before a new run"; exit 1; }
    logged "$(mktemp "$L/arena-contract-XXXXXXXX.log")" python -u "$T" arena-contract
  else
    echo "an uncommitted $A exists: checking it (no new run)"
  fi
  checked=false
  if [ "$rc" -eq 0 ]; then
    logged "$(mktemp "$L/arena-check-XXXXXXXX.log")" python -u "$T" check-contract
    [ "$rc" -eq 0 ] || { echo "STOP: the Arena contract does not check; nothing committed"; exit 1; }
    checked=true
  fi
  paths=()
  shopt -s nullglob
  for f in "$L"/arena-contract-*.log "$L"/arena-check-*.log; do
    n="$(basename "$f" .log)"
    tracked "$D/stage-logs/$n.log" && continue
    python scripts/h7r_d4_records.py copy-logs --resume --src-dir "$L" --pattern "$n.log" \
      --dest-dir "$D/stage-logs" --manifest "$D/stage-logs/$n.manifest.json"
    paths+=("$D/stage-logs/$n.log" "$D/stage-logs/$n.manifest.json")
  done
  if "$checked"; then paths+=("$A"); fi
  [ "${#paths[@]}" -gt 0 ] || { echo "STOP: nothing to record"; exit 1; }
  while IFS= read -r -d '' staged; do
    allowed=false
    for f in "${paths[@]}"; do if [ "$f" = "$staged" ]; then allowed=true; break; fi; done
    "$allowed" || { echo "STOP: unrelated staged path: $staged"; exit 1; }
  done < <(git diff --cached --name-only -z)
  expect=()
  for f in "${paths[@]}"; do expect+=(--expect "$f"); done
  git add -- "${paths[@]}"
  python scripts/h7r_d4_records.py verify-staged "${expect[@]}"
  git commit -m "H7-R PRESERVE: Arena contract exit $rc (records only)"
  git push origin feat/h7-c1
  echo "__RECORDED__ $rc"
  exit "$rc"
)
```

pv-A. Arena 대국과 판독 (tmux, 약 2.5시간)

```bash
(
  set -euo pipefail
  cd ~/renju-h6
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src:$PWD/scripts"
  [ "$(git branch --show-current)" = feat/h7-c1 ] || { echo "STOP: wrong branch"; exit 1; }
  D=docs/h7r/r2/R2-S400-PRESERVE-1
  R=runs/h7r/R2-S400-PRESERVE-1
  L="$R/logs"
  AR="$R/arena"
  A=docs/h7r/contracts-v2/R2-S400-PRESERVE-ARENA-1.json
  T=scripts/h7r_preserve.py
  tracked(){ git cat-file -e "HEAD:$1" 2>/dev/null; }
  if tracked "$D/judge.json"; then echo "STOP: the judge is already recorded"; exit 1; fi
  mkdir -p "$L" "$AR"
  command -v flock > /dev/null || { echo "STOP: flock is missing (util-linux)"; exit 1; }
  exec 9> "$L/arena.lock"
  flock -n 9 || { echo "STOP: another Arena block is running"; exit 1; }
  git diff --quiet || { echo "STOP: unstaged tracked changes"; exit 1; }
  if [ -f "$AR/HEAD" ]; then
    if [ "$(git rev-parse HEAD)" != "$(cat "$AR/HEAD")" ]; then
      echo "STOP: HEAD moved since the Arena started ($(cat "$AR/HEAD")); restore that commit or report"; exit 1
    fi
    echo "resuming at $(cat "$AR/HEAD") (no pull until the judge is recorded)"
  else
    git diff --cached --quiet || { echo "STOP: resolve staged changes before the Arena"; exit 1; }
    [ ! -e "$D/judge.json" ] || { echo "STOP: $D/judge.json exists without an Arena HEAD: report it"; exit 1; }
    git pull --ff-only origin feat/h7-c1
    tracked "$A" || { echo "STOP: the Arena contract is not recorded (run pv-AC first)"; exit 1; }
    git rev-parse HEAD > "$AR/HEAD"
  fi
  tracked "$A" || { echo "STOP: the Arena contract is not recorded (run pv-AC first)"; exit 1; }
  STAMP="$(date +%Y%m%d-%H%M%S)-$$"
  run(){  # log base, then the command; sets rc; the log ends with __EXIT__
    local b="$1"; shift
    if [ -e "$L/$b.log" ]; then echo "STOP: $L/$b.log exists"; exit 1; fi
    set +e
    "$@" 2>&1 | tee "$L/$b.log"
    local codes=("${PIPESTATUS[@]}")
    set -e
    rc="${codes[0]}"
    [ "${codes[1]}" -eq 0 ] || { echo "STOP: tee failed; log: $L/$b.log"; exit 1; }
    echo "__EXIT__ $rc" >> "$L/$b.log"
  }
  if [ ! -e "$D/judge.json" ]; then
    git diff --cached --quiet || { echo "STOP: resolve staged changes before playing"; exit 1; }
    python scripts/check_track_b_frozen.py
    run "check-models-$STAMP" python -u "$T" check-models
    [ "$rc" -eq 0 ] || { echo "STOP: the model files are not the Arena contract's; nothing played or committed"; exit 1; }
    arms="$(python -c "import json, sys; print(' '.join(a['name'] for a in json.load(open(sys.argv[1]))['arms']))" "$A")"
    for arm in $arms; do
      if [ -e "$AR/$arm.json" ]; then echo "done: $arm"; continue; fi
      run "play-$arm-$STAMP" python -u "$T" play --arm "$arm" --device cuda --workers 6 \
        --jsonl "$AR/$arm.jsonl" --output "$AR/$arm.json"
      if [ "$rc" -ne 0 ]; then
        echo "STOP: arm $arm failed (exit $rc); nothing committed, HEAD kept: rerun this block to resume"; exit 1
      fi
    done
    run "judge-$STAMP" python -u "$T" judge --arms-dir "$AR"
    [ "$rc" -eq 0 ] || { echo "STOP: the judge refused (exit $rc); nothing committed, HEAD kept"; exit 1; }
  else
    echo "an uncommitted $D/judge.json exists: verifying it against the arm records (no new judge)"
    run "verify-judge-$STAMP" python -u "$T" verify-judge --arms-dir "$AR"
    [ "$rc" -eq 0 ] || { echo "STOP: $D/judge.json is not what the arm records give: report it; nothing committed"; exit 1; }
  fi
  paths=()
  shopt -s nullglob
  for f in "$L"/check-models-*.log "$L"/play-*.log "$L"/judge-*.log "$L"/verify-judge-*.log; do
    n="$(basename "$f" .log)"
    tracked "$D/stage-logs/$n.log" && continue
    python scripts/h7r_d4_records.py copy-logs --resume --src-dir "$L" --pattern "$n.log" \
      --dest-dir "$D/stage-logs" --manifest "$D/stage-logs/$n.manifest.json"
    paths+=("$D/stage-logs/$n.log" "$D/stage-logs/$n.manifest.json")
  done
  paths+=("$D/judge.json")
  while IFS= read -r -d '' staged; do
    allowed=false
    for f in "${paths[@]}"; do if [ "$f" = "$staged" ]; then allowed=true; break; fi; done
    "$allowed" || { echo "STOP: unrelated staged path: $staged"; exit 1; }
  done < <(git diff --cached --name-only -z)
  expect=()
  for f in "${paths[@]}"; do expect+=(--expect "$f"); done
  git add -- "${paths[@]}"
  python scripts/h7r_d4_records.py verify-staged "${expect[@]}"
  git commit -m "H7-R PRESERVE: Arena and judge (records only)"
  git push origin feat/h7-c1
  echo "__RECORDED__ 0"
)
```
