# Stage 7 Validation Plan

Stage 7은 Stage 6에서 동결한 **PUCT v1 + PolicyValueNet + self-play training loop**가
그램 환경에서 안정적으로 학습 신호를 만들고 있는지 확인하고, Stage 8 장기 학습 전에
검색 품질·병목·장치별 처리량을 측정해 다음 설정을 결정하는 단계다.

Stage 7의 목적은 강한 최종 모델을 만드는 것이 아니다. Stage 6의 재현 가능한 기준선을 보존한 채
**학습 신호 확인 + 병목 분해 + 의미보존 최적화 + 탐색 설정 결정**을 수행한다.

## 1. 기준선과 변경 원칙

- 코드 기준선: MCTS-v7 동결을 포함한 main `6f00c13` (Stage 6 학습 코드는 태그 `v0.4-training`과 동일).
- 학습 기준선: Stage 6 증거 run `stage6_mvp_A` / `stage6_mvp_B` (generation 3, global_step 150,
  replay 1,035). 두 run은 연속 실행과 중단·재개 재현성의 증거물이므로 **원본은 수정하지 않는다.**
- PUCT v1은 Stage 7-A에서 알고리즘적으로 변경하지 않는다.
- **Classical benchmark는 MCTS-v7 FINAL**(Stage 6.5 동결, [mcts-v7.md](mcts-v7.md))이다.
  classical 엔진은 AlphaZero search에 전술 정책을 주입하지 않고 평가 상대로만 쓴다.
- **Stage 7-B 결정(7-A 진단 후):** AlphaZero search에 휴리스틱·threat planner·VCF는 넣지 않되,
  렌주 규칙에서 바로 나오는 **1수 사실**(즉시 승리, 유일한 필수 방어, 규칙으로 증명된 승/패)만 쓰는
  PUCT v2 teacher(`search.tactics`, `tactical_rules`)를 옵션으로 허용한다. 기본값은 끈 상태이며 Stage 5/6
  동작과 hash는 그대로다(§8.3).
- 학습 루프 내부 평가(`evaluation.mcts_v6`)는 Stage 6과의 연속성 때문에 그대로 둔다.
  `evaluation`은 critical config이므로 V7로 바꾸면 resume이 거부된다. **V7 평가는 학습 루프 밖의
  checkpoint 평가**(§3.4)로만 수행한다.
- 의미보존 최적화와 탐색 알고리즘 변경은 같은 PR에 섞지 않는다.
- 한 generation의 2~4판 W/L/D를 기력 향상으로 해석하지 않는다.
- 장시간/대규모 self-play는 Stage 8에서 시작한다.

## 2. Stage 7-A — continuation과 관찰

Stage 6 증거 run 원본은 보존하고 복사본에서 gen 3 → 30 continuation을 수행한다.
training-critical 설정은 Stage 6 MVP와 동일하게 유지한다.

```text
games_per_generation = 4
self-play simulations = 25
batch_size = 32
steps_per_generation = 50
replay_capacity = 10000
```

추가 27 generations × 4 games = 108 games이며, Stage 6의 12 games를 포함해 총 약 120
self-play games다. 이 규모는 기력 결론이 아니라 장기 실행 안정성과 초기 추세를 보는 데 사용한다.

설정은 `configs/stage7a_continuation.yaml`이다. `configs/stage6_mvp.yaml`과 비교해 execution-control
키(`training.generations: 30`, `training.keep_checkpoints: 32`, `output.run_name`)만 다르고
critical config hash는 동일하다(`tests/test_stage7_foundation.py`). `keep_checkpoints: 32`는 gen 3~30
checkpoint를 모두 남겨 probe와 외부 평가에 쓰기 위한 값이다.

`--resume`은 checkpoint 경로의 부모 run 디렉터리에 이어 쓰므로, Stage 6 run을 **복사한 뒤 복사본의
`latest.pt`로 resume**한다. 원본이 변경되지 않음은 같은 테스트에서 확인한다.

```powershell
Copy-Item -Recurse runs\stage6_mvp_B runs\stage7a
python scripts/run_stage6_training.py --config configs/stage7a_continuation.yaml `
  --resume runs\stage7a\checkpoints\latest.pt
```

학습 중에는 profile, 외부 평가, 다른 benchmark를 **동시에 실행하지 않는다.** 같은 장비에서 동시에
돌리면 self-play/학습 시간 지표와 profile이 서로 오염된다. probe와 외부 평가는 continuation 종료 후
저장된 checkpoint에 대해 실행한다.

### 2.1 Stage 7-A 핵심 지표

- crash, NaN/Inf, illegal move/policy mass
- policy/value/total loss moving trend
- replay buffer 크기, 신규 sample 수, sample reuse ratio
- 평균 대국 길이, 흑/백 결과 분리
- Random/Tactical/previous 5-generation 이동 집계
- checkpoint별 tactical/value probe
- self-play/학습 시간과 세부 search profile
- checkpoint/resume 정상성

Random/Tactical의 단일 generation 결과나 Tactical 1승을 milestone으로 사용하지 않는다.

### 2.2 Stage 7-A 결과 (gen 3 → 30, 코드 `adce150`)

실행: 사용자 Windows PC, `configs/stage7a_continuation.yaml`, Stage 6 B run 복사본에서 resume.
시작 checkpoint SHA-256 `d9e62e0e…`(Stage 6 B `latest.pt`), 종료 gen 30 checkpoint `1fd22c6f…`,
global_step 1,500, replay 9,042.

| 항목 | 결과 | 판정 |
|---|---|---|
| 27 generations 실행, resume, checkpoint | crash·NaN·illegal 0, generation당 self-play 22~70 s / 학습 8~13 s | ✅ |
| network 변화 | checkpoint SHA 모두 다름, 같은 오프닝에서 3~6수부터 수순이 달라짐 | ✅ |
| Random (루프 내) | 80승 40패 (66.7%), 5-gen 묶음 55% → 75% (표본 작음) | 🟡 |
| previous (루프 내) | 58승 62패 (48.3%) | 🟡 |
| Tactical (루프 내) | 2승 118패 (1.7%) | ❌ |
| 외부 평가 gen 3/10/20/30 vs mcts_v2 / v321 / v7 | **0승 120패**, 흑·백 모두 0승, 12~22수 패배 | ❌ |
| policy loss / value loss (5-gen 평균) | 4.62 → 4.62 / 0.30 → 0.53 | ❌ 정체 |
| probe gen 30 top-1 (즉시승 / 필수방어 / VCF) | 2.5% / 0% / 0%, mass lift 1.26 / 1.23 / 1.15 | ❌ |
| probe value | separation −0.17, balanced accuracy 0.45, 30 gen 모두 separation ≤ 0 | ❌ |

### 2.3 원인 진단

1. **value target 부호는 정상이다.** 실제 self-play 기록을 텐서 수준에서 확인하면 z가 수마다 +1/−1로
   교대하고 입력의 `current_is_black` plane과 일치한다(`tests/test_stage7_diagnostics.py`).
   음수 separation은 value가 전술이 아니라 **색 사전확률**을 배운 결과다. value와 "흑 차례" 상관
   +0.24~+0.62, self-play 흑:백 승 68:52.
2. **구현은 전술을 학습할 수 있다.** `scripts/run_stage7_supervised_sanity.py`:
   - 암기: probe 160국면에 새 network를 학습하면 100 step에 모든 kind top-1 1.00, value 1.00.
   - 일반화: V7-vs-V5 기보의 전술 국면 1,098개(D4 증강)로 학습하고 V7-vs-V6 기보 probe로 평가하면
     필수방어 top-1 0.67~0.78, forced_loss value 0.75~0.85 (self-play gen 30: 0.00 / 0.45).
3. **병목은 teacher(탐색 target)다.** 거의 균등한 prior + FPU 0 PUCT는 합법수 약 200개 중 10~80개만
   보고 필수방어를 찾지 못한다(학습 안 된 network: 800 simulations에서도 0/20). self-play 방문 비율에
   전술 정보가 거의 없으므로 policy가 학습할 target이 없다(cold start).

결론: Stage 7-A는 **파이프라인 검증 PASS, 기력 학습 FAIL**이며 원인은 구현 버그가 아니라 teacher 품질이다.
같은 설정으로 generation만 늘리지 않는다. 다음 실험은 §2.4.

### 2.4 진단 도구

- `scripts/run_stage7_search_probes.py`: 학습 없이 checkpoint(또는 무작위 초기화)의 PUCT가 probe를
  푸는 비율을 simulations × FPU 격자로 측정한다. `SearchConfig.fpu_reduction`(기본 `None` = Stage 5
  FPU 0, `to_dict`에 나타나지 않아 기존 hash 불변)으로 parent-relative FPU를 켠다.
- `scripts/run_stage7_supervised_sanity.py`: 위 암기/일반화 검증 재현.

## 3. Tactical / Value Probe

구현: probe set `tests/fixtures/stage7_probes_v1.json`(`stage7-probes-v1`, 178국면),
생성기 `scripts/build_stage7_probes.py`, 측정 `src/training/probes.py`,
실행 `scripts/run_stage7_probes.py`(run의 모든 `checkpoint_genNNN.pt` → `<run>/probes/genNNN.json`).

| kind | 출처 | policy 정답 | value 부호 |
|---|---|---|---|
| `immediate_win` (40) | MCTS-v7 benchmark 200판의 모든 국면 | 모든 합법 승리수(규칙 엔진 exact) | + |
| `must_block` (40) | 〃 | 상대의 유일한 승리점 | 없음 |
| `forced_loss` (40) | 〃 (상대 승리점 2개 이상, 내 즉시 승리 없음) | 없음 | − |
| `vcf` (40) | seed 44 VCF 회귀 fixture | VCF를 시작하는 **모든** 첫 수(보수적 VCF solver, 20개 4·20,000 node) | + |
| `avoid` (18) | seed 44 `losing_move_allows_vcf` | 기록된 패배수에 준 mass (낮을수록 좋음) | 없음 |

각 kind는 흑/백 차례를 최대한 반씩 뽑고, 위치 SHA-256 순으로 결정적으로 선택한다. 테스트는 생성기와
독립적으로 규칙 엔진에서 라벨을 다시 계산해 대조한다.

주의 — 설계 결정:

- **흑 금수는 policy 정답으로 쓰지 않는다.** 엔진의 `legal_moves()`가 금수를 이미 제외하고 같은 mask가
  입력 plane과 policy mask에 쓰이므로 masked policy의 금수 mass는 항상 0이다. 대신
  `forbidden_diagnostic`에서 **mask 전 softmax**가 금수점에 주는 mass를 균등 분포 대비 배율로 기록한다.
- 초기 network는 거의 균등하므로 절대 mass 대신 **`mass_lift = correct_mass / uniform_mass`**
  (`uniform_mass = |정답| / |합법수|`)를 핵심 학습 신호로 본다. 1.0이 무작위 수준이다.
- value는 한쪽 부호로 치우친 head가 한 kind에서 100%를 받을 수 있으므로, 핵심 value 지표는 kind를 합친
  `value_overall.separation`(승리 라벨 평균 − 패배 라벨 평균)과 `balanced_sign_accuracy`다.
- D4 증강은 v1에서 사용하지 않는다(필요하면 v2에서 state와 정답 집합을 함께 변환).

아래 §3.1~3.3은 원래 요구사항이며 위 구현이 이를 충족한다.

### 3.1 Policy probe

대상 예시:

- 즉시 승리
- 필수 방어
- open four / 복수 정답 수
- 43 관련 공격·방어
- 금수 때문에 선택하면 안 되는 수

각 국면은 하나 이상의 정답 수 집합 `correct_moves`를 가진다.

측정:

- policy top-1이 `correct_moves`에 포함되는 비율
- `correct_moves` 전체에 배정한 raw policy probability mass
- 필요하면 top-k hit rate

열린4 양끝처럼 정답이 여러 개인 국면은 단일 좌표를 정답으로 강제하지 않는다.

### 3.2 Value probe

value sign 정답은 결과가 실제로 확실한 국면에만 부여한다.

- terminal-near forced win/loss
- 짧은 강제 수순으로 결과를 검증할 수 있는 fixture

단순 must-block 국면처럼 방어 후 최종 승패가 정해지지 않은 상태에는 임의의 value sign을 붙이지 않는다.

### 3.3 고정성

- probe set version을 기록한다.
- D4 변환 사용 시 state와 정답 수 집합을 함께 변환하고 중복을 제거한다.
- checkpoint 비교에서는 동일 probe set을 사용한다.

### 3.4 외부 checkpoint 평가 (MCTS-v7 포함)

`scripts/run_stage7_checkpoint_eval.py`는 학습 루프 밖에서 checkpoint를 고정 classical 상대와 대국시킨다.
모델은 루프 내부 평가와 같은 결정적 PUCT(noise OFF, temperature 0, checkpoint의 `puct_simulations`)를 쓴다.

- 오프닝과 상대 seed는 `(seed, opponent, pair)`에서만 파생하고 **generation과 무관**하다. 모든 checkpoint가
  같은 오프닝·같은 상대를 만나므로 generation 간 비교가 가능하다.
- 상대 사다리(약→강): `random`, `tactical`, `mcts_v2`(순수 MCTS), `mcts_v321`, `mcts_v5`, `mcts_v6`,
  `mcts_v7`. V7은 V6 상대 score 0.90이므로 초기 checkpoint는 V7에 전패할 가능성이 높다. 신호는 probe와
  중간 사다리(`mcts_v2`, `mcts_v321`)에서 먼저 본다.
- 권장: gen 3/10/20/30에서 `--opponents mcts_v2 mcts_v321 mcts_v7 --pairs 5`(상대별 10판).
  경쟁력이 보이기 시작하면 판수를 늘린다. 결과는 `<run>/external_eval/genNNN.json`.

## 4. PUCT profile 분해

현재 `SearchTiming.inference_s`는 순수 NN forward 시간이 아니다. Stage 7에서 최소 다음으로 나눠 측정한다.

```text
legal_moves
snapshot construction
encode / legal mask
NN forward
masked softmax / tensor postprocess
CPU transfer / tolist
evaluation validation
tree selection / backup / play-undo
```

Stage 6의 약 132 ms `inference_s`를 NN 시간으로 직접 해석하지 않는다.

profile 변경은 탐색 결과에 영향을 주지 않는 계측 PR로 분리한다.

구현: `src/training/profiling.py` + `scripts/profile_stage7_search.py`. 탐색 코드는 바꾸지 않는다.

- `ProfiledEvaluator`는 `PolicyValueEvaluator.evaluate_batch`와 같은 연산을 같은 순서로 수행하면서
  encode(legal mask + plane) / stack / forward / masked softmax / CPU 전송 / 결과 객체를 따로 잰다.
- `instrument_search()`는 측정 동안에만 `_legal_moves`(흑/백 차례 분리), PUCT v2 `tactical_filter`,
  `Game.play`/`Game.undo`를 감싼다. snapshot+검증 = `inference_s` − evaluator 시간, tree 기타 =
  `tree_s` − 규칙 필터 − play − undo로 계산한다.
- 계측 전후 방문 수가 같고 감싼 함수가 원래대로 복원됨을 `tests/test_stage7_profiling.py`가 확인한다.
- 입력: 고정 probe 국면(기본 80개), 측정 전 워밍업 5개.

참고(개발 컨테이너, 무작위 64×4 network, 1 thread, 50 simulations, PUCT v2): 착수당 약 170 ms 중
NN forward 48%, **흑 차례 `legal_moves` 27%**(호출당 약 1 ms, 백 차례는 약 0.02 ms — 흑 금수 판정),
tree 기타 10%, encode 4%, 규칙 필터 4%. 확정 수치는 사용자 PC의 학습된 checkpoint로 다시 잰다.

**사용자 PC 결과**(Windows 11, Intel Family 6 Model 170, torch 2.14 CPU, 1 thread, D32 gen 80, probe 80국면,
최적화 전 코드 `a436bfc`):

| | v1 25 sims | v1 50 sims | v2 25 sims | v2 50 sims |
|---|---|---|---|---|
| ms / 착수 | 443 | 780 | 85 | 287 |
| 평가 / 착수 | 24.4 | 47.0 | 6.6 | 12.2 |
| NN forward | 66% | 65% | 47% | 51% |
| **흑 `legal_moves`** | 17% | 17% | **33%** | **30%** |
| 규칙 필터 | — | — | 6% | 5% |
| encode / tree 기타 | 4% / 7% | 4% / 7% | 3% / 6% | 3% / 6% |

- 흑 `legal_moves`는 호출당 2.7~4.5 ms로 백(0.04~0.06 ms)의 70~100배다 → §6에서 최적화.
- NN forward는 평가당 6~12 ms로 같은 실행 안에서도 2배 흔들렸다. P/E 코어 혼합 CPU의 스레드 배치·전원
  상태 영향으로 보고, **절대 시간은 전원 연결·최고 성능 모드에서만 비교**한다(재측정한 §9.1 격자는 탐색 횟수에
  정확히 비례해 안정적이었다).
- v2의 평가 수 감소는 probe 국면(즉시 승리·필수 방어 비중이 큼)의 효과라 일반 국면 추정에 쓰지 않는다.

## 5. Batch microbenchmark

`PolicyValueEvaluator.evaluate_batch`는 이미 batch N을 지원하므로 실제 PUCT를 변경하기 전에
고정 checkpoint/고정 position set에서 다음을 측정한다.

```text
B = 1 / 2 / 4 / 8 / 16
```

기록:

- batch latency
- ms / position
- positions / second
- CPU utilization / 온도(가능한 범위)

CPU에서 batching이 큰 향상을 낸다고 가정하지 않는다. Stage 8의 병렬화 방식은 이 실측과 장치 지원 여부로 결정한다.

구현: `scripts/benchmark_stage7_batch.py`. 같은 고정 국면(기본 128개)을 B = 1/2/4/8/16/32로 나눠
`evaluate_batch` 전체(full)와 forward 단독을 torch thread 수별로 재고(median), batch 결과와 B=1 결과의
최대 prior/value 차이(§5.1)를 함께 기록한다. 탐색·학습 코드는 바꾸지 않는다.

참고(개발 컨테이너, 무작위 64×4): 1 thread에서는 B에 따른 이득이 거의 없고(약 3 ms/국면), 4 threads에서는
B=8이 약 0.8 ms/국면으로 가장 좋았다. batch와 B=1의 차이는 약 1e-6. 공유 CPU라 확정값은 사용자 PC에서 잰다.

**사용자 PC 결과**(같은 PC·checkpoint, 128국면, full `evaluate_batch` ms/국면):

| B | 1 thread | 2 threads | 4 threads |
|---|---|---|---|
| 1 | 8.4 | 8.8 | 6.1 |
| 8 | 8.1 | 5.1 | 3.8 |
| 16 | 8.6 | **4.5** | 3.2 |
| 32 | 8.0 | 4.6 | **2.8** |

- 1 thread에서는 batch 이득이 없다. 4 threads·B=32가 B=1 대비 2.2배(스레드 4배 대비 효율 낮음).
- batch와 B=1의 최대 차이: prior 5×10⁻⁷, value 2×10⁻⁶(§5.1 허용 범위).
- Stage 8 후보: 이 CPU에서는 **1-thread worker 여러 개의 병렬 self-play**가 batch보다 유리할 가능성이 높다.
  worker 수별 처리량을 Stage 8 첫 측정으로 정한다(§11).

### 5.1 수치 재현성 계약

batch 크기가 달라지면 float32 kernel 경로 때문에 bit-exact 출력은 요구하지 않는다.

- Uniform/ScriptedEvaluator: search 결과 exact equality 검증
- 실제 NN: policy/value를 tolerance로 비교
- B=1: Stage 5/6 deterministic baseline과 기존 hash 계약 유지

## 6. `legal_moves` 의미보존 최적화

Stage 6 MVP에서 searched move의 `legal_moves`가 약 56 ms로 관측되었으므로 Stage 7에서 우선 profile한다.

- 흑/백 차례를 분리한다.
- `forbidden_reason` 및 내부 33/44/장목 경로를 profile한다.
- PUCT/NN/training 변경과 같은 PR에 섞지 않는다.
- 규칙 regression, differential tests, frozen MCTS behavior, Stage 5 search 계약을 그대로 통과해야 한다.

최적화 전후에 같은 fixture의 합법수 목록과 순서를 비교한다.

**구현(Stage 7 완료 기준 5).** `Game.legal_moves()`의 흑 차례는 `renju.rules.legal_black_points()`를 쓴다.

- 흑돌마다 네 축의 ±4/±3 이웃 칸에 개수를 더해 **빈칸별 축 개수를 한 번에** 만든다(이웃 관계는 대칭).
  이전에는 빈칸마다 32칸을 다시 셌다.
- 조용한 점 판정(`_quiet_counts`): 어느 축에도 ±4 안에 다른 흑이 3개 이상 없고, ±3 안에 2개인 축이 하나 이하면
  금수가 아니다. 쌍삼 조건을 ±4 대신 `_open_three`의 전제와 같은 ±3으로 좁혀 더 많은 점을 바로 통과시킨다.
- 정밀 판정(`_classify_black_stone`)은 결과가 바뀔 수 없는 축만 건너뛴다. run_length는 ±4에 4개 이상인 축,
  `_fours`는 3개 이상인 축, `_open_three`는 ±3에 2개 이상인 후보 축에서, 삼이 두 개 나올 여지가 있을 때만 부른다.
  모두 기존 함수 안에 이미 있던 필요조건이다.
- 판정 규칙, 결과 문자열, 합법수 순서는 그대로다. `tests/reference_rules.py`와의 차분 테스트(기존 seed 42
  240국면 + 새 seed 7 400국면 + Stage 7 probe 전체, 보드 복원 포함), frozen hash, V6/V7 동작 지문, Stage 5
  golden hash가 모두 통과한다.

효과(개발 컨테이너, 무작위 64×4, v2 50 sims, probe 60국면, 탐색 결과 동일): 흑 `legal_moves` 호출당
1.72 → 0.66 ms(2.6배), 착수당 116 → 91 ms(−22%), 흑 `legal_moves` 비중 36% → 18%.

## 7. Weights-only export

Stage 7-B 전에 Stage 6/7 training checkpoint에서 모델 전용 checkpoint를 만드는 정식 export 도구를 추가한다.

필수 provenance:

- source run
- source generation
- critical config hash
- source git commit
- model config
- encoder/action/checkpoint version

export 후 기존 Stage 4 model loader로 reload하고 state_dict 동일성을 검증한다.

training-critical 설정을 바꾸는 Stage 7-B는 resume snapshot을 변경해 이어가지 않고,
exported weights를 `training.init_checkpoint`로 사용하는 새 run으로 시작한다.

## 8. Stage 7-B — Search budget 실험

### 8.1 Search-only 효과

동일 weights-only checkpoint로 학습 없이 PUCT 탐색 예산만 비교한다.

```text
25 simulations
50 simulations
필요하면 100 simulations
```

기록:

- fixed tactical/search probe
- W/D/L은 충분한 색 교환 평가에서 보조 지표로 사용
- ms/move
- evaluator calls
- probe/search quality per second

simulation 수가 큰 쪽이 계산량도 크므로 절대 승률과 효율을 함께 기록한다.

### 8.2 Training-data 품질 효과

동일 초기 weights에서 별도 학습 run을 만든다.

```text
Run A: self-play 25 simulations
Run B: self-play 50 simulations
```

나머지 설정은 가능한 한 동일하게 유지한다. checkpoint별 probe, fixed evaluation, loss,
opening/game diversity를 비교해 더 비싼 search가 더 좋은 teacher `pi`를 만드는지 본다.

Search-only 실험과 training-data 실험을 혼동하지 않는다.

### 8.3 Stage 7-B teacher 실험 — arm B(주 개발) vs arm A(대조군)

기준점: 코드 `adce150`(브랜치 `checkpoint/stage7a-baseline`, 태그 `stage7a-baseline`), 가중치 `runs/stage7a`
gen 30 checkpoint(SHA-256 `1fd22c6f…`). 두 arm은 같은 export 가중치로 **새 run**을 시작한다(training-critical
설정이 바뀌므로 resume이 아니라 `training.init_checkpoint`; 새 optimizer·빈 replay·generation 0부터).

| | arm A `configs/stage7b_a_scale.yaml` | arm B `configs/stage7b_b_rules.yaml` |
|---|---|---|
| 역할 | 대조군: 규모만 확대(선택지 a) | 주 개발: 규모 + 규칙 teacher(선택지 b) |
| self-play search | PUCT v1, 50 simulations | **PUCT v2**, 50 simulations |
| in-loop 평가 search | PUCT v1, 25 simulations | PUCT v2, 25 simulations |
| games / generation | 16 | 16 |
| 그 외 | Stage 7-A와 동일 (64ch×4, batch 32, 50 steps, replay 10,000, seed 42) | 동일 |

두 arm의 critical config 차이는 `self_play.tactical_rules`, `evaluation.tactical_rules` 두 키뿐이다
(`tests/test_stage7b_teacher.py`). 7-A 대비 두 arm 모두 games 4→16, simulations 25→50을 함께 바꿨으므로
**arm A 대 7-A는 규모 효과, arm B 대 arm A는 teacher 효과**로 읽는다.

PUCT v2 규칙(`search.tactics.tactical_filter`):

- 둘 차례가 즉시 5목을 만들 수 있으면 자식을 승리수로 제한하고, root가 아닌 노드는 +1로 증명 종료한다.
- 상대 승리점이 1개이고 합법이면 자식을 그 방어점 하나로 제한한다.
- 상대 승리점이 2개 이상이거나 유일한 방어점이 흑 금수면 root가 아닌 노드를 −1로 증명 종료한다.
- network 입력은 항상 전체 합법수 mask를 쓰고, prior만 허용 자식 안에서 다시 정규화한다. root는 증명 종료하지 않고
  정확히 N회 탐색하므로 기존 record/replay 계약(방문 합 = N)을 그대로 지킨다.
- 측정: 무작위 초기 network에서도 25 simulations로 즉시 승리·필수 방어 probe 100% 해결, VCF는 규칙 범위 밖(0.05).
  50 simulations self-play 착수당 약 220~260 ms로 v1과 같은 수준이다(증명 노드는 network 호출 생략).

비교 지표(같은 generation끼리):

1. raw network probe(`run_stage7_probes.py`) — teacher와 무관한 network 자체의 학습 신호. **주 지표.**
2. 외부 평가를 **같은 search로** 두 arm에 모두 실행(`--tactical-rules off`와 `on` 각각) — network 비교와
   system(network+search) 비교를 분리한다.
3. in-loop 평가·loss·self-play 길이/흑백 결과.

같은 PC에서 두 arm을 동시에 돌리면 시간 지표가 서로 오염되므로 시간 비교는 하지 않고, 기력·probe만 비교한다.

### 8.4 Stage 7-B 결과와 Stage 7-C 장기 continuation

7-B 결과(gen 30, 외부 평가 상대별 10판): network 자체(v1 search)에서 B가 A보다 Tactical 14/20 대 5/20
(gen 20+30, p=0.005). MCTS-v2 상대 차이(B 3/10 대 A 1/10)는 아직 유의하지 않고, v321/v7에는 두 arm 모두 전패.
A는 v2 search로 평가할 때만 좋아져 규모 확대만으로는 network 향상이 없었다. B self-play는 후반 흑 85%·평균 21수로
치우쳤는데, PUCT v2가 4는 막지만 **열린 3 방어(2수 앞)는 규칙 범위 밖**이기 때문이다(무작위 network, 800
simulations에서도 v2 방어 2/10).

**결정: 탐색 규칙은 v2(1수 사실)에서 고정한다.** 열린 3 방어·VCT·VCF를 규칙으로 계속 추가하면 AlphaZero search가
MCTS-v5~v7의 강제정책을 재구현하게 되어 network가 배울 몫과 V7 대비 측정의 독립성이 사라진다. 2수 이상의 전술은
self-play 학습으로 해결되는지 먼저 확인한다(공격 policy가 배워지면 상대 탐색이 그 응수를 먼저 방문하므로
방어도 따라 배워질 수 있다).

Stage 7-C: 두 arm을 같은 설정 그대로 gen 30 → 110까지 이어간다(각 +80 generation, +1,280 self-play games).

- `configs/stage7c_b_rules_long.yaml`, `configs/stage7c_a_scale_long.yaml`: 7-B 설정과 critical hash가 같고
  `generations`, `keep_checkpoints: 5`, `keep_every: 10`(새 실행 제어 키), `run_name`만 다르다. 7-B run의
  **복사본**을 resume한다.
- 비교는 같은 generation(= 같은 self-play 판수)의 checkpoint끼리 한다(gen 40…110, 10 간격 보존).
- 측정 추가(규칙 아님): `tests/fixtures/stage7_probes_defense_v1.json`(`must_defend_open3` 40국면, 흑/백 20씩;
  상대의 열린 4·4-4 생성수를 없애는 수만 정답, 반격용 4가 없는 국면만). `run_stage7_probes.py --probes ... --suffix _defense`.
- `scripts/summarize_stage7_runs.py`: metrics를 10-gen 묶음 표(흑 승률, 길이, 재사용, loss, in-loop 평가)로 요약.

- 급상승 기록(`milestones`, 실행 제어 전용·critical hash 무관): 매 generation 평가 후 최근 5 generation과 그 전
  5 generation의 in-loop 점수(random/tactical/previous)를 비교해 +0.25 이상 오르면, 그리고 tactical/mcts_v6 첫 승을
  `metrics.jsonl`에 `milestone` 이벤트로 남기고 해당 checkpoint를 `checkpoints/milestone_genNNN.pt`로 보존한다
  (가지치기 대상 아님). 같은 상대의 반복 보고는 5 generation 동안 억제한다. `summarize_stage7_runs.py`는 기록된
  이벤트와 함께 사후 급상승(창·기준 변경 가능)과 self-play 흑 승률 급변을 generation 단위로 출력하고, probe·외부 평가
  스크립트는 가지치기된 generation이면 milestone 파일을 사용한다.
- **probe는 규칙이 아니다.** `training.probes`와 probe fixture/생성기는 `scripts/`·`tests/`에서만 쓰이며 학습 루프,
  self-play, 탐색, 학습 target, loss 어디에서도 import되지 않는다. 저장된 checkpoint를 학습이 끝난 뒤 읽기 전용으로
  채점할 뿐이다. 탐색이 쓰는 규칙은 PUCT v2의 `search.tactics.tactical_filter`(1수 사실)뿐이다.

판단 기준: B의 self-play 흑 승률이 내려오고(평균 길이 증가), `must_defend_open3` probe와 v1 search 외부 평가
(Tactical·MCTS-v2 각 50판)가 오르면 v2 경계를 확정한다. gen 110에서도 변화가 없으면 V7 기보 지도학습 초기화(선택지 c)
또는 Stage 8 규모 확대를 **명시적으로** 선택한다.

### 8.5 Stage 7-C 결과 (gen 30 → 110, arm당 +1,280 self-play games)

외부 평가는 규칙을 끈 PUCT v1(25 simulations)로 network만 비교했다(Tactical·MCTS-v2 50판, v321·v7 10판).
같은 명령을 두 번 실행해 결과가 한 판도 다르지 않음을 확인했다.

| 상대 | B gen 30 → 70 → 110 | A gen 30 → 70 → 110 |
|---|---|---|
| Tactical | 30 → 44 → **47**/50 | 8 → 16 → 3/50 |
| MCTS-v2 | 7 → 10 → **21**/50 (42%) | 0 → 0 → 0/50 |
| MCTS-v2, B 흑 / 백 | 7/25·0/25 → 9/25·1/25 → 11/25·**10/25** | 0 |
| MCTS-v3.2.1 / v7 | 0 → 0 → 1/10 / 0/10 | 0 / 0 |

- **B는 규칙 추가 없이 2수 방어를 학습했다.** MCTS-v2 상대로 백 승리가 0/25(gen 30)에서 10/25(gen 110)로
  늘었다(p = 0.0003). 백이 이기려면 흑의 선공을 막아야 한다. §8.4의 "v2 경계 유지" 판단을 확정한다.
- B gen 110 대 A gen 110, MCTS-v2 21/50 대 0/50(p ≈ 3×10⁻⁸). **A(규모만 확대)는 종료한다.** A의 value는 self-play
  승률을 따라 색을 예측했다(백 차례 상관 최대 −0.48).
- raw network probe(B, gen 30 → 110): 필수 방어 정답 확률 배율 1.60 → 2.85, 열린 3 방어(`must_defend_open3`)
  top-3 10% → 25~38%·배율 2.4 → 3.8~4.4, value 차이 +0.06 → **+0.30**(균형 정답률 0.60, 색 상관 ≈ 0).
  즉시 승리·필수 방어 top-1은 두 arm 모두 0~7%로, **raw policy는 여전히 약하다.**
- 급상승 기록 검증: B gen 48(previous 기준) → 외부 평가 gen 49는 gen 30과 같음(오탐). B gen 106 → gen 107은
  MCTS-v2 19/50으로 gen 70(10/50) 대비 실제 향상(p = 0.04). 루프 내 previous 기준(창당 20판)은 후보 신호로만 쓴다.
- 운영 지표(B 후반): 평균 17수, 샘플 재사용 5.8배, value loss 0.57 → 0.73, self-play 흑 72%. 짧은 대국의
  과다 재사용이 다음 병목이다(§8.6).

### 8.6 Stage 7-D — 샘플 재사용 실험 (D16 대 D32)

7-C B gen 110 가중치를 export해 두 arm을 **새 run으로 동시에** 시작한다. critical 설정 차이는
`training.games_per_generation` 하나뿐이다(`tests/test_stage7d_reuse.py`). 둘 다 새 optimizer·빈 replay로
시작하므로 이어가기 대 새 run의 차이가 섞이지 않는다.

| | D16 `configs/stage7d_b16.yaml` | D32 `configs/stage7d_b32.yaml` |
|---|---|---|
| games / generation | 16 | 32 |
| generations | 160 | 80 |
| 총 self-play | 2,560 | 2,560 |
| 총 학습 step | 8,000 | 4,000 |
| 예상 재사용 | 약 5~6배 | 약 2.5~3배 |
| 체크포인트 보존 | 20 gen마다 | 10 gen마다 (= 같은 판수 지점) |

그 외는 B와 동일하다(PUCT v2, 50 simulations, 64ch×4, batch 32, 50 steps/gen, replay 10,000, seed 42).
비교는 **같은 누적 판수**(320판 간격) checkpoint끼리 외부 평가(v1, MCTS-v2 50판 흑/백 분리)·probe·value loss로 한다.
D32가 같은 판수에서 같거나 더 강하면 Stage 8 기본값을 D32 쪽으로 정한다.

### 8.7 Stage 7-D 결과 (두 arm 모두 2,560 self-play games)

외부 평가는 §8.5와 같다(규칙 끈 PUCT v1 25 sims, 고정 오프닝; Tactical·MCTS-v2 50판, v321·v7 10판).

| checkpoint | Tactical | MCTS-v2 | v2 흑 / 백 | v321 | v7 |
|---|---|---|---|---|---|
| 7-C B gen 110(출발점) | 47 | 21 | 11 / 10 | 1/10 | 0/10 |
| D16 gen 40 / 80 | 49 / **50** | 39 / 39 | 18·21 / 22·17 | — | — |
| **D16 gen 160** | 49 | **43** | 20 / **23** | **7/10** | 0/10 |
| D32 gen 20 / 40 | 49 / 46 | 33 / 18 | 16·17 / 15·3 | — | — |
| D32 gen 80 | 39 | 16 | 15 / **1** | 2/10 | 0/10 |

- **D16이 출발점보다 강해졌다**: MCTS-v2 43 대 21(p = 4×10⁻⁶), 백 23/25 대 10/25(p = 0.0001),
  v321 7/10 대 1/10(p = 0.01). gen 80 → 160은 MCTS-v2 기준 포화(39 → 43, p = 0.22).
- **D32는 출발점보다 약해졌다**: 백 1/25(p = 0.002), Tactical 백 22 → 14/25. 같은 판수 비교에서 D16이 우세
  (1,280판: 39 대 18, p = 2×10⁻⁵; 2,560판: 43 대 16, p = 3×10⁻⁸).
- 원인은 self-play 흑백 흐름이다. D16은 흑 승률 21% → 97%(gen 50~59) → 3%(gen 120~)로 **두 번 뒤집히며**
  공격과 방어를 번갈아 배웠다. D32는 gen 0~9부터 흑 90%, 이후 98~99%로 **한 색에 갇혔고** value loss가
  0.39 → 0.09로 떨어졌다("흑이 이긴다"만 맞히면 되는 목표). 백 방어 예시가 사라져 백 방어가 퇴화했다.
- **주의:** arm당 seed 하나다. 두 arm은 첫 10 generation에 이미 갈렸으므로 "32판이 붕괴를 부른다"가 아니라
  "**v2 경계에서 한 색 붕괴가 실제로 일어나고, 스스로 빠져나오지 못할 수 있다**"까지만 결론으로 삼는다.
  D32 재사용 3.6배, D16 8~10배로 §8.5가 걱정한 과다 재사용은 이번엔 문제가 아니었다.
- raw probe(D16 gen 160, 7-C gen 110 대비): 즉시 승리 배율 1.6 → 5.0(gen 80은 9.6), 열린 3 방어 top-3
  25% → 40%. 필수 방어 top-1은 0~5% 그대로이고 value 분리는 +0.30 → +0.04로 줄었다(흑백 흐름이 극단을 오간 영향).
  D32는 방어 top-3 20%로 7-C보다 낮다.
- 급상승 기록(D16 gen 19/81/144/157, D32 gen 14/30)은 모두 previous 6/20 → 12/20 수준(p ≈ 0.055)으로 잡음이다.
  의미 있는 사건은 `black_rate_shifts`에 잡힌 흑백 우세 전환이었다.

**결정:** Stage 8 기본값은 D16 설정(세대당 16판)이고 최종 Stage 7 checkpoint는 D16 gen 160
(SHA-256 `58aea679…`)이다. self-play 흑 승률이 한쪽으로 오래 머무는지 감시하는 경고를 Stage 8 필수 항목으로 둔다(§10.2).

## 9. FPU 단일 변수 실험

현재 PUCT v1은 미방문 child Q를 0으로 고정한다. 적은 simulation 환경에서는 이 선택이 탐색 폭에
영향을 줄 수 있으므로 Stage 7-B에서 simulation 예산과 분리해 실험한다.

후보:

```text
A. current: FPU = 0
B. parent-value FPU = parent current-player value - reduction
```

현재 Node Q가 `player_who_moved` 관점이므로 non-root의 기준값은 부호를 명시적으로 변환해야 한다.

```text
non-root base = -parent.q
root base     = root NN value
FPU           = base - reduction
```

root는 현재 value_sum을 backup하지 않으므로 root NN value를 별도 보관하는 설계가 필요하다.

먼저 적절한 simulation 예산을 고정한 뒤 FPU만 변경해 비교한다.

### 9.1 Search-only 격자 결과 (완료 기준 7·8)

D16 gen 160, PUCT v2, 학습 없음. 즉시 승리·필수 방어는 모든 설정에서 100%(v2 규칙이 해결)다.

| 설정 | VCF 40 | 열린 3 방어 40 (흑 / 백) | ms / probe |
|---|---|---|---|
| FPU 0, 25 sims | 20% | 22.5% (4 / 5) | 114~136 |
| **FPU 0, 50 sims** | 20% | 27.5% | 226~273 |
| FPU 0, 100 sims | 22.5% | 32.5% (6 / 7) | 437~480 |
| FPU 0.25, 25 / 50 / 100 | 15 / 27.5 / 27.5% | 25 / 30 / 25% | 96~483 |

같은 probe끼리 짝지은 McNemar 검정:

- 25 → 100 sims: 방어는 100에서만 푼 것 4개, 25에서만 푼 것 0개(p = 0.12, 25 → 50 → 100에서 단조 증가).
  VCF는 차이 1개. **탐색을 4배 늘린 이득은 유의하지 않다.**
- FPU 0 대 0.25: 모든 sims에서 p = 0.38~1.0. 0.25는 방문 child 수만 줄인다(방어 100 sims 52 → 31).
- 시간은 탐색 횟수에 비례했다(1 simulation당 약 4.4~5.4 ms).
- D32 gen 80으로 먼저 잰 격자도 결론이 같았다(VCF 12~17%, 방어 10~15%). 모델이 강할수록 해법 비율이 높다.

**결정:**

- **FPU = 0 유지**(기준 8). 0.25는 품질 이득 없이 탐색 폭만 줄인다.
- **self-play는 검증된 50 simulations 유지**(기준 7). 7-B~7-D 학습은 모두 50으로 했다. search-only로는
  25와 50의 차이가 작지만, teacher 분포(`pi`) 품질은 학습 A/B(§8.2) 없이는 판단할 수 없으므로 25로 줄이는 것은
  Stage 8 처리량 후보로만 남긴다.
- 평가·사람 대국에서는 100 simulations를 선택지로 둔다(방어에 약한 상승 경향, 학습 비용과 무관).

## 10. Stage 7 완료 기준

다음을 만족하면 Stage 8로 넘어간다.

1. Stage 7-A continuation이 crash/NaN/illegal 없이 완료된다.
2. tactical/value probe와 moving metrics로 학습 신호 또는 병목을 설명할 수 있다.
3. `inference_s` 내부 비용과 `legal_moves` 비용이 분해되어 있다.
4. batch microbenchmark로 CPU에서 batching 효과를 실제 수치로 알고 있다.
5. 의미보존 `legal_moves` 최적화 여부가 regression으로 검증되어 있다.
6. weights-only export가 round-trip 검증된다.
7. 25/50(필요 시 100) simulation의 품질/시간 trade-off가 기록된다.
8. FPU 실험의 효과가 simulation 수와 분리되어 기록된다.
9. 그램에서 사용할 안정적인 소형 training/search 설정을 하나 선택할 수 있다.

학습 향상이 명확하지 않더라도 원인이 search/data/throughput 중 어디에 가까운지 설명할 수 있으면
Stage 7의 검증 목적은 달성한 것으로 본다.

### 10.1 현황 (Stage 7 종료 시점)

| # | 기준 | 상태 | 근거 |
|---|---|---|---|
| 1 | 7-A continuation 무결성 | ✅ | §2.2 |
| 2 | 학습 신호/병목 설명 | ✅ | teacher 품질(§2.3) → v2(§8.3) → 2수 방어 학습(§8.5) → 한 색 붕괴 위험(§8.7) |
| 3 | `inference_s`·`legal_moves` 비용 분해 | ✅ | §4: NN forward 47~66%, 흑 `legal_moves` 17~33% |
| 4 | batch microbenchmark | ✅ | §5: 1 thread 이득 없음, 4 threads B=32 2.2배, 수치 계약 통과 |
| 5 | `legal_moves` 의미보존 최적화 | ✅ | §6: 흑 호출 2.6배, 차분 테스트·frozen·golden hash 통과 |
| 6 | weights-only export round-trip | ✅ | `scripts/export_stage7_weights.py` |
| 7 | 25/50/100 simulation trade-off | ✅ | §9.1: 100까지 유의한 이득 없음, self-play 50 유지 |
| 8 | FPU 효과(simulation과 분리) | ✅ | §9.1: search-only로 효과 없음, FPU 0 유지 |
| 9 | 그램 소형 설정 선택 | ✅ | §10.2 |

### 10.2 최종 소형 설정 (Stage 8 출발점)

| 항목 | 값 |
|---|---|
| config | `configs/stage7d_b16.yaml` (critical hash `25ab9c5a…`) |
| 초기 가중치 | D16 gen 160 (`runs/stage7d_b16/checkpoints/checkpoint_gen160.pt`, SHA-256 `58aea679…`) |
| network | 64 channels × 4 blocks (Stage 4 기본) |
| search | PUCT v2(`tactical_rules: true`), 50 simulations, c_puct 1.5, FPU 0 |
| self-play | 16 games / generation, temperature 10수, Dirichlet α 0.05 · ε 0.25 |
| training | batch 32, 50 steps / generation, replay 10,000, Adam lr 1e-3, 8-way augmentation |
| 장치 | CPU, torch_threads 1 |
| 평가 | 루프 내 in-loop 평가 + 루프 밖 외부 평가(v1 25 sims, Tactical·MCTS-v2 50판, v321·v7 10판) |

Stage 8 착수 조건으로 함께 넣을 것:

1. **self-play 흑 승률 치우침 경고**: 최근 10 generation 흑 승률이 90% 이상 또는 10% 이하이면 경고를 남긴다
   (D32 붕괴, §8.7). 붕괴가 확인되면 이전 checkpoint로 돌아가거나 seed를 바꾸는 운영 절차를 정한다.
2. 1-thread worker 수별 self-play 처리량 측정으로 병렬화 방식 결정(§5, §11).
3. 외부 평가 주기화: 320판마다 MCTS-v2 50판, 1,280판마다 v321·v7 10판.

## 11. Stage 8로 미루는 구조적 변경

Stage 7 측정 결과 없이 다음을 먼저 구현하지 않는다.

- CPU multiprocessing self-play
- GPU multi-game inference queue
- tree reuse
- within-tree batching / virtual loss
- progressive widening / policy Top-K pruning
- transposition table

Stage 8에서 장치별로 분기한다.

- CPU: 1/2/4 worker를 실측하고 worker당 model copy + `torch_threads=1`을 후보로 둔다.
- GPU: multi-game state machine + shared inference queue + batch evaluation을 우선 검토한다.

병렬 self-play는 game completion 순서가 달라도 game index 순서로 결과를 정렬한 뒤 ReplayBuffer에 넣어
현재 resume/data-order 재현성 계약을 유지한다.

## 12. Tree reuse 선행 계약

Stage 8에서 tree reuse를 구현하기 전에 최소 다음을 테스트/문서로 먼저 고정한다.

1. simulation budget이 추가 방문 수인지 총 방문 수인지
2. training target `pi`에 inherited visits를 포함하는지
3. inherited visits와 새 root Dirichlet noise의 관계
4. own move / opponent move / 미방문 상대 수 / single-legal fast path에서 tree advance 규칙

이 계약이 정해지기 전에는 tree reuse를 의미보존 최적화로 취급하지 않는다.

## 13. 장기 학습 진입

장기 학습은 Stage 7이 아니라 Stage 8에서 시작한다. 장치별 throughput 설정을 검증한 후
작은 pilot에서 단계적으로 늘린다.

```text
500 self-play games
→ 평가
→ 총 2,000 games
→ 평가
→ 개선 추세와 자원 예산을 보고 5,000+ 검토
```

이 판수는 고정 요구사항이 아니라 운영상 pilot 기준이다. self-play loss만으로 중단/계속 여부를 결정하지 않고
fixed baselines, 이전 checkpoint, tactical/value probe와 함께 판단한다.
