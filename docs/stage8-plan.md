# Stage 8 Plan — 데스크톱 이관과 규모 확장

Stage 8은 Stage 7에서 검증한 D16 학습 설정을 **학습 의미를 바꾸지 않고** 데스크톱으로 옮기고,
self-play 처리량을 늘려 장기 학습을 운영할 수 있는 구조를 만드는 단계다. 가장 강한 모델을 만드는 것이
목표가 아니다. 최종 기력 평가는 Stage 9에서 한다.

핵심 원칙: **처리량 개선(execution-only)과 학습 알고리즘 변경(training-critical)을 섞지 않는다.**
worker 수, 평가 주기, 경고는 critical hash에 들어가지 않고, network·simulations·games/generation·
규칙·FPU는 Stage 8 초기에 바꾸지 않는다.

## 1. 기준선

| 항목 | 값 |
|---|---|
| 코드 | main `52fbf91` (PR #17 병합), 태그 `v0.6-stage7` |
| config | `configs/stage7d_b16.yaml`, critical hash `25ab9c5a…` |
| 시작 checkpoint | `runs/stage7d_b16/checkpoints/checkpoint_gen160.pt`, SHA-256 `58aea679…` (generation=160) |
| network | 64 channels × 4 blocks |
| search | PUCT v2(`tactical_rules`), 50 simulations, c_puct 1.5, FPU 0 |
| self-play | 16 games/generation, temperature 10수, Dirichlet α 0.05 · ε 0.25 |
| training | batch 32, 50 steps/generation, replay 10,000, Adam lr 1e-3, 8-way augmentation |
| 루프 내 평가 | 매 generation random·tactical·previous 각 4판(v2 25 sims), 마지막 generation에 MCTS-v6 2판 |
| 장치 | CPU, `torch_threads: 1` |
| Stage 7 외부 평가(gen 160) | Tactical 49/50, MCTS-v2 43/50(흑 20·백 23), v321 7/10, v7 0/10 |

**worker 수를 늘려도 `games_per_generation`은 16으로 둔다.** D32(32판)는 Stage 7-D에서 한 색 붕괴가
관측됐다(stage7-plan §8.7). worker는 같은 16판을 더 빨리 만드는 실행 최적화일 뿐 데이터 양을 바꾸지 않는다.

```text
workers=1 → 16판 직렬 / workers=2 → 8+8 / workers=4 → 4×4 (완료 순서와 무관하게 game index 순으로 합침)
```

## 2. Stage 8-A — 데스크톱 이관 검증

성능 개선 없이 동작만 확인한다.

**run 디렉터리를 통째로 복사한다.** `run_training(resume=...)`는 checkpoint의 상위 디렉터리
(`resume.parent.parent`)를 run 디렉터리로 쓰고 `metadata.json`이 없으면 거부하며, 그 디렉터리의
`metrics.jsonl`에 이어 쓴다(`output.run_name`은 resume 시 무시). 따라서 Stage 7 증거 run을 직접 resume하지 않고
**복사본 `runs/stage8_cpu/`를 만들어** 이어간다.

```text
runs/stage7d_b16/  (원본 보존: Stage 7 증거)
runs/stage8_cpu/   (복사본: Stage 8 학습)
  ├── checkpoints/  ├── self_play/  ├── evaluation/
  ├── metrics.jsonl ├── metadata.json └── config.yaml
```

검증 순서:

1. `git describe --tags` = `v0.6-stage7` 이후, torch `2.14.0+cpu` 동일 설치
2. `checkpoint_gen160.pt` SHA-256 = `58aea679…`, config critical hash = `25ab9c5a…`
3. `scripts/ci_run_tests.py --skip-policy none`, `check_frozen_baseline.py`, V6/V7 동작 지문
4. `run_stage6_training.py --verify-checkpoint …/checkpoint_gen160.pt`(합법 self-play 1판) **두 번** → 같은 game hash
5. worker=1 처리량 기준선(§4)

**재현성의 범위:** 같은 기계 안에서는 같은 checkpoint·seed의 self-play가 bit 단위로 같아야 한다(4번).
그램과 데스크톱 사이에서는 CPU 명령어 집합(AVX2/AVX-512 등)에 따라 float 연산 경로가 달라질 수 있어
**game hash 일치를 요구하지 않는다.** 기계 간 계약은 checkpoint 파일 hash, critical hash, 규칙·frozen 테스트다.

## 3. Stage 8-B — 흑/백 편향 감시

### 3.1 기존 신호의 한계 (Stage 7-D 데이터로 확인)

10-generation 흑 승률 90% 이상 / 10% 이하 기준은 붕괴한 D32만이 아니라 **성공한 D16에도 걸린다.**

| run | 10-gen 흑 승률이 극단(≥90% 또는 ≤10%)이었던 구간 | 결과 |
|---|---|---|
| D16 | gen 50~99 (92~97%), gen 120~159 (2.5~5.6%) | 양쪽을 번갈아 학습, 외부 평가 향상 |
| D32 | gen 0~79 전 구간 (90~99%) | 백 방어 붕괴(MCTS-v2 백 17 → 3 → 1/25) |

지속 기간으로도 둘을 가르기 어렵다(D16도 5개 창 연속 극단 후 회복). 또 **D16 gen 160은 백 우세 구간 안에
있으므로 Stage 8 시작 직후 경고가 뜬다.** 따라서 두 단계로 나눈다.

### 3.2 `color_imbalance` — 정보성 경고 (루프 안)

- 최근 10 generation(= 160판)의 흑 승률, 백 승률, 무승부율, 평균 길이, 연속 극단 창 수를 매 generation 계산한다.
- 흑 승률 ≥ 0.90 또는 ≤ 0.10이면 `{"type": "health_warning", "kind": "color_imbalance", ...}`를
  `metrics.jsonl`에 남긴다. 같은 방향은 10 generation마다 한 번만 기록한다.
- 이 경고만으로 학습을 멈추거나 rollback하지 않는다.

### 3.3 `color_regression` — 조치 신호 (외부 평가)

붕괴를 실제로 가르는 신호는 **색별 외부 평가 하락**이었다(D32 백 17 → 3, D16은 흑·백 모두 유지).
light 평가(§6)마다 MCTS-v2 50판을 흑/백 25판씩 나눠, 한 색의 승수가 직전 light 평가 대비 Fisher 단측
p < 0.05로 떨어지면 `color_regression`을 기록한다. 이 신호가 **두 번 연속**(640판)이면 사람이 판단한다.
판단 선택지는 마지막 정상 checkpoint 검토, seed를 바꾼 분기 run, 또는 계속이다.

자동 rollback은 Stage 8 범위에 넣지 않는다.

## 4. Stage 8-C — CPU worker 처리량 측정

Stage 7 측정(stage7-plan §5)에서 1 thread batch 이득이 없었으므로, **1-thread 독립 worker 여러 개**를 먼저 비교한다.

`scripts/benchmark_stage8_workers.py`: 같은 checkpoint, 같은 16개 seed로 worker 수만 바꾼다.
후보는 1 / 2 / 4, 그리고 물리 코어 수 − 1까지(부모 프로세스·OS용 1코어 남김).

| 기록 | 의미 |
|---|---|
| games/s, games/h, samples/s | 처리량 |
| **generation wall time 분해** | self-play / 학습 50 step / 루프 내 평가 / checkpoint 저장 |
| 평균 대국 길이, ms/move, evaluator calls | 동작 동일성 확인 |
| game hash / record hash (worker 수별) | 재현성(§5)과 같아야 함 |

**Amdahl 주의:** generation 시간에는 self-play 외에 루프 내 평가(매 generation 12판 + previous 모델,
`evaluation.every: 1`)와 학습이 직렬로 들어간다. `evaluation` 설정은 training-critical이라 주기를 바꾸면
resume이 거부되므로, 평가 비중이 크면 **평가 게임도 같은 worker pool로 병렬화**한다. 결과는 같아야 하고,
execution-only로 취급한다. 결정은 wall time 분해를 본 뒤에 한다.

선택 기준은 최대 worker가 아니라 **지속 가능한 최고 처리량**이다. 예: worker 2가 worker 4의 95% 이상이면
발열·메모리 여유를 보고 2를 고른다.

## 5. 병렬 self-play 재현성 계약

현재 코드(`training.loop.generate_self_play`)는 generation마다 `state.self_play_rng.getrandbits(63)`로
게임 seed를 순서대로 뽑고, 각 게임은 자기 `Random(seed)`만 쓴다(`play_self_play_game`). `game_hash`와
`record_hash`는 `runtime_env`와 시간 정보를 제외한다. 따라서 다음을 지키면 worker 수와 무관하게 데이터가 같다.

1. 부모가 16개 seed를 **먼저 같은 순서로** 뽑는다. `self_play_rng`의 소비량과 최종 상태가 직렬과 같다.
2. worker는 `(game_index, seed)`만 받아 1-thread CPU 추론으로 대국한다. 모델은 generation마다 부모의 state_dict를 받는다.
3. 부모가 결과를 **game index 순으로 정렬**한 뒤 기존과 같이 `replay_record` 검증 →
   `samples_from_record(game_id=index)` → `ReplayBuffer`에 넣는다.
4. worker 오류는 generation 실패로 처리한다. 기존 계약대로 마지막 checkpoint에서 generation을 처음부터 다시 돈다.

필수 테스트(`tests/test_stage8_workers.py`, 작은 모델): 같은 checkpoint·seed에서 worker 1 / 2 / 4의
winner, moves, game hash, record hash, sample 순서, `self_play_rng` 최종 상태가 **모두 같다.** 하나라도 다르면
병렬화를 execution-only로 취급하지 않고 버그로 본다. 불법수는 항상 0이다.

구현 주의(Windows):

- multiprocessing 기본이 `spawn`이므로 진입 스크립트에 `if __name__ == '__main__':`이 필요하다.
- pool은 run 동안 유지하고 generation마다 가중치만 보낸다(64×4 모델 ≈ 1~2 MB).
- 각 worker에서 `torch.set_num_threads(1)`을 호출한다.

## 6. Stage 8-D — 외부 평가 주기화

외부 평가는 학습 루프 안에 넣지 않는다. 기존 `scripts/run_stage7_checkpoint_eval.py`와 probe 스크립트를
**orchestration 스크립트**(`scripts/run_stage8_training.py`)가 부른다.

```text
resume → run_training(stop_after=다음 평가 지점) → checkpoint → 외부 평가·probe → 다시 resume → …
```

`run_training(..., stop_after=N)`(CLI `--stop-after-generation N`)으로 멈췄다가 이어가도 RNG·replay·optimizer가
checkpoint에 있으므로 **중단 없이 돈 것과 같은 결과**다(Stage 6 resume 계약). 평가 중에는 학습이 멈춰 있어
처리량 측정이 오염되지 않는다.

| 종류 | 주기 | 내용 |
|---|---|---|
| light | 320판 = 20 generation | 규칙 끈 PUCT v1 25 sims: Tactical 50판, MCTS-v2 50판(25 오프닝 쌍, 흑/백 분리) + 전술/방어 probe |
| heavy | 1,280판 = 80 generation | light + MCTS-v3.2.1 10판, MCTS-v7 10판 |

- 지점(시작 gen 160 기준): light는 gen 180, 200, 220, …이고, heavy는 gen 240, 320, 400, 480이다.
  gen 160 값은 Stage 7 결과를 기준선으로 재사용한다.
- `keep_every: 20`이 이 checkpoint들을 모두 보존한다(16 × 20 = 320판).
- 결과 파일이 있으면 건너뛴다(재실행 idempotent). 평가 결과와 `color_regression`(§3.3)은 run 디렉터리의
  `external_eval/`에 남긴다.
- 평가 주기는 학습 상태가 아니므로 **training config가 아니라 orchestration 인자로 둔다.**
- `training.generations`는 **gate 목표**(192 / 288 / 480)로 두고 구간은 `stop_after`로 나눈다.
  루프 내 MCTS-v6 평가(`final_generation_only`)는 `generations − 1`에서만 돌기 때문이다. 구간마다
  `generations`를 바꾸면 매 구간 끝에 v6 평가가 끼어든다.

## 7. Config 확장

`training.config`는 **모르는 키를 오류로 거부**하므로 새 키는 `DEFAULTS`에 추가해야 한다. Stage 7-C의
`milestones` 선례를 그대로 따른다.

```yaml
parallel:                 # execution-only
  self_play_workers: 1
health:                   # execution-only
  color_imbalance:
    enabled: true
    window: 10
    lower: 0.10
    upper: 0.90
```

- `NON_CRITICAL`에 `('parallel',)`, `('health',)`를 추가한다. critical hash는 그대로다.
- Stage 7 checkpoint에 저장된 config에는 이 키가 없다. `resolve_resume_config`는 checkpoint config를 먼저
  `validate_config`에 통과시키므로, 검증은 `config.get('parallel')`처럼 **키가 없어도 동작**해야 한다
  (`milestones`와 같은 방식).
- regression test(`tests/test_stage8_config.py`)로 고정할 것:
  - `stage7d_b16.yaml`의 critical hash가 `25ab9c5a…` 그대로다(현재 코드에서 확인함).
  - Stage 8 config로 Stage 7 checkpoint를 resume할 수 있다.
  - worker 수만 다른 두 config의 hash가 같다.

## 8. Stage 8-E — 장기 학습 pilot

| gate | 누적 self-play | generation | 확인 |
|---|---|---|---|
| 1 | 512판 | 160 → 192 | crash·NaN·불법수 0, resume 정상, 경고·외부 평가 자동 실행 |
| 2 | 2,048판 | 160 → 288 | 외부 평가 추세(MCTS-v2·Tactical 색별, v321·v7), probe, value 분리, 길이, 흑백 흐름 |
| 3 | 5,120판 (선택) | 160 → 480 | gate 2에서 퇴행이 없을 때만 |

**시간 추정:** Stage 7-D에서 그램의 D16 self-play는 generation당 26~95초였다(D32와 동시에 돌린 상태).
2,048판(128 generation)은 직렬로도 수 시간 규모다. 그래서 **gate 1은 worker=1로 먼저 돌려도 된다.**
이 결과가 데스크톱 직렬 기준선(§2의 5번) 역할도 한다. 병렬화는 gate 3 이상과 Stage 9를 위한 투자다.

gate 판단은 루프 내 previous 승률(창당 20판, 잡음 수준)이 아니라 외부 평가와 probe로 한다.

## 9. Stage 8-F — GPU 경로 (선택)

RX 6600(RDNA2, gfx1032)은 CPU 병렬화의 선행 조건이 아니다. 다음은 **대상 OS에서 실측해 확정**해야 할 사항이다.

- Windows: RDNA2 소비자 GPU용 공식 ROCm PyTorch 지원은 확인되지 않았다. DirectML 플러그인은 지원하는 torch 버전이
  고정 버전(2.14)과 다를 수 있다.
- Linux: ROCm에서 RDNA2 소비자 GPU는 공식 목록 밖이라 호환 설정(예: `HSA_OVERRIDE_GFX_VERSION`)이 필요할 수 있다.
- 64×4 network, 15×15 입력은 매우 작아서 B=1 GPU 호출은 CPU보다 느릴 가능성이 높다. 이득은
  **여러 게임의 요청을 모은 batch**(shared inference queue)에서만 기대할 수 있다.

순서: backend 동작 확인 → checkpoint 추론 수치 비교 → B=1…32 벤치마크(`benchmark_stage7_batch.py`의
device 확장) → CPU worker 방식과 positions/s 비교 → 이득이 클 때만 inference queue를 설계한다.
GPU batch는 float 연산 순서가 달라지므로 CPU run과 **별도 run**으로 취급한다. 이득이 작거나 불안정하면
Stage 8은 CPU로 완료한다.

## 10. Stage 8에서 바꾸지 않는 것

network 크기, games/generation(16), self-play simulations(50), PUCT 규칙, FPU, tree reuse,
within-tree batching, virtual loss, progressive widening, Top-K pruning, transposition table.

- simulations 25는 처리량 후보로만 남긴다(stage7-plan §9.1). 비교하려면 같은 weights에서 **새 run 두 개**를 만든다.
  `self_play.simulations`는 training-critical이라 기존 run을 resume한 채 바꿀 수 없다.
- tree reuse는 stage7-plan §12의 계약(simulation 예산의 의미, `pi`에 이어받은 방문 포함 여부, noise, advance 규칙,
  fast path)을 문서와 테스트로 먼저 고정한 뒤 별도 실험으로 한다.

## 11. 코드 변경 예상과 PR 분리

| PR | 내용 | 주요 파일 |
|---|---|---|
| 8-A | 계획, 이관 절차, 데스크톱 기준선 | `docs/stage8-plan.md` |
| 8-B | 흑백 편향 경고 + config 확장 | `src/training/health.py`, `config.py`, `loop.py`, `tests/test_stage8_health.py`, `tests/test_stage8_config.py` |
| 8-C | worker 벤치마크 → 병렬 self-play(+필요 시 평가) | `src/training/parallel_self_play.py`, `scripts/benchmark_stage8_workers.py`, `tests/test_stage8_workers.py` |
| 8-D | 외부 평가 orchestration + `color_regression` | `scripts/run_stage8_training.py`, `tests/test_stage8_schedule.py` |
| 8-E | pilot 결과와 최종 설정 | `configs/stage8_cpu.yaml`, 문서 |
| (별도) | GPU, tree reuse | — |

8-B가 끝나면 gate 1을 worker=1로 먼저 시작하고, 그동안 8-C를 개발할 수 있다.
규칙 테스트, frozen baseline, Stage 5 golden hash는 모든 PR에서 그대로 통과해야 한다.

## 12. 완료 기준

1. D16 gen 160 checkpoint가 데스크톱에서 복원되고 file SHA-256과 critical hash가 이관 전후 같다.
2. 같은 기계에서 `--verify-checkpoint` 반복 결과가 같고, 규칙·frozen·학습 regression이 통과한다.
3. worker 수별 처리량과 generation wall time 분해를 같은 checkpoint·seed로 측정했다.
4. 사용할 worker 수를 하나 정했다.
5. worker 1과 N의 game/record hash, sample 순서, RNG 상태가 같고 resume 계약이 유지된다.
6. `color_imbalance` 경고와 `color_regression` 신호가 동작한다.
7. 320판 / 1,280판 외부 평가와 probe가 자동으로 돈다.
8. gate 1(512판)이 crash·NaN·불법수 없이 끝나고, 중단 후 resume이 정상이다.
9. gate 2(2,048판)까지 진행할 지속 가능한 CPU 설정이 정해진다.
10. RX 6600 경로를 실측했거나, 쓰지 않는 이유와 CPU 대비 결과를 기록했다.
11. 최종 config와 결과를 문서화하고 `v0.7-stage8` 태그를 만든다.

산출물: 안정적인 장기 self-play 파이프라인, 검증된 병렬화 설정, 색 붕괴 감시, 자동 외부 평가,
그리고 계속 학습할 수 있는 checkpoint.
