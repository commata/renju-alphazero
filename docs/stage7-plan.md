# Stage 7 Validation Plan

Stage 7은 Stage 6에서 동결한 **PUCT v1 + PolicyValueNet + self-play training loop**가
그램 환경에서 안정적으로 학습 신호를 만들고 있는지 확인하고, Stage 8 장기 학습 전에
검색 품질·병목·장치별 처리량을 측정해 다음 설정을 결정하는 단계다.

Stage 7의 목적은 강한 최종 모델을 만드는 것이 아니다. Stage 6의 재현 가능한 기준선을 보존한 채
**학습 신호 확인 + 병목 분해 + 의미보존 최적화 + 탐색 설정 결정**을 수행한다.

## 1. 기준선과 변경 원칙

- 기준선: Stage 6 완료 태그 `v0.4-training`.
- PUCT v1은 Stage 7-A에서 알고리즘적으로 변경하지 않는다.
- MCTS-v6는 frozen benchmark로만 사용하고 AlphaZero search에 전술 정책을 주입하지 않는다.
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

## 3. Tactical / Value Probe

기존 Stage 3~6 테스트 fixture의 전술 국면을 정식 probe로 재구성한다.

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
