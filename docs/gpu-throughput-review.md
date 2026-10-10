# GPU/CUDA 처리량 검토: self-play와 E2 (2026-10-10)

"CPU 멀티프로세싱으로 돌리는 self-play·E2를 GPU/CUDA로 최대한 빠르게"라는 제안(동적 inference batching,
batch 2048, pinned memory, torch.compile, AMP, CUDA stream)을 **저장소의 실측 기록과 코드**에 대조한 결과다.
엔진·설정은 바꾸지 않았다. 결론부터 적는다.

1. **작업마다 병목이 다르다.** Track A AlphaZero self-play는 NN이 착수 시간의 약 79%라 GPU batching이 맞는 방향이다.
   **E1/E2/S3/H5 같은 V8 대국과 Track B(Hybrid) self-play는 NN이 대국 시간의 1% 미만이다.** GPU로 옮길 것이 없다.
2. **E2를 빠르게 하는 길은 CPU뿐이다.** 게임 단위 프로세스 병렬(이미 `--workers`)과
   전술 패턴 스캔 hot path의 컴파일(비트 단위로 같은 결과)이 실제 레버다.
3. **Track A의 cross-game batch 상한은 세대당 게임 수(16)다.** "mean batch ≥ 24", "parallel 64/128"은
   세대당 게임 수나 tree 내 virtual loss를 바꿔야 하고, 둘 다 training-critical이다(stage8-plan §7.1).
4. Track A evaluator의 batch 64 "forward 1.1 ms vs 전체 7.85 ms" 차이는 **GPU 위에서 position마다 encode하는 코드**
   때문이다(§3.2). 제안의 "Python per-position loop가 범인" 추정은 맞고, 위치까지 특정된다.

---

## 1. E2 (V8 대국): GPU가 줄일 수 있는 몫

### 1.1 시간 분해 (커밋된 결과 파일, 새 실행 없음)

E2의 시험 arm(`puct_policy`, `puct_policy_vct2`)과 같은 엔진을 쓴 S3 결과에서 V8 착수 진단의 `seconds`를 모두 더했다.

| 파일 | 대국 | V8 착수 합 | root VCT 안전(V8-C) | 자기 공격(V8-B) | stage VCT(V8-A) | **tree (NN 포함)** | VCT2 | 상대 착수 합 | 대국 합 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `s3_base_8411` | 50 | 25,156 s | 12,999 (52%) | 7,320 (29%) | 2,553 (10%) | **1,938 (7.7%)** | — | 28,382 s | 53,538 s |
| `s3_vct2_8411` | 50 | 26,218 s | 12,751 (49%) | 7,197 (27%) | 2,533 (10%) | **1,916 (7.3%)** | 1,480 (5.6%) | 27,916 s | 54,135 s |
| `h5_policy_8401` | 50 | 23,002 s | 11,336 (49%) | 7,374 (32%) | 1,866 (8%) | **1,991 (8.7%)** | — | 35,188 s | 58,191 s |

- `s3_vct2_8411`의 policy 호출은 50판에 **79,252회**(PUCT 50 sims, 노드 확장마다 1회, batch 1)다.
- E2 상대(`v8:full+vct2atk`)는 여기에 깊이 2 공격(200k)을 더한다. smoke 한 판에서 공격만 960초였다(§12.29). 그래서 E2에서는 상대 몫이 더 커진다.

### 1.2 tree 안의 NN 몫 (이 컨테이너 CPU, 1 thread, 64×4 random init)

`s3_vct2_8411` 첫 대국 ply 14에서 `search_tree_puct`(50 sims, policy prior)를 한 번 재었다.

| 항목 | 시간 |
|---|---:|
| tree 전체 | 1.73 s |
| policy 51회 (encode + forward + softmax) | 0.22 s (13%), 호출당 3.8 ms |
| rollout (`_rollout_v321`) | 0.58 s (33%) |
| 나머지 (후보 생성 `_search_candidates_v321`, play/undo) | 0.93 s (54%) |

### 1.3 Amdahl 상한

```text
NN 시간 ≈ 79,252회 × 약 2–4 ms ≈ 160–320 s
대국 시간 54,135 s 중 NN ≈ 0.3–0.6 %
→ NN을 GPU로 옮겨 0 s가 되어도 E2는 약 1.003–1.006배
```

batch를 모을 수도 없다. 한 tree는 확장마다 policy 1회를 동기적으로 기다리고, 프로세스마다 게임이 하나라 batch는 항상 1이다.
cross-process GPU 서버를 만들어도 batch 상한은 worker 수이고, 줄일 NN 시간 자체가 1% 미만이다.

### 1.4 solver를 GPU로 옮기는 것은 권하지 않는다

V8의 시간 대부분은 예산이 있는 깊이 우선 위협 탐색(VCF/VCT)과 그 안의 패턴 스캔이다.

- 노드마다 다음 노드를 정하는 분기가 있어 GPU에 넘길 같은 연산 묶음이 없다. 노드 하나를 GPU로 판정하면 kernel launch·동기화 지연이 지금의 Python 노드 비용보다 크다.
- 결과는 **노드 예산과 탐색 순서에 묶여 있다**(UNKNOWN = 예산 소진). frontier를 병렬로 펼치는 GPU식 탐색은 같은 예산에서 다른 결과를 내므로 execution-only가 아니다. `mcts_v8.py`·`S3_VCT2_V1` 동결(§12.28)과도 맞지 않는다.
- 패턴 판정(4·열린 3·5목)을 convolution으로 여러 보드에 한꺼번에 적용하는 것은 가능하지만, 탐색이 한 번에 한 보드만 묻기 때문에 batch가 생기지 않는다.

## 2. E2를 실제로 빠르게 하는 방법 (CPU)

### 2.1 hot path: 이 컨테이너에서 V8 착수 3개 cProfile (`s3_base_8411`의 tree 경로 국면, `V8_DEFAULTS`, profiler 포함 48 s)

| 함수 | 호출 | 자체 시간 | 누적 |
|---|---:|---:|---:|
| `search.mcts_v5._threat_windows` | 66,740 | 11.9 s | 15.4 s (32%) |
| `search.mcts_v321._fast_open_three_in_direction` | 99,324 | 1.2 s | 13.1 s (27%) |
| `renju.rules.run_length` | 4.9M | 4.6 s | 6.7 s (14%) |
| `search.mcts_v321._fast_winning_extensions_in_direction` | 364k | 2.2 s | 6.0 s (12%) |
| `search.mcts_v321._run_info` | 2.5M | 2.9 s | 4.0 s |
| `renju.rules.inside` | **21.7M** | 4.0 s | 4.0 s |
| `list.count` | 18.0M | 3.5 s | 3.5 s |

(누적은 겹치므로 더하지 않는다.) cpu-optimization-12600k 문서의 V7 profile과 같은 모양이다. **모두 15×15 보드 위 한 방향 줄을 스칼라로 훑는 순수 Python 함수다.**

### 2.2 권장 순서

| 순서 | 작업 | 기대 효과 | 동결과의 관계 |
|---|---|---|---|
| 1 | **게임 단위 프로세스 병렬 유지**: `run_mcts_v8_benchmark.py --workers`. 12600K(6P+4E)에서 네 실행(8413 base → vct2 → 8414 base → vct2) 모두 **같은 worker 수**로 돌린다. E-core·SMT 경합이 arm 사이에 같아야 시간 비율을 비교할 수 있다 | 코어 수에 가까움 | 영향 없음 |
| 2 | **패턴 스캔 컴파일**: `_threat_windows`, `run_length`, `_run_info`, `_fast_open_three_in_direction`, `_fast_winning_extensions_in_direction`를 Numba(또는 C/Rust 확장)로 옮긴다. 보드는 `int8[225]` 하나로 넘기고 `inside`는 경계 패딩(17×17)으로 없앤다 | 위 함수 묶음이 V8 시간의 대략 50–70%라 이 부분이 10배 빨라지면 전체 약 2–2.5배 | **결과가 비트 단위로 같아야 한다**: 같은 착수, 같은 노드 수. 기존 로그(S3/E1) 재생과 frozen baseline SHA로 검증 |
| 3 | 방향별 line code + LUT(3^9 또는 2-bit 9칸) | 2와 겹침. 2 다음 단계 | 같음 |
| 4 | worker당 `OMP/MKL_NUM_THREADS=1`, policy 1 thread(`RootPolicy(threads=1)` 이미 그렇다) | 과구독 방지 | 영향 없음 |

- 검증 기준은 rule-optimization 때와 같다: 무작위 국면 differential(새 함수 = 기존 함수), V8 착수 진단의 `nodes`·`calls`·`played` 일치.
- E2 실행 중에 구현을 바꾸지 않는다. 바꾼다면 E2 전에 넣고 네 실행 모두 같은 commit으로 돌린다(시간 비율 비교 조건).
- solver 결과 캐시를 착수·대국 사이에 유지하는 방법은 **쓰지 않는다**. 공유 노드 예산(`_BudgetedSolver`)의 소비량이 달라져 같은 예산에서 결과가 바뀐다. 필요하면 별도 arm이다.

## 3. Track A AlphaZero self-play: 제안 검토

### 3.1 맞는 부분

- 단일 프로세스 B=1에서 NN이 착수 시간의 약 79%다(stage8-plan §2.4). 여러 게임의 leaf 요청을 모아 GPU에서 한 번에 평가하는 것이 주 경로라는 판단은 stage8-plan §7(8-E)과 같다.
- 기준을 positions/s가 아니라 세대 wall time으로 잡는 것(stage8-plan §7.4)도 같다.

### 3.2 evaluator overhead의 원인 (코드로 특정)

`PolicyValueEvaluator(device='cuda')`(`model/evaluator.py`)는 position마다

- `legal_moves_to_mask(..., device=cuda)`: GPU에 zeros, GPU index 대입
- `encode_game`: `torch.tensor(board, device=cuda)`(작은 H2D 1회), `torch.zeros(..., device=cuda)`, plane 대입 6회

을 실행한다. position 하나에 작은 kernel 약 10개와 H2D 1회다. 그 뒤 `priors.cpu().tolist()`(64×225 Python float), `validate_evaluation`이 225칸을 Python으로 검사한다.
batch 64에서 forward 밖 약 6.7 ms(position당 약 100 µs)의 대부분이 여기다.

고칠 방향(execution-only, 값은 같은 float32 경로):

1. CPU에서 numpy로 `[B,6,15,15]` float32와 `[B,225]` bool mask를 **미리 잡아 둔 pinned buffer**에 채운다.
2. H2D 1회(`non_blocking=True`) → forward → masked softmax → D2H 1회(`.cpu().numpy()`).
3. 결과 검증은 batch 단위 numpy로 한다(계약은 그대로: 합법수 밖 0, 합 1, value 범위).

`torch.inference_mode()`와 `model.eval()`은 **이미 쓰고 있다**(`model/evaluator.py`, `hybrid/h4_policy.py`). 제안 10번은 완료 상태다.

### 3.3 고쳐야 할 부분

| 제안 | 저장소 기준 | 판단 |
|---|---|---|
| parallel 64/128, mean batch ≥ 24 | `games_per_generation: 16`(stage8 config). tree마다 미완료 NN 요청 1개라 **cross-game batch 상한 = 동시 게임 수 ≤ 16**. 세대 끝에는 남은 게임만큼 더 줄어든다 | 목표를 "16에 가까운 평균"으로 고친다. 16을 넘기려면 세대당 게임 수 증가나 virtual loss가 필요하고 둘 다 별도 arm |
| batch 2048/4096 | 위 상한 때문에 self-play 추론에는 해당 없음. 학습 batch(H3는 1,024)에는 의미 있음 | self-play에서 제외 |
| 단일 프로세스 scheduler면 충분 | 단일 프로세스에서는 CPU(선택·`legal_moves`·규칙 필터·encode)와 GPU가 번갈아 돈다. evaluator를 고쳐도 CPU 잔여(착수당 약 24–32 ms)가 남아 상한은 약 3.4–5.4배다(stage8-plan §2.4) | 다음 단계는 **hybrid**(stage8-plan §9.1): CPU actor 프로세스 k개가 shared memory 요청 큐에 leaf를 넣고, GPU 서버 프로세스 1개가 모아서 평가한다. 상한은 대략 k × (단일 프로세스 배수) |
| `batch_wait_us` 대기 | 세대 안 게임 수가 고정이라 기다려도 더 오지 않는 경우가 많다 | 대기 0(준비된 요청만), actor 수가 많을 때만 짧은 대기를 측정 |
| TF32/BF16 | float 결과가 달라 visit count와 기보 해시가 바뀐다. stage8-plan은 GPU 경로에 해시 동일성을 요구하지 않는다 | 제안대로 policy/value parity와 대국 비교 뒤 채택. batch 16의 64×4 모델에서는 이득이 작을 가능성이 크다 |
| torch.compile | 데스크톱 측정은 Windows다. Inductor(Triton)는 Windows에서 설정이 따로 필요하다 | 고정 batch 크기(예: 1, 2, 4, 8, 16 bucket)로 **CUDA Graph를 직접 capture**하는 편이 작은 모델의 launch overhead 제거에 더 단순하다. compile은 그 뒤에 비교 |
| CUDA stream, 학습/추론 동시 | 세대 단위 순서(self-play → 학습 → 평가)가 재현성 계약이다 | 보류(제안의 우선순위와 같음) |
| 매 iteration `torch.cuda.synchronize()` | 결과를 CPU에서 바로 쓰는 self-play는 D2H 시점에 어차피 동기화된다 | 측정 코드에서만 쓴다 |

## 4. Track B self-play에서 GPU가 쓰이는 곳

- Track B self-play는 V8 엔진이 두 쪽을 모두 둔다. §1의 분해가 그대로 적용된다(NN 1% 미만, solver 90% 이상).
- H6에서 value head가 rollout을 대신해도 바뀌는 것은 tree(V8 시간의 약 7–9%) 안의 rollout 몫(그 33%)뿐이다. Track B self-play는 계속 **CPU solver bound**다.
- 그래서 Track B의 GPU-hours는 학습(H3, B2-H3′, H7 주기 학습)이 차지하고, CPU-hours는 self-play·solver 라벨링이 차지한다(track-b-b1-b2.md의 자원 분리와 같음). self-play를 줄이는 레버는 §2.2다.

## 5. 우선순위 (작업별)

| 작업 | 1순위 | 2순위 | 하지 않음 |
|---|---|---|---|
| E1/E2/S3/H5 (V8 대국) | 게임 단위 `--workers`, arm 간 같은 worker 수 | 패턴 스캔 컴파일(비트 동일 검증) | NN GPU 서버, solver GPU화 |
| Track B self-play / 라벨링 | 위와 같음 | 위와 같음 | 위와 같음 |
| Track A self-play | evaluator CPU batch encode + pinned buffer + 1회 전송(§3.2) → cross-game scheduler(≤16) | hybrid(CPU actor k + GPU 서버), CUDA Graph | batch 2048, 학습/추론 stream 병렬 |
| 학습 | `batch_to_device`(있음), BF16 AMP(`amp_bf16` 옵션 있음) parity 확인 | fused AdamW | — |
