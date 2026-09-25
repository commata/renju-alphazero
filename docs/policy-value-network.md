# Stage 4 Policy-Value Network

Stage 4는 향후 PUCT가 사용할 **고정된 입력·action·checkpoint 계약**을 검증한다.
기력 향상이나 AlphaZero 전체 구현이 목표가 아니다. PUCT, self-play, replay buffer,
noise/temperature, arena, V6 heuristic 통합은 포함하지 않는다. MCTS-v6는 고정
benchmark opponent로 보존한다. Stage 4 모델 구현 커밋에서는 기존 engine/search를 변경하지 않았고,
후속 코드 리뷰에서 발견된 RIF 5목 우선순위 정합성만 `src/renju`와 V3.2.1 fast scanner에
별도 회귀 수정으로 반영한다.

## 설치와 실행

검증 환경: Windows 11 (10.0.26200), Python 3.13.14, PyTorch **2.14.0+cpu**,
Intel64 Family 6 Model 170 Stepping 4, GenuineIntel. CPU thread 수는 1로 고정했다.
PyTorch는 [공식 설치 안내](https://docs.pytorch.org/get-started/locally/)의 CPU index에서
설치하고 import 및 tensor 연산을 직접 검증했다. Optional extra는 공개 버전 `torch==2.14.0`으로
고정한다. CPU local build suffix `+cpu`도 이 버전 조건을 만족한다.

```powershell
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[neural]"
python -m unittest discover -s tests -v
python scripts/run_tiny_overfit.py --seed 42 --samples 32 --max-steps 300 --eval-every 25 --threads 1
python scripts/benchmark_policy_value.py --threads 1 --iterations 100 --warmup 10 --seed 42
```

두 script는 `--output path.json`을 주면 학습 curve 또는 benchmark 전체 통계를 저장한다.
NumPy는 사용하지 않는다. 이 검증 환경에서는 NumPy 미설치로 torch import 시 optional
NumPy 초기화 경고가 출력되지만 tensor 연산, 학습, 저장·복원은 모두 실행되었다.
기본 설치와 engine-only import에는 torch가 필요하지 않다. `model.__init__`도 torch를 import하지 않는다.

## 입력과 action 계약

- `ENCODER_VERSION = "renju-relative-6p-v1"`
- `ACTION_INDEX_VERSION = "row-major-15x15-v1"`
- `CHECKPOINT_FORMAT_VERSION = 1`
- `coordinate_to_action(r,c) = r*15+c`, `action_to_coordinate(a) = divmod(a,15)`.
  범위 밖, 실수, bool 등 잘못된 좌표/index는 `ValueError`.
- Single encoding `[6,15,15]`, `torch.stack`으로 batch `[B,6,15,15]` 구성. `torch.float32`.

| Plane | 이름 | 의미 |
| --- | --- | --- |
| 0 | current_stones | `game.to_play`의 돌 |
| 1 | opponent_stones | 상대 돌 |
| 2 | last_move | 마지막 history 위치 1, history가 없으면 0 |
| 3 | current_is_black | 현재 차례가 BLACK이면 전체 1, WHITE이면 전체 0 |
| 4 | board_constant | 전체 1 |
| 5 | legal_actions | 현재 `Game.legal_moves()`에 포함되는 좌표만 1 |

Plane 5는 **legal action plane**이다. 중앙 opening restriction, occupied point,
흑 33/44/overline, 백의 정상 합법성을 함께 표현한다.
`encode_game(game, legal_mask=mask)`는 해당 상태용 mask를 신뢰하고 재사용하며 legality를
재계산하지 않는다. 호출자가 stale mask를 전달하지 않아야 한다.

```python
import torch
from model.encoding import encode_game
from model.masking import legal_moves_to_mask, masked_softmax
from model.network import PolicyValueNet

model = PolicyValueNet().eval()
# 향후 search 호출자는 game.done 및 legal move 존재 여부를 먼저 검사한다.
mask = legal_moves_to_mask(game.legal_moves())
x = encode_game(game, mask).unsqueeze(0)
with torch.inference_mode():
    logits, value = model(x)
    policy = masked_softmax(logits, mask.unsqueeze(0))
```

Terminal board encoding은 허용한다. 엔진의 `game.to_play`를 그대로 따르며 terminal 결과
label을 encoder가 자동 계산하지 않는다. Terminal/no-legal 상태의 policy 계산은 허용하지 않는다.

## 네트워크

Frozen dataclass `ModelConfig`: board_size=15, input_planes=6, channels=64,
blocks=4, policy_channels=2, value_channels=1, value_hidden=64.
구조 크기는 config로 관리하지만 board_size와 input_planes는 v1 계약상 바꿀 수 없다.

- Stem: 3×3 Conv(6→64, padding=1), BatchNorm, ReLU.
- Residual ×4: Conv3×3 → BN → ReLU → Conv3×3 → BN → skip add → ReLU.
- Policy: Conv1×1(64→2) → BN → ReLU → Flatten → Linear(450→225).
- Value: Conv1×1(64→1) → BN → ReLU → Flatten → Linear(225→64) → ReLU
  → Linear(64→1) → tanh.
- Conv는 bias=False, Kaiming normal 초기화. Linear/BN은 PyTorch 기본 초기화.
- **415,722 trainable parameters** (BN running buffers 제외).
- 출력은 `[B,225]` **raw logits**, `[B,1]` value. Forward 안에는 softmax가 없다.
- Value는 **현재 차례 플레이어 관점**: 승리 +1, 무승부 0, 패배 -1.

## Legal mask와 loss

`legal_moves_to_mask`가 유일한 move-list→mask 변환 함수다. Single `[225]`, batch
`[B,225]`, bool. Logits와 mask는 shape 및 device가 같아야 하며 자동 broadcasting하지 않는다.
각 row에 legal action이 없으면 `ValueError`. Masking 전에 finite logits를 검사한다.

추론: illegal logit을 `-inf`로 바꾼 뒤 softmax. Illegal probability는 정확히 0,
legal probability 합 오차는 1e-5 미만이다.

학습: 동일한 masked logits에 log-softmax를 적용한다. Target은 `[B,225]`, finite,
non-negative, 각 row 합 1(절대오차 1e-5), illegal mass 정확히 0이어야 한다.
불법 target은 자동 정규화하지 않고 거부한다. Illegal log-probability를 곱셈 전에 0으로
바꾸어 `0 * -inf`를 방지한다. Policy loss는 batch mean cross entropy다.
Value loss는 matching `[B,1]` shape, finite `[-1,+1]` target에 대한 MSE다.
Tiny verification에서는 두 loss를 단순 합산하며 최종 학습 weighting/regularization은 다루지 않는다.

## Checkpoint

`save_checkpoint(path, model)`은 다음 dictionary를 저장한다.

```text
checkpoint_format_version: 1
model_state: state_dict (parameters + BatchNorm buffers)
model_config: dataclass의 8개 구조 필드
encoder_version: renju-relative-6p-v1
action_index_version: row-major-15x15-v1
input_plane_names: 위 표의 6개 이름 (순서 포함)
torch_version: 실제 torch.__version__ 문자열
git_commit: 저장 시 소스 저장소 HEAD SHA 또는 null
```

`load_checkpoint(path, expected_config=ModelConfig(), device="cpu")`는 **기대 구조를
호출자가 지정**하고, 모든 구조 필드·plane 이름·계약 버전을 비교한다. Custom 구조는 같은
config를 명시해야 한다. `weights_only=True`, `strict=True`로 읽고 eval model을 반환한다.
누락/추가 weight와 incompatible metadata는 명시적 오류다. Torch 버전은 provenance로
기록하며 버전 문자열 차이 자체를 금지하지 않는다. Optimizer/RNG resume snapshot은 아니다.
Git SHA는 uncommitted 변경까지 식별하지 않으므로 실험에는 커밋된 소스를 사용한다.

임시 디렉터리 저장·복원 테스트에서 eval mode fixed input의 policy/value가
`torch.allclose(atol=1e-6, rtol=0)`를 모두 만족했다.

## D4 convention

행은 아래로, 열은 오른쪽으로 증가한다. Rotation은 **반시계 방향**이며
`(r,c)→(14-c,r)`, mirror는 **좌우 반전** `(r,c)→(r,14-c)`이다.

| id | 변환 |
| --- | --- |
| 0/1/2/3 | identity / CCW90 / CCW180 / CCW270 |
| 4/5/6/7 | mirror 후 CCW0 / CCW90 / CCW180 / CCW270 |

`transform_spatial`은 trailing `[15,15]`를 변환하며 board/planes/batch에 사용한다.
`transform_policy`, `transform_mask`는 row-major index를 같은 공간 변환으로 순열한다.
`transform_coordinate`, `transform_action`, `inverse_symmetry`도 제공한다.
반전 계열은 자기 자신의 inverse이고 rotation inverse는 반대 방향 rotation이다.
Value target은 변하지 않는다. Network equivariance를 요구하거나 검사하지 않는다.

비대칭 board·last move·policy·mask·action의 일치 및 exact inverse round-trip을 검증했다.
삼삼/사사/장목 각 fixture의 흑·백 × 8 transforms **48건**에서 transform(original mask)와
transformed board의 엔진 mask가 정확히 일치했다. 중앙 opening mask도 8변환에서 일치한다.

## Tiny overfit 결과

[전체 curve JSON](policy-value-overfit-results.json). Seed 42, 독립적인 합법 착수 prefix의
non-terminal state 32개, 6~24 plies. Target action은 그 상태의 legal list에서 선택한다.
Value는 -1/0/+1 순환 label이며 **학습 가능성 확인용 합성값**이다. 실제 기력을 의미하지 않는다.
Dataset fingerprint: `f7999e17e6acfe85a684f8d8e9202ebea51834c6a1d3425bf2ae3baf191c5215`.
재생성한 tensor와 fingerprint의 일치도 테스트했다.

Adam lr=0.001, full batch=32, deterministic algorithms, 최대 300 steps,
25 steps마다 평가하며 **50 steps에서 종료**했다. 매 step gradient의 finite 여부를 검사한다.
Train metric 측정이 BN running buffers를 변경하지 않도록 복원한 후 eval metric을 측정한다.

| Step | Train loss | Train top-1 | Eval top-1 (masked) | Eval value MSE |
| --- | ---: | ---: | ---: | ---: |
| 0 | 6.087915 | 3.125% | 0% | 1.022337 |
| 25 | 0.021749 | 100% | 59.375% | 0.577247 |
| 50 | 0.004891 | 100% | **100%** | **0.009394** |

최종 train policy loss=0.004799, train value MSE=0.00009257.
Eval policy loss=0.085961, eval total loss=0.095355. NaN/Inf=0. Eval gate PASS.

## CPU benchmark

[전체 통계·환경 JSON](policy-value-benchmark-results.json). Seed 42의 60-ply black midgame,
eval + inference_mode, 각 항목 warm-up 10회 후 100회 측정. Forward input 반복 구성은 측정 밖이다.
Encode는 미리 계산한 mask를 재사용한다. Full pipeline은 legal_moves → mask 생성 → encode →
forward → masking/softmax 전체를 매번 실행한다. Table의 forward 시간은 batch 전체 latency다.

| 구간 | Mean ms | Median ms | Min ms | Max ms | samples/sec |
| --- | ---: | ---: | ---: | ---: | ---: |
| legal_moves | 2.598 | 2.612 | 1.783 | 4.247 | 384.85 |
| mask 생성 | 0.100 | 0.100 | 0.088 | 0.127 | 9970.69 |
| encode (mask 재사용) | 0.051 | 0.051 | 0.040 | 0.148 | 19572.54 |
| forward B1 | 2.857 | 2.839 | 1.967 | 5.116 | 350.03 |
| forward B8 | 14.150 | 14.252 | 11.129 | 16.726 | 565.36 |
| forward B32 | 55.329 | 54.597 | 46.075 | 74.006 | 578.36 |
| masking + softmax | 0.034 | 0.029 | 0.028 | 0.127 | 29498.53 |
| full pipeline B1 | 6.224 | 5.933 | 4.401 | 11.456 | 160.68 |

구간은 별도로 측정하므로 평균 합과 full pipeline이 정확히 같지는 않다. 열·클럭·다른 프로세스
영향을 받는 로컬 측정이며 latency SLA가 아니다.

## 회귀와 완료 판정

> 아래 수치와 benchmark는 Stage 4 모델 구현 완료 시점의 측정 기록이다. 후속 RIF 규칙 계약
> 교정은 모델 구조/가중치 계약을 바꾸지 않지만 engine/search 회귀 테스트를 함께 갱신하므로,
> 새 기준 성능 수치로 사용할 때는 해당 커밋에서 다시 측정한다.

- `python -m unittest discover -s tests -v`: Stage 4 모델 구현 완료 시점에 기존 149 + Stage 4 23 = **172 PASS**, skip 없음.
- 별도 프로세스에서 import finder로 torch를 차단: 기존 149 + 순수 계약 3 = **152 PASS**,
  neural 20개 skip. Engine/agents/evaluation import에 torch가 포함되지 않음도 확인.
- `python -m pip check`: No broken requirements found.
- `python -m pip wheel . --no-deps --no-build-isolation --wheel-dir <temp>`: wheel build 성공.
- Stage 4 모델 구현 완료 시점에는 `src/renju src/agents src/search src/evaluation` 변경이 없었다.
  후속 규칙 계약 교정에서는 exact-five 우선순위와 동일 의미를 보장하기 위해
  `src/renju/rules.py` 및 `src/search/mcts_v321.py`의 fast scanner만 회귀 수정한다.
- Engine-only `python scripts/benchmark_engine.py --iterations 100 --games 10 --seed 42`:
  구현 전 3.095 ms / 732.560 moves/sec, 구현 후 **2.168 ms / 898.827 moves/sec**.
  사용자 기준 2.435 ms / 약 855 moves/sec와 비교해 악화 징후 없음. 엔진이 동일하므로
  전후 차이는 속도 개선으로 주장하지 않는다. 두 실행 모두 10판, 총 1252 moves.

**PASS**: input/action/version, mask/loss/gradient, strict checkpoint, D4 legality,
eval tiny overfit, 전체 회귀 및 CPU pipeline 측정 기준 충족. Stage 5 PUCT integration을
시작할 수 있는 인터페이스를 확보했다. Stage 5 코드는 추가하지 않았다.
