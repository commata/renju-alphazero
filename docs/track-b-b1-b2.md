# Track B 최종 설계: B1-Champion / B1-Matched / B2

작성 2026-10-09. 상태: **확정**(사용자 승인). 수치는 §13에서 고정했다(2026-10-09, B1 학습 시작 전). 엔진 쪽 운영 규칙(주기 단위 동결, 채택 gate, 새 패배 처리)은
[track-b-post-e1.md](track-b-post-e1.md)를 따른다. §번호는 따로 적지 않으면 [mcts-v8-teacher.md](mcts-v8-teacher.md)의 절이다.

## 0. 결론

Track B를 세 계열로 나눈다.

| 계열 | 역할 | 엔진 | 초기 네트워크 |
|---|---|---|---|
| **B1-Champion** | 프로젝트 최강 모델 | `TB-Engine-v1` → 주기 경계에서 gate 통과 시 `v2` … `vK` | H3 RenjuNet policy |
| **B1-Matched** | RenjuNet 효과를 재는 통제군 | `TB-Engine-vK` 고정 | H3 RenjuNet policy |
| **B2** | RenjuNet-free 실험군, 공개 배포 후보 | `TB-Engine-vK` 고정 | 독립 초기화(B2-H3′ policy bootstrap) |

- B1-Matched와 B2는 엔진, 네트워크 구조, PUCT, solver, H6/H7 프로토콜이 같다. 다른 것은 **RenjuNet provenance 하나**다.
- B1-Champion은 학습 중에 엔진이 바뀔 수 있으므로 RenjuNet 효과 측정에는 쓰지 않는다. B2의 catch-up 목표선으로만 쓴다.
- 단계 이름은 **H7 = 반복 학습 주기(H7-c1, c2, …), H8 = 최종 평가**다(§12.7과 같은 뜻).

## 1. 프로젝트 구조와 연구 질문

```
Renju AI Project
├── Track A — Self-play-only AZ (AZ-Tactical baseline)
│     규칙 + 즉승·즉방 필터, random init, PUCT, self-play. 외부 기보 ✗, solver supervision ✗
│
└── Track B — Hybrid Renju AI (공통 엔진 코드: S3-VCT2-v1 계열)
      ├── B1-Champion   RenjuNet-assisted, 엔진 개선 허용     → 최강 기력
      ├── B1-Matched    RenjuNet-assisted, TB-Engine-vK 고정  ┐
      └── B2            RenjuNet-free,     TB-Engine-vK 고정  ┘→ matched comparison
```

| RQ | 질문 | 비교 |
|---|---|---|
| RQ1 최대 성능 | RenjuNet + Hybrid solver + self-play로 어디까지 강해지는가 | B1-Champion |
| RQ2 RenjuNet 효과 | 다른 조건이 모두 같을 때 RenjuNet 사전학습이 학습 효율과 기력에 주는 영향 | **B1-Matched ↔ B2** |
| RQ3 catch-up 비용 | RenjuNet 없이 B1-Champion 수준에 도달하는 데 드는 추가 자원 | **B2 ↔ B1-Champion** |
| RQ4 Hybrid solver 효과 | 외부 기보 없는 조건에서 solver·전술 계층이 self-play 효율을 얼마나 올리는가 | **Track A ↔ B2** |

각 비교는 질문 하나만 맡는다. RQ4는 엔진 전체(VCF/VCT/VCT2, V8 모듈)를 묶은 효과이고 모듈별 효과로 쪼개지 않는다.

## 2. 정의

- **B1-Champion.** RenjuNet policy 사전학습, solver 증명 라벨(RenjuNet 국면 포함), PUCT, 전술 엔진, Hybrid self-play를 모두 써서
  같은 하드웨어와 합리적인 착수 시간 안에서 가장 강한 Hybrid 렌주 AI를 만드는 모델. 엔진은 H7 주기 경계에서만 바뀐다(track-b-post-e1.md §2).
- **B1-Matched.** B1-Champion이 마지막으로 쓴 `TB-Engine-vK`를 처음부터 고정하고, H3 checkpoint에서 시작해 B2와 같은 H6/H7 프로토콜로 학습하는 통제군.
  RenjuNet 데이터와 RenjuNet 국면의 증명 라벨은 B1-Champion과 같은 규칙으로 쓴다.
- **B2.** B1-Matched와 같은 엔진 코드·설정, 네트워크 구조, solver, 학습 코드, 평가 프로토콜을 쓴다.
  RenjuNet 데이터, RenjuNet 파생 국면·가중치·출력, B1 계열 파생 학습 데이터를 쓰지 않는다.
  네트워크를 독립적으로 초기화하고, 자체 self-play와 자체 국면의 solver 증명만으로 학습한다.

**지름길.** B1-Champion이 학습 중 엔진을 한 번도 바꾸지 않았다면(vK = v1), B1-Matched는 B1-Champion의 H6/H7 궤적과 같다.
이때는 B1-Matched를 다시 돌리지 않고 B1-Champion의 기록을 쓴다. 단 H6/H7 프로토콜이 §6과 같았는지 확인한다.

## 3. B2 provenance 규칙

| 구분 | 항목 |
|---|---|
| **공유 가능** | 렌주 규칙 엔진, 네트워크 구조, 학습 코드, PUCT, VCF/VCT/VCT2 solver, 전술 엔진 코드와 설정, 벤치마크·평가 코드, 고정 probe의 정의 |
| **금지 (학습·초기화·라벨 모두)** | H3 `best.pt`와 그 파생 checkpoint, RenjuNet 기보·국면·policy logits, **RenjuNet 국면에 붙인 solver 라벨**(solver가 독립이어도 국면 출처가 RenjuNet), B1 계열 checkpoint에서 시작하는 fine-tuning, B1 계열 착수·visit 분포를 target으로 쓰는 것, B1 계열 엔진이 둔 대국(self-play, B1과 B2의 대국, 웹 대국, E1 suite, H5·S3 벤치마크)에서 나온 국면 |
| **평가 전용** | B1-Champion·B1-Matched checkpoint, frozen V8·`puct_policy`, E1 suite, 웹 probe, RenjuNet test 국면. B2의 학습·승급 결정에 쓰지 않는다 |

- B2 데이터의 모든 행에 `source`, `game_id`, `model_hash`, `engine_config_hash`를 기록한다(§12.18 계약).
- 학습 시작 전에 금지 출처의 D4 정규형 hash 집합과 B2 데이터의 교차를 검사한다. 하나라도 겹치면 학습을 거부한다.
- B2 승급은 **직전 B2 checkpoint를 상대로만** 판정한다. B1 계열과의 대국 결과는 기록만 한다.

## 4. 엔진 버전 규칙

- B1-Champion: `TB-Engine-v1`로 H6를 시작한다. H7 주기 경계에서 gate를 통과한 변경만 올린다. 버전마다 사용한 주기를 기록한다.
- B1-Matched와 B2: B1-Champion의 최종 엔진 `TB-Engine-vK`로 시작해 끝까지 고정한다.
- **알려진 한계.** c_puct 1.5, S3 예산, H4 policy 순서 같은 엔진 하이퍼파라미터는 RenjuNet policy 기준으로 보정됐다.
  - 이 편향은 B1 계열에 유리한 방향이라 multiplier를 과대 추정하는 쪽이다. 보고서에 적는다.
  - B2에 맞춘 재보정이 필요하면 별도 arm(B2-retuned)으로 돌리고, 보정 비용을 B2 자원에 넣는다. 기본은 off.

## 5. B2-H3′: policy bootstrap 전용

B1의 H3는 policy만 학습했다(checkpoint metadata `policy_trained: true`, `value_trained: false`, `src/hybrid/h3_train.py`).
B2-H3′도 같은 자리에 둔다. **value는 학습하지 않는다.**

```
B2-H3′
  같은 엔진(TB-Engine-vK)을 `puct_heur` prior로 self-play   (H5에서 측정한 arm, RenjuNet 없음)
    → visit 분포 π 저장, PROVEN_LOSS 수에 간 방문은 지움(§5.2 규칙)
    → random init 네트워크에 policy loss만 학습. value loss = MASK
    → 메타데이터 policy_trained: true, value_trained: false
B2-H6
  여기서 처음 value 학습(§12.18 라벨 계약, 증명 WIN/LOSS 우선 + 자체 self-play 결과)
```

| 단계 | B1-Matched | B2 |
|---|---|---|
| H3 / H3′ | RenjuNet policy (기존 H3 checkpoint) | 자체 self-play policy |
| H6 | value 시작 | value 시작 |
| H7 | 반복 학습 주기 | 반복 학습 주기 |

- **B2-H3′ 예산(§13 고정).**
  - self-play 2,000판을 기본으로 하고, D4 정규형 unique position이 100,000개가 될 때까지 계속 생성한다. hard cap은 3,000판이다.
  - 3,000판에서도 100,000개가 안 되면 그 데이터로 시작한다. 이 부족 자체를 RenjuNet-free bootstrap의 결과로 보고한다.
  - 학습은 H3와 같은 네트워크 구조·optimizer 계열·batch 1,024로 **10 epoch**. H3의 step 수(31,030)를 맞추지 않는다(작은 데이터의 과반복 방지).
  - 이 단계의 자원은 모두 B2 고유 비용이다.
- Track A checkpoint로 초기화하는 안(B2-A)은 RQ4와 섞이므로 별도 arm으로만 둔다. 기본은 off.

## 6. 학습 프로토콜 동일성 (B1-Matched = B2)

- **H7 주기 크기:** 모든 계열에서 **1주기 = self-play 1,000판**(H7-c1 = 1,000판, c2 = 누적 2,000판, …). 곡선의 x축 해상도가 이 단위다.
- **같게 유지:** 네트워크 구조, PUCT simulations, optimizer, batch, LR schedule, replay buffer 규칙, 주기당 self-play 양,
  승급 프로토콜, self-play opening 프로토콜, 증명 라벨링 예산, 평가 프로토콜.
- **다른 것:** 초기화와 데이터 provenance.
- B1-Champion도 가능한 한 같은 H6/H7 프로토콜을 쓴다. 다른 점(엔진 버전 변경)은 주기별로 기록한다.
- B2가 정체되어 프로토콜을 바꾸고 싶으면 원래 arm은 그대로 두고 변경을 별도 arm으로 기록한다. 변경 arm의 결과는 RQ2/RQ3 multiplier에 섞지 않는다.
- **실행은 순차로 한다.** 같은 데스크톱에서 두 계열을 동시에 돌리지 않는다(시간 비율 오염, §12.18 단계 B와 같은 이유).

## 7. 자원 기록과 multiplier

### 7.1 주기마다 누적 기록 (모든 계열)

| 항목 | 단위 |
|---|---|
| GPU-hours | 학습·추론, 장치 이름 |
| CPU-hours | self-play, solver 라벨링 |
| wall-clock | 단계별 |
| self-play | 판 수, 국면 수 |
| training | optimizer step 수, 처리 샘플 수 |
| solver | 노드 수, 호출 수, 생성한 증명 라벨 수(깊이별) |
| 메모리 | peak VRAM, peak RAM |

### 7.2 비용 분류

- **B1 고유:** H3 사전학습 + 해당 계열의 H6 이후 전부. B1-Champion과 B1-Matched는 각각 따로 센다(H3 비용은 둘 다에 포함).
- **B2 고유:** B2-H3′ bootstrap, B2 H6 이후 전부, (쓰면) B2-retuned 보정.
- **공통 엔진 개발:** S1–S3, E0–E3, E1 screen, 엔진 gate 측정. 어느 계열에도 넣지 않고 따로 보고한다.
- **외부 지식:** RenjuNet(약 16.5만 판). 계산량으로 환산하지 않는다. "B1 계열은 사람이 만든 기보라는 외부 지식을 썼다"고 질적으로 적는다.
  B1의 계산량이 적다고 해서 그 지식을 공짜로 얻었다는 뜻이 아니다.

### 7.3 multiplier는 지표별로 보고한다

단일 숫자로 합치지 않는다. CPU(self-play, solver)와 GPU(학습·추론)의 비중이 계열마다 다를 수 있기 때문이다.

| multiplier | 정의 (같은 목표 기력에서) |
|---|---|
| GPU | B2 GPU-hours / 기준 GPU-hours |
| CPU | B2 CPU-hours / 기준 CPU-hours |
| self-play | B2 self-play 판 수 / 기준 판 수 |
| training sample | B2 처리 샘플 수 / 기준 처리 샘플 수 |
| wall-clock | B2 wall-clock / 기준 wall-clock |

- 기준은 RQ2에서 B1-Matched, RQ3에서 B1-Champion이다.
- 대표 문장은 두 지표를 함께 쓴다. 예: "GPU-hours 기준 2.1배, CPU-hours 기준 3.4배".
- 최종점 하나에만 의존하지 않도록, 곡선이 겹치는 중간 기력 수준들에서도 multiplier를 표로 낸다.
- 단일 Normalized Compute Cost는 쓰지 않는다. 필요해지면 B2 시작 전에 공식을 따로 정한다.

### 7.4 compute ceiling

- **soft review = 3×.** B2의 GPU-hours 또는 CPU-hours가 B1-Champion 고유 값의 3배에 닿으면 한 번 진단한다(현재 Elo, 주기별 기울기, GPU·CPU 사용량). 멈추지 않는다. 진단 결과로 프로토콜을 바꾸지 않는다(바꾸려면 §6의 별도 arm).
- **hard ceiling = 5×.** GPU-hours 또는 CPU-hours 중 하나라도 B1-Champion 고유 값의 5배에 닿으면 B2를 멈춘다.
- 상한까지 catch-up 기준을 못 넘으면 결과는 "multiplier > k"(censored)로 보고한다. 실패가 아니라 측정 결과다.
- B1-Matched는 자기 수렴 기준(§9)까지 돌린다. B2와의 RQ2 곡선 비교는 두 계열이 모두 지나간 자원 구간에서만 한다.

## 8. 평가: 두 층

### 8.1 H7 Monitoring Arena (개발 중 사용)

- 고정 상대 풀: frozen V8(`v8:full`), H5 `puct_policy`, E2 VCT2 응징 상대, B1-Champion 확정 후에는 B1-Champion-v1.
- 주기마다 모든 checkpoint(모든 계열)를 고정 착수 시간, 양색 균형, 공개 seed·opening으로 평가해 Elo(Bradley–Terry) 척도를 만든다.
- **크기(§13 고정):** 고정 상대당 25쌍(50판) × 상대 3종(`v8:full`, H5 `puct_policy`, E2 상대) = 150판, 그리고 직전 checkpoint와 50쌍(100판). 주기당 약 250판.
  B1-Champion-v1이 확정된 뒤의 B2·B1-Matched 주기에는 B1-Champion-v1과의 25쌍을 기록용으로 더한다.
- 전술 gate: 금수 오류 0, `unsound_witness` 0, P93·P94 value 부호, must_block probe, E1 suite 지표.
- 진행 확인과 곡선 그리기에 쓴다. 승급은 계열 안에서 직전 checkpoint를 상대로만 판정한다(§3).
- 평가 대국은 어느 계열의 학습에도 쓰지 않는다.

### 8.2 H8 Final Holdout (봉인)

- 개발 중 쓰지 않은 seed 블록, opening 세트, probe 세트로 만든다. 만든 시점에 manifest(파일 SHA-256)를 커밋하고 내용은 열지 않는다.
- probe 국면의 D4 정규형은 모든 계열의 학습 데이터에서 제외한다(§12.18 누출 규칙과 같음).
- **봉인 세트는 두 개다.**
  - **Final-1:** B1-Champion H8 한 번에만 연다. B1-Champion-v1 확정용.
    - 고정 상대 3종(`v8:full`, H5 `puct_policy`, E2 상대) × 100쌍 = **600판**
    - 봉인 tactical holdout 약 **200국면**: 금수 판정, must_block, VCF/VCT, VCT2, value 부호, stage 경로 안전
  - **Final-2:** B1-Matched와 B2가 모두 끝난 뒤 한 번만 연다.
    - 직접: B2 최종 ↔ B1-Champion-v1 **200쌍 = 400판**(RQ3 catch-up 판정)
    - 공통: B1-Champion-v1, B1-Matched 최종, B2 최종 각각 × 상대 3종 × 50쌍 = 모델당 300판, **합 900판**(B1-Matched를 넣어야 RQ2를 Final-2로 주장할 수 있다)
    - 합계 **1,300판**. B1-Matched를 지름길(§2)로 생략했다면 1,000판
- **구성 규칙.** Final-1과 Final-2는 seed·opening·국면을 완전히 분리한다. 규칙 엔진으로 새로 생성한 합법 opening·국면을 쓰고,
  RenjuNet train, B1 계열 self-play, 웹 대국, E1-dev, E1-S, 기존 probe와 D4 정규형이 겹치면 뺀다. 기존 probe는 safety test로만 계속 쓴다.
- **재사용 금지.** 연 holdout에서 문제가 보여도 그 모델을 고쳐 같은 holdout으로 다시 재지 않는다. 결과는 그대로 보존하고, 고친 모델은 v2로 등록해 새 봉인 세트로 잰다.
  - Final-1 결과를 본 상태로 B2를 개발하므로, 최종 비교(RQ2, RQ3)는 Final-2로만 주장한다.
- 열기 전에 판정 기준(§9)과 판 수를 고정한다. 연 뒤에 기준을 바꾸지 않는다.

## 9. 사전 고정 기준

**B1 수렴 기준** (B1-Champion H7 시작 전 고정, B1-Matched도 같은 기준). 아래 세 조건을 모두 만족하면 수렴이다.

1. H7 주기를 **최소 3개** 끝냈다.
2. **연속 2개 주기**에서 직전 checkpoint와의 50쌍 대국 점수 향상(score − 0.5)의 쌍 bootstrap 95% 하한이 0 이하다.
3. 그 checkpoint가 §8.1 전술 gate에서 회귀가 없다.

수렴한 주기의 checkpoint 중 마지막 것을 B1 최종 후보로 하고 H8로 간다. 50쌍은 검출력이 낮아 작은 향상을 놓칠 수 있다. 최소 3주기 조건이 그 위험을 줄인다.

**B2 catch-up 기준** (B2 시작 전 고정). 아래 세 조건을 모두 만족하면 목표 수준 도달이다.

1. **직접:** 목표 모델(RQ3는 B1-Champion-v1)과 **200쌍 = 400판**에서 score ≥ 0.45, 쌍 bootstrap 95% 하한 ≥ 0.40.
   - 400판인 이유: 관측 score가 정확히 0.45일 때 이항 근사 95% 하한이 200판 ≈ 0.383, 300판 ≈ 0.395, 400판 ≈ 0.402다. 두 조건을 같이 두려면 400판이 필요하다.
   - score 0.45 ≈ Elo −35다.
2. **공통 척도:** Arena Elo **점추정** 차 ≤ 50. 95% 구간은 보고만 한다(구간 전체를 ±50 안에 넣는 조건은 두지 않는다. 불확실성 조건은 1번이 맡는다).
3. **전술:** §8.1 전술 gate를 모두 통과한다.

- 진행 중에는 주기마다 Monitoring Arena로 후보를 고른다. 후보가 생기면 B1-Champion-v1과 200쌍 직접 대국을 돌려 1·2·3을 확인한다(Final-2 seed는 쓰지 않는다).
- 통과하면 B2를 멈추고 Final-2를 연다. 최종 주장은 Final-2의 같은 기준으로만 한다. Final-2에서 통과하지 못하면 "개발 seed에서는 도달, 봉인 holdout에서는 미도달"로 보고한다.

## 10. 목표선 버전

- `B1-Champion-v1` = checkpoint SHA-256 + `engine_config_hash`(= `TB-Engine-vK`) + Final-1 결과.
- B2 진행 중 B1이 더 좋아져도 v1은 움직이지 않는다. 새 모델은 `B1-Champion-v2`로 따로 등록한다.
- 실험 1(B2 → v1)과 실험 2(B2 → v2)는 별도 실험이다. v2를 쓰면 B1-Matched도 v2의 엔진으로 다시 정의해야 한다.

## 11. 개발 순서

```
현재: S3-VCT2-v1 (H3 RenjuNet policy)
  (7,11) 10M 참값 → E1-dev → E1-S → E2 → (필요하면 E3 한 묶음) → E1-holdout
  → TB-Engine-v1 동결, B1 수렴 기준 고정, Final-1·Final-2 manifest 생성(봉인)
══════════ B1-Champion ══════════
  H6 → H7-c1, c2, … (주기 경계에서만 엔진 v2…vK) → 수렴
  → H8: Final-1 개봉 → ★ B1-Champion-v1 확정
══════════ 비교 실험 준비 ══════════
  §13 수치 재확인(이미 고정, 변경 시 기록), Monitoring Arena 상대 풀 확인
══════════ B1-Matched (vK ≠ v1일 때만) ══════════
  H3 checkpoint + TB-Engine-vK → H6 → H7-c1, c2, … → 수렴
══════════ B2 ══════════
  B2-H3′ policy bootstrap (TB-Engine-vK, value 학습 없음)
  → H6 → H7-c1, c2, … (주기마다 Monitoring Arena + B1-Champion-v1 직접 대국, 기록만)
  → catch-up 도달 또는 compute ceiling 도달
══════════ 최종 ══════════
  H8: Final-2 개봉 → RQ1–RQ4 분석
```

## 12. 최종 산출물

1. **자원 대비 기력 곡선.** x축 누적 GPU-hours(CPU-hours, self-play 판 수 버전도 함께), y축 Arena Elo.
   B1-Champion, B1-Matched, B2, Track A를 같은 그림에 그린다.
2. **multiplier 표.** RQ2(B2 / B1-Matched)와 RQ3(B2 / B1-Champion)를 §7.3 지표별로, 목표 기력과 중간 기력 수준에서.
3. **효과 분해.** RQ2 RenjuNet 효과, RQ4 Hybrid solver 효과.
4. **한계.** 엔진 보정 편향(§4), 외부 지식 비용 미환산(§7.2), 하드웨어·소프트웨어 버전 보정, censored 여부(§7.4), Final-1을 본 뒤 B2를 개발했다는 사실(§8.2).

보고서 문장 예:
> B1-Champion은 프로젝트 내 최강 모델이었다. 같은 최종 Hybrid 엔진을 고정한 B1-Matched와 B2를 비교해 RenjuNet 사전학습의 효과를 측정했고,
> 별도로 B2가 B1-Champion의 기력에 도달하기까지 필요한 GPU-hours, CPU-hours, self-play 양을 측정했다.

## 13. 확정 수치 (2026-10-09 고정)

결과를 본 뒤 바꾸지 않는다. 바꿔야 하면 이유와 날짜를 이 표 아래에 적고, 바꾸기 전 결과와 함께 보고한다.

| 항목 | 값 | 고정 시점(늦어도) |
|---|---|---|
| H7 주기 크기 | self-play **1,000판** | B1-Champion H6 시작 전 |
| B1 최소 H7 주기 | **3** | B1-Champion H7 시작 전 |
| B1 수렴 | 연속 2주기, 직전 checkpoint 50쌍 향상 하한 ≤ 0, 전술 회귀 없음 | B1-Champion H7 시작 전 |
| Monitoring Arena | 상대 3종 × **25쌍** + 직전 checkpoint 50쌍 | B1-Champion H7 시작 전 |
| Final-1 | 상대 3종 × **100쌍 = 600판** + tactical **약 200국면** | `TB-Engine-v1` 동결 때 manifest |
| catch-up 직접 | **200쌍 = 400판**, score **≥ 0.45**, 쌍 bootstrap 하한 **≥ 0.40** | B2-H3′ 시작 전 |
| catch-up Elo | 점추정 차 **≤ 50**, 95% 구간은 보고 | B2-H3′ 시작 전 |
| compute | soft review **3×**, hard ceiling GPU **5×** 또는 CPU **5×** | B2-H3′ 시작 전 |
| B2-H3′ | **2,000판** + unique position **≥ 100,000**, hard cap **3,000판**, 10 epoch, batch 1,024 | B2-H3′ 시작 전 |
| Final-2 | 직접 **400판** + 공통 모델당 **300판**(3종 × 50쌍) | `TB-Engine-v1` 동결 때 manifest |
| B2-retuned, B2-A | **off** | — |
