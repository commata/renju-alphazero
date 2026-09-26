# MCTS-v7 명세 및 구현 기록 (Stage 6.5 Frozen Benchmark)

> 상태: **V7_FINAL 동결 전 구현 후보**. 기준선은 Stage 6 merge `9c9456a`, 개발 브랜치는
> `feat/mcts-v7-benchmark`다. 좌표는 분석 fixture에서 1-based (행, 열), 엔진 내부에서는 0-based다.

## 1. 목적과 원칙

MCTS-v7은 AlphaZero 학습 곡선을 측정하기 위한 **고정 계측기**다. 목표는 최강 classical 엔진이
아니라 V6의 확인된 맹점을 한 차례 정리한 마지막 classical benchmark를 만드는 것이다.

- V6의 `V5_FINAL` 탐색 파라미터와 Stage 1~5, threat planner는 수정하지 않는다.
- V7은 `src/search/mcts_v7.py`, `src/agents/mcts_v7_agent.py`에서만 확장한다.
- 새 모듈 파라미터는 AlphaZero 체크포인트 결과를 보고 조정하지 않는다.
- V7_FINAL 동결 이후 변경이 필요하면 V7 자체를 고치지 않고 별도 후속 버전으로 분리한다.
- VCT 방어와 rollout 내부 VCF, V6 후보 생성/파라미터 변경은 Stage 6.5 범위 밖이다.

## 2. 근거가 된 V6 맹점

완료된 seed 777 10판과 seed 44 100판 분석을 근거로 한다.

- 상대 VCF를 허용한 패배와 자기 VCF를 놓친 국면이 반복적으로 확인됐다.
- 해당 안전수/VCF 첫 수가 V6 root 후보 안에 있었으므로 후보 생성보다 **후보 선택** 문제가 핵심이었다.
- 흑 패배 중 자기 수가 훗날 이용되는 금수점을 만든 사례가 확인됐다.
- VCT 패배는 별도 문제이며 V7에는 넣지 않는다.
- 완료되지 않은 공격 4배 simulation 성공률이나 exploration 변형은 결과로 취급하지 않는다.

## 3. V7 결정 흐름

```text
Stage 1   내 즉시 승리                         V6 그대로
Stage 2   상대 즉시 승리 차단                  V6 그대로
Stage 3   내 unstoppable four                  V6 그대로
Stage 3V  내 VCF 발견 → 첫 수 착수             M1
Stage 4   상대 unstoppable four 방어           V6 + M4
Stage 5   상대 이중위협 방어                    V6 그대로
Stage 6   V6 root 후보 생성
          → 상대 VCF 안전 필터                  M2
          → 흑 자기 금수점 감점                 M3
          → V6 tree/rollout                     V6 그대로
```

## 4. M0/M1 — 보수적 VCF 탐색

`find_vcf(game, attacker, max_fours, node_limit)`를 M1/M2/M4가 공유한다.

- 공격 후보는 attacker 돌이 3개 이상이고 상대 돌이 없는 5칸 window에서 만든다.
- 공격수는 합법적인 4만 사용한다. 흑은 엔진의 정확한 금수 판정을 재사용한다.
- 공격 직후 방어측 즉시 승리가 있으면 해당 수순을 버린다.
- 합법 완성점이 둘 이상이거나 방어측이 완성점을 합법적으로 막을 수 없으면 VCF 승리다.
- 하나뿐인 완성점을 막았을 때 방어측이 즉시 승리하거나 역4를 만들면 그 수순은 버린다.
- node/four 상한을 넘으면 보수적으로 실패 처리한다.
- 모든 임시 배치는 원상 복구되고 후보 순서는 결정론적이다.

M1은 기존 Stage 1~3보다 뒤, Stage 4/5보다 앞에서 own VCF를 찾으면 MCTS를 호출하지 않고 첫 수를 둔다.

## 5. M2 — 상대 VCF 안전 필터

V6 root 후보를 생성한 뒤 후보별로 임시 착수하고 상대 VCF를 검사한다.

- 현재 국면의 상대 VCF를 먼저 한 번 검사한다.
- **내 후보가 4를 만들면 상대의 유일한 합법 완성점 차단까지 강제 응수로 둔 뒤** 상대 VCF를 검사한다. 단순 4가 상대에게 막는 돌만 공짜로 주고 기존 VCF를 남기는 경우를 안전수로 오판하지 않는다.
- 현재 상대 VCF가 없고 후보가 4도 아니라면, 그 한 수가 흑 금수 상태를 위험한 방향으로 바꾸는 경우에만 개별 VCF 검사를 한다. BLACK 수는 새 흑 금수점을 만들어 WHITE forcing line의 방어점을 없앨 수 있고, WHITE 수는 기존 흑 금수를 해제해 BLACK 공격수를 합법화할 수 있다.
- 후보별 VCF는 4,000 node, 한 착수의 M2 전체는 16,000 node로 제한한다. bounded probe가 결론을 내지 못하면 해당 후보를 보수적으로 안전수로 인정하지 않고 budget 소진을 진단에 남긴다.
- 상대 VCF를 남기는 후보를 제외한다.
- 전부 제외되면 현재 상대 VCF 공격/완성점과 내 4 생성점을 보강 후보로 검사한다.
- 여전히 안전수가 없으면 V6 후보를 그대로 사용하고 fallback 진단을 남긴다.
- M1이 이미 수를 결정했다면 M2는 실행되지 않는다.

## 6. M3 — 흑 자기 금수점 감점

흑 후보를 둔 뒤 백 압력이 있는 window에서 **새로 생기는 흑 금수점**을 검사한다.
해당 후보를 제거하지 않고 root 후보 순서의 뒤로 보내며, 동일 그룹 안에서는 V6 순서를 유지한다.

실제 seed777 31수 fixture에서는 end-to-end V7이 M1/M2에서 먼저 나쁜 수를 피할 수 있기 때문에,
M3 자체는 기록된 V6 root 후보를 입력해 bad move가 penalized suffix로 이동하는지 별도로 검증한다.

## 7. M4 — Stage 4 동률 규칙과 fixture 재감사

M4는 V5 Stage 4의 **남은 unstoppable four 최소화**를 최우선으로 그대로 유지한다.
그 값이 같은 방어 후보끼리만 다음을 추가한다.

1. 착수 후 상대 `_double_threat_moves` 수가 적은 후보
2. 착수 후 상대 VCF가 없는 후보
3. 그 외에는 기존 V6/V5 root key

### seed777-g003-p021 재감사 결과

제공 fixture를 현재 엔진과 V7 탐색기로 다시 계산한 결과 Stage 4 방어 후보는 내부 0-based 좌표
`(4,9)`, `(6,9)`, `(9,9)` 세 개였다. 세 후보가 모두 다음 값을 가졌다.

- remaining unstoppable four = 0
- 백 compound creator = `(5,11)` 한 개 (fixture의 1-based `(6,12)`)
- opponent VCF = true

따라서 제공된 M4 기준으로는 이 실제 국면을 **구분할 근거가 없다**. 기존 V6 수 `(6,9)`
(1-based `(7,10)`)을 억지로 변경하지 않는다. 이 fixture는 M4 필요성을 제기한 분석 사례로 보존하되
“반드시 다른 수를 선택”하는 PASS 조건에서는 제외한다.

M4 로직 자체는 동일한 Stage-4 방어력에서 secondary signal이 실제로 다른 synthetic 단위 테스트로 검증한다.
향후 실제 로그에서 구분 가능한 M4 사례가 나오면 fixture를 추가한다.

## 8. 고정 설정 후보

```python
V7_FINAL = {
    **V5_FINAL,
    "own_vcf_max_fours": 10,
    "own_vcf_node_limit": 5000,
    "safety_vcf_max_fours": 10,
    "safety_vcf_node_limit": 4000,
    "safety_total_node_limit": 16000,
    "self_forbidden_min_white": 3,
}
```

현재는 이름만 `V7_FINAL`인 **동결 후보 설정**이다. 완료 게이트 통과 전에는 frozen hash에 넣지 않는다.

## 9. 진단 필드

- `v7_own_vcf_found`, `v7_own_vcf_length`, `v7_own_vcf_nodes`
- `v7_safety_checked`, `v7_safety_removed`, `v7_safety_augmented`, `v7_safety_fallback`
- `v7_safety_nodes`, `v7_safety_precheck_skipped`, `v7_safety_budget_exhausted`
- `v7_self_forbidden_penalized`
- `v7_stage4_tiebreak_applied`
- `v7_module_seconds`

진단 객체는 호출자 소유이며 전역 상태를 사용하지 않는다.

## 10. 현재 fixture와 테스트 범위

`tests/fixtures/mcts_v7_positions.json`에는 제공된 10개 실제 로그 국면을 그대로 저장한다.

- own_vcf 4개
- opponent_vcf_safety 4개
- self_forbidden 1개
- stage4_tiebreak 분석 사례 1개

own VCF 4개는 expected VCF first-move set을 만족하고 M0가 Game 상태를 복원해야 한다.
safety 4개는 V7 선택 후 상대 VCF가 없어야 한다. M3는 기록된 bad move를 실제 root ordering에서 감점한다.

`tests/fixtures/mcts_v7_vcf_regression.json`에는 seed 44 로그에서 추출한 VCF 회귀 124국면을 추가한다.

- `vcf_streak_start` 27개: 현재 차례의 VCF가 검출되어야 한다.
- `missed_own_vcf` 79개: V6가 놓친 own VCF를 5,000-node M1 상한 안에서 모두 검출해야 한다.
- `losing_move_allows_vcf` 18개: 기록된 패배수를 둔 뒤 상대 VCF를 4,000-node safety 상한 안에서 검출해야 한다.
- 각 항목의 first move/fours/nodes 참조값은 50,000-node 분석 탐색 결과와 일치해야 한다.

이 회귀 집합의 최대 참조 node는 3,603이므로 M1 5,000 / M2 per-probe 4,000 상한은 관찰된 124국면을 모두 포괄한다.

### 10.1 M2 재감사 반영

seed 44 재감사에서 기존 M2는 own-four 후보를 상대의 강제 차단 전 상태에서 평가해 단순 4를 과도하게 안전하다고 판정하는 문제가 확인됐다. 수정본은 강제 차단 뒤를 평가하고, 현재 VCF 1회 사전검사 + 흑 금수 변화 precheck + 착수당 총 node budget으로 의미와 최악 VCF 탐색량을 함께 제한한다.

수정 전 표본에서는 안전 fixture 12/12, 평균 착수 시간 V7 1.13초 대 V6 1.01초였지만, 최악 V7 착수는 12.5초(모듈 10.2초)까지 관찰됐다. 이 값은 수정 전 구현의 참고치이며, 동결 전에 같은 장비에서 평균/최악 시간을 다시 측정해야 한다.

## 11. V7_FINAL 동결 게이트

### 기능

- 기존 전체 테스트와 CI 통과
- V6 frozen hash/behavior fingerprint 불변
- 제공된 검증 가능한 M0~M3 fixture 통과
- M4 generic tie-break 단위 테스트 통과
- 불법 착수 0, 동일 seed 결정성 확인

### 비용

같은 그램에서 V6와 V7을 측정한다.

- 평균 착수 시간: V6 기준의 2배 이내를 목표 상한으로 기록
- 최악 착수 시간도 별도 기록
- GitHub Actions runner 시간은 이 비용 게이트의 대체물이 아니다.

### 기력

고정 opening pair와 색 교환을 사용한다.

- V7 vs V6: 최소 100판
- V7 vs V5 FINAL: 최소 100판
- 전체 및 흑/백별 W/L/D와 `score=(W+0.5D)/N`
- 평균/최악 착수 시간
- M1~M4 진단 발동 횟수

특정 목표 승률에 맞춰 V7을 튜닝하지 않는다.

## 12. 동결 절차

완료 게이트를 모두 통과한 뒤에만:

1. 최종 설정을 문서에 확정
2. V7 신규 구현 파일 해시를 frozen manifest에 추가
3. 저비용 V7 behavior fingerprint를 CI에 추가
4. V7-vs-V6/V5 결과와 그램 비용을 이 문서에 기록
5. 이후 AlphaZero 평가에서 이 frozen V7만 사용

그 전까지 PR은 candidate 상태로 유지한다.
