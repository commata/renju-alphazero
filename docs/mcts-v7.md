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
- 후보별 VCF는 4,000 node, root 사전 검사는 8,000 node까지 허용하되 둘 다 **한 착수당 총 8,000-node safety budget을 공유**한다. 알려진 5,498-node "VCF 없음" 국면은 사전 검사에서 끝낼 수 있고, M2 최악 탐색량은 16,000에서 8,000으로 절반이 된다. (착수당 전체 최악값은 §8.1 참고.)
- 상대 VCF가 **확인된 후보만 제거**한다. node 상한으로 결론을 못 낸 후보는 `inconclusive`로 남기되 검증된 안전수 뒤로 보낸다. 따라서 탐색 공간 크기 자체가 제거 사유가 되지 않는다.
- 결과는 **검증된 안전 tier → inconclusive tier** 두 단계이며, 각 tier 안에서는 V6 순서를 유지한다.
- 검증된 안전수가 없고 **예산이 이미 소진됐다면 보강을 건너뛰고** inconclusive 원래 후보만 사용한다. 보강 후보에 대해 새로 알 수 있는 것이 없기 때문이다.
- 예산이 남아 있으면 현재 상대 VCF 공격/완성점과 내 4 생성점을 보강 후보로 검사한다. **안전이 검증된 보강 후보만** 앞에 추가하고, 전체 후보 수는 V6 후보 수(최대 `candidate_limit`)를 넘기지 않는다. 검증되지 않은 보강 후보는 원래 후보가 모두 VCF 확인으로 제거된 경우에만 사용한다.
- 모든 후보가 VCF 확인으로 제거되고 보강도 없으면 V6 후보를 그대로 사용하고 fallback 진단을 남긴다.
- M1이 이미 수를 결정했다면 M2는 실행되지 않는다.
- 기존 V5/V6 Stage 5 강제수는 V7 M2를 거치지 않는 현재 흐름을 **Stage 6.5 범위 결정으로 유지**한다. V7은 Stage 1~5 강제정책을 재정의하지 않고, 이 예외는 동결 문서에 명시한다.

## 6. M3 — 흑 자기 금수점 감점

흑 후보를 둔 뒤 백 압력이 있는 window에서 **새로 생기는 흑 금수점**을 검사한다.
해당 후보를 제거하지 않고 root 후보 순서의 뒤로 보내며, 동일 그룹 안에서는 V6 순서를 유지한다.
M3는 **M2 tier 안에서만** 순서를 바꾼다. 금수 감점은 휴리스틱이고 M2 tier는 VCF 패배 근거이므로, 감점된 검증 안전수도 inconclusive 후보보다 앞에 남는다.

실제 seed777 31수 fixture에서는 end-to-end V7이 M1/M2에서 먼저 나쁜 수를 피할 수 있기 때문에,
M3 자체는 기록된 V6 root 후보를 입력해 bad move가 penalized suffix로 이동하는지 별도로 검증한다.

## 7. M4 — Stage 4 동률 규칙과 fixture 재감사

M4는 V5 Stage 4의 **남은 unstoppable four 최소화**를 최우선으로 그대로 유지한다.
그 값이 같은 방어 후보끼리만 다음을 추가한다.

1. 착수 후 상대 `_double_threat_moves` 수가 적은 후보
2. 착수 후 상대 VCF 판정: **검증된 안전 < inconclusive < 확인된 VCF** 순
3. 그 외에는 기존 V6/V5 root key

VCF 판정은 세 번째 기준이므로 **(remaining, double-threat)가 최소인 동률 그룹만** 검사하고, 그룹이 한 후보뿐이면 VCF 검사를 하지 않는다.
동률 그룹은 root key 순으로 검사하며 **처음 검증된 안전 후보에서 멈춘다.** 뒤의 후보는 key가 더 커서 결과와 무관하게 순위를 이길 수 없다.
VCF 판정은 M2와 같은 `_candidate_allows_vcf`를 사용한다. 즉 4를 만드는 방어수는 상대의 강제 차단까지 둔 뒤 평가한다. 착수 직후 바로 VCF를 찾으면 우리 4가 상대의 모든 4를 무효로 만들어 거의 항상 "VCF 없음"이 나오기 때문이다(fixture 124국면에서 4 생성수 688개 중 182개가 이 오판).
검사는 M2와 같은 크기의 착수당 8,000-node budget으로 제한한다. Stage 4와 M2는 한 착수에서 동시에 실행되지 않는다.

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
    "safety_precheck_node_limit": 8000,
    "safety_total_node_limit": 8000,
    "self_forbidden_min_white": 3,
}
```

현재는 이름만 `V7_FINAL`인 **동결 후보 설정**이다. 완료 게이트 통과 전에는 frozen hash에 넣지 않는다.

### 8.1 착수당 최악 VCF 탐색량

M1(자기 VCF)은 자체 상한 5,000 node로 제한되며 M2/M4 safety budget에 포함되지 않는다. M1이 VCF를 찾으면 바로 착수하므로 M2/M4는 M1이 실패한 뒤에만 실행되고, M2와 M4는 한 착수에서 동시에 실행되지 않는다. 따라서 한 착수의 VCF 탐색 최악값은 **M1 5,000 + safety 8,000 = 13,000 node**이다(측정값 약 0.62ms/node, 약 8초). 관찰된 124국면에서 M1 미발견 탐색은 대부분 수십 node 이하였다. 착수별 합계는 `v7_own_vcf_nodes`(미발견 시에도 기록) + `v7_safety_nodes` + `v7_stage4_vcf_nodes`로 확인한다.

## 9. 진단 필드

- `v7_own_vcf_found`, `v7_own_vcf_length`, `v7_own_vcf_nodes`
- `v7_safety_checked`, `v7_safety_removed`, `v7_safety_augmented`, `v7_safety_fallback`
- `v7_safety_nodes`, `v7_safety_precheck_nodes`, `v7_safety_precheck_skipped`
- `v7_safety_inconclusive`, `v7_safety_budget_exhausted`
- `v7_self_forbidden_penalized`
- `v7_stage4_tiebreak_applied`, `v7_stage4_vcf_nodes`
- `v7_stage4_vcf_inconclusive`, `v7_stage4_vcf_budget_exhausted`
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

첫 M2 수정본(`d2bac56`)을 동일 122-position 표본에서 재측정한 결과 평균 비율은 약 1.13배로 목표 안이었지만 최악값은 개선되지 않았다. 단독 재측정에서 seed44 game 3 ply 109는 V6 1.8초 대비 V7 11.2초(모듈 9.5초), game 44 ply 88은 V6 2.6초 대비 V7 11.5초(모듈 9.0초), game 94 ply 67 Stage 4는 V6 0.01초 대비 V7 3.8초(모듈 3.8초)였다. 이 재측정은 프로세스 3개 동시 실행이라 절대시간보다 비율/급증 지점 진단에 사용한다.

재감사에서 원인은 두 가지로 분리됐다. (1) bounded probe의 `inconclusive`를 VCF 확인과 동일하게 제거해 총예산 소진이 후보 삭제로 이어진 점, (2) root precheck가 후보별 4,000-node 상한을 공유해 5,498-node "VCF 없음" 국면을 끝내지 못한 점이다. 후속 수정은 inconclusive를 제거하지 않고 뒤로 보내고, precheck 8,000 / 총 safety 8,000으로 제한하며, M4도 같은 총예산을 사용한다. 이 후속값으로 동일 장비 timing을 다시 측정해야 한다.

### 10.2 `9a07d13` 재감사 반영

`9a07d13` 재검토에서 추가로 확인된 문제와 수정:

- M4가 동률과 무관한 후보까지, 그리고 첫 안전 후보 이후까지 모두 검사해 예산을 소진했다 → 최소 동률 그룹만, 첫 검증 안전 후보에서 중단.
- M4가 4를 만드는 방어수를 강제 차단 전에 평가했다(M2 1차 수정과 같은 버그) → `_candidate_allows_vcf` 재사용.
- 예산 소진 후에도 검증되지 않은 보강 후보가 추가되어 후보가 20개에서 30개로 늘었다 → 소진 시 보강 생략, 검증된 보강만 추가, 후보 수 상한 유지.
- M3가 M2 tier를 넘어 순서를 바꿀 수 있었다 → tier 안에서만 감점.
- M1 node를 VCF 미발견 시에도 `v7_own_vcf_nodes`에 기록한다.

seed 44 V6 vs V5 로그의 해당 대국을 결정적으로 재생해 단독 실행으로 측정했다(44번 대국은 재생이 원 로그와 66수부터 달라져 같은 유형의 국면이다).

| 국면 | V6 | `d2bac56` (모듈) | `9a07d13` (모듈) | 이번 수정 (모듈) |
|---|---|---|---|---|
| game 3 ply 109 | 1.95초 | 11.8초 (10.1초) | 7.0초 (5.2초) | 7.1초 (5.3초) |
| game 44 ply 88 | 3.1초 | 7.4초 (4.3초) | 5.1초 (2.2초), 후보 30개 | 5.2초 (2.3초), 후보 17개 |
| game 94 ply 67 (Stage 4) | 0.01초 | 5.2초 (5.2초) | 5.1초 (5.1초) | **0.27초 (0.26초)** |

124국면 회귀 fixture 전체에서 최대 모듈 시간은 `d2bac56` 9.2초 → `9a07d13` 5.0초 → 이번 수정 4.5초이고, fixture 124국면의 V7 선택 수는 `9a07d13`과 동일하다. 122-position 평균 비율 표본과 V7 vs V6 / V7 vs V5 비교 대국은 동결 전에 다시 실행해야 한다.

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
