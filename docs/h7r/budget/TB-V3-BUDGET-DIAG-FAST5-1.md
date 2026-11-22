# TB-V3-BUDGET-DIAG-FAST5-1

TB-V3-BUDGET-DIAG-1에서 남은 `8411-p8-white/11`의 `[8,9]`를 깊이 2까지 판정한다. 입력은 `863f808`의 E1 manifest, 진단 계약, deepen 기록이다. 기존 진단에서 사용한 B0–B3의 노드·호출 한도를 적용한다.

1. 각 판정 프로세스에서 TB-V3 fast5를 설치하고 판정기가 실제 네이티브 VCF에 연결됐는지 확인한다. 이 검사는 기보와 착수 좌표로 실행하므로 GPU와 모델 파일이 필요 없다.
2. B0를 최대 5분 동안 실행한다. `UNKNOWN`, 전체 노드 10,000,000, VCF 호출 95,361, 잘린 호출 14를 포함해 기존 replay의 B0 결과와 일치해야 다음 단계로 간다.
3. B1–B3은 같은 빌드에서 각각 별도 프로세스로 실행한다. B1은 B0 실측 벽시계의 2배, B2와 B3은 20배를 적용한다. B2와 B3의 전체 노드 한도가 B0의 10배인 점을 반영한 실행 시간 배정이다. 각 한도는 최소 60초, 최대 1,800초다. B1–B3을 시작하기 전에 실제 한도를 `deepen-plan.json`에 쓴다.
4. 판정기가 반환한 `UNKNOWN`과 시간 초과 `TIMEOUT`을 구분한다. 확정 판정이 서로 충돌하면 `CONFLICT`로 표시한다. `VCT2_CLEAR`의 확인 범위는 깊이 2다.

## 실행

```bash
export PYTHONPATH="$PWD/src"
python3 -m rt30.native.build --if-stale
python3 -m rt30.vcfcore.build --if-stale
python3 scripts/h7r_budget_diag_fast5.py calibrate --output-dir docs/h7r/budget/TB-V3-BUDGET-DIAG-FAST5-1
python3 scripts/h7r_budget_diag_fast5.py deepen --output-dir docs/h7r/budget/TB-V3-BUDGET-DIAG-FAST5-1
```

## 결과

Windows의 Ubuntu 24.04 WSL에서 fast5의 네이티브 모듈을 빌드하고 CPU로 실행했다. 각 결과에 같은 빌드 식별값이 기록돼 있다. B0는 기존 replay의 상태와 사용량에 모두 일치했다. 이 실측으로 B1은 60초, B2와 B3은 각각 347초를 한도로 고정했다.

| 검사 | 깊이 2 판정 | 노드 | VCF 호출 | 잘린 호출 | 벽시계 |
|---|---|---:|---:|---:|---:|
| B0 | UNKNOWN | 10,000,000 | 95,361 | 14 | 17.34초 |
| B1 | UNKNOWN | 10,000,000 | 91,722 | 3 | 16.70초 |
| B2 | VCT2_CLEAR | 22,635,670 | 264,696 | 37 | 54.23초 |
| B3 | VCT2_CLEAR | 24,328,026 | 264,696 | 3 | 65.62초 |

개별 VCF 호출당 노드 한도를 20,000으로 유지한 B2에서 판정이 나왔다. 착수당 노드·호출 한도를 늘리는 것으로 충분했다. B2의 사용량은 B0의 노드 한도 1,000만과 호출 한도 10만을 모두 넘었다. 두 한도를 따로 늘린 실행은 이번 범위에 없었다.

첫 B3 실행은 WSL 세션이 결과 없이 종료됐다. B1·B2 기록을 사용해 같은 한도로 B3만 다시 실행했고, 두 확장 검사 모두 `VCT2_CLEAR`를 반환했다. 확인 범위는 이 한 착수의 깊이 2 강제패 여부다.
