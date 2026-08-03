# 일일 수동 트레이딩 신호 사용법

연구와 검증은 필요할 때만 실행한다.

```powershell
$env:PYTHONPATH = "src"
python scripts/cross_sectional/run_research.py
```

EPS TTM·DCF·P/E·EBITDA 성장 중심 V3 연구는 다음처럼 별도 실행한다.

```powershell
$env:PYTHONPATH = "src"
python scripts/cross_sectional/run_research.py `
  --config config/cross_sectional/research_ttm_financial_momentum.json
```

선택된 파라미터를 그대로 사용해 최신 신호만 다시 만드는 명령은 별도다.

```powershell
$env:PYTHONPATH = "src"
python scripts/cross_sectional/generate_daily_signals.py `
  --strategy "<selected_strategy.json 경로>"
```

`--config`를 생략하면 선택된 전략 파일 안에 동결된 설정을 그대로 재사용한다.
따라서 V3 전략에 실수로 V1 재무 정의를 적용하지 않는다.

결과의 핵심 열은 다음과 같다.

- `DailySignal`: 매일 확인할 BUY·HOLD·REDUCE_WATCH·BUY_WATCH·AVOID
- `TradeAction`: 이번 주 다음 시가에 실행할 BUY·SELL·HOLD; 그 외 날은 NONE
- `TargetWeight`: 모델 포트폴리오의 목표 비중
- `Rank`, `AlphaScore`: 전체 후보 안에서의 상대 우선순위
- `FinancialStale`: 재무제표 요인이 180일 이상 오래되어 중립 처리됐는지 여부
- `EpsTtmGrowthYoY`, `EpsTtmGrowthAcceleration`: EPS TTM 성장과 가속
- `DcfPriceGrowthYoY`, `DcfUpside`: DCF 적정가치 성장과 현재가 대비 여유
- `EbitdaTtmGrowthYoY`, `GrowthAdjustedPe`,
  `GrowthAdjustedEvEbitda`: EBITDA 성장과 성장 대비 밸류에이션 효율

주문 전에는 최신 가격 데이터 날짜와 `IsRebalanceSignal`을 함께 확인한다.

2020–2024년 학습, 2025년 파라미터 선택, 2026년 최종 홀드아웃을 사용하는
V4 연구 명령은 다음과 같다.

```powershell
$env:PYTHONPATH = "src"
python scripts/cross_sectional/run_research.py `
  --config config/cross_sectional/research_ttm_financial_momentum_v4.json
```

손실보호 V5 연구 명령은 다음과 같다.

```powershell
$env:PYTHONPATH = "src"
python scripts/cross_sectional/run_research.py `
  --config config/cross_sectional/research_loss_protected_v5.json
```

동결된 최신 V5 실행을 사용해 주가·완전 진입/청산·신호 강도 HTML/PDF와
상세 CSV를 만드는 명령은 별도다. 이 명령은 재학습하지 않는다.

```powershell
$env:PYTHONPATH = "src"
python scripts/cross_sectional/generate_trade_report.py
```

특정 실행을 고정하려면 해당 실행의 `selected_strategy.json`을 지정한다.

```powershell
$env:PYTHONPATH = "src"
python scripts/cross_sectional/generate_trade_report.py `
  --strategy "<selected_strategy.json 경로>"
```

V6 규칙은 V5를 덮어쓰지 않고 네 개 설정으로 각각 실행한다.

```powershell
$env:PYTHONPATH = "src"
python scripts/cross_sectional/run_research.py `
  --config config/cross_sectional/research_v6_a_winner_retention.json
python scripts/cross_sectional/run_research.py `
  --config config/cross_sectional/research_v6_b_replacement_hurdle.json
python scripts/cross_sectional/run_research.py `
  --config config/cross_sectional/research_v6_c_overheat_guard.json
python scripts/cross_sectional/run_research.py `
  --config config/cross_sectional/research_v6_d_risk_overlay.json
```

V6 추가 진단 열은 다음과 같다.

- `ProfitExitStreak`: 수익 순환매 조건이 연속으로 확인된 횟수
- `BestReplacementAlphaScore`, `ReplacementScoreAdvantage`: 기존 종목과
  최상위 신규 후보의 점수 차이
- `EntryBlocked`, `EntryBlockReason`: 과열 신규 진입 차단 여부와 이유
- `PeakReferenceReturn`, `TrailingDrawdown`: 추적 손절의 최고수익과
  신호일 고점 대비 낙폭

V6는 2025–2026년 사례를 본 뒤 만든 사후진단이다. 실행 결과가 좋아도
`VALIDATED`로 해석하지 않고 2026-07-24 이후 데이터로 전진검증한다.

팩터 가중치와 기본 진입필터를 V5 값으로 고정한 순수 규칙 비교 및
인터랙티브 리밸런싱 리포트는 다음 명령으로 생성한다.

```powershell
$env:PYTHONPATH = "src"
python scripts/cross_sectional/compare_v6_variants.py
```

이 비교는 V5, 네 개 단일 규칙, 결합형을 같은 데이터와 가중치로 실행해
`v6_variant_summary.csv`, 자산곡선, 포지션 원장, 리밸런싱 이벤트와 HTML을
`Results/Cross_Sectional/v6_comparison` 아래에 원자적으로 기록한다.

## 2019–2026 S&P 500 구성종목과 데이터 백필

연도별 S&P 500 멤버십 요약, 연중 변경일 멤버십, 전체 기간 종목 합집합을
생성한다. 이 단계는 가격이나 재무 데이터를 받지 않는다.

```powershell
$env:PYTHONPATH = "src"
python scripts/cross_sectional/build_sp500_universe.py
```

결과의 `sp500_membership.csv`는 매년 1월 1일 요약이고,
`sp500_membership_changes.csv`는 연중 변경일까지 반영한 백테스트
멤버십이다. `sp500_union.csv`는 가격·재무 수집 큐다. 합집합을 고정 과거
유니버스로 사용하면 미래 구성종목을 미리 아는 생존·선택 편향이 생긴다.

가격만 먼저 백필하려면 다음 명령을 실행한다.

```powershell
$env:PYTHONPATH = "src"
python scripts/cross_sectional/backfill_sp500_universe.py --stage price
```

재무만 백필하려면 별도로 실행한다.

```powershell
$env:PYTHONPATH = "src"
python scripts/cross_sectional/backfill_sp500_universe.py --stage financial
```

둘 다 기존 검증을 통과한 파일은 건너뛴다. 장시간 실행은 `--limit` 또는
반복 가능한 `--ticker` 옵션으로 나눌 수 있다. 출력은 sibling OneDrive의
`Results/Cross_Sectional/sp500_backfill_runs`에 원자적으로 체크포인트된다.
상장폐지·인수 종목의 가격이나 현재 웹 재무 페이지가 없으면 실패 상태를
보존하며 다른 종목으로 자동 대체하지 않는다.
`sp500_union.csv`의 `CrawlBlockReason`이 채워진 과거 재사용 티커는 잘못된
현재 증권을 받지 않도록 자동 백필 대상에서 제외된다.
