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
