from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


SCORE_COLUMNS = [
    "Ticker",
    "Company",
    "Close",
    "MarketRiskOn",
    "BaseV7Score",
    "Qualified",
    "Rank",
    "FilingScore",
    "AlphaScore",
    "FilingCriticalFlag",
    "FilingCoverageCount",
    "TargetWeight",
    "ModelSelected",
    "SignalReferenceReturn",
    "HoldingRebalances",
    "ExitReason",
    "TradeAction",
    "MomentumFactor",
    "TrendFactor",
    "GrowthFactor",
    "QualityFactor",
    "RiskControlFactor",
]

ROW_COLUMNS = [
    "Ticker",
    "Company",
    "Bucket",
    "Member",
    "Ready",
    "Close",
    "Alpha",
    "Base",
    "Filing",
    "Momentum",
    "Trend",
    "Growth",
    "Quality",
    "Risk",
    "Qualified",
    "Rank",
    "Weight",
    "Selected",
    "SinceEntry",
    "HoldingRebalances",
    "Action",
    "ExitReason",
    "Critical",
    "FilingCoverage",
]


def build_dashboard_payload(
    signal_history: pd.DataFrame,
    equity: pd.DataFrame,
    membership: pd.DataFrame,
    readiness: pd.DataFrame,
    manifest: dict[str, Any],
    *,
    strategy_series: str = "V7_SEC_COMBINED_OPTIMIZED",
    benchmark_series: str = "SPY_BUY_HOLD",
) -> dict[str, Any]:
    signals = signal_history.copy()
    signals["Date"] = pd.to_datetime(signals["Date"], errors="coerce")
    signals = signals.dropna(subset=["Date", "Ticker"])
    dates = sorted(pd.Timestamp(item) for item in signals["Date"].unique())
    if not dates:
        raise ValueError("Signal history contains no usable dates")

    ready = readiness.copy()
    ready["Ticker"] = ready["Ticker"].astype(str)
    ready_map = ready.set_index("Ticker").to_dict("index")
    tickers = sorted(
        set(ready["Ticker"].astype(str))
        | set(signals["Ticker"].astype(str))
        | set(membership["DataSymbol"].dropna().astype(str))
    )

    member = membership.copy()
    member["AsOfDate"] = pd.to_datetime(member["AsOfDate"], errors="coerce")
    member = member.dropna(subset=["AsOfDate", "DataSymbol"])
    snapshots = sorted(pd.Timestamp(item) for item in member["AsOfDate"].unique())
    snapshot_maps: dict[pd.Timestamp, dict[str, dict[str, Any]]] = {}
    for snapshot in snapshots:
        rows = member.loc[member["AsOfDate"].eq(snapshot)]
        snapshot_maps[snapshot] = {
            str(row["DataSymbol"]): row.to_dict() for _, row in rows.iterrows()
        }

    signal_lookup = {
        (pd.Timestamp(row["Date"]), str(row["Ticker"])): row
        for _, row in signals.iterrows()
    }
    score_dates: list[dict[str, Any]] = []
    snapshot_index = 0
    active_snapshot: pd.Timestamp | None = None
    for signal_date in dates:
        while snapshot_index < len(snapshots) and snapshots[snapshot_index] <= signal_date:
            active_snapshot = snapshots[snapshot_index]
            snapshot_index += 1
        active_members = snapshot_maps.get(active_snapshot, {})
        rows: list[list[Any]] = []
        market_risk_on = False
        for ticker in tickers:
            signal = signal_lookup.get((signal_date, ticker))
            membership_row = active_members.get(ticker)
            readiness_row = ready_map.get(ticker, {})
            ready_status = str(readiness_row.get("Status", "UNKNOWN"))
            company = (
                _value(signal, "Company")
                or (membership_row or {}).get("Company")
                or readiness_row.get("Company")
                or ticker
            )
            is_member = bool(membership_row and membership_row.get("Selected", True))
            bucket = (
                str(membership_row.get("MembershipBucket"))
                if membership_row
                else "OUTSIDE"
            )
            if signal is not None:
                market_risk_on = bool(_value(signal, "MarketRiskOn"))
            values = {
                "Ticker": ticker,
                "Company": company,
                "Bucket": bucket,
                "Member": is_member,
                "Ready": ready_status,
                "Close": _number(signal, "Close", 2),
                "Alpha": _number(signal, "AlphaScore", 5),
                "Base": _number(signal, "BaseV7Score", 5),
                "Filing": _number(signal, "FilingScore", 5),
                "Momentum": _number(signal, "MomentumFactor", 5),
                "Trend": _number(signal, "TrendFactor", 5),
                "Growth": _number(signal, "GrowthFactor", 5),
                "Quality": _number(signal, "QualityFactor", 5),
                "Risk": _number(signal, "RiskControlFactor", 5),
                "Qualified": bool(_value(signal, "Qualified")),
                "Rank": _number(signal, "Rank", 0),
                "Weight": _number(signal, "TargetWeight", 6) or 0.0,
                "Selected": bool(_value(signal, "ModelSelected")),
                "SinceEntry": _number(signal, "SignalReferenceReturn", 6),
                "HoldingRebalances": _number(signal, "HoldingRebalances", 0),
                "Action": _value(signal, "TradeAction") or "NO_SCORE",
                "ExitReason": _value(signal, "ExitReason"),
                "Critical": bool(_value(signal, "FilingCriticalFlag")),
                "FilingCoverage": _number(signal, "FilingCoverageCount", 0),
            }
            rows.append([_json_value(values[column]) for column in ROW_COLUMNS])
        score_dates.append(
            {
                "date": signal_date.strftime("%Y-%m-%d"),
                "riskOn": market_risk_on,
                "rows": rows,
            }
        )

    equity_frame = equity.copy()
    equity_frame["Date"] = pd.to_datetime(equity_frame["Date"], errors="coerce")
    equity_frame["Equity"] = pd.to_numeric(equity_frame["Equity"], errors="coerce")
    pivot = (
        equity_frame.loc[
            equity_frame["Series"].isin([strategy_series, benchmark_series])
        ]
        .dropna(subset=["Date", "Equity"])
        .pivot_table(index="Date", columns="Series", values="Equity", aggfunc="last")
        .sort_index()
        .ffill()
    )
    if strategy_series not in pivot or benchmark_series not in pivot:
        raise ValueError("Equity file is missing the strategy or benchmark series")
    curve = [
        [
            index.strftime("%Y-%m-%d"),
            round(float(row[strategy_series]), 2),
            round(float(row[benchmark_series]), 2),
        ]
        for index, row in pivot.iterrows()
        if pd.notna(row[strategy_series]) and pd.notna(row[benchmark_series])
    ]
    policy = dict(manifest.get("policy", {}))
    return {
        "generatedAt": manifest.get("generated_at"),
        "modelStatus": manifest.get("model_status"),
        "warning": manifest.get("warning"),
        "strategySeries": strategy_series,
        "benchmarkSeries": benchmark_series,
        "initialCapital": round(float(curve[0][1]), 2),
        "filingWeight": float(policy.get("filing_weight", 0.0)),
        "topK": int(policy.get("top_k", 3)),
        "rowColumns": ROW_COLUMNS,
        "scoreDates": score_dates,
        "curve": curve,
    }


def render_dashboard_html(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    encoded = encoded.replace("<", "\\u003c")
    template = _HTML.replace(r'\"', '"')
    return template.replace("__DASHBOARD_DATA__", encoded)


def write_dashboard(path: Path, html: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        suffix=".tmp", prefix=f"{path.stem}_", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(html, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _value(row: pd.Series | None, column: str) -> Any:
    if row is None or column not in row.index:
        return None
    value = row[column]
    if pd.isna(value):
        return None
    return value


def _number(row: pd.Series | None, column: str, digits: int) -> float | int | None:
    value = _value(row, column)
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    rounded = round(numeric, digits)
    return int(rounded) if digits == 0 else rounded


def _json_value(value: Any) -> Any:
    if isinstance(value, (bool, str, int)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return str(value)


_HTML = r'''<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Top 10 + Watchlist 포트폴리오 대시보드</title>
<style>
:root{--bg:#07111f;--panel:#0d1a2b;--panel2:#111f32;--line:#23344b;--text:#edf4ff;--muted:#91a4bd;--blue:#58a6ff;--cyan:#2dd4bf;--gold:#fbbf24;--red:#fb7185;--green:#34d399;--cash:#64748b;--shadow:0 18px 48px rgba(0,0,0,.26)}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 10% 0,#102641 0,transparent 31%),var(--bg);color:var(--text);font-family:Inter,"Segoe UI",Arial,sans-serif;font-size:14px}button,input{font:inherit}button{cursor:pointer}.shell{max-width:1480px;margin:0 auto;padding:24px}.header{display:flex;justify-content:space-between;gap:20px;align-items:flex-start;margin-bottom:18px}.eyebrow{color:var(--cyan);font-size:12px;letter-spacing:.13em;text-transform:uppercase}.title{font-size:28px;font-weight:600;margin:6px 0 5px}.subtitle{color:var(--muted);max-width:800px}.status{padding:8px 12px;border:1px solid #705c22;background:#2a2411;color:#ffe69a;border-radius:999px;font-size:12px;white-space:nowrap}.controls{display:grid;grid-template-columns:auto 1fr auto;gap:12px;align-items:center;padding:15px 18px;background:rgba(13,26,43,.82);border:1px solid var(--line);border-radius:16px;box-shadow:var(--shadow);position:sticky;top:8px;z-index:5;backdrop-filter:blur(12px)}.nav{display:flex;gap:6px}.btn{border:1px solid var(--line);background:var(--panel2);color:var(--text);padding:8px 12px;border-radius:9px}.btn:hover{border-color:var(--blue)}.date-label{font-weight:600;font-variant-numeric:tabular-nums;min-width:105px;text-align:right}.slider{width:100%;accent-color:var(--blue)}.stats{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px;margin:14px 0}.card{background:linear-gradient(145deg,rgba(17,31,50,.96),rgba(10,22,38,.96));border:1px solid var(--line);border-radius:15px;padding:15px;box-shadow:var(--shadow)}.stat-label{color:var(--muted);font-size:12px;margin-bottom:7px}.stat-value{font-size:22px;font-weight:600;font-variant-numeric:tabular-nums}.stat-note{color:var(--muted);font-size:11px;margin-top:4px}.good{color:var(--green)!important}.bad{color:var(--red)!important}.main-grid{display:grid;grid-template-columns:minmax(0,1.7fr) minmax(330px,.8fr);gap:14px}.section{background:rgba(13,26,43,.9);border:1px solid var(--line);border-radius:16px;padding:17px;box-shadow:var(--shadow);min-width:0}.section-head{display:flex;justify-content:space-between;gap:12px;align-items:center;margin-bottom:12px}.section-title{font-size:16px;font-weight:600}.legend{display:flex;gap:15px;color:var(--muted);font-size:12px}.dot{width:9px;height:9px;border-radius:50%;display:inline-block;margin-right:5px}.chart-wrap{height:330px;position:relative}.chart-wrap svg{width:100%;height:100%;display:block}.chart-tooltip{display:none;position:absolute;background:#020814;border:1px solid var(--line);padding:8px 10px;border-radius:8px;pointer-events:none;font-size:12px;box-shadow:var(--shadow);white-space:nowrap;z-index:4}.allocation{display:grid;grid-template-columns:210px 1fr;gap:16px;align-items:center}.donut{width:210px;height:210px}.donut-center{font-size:13px;fill:var(--muted)}.donut-value{font-size:18px;font-weight:600;fill:var(--text)}.alloc-list{display:grid;gap:9px}.alloc-row{display:grid;grid-template-columns:10px minmax(50px,1fr) auto;gap:8px;align-items:center}.swatch{width:10px;height:10px;border-radius:3px}.alloc-meta{text-align:right;font-variant-numeric:tabular-nums}.alloc-sub{color:var(--muted);font-size:11px}.events{margin-top:14px}.event-row{display:flex;gap:8px;flex-wrap:wrap}.event{padding:7px 9px;border-radius:9px;background:var(--panel2);border:1px solid var(--line);font-size:12px}.event strong{margin-right:5px}.tables{display:grid;grid-template-columns:minmax(0,.83fr) minmax(0,1.4fr);gap:14px;margin-top:14px}.table-wrap{overflow:auto;max-height:610px}.table-wrap.score{max-height:760px}table{width:100%;border-collapse:collapse;font-size:12px}th{position:sticky;top:0;background:#0d1a2b;color:var(--muted);font-weight:500;text-align:left;padding:10px 8px;border-bottom:1px solid var(--line);white-space:nowrap;z-index:2}td{padding:9px 8px;border-bottom:1px solid rgba(35,52,75,.7);vertical-align:middle}tbody tr:hover{background:rgba(88,166,255,.06)}.num{text-align:right;font-variant-numeric:tabular-nums}.ticker{font-weight:600;font-size:13px}.company{color:var(--muted);font-size:11px;max-width:155px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.badge{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:3px 7px;color:var(--muted);font-size:10px;white-space:nowrap}.badge.hold,.badge.buy{border-color:#166a57;color:#75ebca;background:#0d2d27}.badge.sell,.badge.critical{border-color:#7f2940;color:#ff9eb1;background:#32131d}.badge.watch{border-color:#705c22;color:#ffe69a;background:#2a2411}.scorebar{display:flex;align-items:center;gap:7px;min-width:98px}.track{position:relative;height:5px;background:#223248;border-radius:99px;flex:1;overflow:hidden}.fill{height:100%;border-radius:99px}.empty{color:var(--muted)}.foot{display:flex;justify-content:space-between;gap:20px;color:var(--muted);font-size:11px;margin:15px 2px 0}.full-span{grid-column:1/-1}.market-off{color:var(--gold)}
@media(max-width:1050px){.stats{grid-template-columns:repeat(3,1fr)}.main-grid,.tables{grid-template-columns:1fr}.allocation{grid-template-columns:190px 1fr}.donut{width:190px;height:190px}}
@media(max-width:650px){.shell{padding:12px}.header{display:block}.status{display:inline-block;margin-top:10px}.controls{grid-template-columns:1fr auto}.nav{grid-column:1/-1}.stats{grid-template-columns:1fr 1fr}.chart-wrap{height:265px}.allocation{grid-template-columns:1fr}.donut{margin:auto}.foot{display:block}.title{font-size:23px}}
</style>
</head>
<body>
<main class="shell">
  <header class="header">
    <div><div class="eyebrow">Point-in-time portfolio research</div><h1 class="title">Top 10 + 관심종목 포트폴리오</h1><div class="subtitle">날짜를 이동하면 당시 보유 구성, 현금 비중, 종목별 통합 점수와 구성요소, 실제 매매 판단이 함께 바뀝니다.</div></div>
    <div class="status" id="model-status"></div>
  </header>
  <section class="controls" aria-label="분석 날짜 선택">
    <div class="nav"><button class="btn" id="prev" type="button">← 이전 주</button><button class="btn" id="next" type="button">다음 주 →</button></div>
    <input class="slider" id="date-slider" type="range" min="0" step="1" aria-label="분석 날짜">
    <div class="date-label" id="date-label"></div>
  </section>
  <section class="stats">
    <div class="card"><div class="stat-label">전략 총자산</div><div class="stat-value" id="equity-value"></div><div class="stat-note" id="equity-roi"></div></div>
    <div class="card"><div class="stat-label">SPY 보유</div><div class="stat-value" id="spy-value"></div><div class="stat-note" id="spy-roi"></div></div>
    <div class="card"><div class="stat-label">SPY 대비 자산 차이</div><div class="stat-value" id="excess-value"></div><div class="stat-note">같은 $100,000 시작 기준</div></div>
    <div class="card"><div class="stat-label">주식 / 현금</div><div class="stat-value" id="exposure-value"></div><div class="stat-note" id="position-count"></div></div>
    <div class="card"><div class="stat-label">시장 상태</div><div class="stat-value" id="regime-value"></div><div class="stat-note">SPY 50일·200일 추세 게이트</div></div>
  </section>
  <div class="main-grid">
    <section class="section">
      <div class="section-head"><div class="section-title">총자산 변화</div><div class="legend"><span><i class="dot" style="background:var(--blue)"></i>전략</span><span><i class="dot" style="background:var(--cyan)"></i>SPY</span></div></div>
      <div class="chart-wrap" id="chart-wrap"><svg id="equity-chart" viewBox="0 0 900 320" role="img" aria-label="전략과 SPY 총자산 곡선"></svg><div class="chart-tooltip" id="chart-tooltip"></div></div>
    </section>
    <section class="section">
      <div class="section-head"><div class="section-title">포트폴리오 구성</div><span class="badge" id="cash-badge"></span></div>
      <div class="allocation"><svg class="donut" id="donut" viewBox="0 0 220 220" role="img" aria-label="현금 포함 포트폴리오 구성"></svg><div class="alloc-list" id="alloc-list"></div></div>
      <div class="events"><div class="section-head"><div class="section-title">이번 주 판단</div></div><div class="event-row" id="events"></div></div>
    </section>
  </div>
  <div class="tables">
    <section class="section">
      <div class="section-head"><div class="section-title">보유 종목</div><span class="badge">매수 후 수익률 포함</span></div>
      <div class="table-wrap"><table><thead><tr><th>종목</th><th class="num">금액</th><th class="num">비중</th><th class="num">통합점수</th><th class="num">매수 후</th><th>판단</th></tr></thead><tbody id="holdings-body"></tbody></table></div>
    </section>
    <section class="section">
      <div class="section-head"><div><div class="section-title">전체 종목 점수</div><div class="stat-note">통합점수 = V7 점수 × (1−공시비중) + 공시점수 × 공시비중</div></div><span class="badge" id="score-count"></span></div>
      <div class="table-wrap score"><table><thead><tr><th>종목</th><th>구분</th><th class="num">순위</th><th>통합점수</th><th class="num">V7</th><th class="num">공시</th><th class="num">모멘텀</th><th class="num">추세</th><th class="num">성장</th><th class="num">품질</th><th class="num">위험</th><th class="num">비중</th><th>판단</th></tr></thead><tbody id="scores-body"></tbody></table></div>
    </section>
  </div>
  <footer class="foot"><span id="formula-note"></span><span id="generated-note"></span></footer>
</main>
<script id="dashboard-data" type="application/json">__DASHBOARD_DATA__</script>
<script>
(() => {
  'use strict';
  const data = JSON.parse(document.getElementById('dashboard-data').textContent);
  const cols = Object.fromEntries(data.rowColumns.map((name, index) => [name, index]));
  const $ = id => document.getElementById(id);
  const state = { index: data.scoreDates.length - 1 };
  const palette = ['#58a6ff','#2dd4bf','#fbbf24','#a78bfa','#fb7185','#38bdf8','#f97316','#34d399'];
  const cashColor = '#64748b';
  const money = value => '$' + Math.round(value).toLocaleString('en-US');
  const pct = value => value == null ? '—' : (value * 100).toFixed(1) + '%';
  const score = value => value == null ? '—' : Number(value).toFixed(4);
  const esc = value => String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
  const rowObj = values => Object.fromEntries(data.rowColumns.map((name, index) => [name, values[index]]));
  const curveDates = data.curve.map(row => row[0]);
  const slider = $('date-slider');
  slider.max = String(data.scoreDates.length - 1);
  slider.value = String(state.index);
  $('model-status').textContent = data.modelStatus === 'RESEARCH_ONLY_NO_STRICT_PASS' ? '연구용 · 엄격 기준 미통과' : data.modelStatus;
  $('formula-note').textContent = `공시 비중 ${(data.filingWeight * 100).toFixed(0)}% · 상위 ${data.topK}종목 · 거래비용 반영 백테스트`;
  $('generated-note').textContent = `생성: ${String(data.generatedAt || '').replace('T',' ').slice(0,19)}`;

  function curveAt(date) {
    let lo = 0, hi = curveDates.length - 1, best = 0;
    while (lo <= hi) { const mid = (lo + hi) >> 1; if (curveDates[mid] <= date) { best = mid; lo = mid + 1; } else { hi = mid - 1; } }
    return data.curve[best];
  }
  function setClass(el, value) { el.classList.remove('good','bad'); if (value > 0) el.classList.add('good'); if (value < 0) el.classList.add('bad'); }
  function render() {
    const snapshot = data.scoreDates[state.index];
    const rows = snapshot.rows.map(rowObj);
    const point = curveAt(snapshot.date);
    const equity = point[1], spy = point[2], initial = data.initialCapital;
    const holdings = rows.filter(row => Number(row.Weight) > 0).sort((a,b) => b.Weight - a.Weight);
    const stockWeight = holdings.reduce((sum,row) => sum + Number(row.Weight), 0);
    const cashWeight = Math.max(0, 1 - stockWeight);
    $('date-label').textContent = snapshot.date;
    $('equity-value').textContent = money(equity);
    $('spy-value').textContent = money(spy);
    const strategyRoi = equity / initial - 1, spyRoi = spy / initial - 1, excess = equity - spy;
    $('equity-roi').textContent = `누적 ROI ${pct(strategyRoi)}`; setClass($('equity-roi'), strategyRoi);
    $('spy-roi').textContent = `누적 ROI ${pct(spyRoi)}`; setClass($('spy-roi'), spyRoi);
    $('excess-value').textContent = (excess >= 0 ? '+' : '−') + money(Math.abs(excess)); setClass($('excess-value'), excess);
    $('exposure-value').textContent = `${(stockWeight*100).toFixed(0)}% / ${(cashWeight*100).toFixed(0)}%`;
    $('position-count').textContent = `보유 ${holdings.length}종목`;
    $('regime-value').textContent = snapshot.riskOn ? '위험선호' : '위험회피';
    $('regime-value').className = 'stat-value ' + (snapshot.riskOn ? 'good' : 'market-off');
    $('cash-badge').textContent = `현금 ${(cashWeight*100).toFixed(1)}%`;
    renderChart(snapshot.date);
    renderAllocation(holdings, cashWeight, equity);
    renderHoldings(holdings, cashWeight, equity);
    renderScores(rows);
    renderEvents(rows, snapshot.riskOn);
    slider.value = String(state.index);
    $('prev').disabled = state.index === 0;
    $('next').disabled = state.index === data.scoreDates.length - 1;
  }
  function renderAllocation(holdings, cashWeight, equity) {
    const items = holdings.map((row,index) => ({name:row.Ticker, weight:Number(row.Weight), color:palette[index % palette.length], ret:row.SinceEntry}));
    if (cashWeight > .00001 || !items.length) items.push({name:'현금',weight:cashWeight || 1,color:cashColor,ret:null});
    let angle = -Math.PI/2; const cx=110,cy=110,r=82,inner=52;
    const arc = (start,end) => { const large=end-start>Math.PI?1:0; const p1=[cx+r*Math.cos(start),cy+r*Math.sin(start)],p2=[cx+r*Math.cos(end),cy+r*Math.sin(end)],q1=[cx+inner*Math.cos(end),cy+inner*Math.sin(end)],q2=[cx+inner*Math.cos(start),cy+inner*Math.sin(start)]; return `M${p1[0]},${p1[1]} A${r},${r} 0 ${large} 1 ${p2[0]},${p2[1]} L${q1[0]},${q1[1]} A${inner},${inner} 0 ${large} 0 ${q2[0]},${q2[1]} Z`; };
    let paths=''; for (const item of items) { const end=angle+item.weight*Math.PI*2; paths += `<path d="${arc(angle,end)}" fill="${item.color}" stroke="#07111f" stroke-width="2"><title>${esc(item.name)} ${(item.weight*100).toFixed(1)}%</title></path>`; angle=end; }
    $('donut').innerHTML = `${paths}<text x="110" y="104" text-anchor="middle" class="donut-center">총자산</text><text x="110" y="127" text-anchor="middle" class="donut-value">${money(equity)}</text>`;
    $('alloc-list').innerHTML = items.map(item => `<div class="alloc-row"><i class="swatch" style="background:${item.color}"></i><div><strong>${esc(item.name)}</strong>${item.ret == null ? '' : `<div class="alloc-sub">매수 후 ${pct(item.ret)}</div>`}</div><div class="alloc-meta"><strong>${(item.weight*100).toFixed(1)}%</strong><div class="alloc-sub">${money(equity*item.weight)}</div></div></div>`).join('');
  }
  function renderHoldings(holdings,cashWeight,equity) {
    const body = holdings.map(row => `<tr><td><div class="ticker">${esc(row.Ticker)}</div><div class="company">${esc(row.Company)}</div></td><td class="num">${money(equity*row.Weight)}</td><td class="num">${pct(row.Weight)}</td><td class="num">${score(row.Alpha)}</td><td class="num ${Number(row.SinceEntry)>=0?'good':'bad'}">${pct(row.SinceEntry)}</td><td>${actionBadge(row.Action)}</td></tr>`).join('');
    const cash = cashWeight > .00001 || !holdings.length ? `<tr><td><div class="ticker">현금</div><div class="company">시장 위험회피 자산</div></td><td class="num">${money(equity*cashWeight)}</td><td class="num">${pct(cashWeight)}</td><td class="num">—</td><td class="num">—</td><td><span class="badge">CASH</span></td></tr>` : '';
    $('holdings-body').innerHTML = body + cash;
  }
  function actionBadge(action) { const normalized=String(action||'NO_SCORE').toLowerCase(); const label={buy:'매수',sell:'매도',hold:'보유',watch:'관찰',avoid:'제외',no_score:'점수없음'}[normalized]||action; return `<span class="badge ${normalized}">${esc(label)}</span>`; }
  function scoreCell(value) { if (value == null) return '<span class="empty">—</span>'; const width=Math.min(100,Math.abs(Number(value))*180); const color=Number(value)>=0?'var(--green)':'var(--red)'; return `<div class="scorebar"><span>${score(value)}</span><i class="track"><i class="fill" style="width:${width}%;background:${color}"></i></i></div>`; }
  function renderScores(rows) {
    const sorted = [...rows].sort((a,b) => { const am=a.Member?0:1,bm=b.Member?0:1;if(am!==bm)return am-bm;const ar=a.Rank??9999,br=b.Rank??9999;if(ar!==br)return ar-br;return (b.Alpha??-999)-(a.Alpha??-999); });
    const scored = sorted.filter(row => row.Alpha != null).length;
    $('score-count').textContent = `${scored}/${sorted.length} 종목 점수 보유`;
    $('scores-body').innerHTML = sorted.map(row => { const bucket=row.Bucket==='TOP_N'?'당시 Top 10':row.Bucket==='WATCHLIST'?'관심종목':row.Ready==='MISSING_FINANCIALS'?'재무 누락':'당시 제외'; const flag=row.Critical?'<span class="badge critical">공시 위험</span> ':''; return `<tr><td><div class="ticker">${esc(row.Ticker)}</div><div class="company">${esc(row.Company)}</div></td><td><span class="badge">${bucket}</span></td><td class="num">${row.Rank??'—'}</td><td>${scoreCell(row.Alpha)}</td><td class="num">${score(row.Base)}</td><td class="num">${score(row.Filing)}</td><td class="num">${score(row.Momentum)}</td><td class="num">${score(row.Trend)}</td><td class="num">${score(row.Growth)}</td><td class="num">${score(row.Quality)}</td><td class="num">${score(row.Risk)}</td><td class="num">${pct(row.Weight)}</td><td>${flag}${actionBadge(row.Action)}</td></tr>`; }).join('');
  }
  function renderEvents(rows,riskOn) {
    const actions=rows.filter(row => ['BUY','SELL'].includes(row.Action));
    if (!riskOn) actions.unshift({Ticker:'시장',Action:'SELL',ExitReason:'SPY 추세 위험회피 → 현금 100%'});
    if (!actions.length) { $('events').innerHTML='<span class="event">신규 매수·매도 없음 · 기존 포지션 유지</span>'; return; }
    $('events').innerHTML=actions.map(row => `<span class="event">${actionBadge(row.Action)} <strong>${esc(row.Ticker)}</strong>${row.ExitReason?esc(reason(row.ExitReason)):''}</span>`).join('');
  }
  function reason(value){return ({UNIVERSE_EXIT:'유니버스 이탈',PROFITABLE_ROTATION:'더 높은 점수로 교체',HARD_STOP:'−25% 손절'}[value]||value);}
  function renderChart(selectedDate) {
    const svg=$('equity-chart'),W=900,H=320,m={l:68,r:22,t:16,b:38},values=data.curve.flatMap(row=>[row[1],row[2]]),min=Math.min(...values)*.92,max=Math.max(...values)*1.04;
    const x=i=>m.l+i/(data.curve.length-1)*(W-m.l-m.r), y=v=>m.t+(max-v)/(max-min)*(H-m.t-m.b);
    const path=series=>data.curve.map((row,i)=>`${i?'L':'M'}${x(i).toFixed(1)},${y(row[series]).toFixed(1)}`).join(' ');
    const ticks=[0,.25,.5,.75,1].map(t=>min+(max-min)*t);
    const yGrid=ticks.map(v=>`<line x1="${m.l}" x2="${W-m.r}" y1="${y(v)}" y2="${y(v)}" stroke="#23344b"/><text x="${m.l-9}" y="${y(v)+4}" text-anchor="end" fill="#91a4bd" font-size="11">${money(v).replace('$','$')}</text>`).join('');
    const years=[...new Set(data.curve.map(row=>row[0].slice(0,4)))]; const xLabels=years.map(year=>{const i=data.curve.findIndex(row=>row[0].startsWith(year));return `<text x="${x(i)}" y="${H-12}" text-anchor="middle" fill="#91a4bd" font-size="11">${year}</text>`}).join('');
    let selected=0; for(let i=0;i<curveDates.length;i++){if(curveDates[i]<=selectedDate)selected=i;else break;}
    svg.innerHTML=`<title>전략과 SPY 총자산</title>${yGrid}${xLabels}<path d="${path(1)}" fill="none" stroke="#58a6ff" stroke-width="2.5"/><path d="${path(2)}" fill="none" stroke="#2dd4bf" stroke-width="2"/><line x1="${x(selected)}" x2="${x(selected)}" y1="${m.t}" y2="${H-m.b}" stroke="#fbbf24" stroke-width="1"/><circle cx="${x(selected)}" cy="${y(data.curve[selected][1])}" r="5" fill="#58a6ff" stroke="#07111f" stroke-width="2"/><circle cx="${x(selected)}" cy="${y(data.curve[selected][2])}" r="4" fill="#2dd4bf" stroke="#07111f" stroke-width="2"/><rect x="${m.l}" y="${m.t}" width="${W-m.l-m.r}" height="${H-m.t-m.b}" fill="transparent" id="chart-hit"/>`;
    const hit=$('chart-hit'),tip=$('chart-tooltip'),wrap=$('chart-wrap');
    hit.addEventListener('pointermove',event=>{const rect=svg.getBoundingClientRect(),sx=W/rect.width,local=(event.clientX-rect.left)*sx,idx=Math.max(0,Math.min(data.curve.length-1,Math.round((local-m.l)/(W-m.l-m.r)*(data.curve.length-1)))),row=data.curve[idx];tip.style.display='block';tip.innerHTML=`<strong>${row[0]}</strong><br>전략 ${money(row[1])}<br>SPY ${money(row[2])}`;const px=x(idx)/W*rect.width;tip.style.left=Math.max(6,Math.min(wrap.clientWidth-tip.offsetWidth-6,px-tip.offsetWidth/2))+'px';tip.style.top=Math.max(6,y(Math.max(row[1],row[2]))/H*rect.height-tip.offsetHeight-9)+'px';});
    hit.addEventListener('pointerleave',()=>tip.style.display='none');
    hit.addEventListener('click',event=>{const rect=svg.getBoundingClientRect(),local=(event.clientX-rect.left)*W/rect.width,idx=Math.max(0,Math.min(data.curve.length-1,Math.round((local-m.l)/(W-m.l-m.r)*(data.curve.length-1)))),target=data.curve[idx][0];let best=0,bestDiff=Infinity;data.scoreDates.forEach((item,i)=>{const diff=Math.abs(new Date(item.date)-new Date(target));if(diff<bestDiff){best=i;bestDiff=diff}});state.index=best;render();});
  }
  slider.addEventListener('input',()=>{state.index=Number(slider.value);render()});
  $('prev').addEventListener('click',()=>{state.index=Math.max(0,state.index-1);render()});
  $('next').addEventListener('click',()=>{state.index=Math.min(data.scoreDates.length-1,state.index+1);render()});
  document.addEventListener('keydown',event=>{if(event.key==='ArrowLeft'){$('prev').click()}if(event.key==='ArrowRight'){$('next').click()}});
  render();
})();
</script>
</body>
</html>
'''
