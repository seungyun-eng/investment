# Codex에 처음 넣을 요청

아래 요청은 이 저장소를 Codex로 연 뒤 그대로 사용하세요.

```text
Read AGENTS.md, README.md, docs/KNOWN_FIXES.md, and docs/NOTEBOOK_MAP.md first.

This repository is inside OneDrive/주식 and must remain there.
Do not create a run-all script. Every workflow must remain independently executable.

First task: audit the integrated v1 against the notebooks in
archive/original_notebooks.

Do not modify strategy or financial formulas yet.

1. Compare each module and script with its source notebook.
2. Create docs/PARITY_AUDIT.md with:
   - matched logic,
   - missing logic,
   - conflicting formulas,
   - changed column names,
   - data-dependent behavior that cannot be verified without real OneDrive files.
3. Pay special attention to:
   - optimization versus simulation signal parity,
   - ROI definitions,
   - Bollinger standard deviation ddof,
   - RSI calculation method,
   - final liquidation,
   - extra-on-buy behavior,
   - DCF and working-capital signs,
   - fiscal-year snapshot logic,
   - Transformer target construction.
4. Run pytest and Python syntax compilation.
5. Do not crawl websites, update market data, train models, or write to real Results.
6. Do not delete or edit original notebooks.
7. Commit only the audit document and any test-only fixes with:
   docs: audit notebook parity
```

감사 결과를 확인한 다음 두 번째 요청:

```text
Using docs/PARITY_AUDIT.md, fix only high-confidence defects in small commits.

For every defect:
1. Add or update a deterministic test first.
2. Make the smallest code change.
3. Run pytest.
4. Update docs/KNOWN_FIXES.md.
5. Do not change ambiguous formulas. Put those in docs/DECISIONS_NEEDED.md.
6. Never touch real OneDrive output folders during tests.

Create separate commits by area:
- preprocessing
- VIX strategy
- technical strategy
- financial analysis
- Transformer
```
