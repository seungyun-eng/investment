# Codex instructions

## Project goal

Maintain one integrated repository whose files live inside OneDrive, while keeping every major workflow independently executable.

## Non-negotiable rules

1. Do not create a single run-all script unless the user explicitly requests it.
2. Do not delete or modify notebooks under `archive/original_notebooks`.
3. Do not hardcode Windows usernames, drive letters, or absolute OneDrive paths.
4. All real data and generated results remain in sibling OneDrive folders.
5. Never commit credentials, `.env`, raw data, result CSVs, model files, or virtual environments.
6. Optimization and simulation must call the same signal functions.
7. ROI means net return: `(final_value / total_injected - 1) * 100`.
8. Do not silently change financial formulas or strategy conditions. Document conflicts.
9. Use atomic file replacement for outputs written into OneDrive.
10. Run `pytest` after changes.

## Source priorities

- Data update and Transformer: `모든 코드 통합.ipynb`
- VIX strategy: `2025-07-10(1).ipynb`
- Technical strategy: `7_11(1).ipynb`
- Financial summary and DCF: `Data summary Code(1).ipynb`
- Financial crawler fallback: both Financial Statement Crawling notebooks
- Old parameter and grid-search notebooks: reference only

## Required evidence after each task

- List changed files.
- Show commands/tests executed.
- Report unresolved data-dependent issues.
- Do not claim real-data parity unless outputs were compared.
