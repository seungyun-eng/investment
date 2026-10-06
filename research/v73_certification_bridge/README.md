# V7.3 2021 PIT Certification Bridge

Purpose: prevent the SEC processing automation from looping forever in validation-only mode after the current raw set is complete, and hand off safely into 2021 strict certification work.

## Safety rules
- Never modify Frozen V7.3 strategy, scoring, universe, backtest, or NAV.
- Never promote a row to CERTIFIED from identity evidence alone.
- Preserve existing certified/identity-ready rows unless new primary evidence proves them wrong.
- Do not use workbook CIK as membership proof.
- Keep share classes distinct.
- Continuing/all-year membership needs the V21 historical-ticker interval layer or equivalent continuity evidence.
- Price, adjusted-basis, corporate-action, filing timing, and financial/factor gates stay separate.

## Current stall signature
The 2026-10-06 core receipt showed queue.sqlite done=18,684 / failed=0 / pending=0, no new raw, no newly-durable done jobs, and validation_only_execution=true. The last reported 2021 strict baseline was 3,938 / 127,009 with 7,384 financial/factor-pass rows and 2,142 identity-pending rows.

The automation should transition to certification bridge work instead of treating a new validation checkpoint as progress.
