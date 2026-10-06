"""Detect a validation-only SEC stall and force the next safe research phase."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

def decide(receipt, strict_certified, strict_total, financial_pass, identity_pending):
    docs = receipt.get("documents", {})
    raw_complete = docs.get("pending") == 0 and docs.get("failed") == 0
    no_new = (
        receipt.get("new_raw_zip_after_parent", 0) == 0
        and receipt.get("jobs_durable_done_added_this_run", 0) == 0
    )
    validation_only = bool(receipt.get("validation_only_execution"))
    stalled = raw_complete and no_new and validation_only and strict_certified < strict_total
    fin_identity_est = max(financial_pass - identity_pending, 0)
    return {
        "phase": "CERTIFICATION_BRIDGE_REQUIRED" if stalled else "SEC_PROCESSING_OR_VALIDATION",
        "action": (
            "STOP_REPEAT_VALIDATION_ONLY_AND_RUN_2021_CERTIFICATION_BRIDGE"
            if stalled else "CONTINUE_EXISTING_SEC_FLOW"
        ),
        "raw_complete_current_set": raw_complete,
        "no_new_raw_or_done_jobs": no_new,
        "validation_only": validation_only,
        "strict": {"certified": strict_certified, "total": strict_total},
        "financial_factor_pass": financial_pass,
        "identity_pending": identity_pending,
        "financial_plus_identity_ready_estimate": fin_identity_est,
        "downstream_gap_after_identity": max(fin_identity_est - strict_certified, 0),
        "upstream_financial_factor_gap": max(strict_total - financial_pass, 0),
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--core-receipt", required=True)
    p.add_argument("--strict-certified", type=int, required=True)
    p.add_argument("--strict-total", type=int, required=True)
    p.add_argument("--financial-pass", type=int, required=True)
    p.add_argument("--identity-pending", type=int, required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    receipt = json.loads(Path(a.core_receipt).read_text(encoding="utf-8"))
    out = decide(receipt, a.strict_certified, a.strict_total, a.financial_pass, a.identity_pending)
    Path(a.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
