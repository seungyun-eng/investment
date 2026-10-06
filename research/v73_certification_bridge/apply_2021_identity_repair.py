"""Fail-closed 2021 S&P500 identity bridge for V7.3 PIT certification.

This module does not certify rows. It only decides whether membership + security
identity can be considered ready after binding actual SEC identity evidence.
All other strict gates remain separate.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from pathlib import Path
import json
import re

def _d(x):
    if isinstance(x, date):
        return x.isoformat()
    return str(x)[:10]

def _z10(x):
    s = "" if x is None else "".join(ch for ch in str(x).strip() if ch.isdigit())
    return s.zfill(10) if s else ""

def accession_cik(accession):
    if not accession:
        return ""
    m = re.match(r"^(\d{1,10})-", str(accession).strip())
    return _z10(m.group(1)) if m else ""

@dataclass(frozen=True)
class IdentityDecision:
    input_ticker: str
    historical_ticker: str
    asof: str
    membership_ready: bool
    membership_ready_basis: str
    canonical_cik: str
    actual_filer_cik: str
    cik_match: bool
    share_class_key: str
    security_identity_ready: bool
    strict_membership_identity_ready: bool
    reason: str

class Repair2021:
    def __init__(self, patch_path):
        d = json.loads(Path(patch_path).read_text(encoding="utf-8"))
        self.intervals = {r[0]: r for r in d["interval_rows"]}
        self.hints = {r[0]: r for r in d["cik_hint_rows"]}
        self.classes = {r[0]: r for r in d["class_overrides"]}
        self.known = {r[0]: r for r in d["known_error_rows"]}

    def canonical(self, ticker):
        r = self.intervals[ticker]
        hist = r[2]
        cik = _z10(r[8])
        share = "COMMON_UNSPECIFIED"
        if ticker in self.hints and self.hints[ticker][2]:
            cik = _z10(self.hints[ticker][2])
        if ticker in self.known:
            k = self.known[ticker]
            hist = k[3]
            cik = _z10(k[5])
        if ticker in self.classes:
            c = self.classes[ticker]
            cik = _z10(c[1])
            share = c[2]
        return hist, cik, share

    def bind(self, ticker, asof, *, actual_filer_cik=None, accession=None,
             v21_continuity=False, share_class_key=None):
        if ticker not in self.intervals:
            raise KeyError(ticker)
        r = self.intervals[ticker]
        ds = _d(asof)
        if not (r[3] <= ds < r[4]):
            raise ValueError(f"{ticker} not in supplied 2021 interval on {ds}")
        hist, canonical, canonical_class = self.canonical(ticker)
        actual = _z10(actual_filer_cik) or accession_cik(accession)
        cik_match = bool(actual and canonical and actual == canonical)

        event_primary = r[6] == "PRIMARY_SPDJI_EVENT_BOUNDARY"
        membership_ready = event_primary or bool(v21_continuity)
        membership_basis = (
            "PRIMARY_SPDJI_EVENT_BOUNDARY" if event_primary
            else "V21_HISTORICAL_TICKER_INTERVAL_CORROBORATED" if v21_continuity
            else "V21_CONTINUITY_REQUIRED"
        )

        class_ok = True
        if ticker in self.classes:
            class_ok = bool(share_class_key and share_class_key == canonical_class)

        security_ready = cik_match and class_ok
        strict = membership_ready and security_ready
        if not membership_ready:
            reason = "MEMBERSHIP_CONTINUITY_NOT_BOUND"
        elif not actual:
            reason = "ACTUAL_SEC_FILER_CIK_OR_ACCESSION_REQUIRED"
        elif not cik_match:
            reason = "SEC_CIK_MISMATCH"
        elif not class_ok:
            reason = "SHARE_CLASS_KEY_REQUIRED_OR_MISMATCH"
        else:
            reason = "READY_MEMBERSHIP_AND_IDENTITY_ONLY"

        return IdentityDecision(
            input_ticker=ticker,
            historical_ticker=hist,
            asof=ds,
            membership_ready=membership_ready,
            membership_ready_basis=membership_basis,
            canonical_cik=canonical,
            actual_filer_cik=actual,
            cik_match=cik_match,
            share_class_key=canonical_class,
            security_identity_ready=security_ready,
            strict_membership_identity_ready=strict,
            reason=reason,
        )
