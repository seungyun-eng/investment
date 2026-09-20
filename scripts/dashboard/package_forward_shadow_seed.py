from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from stock_research.paths import load_paths


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package the minimum sealed inputs for Forward Shadow cloud continuation."
    )
    parser.add_argument("--stock-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = load_paths(args.stock_root)
    results = paths.results
    audit = results / "bravo_desk/alpha_desk_final_champion_audit_v1/20260919"
    comparison = results / "bravo_desk/comparison_2015_2026_v1/20260917_fixed_common_period"
    systemic = results / "bravo_desk/systemic_risk_engine_v1/20260917_fixed_systemic"
    mega = results / "bravo_desk/abe_rates_2015_2026_v1/20260917_fixed_six"
    architecture = results / "bravo_desk/alpha_master_portfolio_architecture_v1/20260919"
    v73 = results / "v73_frozen_macro_annual/20260916T060055_824125Z"
    w100 = results / "bravo_desk/dynamic_universe_ab_v1b/20260919"
    shadow = results / "alpha_desk_forward_shadow_v1"

    files = [
        audit / "FROZEN_SHADOW_MANIFEST.json",
        comparison / "panel.parquet",
        comparison / "core_plan.parquet",
        systemic / "vix.parquet",
        systemic / "accounts/rank1__M0__BASE25/nav.parquet",
        systemic / "accounts/rank1__JOINT__BASE25/controller_audit.parquet",
        systemic / "accounts/rank1__JOINT__BASE25/nav.parquet",
        systemic / "accounts/rank1__JOINT__BASE25/orders.parquet",
        systemic / "accounts/rank1__JOINT__BASE25/holdings.parquet",
        systemic / "accounts/rank1__JOINT__BASE25/metrics.json",
        mega / "accounts/B_BASE/nav.parquet",
        mega / "accounts/B_BASE/orders.parquet",
        mega / "accounts/B_BASE/holdings.parquet",
        mega / "accounts/B_BASE/metrics.json",
        architecture
        / "accounts/TOP1__SYSTEMIC_JOINT__VOL15__C25/daily_ledger.parquet",
        v73 / "base_model_targets.parquet",
        v73 / "accounts/2026-09-11/25bp/V73_U001_FROZEN__M0_NONE/daily_nav.csv",
        v73 / "accounts/2026-09-11/25bp/V73_U001_FROZEN__M0_NONE/orders.csv",
        v73 / "accounts/2026-09-11/25bp/V73_U001_FROZEN__M0_NONE/holdings.parquet",
        v73
        / "research_data/Processed Data/SEC Filings/combined/point_in_time_features.csv",
        w100 / "TRACK1_WIDTH_MEMBERSHIP.parquet",
        w100 / "plans/monthly__W100.parquet",
    ]
    files.extend((v73 / "research_data/Dashboard Data/Prices").glob("*.csv"))
    files.extend(path for path in shadow.rglob("*") if path.is_file())
    files = sorted(set(files))
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"seed inputs missing: {missing}")

    inventory = {
        path.relative_to(paths.stock_root).as_posix(): sha256(path) for path in files
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for path in files:
            archive.write(path, path.relative_to(paths.stock_root).as_posix())
        archive.writestr(
            "FORWARD_SHADOW_SEED_MANIFEST.json",
            json.dumps(
                {
                    "stage": "ALPHA_DESK_FORWARD_SHADOW_V1",
                    "file_count": len(files),
                    "files": inventory,
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "files": len(files),
                "bytes": args.output.stat().st_size,
                "sha256": sha256(args.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
