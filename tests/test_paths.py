from pathlib import Path

from stock_research.paths import load_paths


def test_explicit_stock_root(tmp_path: Path):
    paths = load_paths(tmp_path)
    assert paths.stock_root == tmp_path.resolve()
    assert paths.processed == tmp_path.resolve() / "Processed Data"
    assert paths.parameters.exists()
