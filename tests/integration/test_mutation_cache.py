"""A restored T-MUT source cache must use current assets and remove deleted modules."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from mutmut import __main__ as mutmut
from scripts import mutation_gate

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("leg", ["full", "pr"])
def test_warm_cache_refreshes_assets_and_deletions_before_mutmut(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, leg: str
) -> None:
    """TIER-R6/DOC-R2: both legs refresh stale assets while preserving current mutants."""
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "src"
    cached = tmp_path / "mutants" / "src"
    source.mkdir()
    cached.mkdir(parents=True)
    (source / "openapi.json").write_text('{"current": true}\n')
    (cached / "openapi.json").write_text('{"stale": true}\n')
    (source / "unchanged.py").write_text("value = 1\n")
    (cached / "unchanged.py").write_text("cached mutation trampolines\n")
    (cached / "unchanged.py.meta").write_text("cached verdicts\n")
    (cached / "deleted.py").write_text("obsolete module\n")
    (cached / "deleted.py.meta").write_text("obsolete verdicts\n")
    (cached / "deleted.json").write_text("obsolete asset\n")
    copied_roots = ["tests", "tools", "scripts"]
    for root in copied_roots:
        (tmp_path / root).mkdir()
        (tmp_path / root / "current.py").write_text("current helper\n")
        destination = tmp_path / "mutants" / root
        destination.mkdir()
        (destination / "current.py").write_text("stale helper\n")
        (destination / "deleted.py").write_text("obsolete helper\n")
    (tmp_path / "mutants" / "setup.cfg").write_text("obsolete config\n")
    (tmp_path / "mutants" / "test_deleted.py").write_text("obsolete root test\n")
    monkeypatch.setattr(mutation_gate, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(mutation_gate, "_mutmut_config", lambda: (["src"], []))
    monkeypatch.setattr(
        mutation_gate, "_tracked_files", lambda: {"src/openapi.json", "src/unchanged.py"}
    )
    monkeypatch.setattr(mutation_gate, "_changed_files", lambda _: {"src/unchanged.py"})
    monkeypatch.setattr(mutation_gate, "_restore_commit_mtimes", lambda _: None)
    monkeypatch.setattr(
        mutmut.Config,
        "get",
        lambda: SimpleNamespace(
            source_paths=[Path("src")],
            also_copy=[Path(root) for root in [*copied_roots, "setup.cfg"]],
        ),
    )

    class CampaignReached(Exception):
        pass

    def start_mutmut(patterns: list[str], budget: int) -> tuple[int, bool, str]:
        # Exercise the real library's copy behavior against the warmed cache.
        mutmut.copy_src_dir()
        mutmut.copy_also_copy_files()
        assert (cached / "openapi.json").read_bytes() == (source / "openapi.json").read_bytes()
        assert not (cached / "deleted.py").exists()
        assert not (cached / "deleted.py.meta").exists()
        assert not (cached / "deleted.json").exists()
        assert (cached / "unchanged.py").read_text() == "cached mutation trampolines\n"
        assert (cached / "unchanged.py.meta").read_text() == "cached verdicts\n"
        for root in copied_roots:
            destination = tmp_path / "mutants" / root
            assert not (destination / "deleted.py").exists()
            assert (destination / "current.py").read_text() == "current helper\n"
        assert not (tmp_path / "mutants" / "setup.cfg").exists()
        assert not (tmp_path / "mutants" / "test_deleted.py").exists()
        raise CampaignReached

    monkeypatch.setattr(mutation_gate, "_run_mutmut", start_mutmut)
    with pytest.raises(CampaignReached):
        mutation_gate.main(["--leg", leg])
