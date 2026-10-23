"""The mutation gate survives a ``.meta`` file that mutmut left half-written.

mutmut rewrites each ``mutants/<path>.meta`` in place, so the gate's budget SIGINT can cut a
write short. The gate must neither crash on that file nor let it poison the persisted
mutants/ cache for the next run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts import mutation_gate

pytestmark = pytest.mark.unit

_GOOD = "src/pkg/good.py"
_TORN = "src/pkg/torn.py"


def _mutated_source(*names: str) -> str:
    return "".join(f"def {name}(a):\n    return a\n\n" for name in names)


@pytest.fixture
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A mutants/ tree with one intact and one truncated ``.meta``."""
    monkeypatch.setattr(mutation_gate, "_REPO_ROOT", tmp_path)
    root = tmp_path / "mutants" / "src" / "pkg"
    root.mkdir(parents=True)
    (root / "good.py").write_text(_mutated_source("x_f__mutmut_1", "x_f__mutmut_2"))
    (root / "good.py.meta").write_text(
        json.dumps({"exit_code_by_key": {"pkg.good.x_f__mutmut_1": 1, "pkg.good.x_f__mutmut_2": 0}})
    )
    torn_names = ("x_g__mutmut_1", "x_g__mutmut_2", "x_g__mutmut_3")
    (root / "torn.py").write_text(_mutated_source(*torn_names))
    (root / "torn.py.meta").write_text('{"exit_code_by_key": {"pkg.torn.x_g__mutmut_1": 1, "pkg.to')
    return root


def test_truncated_meta_counts_its_mutants_as_not_checked(cache: Path) -> None:
    """A torn ``.meta`` reads like a budget cut: its mutants are "not checked", no crash."""
    counts, survivors, total = mutation_gate._collect_verdicts([_GOOD, _TORN])

    assert total == 5
    assert counts["killed"] == 1
    assert counts["survived"] == 1
    assert counts["not checked"] == 3
    assert survivors == ["pkg.good.x_f__mutmut_2"]
    # Collection is read-only: it runs more than once per gate (per package, then union).
    assert (cache / "torn.py.meta").exists()


def test_drop_corrupt_metas_removes_only_the_torn_file(cache: Path) -> None:
    """The pre-run sweep drops the torn ``.meta`` and its mutated copy, keeps the intact pair.

    The copy must go too: mutmut does not regenerate a copy whose mtime matches its source,
    and a copy with no ``.meta`` would match no mutant at all on the next run.
    """
    mutation_gate._drop_corrupt_metas()

    assert not (cache / "torn.py.meta").exists()
    assert not (cache / "torn.py").exists()
    assert (cache / "good.py.meta").exists()
    assert (cache / "good.py").exists()


def test_meta_without_verdict_map_is_treated_as_unreadable(cache: Path) -> None:
    """Valid JSON that lacks ``exit_code_by_key`` is as unusable as torn JSON."""
    (cache / "torn.py.meta").write_text(json.dumps({"hash_by_function_name": {}}))

    assert mutation_gate._read_verdicts(cache / "torn.py.meta") is None
    _, _, total = mutation_gate._collect_verdicts([_TORN])
    assert total == 3
