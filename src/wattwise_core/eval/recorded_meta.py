"""Recorded-fixture (cassette) metadata + the stale-cassette static check (QA-EVAL-R12(a)).

The OSS recorded-response fixtures ARE the versioned eval datasets: each dataset file
carries a ``recorded_with`` block — the model identifier and the sha256 of the coach
prompt/persona/language content the recorded outputs were captured under. Refreshing a
fixture is a REVIEWED change via ``just eval-record`` (never a test-run side-effect,
the GOLD-R2 rule applied to cassettes).

:func:`verify_recorded_datasets` is the CI static check: it recomputes the CURRENT pins
from the layered config (:func:`wattwise_core.config.recorded_pins.load_recorded_pins`) and
fails when any dataset's metadata is missing or out of sync — a model/prompt change that
was not re-recorded ("stale cassettes") fails the build instead of silently grading new
behaviour against old recordings. Runs inside ``python -m wattwise_core.eval run`` /
``record`` so the gate (CI-R1 item 6) enforces it on every PR.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from wattwise_core.config.recorded_pins import load_recorded_pins

_DATASETS_DIR = Path(__file__).parent / "datasets"

#: The dataset key carrying the capture-time pins (QA-EVAL-R12(a)).
RECORDED_WITH_KEY = "recorded_with"


def current_pins() -> dict[str, str]:
    """The model + prompt-content pins the fixtures must have been recorded under."""
    return load_recorded_pins()


def verify_recorded_datasets(datasets_dir: Path | None = None) -> tuple[str, ...]:
    """Return one failure line per dataset whose cassette metadata is stale/missing.

    Empty result == every committed dataset was recorded under the CURRENTLY pinned
    model + prompt content. A non-empty result is the QA-EVAL-R12(a) stale-cassette
    condition: regenerate via ``just eval-record`` and commit the reviewed diff.
    """
    pins = current_pins()
    failures: list[str] = []
    directory = datasets_dir if datasets_dir is not None else _DATASETS_DIR
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        meta = payload.get(RECORDED_WITH_KEY)
        if not isinstance(meta, dict):
            failures.append(f"{path.name}: missing {RECORDED_WITH_KEY} cassette metadata")
            continue
        for key, expected in pins.items():
            if str(meta.get(key)) != expected:
                failures.append(
                    f"{path.name}: {RECORDED_WITH_KEY}.{key} is stale "
                    f"(recorded {meta.get(key)!r}, pinned {expected!r}) — "
                    "re-record via `just eval-record` and commit the reviewed diff"
                )
    return tuple(failures)


def stamp_recorded_datasets(datasets_dir: Path | None = None) -> tuple[str, ...]:
    """Stamp every dataset with the CURRENT pins (the ``just eval-record`` refresh step).

    Returns the dataset file names whose metadata changed. The stamp is a MINIMAL text
    edit (insert/replace only the ``recorded_with`` object) so the reviewed diff shows
    exactly the metadata change, never a whole-file reformat. The caller commits the
    diff as a reviewed change with a rationale (QA-EVAL-R12(a)); this function never
    runs as a side-effect of a test.
    """
    pins = current_pins()
    pins_json = json.dumps(pins, sort_keys=True)
    pattern = re.compile(rf'"{RECORDED_WITH_KEY}":\s*\{{[^{{}}]*\}}')
    changed: list[str] = []
    directory = datasets_dir if datasets_dir is not None else _DATASETS_DIR
    for path in sorted(directory.glob("*.json")):
        text = path.read_text(encoding="utf-8")
        if json.loads(text).get(RECORDED_WITH_KEY) == pins:
            continue
        replacement = f'"{RECORDED_WITH_KEY}": {pins_json}'
        if pattern.search(text):
            new_text = pattern.sub(replacement, text, count=1)
        else:
            brace = text.index("{")
            new_text = text[: brace + 1] + replacement + ", " + text[brace + 1 :]
        json.loads(new_text)  # the surgical edit must still be valid JSON
        path.write_text(new_text, encoding="utf-8")
        changed.append(path.name)
    return tuple(changed)


__all__ = [
    "RECORDED_WITH_KEY",
    "current_pins",
    "stamp_recorded_datasets",
    "verify_recorded_datasets",
]
