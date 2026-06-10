"""Shared hypothesis profiles (TIER-R1/TIER-R5): committed, explicit configuration.

Two committed profiles:

* ``default`` — the offline PR-gate profile: bounded examples, shrinking on
  (hypothesis default), and an EXPLICITLY configured example-database path (TIER-R1:
  the database location is committed configuration, not an implicit default).
* ``fuzz-nightly`` — the TIER-R5 mode (b) nightly campaign profile: a far larger
  generative budget with no deadline. Selected by ``just test-fuzz-nightly`` via
  ``--hypothesis-profile=fuzz-nightly``; never used by the PR gate. (A
  coverage-guided Atheris/libFuzzer engine is unavailable on CPython 3.13, so the
  nightly depth comes from the extended generative budget.)

Per-test ``@settings`` decorators still override profile values where a suite needs
tighter bounds.
"""

from __future__ import annotations

from hypothesis import HealthCheck, settings
from hypothesis.database import DirectoryBasedExampleDatabase

settings.register_profile(
    "default",
    database=DirectoryBasedExampleDatabase(".hypothesis/examples"),
)
settings.register_profile(
    "fuzz-nightly",
    database=DirectoryBasedExampleDatabase(".hypothesis/examples"),
    max_examples=5_000,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
settings.load_profile("default")
