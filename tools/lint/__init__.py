"""Custom AST/static lint pack for wattwise-core (QUAL-R9/R10/R11/R13, ARCH-R21/R22, BOOT-R3).

These linters back the CI-R1 "code-craft" (item 14), "content/copy" (item 21),
"no-vendor-SQL" (item 13 static slice), and import-direction (ARCH-R21/R22) gates
that native ruff/flake8 rules cannot express. Each linter is a pure function from a
set of source paths to a list of typed `Violation`s; the runnable entry point
(`python -m tools.lint`) aggregates them and exits non-zero on any *blocking*
violation. Advisory findings (the heuristic non-English-prose layer, QUAL-R11(b))
are reported but never fail the build, so the gate stays non-flaky (QUAL-R11).
"""

from tools.lint.core import Severity, Violation, iter_python_files

__all__ = ["Severity", "Violation", "iter_python_files"]
