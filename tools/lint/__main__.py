"""Console entry point: ``python -m tools.lint [paths]`` (CI-R1 items 13/14/21).

Runs the full custom lint pack and exits non-zero on any blocking violation so the
``just lint`` recipe and CI can gate on it.
"""

from __future__ import annotations

import sys

from tools.lint.runner import run


def main() -> int:
    """Dispatch to the aggregating runner with command-line paths."""
    return run(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
