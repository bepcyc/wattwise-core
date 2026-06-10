"""Import-direction + source-leak architecture linter (ARCH-R21 / ARCH-R22 / ARCH-R2).

Three architecture invariants, all static over the module AST:

  (1) INWARD-ONLY layer imports (ARCH-R21 / ARCH-R1).
      The engine is layered L1..L7, outer (more volatile) -> inner (more stable).
      Every dependency MUST point INWARD: a module in layer ``Ln`` MUST NOT import
      a module in layer ``Lm`` where ``m > n``. Cross-cutting packages (config,
      security, observability) carry no layer rank and may be imported by anyone.

  (2) NO source-name branching via adapter imports (ARCH-R22 / ARCH-R2).
      No consumer module (L3 ingestion service, L5 domain/analytics, L6 edge) may
      import a SOURCE-NAME-SPECIFIC adapter module (e.g.
      ``wattwise_core.ingestion.adapters.intervals_icu``). Consumers select
      adapters through the registry/seam, never by importing a named adapter — the
      "consumers never branch on source" invariant (Principle A). The adapter
      package's ``__init__`` / base / registry modules are NOT source-specific and
      are allowed.

  (3) NO hardcoded source-NAME literal in control flow / logic (ARCH-R2 / ARCH-R22).
      A static scan over L3..L7 (every layered module OUTSIDE the L2 adapter
      packages) for the REGISTERED SET OF SOURCE NAMES appearing as hardcoded string
      literals in code (e.g. ``if source_key == "intervals_icu"``). The registered
      set is read from the adapter registry (the ``wattwise_core.adapters``
      entry-point group) — never a literal baked into this linter (CFG-R1a). Source
      identity that flows at RUNTIME as data is fine; what is barred is source
      identity embedded in CODE. **Exemption (AUTH-R15):** the Connections
      (``/v1/connections``) / Sync (``/v1/sync/*``) / Data-health
      (``/v1/data-health/*``) surface and the internal source-registry that BACKS
      that surface, where a source's name and status ARE legitimate runtime data
      drawn from the registry and the athlete's own connections (e.g. "we couldn't
      reach Garmin"). A bare docstring / string-expression mention of a name is
      prose, not control flow or logic, and is not flagged.

Layer map (package subtree -> layer number), per doc 10 §1:
  L2 adapters         : ingestion/adapters/**
  L3 ingestion/sync   : ingestion/**          (non-adapter)
  L4 canonical store  : persistence/**
  L5 domain analytics : analytics/**
  L6 edge             : api/**, agent/**
Cross-cutting / canonical-model (RANKLESS — importable by any layer): config,
security, observability, eval, testing, seams, AND ``domain/**``. The ``domain``
package holds the GBO canonical VALUE TYPES (closed enums GBO-R12, typed coverage
descriptors GAP-R2) — the data-model vocabulary the L4 ORM models, L5 analytics,
and L6 edge all legitimately reference. It is the GBO model, not an L5 *service*,
so an L4 model importing ``domain.enums`` is NOT an inward-rule violation.
"""

from __future__ import annotations

import ast
import functools
from collections.abc import Iterable
from pathlib import Path

from tools.lint.core import Violation, iter_python_files

_RULE_LAYER = "import-direction"
_RULE_SOURCE = "source-name-import"
_RULE_LITERAL = "source-name-literal"
_REQ_LAYER = "ARCH-R21"
_REQ_SOURCE = "ARCH-R22"
_REQ_LITERAL = "ARCH-R2"

_PKG = "wattwise_core"
_SUPPRESS = "# noqa: import-direction"

# Modules (parts below ``wattwise_core``) that constitute the AUTH-R15 surfaces on
# which a source name is legitimate runtime DATA, plus the internal source-registry
# that BACKS them — explicitly EXEMPT from the source-name-literal scan (ARCH-R2/R22):
# Connections (``/v1/connections``), Sync (``/v1/sync/*``), Data-health
# (``/v1/data-health/*``), the connections catalog backing the Connections surface,
# and the adapter registry. This is the linter's own scope definition (which surfaces
# carry source identity as data), not an operational config value (CFG-R1a concerns
# model ids / thresholds / budgets, not a static rule's exemption set).
_SOURCE_LITERAL_EXEMPT: frozenset[tuple[str, ...]] = frozenset(
    {
        ("api", "routers", "connections"),
        ("api", "routers", "sync"),
        ("api", "routers", "data_health"),
        ("api", "connection_catalog"),
        ("ingestion", "registry"),
        # The adapter-contract capability vocabulary: its SyncMode enum carries the
        # spec's ``file_import`` mode token (ADP-R1/SYN-R1), which textually matches
        # the built-in file-import source_key — contract vocabulary, not a consumer
        # branching on source identity.
        ("ingestion", "capability"),
    }
)

# Subpackage -> import-budget rank, modelling ARCH-R1 "every dependency points
# INWARD toward the L4/L5 canonical core". This is NOT doc 10's presentation L-label:
# the architecture is hexagonal (the L4 store + L5 analytics are the stable core, with
# the ingestion side AND the edge side both pointing inward), so a naive linear
# "doc-10-label > forbidden" rule wrongly bans the L3->L4 write edge that ARCH-R3
# REQUIRES (Ingestion is the ONLY writer to the canonical store). The rank below is the
# module's IMPORT BUDGET: a module may import only equal-or-lower-rank modules. The
# core (persistence, analytics) has the smallest budget; the edge (api/agent) the
# largest; adapters are leaf producers that import only the rankless ``domain`` package.
_LAYER_BY_SUBPACKAGE: dict[str, int] = {
    "persistence": 1,  # L4 canonical store — the core; written by ingestion (ARCH-R3)
    "analytics": 2,  # L5 domain analytics — reads the store only
    "ingestion": 3,  # L3 ingestion/sync — the ONLY writer to the store (ARCH-R3)
    "api": 4,  # L6 edge — reads analytics/store, triggers ingestion
    "agent": 4,  # L6 edge
}
# ``domain`` is deliberately absent: it is the rankless GBO canonical-model package
# (value types), importable inward AND from L4 ORM models without violating ARCH-R21.

# Modules inside ingestion/adapters that are NOT source-specific (registry/base/init).
_ADAPTER_NEUTRAL_MODULES = frozenset({"__init__", "base", "registry", "protocol"})


def _module_parts(path: Path) -> list[str] | None:
    """Return the dotted module path parts below ``wattwise_core``, or None.

    e.g. ``.../wattwise_core/ingestion/adapters/intervals_icu.py`` ->
    ``["ingestion", "adapters", "intervals_icu"]``. Returns None for files outside
    the package (the linter only ranks engine modules).
    """
    parts = path.with_suffix("").parts
    if _PKG not in parts:
        return None
    idx = parts.index(_PKG)
    tail = list(parts[idx + 1 :])
    return tail or None


# Rankless CONTRACT/SEAM modules (importable by any layer, like ``domain``): the
# adapter contract (SourceAdapter Protocol + FetchContext + SourceDescriptorRef) is the
# anti-corruption SEAM L2 adapters implement (spec lists "seams" among the rankless
# packages); it carries no layer rank so an adapter may import it (ARCH-R8 still bars
# adapters from importing the store/analytics, which DO carry ranks).
# ``ingestion.capability`` is the same seam's capability-descriptor vocabulary
# (ADP-R1..R3: CapabilityDescriptor, DiscoveryRef/Page, AuthContext) that every
# adapter DECLARES against, so it is rankless for the identical reason.
_RANKLESS_SEAM_MODULES: frozenset[tuple[str, ...]] = frozenset(
    {("ingestion", "base"), ("ingestion", "capability")}
)


def _layer_of(parts: list[str]) -> int | None:
    """Rank a module by its import budget; adapters are leaf producers (rank 0)."""
    if not parts:
        return None
    if tuple(parts[:2]) in _RANKLESS_SEAM_MODULES:
        return None  # rankless adapter-contract seam
    head = parts[0]
    if head == "ingestion" and len(parts) >= 2 and parts[1] == "adapters":
        return 0  # L2 adapters: import only the rankless domain package (ARCH-R8)
    return _LAYER_BY_SUBPACKAGE.get(head)


def _imported_module_parts(module: str) -> list[str] | None:
    """Parts below ``wattwise_core`` for an imported dotted module name, or None."""
    segments = module.split(".")
    if _PKG not in segments:
        return None
    idx = segments.index(_PKG)
    tail = segments[idx + 1 :]
    return tail or None


def _is_source_specific_adapter(parts: list[str]) -> bool:
    """True when imported parts name a concrete source adapter module (ARCH-R22)."""
    if len(parts) < 3 or parts[0] != "ingestion" or parts[1] != "adapters":
        return False
    return parts[2] not in _ADAPTER_NEUTRAL_MODULES


def _is_adapter_module(parts: list[str]) -> bool:
    """True when a module lives inside the L2 adapter package (scan exempt, ARCH-R2)."""
    return len(parts) >= 2 and parts[0] == "ingestion" and parts[1] == "adapters"


@functools.cache
def _registered_source_names() -> frozenset[str]:
    """The registered set of source names, read from the adapter registry (CFG-R1a).

    Derived from the ``wattwise_core.adapters`` entry-point group via the engine's
    own :func:`load_registry` — the authoritative source identities are each
    adapter's ``source_key`` — so the scan never hardcodes a source-name literal of
    its own. Cached: the registry is process-stable packaging data.

    Fail-CLOSED on the gate path. The ONLY tolerated failure is the engine package
    not being importable (the linter is run standalone outside the installed env):
    that legitimately yields an empty set and rule (3) cannot scan. But if the
    package IS importable and the registry is reachable yet exposes NO source names,
    that is a broken-registry error state — the scan would silently pass every
    consumer — so we raise loudly rather than degrade the gate to a no-op.
    """
    try:
        # Lazy by design: the linter must not hard-depend on the engine package
        # being importable (it runs standalone), nor import-couple tools/ to
        # wattwise_core at module load.
        from wattwise_core.ingestion.registry import load_registry  # noqa: PLC0415
    except ImportError:
        # Engine package absent (standalone run) — the only fail-open we allow.
        return frozenset()

    names = frozenset(load_registry().source_keys())
    if not names:
        raise RuntimeError(
            "adapter registry resolved but exposed no source names; the "
            "source-name-literal scan (ARCH-R2/R22) cannot run fail-closed against "
            "an empty registered set — refusing to degrade the gate to a silent no-op"
        )
    return names


def _docstring_constant_ids(tree: ast.AST) -> set[int]:
    """Ids of string ``Constant`` nodes that are docstrings / bare string statements.

    A string that stands alone as an expression statement (a module/class/function
    docstring or a free-standing string expression) is PROSE, not control flow or
    logic, so a registered source name mentioned there is exempt (ARCH-R2 scopes to
    source identity embedded in *code*).
    """
    out: set[int] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            out.add(id(node.value))
    return out


def _scan_source_name_literals(path: Path, own_parts: list[str], tree: ast.AST) -> list[Violation]:
    """Flag registered source-NAME string literals used in control flow / logic (ARCH-R2).

    Scans L3..L7 modules outside the adapter packages; the AUTH-R15
    Connections/Sync/Data-health surface and the backing source-registry are exempt
    (source identity is runtime data there). Docstrings / bare string expressions are
    prose and not flagged.
    """
    if _is_adapter_module(own_parts) or tuple(own_parts) in _SOURCE_LITERAL_EXEMPT:
        return []
    if _layer_of(own_parts) is None:
        return []  # rankless cross-cutting / contract module — not an L3..L7 layer
    names = _registered_source_names()
    if not names:
        return []
    doc_ids = _docstring_constant_ids(tree)
    found: list[Violation] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        if id(node) in doc_ids or node.value not in names:
            continue
        found.append(
            Violation(
                path=path,
                line=node.lineno,
                rule=_RULE_LITERAL,
                requirement=_REQ_LITERAL,
                message=(
                    f"hardcoded registered source-name literal '{node.value}' in "
                    f"control flow/logic; consumers MUST stay source-blind and select "
                    f"by key via the registry/seam (source identity is runtime data, "
                    f"named only on the Connections/Sync/Data-health surface, AUTH-R15)"
                ),
            )
        )
    return list(dict.fromkeys(found))


def _imported_names(tree: ast.AST, path: Path) -> list[tuple[str, int]]:
    """Collect ``(dotted_module, lineno)`` for every import in a module.

    Relative ``from . import x`` / ``from .adapters import y`` are resolved against
    the importing module's package so intra-package edges are ranked correctly.
    """
    own = _module_parts(path) or []
    own_pkg = own[:-1]  # package containing this module
    results: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                results.append((alias.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                # ``from pkg.a.b import x`` — record the module AND each imported
                # name as ``pkg.a.b.x`` so a ``from ...adapters import intervals_icu``
                # source-leak (ARCH-R22) is visible (the leaked name is the alias).
                results.append((node.module, node.lineno))
                for alias in node.names:
                    results.append((f"{node.module}.{alias.name}", node.lineno))
            else:
                base = _resolve_from(node, own_pkg)
                if base is not None:
                    module = node.module
                    dotted = ".".join([_PKG, *base, *([module] if module else [])])
                    results.append((dotted, node.lineno))
                    for alias in node.names:
                        results.append((f"{dotted}.{alias.name}", node.lineno))
    return results


def _resolve_from(node: ast.ImportFrom, own_pkg: list[str]) -> list[str] | None:
    """Resolve the relative-import base (parts below ``wattwise_core``) or None."""
    if node.level == 0:
        return None
    # level 1 = current package; level N strips (N-1) trailing packages.
    keep = own_pkg[: len(own_pkg) - (node.level - 1)] if node.level > 1 else own_pkg
    return keep


def _line_suppressed(lines: list[str], lineno: int) -> bool:
    """True when the import's physical line carries the suppression token."""
    return 1 <= lineno <= len(lines) and _SUPPRESS in lines[lineno - 1]


def _check_source(path: Path, source: str) -> list[Violation]:
    """Apply all three architecture invariants to one engine module."""
    own_parts = _module_parts(path)
    if own_parts is None:
        return []
    own_layer = _layer_of(own_parts)
    violations: list[Violation] = []
    lines = source.splitlines()
    tree = ast.parse(source, filename=str(path))

    for dotted, lineno in _imported_names(tree, path):
        target = _imported_module_parts(dotted)
        if target is None:
            continue
        if _line_suppressed(lines, lineno):
            continue
        violations.extend(_evaluate_edge(path, lineno, own_parts, own_layer, target))
    violations.extend(_scan_source_name_literals(path, own_parts, tree))
    # The module-name and its alias-qualified form can both yield the same edge
    # finding; de-duplicate so one offending import is reported once.
    return list(dict.fromkeys(violations))


def _evaluate_edge(
    path: Path,
    lineno: int,
    own_parts: list[str],
    own_layer: int | None,
    target: list[str],
) -> list[Violation]:
    """Emit violations for a single import edge (layer-direction + source-leak)."""
    found: list[Violation] = []

    # (2) Source-name adapter import — forbidden for any consumer that is NOT itself
    # inside the adapter package (an adapter may import its own siblings).
    importer_is_adapter = (
        len(own_parts) >= 2
        and own_parts[0] == "ingestion"
        and own_parts[1] == "adapters"
    )
    if _is_source_specific_adapter(target) and not importer_is_adapter:
        found.append(
            Violation(
                path=path,
                line=lineno,
                rule=_RULE_SOURCE,
                requirement=_REQ_SOURCE,
                message=(
                    f"imports source-specific adapter '{'.'.join(target)}'; consumers "
                    f"MUST select adapters via the registry/seam, never branch on source"
                ),
            )
        )

    # (1) Inward-only layer rule: own_layer must be >= target_layer (import inward).
    target_layer = _layer_of(target)
    if own_layer is not None and target_layer is not None and target_layer > own_layer:
        found.append(
            Violation(
                path=path,
                line=lineno,
                rule=_RULE_LAYER,
                requirement=_REQ_LAYER,
                message=(
                    f"L{own_layer} module imports L{target_layer} module "
                    f"'{'.'.join(target)}'; dependencies MUST point inward (Ln may not "
                    f"import Lm where m>n)"
                ),
            )
        )
    return found


def check_paths(paths: Iterable[Path]) -> list[Violation]:
    """Run the import-direction + source-leak linter over engine modules under `paths`."""
    violations: list[Violation] = []
    for path in iter_python_files(paths):
        if _PKG not in path.parts:
            continue
        violations.extend(_check_source(path, path.read_text(encoding="utf-8")))
    return violations
