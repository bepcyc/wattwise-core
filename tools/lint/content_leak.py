"""Content/IP-leak gate: no persona/prompt/skill body embedded in engine source (ARCH-R29 / CFG-R3).

The architecture fitness function that makes the engine/content boundary executable (ARCH-R29):
EVERY coach persona text, system/agent prompt body, and skill definition MUST be loaded from
external config at runtime — the engine source MUST embed NONE inline (CFG-R3 / SKILL-R1). This
linter FAILS THE BUILD if a prompt/persona/skill body is found as a string literal in an engine
``.py`` source, so the four formerly-inline system prompts (``_PLAN_SYSTEM`` / ``_CLAIM_SYSTEM`` /
``_REFLECT_SYSTEM`` / ``_READINESS_SYSTEM``) can never silently return inline, and a new one can
never be added without externalizing it.

It is the COMPLEMENT of the ``content_copy`` gate: that one polices athlete-facing COPY catalogs;
THIS one polices the engine SOURCE for an embedded behavior-asset body. It is deliberately
NON-VACUOUS — planting an inline prompt body trips it — via two AST signals over engine source:

  * NAMED behavior-asset literal (the exact mechanism the removed constants used): a string assigned
    (module- or class-level, incl. a concatenated/joined string) to a name that reads as a
    prompt/persona/skill body — ``*_SYSTEM`` / ``*_PROMPT`` / ``*PERSONA*`` / ``*_PREAMBLE`` /
    ``*_PLAYBOOK*`` / ``*_SKILL_BODY*`` and similar. Such a name is, by convention, a behavior
    asset; its VALUE belongs in the config bundle, never in code.
  * UNNAMED persona/prompt prose literal: a long second-person instruction string (a multi-sentence
    body containing a persona/coach instruction marker such as "You are the …", "system prompt",
    or an imperative coach instruction) — the shape of a prompt body even if assigned to an
    innocuous name or passed positionally.

To avoid false positives on legitimate long docstrings / log templates / SQL, the prose signal is
scoped: a module/class/function DOCSTRING (the first string expression of its body) is exempt, and
the marker set is specific to persona/prompt prose. The named-literal signal needs no prose marker
(the name alone is the tell). Both are required to make the gate non-vacuous yet precise.

Cited requirements: ARCH-R29, CFG-R3, SKILL-R1, SKILL-R6.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable
from pathlib import Path

from tools.lint.core import Violation, iter_python_files

_RULE = "content-leak-prompt"
_REQ = "ARCH-R29"

#: Only ENGINE source is scanned: the application package. Tools/tests are out of scope (a test
#: legitimately plants a prompt body as a fixture to exercise THIS gate — see ``test_lints``).
_APP_SEGMENT = "wattwise_core"

#: A name (the LHS of an assignment / an AnnAssign target) that reads as a behavior-asset body.
#: A string VALUE bound to such a name is a prompt/persona/skill definition that MUST live in the
#: config bundle, not engine code (CFG-R3). Case-insensitive on the bare identifier.
_ASSET_NAME = re.compile(
    r"(?:^|_)(?:system|prompt|persona|preamble|playbook|skill_body|skill_def|"
    r"grounding_rule|coach_voice|instruction)s?(?:$|_)",
    re.IGNORECASE,
)

#: Markers of persona/prompt PROSE: the tell-tale shape of an instruction body addressed to a model
#: in the second person. A long literal carrying one of these is a prompt body even if its binding
#: name is innocuous. Kept specific so a normal log/docstring/comment does not trip the gate.
_PROSE_MARKER = re.compile(
    r"\byou are the\b|\byou are an?\b|\bsystem prompt\b|\bnever (?:call|invent|reveal|emit)\b"
    r"|\breturn only the\b|\bdo not (?:refuse|estimate|use outside)\b",
    re.IGNORECASE,
)

#: A literal this long (chars) AND carrying a prose marker is treated as an unnamed prompt body.
_PROSE_MIN_CHARS = 80

#: A behavior-asset-NAMED literal is treated as an inline body when it is MULTI-LINE or a
#: multi-word string of at least this many chars. An empty/short value (a schema field default
#: ``system_prompt: str = ""``, a one-word token like ``"standard"``, a separator) is NOT a leaked
#: body — a real externalized prompt/persona/skill body is multi-word prose. The whitespace
#: requirement (a space/newline) is what separates a prose BODY from a bare config token.
_ASSET_MIN_CHARS = 24


def _is_body_literal(text: str) -> bool:
    """True when an asset-NAMED literal looks like a prompt/persona BODY (not a short token).

    A body is multi-line, or multi-word prose of non-trivial length; a bare config token
    (``""`` / ``"standard"`` / a separator) has no internal whitespace and/or is short, so it is
    NOT treated as a leaked body. This keeps the named signal precise yet catches a short-but-real
    persona/prompt embedded under a behaviour-asset name.
    """
    stripped = text.strip()
    if "\n" in stripped:
        return True
    return " " in stripped and len(stripped) >= _ASSET_MIN_CHARS


def _const_str(node: ast.expr) -> str | None:
    """Return the static string value of ``node`` if it is a string literal / concat / join.

    Handles the three shapes a prompt body takes in source: a bare ``ast.Constant`` string, an
    implicit/`+` concatenation of string constants, and a ``"".join((...))`` of string constants.
    Anything non-static (an f-string, a name reference, a call to something else) returns ``None``
    — only a literal BODY embedded in code is a leak.
    """
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _const_str(node.left), _const_str(node.right)
        return None if left is None or right is None else left + right
    return None


def _target_names(node: ast.Assign | ast.AnnAssign) -> list[str]:
    """The simple identifier names this assignment binds (for the asset-name signal)."""
    names: list[str] = []
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    for tgt in targets:
        if isinstance(tgt, ast.Name):
            names.append(tgt.id)
    return names


def _docstring_lines(tree: ast.Module) -> set[int]:
    """Line numbers of every module/class/function DOCSTRING literal (exempt from the prose scan).

    A docstring is the first statement of a module/class/function body when it is a bare string
    expression. Engine docstrings legitimately describe behavior in prose; they are NOT prompt
    bodies, so the unnamed-prose signal skips them (the named-asset signal never matched a docstring
    anyway, as a docstring has no binding name).
    """
    exempt: set[int] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list) or not body:
            continue
        first = body[0]
        if (
            isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
            and isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            exempt.add(first.value.lineno)
    return exempt


def _scan_source(path: Path, source: str) -> list[Violation]:
    """Flag inline prompt/persona/skill bodies in one engine source file (ARCH-R29)."""
    out: list[Violation] = []
    tree = ast.parse(source, filename=str(path))
    doc_lines = _docstring_lines(tree)

    for node in ast.walk(tree):
        # Signal 1: a string assigned to a behavior-asset NAME (the removed-constant mechanism).
        if isinstance(node, ast.Assign | ast.AnnAssign):
            value = node.value
            if value is None:
                continue
            literal = _const_str(value)
            if literal is None or not _is_body_literal(literal):
                continue
            for name in _target_names(node):
                if _ASSET_NAME.search(name):
                    out.append(
                        Violation(
                            path=path,
                            line=node.lineno,
                            rule=_RULE,
                            requirement=_REQ,
                            message=(
                                f"'{name}' embeds a prompt/persona/skill body inline; "
                                f"externalize it to the coach-config bundle (CFG-R3/SKILL-R1)"
                            ),
                        )
                    )

    # Signal 2: an unnamed long persona/prompt PROSE literal anywhere (a prompt body even under an
    # innocuous name / passed positionally), excluding docstrings.
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        if node.lineno in doc_lines:
            continue
        text = node.value
        if len(text) >= _PROSE_MIN_CHARS and _PROSE_MARKER.search(text):
            out.append(
                Violation(
                    path=path,
                    line=node.lineno,
                    rule=_RULE,
                    requirement=_REQ,
                    message=(
                        "inline persona/prompt prose literal in engine source; "
                        "externalize it to the coach-config bundle (CFG-R3/SKILL-R1)"
                    ),
                )
            )
    return out


def _is_engine_source(path: Path) -> bool:
    """True for an engine ``.py`` under the application package (tests/tools are out of scope)."""
    parts = path.parts
    out_of_scope = path.name.startswith("test_") or "tests" in parts or "tools" in parts
    return _APP_SEGMENT in parts and not out_of_scope


def check_paths(paths: Iterable[Path]) -> list[Violation]:
    """Run the content/IP-leak gate over engine source (ARCH-R29 / CFG-R3)."""
    violations: list[Violation] = []
    for path in iter_python_files(paths):
        if _is_engine_source(path):
            violations.extend(_scan_source(path, path.read_text(encoding="utf-8")))
    return violations


__all__ = ["check_paths"]
