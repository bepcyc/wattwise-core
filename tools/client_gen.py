"""Generate TypeScript interfaces + type guards from the OpenAPI document (DOC-R5/R6).

The CI client-generation step: builds typed TS interfaces for every component schema
of the emitted OpenAPI document plus shape-discriminating type guards, and FAILS (
non-zero exit) on any unresolved ``$ref``, unknown schema type, or a required field
the guard cannot check — so the document is proven sufficient to generate a typed
client with no manual fixups (DOC-R5). The artifacts include ``isProblem(value)`` for
the RFC 9457 error contract and a per-resource guard for each object schema, so any
client can discriminate success vs ``problem+json`` purely by shape (DOC-R6).

Usage::

    uv run python -m tools.client_gen [OUT_FILE]   # default: generated/client.ts
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from tools.openapi_artifact import build_reference_document


class ClientGenError(Exception):
    """A document defect that makes typed-client generation impossible (DOC-R5)."""


def _sanitize(name: str) -> str:
    """A TS-safe identifier for a component schema name."""
    out = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)
    return out if out and not out[0].isdigit() else f"_{out}"


def _ref_name(ref: str, schemas: dict[str, Any]) -> str:
    """Resolve a ``$ref`` to its component name, failing on an unresolved target."""
    prefix = "#/components/schemas/"
    if not ref.startswith(prefix) or ref[len(prefix) :] not in schemas:
        raise ClientGenError(f"unresolved $ref: {ref}")
    return _sanitize(ref[len(prefix) :])


def _ts_composite(schema: dict[str, Any], schemas: dict[str, Any]) -> str | None:
    """The TS expression for a $ref/combinator/enum/const schema, else ``None``."""
    if "$ref" in schema:
        return _ref_name(schema["$ref"], schemas)
    for combinator, joiner in (("anyOf", " | "), ("oneOf", " | "), ("allOf", " & ")):
        if combinator in schema:
            return joiner.join(ts_type(member, schemas) for member in schema[combinator])
    if "enum" in schema:
        return " | ".join(
            "null" if value is None else f'"{value}"' if isinstance(value, str) else str(value)
            for value in schema["enum"]
        )
    if "const" in schema:
        value = schema["const"]
        return f'"{value}"' if isinstance(value, str) else str(value)
    return None


def _ts_object(schema: dict[str, Any], schemas: dict[str, Any]) -> str:
    """The TS expression for an object schema (inline shape or a record type)."""
    if "properties" in schema:
        required = set(schema.get("required", []))
        inner = ", ".join(
            f'"{prop}"{"" if prop in required else "?"}: ' + ts_type(sub, schemas)
            for prop, sub in schema["properties"].items()
        )
        return "{ " + inner + " }" if inner else "Record<string, unknown>"
    extra = schema.get("additionalProperties")
    if isinstance(extra, dict):
        return f"Record<string, {ts_type(extra, schemas)}>"
    return "Record<string, unknown>"


def ts_type(schema: dict[str, Any], schemas: dict[str, Any]) -> str:
    """The TypeScript type expression for one (sub)schema; unknown types FAIL."""
    composite = _ts_composite(schema, schemas)
    if composite is not None:
        return composite
    kind = schema.get("type")
    if isinstance(kind, list):
        return " | ".join(ts_type({**schema, "type": member}, schemas) for member in kind)
    if kind == "array":
        items = schema.get("items", {})
        return f"Array<{ts_type(items, schemas)}>" if items else "Array<unknown>"
    if kind == "object" or (kind is None and ("properties" in schema or schema == {})):
        return _ts_object(schema, schemas)
    simple = {
        "string": "string",
        "integer": "number",
        "number": "number",
        "boolean": "boolean",
        "null": "null",
    }
    if isinstance(kind, str) and kind in simple:
        return simple[kind]
    raise ClientGenError(f"unknown schema type: {schema!r}")


def _guard_check(prop: str, schema: dict[str, Any], schemas: dict[str, Any]) -> str:
    """One required-field runtime check inside a type guard (DOC-R5: every required
    field is guarded; an uncheckable one fails generation)."""
    accessor = f'(value as Record<string, unknown>)["{prop}"]'
    resolved = schema
    if "$ref" in schema:
        target = schema["$ref"].rsplit("/", 1)[-1]
        resolved = schemas.get(target, {})
    kind = resolved.get("type")
    if isinstance(kind, list):
        kind = next((k for k in kind if k != "null"), None)
    primitive = {"string": "string", "integer": "number", "number": "number", "boolean": "boolean"}
    if isinstance(kind, str) and kind in primitive:
        nullable = (
            "null" in (resolved.get("type") or [])
            if isinstance(resolved.get("type"), list)
            else False
        )
        check = f'typeof {accessor} === "{primitive[kind]}"'
        return f"({check} || {accessor} === null)" if nullable else check
    if kind == "array":
        return f"Array.isArray({accessor})"
    # objects / unions / enums / $ref-typed members: presence is the shape signal
    return f'"{prop}" in (value as Record<string, unknown>)'


def _emit_guard(name: str, schema: dict[str, Any], schemas: dict[str, Any]) -> str:
    """A per-resource shape guard ``is<Name>`` over the schema's required fields."""
    checks = ['typeof value === "object"', "value !== null"]
    for prop in schema.get("required", []):
        sub = schema.get("properties", {}).get(prop)
        if sub is None:
            raise ClientGenError(f"{name}: required field {prop!r} has no schema")
        checks.append(_guard_check(prop, sub, schemas))
    body = " &&\n    ".join(checks)
    return (
        f"export function is{name}(value: unknown): value is {name} {{\n"
        f"  return (\n    {body}\n  );\n}}\n"
    )


def generate(document: dict[str, Any]) -> str:
    """Render the full TS module: interfaces + guards for every component schema."""
    schemas: dict[str, Any] = document.get("components", {}).get("schemas", {})
    if not schemas:
        raise ClientGenError("document has no component schemas")
    parts = ["/* Generated by tools/client_gen.py from /v1/openapi.json — do not edit. */\n"]
    for raw_name in sorted(schemas):
        name = _sanitize(raw_name)
        schema = schemas[raw_name]
        if schema.get("type") == "object" or "properties" in schema:
            required = set(schema.get("required", []))
            lines = [f"export interface {name} {{"]
            for prop, sub in schema.get("properties", {}).items():
                optional = "" if prop in required else "?"
                lines.append(f'  "{prop}"{optional}: {ts_type(sub, schemas)};')
            lines.append("}\n")
            parts.append("\n".join(lines))
            parts.append(_emit_guard(name, schema, schemas))
        else:
            parts.append(f"export type {name} = {ts_type(schema, schemas)};\n")
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    """Generate the typed client; exit non-zero on any DOC-R5 defect."""
    args = argv if argv is not None else sys.argv[1:]
    out_path = Path(args[0]) if args else Path("generated") / "client.ts"
    try:
        rendered = generate(build_reference_document())
    except ClientGenError as exc:
        sys.stderr.write(f"client generation failed (DOC-R5): {exc}\n")
        return 1
    if "export function isProblem(" not in rendered:
        sys.stderr.write("client generation failed (DOC-R6): no isProblem guard\n")
        return 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered)
    sys.stdout.write(f"wrote {out_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
