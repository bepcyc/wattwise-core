"""Credential-probe + credential-store seams for the connections router (AUTH-R16/R17).

Factored out of :mod:`wattwise_core.api.routers.connections` (QUAL-R9 module-size
decomposition) as the cohesive "how the router reaches a secret" surface: the
read-only probe seam (AUTH-R17) and the envelope-encryption store seam (AUTH-R16).
Both are injectable :func:`fastapi.Depends` providers the app factory overrides with
the registered adapter's probe and the process credential store, so the connections
router never imports a concrete source adapter (ARCH-R22 / ONB-R4); tests inject fakes.

The connections router re-exports every public name here, so existing importers and
in-test ``dependency_overrides`` keys (``credential_probe`` / ``credential_sink``)
keep their identity unchanged.

Requirement IDs: AUTH-R16, AUTH-R17, API-R44, ARCH-R22, ONB-R4, QUAL-R13(d)/(e).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Annotated

from fastapi import Depends, Path
from pydantic import BaseModel, ConfigDict

from wattwise_core.api.copy import message as _copy
from wattwise_core.api.errors import FieldError, ProblemError

# --------------------------------------------------------- credential-probe seam


class CredentialProbeError(Exception):
    """A read-only credential probe rejected the supplied secret (AUTH-R17).

    Raised by the probe seam when the adapter's read-only check fails (bad key /
    revoked / unreachable-with-this-credential). Carries no secret material and no
    source-specific detail; the router maps it to ``422 credential-invalid``.
    """


class CredentialProbeUnavailable(CredentialProbeError):
    """No credential probe is wired — the connector is inert (API-R44, QUAL-R13(d)/(e)).

    A SUBCLASS of :class:`CredentialProbeError` (so any ``except CredentialProbeError``
    still fails closed), raised by the unconfigured default seam. The router catches THIS
    first and maps it to the DISTINCT ``422 connector-unavailable`` — a connector merely
    not enabled in this build is never reported as a bad credential, valid key or not.
    """


#: The probe seam: given a source key + the raw secret, run the adapter's MANDATORY
#: read-only check (AUTH-R17). Returns nothing on success; raises
#: :class:`CredentialProbeError` on a bad credential. The app factory overrides this
#: with the registered adapter's probe so this router never imports a named adapter
#: (ARCH-R22 / ONB-R4); tests inject a mock probe.
CredentialProbe = Callable[[str, str], Awaitable[None]]


async def _unconfigured_probe(source: str, secret: str) -> None:
    """Fail-closed default probe: refuse every credential until the factory wires one.

    The real probe is the registered adapter's read-only check, injected by the app
    factory. Until then no credential can pass (AUTH-R17, fail-closed). Because the
    connector is inert (not a bad key), this raises the distinct
    :class:`CredentialProbeUnavailable` → ``connector-unavailable`` (API-R44,
    QUAL-R13(d)/(e)), never blaming the athlete's credential.
    """
    raise CredentialProbeUnavailable(source)


def credential_probe() -> CredentialProbe:
    """Provide the credential-probe seam; the app factory overrides it (AUTH-R17)."""
    return _unconfigured_probe


# --------------------------------------------------------- credential-store seam


class CredentialSink(BaseModel):
    """The minimal credential-store surface this router needs (AUTH-R16).

    A structural seam (``store`` only) so the router depends on a capability, not on
    the security package's concrete store. The app factory binds the process
    :class:`~wattwise_core.security.credentials.CredentialStore`; tests inject a fake.
    Envelope encryption + opaque-ref issuance live behind it; the raw secret is never
    persisted here (AUTH-R16).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    store: Callable[[str], str]


def credential_sink() -> CredentialSink:
    """Provide the credential-store seam; keyless apps keep credential storage disabled.

    ``create_app`` overrides this dependency only when an envelope encryption root key is
    configured. Without that key, file upload/import remains available but API-key connector
    completion fails closed with a typed operator-actionable 4xx instead of storing a secret
    insecurely or surfacing a generic internal error.
    """
    raise ProblemError(
        "credential-storage-disabled",
        detail=_copy("connection.credential_storage_disabled_detail"),
        errors=[
            FieldError(
                code="credential_storage_disabled",
                message=_copy("connection.credential_storage_disabled"),
                parameter="source",
            )
        ],
    )


ProbeDep = Annotated[CredentialProbe, Depends(credential_probe)]
SinkDep = Annotated[CredentialSink, Depends(credential_sink)]
SourcePath = Annotated[str, Path(description="The connectable source key (catalog).")]


__all__ = [
    "CredentialProbe",
    "CredentialProbeError",
    "CredentialProbeUnavailable",
    "CredentialSink",
    "ProbeDep",
    "SinkDep",
    "SourcePath",
    "credential_probe",
    "credential_sink",
]
