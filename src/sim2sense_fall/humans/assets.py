"""Registry and loader for licence-gated body models and motion-capture sources.

Both assets this stage needs are registration-gated research releases, so the
pipeline has to be honest about three separate things:

1. **What the asset is** -- file name, version, expected contents, joint count.
2. **What the licence is** -- and therefore whether the file may even be fetched
   automatically (it may not: both require an account and licence acceptance).
3. **Whether it is actually present on this machine** -- which is a *runtime*
   fact, re-checked on every run and reported as such.

Consequently nothing here downloads anything. When a model is missing,
:func:`require_model` raises :class:`AssetUnavailable` carrying the registration
URL and the exact search roots that were tried, and every caller must degrade to
the procedural skeleton rather than pretending the licensed body was imported.

A note on pickle safety: SMPL `basicModel_*.pkl` files are Python pickles, and
unpickling executes arbitrary code. The loader only accepts files under an
explicitly configured asset root and refuses anything else, which is the same
trust boundary as running a model you downloaded from the official site.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import pickle
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .motion import body_frame_conversion
from .skeleton import SMPL_JOINT_NAMES, SkeletonTopology, skeleton_from_kintree_table

__all__ = [
    "ASSET_REGISTRY_FILENAME",
    "AssetUnavailable",
    "AssetRegistry",
    "ModelAsset",
    "MotionSource",
    "SmplModel",
    "audit_registry",
    "default_asset_roots",
    "load_asset_registry",
    "load_model_payload",
    "load_smpl_model",
    "select_body",
    "BodySelection",
    "require_model",
    "sha256_file",
]

LOGGER = logging.getLogger(__name__)

#: The two representations a human product may carry, mirrored in export.TrialRecord.
REPRESENTATION_SKIN_MESH = "smpl_skin_mesh"
REPRESENTATION_PROXY = "capsule_proxy_surface"

ASSET_REGISTRY_FILENAME = "configs/humans/assets.yaml"

#: Environment variable that prepends extra asset search roots.
ASSET_ROOT_ENV = "SIM2SENSE_HUMAN_ASSETS"

_MODEL_FIELDS = {
    "id",
    "family",
    "version",
    "filename",
    "filename_candidates",
    "subdirectory",
    "pickle_python_major",
    "contains_chumpy_reported",
    "requires_registration",
    "registration_url",
    "license",
    "license_url",
    "joint_count",
    "betas",
    "pose_parameters",
    "pickle_layout",
    "provenance",
    "verified",
    "notes",
}

_AUXILIARY_FIELDS = {
    "id",
    "kind",
    "filename",
    "filename_candidates",
    "subdirectory",
    "requires_registration",
    "registration_url",
    "license",
    "license_url",
    "provenance",
    "verified",
    "notes",
}

_MOTION_FIELDS = {
    "id",
    "dataset",
    "representation",
    "pose_parameters",
    "body_joints",
    "betas",
    "dmpls",
    "subsets",
    "requires_registration",
    "registration_url",
    "license",
    "license_url",
    "provenance",
    "verified",
    "notes",
}

_MODEL_FAMILIES = ("smpl", "smplh", "smplx")


class AssetUnavailable(RuntimeError):
    """Raised when a licence-gated asset is required but is not on this machine."""


@dataclass(frozen=True, slots=True)
class ModelAsset:
    """One downloadable body-model file and everything needed to audit it."""

    id: str
    family: str
    version: str
    filename: str
    subdirectory: str
    requires_registration: bool
    registration_url: str
    license: str
    license_url: str
    joint_count: int
    betas: int
    pose_parameters: int
    pickle_layout: str
    provenance: str
    verified: bool
    pickle_python_major: int = 3
    contains_chumpy_reported: bool = False
    filename_candidates: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        for name in (
            "id",
            "family",
            "version",
            "filename",
            "registration_url",
            "license",
            "license_url",
            "pickle_layout",
            "provenance",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"model asset field {name!r} must be a non-empty string")
        if self.family not in _MODEL_FAMILIES:
            raise ValueError(f"{self.id}: family must be one of {_MODEL_FAMILIES}")
        if isinstance(self.joint_count, bool) or self.joint_count <= 0:
            raise ValueError(f"{self.id}: joint_count must be a positive integer")
        if isinstance(self.betas, bool) or self.betas <= 0:
            raise ValueError(f"{self.id}: betas must be a positive integer")
        if isinstance(self.pose_parameters, bool) or self.pose_parameters <= 0:
            raise ValueError(f"{self.id}: pose_parameters must be a positive integer")
        if self.pose_parameters != self.joint_count * 3:
            raise ValueError(
                f"{self.id}: pose_parameters {self.pose_parameters} is not 3 x joint_count "
                f"{self.joint_count}"
            )
        for name in ("requires_registration", "verified"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{self.id}: {name} must be a boolean")
        if Path(self.filename).name != self.filename:
            raise ValueError(f"{self.id}: filename must not contain a directory")
        for candidate in self.filename_candidates:
            if not isinstance(candidate, str) or Path(candidate).name != candidate:
                raise ValueError(
                    f"{self.id}: filename candidate {candidate!r} must be a bare file name"
                )
        if isinstance(self.pickle_python_major, bool) or self.pickle_python_major not in (1, 2, 3):
            raise ValueError(
                f"{self.id}: pickle_python_major must be 1, 2 or 3, got "
                f"{self.pickle_python_major!r}"
            )
        if not isinstance(self.contains_chumpy_reported, bool):
            raise ValueError(f"{self.id}: contains_chumpy_reported must be a boolean")
        if not isinstance(self.subdirectory, str):
            raise ValueError(f"{self.id}: subdirectory must be a string (may be empty)")
        if Path(self.subdirectory).is_absolute() or ".." in Path(self.subdirectory).parts:
            raise ValueError(f"{self.id}: subdirectory must be a relative path without '..'")

    @property
    def all_filenames(self) -> tuple[str, ...]:
        """Every file name this entry accepts, primary first, deduplicated.

        The official download page does not publish file names, so the archive layout
        has to be discovered after downloading. Declaring candidates means a name
        mismatch shows up as "found under an alternate name" rather than as "asset
        missing" -- which would look identical to not having downloaded it at all.
        """

        seen: list[str] = []
        for name in (self.filename, *self.filename_candidates):
            if name and name not in seen:
                seen.append(name)
        return tuple(seen)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "family": self.family,
            "version": self.version,
            "filename": self.filename,
            "filename_candidates": list(self.filename_candidates),
            "all_filenames": list(self.all_filenames),
            "subdirectory": self.subdirectory,
            "requires_registration": self.requires_registration,
            "registration_url": self.registration_url,
            "license": self.license,
            "license_url": self.license_url,
            "joint_count": self.joint_count,
            "betas": self.betas,
            "pose_parameters": self.pose_parameters,
            "pickle_layout": self.pickle_layout,
            "pickle_python_major": self.pickle_python_major,
            "contains_chumpy_reported": self.contains_chumpy_reported,
            "provenance": self.provenance,
            "verified": self.verified,
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class AuxiliaryAsset:
    """A downloadable file that supports the body model but is not one.

    The SMPL download page also offers a UV map in OBJ format. It carries no joints
    and no skinning weights, so it does not belong in :class:`ModelAsset`, but it is
    still a registration-gated file this project will want for texturing -- so it is
    declared and audited on the same footing instead of being left in a comment.
    """

    id: str
    kind: str
    filename: str
    subdirectory: str
    requires_registration: bool
    registration_url: str
    license: str
    license_url: str
    provenance: str
    verified: bool
    filename_candidates: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        for name in (
            "id",
            "kind",
            "filename",
            "registration_url",
            "license",
            "license_url",
            "provenance",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"auxiliary asset field {name!r} must be a non-empty string")
        if not self.kind.strip():
            raise ValueError(f"{self.id}: kind must be a non-empty string")
        if Path(self.filename).name != self.filename:
            raise ValueError(f"{self.id}: filename must not contain a directory")
        for name in ("requires_registration", "verified"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{self.id}: {name} must be a boolean")
        if Path(self.subdirectory).is_absolute() or ".." in Path(self.subdirectory).parts:
            raise ValueError(f"{self.id}: subdirectory must be a relative path without '..'")

    @property
    def all_filenames(self) -> tuple[str, ...]:
        seen: list[str] = []
        for name in (self.filename, *self.filename_candidates):
            if name and name not in seen:
                seen.append(name)
        return tuple(seen)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "filename": self.filename,
            "filename_candidates": list(self.filename_candidates),
            "all_filenames": list(self.all_filenames),
            "subdirectory": self.subdirectory,
            "requires_registration": self.requires_registration,
            "registration_url": self.registration_url,
            "license": self.license,
            "license_url": self.license_url,
            "provenance": self.provenance,
            "verified": self.verified,
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class MotionSource:
    """One motion-capture collection and how its representation maps to SMPL."""

    id: str
    dataset: str
    representation: str
    pose_parameters: int
    body_joints: int
    betas: int
    dmpls: int
    subsets: tuple[str, ...]
    requires_registration: bool
    registration_url: str
    license: str
    license_url: str
    provenance: str
    verified: bool
    notes: str = ""

    def __post_init__(self) -> None:
        for name in (
            "id",
            "dataset",
            "representation",
            "registration_url",
            "license",
            "license_url",
            "provenance",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"motion source field {name!r} must be a non-empty string")
        for name in ("pose_parameters", "body_joints", "betas", "dmpls"):
            value = getattr(self, name)
            if isinstance(value, bool) or value < 0:
                raise ValueError(f"{self.id}: {name} must be a non-negative integer")
        if self.pose_parameters <= 0:
            raise ValueError(f"{self.id}: pose_parameters must be positive")
        if self.pose_parameters % 3:
            raise ValueError(f"{self.id}: pose_parameters must be a multiple of 3")
        if self.pose_parameters // 3 < self.body_joints:
            raise ValueError(f"{self.id}: pose_parameters describe fewer joints than body_joints")
        for name in ("requires_registration", "verified"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{self.id}: {name} must be a boolean")
        if not isinstance(self.subsets, tuple) or not self.subsets:
            raise ValueError(f"{self.id}: subsets must be a non-empty tuple")

    @property
    def hand_joints(self) -> int:
        """Joints beyond the body block, i.e. fingers."""

        return self.pose_parameters // 3 - self.body_joints


@dataclass(frozen=True, slots=True)
class AssetRegistry:
    """The declared assets plus the roots that are searched for them."""

    models: Mapping[str, ModelAsset]
    motions: Mapping[str, MotionSource]
    roots: tuple[Path, ...]
    auxiliary: Mapping[str, AuxiliaryAsset] = field(default_factory=dict)
    source_path: Path | None = None
    notes: str = field(default="")

    def __post_init__(self) -> None:
        if not self.models and not self.motions:
            raise ValueError("asset registry declares no models and no motions")
        if len(set(self.roots)) != len(self.roots):
            raise ValueError("asset registry has duplicate search roots")
        if not self.roots:
            raise ValueError("asset registry must declare at least one search root")
        for root in self.roots:
            if not isinstance(root, Path):
                raise ValueError(f"asset search roots must be Path objects, got {root!r}")

    def model(self, model_id: str) -> ModelAsset:
        if model_id not in self.models:
            raise KeyError(f"unknown model asset {model_id!r}; known: {sorted(self.models)}")
        return self.models[model_id]

    def aux(self, auxiliary_id: str) -> AuxiliaryAsset:
        if auxiliary_id not in self.auxiliary:
            raise KeyError(
                f"unknown auxiliary asset {auxiliary_id!r}; known: {sorted(self.auxiliary)}"
            )
        return self.auxiliary[auxiliary_id]

    def motion(self, motion_id: str) -> MotionSource:
        if motion_id not in self.motions:
            raise KeyError(f"unknown motion source {motion_id!r}; known: {sorted(self.motions)}")
        return self.motions[motion_id]


def default_asset_roots(project_root: Path | None = None) -> tuple[Path, ...]:
    """Search roots for licensed assets, most specific first.

    Assets never live inside the repository tree that is committed: they are
    large, licence-gated, and must not be redistributed.
    """

    roots: list[Path] = []
    override = os.environ.get(ASSET_ROOT_ENV)
    if override:
        roots.extend(Path(entry).expanduser() for entry in override.split(os.pathsep) if entry)
    if project_root is not None:
        roots.append(project_root / "data" / "humans")
    roots.append(Path.home() / ".cache" / "sim2sense-fall" / "humans")
    seen: list[Path] = []
    for root in roots:
        if root not in seen:
            seen.append(root)
    return tuple(seen)


def _require_string(mapping: Mapping[str, Any], key: str, *, where: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}: {key!r} must be a non-empty string")
    return value


def _require_int(
    mapping: Mapping[str, Any],
    key: str,
    *,
    where: str,
    minimum: int = 0,
    default: int | None = None,
) -> int:
    if key not in mapping:
        if default is None:
            raise ValueError(f"{where}: missing required integer field {key!r}")
        return int(default)
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{where}: {key!r} must be an integer >= {minimum}, got {value!r}")
    return value


def _require_bool(
    mapping: Mapping[str, Any], key: str, *, where: str, default: bool | None = None
) -> bool:
    if key not in mapping:
        if default is None:
            raise ValueError(f"{where}: missing required boolean field {key!r}")
        return bool(default)
    value = mapping[key]
    if not isinstance(value, bool):
        raise ValueError(f"{where}: {key!r} must be a boolean, got {value!r}")
    return value


def _require_names(mapping: Mapping[str, Any], key: str, *, where: str) -> tuple[str, ...]:
    value = mapping.get(key)
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{where}: {key!r} must be a list of file names")
    return tuple(str(entry) for entry in value)


def _reject_unknown(mapping: Mapping[str, Any], allowed: set[str], *, where: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"{where}: unsupported keys {unknown}")


def _model_from_mapping(payload: Mapping[str, Any]) -> ModelAsset:
    if not isinstance(payload, Mapping):
        raise ValueError("each model entry must be a mapping")
    identifier = _require_string(payload, "id", where="model")
    _reject_unknown(payload, _MODEL_FIELDS, where=f"model {identifier}")
    return ModelAsset(
        id=identifier,
        family=_require_string(payload, "family", where=f"model {identifier}"),
        version=_require_string(payload, "version", where=f"model {identifier}"),
        filename=_require_string(payload, "filename", where=f"model {identifier}"),
        subdirectory=payload.get("subdirectory", "") or "",
        requires_registration=_require_bool(
            payload, "requires_registration", where=f"model {identifier}"
        ),
        registration_url=_require_string(payload, "registration_url", where=f"model {identifier}"),
        license=_require_string(payload, "license", where=f"model {identifier}"),
        license_url=_require_string(payload, "license_url", where=f"model {identifier}"),
        joint_count=_require_int(payload, "joint_count", where=f"model {identifier}", minimum=1),
        betas=_require_int(payload, "betas", where=f"model {identifier}", minimum=1),
        pose_parameters=_require_int(
            payload, "pose_parameters", where=f"model {identifier}", minimum=3
        ),
        pickle_layout=_require_string(payload, "pickle_layout", where=f"model {identifier}"),
        provenance=_require_string(payload, "provenance", where=f"model {identifier}"),
        verified=_require_bool(payload, "verified", where=f"model {identifier}"),
        # Optional: default to a modern pickle with no chumpy wrappers, which is the
        # safe assumption for a converted file, and which the loader re-checks anyway.
        pickle_python_major=_require_int(
            payload,
            "pickle_python_major",
            where=f"model {identifier}",
            minimum=1,
            default=3,
        ),
        contains_chumpy_reported=_require_bool(
            payload, "contains_chumpy_reported", where=f"model {identifier}", default=False
        ),
        filename_candidates=_require_names(
            payload, "filename_candidates", where=f"model {identifier}"
        ),
        notes=str(payload.get("notes", "")),
    )


def _auxiliary_from_mapping(payload: Mapping[str, Any]) -> AuxiliaryAsset:
    if not isinstance(payload, Mapping):
        raise ValueError("each auxiliary entry must be a mapping")
    identifier = _require_string(payload, "id", where="auxiliary")
    _reject_unknown(payload, _AUXILIARY_FIELDS, where=f"auxiliary {identifier}")
    return AuxiliaryAsset(
        id=identifier,
        kind=_require_string(payload, "kind", where=f"auxiliary {identifier}"),
        filename=_require_string(payload, "filename", where=f"auxiliary {identifier}"),
        subdirectory=payload.get("subdirectory", "") or "",
        requires_registration=_require_bool(
            payload, "requires_registration", where=f"auxiliary {identifier}"
        ),
        registration_url=_require_string(
            payload, "registration_url", where=f"auxiliary {identifier}"
        ),
        license=_require_string(payload, "license", where=f"auxiliary {identifier}"),
        license_url=_require_string(payload, "license_url", where=f"auxiliary {identifier}"),
        provenance=_require_string(payload, "provenance", where=f"auxiliary {identifier}"),
        verified=_require_bool(payload, "verified", where=f"auxiliary {identifier}"),
        filename_candidates=_require_names(
            payload, "filename_candidates", where=f"auxiliary {identifier}"
        ),
        notes=str(payload.get("notes", "")),
    )


def _motion_from_mapping(payload: Mapping[str, Any]) -> MotionSource:
    if not isinstance(payload, Mapping):
        raise ValueError("each motion entry must be a mapping")
    identifier = _require_string(payload, "id", where="motion")
    _reject_unknown(payload, _MOTION_FIELDS, where=f"motion {identifier}")
    subsets = payload.get("subsets")
    if not isinstance(subsets, list) or not subsets:
        raise ValueError(f"motion {identifier}: 'subsets' must be a non-empty list")
    return MotionSource(
        id=identifier,
        dataset=_require_string(payload, "dataset", where=f"motion {identifier}"),
        representation=_require_string(payload, "representation", where=f"motion {identifier}"),
        pose_parameters=_require_int(
            payload, "pose_parameters", where=f"motion {identifier}", minimum=3
        ),
        body_joints=_require_int(payload, "body_joints", where=f"motion {identifier}", minimum=1),
        betas=_require_int(payload, "betas", where=f"motion {identifier}"),
        dmpls=_require_int(payload, "dmpls", where=f"motion {identifier}"),
        subsets=tuple(str(entry) for entry in subsets),
        requires_registration=_require_bool(
            payload, "requires_registration", where=f"motion {identifier}"
        ),
        registration_url=_require_string(payload, "registration_url", where=f"motion {identifier}"),
        license=_require_string(payload, "license", where=f"motion {identifier}"),
        license_url=_require_string(payload, "license_url", where=f"motion {identifier}"),
        provenance=_require_string(payload, "provenance", where=f"motion {identifier}"),
        verified=_require_bool(payload, "verified", where=f"motion {identifier}"),
        notes=str(payload.get("notes", "")),
    )


def asset_registry_from_mapping(
    payload: Mapping[str, Any], *, project_root: Path | None = None, source_path: Path | None = None
) -> AssetRegistry:
    """Build an :class:`AssetRegistry` from a parsed YAML mapping."""

    if not isinstance(payload, Mapping):
        raise ValueError("asset registry must be a mapping")
    _reject_unknown(
        payload,
        {"version", "notes", "roots", "models", "motions", "auxiliary"},
        where="registry",
    )
    models_payload = payload.get("models")
    if not isinstance(models_payload, list) or not models_payload:
        raise ValueError("asset registry must declare a non-empty 'models' list")
    motions_payload = payload.get("motions")
    if not isinstance(motions_payload, list) or not motions_payload:
        raise ValueError("asset registry must declare a non-empty 'motions' list")
    models = [_model_from_mapping(entry) for entry in models_payload]
    motions = [_motion_from_mapping(entry) for entry in motions_payload]
    auxiliary_payload = payload.get("auxiliary", [])
    if not isinstance(auxiliary_payload, list):
        raise ValueError("'auxiliary' must be a list")
    auxiliary = [_auxiliary_from_mapping(entry) for entry in auxiliary_payload]
    for collection, label in (
        (models, "model"),
        (motions, "motion"),
        (auxiliary, "auxiliary"),
    ):
        identifiers = [entry.id for entry in collection]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError(f"duplicate {label} ids in {sorted(identifiers)}")
    roots_payload = payload.get("roots")
    if roots_payload is None:
        roots = default_asset_roots(project_root)
    else:
        if not isinstance(roots_payload, list) or not roots_payload:
            raise ValueError("'roots' must be a non-empty list when present")
        base = source_path.parent if source_path is not None else Path.cwd()
        roots = tuple(
            _resolve_root(str(entry), base=base, project_root=project_root)
            for entry in roots_payload
        )
    return AssetRegistry(
        models={entry.id: entry for entry in models},
        motions={entry.id: entry for entry in motions},
        auxiliary={entry.id: entry for entry in auxiliary},
        roots=roots,
        source_path=source_path,
        notes=str(payload.get("notes", "")),
    )


def _resolve_root(value: str, *, base: Path, project_root: Path | None) -> Path:
    text = value.strip()
    if not text:
        raise ValueError("asset root entries must be non-empty strings")
    if text.startswith("~"):
        return Path(text).expanduser()
    for token in ("$PROJECT", "%PROJECT%"):
        if text == token or text.startswith(f"{token}/"):
            if project_root is None:
                raise ValueError("a '$PROJECT' root needs the project root to be known")
            remainder = text[len(token) :].lstrip("/")
            base_path = project_root.resolve()
            return (base_path / remainder).resolve() if remainder else base_path
    candidate = Path(text)
    if candidate.is_absolute():
        return candidate
    return (base / candidate).resolve()


def load_asset_registry(path: str | Path, *, project_root: Path | None = None) -> AssetRegistry:
    """Load and strictly validate the asset registry YAML."""

    registry_path = Path(path)
    if not registry_path.is_file():
        raise FileNotFoundError(f"asset registry not found: {registry_path}")
    with registry_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{registry_path}: top level must be a mapping")
    return asset_registry_from_mapping(
        payload, project_root=project_root, source_path=registry_path
    )


def candidate_paths(registry: AssetRegistry, model_id: str) -> tuple[Path, ...]:
    """Every path where ``model_id`` could live, in search order."""

    model = registry.model(model_id)
    base = Path(model.subdirectory) if model.subdirectory else Path()
    return tuple((root / base / name) for root in registry.roots for name in model.all_filenames)


def find_model(registry: AssetRegistry, model_id: str) -> Path | None:
    """Return the first existing file for ``model_id``, or ``None``."""

    for candidate in candidate_paths(registry, model_id):
        if candidate.is_file():
            return candidate
    return None


def require_model(registry: AssetRegistry, model_id: str) -> Path:
    """Return the model file path or fail with an actionable message."""

    model = registry.model(model_id)
    found = find_model(registry, model_id)
    if found is not None:
        return found
    searched = "\n".join(f"    - {path}" for path in candidate_paths(registry, model_id))
    raise AssetUnavailable(
        f"body model {model_id!r} ({model.family} v{model.version}, "
        f"{model.filename}) is not present on this machine.\n"
        f"  licence: {model.license}\n"
        f"  register at: {model.registration_url}\n"
        f"  searched:\n{searched}\n"
        f"  set {ASSET_ROOT_ENV} or drop the file into one of the roots above.\n"
        "  This model is registration-gated, so the pipeline does not download it. "
        "Re-run with --allow-procedural-skeleton to use the procedural stand-in; "
        "artefacts built that way record honest provenance."
    )


def sha256_file(path: str | Path, *, chunk_bytes: int = 1 << 20) -> str:
    """Streaming SHA-256 so large model files do not have to fit in memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ModelAudit:
    """Resolved availability of one declared model asset."""

    model_id: str
    available: bool
    path: Path | None
    sha256: str | None
    size_bytes: int | None
    verified_metadata: bool
    license: str
    registration_url: str
    message: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "available": self.available,
            "path": None if self.path is None else str(self.path),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "metadata_verified": self.verified_metadata,
            "license": self.license,
            "registration_url": self.registration_url,
            "message": self.message,
        }


def audit_registry(registry: AssetRegistry, *, hash_files: bool = False) -> dict[str, Any]:
    """Report which declared assets are actually present, without downloading any.

    ``hash_files=False`` by default because hashing a 100 MB model on every dry-run
    is wasted work; the simulation entry points turn it on when they record
    provenance.
    """

    audits: list[ModelAudit] = []
    for model_id, model in sorted(registry.models.items()):
        path = find_model(registry, model_id)
        if path is None:
            audits.append(
                ModelAudit(
                    model_id=model_id,
                    available=False,
                    path=None,
                    sha256=None,
                    size_bytes=None,
                    verified_metadata=model.verified,
                    license=model.license,
                    registration_url=model.registration_url,
                    message=(
                        "not present; registration-gated so it cannot be fetched automatically"
                    ),
                )
            )
            continue
        audits.append(
            ModelAudit(
                model_id=model_id,
                available=True,
                path=path,
                sha256=sha256_file(path) if hash_files else None,
                size_bytes=path.stat().st_size,
                verified_metadata=model.verified,
                license=model.license,
                registration_url=model.registration_url,
                message="present",
            )
        )
    return {
        "roots": [str(root) for root in registry.roots],
        "registry_source": None if registry.source_path is None else str(registry.source_path),
        "models": [audit.as_dict() for audit in audits],
        "available_model_count": sum(audit.available for audit in audits),
        "auxiliary": [
            {
                **entry.as_dict(),
                "paths": [
                    str(root / (Path(entry.subdirectory) if entry.subdirectory else Path()) / name)
                    for root in registry.roots
                    for name in entry.all_filenames
                ],
                "available": any(
                    (
                        root / (Path(entry.subdirectory) if entry.subdirectory else Path()) / name
                    ).is_file()
                    for root in registry.roots
                    for name in entry.all_filenames
                ),
            }
            for entry in sorted(registry.auxiliary.values(), key=lambda item: item.id)
        ],
        "models_detail": [
            model.as_dict() for model in sorted(registry.models.values(), key=lambda item: item.id)
        ],
        "motions": [
            {
                "motion_id": source.id,
                "dataset": source.dataset,
                "representation": source.representation,
                "body_joints": source.body_joints,
                "hand_joints": source.hand_joints,
                "betas": source.betas,
                "dmpls": source.dmpls,
                "subsets": list(source.subsets),
                "requires_registration": source.requires_registration,
                "registration_url": source.registration_url,
                "license": source.license,
                "metadata_verified": source.verified,
                "notes": source.notes,
            }
            for source in sorted(registry.motions.values(), key=lambda entry: entry.id)
        ],
        "notes": registry.notes,
    }


@dataclass(frozen=True, slots=True)
class SmplModel:
    """The parts of a loaded SMPL model this pipeline actually consumes.

    Templates, shape directions and skinning weights are kept because the mesh
    stage needs them; the joint centres and the topology are derived here so the
    rig and the motion contract agree with the file rather than with a constant.
    """

    path: Path
    topology: SkeletonTopology
    joint_positions: tuple[tuple[float, float, float], ...]
    vertex_count: int
    face_count: int
    has_skinning_weights: bool
    has_shape_directions: bool
    source_sha256: str
    declared_asset_id: str | None = None
    pickle_python_major: int = 3
    #: Rest geometry, already rotated into the pipeline's Z-up frame.
    vertices: np.ndarray | None = None
    faces: np.ndarray | None = None
    weights: np.ndarray | None = None
    shape_directions: np.ndarray | None = None
    up_axis_source: str = "y"
    up_axis_target: str = "z"
    #: The source file's full axis triple, e.g. ``up=y forward=z left=x``. Unlike a bare
    #: up axis this pins the horizontal convention, so it is what makes the import
    #: reproducible; a consumer can rebuild the same basis from it.
    source_frame: str = ""
    stature_m: float = 0.0
    beta_count: int = 0

    @property
    def joint_names(self) -> tuple[str, ...]:
        return self.topology.joint_names

    @property
    def has_mesh(self) -> bool:
        """True only when vertices, faces and skinning weights all loaded.

        This, not "a registered path exists", is what lets a product claim the real
        skin mesh was imported.
        """

        return (
            self.vertices is not None
            and self.faces is not None
            and self.weights is not None
            and self.weights.shape == (self.vertices.shape[0], self.topology.joint_count)
        )

    def mesh(self) -> Any:
        """The loaded skin as an :class:`~sim2sense_fall.humans.skinning.SmplMesh`.

        Raises rather than synthesising a stand-in: a body with no real weights is the
        capsule proxy, and it must be reported as that.
        """

        if not self.has_mesh:
            raise AssetUnavailable(
                f"{self.path}: no complete mesh (vertices, faces and {(self.topology.joint_count)}"
                "-column weights are required), so the skin mesh cannot be built"
            )
        from .skeleton import smpl_skeleton
        from .skinning import SmplMesh

        rest = np.asarray(self.joint_positions, dtype=np.float64)
        return SmplMesh(
            vertices=np.asarray(self.vertices, dtype=np.float64),
            faces=np.asarray(self.faces),
            weights=np.asarray(self.weights, dtype=np.float64),
            topology=smpl_skeleton(),
            rest_joints=rest,
            shape_directions=None
            if self.shape_directions is None
            else np.asarray(self.shape_directions, dtype=np.float64),
            source=self.declared_asset_id or self.path.name,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "joint_count": self.topology.joint_count,
            "joint_names": list(self.topology.joint_names),
            "parents": list(self.topology.parents),
            "vertex_count": self.vertex_count,
            "face_count": self.face_count,
            "has_skinning_weights": self.has_skinning_weights,
            "has_shape_directions": self.has_shape_directions,
            "source_sha256": self.source_sha256,
            "declared_asset_id": self.declared_asset_id,
            "pickle_python_major": self.pickle_python_major,
            "has_mesh": self.has_mesh,
            "up_axis": f"{self.up_axis_source}->{self.up_axis_target}",
            "source_frame": self.source_frame,
            "stature_m": round(self.stature_m, 6),
            "beta_count": self.beta_count,
        }


class _ChumpyPlaceholder:
    """Stand-in for a ``chumpy`` object, so a pickle that references chumpy can load.

    The SMPL downloads are labelled "for Python 2.7" and the ``shapedirs`` /
    ``posedirs`` entries are commonly stored as ``chumpy`` arrays. ``chumpy`` is
    unmaintained and does not install cleanly on modern Python, so instead of making
    it a hard dependency the unpickler substitutes this class and
    :func:`_coerce_array` digs the underlying NumPy array back out.

    If the wrapped array cannot be found the loader raises and says so, rather than
    silently returning a placeholder that would blow up later as an obscure shape
    error.
    """

    _WRAPPED_ATTRIBUTES = ("x", "r", "_data", "v", "a")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._chumpy_args = args
        self._chumpy_kwargs = kwargs

    def __setstate__(self, state: Any) -> None:
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.__dict__["_chumpy_state"] = state

    def __reduce__(self) -> tuple[Any, ...]:
        # Only needed so the placeholder can itself be pickled in tests.
        return (_ChumpyPlaceholder, (), self.__dict__)

    def unwrap(self) -> np.ndarray | None:
        for attribute in self._WRAPPED_ATTRIBUTES:
            candidate = self.__dict__.get(attribute)
            if candidate is None:
                continue
            try:
                array = np.asarray(candidate)
            except Exception:  # noqa: BLE001 - tried and rejected
                continue
            if array.dtype != object and array.size:
                return array
        return None

    def describe(self) -> str:
        return f"chumpy-backed object with attributes {sorted(self.__dict__)}"


class _ModelUnpickler(pickle.Unpickler):
    """Unpickler that resolves ``chumpy`` classes to a NumPy-carrying placeholder."""

    _PLACEHOLDER_MODULES = ("chumpy", "ch", "scipy.sparse")

    def find_class(self, module: str, name: str) -> Any:
        root = module.split(".")[0]
        if root in self._PLACEHOLDER_MODULES:
            return _ChumpyPlaceholder
        return super().find_class(module, name)


def _unpickle_with_encoding(path: Path, encoding: str) -> Mapping[str, Any]:
    with path.open("rb") as handle:
        unpickler = _ModelUnpickler(handle, encoding=encoding)
        payload = unpickler.load()
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path}: expected the model pickle to contain a mapping")
    return payload


def load_model_payload(path: Path, *, python_major: int = 3) -> Mapping[str, Any]:
    """Load a model pickle, retrying with ``latin1`` for Python 2 files.

    ``python_major`` is the declared pickle Python version from the asset registry; it
    selects which decoding is tried first. Both are tried, because a wrong declaration
    must not look like a corrupt file.
    """

    order = ["latin1", "ASCII"] if python_major == 2 else ["ASCII", "latin1"]
    errors: list[str] = []
    for encoding in order:
        try:
            payload = _unpickle_with_encoding(path, encoding)
        except (UnicodeDecodeError, ValueError, pickle.UnpicklingError) as exc:
            errors.append(f"{encoding}: {type(exc).__name__}: {exc}")
            continue
        LOGGER.info("%s loaded with pickle encoding %s", path.name, encoding)
        return payload
    raise ValueError(
        f"{path} could not be loaded as a pickle with any encoding "
        f"({'; '.join(errors)}). The released SMPL files are Python 2.7 pickles; check "
        "that the download completed and is not a partial archive."
    )


def _as_float_array(value: Any, name: str, *, ndim: int | None = None) -> np.ndarray:
    """Coerce a model value to a float array, unwrapping chumpy wrappers.

    Handles four shapes the released files actually use: a plain array, a ``chumpy``
    wrapper carrying the array in one of its attributes, an object array of chumpy
    wrappers, and a value that exposes ``.r`` as its numpy view.
    """

    array = _coerce_array(value, name)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name}: expected {ndim} dimensions, got {array.ndim}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name}: contains a non-finite value")
    return array.astype(np.float64, copy=False)


def _coerce_array(value: Any, name: str, *, depth: int = 0) -> np.ndarray:
    """Turn one stored model value into a plain NumPy array.

       The released v1.1.0 pickles do not store everything as arrays: ``J_regressor`` is a
       ``scipy`` sparse matrix and several entries are ``chumpy`` nodes. A sparse matrix
       also converts to a zero-dimensional object array, so the object branch below would
    otherwise recurse into the same object forever -- which is exactly what happened on the
       first load of a real model file. Sparse is therefore expanded first, and the object
       branch is depth-bounded so an unexpected layout fails with a message.
    """

    if hasattr(value, "toarray") and not isinstance(value, np.ndarray):
        value = np.asarray(value.toarray())
    if isinstance(value, _ChumpyPlaceholder):
        unwrapped = value.unwrap()
        if unwrapped is None:
            raise ValueError(
                f"{name}: this model stores a chumpy object whose array could not be "
                f"recovered ({value.describe()}). chumpy is unmaintained, so this loader "
                "substitutes it; if the file uses a chumpy layout this build does not "
                "know, convert the model once with the reference SMPL loader and pass "
                "the converted plain-NumPy pickle instead. The pipeline will not silently "
                "depend on chumpy."
            )
        return unwrapped
    if hasattr(value, "r") and not isinstance(value, np.ndarray):
        candidate = value.r
        if not np.isscalar(candidate):
            value = candidate
    array = np.asarray(value)
    if array.dtype != object:
        return array
    if depth >= 8 or array.shape == ():
        raise ValueError(
            f"{name}: stored as an object array this loader cannot flatten "
            f"(shape {array.shape}, type {type(value).__name__}). Convert the model once "
            "with the reference SMPL loader and pass the plain-NumPy pickle instead."
        )
    # An object array of chumpy wrappers, as some of the released files store.
    flat = [_coerce_array(entry, name, depth=depth + 1) for entry in array.ravel().tolist()]
    shapes = {entry.shape for entry in flat}
    if len(shapes) != 1:
        raise ValueError(
            f"{name}: object array holds inconsistent shapes {sorted(shapes)}; cannot assemble it"
        )
    return np.stack(flat).reshape((*array.shape, *flat[0].shape))


def declared_model_for(registry: AssetRegistry, path: Path) -> ModelAsset | None:
    """Which declared entry a resolved file path belongs to, if any."""

    resolved = path.resolve()
    for model_id in registry.models:
        if resolved in {candidate.resolve() for candidate in candidate_paths(registry, model_id)}:
            return registry.model(model_id)
    return None


def load_smpl_model(
    path: str | Path,
    *,
    registry: AssetRegistry | None = None,
    python_major: int | None = None,
) -> SmplModel:
    """Load a SMPL ``basicModel_*.pkl`` and derive its skeleton.

    Refuses files outside a configured asset root so that the trust boundary for
    ``pickle.load`` is explicit rather than implicit. Tolerates the released files'
    Python 2 pickle encoding and ``chumpy`` array wrappers; if a chumpy layout cannot
    be unwrapped, the error says what to do instead of failing later with a shape
    mismatch.
    """

    model_path = Path(path).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"model file not found: {model_path}")
    if registry is not None:
        allowed = [
            root.resolve() for root in registry.roots if model_path.is_relative_to(root.resolve())
        ]
        if not allowed:
            raise AssetUnavailable(
                f"{model_path} is outside every configured asset root "
                f"({[str(r) for r in registry.roots]}). Refusing to unpickle it. "
                f"Move it into an asset root or set {ASSET_ROOT_ENV}."
            )
    declared = declared_model_for(registry, model_path) if registry is not None else None
    pickle_python_major = (
        python_major
        if python_major is not None
        else (declared.pickle_python_major if declared is not None else 3)
    )
    payload = load_model_payload(model_path, python_major=pickle_python_major)
    for key in ("v_template", "f", "kintree_table", "J_regressor"):
        if key not in payload:
            raise ValueError(f"{model_path}: missing required key {key!r} (see pickle_layout)")

    template = _as_float_array(payload["v_template"], "v_template", ndim=2)
    faces_raw = _as_float_array(payload["f"], "f", ndim=2)
    if not np.all(faces_raw == np.round(faces_raw)):
        raise ValueError(f"{model_path}: f holds non-integer indices; faces cannot be built")
    faces = faces_raw.round().astype(np.int64)
    regressor = _as_float_array(payload["J_regressor"], "J_regressor", ndim=2)
    if template.shape[1] != 3:
        raise ValueError(f"v_template must be (V, 3), got {template.shape}")
    if faces.shape[1] != 3:
        raise ValueError(f"f must be (F, 3), got {faces.shape}")
    if regressor.shape[0] != len(SMPL_JOINT_NAMES):
        raise ValueError(
            f"J_regressor must have {len(SMPL_JOINT_NAMES)} rows, got {regressor.shape[0]}"
        )
    if regressor.shape[1] != template.shape[0]:
        raise ValueError("J_regressor columns must match v_template vertex count")
    if faces.size and (int(faces.max()) >= template.shape[0] or int(faces.min()) < 0):
        raise ValueError(
            f"{model_path}: face indices run [{int(faces.min())}, {int(faces.max())}] but the "
            f"template has {template.shape[0]} vertices"
        )

    weights = None
    if "weights" in payload:
        weights = _as_float_array(payload["weights"], "weights", ndim=2)
        if weights.shape != (template.shape[0], len(SMPL_JOINT_NAMES)):
            raise ValueError(
                f"weights must be ({template.shape[0]}, {len(SMPL_JOINT_NAMES)}), got "
                f"{weights.shape}"
            )
        deviation = float(np.abs(weights.sum(axis=1) - 1.0).max())
        if deviation > 1e-6:
            raise ValueError(
                f"{model_path}: skinning weights do not sum to 1 per vertex (largest deviation "
                f"{deviation:.3e}); a mesh built on them would tear or collapse"
            )
    shape_directions = None
    if "shapedirs" in payload:
        shape_directions = _as_float_array(payload["shapedirs"], "shapedirs", ndim=3)
        if shape_directions.shape[:2] != (template.shape[0], 3):
            raise ValueError(f"shapedirs must be (V, 3, B), got {shape_directions.shape}")

    topology = skeleton_from_kintree_table(payload["kintree_table"])
    joints = regressor @ template

    # Re-express the file in the pipeline body frame. Measuring only the up axis is not
    # enough: the released template's *lateral* axis (the shoulder span) is where the
    # pipeline expects forward, so a basis that only guarantees "up is up" yaws the body
    # 90 degrees and every exported mesh lies on its side while still being human-sized.
    # The frame is therefore derived from three independent anatomical directions, all
    # read off the joint positions.
    source_frame = _derive_body_frame(joints, topology, model_path)
    basis = body_frame_conversion(
        up=source_frame.up,
        forward=source_frame.forward,
        left=source_frame.left,
        source=str(model_path),
    )
    template = (basis @ template.T).T
    joints = (basis @ joints.T).T
    if shape_directions is not None:
        shape_directions = np.einsum("ij,vjk->vik", basis, shape_directions)
    heights = template[:, 2]
    stature = float(heights.max() - heights.min())
    if not 0.8 <= stature <= 2.5:
        raise ValueError(
            f"{model_path}: the rest template spans {stature:.3f} m along the derived "
            f"{TARGET_UP_AXIS.upper()}-up axis. A human body model should span roughly 1.7 m; "
            "a centimetres-scale or unit-mismatched file must not be imported silently."
        )
    _check_standing_frame(joints, topology, model_path, source_frame)
    LOGGER.info(
        "loaded %s: %d vertices, %d faces, %d joints, body frame %s, stature %.3f m",
        model_path,
        template.shape[0],
        faces.shape[0],
        topology.joint_count,
        source_frame.describe(),
        stature,
    )
    return SmplModel(
        path=model_path,
        topology=topology,
        joint_positions=tuple((float(r[0]), float(r[1]), float(r[2])) for r in joints),  # type: ignore[arg-type]
        vertex_count=int(template.shape[0]),
        face_count=int(faces.shape[0]),
        has_skinning_weights=weights is not None,
        has_shape_directions=shape_directions is not None,
        source_sha256=sha256_file(model_path),
        declared_asset_id=None if declared is None else declared.id,
        pickle_python_major=pickle_python_major,
        vertices=template,
        faces=faces,
        weights=weights,
        shape_directions=shape_directions,
        up_axis_source=source_frame.up,
        up_axis_target=TARGET_UP_AXIS,
        stature_m=stature,
        beta_count=0 if shape_directions is None else int(shape_directions.shape[2]),
        source_frame=source_frame.describe(),
    )


#: Joint indices the released templates use for the root and the head.
_PELVIS_INDEX = 0
_HEAD_INDEX = 15
#: The frame every downstream product is authored in.
TARGET_UP_AXIS = "z"


@dataclass(frozen=True, slots=True)
class _BodyFrame:
    """The axis triple a body file is authored in, measured from its joints.

    ``up``/``forward``/``left`` are ``x``/``y``/``z`` tokens naming the file's own
    axes, so they can be handed straight to
    :func:`~sim2sense_fall.humans.motion.body_frame_conversion`.
    """

    up: str
    forward: str
    left: str
    evidence: tuple[str, ...] = ()

    def describe(self) -> str:
        return f"up={self.up} forward={self.forward} left={self.left}"


def _axis_token(vector: np.ndarray, source: Path | str, role: str) -> str:
    """Which signed axis ``vector`` lies along, refusing a diagonal or zero vector.

    Axis-alignment is required rather than assumed because a diagonal vector means the
    file is pre-rotated, and silently picking its largest component would bake that
    rotation into every pose downstream.
    """

    row = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(row))
    if not math.isfinite(norm) or norm < 1e-9:
        raise ValueError(f"{source}: the {role} reference vector is degenerate; no axis to derive")
    unit = row / norm
    index = int(np.argmax(np.abs(unit)))
    if abs(unit[index]) < 0.99:
        raise ValueError(
            f"{source}: the {role} reference direction {np.round(unit, 3).tolist()} is not "
            f"axis-aligned (best fit {'xyz'[index]} at {abs(unit[index]):.3f}); this template "
            "is pre-rotated or malformed, and the body frame cannot be derived from it"
        )
    sign = "" if unit[index] > 0 else "-"
    return f"{sign}{'xyz'[index]}"


def _joint(joints: np.ndarray, topology: Any, name: str, source: Path | str) -> np.ndarray:
    """Look a joint up by name, raising rather than silently using index 0."""

    try:
        index = topology.index(name)
    except (KeyError, ValueError) as error:
        raise ValueError(
            f"{source}: the model has no {name!r} joint, so its body frame cannot be "
            f"derived; joints present: {list(topology.joint_names)}"
        ) from error
    return np.asarray(joints[index], dtype=np.float64)


def _derive_body_frame(joints: np.ndarray, topology: Any, source: Path | str) -> _BodyFrame:
    """Measure the file's body frame from its rest joint positions.

    Three independent anatomical directions are used, so a single misleading vector
    cannot rotate the body:

    ``up``
        The pelvis-to-head vector, cross-checked against hip-to-knee, which must agree
        after sign correction. A body whose torso and thigh disagree about which way is
        up is not something this loader will guess about.
    ``left``
        The left-hip-to-right-hip vector, reversed. The shoulder line is checked to be
        parallel to it.
    ``forward``
        The remaining axis, fixed by the right-handedness of the triple and verified
        against the ankle-to-foot vector, which points forward on an upright body.

    The frame is derived rather than assumed because body models disagree: the released
    SMPL template is Y-up with its *lateral* axis on X, while this pipeline authors
    ``+X`` forward / ``+Y`` left / ``+Z`` up. Measuring only the up axis, as an earlier
    version of this loader did, silently yawed every body 90 degrees.
    """

    pelvis = _joint(joints, topology, "pelvis", source)
    head = _joint(joints, topology, "head", source)
    left_hip = _joint(joints, topology, "left_hip", source)
    right_hip = _joint(joints, topology, "right_hip", source)
    left_shoulder = _joint(joints, topology, "left_shoulder", source)
    right_shoulder = _joint(joints, topology, "right_shoulder", source)

    evidence: list[str] = []

    up = _axis_token(head - pelvis, source, "pelvis-to-head")
    hip_to_knee = _axis_token(
        _joint(joints, topology, "left_knee", source)
        - _joint(joints, topology, "left_hip", source),
        source,
        "hip-to-knee",
    )
    # hip->knee points down, so it must be the exact negation of the up token.
    expected_down = up.lstrip("-") if up.startswith("-") else f"-{up}"
    if hip_to_knee != expected_down:
        raise ValueError(
            f"{source}: the torso says up is {up!r} but the thigh points {hip_to_knee!r} "
            f"(expected {expected_down!r}); the rest pose is not a plausible upright body "
            "and the frame will not be guessed"
        )
    evidence.append(f"up={up} (pelvis->head, thigh agrees)")

    hip_axis = _axis_token(right_hip - left_hip, source, "hip line (right relative to left)")
    left = f"-{hip_axis.lstrip('-')}" if not hip_axis.startswith("-") else hip_axis.lstrip("-")
    shoulder_axis = _axis_token(
        right_shoulder - left_shoulder, source, "shoulder line (right relative to left)"
    )
    if shoulder_axis != hip_axis:
        raise ValueError(
            f"{source}: the hip line points {hip_axis!r} but the shoulder line points "
            f"{shoulder_axis!r}; the rest pose is twisted, not upright"
        )
    evidence.append(f"left=-{hip_axis} (hip and shoulder lines agree)")

    axes = {"x", "y", "z"}
    remaining = axes - {up.lstrip("-"), left.lstrip("-")}
    if len(remaining) != 1:
        raise ValueError(f"{source}: up={up!r} and left={left!r} do not leave one axis for forward")
    forward_axis = remaining.pop()

    # Right-handedness fixes the forward sign. A right-handed triple has the up axis
    # equal to left x forward, so pick the sign that satisfies it.
    for sign in ("", "-"):
        candidate = f"{sign}{forward_axis}"
        triple = body_frame_conversion(up=up, forward=candidate, left=left, source=str(source))
        if abs(float(np.linalg.det(triple)) - 1.0) <= 1e-9:
            forward = candidate
            break
    else:  # pragma: no cover - one of the two signs is always right-handed
        raise ValueError(
            f"{source}: no right-handed forward axis found for up={up!r}, left={left!r}"
        )

    # The ankle-to-foot vector is a poor axis probe -- a foot is a short, angled segment
    # (measured 0.888 on its dominant axis), so axis-alignment is not required here.
    # Only the sign of its forward component is meaningful, and it is used to confirm the
    # forward direction the right-handedness argument already chose.
    ankle_to_foot = _joint(joints, topology, "left_foot", source) - _joint(
        joints, topology, "left_ankle", source
    )
    forward_index = "xyz".index(forward_axis)
    forward_sign = -1.0 if forward.startswith("-") else 1.0
    toe_component = forward_sign * float(ankle_to_foot[forward_index])
    if toe_component <= 0:
        raise ValueError(
            f"{source}: with forward on {forward!r}, the toes point backwards "
            f"(component {toe_component:+.3f}); the rest pose is inconsistent"
        )
    evidence.append(f"forward={forward} (toes lead by {toe_component:+.3f} m)")

    return _BodyFrame(up=up, forward=forward, left=left, evidence=tuple(evidence))


def _check_standing_frame(
    joints: np.ndarray, topology: Any, source: Path | str, frame: _BodyFrame
) -> None:
    """Reject a body that is not upright in the frame that was just applied.

    This is the guard the old up-axis check lacked: an up-only basis could yaw the body
    90 degrees and still pass every extent test, because a yawed body is human-sized
    along *some* axis. Here the vertical extent is required to come from the joints that
    actually describe height -- the pelvis-to-head rise -- and the lateral span from the
    shoulders, rather than from whichever axis happens to be longest.
    """

    pelvis = _joint(joints, topology, "pelvis", source)
    head = _joint(joints, topology, "head", source)
    left_foot = _joint(joints, topology, "left_foot", source)
    left_shoulder = _joint(joints, topology, "left_shoulder", source)
    right_shoulder = _joint(joints, topology, "right_shoulder", source)

    rise = float(head[2] - pelvis[2])
    if rise < 0.4:
        raise ValueError(
            f"{source}: the head rises only {rise:.3f} m above the pelvis along the declared "
            f"up axis ({frame.up!r}); an upright body model needs roughly 0.5 m of torso. "
            "The body frame is wrong and downstream meshes would be exported lying down"
        )
    drop = float(pelvis[2] - left_foot[2])
    if drop < 0.4:
        raise ValueError(
            f"{source}: the foot hangs only {drop:.3f} m below the pelvis along the declared "
            f"up axis ({frame.up!r}); the body is not upright"
        )
    span = float(np.linalg.norm(right_shoulder - left_shoulder))
    if span < 0.15:
        raise ValueError(
            f"{source}: the shoulders are only {span:.3f} m apart in the rest pose, so the "
            "lateral axis cannot be identified"
        )


@dataclass(frozen=True, slots=True)
class BodySelection:
    """What the pipeline may honestly claim about the human body it is running.

    ``representation`` is decided by the geometry that loaded, never by whether a
    registered path exists: the stage-7 review found a text file named ``basicModel_*.pkl``
    producing a ``smpl_skin_mesh`` summary while the build still ran the procedural
    skeleton. ``degraded`` says whether falling back to the proxy was permitted by the
    configuration, so a silent fallback cannot be mistaken for an imported body.
    """

    representation: str
    model: SmplModel | None = None
    error: str | None = None
    model_id: str | None = None
    path: Path | None = None

    @property
    def has_skin_mesh(self) -> bool:
        return self.representation == REPRESENTATION_SKIN_MESH

    def as_dict(self) -> dict[str, Any]:
        return {
            "body_representation": self.representation,
            "model_asset_id": self.model_id,
            "model_asset_path": None if self.path is None else str(self.path),
            "model_import_error": self.error,
            "imported_model": None if self.model is None else self.model.as_dict(),
        }


def select_body(
    registry: AssetRegistry,
    *,
    model_id: str = "smpl_neutral_v1_1_0",
    allow_procedural: bool = True,
) -> BodySelection:
    """Load the selected licensed model, or fail rather than pretend.

    ``allow_procedural`` comes from ``skeleton.allow_procedural_skeleton``. When it is
    false the result still says ``capsule_proxy_surface`` but carries the reason, so the
    caller fails the run instead of quietly building a mannequin and labelling it SMPL.
    """

    try:
        path = require_model(registry, model_id)
    except AssetUnavailable as exc:
        # Keep the whole message: the registration URL and the searched paths are the
        # actionable part, and the first line alone says only "not present".
        reason = " ".join(str(exc).split())[:600]
        return BodySelection(
            representation=REPRESENTATION_PROXY,
            model_id=model_id,
            error=reason if allow_procedural else f"{reason} (degrading is not allowed)",
        )
    try:
        model = load_smpl_model(path, registry=registry)
    except Exception as exc:  # noqa: BLE001 - reported to the caller, never swallowed
        reason = f"{type(exc).__name__}: {exc}"
        LOGGER.warning("%s could not be imported as a body model: %s", path, reason)
        return BodySelection(
            representation=REPRESENTATION_PROXY,
            model_id=model_id,
            path=path,
            error=reason if allow_procedural else f"{reason} (degrading is not allowed)",
        )
    if not model.has_mesh:
        reason = (
            f"{path.name} loaded but exposes no vertices, faces and skinning weights for "
            f"{model.vertex_count} vertices"
        )
        return BodySelection(
            representation=REPRESENTATION_PROXY,
            model=model,
            model_id=model_id,
            path=path,
            error=reason if allow_procedural else f"{reason} (degrading is not allowed)",
        )
    return BodySelection(
        representation=REPRESENTATION_SKIN_MESH, model=model, model_id=model_id, path=path
    )


def registry_from_sequences(
    model_entries: Sequence[Mapping[str, Any]],
    motion_entries: Sequence[Mapping[str, Any]],
    *,
    roots: Sequence[Path],
    notes: str = "",
) -> AssetRegistry:
    """Build a registry directly from mappings; used by tests and tooling."""

    models = [_model_from_mapping(dict(entry)) for entry in model_entries]
    motions = [_motion_from_mapping(dict(entry)) for entry in motion_entries]
    return AssetRegistry(
        models={entry.id: entry for entry in models},
        motions={entry.id: entry for entry in motions},
        roots=tuple(Path(root) for root in roots),
        notes=notes,
    )
