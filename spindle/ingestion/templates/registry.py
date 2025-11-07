"""Template loading and registry utilities."""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Iterator, List, MutableSequence, Sequence

import yaml

from spindle.ingestion.errors import TemplateResolutionError
from spindle.ingestion.types import TemplateSelector, TemplateSpec

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - compatibility shim
    import tomli as tomllib  # type: ignore[no-redef]


LOGGER = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".yaml", ".yml", ".json", ".toml"}


def _coerce_sequence(value: Any) -> Sequence[str]:
    if value is None:
        return tuple()
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(str(v) for v in value)
    return (str(value),)


def _load_raw_template(path: Path) -> Iterable[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported template file type: {suffix}")
    LOGGER.debug("Loading template file: %s", path)
    with path.open("rb") as file_obj:
        if suffix in {".yaml", ".yml"}:
            data = yaml.safe_load(file_obj)
        elif suffix == ".json":
            data = json.load(file_obj)
        else:  # .toml
            data = tomllib.load(file_obj)

    if data is None:
        return []
    if isinstance(data, dict):
        if "templates" in data and isinstance(data["templates"], list):
            return data["templates"]
        return [data]
    if isinstance(data, list):
        return data
    raise TypeError(f"Template file {path} has unsupported structure: {type(data)!r}")


def _normalise_template_dict(raw: dict[str, Any]) -> TemplateSpec:
    selector_data = raw.get("selector", {}) or {}
    selector = TemplateSelector(
        mime_types=_coerce_sequence(selector_data.get("mime")),
        path_globs=_coerce_sequence(selector_data.get("glob")),
        file_extensions=_coerce_sequence(selector_data.get("extensions")),
    )
    if "name" not in raw or not raw["name"]:
        raise ValueError("Template definition missing required 'name' field")
    return TemplateSpec(
        name=str(raw.get("name")),
        selector=selector,
        loader=raw.get("loader", ""),
        preprocessors=_coerce_sequence(raw.get("preprocessors")),
        splitter=raw.get("splitter", {}),
        metadata_extractors=_coerce_sequence(raw.get("metadata_extractors")),
        postprocessors=_coerce_sequence(raw.get("postprocessors")),
        graph_hooks=_coerce_sequence(raw.get("graph_hooks")),
        description=raw.get("description"),
    )


def load_templates_from_paths(paths: Sequence[Path]) -> List[TemplateSpec]:
    """Load template specifications from disk."""

    specs: List[TemplateSpec] = []
    for path in paths:
        if not path.exists():
            LOGGER.debug("Skipping missing template path: %s", path)
            continue
        if path.is_dir():
            for file_path in sorted(path.iterdir()):
                if file_path.suffix.lower() in SUPPORTED_SUFFIXES:
                    specs.extend(load_templates_from_paths([file_path]))
            continue
        for template_dict in _load_raw_template(path):
            specs.append(_normalise_template_dict(template_dict))
    return specs


class TemplateRegistry:
    """Registry that resolves documents to template specifications."""

    def __init__(self, templates: Iterable[TemplateSpec] | None = None) -> None:
        self._templates: MutableSequence[TemplateSpec] = list(templates or ())

    def __iter__(self) -> Iterator[TemplateSpec]:
        return iter(self._templates)

    def __len__(self) -> int:
        return len(self._templates)

    def register(self, spec: TemplateSpec) -> None:
        LOGGER.debug("Registering template: %s", spec.name)
        self._templates.append(spec)

    def extend(self, specs: Iterable[TemplateSpec]) -> None:
        for spec in specs:
            self.register(spec)

    def resolve(self, *, path: Path, mime_type: str | None = None) -> TemplateSpec:
        LOGGER.debug("Resolving template for path=%s mime_type=%s", path, mime_type)
        matches: list[TemplateSpec] = []
        for spec in self._templates:
            if _matches_selector(spec.selector, path, mime_type):
                matches.append(spec)
        if not matches:
            raise TemplateResolutionError(
                f"No template found for path='{path}' mime_type='{mime_type}'"
            )
        if len(matches) == 1:
            return matches[0]
        LOGGER.debug("Multiple templates matched; selecting the most specific")
        matches.sort(key=_selector_specificity, reverse=True)
        return matches[0]

    def clone_with_overrides(
        self, overrides: Iterable[TemplateSpec]
    ) -> "TemplateRegistry":
        merged = TemplateRegistry(self._templates)
        merged.extend(overrides)
        return merged

    def with_default_template(self, fallback: TemplateSpec) -> "TemplateRegistry":
        has_default = any(spec.name == fallback.name for spec in self._templates)
        if not has_default:
            self.register(fallback)
        return self

    def descriptions(self) -> List[str]:
        return [spec.description or spec.name for spec in self._templates]


def _matches_selector(selector: TemplateSelector, path: Path, mime_type: str | None) -> bool:
    if selector.mime_types and mime_type not in selector.mime_types:
        return False
    if selector.file_extensions:
        if path.suffix.lower() not in {ext.lower() for ext in selector.file_extensions}:
            return False
    if selector.path_globs:
        matched = any(path.match(pattern) for pattern in selector.path_globs)
        if not matched:
            return False
    return True


def _selector_specificity(selector: TemplateSelector) -> int:
    score = 0
    score += len(selector.mime_types) * 2
    score += len(selector.file_extensions) * 3
    score += len(selector.path_globs)
    return score


def merge_template_sequences(
    base: Sequence[TemplateSpec],
    overrides: Sequence[TemplateSpec],
) -> List[TemplateSpec]:
    registry = TemplateRegistry(base)
    for override in overrides:
        replaced = False
        for index, template in enumerate(list(registry._templates)):
            if template.name == override.name:
                registry._templates[index] = replace(template, **override.__dict__)
                replaced = True
                break
        if not replaced:
            registry.register(override)
    return list(registry)

