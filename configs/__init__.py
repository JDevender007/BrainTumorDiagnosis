"""
Configuration loading for BrainTumorDiagnosis.

YAML files are merged into dataclass instances. Sections group
related keys; a single instance can be filled from several
sections (later sections win).
"""

from __future__ import annotations

import logging
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Iterable, TypeVar, get_type_hints


LOGGER = logging.getLogger(__name__)

T = TypeVar("T")


def load_merged_config(
    instance: T,
    path: str | Path,
    sections: Iterable[str] = (),
) -> T:
    """
    Load YAML sections into a dataclass instance.

    Args:
        instance: Dataclass instance receiving the values.
        path: YAML configuration file.
        sections: Section names to merge (later sections win).
            When empty, the whole YAML root is used.

    Returns:
        The same instance, updated in place.
    """
    if not is_dataclass(instance):
        raise TypeError(
            f"load_merged_config expects a dataclass, got {type(instance)}"
        )

    config_path = Path(path)

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    import yaml

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(
            f"Configuration root must be a mapping: {config_path}"
        )

    sections_to_load = list(sections)

    if sections_to_load:
        merged: dict[str, Any] = {}

        for section_name in sections_to_load:
            section = data.get(section_name, {})

            if not isinstance(section, dict):
                raise ValueError(
                    f"Configuration section '{section_name}' "
                    f"must be a mapping: {config_path}"
                )

            merged.update(section)
    else:
        merged = data

    known_field_names = {
        field.name for field in fields(instance)
    }

    field_type_hints = get_type_hints(type(instance))

    unknown_keys = sorted(
        set(merged) - known_field_names
    )

    if unknown_keys:
        LOGGER.warning(
            "Ignoring unknown configuration keys: %s",
            unknown_keys,
        )

    for key, value in merged.items():
        if key in known_field_names:
            setattr(
                instance,
                key,
                _coerce_value(
                    value,
                    type_hint=field_type_hints.get(key),
                ),
            )

    return instance


def _coerce_value(
    value: Any,
    type_hint: Any = None,
) -> Any:
    """
    Convert YAML values to the dataclass field types.

    Lists become tuples when the field is annotated as a tuple,
    keeping dataclass defaults consistent.
    """

    if type_hint is not None and isinstance(value, list):

        origin = getattr(
            type_hint,
            "__origin__",
            None,
        )

        if origin is tuple:
            return tuple(value)

    return value