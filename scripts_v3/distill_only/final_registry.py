#!/usr/bin/env python3
"""Resolve selected model entries from a Stage-F registry."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def resolve(registry: Path, identifiers: list[str]) -> list[tuple[str, Path]]:
    value = yaml.safe_load(registry.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("registry must be a mapping")
    rows = []
    for identifier in identifiers:
        entry = value.get(identifier)
        if not isinstance(entry, dict) or not entry.get("checkpoint"):
            raise ValueError(f"registry checkpoint is unresolved: {identifier}")
        checkpoint = Path(entry["checkpoint"]).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = registry.parent / checkpoint
        checkpoint = checkpoint.resolve()
        if not checkpoint.is_file():
            raise ValueError(f"registry checkpoint does not exist: {checkpoint}")
        rows.append((identifier, checkpoint))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--models", required=True)
    args = parser.parse_args()
    identifiers = args.models.split(",")
    if not identifiers or any(not item for item in identifiers):
        parser.error("--models must contain comma-separated model IDs")
    try:
        rows = resolve(args.registry.resolve(), identifiers)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    for identifier, checkpoint in rows:
        print(f"{identifier}\t{checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
