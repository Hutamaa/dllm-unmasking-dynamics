"""Repository paths and exact reconstruction of pre-move import/path-only sources.

This changes no model computation. A historical match requires reconstructing the
original bytes and matching their recorded SHA256; arbitrary source edits fail.
"""
import hashlib
import json
import os
from pathlib import Path


def repository_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / 'pyproject.toml').is_file() and (parent / 'configs').is_dir():
            return parent
    raise RuntimeError('Use an editable installation from the research checkout; configs are required')


def enter_repository():
    """CLI paths, defaults and output guards are relative to the checkout root."""
    os.chdir(repository_root())


def source_records():
    return json.loads(Path(__file__).with_name('source_layout.json').read_text())['files']


def source_path(path):
    path = Path(path)
    root = repository_root()
    candidate = path if path.is_absolute() else root / path
    if candidate.exists():
        return candidate
    try:
        legacy = candidate.relative_to(root).as_posix()
    except ValueError:
        return candidate
    record = source_records().get(legacy)
    return root / record['path'] if record else candidate


def historical_bytes(path, content=None):
    """Reverse the finite recorded replacements, then require exact original bytes."""
    physical = source_path(path).resolve()
    root = repository_root()
    record = next((r for r in source_records().values() if (root / r['path']).resolve() == physical), None)
    raw = physical.read_bytes() if content is None else content
    if record is None:
        return raw
    text = raw.decode('utf8')
    for before, after in reversed(record['replacements']):
        text = text.replace(after, before)
    original = text.encode('utf8')
    if hashlib.sha256(original).hexdigest() != record['sha256']:
        raise ValueError(f'Source has changes beyond the recorded import/path substitutions: {physical.name}')
    return original


def historical_digest(path):
    return hashlib.sha256(historical_bytes(path)).hexdigest()


def source_matches(path, expected):
    physical = source_path(path)
    if not physical.is_file():
        return False
    if hashlib.sha256(physical.read_bytes()).hexdigest() == expected:
        return True
    try:
        return historical_digest(physical) == expected
    except (ValueError, UnicodeError):
        return False
