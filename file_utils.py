from __future__ import annotations

import glob
from pathlib import Path


def expand_file_patterns(patterns: list[str]) -> list[Path]:
    """Expand file paths and Windows glob patterns into existing files."""
    paths: list[Path] = []
    seen: set[Path] = set()
    unmatched_patterns: list[str] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if not matches:
            if glob.has_magic(pattern):
                unmatched_patterns.append(pattern)
                continue
            matches = [pattern]
        for match in matches:
            path = Path(match).resolve()
            if path not in seen:
                seen.add(path)
                paths.append(path)
    missing = [path for path in paths if not path.is_file()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"file not found:\n{missing_text}")
    if unmatched_patterns:
        unmatched_text = "\n".join(unmatched_patterns)
        raise FileNotFoundError(f"no files matched pattern:\n{unmatched_text}")
    return sorted(paths)
