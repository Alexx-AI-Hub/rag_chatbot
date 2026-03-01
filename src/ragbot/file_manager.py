from __future__ import annotations
import os
import hashlib
import logging as log
import shutil
from pathlib import Path
from typing import Any

from config import ALLOWED_UPLOAD_TYPES, MAX_UPLOAD_SIZE, BASE_DIR, SESSION_DIR, BASE_INDEX_DIR, SESSION_INDEX_DIR

UNIQUE_NAME_MAX_ATTEMPTS = 99

def clean_session_dirs():
    if any(directory.exists() and any(directory.iterdir()) for directory in (SESSION_DIR, SESSION_INDEX_DIR)):
        reset_directory(SESSION_DIR)
        reset_directory(SESSION_INDEX_DIR)
        log.info("Startup cleanup removed leftover session files.")


def ensure_data_dirs() -> None:
    """Create required data directories if they do not already exist."""
    dirs = [BASE_DIR, BASE_INDEX_DIR, SESSION_DIR, SESSION_INDEX_DIR]
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)


def get_valid_path(file: Any) -> Path | None:
    """Return a valid file path if found, otherwise None."""

    if isinstance(file, (str, os.PathLike)):
        if Path(file).is_file():
            log.debug("Found valid path from (str/os.PathLike): %s", file)
            return Path(file)
    elif isinstance(file, dict):
        for key in file:
            if isinstance(file[key], (str, os.PathLike)) and Path(file[key]).is_file():
                log.debug("Found valid path in file Key: %s. Path: %s", key, file[key])
                return Path(file[key])
    else:
        for attr in dir(file):
            if attr.startswith("_"):
                continue
            try:
                value = getattr(file, attr)
                if not callable(value) and isinstance(value, (str, os.PathLike)) and Path(value).is_file():
                    log.debug("Found valid path in file Attribute: %s. Path: %s", attr, value)
                    return Path(value)
            except Exception:
                continue

    log.debug("No path found -> None. File: %s Type: %s", file, type(file))
    return None


def copy_files_from_dir(src_dir_path: str | Path, dest_dir_path: str | Path)->bool:
    src_dir_path = Path(src_dir_path)
    dest_path = Path(dest_dir_path)

    if not src_dir_path.exists() or not src_dir_path.is_dir() or not any(src_dir_path.iterdir()):
        log.debug("Source directory does not exist or is not a directory: %s", src_dir_path)
        return False

    if not dest_path.exists() or not dest_path.is_dir():
        log.debug("Destination directory does not exist or is not a directory: %s", dest_path)
        return False

    files = get_files_from_dir(src_dir_path)
    if not files:
        log.info("Source directory is empty, nothing to copy: %s", src_dir_path)
        return False

    copied = 0
    for file_path in files:
        file_path = Path(file_path)
        if copy_if_allowed(file_path, dest_path, file_path.name):
            copied += 1
            log.info("Copied %s from: %s --> %s", file_path.name, src_dir_path, dest_dir_path)
            continue
        log.debug("Failed to copy %s from: %s --> %s", file_path.name, src_dir_path, dest_dir_path)
    return True if copied == len(files) else False



def copy_if_allowed(file_path: str | Path, dest_dir_path: str | Path, file_name: str,
                    max_file_size: int = MAX_UPLOAD_SIZE, allowed_file_types: set[str] | list[str] = ALLOWED_UPLOAD_TYPES) -> bool:
    """Copy a source file into a destination directory if suffix and size are allowed."""
    file_path = get_valid_path(file_path)
    dest_dir_path = Path(dest_dir_path)
    if file_path:
        if allowed_file_types and not file_path.suffix.lower() in allowed_file_types:
            log.debug("Unallowed file type:(%s)", file_path.suffix.lower())
            return False

        if file_path.stat().st_size > max_file_size:
            log.debug("File size is over allowed size. Size: %s bytes", file_path.stat().st_size)
            return False

        if not dest_dir_path.is_dir():
            log.debug("Dir Path must point to a Directory. DirPath: %s", dest_dir_path)
            return False

        file_name_unique = _file_name_unique(dest_dir_path, file_name)
        if file_name_unique:
            shutil.copy2(file_path, dest_dir_path / file_name_unique)
            log.debug("Succeeded to copy: %s. -to-> %s",file_name_unique, dest_dir_path)

            return True

    log.debug("get_valid_path() Found no valid path. Path: %s", file_path)
    return False


def get_files_from_dir(dir_path:str|Path) -> list[str]:
    """Return file paths found directly in a directory as strings."""
    dir_path = Path(dir_path)
    if not dir_path.exists():
        log.debug("Directory: %s. Did not exist In: get_files_from_dir()", dir_path)
        return []
    return [str(path) for path in sorted(dir_path.iterdir()) if path.is_file()]


def compute_checksum(src_path: str | Path) -> str:
    """Compute a SHA256 checksum for file content."""
    with Path(src_path).open("rb") as file_handle:
        return hashlib.file_digest(file_handle, "sha256").hexdigest()


def reset_directory(dir_path: str | Path) -> None:
    """Reset a directory by recreating it as an empty folder."""
    path = Path(dir_path)
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


def _file_name_unique(dir_path: Path, file_name: str) -> str:
    """Return a non-conflicting file name for directory."""
    path = dir_path / Path(file_name).name
    if not path.exists():
        log.debug("Good!. %s: did was unique In: %s", file_name, dir_path)
        return file_name

    file_stem = path.stem  # file_name without .suffix(ex: .txt)
    file_type = path.suffix
    counter = 1
    while counter <= UNIQUE_NAME_MAX_ATTEMPTS:
        next_file_name = dir_path / f"{file_stem}_({counter}){file_type}"
        if not next_file_name.exists():
            log.debug("Good!. %s: did was unique In: %s", file_name, dir_path)
            return next_file_name.name
        counter += 1
    return ""
