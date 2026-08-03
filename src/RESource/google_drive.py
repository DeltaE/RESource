"""Mount and access Google Drive files through an ``rclone`` remote.

This module deliberately delegates authentication and FUSE mounting to rclone.
Credentials therefore remain in rclone's user configuration and never enter a
RESource configuration file, notebook, or log.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any


class GoogleDriveError(RuntimeError):
    """Base error raised by the Google Drive integration."""


class GoogleDriveMountError(GoogleDriveError):
    """Raised when the Drive remote cannot be mounted."""


class AmbiguousDriveFileError(GoogleDriveError):
    """Raised when a filename identifies more than one Drive file."""


class GoogleDriveFiles:
    """Find and open files below an already-mounted Google Drive directory.

    Args:
        root: Root directory of the mounted Drive or mounted subdirectory.

    File names in Google Drive are not unique. ``path()`` raises
    :class:`AmbiguousDriveFileError` when an exact filename has multiple matches;
    callers can use ``find_all()`` or provide a relative path to disambiguate.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()

    def find_all(self, filename: str, *, case_sensitive: bool = True) -> list[Path]:
        """Return every regular file whose basename matches ``filename``.

        Args:
            filename: Basename to locate. Directory components are not allowed.
            case_sensitive: Whether filename case must match exactly.

        Returns:
            Sorted absolute paths to matching files.
        """
        self._require_root()
        if not filename or Path(filename).name != filename:
            raise ValueError("filename must be a non-empty basename")

        expected = filename if case_sensitive else filename.casefold()
        matches: list[Path] = []
        for directory, _subdirectories, files in os.walk(self.root):
            for candidate in files:
                comparable = candidate if case_sensitive else candidate.casefold()
                if comparable == expected:
                    matches.append(Path(directory, candidate))
        return sorted(matches, key=lambda item: item.as_posix())

    def path(self, name_or_relative_path: str | Path) -> Path:
        """Resolve a unique filename or a path relative to the mounted root.

        Args:
            name_or_relative_path: A basename to search recursively, or a relative
                path containing one or more directory components.

        Returns:
            Absolute path to the requested file.

        Raises:
            FileNotFoundError: If no regular file matches.
            AmbiguousDriveFileError: If a basename has multiple matches.
            ValueError: If an absolute path or parent traversal is supplied.
        """
        requested = Path(name_or_relative_path)
        if requested.is_absolute() or ".." in requested.parts:
            raise ValueError("Drive paths must be relative and cannot contain '..'")

        if len(requested.parts) > 1:
            self._require_root()
            candidate = self.root.joinpath(requested)
            if not candidate.is_file():
                raise FileNotFoundError(f"Google Drive file not found: {requested}")
            return candidate

        matches = self.find_all(requested.name)
        if not matches:
            raise FileNotFoundError(f"Google Drive file not found: {requested.name}")
        if len(matches) > 1:
            relative_matches = [str(match.relative_to(self.root)) for match in matches]
            raise AmbiguousDriveFileError(
                f"{requested.name!r} matches multiple Drive files: " + ", ".join(relative_matches)
            )
        return matches[0]

    def open(
        self,
        name_or_relative_path: str | Path,
        mode: str = "rb",
        **kwargs: Any,
    ) -> IO[Any]:
        """Open a uniquely resolved Drive file using Python's built-in ``open``."""
        return self.path(name_or_relative_path).open(mode, **kwargs)

    def _require_root(self) -> None:
        if not self.root.is_dir():
            raise GoogleDriveMountError(f"Google Drive mount is unavailable: {self.root}")


class GoogleDriveMount:
    """Manage a read-only Google Drive mount backed by ``rclone mount``.

    Args:
        remote: Configured rclone remote, optionally followed by a subdirectory,
            for example ``"gdrive:"`` or ``"gdrive:RESource/input"``.
        mount_point: Empty local directory where the Drive will be mounted.
        read_only: Add rclone's read-only protection. Defaults to ``True``.
        extra_args: Additional trusted arguments passed directly to rclone.
        mount_timeout: Seconds to wait for the FUSE mount to become available.
        rclone_binary: Binary name or explicit path, primarily useful in deployment.

    The mount runs as a child process. Use the object as a context manager or call
    ``unmount()`` during shutdown so the child process is always cleaned up.
    """

    def __init__(
        self,
        remote: str,
        mount_point: str | Path,
        *,
        read_only: bool = True,
        extra_args: Sequence[str] = (),
        mount_timeout: float = 15.0,
        rclone_binary: str = "rclone",
    ) -> None:
        if not remote or ":" not in remote:
            raise ValueError("remote must be an rclone remote such as 'gdrive:'")
        if mount_timeout <= 0:
            raise ValueError("mount_timeout must be greater than zero")

        self.remote = remote
        self.mount_point = Path(mount_point).expanduser().resolve()
        self.read_only = read_only
        self.extra_args = tuple(extra_args)
        self.mount_timeout = mount_timeout
        self.rclone_binary = rclone_binary
        self._process: subprocess.Popen[str] | None = None
        self.files = GoogleDriveFiles(self.mount_point)

    @property
    def is_mounted(self) -> bool:
        """Return whether the local path is currently a mounted filesystem."""
        return os.path.ismount(self.mount_point)

    def mount(self) -> GoogleDriveFiles:
        """Start rclone and wait until the Drive filesystem is available."""
        if self.is_mounted:
            return self.files
        if self._process is not None and self._process.poll() is None:
            raise GoogleDriveMountError("rclone is running but the mount is unavailable")

        executable = shutil.which(self.rclone_binary)
        if executable is None:
            raise GoogleDriveMountError(
                "rclone was not found; install it and run 'rclone config' first"
            )

        self.mount_point.mkdir(parents=True, exist_ok=True)
        if any(self.mount_point.iterdir()):
            raise GoogleDriveMountError(
                f"mount point must be empty to avoid hiding local files: {self.mount_point}"
            )

        command = [executable, "mount", self.remote, str(self.mount_point)]
        if self.read_only:
            command.append("--read-only")
        command.extend(self.extra_args)
        self._process = subprocess.Popen(  # noqa: S603 - no shell; arguments are explicit
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )

        deadline = time.monotonic() + self.mount_timeout
        while time.monotonic() < deadline:
            if self.is_mounted:
                return self.files
            if self._process.poll() is not None:
                error = self._read_error()
                self._process = None
                raise GoogleDriveMountError(f"rclone mount failed: {error}")
            time.sleep(0.05)

        self.unmount()
        raise GoogleDriveMountError(
            f"Google Drive did not mount within {self.mount_timeout:g} seconds"
        )

    def unmount(self) -> None:
        """Stop the managed rclone process and release its FUSE mount."""
        process = self._process
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        self._process = None

    def __enter__(self) -> GoogleDriveFiles:
        return self.mount()

    def __exit__(self, *_exc_info: object) -> None:
        self.unmount()

    def _read_error(self) -> str:
        if self._process is None or self._process.stderr is None:
            return "unknown error"
        return self._process.stderr.read().strip() or "unknown error"


@contextmanager
def mounted_google_drive(
    remote: str,
    mount_point: str | Path,
    **kwargs: Any,
) -> Iterator[GoogleDriveFiles]:
    """Mount a Drive for the duration of a ``with`` block."""
    mount = GoogleDriveMount(remote, mount_point, **kwargs)
    try:
        yield mount.mount()
    finally:
        mount.unmount()
