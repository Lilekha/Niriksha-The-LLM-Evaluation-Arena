"""Software provenance recorded in run manifests.

Unknown values are ``None``. Nothing is guessed, and nothing identifying the machine is recorded:
no hostname, username or filesystem path (only ``sys.platform``, e.g. ``"win32"``).

Limits: this records which code and library versions produced a run. It does not make the run
reproducible. Provider-side nondeterminism or silent model updates, hardware, unlocked
dependencies (there is no lockfile yet) and a dirty working tree are all outside what it captures.
"""

import platform
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import pydantic
from pydantic import BaseModel, ConfigDict, StringConstraints, model_validator

import niriksha

_NonEmpty = Annotated[str, StringConstraints(min_length=1)]
_COMMIT = re.compile(r"[0-9a-f]{40}")

GitLookup = Callable[[], tuple[str | None, bool | None]]


class SoftwareInfo(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    niriksha_version: _NonEmpty
    python_version: _NonEmpty
    pydantic_version: _NonEmpty
    platform: _NonEmpty
    git_commit: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")] | None = None
    git_dirty: bool | None = None  # includes untracked files; None when the commit is unknown

    @model_validator(mode="after")
    def _git_fields_agree(self):
        if (self.git_commit is None) != (self.git_dirty is None):
            raise ValueError("git_commit and git_dirty must both be known or both be null")
        return self


def git_state(cwd: Path | None = None) -> tuple[str | None, bool | None]:
    """``(commit, dirty)`` of the repository containing the niriksha source, or ``(None, None)``.

    Returns ``(None, None)`` when git is missing, the directory is not a repository, or the
    repository is not the niriksha source tree (for example, an installed copy sitting inside some
    unrelated repository). Runs ``git`` locally; no network.
    """
    cwd = cwd or Path(__file__).resolve().parent
    run = {"cwd": cwd, "capture_output": True, "text": True, "errors": "replace", "timeout": 10}
    try:
        head = subprocess.run(["git", "rev-parse", "--show-toplevel", "HEAD"], check=False, **run)
        if head.returncode != 0:
            return None, None
        top, _, commit = head.stdout.strip().partition("\n")
        if not _COMMIT.fullmatch(commit) or not (Path(top) / "src" / "niriksha").is_dir():
            return None, None
        status = subprocess.run(["git", "status", "--porcelain"], check=False, **run)
        if status.returncode != 0:
            return None, None
        return commit, bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return None, None


def collect_software_info(git_lookup: GitLookup = git_state) -> SoftwareInfo:
    commit, dirty = git_lookup()
    return SoftwareInfo(
        niriksha_version=niriksha.__version__,
        python_version=platform.python_version(),
        pydantic_version=pydantic.VERSION,
        platform=sys.platform,
        git_commit=commit,
        git_dirty=dirty,
    )
