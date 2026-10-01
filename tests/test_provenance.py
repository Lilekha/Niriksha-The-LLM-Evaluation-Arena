import platform
import socket
import subprocess
import sys
from pathlib import Path

import pydantic
import pytest
from pydantic import ValidationError

import niriksha
import niriksha.core.provenance as provenance
from niriksha.core.provenance import SoftwareInfo, collect_software_info, git_state

COMMIT = "0123456789abcdef0123456789abcdef01234567"  # pragma: allowlist secret


def test_collects_versions_and_injected_git_state():
    info = collect_software_info(lambda: (COMMIT, True))
    assert info.niriksha_version == niriksha.__version__
    assert info.python_version == platform.python_version()
    assert info.pydantic_version == pydantic.VERSION
    assert info.platform == sys.platform
    assert (info.git_commit, info.git_dirty) == (COMMIT, True)


def test_unavailable_git_is_null_not_invented():
    info = collect_software_info(lambda: (None, None))
    assert info.git_commit is None and info.git_dirty is None


@pytest.mark.parametrize(
    "fields",
    [
        {"git_commit": COMMIT},  # commit without dirty flag
        {"git_dirty": False},  # dirty flag without commit
        {"git_commit": "not-a-commit", "git_dirty": False},
        {"git_commit": COMMIT.upper(), "git_dirty": False},
        {"unknown": 1},
        {"platform": ""},
    ],
)
def test_software_info_validation(fields):
    base = {
        "niriksha_version": "0.0.0",
        "python_version": "3.13.3",
        "pydantic_version": "2.13.5",
        "platform": "win32",
    }
    with pytest.raises(ValidationError):
        SoftwareInfo(**{**base, **fields})


def test_software_info_contains_no_machine_identity():
    dumped = collect_software_info(lambda: (None, None)).model_dump_json()
    assert str(Path.home()) not in dumped and Path.home().name not in dumped
    assert socket.gethostname().lower() not in dumped.lower()


def test_git_state_outside_a_repository_is_unknown(tmp_path):
    assert git_state(tmp_path) == (None, None)


def test_git_state_in_this_checkout_is_well_formed():
    commit, dirty = git_state()
    assert (commit is None and dirty is None) or (len(commit) == 40 and isinstance(dirty, bool))


def _fake_git(monkeypatch, tmp_path, *, top_has_source=True, commit=COMMIT, status="", rc=(0, 0)):
    top = tmp_path / "repo"
    (top / "src" / "niriksha").mkdir(parents=True) if top_has_source else top.mkdir()

    def run(args, **kwargs):
        if args[1] == "rev-parse":
            return subprocess.CompletedProcess(args, rc[0], f"{top}\n{commit}\n", "")
        return subprocess.CompletedProcess(args, rc[1], status, "")

    monkeypatch.setattr(provenance.subprocess, "run", run)


@pytest.mark.parametrize(
    ("status", "dirty"), [("", False), (" M file.py\n", True), ("?? new\n", True)]
)
def test_git_state_reports_commit_and_dirty_flag(monkeypatch, tmp_path, status, dirty):
    _fake_git(monkeypatch, tmp_path, status=status)
    assert git_state(tmp_path) == (COMMIT, dirty)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"top_has_source": False},  # some unrelated repository
        {"commit": "HEAD"},
        {"rc": (128, 0)},
        {"rc": (0, 1)},
    ],
)
def test_git_state_unknown_for_unusable_git_output(monkeypatch, tmp_path, kwargs):
    _fake_git(monkeypatch, tmp_path, **kwargs)
    assert git_state(tmp_path) == (None, None)


@pytest.mark.parametrize(
    "exc", [FileNotFoundError("git"), subprocess.TimeoutExpired("git", 10), PermissionError()]
)
def test_git_state_unknown_when_git_cannot_run(monkeypatch, tmp_path, exc):
    def run(*args, **kwargs):
        raise exc

    monkeypatch.setattr(provenance.subprocess, "run", run)
    assert git_state(tmp_path) == (None, None)
