"""Builders for the run-reader and scorer tests. Test code only."""

import itertools
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from ds_helpers import FIXTURES
from e2e_support import Crash, CrashingProvider, make_config
from niriksha.core.dataset import load_dataset
from niriksha.core.execution import execute_run, resume_run
from niriksha.core.provenance import collect_software_info

NOW = datetime(2026, 10, 1, 9, 30, tzinfo=UTC)
SOFTWARE = collect_software_info(lambda: (None, None))
SPLITS = {"tiny_qa": ("dev", "test", "holdout"), "tiny_extraction": ("dev", "test")}


def deterministic():
    ticks = itertools.count()
    return {"clock": lambda: next(ticks) * 0.5, "now": lambda: NOW, "software": SOFTWARE}


def copy_fixture(tmp_path: Path, name: str) -> Path:
    return Path(shutil.copytree(FIXTURES / name, tmp_path / name))


def finished_run(tmp_path, fixture="tiny_qa", script=None, run_id="r1", crash_after=None):
    """Execute a run on a private copy of a fixture dataset.

    With ``crash_after`` the run is interrupted and then resumed with the same provider class, so
    the result is a complete run that went through a crash.
    """
    dataset_dir = copy_fixture(tmp_path, fixture)
    config = make_config(run_id, SPLITS[fixture])
    dataset = load_dataset(dataset_dir)
    runs = tmp_path / "runs"
    if crash_after is None:
        execute_run(config, dataset, CrashingProvider(script), runs, **deterministic())
    else:
        with pytest.raises(Crash):
            execute_run(
                config,
                dataset,
                CrashingProvider(script, crash_after=crash_after),
                runs,
                **deterministic(),
            )
        resume_run(
            config,
            dataset,
            CrashingProvider(script and script[crash_after:]),
            runs,
            **deterministic(),
        )
    return SimpleNamespace(run_dir=runs / run_id, dataset_dir=dataset_dir, config=config)


def interrupted_run(tmp_path, fixture="tiny_qa", completed=2, run_id="r1"):
    """A run that stopped after ``completed`` results and was not resumed."""
    dataset_dir = copy_fixture(tmp_path, fixture)
    config = make_config(run_id, SPLITS[fixture])
    with pytest.raises(Crash):
        execute_run(
            config,
            load_dataset(dataset_dir),
            CrashingProvider(crash_after=completed),
            tmp_path / "runs",
            **deterministic(),
        )
    return SimpleNamespace(
        run_dir=tmp_path / "runs" / run_id, dataset_dir=dataset_dir, config=config
    )


def snapshot(*directories: Path) -> dict:
    return {
        str(path): path.read_bytes()
        for directory in directories
        for path in sorted(Path(directory).rglob("*"))
        if path.is_file()
    }


def edit_manifest(run_dir: Path, change) -> None:
    path = Path(run_dir) / "manifest.json"
    manifest = json.loads(path.read_bytes())
    change(manifest)
    path.write_bytes(json.dumps(manifest, indent=2).encode() + b"\n")


def result_lines(run_dir: Path) -> list[bytes]:
    return [line for line in (Path(run_dir) / "results.jsonl").read_bytes().split(b"\n") if line]


def write_result_lines(run_dir: Path, lines: list[bytes]) -> None:
    (Path(run_dir) / "results.jsonl").write_bytes(b"".join(line + b"\n" for line in lines))
