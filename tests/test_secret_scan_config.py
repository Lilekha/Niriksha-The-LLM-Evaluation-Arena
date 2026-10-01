"""Locks the meaning of the detect-secrets baseline's line exclusion.

The baseline skips only lines that consist solely of a ``content_sha256`` pin (a public SHA-256
digest). Everything else on every other line, including a second secret next to a pin, is still
scanned. Fake secrets are assembled from pieces so this file does not trip the scanner itself.
"""

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
AWS = "AKIA" + "IOSFODNN7EXAMPLE"  # the AWS documentation example key
DIGEST = hashlib.sha256(b"pin").hexdigest()  # computed, so no hex literal sits in this file
OTHER_HEX = hashlib.sha256(b"not a pin").hexdigest()  # a different 64-hex value, as a fake "secret"
HOOK = "import sys; from detect_secrets.pre_commit_hook import main; sys.exit(main(sys.argv[1:]))"
KEYWORD = "pass" + 'word = "hunter2hunter2xyz"'


def pin(value=DIGEST, indent="  ", comma=","):
    return f'{indent}"content_sha256": "{value}"{comma}'


def run_hook(baseline: Path, target: Path) -> int:
    return subprocess.run(
        [sys.executable, "-c", HOOK, "--baseline", str(baseline), str(target)],
        capture_output=True,
        text=True,
        check=False,
    ).returncode


def flagged(tmp_path, text: str) -> bool:
    """Run the real hook (in a subprocess, on a copy of the baseline) over ``text``."""
    baseline = tmp_path / "baseline"
    if not baseline.exists():
        shutil.copy(ROOT / ".secrets.baseline", baseline)  # the hook may rewrite its baseline
    target = tmp_path / "sample.txt"
    target.write_bytes(text.encode())
    return run_hook(baseline, target) != 0


@pytest.mark.parametrize(
    ("label", "text"),
    [
        ("pin line in pretty JSON", "{\n" + pin() + '\n  "version": "0.1.0"\n}\n'),
        ("last key without a comma", '{\r\n  "name": "x",\r\n' + pin(comma="") + "\r\n}\r\n"),
        ("tab-indented", "{\n" + pin(indent="\t") + '\n"v": 1}\n'),
    ],
)
def test_a_line_holding_only_the_pin_is_not_flagged(tmp_path, label, text):
    assert not flagged(tmp_path, text), label


@pytest.mark.parametrize(
    ("label", "text"),
    [
        ("fake AWS key", f"aws_access_key_id = {AWS}\n"),
        ("pin and AWS key on one line", f'{{"content_sha256": "{DIGEST}", "k": "{AWS}"}}\n'),
        ("AWS key after the marker", f'"content_sha256": "x", aws_access_key_id = "{AWS}"\n'),
        ("compact JSON, pin among keys", f'{{"name": "x", "content_sha256": "{DIGEST}"}}\n'),
        ("a second secret on the pin line", pin() + f' "{OTHER_HEX}"\n'),
        ("keyword secret after the marker", '"content_sha256": "x", ' + KEYWORD + "\n"),
        ("64-hex under another key", f'  "token": "{OTHER_HEX}",\n'),
        ("marker mentioned in prose", f'Example "content_sha256": "{DIGEST}" and {OTHER_HEX}\n'),
        ("non-sha256 value under the marker key", pin(OTHER_HEX[:40]) + "\n"),
        ("uppercase hex under the marker key", pin(DIGEST.upper()) + "\n"),
    ],
)
def test_everything_else_is_still_flagged(tmp_path, label, text):
    assert flagged(tmp_path, text), label


@pytest.mark.parametrize("name", ["tiny_qa", "tiny_extraction"])
def test_the_committed_fixture_datasets_pass(tmp_path, name):
    baseline = tmp_path / "baseline"
    shutil.copy(ROOT / ".secrets.baseline", baseline)
    path = ROOT / "tests" / "fixtures" / "datasets" / name / "dataset.json"
    assert run_hook(baseline, path) == 0
