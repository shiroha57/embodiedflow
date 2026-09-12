"""Package import boundaries for the 5090 checkpoint bundle.

The official load path (import package -> sha256 check -> torch.load ->
rebuild -> inference) must work with only model.py + config.py present;
dataset.py / stats.py / numpy / jsonschema must not be pulled in.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT / "artifacts" / "checkpoints" / "overfit-fixture-v0"


def _run(code: str, pythonpath: Path) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(pythonpath)}
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=300, env=env,
    )


def test_package_import_is_lightweight():
    code = (
        "import embodiedflow.dataplane, sys\n"
        "bad = [m for m in ('numpy', 'yaml', 'torch',\n"
        "                   'embodiedflow.dataplane.dataset',\n"
        "                   'embodiedflow.dataplane.stats',\n"
        "                   'embodiedflow.dataplane.checkpoint')\n"
        "       if m in sys.modules]\n"
        "assert not bad, bad\n"
        "print('lightweight ok')\n"
    )
    proc = _run(code, ROOT / "src")
    assert proc.returncode == 0, proc.stderr
    assert "lightweight ok" in proc.stdout


def test_official_load_path_in_minimal_bundle(tmp_path: Path):
    """Copy only the files the 5090 bundle needs; run the documented
    commands; no dataset.py / stats.py anywhere on the path."""
    bundle = tmp_path / "bundle"
    (bundle / "embodiedflow" / "dataplane").mkdir(parents=True)
    shutil.copy(ROOT / "src" / "embodiedflow" / "__init__.py", bundle / "embodiedflow")
    for name in ("__init__.py", "config.py", "model.py"):
        shutil.copy(
            ROOT / "src" / "embodiedflow" / "dataplane" / name,
            bundle / "embodiedflow" / "dataplane",
        )
    shutil.copy(CHECKPOINT_DIR / "model.pt", tmp_path / "model.pt")
    shutil.copy(CHECKPOINT_DIR / "model_manifest.json", tmp_path / "model_manifest.json")

    code = f"""
import hashlib, json, sys, torch
import embodiedflow.dataplane
# numpy may come in via torch's own import chain; our dataset/stats modules
# must NOT be required for the load path
assert 'embodiedflow.dataplane.dataset' not in sys.modules, sys.modules.keys()
assert 'embodiedflow.dataplane.stats' not in sys.modules, sys.modules.keys()

manifest = json.load(open({str(tmp_path / 'model_manifest.json')!r}, encoding='utf-8'))
sha = hashlib.sha256(open({str(tmp_path / 'model.pt')!r}, 'rb').read()).hexdigest()
assert sha == manifest['checkpoint_sha256'], (sha, manifest['checkpoint_sha256'])

ckpt = torch.load({str(tmp_path / 'model.pt')!r}, map_location='cpu', weights_only=False)
from embodiedflow.dataplane.model import ChunkedActionTransformerMinimal
from embodiedflow.dataplane.config import ModelConfig
model = ChunkedActionTransformerMinimal(ModelConfig(**ckpt['model_config']))
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
rgb = torch.rand(1, 3, 32, 32)
state = torch.randn(1, 13)
with torch.no_grad():
    chunk = model(rgb, state)
assert tuple(chunk.shape) == (1, 20, 7), tuple(chunk.shape)
# still absent after the full load path
assert 'embodiedflow.dataplane.dataset' not in sys.modules, sys.modules.keys()
assert 'embodiedflow.dataplane.stats' not in sys.modules, sys.modules.keys()
print('minimal-bundle load ok', tuple(chunk.shape), 'step', ckpt['global_step'])
"""
    proc = _run(code, bundle)
    assert proc.returncode == 0, proc.stderr
    assert "minimal-bundle load ok (1, 20, 7)" in proc.stdout
