# Environment

## Verified development snapshot

The repository unit tests were run on 2026-08-03 with:

- Windows and PowerShell;
- Python 3.12.7 from Anaconda;
- PyTorch 2.6.0+cu124;
- Ultralytics 8.4.51;
- NVIDIA GeForce RTX 4070 Laptop GPU, 8 GB VRAM;
- NVIDIA driver 592.00; CUDA available to PyTorch.

This records what was observed, not a clean-environment guarantee. The initial checks used the
Anaconda base environment, which is convenient for validation but is not the recommended final
experiment environment.

## Recommended clean setup

Create a dedicated Python 3.11 or 3.12 environment, install the PyTorch build appropriate for the
host GPU from the official PyTorch selector, then install this repository:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[train,dev]"
python -m pytest -q
```

Before publishing detector results, capture the exact Python package list, GPU model, driver,
Ultralytics version, command, seed, run directory, and best-checkpoint SHA-256 in the run manifest.
