"""Set up BeatMap AI on a new PC: detect the graphics card, create a Python environment and
install the PyTorch build that can use it.

    NVIDIA  -> .venv-cuda  (PyTorch with CUDA, from download.pytorch.org)
    AMD     -> .venv-rocm  (PyTorch with ROCm for Windows, from repo.radeon.com; Python 3.12)
    none    -> .venv-cpu   (PyTorch for the processor only)

Run it with the Python it should use (3.10-3.12; 3.12 for AMD), e.g. via
"BeatMap AI einrichten.bat". ``--target nvidia|amd|cpu`` overrides the detection.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROCM = "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/"
AMD_PACKAGES = [
    ROCM + "rocm_sdk_core-7.2.1-py3-none-win_amd64.whl",
    ROCM + "rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl",
    ROCM + "rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl",
    ROCM + "rocm-7.2.1.tar.gz",
]
AMD_TORCH = [ROCM + "torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl",
             ROCM + "torchaudio-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl"]
TARGETS = {"nvidia": ".venv-cuda", "amd": ".venv-rocm", "cpu": ".venv-cpu"}


def graphics_cards() -> list[str]:
    if os.name != "nt":
        return []
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_VideoController).Name"],
                             capture_output=True, text=True, timeout=15).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def detect_target(cards: list[str]) -> str:
    names = " ".join(cards).lower()
    if "nvidia" in names:
        return "nvidia"
    if "radeon" in names or "amd" in names:
        return "amd"
    return "cpu"


def run(*args: str) -> None:
    print(">", " ".join(args), flush=True)
    subprocess.run(args, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", choices=tuple(TARGETS))
    parser.add_argument("--yes", action="store_true", help="do not ask before installing")
    args = parser.parse_args()

    cards = graphics_cards()
    target = args.target or detect_target(cards)
    print("Grafikkarte(n):", ", ".join(cards) if cards else "keine erkannt")
    if target == "amd" and sys.version_info[:2] != (3, 12):
        print("AMD (ROCm unter Windows) braucht Python 3.12. Installiere Python 3.12 und starte "
              "diese Einrichtung damit, oder wähle --target cpu.")
        sys.exit(1)
    venv = ROOT / TARGETS[target]
    labels = {"nvidia": "NVIDIA (CUDA)", "amd": "AMD (ROCm)", "cpu": "nur Prozessor (CPU)"}
    print(f"Einrichtung für {labels[target]} in {venv.name} (Download ca. 1-3 GB).")
    if not args.yes and input("Fortfahren? [j/N] ").strip().lower() not in ("j", "ja", "y", "yes"):
        print("Abgebrochen.")
        return

    if not venv.exists():
        run(sys.executable, "-m", "venv", str(venv))
    python = str(venv / ("Scripts" if os.name == "nt" else "bin") / "python")
    run(python, "-m", "pip", "install", "--upgrade", "pip")
    if target == "nvidia":
        run(python, "-m", "pip", "install", "torch", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu128")
    elif target == "amd":
        run(python, "-m", "pip", "install", "--no-cache-dir", *AMD_PACKAGES)
        run(python, "-m", "pip", "install", "--no-cache-dir", *AMD_TORCH)
    else:
        run(python, "-m", "pip", "install", "torch", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cpu")
    run(python, "-m", "pip", "install", "-e", str(ROOT), "rosu-pp-py")
    check = ("import torch; ok = torch.cuda.is_available(); "
             "print('Grafikkarte nutzbar:', torch.cuda.get_device_name(0) if ok else 'nein (CPU)')")
    run(python, "-c", check)
    print("Fertig. Starte BeatMap AI mit „Start BeatMap AI.bat“.")


if __name__ == "__main__":
    main()
