"""Regenerate every figure and table in results/ from data/ (about a minute on a laptop)."""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for name in ['tts_stt.py']:
    print("==", name, flush=True)
    subprocess.run([sys.executable, str(HERE / name)], check=True, cwd=HERE.parent)
