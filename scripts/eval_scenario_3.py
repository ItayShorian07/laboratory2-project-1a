"""Shortcut for scenario 3."""
import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.argv = [str(Path(__file__).name), "--scenario", "3"] + sys.argv[1:]
    runpy.run_path(str(Path(__file__).resolve().parent / "eval_scenario.py"), run_name="__main__")
