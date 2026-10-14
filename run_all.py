"""
Master script: run all steps in sequence.
Usage: python run_all.py
"""
import subprocess, sys, os

BASE = os.path.dirname(os.path.abspath(__file__))
steps = [
    ("Step 1: Data Preparation",  os.path.join(BASE, "01_data_prep.py")),
    ("Step 2: Rolling Training",  os.path.join(BASE, "03_rolling_train.py")),
    ("Step 3: Results & Figures", os.path.join(BASE, "04_results.py")),
]

for desc, script in steps:
    print(f"\n{'#'*70}\n# {desc}\n{'#'*70}")
    ret = subprocess.run([sys.executable, script], cwd=BASE)
    if ret.returncode != 0:
        print(f"ERROR in {script}, stopping.")
        sys.exit(1)

print("\nAll steps complete.")
