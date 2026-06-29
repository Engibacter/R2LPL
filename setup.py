import os
from pathlib import Path

from setuptools import setup, find_packages

# Change directory to allow installation from anywhere
script_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_folder)

with open("README.md", "r") as fh:
    long_description = fh.read()

repo_root = Path(script_folder).resolve()
results_root = Path(os.environ.get("R2LPL_RESULTS_ROOT", repo_root / "results")).expanduser()
cache_root = Path(os.environ.get("R2LPL_CACHE_ROOT", results_root / "cache")).expanduser()
nuplan_data_root = Path(os.environ.get("NUPLAN_DATA_ROOT", Path.home() / "nuplan" / "dataset")).expanduser()

os.environ.setdefault("R2LPL_ROOT", str(repo_root))
os.environ.setdefault("R2LPL_RESULTS_ROOT", str(results_root))
os.environ.setdefault("R2LPL_CACHE_ROOT", str(cache_root))
os.environ.setdefault("NUPLAN_DATA_ROOT", str(nuplan_data_root))
os.environ.setdefault("NUPLAN_MAPS_ROOT", str(nuplan_data_root / "maps"))
os.environ.setdefault("NUPLAN_EXP_ROOT", str(results_root / "nuplan_exp"))

for path in (
    results_root,
    cache_root,
    results_root / "checkpoints",
    results_root / "planner_anchors",
    results_root / "rollout",
    results_root / "rollout_data",
    results_root / "logs",
):
    path.mkdir(parents=True, exist_ok=True)

setup(
    name="R2LPL",
    version="0.0.1",
    author="Gong Cheng",
    author_email="chenggong@bit.edu.cn",
    description="a testing package for hybrind decision combining mpc and learning",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
        ],
    packages=find_packages(script_folder, exclude=['*test']),
    package_dir={"": "."},
    )
