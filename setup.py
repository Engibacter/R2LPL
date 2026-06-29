from pathlib import Path

from setuptools import setup, find_packages

repo_root = Path(__file__).resolve().parent

with (repo_root / "README.md").open("r", encoding="utf-8") as fh:
    long_description = fh.read()

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
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=['*test']),
    package_dir={"": "."},
    include_package_data=True,
    package_data={"lpl_planner": ["config/**/*.yaml"]},
    )
