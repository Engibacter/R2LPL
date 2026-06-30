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
    description="Rollout-Retrieval Lifelong Policy Learning for autonomous driving",
    url="https://github.com/Engibacter/R2LPL",
    license="Apache-2.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent"
        ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=['*test']),
    package_dir={"": "."},
    include_package_data=True,
    package_data={"lpl_planner": ["config/**/*.yaml"]},
    python_requires=">=3.9",
    )
