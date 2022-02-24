# FS-GRL

<p align="center">
    <a href="https://github.com/lucmos/nn-template"><img alt="NN Template" src="https://shields.io/badge/nn--template-0.0.2-emerald?style=flat&labelColor=gray"></a>
    <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-3.8-blue.svg"></a>
    <a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

A new awesome project.


## Installation

```bash
pip install git+ssh://git@github.com/crisostomi/fs-grl.git
```


## Quickstart

[comment]: <> (> Fill me!)


## Development installation

Setup the development environment:

```bash
git clone git+ssh://git@github.com/crisostomi/fs-grl.git
conda env create -f env.yaml
conda activate fs-grl
pre-commit install
```

Run the tests:

```bash
pre-commit run --all-files
pytest -v
```


### Download data

Download the versioned datasets:
```bash
dvc pull  # Login with your @di email when prompted
dvc checkout
```


### Update the dependencies

Re-install the project in edit mode:

```bash
pip install -e .[dev]
```
