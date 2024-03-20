# Metric Based Few-Shot Graph Classification

<p align="center">
    <a href="https://github.com/lucmos/nn-template"><img alt="NN Template" src="https://shields.io/badge/nn--template-0.0.2-emerald?style=flat&labelColor=gray"></a>
    <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-3.8-blue.svg"></a>
    <a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

Codebase for the paper [Metric Based Few-Shot Graph Classification](https://proceedings.mlr.press/v198/crisostomi22a.html), published at Learning on Graphs (2022).

## Installation

Setup the development environment:

```bash
conda create --name fs-grl python=3.9
conda activate fs-grl
```
Install PyTorch with CUDA support according to https://pytorch.org/get-started/locally/.

Install PyG
```bash
conda install pyg -c pyg
```

Install the project in edit mode:
```bash
pip install -e .
```

### Download data

Download the versioned datasets:
```bash
dvc pull
dvc checkout
```

### Training a model
You can train and evaluate various families of models by running the corresponding script in the `scripts` folder. For example, to train a Distance Metric Learning model, you can run:
```bash
python fs_grl/scripts/run_dml.py
```
