<p align="center">
<h2 align="center">Jina Models on AWS SageMaker</h2>
</p>


<p align=center>
<a href="https://pypi.org/project/jina-sagemaker/"><img alt="PyPI" src="https://img.shields.io/pypi/v/jina-sagemaker?label=Release&style=flat-square"></a>
<a href="https://discord.jina.ai"><img src="https://img.shields.io/discord/1106542220112302130?logo=discord&logoColor=white&style=flat-square"></a>
<a href="https://pypistats.org/packages/jina-sagemaker"><img alt="PyPI - Downloads from official pypistats" src="https://img.shields.io/pypi/dm/jina-sagemaker?style=flat-square"></a>
</p>

`jina-sagemaker` package offers streamlined tools for interacting with [Jina Embedding Models through the AWS SageMaker Marketplace](), all within a Python environment.

## Installation

Install from PyPI:

```bash
pip install --upgrade jina-sagemaker
```

Install from source:

```bash
pip install .
```

Build distributions locally:

```bash
pip install build
python -m build
```

## Usage

Please configure your AWS credentials before using this package. You can do this by following the instructions [here](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html).

Please follow the examples in `notebooks` to get an overview of how to use model packages offered for real time inference and batch transform jobs.

## Development

```bash
pip install -e .[test]
pytest
```

Formatting and linting use [`ruff`](https://github.com/astral-sh/ruff); configuration lives in `pyproject.toml`:

```bash
pip install ruff
ruff format .       # apply formatting
ruff check --fix .  # apply lint autofixes
```

A pre-commit hook is provided:

```bash
pip install pre-commit
pre-commit install
```
