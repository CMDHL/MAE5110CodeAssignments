# MAE 5110 Code Assignments

Code assignments for MAE 5110.

## Installation

Install [Git](https://git-scm.com/downloads) and [uv](https://docs.astral.sh/uv/getting-started/installation/). After cloning this repository, run the following command from its root directory:

```console
uv sync --python 3.14
```

This creates a local `.venv` and installs the required dependencies. Run Python commands inside the environment with `uv run`, for example:

```console
uv run python assignment_0.py
```

## Assignments

- [Assignment 0](assignments/assignment_0.md)
- [Assignment 1](assignments/assignment_1.md)

### Assignment 1 results

The completed analysis and discussion are in
[the Assignment 1 report](assignment_1_report.md). Regenerate every figure and
numeric table with:

```console
uv run python assignment_1.py
```

For a faster smoke run, add `--quick`. Run the focused checks with:

```console
uv run pytest -q
uv run ruff check assignment_1.py models/rimless_wheel.py tests/
```
