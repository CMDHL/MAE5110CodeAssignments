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
[the Assignment 1 report](assignment_1_report.md). Run each experiment on its
own with:

```console
uv run python assignment_1.py sanity
uv run python assignment_1.py roa
uv run python assignment_1.py return-map
uv run python assignment_1.py floquet
uv run python assignment_1.py slope-sweep
uv run python assignment_1.py spoke-sweep
```

Each command writes the figure or table for that specific part of the
assignment into `assignment_1_results/`.
