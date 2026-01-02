## Purpose
Provide concise, actionable guidance for AI coding agents working in this repository so they can be immediately productive with the project's notebooks and local virtual environment.

## Big picture (what this repo contains)
- Primary artifacts: Jupyter notebooks in `notebooks/` (e.g. `notebooks/hello_world.ipynb`, `notebooks/qiskit_fundamentals.ipynb`).
- A prebuilt Python virtual environment is checked in under `qfhack/` (has `pyvenv.cfg`, `Scripts/`, `Lib/site-packages/`). The notebooks run against that environment.

## Where to focus edits
- Most living code is inside the notebooks. Small, reusable Python modules are uncommon in this repo—check for a `src/` or top-level .py files before adding one.
- Do NOT edit `qfhack/Lib/site-packages/` or other vendored site-packages files. Treat `qfhack/` as the project's virtualenv.

## How to run / reproduce locally (PowerShell)
1. Open PowerShell at the repo root.
2. Activate the included virtualenv:

   & .\qfhack\Scripts\Activate.ps1

3. Start the notebook server (if needed):

   jupyter lab

Or run a single notebook headlessly (useful for CI/emulation):

   jupyter nbconvert --to notebook --execute notebooks/hello_world.ipynb --ExecutePreprocessor.timeout=600 --output executed_hello_world.ipynb

Notes:
- The checked-in venv already contains many packages (see `qfhack/Lib/site-packages/`); prefer using it rather than installing globally.
- If a package is missing, modify the active environment (pip install into the `qfhack` venv) rather than the system Python.

## Project-specific patterns & conventions
- Notebooks are the source of truth. When adding functionality that should be reused across notebooks, prefer extracting it to a new top-level Python module and reference it explicitly from notebooks (or add a small package and document install steps).
- Kernel behavior: notebooks are expected to run against the `qfhack` environment. When testing execution, make sure the kernel matches that environment.
- Avoid changing files in `qfhack/Scripts` or `qfhack/Lib` unless you understand virtualenv semantics.

## Integration and external dependencies
- Jupyter and ipykernel are present in `qfhack/Lib/site-packages/` — kernel launch and notebook execution rely on these.
- The repo contains several third-party wheels/packages inside the venv (example: a `scipy` wheel and `ipykernel`). Treat them as part of the runtime image.

## Examples (explicit pointers)
- To inspect the basic example, open `notebooks/hello_world.ipynb` and run cells after activating the venv above.
- To run an end-to-end execution from the command line, use the nbconvert example above.

## When in doubt
- Prefer non-destructive edits to notebooks (add new cells or new notebooks rather than overwriting historically important ones).
- If you need to add project-level automation (tests, packaging), add a new top-level directory (for example `tools/` or `src/`) and document activation + install instructions in this file.

If anything here is unclear or you want more detailed instructions (CI, packaging, extracting reusable modules), tell me what to expand and I'll iterate.
