"""Execute the trusted bundled notebook's print-based cells without a kernel.

This deliberately small runner supports sequential Python cells and stdout;
it does not implement Jupyter magics, widgets, rich displays or a kernel.
"""

import contextlib
import io
from pathlib import Path

import nbformat


def execute(path):
    notebook = nbformat.read(path, as_version=4)
    namespace = {"__name__": "__main__"}
    count = 0
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        count += 1
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(compile(cell.source, str(path), "exec"), namespace)
        cell.execution_count = count
        cell.outputs = []
        if output.getvalue():
            cell.outputs = [
                nbformat.v4.new_output("stream", name="stdout", text=output.getvalue())
            ]
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    return count


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    # Module imports are resolved after `pip install -e .` or from repo root.
    print(f"Executed {execute(root / 'notebooks/quickstart.ipynb')} Python cells")
