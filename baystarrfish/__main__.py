"""Command-line entry point: ``python -m baystarrfish <command> [options]``.

Lives here rather than behind ``if __name__ == "__main__"`` in each module: a
submodule that its package's ``__init__`` already imports cannot be run with
``python -m`` without runpy warning that it executed twice.

Fitting is deliberately absent. It needs a GPU, tens of hours and a slurm
allocation, and the runners under ``revision/bayesian_vs_fold_change/code/``
carry the output-directory conventions that go with it.
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

COMMANDS = {
    "copy-number": (
        "baystarrfish.inference.copy_number",
        "infer the cell x cCRE matrix of latent AAV copy number from a fit",
    ),
    "recovery": (
        "baystarrfish.simulate.recovery",
        "simulate from known parameters, fit, and check the truth is recovered",
    ),
}


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="python -m baystarrfish",
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="commands:\n"
        + "\n".join(f"  {name:<14} {help}" for name, (_, help) in COMMANDS.items()),
    )
    parser.add_argument("command", choices=sorted(COMMANDS), help=argparse.SUPPRESS)
    known, rest = parser.parse_known_args(argv[:1] if argv else argv)
    module_path = COMMANDS[known.command][0]

    from importlib import import_module

    module = import_module(module_path)
    entry = getattr(module, "_main", None) or getattr(module, "main")
    return int(entry(argv[1:]) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
