"""Import hygiene, and the compatibility surface downstream code depends on."""

from __future__ import annotations

import subprocess
import sys

import pytest

PYTHON = sys.executable


def _in_fresh_process(code: str) -> str:
    result = subprocess.run(
        [PYTHON, "-c", code], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return result.stdout.strip()


def test_the_light_layers_do_not_import_jax_or_numpyro():
    """The plotting environment must be able to read fits without the fit stack."""
    out = _in_fresh_process(
        "import sys, baystarrfish, baystarrfish.data, baystarrfish.stats, baystarrfish.io;"
        "print('jax' in sys.modules, 'numpyro' in sys.modules)"
    )
    assert out == "False False"


def test_touching_a_model_symbol_pulls_in_numpyro():
    out = _in_fresh_process(
        "import sys, baystarrfish as b; b.MODEL_FAMILIES;"
        "print('numpyro' in sys.modules)"
    )
    assert out == "True"


def test_x64_is_active_whichever_import_comes_first():
    for code in (
        "import baystarrfish, jax.numpy as jnp; print(jnp.zeros(1).dtype)",
        "import jax.numpy as jnp, baystarrfish; baystarrfish._jax_setup.assert_x64_enabled();"
        "print(jnp.zeros(1).dtype)",
    ):
        assert _in_fresh_process(code) == "float64"


def test_every_advertised_export_resolves():
    import baystarrfish as bsf

    for name in bsf.__all__:
        assert getattr(bsf, name) is not None, name


def test_fit_aliases_point_at_the_run_drivers():
    import baystarrfish as bsf

    assert bsf.fit is bsf.run_model
    assert bsf.fit_decoupled is bsf.run_decoupled_model


def test_unknown_attribute_raises_attribute_error():
    import baystarrfish as bsf

    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        bsf.nope


def test_the_legacy_starrfish_module_still_works_with_a_warning():
    """revision/archive/ scripts import it by the old name; keep them running."""
    out = _in_fresh_process(
        "import sys, warnings, pathlib;"
        "sys.path.insert(0, str(pathlib.Path('STARRFISH_in_vivo/STARRFISH').resolve()));"
        "warnings.simplefilter('always');"
        "\nwith warnings.catch_warnings(record=True) as w:\n"
        "    import bayesian_hierarchical as bh\n"
        "print(any(issubclass(x.category, DeprecationWarning) for x in w),"
        " bh.run_model.__module__)"
    )
    assert out == "True baystarrfish.inference.run"


def test_analysis_utils_shim_still_exposes_what_the_scripts_import():
    import ast
    import pathlib

    code_dir = pathlib.Path("revision/bayesian_vs_fold_change/code")
    if not code_dir.exists():
        pytest.skip("analysis directory not present in this checkout")
    sys.path.insert(0, str(code_dir))
    try:
        import analysis_utils
    finally:
        sys.path.pop(0)

    wanted = set()
    for path in code_dir.glob("*.py"):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "analysis_utils":
                wanted |= {alias.name for alias in node.names}
    missing = sorted(name for name in wanted if not hasattr(analysis_utils, name))
    assert not missing, f"analysis_utils shim lost: {missing}"


def test_the_package_cli_dispatches_to_both_commands():
    from baystarrfish.__main__ import COMMANDS

    assert set(COMMANDS) == {"copy-number", "recovery"}
    for command in COMMANDS:
        out = subprocess.run(
            [PYTHON, "-m", "baystarrfish", command, "--help"],
            capture_output=True, text=True, timeout=300,
        )
        assert out.returncode == 0, out.stderr[-1000:]
        assert f"python -m baystarrfish {command}" in out.stdout
        # runpy warns when a module its package already imported is run with -m;
        # the dispatcher exists precisely so that does not happen.
        assert "RuntimeWarning" not in out.stderr


def test_the_cli_reports_an_unknown_command():
    out = subprocess.run(
        [PYTHON, "-m", "baystarrfish", "nonsense"], capture_output=True, text=True, timeout=300
    )
    assert out.returncode != 0
    assert "invalid choice" in out.stderr
