"""Guards for the lazy ``verso.engine`` facade (PEP 562 ``__getattr__``).

The facade resolves each public name to its submodule on first access instead
of importing everything eagerly (which pulled ``scipy.spatial`` in at GUI
startup). These tests ensure the lazy table stays honest: every advertised name
resolves, the table and ``__all__`` agree, and unknown names still raise.
"""

import importlib

import verso.engine as engine


def test_every_public_name_resolves():
    """Each entry in ``__all__`` imports without error and is the real symbol."""
    for name in engine.__all__:
        obj = getattr(engine, name)
        assert obj is not None
        # Re-importing the source module and fetching the name yields the same object.
        module_path = engine._EXPORTS[name]
        source_obj = getattr(importlib.import_module(module_path), name)
        assert obj is source_obj


def test_all_and_exports_are_in_sync():
    """``__all__`` and the lazy ``_EXPORTS`` table cover exactly the same names."""
    assert set(engine.__all__) == set(engine._EXPORTS)


def test_dir_lists_public_api():
    assert set(dir(engine)) >= set(engine.__all__)


def test_unknown_attribute_raises():
    try:
        engine.does_not_exist  # noqa: B018
    except AttributeError:
        return
    raise AssertionError("expected AttributeError for unknown attribute")
