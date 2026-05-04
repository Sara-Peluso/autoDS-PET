"""Tests for autods_pet.__init__ - setup_logging."""

import logging

import pytest

from autods_pet import setup_logging


def test_setup_logging_default_level_sets_info():
    # Reset root logger to a known state
    root = logging.getLogger()
    root.setLevel(logging.WARNING)
    root.handlers.clear()
    setup_logging()
    assert root.level == logging.INFO


def test_setup_logging_custom_level_debug():
    root = logging.getLogger()
    root.handlers.clear()
    setup_logging(logging.DEBUG)
    assert root.level == logging.DEBUG


def test_setup_logging_format_includes_levelname():
    root = logging.getLogger()
    root.handlers.clear()
    setup_logging()
    handlers = [h for h in root.handlers if isinstance(h, logging.StreamHandler)]
    assert any(
        "levelname" in (h.formatter._fmt if h.formatter else "") for h in handlers
    )


def test_setup_logging_idempotent():
    root = logging.getLogger()
    root.handlers.clear()
    setup_logging()
    n_before = len(root.handlers)
    setup_logging()
    n_after = len(root.handlers)
    # basicConfig is a no-op when handlers already exist
    assert n_after == n_before


def test_lazy_import_known_attribute():
    """Accessing a lazy-import name returns the real callable."""
    import autods_pet

    assert callable(autods_pet.assign_ds)


def test_lazy_import_caches_on_second_access():
    """Second access returns the cached object (same id)."""
    import autods_pet

    first = autods_pet.ROIResult
    second = autods_pet.ROIResult
    assert first is second


def test_lazy_import_unknown_raises_attributeerror():
    """Unknown attribute raises AttributeError with descriptive message."""
    import autods_pet

    with pytest.raises(AttributeError, match="has no attribute"):
        _ = autods_pet.nonexistent_thing_xyz


def test_all_names_in_dunder_all_importable():
    """Every name in __all__ can be resolved via __getattr__."""
    import autods_pet

    for name in autods_pet.__all__:
        obj = getattr(autods_pet, name)
        assert obj is not None


def test_version_is_string():
    """__version__ is a string."""
    import autods_pet

    assert isinstance(autods_pet.__version__, str)
