"""Tests for autods_pet.manifest - reproducibility manifest."""

import json

from autods_pet.manifest import collect_environment, write_manifest


def test_collect_environment_keys():
    env = collect_environment()
    assert "autods_pet_version" in env
    assert "python_version" in env
    assert "platform" in env
    assert "packages" in env


def test_collect_environment_version_string():
    env = collect_environment()
    from autods_pet import __version__

    assert env["autods_pet_version"] == __version__


def test_collect_environment_packages_are_dict():
    env = collect_environment()
    assert isinstance(env["packages"], dict)
    assert "numpy" in env["packages"]
    assert "SimpleITK" in env["packages"]


def test_collect_environment_installed_package_has_version():
    env = collect_environment()
    assert env["packages"]["numpy"] != "not installed"


def test_write_manifest_creates_json(tmp_path):
    cfg = {"paths": {"basepath": "/data"}, "liver": {"erosion_mm": 10.0}}
    path = write_manifest(cfg, tmp_path)
    assert path is not None
    assert path.name == "manifest.json"
    assert path.exists()

    data = json.loads(path.read_text())
    assert data["config"] == cfg
    assert "timestamp" in data
    assert "autods_pet_version" in data
    assert "packages" in data


def test_write_manifest_returns_none_on_bad_dir(tmp_path):
    cfg = {"paths": {"basepath": "/data"}}
    bad_dir = tmp_path / "nonexistent" / "nested"
    result = write_manifest(cfg, bad_dir)
    assert result is None
