"""Configuration of the rotational broadening function (RBF) kernel."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from elisa import settings
from elisa.base.error import ElisaSettingsError

if TYPE_CHECKING:
    from pathlib import Path


def test_rbf_defaults() -> None:
    assert settings.RBF_MESH_SMOOTHING == 1.5
    assert settings.RBF_SIGMA_SMOOTHING_PASSES == 2
    assert settings.RBF_GRID_TOLERANCE == 1e-4


def test_rbf_can_be_configured() -> None:
    with settings.override():
        settings.configure(
            RBF_MESH_SMOOTHING=2.0,
            RBF_SIGMA_SMOOTHING_PASSES=0,
            RBF_GRID_TOLERANCE=1e-3,
        )
        assert settings.RBF_MESH_SMOOTHING == 2.0
        assert settings.RBF_SIGMA_SMOOTHING_PASSES == 0
        assert settings.RBF_GRID_TOLERANCE == 1e-3


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("RBF_MESH_SMOOTHING", 0.0),
        ("RBF_MESH_SMOOTHING", -1.0),
        ("RBF_SIGMA_SMOOTHING_PASSES", -1),
        ("RBF_GRID_TOLERANCE", -1e-6),
    ],
)
def test_invalid_rbf_settings_raise(name: str, value: float) -> None:
    with settings.override(), pytest.raises(ElisaSettingsError):
        settings.configure(**{name: value})


def test_rbf_section_is_read_from_ini(tmp_path: Path) -> None:
    config = tmp_path / "elisa_conf.ini"
    config.write_text(
        "[rbf]\n"
        "mesh_smoothing = 2.25\n"
        "sigma_smoothing_passes = 1\n"
        "grid_tolerance = 0.005\n",
        encoding="utf-8",
    )

    with settings.override():
        settings.read_and_update_config(conf_path=str(config))
        assert settings.RBF_MESH_SMOOTHING == 2.25
        assert settings.RBF_SIGMA_SMOOTHING_PASSES == 1
        assert settings.RBF_GRID_TOLERANCE == 0.005


def test_invalid_rbf_ini_raises(tmp_path: Path) -> None:
    config = tmp_path / "elisa_conf.ini"
    config.write_text("[rbf]\nmesh_smoothing = 0\n", encoding="utf-8")

    with settings.override(), pytest.raises(ElisaSettingsError):
        settings.read_and_update_config(conf_path=str(config))
