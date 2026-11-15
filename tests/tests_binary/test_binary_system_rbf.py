"""Binary-system level checks of the rotational broadening function."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from elisa import settings
from elisa import units as u
from elisa.base.error import ElisaOperationInputError
from elisa.base.star import Star
from elisa.binary_system.system import BinarySystem
from tests import set_astropy_units
from tests.utils import BINARY_SYSTEM_PARAMS, ElisaTestCase, prepare_binary_system

set_astropy_units()


class BinarySystemRBFTestCase(ElisaTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.write_default_support(ld_tables=settings.LD_TABLES, atm_tables=settings.CK04_ATM_TABLES)
        self.system = prepare_binary_system(BINARY_SYSTEM_PARAMS["detached-physical"])
        self.phases = np.array([0.0, 0.25, 0.5])
        self.velocity_grid = np.arange(-400_000.0, 400_001.0, 5_000.0)

    def _compute(self, system: BinarySystem, phases: np.ndarray) -> dict[str, np.ndarray]:
        return system.compute_rbf(
            phases=phases,
            velocity_grid=self.velocity_grid,
            position_method=system.get_positions_method(),
        )

    def test_compute_rbf_shapes_and_finiteness(self) -> None:
        curves = self._compute(self.system, self.phases)

        assert set(curves) == {"primary", "secondary"}
        for component, values in curves.items():
            assert values.shape == (self.phases.size, self.velocity_grid.size), component
            assert np.isfinite(values).all(), component
            assert np.all(values.sum(axis=1) > 0.0), component

    def test_compute_rbf_rejects_invalid_grid(self) -> None:
        with pytest.raises(ElisaOperationInputError):
            self.system.compute_rbf(
                phases=self.phases,
                velocity_grid=np.array([0.0]),
                position_method=self.system.get_positions_method(),
            )

    def _prepare_pulsating_system(self) -> BinarySystem:
        pulse_meta = [
            {
                "l": 2,
                "m": 1,
                "amplitude": 1.0 * u.km / u.s,
                "frequency": 16 / u.d,
                "start_phase": 0.0,
                "horizontal_to_radial_amplitude_ratio": 0.0,
                "temperature_amplitude_factor": 0.01,
            },
        ]
        star_params = {
            "mass": 2.0 * u.solMass,
            "t_eff": 10000 * u.K,
            "gravity_darkening": 1.0,
            "discretization_factor": 5,
            "albedo": 1.0,
            "metallicity": 0.0,
            "synchronicity": 1.0,
            "surface_potential": 10,
        }
        primary = Star(pulsations=pulse_meta, **star_params)
        secondary = Star(pulsations=pulse_meta, **star_params)
        return BinarySystem(
            primary=primary,
            secondary=secondary,
            argument_of_periastron=0 * u.deg,
            gamma=0 * u.km / u.s,
            period=2.0 * u.d,
            eccentricity=0.0,
            inclination=87 * u.deg,
            primary_minimum_time=0 * u.d,
            phase_shift=0.0,
        )

    def test_compute_rbf_pulsating_system(self) -> None:
        system = self._prepare_pulsating_system()
        phases = np.array([0.0, 0.3])
        curves = system.compute_rbf(
            phases=phases,
            velocity_grid=self.velocity_grid,
            position_method=system.get_positions_method(),
        )

        assert set(curves) == {"primary", "secondary"}
        for component, values in curves.items():
            assert values.shape == (phases.size, self.velocity_grid.size), component
            assert np.isfinite(values).all(), component
            assert np.all(values.sum(axis=1) > 0.0), component

    def test_compute_rbf_eccentric_system_uses_exact_integration(self) -> None:
        system = prepare_binary_system(BINARY_SYSTEM_PARAMS["detached.ecc"])
        curves = self._compute(system, np.array([0.0, 0.2, 0.45]))

        assert set(curves) == {"primary", "secondary"}
        for component, values in curves.items():
            assert values.shape == (3, self.velocity_grid.size), component
            assert np.isfinite(values).all(), component
            assert np.all(values.sum(axis=1) > 0.0), component

    def test_compute_rbf_single_vs_multiprocess(self) -> None:
        settings.configure(NUMBER_OF_PROCESSES=1)
        single = self._compute(self.system, self.phases)

        settings.configure(NUMBER_OF_PROCESSES=2)
        multi = self._compute(self.system, self.phases)

        for component in single:
            assert_allclose(single[component], multi[component], rtol=1e-8, atol=0.0)
