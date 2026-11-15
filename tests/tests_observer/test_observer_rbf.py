"""Observer-level checks of the rotational broadening function API."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from elisa import settings
from elisa import units as u
from elisa.base.error import ElisaNotImplementedError, ElisaOperationInputError
from elisa.base.star import Star
from elisa.binary_system.system import BinarySystem
from elisa.observer.observer import Observer
from tests import set_astropy_units
from tests.utils import BINARY_SYSTEM_PARAMS, ElisaTestCase, prepare_binary_system

set_astropy_units()


class ObserverRBFTestCase(ElisaTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.write_default_support(ld_tables=settings.LD_TABLES, atm_tables=settings.CK04_ATM_TABLES)
        self.system = prepare_binary_system(BINARY_SYSTEM_PARAMS["detached-physical"])
        self.phases = np.array([0.0, 0.25, 0.5])

    def test_shapes_units_and_finiteness(self) -> None:
        observer = Observer(system=self.system)
        phases, velocity_grid, curves = observer.rbf(
            phases=self.phases,
            v_start=-400,
            v_stop=400,
            v_step=5,
        )

        assert_allclose(phases, self.phases + self.system.phase_shift)
        assert velocity_grid.unit == u.m / u.s
        assert velocity_grid.size == 160
        assert set(curves) == {"primary", "secondary"}
        for component, values in curves.items():
            assert values.shape == (self.phases.size, velocity_grid.size), component
            assert np.isfinite(values).all(), component
            assert np.all(values.sum(axis=1) > 0.0), component

    def test_accepts_quantity_velocity_parameters(self) -> None:
        observer = Observer(system=self.system)
        _, grid_plain, curves_plain = observer.rbf(
            phases=self.phases,
            v_start=-400,
            v_stop=400,
            v_step=5,
        )
        _, grid_quantity, curves_quantity = observer.rbf(
            phases=self.phases,
            v_start=-400 * u.km / u.s,
            v_stop=400 * u.km / u.s,
            v_step=5 * u.km / u.s,
        )

        assert_allclose(grid_plain.value, grid_quantity.value)
        for component in curves_plain:
            assert_allclose(curves_plain[component], curves_quantity[component], rtol=1e-12)

    def test_invalid_velocity_grid(self) -> None:
        observer = Observer(system=self.system)
        with pytest.raises(ElisaOperationInputError):
            observer.rbf(phases=self.phases, v_start=0, v_stop=100, v_step=0)
        with pytest.raises(ElisaOperationInputError):
            observer.rbf(phases=self.phases, v_start=100, v_stop=0, v_step=1)

    def test_single_star_system_is_rejected(self) -> None:
        observer = Observer()
        with pytest.raises(ElisaNotImplementedError):
            observer.rbf(phases=self.phases)

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

    def test_pulsating_system(self) -> None:
        observer = Observer(system=self._prepare_pulsating_system())
        phases, velocity_grid, curves = observer.rbf(
            phases=np.array([0.0, 0.3]),
            v_start=-400,
            v_stop=400,
            v_step=10,
        )

        assert phases.size == 2
        assert velocity_grid.size == 80
        assert set(curves) == {"primary", "secondary"}
        for component, values in curves.items():
            assert values.shape == (2, 80), component
            assert np.isfinite(values).all(), component
            assert np.all(values.sum(axis=1) > 0.0), component

    def test_single_vs_multiprocess(self) -> None:
        observer = Observer(system=self.system)

        settings.configure(NUMBER_OF_PROCESSES=1)
        _, _, single = observer.rbf(phases=self.phases, v_start=-400, v_stop=400, v_step=5)

        settings.configure(NUMBER_OF_PROCESSES=2)
        _, _, multi = observer.rbf(phases=self.phases, v_start=-400, v_stop=400, v_step=5)

        for component in single:
            assert_allclose(single[component], multi[component], rtol=1e-8, atol=0.0)
