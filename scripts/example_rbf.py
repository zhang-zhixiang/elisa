# ruff: noqa: INP001, T201
"""Minimal example of rotational broadening function (RBF) calculation.

The example builds a circular binary system and evaluates the RBF on a
uniform velocity grid.  It assumes that the ELISa atmosphere and
limb-darkening tables are configured in the standard way (see README).
"""

import numpy as np
from astropy import units as u

from elisa import BinarySystem, Observer, Star

primary = Star(
    mass=2.0 * u.solMass,
    t_eff=10000 * u.K,
    gravity_darkening=1.0,
    albedo=1.0,
    metallicity=0.0,
    synchronicity=1.0,
    surface_potential=15.0,
)
secondary = Star(
    mass=1.0 * u.solMass,
    t_eff=5000 * u.K,
    gravity_darkening=1.0,
    albedo=1.0,
    metallicity=0.0,
    synchronicity=1.0,
    surface_potential=15.0,
)

system = BinarySystem(
    primary=primary,
    secondary=secondary,
    argument_of_periastron=0.0 * u.deg,
    gamma=0.0 * u.km / u.s,
    period=5.0 * u.d,
    eccentricity=0.0,
    inclination=87.0 * u.deg,
    primary_minimum_time=0.0 * u.d,
    phase_shift=0.0,
)

observer = Observer(system=system)
phases, velocity_grid, rbf = observer.observe.rbf(
    from_phase=-0.2,
    to_phase=0.2,
    phase_step=0.05,
    v_start=-300.0,
    v_stop=300.0,
    v_step=1.0,
)

print(f"phases: {phases.shape}")
print(f"velocity grid: {velocity_grid.shape} ({velocity_grid.unit})")
for component, profile in rbf.items():
    print(f"{component}: shape={profile.shape}, integrated={np.sum(profile, axis=1)}")
