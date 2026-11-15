from __future__ import annotations

# The kernel uses fixed Gaussian truncation factors and bin-edge arithmetic.
# ruff: noqa: PLR2004
import math
import warnings
from typing import TYPE_CHECKING

import numpy as np
from numba import njit
from scipy.special import erf

from elisa import const, settings
from elisa.base.curves import utils as crv_utils
from elisa.base.error import raise_elisa_operation_input_error

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa.base.container import StarContainer
    from elisa.binary_system.container import OrbitalPositionContainer


# (component, grid lower bound, grid upper bound) of the coverage warnings that
# have already been issued.  In a multi-phase run the same warning would
# otherwise be repeated for every orbital position.
_WARNED_GRIDS: set[tuple[str | None, float, float]] = set()


class RBFGridCoverageWarning(UserWarning):
    """Issued when the velocity grid does not cover the line-of-sight velocities.

    In that case part of the visible flux falls outside the grid and the
    integral of the RBF is smaller than the total flux of the component.
    """



def validate_velocity_grid(velocity_grid: NDArray) -> NDArray[np.float64]:
    """Validate and normalize a velocity grid used by the RBF calculation.

    The grid has to be one-dimensional, strictly increasing, finite and
    uniformly spaced. Values are expected in metres per second.

    :param velocity_grid: Candidate velocity grid.
    :type velocity_grid: numpy.typing.NDArray
    :raises ElisaOperationInputError: If the grid is not one-dimensional,
        contains fewer than two points, is not finite, is not strictly
        increasing or is not uniformly spaced.
    :return: Validated grid as a ``float64`` array.
    :rtype: numpy.typing.NDArray[numpy.float64]
    """
    grid = np.asarray(velocity_grid, dtype=np.float64)
    if grid.ndim != 1:
        raise_elisa_operation_input_error("velocity_grid must be a one-dimensional array")
    if grid.size < 2:
        raise_elisa_operation_input_error("velocity_grid must contain at least two points")
    if not np.all(np.isfinite(grid)):
        raise_elisa_operation_input_error("velocity_grid must contain only finite values")

    diff = np.diff(grid)
    if np.any(diff <= 0.0):
        raise_elisa_operation_input_error("velocity_grid must be strictly increasing")
    if not np.allclose(diff, diff[0], rtol=1e-7, atol=1e-9):
        raise_elisa_operation_input_error("velocity_grid must be uniformly spaced")
    return grid


@njit(cache=True)
def accumulate_gaussian_rbf(
    velocities: NDArray,
    fluxes: NDArray,
    velocity_grid: NDArray,
    sigmas: NDArray,
) -> NDArray:
    """Accumulate the Gaussian-broadened RBF on a velocity grid.

    Each visible surface element contributes a Gaussian centred at its
    line-of-sight velocity ``velocities[i]`` with local width ``sigmas[i]``.
    The contribution is integrated exactly over each velocity bin using the
    error function, so the result is the flux-weighted distribution of
    line-of-sight velocities smoothed by a local Gaussian kernel.

    :param velocities: Line-of-sight velocities of visible surface elements.
    :type velocities: numpy.typing.NDArray
    :param fluxes: Flux weights of the visible surface elements.
    :type fluxes: numpy.typing.NDArray
    :param velocity_grid: Uniform, strictly increasing velocity grid.
    :type velocity_grid: numpy.typing.NDArray
    :param sigmas: Local Gaussian widths for each surface element.
    :type sigmas: numpy.typing.NDArray
    :return: RBF values evaluated at the velocity grid.
    :rtype: numpy.typing.NDArray
    """
    n_out = len(velocity_grid)
    n_elements = len(velocities)
    rbf = np.zeros(n_out)
    if n_out < 2 or n_elements == 0:
        return rbf

    delta_v = velocity_grid[1] - velocity_grid[0]
    grid_start_edge = velocity_grid[0] - 0.5 * delta_v
    sqrt_two = 1.41421356

    for i in range(n_elements):
        v_center = velocities[i]
        flux = fluxes[i]
        sigma = sigmas[i]

        calc_sigma = max(0.1 * delta_v, sigma)
        v_min = v_center - 4.0 * calc_sigma
        v_max = v_center + 4.0 * calc_sigma

        idx_start = int(np.floor((v_min - grid_start_edge) / delta_v))
        idx_end = int(np.floor((v_max - grid_start_edge) / delta_v)) + 1

        idx_start = max(idx_start, 0)
        idx_end = min(idx_end, n_out)

        for j in range(idx_start, idx_end):
            edge_left = grid_start_edge + j * delta_v
            edge_right = edge_left + delta_v

            if sigma == 0.0:
                weight = 1.0 if edge_left <= v_center < edge_right else 0.0
                rbf[j] += flux * weight
                continue

            denom = 1.0 / (sigma * sqrt_two)
            z_right = (edge_right - v_center) * denom
            z_left = (edge_left - v_center) * denom
            weight = 0.5 * (math.erf(z_right) - math.erf(z_left))
            rbf[j] += flux * weight

    return rbf


def _los_velocity_gradient(star: StarContainer) -> NDArray[np.float64]:
    """Return the constant gradient of the line-of-sight velocity field.

    For a rigidly rotating star the surface velocity is an affine function of
    the position relative to the stellar centre,
    ``v(rho) = v_bulk + omega * (r_hat x rho)``.  Its Jacobian is therefore
    constant and antisymmetric.  The face centres and face velocities stored
    on ``star`` describe exactly such a field: a face velocity is the mean of
    the three vertex velocities and the velocity is linear in position, so the
    least-squares solve below is an exact linear solve, not a statistical fit.
    Taking the antisymmetric part of the Jacobian isolates the rigid rotation
    and discards any symmetric contribution (bulk motion is absorbed by the
    intercept).

    The returned gradient has units of ``(m/s) / length_unit`` where
    ``length_unit`` is the unit of ``star.points`` (metres for single systems,
    semi-major axis for binary systems).  Multiplying it by vertex positions
    expressed in the same unit yields m/s, so callers do not need to know the
    unit.

    :param star: Star container with populated ``face_centres``,
        ``velocities`` and ``com``.
    :type star: elisa.base.container.StarContainer
    :return: Gradient ``g`` of the line-of-sight velocity with respect to
        position.
    :rtype: numpy.typing.NDArray
    """
    origin = np.asarray(star.face_centres, dtype=np.float64) - np.asarray(star.com, dtype=np.float64)
    design = np.column_stack((origin, np.ones(origin.shape[0], dtype=np.float64)))
    velocities = np.asarray(star.velocities, dtype=np.float64)

    coeffs, *_ = np.linalg.lstsq(design, velocities, rcond=None)
    jacobian = coeffs[:3].T
    jacobian = 0.5 * (jacobian - jacobian.T)
    # the RBF velocity axis is +x while const.LINE_OF_SIGHT is -x; the sign is
    # irrelevant for a standard deviation
    return jacobian.T @ const.LINE_OF_SIGHT


def _smooth_face_values(values: NDArray, faces: NDArray, passes: int) -> NDArray[np.float64]:
    """Average per-face values with the values of topologically adjacent faces.

    Two faces are neighbours when they share at least one vertex.  Each pass
    replaces a face value by ``0.5 * value + 0.5 * neighbour_mean``.

    :param values: Per-face values, shape ``(M,)``.
    :type values: numpy.typing.NDArray
    :param faces: Triangulation index array, shape ``(M, 3)``.
    :type faces: numpy.typing.NDArray
    :param passes: Number of averaging passes; non-positive values disable the
        smoothing.
    :type passes: int
    :return: Smoothed per-face values.
    :rtype: numpy.typing.NDArray
    """
    smoothed = np.asarray(values, dtype=np.float64).copy()
    if passes <= 0 or len(faces) == 0:
        return smoothed

    n_vertices = int(faces.max()) + 1
    flat_faces = np.asarray(faces).ravel()
    vertices_per_face = faces.shape[1]
    for _ in range(passes):
        vertex_sum = np.bincount(
            flat_faces, weights=np.repeat(smoothed, vertices_per_face), minlength=n_vertices,
        )
        vertex_count = np.bincount(flat_faces, minlength=n_vertices).astype(np.float64)
        neighbour_mean = vertex_sum[faces].sum(axis=1) / vertex_count[faces].sum(axis=1)
        smoothed = 0.5 * smoothed + 0.5 * neighbour_mean
    return smoothed


def _flux_fraction_outside_grid(
    velocities: NDArray,
    fluxes: NDArray,
    sigmas: NDArray,
    velocity_grid: NDArray,
) -> float:
    """Return the fraction of flux that falls outside the velocity grid.

    The Gaussian tail of every surface element is integrated analytically, so
    the result accounts both for elements whose centre lies outside the grid and
    for elements whose wings are clipped by the grid edges.

    :param velocities: Line-of-sight velocity of each visible element.
    :type velocities: numpy.typing.NDArray
    :param fluxes: Flux weight of each visible element.
    :type fluxes: numpy.typing.NDArray
    :param sigmas: Gaussian width of each visible element.
    :type sigmas: numpy.typing.NDArray
    :param velocity_grid: Uniform velocity grid.
    :type velocity_grid: numpy.typing.NDArray
    :return: Fraction of the total flux outside the grid.
    :rtype: float
    """
    total_flux = float(np.sum(fluxes))
    if total_flux <= 0.0:
        return 0.0

    sigma = np.maximum(np.asarray(sigmas, dtype=np.float64), 1e-12)
    # only elements whose Gaussian extends beyond a grid edge can lose flux
    affected = (velocities - 4.0 * sigma < velocity_grid[0]) | (velocities + 4.0 * sigma > velocity_grid[-1])
    if not np.any(affected):
        return 0.0

    sqrt_two = np.sqrt(2.0)
    z_high = (velocity_grid[-1] - velocities[affected]) / sigma[affected]
    z_low = (velocity_grid[0] - velocities[affected]) / sigma[affected]
    inside = 0.5 * (erf(z_high / sqrt_two) - erf(z_low / sqrt_two))
    outside_flux = np.sum(fluxes[affected] * (1.0 - inside))
    return float(outside_flux / total_flux)


def _warn_on_partial_grid_coverage(
    velocities: NDArray,
    fluxes: NDArray,
    sigmas: NDArray,
    velocity_grid: NDArray,
    component: str | None,
) -> None:
    """Warn when the velocity grid truncates a non-negligible part of the flux."""
    lost = _flux_fraction_outside_grid(velocities, fluxes, sigmas, velocity_grid)
    if lost <= settings.RBF_GRID_TOLERANCE:
        return

    key = (component, float(velocity_grid[0]), float(velocity_grid[-1]))
    if key in _WARNED_GRIDS:
        return
    _WARNED_GRIDS.add(key)

    name = component if component is not None else "unknown"
    warnings.warn(
        f"RBF velocity grid [{velocity_grid[0]:.0f}, {velocity_grid[-1]:.0f}] m/s "
        f"does not cover component '{name}': line-of-sight velocities span "
        f"[{velocities.min():.0f}, {velocities.max():.0f}] m/s. About "
        f"{100.0 * lost:.2f} per cent of the visible flux lies outside the grid, "
        "so the integral of the RBF is smaller than the total flux. Widen "
        "v_start/v_stop to include the full velocity range.",
        RBFGridCoverageWarning,
        stacklevel=3,
    )


def _calculate_rbf_point(
    star: StarContainer,
    velocity_grid: NDArray,
    component: str | None = None,
) -> NDArray:
    """Calculate the RBF for a single component at one orbital position.

    Each visible surface element contributes a Gaussian centred at its
    line-of-sight velocity.  The Gaussian width is the RMS spread of the
    line-of-sight velocity over the vertices of that face, which for rigid
    rotation follows from the constant gradient of the velocity field:
    ``sigma_face = std(vertices . g) = omega * sin(i) * s_face``.  This is a
    resolution-scale width (it vanishes as the mesh is refined) that restores
    the variance lost by representing every face by a single velocity.  The
    vertex spread is multiplied by ``settings.RBF_MESH_SMOOTHING`` so that the
    kernel is wider than the mesh sampling step and the periodic density
    modulation of a regular mesh does not show up as ripple in the profile.
    The width field is additionally averaged over topologically adjacent faces
    (``settings.RBF_SIGMA_SMOOTHING_PASSES``), because the face-to-face scatter
    of the discrete geometry would otherwise modulate the flat top of the RBF.

    A :class:`RBFGridCoverageWarning` is issued when the supplied velocity grid
    truncates a non-negligible fraction of the visible flux.

    :param star: Star container with pre-calculated surface quantities.
    :type star: elisa.base.container.StarContainer
    :param velocity_grid: Uniform velocity grid in metres per second.
    :type velocity_grid: numpy.typing.NDArray
    :param component: Component label used in warnings; optional.
    :type component: str | None
    :return: RBF values evaluated at ``velocity_grid``.
    :rtype: numpy.typing.NDArray
    """
    indices = star.indices
    velocities = star.velocities[indices][:, 0]
    fluxes = crv_utils.calculate_surface_element_fluxes("rv_band", star)

    if len(velocities) == 0:
        return np.zeros_like(velocity_grid, dtype=np.float64)

    faces = np.asarray(star.faces)
    if faces.size == 0:
        local_sigmas = np.zeros_like(velocities, dtype=np.float64)
    else:
        gradient = _los_velocity_gradient(star)
        positions = np.asarray(star.points, dtype=np.float64) - np.asarray(star.com, dtype=np.float64)
        sigma_all = (positions @ gradient)[faces].std(axis=1)
        sigma_all = _smooth_face_values(sigma_all, faces, settings.RBF_SIGMA_SMOOTHING_PASSES)
        local_sigmas = settings.RBF_MESH_SMOOTHING * sigma_all[indices]

    _warn_on_partial_grid_coverage(velocities, fluxes, local_sigmas, velocity_grid, component)

    return accumulate_gaussian_rbf(velocities, fluxes, velocity_grid, local_sigmas)


def compute_rbf_at_pos(
    velocities: dict[str, NDArray],
    pos_idx: int,
    crv_labels: list[str],
    system: OrbitalPositionContainer,
    velocity_grid: NDArray,
) -> dict[str, NDArray]:
    """Compute RBF values for all requested components at one orbital position.

    :param velocities: Mapping from component label to the curves array.
    :type velocities: dict[str, numpy.typing.NDArray]
    :param pos_idx: Index of the current orbital position.
    :type pos_idx: int
    :param crv_labels: Component labels to compute.
    :type crv_labels: list[str]
    :param system: Orbital-position container.
    :type system: elisa.binary_system.container.OrbitalPositionContainer
    :param velocity_grid: Uniform velocity grid in metres per second.
    :type velocity_grid: numpy.typing.NDArray
    :return: The updated ``velocities`` mapping.
    :rtype: dict[str, numpy.typing.NDArray]
    """
    for component in crv_labels:
        star = getattr(system, component)
        velocities[component][pos_idx] = _calculate_rbf_point(star, velocity_grid, component=component)
    return velocities


class RBFPointComputer:
    """Callable that binds a velocity grid to the per-position RBF callback.

    The curve-routing layer only knows the standard
    ``curve_fn(curves, pos_idx, crv_labels, system)`` contract. Keeping the
    velocity grid on the callable allows the generic routing functions to stay
    independent of the RBF observable while still making the grid available to
    the actual computation.

    :param velocity_grid: Uniform velocity grid in metres per second.
    :type velocity_grid: numpy.typing.NDArray
    """

    curve_kind = "rbf"
    force_exact = True

    def __init__(self, velocity_grid: NDArray) -> None:
        self.velocity_grid = validate_velocity_grid(velocity_grid)

    @property
    def n_outputs(self) -> int:
        """Return the number of velocity bins produced by this callable."""
        return int(self.velocity_grid.size)

    def __call__(
        self,
        velocities: dict[str, NDArray],
        pos_idx: int,
        crv_labels: list[str],
        system: OrbitalPositionContainer,
    ) -> dict[str, NDArray]:
        """Compute RBF values at one orbital position."""
        return compute_rbf_at_pos(
            velocities,
            pos_idx,
            crv_labels,
            system,
            self.velocity_grid,
        )
