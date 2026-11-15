import math
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from elisa import const, settings
from elisa.base.curves import rbf
from elisa.base.curves.utils import calculate_surface_element_fluxes
from elisa.base.error import ElisaOperationInputError
from tests import set_astropy_units
from tests.utils import ElisaTestCase

# Test helpers build small lists incrementally; readability matters more here.
# ruff: noqa: PERF401

set_astropy_units()


def _rotation_axis(inclination: float) -> np.ndarray:
    """Return the rotation axis that makes angle ``inclination`` with the LOS."""
    inc = np.radians(inclination)
    return np.array([-np.cos(inc), 0.0, np.sin(inc)])


def _uv_sphere(n_lat: int, n_lon: int, radius: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Return a triangulated UV sphere (north pole, rings, south pole)."""
    theta = np.linspace(0.0, np.pi, n_lat + 1)
    phi = np.linspace(0.0, 2.0 * np.pi, n_lon, endpoint=False)

    points = [np.array([0.0, 0.0, radius])]
    for k in range(1, n_lat):
        points.append(
            np.column_stack(
                (
                    radius * np.sin(theta[k]) * np.cos(phi),
                    radius * np.sin(theta[k]) * np.sin(phi),
                    np.full(n_lon, radius * np.cos(theta[k])),
                ),
            ),
        )
    points.append(np.array([0.0, 0.0, -radius]))
    points = np.vstack(points)
    south = len(points) - 1

    faces = []
    for j in range(n_lon):
        faces.append([0, 1 + j, 1 + (j + 1) % n_lon])
    for k in range(n_lat - 2):
        base_0 = 1 + k * n_lon
        base_1 = 1 + (k + 1) * n_lon
        for j in range(n_lon):
            jn = (j + 1) % n_lon
            faces.append([base_0 + j, base_1 + j, base_1 + jn])
            faces.append([base_0 + j, base_1 + jn, base_0 + jn])
    base = 1 + (n_lat - 2) * n_lon
    for j in range(n_lon):
        faces.append([south, base + (j + 1) % n_lon, base + j])

    return points, np.asarray(faces, dtype=np.int64)


def _make_star(
    vsini: float,
    n: int = 2000,
    ld_coefficient: float = 0.6,
    systemic_velocity: float = 0.0,
    inclination: float = 90.0,
    n_lat: int | None = None,
    n_lon: int | None = None,
    radius: float = 1.0,
) -> SimpleNamespace:
    """Create a lightweight, rigidly rotating spherical star container.

    The velocity field is ``v = omega * (r_hat x rho)`` evaluated at the
    vertices and averaged over each face, i.e. exactly the convention used by
    ELISa's ``build_velocities`` for a single star.  The rotation axis makes
    angle ``inclination`` with the line of sight ``const.LINE_OF_SIGHT``.
    """
    if n_lat is None:
        n_lat = max(6, round(np.sqrt(n / 2.0)))
    if n_lon is None:
        n_lon = 2 * n_lat

    points, faces = _uv_sphere(n_lat, n_lon, radius=radius)
    face_centres = points[faces].mean(axis=1)

    rhat = _rotation_axis(inclination)
    omega = vsini / radius
    vertex_velocities = omega * np.cross(rhat[None, :], points)
    velocities = vertex_velocities[faces].mean(axis=1)
    velocities[:, 0] += systemic_velocity

    normals = face_centres / np.linalg.norm(face_centres, axis=1)[:, None]
    los_cosines = const.LINE_OF_SIGHT[0] * normals[:, 0]
    indices = np.where(los_cosines > 0.0)[0]

    n_faces = len(faces)
    return SimpleNamespace(
        points=points,
        faces=faces,
        face_centres=face_centres,
        com=np.zeros(3),
        velocities=velocities,
        normals=normals,
        los_cosines=los_cosines,
        indices=indices,
        normal_radiance={"rv_band": np.ones(n_faces)},
        ld_cfs={"rv_band": np.full((n_faces, 1), ld_coefficient)},
        coverage=np.ones(n_faces),
    )


def _face_sigmas(star: SimpleNamespace) -> np.ndarray:
    """Per-face RMS spread of the LOS velocity, using the fitted gradient."""
    gradient = rbf._los_velocity_gradient(star)
    positions = star.points - star.com
    return (positions @ gradient)[star.faces].std(axis=1)


def _legacy_accumulate(
    velocities: np.ndarray,
    fluxes: np.ndarray,
    velocity_grid: np.ndarray,
    sigmas: np.ndarray,
) -> np.ndarray:
    """Line-by-line copy of the original fork's accumulation kernel."""
    n_out = len(velocity_grid)
    n_elements = len(velocities)
    result = np.zeros(n_out)

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
            denom = 1.0 / (sigma * sqrt_two + 1e-20)
            z_right = (edge_right - v_center) * denom
            z_left = (edge_left - v_center) * denom
            weight = 0.5 * (math.erf(z_right) - math.erf(z_left))
            result[j] += flux * weight

    return result


def _fwhm(profile: np.ndarray, grid: np.ndarray) -> np.floating:
    half_max = profile.max() / 2.0
    indices = np.where(profile >= half_max)[0]
    return grid[indices[-1]] - grid[indices[0]]


class RBFKernelTestCase(ElisaTestCase):
    def setUp(self) -> None:
        super().setUp()
        settings.configure(LIMB_DARKENING_LAW="cosine")
        rbf._WARNED_GRIDS.clear()

    def test_kernel_normalization_single_element(self) -> None:
        grid = np.arange(-40000.0, 40001.0, 1000.0)
        profile = rbf.accumulate_gaussian_rbf(
            np.array([1234.0]),
            np.array([2.5]),
            grid,
            np.array([3000.0]),
        )
        assert_allclose(profile.sum(), 2.5, rtol=1e-4)

    def test_kernel_normalization_many_elements(self) -> None:
        rng = np.random.default_rng(42)
        grid = np.arange(-200000.0, 200001.0, 1000.0)
        velocities = rng.uniform(-50000.0, 50000.0, 200)
        fluxes = rng.uniform(0.5, 2.0, 200)
        sigmas = rng.uniform(500.0, 5000.0, 200)
        profile = rbf.accumulate_gaussian_rbf(velocities, fluxes, grid, sigmas)
        assert_allclose(profile.sum(), fluxes.sum(), rtol=1e-4)

    def test_rotation_speed_broadens_profile(self) -> None:
        grid = np.arange(-200000.0, 200001.0, 1000.0)
        widths = []
        sums = []
        for vsini in (10e3, 20e3, 40e3):
            star = _make_star(vsini)
            profile = rbf._calculate_rbf_point(star, grid)
            widths.append(_fwhm(profile, grid))
            sums.append(profile.sum())

        assert widths[0] < widths[1] < widths[2]
        assert 1.6 < widths[1] / widths[0] < 2.4
        assert 1.6 < widths[2] / widths[1] < 2.4
        assert_allclose(sums, sums[0], rtol=1e-4)

    def test_limb_darkening_changes_profile(self) -> None:
        grid = np.arange(-200000.0, 200001.0, 1000.0)
        star_no_ld = _make_star(30e3, ld_coefficient=0.0)
        star_strong_ld = _make_star(30e3, ld_coefficient=0.9)

        profile_no_ld = rbf._calculate_rbf_point(star_no_ld, grid)
        profile_strong_ld = rbf._calculate_rbf_point(star_strong_ld, grid)

        assert np.abs(profile_no_ld - profile_strong_ld).max() > 0.01 * profile_no_ld.max()
        assert_allclose(
            profile_no_ld.sum(),
            calculate_surface_element_fluxes("rv_band", star_no_ld).sum(),
            rtol=1e-4,
        )
        assert_allclose(
            profile_strong_ld.sum(),
            calculate_surface_element_fluxes("rv_band", star_strong_ld).sum(),
            rtol=1e-4,
        )

    def test_matches_legacy_accumulator(self) -> None:
        rng = np.random.default_rng(7)
        grid = np.arange(-200000.0, 200001.0, 1000.0)
        velocities = rng.uniform(-50000.0, 50000.0, 150)
        fluxes = rng.uniform(0.1, 2.0, 150)
        sigmas = rng.uniform(200.0, 5000.0, 150)

        obtained = rbf.accumulate_gaussian_rbf(velocities, fluxes, grid, sigmas)
        expected = _legacy_accumulate(velocities, fluxes, grid, sigmas)
        assert_allclose(obtained, expected, rtol=1e-12, atol=1e-15)

    def test_zero_rotation_is_delta(self) -> None:
        grid = np.arange(-200000.0, 200001.0, 1000.0)
        star = _make_star(0.0)
        profile = rbf._calculate_rbf_point(star, grid)

        assert np.isfinite(profile).all()
        assert np.count_nonzero(profile) == 1
        assert_allclose(
            profile.sum(),
            calculate_surface_element_fluxes("rv_band", star).sum(),
            rtol=1e-4,
        )

    def test_zero_rotation_with_systemic_velocity(self) -> None:
        grid = np.arange(-200000.0, 200001.0, 1000.0)
        star = _make_star(0.0, systemic_velocity=50000.0)
        profile = rbf._calculate_rbf_point(star, grid)

        assert np.isfinite(profile).all()
        assert np.count_nonzero(profile) == 1
        assert_allclose(profile.sum(), calculate_surface_element_fluxes("rv_band", star).sum(), rtol=1e-4)

    def test_velocity_grid_validation(self) -> None:
        with pytest.raises(ElisaOperationInputError):
            rbf.validate_velocity_grid(np.array([0.0]))
        with pytest.raises(ElisaOperationInputError):
            rbf.validate_velocity_grid(np.array([0.0, 2.0, 1.0]))
        with pytest.raises(ElisaOperationInputError):
            rbf.validate_velocity_grid(np.array([0.0, 1.0, 3.0]))
        with pytest.raises(ElisaOperationInputError):
            rbf.validate_velocity_grid(np.zeros((2, 2)))

    def test_rbf_point_computer_binds_grid(self) -> None:
        grid = np.arange(-100000.0, 100001.0, 1000.0)
        computer = rbf.RBFPointComputer(grid)
        star = _make_star(30e3)
        system = SimpleNamespace(primary=star, secondary=star)
        curves = {"primary": np.zeros((1, grid.size))}

        result = computer(curves, 0, ["primary"], system)

        assert computer.n_outputs == grid.size
        assert result["primary"].shape == (1, grid.size)
        assert np.isfinite(result["primary"]).all()


class RBFLocalWidthTestCase(ElisaTestCase):
    """Tests for the rigid-rotation local width model."""

    def setUp(self) -> None:
        super().setUp()
        settings.configure(LIMB_DARKENING_LAW="cosine")
        rbf._WARNED_GRIDS.clear()

    def test_gradient_matches_analytic_rotation(self) -> None:
        vsini = 30e3
        for inclination in (90.0, 60.0, 30.0):
            star = _make_star(vsini, inclination=inclination)
            gradient = rbf._los_velocity_gradient(star)
            expected = vsini * np.cross(const.LINE_OF_SIGHT, _rotation_axis(inclination))
            assert_allclose(gradient, expected, rtol=1e-8, atol=1e-6, err_msg=f"i={inclination}")

    def test_sigma_matches_analytic_vertex_spread(self) -> None:
        star = _make_star(30e3, inclination=90.0)
        gradient = rbf._los_velocity_gradient(star)
        sigma = ((star.points - star.com) @ gradient)[star.faces].std(axis=1)

        g_true = 30e3 * np.cross(const.LINE_OF_SIGHT, _rotation_axis(90.0))
        sigma_true = (star.points @ g_true)[star.faces].std(axis=1)

        assert_allclose(sigma, sigma_true, rtol=1e-6, atol=1e-6)
        assert sigma.max() > 0.0

    def test_pole_on_rotation_has_zero_width(self) -> None:
        star = _make_star(30e3, inclination=0.0)
        gradient = rbf._los_velocity_gradient(star)
        sigma = _face_sigmas(star)

        assert_allclose(gradient, 0.0, atol=1e-6)
        assert_allclose(sigma, 0.0, atol=1e-6)

    def test_sigma_scales_with_sine_inclination(self) -> None:
        medians = []
        for inclination in (90.0, 60.0, 30.0):
            star = _make_star(30e3, inclination=inclination)
            medians.append(np.median(_face_sigmas(star)))

        expected = np.sin(np.radians([90.0, 60.0, 30.0]))
        assert_allclose(np.asarray(medians) / medians[0], expected, rtol=2e-2)

    def test_sigma_scales_with_face_size(self) -> None:
        coarse = _make_star(30e3, n_lat=12, n_lon=24)
        fine = _make_star(30e3, n_lat=24, n_lon=48)

        ratio = np.median(_face_sigmas(coarse)) / np.median(_face_sigmas(fine))
        assert 1.7 < ratio < 2.3

    def test_systemic_velocity_does_not_change_gradient(self) -> None:
        star_rest = _make_star(30e3, systemic_velocity=0.0)
        star_moving = _make_star(30e3, systemic_velocity=50000.0)

        gradient_rest = rbf._los_velocity_gradient(star_rest)
        gradient_moving = rbf._los_velocity_gradient(star_moving)
        assert_allclose(gradient_rest, gradient_moving, rtol=1e-8, atol=1e-6)

    def test_profile_uses_mesh_smoothing_factor(self) -> None:
        grid = np.arange(-200000.0, 200001.0, 1000.0)
        star = _make_star(30e3)

        profile = rbf._calculate_rbf_point(star, grid)

        velocities = star.velocities[star.indices][:, 0]
        fluxes = calculate_surface_element_fluxes("rv_band", star)
        sigma_all = rbf._smooth_face_values(
            _face_sigmas(star), star.faces, settings.RBF_SIGMA_SMOOTHING_PASSES,
        )
        sigmas = settings.RBF_MESH_SMOOTHING * sigma_all[star.indices]
        expected = rbf.accumulate_gaussian_rbf(velocities, fluxes, grid, sigmas)

        assert_allclose(profile, expected, rtol=1e-12, atol=1e-12)

    def test_sigma_smoothing_reduces_face_scatter(self) -> None:
        star = _make_star(30e3)
        faces = np.asarray(star.faces)
        raw = _face_sigmas(star)
        smoothed = rbf._smooth_face_values(raw, faces, settings.RBF_SIGMA_SMOOTHING_PASSES)

        def neighbour_roughness(values: np.ndarray) -> np.floating:
            n_vertices = int(faces.max()) + 1
            flat = faces.ravel()
            vertex_sum = np.bincount(flat, weights=np.repeat(values, 3), minlength=n_vertices)
            vertex_count = np.bincount(flat, minlength=n_vertices).astype(np.float64)
            neighbour_mean = vertex_sum[faces].sum(axis=1) / vertex_count[faces].sum(axis=1)
            return np.mean(np.abs(values - neighbour_mean)) / np.mean(values)

        assert neighbour_roughness(smoothed) < 0.7 * neighbour_roughness(raw)
        assert_allclose(smoothed.mean(), raw.mean(), rtol=1e-3)
        assert np.all(smoothed > 0.0)

    def test_narrow_grid_emits_warning(self) -> None:
        rbf._WARNED_GRIDS.clear()
        grid = np.arange(-10000.0, 10001.0, 1000.0)
        star = _make_star(30e3)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rbf._calculate_rbf_point(star, grid, component="primary")
        matching = [item for item in caught if issubclass(item.category, rbf.RBFGridCoverageWarning)]
        assert matching, "narrow velocity grid should emit RBFGridCoverageWarning"

    def test_wide_grid_does_not_warn(self) -> None:
        rbf._WARNED_GRIDS.clear()
        grid = np.arange(-200000.0, 200001.0, 1000.0)
        star = _make_star(30e3)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rbf._calculate_rbf_point(star, grid, component="primary")
        matching = [item for item in caught if issubclass(item.category, rbf.RBFGridCoverageWarning)]
        assert not matching

    def test_profile_is_shifted_by_systemic_velocity(self) -> None:
        grid = np.arange(-200000.0, 200001.0, 1000.0)
        profile_rest = rbf._calculate_rbf_point(_make_star(30e3, systemic_velocity=0.0), grid)
        profile_moving = rbf._calculate_rbf_point(_make_star(30e3, systemic_velocity=50000.0), grid)

        shift = round(50000.0 / (grid[1] - grid[0]))
        assert_allclose(profile_moving[shift:], profile_rest[:-shift], rtol=1e-6, atol=1e-9)
