from __future__ import annotations

from typing import TYPE_CHECKING, Any

from elisa import settings
from elisa.base.curves import rbf as rbf_point
from elisa.binary_system.curves import c_router

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa.binary_system.system import BinarySystem
    from elisa.types import Float


def _prepare_curve_fn(kwargs: dict[str, Any]) -> rbf_point.RBFPointComputer:
    """Pop the velocity grid from kwargs and bind it to a curve callable."""
    velocity_grid = rbf_point.validate_velocity_grid(kwargs.pop("velocity_grid"))
    return rbf_point.RBFPointComputer(velocity_grid)


def compute_circular_synchronous_rbf_curve(
    binary: BinarySystem,
    **kwargs: Any,
) -> dict[str, NDArray[Float]]:
    """Compute the RBF curve for a circular synchronous binary system.

    :param binary: Binary system instance.
    :type binary: BinarySystem
    :param kwargs: Additional keyword arguments.

        Supported options include:

        - ``velocity_grid`` - uniform velocity grid in metres per second
        - ``phases`` - photometric phases
        - ``position_method`` - callable returning orbital positions
        - ``passband`` - passband data injected by the system layer
    :type kwargs: Any
    :return: RBF curves for each component.
    :rtype: dict[str, numpy.typing.NDArray[Float]]
    """
    initial_system = c_router.prep_initial_system(binary)
    crv_labels = list(settings.BINARY_COUNTERPARTS.keys())
    curve_fn = _prepare_curve_fn(kwargs)
    args = (
        binary,
        initial_system,
        kwargs.pop("phases"),
        curve_fn,
        crv_labels,
    )
    return c_router.produce_circular_sync_curves(*args, **kwargs)


def compute_circular_spotty_asynchronous_rbf_curve(
    binary: BinarySystem,
    **kwargs: Any,
) -> dict[str, NDArray[Float]]:
    """Compute the RBF curve for a circular asynchronous spotty binary system.

    :param binary: Binary system instance.
    :type binary: BinarySystem
    :param kwargs: Additional keyword arguments.
    :type kwargs: Any
    :return: RBF curves for each component.
    :rtype: dict[str, numpy.typing.NDArray[Float]]
    """
    crv_labels = list(settings.BINARY_COUNTERPARTS.keys())
    curve_fn = _prepare_curve_fn(kwargs)
    return c_router.produce_circular_spotty_async_curves(
        binary,
        curve_fn,
        crv_labels,
        **kwargs,
    )


def compute_circular_pulsating_rbf_curve(
    binary: BinarySystem,
    **kwargs: Any,
) -> dict[str, NDArray[Float]]:
    """Compute the RBF curve for a circular pulsating binary system.

    :param binary: Binary system instance.
    :type binary: BinarySystem
    :param kwargs: Additional keyword arguments.
    :type kwargs: Any
    :return: RBF curves for each component.
    :rtype: dict[str, numpy.typing.NDArray[Float]]
    """
    initial_system = c_router.prep_initial_system(
        binary,
        build_pulsations=False,
    )
    crv_labels = list(settings.BINARY_COUNTERPARTS.keys())
    curve_fn = _prepare_curve_fn(kwargs)
    args = (
        binary,
        initial_system,
        kwargs.pop("phases"),
        curve_fn,
        crv_labels,
    )
    return c_router.produce_circular_pulsating_curves(*args, **kwargs)


def compute_eccentric_rbf_curve_no_spots(
    binary: BinarySystem,
    **kwargs: Any,
) -> dict[str, NDArray[Float]]:
    """Compute the RBF curve for an eccentric binary system without spots.

    :param binary: Binary system instance.
    :type binary: BinarySystem
    :param kwargs: Additional keyword arguments.
    :type kwargs: Any
    :return: RBF curves for each component.
    :rtype: dict[str, numpy.typing.NDArray[Float]]
    """
    crv_labels = list(settings.BINARY_COUNTERPARTS.keys())
    curve_fn = _prepare_curve_fn(kwargs)
    return c_router.produce_ecc_curves_no_spots(
        binary,
        curve_fn,
        crv_labels,
        **kwargs,
    )


def compute_eccentric_spotty_rbf_curve(
    binary: BinarySystem,
    **kwargs: Any,
) -> dict[str, NDArray[Float]]:
    """Compute the RBF curve for an eccentric binary system with spots.

    :param binary: Binary system instance.
    :type binary: BinarySystem
    :param kwargs: Additional keyword arguments.
    :type kwargs: Any
    :return: RBF curves for each component.
    :rtype: dict[str, numpy.typing.NDArray[Float]]
    """
    crv_labels = list(settings.BINARY_COUNTERPARTS.keys())
    curve_fn = _prepare_curve_fn(kwargs)
    return c_router.produce_ecc_curves_with_spots(
        binary,
        curve_fn,
        crv_labels,
        **kwargs,
    )
