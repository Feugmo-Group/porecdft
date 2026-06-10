"""Tests for porecdft.warp_backend — import guards and numpy CPU path."""
import numpy as np
import pytest

from porecdft.warp_backend import WARP_AVAILABLE, WARP_VERSION, get_warp_device
from porecdft.warp_backend.interop import warp_kernel, warp_callable


def test_warp_available_is_bool():
    assert isinstance(WARP_AVAILABLE, bool)


def test_warp_version_type():
    if WARP_AVAILABLE:
        assert isinstance(WARP_VERSION, str)
    else:
        assert WARP_VERSION is None


def test_get_warp_device_no_crash():
    device = get_warp_device()
    if not WARP_AVAILABLE:
        assert device is None


def test_warp_kernel_raises_without_warp():
    if not WARP_AVAILABLE:
        with pytest.raises(RuntimeError, match="warp-lang"):
            warp_kernel(lambda: None)


def test_warp_callable_raises_without_warp():
    if not WARP_AVAILABLE:
        with pytest.raises(RuntimeError, match="warp-lang"):
            warp_callable(lambda: None)


@pytest.mark.skipif(WARP_AVAILABLE, reason="test the no-warp error path only")
def test_rho_bar_sphere_warp_raises_without_warp():
    from porecdft.warp_backend.kernels import rho_bar_sphere_warp
    with pytest.raises(RuntimeError, match="warp-lang"):
        rho_bar_sphere_warp(
            np.zeros((4, 3), dtype=np.float32),
            np.zeros(4, dtype=np.float32),
            np.zeros((2, 3), dtype=np.float32),
            np.ones(2, dtype=np.float32),
            np.ones(2, dtype=np.float32),
            1.0,
        )
