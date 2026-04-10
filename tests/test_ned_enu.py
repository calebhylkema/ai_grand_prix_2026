"""Tests for NED ↔ ENU frame conversions in transforms module."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from drone_racing.utils.transforms import (
    enu_to_ned,
    ned_to_enu,
    quat_enu_to_ned,
    quat_ned_to_enu,
)

# ---------------------------------------------------------------------------
# Position / velocity conversions
# ---------------------------------------------------------------------------


class TestNedEnuPosition:
    """NED ↔ ENU position and velocity conversions."""

    def test_ned_to_enu_basic(self) -> None:
        """1m North, 2m East, 3m Down → 2m East, 1m North, 3m Up."""
        v_ned = np.array([1.0, 2.0, -3.0])
        v_enu = ned_to_enu(v_ned)
        np.testing.assert_allclose(v_enu, [2.0, 1.0, 3.0])

    def test_enu_to_ned_basic(self) -> None:
        """2m East, 1m North, 3m Up → 1m North, 2m East, 3m Down."""
        v_enu = np.array([2.0, 1.0, 3.0])
        v_ned = enu_to_ned(v_enu)
        np.testing.assert_allclose(v_ned, [1.0, 2.0, -3.0])

    def test_zero_vector(self) -> None:
        """Origin maps to origin."""
        np.testing.assert_allclose(ned_to_enu(np.zeros(3)), np.zeros(3))
        np.testing.assert_allclose(enu_to_ned(np.zeros(3)), np.zeros(3))

    @given(
        arrays(
            dtype=np.float64,
            shape=(3,),
            elements=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False),
        )
    )
    def test_round_trip_ned(self, v_ned: np.ndarray) -> None:
        """NED → ENU → NED round-trip is identity."""
        np.testing.assert_allclose(enu_to_ned(ned_to_enu(v_ned)), v_ned, atol=1e-10)

    @given(
        arrays(
            dtype=np.float64,
            shape=(3,),
            elements=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False),
        )
    )
    def test_round_trip_enu(self, v_enu: np.ndarray) -> None:
        """ENU → NED → ENU round-trip is identity."""
        np.testing.assert_allclose(ned_to_enu(enu_to_ned(v_enu)), v_enu, atol=1e-10)


# ---------------------------------------------------------------------------
# Quaternion conversions
# ---------------------------------------------------------------------------


class TestNedEnuQuaternion:
    """NED ↔ ENU quaternion conversions."""

    def test_identity_quaternion_round_trip(self) -> None:
        """Identity quaternion round-trips through NED/ENU conversion."""
        q_ned = np.array([0.0, 0.0, 0.0, 1.0])
        q_enu = quat_ned_to_enu(q_ned)
        q_back = quat_enu_to_ned(q_enu)
        # The quaternion may differ by sign (q and -q are the same rotation)
        assert np.allclose(q_ned, q_back, atol=1e-10) or np.allclose(
            q_ned, -q_back, atol=1e-10
        )

    def test_quaternion_unit_norm(self) -> None:
        """Converted quaternions should remain unit norm."""
        q_ned = np.array([0.1, 0.2, 0.3, 0.9])
        q_ned = q_ned / np.linalg.norm(q_ned)
        q_enu = quat_ned_to_enu(q_ned)
        assert abs(np.linalg.norm(q_enu) - 1.0) < 1e-10

    @given(
        arrays(
            dtype=np.float64,
            shape=(4,),
            elements=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
        )
    )
    def test_round_trip_quaternion(self, q_raw: np.ndarray) -> None:
        """Quaternion NED → ENU → NED round-trip preserves rotation."""
        norm = np.linalg.norm(q_raw)
        if norm < 1e-6:
            return  # Skip degenerate quaternions

        q_ned = q_raw / norm
        q_enu = quat_ned_to_enu(q_ned)
        q_back = quat_enu_to_ned(q_enu)

        # q and -q represent the same rotation
        assert np.allclose(q_ned, q_back, atol=1e-8) or np.allclose(
            q_ned, -q_back, atol=1e-8
        )
