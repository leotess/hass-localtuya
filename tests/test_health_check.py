"""Tests for the LocalTuya health check system."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from custom_components.localtuya.const import (
    DeviceHealthState,
    HEALTH_CHECK_MAX_RETRIES,
    HEALTH_CHECK_RECOVERY_INTERVAL,
)
from custom_components.localtuya.health_check import (
    DeviceHealthInfo,
    HealthCheckManager,
)


def _make_device(dev_id="test_dev_001", connected=True, is_subdevice=False):
    """Create a mock TuyaDevice."""
    device = MagicMock()
    device.id = dev_id
    device.friendly_name = f"Test Device {dev_id}"
    device.connected = connected
    device.is_subdevice = is_subdevice
    device.is_closing = False
    device.is_connecting = False
    device.async_connect = AsyncMock()
    device._health_manager = None
    return device


def _make_manager(devices=None, no_cloud=False):
    """Create a HealthCheckManager with mocked dependencies."""
    hass = MagicMock()
    entry = MagicMock()
    entry.data = {"no_cloud": no_cloud}

    cloud_api = AsyncMock()
    cloud_api.async_send_commands = AsyncMock(return_value={"success": True})
    cloud_api.async_get_device_status = AsyncMock(return_value=None)

    hass_entry_data = MagicMock()
    hass_entry_data.cloud_data = cloud_api
    hass_entry_data.devices = devices or {}

    manager = HealthCheckManager(hass, entry, hass_entry_data, check_interval=30)
    return manager


# --- DeviceHealthInfo tests ---


class TestDeviceHealthInfo:
    def test_initial_state(self):
        info = DeviceHealthInfo(device_id="dev1")
        assert info.health_state == DeviceHealthState.HEALTHY
        assert info.reconnect_attempts == 0
        assert info.cloud_fallback_active is False

    def test_reset(self):
        info = DeviceHealthInfo(device_id="dev1")
        info.reconnect_attempts = 3
        info.health_state = DeviceHealthState.CLOUD_FALLBACK
        info.cloud_fallback_active = True
        info.reset()
        assert info.health_state == DeviceHealthState.HEALTHY
        assert info.reconnect_attempts == 0
        assert info.cloud_fallback_active is False

    def test_mark_cloud_fallback(self):
        info = DeviceHealthInfo(device_id="dev1")
        info.mark_cloud_fallback()
        assert info.health_state == DeviceHealthState.CLOUD_FALLBACK
        assert info.cloud_fallback_active is True

    def test_mark_unhealthy(self):
        info = DeviceHealthInfo(device_id="dev1")
        info.mark_unhealthy()
        assert info.health_state == DeviceHealthState.UNHEALTHY
        assert info.last_recovery_attempt > 0


# --- HealthCheckManager tests ---


class TestHealthCheckManager:
    def test_get_health_state_default(self):
        manager = _make_manager()
        assert manager.get_health_state("nonexistent") == DeviceHealthState.HEALTHY

    def test_is_cloud_fallback_default(self):
        manager = _make_manager()
        assert manager.is_cloud_fallback("nonexistent") is False

    def test_start_initializes_health_info(self):
        device = _make_device("dev1")
        manager = _make_manager(devices={"192.168.1.1": device})

        with patch(
            "custom_components.localtuya.health_check.async_track_time_interval"
        ) as mock_track:
            manager.start()

        assert "dev1" in manager._device_health
        assert manager._device_health["dev1"].health_state == DeviceHealthState.HEALTHY
        assert manager._running is True

    def test_stop(self):
        device = _make_device("dev1")
        manager = _make_manager(devices={"192.168.1.1": device})

        unsub = MagicMock()
        with patch(
            "custom_components.localtuya.health_check.async_track_time_interval",
            return_value=unsub,
        ):
            manager.start()
            manager.stop()

        unsub.assert_called_once()
        assert manager._running is False

    @pytest.mark.asyncio
    async def test_healthy_device_resets_info(self):
        device = _make_device("dev1", connected=True)
        manager = _make_manager(devices={"192.168.1.1": device})
        manager._device_health["dev1"] = DeviceHealthInfo(device_id="dev1")
        manager._device_health["dev1"].reconnect_attempts = 3
        manager._device_health["dev1"].health_state = DeviceHealthState.CLOUD_FALLBACK

        await manager._check_device(device)

        info = manager._device_health["dev1"]
        assert info.health_state == DeviceHealthState.HEALTHY
        assert info.reconnect_attempts == 0

    @pytest.mark.asyncio
    async def test_disconnected_device_increments_attempts(self):
        device = _make_device("dev1", connected=False)
        manager = _make_manager(devices={"192.168.1.1": device}, no_cloud=True)
        manager._device_health["dev1"] = DeviceHealthInfo(device_id="dev1")

        await manager._check_device(device)

        info = manager._device_health["dev1"]
        assert info.reconnect_attempts == 1
        device.async_connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_cloud_fallback_after_first_failed_attempt(self):
        device = _make_device("dev1", connected=False)
        manager = _make_manager(devices={"192.168.1.1": device}, no_cloud=False)
        manager._device_health["dev1"] = DeviceHealthInfo(device_id="dev1")

        await manager._check_device(device)

        info = manager._device_health["dev1"]
        assert info.cloud_fallback_active is True
        assert info.health_state == DeviceHealthState.CLOUD_FALLBACK

    @pytest.mark.asyncio
    async def test_no_cloud_fallback_when_cloud_disabled(self):
        device = _make_device("dev1", connected=False)
        manager = _make_manager(devices={"192.168.1.1": device}, no_cloud=True)
        manager._device_health["dev1"] = DeviceHealthInfo(device_id="dev1")

        await manager._check_device(device)

        info = manager._device_health["dev1"]
        assert info.cloud_fallback_active is False

    @pytest.mark.asyncio
    async def test_unhealthy_after_max_retries(self):
        device = _make_device("dev1", connected=False)
        manager = _make_manager(devices={"192.168.1.1": device}, no_cloud=True)
        manager._device_health["dev1"] = DeviceHealthInfo(device_id="dev1")

        for _ in range(HEALTH_CHECK_MAX_RETRIES):
            await manager._check_device(device)

        info = manager._device_health["dev1"]
        assert info.health_state == DeviceHealthState.UNHEALTHY
        assert info.reconnect_attempts == HEALTH_CHECK_MAX_RETRIES

    @pytest.mark.asyncio
    async def test_hourly_recovery_not_triggered_before_interval(self):
        device = _make_device("dev1", connected=False)
        manager = _make_manager(devices={"192.168.1.1": device})
        info = DeviceHealthInfo(device_id="dev1")
        info.mark_unhealthy()
        # Simulate that the last recovery was just now.
        info.last_recovery_attempt = time.monotonic()
        manager._device_health["dev1"] = info

        device.async_connect.reset_mock()
        await manager._check_device(device)

        # Should not attempt connect because interval hasn't elapsed.
        device.async_connect.assert_not_called()

    @pytest.mark.asyncio
    async def test_hourly_recovery_triggered_after_interval(self):
        device = _make_device("dev1", connected=False)
        manager = _make_manager(devices={"192.168.1.1": device})
        info = DeviceHealthInfo(device_id="dev1")
        info.mark_unhealthy()
        # Simulate that the last recovery was long ago.
        info.last_recovery_attempt = (
            time.monotonic() - HEALTH_CHECK_RECOVERY_INTERVAL - 1
        )
        manager._device_health["dev1"] = info

        await manager._check_device(device)

        device.async_connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_hourly_recovery_success_resets_state(self):
        device = _make_device("dev1", connected=False)
        manager = _make_manager(devices={"192.168.1.1": device})
        info = DeviceHealthInfo(device_id="dev1")
        info.mark_unhealthy()
        info.last_recovery_attempt = (
            time.monotonic() - HEALTH_CHECK_RECOVERY_INTERVAL - 1
        )
        manager._device_health["dev1"] = info

        # After connect, device becomes connected.
        async def side_effect():
            device.connected = True

        device.async_connect = AsyncMock(side_effect=side_effect)

        await manager._check_device(device)

        assert info.health_state == DeviceHealthState.HEALTHY
        assert info.reconnect_attempts == 0

    @pytest.mark.asyncio
    async def test_reconnect_success_mid_attempts_resets(self):
        device = _make_device("dev1", connected=False)
        manager = _make_manager(devices={"192.168.1.1": device}, no_cloud=False)
        manager._device_health["dev1"] = DeviceHealthInfo(device_id="dev1")

        # First two attempts fail.
        await manager._check_device(device)
        await manager._check_device(device)
        assert manager._device_health["dev1"].reconnect_attempts == 2

        # Third attempt succeeds.
        async def side_effect():
            device.connected = True

        device.async_connect = AsyncMock(side_effect=side_effect)
        await manager._check_device(device)

        info = manager._device_health["dev1"]
        assert info.health_state == DeviceHealthState.HEALTHY
        assert info.reconnect_attempts == 0
        assert info.cloud_fallback_active is False

    @pytest.mark.asyncio
    async def test_subdevices_skipped_in_health_check(self):
        device = _make_device("dev1", connected=False, is_subdevice=True)
        manager = _make_manager(devices={"192.168.1.1_node": device})
        manager._running = True

        await manager._async_health_check()

        device.async_connect.assert_not_called()

    @pytest.mark.asyncio
    async def test_async_send_via_cloud_success(self):
        manager = _make_manager(no_cloud=False)
        info = DeviceHealthInfo(device_id="dev1")
        info.mark_cloud_fallback()
        manager._device_health["dev1"] = info

        result = await manager.async_send_via_cloud("dev1", {"1": True})
        assert result is True
        manager.cloud_api.async_send_commands.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_send_via_cloud_no_cloud(self):
        manager = _make_manager(no_cloud=True)
        result = await manager.async_send_via_cloud("dev1", {"1": True})
        assert result is False

    @pytest.mark.asyncio
    async def test_async_send_via_cloud_failure(self):
        manager = _make_manager(no_cloud=False)
        manager.cloud_api.async_send_commands = AsyncMock(
            return_value={"success": False}
        )
        result = await manager.async_send_via_cloud("dev1", {"1": True})
        assert result is False

    def test_get_all_health_states(self):
        manager = _make_manager()
        manager._device_health["dev1"] = DeviceHealthInfo(device_id="dev1")
        manager._device_health["dev2"] = DeviceHealthInfo(device_id="dev2")
        manager._device_health["dev2"].mark_cloud_fallback()

        states = manager.get_all_health_states()
        assert states["dev1"]["health_state"] == DeviceHealthState.HEALTHY
        assert states["dev2"]["health_state"] == DeviceHealthState.CLOUD_FALLBACK
        assert states["dev2"]["cloud_fallback_active"] is True

    @pytest.mark.asyncio
    async def test_closing_device_skipped(self):
        device = _make_device("dev1", connected=False)
        device.is_closing = True
        manager = _make_manager(devices={"192.168.1.1": device})
        manager._device_health["dev1"] = DeviceHealthInfo(device_id="dev1")

        await manager._check_device(device)

        device.async_connect.assert_not_called()
        assert manager._device_health["dev1"].reconnect_attempts == 0
