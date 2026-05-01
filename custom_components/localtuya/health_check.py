"""Health check system for LocalTuya devices.

Monitors device connectivity, attempts reconnection, falls back to Tuya Cloud API
when local connections fail, and periodically retries local recovery.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from homeassistant.core import HomeAssistant, CALLBACK_TYPE, callback
from homeassistant.helpers.event import async_track_time_interval

from .const import (
    CONF_NO_CLOUD,
    DeviceHealthState,
    HEALTH_CHECK_INTERVAL_DEFAULT,
    HEALTH_CHECK_MAX_RETRIES,
    HEALTH_CHECK_RECOVERY_INTERVAL,
)

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry
    from .coordinator import TuyaDevice, HassLocalTuyaData
    from .core.cloud_api import TuyaCloudApi

_LOGGER = logging.getLogger(__name__)


@dataclass
class DeviceHealthInfo:
    """Track health state for a single device."""

    device_id: str
    health_state: str = DeviceHealthState.HEALTHY
    reconnect_attempts: int = 0
    last_healthy_time: float = field(default_factory=time.monotonic)
    last_recovery_attempt: float = 0.0
    cloud_fallback_active: bool = False

    def reset(self):
        """Reset health info to healthy state."""
        self.health_state = DeviceHealthState.HEALTHY
        self.reconnect_attempts = 0
        self.last_healthy_time = time.monotonic()
        self.cloud_fallback_active = False

    def mark_cloud_fallback(self):
        """Mark device as using cloud fallback."""
        self.health_state = DeviceHealthState.CLOUD_FALLBACK
        self.cloud_fallback_active = True

    def mark_unhealthy(self):
        """Mark device as unhealthy after max retries exhausted."""
        self.health_state = DeviceHealthState.UNHEALTHY
        self.last_recovery_attempt = time.monotonic()


class HealthCheckManager:
    """Manages health monitoring for all LocalTuya devices in a config entry."""

    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        hass_entry_data: HassLocalTuyaData,
        check_interval: int = HEALTH_CHECK_INTERVAL_DEFAULT,
    ):
        self._hass = hass
        self._entry = entry
        self._hass_entry_data = hass_entry_data
        self._check_interval = check_interval
        self._device_health: dict[str, DeviceHealthInfo] = {}
        self._unsub_interval: CALLBACK_TYPE | None = None
        self._running = False
        self._lock = asyncio.Lock()

    @property
    def cloud_api(self) -> TuyaCloudApi:
        """Return the cloud API instance."""
        return self._hass_entry_data.cloud_data

    @property
    def no_cloud(self) -> bool:
        """Return whether cloud API is disabled."""
        return self._entry.data.get(CONF_NO_CLOUD, True)

    @property
    def devices(self) -> dict[str, TuyaDevice]:
        """Return all tracked devices."""
        return self._hass_entry_data.devices

    def get_health_info(self, device_id: str) -> DeviceHealthInfo | None:
        """Return health info for a device by its Tuya device ID."""
        return self._device_health.get(device_id)

    def get_health_state(self, device_id: str) -> str:
        """Return the health state string for a device."""
        if info := self._device_health.get(device_id):
            return info.health_state
        return DeviceHealthState.HEALTHY

    def is_cloud_fallback(self, device_id: str) -> bool:
        """Return whether a device is currently using cloud fallback."""
        if info := self._device_health.get(device_id):
            return info.cloud_fallback_active
        return False

    def get_all_health_states(self) -> dict[str, dict[str, Any]]:
        """Return a summary of all device health states."""
        return {
            dev_id: {
                "health_state": info.health_state,
                "reconnect_attempts": info.reconnect_attempts,
                "cloud_fallback_active": info.cloud_fallback_active,
            }
            for dev_id, info in self._device_health.items()
        }

    @callback
    def start(self):
        """Start the periodic health check."""
        if self._running:
            return

        self._running = True

        # Initialize health info for all devices.
        for device in self.devices.values():
            if device.id not in self._device_health:
                self._device_health[device.id] = DeviceHealthInfo(device_id=device.id)

        self._unsub_interval = async_track_time_interval(
            self._hass,
            self._async_health_check,
            timedelta(seconds=self._check_interval),
        )
        _LOGGER.debug("Health check started (interval=%ds)", self._check_interval)

    @callback
    def stop(self):
        """Stop the periodic health check."""
        self._running = False
        if self._unsub_interval:
            self._unsub_interval()
            self._unsub_interval = None
        _LOGGER.debug("Health check stopped")

    async def _async_health_check(self, _now=None):
        """Run a single health check cycle across all devices."""
        if not self._running:
            return

        async with self._lock:
            for device in list(self.devices.values()):
                # Skip sub-devices; they are managed via their gateway.
                if device.is_subdevice:
                    continue
                await self._check_device(device)

    async def _check_device(self, device: TuyaDevice):
        """Check and manage health for a single device."""
        dev_id = device.id
        if dev_id not in self._device_health:
            self._device_health[dev_id] = DeviceHealthInfo(device_id=dev_id)

        info = self._device_health[dev_id]

        # If the device is connected and healthy, reset tracking.
        if device.connected:
            if info.health_state != DeviceHealthState.HEALTHY:
                _LOGGER.info(
                    "Device %s (%s) recovered to healthy state",
                    device.friendly_name,
                    dev_id,
                )
            info.reset()
            return

        # Device is not connected: handle based on current state.
        if device.is_closing:
            return

        if info.health_state == DeviceHealthState.UNHEALTHY:
            await self._try_hourly_recovery(device, info)
            return

        # Increment reconnect attempts.
        info.reconnect_attempts += 1
        attempt = info.reconnect_attempts

        _LOGGER.warning(
            "Device %s (%s) not connected - reconnect attempt %d/%d",
            device.friendly_name,
            dev_id,
            attempt,
            HEALTH_CHECK_MAX_RETRIES,
        )

        # After the first failed attempt, immediately enable cloud fallback.
        if attempt >= 1 and not info.cloud_fallback_active:
            if not self.no_cloud:
                info.mark_cloud_fallback()
                _LOGGER.info(
                    "Device %s (%s): enabling cloud fallback after attempt %d",
                    device.friendly_name,
                    dev_id,
                    attempt,
                )
            else:
                _LOGGER.debug(
                    "Device %s (%s): cloud API not configured, cannot fallback",
                    device.friendly_name,
                    dev_id,
                )

        # Try to reconnect locally.
        await self._try_reconnect(device, info)

        # If reconnected, reset.
        if device.connected:
            _LOGGER.info(
                "Device %s (%s) reconnected on attempt %d",
                device.friendly_name,
                dev_id,
                attempt,
            )
            info.reset()
            return

        # After max retries, mark as unhealthy.
        if attempt >= HEALTH_CHECK_MAX_RETRIES:
            info.mark_unhealthy()
            _LOGGER.error(
                "Device %s (%s) marked unhealthy after %d failed attempts. "
                "Will retry local recovery every %d seconds",
                device.friendly_name,
                dev_id,
                HEALTH_CHECK_MAX_RETRIES,
                HEALTH_CHECK_RECOVERY_INTERVAL,
            )

    async def _try_reconnect(self, device: TuyaDevice, info: DeviceHealthInfo):
        """Attempt a single local reconnection."""
        try:
            if not device.is_connecting and not device.connected:
                await device.async_connect()
        except Exception:
            _LOGGER.debug(
                "Reconnect attempt %d failed for %s",
                info.reconnect_attempts,
                device.friendly_name,
                exc_info=True,
            )

    async def _try_hourly_recovery(self, device: TuyaDevice, info: DeviceHealthInfo):
        """For unhealthy devices, attempt local recovery once per hour."""
        elapsed = time.monotonic() - info.last_recovery_attempt
        if elapsed < HEALTH_CHECK_RECOVERY_INTERVAL:
            return

        info.last_recovery_attempt = time.monotonic()
        _LOGGER.info(
            "Attempting hourly local recovery for device %s (%s)",
            device.friendly_name,
            info.device_id,
        )

        try:
            if not device.is_connecting and not device.connected:
                await device.async_connect()
        except Exception:
            _LOGGER.debug(
                "Hourly recovery failed for %s",
                device.friendly_name,
                exc_info=True,
            )

        if device.connected:
            _LOGGER.info(
                "Hourly recovery succeeded for device %s (%s)",
                device.friendly_name,
                info.device_id,
            )
            info.reset()
        else:
            _LOGGER.warning(
                "Hourly recovery failed for device %s (%s), "
                "next retry in %d seconds",
                device.friendly_name,
                info.device_id,
                HEALTH_CHECK_RECOVERY_INTERVAL,
            )

    async def async_send_via_cloud(self, device_id: str, dps: dict[str, Any]) -> bool:
        """Send a DP command via the Tuya Cloud API as fallback.

        Args:
            device_id: The Tuya device ID.
            dps: Dict of DP index to value, e.g. {"1": True}.

        Returns:
            True if the cloud command was sent successfully.
        """
        if self.no_cloud:
            _LOGGER.debug("Cloud API not configured; cannot send cloud command")
            return False

        commands = [{"code": dp, "value": val} for dp, val in dps.items()]
        resp = await self.cloud_api.async_send_commands(device_id, commands)
        if resp and resp.get("success"):
            _LOGGER.debug(
                "Cloud command sent successfully for device %s: %s",
                device_id,
                dps,
            )
            return True

        _LOGGER.warning("Cloud command failed for device %s: %s", device_id, dps)
        return False

    async def async_get_cloud_status(self, device_id: str) -> dict[str, Any] | None:
        """Get device status from the Tuya Cloud API.

        Returns:
            A dict mapping DP codes to values, or None on failure.
        """
        if self.no_cloud:
            return None

        result = await self.cloud_api.async_get_device_status(device_id)
        if result is None:
            return None

        # Convert the cloud status list to a simple dict.
        status = {}
        for item in result:
            code = item.get("code", "")
            value = item.get("value")
            if dp_id := item.get("dp_id"):
                status[str(dp_id)] = value
            else:
                status[code] = value

        return status
