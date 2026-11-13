"""Weather forecast tool."""

import logging
from collections.abc import Callable
from datetime import date, datetime, timedelta
from typing import Any

import voluptuous as vol
from homeassistant.components.weather import WeatherEntityFeature
from homeassistant.const import UnitOfTemperature
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.llm import ToolResult

from .base_tool import BaseTool
from .const import (
    CONF_DAILY_WEATHER_ENTITY,
    CONF_HOURLY_WEATHER_ENTITY,
    CONF_WEATHER_TEMPERATURE_SENSOR,
)

_LOGGER = logging.getLogger(__name__)

# Precipitation chance thresholds for friendly categorization
PRECIPITATION_THRESHOLDS: dict[int, str] = {
    0: "none",
    5: "very unlikely",
    15: "unlikely",
    30: "possible",
    50: "moderate",
    70: "likely",
    85: "very likely",
    95: "extremely likely",
    100: "almost guaranteed",
}


class WeatherToolError(Exception):
    """Base exception for weather tool errors."""

    def __init__(self, message: str = "Weather tool encountered an error") -> None:
        """Init exception with default message."""
        super().__init__(message)


class WeatherEntityNotFoundError(WeatherToolError):
    """Raised when a weather entity is not found."""

    def __init__(self, message: str = "Weather entity not found") -> None:
        """Init exception with default message."""
        super().__init__(message)


class ForecastRetrievalError(WeatherToolError):
    """Raised when forecast data cannot be retrieved."""

    def __init__(self, message: str = "Failed to retrieve forecast") -> None:
        """Init exception with default message."""
        super().__init__(message)


def _friendly_precipitation_chance(precipitation_chance: int) -> str:
    """Format the precipitation chance into string categories for the LLM."""
    for threshold, value in PRECIPITATION_THRESHOLDS.items():
        if precipitation_chance <= threshold:
            return value
    return PRECIPITATION_THRESHOLDS[100]


def _format_percent(value: Any) -> str:
    """Format a percentage value for the LLM."""
    return f"{round(float(value))}%"


class WeatherAttribute:
    """Represent a weather attribute."""

    def __init__(
        self,
        key: str,
        name: str,
        formatter: Callable[[Any], str] | None = None,
    ) -> None:
        """Init our WeatherAttribute."""
        super().__init__()
        self.formatter = formatter
        self.key: str = key
        self.name: str = name


def _build_attributes(
    attribute_list: list[WeatherAttribute],
    weather_data: dict,
) -> list[str]:
    """Build our attributes in a friendly manner for the LLM."""
    output = []
    for attribute in attribute_list:
        if attribute.key in weather_data:
            attr_data = weather_data.get(attribute.key)
            if attr_data is None:
                continue
            output.append(
                f"  {attribute.name}: {attribute.formatter(attr_data) if attribute.formatter else attr_data}",
            )
    return output


# WMO weather codes (Open-Meteo) mapped to Home Assistant condition strings
WMO_CONDITIONS: dict[int, str] = {
    0: "sunny",
    1: "sunny",
    2: "partlycloudy",
    3: "cloudy",
    45: "fog",
    48: "fog",
    51: "rainy",
    53: "rainy",
    55: "rainy",
    56: "rainy",
    57: "rainy",
    61: "rainy",
    63: "rainy",
    65: "pouring",
    66: "rainy",
    67: "pouring",
    71: "snowy",
    73: "snowy",
    75: "snowy",
    77: "snowy",
    80: "rainy",
    81: "rainy",
    82: "pouring",
    85: "snowy",
    86: "snowy",
    95: "lightning",
    96: "lightning-rainy",
    99: "lightning-rainy",
}


class WeatherForecastTool(BaseTool):
    """Tool for weather forecast data."""

    name = "get_weather_forecast"
    description = (
        "Use this tool to retrieve the current weather conditions and weather forecasts for this home, from now up until a week in advance.\n"
        "Current conditions include temperature, feels-like temperature, humidity, wind, and pressure: use this tool for questions about current outdoor conditions instead of looking for a sensor.\n"
        "If the user requests data for `tonight`, use the `today` argument.\n"
        "For weather in another place, pass the `location` argument with the name of the place (for example 'Orlando'): current conditions and the forecast for that place are retrieved from Open-Meteo.\n"
        "Cannot be used to retrieve past or historic weather data: local sensor history can be queried via the device history tool."
    )
    prompt_description = None

    parameters = vol.Schema(
        {
            vol.Required(
                "range",
                description="Specify whether to receive data for the week ahead, or for a particular upcoming day",
            ): vol.In(
                [
                    "week",
                    "today",
                    "tomorrow",
                    "monday",
                    "tuesday",
                    "wednesday",
                    "thursday",
                    "friday",
                    "saturday",
                    "sunday",
                ],
            ),
            vol.Optional(
                "location",
                description="Name of another place to get the weather for, e.g. a city like 'Orlando'. Omit to get the weather for this home.",
            ): str,
        },
    )

    @staticmethod
    def _find_target_date(date_range: str) -> date | None:
        """Find our target date based on the input."""
        now = datetime.now().astimezone()

        # Determine target date
        if date_range.lower() == "today":
            target_date = now.date()
        elif date_range.lower() == "tomorrow":
            target_date = (now + timedelta(days=1)).date()
        else:
            # Map weekday names to numbers
            weekdays = {
                "monday": 0,
                "tuesday": 1,
                "wednesday": 2,
                "thursday": 3,
                "friday": 4,
                "saturday": 5,
                "sunday": 6,
            }
            target_weekday = weekdays.get(date_range.lower())
            if target_weekday is None:
                return None

            # Find next matching weekday (not necessarily next calendar week)
            days_ahead = (target_weekday - now.weekday() + 7) % 7
            if days_ahead == 0:
                target_date = now.date()
            else:
                target_date = (now + timedelta(days=days_ahead)).date()
        return target_date

    @staticmethod
    def _filter_forecast_by_day(forecast: list[dict], target_date: date) -> list[dict]:
        """Filter forecast entries for the target date."""
        result = []
        for entry in forecast:
            dt = datetime.fromisoformat(entry["datetime"]).astimezone()
            if dt.date() == target_date:
                result.append(entry)

        return result

    @staticmethod
    def _format_time(iso_str: str) -> str:
        """Format our time nicely for the LLM."""
        dt = datetime.fromisoformat(iso_str).astimezone()
        next_hour = dt + timedelta(hours=1)
        return next_hour.strftime("%-I%p").lower()

    @staticmethod
    def _format_date(iso_str: str) -> str:
        """Format our date nicely for the LLM."""
        dt = datetime.fromisoformat(iso_str).astimezone()
        now = datetime.now().astimezone()
        date = dt.strftime("%A")

        if now.date() == dt.date():
            return f"Today ({date})"

        return date

    def has_twice_daily_data(self, entity_id: str) -> bool:
        """Check if our daily entity data provides twice-daily data."""
        entity = self.hass.states.get(entity_id)
        if entity is None:
            message = f"Weather entity {entity_id} not found."
            raise WeatherEntityNotFoundError(message)
        features = entity.attributes.get("supported_features", 0)
        return bool(features & WeatherEntityFeature.FORECAST_TWICE_DAILY)

    async def _get_daily_forecast(
        self,
        hass: HomeAssistant,
        entity_id: str,
        target_date: date | None,
    ) -> str:
        """Build the daily forecast data."""
        forecast = await hass.services.async_call(
            "weather",
            "get_forecasts",
            {"entity_id": entity_id, "type": "daily"},
            blocking=True,
            return_response=True,
        )
        forecast = forecast.get(entity_id, {}).get("forecast")
        if not forecast:
            raise ForecastRetrievalError

        if target_date:
            forecast = self._filter_forecast_by_day(forecast, target_date)

        daily_attributes = [
            WeatherAttribute(key="condition", name="General Condition", formatter=None),
            WeatherAttribute(
                key="precipitation_probability",
                name="Chance of Precipitation",
                formatter=_friendly_precipitation_chance,
            ),
            WeatherAttribute(
                key="humidity", name="Humidity", formatter=_format_percent
            ),
        ]

        output = []
        for day in forecast:
            temp_low = day.get("templow")
            temperature = (
                f"{round(temp_low)} - {round(day['temperature'])}"
                if temp_low is not None
                else round(day["temperature"])
            )
            output.append(
                "\n".join(
                    [
                        f"- Date: {self._format_date(day['datetime'])}",
                        f"  Temperature: {temperature}",
                        *_build_attributes(daily_attributes, day),
                    ],
                ),
            )

        return "\n".join(output)

    async def _get_twice_daily_forecast(
        self,
        hass: HomeAssistant,
        entity_id: str,
        target_date: date | None,
    ) -> str:
        """Build the twice daily forecast data."""
        forecast = await hass.services.async_call(
            "weather",
            "get_forecasts",
            {"entity_id": entity_id, "type": "twice_daily"},
            blocking=True,
            return_response=True,
        )
        forecast = forecast.get(entity_id, {}).get("forecast")
        if not forecast:
            raise ForecastRetrievalError

        if target_date:
            forecast = self._filter_forecast_by_day(forecast, target_date)

        daily_attributes = [
            WeatherAttribute(key="condition", name="General Condition", formatter=None),
            WeatherAttribute(
                key="precipitation_probability",
                name="Chance of Precipitation",
                formatter=_friendly_precipitation_chance,
            ),
            WeatherAttribute(
                key="humidity", name="Humidity", formatter=_format_percent
            ),
        ]

        days = {}
        for day in forecast:
            dt = datetime.fromisoformat(day["datetime"]).astimezone()
            target_date = dt.date()
            day_night = "day" if day.get("is_daytime", True) else "night"
            date_str = target_date.strftime("%A %-d %B")

            days[date_str] = days.get(
                date_str,
                {
                    "day": {},
                    "night": {},
                },
            )

            days[date_str][day_night] = day

        output = []
        for day in forecast:
            temp_low = day.get("templow")
            temperature = (
                f"{round(temp_low)} - {round(day['temperature'])}"
                if temp_low is not None
                else round(day["temperature"])
            )
            output.append(
                "\n".join(
                    [
                        f"- Date: {self._format_date(day['datetime'])} {'daytime' if day['is_daytime'] else 'nighttime'}",
                        f"  Temperature: {temperature}",
                        *_build_attributes(daily_attributes, day),
                    ],
                ),
            )
        return "\n".join(output)

    async def _get_hourly_forecast(
        self,
        hass: HomeAssistant,
        entity_id: str,
        target_date: date,
    ) -> str:
        """Build the hourly forecast data."""
        forecast = await hass.services.async_call(
            "weather",
            "get_forecasts",
            {"entity_id": entity_id, "type": "hourly"},
            blocking=True,
            return_response=True,
        )
        forecast = forecast.get(entity_id, {}).get("forecast")
        if not forecast:
            raise ForecastRetrievalError

        forecast = self._filter_forecast_by_day(forecast, target_date)

        hourly_attributes = [
            WeatherAttribute(name="General Condition", key="condition", formatter=None),
            WeatherAttribute(
                name="Chance of Precipitation",
                key="precipitation_probability",
                formatter=_friendly_precipitation_chance,
            ),
            WeatherAttribute(
                key="humidity", name="Humidity", formatter=_format_percent
            ),
        ]

        output = []
        for hour in forecast:
            output.append(
                "\n".join(
                    [
                        f"- Time: {self._format_time(hour['datetime'])}",
                        f"  Temperature: {round(hour['temperature'])}",
                        *_build_attributes(hourly_attributes, hour),
                    ],
                ),
            )

        return "\n".join(output)

    @staticmethod
    def _get_current_temperature_sensor_data(
        hass: HomeAssistant,
        temperature_entity_id: str,
    ) -> str | None:
        output = []
        # Add current temperature from sensor if provided and this is today's forecast
        sensor_state = hass.states.get(temperature_entity_id)
        if sensor_state and sensor_state.state not in ("unknown", "unavailable"):
            try:
                current_temp = round(float(sensor_state.state))
                output.append(
                    "\n".join(
                        [
                            "- Time: current",
                            f"  Temperature: {current_temp}",
                        ],
                    ),
                )
            except (ValueError, TypeError):
                _LOGGER.warning(
                    "Could not parse temperature sensor value: %s",
                    sensor_state.state,
                )
                return None
        return "\n".join(output)

    @staticmethod
    def _get_current_conditions(
        hass: HomeAssistant,
        entity_id: str | None,
    ) -> str | None:
        """Build current conditions from a weather entity's state attributes."""
        if not entity_id:
            return None
        entity = hass.states.get(entity_id)
        if entity is None or entity.state in ("unknown", "unavailable"):
            return None
        data = dict(entity.attributes)
        data["condition"] = entity.state
        current_attributes = [
            WeatherAttribute(key="condition", name="General Condition", formatter=None),
            WeatherAttribute(
                key="temperature",
                name="Temperature",
                formatter=lambda v: round(float(v)),
            ),
            WeatherAttribute(
                key="apparent_temperature",
                name="Feels Like",
                formatter=lambda v: round(float(v)),
            ),
            WeatherAttribute(
                key="humidity", name="Humidity", formatter=_format_percent
            ),
            WeatherAttribute(
                key="dew_point", name="Dew Point", formatter=lambda v: round(float(v))
            ),
            WeatherAttribute(
                key="wind_speed", name="Wind Speed", formatter=lambda v: round(float(v))
            ),
            WeatherAttribute(
                key="wind_gust_speed",
                name="Wind Gusts",
                formatter=lambda v: round(float(v)),
            ),
            WeatherAttribute(
                key="pressure", name="Pressure", formatter=lambda v: round(float(v))
            ),
        ]
        lines = _build_attributes(current_attributes, data)
        if not lines:
            return None
        return "\n".join(["- Time: current", *lines])

    @staticmethod
    def _remote_target_date(date_range: str, place_today: date) -> date | None:
        """Find the target date in the remote place's own calendar."""
        if date_range == "week":
            return None
        if date_range == "today":
            return place_today
        if date_range == "tomorrow":
            return place_today + timedelta(days=1)
        weekdays = {
            "monday": 0,
            "tuesday": 1,
            "wednesday": 2,
            "thursday": 3,
            "friday": 4,
            "saturday": 5,
            "sunday": 6,
        }
        target_weekday = weekdays.get(date_range)
        if target_weekday is None:
            return None
        days_ahead = (target_weekday - place_today.weekday() + 7) % 7
        return place_today + timedelta(days=days_ahead)

    async def _get_open_meteo_forecast(
        self,
        hass: HomeAssistant,
        location: str,
        date_range: str,
    ) -> str:
        """
        Get current conditions and forecast for another place from Open-Meteo.

        Open-Meteo's free API explicitly permits personal home automation use;
        data is CC BY 4.0, hence the source attribution in the output.
        """
        session = async_get_clientsession(hass)

        async with session.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={
                "name": location,
                "count": 1,
                "language": "en",
                "format": "json",
            },
        ) as resp:
            resp.raise_for_status()
            geo = await resp.json()

        results = geo.get("results") or []
        if not results:
            message = f"Could not find a place named '{location}'"
            raise ForecastRetrievalError(message)
        place = results[0]

        params: dict[str, Any] = {
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current": (
                "temperature_2m,relative_humidity_2m,apparent_temperature,"
                "dew_point_2m,weather_code,wind_speed_10m,wind_gusts_10m,"
                "surface_pressure"
            ),
            "hourly": "temperature_2m,precipitation_probability,relative_humidity_2m",
            "daily": (
                "weather_code,temperature_2m_max,temperature_2m_min,"
                "precipitation_probability_max,relative_humidity_2m_mean"
            ),
            "timezone": "auto",
            "forecast_days": 8,
        }
        if hass.config.units.temperature_unit != UnitOfTemperature.CELSIUS:
            params["temperature_unit"] = "fahrenheit"
            params["wind_speed_unit"] = "mph"

        async with session.get(
            "https://api.open-meteo.com/v1/forecast",
            params=params,
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()

        place_name = ", ".join(
            part
            for part in [place.get("name"), place.get("admin1"), place.get("country")]
            if part
        )
        lines = [f"Weather for {place_name} (source: Open-Meteo, CC BY 4.0)"]

        current = data.get("current") or {}
        place_today = (
            date.fromisoformat(current["time"][:10])
            if current.get("time")
            else datetime.now().astimezone().date()
        )
        target_date = self._remote_target_date(date_range, place_today)

        if current:
            current_data = {
                "condition": WMO_CONDITIONS.get(current.get("weather_code")),
                "temperature": current.get("temperature_2m"),
                "apparent_temperature": current.get("apparent_temperature"),
                "humidity": current.get("relative_humidity_2m"),
                "dew_point": current.get("dew_point_2m"),
                "wind_speed": current.get("wind_speed_10m"),
                "wind_gust_speed": current.get("wind_gusts_10m"),
                "pressure": current.get("surface_pressure"),
            }
            current_attributes = [
                WeatherAttribute(
                    key="condition", name="General Condition", formatter=None
                ),
                WeatherAttribute(
                    key="temperature",
                    name="Temperature",
                    formatter=lambda v: round(float(v)),
                ),
                WeatherAttribute(
                    key="apparent_temperature",
                    name="Feels Like",
                    formatter=lambda v: round(float(v)),
                ),
                WeatherAttribute(
                    key="humidity", name="Humidity", formatter=_format_percent
                ),
                WeatherAttribute(
                    key="dew_point",
                    name="Dew Point",
                    formatter=lambda v: round(float(v)),
                ),
                WeatherAttribute(
                    key="wind_speed",
                    name="Wind Speed",
                    formatter=lambda v: round(float(v)),
                ),
                WeatherAttribute(
                    key="wind_gust_speed",
                    name="Wind Gusts",
                    formatter=lambda v: round(float(v)),
                ),
                WeatherAttribute(
                    key="pressure", name="Pressure", formatter=lambda v: round(float(v))
                ),
            ]
            built = _build_attributes(current_attributes, current_data)
            if built:
                lines.append("\n".join(["- Time: current", *built]))

        daily = data.get("daily") or {}
        daily_attributes = [
            WeatherAttribute(key="condition", name="General Condition", formatter=None),
            WeatherAttribute(
                key="precipitation_probability",
                name="Chance of Precipitation",
                formatter=_friendly_precipitation_chance,
            ),
            WeatherAttribute(
                key="humidity", name="Humidity", formatter=_format_percent
            ),
        ]
        day_count = len(daily.get("time", []))
        precip_max = daily.get("precipitation_probability_max") or [None] * day_count
        humidity_mean = daily.get("relative_humidity_2m_mean") or [None] * day_count
        for i, day_str in enumerate(daily.get("time", [])):
            if target_date and day_str != target_date.isoformat():
                continue
            day_data = {
                "condition": WMO_CONDITIONS.get(daily["weather_code"][i]),
                "precipitation_probability": precip_max[i],
                "humidity": humidity_mean[i],
            }
            temp_low = daily["temperature_2m_min"][i]
            temp_high = daily["temperature_2m_max"][i]
            temperature = (
                f"{round(temp_low)} - {round(temp_high)}"
                if temp_low is not None
                else round(temp_high)
            )
            weekday = date.fromisoformat(day_str).strftime("%A")
            label = (
                f"Today ({weekday})" if day_str == place_today.isoformat() else weekday
            )
            lines.append(
                "\n".join(
                    [
                        f"- Date: {label}",
                        f"  Temperature: {temperature}",
                        *_build_attributes(daily_attributes, day_data),
                    ],
                ),
            )

        if target_date:
            hourly = data.get("hourly") or {}
            hour_count = len(hourly.get("time", []))
            h_precip = hourly.get("precipitation_probability") or [None] * hour_count
            h_humidity = hourly.get("relative_humidity_2m") or [None] * hour_count
            hourly_attributes = [
                WeatherAttribute(
                    name="Chance of Precipitation",
                    key="precipitation_probability",
                    formatter=_friendly_precipitation_chance,
                ),
                WeatherAttribute(
                    key="humidity", name="Humidity", formatter=_format_percent
                ),
            ]
            for i, time_str in enumerate(hourly.get("time", [])):
                if not time_str.startswith(target_date.isoformat()):
                    continue
                hour_data = {
                    "precipitation_probability": h_precip[i],
                    "humidity": h_humidity[i],
                }
                hour_label = datetime.fromisoformat(time_str).strftime("%-I%p").lower()
                lines.append(
                    "\n".join(
                        [
                            f"- Time: {hour_label}",
                            f"  Temperature: {round(hourly['temperature_2m'][i])}",
                            *_build_attributes(hourly_attributes, hour_data),
                        ],
                    ),
                )

        return "\n".join(lines)

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> ToolResult:
        """Call the tool."""
        config_data = self.config

        date_range = tool_input.tool_args.get("range", "week").lower()
        _LOGGER.info("Weather forecast for the period: %s", date_range)

        location = tool_input.tool_args.get("location")
        if location:
            try:
                return ToolResult(
                    data={
                        "result": await self._get_open_meteo_forecast(
                            hass, location, date_range
                        )
                    }
                )
            except Exception as e:
                _LOGGER.exception(msg="Open-Meteo weather lookup encountered an error")
                return ToolResult(
                    data={"error": f"Error retrieving weather for {location}: {e!s}"},
                    error=True,
                )

        try:
            hourly_entity_id = config_data.get(CONF_HOURLY_WEATHER_ENTITY)
            daily_entity_id = config_data.get(CONF_DAILY_WEATHER_ENTITY)
            current_temperature_entity_id = config_data.get(
                CONF_WEATHER_TEMPERATURE_SENSOR,
            )

            forecast = None
            target_date = self._find_target_date(date_range)

            if date_range != "week" and hourly_entity_id and target_date:
                forecast = await self._get_hourly_forecast(
                    hass,
                    hourly_entity_id,
                    target_date,
                )

            if not forecast and daily_entity_id:
                is_twice_daily = self.has_twice_daily_data(daily_entity_id)

                if is_twice_daily:
                    forecast = await self._get_twice_daily_forecast(
                        hass,
                        daily_entity_id,
                        target_date,
                    )
                else:
                    forecast = await self._get_daily_forecast(
                        hass,
                        daily_entity_id,
                        target_date,
                    )

            if (
                forecast
                and current_temperature_entity_id
                and target_date == datetime.now().astimezone().date()
            ):
                current_temperature_data = self._get_current_temperature_sensor_data(
                    hass,
                    current_temperature_entity_id,
                )
                if current_temperature_data:
                    forecast = current_temperature_data + "\n" + forecast

            if forecast and target_date == datetime.now().astimezone().date():
                current_conditions = self._get_current_conditions(
                    hass,
                    daily_entity_id or hourly_entity_id,
                )
                if current_conditions:
                    forecast = current_conditions + "\n" + forecast

            if not forecast:
                forecast = "No weather forecast available for the selected range"

            return ToolResult(data={"result": forecast})
        except Exception as e:
            _LOGGER.exception(msg="Weather forecast encountered an error")
            return ToolResult(
                data={"error": f"Error retrieving weather forecast: {e!s}"},
                error=True,
            )
