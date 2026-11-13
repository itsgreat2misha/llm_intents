"""Tests for the WeatherForecastTool."""

import datetime as dt
from datetime import date, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.components.weather import WeatherEntityFeature
from homeassistant.const import UnitOfTemperature
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm
from homeassistant.helpers.llm import ToolResult
from pytest_homeassistant.common import MockConfigEntry

from custom_components.llm_intents.const import DOMAIN
from custom_components.llm_intents.weather import (
    ForecastRetrievalError,
    WeatherAttribute,
    WeatherEntityNotFoundError,
    WeatherForecastTool,
    _build_attributes,
    _format_temperature,
    _format_temperature_range,
    _friendly_precipitation_chance,
)

from .utils import MockContext


@pytest.fixture
def tool(hass: HomeAssistant) -> WeatherForecastTool:
    """Return an instance of the WeatherForecastTool."""
    return WeatherForecastTool({}, hass)


@pytest.mark.parametrize(
    ("precipitation_chance", "expected"),
    [
        (0, "none"),
        (1, "very unlikely"),
        (5, "very unlikely"),
        (6, "unlikely"),
        (15, "unlikely"),
        (16, "possible"),
        (30, "possible"),
        (31, "moderate"),
        (50, "moderate"),
        (51, "likely"),
        (70, "likely"),
        (71, "very likely"),
        (85, "very likely"),
        (95, "extremely likely"),
        (100, "almost guaranteed"),
        (101, "almost guaranteed"),
    ],
)
def test_friendly_precipitation_chance(
    precipitation_chance: int, expected: str
) -> None:
    """Test precipitation chance categorization at various thresholds."""
    result = _friendly_precipitation_chance(precipitation_chance)
    assert result == expected


@pytest.mark.parametrize(
    ("value", "unit", "expected"),
    [
        (41, UnitOfTemperature.FAHRENHEIT, "41°F (5°C)"),
        (66, UnitOfTemperature.FAHRENHEIT, "66°F (19°C)"),
        (32, UnitOfTemperature.FAHRENHEIT, "32°F (0°C)"),
        (-40, UnitOfTemperature.FAHRENHEIT, "-40°F (-40°C)"),
        (0, UnitOfTemperature.CELSIUS, "0°C (32°F)"),
        (20, UnitOfTemperature.CELSIUS, "20°C (68°F)"),
        (100, UnitOfTemperature.CELSIUS, "100°C (212°F)"),
        (21.6, UnitOfTemperature.CELSIUS, "22°C (71°F)"),
    ],
)
def test_format_temperature(
    value: float, unit: UnitOfTemperature, expected: str
) -> None:
    """Test temperatures are rendered in both Fahrenheit and Celsius."""
    assert _format_temperature(value, unit) == expected


def test_format_temperature_range() -> None:
    """Test low/high pairs render both units; a missing low gives one value."""
    assert (
        _format_temperature_range(32, 49, UnitOfTemperature.FAHRENHEIT)
        == "32°F (0°C) - 49°F (9°C)"
    )
    assert (
        _format_temperature_range(None, 49, UnitOfTemperature.FAHRENHEIT)
        == "49°F (9°C)"
    )


def test_build_attributes() -> None:
    """Test building attributes function with and without a formatter provided."""
    weather_data = {"precipitation_probability": 75, "condition": "Sunny"}
    attributes = [
        WeatherAttribute(
            name="Condition",
            key="condition",
            formatter=None,
        ),
        WeatherAttribute(
            name="Chance of Precipitation",
            key="precipitation_probability",
            formatter=_friendly_precipitation_chance,
        ),
    ]
    # Set the name attribute after creation
    result = _build_attributes(attributes, weather_data)

    assert result[0] == "  Condition: Sunny"
    assert result[1] == "  Chance of Precipitation: very likely"


def test_build_attributes_missing_key() -> None:
    """Test that missing keys are skipped."""
    weather_data = {}
    attributes = [
        WeatherAttribute(name="Condition", key="condition", formatter=None),
    ]
    result = _build_attributes(attributes, weather_data)
    assert result == []


def test_build_attributes_none_value_and_missing_key() -> None:
    """Test that None values are skipped (not formatted or included) and missing keys are also skipped."""
    weather_data = {
        "condition": "Sunny",
        "precipitation_probability": None,
        # "wind_speed" is intentionally absent to test missing key handling
    }
    attributes = [
        WeatherAttribute(name="Condition", key="condition", formatter=None),
        WeatherAttribute(
            name="Chance of Precipitation",
            key="precipitation_probability",
            formatter=_friendly_precipitation_chance,
        ),
        WeatherAttribute(name="Wind Speed", key="wind_speed", formatter=None),
    ]
    result = _build_attributes(attributes, weather_data)

    assert len(result) == 1
    assert result[0] == "  Condition: Sunny"
    # precipitation_probability with None value should be skipped
    assert not any("Precipitation" in line for line in result)
    # wind_speed key is missing so should also be skipped
    assert not any("Wind Speed" in line for line in result)


# =============================================================================
# _find_target_date() tests
# =============================================================================


@pytest.mark.parametrize(
    ("date_range", "expected_delta_days"),
    [
        ("today", 0),
        ("tomorrow", 1),
        ("tuesday", 5),
        ("wednesday", 6),
        ("thursday", 0),
        ("friday", 1),
        ("saturday", 2),
        ("sunday", 3),
    ],
)
@pytest.mark.freeze_time("2026-01-01")
def test_find_target_date_relative(
    date_range: str,
    expected_delta_days: int,
) -> None:
    """Test finding target date for relative expressions."""
    result = WeatherForecastTool._find_target_date(date_range)
    assert isinstance(result, date)
    assert result == datetime.now().astimezone().date() + timedelta(
        days=expected_delta_days
    )


def test_find_target_date_invalid() -> None:
    """Test that invalid date ranges return None."""
    result = WeatherForecastTool._find_target_date("next week")
    assert result is None


# =============================================================================
# _filter_forecast_by_day() tests
# =============================================================================


def test_filter_forecast_by_day_matches() -> None:
    """Test filtering forecast when entries match target date."""
    forecast = [
        {"datetime": "2026-05-01T00:00:00+00:00", "temperature": 20},
        {"datetime": "2026-05-03T00:00:00+00:00", "temperature": 22},
    ]
    target = date(2026, 5, 3)
    result = WeatherForecastTool._filter_forecast_by_day(forecast, target)
    assert len(result) == 1
    assert result[0]["temperature"] == 22


def test_filter_forecast_by_day_no_matches() -> None:
    """Test filtering forecast when no entries match target date."""
    forecast = [
        {"datetime": "2026-05-01T00:00:00+00:00", "temperature": 20},
    ]
    target = date(2026, 5, 3)
    result = WeatherForecastTool._filter_forecast_by_day(forecast, target)
    assert result == []


def test_filter_forecast_by_day_multiple_matches() -> None:
    """Test filtering forecast when multiple entries match target date."""
    forecast = [
        {"datetime": "2026-05-03T08:00:00+00:00", "temperature": 18},
        {"datetime": "2026-05-03T12:00:00+00:00", "temperature": 24},
        {"datetime": "2026-05-03T04:00:00+00:00", "temperature": 16},
    ]
    target = date(2026, 5, 3)
    result = WeatherForecastTool._filter_forecast_by_day(forecast, target)
    assert len(result) == 3


# =============================================================================
# _format_time() tests
# =============================================================================


@pytest.mark.parametrize(
    ("input_hour", "expected_output"),
    [
        (8, "9am"),
        (14, "3pm"),
        (0, "1am"),
        (12, "1pm"),
        (1, "2am"),
        (13, "2pm"),
        (23, "12am"),
    ],
)
def test_format_time(
    input_hour: int,
    expected_output: str,
) -> None:
    """Test formatting time - returns NEXT hour in local timezone."""
    # Mock datetime module to control timezone behavior
    with patch("custom_components.llm_intents.weather.datetime") as mock_dt:
        # Mock fromisoformat to return a mock datetime instance
        mock_datetime_instance = MagicMock()
        mock_dt.fromisoformat.return_value = mock_datetime_instance

        # Mock astimezone to return a datetime at the input hour
        mock_astimezone_result = dt.datetime(
            2026, 5, 3, input_hour, 0, 0, tzinfo=dt.UTC
        )
        mock_datetime_instance.astimezone.return_value = mock_astimezone_result

        result = WeatherForecastTool._format_time(
            f"2026-05-03T{input_hour:02d}:00:00+00:00"
        )
        assert result == expected_output


# =============================================================================
# _format_date() tests
# =============================================================================


@pytest.mark.freeze_time("2026-01-01")
def test_format_date_today() -> None:
    """Test formatting today's date."""
    # Use a date that is definitely in the future to avoid "today" ambiguity
    result = WeatherForecastTool._format_date("2026-06-01T00:00:00+00:00")
    assert "Today" not in result
    assert "Monday" in result  # 2026-06-01 is a Monday


@pytest.mark.freeze_time("2026-01-01")
def test_format_date_future() -> None:
    """Test formatting a future date."""
    # Use a date that is definitely in the future to avoid "today" ambiguity
    result = WeatherForecastTool._format_date("2026-06-15T00:00:00+00:00")
    assert "Today" not in result
    assert "Monday" in result  # 2026-06-15 is a Monday


# =============================================================================
# has_twice_daily_data() tests
# =============================================================================


def test_has_twice_daily_data_entity_not_found(tool: WeatherForecastTool) -> None:
    """Test that WeatherEntityNotFoundError is raised when entity is not found."""
    # Create states mock before patching
    mock_states = MagicMock()
    mock_states.get.return_value = None
    tool.hass.states = mock_states
    with pytest.raises(WeatherEntityNotFoundError):
        tool.has_twice_daily_data("sensor.test_weather")


def test_has_twice_daily_data_not_supported(tool: WeatherForecastTool) -> None:
    """Test when entity exists but doesn't support twice-daily data."""
    mock_states = MagicMock()
    entity = MagicMock()
    entity.attributes = {"supported_features": 0}
    mock_states.get.return_value = entity
    tool.hass.states = mock_states
    result = tool.has_twice_daily_data("sensor.test_weather")
    assert not result


def test_has_twice_daily_data_supported(tool: WeatherForecastTool) -> None:
    """Test when entity supports twice-daily data."""
    mock_states = MagicMock()
    entity = MagicMock()
    entity.attributes = {
        "supported_features": WeatherEntityFeature.FORECAST_TWICE_DAILY
    }
    mock_states.get.return_value = entity
    tool.hass.states = mock_states
    result = tool.has_twice_daily_data("sensor.test_weather")
    assert result


# =============================================================================
# _get_current_temperature_sensor_data() tests
# =============================================================================


@pytest.mark.parametrize(
    ("state", "expected_result"),
    [
        ("22.5", "- Time: current\n  Temperature: 22°C (72°F)"),
        ("23.9", "- Time: current\n  Temperature: 24°C (75°F)"),
        ("0.0", "- Time: current\n  Temperature: 0°C (32°F)"),
        ("100.1", "- Time: current\n  Temperature: 100°C (212°F)"),
        ("unknown", ""),
        ("unavailable", ""),
        ("invalid", None),
        ("-5.7", "- Time: current\n  Temperature: -6°C (22°F)"),
        ("22", "- Time: current\n  Temperature: 22°C (72°F)"),
        ("-10.0", "- Time: current\n  Temperature: -10°C (14°F)"),
    ],
)
def test_get_current_temperature_sensor_data(
    tool: WeatherForecastTool,
    hass: HomeAssistant,
    state: str,
    expected_result: str,
) -> None:
    """Test getting current temperature sensor data with various states."""
    mock_states = MagicMock()
    entity = MagicMock()
    entity.state = state
    mock_states.get.return_value = entity
    tool.hass.states = mock_states
    result = tool._get_current_temperature_sensor_data(hass, "sensor.temperature")
    assert result == expected_result


# =============================================================================
# _get_daily_forecast() tests
# =============================================================================


@pytest.mark.asyncio
async def test_get_daily_forecast_with_target_date(
    tool: WeatherForecastTool, hass: HomeAssistant
) -> None:
    """Test getting daily forecast with a target date."""
    tool: WeatherForecastTool  # Set up mock hass data and states
    mock_entry = MockConfigEntry(domain=DOMAIN, options={})
    hass.data = {DOMAIN: {"config": {}}}
    mock_entry.add_to_hass(hass)

    forecast_data = [
        {
            "datetime": "2026-05-01T00:00:00+00:00",
            "temperature": 20,
            "templow": 15,
            "condition": "Sunny",
            "precipitation_probability": 30,
        },
        {
            "datetime": "2026-05-03T00:00:00+00:00",
            "temperature": 22,
            "templow": 17,
            "condition": "Cloudy",
            "precipitation_probability": 50,
        },
    ]

    mock_service = AsyncMock()
    mock_service.return_value = {"sensor.test_weather": {"forecast": forecast_data}}
    mock_services = MagicMock()
    mock_services.async_call = mock_service
    tool.hass.services = mock_services

    target_date = date(2026, 5, 3)
    result = await tool._get_daily_forecast(
        hass,
        "sensor.test_weather",
        target_date,
    )

    assert "Sunday" in result  # 2026-05-03 is a Sunday
    assert "Cloudy" in result
    assert "Precipitation: moderate" in result


@pytest.mark.asyncio
async def test_get_daily_forecast_without_target_date(
    tool: WeatherForecastTool, hass: HomeAssistant
) -> None:
    """Test getting daily forecast without a target date (all days)."""
    # Set up mock hass data and states
    mock_entry = MockConfigEntry(domain=DOMAIN, options={})
    hass.data = {DOMAIN: {"config": {}}}
    mock_entry.add_to_hass(hass)

    forecast_data = [
        {
            "datetime": "2026-05-01T00:00:00+00:00",
            "temperature": 20,
            "templow": 15,
            "condition": "Sunny",
            "precipitation_probability": 30,
        },
        {
            "datetime": "2026-05-02T00:00:00+00:00",
            "temperature": 21,
            "templow": 16,
            "condition": "Partly Cloudy",
            "precipitation_probability": 40,
        },
    ]

    mock_service = AsyncMock()
    mock_service.return_value = {"sensor.test_weather": {"forecast": forecast_data}}
    mock_services = MagicMock()
    mock_services.async_call = mock_service
    tool.hass.services = mock_services

    result = await tool._get_daily_forecast(hass, "sensor.test_weather", None)

    assert len(result.split("\n")) >= 2
    assert "Sunny" in result
    assert "Partly Cloudy" in result


@pytest.mark.asyncio
async def test_get_daily_forecast_no_forecast(
    tool: WeatherForecastTool, hass: HomeAssistant
) -> None:
    """Test when no forecast data is available."""
    # Set up mock hass data and states
    mock_entry = MockConfigEntry(domain=DOMAIN, options={})
    hass.data = {DOMAIN: {"config": {}}}
    mock_entry.add_to_hass(hass)

    mock_service = AsyncMock()
    mock_service.return_value = {"sensor.test_weather": {"forecast": []}}
    mock_services = MagicMock()
    mock_services.async_call = mock_service
    tool.hass.services = mock_services

    with pytest.raises(ForecastRetrievalError):
        await tool._get_daily_forecast(hass, "sensor.test_weather", None)


# =============================================================================
# _get_twice_daily_forecast() tests
# =============================================================================


@pytest.mark.asyncio
async def test_get_twice_daily_forecast(
    tool: WeatherForecastTool, hass: HomeAssistant
) -> None:
    """Test getting twice-daily forecast."""
    # Set up mock hass data and states
    mock_entry = MockConfigEntry(domain=DOMAIN, options={})
    hass.data = {DOMAIN: {"config": {}}}
    mock_entry.add_to_hass(hass)

    forecast_data = [
        {
            "datetime": "2026-05-03T00:00:00+00:00",
            "temperature": 20,
            "templow": 15,
            "condition": "Sunny",
            "precipitation_probability": 30,
            "is_daytime": True,
        },
        {
            "datetime": "2026-05-03T00:00:00+00:00",
            "temperature": 16,
            "templow": 12,
            "condition": "Clear",
            "precipitation_probability": 10,
            "is_daytime": False,
        },
    ]

    mock_service = AsyncMock()
    mock_service.return_value = {"sensor.test_weather": {"forecast": forecast_data}}
    mock_services = MagicMock()
    mock_services.async_call = mock_service
    tool.hass.services = mock_services

    target_date = date(2026, 5, 3)
    result = await tool._get_twice_daily_forecast(
        hass,
        "sensor.test_weather",
        target_date,
    )

    assert "daytime" in result
    assert "nighttime" in result
    assert "Sunny" in result
    assert "Clear" in result


@pytest.mark.asyncio
async def test_get_twice_daily_forecast_no_twice_daily_support(
    tool: WeatherForecastTool, hass: HomeAssistant
) -> None:
    """Test that forecast is still returned even if entity doesn't support twice-daily."""
    # Set up mock hass data and states
    mock_entry = MockConfigEntry(domain=DOMAIN, options={})
    hass.data = {DOMAIN: {"config": {}}}
    mock_entry.add_to_hass(hass)

    forecast_data = [
        {
            "datetime": "2026-05-03T00:00:00+00:00",
            "temperature": 20,
            "templow": 15,
            "condition": "Sunny",
            "precipitation_probability": 30,
            "is_daytime": True,
        },
    ]

    mock_service = AsyncMock()
    mock_service.return_value = {"sensor.test_weather": {"forecast": forecast_data}}
    mock_services = MagicMock()
    mock_services.async_call = mock_service
    tool.hass.services = mock_services

    target_date = date(2026, 5, 3)
    result = await tool._get_twice_daily_forecast(
        hass,
        "sensor.test_weather",
        target_date,
    )

    assert "Sunny" in result


@pytest.mark.asyncio
async def test_get_twice_daily_forecast_no_forecast(
    tool: WeatherForecastTool, hass: HomeAssistant
) -> None:
    """Test when no twice-daily forecast data is available."""
    # Set up mock hass data and states
    mock_entry = MockConfigEntry(domain=DOMAIN, options={})
    hass.data = {DOMAIN: {"config": {}}}
    mock_entry.add_to_hass(hass)

    mock_service = AsyncMock()
    mock_service.return_value = {"sensor.test_weather": {"forecast": []}}
    mock_services = MagicMock()
    mock_services.async_call = mock_service
    tool.hass.services = mock_services

    with pytest.raises(ForecastRetrievalError):
        await tool._get_twice_daily_forecast(hass, "sensor.test_weather", None)


# =============================================================================
# _get_hourly_forecast() tests
# =============================================================================


@pytest.mark.asyncio
async def test_get_hourly_forecast(
    tool: WeatherForecastTool,
    hass: HomeAssistant,
) -> None:
    """Test getting hourly forecast."""
    # Set up mock hass data and states
    mock_entry = MockConfigEntry(domain=DOMAIN, options={})
    hass.data = {DOMAIN: {"config": {}}}
    mock_entry.add_to_hass(hass)

    forecast_data = [
        {
            "datetime": "2026-05-03T06:00:00+00:00",
            "temperature": 18,
            "condition": "Partly Cloudy",
            "precipitation_probability": 20,
        },
        {
            "datetime": "2026-05-03T07:00:00+00:00",
            "temperature": 20,
            "condition": "Sunny",
            "precipitation_probability": 10,
        },
        {
            "datetime": "2026-05-03T12:00:00+00:00",
            "temperature": 25,
            "condition": "Sunny",
            "precipitation_probability": 0,
        },
        {
            "datetime": "2026-05-03T18:00:00+00:00",
            "temperature": 22,
            "condition": "Cloudy",
            "precipitation_probability": 40,
        },
    ]

    mock_service = AsyncMock()
    mock_service.return_value = {"sensor.test_weather": {"forecast": forecast_data}}
    mock_services = MagicMock()
    mock_services.async_call = mock_service
    tool.hass.services = mock_services

    target_date = date(2026, 5, 3)

    # Mock datetime to control timezone behavior
    with patch("custom_components.llm_intents.weather.datetime") as mock_dt:
        mock_dt_instance = mock_dt.return_value

        # Mock astimezone to return UTC timezone (no offset) so dates match
        mock_dt_instance.astimezone.return_value = dt.datetime(
            2026, 5, 3, 6, 0, 0, tzinfo=dt.UTC
        )
        mock_dt_instance.now.return_value = dt.datetime(
            2026, 5, 3, 0, 0, 0, tzinfo=dt.UTC
        )

        # Mock fromisoformat to return a UTC datetime
        mock_dt.fromisoformat.return_value = dt.datetime(
            2026, 5, 3, 6, 0, 0, tzinfo=dt.UTC
        )

        result = await tool._get_hourly_forecast(
            hass,
            "sensor.test_weather",
            target_date,
        )

        assert "Partly Cloudy" in result
        assert "Sunny" in result
        assert "Cloudy" in result
        assert "Precipitation: possible" in result  # 20% is "possible"
        assert "Precipitation: unlikely" in result  # 10% is "unlikely"
        assert "Precipitation: none" in result  # 0% is "none"
        assert "Temperature: 18" in result
        assert "Temperature: 25" in result
        # Verify all 4 hourly entries are present
        assert result.count("- Time:") == 4


@pytest.mark.asyncio
async def test_get_hourly_forecast_no_forecast(
    tool: WeatherForecastTool, hass: HomeAssistant
) -> None:
    """Test when no hourly forecast data is available."""
    # Set up mock hass data and states
    mock_entry = MockConfigEntry(domain=DOMAIN, options={})
    hass.data = {DOMAIN: {"config": {}}}
    mock_entry.add_to_hass(hass)

    mock_service = AsyncMock()
    mock_service.return_value = {"sensor.test_weather": {"forecast": []}}
    mock_services = MagicMock()
    mock_services.async_call = mock_service
    tool.hass.services = mock_services

    target_date = date(2026, 5, 3)
    with pytest.raises(ForecastRetrievalError):
        await tool._get_hourly_forecast(hass, "sensor.test_weather", target_date)


# =============================================================================
# async_call() integration tests
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.freeze_time("2026-05-03")
async def test_async_call_daily_forecast(hass: HomeAssistant) -> None:
    """Test async_call with daily forecast."""
    tool = WeatherForecastTool(
        {"weather_daily_entity": "sensor.test_weather"},
        hass,
    )
    tool_input = llm.ToolInput(
        tool_args={"range": "tomorrow"},
        tool_name="get_weather_forecast",
    )

    forecast_data = [
        {
            "datetime": "2026-05-04T00:00:00+00:00",
            "temperature": 22,
            "templow": 17,
            "condition": "Sunny",
            "precipitation_probability": 30,
        }
    ]

    mock_service = AsyncMock()
    mock_service.return_value = {"sensor.test_weather": {"forecast": forecast_data}}
    mock_services = MagicMock()
    tool.hass.services = mock_services
    mock_states = MagicMock()
    mock_entity = MagicMock()
    mock_entity.attributes = {"supported_features": 0}  # Not twice daily
    mock_states.get.return_value = mock_entity
    tool.hass.states = mock_states

    async def mock_async_call(*args: object, **kwargs: Any) -> dict:
        return {"sensor.test_weather": {"forecast": forecast_data}}

    mock_services.async_call = mock_async_call

    result = await tool.async_call(
        hass,
        tool_input,
        MagicMock(spec=llm.LLMContext),
    )
    assert isinstance(result, ToolResult)

    assert "Sunny" in result.data["result"]
    assert "Precipitation: possible" in result.data["result"]  # 30% is "possible"


@pytest.mark.asyncio
@pytest.mark.freeze_time("2026-05-03")
async def test_async_call_with_temperature_sensor(hass: HomeAssistant) -> None:
    """Test async_call with current temperature sensor included."""
    tool = WeatherForecastTool(
        {
            "weather_hourly_entity": "sensor.test_weather",
            "current_temperature_entity": "sensor.temperature",
        },
        hass,
    )

    tool_input = llm.ToolInput(
        tool_args={"range": "today"},
        tool_name="get_weather_forecast",
    )

    forecast_data = [
        {
            "datetime": "2026-05-03T00:00:00+00:00",
            "temperature": 22,
            "templow": 17,
            "condition": "Sunny",
            "precipitation_probability": 30,
        }
    ]

    mock_service = AsyncMock()
    mock_service.return_value = {"sensor.test_weather": {"forecast": forecast_data}}
    mock_services = MagicMock()
    mock_services.async_call = mock_service
    tool.hass.services = mock_services

    mock_states = MagicMock()
    entity = MagicMock()
    entity.state = "23"
    entity.attributes = {"supported_features": 0}  # Not twice daily
    mock_states.get.return_value = entity
    tool.hass.states = mock_states

    result = await tool.async_call(
        hass,
        tool_input,
        MagicMock(spec=llm.LLMContext),
    )
    assert isinstance(result, ToolResult)

    assert "current" in result.data["result"]
    assert "Temperature: 23" in result.data["result"]

    # Verify the service was called for hourly forecast (not daily)
    mock_services.async_call.assert_called_once_with(
        "weather",
        "get_forecasts",
        {"entity_id": "sensor.test_weather", "type": "hourly"},
        blocking=True,
        return_response=True,
    )


@pytest.mark.asyncio
async def test_async_call_no_forecast_available(hass: HomeAssistant) -> None:
    """Test async_call when no forecast is available."""
    tool = WeatherForecastTool(
        {"weather_daily_entity": "sensor.test_weather"},
        hass,
    )
    tool_input = llm.ToolInput(
        tool_args={"range": "week"},
        tool_name="get_weather_forecast",
    )

    mock_service = AsyncMock(side_effect=ForecastRetrievalError)
    mock_services = MagicMock()
    mock_services.async_call = mock_service
    tool.hass.services = mock_services

    # Mock entity as existing but not twice-daily
    mock_entity = MagicMock()
    mock_entity.attributes = {"supported_features": 0}
    tool.hass.states = MagicMock()
    tool.hass.states.get.return_value = mock_entity

    result = await tool.async_call(
        hass,
        tool_input,
        MagicMock(spec=llm.LLMContext),
    )
    assert isinstance(result, ToolResult)

    # When forecast retrieval fails, return error message
    assert "Error retrieving weather forecast" in result.data.get("error", "")


@pytest.mark.asyncio
async def test_async_call_error_handling(hass: HomeAssistant) -> None:
    """Test async_call error handling."""
    tool = WeatherForecastTool(
        {"weather_daily_entity": "sensor.test_weather"},
        hass,
    )
    tool_input = llm.ToolInput(
        tool_args={"range": "today"},
        tool_name="get_weather_forecast",
    )

    mock_service = AsyncMock(side_effect=Exception("Service error"))
    mock_services = MagicMock()
    mock_services.async_call = mock_service
    tool.hass.services = mock_services
    tool.hass.states = MagicMock()
    tool.hass.states.get.return_value = None

    result = await tool.async_call(
        hass,
        tool_input,
        MagicMock(spec=llm.LLMContext),
    )
    assert isinstance(result, ToolResult)

    assert "error" in result.data
    assert "Error retrieving weather forecast" in result.data["error"]


@pytest.mark.freeze_time("2026-01-01")
def test_format_date_returns_today() -> None:
    """Test formatting when the forecast date matches today."""
    result = WeatherForecastTool._format_date("2026-01-01T00:00:00+00:00")
    assert result == "Today (Thursday)"


@pytest.mark.asyncio
@pytest.mark.freeze_time("2026-05-03")
async def test_async_call_twice_daily_forecast(hass: HomeAssistant) -> None:
    """Test async_call takes the twice-daily forecast path."""
    tool = WeatherForecastTool(
        {"weather_daily_entity": "sensor.test_weather"},
        hass,
    )
    tool_input = llm.ToolInput(
        tool_args={"range": "tomorrow"},
        tool_name="get_weather_forecast",
    )

    forecast_data = [
        {
            "datetime": "2026-05-04T00:00:00+00:00",
            "temperature": 22,
            "templow": 17,
            "condition": "Sunny",
            "precipitation_probability": 30,
            "is_daytime": True,
        },
        {
            "datetime": "2026-05-04T00:00:00+00:00",
            "temperature": 16,
            "templow": 12,
            "condition": "Clear",
            "precipitation_probability": 10,
            "is_daytime": False,
        },
    ]

    mock_service = AsyncMock()
    mock_service.return_value = {"sensor.test_weather": {"forecast": forecast_data}}
    mock_services = MagicMock()
    mock_services.async_call = mock_service
    tool.hass.services = mock_services

    mock_entity = MagicMock()
    mock_entity.attributes = {
        "supported_features": WeatherEntityFeature.FORECAST_TWICE_DAILY
    }
    tool.hass.states = MagicMock()
    tool.hass.states.get.return_value = mock_entity

    result = await tool.async_call(
        hass,
        tool_input,
        MagicMock(spec=llm.LLMContext),
    )
    assert isinstance(result, ToolResult)

    assert "daytime" in result.data["result"]
    assert "nighttime" in result.data["result"]
    assert "Sunny" in result.data["result"]
    assert "Clear" in result.data["result"]


@pytest.mark.asyncio
async def test_async_call_no_forecast_fallback(
    tool: WeatherForecastTool, hass: HomeAssistant
) -> None:
    """Test async_call returns fallback message when no forecast is available."""
    tool_input = llm.ToolInput(
        tool_args={"range": "week"},
        tool_name="get_weather_forecast",
    )

    mock_entry = MockConfigEntry(domain=DOMAIN, options={})
    hass.data = {
        DOMAIN: {
            "config": {},
        },
    }
    mock_entry.add_to_hass(hass)

    result = await tool.async_call(
        hass,
        tool_input,
        MagicMock(spec=llm.LLMContext),
    )
    assert isinstance(result, ToolResult)

    assert (
        result.data["result"] == "No weather forecast available for the selected range"
    )


# =============================================================================
# Current conditions and Open-Meteo location tests
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.freeze_time("2026-05-03")
async def test_async_call_prepends_current_conditions(hass: HomeAssistant) -> None:
    """Current conditions from the weather entity precede today's forecast."""
    tool = WeatherForecastTool(
        {"weather_daily_entity": "weather.test"},
        hass,
    )
    tool_input = llm.ToolInput(
        tool_args={"range": "today"},
        tool_name="get_weather_forecast",
    )

    forecast_data = [
        {
            "datetime": "2026-05-03T12:00:00+00:00",
            "temperature": 22,
            "templow": 17,
            "condition": "Sunny",
            "precipitation_probability": 30,
            "humidity": 65,
        }
    ]

    async def mock_async_call(*args: object, **kwargs: Any) -> dict:
        return {"weather.test": {"forecast": forecast_data}}

    mock_services = MagicMock()
    mock_services.async_call = mock_async_call
    tool.hass.services = mock_services

    mock_states = MagicMock()
    entity = MagicMock()
    entity.state = "partlycloudy"
    entity.attributes = {
        "supported_features": 0,
        "temperature": 21.6,
        "apparent_temperature": 20.2,
        "humidity": 78,
        "wind_speed": 12.4,
        "pressure": 1013,
    }
    mock_states.get.return_value = entity
    tool.hass.states = mock_states

    result = await tool.async_call(
        hass,
        tool_input,
        MagicMock(spec=llm.LLMContext),
    )
    assert isinstance(result, ToolResult)

    text = result.data["result"]
    assert "- Time: current" in text
    assert "General Condition: partlycloudy" in text
    assert "Temperature: 22°C (71°F)" in text
    assert "Feels Like: 20°C (68°F)" in text
    assert "Humidity: 78%" in text
    # Humidity is also reported inside the daily forecast itself.
    assert "Humidity: 65%" in text
    # Current conditions come before the day's forecast block.
    assert text.index("- Time: current") < text.index("- Date:")


def test_get_current_conditions_sparse_and_missing() -> None:
    """Current conditions degrade gracefully with sparse attributes."""
    hass = MagicMock()
    entity = MagicMock()
    entity.state = "sunny"
    entity.attributes = {"humidity": 55}
    hass.states.get.return_value = entity

    result = WeatherForecastTool._get_current_conditions(hass, "weather.test")
    assert result is not None
    assert "Humidity: 55%" in result
    assert "Temperature" not in result

    hass.states.get.return_value = None
    assert WeatherForecastTool._get_current_conditions(hass, "weather.test") is None
    assert WeatherForecastTool._get_current_conditions(hass, None) is None


def test_remote_target_date() -> None:
    """Remote target dates resolve on the place's own calendar."""
    sunday = date(2026, 5, 3)  # 2026-05-03 is a Sunday
    assert WeatherForecastTool._remote_target_date("today", sunday) == sunday
    assert WeatherForecastTool._remote_target_date("tomorrow", sunday) == date(
        2026, 5, 4
    )
    assert WeatherForecastTool._remote_target_date("week", sunday) is None
    assert WeatherForecastTool._remote_target_date("monday", sunday) == date(2026, 5, 4)
    assert WeatherForecastTool._remote_target_date("sunday", sunday) == sunday


def _open_meteo_session(geo_data: dict, forecast_data: dict | None = None) -> MagicMock:
    """Build a mock client session dispatching Open-Meteo GETs by URL."""

    def mock_get(url: str, **_kwargs: Any) -> MockContext:
        response = MagicMock()
        response.status = 200
        response.raise_for_status = MagicMock()
        if "geocoding" in url:
            response.json = AsyncMock(return_value=geo_data)
        else:
            response.json = AsyncMock(return_value=forecast_data)
        return MockContext(response)

    session = MagicMock()
    session.get = MagicMock(side_effect=mock_get)
    return session


@pytest.mark.asyncio
async def test_async_call_location_uses_open_meteo(hass: HomeAssistant) -> None:
    """A location argument answers from Open-Meteo for that place."""
    tool = WeatherForecastTool({}, hass)
    tool_input = llm.ToolInput(
        tool_args={"range": "today", "location": "Orlando"},
        tool_name="get_weather_forecast",
    )
    geo_data = {
        "results": [
            {
                "name": "Orlando",
                "admin1": "Florida",
                "country": "United States",
                "latitude": 28.5455,
                "longitude": -81.3392,
            }
        ]
    }
    forecast_data = {
        "current": {
            "time": "2026-05-03T12:00",
            "temperature_2m": 30.2,
            "relative_humidity_2m": 70,
            "apparent_temperature": 33.1,
            "dew_point_2m": 22.0,
            "weather_code": 2,
            "wind_speed_10m": 10.0,
            "wind_gusts_10m": 20.0,
            "surface_pressure": 1015.0,
        },
        "daily": {
            "time": ["2026-05-03", "2026-05-04"],
            "weather_code": [2, 61],
            "temperature_2m_max": [31.0, 29.0],
            "temperature_2m_min": [22.0, 21.0],
            "precipitation_probability_max": [20, 60],
            "relative_humidity_2m_mean": [70, 75],
        },
        "hourly": {
            "time": ["2026-05-03T00:00", "2026-05-03T01:00"],
            "temperature_2m": [25.0, 24.5],
            "precipitation_probability": [10, 20],
            "relative_humidity_2m": [80, 82],
        },
    }
    session = _open_meteo_session(geo_data, forecast_data)

    with patch(
        "custom_components.llm_intents.weather.async_get_clientsession",
        return_value=session,
    ):
        result = await tool.async_call(
            hass,
            tool_input,
            MagicMock(spec=llm.LLMContext),
        )

    assert isinstance(result, ToolResult)
    text = result.data["result"]
    assert text.startswith(
        "Weather for Orlando, Florida, United States (source: Open-Meteo, CC BY 4.0)"
    )
    assert "- Time: current" in text
    assert "Temperature: 30°C (86°F)" in text
    assert "Temperature: 22°C (72°F) - 31°C (88°F)" in text
    assert "Humidity: 70%" in text
    assert "Today (Sunday)" in text
    assert "- Time: 12am" in text
    assert "Humidity: 80%" in text


@pytest.mark.asyncio
async def test_async_call_location_unknown_place(hass: HomeAssistant) -> None:
    """An unresolvable location surfaces a clear error."""
    tool = WeatherForecastTool({}, hass)
    tool_input = llm.ToolInput(
        tool_args={"range": "today", "location": "Nowhereville"},
        tool_name="get_weather_forecast",
    )
    session = _open_meteo_session({"results": []})

    with patch(
        "custom_components.llm_intents.weather.async_get_clientsession",
        return_value=session,
    ):
        result = await tool.async_call(
            hass,
            tool_input,
            MagicMock(spec=llm.LLMContext),
        )

    assert isinstance(result, ToolResult)
    assert result.data == {
        "error": (
            "Error retrieving weather for Nowhereville: "
            "Could not find a place named 'Nowhereville'"
        )
    }
