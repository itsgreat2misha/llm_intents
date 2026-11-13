"""Google Places tool."""

import logging
import re
from http import HTTPStatus

import voluptuous as vol
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.util import dt
from homeassistant.util.json import JsonObjectType

from .base_tool import BaseTool
from .cache import InMemoryCache
from .const import (
    CONF_GOOGLE_PLACES_LATITUDE,
    CONF_GOOGLE_PLACES_LONGITUDE,
    CONF_GOOGLE_PLACES_NUM_RESULTS,
    CONF_GOOGLE_PLACES_RADIUS,
    CONF_GOOGLE_PLACES_RANKING,
    CONF_PROVIDER_API_KEYS,
    PROVIDER_GOOGLE,
    SERVICE_DEFAULTS,
)

_LOGGER = logging.getLogger(__name__)


class FindPlacesTool(BaseTool):
    """Tool for finding places."""

    name = "find_places"

    description = (
        "Use this tool to search Google Places when the user requests or infers they are after any of the following information for a business or location:\n"
        "- Address\n"
        "- Contact telephone number\n"
        "- Regular Open & Close times\n"
        "- Average user rating"
    )

    prompt_description = (
        f"Use the `{name}` tool to search for information about a business or public location:\n"
        "- This includes address, phone number, and opening hours."
    )

    response_directive = (
        "Use the search results to answer the users query.\n"
        "Focus on answering the users request directly, rather than repeating the entirety of the results to them."
    )

    parameters = vol.Schema(
        {
            vol.Required(
                "query",
                description="The place or location to search for",
            ): str,
        },
    )

    def wrap_response(self, response: dict) -> dict:
        """Wrap the response with our instructions."""
        response["instruction"] = self.response_instruction
        return response

    async def _search_openstreetmap(
        self,
        hass: HomeAssistant,
        query: str,
        latitude,
        longitude,
        num_results: int,
    ) -> list:
        """Search OpenStreetMap (Nominatim) for places, including opening hours.

        Data is ODbL (c) OpenStreetMap contributors. Nominatim's usage policy
        requires an identifying User-Agent and light, non-bulk use, which a
        voice assistant's occasional lookups satisfy.
        """
        lat = latitude if latitude is not None else hass.config.latitude
        lon = longitude if longitude is not None else hass.config.longitude

        base_params = {
            "q": query,
            "format": "jsonv2",
            "addressdetails": 1,
            "extratags": 1,
            "namedetails": 1,
            "limit": 10,
            "accept-language": "en",
        }
        session = async_get_clientsession(hass)

        async def _fetch(extra: dict) -> list:
            async with session.get(
                "https://nominatim.openstreetmap.org/search",
                params={**base_params, **extra},
                headers={
                    "User-Agent": "HomeAssistant-llm_intents (personal home automation)",
                },
            ) as resp:
                if resp.status != HTTPStatus.OK:
                    _LOGGER.warning(
                        "OpenStreetMap search returned HTTP %s",
                        resp.status,
                    )
                    return []
                return await resp.json()

        data = []
        if lat is not None and lon is not None:
            # Search the home area first (bounded): Nominatim's prominence
            # ranking otherwise buries the nearest match for chain names.
            viewbox = f"{lon - 0.5},{lat + 0.5},{lon + 0.5},{lat - 0.5}"
            data = await _fetch({"viewbox": viewbox, "bounded": 1})
        if not data:
            data = await _fetch({})

        if lat is not None and lon is not None:

            def _distance_sq(place):
                try:
                    plat = float(place["lat"])
                    plon = float(place["lon"])
                except (KeyError, TypeError, ValueError):
                    return float("inf")
                return (plat - lat) ** 2 + (plon - lon) ** 2

            data = sorted(data, key=_distance_sq)
        data = data[:num_results]

        results = []
        for place in data:
            extratags = place.get("extratags") or {}
            address = place.get("address") or {}
            street = " ".join(
                part
                for part in [address.get("house_number"), address.get("road")]
                if part
            )
            city = (
                address.get("city")
                or address.get("town")
                or address.get("village")
                or address.get("hamlet")
            )
            address_str = ", ".join(
                part
                for part in [
                    street,
                    city,
                    address.get("state"),
                    address.get("postcode"),
                ]
                if part
            )
            this_place = {
                "name": (place.get("namedetails") or {}).get("name")
                or place.get("display_name", "").split(",")[0],
                "address": address_str or place.get("display_name", ""),
                "rating": "Not rated",
                "phone": extratags.get("phone")
                or extratags.get("contact:phone")
                or "Not available",
                "source": "OpenStreetMap",
            }
            if extratags.get("opening_hours"):
                this_place["opening_hours"] = extratags["opening_hours"]
            website = extratags.get("website") or extratags.get("contact:website")
            if website:
                this_place["website"] = website
            results.append(this_place)

        return results

    async def _osm_fallback(
        self,
        hass: HomeAssistant,
        query: str,
        latitude,
        longitude,
        num_results: int,
        error: dict,
    ) -> dict:
        """Try OpenStreetMap; if it has nothing either, return the original error."""
        try:
            results = await self._search_openstreetmap(
                hass,
                query,
                latitude,
                longitude,
                num_results,
            )
        except Exception:
            _LOGGER.exception("OpenStreetMap fallback encountered an error")
            results = []
        if results:
            return {"results": results, "instruction": self.response_directive}
        return error

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> JsonObjectType:
        """Call the tool."""
        config_data = self.config

        query = tool_input.tool_args["query"]

        provider_keys = config_data.get(CONF_PROVIDER_API_KEYS) or {}
        api_key = provider_keys.get(PROVIDER_GOOGLE, "")
        num_results = int(
            config_data.get(
                CONF_GOOGLE_PLACES_NUM_RESULTS,
                SERVICE_DEFAULTS.get(CONF_GOOGLE_PLACES_NUM_RESULTS),
            ),
        )
        latitude = config_data.get(CONF_GOOGLE_PLACES_LATITUDE)
        longitude = config_data.get(CONF_GOOGLE_PLACES_LONGITUDE)
        radius = config_data.get(
            CONF_GOOGLE_PLACES_RADIUS,
            SERVICE_DEFAULTS.get(CONF_GOOGLE_PLACES_RADIUS),
        )
        rank_pref = config_data.get(
            CONF_GOOGLE_PLACES_RANKING,
            SERVICE_DEFAULTS.get(CONF_GOOGLE_PLACES_RANKING),
        ).upper()

        if not api_key:
            return await self._osm_fallback(
                hass,
                query,
                latitude,
                longitude,
                num_results,
                {"error": "Google Places API key not configured"},
            )

        try:
            session = async_get_clientsession(hass)
            params = {
                "textQuery": query,
                "pageSize": num_results,
            }

            if rank_pref != "None":
                params["rankPreference"] = rank_pref.upper()

            if latitude and longitude:
                params["locationBias"] = {
                    "circle": {
                        "center": {
                            "latitude": latitude,
                            "longitude": longitude,
                        },
                        "radius": radius * 1000,
                    },
                }

            cache = InMemoryCache()
            cached_response = cache.get(__name__, params)
            if cached_response:
                return cached_response

            field_mask = (
                "places.displayName,"
                "places.location,"
                "places.rating,"
                "places.nationalPhoneNumber,"
                "places.regularOpeningHours,"
                "places.shortFormattedAddress"
            )

            headers = {
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Goog-Api-Key": api_key,
                "X-Goog-FieldMask": field_mask,
            }

            async with session.post(
                "https://places.googleapis.com/v1/places:searchText",
                json=params,
                headers=headers,
            ) as resp:
                if resp.status == HTTPStatus.OK:
                    data = await resp.json()
                    results = []

                    for place in data.get("places", []):
                        this_place = {
                            "name": place.get("displayName", {}).get("text", None),
                            "address": place.get("shortFormattedAddress", None),
                            "rating": f"{place.get('rating')} out of 5"
                            if place.get("rating")
                            else "Not rated",
                            "phone": place.get("nationalPhoneNumber", "Not available"),
                        }

                        opening_hours = place.get("regularOpeningHours")
                        if opening_hours:
                            this_place["open_now"] = opening_hours.get("openNow", False)
                            next_closes = opening_hours.get("nextCloseTime")
                            next_opens = opening_hours.get("nextOpenTime")
                            weekday_descriptions = opening_hours.get(
                                "weekdayDescriptions",
                            )

                            if next_closes:
                                utc_time = dt.parse_datetime(next_closes)
                                local_time = dt.as_local(utc_time).strftime(
                                    "%Y-%m-%d %H:%M",
                                )
                                this_place["next_closes_at"] = local_time

                            if next_opens:
                                utc_time = dt.parse_datetime(next_opens)
                                local_time = dt.as_local(utc_time).strftime(
                                    "%Y-%m-%d %H:%M",
                                )
                                this_place["next_opens_at"] = local_time

                            if weekday_descriptions:
                                this_place["regular_open_hours"] = [
                                    re.sub(r"[\u2000-\u200F\u202F]", " ", desc)
                                    for desc in weekday_descriptions
                                ]

                        results.append(this_place)

                    if results:
                        cache.set(
                            __name__,
                            params,
                            {
                                "results": results,
                                "instructions": self.response_directive,
                            },
                        )

                    return (
                        {"results": results, "instruction": self.response_directive}
                        if results
                        else {"result": "No places found"}
                    )

                error_msg = f"Places search received a HTTP {resp.status} error from Google: {await resp.text()}"
                _LOGGER.error(error_msg)
                return await self._osm_fallback(
                    hass,
                    query,
                    latitude,
                    longitude,
                    num_results,
                    {"error": f"Places search error: {resp.status}"},
                )

        except Exception as e:
            _LOGGER.exception("Places search encountered an error")
            return await self._osm_fallback(
                hass,
                query,
                latitude,
                longitude,
                num_results,
                {"error": f"Error finding places: {e!s}"},
            )
