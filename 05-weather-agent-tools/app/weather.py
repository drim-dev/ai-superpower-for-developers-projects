"""Weather service using Open-Meteo API (free, no API key required)."""

import logging
import httpx

logger = logging.getLogger(__name__)

# Weather code descriptions from WMO
WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Foggy",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


async def geocode_location(location: str) -> dict | None:
    """Convert location name to coordinates using Open-Meteo Geocoding API."""
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": location, "count": 1, "language": "en", "format": "json"}
    logger.info(f"[WEATHER API] Geocoding: {url}?name={location}")

    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        data = response.json()

        if "results" not in data or not data["results"]:
            logger.info(f"[WEATHER API] Location not found: {location}")
            return None

        result = data["results"][0]
        geo = {
            "name": result.get("name"),
            "country": result.get("country"),
            "latitude": result.get("latitude"),
            "longitude": result.get("longitude"),
        }
        logger.info(f"[WEATHER API] Found: {geo['name']}, {geo['country']} ({geo['latitude']}, {geo['longitude']})")
        return geo


async def get_current_weather(location: str) -> dict:
    """Get current weather for a location."""
    logger.info(f"[WEATHER API] Getting current weather for: {location}")
    geo = await geocode_location(location)
    if not geo:
        return {"error": f"Location '{location}' not found"}

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": geo["latitude"],
        "longitude": geo["longitude"],
        "current": "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m",
        "timezone": "auto",
    }
    logger.info(f"[WEATHER API] Request: {url} (lat={geo['latitude']}, lon={geo['longitude']})")

    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        data = response.json()

        current = data.get("current", {})
        weather_code = current.get("weather_code", 0)

        result = {
            "location": f"{geo['name']}, {geo['country']}",
            "temperature_celsius": current.get("temperature_2m"),
            "feels_like_celsius": current.get("apparent_temperature"),
            "humidity_percent": current.get("relative_humidity_2m"),
            "wind_speed_kmh": current.get("wind_speed_10m"),
            "conditions": WEATHER_CODES.get(weather_code, "Unknown"),
        }
        logger.info(f"[WEATHER API] Response: {result['temperature_celsius']}°C, {result['conditions']}")
        return result


async def get_weather_forecast(location: str, days: int = 3) -> dict:
    """Get weather forecast for a location."""
    logger.info(f"[WEATHER API] Getting {days}-day forecast for: {location}")
    geo = await geocode_location(location)
    if not geo:
        return {"error": f"Location '{location}' not found"}

    days = min(max(days, 1), 7)  # Clamp between 1 and 7

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": geo["latitude"],
        "longitude": geo["longitude"],
        "daily": "temperature_2m_max,temperature_2m_min,weather_code,precipitation_probability_max",
        "timezone": "auto",
        "forecast_days": days,
    }
    logger.info(f"[WEATHER API] Request: {url} (lat={geo['latitude']}, lon={geo['longitude']}, days={days})")

    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        data = response.json()

        daily = data.get("daily", {})
        dates = daily.get("time", [])
        max_temps = daily.get("temperature_2m_max", [])
        min_temps = daily.get("temperature_2m_min", [])
        weather_codes = daily.get("weather_code", [])
        precip_probs = daily.get("precipitation_probability_max", [])

        forecast = []
        for i in range(len(dates)):
            forecast.append({
                "date": dates[i],
                "high_celsius": max_temps[i] if i < len(max_temps) else None,
                "low_celsius": min_temps[i] if i < len(min_temps) else None,
                "conditions": WEATHER_CODES.get(weather_codes[i] if i < len(weather_codes) else 0, "Unknown"),
                "precipitation_chance_percent": precip_probs[i] if i < len(precip_probs) else None,
            })

        result = {
            "location": f"{geo['name']}, {geo['country']}",
            "forecast": forecast,
        }
        logger.info(f"[WEATHER API] Response: {len(forecast)} days forecast received")
        return result
