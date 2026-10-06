"""Stage 1: download raw data and save it to data/raw/.

Three sources, one CSV each:
  pm25_stations.csv  daily PM2.5 measured at each ground station (OpenAQ)
  weather.csv        daily weather for each city centre (Open-Meteo archive)
  cams_pm25.csv      daily PM2.5 *forecast* by the CAMS model (Open-Meteo), our baseline to beat

Run it with:
    uv run python -m aqi_forecast.ingest --days 90
"""

import argparse
import os
import time
from datetime import date, timedelta

import pandas as pd
import requests
from dotenv import load_dotenv

from aqi_forecast.config import CITIES, RADIUS_M, RAW_DIR, TIMEZONE

OPENAQ_URL = "https://api.openaq.org/v3"
PM25_PARAMETER_ID = 2  # OpenAQ's id for PM2.5
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
CAMS_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_VARS = [
    "temperature_2m_mean",
    "relative_humidity_2m_mean",
    "precipitation_sum",
    "wind_speed_10m_max",
    "wind_direction_10m_dominant",
]


# ---------- OpenAQ: measured PM2.5 ----------

def openaq_get(path: str, api_key: str, params: dict) -> dict:
    """GET one OpenAQ endpoint, waiting and retrying if we hit the rate limit."""
    for attempt in range(3):
        resp = requests.get(
            f"{OPENAQ_URL}{path}", headers={"X-API-Key": api_key}, params=params, timeout=30
        )
        if resp.status_code == 429:  # too many requests: wait, then retry
            time.sleep(10 * (attempt + 1))
            continue
        resp.raise_for_status()
        return resp.json()
    resp.raise_for_status()
    return resp.json()


def pm25_sensors_near(city: str, api_key: str) -> list[dict]:
    """Find every PM2.5 sensor within RADIUS_M of a city centre."""
    lat, lon = CITIES[city]
    data = openaq_get(
        "/locations",
        api_key,
        {
            "coordinates": f"{lat},{lon}",
            "radius": RADIUS_M,
            "parameters_id": PM25_PARAMETER_ID,
            "limit": 1000,
        },
    )
    return parse_pm25_sensors(data, city)


def parse_pm25_sensors(data: dict, city: str) -> list[dict]:
    """Pick out the PM2.5 sensors from an OpenAQ /locations response."""
    sensors = []
    for loc in data["results"]:
        for sensor in loc["sensors"]:
            if sensor["parameter"]["id"] == PM25_PARAMETER_ID:
                sensors.append(
                    {
                        "city": city,
                        "location_id": loc["id"],
                        "location_name": loc["name"],
                        "sensor_id": sensor["id"],
                    }
                )
    return sensors


def daily_pm25(sensor: dict, start: date, end: date, api_key: str) -> list[dict]:
    """Daily average PM2.5 for one sensor between start and end (inclusive)."""
    rows, page = [], 1
    while True:
        data = openaq_get(
            f"/sensors/{sensor['sensor_id']}/days",
            api_key,
            {
                "datetime_from": start.isoformat(),
                "datetime_to": (end + timedelta(days=1)).isoformat(),
                "limit": 1000,
                "page": page,
            },
        )
        rows += parse_daily_pm25(data, sensor)
        if len(data["results"]) < 1000:
            return rows
        page += 1


def parse_daily_pm25(data: dict, sensor: dict) -> list[dict]:
    """Turn an OpenAQ /sensors/{id}/days response into flat rows."""
    return [
        {
            **sensor,
            "date": r["period"]["datetimeFrom"]["local"][:10],
            "pm25": r["value"],
            "coverage_pct": (r.get("coverage") or {}).get("percentComplete"),
        }
        for r in data["results"]
    ]


# ---------- Open-Meteo: weather and the CAMS forecast ----------

def open_meteo_daily(url: str, city: str, start: date, end: date, params: dict) -> pd.DataFrame:
    lat, lon = CITIES[city]
    resp = requests.get(
        url,
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "timezone": TIMEZONE,
            **params,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return parse_open_meteo(resp.json(), city)


def parse_open_meteo(data: dict, city: str) -> pd.DataFrame:
    """Open-Meteo returns columns as lists under 'daily' or 'hourly'; make a table."""
    key = "daily" if "daily" in data else "hourly"
    df = pd.DataFrame(data[key]).rename(columns={"time": "date"})
    df.insert(0, "city", city)
    return df


def cams_daily_pm25(city: str, start: date, end: date) -> pd.DataFrame:
    """CAMS only gives hourly PM2.5, so we average it per day."""
    hourly = open_meteo_daily(CAMS_URL, city, start, end, {"hourly": "pm2_5"})
    hourly["date"] = hourly["date"].str[:10]
    return (
        hourly.groupby(["city", "date"], as_index=False)["pm2_5"]
        .mean()
        .rename(columns={"pm2_5": "cams_pm25"})
    )


# ---------- Put it together ----------

def main() -> None:
    parser = argparse.ArgumentParser(description="Download raw AQI and weather data.")
    parser.add_argument("--days", type=int, default=90, help="how many past days to fetch")
    args = parser.parse_args()

    load_dotenv()  # reads OPENAQ_API_KEY from the .env file
    api_key = os.environ.get("OPENAQ_API_KEY")
    if not api_key:
        raise SystemExit("OPENAQ_API_KEY is missing. Copy .env.example to .env and add your key.")

    end = date.today() - timedelta(days=1)  # yesterday is the last complete day
    start = end - timedelta(days=args.days - 1)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Fetching {start} to {end} for {', '.join(CITIES)}")

    pm25_rows, weather, cams = [], [], []
    for city in CITIES:
        sensors = pm25_sensors_near(city, api_key)
        print(f"  {city}: {len(sensors)} PM2.5 sensors")
        for sensor in sensors:
            pm25_rows += daily_pm25(sensor, start, end, api_key)
        weather.append(
            open_meteo_daily(WEATHER_URL, city, start, end, {"daily": ",".join(WEATHER_VARS)})
        )
        cams.append(cams_daily_pm25(city, start, end))  # CAMS history starts Aug 2022

    outputs = {
        "pm25_stations.csv": pd.DataFrame(pm25_rows),
        "weather.csv": pd.concat(weather),
        "cams_pm25.csv": pd.concat(cams),
    }
    for name, df in outputs.items():
        df.to_csv(RAW_DIR / name, index=False)
        print(f"Wrote {len(df):>6} rows to data/raw/{name}")


if __name__ == "__main__":
    main()
