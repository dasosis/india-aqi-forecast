"""Tests for the parsing code, using small made-up API responses (no internet needed)."""

from aqi_forecast.ingest import parse_daily_pm25, parse_open_meteo, parse_pm25_sensors

LOCATIONS_RESPONSE = {
    "meta": {"found": 1},
    "results": [
        {
            "id": 50,
            "name": "ITO, Delhi - CPCB",
            "sensors": [
                {"id": 101, "parameter": {"id": 2, "name": "pm25"}},
                {"id": 102, "parameter": {"id": 1, "name": "pm10"}},
            ],
        }
    ],
}

DAYS_RESPONSE = {
    "results": [
        {
            "value": 142.5,
            "period": {"datetimeFrom": {"local": "2026-10-01T00:00:00+05:30"}},
            "coverage": {"percentComplete": 95.8},
        }
    ]
}


def test_only_pm25_sensors_are_kept():
    sensors = parse_pm25_sensors(LOCATIONS_RESPONSE, "Delhi")
    assert sensors == [
        {"city": "Delhi", "location_id": 50, "location_name": "ITO, Delhi - CPCB", "sensor_id": 101}
    ]


def test_daily_rows_use_the_local_date():
    sensor = {"city": "Delhi", "sensor_id": 101}
    rows = parse_daily_pm25(DAYS_RESPONSE, sensor)
    assert rows == [
        {"city": "Delhi", "sensor_id": 101, "date": "2026-10-01", "pm25": 142.5, "coverage_pct": 95.8}
    ]


def test_open_meteo_lists_become_a_table():
    data = {"daily": {"time": ["2026-10-01", "2026-10-02"], "temperature_2m_mean": [30.1, 29.4]}}
    df = parse_open_meteo(data, "Mumbai")
    assert list(df.columns) == ["city", "date", "temperature_2m_mean"]
    assert df["city"].tolist() == ["Mumbai", "Mumbai"]
