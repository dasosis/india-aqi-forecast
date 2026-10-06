from aqi_forecast import CITIES


def test_package_imports():
    assert CITIES == ["Delhi", "Mumbai", "Bengaluru"]
