import pandas as pd
from script.preprocessing import fn_fare_per_mile, fn_fare_per_sec, fn_mile_per_sec


def test_fn_fare_per_mile():
    df = pd.DataFrame({
        'fare': [1, 2, 3],
        'trip_miles': [1, 0, 2]
    })
    assert (fn_fare_per_mile(df).values == [0.5, 2.0, 1.0]).all()


def test_fn_fare_per_sec():
    df = pd.DataFrame({
        'fare': [1, 2, 3],
        'trip_seconds': [1, 0, 2]
    })
    assert (fn_fare_per_sec(df).values == [0.5, 2.0, 1.0]).all()


def test_fn_miles_per_sec():
    df = pd.DataFrame({
        'trip_miles': [1, 2, 3],
        'trip_seconds': [1, 0, 2]
    })
    assert (fn_mile_per_sec(df).values == [0.5, 2.0, 1.0]).all()
