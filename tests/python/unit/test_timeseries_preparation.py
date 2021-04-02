import numpy as np
import pandas as pd
import pytest

from dku_config.stl_config import STLConfig
from timeseries_preparation.preparation import TimeseriesPreparator


@pytest.fixture
def time_column_name():
    return "date"


@pytest.fixture
def timeseries_identifiers_names():
    return ["id"]


@pytest.fixture
def basic_config(time_column_name):
    config = {"transformation_type": "seasonal_decomposition", "time_decomposition_method": "STL",
              "frequency_unit": "M", "season_length_M": 12, "time_column": time_column_name, "target_columns": ["target"],
              "long_format": False, "decomposition_model": "multiplicative", "expert": False}
    return config


def test_duplicate_dates(time_column_name, timeseries_identifiers_names, basic_config):
    df = pd.DataFrame(
        {
            "date": [
                "2021-01-01 12:12:00",
                "2021-01-01 17:35:00",
                "2021-01-02 14:55:00",
            ],
            "id": [1, 1, 1],
            "target": [1, 2, 3]
        }
    )
    dku_config = STLConfig()
    basic_config["frequency"] = "D"
    dku_config.add_parameters(basic_config, list(df.columns))
    df[time_column_name] = pd.to_datetime(df[time_column_name]).dt.tz_localize(tz=None)
    preparator = TimeseriesPreparator(dku_config)
    with pytest.raises(ValueError):
        _ = preparator._truncate_dates(df)


def test_minutes_truncation(time_column_name, basic_config):
    df = pd.DataFrame(
        {
            "date": [
                "2021-01-01 12:17:42",
                "2021-01-01 12:30:00",
                "2021-01-01 12:46:00",
            ],
            "id": [1, 1, 1],
            "target": [1, 2, 3]
        }
    )
    dku_config = STLConfig()
    basic_config["frequency_step_minutes"] = "15"
    basic_config["frequency_unit"] = "min"
    basic_config["season_length_min"] = 4
    dku_config.add_parameters(basic_config, list(df.columns))
    df[time_column_name] = pd.to_datetime(df[time_column_name]).dt.tz_localize(tz=None)
    preparator = TimeseriesPreparator(dku_config)
    dataframe_prepared = preparator._truncate_dates(df)
    dataframe_prepared = preparator._sort(dataframe_prepared)
    preparator._check_regular_frequency(dataframe_prepared)

    assert dataframe_prepared[time_column_name][0] == pd.Timestamp("2021-01-01  12:15:00")
    assert dataframe_prepared[time_column_name][2] == pd.Timestamp("2021-01-01 12:45:00")


def test_hour_truncation(time_column_name, timeseries_identifiers_names, basic_config):
    df = pd.DataFrame(
        {
            "date": [
                "2020-01-07 12:12:00",
                "2020-01-07 17:35:00",
                "2020-01-07 14:55:00",
                "2020-01-07 18:06:00",
                "2020-01-08 04:40:00",
                "2020-01-08 06:13:00",
                "2020-01-08 03:23:00",
            ],
            "id": [1, 1, 1, 1, 2, 2, 2],
            "target": [1, 2, 3, 4, 5, 6, 7]
        }
    )
    df[time_column_name] = pd.to_datetime(df[time_column_name]).dt.tz_localize(tz=None)
    dku_config = STLConfig()
    basic_config["frequency_step_hours"] = "2"
    basic_config["frequency_unit"] = "H"
    basic_config["long_format"] = True
    basic_config["timeseries_identifiers"] = timeseries_identifiers_names
    dku_config.add_parameters(basic_config, list(df.columns))
    preparator = TimeseriesPreparator(dku_config, max_timeseries_length=2)
    dataframe_prepared = preparator._truncate_dates(df)
    dataframe_prepared = preparator._sort(dataframe_prepared)
    preparator._check_regular_frequency(dataframe_prepared)
    dataframe_prepared = preparator._keep_last_dates(dataframe_prepared)
    assert dataframe_prepared[time_column_name][0] == pd.Timestamp("2020-01-07 16:00:00")
    assert dataframe_prepared[time_column_name][3] == pd.Timestamp("2020-01-08 06:00:00")


def test_day_truncation(time_column_name, timeseries_identifiers_names, basic_config):
    df = pd.DataFrame(
        {
            "date": [
                "2021-01-01 12:17:42",
                "2021-01-02 00:00:00",
                "2021-01-03 12:46:00",
            ],
            "id": [1, 1, 1],
            "target": [1, 2, 3]
        }
    )
    dku_config = STLConfig()
    basic_config["frequency_unit"] = "D"
    basic_config["long_format"] = True
    basic_config["timeseries_identifiers"] = timeseries_identifiers_names
    dku_config.add_parameters(basic_config, list(df.columns))
    preparator = TimeseriesPreparator(dku_config)
    df[time_column_name] = pd.to_datetime(df[time_column_name]).dt.tz_localize(tz=None)
    dataframe_prepared = preparator._truncate_dates(df)
    dataframe_prepared = preparator._sort(dataframe_prepared)
    preparator._check_regular_frequency(dataframe_prepared)

    assert dataframe_prepared[time_column_name][0] == pd.Timestamp("2021-01-01")
    assert dataframe_prepared[time_column_name][2] == pd.Timestamp("2021-01-03")


def test_business_day_truncation(time_column_name, timeseries_identifiers_names, basic_config):
    df = pd.DataFrame(
        {
            "date": [
                "2021-01-04 12:17:42",
                "2021-01-05 00:00:00",
                "2021-01-06 12:46:00",
            ],
            "id": [1, 1, 1],
            "target": [1, 2, 3]
        }
    )
    dku_config = STLConfig()
    basic_config["frequency_unit"] = "B"
    basic_config["long_format"] = True
    basic_config["timeseries_identifiers"] = timeseries_identifiers_names
    dku_config.add_parameters(basic_config, list(df.columns))
    preparator = TimeseriesPreparator(dku_config)
    df[time_column_name] = pd.to_datetime(df[time_column_name]).dt.tz_localize(tz=None)
    dataframe_prepared = preparator._truncate_dates(df)
    dataframe_prepared = preparator._sort(dataframe_prepared)
    preparator._check_regular_frequency(dataframe_prepared)

    assert dataframe_prepared[time_column_name][0] == pd.Timestamp("2021-01-04")
    assert dataframe_prepared[time_column_name][2] == pd.Timestamp("2021-01-06")


def test_week_sunday_truncation(time_column_name, timeseries_identifiers_names, basic_config):
    df = pd.DataFrame(
        {
            "date": [
                "2021-01-03 12:12:00",
                "2021-01-05 17:35:00",
                "2021-01-15 14:55:00",
            ],
            "id": [1, 1, 1],
            "target": [1, 2, 3]
        }
    )
    dku_config = STLConfig()
    basic_config["frequency_unit"] = "W"
    basic_config["frequency_end_of_week"] = "SUN"
    basic_config["long_format"] = True
    basic_config["timeseries_identifiers"] = timeseries_identifiers_names
    dku_config.add_parameters(basic_config, list(df.columns))
    preparator = TimeseriesPreparator(dku_config, max_timeseries_length=2)
    df[time_column_name] = pd.to_datetime(df[time_column_name]).dt.tz_localize(tz=None)
    dataframe_prepared = preparator._truncate_dates(df)
    dataframe_prepared = preparator._sort(dataframe_prepared)
    preparator._check_regular_frequency(dataframe_prepared)

    dataframe_prepared = preparator._keep_last_dates(dataframe_prepared)
    assert dataframe_prepared[time_column_name][0] == pd.Timestamp("2021-01-10")
    assert dataframe_prepared[time_column_name][1] == pd.Timestamp("2021-01-17")


def test_quarter_truncation(time_column_name, timeseries_identifiers_names, basic_config):
    df = pd.DataFrame(
        {
            "date": [
                "2020-12-15",
                "2021-03-28",
                "2021-06-11",
            ],
            "id": [1, 1, 1],
            "target": [1, 2, 3]
        }
    )
    dku_config = STLConfig()
    basic_config["frequency_unit"] = "3M"
    basic_config["long_format"] = True
    basic_config["timeseries_identifiers"] = timeseries_identifiers_names
    dku_config.add_parameters(basic_config, list(df.columns))
    preparator = TimeseriesPreparator(dku_config)
    df[time_column_name] = pd.to_datetime(df[time_column_name]).dt.tz_localize(tz=None)
    dataframe_prepared = preparator._truncate_dates(df)
    dataframe_prepared = preparator._sort(dataframe_prepared)
    preparator._check_regular_frequency(dataframe_prepared)

    assert dataframe_prepared[time_column_name][0] == pd.Timestamp("2020-12-31")
    assert dataframe_prepared[time_column_name][2] == pd.Timestamp("2021-06-30")


def test_semester_truncation(time_column_name, timeseries_identifiers_names, basic_config):
    df = pd.DataFrame(
        {
            "date": [
                "2020-12-15",
                "2021-06-28",
                "2021-12-01",
            ],
            "id": [1, 1, 1],
            "target": [1, 2, 3]
        }
    )
    dku_config = STLConfig()
    basic_config["frequency_unit"] = "6M"
    basic_config["long_format"] = True
    basic_config["timeseries_identifiers"] = timeseries_identifiers_names
    dku_config.add_parameters(basic_config, list(df.columns))
    preparator = TimeseriesPreparator(dku_config)
    df[time_column_name] = pd.to_datetime(df[time_column_name]).dt.tz_localize(tz=None)
    dataframe_prepared = preparator._truncate_dates(df)
    dataframe_prepared = preparator._sort(dataframe_prepared)
    preparator._check_regular_frequency(dataframe_prepared)

    assert dataframe_prepared[time_column_name][0] == pd.Timestamp("2020-12-31")
    assert dataframe_prepared[time_column_name][1] == pd.Timestamp("2021-06-30")
    assert dataframe_prepared[time_column_name][2] == pd.Timestamp("2021-12-31")


def test_year_truncation(time_column_name, timeseries_identifiers_names, basic_config):
    df = pd.DataFrame(
        {
            "date": [
                "2020-12-31",
                "2021-12-15",
                "2022-12-01",
            ],
            "id": [1, 1, 1],
            "target": [1, 2, 3]
        }
    )
    dku_config = STLConfig()
    basic_config["frequency_unit"] = "12M"
    basic_config["long_format"] = True
    basic_config["timeseries_identifiers"] = timeseries_identifiers_names
    basic_config["season_length_12M"] = 4
    dku_config.add_parameters(basic_config, list(df.columns))
    preparator = TimeseriesPreparator(dku_config)
    df[time_column_name] = pd.to_datetime(df[time_column_name]).dt.tz_localize(tz=None)
    dataframe_prepared = preparator._truncate_dates(df)
    dataframe_prepared = preparator._sort(dataframe_prepared)
    preparator._check_regular_frequency(dataframe_prepared)

    assert dataframe_prepared[time_column_name][0] == pd.Timestamp("2020-12-31")
    assert dataframe_prepared[time_column_name][1] == pd.Timestamp("2021-12-31")
    assert dataframe_prepared[time_column_name][2] == pd.Timestamp("2022-12-31")


def test_target_column_preparation(time_column_name, timeseries_identifiers_names, basic_config):
    df = pd.DataFrame(
        {
            "date": [
                "2020-12-31",
                "2021-12-15",
                "2022-12-01",
            ],
            "id": [1, 1, 1],
            "target": [1, 2, 3],
            "invalid_target": ["a", "b", "c"],
            "missing_target": [1, np.nan, 2],
            "unformatted_target": ["1", "2", "3"]
        }
    )
    dku_config = STLConfig()
    basic_config["target_columns"] = ["target"]
    basic_config["frequency_unit"] = "12M"
    basic_config["season_length_12M"] = 4
    dku_config.add_parameters(basic_config, list(df.columns))
    preparator = TimeseriesPreparator(dku_config)
    df_prepared = preparator.prepare_timeseries_dataframe(df)
    assert df_prepared.loc[0,"target"] == 1

    dku_config = STLConfig()
    basic_config["target_columns"] = ["unformatted_target"]
    basic_config["frequency_unit"] = "12M"
    basic_config["season_length_12M"] = 4
    dku_config.add_parameters(basic_config, list(df.columns))
    preparator = TimeseriesPreparator(dku_config)
    df_prepared_unformatted = preparator.prepare_timeseries_dataframe(df)
    assert df_prepared_unformatted.loc[0, "unformatted_target"] == 1



