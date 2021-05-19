import numpy as np
import pandas as pd
import pytest

from dku_timeseries import IntervalRestrictor
from recipe_config_loading import get_interval_restriction_params


@pytest.fixture
def columns():
    class COLUMNS:
        date = "Date"
        category = "categorical"
        data = "value1"

    return COLUMNS


@pytest.fixture
def config(columns):
    config = {u'max_threshold': 400, u'min_threshold': 300, u'datetime_column': columns.date, u'advanced_activated': False, u'time_unit': u'days',
              u'min_deviation_duration_value': 0, u'value_column': columns.data, u'min_valid_values_duration_value': 0}
    return config


@pytest.fixture
def threshold_dict(config):
    min_threshold = config.get('min_threshold')
    max_threshold = config.get('max_threshold')
    value_column = config.get('value_column')
    threshold_dict = {value_column: (min_threshold, max_threshold)}
    return threshold_dict


class TestIntervalFrequencies:
    def test_day(self, config, threshold_dict, columns):
        config["time_unit"] = "days"
        params = get_interval_restriction_params(config)
        df = get_df_DST("W", columns)
        interval_restrictor = IntervalRestrictor(params)
        output_df = interval_restrictor.compute(df, columns.date, threshold_dict)
        expected_dates = pd.DatetimeIndex(['2019-02-03T00:59:00.000000000', '2019-02-10T00:59:00.000000000',
                                           '2019-02-17T00:59:00.000000000', '2019-02-24T00:59:00.000000000'])
        np.testing.assert_array_equal(expected_dates, output_df[columns.date].values)

    def test_hours(self, config, threshold_dict, columns):
        config["time_unit"] = "hours"
        params = get_interval_restriction_params(config)
        df = get_df_DST("H", columns)
        interval_restrictor = IntervalRestrictor(params)
        output_df = interval_restrictor.compute(df, columns.date, threshold_dict)
        expected_dates = pd.DatetimeIndex(['2019-01-31T00:59:00.000000000', '2019-01-31T01:59:00.000000000',
                                           '2019-01-31T02:59:00.000000000', '2019-01-31T03:59:00.000000000'])
        np.testing.assert_array_equal(expected_dates, output_df[columns.date].values)
        assert np.all(output_df["interval_id"].values == "0")

    def test_minutes(self, config, threshold_dict, columns):
        config["time_unit"] = "minutes"
        params = get_interval_restriction_params(config)
        df = get_df_DST("T", columns)
        interval_restrictor = IntervalRestrictor(params)
        output_df = interval_restrictor.compute(df, columns.date, threshold_dict)
        expected_dates = pd.DatetimeIndex(['2019-01-31T00:59:00.000000000', '2019-01-31T01:00:00.000000000',
                                           '2019-01-31T01:01:00.000000000', '2019-01-31T01:02:00.000000000'])
        np.testing.assert_array_equal(expected_dates, output_df[columns.date].values)
        assert np.all(output_df["interval_id"].values == "0")

    def test_seconds(self, config, threshold_dict, columns):
        config["time_unit"] = "seconds"
        params = get_interval_restriction_params(config)
        df = get_df_DST("S", columns)
        interval_restrictor = IntervalRestrictor(params)
        output_df = interval_restrictor.compute(df, columns.date, threshold_dict)
        expected_dates = pd.DatetimeIndex(['2019-01-31T00:59:00.000000000', '2019-01-31T00:59:01.000000000',
                                           '2019-01-31T00:59:02.000000000', '2019-01-31T00:59:03.000000000'])
        np.testing.assert_array_equal(expected_dates, output_df[columns.date].values)
        assert np.all(output_df["interval_id"].values == "0")

    def test_milliseconds(self, config, threshold_dict, columns):
        config["time_unit"] = "milliseconds"
        params = get_interval_restriction_params(config)
        df = get_df_DST("L", columns)
        interval_restrictor = IntervalRestrictor(params)
        output_df = interval_restrictor.compute(df, columns.date, threshold_dict)
        expected_dates = pd.DatetimeIndex(['2019-01-31T00:59:00.000000000', '2019-01-31T00:59:00.001000000',
                                           '2019-01-31T00:59:00.002000000', '2019-01-31T00:59:00.003000000'])
        np.testing.assert_array_equal(expected_dates, output_df[columns.date].values)
        assert np.all(output_df["interval_id"].values == "0")

    def test_microseconds(self, config, threshold_dict, columns):
        config["time_unit"] = "microseconds"
        params = get_interval_restriction_params(config)
        df = get_df_DST("U", columns)
        interval_restrictor = IntervalRestrictor(params)
        output_df = interval_restrictor.compute(df, columns.date, threshold_dict)
        expected_dates = pd.DatetimeIndex(['2019-01-31T00:59:00.000000000', '2019-01-31T00:59:00.000001000',
                                           '2019-01-31T00:59:00.000002000', '2019-01-31T00:59:00.000003000'])
        np.testing.assert_array_equal(expected_dates, output_df[columns.date].values)
        assert np.all(output_df["interval_id"].values == "0")

    def test_nanoseconds(self, config, threshold_dict, columns):
        config["time_unit"] = "nanoseconds"
        params = get_interval_restriction_params(config)
        df = get_df_DST("N", columns)
        interval_restrictor = IntervalRestrictor(params)
        output_df = interval_restrictor.compute(df, columns.date, threshold_dict)
        expected_dates = pd.DatetimeIndex(['2019-01-31T00:59:00.000000000', '2019-01-31T00:59:00.000000001',
                                           '2019-01-31T00:59:00.000000002', '2019-01-31T00:59:00.000000003'])
        np.testing.assert_array_equal(expected_dates, output_df[columns.date].values)
        assert np.all(output_df["interval_id"].values == "0")


def get_df_DST(frequency, columns):
    JUST_BEFORE_SPRING_DST = pd.Timestamp('20190131 01:59:0000000').tz_localize('CET')
    co2 = [315.58, 316.39, 316.79, 316.2, 666, 888]
    co = [315.58, 77, 316.79, 66, 666, 888]
    categorical = ["first", "first", "second", "second", "second", "second"]
    time_index = pd.date_range(JUST_BEFORE_SPRING_DST, periods=6, freq=frequency)
    df = pd.DataFrame.from_dict({columns.data: co2, "value2": co, columns.category: categorical, columns.date: time_index})
    return df
