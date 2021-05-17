import numpy as np
import pandas as pd
import pytest

from dku_timeseries import Resampler
from recipe_config_loading import get_resampling_params


@pytest.fixture
def columns():
    class COLUMNS:
        date = "Date"
        category = "categorical"
        data = "value1"

    return COLUMNS


@pytest.fixture
def config(columns):
    config = {u'clip_end': 0, u'constant_value': 0, u'extrapolation_method': u'clip', u'shift': 0,
              u'datetime_column': columns.date, u'advanced_activated': False, u"groupby_columns": [], u'time_unit': u'weeks', u'clip_start': 0,
              u'time_step': 2, u'interpolation_method': u'linear'}
    return config


class TestResamplingFrequencies:
    def test_year(self, config, columns):
        config["time_unit"] = "years"
        params = get_resampling_params(config)
        resampler = Resampler(params)
        df = get_df("Y", columns)
        output_df = resampler.transform(df, columns.date)

        assert np.mean(output_df[columns.data]) == 316.19
        expected_dates = pd.DatetimeIndex(['1959-12-31T00:00:00.000000000', '1961-12-31T00:00:00.000000000',
                                           '1963-12-31T00:00:00.000000000'])
        np.testing.assert_array_equal(output_df[columns.date].values, expected_dates)

    def test_month(self, config, columns):
        config["time_unit"] = "months"
        params = get_resampling_params(config)
        resampler = Resampler(params)
        df = get_df("Y", columns)
        output_df = resampler.transform(df, columns.date)

        assert np.mean(output_df[columns.data]) == 316.32550000000003
        expected_dates = pd.DatetimeIndex(['1959-12-31T00:00:00.000000000', '1960-02-29T00:00:00.000000000',
                                           '1960-04-30T00:00:00.000000000', '1960-06-30T00:00:00.000000000',
                                           '1960-08-31T00:00:00.000000000', '1960-10-31T00:00:00.000000000',
                                           '1960-12-31T00:00:00.000000000', '1961-02-28T00:00:00.000000000',
                                           '1961-04-30T00:00:00.000000000', '1961-06-30T00:00:00.000000000',
                                           '1961-08-31T00:00:00.000000000', '1961-10-31T00:00:00.000000000',
                                           '1961-12-31T00:00:00.000000000', '1962-02-28T00:00:00.000000000',
                                           '1962-04-30T00:00:00.000000000', '1962-06-30T00:00:00.000000000',
                                           '1962-08-31T00:00:00.000000000', '1962-10-31T00:00:00.000000000',
                                           '1962-12-31T00:00:00.000000000', '1963-02-28T00:00:00.000000000'])
        np.testing.assert_array_equal(output_df[columns.date].values, expected_dates)

    def test_weeks_sunday_end(self, config, columns):
        config["time_unit"] = "weeks"
        params = get_resampling_params(config)
        resampler = Resampler(params)
        df = get_df("M", columns)
        output_df = resampler.transform(df, columns.date)

        assert np.mean(output_df[columns.data]) == 316.36625000000004
        expected_dates = pd.DatetimeIndex(['1959-02-01T00:00:00.000000000', '1959-02-15T00:00:00.000000000',
                                           '1959-03-01T00:00:00.000000000', '1959-03-15T00:00:00.000000000',
                                           '1959-03-29T00:00:00.000000000', '1959-04-12T00:00:00.000000000',
                                           '1959-04-26T00:00:00.000000000', '1959-05-10T00:00:00.000000000'])
        np.testing.assert_array_equal(output_df[columns.date].values, expected_dates)

    def test_weeks_monday_end(self, config, columns):
        config["time_unit"] = "weeks"
        config["time_unit_end_of_week"] = "MON"
        params = get_resampling_params(config)
        resampler = Resampler(params)
        df = get_df("M", columns)
        output_df = resampler.transform(df, columns.date)
        assert np.mean(output_df[columns.data]) == 316.36625000000004
        expected_dates = pd.DatetimeIndex(['1959-02-02T00:00:00.000000000', '1959-02-16T00:00:00.000000000',
                                           '1959-03-02T00:00:00.000000000', '1959-03-16T00:00:00.000000000',
                                           '1959-03-30T00:00:00.000000000', '1959-04-13T00:00:00.000000000',
                                           '1959-04-27T00:00:00.000000000', '1959-05-11T00:00:00.000000000'])
        np.testing.assert_array_equal(output_df[columns.date].values, expected_dates)

    def test_days(self, config, columns):
        config["time_unit"] = "days"
        params = get_resampling_params(config)
        resampler = Resampler(params)
        df = get_df("W-TUE", columns)
        output_df = resampler.transform(df, columns.date)
        assert np.mean(output_df[columns.data]) == 316.3254545454545
        expected_dates = pd.DatetimeIndex(['1959-01-06T00:00:00.000000000', '1959-01-08T00:00:00.000000000',
                                           '1959-01-10T00:00:00.000000000', '1959-01-12T00:00:00.000000000',
                                           '1959-01-14T00:00:00.000000000', '1959-01-16T00:00:00.000000000',
                                           '1959-01-18T00:00:00.000000000', '1959-01-20T00:00:00.000000000',
                                           '1959-01-22T00:00:00.000000000', '1959-01-24T00:00:00.000000000',
                                           '1959-01-26T00:00:00.000000000'])
        np.testing.assert_array_equal(output_df[columns.date].values, expected_dates)

    def test_days_DST(self, config, columns):
        config["time_unit"] = "days"
        params = get_resampling_params(config)
        resampler = Resampler(params)
        df_DST = get_df_DST("W-WED", columns)
        output_df = resampler.transform(df_DST, columns.date)
        assert np.mean(output_df[columns.data]) == 316.3072727272727
        expected_dates = pd.DatetimeIndex(['2019-02-05T23:00:00.000000000', '2019-02-07T23:00:00.000000000',
                                           '2019-02-09T23:00:00.000000000', '2019-02-11T23:00:00.000000000',
                                           '2019-02-13T23:00:00.000000000', '2019-02-15T23:00:00.000000000',
                                           '2019-02-17T23:00:00.000000000', '2019-02-19T23:00:00.000000000',
                                           '2019-02-21T23:00:00.000000000', '2019-02-23T23:00:00.000000000',
                                           '2019-02-25T23:00:00.000000000'])
        np.testing.assert_array_equal(output_df[columns.date].values, expected_dates)

    def test_hours_DST(self, config, columns):
        config["time_unit"] = "hours"
        params = get_resampling_params(config)
        resampler = Resampler(params)
        df_DST = get_df_DST("4H", columns)
        output_df = resampler.transform(df_DST, columns.date)
        assert np.mean(output_df[columns.data]) == 316.33428571428567
        expected_dates = pd.DatetimeIndex(['2019-01-31T01:00:00.000000000', '2019-01-31T03:00:00.000000000',
                                           '2019-01-31T05:00:00.000000000', '2019-01-31T07:00:00.000000000',
                                           '2019-01-31T09:00:00.000000000', '2019-01-31T11:00:00.000000000',
                                           '2019-01-31T13:00:00.000000000'])
        np.testing.assert_array_equal(output_df[columns.date].values, expected_dates)

    def test_minutes_DST(self, config, columns):
        config["time_unit"] = "minutes"
        config["time_step"] = 30
        params = get_resampling_params(config)
        resampler = Resampler(params)
        df_DST = get_df_DST("H", columns)
        output_df = resampler.transform(df_DST, columns.date)
        assert np.mean(output_df[columns.data]) == 316.28999999999996

        expected_dates = pd.DatetimeIndex(['2019-01-31T00:59:00.000000000', '2019-01-31T01:29:00.000000000',
                                           '2019-01-31T01:59:00.000000000', '2019-01-31T02:29:00.000000000',
                                           '2019-01-31T02:59:00.000000000', '2019-01-31T03:29:00.000000000',
                                           '2019-01-31T03:59:00.000000000'])
        np.testing.assert_array_equal(output_df[columns.date].values, expected_dates)

    def test_seconds(self, config, columns):
        config["time_unit"] = "seconds"
        config["time_step"] = 30
        params = get_resampling_params(config)
        resampler = Resampler(params)
        df_DST = get_df_DST("min", columns)
        output_df = resampler.transform(df_DST, columns.date)
        print(output_df[columns.date].values)
        assert np.mean(output_df[columns.data]) == 316.28999999999996
        expected_dates = pd.DatetimeIndex(['2019-01-31T00:59:00.000000000', '2019-01-31T00:59:30.000000000',
                                           '2019-01-31T01:00:00.000000000', '2019-01-31T01:00:30.000000000',
                                           '2019-01-31T01:01:00.000000000', '2019-01-31T01:01:30.000000000',
                                           '2019-01-31T01:02:00.000000000'])
        np.testing.assert_array_equal(output_df[columns.date].values, expected_dates)


def get_df(frequency, columns):
    co2 = [315.58, 316.39, 316.79, 316.2]
    categorical = ["first", "first", "second", "second"]
    time_index = pd.date_range("1-1-1959", periods=4, freq=frequency)
    df = pd.DataFrame.from_dict(
        {columns.data: co2, "value2": co2, columns.category: categorical, columns.date: time_index})
    return df


def get_df_DST(frequency, columns):
    JUST_BEFORE_SPRING_DST = pd.Timestamp('20190131 01:59:00').tz_localize('CET')
    co2 = [315.58, 316.39, 316.79, 316.2]
    categorical = ["first", "first", "second", "second"]
    time_index = pd.date_range(JUST_BEFORE_SPRING_DST, periods=4, freq=frequency)
    df = pd.DataFrame.from_dict(
        {columns.data: co2, "value2": co2, columns.category: categorical, columns.date: time_index})
    return df
