import math

import numpy as np
import pandas as pd
import pytest

from dku_timeseries import ResamplerParams, Resampler


@pytest.fixture
def df():
    co2 = [315.58, 316.39, 316.79, 316.2]
    categorical = ["first", "first", "second", "second"]
    time_index = pd.date_range("1-1-1959", periods=4, freq="M")
    df = pd.DataFrame.from_dict(
        {"value1": co2, "value2": co2, "categorical": categorical, "Date": time_index})
    return df


@pytest.fixture
def df2():
    dates = np.array(['2013-01-01T00:00:00.000000000', '2013-01-03T00:00:00.000000000',
                      '2013-01-04T00:00:00.000000000', '2013-01-05T00:00:00.000000000',
                      '2013-01-06T00:00:00.000000000'], dtype='datetime64[ns]')
    holiday = np.array([0, 1, 2, 3, 4])
    sales_1 = np.array([13, 14, 13, 10, 12])
    categorical = ["first", "first", "second", "second", "third"]
    df2 = pd.DataFrame.from_dict({"Date": dates, "holiday": holiday, "sales_1": sales_1, "categorical": categorical})
    return df2


@pytest.fixture
def long_df():
    co2 = [315.58, 316.39, 316.79, 316.2]
    country = [0, 0, 1, 1]
    categorical = ["first", "second", "third", "fourth"]
    time_index = pd.date_range("1-1-1959", periods=2, freq="M").append(pd.date_range("1-1-1959", periods=2, freq="M"))
    long_df = pd.DataFrame.from_dict(
        {"value1": co2, "value2": co2, "country": country, "categorical": categorical, "Date": time_index})
    return long_df


@pytest.fixture
def df_multiple_dates():
    dates = np.array(['2013-01-01T00:00:00.000000000', '2013-01-03T00:00:00.000000000',
                      '2013-01-04T00:00:00.000000000', '2013-01-05T00:00:00.000000000',
                      '2013-01-06T00:00:00.000000000'], dtype='datetime64[ns]')
    holiday = np.array([0, 1, 2, 3, 4])
    sales_1 = np.array([13, 14, 13, 10, 12])
    categorical = ["first", "first", "second", "second", "third"]
    df2 = pd.DataFrame.from_dict({"Date": dates, "holiday": holiday, "sales_1": sales_1, "categorical": categorical, "date2": dates})
    return df2


@pytest.fixture
def bool_df():
    dates = np.array(['2013-01-01T00:00:00.000000000', '2013-01-03T00:00:00.000000000',
                      '2013-01-04T00:00:00.000000000', '2013-01-05T00:00:00.000000000',
                      '2013-01-06T00:00:00.000000000'], dtype='datetime64[ns]')
    holiday = np.array([0, 1, 2, 3, 4])
    sales_1 = np.array([13, 14, 13, 10, 12])
    categorical = [True, True, False, False, False]
    df2 = pd.DataFrame.from_dict({"Date": dates, "holiday": holiday, "sales_1": sales_1, "categorical": categorical})
    return df2


@pytest.fixture
def config():
    config = {u'clip_end': 0, u'constant_value': 0, u'extrapolation_method': u'clip', u'shift': 0, u'time_unit_end_of_week': u'SUN',
              u'datetime_column': u'Date', u'advanced_activated': False, u"groupby_columns": [], u'time_unit': u'weeks', u'clip_start': 0,
              u'time_step': 2, "category_column_method": "empty", u'interpolation_method': u'linear'}
    return config


def get_params(config):
    def _p(param_name, default=None):
        return config.get(param_name, default)

    interpolation_method = _p('interpolation_method')
    extrapolation_method = _p('extrapolation_method')
    category_column_method = _p('category_column_method', 'empty')
    category_custom_value = _p('category_custom_value', '')
    constant_value = _p('constant_value')
    time_step = _p('time_step')
    time_unit = _p('time_unit')
    params = ResamplerParams(interpolation_method=interpolation_method,
                             extrapolation_method=extrapolation_method,
                             constant_value=constant_value,
                             category_column_method=category_column_method,
                             category_custom_value=category_custom_value,
                             time_step=time_step,
                             time_unit=time_unit)
    return params


class TestCategoryMethods:
    def test_extrapolation(self, df, config):
        params = get_params(config)
        resampler = Resampler(params)
        datetime_column = config.get('datetime_column')
        output_df = resampler.transform(df, datetime_column)
        assert output_df.loc[7, "value1"] == 316.2
        assert math.isnan(output_df.loc[7, "categorical"])

        config.pop("category_column_method")
        resampler = Resampler(params)
        datetime_column = config.get('datetime_column')
        output_df = resampler.transform(df, datetime_column)
        assert output_df.loc[7, "value1"] == 316.2
        assert math.isnan(output_df.loc[7, "categorical"])

        config["extrapolation_method"] = "none"
        params = get_params(config)
        resampler = Resampler(params)
        datetime_column = config.get('datetime_column')
        output_df = resampler.transform(df, datetime_column)
        assert math.isnan(output_df.loc[6, "categorical"])
        category_results = np.array(output_df["categorical"].values, dtype=np.float64)
        assert np.isnan(category_results).all()

        config["extrapolation_method"] = "interpolation"
        params = get_params(config)
        resampler = Resampler(params)
        datetime_column = config.get('datetime_column')
        output_df = resampler.transform(df, datetime_column)
        assert np.round(output_df.loc[7, "value1"], 3) == 316.003
        assert math.isnan(output_df.loc[7, "categorical"])

    def test_empty_filling(self, df2, config):
        config["time_unit"] = "hours"
        config["time_step"] = 12
        params = get_params(config)
        resampler = Resampler(params)
        datetime_column = config.get('datetime_column')
        output_df = resampler.transform(df2, datetime_column)
        assert output_df.loc[0, "categorical"] == "first"
        assert math.isnan(output_df.loc[1, "categorical"])
        assert math.isnan(output_df.loc[2, "categorical"])
        assert output_df.loc[6, "categorical"] == "second"
        assert math.isnan(output_df.loc[7, "categorical"])

    def test_custom_value_filling(self, df2, config):
        config["category_column_method"] = "custom"
        config["category_custom_value"] = "myvalue"
        config["time_unit"] = "hours"
        config["time_step"] = 12
        params = get_params(config)
        resampler = Resampler(params)
        datetime_column = config.get('datetime_column')
        output_df = resampler.transform(df2, datetime_column)
        assert output_df.loc[0, "categorical"] == "first"
        assert output_df.loc[1, "categorical"] == "myvalue"
        assert output_df.loc[2, "categorical"] == "myvalue"
        assert output_df.loc[6, "categorical"] == "second"
        assert output_df.loc[7, "categorical"] == "myvalue"

    def test_previous_filling(self, df2, config):
        config["category_column_method"] = "previous"
        config["time_unit"] = "hours"
        config["time_step"] = 12
        params = get_params(config)
        resampler = Resampler(params)
        datetime_column = config.get('datetime_column')
        output_df = resampler.transform(df2, datetime_column)
        np.testing.assert_array_equal(output_df.categorical.values,
                                      np.array(['first', 'first', 'first', 'first', 'first', 'first', 'second', 'second', 'second', 'second', 'third']))

    def test_next_filling(self, df2, config):
        config["category_column_method"] = "next"
        config["time_unit"] = "hours"
        config["time_step"] = 12
        params = get_params(config)
        resampler = Resampler(params)
        datetime_column = config.get('datetime_column')
        output_df = resampler.transform(df2, datetime_column)
        assert output_df.loc[0, "categorical"] == "first"
        assert output_df.loc[1, "categorical"] == "first"
        assert output_df.loc[3, "categorical"] == "first"
        assert output_df.loc[5, "categorical"] == "second"
        assert output_df.loc[9, "categorical"] == "third"

    def test_no_category_values(self, df, config):
        config["category_column_method"] = "previous"
        params = get_params(config)
        resampler = Resampler(params)
        datetime_column = config.get('datetime_column')
        output_df_first = resampler.transform(df, datetime_column)
        np.testing.assert_array_equal(output_df_first.categorical.values,
                                      np.array(['first', 'first', 'first', 'first', 'first', 'second', 'second', 'second']))

        config["category_column_method"] = "empty"
        params = get_params(config)
        resampler = Resampler(params)
        datetime_column = config.get('datetime_column')
        output_df_empty = resampler.transform(df, datetime_column)
        assert math.isnan(output_df_empty.loc[0, "categorical"])

    def test_previous_filling_long_format(self, long_df, config):
        config["category_column_method"] = "previous"
        config["time_unit"] = "weeks"
        config["time_step"] = 1
        params = get_params(config)
        resampler = Resampler(params)
        datetime_column = config.get('datetime_column')
        output_df = resampler.transform(long_df, datetime_column, groupby_columns=["country"])
        expected_dates = pd.DatetimeIndex(['1959-02-01T00:00:00.000000000', '1959-02-08T00:00:00.000000000',
                                           '1959-02-15T00:00:00.000000000', '1959-02-22T00:00:00.000000000',
                                           '1959-03-01T00:00:00.000000000', '1959-02-01T00:00:00.000000000',
                                           '1959-02-08T00:00:00.000000000', '1959-02-15T00:00:00.000000000',
                                           '1959-02-22T00:00:00.000000000', '1959-03-01T00:00:00.000000000'])
        np.testing.assert_array_equal(output_df.Date.values, expected_dates)
        expected_categorical = np.array(['first', 'first', 'first', 'first', 'second', 'third', 'third', 'third', 'third', 'fourth'])
        np.testing.assert_array_equal(output_df.categorical.values, expected_categorical)

    def test_next_filling_long_format(self, long_df, config):
        config["category_column_method"] = "next"
        config["time_unit"] = "weeks"
        config["time_step"] = 1
        params = get_params(config)
        resampler = Resampler(params)
        datetime_column = config.get('datetime_column')
        output_df = resampler.transform(long_df, datetime_column, groupby_columns=["country"])
        assert math.isnan(output_df.loc[4, "categorical"])
        assert output_df.loc[3, "categorical"] == "second"

    def test_clip_filling(self, long_df, config):
        config["category_column_method"] = "clip"
        config["time_unit"] = "weeks"
        config["time_step"] = 1
        params = get_params(config)
        resampler = Resampler(params)
        datetime_column = config.get('datetime_column')
        output_df = resampler.transform(long_df, datetime_column, groupby_columns=["country"])
        assert output_df.loc[3, "categorical"] == "first"

    def test_df_multiple_dates(self, df_multiple_dates, config):
        config["category_column_method"] = "previous"
        config["time_unit"] = "hours"
        config["time_step"] = 12
        params = get_params(config)
        resampler = Resampler(params)
        datetime_column = config.get('datetime_column')
        output_df = resampler.transform(df_multiple_dates, datetime_column)
        assert pd.isnull(output_df.loc[1, "date2"])

    def test_bool_column(self, bool_df, config):
        config["category_column_method"] = "previous"
        config["time_unit"] = "hours"
        config["time_step"] = 12
        params = get_params(config)
        resampler = Resampler(params)
        datetime_column = config.get('datetime_column')
        output_df = resampler.transform(bool_df, datetime_column)
        np.testing.assert_array_equal(output_df.categorical.values, np.array([True, True, True, True, True, True, False, False, False, False, False]))
