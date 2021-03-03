import numpy as np
import pandas as pd
import pytest

from commons import get_resampling_params
from dku_timeseries import Resampler


@pytest.fixture
def long_df():
    co2 = [315.58, 316.39, 316.79, 316.2]
    country = ["first", "first", "second", "second"]
    time_index = pd.date_range("1-1-1959", periods=2, freq="M").append(pd.date_range("1-1-1959", periods=2, freq="M"))
    df = pd.DataFrame.from_dict(
        {"value1": co2, "value2": co2, "country": country, "Date": time_index})
    return df


@pytest.fixture
def long_df_2():
    co2 = [315.58, 316.39, 316.79, 316.2,9,10]
    country = ["first", "first", "second", "second","third","third"]
    country_2 = ["first", "first", "second", "second","third","third"]
    time_index = pd.date_range("1-1-1959", periods=2, freq="M").append(pd.date_range("1-1-1959", periods=2, freq="M")).append(
        pd.date_range("1-1-1959", periods=2, freq="M"))
    df = pd.DataFrame.from_dict(
        {"value1": co2, "value2": co2, "country": country, "item": country_2, "Date": time_index})
    return df


@pytest.fixture
def config():
    config = {u'clip_end': 0, u'constant_value': 0, u'extrapolation_method': u'clip', u'shift': 0, u'time_unit_end_of_week': u'SUN',
              u'datetime_column': u'Date', u'advanced_activated': True, u"groupby_columns": ["country"], u'time_unit': u'weeks', u'clip_start': 0,
              u'time_step': 2,
              u'interpolation_method': u'linear'}
    return config


class TestTransform:
    def test_long_format(self, long_df, config):
        params = get_resampling_params(config)
        resampler = Resampler(params)
        groupby_columns = ["country"]
        datetime_column = config.get('datetime_column')
        output_df = resampler.transform(long_df, datetime_column, groupby_columns=groupby_columns)
        np.testing.assert_array_equal(output_df[datetime_column].values,
                                      pd.DatetimeIndex(["1959-02-01", "1959-02-15", "1959-03-01", "1959-02-01", "1959-02-15", "1959-03-01"]))

    def test_multiple_identifiers(self, long_df_2, config):
        params = get_resampling_params(config)
        resampler = Resampler(params)
        groupby_columns = ["country","item"]
        datetime_column = config.get('datetime_column')
        output_df = resampler.transform(long_df_2, datetime_column, groupby_columns=groupby_columns)

        np.testing.assert_array_equal(output_df[datetime_column].values,
                                      pd.DatetimeIndex(["1959-02-01", "1959-02-15", "1959-03-01", "1959-02-01", "1959-02-15", "1959-03-01","1959-02-01",
        "1959-02-15", "1959-03-01"]))

