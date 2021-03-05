import numpy as np
import pandas as pd
import pytest

from dku_timeseries import ResamplerParams, Resampler


@pytest.fixture
def long_df_numerical():
    co2 = [315.58, 316.39, 316.79, 316.2]
    country = [0, 0, 1, 1]
    time_index = pd.date_range("1-1-1959", periods=2, freq="M").append(pd.date_range("1-1-1959", periods=2, freq="M"))
    df = pd.DataFrame.from_dict(
        {"value1": co2, "value2": co2, "country": country, "Date": time_index})
    return df


@pytest.fixture
def config():
    config = {u'clip_end': 0, u'constant_value': 0, u'extrapolation_method': u'clip', u'shift': 0, u'time_unit_end_of_week': u'SUN',
              u'datetime_column': u'Date', u'advanced_activated': True, u"groupby_columns": ["country"], u'time_unit': u'weeks', u'clip_start': 0,
              u'time_step': 2,
              u'interpolation_method': u'linear'}
    return config


@pytest.fixture
def params(config):
    def _p(param_name, default=None):
        return config.get(param_name, default)

    interpolation_method = _p('interpolation_method')
    extrapolation_method = _p('extrapolation_method')
    constant_value = _p('constant_value')
    time_step = _p('time_step')
    time_unit = _p('time_unit')

    params = ResamplerParams(interpolation_method=interpolation_method,
                             extrapolation_method=extrapolation_method,
                             constant_value=constant_value,
                             time_step=time_step,
                             time_unit=time_unit)
    return params


class TestResamplerLongFormat:
    def test_numerical_long_format(self, long_df_numerical, params, config):
        resampler = Resampler(params)
        groupby_columns = ["country"]
        datetime_column = config.get('datetime_column')
        output_df = resampler.transform(long_df_numerical, datetime_column, groupby_columns=groupby_columns)
        np.testing.assert_array_equal(output_df[datetime_column].values,
                                      pd.DatetimeIndex(["1959-02-01", "1959-02-15", "1959-03-01", "1959-02-01", "1959-02-15", "1959-03-01"]))
        np.testing.assert_array_equal(output_df["country"].values, np.array([0, 0, 0, 1, 1, 1]))
