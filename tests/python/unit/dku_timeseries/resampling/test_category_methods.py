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
def config():
    config = {u'clip_end': 0, u'constant_value': 0, u'extrapolation_method': u'clip', u'shift': 0, u'time_unit_end_of_week': u'SUN',
              u'datetime_column': u'Date', u'advanced_activated': True, u"groupby_columns": ["categorical"], u'time_unit': u'weeks', u'clip_start': 0,
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

