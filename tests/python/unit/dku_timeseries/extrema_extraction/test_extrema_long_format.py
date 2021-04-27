import os
import sys

import numpy as np
import pandas as pd
import pytest

plugin_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname((os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))))
sys.path.append(os.path.join(plugin_root, 'python-lib'))

from dku_timeseries import ExtremaExtractor
from recipe_config_loading import get_extrema_extraction_params


@pytest.fixture
def columns():
    class COLUMNS:
        date = "Date"
        extrema = "value1"

    return COLUMNS


@pytest.fixture
def df(columns):
    co2 = [315.58, 316.39, 316.79, 316.2]
    country = ["first", "first", "second", "second"]
    time_index = pd.date_range("1-1-1959", periods=4, freq="M")
    df = pd.DataFrame.from_dict(
        {columns.extrema: co2, "value2": co2, "country": country, columns.date: time_index})
    return df


@pytest.fixture
def long_df(columns):
    co2 = [315.58, 316.39, 316.79, 316.2, 345, 234, 100, 299]
    nan = np.ones(8) * np.nan
    country = ["first", "first", "first", "first", "second", "second", "second", "second"]
    half_nan = [np.nan, np.nan, np.nan, np.nan, 1, 2, 3, 4]
    time_index = pd.date_range("1-1-1959", periods=4, freq="D").append(pd.date_range("1-1-1959", periods=4, freq="D"))
    df = pd.DataFrame.from_dict(
        {columns.extrema: co2, "value2": co2, "country": country, "nan": nan, "half_nan": half_nan, columns.date: time_index})
    return df


@pytest.fixture
def long_df_2(columns):
    co2 = [315.58, 316.39, 316.79, 316.2, 9, 10]
    half_nan = [np.nan, np.nan, 7, 2, 1, 2]
    country = ["first", "first", "second", "second", "third", "third"]
    country_2 = ["first", "first", "second", "second", "third", "third"]
    time_index = pd.date_range("1-1-1959", periods=2, freq="M").append(pd.date_range("1-1-1959", periods=2, freq="M")).append(
        pd.date_range("1-1-1959", periods=2, freq="M"))
    df = pd.DataFrame.from_dict(
        {columns.extrema: co2, "value2": co2, "country": country, "item": country_2, "half_nan": half_nan, columns.date: time_index})
    return df


@pytest.fixture
def long_df_3(columns):
    co2 = [315.58, 316.39, 316.79, 316.2, 9, 10, 2, 3]
    country = ["first", "first", "second", "second", "third", "third", "fourth", "fourth"]
    country_2 = ["first", "first", "second", "second", "third", "third", "fourth", "fourth"]
    country_3 = ["first", "first", "second", "second", "third", "third", "fourth", "fourth"]
    time_index = pd.date_range("1-1-1959", periods=2, freq="M").append(pd.date_range("1-1-1959", periods=2, freq="M")).append(
        pd.date_range("1-1-1959", periods=2, freq="M")).append(pd.date_range("1-1-1959", periods=2, freq="M"))
    df = pd.DataFrame.from_dict(
        {columns.extrema: co2, "value2": co2, "country": country, "item": country_2, "store": country_3, columns.date: time_index})
    return df


@pytest.fixture
def long_df_4(columns):
    co2 = [315.58, 316.39, 316.79, 316.2, 9, 10, 2, 3]
    country = ["first", "first", "second", "second", "third", "third", "first", "first"]
    country_2 = ["first", "first", "second", "second", "third", "third", "second", "first"]
    country_3 = ["first", "first", "second", "second", "third", "third", "third", "fourth"]
    time_index = pd.date_range("1-1-2020", periods=2, freq="M").append(pd.date_range("1-1-2020", periods=2, freq="M")).append(
        pd.date_range("1-1-2020", periods=2, freq="M")).append(pd.date_range("1-1-2020", periods=2, freq="M"))
    df = pd.DataFrame.from_dict(
        {columns.extrema: co2, "value2": co2, "country": country, "item": country_2, "store": country_3, columns.date: time_index})
    return df


@pytest.fixture
def long_df_numerical(columns):
    co2 = [315.58, 316.39, 316.79, 316.2, 345, 234, 100, 299]
    country = [1, 1, 1, 1, 2, 2, 2, 2]
    time_index = pd.date_range("1-1-1959", periods=4, freq="D").append(pd.date_range("1-1-1959", periods=4, freq="D"))
    df = pd.DataFrame.from_dict(
        {columns.extrema: co2, "value2": co2, "country": country, columns.date: time_index})
    return df


@pytest.fixture
def recipe_config(columns):
    config = {u'window_type': u'none', u'groupby_columns': [u'country'], u'closed_option': u'left', u'window_unit': u'seconds', u'window_width': 1,
              u'causal_window': False, columns.date: u'Date', u'advanced_activated': True, u'extrema_column': columns.extrema, u'extrema_type': u'max',
              u'aggregation_types': [u'average'], u'gaussian_std': 1}
    return config


@pytest.fixture
def params(recipe_config):
    return get_extrema_extraction_params(recipe_config)


class TestExtremaLongFormat:
    def test_long_format(self, long_df, params, recipe_config, columns):
        groupby_columns = ["country"]
        extrema_extractor = ExtremaExtractor(params)
        output_df = extrema_extractor.compute(long_df, columns.date, columns.extrema, groupby_columns=groupby_columns)
        np.testing.assert_array_equal(output_df[columns.date].values, pd.DatetimeIndex(['1959-01-03T00:00:00.000000000', '1959-01-01T00:00:00.000000000']))
        np.testing.assert_array_equal(output_df.country.values, np.array(["first", "second"]))

        extrema_column = "nan"
        output_df = extrema_extractor.compute(long_df, columns.date, extrema_column, groupby_columns=groupby_columns)
        assert output_df.shape == (2, 1)

        extrema_column = "half_nan"
        output_df = extrema_extractor.compute(long_df, columns.date, extrema_column, groupby_columns=groupby_columns)
        assert output_df.shape == (2, 6)

    def test_two_identifiers(self, long_df_2, params, recipe_config, columns):
        groupby_columns = ["country", "item"]
        extrema_extractor = ExtremaExtractor(params)
        output_df = extrema_extractor.compute(long_df_2, columns.date, columns.extrema, groupby_columns=groupby_columns)
        np.testing.assert_array_equal(output_df[columns.date].values, pd.DatetimeIndex(['1959-02-28T00:00:00.000000000', '1959-01-31T00:00:00.000000000',
                                                                                        '1959-02-28T00:00:00.000000000']))
        extrema_column = "half_nan"
        output_df = extrema_extractor.compute(long_df_2, columns.date, extrema_column, groupby_columns=groupby_columns)
        np.testing.assert_array_equal(output_df.half_nan.values, np.array([np.nan, 7., 2.]))
        assert output_df.shape == (3, 6)

    def test_three_identifiers(self, long_df_3, params, recipe_config, columns):
        groupby_columns = ["country", "item", "store"]
        extrema_extractor = ExtremaExtractor(params)
        output_df = extrema_extractor.compute(long_df_3, columns.date, columns.extrema, groupby_columns=groupby_columns)
        np.testing.assert_array_equal(output_df[columns.date].values, pd.DatetimeIndex(['1959-02-28T00:00:00.000000000', '1959-02-28T00:00:00.000000000',
                                                                                        '1959-01-31T00:00:00.000000000', '1959-02-28T00:00:00.000000000']))

    def test_mix_identifiers(self, long_df_4, params, recipe_config, columns):
        groupby_columns = ["country", "item", "store"]
        extrema_extractor = ExtremaExtractor(params)
        output_df = extrema_extractor.compute(long_df_4, columns.date, columns.extrema, groupby_columns=groupby_columns)
        np.testing.assert_array_equal(output_df[columns.date].values, pd.DatetimeIndex(['2020-02-29T00:00:00.000000000', '2020-02-29T00:00:00.000000000',
                                                                                        '2020-01-31T00:00:00.000000000', '2020-01-31T00:00:00.000000000',
                                                                                        '2020-02-29T00:00:00.000000000']))

    def test_empty_identifiers(self, df, params, recipe_config, columns):
        extrema_extractor = ExtremaExtractor(params)
        output_df = extrema_extractor.compute(df, columns.date, columns.extrema, groupby_columns=[])
        assert output_df.shape == (1, 4)
        output_df = extrema_extractor.compute(df, columns.date, columns.extrema)
        assert output_df.shape == (1, 4)
        output_df = extrema_extractor.compute(df, columns.date, columns.extrema, groupby_columns=None)
        assert output_df.shape == (1, 4)

    def test_long_format_numerical(self, long_df_numerical, params, recipe_config, columns):
        groupby_columns = ["country"]
        extrema_extractor = ExtremaExtractor(params)
        output_df = extrema_extractor.compute(long_df_numerical, columns.date, columns.extrema, groupby_columns=groupby_columns)
        np.testing.assert_array_equal(output_df[columns.date].values, pd.DatetimeIndex(['1959-01-03T00:00:00.000000000', '1959-01-01T00:00:00.000000000']))
        np.testing.assert_array_equal(output_df.country.values, np.array([1, 2]))
