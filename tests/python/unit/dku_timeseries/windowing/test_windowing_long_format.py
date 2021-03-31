import numpy as np
import pandas as pd
import pytest

from dku_timeseries import WindowAggregator
from recipe_config_loading import get_windowing_params


@pytest.fixture
def df():
    co2 = [315.58, 316.39, 316.79, 316.2]
    country = ["first", "first", "second", "second"]
    time_index = pd.date_range("1-1-1959", periods=4, freq="M")
    df = pd.DataFrame.from_dict(
        {"value1": co2, "value2": co2, "country": country, "Date": time_index})
    return df


@pytest.fixture
def long_df():
    co2 = [315.58, 316.39, 316.79, 316.2, 345, 234, 100, 299]
    country = ["first", "first", "first", "first", "second", "second", "second", "second"]
    time_index = pd.date_range("1-1-1959", periods=4, freq="D").append(pd.date_range("1-1-1959", periods=4, freq="D"))
    df = pd.DataFrame.from_dict(
        {"value1": co2, "value2": co2, "country": country, "Date": time_index})
    return df


@pytest.fixture
def long_df_2():
    co2 = [315.58, 316.39, 316.79, 316.2, 9, 10]
    country = ["first", "first", "second", "second", "third", "third"]
    country_2 = ["first", "first", "second", "second", "third", "third"]
    time_index = pd.date_range("1-1-1959", periods=2, freq="M").append(pd.date_range("1-1-1959", periods=2, freq="M")).append(
        pd.date_range("1-1-1959", periods=2, freq="M"))
    df = pd.DataFrame.from_dict(
        {"value1": co2, "value2": co2, "country": country, "item": country_2, "Date": time_index})
    return df


@pytest.fixture
def long_df_3():
    co2 = [315.58, 316.39, 316.79, 316.2, 9, 10, 2, 3]
    country = ["first", "first", "second", "second", "third", "third", "fourth", "fourth"]
    country_2 = ["first", "first", "second", "second", "third", "third", "fourth", "fourth"]
    country_3 = ["first", "first", "second", "second", "third", "third", "fourth", "fourth"]
    time_index = pd.date_range("1-1-1959", periods=2, freq="M").append(pd.date_range("1-1-1959", periods=2, freq="M")).append(
        pd.date_range("1-1-1959", periods=2, freq="M")).append(pd.date_range("1-1-1959", periods=2, freq="M"))
    df = pd.DataFrame.from_dict(
        {"value1": co2, "value2": co2, "country": country, "item": country_2, "store": country_3, "Date": time_index})
    return df


@pytest.fixture
def long_df_4():
    co2 = [315.58, 316.39, 316.79, 316.2, 9, 10, 2, 3]
    country = ["first", "first", "second", "second", "third", "third", "first", "first"]
    country_2 = ["first", "first", "second", "second", "third", "third", "second", "first"]
    country_3 = ["first", "first", "second", "second", "third", "third", "third", "fourth"]
    time_index = pd.date_range("1-1-2020", periods=2, freq="M").append(pd.date_range("1-1-2020", periods=2, freq="M")).append(
        pd.date_range("1-1-2020", periods=2, freq="M")).append(pd.date_range("1-1-2020", periods=2, freq="M"))
    df = pd.DataFrame.from_dict(
        {"value1": co2, "value2": co2, "country": country, "item": country_2, "store": country_3, "Date": time_index})
    return df


@pytest.fixture
def long_df_numerical():
    co2 = [315.58, 316.39, 316.79, 316.2, 345, 234, 100, 299]
    country = [1, 1, 1, 1, 2, 2, 2, 2]
    time_index = pd.date_range("1-1-1959", periods=4, freq="D").append(pd.date_range("1-1-1959", periods=4, freq="D"))
    df = pd.DataFrame.from_dict(
        {"value1": co2, "value2": co2, "country": country, "Date": time_index})
    return df


@pytest.fixture
def recipe_config():
    config = {u'window_type': u'none', u'groupby_columns': [u'country'], u'closed_option': u'left', u'window_unit': u'days', u'window_width': 3,
              u'causal_window': True, u'datetime_column': u'Date', u'advanced_activated': True, u'aggregation_types': [u'retrieve', u'average'],
              u'gaussian_std': 1}
    return config


@pytest.fixture
def params(recipe_config):
    return get_windowing_params(recipe_config)


@pytest.fixture
def params_no_causal(recipe_config):
    recipe_config["causal_window"] = False
    return get_windowing_params(recipe_config)


class TestWindowingLongFormat:
    def test_long_format(self, long_df, params, recipe_config):
        window_aggregator = WindowAggregator(params)
        groupby_columns = ["country"]
        datetime_column = recipe_config.get('datetime_column')
        output_df = window_aggregator.compute(long_df, datetime_column, groupby_columns=groupby_columns)
        np.testing.assert_array_equal(np.round(output_df.value1_avg.values, 2), np.array([np.nan, 315.58, 315.98, 316.25, np.nan, 345.,
                                                                                          289.5, 226.33]))
        np.testing.assert_array_equal(output_df.country.values, np.array(['first', 'first', 'first', 'first', 'second', 'second', 'second', 'second']))

    def test_two_identifiers(self, long_df_2, params, recipe_config):
        window_aggregator = WindowAggregator(params)
        groupby_columns = ["country", "item"]
        datetime_column = recipe_config.get('datetime_column')
        output_df = window_aggregator.compute(long_df_2, datetime_column, groupby_columns=groupby_columns)
        np.testing.assert_array_equal(output_df[datetime_column].values,
                                      pd.DatetimeIndex(['1959-01-31T00:00:00.000000000', '1959-02-28T00:00:00.000000000',
                                                        '1959-01-31T00:00:00.000000000', '1959-02-28T00:00:00.000000000',
                                                        '1959-01-31T00:00:00.000000000', '1959-02-28T00:00:00.000000000']))

    def test_three_identifiers(self, long_df_3, params, recipe_config):
        window_aggregator = WindowAggregator(params)
        groupby_columns = ["country", "item", "store"]
        datetime_column = recipe_config.get('datetime_column')
        output_df = window_aggregator.compute(long_df_3, datetime_column, groupby_columns=groupby_columns)
        np.testing.assert_array_equal(output_df[datetime_column].values,
                                      pd.DatetimeIndex(['1959-01-31T00:00:00.000000000', '1959-02-28T00:00:00.000000000',
                                                        '1959-01-31T00:00:00.000000000', '1959-02-28T00:00:00.000000000',
                                                        '1959-01-31T00:00:00.000000000', '1959-02-28T00:00:00.000000000',
                                                        '1959-01-31T00:00:00.000000000', '1959-02-28T00:00:00.000000000']))

    def test_mix_identifiers(self, long_df_4, params, recipe_config):
        window_aggregator = WindowAggregator(params)
        groupby_columns = ["country", "item", "store"]
        datetime_column = recipe_config.get('datetime_column')
        output_df = window_aggregator.compute(long_df_4, datetime_column, groupby_columns=groupby_columns)
        expected_dates = pd.DatetimeIndex(['2020-01-31T00:00:00.000000000', '2020-02-29T00:00:00.000000000',
                                           '2020-02-29T00:00:00.000000000', '2020-01-31T00:00:00.000000000',
                                           '2020-01-31T00:00:00.000000000', '2020-02-29T00:00:00.000000000',
                                           '2020-01-31T00:00:00.000000000', '2020-02-29T00:00:00.000000000'])
        np.testing.assert_array_equal(output_df[datetime_column].values, expected_dates)

    def test_empty_identifiers(self, df, params, recipe_config):
        window_aggregator = WindowAggregator(params)
        datetime_column = recipe_config.get('datetime_column')
        output_df = window_aggregator.compute(df, datetime_column, groupby_columns=[])
        assert output_df.shape == (4, 5)
        output_df = window_aggregator.compute(df, datetime_column)
        assert output_df.shape == (4, 5)
        output_df = window_aggregator.compute(df, datetime_column, groupby_columns=None)
        assert output_df.shape == (4, 5)

    def test_long_format_no_causal(self, long_df, params_no_causal, recipe_config):
        window_aggregator = WindowAggregator(params_no_causal)
        groupby_columns = ["country"]
        datetime_column = recipe_config.get('datetime_column')
        output_df = window_aggregator.compute(long_df, datetime_column, groupby_columns=groupby_columns)

        np.testing.assert_array_equal(np.round(output_df.value1_avg.values, 2), np.array([np.nan, 316.25, 316.46, np.nan, np.nan, 226.33,
                                                                                          211., np.nan]))
        np.testing.assert_array_equal(output_df.country.values, np.array(['first', 'first', 'first', 'first', 'second', 'second', 'second', 'second']))

    def test_long_format_numerical(self, long_df_numerical, params, recipe_config):
        window_aggregator = WindowAggregator(params)
        groupby_columns = ["country"]
        datetime_column = recipe_config.get('datetime_column')
        output_df = window_aggregator.compute(long_df_numerical, datetime_column, groupby_columns=groupby_columns)
        np.testing.assert_array_equal(output_df.country.values, np.array([1, 1, 1, 1, 2, 2, 2, 2]))
