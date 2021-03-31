import numpy as np
import pandas as pd
import pytest

from dku_timeseries import IntervalRestrictor
from recipe_config_loading import get_interval_restriction_params


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
    co2 = [315.58, 316.39, 100, 116.2, 345, 234, 201, 100]
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
    co2 = [315.58, 316.39, 316.79, 316.2, 9, 319, 250, 300]
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
    co2 = [315.58, 316.39, 316.79, 316.2, 9, 319, 250, 300]
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
    co2 = [315.58, 316.39, 100, 116.2, 345, 234, 201, 100]
    country = [1, 1, 1, 1, 2, 2, 2, 2]
    time_index = pd.date_range("1-1-1959", periods=4, freq="D").append(pd.date_range("1-1-1959", periods=4, freq="D"))
    df = pd.DataFrame.from_dict(
        {"value1": co2, "value2": co2, "country": country, "Date": time_index})
    return df


@pytest.fixture
def recipe_config():
    config = {u'groupby_columns': [u'country'], u'max_threshold': 320, u'min_threshold': 200, u'datetime_column': u'Date', u'advanced_activated': True,
              u'time_unit': u'days', u'min_deviation_duration_value': 0, u'value_column': u'value1', u'min_valid_values_duration_value': 0}
    return config


@pytest.fixture
def threshold_dict(recipe_config):
    min_threshold = recipe_config.get('min_threshold')
    max_threshold = recipe_config.get('max_threshold')
    value_column = recipe_config.get('value_column')
    threshold_dict = {value_column: (min_threshold, max_threshold)}
    return threshold_dict


@pytest.fixture
def params(recipe_config):
    return get_interval_restriction_params(recipe_config)


class TestIntervalLongFormat:
    def test_long_format(self, long_df, params, recipe_config, threshold_dict):
        groupby_columns = ["country"]
        datetime_column = recipe_config.get('datetime_column')

        interval_restrictor = IntervalRestrictor(params)
        output_df = interval_restrictor.compute(long_df, datetime_column, threshold_dict, groupby_columns=groupby_columns)
        np.testing.assert_array_equal(output_df.Date.values, pd.DatetimeIndex(['1959-01-01T00:00:00.000000000', '1959-01-02T00:00:00.000000000',
                                                                               '1959-01-02T00:00:00.000000000', '1959-01-03T00:00:00.000000000']))

    def test_two_identifiers(self, long_df_2, params, recipe_config, threshold_dict):
        groupby_columns = ["country", "item"]
        datetime_column = recipe_config.get('datetime_column')
        interval_restrictor = IntervalRestrictor(params)
        output_df = interval_restrictor.compute(long_df_2, datetime_column, threshold_dict, groupby_columns=groupby_columns)
        np.testing.assert_array_equal(output_df.Date.values, pd.DatetimeIndex(['1959-01-31T00:00:00.000000000', '1959-02-28T00:00:00.000000000',
                                                                               '1959-01-31T00:00:00.000000000', '1959-02-28T00:00:00.000000000']))

    def test_three_identifiers(self, long_df_3, params, recipe_config, threshold_dict):
        groupby_columns = ["country", "item", "store"]
        datetime_column = recipe_config.get('datetime_column')

        interval_restrictor = IntervalRestrictor(params)
        output_df = interval_restrictor.compute(long_df_3, datetime_column, threshold_dict, groupby_columns=groupby_columns)
        np.testing.assert_array_equal(output_df.Date.values, pd.DatetimeIndex(['1959-01-31T00:00:00.000000000', '1959-02-28T00:00:00.000000000',
                                                                               '1959-01-31T00:00:00.000000000', '1959-02-28T00:00:00.000000000',
                                                                               '1959-01-31T00:00:00.000000000', '1959-02-28T00:00:00.000000000',
                                                                               '1959-02-28T00:00:00.000000000']))

    def test_mix_identifiers(self, long_df_4, params, recipe_config, threshold_dict):
        groupby_columns = ["country", "item", "store"]
        datetime_column = recipe_config.get('datetime_column')
        interval_restrictor = IntervalRestrictor(params)
        output_df = interval_restrictor.compute(long_df_4, datetime_column, threshold_dict, groupby_columns=groupby_columns)
        np.testing.assert_array_equal(output_df.Date.values, pd.DatetimeIndex(['2020-01-31T00:00:00.000000000', '2020-02-29T00:00:00.000000000',
                                                                               '2020-02-29T00:00:00.000000000', '2020-01-31T00:00:00.000000000',
                                                                               '2020-01-31T00:00:00.000000000', '2020-02-29T00:00:00.000000000',
                                                                               '2020-02-29T00:00:00.000000000']))

    def test_empty_identifiers(self, df, params, recipe_config, threshold_dict):
        datetime_column = recipe_config.get('datetime_column')
        interval_restrictor = IntervalRestrictor(params)
        output_df = interval_restrictor.compute(df, datetime_column, threshold_dict, groupby_columns=[])
        assert output_df.shape == (4, 5)
        output_df = interval_restrictor.compute(df, datetime_column, threshold_dict)
        assert output_df.shape == (4, 5)
        output_df = interval_restrictor.compute(df, datetime_column, threshold_dict, groupby_columns=None)
        assert output_df.shape == (4, 5)

    def test_long_format_numerical(self, long_df_numerical, params, recipe_config, threshold_dict):
        groupby_columns = ["country"]
        datetime_column = recipe_config.get('datetime_column')

        interval_restrictor = IntervalRestrictor(params)
        output_df = interval_restrictor.compute(long_df_numerical, datetime_column, threshold_dict, groupby_columns=groupby_columns)
        np.testing.assert_array_equal(output_df.Date.values, pd.DatetimeIndex(['1959-01-01T00:00:00.000000000', '1959-01-02T00:00:00.000000000',
                                                                               '1959-01-02T00:00:00.000000000', '1959-01-03T00:00:00.000000000']))
        np.testing.assert_array_equal(output_df.country.values, np.array([1, 1, 2, 2]))
