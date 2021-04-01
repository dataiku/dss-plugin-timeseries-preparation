import numpy as np
import pandas as pd
import pytest

from dku_timeseries import WindowAggregatorParams, WindowAggregator


@pytest.fixture
def monthly_df():
    co2 = [4, 9, 4, 2, 5, 1]
    time_index = pd.date_range("1-1-2015", periods=6, freq="M")
    df = pd.DataFrame.from_dict(
        {"value1": co2, "value2": co2, "Date": time_index})
    return df


@pytest.fixture
def monthly_start_df():
    co2 = [4, 9, 4, 2, 5, 1]
    time_index = pd.date_range("1-1-2015", periods=6, freq="MS")
    df = pd.DataFrame.from_dict(
        {"value1": co2, "value2": co2, "Date": time_index})
    return df


@pytest.fixture
def weekly_df():
    co2 = [4, 9, 4, 2, 5, 1, 2, 3, 4]
    time_index = pd.date_range("1-1-2015", periods=9, freq="2W-MON")
    df = pd.DataFrame.from_dict(
        {"value1": co2, "Date": time_index})
    return df


@pytest.fixture
def annual_df():
    co2 = [4, 9, 4, 2, 5, 1]
    time_index = pd.date_range("1-1-2015", periods=6, freq="Y")
    df = pd.DataFrame.from_dict(
        {"value1": co2, "Date": time_index})
    return df


@pytest.fixture
def annual_start_df():
    co2 = [4, 9, 4, 2, 5, 1]
    time_index = pd.date_range("1-1-2015", periods=6, freq="YS")
    df = pd.DataFrame.from_dict(
        {"value1": co2, "Date": time_index})
    return df


@pytest.fixture
def recipe_config():
    config = {u'window_type': u'none', u'groupby_columns': [u'country'], u'closed_option': u'left', u'window_unit': u'months', u'window_width': 3,
              u'causal_window': False, u'datetime_column': u'Date', u'advanced_activated': False, u'aggregation_types': [u'average', 'retrieve', 'sum'],
              u'gaussian_std': 1}
    return config


def get_params(recipe_config):
    def _p(param_name, default=None):
        return recipe_config.get(param_name, default)

    causal_window = _p('causal_window')
    window_unit = _p('window_unit')
    window_width = int(_p('window_width'))
    if _p('window_type') == 'none':
        window_type = None
    else:
        window_type = _p('window_type')

    if window_type == 'gaussian':
        gaussian_std = _p('gaussian_std')
    else:
        gaussian_std = None

    closed_option = _p('closed_option')
    aggregation_types = _p('aggregation_types')

    params = WindowAggregatorParams(window_unit=window_unit,
                                    window_width=window_width,
                                    window_type=window_type,
                                    gaussian_std=gaussian_std,
                                    closed_option=closed_option,
                                    causal_window=causal_window,
                                    aggregation_types=aggregation_types)

    params.check()
    return params


class TestWindowFrequencies:
    def test_monthly_no_causal(self, monthly_df, recipe_config):
        params_no_causal = get_params(recipe_config)
        window_aggregator = WindowAggregator(params_no_causal)
        datetime_column = recipe_config.get('datetime_column')
        output_df = window_aggregator.compute(monthly_df, datetime_column)
        assert output_df.shape == (6, 7)
        np.testing.assert_array_equal(output_df.value1_sum.values, np.array([np.nan, 17, 15, 11, 8, np.nan]))

    def test_monthly_causal(self, monthly_df, recipe_config):
        recipe_config["causal_window"] = True
        recipe_config["window_type"] = "triang"
        params_causal = get_params(recipe_config)
        window_aggregator = WindowAggregator(params_causal)
        datetime_column = recipe_config.get('datetime_column')
        output_df = window_aggregator.compute(monthly_df, datetime_column)
        assert output_df.shape == (6, 7)

    def test_weekly_no_causal(self, weekly_df, recipe_config):
        params_no_causal = get_params(recipe_config)
        window_aggregator = WindowAggregator(params_no_causal)
        datetime_column = recipe_config.get('datetime_column')
        output_df = window_aggregator.compute(weekly_df, datetime_column)
        assert output_df.shape == (9, 4)
        np.testing.assert_array_equal(output_df.value1_sum.values, np.array([np.nan, np.nan, np.nan, 27, 26, 21, np.nan, np.nan, np.nan]))

    def test_weekly_causal(self, weekly_df, recipe_config):
        recipe_config["causal_window"] = True
        recipe_config["window_type"] = "triang"
        params_causal = get_params(recipe_config)
        window_aggregator = WindowAggregator(params_causal)
        datetime_column = recipe_config.get('datetime_column')
        output_df = window_aggregator.compute(weekly_df, datetime_column)
        np.testing.assert_array_equal(output_df.value1_sum, np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 15.25, 13.25]))
        assert output_df.shape == (9, 4)

    def test_annual_no_causal(self, annual_df, recipe_config):
        recipe_config["window_unit"] = "years"
        params_no_causal = get_params(recipe_config)
        window_aggregator = WindowAggregator(params_no_causal)
        datetime_column = recipe_config.get('datetime_column')
        output_df = window_aggregator.compute(annual_df, datetime_column)
        assert output_df.shape == (6, 4)
        np.testing.assert_array_equal(output_df.value1_sum, np.array([np.nan, 17, 15, 11, 8, np.nan]))

    def test_annual_causal(self, annual_df, recipe_config):
        recipe_config["causal_window"] = True
        recipe_config["window_type"] = "triang"
        recipe_config["window_unit"] = "years"
        params_causal = get_params(recipe_config)
        window_aggregator = WindowAggregator(params_causal)
        datetime_column = recipe_config.get('datetime_column')
        output_df = window_aggregator.compute(annual_df, datetime_column)
        np.testing.assert_array_equal(output_df.value1_avg, np.array([np.nan, np.nan, np.nan, 6.5, 4.75, 3.25]))
        assert output_df.shape == (6, 4)

    def test_invalid_frequencies(self, annual_df, recipe_config):
        params_no_causal = get_params(recipe_config)
        window_aggregator = WindowAggregator(params_no_causal)
        datetime_column = recipe_config.get('datetime_column')
        with pytest.raises(Exception) as err:
            _ = window_aggregator.compute(annual_df, datetime_column)
        assert "smaller than the timeseries frequency" in str(err.value)

        recipe_config["causal_window"] = True
        recipe_config["window_type"] = "triang"
        params_causal = get_params(recipe_config)
        window_aggregator = WindowAggregator(params_causal)
        with pytest.raises(Exception) as err:
            _ = window_aggregator.compute(annual_df, datetime_column)
        assert "smaller than the timeseries frequency" in str(err.value)

        recipe_config["window_type"] = "none"
        params_causal = get_params(recipe_config)
        window_aggregator = WindowAggregator(params_causal)
        output_df = window_aggregator.compute(annual_df, datetime_column)
        np.testing.assert_array_equal(output_df.value1_sum, np.nan * np.ones(6))

    def test_month_start(self, monthly_start_df, recipe_config):
        recipe_config["window_width"] = 1
        recipe_config["aggregation_types"] = [u'average', 'retrieve']
        params = get_params(recipe_config)
        window_aggregator = WindowAggregator(params)
        datetime_column = recipe_config.get('datetime_column')
        output_df = window_aggregator.compute(monthly_start_df, datetime_column)
        assert output_df.shape == (6, 5)
        np.testing.assert_array_equal(output_df.Date.values, pd.DatetimeIndex(['2015-01-01T00:00:00.000000000', '2015-02-01T00:00:00.000000000',
                                                                               '2015-03-01T00:00:00.000000000', '2015-04-01T00:00:00.000000000',
                                                                               '2015-05-01T00:00:00.000000000', '2015-06-01T00:00:00.000000000']))

    def test_year_start(self, annual_start_df, recipe_config):
        recipe_config["window_unit"] = "years"
        params = get_params(recipe_config)
        window_aggregator = WindowAggregator(params)
        datetime_column = recipe_config.get('datetime_column')
        output_df = window_aggregator.compute(annual_start_df, datetime_column)
        assert output_df.shape == (6, 4)
        np.testing.assert_array_equal(output_df.Date.values, pd.DatetimeIndex(['2015-01-01T00:00:00.000000000', '2016-01-01T00:00:00.000000000',
                                                                               '2017-01-01T00:00:00.000000000', '2018-01-01T00:00:00.000000000',
                                                                               '2019-01-01T00:00:00.000000000', '2020-01-01T00:00:00.000000000']))
