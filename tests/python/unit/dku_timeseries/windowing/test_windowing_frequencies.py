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
def weekly_df():
    co2 = [4, 9, 4, 2, 5, 1]
    time_index = pd.date_range("1-1-2015", periods=6, freq="W-MON")
    df = pd.DataFrame.from_dict(
        {"value1": co2, "value2": co2, "Date": time_index})
    return df

@pytest.fixture
def annual_df():
    co2 = [4, 9, 4, 2, 5, 1]
    time_index = pd.date_range("1-1-2015", periods=6, freq="W-MON")
    df = pd.DataFrame.from_dict(
        {"value1": co2, "value2": co2, "Date": time_index})
    return df


@pytest.fixture
def recipe_config():
    config = {u'window_type': u'none', u'groupby_columns': [u'country'], u'closed_option': u'left', u'window_unit': u'months', u'window_width': 3,
              u'causal_window': False, u'datetime_column': u'Date', u'advanced_activated': False, u'aggregation_types': [u'average', 'retrieve','sum'],
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
        assert output_df.shape == (6,7)
        np.testing.assert_array_equal(output_df.value1_sum.values, np.array([np.nan, 17, 15, 11, 8, np.nan]))

    def test_monthly_causal(self, monthly_df, recipe_config):
        recipe_config["causal_window"] = True
        recipe_config["window_type"] = "triang"
        params_causal = get_params(recipe_config)
        window_aggregator = WindowAggregator(params_causal)
        datetime_column = recipe_config.get('datetime_column')
        output_df = window_aggregator.compute(monthly_df, datetime_column)
        assert output_df.shape == (6,7)
