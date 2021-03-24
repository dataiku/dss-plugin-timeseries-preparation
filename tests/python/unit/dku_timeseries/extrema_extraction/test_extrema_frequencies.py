import pandas as pd
import pytest

from dku_timeseries import WindowAggregator, ExtremaExtractorParams, WindowAggregatorParams, ExtremaExtractor


@pytest.fixture
def monthly_df():
    co2 = [4, 9, 4, 2, 5, 1]
    time_index = pd.date_range("1-1-2015", periods=6, freq="M")
    df = pd.DataFrame.from_dict(
        {"value1": co2, "value2": co2, "Date": time_index})
    return df


@pytest.fixture
def recipe_config():
    config = {u'window_type': u'none', u'groupby_columns': [u'country'], u'closed_option': u'left', u'window_unit': u'months', u'window_width': 2,
              u'causal_window': False, u'datetime_column': u'Date', u'advanced_activated': True, u'extrema_column': u'value1', u'extrema_type': u'max',
              u'aggregation_types': [u'average'], u'gaussian_std': 1}
    return config


@pytest.fixture
def params(recipe_config):
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
    extrema_type = _p('extrema_type')
    aggregation_types = _p('aggregation_types') + ['retrieve']

    window_params = WindowAggregatorParams(window_unit=window_unit,
                                           window_width=window_width,
                                           window_type=window_type,
                                           gaussian_std=gaussian_std,
                                           closed_option=closed_option,
                                           causal_window=causal_window,
                                           aggregation_types=aggregation_types)

    window_aggregator = WindowAggregator(window_params)
    params = ExtremaExtractorParams(window_aggregator=window_aggregator, extrema_type=extrema_type)
    params.check()
    return params


class TestExtremaFrequencies:
    def test_monthly_frequency(self, monthly_df, params, recipe_config):
        datetime_column = recipe_config.get('datetime_column')
        extrema_column = "value1"
        extrema_extractor = ExtremaExtractor(params)
        output_df = extrema_extractor.compute(monthly_df, datetime_column, extrema_column)
        assert output_df.loc[0, "value1_avg"] == 6.5
        assert output_df.shape == (1, 5)
