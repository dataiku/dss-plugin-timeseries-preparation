import pandas as pd
import pytest

from dku_timeseries import IntervalRestrictorParams, IntervalRestrictor


@pytest.fixture
def config():
    config = {u'max_threshold': 35000, u'min_threshold': 25000, u'datetime_column': u'date', u'advanced_activated': False, u'time_unit': u'days',
              u'min_deviation_duration_value': 0, u'value_column': u'revenue', u'min_valid_values_duration_value': 0}
    return config

def get_params(config):
    def _p(param_name, default=None):
        return config.get(param_name, default)
    min_valid_values_duration_value = _p('min_valid_values_duration_value')
    min_deviation_duration_value = _p('min_deviation_duration_value')
    time_unit = _p('time_unit')

    params = IntervalRestrictorParams(min_valid_values_duration_value=min_valid_values_duration_value,
                                      max_deviation_duration_value=min_deviation_duration_value,
                                      time_unit=time_unit)

    params.check()
    return params

@pytest.fixture
def edge_df():
    time_index = pd.date_range("7-1-2020", periods=8, freq="D")
    revenue = [26000, 20000, 25000, 34000,40000, 43000, 30000, 27000]
    edge_df = pd.DataFrame.from_dict(
        {"date": time_index, "revenue": revenue})
    return edge_df

class TestEdgeCases():
    def test_zero_deviation(self, edge_df,config):
        params = get_params(config)
        interval_restrictor = IntervalRestrictor(params)
        datetime_column = config.get('datetime_column')
        value_column = config.get('value_column')
        min_threshold = config.get('min_threshold')
        max_threshold = config.get('max_threshold')
        threshold_dict = {value_column: (min_threshold, max_threshold)}
        output_df = interval_restrictor.compute(edge_df, datetime_column, threshold_dict)
        assert output_df.loc[0,"date"] == pd.Timestamp("2020-07-01")
