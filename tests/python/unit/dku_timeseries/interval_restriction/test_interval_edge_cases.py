import numpy as np
import pandas as pd
import pytest

from dku_timeseries import IntervalRestrictor
from recipe_config_loading import get_interval_restriction_params


@pytest.fixture
def datetime_column():
    return "date"


@pytest.fixture
def config(datetime_column):
    config = {u'max_threshold': 35000, u'min_threshold': 25000, u'datetime_column': datetime_column, u'advanced_activated': False, u'time_unit': u'days',
              u'min_deviation_duration_value': 0, u'value_column': u'revenue', u'min_valid_values_duration_value': 0}
    return config


@pytest.fixture
def threshold_dict(config):
    min_threshold = config.get('min_threshold')
    max_threshold = config.get('max_threshold')
    value_column = config.get('value_column')
    threshold_dict = {value_column: (min_threshold, max_threshold)}
    return threshold_dict


@pytest.fixture
def edge_df(datetime_column):
    time_index = pd.date_range("7-1-2020", periods=12, freq="D")
    revenue = [26000, 20000, 25000, 34000, 40000, 43000, 30000, 27000, 20000, 26000, 20000, 27000]
    edge_df = pd.DataFrame.from_dict(
        {datetime_column: time_index, "revenue": revenue})
    return edge_df


@pytest.fixture
def annual_edge_df(datetime_column):
    time_index = np.array(['2011',
                           '2012',
                           '2013',
                           '2014',
                           '2015',
                           '2016',
                           '2017',
                           '2018',
                           '2019',
                           '2020',
                           '2021',
                           '2022'])
    revenue = [26000, 20000, 25000, 34000, 40000, 43000, 30000, 27000, 20000, 26000, 20000, 27000]
    edge_df = pd.DataFrame.from_dict(
        {datetime_column: time_index, "revenue": revenue})
    return edge_df


@pytest.fixture
def edge_df_without_1st_row(datetime_column):
    time_index = pd.date_range("7-1-2020", periods=11, freq="D")
    revenue = [18000, 20000, 25000, 34000, 40000, 43000, 30000, 27000, 20000, 20000, 20000]
    edge_df = pd.DataFrame.from_dict(
        {datetime_column: time_index, "revenue": revenue})
    return edge_df


@pytest.fixture
def edge_df_segment(datetime_column):
    time_index = pd.date_range("7-1-2020", periods=12, freq="D")
    revenue = [26000, 26000, 25000, 34000, 40000, 43000, 30000, 27000, 20000, 26000, 20000, 26000]
    edge_df = pd.DataFrame.from_dict(
        {datetime_column: time_index, "revenue": revenue})
    return edge_df


class TestEdgeCases():
    def test_zero_deviation_edges(self, edge_df, config, threshold_dict, datetime_column):
        # [ch54733] - check if the recipe properly handles the first and the last rows
        params = get_interval_restriction_params(config)
        interval_restrictor = IntervalRestrictor(params)
        output_df = interval_restrictor.compute(edge_df, datetime_column, threshold_dict)
        assert len(output_df.index) == 7
        assert output_df.loc[0, datetime_column] == pd.Timestamp("2020-07-01")
        assert output_df.loc[6, datetime_column] == pd.Timestamp("2020-07-12")

    def test_zero_deviation_without_1st_row(self, edge_df_without_1st_row, config, threshold_dict, datetime_column):
        params = get_interval_restriction_params(config)
        interval_restrictor = IntervalRestrictor(params)
        output_df = interval_restrictor.compute(edge_df_without_1st_row, datetime_column, threshold_dict)
        assert output_df.loc[0, datetime_column] == pd.Timestamp("2020-07-03")
        assert output_df.loc[3, datetime_column] == pd.Timestamp("2020-07-08")
        assert len(output_df.index) == 4

    def test_segment_beginning(self, edge_df_segment, config, threshold_dict, datetime_column):
        params = get_interval_restriction_params(config)
        interval_restrictor = IntervalRestrictor(params)
        output_df = interval_restrictor.compute(edge_df_segment, datetime_column, threshold_dict)
        assert np.all(output_df.interval_id.values[:4] == "0")
        assert output_df.loc[0, datetime_column] == pd.Timestamp("2020-07-01")

    def test_zero_deviation_annual_edges(self, annual_edge_df, config, threshold_dict, datetime_column):
        params = get_interval_restriction_params(config)
        interval_restrictor = IntervalRestrictor(params)
        df_test = annual_edge_df.copy()
        df_test.loc[:, datetime_column] = pd.to_datetime(df_test[datetime_column])
        df_test = df_test.set_index(datetime_column).sort_index()
        df_initialized = interval_restrictor._initialize_edges(df_test)
        assert df_initialized.index[0] == pd.Timestamp("2010-12-31")
        assert df_initialized.index[-1] == pd.Timestamp("2022-01-02")

        output_df = interval_restrictor.compute(annual_edge_df, datetime_column, threshold_dict)
        assert len(output_df.index) == 7
        assert output_df.loc[0, datetime_column] == pd.Timestamp("2011-01-01")
        assert output_df.loc[6, datetime_column] == pd.Timestamp("2022-01-01")
