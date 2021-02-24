import numpy as np
import pandas as pd
import pytest

from dku_config.classical_config import ClassicalConfig
from dku_timeseries.dku_decomposition.classical_decomposition import ClassicalDecomposition
from timeseries_preparation.preparation import TimeseriesPreparator


@pytest.fixture
def data():
    return [855404., 912462., 870896., 640361., 319947., 276845.,
            208366., 192450., 200367., 347625., 459965., 641737.,
            833240., 744755., 755849., 511676., 359276., 202110.,
            174317., 141332., 186421., 376528., 525109., 759468.,
            1030616., 976795.]


@pytest.fixture
def input_df(data):
    df = pd.DataFrame.from_dict(
        {"value1": data, "value2": data, "date": pd.date_range("1-1-1959", periods=len(data), freq="M")})
    return df


@pytest.fixture
def dku_config():
    config = {"transformation_type": "seasonal_decomposition", "time_decomposition_method": "classical",
              "frequency_unit": "M", "season_length_M": 12, "time_column": "date", "target_columns": ["value1"],
              "long_format": False, "decomposition_model": "additive", "expert": False}
    input_dataset_columns = ["value1", "value2", "country", "date"]
    dku_config = ClassicalConfig()
    dku_config.add_parameters(config, input_dataset_columns)
    return dku_config


@pytest.fixture
def advanced_dku_config_empty():
    config = {"transformation_type": "seasonal_decomposition", "time_decomposition_method": "classical",
              "frequency_unit": "M", "season_length_M": 12, "time_column": "date", "target_columns": ["value1"],
              "long_format": False, "decomposition_model": "additive", "expert": True, "advanced_params_classical": {}}
    input_dataset_columns = ["value1", "value2", "country", "date"]
    dku_config = ClassicalConfig()
    dku_config.add_parameters(config, input_dataset_columns)
    return dku_config


@pytest.fixture
def advanced_dku_config():
    config = {"transformation_type": "seasonal_decomposition", "time_decomposition_method": "classical",
              "frequency_unit": "M", "season_length_M": 12, "time_column": "date", "target_columns": ["value1"],
              "long_format": False, "decomposition_model": "additive", "expert": True,
              "advanced_params_classical": {"filt": "[1,2,3]", "two_sided": "False", "extrapolate_trend": ""}}
    input_dataset_columns = ["value1", "value2", "country", "date"]
    dku_config = ClassicalConfig()
    dku_config.add_parameters(config, input_dataset_columns)
    return dku_config


@pytest.fixture
def expected_array():
    expected_array = np.array([np.nan, np.nan, np.nan, np.nan,
                               np.nan, np.nan, 492945.25, 485033.95833333,
                               473252.54166667, 463097.04166667, 459373.875, 457898.625,
                               453365.95833333, 449817.33333333, 447106.33333333, 447729.54166667,
                               451648.16666667, 459267.95833333, 472397.41666667, 490289.75,
                               np.nan, np.nan, np.nan, np.nan,
                               np.nan, np.nan])
    return expected_array


class TestClassicalDecomposition:
    def test_classical_decomposition(self, dku_config, input_df, expected_array):
        timeseries_preparator = TimeseriesPreparator(dku_config)
        df_prepared = timeseries_preparator.prepare_timeseries_dataframe(input_df)
        decomposition = ClassicalDecomposition(dku_config)
        results = decomposition.fit(df_prepared)
        rounded_results = np.round(results["value1_trend"].values, 8)
        np.testing.assert_equal(expected_array, rounded_results)

    def test_advanced_classical_empty_params(self, advanced_dku_config_empty, input_df, expected_array):
        timeseries_preparator = TimeseriesPreparator(advanced_dku_config_empty)
        df_prepared = timeseries_preparator.prepare_timeseries_dataframe(input_df)
        decomposition = ClassicalDecomposition(advanced_dku_config_empty)
        results = decomposition.fit(df_prepared)
        rounded_results = np.round(results["value1_trend"].values, 8)
        np.testing.assert_equal(expected_array, rounded_results)

    def test_advanced_classical(self, advanced_dku_config, input_df):
        timeseries_preparator = TimeseriesPreparator(advanced_dku_config)
        df_prepared = timeseries_preparator.prepare_timeseries_dataframe(input_df)
        decomposition = ClassicalDecomposition(advanced_dku_config)
        results = decomposition.fit(df_prepared)
        expected_trend = np.array([np.nan, np.nan, 5262032., 5119539., 4213357., 2837822, 1721897, 1439717,
                                   1210365, 1325709, 1756316, 2604542, 3496609, 4336446, 4745079, 4257639,
                                   3650175, 2455690, 1656365, 1096296, 992036, 1173366, 1837428, 2939270,
                                   4124879, 5316431])
        rounded_results = np.round(results["value1_trend"].values, 8)
        np.testing.assert_equal(expected_trend, rounded_results)
