import numpy as np
import pandas as pd
import pytest

from dku_config.decomposition_config import DecompositionConfig
from dku_timeseries.decomposition import TimeseriesDecomposition
from timeseries_preparation.preparation import TimeseriesPreparator


class MockResults(object):
    def __init__(self, size, counter=1):
        time_index = pd.date_range("1-1-1959", periods=size, freq="M")
        self.trend = pd.Series(np.ones(size) * counter, index=time_index)
        self.seasonal = pd.Series(2 * np.ones(size) * counter, index=time_index)
        self.resid = pd.Series(3 * np.ones(size) * counter, index=time_index)


class MockDecomposition(TimeseriesDecomposition):
    def __init__(self, dku_config):
        super().__init__(dku_config)
        self.counter = 1

    def _decompose(self, ts):
        size = ts.shape[0]
        if self.dku_config.long_format:
            results = MockResults(size, self.counter)
            self.counter += 1
            return results
        else:
            return MockResults(size)


@pytest.fixture
def input_df():
    co2 = [315.58, 316.39, 316.79]
    df = pd.DataFrame.from_dict(
        {"value1": co2, "value2": co2, "date": pd.date_range("1-1-1959", periods=len(co2), freq="M")})
    return df


@pytest.fixture
def long_df():
    co2 = [315.58, 316.39, 316.79, 316.2]
    country = [0, 0, 1, 1]
    time_index = pd.date_range("1-1-1959", periods=2, freq="M").append(pd.date_range("1-1-1959", periods=2, freq="M"))
    df = pd.DataFrame.from_dict(
        {"value1": co2, "value2": co2, "country": country, "date": time_index})
    return df

@pytest.fixture
def basic_dku_config():
    input_dataset_columns = ["value1", "value2", "country", "date"]
    dku_config = DecompositionConfig()
    config = {"transformation_type": "seasonal_decomposition", "time_decomposition_method": "STL",
              "frequency_unit": "M", "time_column": "date", "target_columns": ["value1", "value2"],
              "long_format": False, "decomposition_model": "multiplicative", "seasonal_stl": 13, "expert": False}
    dku_config.add_parameters(config, input_dataset_columns)
    return dku_config


class TestDecomposition:
    def test_single_target(self, basic_dku_config, input_df):
        basic_dku_config.target_columns = ["value1"]
        timeseries_preparator = TimeseriesPreparator(
            time_column_name=basic_dku_config.time_column,
            frequency=basic_dku_config.frequency,
            target_columns_names=basic_dku_config.target_columns,
            timeseries_identifiers_names=basic_dku_config.timeseries_identifiers
        )
        df_prepared = timeseries_preparator.prepare_timeseries_dataframe(input_df)
        decomposition = MockDecomposition(basic_dku_config)
        df_results = decomposition.fit(df_prepared)
        size = df_prepared.shape[0]

        assert np.array_equal(df_results["value1_trend"], np.ones(size))
        assert np.array_equal(df_results["value1_seasonal"], 2 * np.ones(size))
        assert np.array_equal(df_results["value1_residuals"], 3 * np.ones(size))

    def test_multiple_targets(self, basic_dku_config, input_df):
        timeseries_preparator = TimeseriesPreparator(
            time_column_name=basic_dku_config.time_column,
            frequency=basic_dku_config.frequency,
            target_columns_names=basic_dku_config.target_columns,
            timeseries_identifiers_names=basic_dku_config.timeseries_identifiers
        )
        basic_df_prepared = timeseries_preparator.prepare_timeseries_dataframe(input_df)
        decomposition = MockDecomposition(basic_dku_config)
        df_results = decomposition.fit(basic_df_prepared)
        size = basic_df_prepared.shape[0]

        assert np.array_equal(df_results["value1_trend"], np.ones(size))
        assert np.array_equal(df_results["value2_trend"], np.ones(size))

        assert np.array_equal(df_results["value1_seasonal"], 2 * np.ones(size))
        assert np.array_equal(df_results["value2_seasonal"], 2 * np.ones(size))

        assert np.array_equal(df_results["value1_residuals"], 3 * np.ones(size))
        assert np.array_equal(df_results["value2_residuals"], 3 * np.ones(size))

    def test_long_format(self, basic_dku_config, long_df):
        basic_dku_config.long_format = True
        basic_dku_config.timeseries_identifiers = ["country"]
        timeseries_preparator = TimeseriesPreparator(
            time_column_name=basic_dku_config.time_column,
            frequency=basic_dku_config.frequency,
            target_columns_names=basic_dku_config.target_columns,
            timeseries_identifiers_names=basic_dku_config.timeseries_identifiers
        )
        df_long_prepared = timeseries_preparator.prepare_timeseries_dataframe(long_df)
        decomposition = MockDecomposition(basic_dku_config)
        df_results = decomposition.fit(df_long_prepared)
        assert np.array_equal(df_results["value1_trend"], np.array([1, 1, 3, 3]))
        assert np.array_equal(df_results["value2_trend"], np.array([2, 2, 4, 4]))
        assert np.array_equal(df_results["value1_seasonal"], np.array([2, 2, 6, 6]))
        assert np.array_equal(df_results["value2_residuals"], np.array([6, 6, 12, 12]))

    def test_collision(self, basic_dku_config, input_df):
        basic_dku_config.target_columns = ["value1"]
        input_df = input_df.rename(columns={"value2": "value1_trend"})
        timeseries_preparator = TimeseriesPreparator(
            time_column_name=basic_dku_config.time_column,
            frequency=basic_dku_config.frequency,
            target_columns_names=basic_dku_config.target_columns,
            timeseries_identifiers_names=basic_dku_config.timeseries_identifiers
        )
        df_prepared = timeseries_preparator.prepare_timeseries_dataframe(input_df)
        decomposition = MockDecomposition(basic_dku_config)
        df_results = decomposition.fit(df_prepared)
        assert df_results.columns[3] == "value1_trend_0"
        assert df_results.columns[4] == "value1_seasonal"

