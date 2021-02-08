import numpy as np
import pandas as pd
import pytest

from dku_config.stl_config import STLConfig
from dku_timeseries.stl_decomposition import STLDecomposition
from timeseries_preparation.preparation import TimeseriesPreparator


@pytest.fixture
def input_df(data):
    df = pd.DataFrame.from_dict(
        {"value1": data, "value2": data, "date": pd.date_range("1-1-1959", periods=len(data), freq="M")})
    return df


@pytest.fixture
def dku_config():
    config = {"transformation_type": "seasonal_decomposition", "time_decomposition_method": "STL",
              "frequency_unit": "M", "time_column": "date", "target_columns": ["value1"],
              "long_format": False, "decomposition_model": "multiplicative", "seasonal_stl": 13, "expert": False}
    input_dataset_columns = ["value1", "value2", "country", "date"]
    dku_config = STLConfig()
    dku_config.add_parameters(config, input_dataset_columns)
    return dku_config


class TestSTLDecomposition:
    def test_STL_multiplicative(self, dku_config, input_df):
        timeseries_preparator = TimeseriesPreparator(
            time_column_name=dku_config.time_column,
            frequency=dku_config.frequency,
            target_columns_names=dku_config.target_columns,
            timeseries_identifiers_names=dku_config.timeseries_identifiers
        )
        df_prepared = timeseries_preparator.prepare_timeseries_dataframe(input_df)
        decomposition = STLDecomposition(dku_config)
        results = decomposition.fit(df_prepared)
        assert np.all(results["value1_seasonal"] < 3)

    def test_several_frequencies(self, dku_config):
        quarterly_prepared = df_from_freq("3M", dku_config)
        decomposition = STLDecomposition(dku_config)
        results_df = decomposition.fit(quarterly_prepared)
        assert results_df.shape == (7, 5)

        monthly_prepared = df_from_freq("M", dku_config)
        decomposition = STLDecomposition(dku_config)
        results_df = decomposition.fit(monthly_prepared)
        assert results_df.shape == (7, 5)

        weekly_prepared = df_from_freq("W", dku_config)
        decomposition = STLDecomposition(dku_config)
        results_df = decomposition.fit(weekly_prepared)
        assert results_df.shape == (7, 5)

        weekly_prepared = df_from_freq("B", dku_config)
        decomposition = STLDecomposition(dku_config)
        results_df = decomposition.fit(weekly_prepared)
        assert results_df.shape == (7, 5)

        weekly_prepared = df_from_freq("H", dku_config)
        decomposition = STLDecomposition(dku_config)
        results_df = decomposition.fit(weekly_prepared)
        assert results_df.shape == (7, 5)

        weekly_prepared = df_from_freq("6H", dku_config)
        decomposition = STLDecomposition(dku_config)
        results_df = decomposition.fit(weekly_prepared)
        assert results_df.shape == (7, 5)

        weekly_prepared = df_from_freq("D", dku_config)
        decomposition = STLDecomposition(dku_config)
        results_df = decomposition.fit(weekly_prepared)
        assert results_df.shape == (7, 5)


def df_from_freq(freq, dku_config):
    data = [315.58, 316.39, 316.79, 312.09, 321.08, 450.08, 298.79]
    dku_config.frequency = freq
    quarterly_df = pd.DataFrame.from_dict(
        {"value1": data, "date": pd.date_range("1-1-1959", periods=len(data), freq=freq)})
    timeseries_preparator = TimeseriesPreparator(
        time_column_name=dku_config.time_column,
        frequency=dku_config.frequency,
        target_columns_names=dku_config.target_columns,
        timeseries_identifiers_names=dku_config.timeseries_identifiers
    )
    return timeseries_preparator.prepare_timeseries_dataframe(quarterly_df)
