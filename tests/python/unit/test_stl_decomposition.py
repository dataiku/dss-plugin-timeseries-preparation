import numpy as np
import pandas as pd
import pytest

from dku_config.stl_config import STLConfig
from dku_timeseries.stl_decomposition import STLDecomposition
from timeseries_preparation.preparation import TimeseriesPreparator


@pytest.fixture
def data():
    return [315.58, 316.39, 316.79, 312.09, 321.08, 450.08, 298.79]


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

    def test_several_frequencies(self):
        quarterly_config = config_from_freq("3M")
        quarterly_prepared = df_from_freq("3M", quarterly_config)
        decomposition = STLDecomposition(quarterly_config)
        results_df = decomposition.fit(quarterly_prepared)
        assert results_df.shape == (7, 5)

        semiannual_config = config_from_freq("6M")
        semiannual_prepared = df_from_freq("6M", semiannual_config)
        decomposition = STLDecomposition(semiannual_config)
        results_df = decomposition.fit(semiannual_prepared)
        assert results_df.shape == (7, 5)

        monthly_config = config_from_freq("M")
        monthly_prepared = df_from_freq("M", monthly_config)
        decomposition = STLDecomposition(monthly_config)
        results_df = decomposition.fit(monthly_prepared)
        assert results_df.shape == (7, 5)

        weekly_config = config_from_freq("W")
        weekly_prepared = df_from_freq("W", weekly_config)
        decomposition = STLDecomposition(weekly_config)
        results_df = decomposition.fit(weekly_prepared)
        assert results_df.shape == (7, 5)

        b_weekly_config = config_from_freq("B")
        b_weekly_prepared = df_from_freq("B", b_weekly_config)
        decomposition = STLDecomposition(b_weekly_config)
        results_df = decomposition.fit(b_weekly_prepared)
        assert results_df.shape == (7, 5)

        hourly_config = config_from_freq("H")
        hourly_prepared = df_from_freq("H", hourly_config)
        decomposition = STLDecomposition(hourly_config)
        results_df = decomposition.fit(hourly_prepared)
        assert results_df.shape == (7, 5)

        daily_config = config_from_freq("D")
        daily_prepared = df_from_freq("D", daily_config)
        decomposition = STLDecomposition(daily_config)
        results_df = decomposition.fit(daily_prepared)
        assert results_df.shape == (7, 5)


def df_from_freq(freq, dku_config):
    data = [315.58, 316.39, 316.79, 312.09, 321.08, 450.08, 298.79]
    df = pd.DataFrame.from_dict(
        {"value1": data, "date": pd.date_range("1-1-1959", periods=len(data), freq=freq)})
    timeseries_preparator = TimeseriesPreparator(
        time_column_name=dku_config.time_column,
        frequency=dku_config.frequency,
        target_columns_names=dku_config.target_columns,
        timeseries_identifiers_names=dku_config.timeseries_identifiers
    )
    df_prepared = timeseries_preparator.prepare_timeseries_dataframe(df)
    return df_prepared

def config_from_freq(freq):
    config = {"transformation_type": "seasonal_decomposition", "time_decomposition_method": "STL",
              "frequency_unit": freq, "time_column": "date", "target_columns": ["value1"],
              "long_format": False, "decomposition_model": "multiplicative", "seasonal_stl": 13, "expert": False}
    input_dataset_columns = ["value1", "value2", "country", "date"]
    dku_config = STLConfig()
    dku_config.add_parameters(config, input_dataset_columns)
    return dku_config
