import numpy as np
import pandas as pd
import pytest

from dku_config.classical_config import ClassicalConfig
from dku_timeseries.classical_decomposition import ClassicalDecomposition
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


class TestClassicalDecomposition:
    def test_classical_decomposition(self, dku_config, input_df):
        dku_config.model = "additive"
        timeseries_preparator = TimeseriesPreparator(
            time_column_name=dku_config.time_column,
            frequency=dku_config.frequency,
            target_columns_names=dku_config.target_columns,
            timeseries_identifiers_names=dku_config.timeseries_identifiers
        )
        df_prepared = timeseries_preparator.prepare_timeseries_dataframe(input_df)
        decomposition = ClassicalDecomposition(dku_config)
        results = decomposition.fit(df_prepared)
        expected_array = np.array([np.nan, np.nan, np.nan, np.nan,
                                   np.nan, np.nan, 492945.25, 485033.95833333,
                                   473252.54166667, 463097.04166667, 459373.875, 457898.625,
                                   453365.95833333, 449817.33333333, 447106.33333333, 447729.54166667,
                                   451648.16666667, 459267.95833333, 472397.41666667, 490289.75,
                                   np.nan, np.nan, np.nan, np.nan,
                                   np.nan, np.nan])
        rounded_results = np.round(results["value1_trend"].values, 8)
        np.testing.assert_equal(expected_array, rounded_results)
