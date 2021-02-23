import numpy as np
import pandas as pd
import pytest

from dku_config.stl_config import STLConfig
from dku_timeseries.dku_decomposition.stl_decomposition import STLDecomposition
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
    config = {"transformation_type": "seasonal_decomposition", "time_decomposition_method": "STL",
              "frequency_unit": "M", "season_length_M": 12, "time_column": "date", "target_columns": ["value1"],
              "long_format": False, "decomposition_model": "multiplicative", "seasonal_stl": 13, "expert": False}
    input_dataset_columns = ["value1", "value2", "country", "date"]
    dku_config = STLConfig()
    dku_config.add_parameters(config, input_dataset_columns)
    return dku_config


class TestSTLDecomposition:
    def test_STL_multiplicative(self, dku_config, input_df):
        timeseries_preparator = TimeseriesPreparator(dku_config)
        df_prepared = timeseries_preparator.prepare_timeseries_dataframe(input_df)
        decomposition = STLDecomposition(dku_config)
        results = decomposition.fit(df_prepared)
        expected_array = [1.87080328, 1.94864198, 1.97546651, 1.47349625, 0.74672304,
                          0.6552587, 0.5000725, 0.46825876, 0.49417933, 0.86890043,
                          1.16434155, 1.63725892, 2.17084151, 2.106642, 1.95377386,
                          1.32400823, 0.92620183, 0.51855162, 0.44493062, 0.35877353,
                          0.47054681, 0.94481716, 1.30967762, 1.88240591, 2.51946737,
                          2.28270725]
        rounded_results = np.round(results["value1_seasonal"].values, 8)
        np.testing.assert_equal(rounded_results, expected_array)

    def test_STL_additive(self, dku_config, input_df):
        dku_config.model = "additive"
        timeseries_preparator = TimeseriesPreparator(dku_config)
        df_prepared = timeseries_preparator.prepare_timeseries_dataframe(input_df)
        decomposition = STLDecomposition(dku_config)
        results = decomposition.fit(df_prepared)
        expected_array = np.array([547017.83142434, 537486.72199014, 528097.19544005, 518846.2604546,
                                   509728.89885944, 500744.20335335, 491895.32401111, 483188.51149881,
                                   474630.52991124, 466256.27823943, 458496.28687542, 454985.69345522,
                                   453114.06254627, 452740.14895286, 453810.18657759, 456404.77681914,
                                   463218.97665376, 470913.29201871, 478947.25224629, 487217.22901328,
                                   495684.78235812, 504325.60788619, 513126.17464565, 522081.85641882,
                                   531195.14275262, 540473.28351069])
        rounded_results = np.round(results["value1_trend"].values, 8)
        np.testing.assert_equal(rounded_results, expected_array)

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
    timeseries_preparator = TimeseriesPreparator(dku_config)
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
