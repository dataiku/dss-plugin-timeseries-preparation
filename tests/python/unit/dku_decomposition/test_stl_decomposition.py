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
              "long_format": False, "decomposition_model": "multiplicative", "seasonal_stl": 13, "expert": True, "stl_degree_kwargs": {},
              "stl_speed_jump_kwargs": {},
              "stl_smoothers_kwargs": {}}
    input_dataset_columns = ["value1", "value2", "country", "date"]
    dku_config = STLConfig()
    dku_config.add_parameters(config, input_dataset_columns)
    return dku_config


@pytest.fixture
def advanced_dku_config():
    config = {"transformation_type": "seasonal_decomposition", "time_decomposition_method": "STL",
              "frequency_unit": "M", "season_length_M": 12, "time_column": "date", "target_columns": ["value1"],
              "long_format": False, "decomposition_model": "additive", "seasonal_stl": 13, "expert": True,
              "stl_degree_kwargs": {"seasonal_deg": "1", "trend_deg": "0", "low_pass_deg": "1"},
              "stl_speed_jump_kwargs": {"seasonal_jump": '', "trend_jump": "12", "low_pass_jump": ''},
              "stl_smoothers_kwargs": {"trend": "13", "low_pass": ''}}
    input_dataset_columns = ["value1", "value2", "country", "date"]
    dku_config = STLConfig()
    dku_config.add_parameters(config, input_dataset_columns)
    return dku_config


@pytest.fixture
def expected_dates():
    expected = {"3M": np.array(['1959-01-31T00:00:00.000000000', '1959-04-30T00:00:00.000000000',
                                '1959-07-31T00:00:00.000000000', '1959-10-31T00:00:00.000000000',
                                '1960-01-31T00:00:00.000000000', '1960-04-30T00:00:00.000000000',
                                '1960-07-31T00:00:00.000000000'], dtype='datetime64[ns]'),
                "6M": np.array(['1959-01-31T00:00:00.000000000', '1959-07-31T00:00:00.000000000',
                                '1960-01-31T00:00:00.000000000', '1960-07-31T00:00:00.000000000',
                                '1961-01-31T00:00:00.000000000', '1961-07-31T00:00:00.000000000',
                                '1962-01-31T00:00:00.000000000'], dtype='datetime64[ns]'),
                "M": np.array(['1959-01-31T00:00:00.000000000', '1959-02-28T00:00:00.000000000',
                               '1959-03-31T00:00:00.000000000', '1959-04-30T00:00:00.000000000',
                               '1959-05-31T00:00:00.000000000', '1959-06-30T00:00:00.000000000',
                               '1959-07-31T00:00:00.000000000'], dtype='datetime64[ns]'),
                "W": np.array(['1959-01-04T00:00:00.000000000', '1959-01-11T00:00:00.000000000',
                               '1959-01-18T00:00:00.000000000', '1959-01-25T00:00:00.000000000',
                               '1959-02-01T00:00:00.000000000', '1959-02-08T00:00:00.000000000',
                               '1959-02-15T00:00:00.000000000'], dtype='datetime64[ns]'),
                "W-FRI": np.array(['1959-01-02T00:00:00.000000000', '1959-01-09T00:00:00.000000000',
                                   '1959-01-16T00:00:00.000000000', '1959-01-23T00:00:00.000000000',
                                   '1959-01-30T00:00:00.000000000', '1959-02-06T00:00:00.000000000',
                                   '1959-02-13T00:00:00.000000000'], dtype='datetime64[ns]'),
                "B": np.array(['1959-01-01T00:00:00.000000000', '1959-01-02T00:00:00.000000000',
                               '1959-01-05T00:00:00.000000000', '1959-01-06T00:00:00.000000000',
                               '1959-01-07T00:00:00.000000000', '1959-01-08T00:00:00.000000000',
                               '1959-01-09T00:00:00.000000000'], dtype='datetime64[ns]'),
                "H": np.array(['1959-01-01T00:00:00.000000000', '1959-01-01T01:00:00.000000000',
                               '1959-01-01T02:00:00.000000000', '1959-01-01T03:00:00.000000000',
                               '1959-01-01T04:00:00.000000000', '1959-01-01T05:00:00.000000000',
                               '1959-01-01T06:00:00.000000000'], dtype='datetime64[ns]'),
                "4H": np.array(['1959-01-01T00:00:00.000000000', '1959-01-01T04:00:00.000000000',
                                '1959-01-01T08:00:00.000000000', '1959-01-01T12:00:00.000000000',
                                '1959-01-01T16:00:00.000000000', '1959-01-01T20:00:00.000000000',
                                '1959-01-02T00:00:00.000000000'], dtype='datetime64[ns]'),
                "D": np.array(['1959-01-01T00:00:00.000000000', '1959-01-02T00:00:00.000000000',
                               '1959-01-03T00:00:00.000000000', '1959-01-04T00:00:00.000000000',
                               '1959-01-05T00:00:00.000000000', '1959-01-06T00:00:00.000000000',
                               '1959-01-07T00:00:00.000000000'], dtype='datetime64[ns]'),
                "min": np.array(['1959-01-01T00:00:00.000000000', '1959-01-01T00:01:00.000000000',
                                 '1959-01-01T00:02:00.000000000', '1959-01-01T00:03:00.000000000',
                                 '1959-01-01T00:04:00.000000000', '1959-01-01T00:05:00.000000000',
                                 '1959-01-01T00:06:00.000000000'], dtype='datetime64[ns]'),
                "12M": np.array(['1959-01-31T00:00:00.000000000', '1960-01-31T00:00:00.000000000',
                                 '1961-01-31T00:00:00.000000000', '1962-01-31T00:00:00.000000000',
                                 '1963-01-31T00:00:00.000000000', '1964-01-31T00:00:00.000000000',
                                 '1965-01-31T00:00:00.000000000'], dtype='datetime64[ns]')
                }
    return expected


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

    def test_advanced_STL(self, advanced_dku_config, input_df):
        timeseries_preparator = TimeseriesPreparator(advanced_dku_config)
        df_prepared = timeseries_preparator.prepare_timeseries_dataframe(input_df)
        decomposition = STLDecomposition(advanced_dku_config)
        results = decomposition.fit(df_prepared)
        expected_array = np.array([507693, 502152, 496611, 491070, 485529, 479988, 474447, 468906, 463365,
                                   457824, 452283, 446742, 441201, 445989, 450777, 455565, 460353, 465141,
                                   469928, 474716, 479504, 484292, 489080, 493868, 498655, 499737])
        rounded_results = np.round(results["value1_trend"].values)
        np.testing.assert_equal(rounded_results, expected_array)

    def test_several_frequencies(self, expected_dates):
        results_df = get_result_df("3M")
        assert results_df.shape == (7, 5)
        np.testing.assert_equal(results_df["date"].values, expected_dates["3M"])

        results_df = get_result_df("6M")
        assert results_df.shape == (7, 5)
        np.testing.assert_equal(results_df["date"].values, expected_dates["6M"])

        results_df = get_result_df("M")
        assert results_df.shape == (7, 5)
        np.testing.assert_equal(results_df["date"].values, expected_dates["M"])

        results_df = get_result_df("W")
        assert results_df.shape == (7, 5)
        np.testing.assert_equal(results_df["date"].values, expected_dates["W"])

        results_df = get_result_df("W", frequency_end_of_week="FRI")
        assert results_df.shape == (7, 5)
        np.testing.assert_equal(results_df["date"].values, expected_dates["W-FRI"])

        results_df = get_result_df("B")
        assert results_df.shape == (7, 5)
        np.testing.assert_equal(results_df["date"].values, expected_dates["B"])

        results_df = get_result_df("H")
        assert results_df.shape == (7, 5)
        np.testing.assert_equal(results_df["date"].values, expected_dates["H"])

        results_df = get_result_df("H", frequency_step_hours="4")
        assert results_df.shape == (7, 5)
        np.testing.assert_equal(results_df["date"].values, expected_dates["4H"])

        results_df = get_result_df("D")
        assert results_df.shape == (7, 5)
        np.testing.assert_equal(results_df["date"].values, expected_dates["D"])

        results_df = get_result_df("min")
        assert results_df.shape == (7, 5)
        np.testing.assert_equal(results_df["date"].values, expected_dates["min"])

        results_df = get_result_df("12M")
        assert results_df.shape == (7, 5)
        np.testing.assert_equal(results_df["date"].values, expected_dates["12M"])


def df_from_freq(dku_config):
    data = [315.58, 316.39, 316.79, 312.09, 321.08, 450.08, 298.79]
    freq = dku_config.frequency
    df = pd.DataFrame.from_dict(
        {"value1": data, "date": pd.date_range("1-1-1959", periods=len(data), freq=freq)})
    timeseries_preparator = TimeseriesPreparator(dku_config)
    df_prepared = timeseries_preparator.prepare_timeseries_dataframe(df)
    return df_prepared


def config_from_freq(freq, frequency_end_of_week=None, frequency_step_hours=None, frequency_step_minutes=None):
    default_season_length = {"12M": 4, "6M": 2, "3M": 4, "M": 12, "W": 52, "D": 7, "B": 5, "H": 24, "min": 60}
    config = {"transformation_type": "seasonal_decomposition", "time_decomposition_method": "STL", "frequency_unit": freq, "time_column": "date",
              "target_columns": ["value1"], "long_format": False, "decomposition_model": "multiplicative", "seasonal_stl": 13, "expert": True,
              "season_length_{}".format(freq): default_season_length[freq]}
    if frequency_end_of_week:
        config["frequency_end_of_week"] = frequency_end_of_week
    elif frequency_step_hours:
        config["frequency_step_hours"] = frequency_step_hours
    elif frequency_step_minutes:
        config["frequency_step_minutes"] = frequency_step_minutes
    input_dataset_columns = ["value1", "value2", "country", "date"]
    dku_config = STLConfig()
    dku_config.add_parameters(config, input_dataset_columns)
    return dku_config


def get_result_df(freq, frequency_end_of_week=None, frequency_step_hours=None, frequency_step_minutes=None):
    dku_config = config_from_freq(freq, frequency_end_of_week, frequency_step_hours, frequency_step_minutes)
    df_prepared = df_from_freq(dku_config)
    decomposition = STLDecomposition(dku_config)
    results_df = decomposition.fit(df_prepared)
    return results_df
