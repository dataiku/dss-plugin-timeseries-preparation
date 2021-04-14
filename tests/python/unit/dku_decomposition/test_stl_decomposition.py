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
def config():
    config = {"frequency_unit": "M", "season_length_M": 12, "time_column": "date", "target_columns": ["value1"],
              "long_format": False, "decomposition_model": "additive", "seasonal_stl": 13, "expert": True, "additional_parameters_STL": {}}
    return config


@pytest.fixture
def input_dataset_columns():
    return ["value1", "value2", "country", "date"]


@pytest.fixture
def dku_config(config, input_dataset_columns):
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
                               '1959-01-07T00:00:00.000000000'], dtype='datetime64[ns]')
                }
    return expected


class TestSTLDecomposition:
    def test_STL_multiplicative(self, dku_config, input_df):
        dku_config.model = "multiplicative"
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

        assert np.mean(results["value1_trend"]) == 409265.35453951
        assert np.mean(results["value1_seasonal"]) == 1.2698748679749627
        assert np.mean(results["value1_residuals"]) == 0.9941032097902623

    def test_STL_additive(self, dku_config, input_df):
        timeseries_preparator = TimeseriesPreparator(dku_config)
        df_prepared = timeseries_preparator.prepare_timeseries_dataframe(input_df)
        decomposition = STLDecomposition(dku_config)
        results = decomposition.fit(df_prepared)
        expected_array = np.array(
            [547017.8314, 537486.722, 528097.1954, 518846.2605, 509728.8989, 500744.2034, 491895.324, 483188.5115, 474630.5299, 466256.2782, 458496.2869,
             454985.6935, 453114.0625, 452740.149, 453810.1866, 456404.7768, 463218.9767, 470913.292, 478947.2522, 487217.229, 495684.7824, 504325.6079,
             513126.1746, 522081.8564, 531195.1428, 540473.2835])
        rounded_results = np.round(results["value1_trend"].values, 4)
        np.testing.assert_equal(rounded_results, expected_array)
        assert np.mean(results["value1_trend"]) == 492101.0195351211
        assert np.mean(results["value1_seasonal"]) == 32625.652227975654
        assert np.mean(results["value1_residuals"]) == -5345.248686173698

    def test_quarter_frequency(self, expected_dates):
        results_df = get_additive_result_df("3M")
        assert results_df.shape == (7, 5)
        np.testing.assert_equal(results_df["date"].values, expected_dates["3M"])
        np.testing.assert_equal(np.round(results_df["value1_trend"].values, 4), [301.5621, 311.6514, 321.7407, 331.83, 341.9193, 352.0086, 362.0979])
        assert np.mean(results_df["value1_trend"]) == 331.8299999999998
        assert np.mean(results_df["value1_seasonal"]) == 1.141428571428658
        assert np.mean(results_df["value1_residuals"]) == 9.74458609042423e-14

    def test_semiannual_frequency(self, expected_dates):
        results_df = get_additive_result_df("6M")
        assert results_df.shape == (7, 5)
        np.testing.assert_equal(results_df["date"].values, expected_dates["6M"])
        np.testing.assert_equal(np.round(results_df["value1_trend"].values, 4), [300.893, 309.6409, 314.1272, 319.7546, 349.1279, 372.5453, 390.7896])
        assert np.mean(results_df["value1_trend"]) == 336.6969211857202
        assert np.mean(results_df["value1_seasonal"]) == -3.868590336224215
        assert np.mean(results_df["value1_residuals"]) == 0.1430977219325931

    def test_monthly_frequency(self, expected_dates):
        results_df = get_additive_result_df("M")
        assert results_df.shape == (7, 5)
        np.testing.assert_equal(results_df["date"].values, expected_dates["M"])
        np.testing.assert_equal(np.round(results_df["value1_trend"].values, 4), [336.8828, 336.8828, 336.8828, 336.8828, 336.8828, 336.8828, 336.8828])
        assert np.mean(results_df["value1_trend"]) == 336.88280446244863
        assert np.mean(results_df["value1_seasonal"]) == -3.9113758910199925
        assert np.mean(results_df["value1_residuals"]) == -8.120488408686859e-15

    def test_weekly_frequencies(self, expected_dates):
        results_df = get_additive_result_df("W")
        assert results_df.shape == (7, 5)
        np.testing.assert_equal(results_df["date"].values, expected_dates["W"])
        np.testing.assert_equal(np.round(results_df["value1_trend"].values, 4), [175.9658, 175.9658, 175.9658, 175.9658, 175.9658, 175.9658, 175.9658])
        assert np.mean(results_df["value1_trend"]) == 175.9658401676288
        assert np.mean(results_df["value1_seasonal"]) == 157.00558840379978
        assert np.mean(results_df["value1_residuals"]) == -1.6240976817373718e-14

        results_df = get_additive_result_df("W", frequency_end_of_week="FRI")
        assert results_df.shape == (7, 5)
        np.testing.assert_equal(results_df["date"].values, expected_dates["W-FRI"])
        np.testing.assert_equal(np.round(results_df["value1_trend"].values, 4), [175.9658, 175.9658, 175.9658, 175.9658, 175.9658, 175.9658, 175.9658])
        assert np.mean(results_df["value1_trend"]) == 175.9658401676288
        assert np.mean(results_df["value1_seasonal"]) == 157.00558840379978
        assert np.mean(results_df["value1_residuals"]) == -1.6240976817373718e-14

    def test_business_day_frequencies(self, expected_dates):
        results_df = get_additive_result_df("B")
        assert results_df.shape == (7, 5)
        np.testing.assert_equal(results_df["date"].values, expected_dates["B"])
        np.testing.assert_equal(np.round(results_df["value1_trend"].values, 4), [298.775, 309.556, 320.337, 331.118, 341.899, 352.68, 363.461])
        assert np.mean(results_df["value1_trend"]) == 331.1179999999999
        assert np.mean(results_df["value1_seasonal"]) == 1.8534285714286125
        assert np.mean(results_df["value1_residuals"]) == 7.308439567818174e-14

    def test_hour_frequencies(self, expected_dates):
        results_df = get_additive_result_df("H")
        assert results_df.shape == (7, 5)
        np.testing.assert_equal(results_df["date"].values, expected_dates["H"])
        np.testing.assert_equal(np.round(results_df["value1_trend"].values, 4), [285.7512, 285.7512, 285.7512, 285.7512, 285.7512, 285.7512, 285.7512])
        assert np.mean(results_df["value1_trend"]) == 285.7511909139499
        assert np.mean(results_df["value1_seasonal"]) == 47.2202376574786
        assert np.mean(results_df["value1_residuals"]) == 0.0

        results_df = get_additive_result_df("H", frequency_step_hours="4")
        assert results_df.shape == (7, 5)
        np.testing.assert_equal(results_df["date"].values, expected_dates["4H"])
        np.testing.assert_equal(np.round(results_df["value1_trend"].values, 4), [285.7512, 285.7512, 285.7512, 285.7512, 285.7512, 285.7512, 285.7512])
        assert np.mean(results_df["value1_trend"]) == 285.7511909139499
        assert np.mean(results_df["value1_seasonal"]) == 47.2202376574786
        assert np.mean(results_df["value1_residuals"]) == 0.0

    def test_day_frequencies(self, expected_dates):
        results_df = get_additive_result_df("D")
        assert results_df.shape == (7, 5)
        np.testing.assert_equal(results_df["date"].values, expected_dates["D"])
        np.testing.assert_equal(np.round(results_df["value1_trend"].values, 4), [332.9714, 332.9714, 332.9714, 332.9714, 332.9714, 332.9714, 332.9714])
        assert np.mean(results_df["value1_trend"]) == 332.97142857142853
        assert np.mean(results_df["value1_seasonal"]) == -1.0150610510858574e-15
        assert np.mean(results_df["value1_residuals"]) == -1.6240976817373718e-14

    def test_advanced_smoothers(self, config, input_df, input_dataset_columns):
        config["decomposition_model"] = "additive"
        config["additional_parameters_STL"] = {"trend": "35", "low_pass": "31"}
        dku_config = STLConfig()
        dku_config.add_parameters(config, input_dataset_columns)
        timeseries_preparator = TimeseriesPreparator(dku_config)
        df_prepared = timeseries_preparator.prepare_timeseries_dataframe(input_df)
        decomposition = STLDecomposition(dku_config)
        result_df = decomposition.fit(df_prepared)
        assert result_df.shape == (26, 6)
        np.testing.assert_array_equal(np.round(result_df.value1_seasonal.values, 4), np.array(
            [329279.6394, 360305.5117, 378691.0343, 151319.491, -166075.4661, -206300.4391, -272041.7161, -285356.053, -274969.4078, -125368.4261, -10804.3636,
             173084.5489, 421640.9531, 393264.9995, 288207.4229, 42573.3565, -111402.3446, -270267.5348, -299889.3857, -334837.5864, -291850.134, -103986.6224,
             42205.6726, 274027.7075, 515335.6499, 429183.6225]))
        assert np.mean(result_df["value1_trend"]) == 482542.4367257319
        assert np.mean(result_df["value1_seasonal"]) == 40229.62038767122
        assert np.mean(result_df["value1_residuals"]) == -3390.634036480091

        config["additional_parameters_STL"] = {"trend": "2999999", "low_pass": "13"}
        dku_config = STLConfig()
        dku_config.add_parameters(config, input_dataset_columns)
        timeseries_preparator = TimeseriesPreparator(dku_config)
        df_prepared = timeseries_preparator.prepare_timeseries_dataframe(input_df)
        decomposition = STLDecomposition(dku_config)
        result_df = decomposition.fit(df_prepared)
        assert result_df.shape == (26, 6)
        assert np.mean(result_df["value1_trend"]) == 476077.5935197392
        assert np.mean(result_df["value1_seasonal"]) == 43303.82955718398
        assert np.mean(result_df["value1_residuals"]) == -3.134258664571322e-11

    def test_advanced_degrees(self, config, input_df, input_dataset_columns):
        config["additional_parameters_STL"] = {"seasonal_deg": "1", "trend_deg": "1", "low_pass_deg": "1"}
        dku_config = STLConfig()
        dku_config.add_parameters(config, input_dataset_columns)
        timeseries_preparator = TimeseriesPreparator(dku_config)
        df_prepared = timeseries_preparator.prepare_timeseries_dataframe(input_df)
        decomposition = STLDecomposition(dku_config)
        result_df = decomposition.fit(df_prepared)
        assert result_df.shape == (26, 6)
        np.testing.assert_array_equal(np.round(result_df.value1_trend.values, 4),
                                      np.array(
                                          [547017.8314, 537486.722, 528097.1954, 518846.2605, 509728.8989, 500744.2034, 491895.324, 483188.5115, 474630.5299,
                                           466256.2782, 458496.2869, 454985.6935, 453114.0625, 452740.149, 453810.1866, 456404.7768, 463218.9767, 470913.292,
                                           478947.2522, 487217.229, 495684.7824, 504325.6079, 513126.1746, 522081.8564, 531195.1428, 540473.2835]))
        assert np.mean(result_df["value1_trend"]) == 492101.0195351211
        assert np.mean(result_df["value1_seasonal"]) == 32625.652227975654
        assert np.mean(result_df["value1_residuals"]) == -5345.248686173698

        config["additional_parameters_STL"] = {"seasonal_deg": "1", "trend_deg": "0"}
        dku_config = STLConfig()
        dku_config.add_parameters(config, input_dataset_columns)
        timeseries_preparator = TimeseriesPreparator(dku_config)
        df_prepared = timeseries_preparator.prepare_timeseries_dataframe(input_df)
        decomposition = STLDecomposition(dku_config)
        result_df = decomposition.fit(df_prepared)
        assert result_df.shape == (26, 6)
        np.testing.assert_array_equal(np.round(result_df.value1_seasonal.values, 4),
                                      np.array([334926.5396, 363552.8324, 380642.7497, 151182.772, -168020.8919, -209675.4339, -276299.0916, -289677.7104,
                                                -278165.1873, -126041.2679, -8513.4181, 175315.4394, 425222.6624, 396736.9844, 290811.7923, 45628.9471,
                                                -110941.82, -272356.2149, -303391.2037, -338667.781, -295226.877, -106373.2845, 41186.7333, 274657.8578,
                                                516720.1595, 432742.083]))
        assert np.mean(result_df["value1_trend"]) == 470658.0934271346
        assert np.mean(result_df["value1_seasonal"]) == 40229.89887290871
        assert np.mean(result_df["value1_residuals"]) == 8493.430776879803


def df_from_freq(dku_config):
    data = [315.58, 316.39, 316.79, 312.09, 321.08, 450.08, 298.79]
    freq = dku_config.frequency
    df = pd.DataFrame.from_dict(
        {"value1": data, "date": pd.date_range("1-1-1959", periods=len(data), freq=freq)})
    timeseries_preparator = TimeseriesPreparator(dku_config)
    df_prepared = timeseries_preparator.prepare_timeseries_dataframe(df)
    return df_prepared


def additive_config_from_freq(freq, frequency_end_of_week=None, frequency_step_hours=None, frequency_step_minutes=None):
    default_season_length = {"12M": 4, "6M": 2, "3M": 4, "M": 12, "W": 52, "D": 7, "B": 5, "H": 24, "min": 60}
    config = {"frequency_unit": freq, "time_column": "date", "target_columns": ["value1"], "season_length_{}".format(freq): default_season_length[freq],
              "long_format": False, "decomposition_model": "additive", "seasonal_stl": 13, "expert": True, "additional_parameters_STL": {}}
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


def get_additive_result_df(freq, frequency_end_of_week=None, frequency_step_hours=None, frequency_step_minutes=None):
    dku_config = additive_config_from_freq(freq, frequency_end_of_week, frequency_step_hours, frequency_step_minutes)
    df_prepared = df_from_freq(dku_config)
    decomposition = STLDecomposition(dku_config)
    results_df = decomposition.fit(df_prepared)
    return results_df
