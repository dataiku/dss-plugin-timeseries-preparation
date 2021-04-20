import pandas as pd
import pytest

from dku_input_validator.decomposition_input_validator import DecompositionInputValidator
from dku_config.decomposition_config import DecompositionConfig
from timeseries_preparation.preparation import TimeseriesPreparator


@pytest.fixture
def basic_dku_config():
    input_dataset_columns = ["value1","value2","date"]
    dku_config = DecompositionConfig()
    config = {"transformation_type": "seasonal_decomposition", "time_decomposition_method": "STL",
              "frequency_unit": "M", "season_length_M": 12, "time_column": "date", "target_columns": ["value1", "value2"],
              "long_format": False, "decomposition_model": "multiplicative", "expert": False}
    dku_config.add_parameters(config, input_dataset_columns)
    return dku_config

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
    item = [1,2,1,2]
    time_index = pd.date_range("1-1-1959", periods=2, freq="M").append(pd.date_range("1-1-1959", periods=2, freq="M"))
    df = pd.DataFrame.from_dict(
        {"value1": co2, "value2": co2, "country": country, "item": item, "date": time_index})
    return df


class TestInputValidator:
    def test_multiplicative_model_with_negative_values(self, basic_dku_config, input_df):
        input_df.loc[0, "value1"] = -2

        timeseries_preparator = TimeseriesPreparator(basic_dku_config)
        df_prepared = timeseries_preparator.prepare_timeseries_dataframe(input_df)

        input_validator = DecompositionInputValidator(basic_dku_config)
        with pytest.raises(ValueError) as err:
            _ = input_validator.check(df_prepared)
        assert "multiplicative" in str(err.value)
        assert "negative" in str(err.value)
        assert "value1" in str(err.value)

    def test_insufficient_samples_one_ts_identifier(self, basic_dku_config, long_df):
        basic_dku_config.long_format = True
        basic_dku_config.timeseries_identifiers = ["country"]
        timeseries_preparator = TimeseriesPreparator(basic_dku_config)
        df_too_short = timeseries_preparator.prepare_timeseries_dataframe(long_df)
        input_validator = DecompositionInputValidator(basic_dku_config)

        with pytest.raises(ValueError) as err:
            _ = input_validator.check(df_too_short)
        assert "need at least" in str(err.value)
        assert "country" in str(err.value)
        assert "[2 2]" in str(err.value)

    def test_insufficient_samples_2_ts_identifiers(self, basic_dku_config, long_df):
        basic_dku_config.long_format = True
        basic_dku_config.timeseries_identifiers = ["country","item"]
        timeseries_preparator = TimeseriesPreparator(basic_dku_config)
        df_too_short = timeseries_preparator.prepare_timeseries_dataframe(long_df)
        input_validator = DecompositionInputValidator(basic_dku_config)
        with pytest.raises(ValueError) as err:
            _ = input_validator.check(df_too_short)
        assert "need at least" in str(err.value)
        assert "country" in str(err.value)
        assert "item" in str(err.value)
        assert "[1 1 1 1]" in str(err.value)





