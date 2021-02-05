import pandas as pd
import pytest

from dku_config.dss_parameter import DSSParameterError
from dku_config.decomposition_config import DecompositionConfig


@pytest.fixture
def basic_config():
    config = {"transformation_type": "seasonal_decomposition", "time_decomposition_method": "STL",
              "frequency_unit": "M", "time_column": "date", "target_columns": ["value1", "value2"],
              "long_format": False, "decomposition_model": "multiplicative", "expert": False}
    return config


@pytest.fixture
def input_df():
    co2 = [315.58, 316.39, 316.79]
    df = pd.DataFrame.from_dict(
        {"value1": co2, "value2": co2, "date": pd.date_range("1-1-1959", periods=len(co2), freq="M")})
    return df


class TestDecompositionConfig:
    def test_input_parameters(self, basic_config, input_df):
        dku_config = DecompositionConfig()
        assert dku_config.add_parameters(basic_config, input_df) is None

    def test_invalid_time_column(self, basic_config, input_df):
        dku_config = DecompositionConfig()
        basic_config["time_column"] = "date_different"
        with pytest.raises(DSSParameterError) as err:
            _ = dku_config.add_parameters(basic_config, input_df)
        assert "Invalid time column selection:" in str(err.value)

    def test_invalid_target_columns(self,basic_config, input_df):
        dku_config = DecompositionConfig()
        basic_config["target_columns"] = ["wrong_target"]
        with pytest.raises(DSSParameterError) as err:
            _ = dku_config.add_parameters(basic_config, input_df)
        assert "Invalid target column(s) selection:" in str(err.value)

    def test_long_format(self, basic_config, input_df):
        basic_config["long_format"] = True
        basic_config["timeseries_identifiers"] = ["value1"]
        basic_config["target_columns"] = ["value2"]
        dku_config = DecompositionConfig()
        assert dku_config.add_parameters(basic_config, input_df) is None

        basic_config["timeseries_identifiers"] = []
        with pytest.raises(DSSParameterError) as identifier_err:
            _ = dku_config.add_parameters(basic_config, input_df)
        assert "no time series identifiers" in str(identifier_err.value)

        basic_config.pop("timeseries_identifiers")
        with pytest.raises(DSSParameterError) as missing_id_err:
            _ = dku_config.add_parameters(basic_config, input_df)
        assert "no time series identifiers" in str(missing_id_err.value)

        basic_config["timeseries_identifiers"] = "column"
        with pytest.raises(DSSParameterError) as type_err:
            _ = dku_config.add_parameters(basic_config, input_df)
        assert "Should be of type" in str(type_err.value)
        assert "list" in str(type_err.value)

        basic_config["timeseries_identifiers"] = ["column"]
        with pytest.raises(DSSParameterError) as in_err:
            _ = dku_config.add_parameters(basic_config, input_df)
        assert "Invalid time series identifiers" in str(in_err.value)

    def test_multiplicative_model_with_negative_values(self, basic_config, input_df):
        dku_config = DecompositionConfig()
        input_df.loc[0, "value1"] = -2
        with pytest.raises(DSSParameterError) as err:
            _ = dku_config.add_parameters(basic_config, input_df)

        assert "Error for parameter" in str(err.value)
        assert "multiplicative" in str(err.value)
        assert "negative" in str(err.value)
