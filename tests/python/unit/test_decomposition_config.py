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
def input_dataset_columns():
    return ["value1", "value2", "date"]


class TestDecompositionConfig:
    def test_input_parameters(self, basic_config, input_dataset_columns):
        dku_config = DecompositionConfig()
        assert dku_config.add_parameters(basic_config, input_dataset_columns) is None

    def test_invalid_time_column(self, basic_config, input_dataset_columns):
        dku_config = DecompositionConfig()
        basic_config["time_column"] = "date_different"
        with pytest.raises(DSSParameterError) as err:
            _ = dku_config.add_parameters(basic_config, input_dataset_columns)
        assert "Invalid time column selection:" in str(err.value)

    def test_invalid_target_columns(self, basic_config, input_dataset_columns):
        dku_config = DecompositionConfig()
        basic_config["target_columns"] = ["wrong_target"]
        with pytest.raises(DSSParameterError) as err:
            _ = dku_config.add_parameters(basic_config, input_dataset_columns)
        assert "Invalid target column(s) selection:" in str(err.value)

    def test_long_format(self, basic_config, input_dataset_columns):
        basic_config["long_format"] = True
        basic_config["timeseries_identifiers"] = ["value1"]
        basic_config["target_columns"] = ["value2"]
        dku_config = DecompositionConfig()
        assert dku_config.add_parameters(basic_config, input_dataset_columns) is None

        basic_config["timeseries_identifiers"] = []
        with pytest.raises(DSSParameterError) as identifier_err:
            _ = dku_config.add_parameters(basic_config, input_dataset_columns)
        assert "no time series identifiers" in str(identifier_err.value)

        basic_config.pop("timeseries_identifiers")
        with pytest.raises(DSSParameterError) as missing_id_err:
            _ = dku_config.add_parameters(basic_config, input_dataset_columns)
        assert "no time series identifiers" in str(missing_id_err.value)

        basic_config["timeseries_identifiers"] = "column"
        with pytest.raises(DSSParameterError) as type_err:
            _ = dku_config.add_parameters(basic_config, input_dataset_columns)
        assert "Should be of type" in str(type_err.value)
        assert "list" in str(type_err.value)

        basic_config["timeseries_identifiers"] = ["column"]
        with pytest.raises(DSSParameterError) as in_err:
            _ = dku_config.add_parameters(basic_config, input_dataset_columns)
        assert "Invalid time series identifiers" in str(in_err.value)
