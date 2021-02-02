import pytest

from dku_config.dss_parameter import DSSParameterError
from dku_config.decomposition_config import DecompositionConfig


class TestDecompositionConfig:
    def test_load_input_parameters(self):
        input_dataset_columns = ["date", "value1", "value2"]
        valid_config = {"time_column": "date", "target_columns": ["value1", "value2"], "frequency_unit": "D"}
        dku_config = DecompositionConfig()
        assert dku_config._load_input_parameters(valid_config, input_dataset_columns) is None

        invalid_target_config = {"time_column": "date", "target_columns": ["not_here", "value2"], "frequency_unit": "D"}
        with pytest.raises(DSSParameterError):
            _ = dku_config._load_input_parameters(invalid_target_config, input_dataset_columns)
        invalid_time_config = {"time_column": "date_different", "target_columns": ["value1", "value2"],
                               "frequency_unit": "D"}
        with pytest.raises(DSSParameterError):
            _ = dku_config._load_input_parameters(invalid_time_config, input_dataset_columns)

    def test_long_format(self):
        input_dataset_columns = ["date", "value1", "value2"]
        config = {"time_column": "date", "target_columns": ["value1", "value2"], "frequency_unit": "D",
                        "long_format": True,
                        "timeseries_identifiers": ["value1"]}
        dku_config = DecompositionConfig()
        assert dku_config._load_input_parameters(config, input_dataset_columns) is None

        config["timeseries_identifiers"] = []
        with pytest.raises(DSSParameterError):
            _ = dku_config._load_input_parameters(config, input_dataset_columns)
        config.pop("timeseries_identifiers")
        with pytest.raises(DSSParameterError):
            _ = dku_config._load_input_parameters(config, input_dataset_columns)
        config["timeseries_identifiers"] = "column"
        with pytest.raises(DSSParameterError):
            _ = dku_config._load_input_parameters(config, input_dataset_columns)
        config["timeseries_identifiers"] = ["column"]
        with pytest.raises(DSSParameterError):
            _ = dku_config._load_input_parameters(config, input_dataset_columns)

