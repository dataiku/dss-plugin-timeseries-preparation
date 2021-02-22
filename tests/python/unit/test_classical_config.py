import numpy as np
import pytest

from dku_config.dss_parameter import DSSParameterError
from dku_config.classical_config import ClassicalConfig


@pytest.fixture
def basic_config():
    config = {"transformation_type": "seasonal_decomposition", "time_decomposition_method": "classical",
              "frequency_unit": "M", "season_length_M": 12, "time_column": "date", "target_columns": ["target"],
              "long_format": False, "decomposition_model": "multiplicative", "expert": False}
    return config


@pytest.fixture
def advanced_config():
    config = {"transformation_type": "seasonal_decomposition", "time_decomposition_method": "classical",
              "frequency_unit": "M", "season_length_M": 12, "time_column": "date", "target_columns": ["target"],
              "long_format": False, "decomposition_model": "multiplicative", "expert": True,
              "advanced_params_classical": {"filt": "[1,2,3]", "two_sided": "False", "extrapolate_trend": ""}
              }
    return config


@pytest.fixture
def input_dataset_columns():
    return ["target", "date", "something"]


class TestClassicalConfig:
    def test_load_settings(self, basic_config, input_dataset_columns):
        dku_config = ClassicalConfig()
        assert dku_config.add_parameters(basic_config, input_dataset_columns) is None

    def test_load_advanced_parameters(self, advanced_config, input_dataset_columns):
        dku_config = ClassicalConfig()
        assert dku_config.add_parameters(advanced_config, input_dataset_columns) is None

        advanced_config["advanced_params_classical"]["extrapolate_trend"] = "freq"
        assert dku_config.add_parameters(advanced_config, input_dataset_columns) is None

        advanced_config["advanced_params_classical"]["extrapolate_trend"] = "2"
        assert dku_config.add_parameters(advanced_config, input_dataset_columns) is None

        advanced_config["advanced_params_classical"]["wrong_key"] = "2"
        with pytest.raises(DSSParameterError) as wrong_key_err:
            _ = dku_config.add_parameters(advanced_config, input_dataset_columns)
        assert "field is invalid" in str(wrong_key_err.value)

        advanced_config["advanced_params_classical"].pop("wrong_key")
        advanced_config["advanced_params_classical"]["filt"] = "[2,3,f]"
        with pytest.raises(DSSParameterError) as array_err:
            _ = dku_config.add_parameters(advanced_config, input_dataset_columns)
        assert "Should be of type" in str(array_err.value)
        assert "numpy.ndarray" in str(array_err.value)

        advanced_config["advanced_params_classical"].pop("filt")
        advanced_config["advanced_params_classical"]["two_sided"] = "not_a_bool"
        with pytest.raises(DSSParameterError) as bool_err:
            _ = dku_config.add_parameters(advanced_config, input_dataset_columns)
        assert "Should be of type" in str(bool_err.value)
        assert "bool" in str(bool_err.value)

    def test_parsing(self, advanced_config, input_dataset_columns):
        dku_config = ClassicalConfig()
        advanced_config["advanced_params_classical"]["extrapolate_trend"] = "2"
        advanced_config["advanced_params_classical"]["filt"] = "[2,3]"
        dku_config.add_parameters(advanced_config, input_dataset_columns)
        assert dku_config.extrapolate_trend == 2
        np.testing.assert_equal(dku_config.filt, np.array([2, 3]))
        assert dku_config.two_sided is False
