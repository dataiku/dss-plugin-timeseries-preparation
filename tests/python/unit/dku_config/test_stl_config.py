import pytest

from dku_config.dss_parameter import DSSParameterError
from dku_config.stl_config import STLConfig


@pytest.fixture
def basic_config():
    config = {"transformation_type": "seasonal_decomposition", "time_decomposition_method": "STL",
              "frequency_unit": "M", "season_length_M": 12, "time_column": "date", "target_columns": ["target"],
              "long_format": False, "decomposition_model": "multiplicative", "seasonal_stl": 7, "expert": False}
    return config


@pytest.fixture
def advanced_config():
    config = {"transformation_type": "seasonal_decomposition", "time_decomposition_method": "STL",
              "frequency_unit": "D", "season_length_D": 7,
              "decomposition_model": "additive", "seasonal_stl": 7, "expert": True, "robust_stl": True,
              "seasonal_degree_stl": "1",
              "trend_degree_stl": "0", "lowpass_degree_stl": "0", "time_column": "date",
              "long_format": False, "target_columns": ["target"],
              "stl_degree_kwargs": {"seasonal_deg": "1", "trend_deg": "", "low_pass_deg": "1"},
              "stl_speed_jump_kwargs": {"seasonal_jump": '', "trend_jump": "12", "low_pass_jump": ''},
              "stl_smoothers_kwargs": {"trend": "13", "low_pass": ''}}
    return config


@pytest.fixture
def input_dataset_columns():
    return ["target", "date", "something"]


class TestSTLConfig:
    def test_add_parameters(self, basic_config, input_dataset_columns):
        dku_config = STLConfig()
        assert dku_config.add_parameters(basic_config, input_dataset_columns) is None

    def test_invalid_seasonal(self, basic_config, input_dataset_columns):
        dku_config = STLConfig()
        basic_config["seasonal_stl"] = 8
        with pytest.raises(DSSParameterError) as odd_err:
            _ = dku_config.add_parameters(basic_config, input_dataset_columns)
        assert "odd" in str(odd_err.value)

        basic_config["seasonal_stl"] = 9.5
        with pytest.raises(DSSParameterError) as double_err:
            _ = dku_config.add_parameters(basic_config, input_dataset_columns)
        assert "type" in str(double_err.value)
        assert "int" in str(double_err.value)

        basic_config["seasonal_stl"] = "string"
        with pytest.raises(DSSParameterError) as str_err:
            _ = dku_config.add_parameters(basic_config, input_dataset_columns)
        assert "type" in str(str_err.value)
        assert "int" in str(str_err.value)

        basic_config.pop("seasonal_stl")
        with pytest.raises(DSSParameterError) as required_err:
            _ = dku_config.add_parameters(basic_config, input_dataset_columns)
        assert "required" in str(required_err.value)

    def test_advanced_parameters(self, advanced_config, input_dataset_columns):
        dku_config = STLConfig()
        assert dku_config.add_parameters(advanced_config, input_dataset_columns) is None

    def test_invalid_advanced_parameters(self, advanced_config, input_dataset_columns):
        dku_config = STLConfig()
        advanced_config["stl_smoothers_kwargs"]["trend"] = "200"
        with pytest.raises(DSSParameterError) as odd_err:
            _ = dku_config.add_parameters(advanced_config, input_dataset_columns)
        assert "odd" in str(odd_err.value)

        advanced_config["stl_smoothers_kwargs"]["trend"] = "4"
        with pytest.raises(DSSParameterError) as min_err:
            _ = dku_config.add_parameters(advanced_config, input_dataset_columns)
        assert "greater" in str(min_err.value)
        assert "period (= 7)" in str(min_err.value)

        advanced_config["stl_smoothers_kwargs"]["trend"] = "3"
        advanced_config["stl_degree_kwargs"]["seasonal_deg"] = "2"
        with pytest.raises(DSSParameterError) as non_binary_err:
            _ = dku_config.add_parameters(advanced_config, input_dataset_columns)
        assert "0" in str(non_binary_err.value)
        assert "1" in str(non_binary_err.value)

        advanced_config["stl_degree_kwargs"]["seasonal_deg"] = "1"
        advanced_config["stl_smoothers_kwargs"]["wrong_field"] = "3"
        with pytest.raises(DSSParameterError) as invalid_field_err:
            _ = dku_config.add_parameters(advanced_config, input_dataset_columns)
        assert "This field is invalid" in str(invalid_field_err.value)

    def test_parsing(self, advanced_config, input_dataset_columns):
        dku_config = STLConfig()
        dku_config.add_parameters(advanced_config, input_dataset_columns)
        assert dku_config.loess_degrees.get("seasonal_deg") == 1
        assert dku_config.speed_jumps.get("trend_jump") == 12
        assert dku_config.additional_smoothers.get("trend") == 13
