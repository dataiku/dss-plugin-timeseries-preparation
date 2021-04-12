import pytest

from dku_config.dss_parameter import DSSParameterError
from dku_config.stl_config import STLConfig


@pytest.fixture
def basic_config():
    config = {"transformation_type": "seasonal_decomposition", "frequency_unit": "M", "season_length_M": 12, "time_column": "date",
              "target_columns": ["target"], "long_format": False, "decomposition_model": "multiplicative", "expert": False}
    return config


@pytest.fixture
def advanced_config():
    config = {"transformation_type": "seasonal_decomposition","time_column": "date","target_columns": ["target"],
              "frequency_unit": "D", "season_length_D": 7,
              "decomposition_model": "additive", "seasonal_stl": 7, "expert": True, "robust_stl": True,
              "seasonal_degree_stl": "1", "advanced_params_STL": {}}
    return config


@pytest.fixture
def input_dataset_columns():
    return ["target", "date", "something"]


class TestSTLConfig:
    def test_add_parameters(self, basic_config, input_dataset_columns):
        dku_config = STLConfig()
        assert dku_config.add_parameters(basic_config, input_dataset_columns) is None

    def test_advanced_params(self, advanced_config, input_dataset_columns):
        dku_config = STLConfig()
        assert dku_config.add_parameters(advanced_config, input_dataset_columns) is None

    def test_robust_params(self, advanced_config, input_dataset_columns):
        dku_config = STLConfig()
        advanced_config["robust_stl"] = False
        assert dku_config.add_parameters(advanced_config, input_dataset_columns) is None

        advanced_config["robust_stl"] = 42
        with pytest.raises(DSSParameterError) as err:
            _ = dku_config.add_parameters(advanced_config, input_dataset_columns)
        assert "bool" in str(err.value)

        advanced_config.pop("robust_stl")
        dku_config.add_parameters(advanced_config, input_dataset_columns)
        assert dku_config.robust_stl == False


    def test_invalid_seasonal(self, advanced_config, input_dataset_columns):
        dku_config = STLConfig()
        advanced_config["seasonal_stl"] = 8
        with pytest.raises(DSSParameterError) as odd_err:
            _ = dku_config.add_parameters(advanced_config, input_dataset_columns)
        assert "odd" in str(odd_err.value)

        advanced_config["seasonal_stl"] = 9.5
        with pytest.raises(DSSParameterError) as double_err:
            _ = dku_config.add_parameters(advanced_config, input_dataset_columns)
        assert "type" in str(double_err.value)
        assert "int" in str(double_err.value)

        advanced_config["seasonal_stl"] = "string"
        with pytest.raises(DSSParameterError) as str_err:
            _ = dku_config.add_parameters(advanced_config, input_dataset_columns)
        assert "type" in str(str_err.value)
        assert "int" in str(str_err.value)

        advanced_config.pop("seasonal_stl")
        with pytest.raises(DSSParameterError) as str_err:
            _ = dku_config.add_parameters(advanced_config, input_dataset_columns)
        assert "required" in str(str_err.value)

    def test_advanced_degrees(self, advanced_config, input_dataset_columns):
        dku_config = STLConfig()
        advanced_config["advanced_params_STL"] = {"seasonal_deg":"1", "trend_deg":"1", "low_pass_deg":"1"}
        dku_config.add_parameters(advanced_config, input_dataset_columns)
        assert dku_config.advanced_params_STL == {"seasonal_deg":"1", "trend_deg":"1", "low_pass_deg":"1"}
        assert dku_config.seasonal_deg == 1
        assert dku_config.trend_deg == 1
        assert dku_config.low_pass_deg == 1

        dku_config = STLConfig()
        advanced_config["advanced_params_STL"] = {"seasonal_deg": "1000"}
        with pytest.raises(DSSParameterError):
            _ = dku_config.add_parameters(advanced_config, input_dataset_columns)

        dku_config = STLConfig()
        advanced_config["advanced_params_STL"] = {"seasonal_deg": "string"}
        with pytest.raises(DSSParameterError):
            _ = dku_config.add_parameters(advanced_config, input_dataset_columns)

    def test_advanced_speed_jumps(self, advanced_config, input_dataset_columns):
        dku_config = STLConfig()
        advanced_config["advanced_params_STL"] = {"seasonal_jump":"2", "trend_jump":"3", "low_pass_jump":"4"}
        dku_config.add_parameters(advanced_config, input_dataset_columns)
        assert dku_config.advanced_params_STL == {"seasonal_jump":"2", "trend_jump":"3", "low_pass_jump":"4"}
        assert dku_config.seasonal_jump == 2
        assert dku_config.trend_jump == 3
        assert dku_config.low_pass_jump == 4

        dku_config = STLConfig()
        advanced_config["advanced_params_STL"] = {"seasonal_jump": "-1000"}
        with pytest.raises(DSSParameterError) as str_err:
            _ = dku_config.add_parameters(advanced_config, input_dataset_columns)
        assert "positive integer" in str(str_err.value)

        dku_config = STLConfig()
        advanced_config["advanced_params_STL"] = {"seasonal_jump": "string"}
        with pytest.raises(DSSParameterError):
            _ = dku_config.add_parameters(advanced_config, input_dataset_columns)

    def test_advanced_smoothers(self, advanced_config, input_dataset_columns):
        dku_config = STLConfig()
        advanced_config["advanced_params_STL"] = {"trend":"23", "low_pass":"31"}
        dku_config.add_parameters(advanced_config, input_dataset_columns)
        assert dku_config.advanced_params_STL == {"trend":"23", "low_pass":"31"}
        assert dku_config.trend == 23
        assert dku_config.low_pass == 31

        dku_config = STLConfig()
        advanced_config["advanced_params_STL"] = {"trend": "2"}
        with pytest.raises(DSSParameterError) as str_err:
            _ = dku_config.add_parameters(advanced_config, input_dataset_columns)
        assert "odd" in str(str_err.value)

        advanced_config["advanced_params_STL"] = {"trend": "-2"}
        with pytest.raises(DSSParameterError):
            _ = dku_config.add_parameters(advanced_config, input_dataset_columns)

        dku_config = STLConfig()
        advanced_config["advanced_params_STL"] = {"trend": "5"}
        with pytest.raises(DSSParameterError):
            _ = dku_config.add_parameters(advanced_config, input_dataset_columns)

        dku_config = STLConfig()
        advanced_config["advanced_params_STL"] = {"trend": "string"}
        with pytest.raises(DSSParameterError):
            _ = dku_config.add_parameters(advanced_config, input_dataset_columns)






