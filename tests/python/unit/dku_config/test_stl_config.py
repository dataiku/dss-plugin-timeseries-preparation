import os
import sys

import pytest

plugin_root = os.path.dirname(os.path.dirname(os.path.dirname((os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))))
sys.path.append(os.path.join(plugin_root, 'python-lib'))

from dku_config.dss_parameter import DSSParameterError
from dku_config.stl_config import STLConfig


@pytest.fixture
def basic_config():
    config = {"frequency_unit": "M", "season_length_M": 12, "time_column": "date",
              "target_columns": ["target"], "long_format": False, "decomposition_model": "multiplicative", "expert": False}
    return config


@pytest.fixture
def advanced_config():
    config = {"time_column": "date", "target_columns": ["target"],
              "frequency_unit": "D", "season_length_D": 7,
              "decomposition_model": "additive", "seasonal_stl": 7, "expert": True, "robust_stl": True,
              "seasonal_degree_stl": "1", "additional_parameters_STL": {}}
    return config


@pytest.fixture
def input_dataset_columns():
    return ["target", "date", "something"]

@pytest.mark.skipif(sys.version_info < (3, 0), reason="requires Python3")
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

    def test_additional_parameters(self, advanced_config, input_dataset_columns):
        dku_config = STLConfig()
        advanced_config["additional_parameters_STL"] = {"invalid_key": ""}
        with pytest.raises(DSSParameterError) as str_err:
            _ = dku_config.add_parameters(advanced_config, input_dataset_columns)
        assert "keys" in str(str_err.value)

        dku_config = STLConfig()
        advanced_config["additional_parameters_STL"] = {"invalid_key": "5"}
        with pytest.raises(DSSParameterError) as str_err:
            _ = dku_config.add_parameters(advanced_config, input_dataset_columns)
        assert "keys" in str(str_err.value)

    def test_advanced_degrees(self, advanced_config, input_dataset_columns):
        dku_config = STLConfig()
        advanced_config["additional_parameters_STL"] = {"seasonal_deg": "1", "trend_deg": "1", "low_pass_deg": "1"}
        dku_config.add_parameters(advanced_config, input_dataset_columns)
        assert dku_config.additional_parameters_STL == {"seasonal_deg": 1, "trend_deg": 1, "low_pass_deg": 1}

        dku_config = STLConfig()
        advanced_config["additional_parameters_STL"] = {"seasonal_deg": "1000"}
        with pytest.raises(ValueError) as str_err:
            _ = dku_config.add_parameters(advanced_config, input_dataset_columns)
        assert "must be equal to 0 or 1" in str(str_err.value)

        dku_config = STLConfig()
        advanced_config["additional_parameters_STL"] = {"seasonal_deg": "string"}
        with pytest.raises(ValueError) as str_err:
            _ = dku_config.add_parameters(advanced_config, input_dataset_columns)
        assert "must be equal to 0 or 1" in str(str_err.value)

        dku_config = STLConfig()
        advanced_config["additional_parameters_STL"] = {"low_pass_deg": ""}
        with pytest.raises(ValueError) as str_err:
            _ = dku_config.add_parameters(advanced_config, input_dataset_columns)
        assert "must be equal to 0 or 1" in str(str_err.value)

    def test_advanced_speed_jumps(self, advanced_config, input_dataset_columns):
        dku_config = STLConfig()
        advanced_config["additional_parameters_STL"] = {"seasonal_jump": "2", "trend_jump": "3", "low_pass_jump": "4"}
        dku_config.add_parameters(advanced_config, input_dataset_columns)
        assert dku_config.additional_parameters_STL == {"seasonal_jump": 2, "trend_jump": 3, "low_pass_jump": 4}

        dku_config = STLConfig()
        advanced_config["additional_parameters_STL"] = {"seasonal_jump": "-1000"}
        with pytest.raises(ValueError) as str_err:
            _ = dku_config.add_parameters(advanced_config, input_dataset_columns)
        assert "positive integer" in str(str_err.value)

        dku_config = STLConfig()
        advanced_config["additional_parameters_STL"] = {"seasonal_jump": "string"}
        with pytest.raises(ValueError):
            _ = dku_config.add_parameters(advanced_config, input_dataset_columns)
        assert "positive integer" in str(str_err.value)

        dku_config = STLConfig()
        advanced_config["additional_parameters_STL"] = {"seasonal_jump": ""}
        with pytest.raises(ValueError):
            _ = dku_config.add_parameters(advanced_config, input_dataset_columns)
        assert "positive integer" in str(str_err.value)

    def test_advanced_smoothers(self, advanced_config, input_dataset_columns):
        dku_config = STLConfig()
        advanced_config["additional_parameters_STL"] = {"trend": "23", "low_pass": "31"}
        dku_config.add_parameters(advanced_config, input_dataset_columns)
        assert dku_config.additional_parameters_STL == {"trend": 23, "low_pass": 31}

        dku_config = STLConfig()
        advanced_config["additional_parameters_STL"] = {"trend": "None", "low_pass": "31"}
        dku_config.add_parameters(advanced_config, input_dataset_columns)
        assert dku_config.additional_parameters_STL == {"trend": None, "low_pass": 31}

        advanced_config["additional_parameters_STL"] = {"trend": ""}
        dku_config.add_parameters(advanced_config, input_dataset_columns)
        assert dku_config.additional_parameters_STL == {"trend": None}

        dku_config = STLConfig()
        advanced_config["additional_parameters_STL"] = {"trend": "2"}
        with pytest.raises(ValueError) as str_err:
            _ = dku_config.add_parameters(advanced_config, input_dataset_columns)
        assert "odd" in str(str_err.value)

        advanced_config["additional_parameters_STL"] = {"trend": "-2"}
        with pytest.raises(ValueError):
            _ = dku_config.add_parameters(advanced_config, input_dataset_columns)
        assert "positive" in str(str_err.value)

        dku_config = STLConfig()
        advanced_config["additional_parameters_STL"] = {"trend": "5"}
        with pytest.raises(ValueError):
            _ = dku_config.add_parameters(advanced_config, input_dataset_columns)
        assert "greater" in str(str_err.value)

        dku_config = STLConfig()
        advanced_config["additional_parameters_STL"] = {"trend": "string"}
        with pytest.raises(ValueError):
            _ = dku_config.add_parameters(advanced_config, input_dataset_columns)
        assert "integer" in str(str_err.value)

    def test_frequencies_with_default_season_lengths(self):
        annual_config = config_with_default_periods("12M")
        assert annual_config.season_length == 4

        quarterly_config = config_with_default_periods("3M")
        assert quarterly_config.season_length == 4

        semiannual_config = config_with_default_periods("6M")
        assert semiannual_config.season_length == 2

        monthly_config = config_with_default_periods("M")
        assert monthly_config.season_length == 12

        weekly_config = config_with_default_periods("W", frequency_end_of_week="WED")
        assert weekly_config.season_length == 52

        b_weekly_config = config_with_default_periods("B")
        assert b_weekly_config.season_length == 5

        hourly_config = config_with_default_periods("H")
        assert hourly_config.season_length == 24

        hourly_config_3 = config_with_default_periods("H", frequency_step_hours=3)
        assert hourly_config_3.season_length == 24

        daily_config = config_with_default_periods("D")
        assert daily_config.season_length == 7

        min_config = config_with_default_periods("min")
        assert min_config.season_length == 60

        min_config_30 = config_with_default_periods("min", frequency_step_minutes=30)
        assert min_config_30.season_length == 60


def config_with_default_periods(freq, frequency_end_of_week=None, frequency_step_hours=None, frequency_step_minutes=None):
    config = {"frequency_unit": freq, "time_column": "date", "target_columns": ["value1"],
              "long_format": False, "decomposition_model": "multiplicative", "expert": False}
    default_season_length = {"12M": 4, "6M": 2, "3M": 4, "M": 12, "W": 52, "D": 7, "B": 5, "H": 24, "min": 60}
    if frequency_end_of_week:
        config["frequency_end_of_week"] = frequency_end_of_week
    elif frequency_step_hours:
        config["frequency_step_hours"] = frequency_step_hours
    elif frequency_step_minutes:
        config["frequency_step_minutes"] = frequency_step_minutes
    input_dataset_columns = ["value1", "value2", "country", "date"]
    config[f"season_length_{freq}"] = default_season_length[freq]
    dku_config = STLConfig()
    dku_config.add_parameters(config, input_dataset_columns)
    return dku_config
