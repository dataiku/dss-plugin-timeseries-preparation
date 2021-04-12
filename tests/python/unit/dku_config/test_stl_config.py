import pytest

from dku_config.dss_parameter import DSSParameterError
from dku_config.stl_config import STLConfig


@pytest.fixture
def basic_config():
    config = {"transformation_type": "seasonal_decomposition", "time_decomposition_method": "STL",
              "frequency_unit": "M", "season_length_M": 12, "time_column": "date", "target_columns": ["target"],
              "long_format": False, "decomposition_model": "multiplicative", "expert": False}
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

    def test_missing_values(self, advanced_config, input_dataset_columns):
        dku_config = STLConfig()
        advanced_config.pop("seasonal_stl")
        with pytest.raises(DSSParameterError) as str_err:
            _ = dku_config.add_parameters(advanced_config, input_dataset_columns)
        assert "required" in str(str_err.value)

    def test_advanced_parameters(self, advanced_config, input_dataset_columns):
        dku_config = STLConfig()
        assert dku_config.add_parameters(advanced_config, input_dataset_columns) is None
        assert dku_config.get_param("loess_degrees") is not None

    def test_missing_loess_degrees(self, advanced_config, input_dataset_columns):
        dku_config = STLConfig()
        advanced_config["stl_degree_kwargs"] = {}
        dku_config.add_parameters(advanced_config, input_dataset_columns)
        assert dku_config.get_param("loess_degrees") is None
        advanced_config.pop("stl_degree_kwargs")
        dku_config.add_parameters(advanced_config, input_dataset_columns)
        assert dku_config.get_param("loess_degrees") is None

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

    def test_frequencies_with_default_season_lengths(self):
        annual_config = config_with_default_periods("12M")
        assert annual_config.period == 4

        quarterly_config = config_with_default_periods("3M")
        assert quarterly_config.period == 4

        semiannual_config = config_with_default_periods("6M")
        assert semiannual_config.period == 2

        monthly_config = config_with_default_periods("M")
        assert monthly_config.period == 12

        weekly_config = config_with_default_periods("W", frequency_end_of_week="WED")
        assert weekly_config.period == 52

        b_weekly_config = config_with_default_periods("B")
        assert b_weekly_config.period == 5

        hourly_config = config_with_default_periods("H")
        assert hourly_config.period == 24

        hourly_config_3 = config_with_default_periods("H", frequency_step_hours=3)
        assert hourly_config_3.period == 24

        daily_config = config_with_default_periods("D")
        assert daily_config.period == 7

        min_config = config_with_default_periods("min")
        assert min_config.period == 60

        min_config_30 = config_with_default_periods("min", frequency_step_minutes=30)
        assert min_config_30.period == 60


def config_with_default_periods(freq, frequency_end_of_week=None, frequency_step_hours=None, frequency_step_minutes=None):
    config = {"transformation_type": "seasonal_decomposition", "time_decomposition_method": "STL",
              "frequency_unit": freq, "time_column": "date", "target_columns": ["value1"],
              "long_format": False, "decomposition_model": "multiplicative", "expert": False}
    default_season_length = {"12M": 4, "6M":2, "3M": 4, "M": 12, "W": 52, "D": 7, "B": 5, "H": 24, "min": 60}
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
