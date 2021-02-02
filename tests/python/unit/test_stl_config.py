import pandas as pd
import pytest

from dku_config.dss_parameter import DSSParameterError
from dku_config.stl_config import STLConfig


class TestSTLConfig:

    def test_load_settings(self):
        dku_config = STLConfig()
        co2 = [315.58, 316.39, 316.79]
        input_df = pd.DataFrame.from_dict(
            {"target": co2, "date": pd.date_range("1-1-1959", periods=len(co2), freq="M")})

        config = {"transformation_type": "seasonal_decomposition", "time_decomposition_method": "STL",
                  "frequency_unit": "M", "seasonal": 13, "time_column": "date", "target_columns": ["target"],
                  "long_format": False, "model_stl": "multiplicative", "seasonal_stl": 7, "expert_stl": False}
        input_dataset_columns = ["date", "target", "value2"]
        dku_config._load_input_parameters(config, input_dataset_columns)
        assert dku_config._load_settings(config, input_df) is None

        negative_co2 = [315.58, 316.39, -316.79]
        negative_input_df = pd.DataFrame.from_dict(
            {"target": negative_co2, "date": pd.date_range("1-1-1959", periods=len(negative_co2), freq="M")})
        with pytest.raises(DSSParameterError):
            _ = dku_config._load_settings(config, negative_input_df)

        config["seasonal_stl"] = 2
        with pytest.raises(DSSParameterError):
            _ = dku_config._load_settings(config, input_df)

        config["seasonal_stl"] = 2.5
        with pytest.raises(DSSParameterError):
            _ = dku_config._load_settings(config, input_df)

        config.pop("seasonal_stl")
        with pytest.raises(DSSParameterError):
            _ = dku_config._load_settings(config, input_df)

    def test_load_advanced_parameters(self):
        dku_config = STLConfig()
        config = {"transformation_type": "seasonal_decomposition", "time_decomposition_method": "STL",
                  "frequency_unit": "D",
                  "model_stl": "additive", "seasonal_stl": "7", "expert_stl": True, "robust_stl": True,
                  "seasonal_degree_stl": "1",
                  "trend_degree_stl": "0", "lowpass_degree_stl": "0", "time_column": "date",
                  "long_format": False, "target_columns": ["target"],
                  "stl_degree_kwargs": {"seasonal_deg": "1", "trend_deg": "", "low_pass_deg": "1"},
                  "stl_speed_jump_kwargs": {"seasonal_jump": '', "trend_jump": "12", "low_pass_jump": ''},
                  "stl_smoothers_kwargs": {"trend": "13", "low_pass": ''}}

        assert dku_config._load_advanced_parameters(config) is None

        config["stl_smoothers_kwargs"]["trend"] = "2"
        with pytest.raises(DSSParameterError):
            _ = dku_config._load_advanced_parameters(config)

        config["stl_smoothers_kwargs"]["trend"] = "3"
        config["stl_degree_kwargs"]["seasonal_deg"] = "2"
        with pytest.raises(DSSParameterError):
            _ = dku_config._load_advanced_parameters(config)
        config["stl_degree_kwargs"]["seasonal_deg"] = "1"

        config["stl_smoothers_kwargs"]["wrong_field"] = "3"
        with pytest.raises(DSSParameterError):
            _ = dku_config._load_advanced_parameters(config)

        config["stl_smoothers_kwargs"].pop("wrong_field")
        config["stl_smoothers_kwargs"]["trend"] = "12"
        with pytest.raises(DSSParameterError):
            _ = dku_config._load_advanced_parameters(config)
