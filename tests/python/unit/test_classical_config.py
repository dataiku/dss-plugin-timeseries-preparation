import pandas as pd
import pytest

from dku_config.dss_parameter import DSSParameterError
from dku_config.classical_config import ClassicalConfig


class TestClassicalConfig:

    def test_load_settings(self):
        dku_config = ClassicalConfig()
        config = {"transformation_type": "seasonal_decomposition", "time_decomposition_method": "classical",
                  "classical_model": "additive", "time_column": "date", "target_columns": ["target"],
                  "frequency_unit": "D"}
        input_dataset_columns = ["date", "target", "value2"]
        dku_config._load_input_parameters(config, input_dataset_columns)
        co2 = [315.58, 316.39, 316.79]
        input_df = pd.DataFrame.from_dict(
            {"target": co2, "date": pd.date_range("1-1-1959", periods=len(co2), freq="M")})
        assert dku_config._load_settings(config, input_df) is None

        config.pop("classical_model")
        assert dku_config._load_settings(config, input_df) is None

        config["classical_model"] = "multiplicative"
        assert dku_config._load_settings(config, input_df) is None

        negative_co2 = [315.58, 316.39, -316.79]
        negative_input_df = pd.DataFrame.from_dict(
            {"target": negative_co2, "date": pd.date_range("1-1-1959", periods=len(negative_co2), freq="M")})
        with pytest.raises(DSSParameterError):
            _ = dku_config._load_settings(config, negative_input_df)

    def test_load_advanced_parameters(self):
        dku_config = ClassicalConfig()
        config = {"transformation_type": "seasonal_decomposition", "time_decomposition_method": "classical",
                  "classical_model": "additive", "time_column": "date", "target_columns": ["target"],
                  "frequency_unit": "D", "expert_classical": True,
                  "advanced_params_classical": {"filt": "[1,2,3]", "two_sided": "False","extrapolate_trend":""}}
        assert dku_config._load_advanced_parameters(config) is None

        config["advanced_params_classical"]["extrapolate_trend"] = "freq"
        assert dku_config._load_advanced_parameters(config) is None

        config["advanced_params_classical"]["extrapolate_trend"] = "2"
        assert dku_config._load_advanced_parameters(config) is None

        config["advanced_params_classical"]["wrong_key"] = "2"
        with pytest.raises(DSSParameterError):
            _ = dku_config._load_advanced_parameters(config)

        config["advanced_params_classical"]["filt"] = "4"
        with pytest.raises(DSSParameterError):
            _ = dku_config._load_advanced_parameters(config)

