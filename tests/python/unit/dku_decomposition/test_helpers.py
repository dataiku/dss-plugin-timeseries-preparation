import pytest

from dku_timeseries.dku_decomposition.helpers import check_and_load_params


@pytest.fixture
def config():
    config = {"transformation_type": "seasonal_decomposition", "time_decomposition_method": "STL",
              "frequency_unit": "M", "season_length_M": 12, "time_column": "date", "target_columns": ["value1", "value2"],
              "long_format": False, "decomposition_model": "multiplicative", "seasonal_stl": 13, "expert": False}
    return config


@pytest.fixture
def columns():
    return ["value1", "value2", "value3", "date"]


class TestHelpers:
    def test_check_and_load_params(self, config, columns):
        (dku_config, input_validator, decomposition) = check_and_load_params(config, columns)
        assert input_validator.dku_config == dku_config == decomposition.dku_config