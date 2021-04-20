import pytest

from dku_config.decomposition_config import DecompositionConfig
from dku_config.dss_parameter import DSSParameterError


@pytest.fixture
def basic_config():
    config = {"time_decomposition_method": "STL", "frequency_unit": "M", "season_length_M": 12, "time_column": "date",
              "target_columns": ["value1", "value2"],
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

    def test_missing_season_length(self, input_dataset_columns):
        annual_config = config_missing_season_length("12M")
        dku_config = DecompositionConfig()
        with pytest.raises(DSSParameterError) as in_err:
            _ = dku_config.add_parameters(annual_config, input_dataset_columns)
        assert "required" in str(in_err.value)

        quarterly_config = config_missing_season_length("3M")
        with pytest.raises(DSSParameterError) as in_err:
            _ = dku_config.add_parameters(quarterly_config, input_dataset_columns)
        assert "required" in str(in_err.value)

        semiannual_config = config_missing_season_length("6M")
        with pytest.raises(DSSParameterError) as in_err:
            _ = dku_config.add_parameters(semiannual_config, input_dataset_columns)
        assert "required" in str(in_err.value)

        monthly_config = config_missing_season_length("M")
        with pytest.raises(DSSParameterError) as in_err:
            _ = dku_config.add_parameters(monthly_config, input_dataset_columns)
        assert "required" in str(in_err.value)

        weekly_config = config_missing_season_length("W")
        with pytest.raises(DSSParameterError) as in_err:
            _ = dku_config.add_parameters(weekly_config, input_dataset_columns)
        assert "required" in str(in_err.value)

        weekly_config_monday = config_missing_season_length("W", frequency_end_of_week="MON")
        with pytest.raises(DSSParameterError) as in_err:
            _ = dku_config.add_parameters(weekly_config_monday, input_dataset_columns)
        assert "required" in str(in_err.value)

        b_weekly_config = config_missing_season_length("B")
        with pytest.raises(DSSParameterError) as in_err:
            _ = dku_config.add_parameters(b_weekly_config, input_dataset_columns)
        assert "required" in str(in_err.value)

        hourly_config = config_missing_season_length("H")
        with pytest.raises(DSSParameterError) as in_err:
            _ = dku_config.add_parameters(hourly_config, input_dataset_columns)
        assert "required" in str(in_err.value)

        hourly_config_3 = config_missing_season_length("H", frequency_step_hours=3)
        with pytest.raises(DSSParameterError) as in_err:
            _ = dku_config.add_parameters(hourly_config_3, input_dataset_columns)
        assert "required" in str(in_err.value)

        daily_config = config_missing_season_length("D")
        with pytest.raises(DSSParameterError) as in_err:
            _ = dku_config.add_parameters(daily_config, input_dataset_columns)
        assert "required" in str(in_err.value)

        min_config = config_missing_season_length("min")
        with pytest.raises(DSSParameterError) as in_err:
            _ = dku_config.add_parameters(min_config, input_dataset_columns)
        assert "required" in str(in_err.value)

        min_config_30 = config_missing_season_length("min", frequency_step_minutes=30)
        with pytest.raises(DSSParameterError) as in_err:
            _ = dku_config.add_parameters(min_config_30, input_dataset_columns)
        assert "required" in str(in_err.value)

    def test_frequencies_with_constant_season_lengths(self):
        annual_config = config_with_constant_season_length("12M")
        assert annual_config.season_length == 5

        quarterly_config = config_with_constant_season_length("3M")
        assert quarterly_config.season_length == 5

        semiannual_config = config_with_constant_season_length("6M")
        assert semiannual_config.season_length == 5

        monthly_config = config_with_constant_season_length("M")
        assert monthly_config.season_length == 5

        weekly_config = config_with_constant_season_length("W", frequency_end_of_week="WED")
        assert weekly_config.season_length == 5

        b_weekly_config = config_with_constant_season_length("B")
        assert b_weekly_config.season_length == 5

        hourly_config = config_with_constant_season_length("H")
        assert hourly_config.season_length == 5

        hourly_config_3 = config_with_constant_season_length("H", frequency_step_hours=3)
        assert hourly_config_3.season_length == 5

        daily_config = config_with_constant_season_length("D")
        assert daily_config.season_length == 5

        min_config = config_with_constant_season_length("min")
        assert min_config.season_length == 5

        min_config_30 = config_with_constant_season_length("min", frequency_step_minutes=30)
        assert min_config_30.season_length == 5

    def test_frequencies_with_default_season_lengths(self):
        annual_config = config_with_default_season_length("12M")
        assert annual_config.season_length == 4

        quarterly_config = config_with_default_season_length("3M")
        assert quarterly_config.season_length == 4

        semiannual_config = config_with_default_season_length("6M")
        assert semiannual_config.season_length == 2

        monthly_config = config_with_default_season_length("M")
        assert monthly_config.season_length == 12

        weekly_config = config_with_default_season_length("W", frequency_end_of_week="WED")
        assert weekly_config.season_length == 52

        b_weekly_config = config_with_default_season_length("B")
        assert b_weekly_config.season_length == 5

        hourly_config = config_with_default_season_length("H")
        assert hourly_config.season_length == 24

        hourly_config_3 = config_with_default_season_length("H", frequency_step_hours=3)
        assert hourly_config_3.season_length == 24

        daily_config = config_with_default_season_length("D")
        assert daily_config.season_length == 7

        min_config = config_with_default_season_length("min")
        assert min_config.season_length == 60

        min_config_30 = config_with_default_season_length("min", frequency_step_minutes=30)
        assert min_config_30.season_length == 60

    def test_invalid_season_length(self, basic_config, input_dataset_columns):
        dku_config = DecompositionConfig()
        basic_config["season_length_M"] = 0
        with pytest.raises(DSSParameterError) as err:
            _ = dku_config.add_parameters(basic_config, input_dataset_columns)
        assert "season_length" in str(err.value)
        assert "greater than or equal to 1" in str(err.value)

        dku_config = DecompositionConfig()
        basic_config["season_length_M"] = -3
        with pytest.raises(DSSParameterError) as err:
            _ = dku_config.add_parameters(basic_config, input_dataset_columns)
        assert "season_length" in str(err.value)
        assert "greater than or equal to 1" in str(err.value)

        dku_config = DecompositionConfig()
        basic_config["season_length_M"] = "string"
        with pytest.raises(DSSParameterError) as err:
            _ = dku_config.add_parameters(basic_config, input_dataset_columns)
        assert "season_length" in str(err.value)
        assert "int" in str(err.value)


def config_missing_season_length(freq, frequency_end_of_week=None, frequency_step_hours=None, frequency_step_minutes=None):
    config = {"transformation_type": "seasonal_decomposition", "time_decomposition_method": "STL",
              "frequency_unit": freq, "time_column": "date", "target_columns": ["value1"],
              "long_format": False, "decomposition_model": "multiplicative", "expert": False}
    if frequency_end_of_week:
        config["frequency_end_of_week"] = frequency_end_of_week
    elif frequency_step_hours:
        config["frequency_step_hours"] = frequency_step_hours
    elif frequency_step_minutes:
        config["frequency_step_minutes"] = frequency_step_minutes
    return config


def config_with_constant_season_length(freq, frequency_end_of_week=None, frequency_step_hours=None, frequency_step_minutes=None):
    config = {"transformation_type": "seasonal_decomposition", "time_decomposition_method": "STL",
              "frequency_unit": freq, "time_column": "date", "target_columns": ["value1"],
              "long_format": False, "decomposition_model": "multiplicative", "expert": False}
    if frequency_end_of_week:
        config["frequency_end_of_week"] = frequency_end_of_week
    elif frequency_step_hours:
        config["frequency_step_hours"] = frequency_step_hours
    elif frequency_step_minutes:
        config["frequency_step_minutes"] = frequency_step_minutes
    input_dataset_columns = ["value1", "value2", "country", "date"]
    config[f"season_length_{freq}"] = 5
    dku_config = DecompositionConfig()
    dku_config.add_parameters(config, input_dataset_columns)
    return dku_config


def config_with_default_season_length(freq, frequency_end_of_week=None, frequency_step_hours=None, frequency_step_minutes=None):
    config = {"transformation_type": "seasonal_decomposition", "time_decomposition_method": "STL",
              "frequency_unit": freq, "time_column": "date", "target_columns": ["value1"],
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
    dku_config = DecompositionConfig()
    dku_config.add_parameters(config, input_dataset_columns)
    return dku_config
