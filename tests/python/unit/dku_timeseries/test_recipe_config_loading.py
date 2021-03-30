import pytest

from recipe_config_loading import check_and_get_groupby_columns


@pytest.fixture
def config():
    config = {u'clip_end': 0, u'constant_value': 0, u'extrapolation_method': u'clip', u'shift': 0, u'time_unit_end_of_week': u'SUN',
              u'datetime_column': u'Date', u'advanced_activated': True, u"groupby_columns": ["country", "old"], u'time_unit': u'weeks', u'clip_start': 0,
              u'time_step': 2,
              u'interpolation_method': u'linear'}
    return config


@pytest.fixture
def dataset_columns():
    return ["value1", "country", "old"]


class TestRecipeConfigLoading:
    def test_check_and_get_groupby_columns(self, config):
        dataset_columns = ["value1", "country", "old"]
        groupby_colums = check_and_get_groupby_columns(config, dataset_columns)
        assert groupby_colums == ["country", "old"]

        config.pop("groupby_columns")
        config["groupby_column"] = "country"
        groupby_colums = check_and_get_groupby_columns(config, dataset_columns)
        assert groupby_colums == ["country"]

    def test_invalid_groupby_columns(self, config, dataset_columns):
        config["groupby_columns"] = ["country", ""]
        restricted_dataset_columns = ["value1", "country"]
        with pytest.raises(ValueError) as err:
            _ = check_and_get_groupby_columns(config, restricted_dataset_columns)
        assert "Invalid time series identifiers selection" in str(err.value)

        with pytest.raises(ValueError) as err:
            _ = check_and_get_groupby_columns(config, restricted_dataset_columns)
        assert "Invalid time series identifiers selection" in str(err.value)

        config.pop("groupby_columns")
        with pytest.raises(ValueError) as err:
            _ = check_and_get_groupby_columns(config, restricted_dataset_columns)
        assert "Long format is activated but no time series identifiers have been provided" in str(err.value)

        config["groupby_column"] = "not_ok"
        with pytest.raises(ValueError) as err:
            _ = check_and_get_groupby_columns(config, dataset_columns)
        assert "Invalid time series identifiers selection" in str(err.value)

    def test_two_identifiers_fields(self, config, dataset_columns):
        config["groupby_column"] = "notice"
        config["groupby_columns"] = ["country"]
        groupby_colums = check_and_get_groupby_columns(config, dataset_columns)
        assert groupby_colums == ["country"]
