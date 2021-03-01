import pytest

from commons import get_resampling_params


@pytest.fixture
def config():
    config = {u'clip_end': 0, u'constant_value': 0, u'extrapolation_method': u'none', u'shift': 0, u'time_unit_end_of_week': u'SUN',
              u'datetime_column': u'Date', u'advanced_activated': False, u'time_unit': u'quarters', u'clip_start': 0, u'time_step': 2,
              u'interpolation_method': u'linear'}
    return config


class TestResamplerParams:
    def test_quarter_params(self, config):
        params = get_resampling_params(config)
        assert params.time_step == 6
        assert params.resampling_step == "6M"

    def test_semi_annual_params(self, config):
        config["time_unit"] = "semi_annual"
        params = get_resampling_params(config)
        assert params.time_step == 12
        assert params.resampling_step == "12M"
        config["time_step"] = 1.5
        params = get_resampling_params(config)
        assert params.time_step == 9
        assert params.resampling_step == "9M"

    def test_weekly_params(self, config):
        config["time_unit"] = "weeks"
        params = get_resampling_params(config)
        assert params.resampling_step == "2W-SUN"
        config["time_unit_end_of_week"] = "MON"
        params = get_resampling_params(config)
        assert params.time_unit_end_of_week == "MON"
        assert params.resampling_step == "2W-MON"

    def test_b_days_params(self, config):
        config["time_unit"] = "business_days"
        params = get_resampling_params(config)
        assert params.resampling_step == "2B"
        assert params.time_step == 2
