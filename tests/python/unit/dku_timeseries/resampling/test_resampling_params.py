import pytest

from dku_timeseries import Resampler, ResamplerParams


@pytest.fixture
def config():
    config = {u'clip_end': 0, u'constant_value': 0, u'extrapolation_method': u'none', u'shift': 0, u'time_unit_end_of_week': u'SUN',
              u'datetime_column': u'Date', u'advanced_activated': False, u'time_unit': u'quarters', u'clip_start': 0, u'time_step': 2,
              u'interpolation_method': u'linear'}
    return config


def get_params(config):
    def _p(param_name, default=None):
        return config.get(param_name, default)

    interpolation_method = _p('interpolation_method')
    extrapolation_method = _p('extrapolation_method')
    constant_value = _p('constant_value')
    time_step = _p('time_step')
    time_unit = _p('time_unit')
    time_unit_end_of_week = _p('time_unit_end_of_week')

    params = ResamplerParams(interpolation_method=interpolation_method,
                             extrapolation_method=extrapolation_method,
                             constant_value=constant_value,
                             time_step=time_step,
                             time_unit=time_unit,
                             time_unit_end_of_week=time_unit_end_of_week)
    return params


class TestResamplerParams:
    def test_invalid_time_step(self, config):
        config.pop("time_step")
        with pytest.raises(ValueError) as err:
            _ = get_params(config)
        assert "Invalid time step" in str(err.value)

        config["time_step"] = 0
        params = get_params(config)
        with pytest.raises(ValueError) as err:
            _ = Resampler(params)
        assert "Time step can not be null or negative" in str(err.value)

    def test_quarter_params(self, config):
        params = get_params(config)
        assert params.time_step == 6
        assert params.resampling_step == "6M"

    def test_semi_annual_params(self, config):
        config["time_unit"] = "semi_annual"
        params = get_params(config)
        assert params.time_step == 12
        assert params.resampling_step == "12M"
        config["time_step"] = 1.5
        params = get_params(config)
        assert params.time_step == 9
        assert params.resampling_step == "9M"

    def test_weekly_params(self, config):
        config["time_unit"] = "weeks"
        params = get_params(config)
        assert params.resampling_step == "2W-SUN"
        config["time_unit_end_of_week"] = "MON"
        params = get_params(config)
        assert params.time_unit_end_of_week == "MON"
        assert params.resampling_step == "2W-MON"

    def test_b_days_params(self, config):
        config["time_unit"] = "business_days"
        params = get_params(config)
        assert params.resampling_step == "2B"
        assert params.time_step == 2



