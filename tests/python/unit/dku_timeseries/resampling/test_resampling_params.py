import pytest

from dku_timeseries import Resampler, ResamplerParams


@pytest.fixture
def config():
    config = {u'clip_end': 0, u'constant_value': 0, u'extrapolation_method': u'clip', u'shift': 0, u'time_unit_end_of_week': u'SUN',
              u'datetime_column': u'Date', u'advanced_activated': True, u"groupby_columns": ["country"], u'time_unit': u'weeks', u'clip_start': 0,
              u'time_step': 2,
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

    params = ResamplerParams(interpolation_method=interpolation_method,
                             extrapolation_method=extrapolation_method,
                             constant_value=constant_value,
                             time_step=time_step,
                             time_unit=time_unit)
    return params

class TestResamplerParams():
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

