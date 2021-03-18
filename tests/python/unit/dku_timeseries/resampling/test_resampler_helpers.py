import numpy as np
import pandas as pd
import pytest

from dku_timeseries.resampling.resampling import ResamplerParams
from dku_timeseries.timeseries_helpers import generate_date_range, get_date_offset


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


class TestResamplerHelpers:
    def test_date_offset(self):
        time_unit = "business_days"
        offset_value = 0
        sunday = pd.Timestamp('2021-01-31 10:00:00')
        offset = get_date_offset(time_unit, offset_value)
        assert sunday + offset == sunday

        sunday = pd.Timestamp('2021-01-31 00:00:00')
        offset = get_date_offset(time_unit, 1)
        assert sunday + offset == pd.Timestamp('2021-02-01 00:00:00')
        assert sunday - offset == pd.Timestamp('2021-01-29 00:00:00')
        assert sunday + offset + offset == pd.Timestamp('2021-02-02 00:00:00')

        friday = pd.Timestamp('2021-01-29 00:00:00')
        offset = get_date_offset(time_unit, 1)
        assert friday + offset == pd.Timestamp('2021-02-01 00:00:00')

        friday = pd.Timestamp('2021-01-29 00:00:00')
        offset = get_date_offset(time_unit, 2)
        assert friday + offset == pd.Timestamp('2021-02-02 00:00:00')

        saturday = pd.Timestamp('2021-01-30 00:00:00')
        offset = get_date_offset(time_unit, 1)
        assert saturday + offset == pd.Timestamp('2021-02-01 00:00:00')

        saturday = pd.Timestamp('2021-02-04 00:00:00')
        offset = get_date_offset(time_unit, 1)
        assert saturday + offset == pd.Timestamp('2021-02-05 00:00:00')

    def test_generate_date_range_month(self, config):
        config["time_unit"] = "months"
        params = get_params(config)
        frequency = params.resampling_step
        time_unit = params.time_unit
        time_step = params.time_step

        end_time = pd.Timestamp('2021-06-20 00:00:00')

        start_time = pd.Timestamp('2021-01-31 00:00:00')
        date_range = generate_date_range(start_time, end_time, 0, 0, 0, frequency, time_step, time_unit)
        np.testing.assert_array_equal(date_range, pd.DatetimeIndex(['2021-01-31', '2021-03-31', '2021-05-31', '2021-07-31']))

        start_time = pd.Timestamp('2021-01-23 00:00:00')
        date_range = generate_date_range(start_time, end_time, 0, 0, 0, frequency, time_step, time_unit)
        np.testing.assert_array_equal(date_range, pd.DatetimeIndex(['2021-01-31', '2021-03-31', '2021-05-31', '2021-07-31']))

        start_time = pd.Timestamp('2021-01-31 10:00:00')
        date_range = generate_date_range(start_time, end_time, 0, 0, 0, frequency, time_step, time_unit)
        np.testing.assert_array_equal(date_range, pd.DatetimeIndex(['2021-01-31', '2021-03-31', '2021-05-31', '2021-07-31']))

        start_time = pd.Timestamp('2021-01-31 10:00:00').tz_localize("CET")
        end_time = pd.Timestamp('2021-06-20 00:00:00').tz_localize("CET")
        date_range = generate_date_range(start_time, end_time, 0, 0, 0, frequency, time_step, time_unit)
        np.testing.assert_array_equal(date_range, pd.DatetimeIndex(
            ['2021-01-31 00:00:00+01:00', '2021-03-31 00:00:00+02:00', '2021-05-31 00:00:00+02:00', '2021-07-31 00:00:00+02:00']))

        start_time = pd.Timestamp('2021-01-31 10:00:00')
        end_time = pd.Timestamp('2021-06-20 00:00:00')
        date_range = generate_date_range(start_time, end_time, 1, 0, 1, frequency, time_step, time_unit)
        np.testing.assert_array_equal(date_range, pd.DatetimeIndex(['2021-03-31', '2021-05-31', '2021-07-31']))

    def test_generate_date_range_week(self, config):
        config["time_unit"] = "weeks"
        params = get_params(config)
        frequency = params.resampling_step
        time_unit = params.time_unit
        time_step = params.time_step

        start_time = pd.Timestamp('2020-12-23 00:00:00')
        end_time = pd.Timestamp('2021-01-18 00:00:00')

        date_range = generate_date_range(start_time, end_time, 0, 0, 0, frequency, time_step, time_unit)
        np.testing.assert_array_equal(date_range, pd.DatetimeIndex(['2020-12-27', '2021-01-10', '2021-01-24']))

        end_time = pd.Timestamp('2021-01-24 00:00:00')
        date_range = generate_date_range(start_time, end_time, 0, 0, 0, frequency, time_step, time_unit)
        np.testing.assert_array_equal(date_range, pd.DatetimeIndex(['2020-12-27', '2021-01-10', '2021-01-24', '2021-02-07']))

        date_range = generate_date_range(start_time, end_time, 1, 0, 1, frequency, time_step, time_unit)
        np.testing.assert_array_equal(date_range, pd.DatetimeIndex(['2021-01-10', '2021-01-24', '2021-02-07']))

        config["time_unit"] = "weeks"
        config["time_unit_end_of_week"] = "WED"
        params = get_params(config)
        frequency = params.resampling_step
        time_unit = params.time_unit
        time_step = params.time_step
        date_range = generate_date_range(start_time, end_time, 0, 0, 0, frequency, time_step, time_unit)
        np.testing.assert_array_equal(date_range, pd.DatetimeIndex(['2020-12-23', '2021-01-6', '2021-01-20', '2021-02-03']))

    def test_generate_date_range_quarters(self, config):
        config["time_step"] = 1
        config["time_unit"] = "quarters"
        start_time = pd.Timestamp('2020-01-23 00:00:00')
        end_time = pd.Timestamp('2021-01-18 00:00:00')

        params = get_params(config)
        frequency = params.resampling_step
        time_unit = params.time_unit
        time_step = params.time_step

        date_range = generate_date_range(start_time, end_time, 0, 0, 0, frequency, time_step, time_unit)
        np.testing.assert_array_equal(date_range, pd.DatetimeIndex(['2020-01-31', '2020-04-30', '2020-07-31', '2020-10-31', '2021-01-31']))

    def test_generate_date_range_half_year(self, config):
        config["time_step"] = 1
        config["time_unit"] = "semi_annual"
        start_time = pd.Timestamp('2020-01-01 00:00:00')
        end_time = pd.Timestamp('2021-06-18 00:00:00')

        params = get_params(config)
        frequency = params.resampling_step
        time_unit = params.time_unit
        time_step = params.time_step

        date_range = generate_date_range(start_time, end_time, 0, 0, 0, frequency, time_step, time_unit)
        np.testing.assert_array_equal(date_range, pd.DatetimeIndex(['2020-01-31', '2020-07-31', '2021-01-31', '2021-07-31']))

    def test_generate_date_range_b_days(self, config):
        config["time_unit"] = "business_days"
        config["time_step"] = 1
        start_time = pd.Timestamp('2021-01-02 00:00:00')
        end_time = pd.Timestamp('2021-01-10 00:00:00')

        params = get_params(config)
        frequency = params.resampling_step
        time_unit = params.time_unit
        time_step = params.time_step

        date_range = generate_date_range(start_time, end_time, 0, 0, 0, frequency, time_step, time_unit)
        np.testing.assert_array_equal(date_range, pd.DatetimeIndex(['2021-01-04', '2021-01-05', '2021-01-06', '2021-01-07', '2021-01-08', '2021-01-11']))

        clip_start = 1
        clip_end = 1
        shift = 0
        date_range = generate_date_range(start_time, end_time, clip_start, clip_end, shift, frequency, time_step, time_unit)
        np.testing.assert_array_equal(date_range, pd.DatetimeIndex(['2021-01-04', '2021-01-05', '2021-01-06', '2021-01-07', '2021-01-08', '2021-01-11']))

        clip_start = 2
        clip_end = 2
        shift = 0
        date_range = generate_date_range(start_time, end_time, clip_start, clip_end, shift, frequency, time_step, time_unit)
        np.testing.assert_array_equal(date_range, pd.DatetimeIndex(['2021-01-05', '2021-01-06', '2021-01-07', '2021-01-08']))
