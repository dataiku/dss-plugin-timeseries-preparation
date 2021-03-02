import numpy as np
import pandas as pd
import pytest

from commons import get_resampling_params
from dku_timeseries.timeseries_helpers import generate_date_range, get_period_end_date


@pytest.fixture
def config():
    config = {u'clip_end': 0, u'constant_value': 0, u'extrapolation_method': u'none', u'shift': 0, u'time_unit_end_of_week': u'SUN',
              u'datetime_column': u'Date', u'advanced_activated': False, u'time_unit': u'quarters', u'clip_start': 0, u'time_step': 2,
              u'interpolation_method': u'linear'}
    return config


class TestResamplerHelpers:
    def test_period_end_date(self, config):
        frequency = "M"
        middle_month = pd.Timestamp('2021-01-23 00:00:00')
        period_end_middle_month = get_period_end_date(middle_month, frequency)
        assert period_end_middle_month == pd.Timestamp('2021-01-31 00:00:00')
        end_month = pd.Timestamp('2021-01-31 00:00:00')
        period_end_month = get_period_end_date(end_month, frequency)
        assert period_end_month == pd.Timestamp('2021-01-31 00:00:00')
        tz_time = pd.Timestamp('2021-01-23 00:00:00').tz_localize('CET')
        period_end = get_period_end_date(tz_time, frequency)
        assert period_end == pd.Timestamp('2021-01-31 00:00:00').tz_localize('CET')

        frequency = "B"
        saturday = pd.Timestamp('2021-01-23 00:00:00').tz_localize('CET')
        period_end = get_period_end_date(saturday, frequency)
        assert period_end == pd.Timestamp('2021-01-25 00:00:00').tz_localize('CET')
        friday = pd.Timestamp('2021-01-22 00:00:00').tz_localize('CET')
        period_end = get_period_end_date(friday, frequency)
        assert period_end == pd.Timestamp('2021-01-22 00:00:00').tz_localize('CET')

        frequency = "6M"
        time = pd.Timestamp('2021-01-23 00:00:00').tz_localize('CET')
        period_end = get_period_end_date(time, frequency)
        assert period_end == pd.Timestamp('2021-06-30 00:00:00').tz_localize('CET')

        frequency = "3M"
        time = pd.Timestamp('2021-01-23 00:00:00').tz_localize('CET')
        period_end = get_period_end_date(time, frequency)
        assert period_end == pd.Timestamp('2021-03-31 00:00:00').tz_localize('CET')

        frequency = "Y"
        time = pd.Timestamp('2021-01-23 00:00:00').tz_localize('CET')
        period_end = get_period_end_date(time, frequency)
        assert period_end == pd.Timestamp('2021-12-31 00:00:00').tz_localize('CET')

    def test_generate_date_range_month(self, config):
        config["time_unit"] = "months"
        params = get_resampling_params(config)
        frequency = params.resampling_step
        time_unit = params.time_unit

        end_time = pd.Timestamp('2021-06-20 00:00:00')

        extrapolation_method = "none"
        start_time = pd.Timestamp('2021-01-31 00:00:00')
        date_range = generate_date_range(start_time, end_time, 0, 0, 0, frequency, time_unit, extrapolation_method)
        np.testing.assert_array_equal(date_range, pd.DatetimeIndex(['2021-01-31', '2021-03-31', '2021-05-31']))

        start_time = pd.Timestamp('2021-01-23 00:00:00')
        date_range = generate_date_range(start_time, end_time, 0, 0, 0, frequency, time_unit, extrapolation_method)
        np.testing.assert_array_equal(date_range, pd.DatetimeIndex(['2021-01-31', '2021-03-31', '2021-05-31']))

        start_time = pd.Timestamp('2021-01-31 10:00:00')
        date_range = generate_date_range(start_time, end_time, 0, 0, 0, frequency, time_unit, extrapolation_method)
        np.testing.assert_array_equal(date_range, pd.DatetimeIndex(['2021-03-31', '2021-05-31']))

        extrapolation_method = "clip"
        date_range_extrapolation = generate_date_range(start_time, end_time, 0, 0, 0, frequency, time_unit, extrapolation_method)
        np.testing.assert_array_equal(date_range_extrapolation, pd.DatetimeIndex(['2021-01-31', '2021-03-31', '2021-05-31', '2021-07-31']))

    def test_generate_date_range_week(self, config):
        config["time_unit"] = "weeks"
        params = get_resampling_params(config)
        frequency = params.resampling_step
        time_unit = params.time_unit

        start_time = pd.Timestamp('2020-12-23 00:00:00')
        end_time = pd.Timestamp('2021-01-18 00:00:00')

        #extrapolation_method = "none"
        #date_range = generate_date_range(start_time, end_time, 0, 0, 0, frequency,  time_unit, extrapolation_method)
        #np.testing.assert_array_equal(date_range, pd.DatetimeIndex(['2021-01-03', '2021-01-17']))

        extrapolation_method = "clip"
        date_range_extrapolation = generate_date_range(start_time, end_time, 0, 0, 0, frequency,  time_unit, extrapolation_method)
        np.testing.assert_array_equal(date_range_extrapolation, pd.DatetimeIndex(['2020-12-27', '2021-01-10', '2021-01-24']))

        end_time = pd.Timestamp('2021-01-24 00:00:00')
        date_range_extrapolation = generate_date_range(start_time, end_time, 0, 0, 0, frequency,  time_unit, extrapolation_method)
        np.testing.assert_array_equal(date_range_extrapolation, pd.DatetimeIndex(['2020-12-27', '2021-01-10', '2021-01-24']))

    def test_generate_date_range_quarters(self, config):
        config["time_step"] = 1
        config["time_unit"] = "quarters"
        start_time = pd.Timestamp('2020-01-23 00:00:00')
        end_time = pd.Timestamp('2021-01-18 00:00:00')

        params = get_resampling_params(config)
        frequency = params.resampling_step
        time_unit = params.time_unit

        extrapolation_method = "none"
        date_range = generate_date_range(start_time, end_time, 0, 0, 0, frequency,  time_unit, extrapolation_method)
        np.testing.assert_array_equal(date_range, pd.DatetimeIndex(['2020-01-31', '2020-04-30', '2020-07-31', '2020-10-31']))

        extrapolation_method = "clip"
        date_range = generate_date_range(start_time, end_time, 0, 0, 0, frequency,  time_unit, extrapolation_method)
        np.testing.assert_array_equal(date_range, pd.DatetimeIndex(['2020-01-31', '2020-04-30', '2020-07-31', '2020-10-31','2021-01-31']))

    def test_generate_date_range_half_year(self, config):
        config["time_step"] = 1
        config["time_unit"] = "semi_annual"
        start_time = pd.Timestamp('2020-01-01 00:00:00')
        end_time = pd.Timestamp('2021-06-18 00:00:00')

        params = get_resampling_params(config)
        frequency = params.resampling_step
        time_unit = params.time_unit

        extrapolation_method = "none"
        date_range = generate_date_range(start_time, end_time, 0, 0, 0, frequency,  time_unit, extrapolation_method)
        np.testing.assert_array_equal(date_range, pd.DatetimeIndex(['2020-01-31','2020-07-31', '2021-01-31']))
        date_range_year = generate_date_range(start_time, end_time, 0, 0, 0, frequency,"years", extrapolation_method)
        np.testing.assert_array_equal(date_range, date_range_year)

        extrapolation_method = "clip"
        date_range = generate_date_range(start_time, end_time, 0, 0, 0, frequency,  time_unit, extrapolation_method)
        np.testing.assert_array_equal(date_range, pd.DatetimeIndex(['2020-01-31', '2020-07-31', '2021-01-31', '2021-07-31']))

    def test_generate_date_range_b_days(self, config):
        config["time_unit"] = "business_days"
        config["time_step"] = 1
        start_time = pd.Timestamp('2021-01-02 00:00:00')
        end_time = pd.Timestamp('2021-01-10 00:00:00')

        params = get_resampling_params(config)
        frequency = params.resampling_step
        time_unit = params.time_unit

        extrapolation_method = "none"
        date_range = generate_date_range(start_time, end_time, 0, 0, 0, frequency,  time_unit, extrapolation_method)
        np.testing.assert_array_equal(date_range, pd.DatetimeIndex(['2021-01-04', '2021-01-05', '2021-01-06', '2021-01-07', '2021-01-08']))

    def test_generate_date_range_end_of_period(self, config):
        extrapolation_method = "clip"
        start_time = pd.Timestamp('2021-01-23 00:00:00')
        end_time = pd.Timestamp('2021-06-30 00:00:00')

        config["time_unit"] = "months"
        params = get_resampling_params(config)
        frequency = params.resampling_step
        time_unit = params.time_unit
        date_range = generate_date_range(start_time, end_time, 0, 0, 0, frequency,  time_unit, extrapolation_method)
        np.testing.assert_array_equal(date_range, pd.DatetimeIndex(['2021-01-31', '2021-03-31', '2021-05-31', '2021-07-31']))

        config["time_unit"] = "weeks"
        end_time = pd.Timestamp('2021-02-07 00:00:00')
        params = get_resampling_params(config)
        frequency = params.resampling_step
        time_unit = params.time_unit
        date_range = generate_date_range(start_time, end_time, 0, 0, 0, frequency,  time_unit, extrapolation_method)
        np.testing.assert_array_equal(date_range, pd.DatetimeIndex(['2021-01-24', '2021-02-07']))

        config["time_unit"] = "days"
        end_time = pd.Timestamp('2021-01-24 12:00:00')
        params = get_resampling_params(config)
        frequency = params.resampling_step
        time_unit = params.time_unit
        date_range = generate_date_range(start_time, end_time, 0, 0, 0, frequency,  time_unit, extrapolation_method)
        np.testing.assert_array_equal(date_range, pd.DatetimeIndex(['2021-01-23', '2021-01-25']))
