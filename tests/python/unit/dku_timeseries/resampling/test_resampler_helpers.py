import numpy as np
import pandas as pd
import pytest

from dku_timeseries.timeseries_helpers import generate_date_range, get_date_offset
from recipe_config_loading import get_resampling_params


@pytest.fixture
def config():
    config = {u'clip_end': 0, u'constant_value': 0, u'extrapolation_method': u'none', u'shift': 0, u'time_unit_end_of_week': u'SUN',
              u'datetime_column': u'Date', u'advanced_activated': False, u'time_unit': u'quarters', u'clip_start': 0, u'time_step': 2,
              u'interpolation_method': u'linear'}
    return config


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
        params = get_resampling_params(config)
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
        params = get_resampling_params(config)
        frequency = params.resampling_step
        time_unit = params.time_unit
        time_step = params.time_step

        start_time = pd.Timestamp('2020-12-23 00:00:00')
        end_time = pd.Timestamp('2021-01-18 00:00:00')

        date_range = generate_date_range(start_time, end_time, 0, 0, 0, frequency, time_step, time_unit)
        np.testing.assert_array_equal(date_range, pd.DatetimeIndex(['2020-12-27', '2021-01-10', '2021-01-24']))

        end_time = pd.Timestamp('2021-01-24 00:00:00')
        date_range = generate_date_range(start_time, end_time, 0, 0, 0, frequency, time_step, time_unit)
        np.testing.assert_array_equal(date_range, pd.DatetimeIndex(['2020-12-27', '2021-01-10', '2021-01-24']))

        date_range = generate_date_range(start_time, end_time, 1, 0, 1, frequency, time_step, time_unit)
        np.testing.assert_array_equal(date_range, pd.DatetimeIndex(['2021-01-10', '2021-01-24', '2021-02-07']))

        config["time_unit"] = "weeks"
        config["time_unit_end_of_week"] = "WED"
        params = get_resampling_params(config)
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

        params = get_resampling_params(config)
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

        params = get_resampling_params(config)
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

        params = get_resampling_params(config)
        frequency = params.resampling_step
        time_unit = params.time_unit
        time_step = params.time_step

        date_range = generate_date_range(start_time, end_time, 0, 0, 0, frequency, time_step, time_unit)
        np.testing.assert_array_equal(date_range, pd.DatetimeIndex(['2021-01-04', '2021-01-05', '2021-01-06', '2021-01-07', '2021-01-08']))

        clip_start = 1
        clip_end = 1
        shift = 0
        date_range = generate_date_range(start_time, end_time, clip_start, clip_end, shift, frequency, time_step, time_unit)
        np.testing.assert_array_equal(date_range, pd.DatetimeIndex(['2021-01-05', '2021-01-06', '2021-01-07']))

        clip_start = 2
        clip_end = 2
        shift = 0
        date_range = generate_date_range(start_time, end_time, clip_start, clip_end, shift, frequency, time_step, time_unit)
        np.testing.assert_array_equal(date_range, pd.DatetimeIndex(['2021-01-06']))

    def test_generate_date_range_days(self, config):
        config["time_unit"] = "days"
        config["time_step"] = 1
        start_time = pd.Timestamp('20190131 01:59:00').tz_localize('CET')
        end_time = pd.Timestamp('20190214 01:59:00').tz_localize('CET')

        params = get_resampling_params(config)
        frequency = params.resampling_step
        time_unit = params.time_unit
        time_step = params.time_step

        clip_start = 5
        shift = 2
        clip_end = 3

        date_range = generate_date_range(start_time, end_time, clip_start, clip_end, shift, frequency, time_step, time_unit)
        expected_range = pd.DatetimeIndex(['2019-02-07 00:00:00+01:00', '2019-02-08 00:00:00+01:00',
                                           '2019-02-09 00:00:00+01:00', '2019-02-10 00:00:00+01:00',
                                           '2019-02-11 00:00:00+01:00', '2019-02-12 00:00:00+01:00',
                                           '2019-02-13 00:00:00+01:00'])
        np.testing.assert_array_equal(date_range, expected_range)

    def test_generate_date_range_hours(self, config):
        config["time_unit"] = "hours"
        config["time_step"] = 1
        start_time = pd.Timestamp('20190131 01:59:00').tz_localize('CET')
        end_time = pd.Timestamp('20190131 11:59:00').tz_localize('CET')

        params = get_resampling_params(config)
        frequency = params.resampling_step
        time_unit = params.time_unit
        time_step = params.time_step

        clip_start = 5
        shift = 2
        clip_end = 3

        date_range = generate_date_range(start_time, end_time, clip_start, clip_end, shift, frequency, time_step, time_unit)
        expected_range = pd.DatetimeIndex(['2019-01-31 09:00:00+01:00', '2019-01-31 10:00:00+01:00',
                                           '2019-01-31 11:00:00+01:00'])
        np.testing.assert_array_equal(date_range, expected_range)

    def test_generate_date_range_minutes(self, config):
        config["time_unit"] = "minutes"
        config["time_step"] = 1
        start_time = pd.Timestamp('20190131 01:59:00').tz_localize('CET')
        end_time = pd.Timestamp('20190131 02:15:00').tz_localize('CET')

        params = get_resampling_params(config)
        frequency = params.resampling_step
        time_unit = params.time_unit
        time_step = params.time_step

        clip_start = 5
        shift = 2
        clip_end = 3

        date_range = generate_date_range(start_time, end_time, clip_start, clip_end, shift, frequency, time_step, time_unit)
        expected_range = pd.DatetimeIndex(['2019-01-31 02:06:00+01:00', '2019-01-31 02:07:00+01:00',
                                           '2019-01-31 02:08:00+01:00', '2019-01-31 02:09:00+01:00',
                                           '2019-01-31 02:10:00+01:00', '2019-01-31 02:11:00+01:00',
                                           '2019-01-31 02:12:00+01:00', '2019-01-31 02:13:00+01:00',
                                           '2019-01-31 02:14:00+01:00'])
        np.testing.assert_array_equal(date_range, expected_range)

    def test_generate_date_range_seconds(self, config):
        config["time_unit"] = "seconds"
        config["time_step"] = 1
        start_time = pd.Timestamp('20190131 01:59:00').tz_localize('CET')
        end_time = pd.Timestamp('20190131 01:59:12').tz_localize('CET')

        params = get_resampling_params(config)
        frequency = params.resampling_step
        time_unit = params.time_unit
        time_step = params.time_step

        clip_start = 5
        shift = 2
        clip_end = 3

        date_range = generate_date_range(start_time, end_time, clip_start, clip_end, shift, frequency, time_step, time_unit)
        expected_range = pd.DatetimeIndex(['2019-01-31 01:59:07+01:00', '2019-01-31 01:59:08+01:00',
                                           '2019-01-31 01:59:09+01:00', '2019-01-31 01:59:10+01:00',
                                           '2019-01-31 01:59:11+01:00'])
        np.testing.assert_array_equal(date_range, expected_range)

    def test_generate_date_range_milliseconds(self, config):
        config["time_unit"] = "milliseconds"
        config["time_step"] = 1
        start_time = pd.Timestamp('20190131 01:59:00').tz_localize('CET')
        end_time = pd.Timestamp('2019-01-31 01:59:00.015000').tz_localize('CET')

        params = get_resampling_params(config)
        frequency = params.resampling_step
        time_unit = params.time_unit
        time_step = params.time_step

        clip_start = 5
        shift = 2
        clip_end = 3

        date_range = generate_date_range(start_time, end_time, clip_start, clip_end, shift, frequency, time_step, time_unit)
        expected_range = pd.DatetimeIndex(['2019-01-31 01:59:00.007000+01:00',
                                           '2019-01-31 01:59:00.008000+01:00',
                                           '2019-01-31 01:59:00.009000+01:00',
                                           '2019-01-31 01:59:00.010000+01:00',
                                           '2019-01-31 01:59:00.011000+01:00',
                                           '2019-01-31 01:59:00.012000+01:00',
                                           '2019-01-31 01:59:00.013000+01:00',
                                           '2019-01-31 01:59:00.014000+01:00'])
        np.testing.assert_array_equal(date_range, expected_range)

    def test_generate_date_range_microseconds(self, config):
        config["time_unit"] = "microseconds"
        config["time_step"] = 1
        start_time = pd.Timestamp('20190131 01:59:00').tz_localize('CET')
        end_time = pd.Timestamp('2019-01-31 01:59:00.000016').tz_localize('CET')

        params = get_resampling_params(config)
        frequency = params.resampling_step
        time_unit = params.time_unit
        time_step = params.time_step

        clip_start = 5
        shift = 2
        clip_end = 3

        date_range = generate_date_range(start_time, end_time, clip_start, clip_end, shift, frequency, time_step, time_unit)
        expected_range = pd.DatetimeIndex(['2019-01-31 01:59:00.000007+01:00',
                                           '2019-01-31 01:59:00.000008+01:00',
                                           '2019-01-31 01:59:00.000009+01:00',
                                           '2019-01-31 01:59:00.000010+01:00',
                                           '2019-01-31 01:59:00.000011+01:00',
                                           '2019-01-31 01:59:00.000012+01:00',
                                           '2019-01-31 01:59:00.000013+01:00',
                                           '2019-01-31 01:59:00.000014+01:00',
                                           '2019-01-31 01:59:00.000015+01:00'])
        np.testing.assert_array_equal(date_range, expected_range)

    def test_generate_date_range_nanoseconds(self, config):
        config["time_unit"] = "nanoseconds"
        config["time_step"] = 1
        start_time = pd.Timestamp('2019-01-31T00:59:00.000000000')
        end_time = pd.Timestamp('2019-01-31T00:59:00.000000009')

        params = get_resampling_params(config)
        frequency = params.resampling_step
        time_unit = params.time_unit
        time_step = params.time_step

        clip_start = 5
        shift = 2
        clip_end = 3

        date_range = generate_date_range(start_time, end_time, clip_start, clip_end, shift, frequency, time_step, time_unit)
        np.testing.assert_array_equal(date_range, pd.DatetimeIndex(['2019-01-31 00:59:00.000000007',
                                                                    '2019-01-31 00:59:00.000000008']))
