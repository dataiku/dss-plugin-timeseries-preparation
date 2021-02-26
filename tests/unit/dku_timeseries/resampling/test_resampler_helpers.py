import numpy as np
import pandas as pd
import pytest

from commons import get_resampling_params
from dku_timeseries.timeseries_helpers import generate_date_range


@pytest.fixture
def config():
    config = {u'clip_end': 0, u'constant_value': 0, u'extrapolation_method': u'none', u'shift': 0, u'time_unit_end_of_week': u'SUN',
              u'datetime_column': u'Date', u'advanced_activated': False, u'time_unit': u'quarters', u'clip_start': 0, u'time_step': 2,
              u'interpolation_method': u'linear'}
    return config


class TestResamplerHelpers:
    def test_generate_date_range_month(self,config):
        config["time_step"] = 2
        config["time_unit"] = "months"
        params = get_resampling_params(config)
        frequency = params.resampling_step
        time_step = params.time_step
        time_unit = params.time_unit

        start_time = pd.Timestamp('2021-01-23 00:00:00')
        end_time = pd.Timestamp('2021-06-20 00:00:00')
        extrapolation_method = "none"
        date_range = generate_date_range(start_time, end_time, 0, 0, 0, frequency, time_step, time_unit, extrapolation_method)
        assert date_range.shape[0] == 3
        extrapolation_method = "clip"
        date_range_extrapolation = generate_date_range(start_time, end_time, 0, 0, 0, frequency, time_step, time_unit, extrapolation_method)
        assert date_range_extrapolation.shape[0] == 4

    def test_generate_date_range_week(self,config):
        config["time_step"] = 2
        config["time_unit"] = "weeks"
        start_time = pd.Timestamp('2020-12-23 00:00:00')
        end_time = pd.Timestamp('2021-01-18 00:00:00')
        params = get_resampling_params(config)
        frequency = params.resampling_step
        time_step = params.time_step
        time_unit = params.time_unit

        extrapolation_method = "none"
        date_range = generate_date_range(start_time, end_time, 0, 0, 0, frequency, time_step, time_unit, extrapolation_method)
        assert date_range.shape[0] == 2

        extrapolation_method = "clip"
        date_range_extrapolation = generate_date_range(start_time, end_time, 0, 0, 0, frequency, time_step, time_unit, extrapolation_method)
        assert date_range_extrapolation.shape[0] == 3

        end_time = pd.Timestamp('2021-01-31 00:00:00')
        date_range = generate_date_range(start_time, end_time, 0, 0, 0, frequency, time_step, time_unit, extrapolation_method)
        assert date_range.shape[0] == 3

    def test_generate_date_range_quarters(self,config):
        config["time_step"] = 1
        config["time_unit"] = "quarters"
        start_time = pd.Timestamp('2020-01-23 00:00:00')
        end_time = pd.Timestamp('2021-01-18 00:00:00')

        params = get_resampling_params(config)
        frequency = params.resampling_step
        time_step = params.time_step
        time_unit = params.time_unit

        extrapolation_method = "none"
        date_range = generate_date_range(start_time, end_time, 0, 0, 0, frequency, time_step, time_unit, extrapolation_method)
        assert date_range[-1] == pd.Timestamp('2020-10-31')

        extrapolation_method = "clip"
        date_range = generate_date_range(start_time, end_time, 0, 0, 0, frequency, time_step, time_unit, extrapolation_method)
        assert date_range[-1] == pd.Timestamp('2021-01-31')

    def test_generate_date_range_half_year(self,config):
        config["time_step"] = 1
        config["time_unit"] = "semi_annual"
        start_time = pd.Timestamp('2020-01-23 00:00:00')
        end_time = pd.Timestamp('2021-01-18 00:00:00')

        params = get_resampling_params(config)
        frequency = params.resampling_step
        time_step = params.time_step
        time_unit = params.time_unit

        extrapolation_method = "none"
        date_range = generate_date_range(start_time, end_time, 0, 0, 0, frequency, time_step, time_unit, extrapolation_method)
        assert date_range[-1] == pd.Timestamp('2020-07-31')
        date_range_year = generate_date_range(start_time, end_time, 0, 0, 0, frequency, 1, "years", extrapolation_method)
        np.testing.assert_array_equal(date_range, date_range_year)

        extrapolation_method = "clip"
        date_range = generate_date_range(start_time, end_time, 0, 0, 0, frequency, time_step, time_unit, extrapolation_method)
        assert date_range[-1] == pd.Timestamp('2021-01-31 00:00:00')
