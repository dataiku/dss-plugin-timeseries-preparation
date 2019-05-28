# coding: utf-8
import pandas as pd
# Frequency strings as defined in https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
FREQUENCY_STRINGS = {
    'years': 'A',
    'months': 'M',
    'weeks': 'W',
    'days': 'D',
    'hours': 'H',
    'minutes': 'T',
    'seconds': 'S',
    'milliseconds': 'L',
    'microseconds': 'us',
    'nanoseconds': 'ns'
}

ROUND_COMPATIBLE_TIME_UNIT = ['days', 'hours', 'minutes', 'seconds', 'milliseconds', 'microseconds', 'nanoseconds']


def get_date_offset(time_unit, offset_value):
    return pd.DateOffset(**{time_unit: offset_value})


def generate_date_range(start_time, end_time, clip_start, clip_end, frequency, time_step, time_unit):
    rounding_freq_string = FREQUENCY_STRINGS.get(time_unit)
    clip_start_value = get_date_offset(time_unit, clip_start)
    clip_end_value = get_date_offset(time_unit, clip_end)
    if time_unit in ROUND_COMPATIBLE_TIME_UNIT:
        start_index = start_time.round(rounding_freq_string) + clip_start_value
        end_index = end_time.round(rounding_freq_string) - clip_end_value
    else:  # for week, month, year we round up to closest day
        start_index = start_time.round('D') + clip_start_value
        # for some reason date_range omit the last entry when dealing with months, years
        end_index = end_time.round('D') - clip_end_value + get_date_offset(time_unit, time_step)

    return pd.date_range(start=start_index, end=end_index, freq=frequency)