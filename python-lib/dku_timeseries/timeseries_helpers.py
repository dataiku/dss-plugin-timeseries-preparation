# coding: utf-8
import logging
import math

import pandas as pd
from pandas.tseries.frequencies import to_offset
from pandas.tseries.offsets import BDay

logger = logging.getLogger(__name__)

# Frequency strings as defined in https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
FREQUENCY_STRINGS = {
    'years': 'A',
    'semi_annual': 'M',
    'quarters': 'M',
    'months': 'M',
    'weeks': 'W',
    'days': 'D',
    'business_days': 'B',
    'hours': 'H',
    'minutes': 'T',
    'seconds': 'S',
    'milliseconds': 'L',
    'microseconds': 'us',
    'nanoseconds': 'ns'
}

ROUND_COMPATIBLE_TIME_UNIT = ['days', 'hours', 'minutes', 'seconds', 'milliseconds', 'microseconds', 'nanoseconds']
UNIT_ORDER = ['years', 'months', 'semi_annual', 'quarters', 'weeks', 'days', 'business_days', 'hours', 'minutes', 'seconds', 'milliseconds', 'microseconds',
              'nanoseconds']


def reformat_time_step(time_step, time_unit):
    if time_step is not None:
        reformatted_time_step = float(time_step)
        if time_unit == "semi_annual":
            reformatted_time_step = 6 * reformatted_time_step
        elif time_unit == "quarters":
            reformatted_time_step = 3 * reformatted_time_step
        return reformat_time_value(reformatted_time_step, time_unit)
    else:
        raise ValueError("Invalid time step, it must be a number greater than 0")


def reformat_time_value(time_value, time_unit):
    formatted_time_value = time_value
    if time_unit not in ROUND_COMPATIBLE_TIME_UNIT:
        if time_value.is_integer():
            formatted_time_value = int(time_value)
        else:
            raise ValueError("Can not use non-integer time value with time unit '{}'".format(time_unit))
    return formatted_time_value


def format_resampling_step(time_unit, time_step, time_unit_end_of_week):
    frequency = FREQUENCY_STRINGS.get(time_unit, '')
    if frequency == "W" and time_unit_end_of_week:
        frequency = "W-{}".format(time_unit_end_of_week)
    return str(time_step) + frequency


def get_date_offset(time_unit, offset_value):
    if time_unit == "business_days":
        # adding a Bday converts the timestamps into a business day, so BDay(0) + Saturday January 1st =  Monday January 4th
        if offset_value != 0:
            return BDay(offset_value)
        else:
            formatted_time_unit = "days"
    elif time_unit == "semi_annual" or time_unit == "quarters":
        formatted_time_unit = "months"
    else:
        formatted_time_unit = time_unit
    return pd.DateOffset(**{formatted_time_unit: offset_value})


def generate_date_range(start_time, end_time, clip_start, clip_end, shift, frequency, time_step, time_unit):
    rounding_freq_string = FREQUENCY_STRINGS.get(time_unit)
    clip_start_value = get_date_offset(time_unit, clip_start)
    clip_end_value = get_date_offset(time_unit, clip_end)
    shift_value = get_date_offset(time_unit, shift)
    if time_unit in ROUND_COMPATIBLE_TIME_UNIT:
        start_index = start_time.round(rounding_freq_string) + clip_start_value + shift_value
        end_index = end_time.round(rounding_freq_string) - clip_end_value + shift_value + shift_value
    else:  # for week, month, year we round up to closest day
        start_index = start_time.round("D") + clip_start_value + shift_value
        # for some reason date_range omit the last entry when dealing with months, years
        end_index = end_time.round("D") - clip_end_value + get_date_offset(time_unit, time_step) + shift_value
    return pd.date_range(start=start_index, end=end_index, freq=frequency)


def get_smaller_unit(window_unit):
    index = UNIT_ORDER.index(window_unit)
    next_index = index + 1
    if next_index >= len(UNIT_ORDER):
        next_index = len(UNIT_ORDER) - 1
    return UNIT_ORDER[next_index]


def convert_time_freq_to_row_freq(frequency, window_description):
    data_frequency = pd.to_timedelta(to_offset(frequency))
    demanded_frequency = pd.to_timedelta(to_offset(window_description))
    n = demanded_frequency / data_frequency
    if n < 1:
        logger.error('The requested window width ({1}) is smaller than the timeseries frequency ({0}).'.format(data_frequency, demanded_frequency))
        raise ValueError('The requested window width ({1}) is smaller than the timeseries frequency ({0}).'.format(data_frequency, demanded_frequency))
    return int(math.ceil(n))  # always round up so that we dont miss any data


def infer_frequency(df):
    if len(df) > 2:
        frequency = pd.infer_freq(df[~df.index.duplicated()][:10000].index)
    elif len(df) == 2:
        frequency = df.index[1] - df.index[0]
    else:
        frequency = None
    return frequency


def format_group_id(group_id, identifiers_number):
    if identifiers_number == 1:
        group_id = [group_id]
    else:
        group_id = list(group_id)
    return group_id
