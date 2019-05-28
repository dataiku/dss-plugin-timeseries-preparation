# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import logging
import re
from pandas.tseries.frequencies import to_offset
import math

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='timeseries-preparation plugin %(levelname)s - %(message)s')

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

TIMEDELTA_STRINGS = {
    'years': 'Y',
    'months': 'M',
    'weeks': 'W',
    'days': 'D',
    'hours': 'h',
    'minutes': 'm',
    'seconds': 's',
    'milliseconds': 'ms',
    'microseconds': 'us',
    'nanoseconds': 'ns'
}

UNIT_ORDER = ['years', 'months', 'weeks', 'days', 'hours', 'minutes', 'seconds', 'milliseconds', 'microseconds', 'nanoseconds']

WINDOW_TYPES = ['boxcar', 'triang', 'blackman', 'hamming', 'bartlett', 'parzen', 'gaussian'] # None for global extrema
WINDOW_UNITS = list(FREQUENCY_STRINGS.keys()) + ['rows']
CLOSED_OPTIONS = ['right', 'left', 'both', 'neither']
AGGREGATION_TYPES = [
    'retrieve',
    'average',
    'min',
    'max',
    'std',
    'q25',
    'median',
    'q75',
    'sum',
    'first_order_derivative',
    'second_order_derivative',
    'count'
    # No lag, UI concern (where to put offset value)
    ]


def get_smaller_unit(window_unit):
    index = UNIT_ORDER.index(window_unit)
    next_index = index + 1
    if next_index >= len(UNIT_ORDER):
        next_index = len(UNIT_ORDER) - 1

    return UNIT_ORDER[next_index]


class WindowRollerParams:  # TODO better naming ?

    def __init__(self,
                 window_width=1,
                 window_unit='seconds',
                 min_period=1,
                 closed_option='left',
                 center=False,
                 window_type=None,
                 gaussian_std=1.0,
                 aggregation_types=AGGREGATION_TYPES):

        self.window_width = window_width
        self.window_unit = window_unit

        if window_unit == 'rows':
            self.window_description = self.window_width
        else:
            self.window_description = str(self.window_width) + FREQUENCY_STRINGS.get(self.window_unit, '')
        self.window_description_in_row = None
        self.min_period = min_period
        self.closed_option = closed_option
        self.center = center
        self.window_type = window_type
        self.gaussian_std = gaussian_std
        self.aggregation_types = aggregation_types

    def check(self):

        if self.window_type is not None and self.window_type not in WINDOW_TYPES:
            raise ValueError('{0} is not a valid window type. Possible options are: {1}'.format(self.window_type, WINDOW_TYPES))
        if self.window_width < 0:
            raise ValueError('Window width can not be negative.')
        if self.window_unit not in WINDOW_UNITS:
            raise ValueError('"{0}" is not a valid unit. Possible window units are: {1}'.format(self.window_unit, WINDOW_UNITS))
        if self.min_period < 1:
            raise ValueError('Min period must be positive.')
        if self.closed_option not in CLOSED_OPTIONS:
            raise ValueError('"{0}" is not a valid closed option. Possible values are: {1}'.format(self.closed_option, CLOSED_OPTIONS))


class WindowRoller:

    def __init__(self, params=None):

        self.params = params
        if params is None:
            raise ValueError('WindowRollerParams instance is not specified.')
        self.params.check()

    def compute(self, raw_df, datetime_column, groupby_columns=None):

        if not isinstance(datetime_column, basestring):
            raise ValueError('datetime_column param must be string. Got: ' + str(datetime_column))
        if groupby_columns:
            if not isinstance(groupby_columns, list):
                raise ValueError('groupby_columns param must be an array of strings. Got: ' + str(groupby_columns))
            for col in groupby_columns:
                if not isinstance(col, basestring):
                    raise ValueError('groupby_columns param must be an array of strings. Got: ' + str(col))

        df = raw_df.copy()
        df.loc[:, datetime_column] = pd.to_datetime(df[datetime_column])
        raw_columns = df.select_dtypes(include=['float', 'int']).columns.tolist()

        if groupby_columns:
            grouped = df.groupby(groupby_columns)
            computed_groups = []
            for group_id, group in grouped:
                computed_df = self._compute_rolling_stats(group, datetime_column, raw_columns)
                computed_df[groupby_columns[0]] = group_id  # TODO generalize to multiple groupby cols
                computed_groups.append(computed_df)
            final_df = pd.concat(computed_groups)
        else:
            final_df = self._compute_rolling_stats(df, datetime_column, raw_columns)
            # return final_df

        final_df = final_df.reset_index(drop=True)

        return final_df

    def _nothing_to_do(self, df):
        return len(df) < 2

    def get_window_date_offset(self):
        return pd.DateOffset(**{self.params.window_unit: self.params.window_width})

    def _check_valid_data(self, df, frequency):
        # if non-equispaced + time-based window + using window, it is not possible (scipy limitation)
        if not frequency and self.params.window_unit != 'rows' and self.params.window_type is not None:
            raise ValueError('The input dataset is not equispaced thus we can not apply rolling {} window with time unit.'.format(self.params.window_type))

    def _convert_time_freq_to_row_freq(self, frequency):

        data_frequency = pd.to_timedelta(to_offset(frequency))
        demanded_frequency = pd.to_timedelta(to_offset(self.params.window_description))
        n = demanded_frequency / data_frequency
        if n < 1:
            raise ValueError('The requested window width ({0}) is smaller than the timeseries frequency ({1}).'.format(data_frequency, demanded_frequency))
        return int(math.ceil(n))  # always round up so that we dont miss any data

    def _compute_rolling_stats(self, df, datetime_column, raw_columns):

        if self._nothing_to_do(df):
            return df

        df = df.set_index(datetime_column).sort_index()
        # return df

        frequency = None
        if len(df) > 2:
            frequency = pd.infer_freq(df[~df.index.duplicated()][:1000].index)
            logger.info('Timeseries frequency: ', frequency)
        elif len(df) == 2:
            frequency = df.index[1] - df.index[0]

        self._check_valid_data(df, frequency)

        if frequency and self.params.window_unit != 'rows' and self.params.window_type is not None:
            self.params.window_description_in_row = self._convert_time_freq_to_row_freq(frequency)

        new_df = pd.DataFrame(index=df.index)
        # compute all stats except mean and sum, does not change whether or not we have a window type
        if self.params.window_unit == 'rows':  # for now `closed` is only implemented for time-based window
            rolling_without_window = df.rolling(window=self.params.window_description)
        else:
            rolling_without_window = df.rolling(window=self.params.window_description, closed=self.params.closed_option)

        if 'retrieve' in self.params.aggregation_types:
            new_df[raw_columns] = df[raw_columns]
        if 'min' in self.params.aggregation_types:
            col_names = ['{}_min'.format(col) for col in raw_columns]
            new_df[col_names] = rolling_without_window[raw_columns].apply(min, raw=True)  # raw=True
        if 'max' in self.params.aggregation_types:
            col_names = ['{}_max'.format(col) for col in raw_columns]
            new_df[col_names] = rolling_without_window[raw_columns].apply(max, raw=True)
        if 'q25' in self.params.aggregation_types:
            col_names = ['{}_q25'.format(col) for col in raw_columns]
            new_df[col_names] = rolling_without_window[raw_columns].quantile(0.25, raw=True)
        if 'median' in self.params.aggregation_types:
            col_names = ['{}_median'.format(col) for col in raw_columns]
            new_df[col_names] = rolling_without_window[raw_columns].quantile(0.5, raw=True)
        if 'q75' in self.params.aggregation_types:
            col_names = ['{}_q75'.format(col) for col in raw_columns]
            new_df[col_names] = rolling_without_window[raw_columns].quantile(0.75, raw=True)
        if 'first_order_derivative' in self.params.aggregation_types:
            col_names = ['{}_1st_derivative'.format(col) for col in raw_columns]
            if self.params.window_width < 1:
                derivative_time_unit = get_smaller_unit(self.params.window_unit)
            else:
                derivative_time_unit = TIMEDELTA_STRINGS.get(self.params.window_unit)

            data_lag_diff = df[raw_columns].diff()
            # the division is to express the diff in the correct time unit
            time_lag_diff = df.index.to_series().diff()
            time_lag_diff_normalized = time_lag_diff / (np.timedelta64(1, derivative_time_unit))

            timedelta_unit = TIMEDELTA_STRINGS.get(self.params.window_unit)
            is_inside_window_mask = (time_lag_diff <= self.params.window_width * np.timedelta64(1, timedelta_unit)).astype(
                'float').replace({0: np.nan})
            new_df[col_names] = (data_lag_diff.div(time_lag_diff_normalized, axis=0)).multiply(is_inside_window_mask,
                                                                                               axis=0)

        if 'second_order_derivative' in self.params.aggregation_types:
            col_names = ['{}_2nd_derivative'.format(col) for col in raw_columns]
            if self.params.window_width < 1:
                derivative_time_unit = get_smaller_unit(self.params.window_unit)
            else:
                derivative_time_unit = TIMEDELTA_STRINGS.get(self.params.window_unit)

            data_lag_two_diff = df[raw_columns].diff().diff()
            # the division is to express the diff in the correct time unit
            time_lag_diff = df.index.to_series().diff()
            time_lag_diff_normalized = time_lag_diff / (np.timedelta64(1, derivative_time_unit))

            timedelta_unit = TIMEDELTA_STRINGS.get(self.params.window_unit)
            is_inside_window_mask = (time_lag_diff <= self.params.window_width * np.timedelta64(1, timedelta_unit)).astype(
                'float').replace({0: np.nan})
            new_df[col_names] = (data_lag_two_diff.div(time_lag_diff_normalized, axis=0)).multiply(
                is_inside_window_mask, axis=0)

        if 'std' in self.params.aggregation_types:
            col_names = ['{}_std'.format(col) for col in raw_columns]
            new_df[col_names] = rolling_without_window[raw_columns].std()

        # compute mean, std and sum
        if self.params.window_type:
            if self.params.closed_option == 'left':
                shifted_df = df.shift(1)
            else:
                shifted_df = df

            if self.params.window_unit == 'rows':  #
                rolling_with_window = shifted_df.rolling(window=self.params.window_description,
                                                         win_type=self.params.window_type)
            else:
                rolling_with_window = shifted_df.rolling(window=self.params.window_description_in_row,
                                                         win_type=self.params.window_type,
                                                         closed=self.params.closed_option)

            if 'average' in self.params.aggregation_types:
                col_names = ['{}_avg'.format(col) for col in raw_columns]
                if self.params.window_type == 'gaussian':
                    new_df[col_names] = rolling_with_window[raw_columns].mean(std=self.params.gaussian_std)
                else:
                    new_df[col_names] = rolling_with_window[raw_columns].mean()
            if 'sum' in self.params.aggregation_types:
                col_names = ['{}_sum'.format(col) for col in raw_columns]
                if self.params.window_type == 'gaussian':
                    new_df[col_names] = rolling_with_window[raw_columns].sum(std=self.params.gaussian_std)
                else:
                    new_df[col_names] = rolling_with_window[raw_columns].sum()
        else:
            if 'average' in self.params.aggregation_types:
                col_names = ['{}_avg'.format(col) for col in raw_columns]
                new_df[col_names] = rolling_without_window[raw_columns].mean()
            if 'sum' in self.params.aggregation_types:
                col_names = ['{}_sum'.format(col) for col in raw_columns]
                new_df[col_names] = rolling_without_window[raw_columns].sum()

        new_df = new_df.rename_axis(datetime_column).reset_index()
        return new_df