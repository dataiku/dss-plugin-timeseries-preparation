# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import logging
import sys
import re
from pandas.tseries.frequencies import to_offset
import math

from dku_timeseries.dataframe_helpers import has_duplicates, nothing_to_do, filter_empty_columns, generic_check_compute_arguments
from dku_timeseries.timeseries_helpers import convert_time_freq_to_row_freq, get_smaller_unit, infer_frequency, FREQUENCY_STRINGS, UNIT_ORDER

logger = logging.getLogger(__name__)

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

WINDOW_TYPES = ['triang', 'blackman', 'hamming', 'bartlett', 'parzen', 'gaussian', None]  # None for global extrema
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
    'second_order_derivative'
    # No lag, UI concern (where to put offset value)
]


class WindowAggregatorParams:  # TODO better naming ?

    def __init__(self,
                 causal_window=True,
                 window_width=1,
                 window_unit='seconds',
                 min_period=1,
                 closed_option='left',
                 window_type=None,
                 gaussian_std=1.0,
                 aggregation_types=AGGREGATION_TYPES):

        self.causal_window = causal_window
        self.window_width = window_width
        self.window_unit = window_unit
        self.window_description = str(self.window_width) + FREQUENCY_STRINGS.get(self.window_unit, '')
        self.min_period = min_period
        self.closed_option = closed_option
        self.window_type = window_type
        self.gaussian_std = gaussian_std
        self.aggregation_types = aggregation_types

    def check(self):

        if self.window_type is not None and self.window_type not in WINDOW_TYPES:
            raise ValueError(
                '{0} is not a valid window type. Possible options are: {1}'.format(self.window_type, WINDOW_TYPES))
        if self.window_width < 0:
            raise ValueError('Window width can not be negative.')
        if self.window_unit not in WINDOW_UNITS:
            raise ValueError(
                '"{0}" is not a valid unit. Possible window units are: {1}'.format(self.window_unit, WINDOW_UNITS))
        if self.min_period < 1:
            raise ValueError('Min period must be positive.')
        if self.closed_option not in CLOSED_OPTIONS:
            raise ValueError('"{0}" is not a valid closed option. Possible values are: {1}'.format(self.closed_option, CLOSED_OPTIONS))
        if self.window_unit == 'rows':
            raise NotImplementedError


class WindowAggregator:

    def __init__(self, params=None):

        self.params = params
        if params is None:
            raise ValueError('WindowAggregatorParams instance is not specified.')
        self.params.check()

    def compute(self, df, datetime_column, groupby_columns=None):

        generic_check_compute_arguments(datetime_column, groupby_columns)

        # drop all rows where the timestamp is null
        df_copy = df.dropna(subset=[datetime_column]).copy()
        if nothing_to_do(df_copy, min_len=2):
            logger.warning('The time series has less than 2 rows with values, can not apply window.')
            return df_copy

        df_copy.loc[:, datetime_column] = pd.to_datetime(df_copy[datetime_column])
        raw_columns = df_copy.select_dtypes(include=['float', 'int']).columns.tolist()

        if groupby_columns:
            grouped = df_copy.groupby(groupby_columns)
            computed_groups = []
            for group_id, group in grouped:
                logger.info("Computing for group {}".format(group_id))

                try:
                    if self.params.causal_window:
                        computed_df = self._compute_causal_stats(group, datetime_column, raw_columns, df_id=group_id)
                    else:
                        computed_df = self._compute_bilateral_stats(group, datetime_column, raw_columns, df_id=group_id)
                except Exception as e:
                    from future.utils import raise_
                    # issues with left border, cf https://github.com/pandas-dev/pandas/issues/26005
                    if e.message == ('skiplist_init failed'):
                        raise_(Exception, "Window width is too small", sys.exc_info()[2])
                    else:
                        raise_(Exception, "Compute stats failed", sys.exc_info()[2])

                computed_df[groupby_columns[0]] = group_id  # TODO generalize to multiple groupby cols
                computed_groups.append(computed_df)
            final_df = pd.concat(computed_groups)
        else:
            try:
                if self.params.causal_window:
                    final_df = self._compute_causal_stats(df_copy, datetime_column, raw_columns)
                else:
                    final_df = self._compute_bilateral_stats(df_copy, datetime_column, raw_columns)
            except Exception as e:
                from future.utils import raise_
                if e.message == ('skiplist_init failed'):
                    raise_(Exception, "Window width is too small", sys.exc_info()[2])
                else:
                    raise_(Exception, "Compute stats failed", sys.exc_info()[2])

        return final_df.reset_index(drop=True)

    def _check_valid_timeseries(self, frequency):
        if not frequency and self.params.window_type is not None:
            raise ValueError('The input time series is not equispaced. Cannot apply window with time unit.')  # pandas limitation

    def _compute_causal_stats(self, df, datetime_column, raw_columns, df_id=''):

        if nothing_to_do(df, min_len=2):
            logger.info('The time series {} has less than 2 rows with values, can not apply window.'.format(df_id))
            return df
        if has_duplicates(df, datetime_column):
            raise ValueError('The time series {} contain duplicate timestamps.'.format(df_id))

        reference_df = df.set_index(datetime_column).sort_index().copy()
        new_df = pd.DataFrame(index=reference_df.index)

        # compute all stats except mean and sum, the syntax does not change whether or not we have a window type
        roller_without_window_type = reference_df.rolling(window=self.params.window_description, closed=self.params.closed_option)
        new_df = self._compute_stats_without_win_type(roller_without_window_type, raw_columns, new_df, reference_df)

        # compute mean and sum, the only operations that might need a win_type
        # when using win_type, window must be defined in terms of rows and not time unit (pandas limitation)
        compute_sum_and_mean = len(set(self.params.aggregation_types).intersection(set(['average', 'sum']))) > 0
        if compute_sum_and_mean and self.params.window_type:
            # row-based rolling is always bound both side of the window, we thus shift 1 row down when closed is left
            if self.params.closed_option == 'left':
                shifted_df = reference_df.shift(1)
            else:
                shifted_df = reference_df

            frequency = infer_frequency(reference_df)
            if frequency:
                window_description_in_row = convert_time_freq_to_row_freq(frequency, self.params.window_description)
            else:
                raise ValueError('The input time series is not equispaced. Cannot apply window with time unit.')  # pandas limitation

            roller_with_window = shifted_df.rolling(window=window_description_in_row, win_type=self.params.window_type, closed=self.params.closed_option)
            new_df = self._compute_stats_with_win_type(roller_with_window, raw_columns, new_df)

        return new_df.rename_axis(datetime_column).reset_index()

    def _compute_bilateral_stats(self, df, datetime_column, raw_columns, df_id=''):

        if nothing_to_do(df, min_len=2):
            logger.info('The time series {} has less than 2 rows with values, can not apply window.'.format(df_id))
            return df
        if has_duplicates(df, datetime_column):
            raise ValueError('The time series {} contain duplicate timestamps.'.format(df_id))

        reference_df = df.set_index(datetime_column).sort_index().copy()
        new_df = pd.DataFrame(index=reference_df.index)

        frequency = infer_frequency(reference_df)
        if frequency:
            window_description_in_row = convert_time_freq_to_row_freq(frequency, self.params.window_description)
        else:
            raise ValueError('The input time series is not equispaced. Cannot compute bilateral window.')  # pandas limitation

        # compute all stats except mean and sum, these stats dont need a win_type
        roller_without_win_type = reference_df.rolling(window=window_description_in_row, center=True)
        new_df = self._compute_stats_without_win_type(roller_without_win_type, raw_columns, new_df, reference_df)

        # compute mean and sum, the only operations that win_type has an effect
        roller_with_win_type = reference_df.rolling(window=window_description_in_row, win_type=self.params.window_type, center=True)
        new_df = self._compute_stats_with_win_type(roller_with_win_type, raw_columns, new_df)

        return new_df.rename_axis(datetime_column).reset_index()

    def _compute_stats_without_win_type(self, roller, raw_columns, new_df, df_ref):

        if 'retrieve' in self.params.aggregation_types:
            new_df[raw_columns] = df_ref[raw_columns]
        if 'min' in self.params.aggregation_types:
            col_names = ['{}_min'.format(col) for col in raw_columns]
            new_df[col_names] = roller[raw_columns].apply(min, raw=True)
        if 'max' in self.params.aggregation_types:
            col_names = ['{}_max'.format(col) for col in raw_columns]
            new_df[col_names] = roller[raw_columns].apply(max, raw=True)
        if 'q25' in self.params.aggregation_types:
            col_names = ['{}_q25'.format(col) for col in raw_columns]
            new_df[col_names] = roller[raw_columns].quantile(0.25, raw=True)
        if 'median' in self.params.aggregation_types:
            col_names = ['{}_median'.format(col) for col in raw_columns]
            new_df[col_names] = roller[raw_columns].quantile(0.5, raw=True)
        if 'q75' in self.params.aggregation_types:
            col_names = ['{}_q75'.format(col) for col in raw_columns]
            new_df[col_names] = roller[raw_columns].quantile(0.75, raw=True)
        if 'first_order_derivative' in self.params.aggregation_types:
            col_names = ['{}_1st_derivative'.format(col) for col in raw_columns]
            if self.params.window_width < 1:
                derivative_time_unit = get_smaller_unit(self.params.window_unit)
            else:
                derivative_time_unit = TIMEDELTA_STRINGS.get(self.params.window_unit)

            data_lag_diff = df_ref[raw_columns].diff()
            # the division is to express the diff in the correct time unit
            time_lag_diff = df_ref.index.to_series().diff()
            time_lag_diff_normalized = time_lag_diff / (np.timedelta64(1, derivative_time_unit))

            timedelta_unit = TIMEDELTA_STRINGS.get(self.params.window_unit)
            is_inside_window_mask = (time_lag_diff <= self.params.window_width * np.timedelta64(1, timedelta_unit)).astype('float').replace({0: np.nan})
            new_df[col_names] = (data_lag_diff.div(time_lag_diff_normalized, axis=0)).multiply(is_inside_window_mask, axis=0)

        if 'second_order_derivative' in self.params.aggregation_types:
            col_names = ['{}_2nd_derivative'.format(col) for col in raw_columns]
            if self.params.window_width < 1:
                derivative_time_unit = get_smaller_unit(self.params.window_unit)
            else:
                derivative_time_unit = TIMEDELTA_STRINGS.get(self.params.window_unit)
            data_lag_two_diff = df_ref[raw_columns].diff().diff()
            # the division is to express the diff in the correct time unit
            time_lag_diff = df_ref.index.to_series().diff()
            time_lag_diff_normalized = time_lag_diff / (np.timedelta64(1, derivative_time_unit))
            timedelta_unit = TIMEDELTA_STRINGS.get(self.params.window_unit)
            is_inside_window_mask = (time_lag_diff <= self.params.window_width * np.timedelta64(1, timedelta_unit)).astype('float').replace({0: np.nan})
            new_df[col_names] = (data_lag_two_diff.div(time_lag_diff_normalized, axis=0)).multiply(is_inside_window_mask, axis=0)

        if 'std' in self.params.aggregation_types:
            col_names = ['{}_std'.format(col) for col in raw_columns]
            new_df[col_names] = roller[raw_columns].std()

        if 'average' in self.params.aggregation_types and self.params.window_type is None:
            col_names = ['{}_avg'.format(col) for col in raw_columns]
            new_df[col_names] = roller[raw_columns].mean()

        if 'sum' in self.params.aggregation_types and self.params.window_type is None:
            col_names = ['{}_sum'.format(col) for col in raw_columns]
            new_df[col_names] = roller[raw_columns].sum()

        return new_df

    def _compute_stats_with_win_type(self, roller, raw_columns, new_df):

        if 'average' in self.params.aggregation_types:
            col_names = ['{}_avg'.format(col) for col in raw_columns]
            if self.params.window_type == 'gaussian':
                new_df[col_names] = roller[raw_columns].mean(std=self.params.gaussian_std)
            else:
                new_df[col_names] = roller[raw_columns].mean()
        if 'sum' in self.params.aggregation_types:
            col_names = ['{}_sum'.format(col) for col in raw_columns]
            if self.params.window_type == 'gaussian':
                new_df[col_names] = roller[raw_columns].sum(std=self.params.gaussian_std)
            else:
                new_df[col_names] = roller[raw_columns].sum()
        return new_df