# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import logging
from operator import itemgetter
from itertools import *

from dataframe_helpers import has_duplicates, nothing_to_do, filter_empty_columns, generic_check_compute_arguments
from timeseries_helpers import get_date_offset, generate_date_range

logger = logging.getLogger(__name__)

# Frequency strings as defined in https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
FREQUENCY_STRINGS = {
    'days': 'D',
    'hours': 'h',
    'minutes': 'm',
    'seconds': 's',
    'milliseconds': 'ms',
    'microseconds': 'us',
    'nanoseconds': 'ns',
    'rows': ''
}


class IntervalRestrictorParams:

    def __init__(self, min_valid_values_duration_value=1, max_deviation_duration_value=0, time_unit='seconds'):

        self.min_valid_values_duration_value = min_valid_values_duration_value
        self.min_deviation_duration_value = max_deviation_duration_value
        self.time_unit = time_unit
        self.min_valid_values_duration = pd.Timedelta(
            str(self.min_valid_values_duration_value) + FREQUENCY_STRINGS.get(self.time_unit, ''))
        self.min_deviation_duration = pd.Timedelta(
            str(self.min_deviation_duration_value) + FREQUENCY_STRINGS.get(self.time_unit, ''))

    def check(self):

        # if self.min_valid_values_duration_value <= 0:
        #    raise ValueError('Min valid values duration must be positive.')
        if self.min_deviation_duration_value < 0:
            raise ValueError('Min deviation duration cannot be negative.')
        if self.time_unit not in FREQUENCY_STRINGS:
            raise ValueError('{0} is not a valid time unit. Possible options are: {1}'.format(self.time_unit,
                                                                                              FREQUENCY_STRINGS.keys()))


class IntervalRestrictor:

    def __init__(self, params):
        if params is None:
            raise ValueError('IntervalRestrictorParams instance is not specified.')
        self.params = params
        self.params.check()

    def compute(self, df, datetime_column, threshold_dict, groupby_columns=None):

        generic_check_compute_arguments(datetime_column, groupby_columns)
        df_copy = df.copy()

        # drop all rows where the timestamp is null
        df_copy = df_copy.dropna(subset=[datetime_column])
        if nothing_to_do(df_copy, min_len=0):
            logger.warning('The time series is empty, can not compute.')
            return pd.DataFrame(columns=df_copy.columns)

        lower_threshold, upper_threshold, filter_column = None, None, None
        for column, threshold_tuple in threshold_dict.items():
            filter_column = column
            lower_threshold, upper_threshold = threshold_tuple

        filter_function = self._between_min_max_mask(lower_threshold, upper_threshold)

        if groupby_columns:
            grouped = df.groupby(groupby_columns)
            filtered_groups = []
            for _, group in grouped:
                filtered_df = self._detect_segment(group, datetime_column, filter_column, filter_function)
                filtered_groups.append(filtered_df)
            return pd.concat(filtered_groups).reset_index(drop=True)
        else:
            return self._detect_segment(df, datetime_column, filter_column, filter_function)

    def _between_min_max_mask(self, lower_threshold, upper_threshold):
        def _func(x):
            return (x >= lower_threshold) & (x <= upper_threshold) & (~np.isnan(x))
        return _func

    def _detect_time_segment(self, df, filter_column, filter_function):

        """
        We start with the search of deviation segments (the one that are NOT valid) -> binary mask
        If the deviation duration is less than the max_deviation_duration allowed, we accept them as valid values.
        Once dealt with deviation, we end up with the valid segments.
        The final filter will be apply on them and we keep thoses that last at least min_valid_values_duration

        TODO: should take a dict of functions ? -> different conditions for each column for example
        """
        new_df = df.copy()
        new_df['numerical_index'] = range(len(new_df))

        # find index of invalid values
        invalid_values_numerical_index = new_df[~filter_function(new_df[filter_column])]['numerical_index'].values

        if len(invalid_values_numerical_index) == len(df):  # all data is artefact
            return []

        artefact_index_list = []
        # [1,2,3,5,6] -> [[1,2,3], [5,6]
        for k, g in groupby(enumerate(invalid_values_numerical_index), lambda (i, x): i - x):
            artefact_index_list.append(map(itemgetter(1), g))

        border_list = []
        for artefact_indexes in artefact_index_list:
            new_list = list(artefact_indexes)
            new_list.insert(0, max(artefact_indexes[0] - 1, 0))
            new_list.append(min(artefact_indexes[-1] + 1, len(new_df) - 1))
            border_list.append([new_list[0], new_list[1], new_list[-2], new_list[-1]])

        border_timestamp_list = []
        for border in border_list:
            temp_timestamp = []
            for border_index in border:
                temp_timestamp.append(new_df.loc[new_df['numerical_index'] == border_index].index[0])
            border_timestamp_list.append(temp_timestamp)

        deviations_indices = []
        for border_timestamp in border_timestamp_list:
            start = border_timestamp[1]
            end = border_timestamp[-2]
            is_deviation = (end - start) >= self.params.min_deviation_duration
            if is_deviation:
                deviations_indices.extend([border_timestamp[0], border_timestamp[-1]])

        if len(deviations_indices) > 0:
            if (deviations_indices[0] == new_df.index[0]) and (deviations_indices[-1] == new_df.index[-1]):
                proposed_indexes = deviations_indices
            elif deviations_indices[0] == new_df.index[0]:
                proposed_indexes = deviations_indices + [new_df.index[-1]]
            elif deviations_indices[-1] == new_df.index[-1]:
                proposed_indexes = [new_df.index[0]] + deviations_indices
            else:
                proposed_indexes = [new_df.index[0]] + deviations_indices + [new_df.index[-1]]
        else:  # no artifact
            proposed_indexes = [new_df.index[0], new_df.index[-1]]

        list_of_groups = zip(*(iter(proposed_indexes),) * 2)  # [a,b,c,d] -> [(a,b), (c,d)]
        final_indexes = []
        for group in list_of_groups:
            duration = group[1] - group[0]
            if duration >= self.params.min_valid_values_duration:
                final_indexes.append(group)

        return final_indexes

    def _detect_segment(self, df, datetime_column, filter_column, filter_function, df_id=''):

        if has_duplicates(df, datetime_column):
            raise ValueError('The time series {} contain duplicate timestamps.'.format(df_id))

        if nothing_to_do(df, min_len=0):
            logger.warning('The time series {} is empty, can not compute.'.format(df_id))
            return pd.DataFrame(columns=df.columns)

        df_copy = df.copy()
        df_copy.loc[:, datetime_column] = pd.to_datetime(df_copy[datetime_column])
        df_copy = df_copy.set_index(datetime_column).sort_index()

        segment_indexes = self._detect_time_segment(df_copy, filter_column, filter_function)

        mask_dict = {}
        if len(segment_indexes) > 0:
            for segment_index, (start, end) in enumerate(segment_indexes):
                mask = (df_copy.index >= start) & (df_copy.index <= end)
                mask_dict[segment_index] = mask

            df_labeled = df_copy.copy()
            df_labeled['interval_id'] = np.nan

            for k, v in mask_dict.items():
                df_labeled.loc[v, 'interval_id'] = str(int(k))

            segment_df = df_labeled.loc[np.logical_or.reduce(mask_dict.values())].sort_index()
        else:
            segment_df = pd.DataFrame(columns=df_copy.columns)

        return segment_df.rename_axis(datetime_column).reset_index()