# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import logging
from operator import itemgetter
from itertools import *

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='timeseries-preparation plugin %(levelname)s - %(message)s')

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

    def __init__(self,
                 min_valid_values_duration_value=1,
                 min_deviation_duration_value=0,
                 time_unit='seconds'):

        self.min_valid_values_duration_value = min_valid_values_duration_value
        self.min_deviation_duration_value = min_deviation_duration_value
        self.time_unit = time_unit

        if self.time_unit == 'rows':
            self.min_valid_values_duration = self.min_valid_values_duration_value
            self.min_deviation_duration = self.min_deviation_duration_value
        else:
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
            raise ValueError('{0} is not a valid time unit. Possible options are: {1}'.format(self.time_unit, FREQUENCY_STRINGS.keys()))


class IntervalRestrictor:

    def __init__(self, params):
        if params is None:
            raise ValueError('IntervalRestrictorParams instance is not specified.')
        self.params = params
        self.params.check()

    def compute(self, raw_df, datetime_column, threshold_dict, groupby_columns=None):

        if not isinstance(datetime_column, basestring):
            raise ValueError('datetime_column param must be string. Got: ' + str(datetime_column))
        if groupby_columns:
            if not isinstance(groupby_columns, list):
                raise ValueError('groupby_columns param must be an array of strings. Got: ' + str(groupby_columns))
            for col in groupby_columns:
                if not isinstance(col, basestring):
                    raise ValueError('groupby_columns param must be an array of strings. Got: ' + str(col))

        if groupby_columns:
            grouped = raw_df.groupby(groupby_columns)
            filtered_groups = []
            for _, group in grouped:
                filtered_df = self._detect_segment(group, datetime_column, threshold_dict)
                filtered_groups.append(filtered_df)
            return pd.concat(filtered_groups).reset_index(drop=True)
        else:
            return self._detect_segment(raw_df, datetime_column, threshold_dict)

    def _nothing_to_do(self, df):
        return len(df) == 0

    def _detect_time_segment(self, df, chosen_col, lower_threshold, upper_threshold):

        new_df = df.copy()
        new_df['numerical_index'] = range(len(new_df))

        filtered_numerical_index = \
            new_df[(new_df[chosen_col] < lower_threshold) | (new_df[chosen_col] > upper_threshold)][
                'numerical_index'].values

        if len(filtered_numerical_index) == len(df):  # all data is artefact
            return []

        artefact_index_list = []
        # [1,2,3,5,6] -> [[1,2,3], [5,6]
        for k, g in groupby(enumerate(filtered_numerical_index), lambda (i, x): i - x):
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
        else: # no artifact
            proposed_indexes = [new_df.index[0], new_df.index[-1]]

        list_of_groups = zip(*(iter(proposed_indexes),) * 2)  # [a,b,c,d] -> [(a,b), (c,d)]

        final_indexes = []
        for group in list_of_groups:
            duration = group[1] - group[0]
            if duration >= self.params.min_valid_values_duration:
                final_indexes.append(group)

        return final_indexes

    def _detect_row_segment(self, df, chosen_col, lower_threshold, upper_threshold):

        new_df = df.copy()
        filtered_numerical_index = \
            np.nonzero((new_df[chosen_col] < lower_threshold) | (new_df[chosen_col] > upper_threshold))[0]
        if len(filtered_numerical_index) == len(df):  # all data is artefact
            return []

        artefact_index_list = []
        for k, g in groupby(enumerate(filtered_numerical_index), lambda (i, x): i - x):
            artefact_index_list.append(map(itemgetter(1), g))

        border_list = []
        for artefact_indexes in artefact_index_list:
            new_list = list(artefact_indexes)
            new_list.insert(0, max(artefact_indexes[0] - 1, 0))
            new_list.append(min(artefact_indexes[-1] + 1, len(df) - 1))
            border_list.append([new_list[0], new_list[1], new_list[-2], new_list[-1]])

        deviations_indices = []
        for border_index in border_list:
            start = border_index[1]
            end = border_index[-2]
            is_deviation = (end - start) >= self.params.min_deviation_duration
            if is_deviation:
                deviations_indices.extend([border_index[0], border_index[-1]])

        if len(deviations_indices) > 0:
            if (deviations_indices[0] == new_df.index[0]) and (deviations_indices[-1] == new_df.index[-1]):
                proposed_indexes = deviations_indices
            elif deviations_indices[0] == new_df.index[0]:
                proposed_indexes = deviations_indices + [new_df.index[-1]]
            elif deviations_indices[-1] == new_df.index[-1]:
                proposed_indexes = [new_df.index[0]] + deviations_indices
            else:
                proposed_indexes = [new_df.index[0]] + deviations_indices + [new_df.index[-1]]
        else: # no artifact
            proposed_indexes = [new_df.index[0], new_df.index[-1]]

        list_of_groups = zip(*(iter(proposed_indexes),) * 2)  # [a,b,c,d] -> [(a,b), (c,d)]

        final_indexes = []
        for group in list_of_groups:
            duration = group[1] - group[0]
            print(group, duration)
            if duration >= self.params.min_valid_values_duration:
                final_indexes.append(group)

        return final_indexes

    def _detect_segment(self, raw_df, datetime_column, threshold_dict):

        if self._nothing_to_do(raw_df):
            return raw_df

        # TODO add support for multiple threshold column
        if self.params.time_unit == 'rows':
            df = raw_df.sort_values(datetime_column)
            df = df.reset_index(drop=True)
        else:
            df = raw_df.copy()
            df.loc[:, datetime_column] = pd.to_datetime(df[datetime_column])
            df = df.set_index(datetime_column).sort_index()

        segment_indexes = []
        for chosen_column, threshold_tuple in threshold_dict.items():
            lower_threshold, upper_threshold = threshold_tuple
            if self.params.time_unit == 'rows':
                segment_indexes = self._detect_row_segment(df, chosen_column, lower_threshold, upper_threshold)
            else:
                segment_indexes = self._detect_time_segment(df, chosen_column, lower_threshold, upper_threshold)

        mask_list = []
        if len(segment_indexes) > 0:
            for start, end in segment_indexes:
                mask = (df.index >= start) & (df.index <= end)
                mask_list.append(mask)
            segment_df = df.loc[np.logical_or.reduce(mask_list)]
        else:
            segment_df = pd.DataFrame(columns=df.columns)

        if self.params.time_unit != 'rows':
            segment_df = segment_df.rename_axis(datetime_column).reset_index()

        return segment_df
