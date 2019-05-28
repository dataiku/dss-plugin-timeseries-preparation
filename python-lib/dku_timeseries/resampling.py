# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import logging
from scipy import interpolate
from dataframe_helpers import have_duplicate, nothing_to_do, filter_empty_columns, check_transform_arguments
from timeseries_helpers import  get_date_offset, generate_date_range

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

ROUND_COMPATIBLE_TIME_UNIT = ['days', 'hours', 'minutes', 'seconds', 'milliseconds', 'microseconds', 'nanoseconds']
INTERPOLATION_METHODS = ['None', 'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', 'next']
EXTRAPOLATION_METHODS = ['None', 'clip', 'interpolation']
TIME_UNITS = list(FREQUENCY_STRINGS.keys()) + ['rows']


class ResamplerParams:

    def __init__(self,
                 interpolation_method='linear',
                 extrapolation_method='clip',
                 time_step=1,
                 time_unit='seconds',
                 clip_start=0,
                 clip_end=0):

        self.interpolation_method = interpolation_method
        self.extrapolation_method = extrapolation_method
        self.time_step = float(time_step)
        self.time_unit = time_unit
        if self.time_unit not in ROUND_COMPATIBLE_TIME_UNIT:
            if self.time_step.is_integer():
                self.time_step = int(self.time_step)
            else:
                raise ValueError("Can not use non-integer time step with time unit {}".format(self.time_unit))
        self.resampling_step = str(self.time_step) + FREQUENCY_STRINGS.get(self.time_unit, '')
        self.clip_start = clip_start
        self.clip_end = clip_end

    def check(self):

        if self.interpolation_method not in INTERPOLATION_METHODS:
            raise ValueError(
                'Method "{0}" is not valid. Possible interpolation methods are: {1}.'.format(self.interpolation_method,
                                                                                             INTERPOLATION_METHODS))
        if self.extrapolation_method not in EXTRAPOLATION_METHODS:
            raise ValueError(
                'Method "{0}" is not valid. Possible extrapolation methods are: {1}.'.format(self.extrapolation_method,
                                                                                             EXTRAPOLATION_METHODS))
        if self.time_step < 0:
            raise ValueError('Time step can not be negative.')
        if self.time_unit not in TIME_UNITS:
            raise ValueError(
                '"{0}" is not a valid unit. Possible time units are: {1}'.format(self.time_unit, TIME_UNITS))


class Resampler:

    def __init__(self, params=None):

        if params is None:
            raise ValueError('ResamplerParams not specified.')
        self.params = params
        self.params.check()

    def transform(self, raw_df, datetime_column, groupby_columns=None):

        check_transform_arguments(datetime_column, groupby_columns)

        df = raw_df.copy()

        # drop all rows where the timestamp is null
        df = df.dropna(subset=[datetime_column])
        if nothing_to_do(df, min_len=2):
            logger.warning('The timeseries has less than 2 rows with values, can not resample.')
            return df

        df.loc[:, datetime_column] = pd.to_datetime(df[datetime_column])
        # when having multiple partitions, their time range is not necessarily the same
        # we thus compute a unified time index for all partitions
        reference_time_index = self._compute_full_time_index(df, datetime_column)
        columns_to_resample = [col for col in df.select_dtypes([int, float]).columns.tolist() if col != datetime_column]

        if groupby_columns:
            print('*********')
            print(groupby_columns)
            grouped = df.groupby(groupby_columns)
            resampled_groups = []

            for group_id, group in grouped:
                group_resampled = self._resample(group, datetime_column, columns_to_resample, reference_time_index,
                                                 df_id=group_id)
                group_resampled[groupby_columns] = group_id
                resampled_groups.append(group_resampled)

            df_resampled = pd.concat(resampled_groups)
        else:
            df_resampled = self._resample(df, datetime_column, columns_to_resample, reference_time_index)

        return df_resampled.reset_index(drop=True)

    def _compute_full_time_index(self, df, datetime_column):
        """
        From the resampling config, create the full index of the output dataframe.
        """
        start_time = df[datetime_column].min()
        end_time = df[datetime_column].max()
        clip_start = self.params.clip_start
        clip_end = self.params.clip_end
        frequency = self.params.resampling_step
        time_step = self.params.time_step
        time_unit = self.params.time_unit

        return generate_date_range(start_time, end_time, clip_start, clip_end, frequency, time_step, time_unit)

    def _resample(self, df, datetime_column, columns_to_resample, reference_time_index, df_id=''):

        """
        1. Move datetime column to the index.
        2. Merge the original datetime index with the full_time_index.
        3. Create a numerical index of the df and save the correspond index.
        """

        if have_duplicate(df, datetime_column):
            raise ValueError('The timeseries {} contain duplicate timestamps.'.format(df_id))

        if nothing_to_do(df, min_len=2):
            logger.warning('The timeseries {} has less than 2 rows with values, can not resample.'.format(df_id))
            return df

        # `scipy.interpolate.interp1d` does not like empty columns, so we need to filter these out first
        filtered_columns_to_resample = filter_empty_columns(df, columns_to_resample)
        if len(filtered_columns_to_resample) == 0:
            logger.warning('All numerical columns are empty for the timeseries {}.'.format(df_id))
            return pd.DataFrame({datetime_column: reference_time_index},
                                columns=[datetime_column] + columns_to_resample)

        df_resample = df.set_index(datetime_column).sort_index().copy()
        # merge the reference time index with the original ones that has data
        # cf: https://stackoverflow.com/questions/47148446/pandas-resample-interpolate-is-producing-nans
        df_resample = df_resample.reindex(df_resample.index | reference_time_index)

        # `scipy.interpolate.interp1d` only works with numerical index, so we create one
        df_resample['numerical_index'] = range(len(df_resample))
        reference_index = df_resample.loc[reference_time_index, 'numerical_index']

        df_resample = df_resample.rename_axis(datetime_column).reset_index()

        interpolation_index_mask = (df_resample[datetime_column] >= df[datetime_column].min()) & (
                df_resample[datetime_column] <= df[datetime_column].max())
        interpolation_index = df_resample.index[interpolation_index_mask]

        extrapolation_index_mask = (df_resample[datetime_column] < df[datetime_column].min()) | (
                df_resample[datetime_column] > df[datetime_column].max())
        extrapolation_index = df_resample.index[extrapolation_index_mask]

        index_with_data = df_resample.loc[interpolation_index, filtered_columns_to_resample].dropna(how='all').index
        interpolation_function = interpolate.interp1d(index_with_data,
                                                      df_resample.loc[index_with_data, filtered_columns_to_resample],
                                                      kind=self.params.interpolation_method,
                                                      axis=0,
                                                      fill_value='extrapolate')

        df_resample.loc[interpolation_index, filtered_columns_to_resample] = interpolation_function(
            df_resample.loc[interpolation_index].index)

        if self.params.extrapolation_method == "interpolation":
            df_resample.loc[extrapolation_index, filtered_columns_to_resample] = interpolation_function(
                df_resample.loc[extrapolation_index].index)

        if self.params.extrapolation_method == "clip":
            df_resample = df_resample.ffill().bfill()

        return df_resample.loc[reference_index].drop('numerical_index', axis=1)
