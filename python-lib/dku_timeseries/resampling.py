# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import logging
from scipy import interpolate

from dku_timeseries.dataframe_helpers import has_duplicates, nothing_to_do, filter_empty_columns, generic_check_compute_arguments
from dku_timeseries.timeseries_helpers import FREQUENCY_STRINGS, ROUND_COMPATIBLE_TIME_UNIT, generate_date_range, reformat_time_value

logger = logging.getLogger(__name__)

INTERPOLATION_METHODS = ['linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', 'next']
EXTRAPOLATION_METHODS = ['None', 'clip', 'interpolation']
TIME_UNITS = list(FREQUENCY_STRINGS.keys()) + ['rows']


class ResamplerParams:

    def __init__(self,
                 interpolation_method='linear',
                 extrapolation_method='clip',
                 time_step=1,
                 time_unit='seconds',
                 clip_start=0,
                 clip_end=0,
                 shift=0):

        self.interpolation_method = interpolation_method
        self.extrapolation_method = extrapolation_method
        self.time_step = reformat_time_value(float(time_step), time_unit)
        self.time_unit = time_unit
        self.resampling_step = str(self.time_step) + FREQUENCY_STRINGS.get(self.time_unit, '')
        self.clip_start = reformat_time_value(float(clip_start), time_unit)
        self.clip_end = reformat_time_value(float(clip_end), time_unit)
        self.shift = reformat_time_value(float(shift), time_unit)

    def check(self):

        if self.interpolation_method not in INTERPOLATION_METHODS:
            raise ValueError(
                'Method "{0}" is not valid. Possible interpolation methods are: {1}.'.format(self.interpolation_method, INTERPOLATION_METHODS))
        if self.extrapolation_method not in EXTRAPOLATION_METHODS:
            raise ValueError(
                'Method "{0}" is not valid. Possible extrapolation methods are: {1}.'.format(self.extrapolation_method, EXTRAPOLATION_METHODS))
        if self.time_step < 0:
            raise ValueError('Time step can not be negative.')
        if self.time_unit not in TIME_UNITS:
            raise ValueError(
                '"{0}" is not a valid unit. Possible time units are: {1}'.format(self.time_unit, TIME_UNITS))
        if self.time_unit == 'rows':
            raise NotImplementedError


class Resampler:

    def __init__(self, params=None):

        if params is None:
            raise ValueError('ResamplerParams not specified.')
        self.params = params
        self.params.check()

    def transform(self, df, datetime_column, groupby_columns=None):

        generic_check_compute_arguments(datetime_column, groupby_columns)
        df_copy = df.copy()

        # drop all rows where the timestamp is null
        df_copy = df_copy.dropna(subset=[datetime_column])
        if nothing_to_do(df_copy, min_len=2):
            logger.warning('The timeseries has less than 2 rows with values, can not resample.')
            return df_copy

        df_copy.loc[:, datetime_column] = pd.to_datetime(df_copy[datetime_column])
        # when having multiple partitions, their time range is not necessarily the same
        # we thus compute a unified time index for all partitions
        reference_time_index = self._compute_full_time_index(df_copy, datetime_column)
        columns_to_resample = [col for col in df_copy.select_dtypes([int, float]).columns.tolist() if col != datetime_column]

        if groupby_columns:
            grouped = df_copy.groupby(groupby_columns)
            resampled_groups = []
            for group_id, group in grouped:
                logger.info("Computing for group: {}".format(group_id))
                group_resampled = self._resample(group.drop(groupby_columns, axis=1), datetime_column, columns_to_resample, reference_time_index, df_id=group_id)
                group_resampled.loc[:, groupby_columns[0]] = group_id  # TODO make this work with multiple group cols
                resampled_groups.append(group_resampled)
            df_resampled = pd.concat(resampled_groups)
        else:
            df_resampled = self._resample(df_copy, datetime_column, columns_to_resample, reference_time_index)

        df_resampled = df_resampled[df.columns].reset_index(drop=True)

        return df_resampled

    def _compute_full_time_index(self, df, datetime_column):
        """
        From the resampling config, create the full index of the output dataframe.
        """
        start_time = df[datetime_column].min()
        end_time = df[datetime_column].max()
        clip_start = self.params.clip_start
        clip_end = self.params.clip_end
        shift = self.params.shift
        frequency = self.params.resampling_step
        time_step = self.params.time_step
        time_unit = self.params.time_unit
        return generate_date_range(start_time, end_time, clip_start, clip_end, shift, frequency, time_step, time_unit)

    def _resample(self, df, datetime_column, columns_to_resample, reference_time_index, df_id=''):
        """
        1. Move datetime column to the index.
        2. Merge the original datetime index with the full_time_index.
        3. Create a numerical index of the df and save the correspond index.
        """

        if has_duplicates(df, datetime_column):
            raise ValueError('The time series {} contain duplicate timestamps.'.format(df_id))

        if nothing_to_do(df, min_len=2):
            logger.warning('The time series {} has less than 2 rows with values, can not resample.'.format(df_id))
            return df

        # `scipy.interpolate.interp1d` does not like empty columns, so we need to filter these out first
        filtered_columns_to_resample = filter_empty_columns(df, columns_to_resample)
        if len(filtered_columns_to_resample) == 0:
            logger.warning('All numerical columns are empty for the time series {}.'.format(df_id))
            return pd.DataFrame({datetime_column: reference_time_index}, columns=[datetime_column] + columns_to_resample)

        df_resample = df.set_index(datetime_column).sort_index().copy()
        # merge the reference time index with the original ones that has data
        # cf: https://stackoverflow.com/questions/47148446/pandas-resample-interpolate-is-producing-nans
        df_resample = df_resample.reindex(df_resample.index | reference_time_index)

        # `scipy.interpolate.interp1d` only works with numerical index, so we create one
        df_resample['numerical_index'] = range(len(df_resample))
        reference_index = df_resample.loc[reference_time_index, 'numerical_index']

        df_resample = df_resample.rename_axis(datetime_column).reset_index()

        # TODO different columns might start to have values at different rows -> how to handle
        df_without_nan = df.dropna(subset=filtered_columns_to_resample)
        interpolation_index_mask = (df_resample[datetime_column] >= df_without_nan[datetime_column].min()) & (df_resample[datetime_column] <= df_without_nan[datetime_column].max())
        interpolation_index = df_resample.index[interpolation_index_mask]

        extrapolation_index_mask = (df_resample[datetime_column] < df_without_nan[datetime_column].min()) | (df_resample[datetime_column] > df_without_nan[datetime_column].max())
        extrapolation_index = df_resample.index[extrapolation_index_mask]

        index_with_data = df_resample.loc[interpolation_index, filtered_columns_to_resample].dropna(how='all').index

        interpolation_function = interpolate.interp1d(index_with_data,
                                                      df_resample.loc[index_with_data, filtered_columns_to_resample],
                                                      kind=self.params.interpolation_method,
                                                      axis=0,
                                                      fill_value='extrapolate')

        df_resample.loc[interpolation_index, filtered_columns_to_resample] = interpolation_function(df_resample.loc[interpolation_index].index)

        if self.params.extrapolation_method == "interpolation":
            df_resample.loc[extrapolation_index, filtered_columns_to_resample] = interpolation_function(df_resample.loc[extrapolation_index].index)

        if self.params.extrapolation_method == "clip":
            df_resample = df_resample.ffill().bfill()

        return df_resample.loc[reference_index].drop('numerical_index', axis=1)