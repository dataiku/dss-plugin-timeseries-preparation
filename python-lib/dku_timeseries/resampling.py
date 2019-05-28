# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import logging
from scipy import interpolate

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
        full_time_index = self._compute_full_time_index(df, datetime_column)
        columns_to_resample = [col for col in df.select_dtypes([int, float]).columns.tolist() if col != 'time_col']

        if groupby_columns:
            print('*********')
            print(groupby_columns)
            grouped = df.groupby(groupby_columns)
            resampled_groups = []

            for group_id, group in grouped:
                group_resampled = self._resample(group, datetime_column, columns_to_resample, full_time_index)
                group_resampled[groupby_columns] = group_id
                resampled_groups.append(group_resampled)

            df_resampled = pd.concat(resampled_groups)
        else:
            df_resampled = self._resample(df, datetime_column, columns_to_resample, full_time_index)

        # put the datetime index back to the dataframe
        df_final = df_resampled.reset_index(drop=True)
        return df_final

    def _compute_full_time_index(self, df, datetime_column):
        """
        From the resampling config, create the full index of the output dataframe.
        """
        col = df[datetime_column]
        if len(col):
            rounding_freq_string = FREQUENCY_STRINGS.get(self.params.time_unit)
            clip_start_value = self._get_date_offset(self.params.clip_start)
            clip_end_value = self._get_date_offset(self.params.clip_end)
            if self.params.time_unit in ROUND_COMPATIBLE_TIME_UNIT:
                start_index = col.min().round(rounding_freq_string) + clip_start_value
                end_index = col.max().round(rounding_freq_string) - clip_end_value
                return pd.date_range(start=start_index, end=end_index, freq=self.params.resampling_step)
            else:  # for week, month, year we round up to closest day
                start_index = col.min().round('D') + clip_start_value
                end_index = col.max().round('D') - clip_end_value
                # for some reason date_range omit the last entry when dealing with months, years
                return pd.date_range(start=start_index, end=end_index + self._get_date_offset(self.params.time_step),
                                     freq=self.params.resampling_step)
        else:
            return None

    def _nothing_to_do(self, df):
        return len(df) < 2

    def _get_date_offset(self, offset_value):

        return pd.DateOffset(**{self.params.time_unit: offset_value})

    def _filter_empty_columns(self, df, columns_to_resample):
        filtered_columns_to_resample = []
        for col in columns_to_resample:
            if np.sum(df[col].notnull()) > 0:
                filtered_columns_to_resample.append(col)
        return filtered_columns_to_resample

    def _resample(self, df, datetime_column, columns_to_resample, full_time_index):

        """
        1. Move datetime column to the index.
        2. Merge the original datetime index with the full_time_index.
        3. Create a numerical index of the df and save the correspond index.
        """
        if self._nothing_to_do(df):
            logger.warning('The partition/dataset has less than 2 rows, can not resample.')
            return df

        df_resample = df.set_index(datetime_column).sort_index()
        try:
            df_resample = df_resample.reindex(df_resample.index | full_time_index)
            df_resample['reference_index'] = range(len(df_resample))
            reference_index = df_resample.loc[full_time_index, 'reference_index']
        except Exception as e:
            if e.message == 'cannot reindex from a duplicate axis':
                raise ValueError('{}: The timeseries contain duplicate timestamps.'.format(str(e)))
            else:
                raise e

        df_resample = df_resample.rename_axis(datetime_column).reset_index()
        # if we pass an empty column through `interp1d`, it will return an error, so we need to filter these out first
        filtered_columns_to_resample = self._filter_empty_columns(df_resample, columns_to_resample)
        if len(filtered_columns_to_resample) == 0:
            logger.warning('All numerical columns are empty for this group.')
            df_final = df_resample.loc[reference_index]
            df_final = df_final.drop('reference_index', axis=1)
            return df_final

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

        return df_resample.loc[reference_index].drop('reference_index', axis=1)
