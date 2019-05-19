# -*- coding: utf-8 -*-
import pandas as pd
import logging
from scipy import interpolate

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='timeseries-preparation plugin %(levelname)s - %(message)s')

TIME_STEP_MAPPING = {
    'year': 'A',
    'month': 'M',
    'week': 'W',
    'day': 'D',
    'hour': 'H',
    'minute': 'T',
    'second': 'S',
    'millisecond': 'L',
    'microsecond': 'us',
    'nanosecond': 'ns'
}

ROUND_COMPATIBLE_TIME_UNIT = ['day', 'hour', 'minute', 'second', 'millisecond', 'microsecond', 'nanosecond']
INTERPOLATION_METHODS = ['None', 'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', 'next']
EXTRAPOLATION_METHODS = ['None', 'clip', 'interpolation']
TIME_UNITS = list(TIME_STEP_MAPPING.keys()) + ['row']


class ResamplerParams:

    def __init__(self,
                 interpolation_method='linear',
                 extrapolation_method='clip',
                 time_step=1,
                 time_unit='second',
                 offset=0,
                 crop=0):

        self.interpolation_method = interpolation_method
        self.extrapolation_method = extrapolation_method
        self.time_step = time_step
        self.time_unit = time_unit
        if self.time_unit not in ROUND_COMPATIBLE_TIME_UNIT:
            if self.time_step.is_integer():
                self.time_step = int(self.time_step)
            else:
                raise ValueError("Can not use non-integer time step with time unit {}".format(self.time_unit))
        self.resampling_step = str(self.time_step) + TIME_STEP_MAPPING.get(self.time_unit, '')
        self.offset = offset
        self.crop = crop

    def check(self):

        if self.interpolation_method not in INTERPOLATION_METHODS:
            raise ValueError('Method "{0}" is not valid. Possible interpolation methods are: {1}.'.format(
                self.interpolation_method, INTERPOLATION_METHODS))
        if self.extrapolation_method not in EXTRAPOLATION_METHODS:
            raise ValueError('Method "{0}" is not valid. Possible extrapolation methods are: {1}.'.format(
                self.extrapolation_method, EXTRAPOLATION_METHODS))
        if self.time_step < 0:
            raise ValueError('Time step can not be negative.')
        if self.time_unit not in TIME_UNITS:
            raise ValueError('"{0}" is not a valid unit. Possible time units are: {1}'.format(
                self.time_unit, TIME_UNITS))


class Resampler:

    def __init__(self, params=None):

        assert params is not None, "ResamplerParams not specified."
        self.params = params
        self.params.check()

    def _compute_full_time_index(self, df):
        """
        From the resampling config, create the full index of the output dataframe.
        """
        time_unit = self.params.time_unit

        offset_value = self._get_date_offset(self.params.offset)
        crop_value = self._get_date_offset(self.params.crop)

        if time_unit in ROUND_COMPATIBLE_TIME_UNIT:
            start_index = df[self.datetime_column].min().round(TIME_STEP_MAPPING.get(time_unit)) + offset_value
            end_index = df[self.datetime_column].max().round(TIME_STEP_MAPPING.get(time_unit)) + crop_value
        else:  # for week, month, year we round up to closest day
            start_index = df[self.datetime_column].min().round('D') + offset_value
            end_index = df[self.datetime_column].max().round('D') + crop_value

        full_time_index = pd.date_range(start=start_index,
                                        end=end_index,
                                        freq=self.params.resampling_step)

        return full_time_index

    def _check_df(self, df):

        if len(df) < 2:
            return False
        else:
            return True

    def _get_date_offset(self, offset_value):

        return pd.DateOffset(**{self.params.time_unit + 's': offset_value})

    def _resample(self, df):

        """
        1. Move datetime column to the index.
        2. Merge the original datetime index with the full_time_index.
        3. Create a numerical index of the df and save the correspond index.
        """
        temp_df = df.set_index(self.datetime_column).sort_index()
        try:
            temp_df = temp_df.reindex(temp_df.index | self.full_time_index)
            temp_df['reference_index'] = range(len(temp_df))
            reference_index = temp_df.loc[self.full_time_index, 'reference_index']
        except Exception as e:
            if e.message == 'cannot reindex from a duplicate axis':
                raise ValueError('{}: Your timeseries contain duplicate timestamps.'.format(str(e)))
            else:
                raise ValueError(str(e))

        temp_df = temp_df.rename_axis(self.datetime_column).reset_index()

        interpolation_index_mask = (temp_df[self.datetime_column] >= df[self.datetime_column].min()) & (
                temp_df[self.datetime_column] <= df[self.datetime_column].max())
        interpolation_index = temp_df.index[interpolation_index_mask]
        df_to_interpolate = temp_df.loc[interpolation_index]

        extrapolation_index_mask = (temp_df[self.datetime_column] < df[self.datetime_column].min()) | (
                temp_df[self.datetime_column] > df[self.datetime_column].max())
        extrapolation_index = temp_df.index[extrapolation_index_mask]
        df_to_extrapolate = temp_df.loc[extrapolation_index]

        df_to_fit = df_to_interpolate.dropna()
        interpolation_function = interpolate.interp1d(df_to_fit.index,
                                                      df_to_fit[self.columns_to_resample],
                                                      kind=self.params.interpolation_method,
                                                      axis=0,
                                                      fill_value='extrapolate')

        df_to_interpolate[self.columns_to_resample] = interpolation_function(df_to_interpolate.index)

        if self.params.extrapolation_method == "interpolate":
            df_to_extrapolate[self.columns_to_resample] = interpolation_function(df_to_extrapolate.index)

        df_resampled = pd.concat([df_to_interpolate, df_to_extrapolate])
        df_resampled = df_resampled.sort_index()

        if self.params.extrapolation_method == "clip":
            df_resampled = df_resampled.ffill().bfill()

        df_final = df_resampled.loc[reference_index]
        df_final = df_final.drop('reference_index', axis=1)

        return df_final

    def transform(self, raw_df, datetime_column, groupby_columns=None):

        passed = self._check_df(raw_df)
        if not passed:
            return raw_df

        df = raw_df.copy()
        df[datetime_column] = pd.to_datetime(df[datetime_column])
        # df = raw_df.set_index(datetime_column).sort_index()

        # TODO is it good practice to do the following ?
        self.datetime_column = datetime_column
        self.columns_to_resample = [col for col in df.select_dtypes([int, float]).columns.tolist() if col != 'time_col']
        self.full_time_index = self._compute_full_time_index(df)

        if groupby_columns:
            grouped = df.groupby(groupby_columns)
            resampled_groups = []

            for group_id, group in grouped:
                group_resampled = self._resample(group)
                group_resampled[groupby_columns] = group_id
                resampled_groups.append(group_resampled)

            df_resampled = pd.concat(resampled_groups)

        else:
            df_resampled = self._resample(df)

        # put the datetime index back to the dataframe
        df_final = df_resampled.reset_index(drop=True)
        return df_final
