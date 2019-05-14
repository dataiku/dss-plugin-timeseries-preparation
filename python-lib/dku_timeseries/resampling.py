# -*- coding: utf-8 -*-
import dataiku
import pandas as pd
import logging


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

INTERPOLATION_METHODS = ['None', 'nearest', 'previous',
                         'next', 'linear', 'quadratic', 'cubic', 'barycentric']
EXTRAPOLATION_METHODS = ['None', 'clip', 'interpolation']
TIME_UNITS = list(TIME_STEP_MAPPING.keys()) + ['row']


class ResamplerParams:

    def __init__(self,
                 interpolation_method='linear',
                 extrapolation_method='clip',
                 time_step_size=1,
                 time_unit='second',
                 offset=0,
                 crop=0,
                 groupby_cols=None):

        self.interpolation_method = interpolation_method
        self.extrapolation_method = extrapolation_method
        self.time_step_size = time_step_size
        self.time_unit = time_unit
        self.resampling_step = str(self.time_step_size) + TIME_STEP_MAPPING.get(self.time_unit, '')
        self.offset = offset
        self.crop = crop
        self.groupby_cols = groupby_cols

    def check(self):

        if self.interpolation_method not in INTERPOLATION_METHODS:
            raise ValueError('Method "{0}" is not valid. Possible interpolation methods are: {1}.'.format(
                self.interpolation_method, INTERPOLATION_METHODS))
        if self.extrapolation_method not in EXTRAPOLATION_METHODS:
            raise ValueError('Method "{0}" is not valid. Possible extrapolation methods are: {1}.'.format(
                self.extrapolation_method, EXTRAPOLATION_METHODS))
        if self.time_step_size < 0:
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

        offset_value = self._get_date_offset(self.params.offset) 
        crop_value = self._get_date_offset(self.params.crop)

        start_index = df.index.min().round(TIME_STEP_MAPPING.get(self.params.time_unit)) + offset_value
        end_index = df.index.max().round(TIME_STEP_MAPPING.get(self.params.time_unit)) + crop_value

        full_time_index = pd.date_range(start=start_index,
                                        end=end_index,
                                        freq=self.params.resampling_step)

        return full_time_index

    def _find_datetime_column(self, df):
        datetime_cols = df.select_dtypes(include='datetime').columns.tolist()
        if len(datetime_cols) == 0:
            raise ValueError('Datetime column not specified.')
        elif len(datetime_cols) == 1:
            return datetime_cols[0]
        else:
            raise ValueError('Multiple datetime columns detected.')

    def _check_df(self, df):

        if len(df) == 0:
            return False
        else:
            return True

    def _get_date_offset(self, offset_value):
        
        if self.params.time_unit == 'year':
            return pd.DateOffset(years=offset_value)
        elif self.params.time_unit == 'month':
            return pd.DateOffset(months=offset_value)
        elif self.params.time_unit == 'week':
            return pd.DateOffset(weeks=offset_value)
        elif self.params.time_unit == 'day':
            return pd.DateOffset(days=offset_value)
        elif self.params.time_unit == 'hour':
            return pd.DateOffset(hours=offset_value)
        elif self.params.time_unit == 'minute':
            return pd.DateOffset(minutes=offset_value)
        elif self.params.time_unit == 'second':
            return pd.DateOffset(seconds=offset_value)
        elif self.params.time_unit == 'microsecond':
            return pd.DateOffset(microseconds=offset_value)
        elif self.params.time_unit == 'nanosecond':
            return pd.DateOffset(nanoseconds=offset_value)

    def _resample(self, df): 
        
        try:
            temp_df = df.reindex(df.index | self.full_time_index)
        except Exception, e:
            raise ValueError('{}: Your timeseries contain dupplicate timestamps.'.format(str(e)))

        if self.params.interpolation_method == 'next':  # no `next` method with pd.interpolate()
            df_interpolated = temp_df.bfill()
        elif self.params.interpolation_method == 'previous':  # no `previous` method with pd.interpolate()
            df_interpolated = temp_df.ffill()
        else:
            df_interpolated = temp_df.interpolate(
                self.params.interpolation_method)

        if self.params.extrapolation_method == "clip":
            df_extrapolated = df_interpolated.ffill().bfill()
        elif self.params.interpolation_method == 'next':
            df_extrapolated = df_interpolated.bfill()
        elif self.params.interpolation_method == 'previous':
            df_extrapolated = df_interpolated.ffill()
        else:
            df_extrapolated = df_interpolated

        groupby_cols = self.params.groupby_cols
        if groupby_cols:
            # when having groups, there will very likely be empty rows before and after 
            # (because we use full index parameters) 
            df_extrapolated[groupby_cols] = df_extrapolated[groupby_cols].ffill().bfill()

        df_resampled = df_extrapolated.reindex(self.full_time_index)
        return df_resampled

    def transform(self, raw_df, datetime_column):

        passed = self._check_df(raw_df)
        if not passed:
            return raw_df
        df = raw_df.set_index(datetime_column).sort_index()
        self.full_time_index = self._compute_full_time_index(df)

        if self.params.groupby_cols:
            grouped = df.groupby(self.params.groupby_cols)
            resampled_groups = []

            for group_id, group in grouped:
                df_extrapolated = self._resample(group)
                resampled_groups.append(df_extrapolated)

            df_transformed = pd.concat(resampled_groups)

        else:
            df_transformed = self._resample(df)

        # put the datetime index back to the dataframe
        final_df = df_transformed.rename_axis(datetime_column).reset_index()
        return final_df