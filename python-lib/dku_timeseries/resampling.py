# -*- coding: utf-8 -*-
import dataiku
import pandas as pd
import logging


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='timeseries-preparation plugin %(levelname)s - %(message)s')


TIME_STEP_MAPPING = {
    'year': 'A',
    'quarter': 'Q',
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

OFFSET_MAPPING = {
    'day': 'D',
    'hour': 'h',
    'minute': 'm',
    'second': 's',
    'millisecond': 'ms',
    'microsecond': 'us',
    'nanosecond': 'ns'
}

INTERPOLATION_METHODS = ['None', 'nearest', 'previous',
                         'next', 'linear', 'quadratic', 'cubic', 'barycentric']
EXTRAPOLATION_METHODS = ['None', 'clip', 'interpolation']
TIME_UNITS = TIME_STEP_MAPPING.keys() + ['row']
OFFSET_UNITS = OFFSET_MAPPING.keys() + ['row']


class ResamplerParams:

    def __init__(self,
                 datetime_column = None,
                 interpolation_method='linear',
                 extrapolation_method='clip',
                 time_step_size=1,
                 time_unit='second',
                 offset=0,
                 crop=0,
                 groupby_cols=None):

        self.datetime_column = datetime_column
        self.interpolation_method = interpolation_method
        self.extrapolation_method = extrapolation_method
        self.time_step_size = time_step_size
        self.time_unit = time_unit
        self.offset = offset
        self.crop = crop
        self.groupby_cols = groupby_cols

    def check(self):

        if self.datetime_column is None:
            raise ValueError('Timestamp column not defined.')

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
        """
        if self.offset != 0 and self.time_unit not in OFFSET_UNITS:
            raise ValueError('Can not use offset with "{0}" unit. Possible time units are: {1}'.format(
                self.time_unit, OFFSET_UNITS))
        if self.crop != 0 and self.time_unit not in OFFSET_UNITS:
            raise ValueError('Can not use crop with "{0}" unit. Possible time units are: {1}'.format(
                self.time_unit, OFFSET_UNITS))
        """ 

class Resampler:

    def __init__(self, params=None):

        assert params is not None, "ResamplerParams not specified."
        self.params = params
        self.params.check()

    def _compute_full_time_index(self, df):
        """
        From the resampling config, create the full index of the output dataframe.
        """

        resampling_step = str(self.params.time_step_size) + TIME_STEP_MAPPING.get(self.params.time_unit, '')

        # pd.Timedelta does not have this units, so we convert them to `day` 
        if self.params.time_unit in ['week', 'month', 'quarter', 'year']:
            day_conversion = {
                'year': 365,
                'quarter': 90,
                'month': 30,
                'week': 7
            }
            offset_value = self.params.offset * day_conversion.get(self.params.time_unit)
            crop_value = self.params.crop * day_conversion.get(self.params.time_unit)
            offset_step = str(offset_value) + 'day'
            crop_step = str(crop_value) + 'day'
        else:

            offset_step = str(self.params.offset) + OFFSET_MAPPING.get(self.params.time_unit, '')
            crop_step = str(self.params.crop) + OFFSET_MAPPING.get(self.params.time_unit, '')

        offset_time_delta = pd.Timedelta(offset_step)
        crop_time_delta = pd.Timedelta(crop_step)

        start_index = df.index.min() + offset_time_delta
        end_index = df.index.max() + crop_time_delta

        full_time_index = pd.date_range(start=start_index,
                                        end=end_index,
                                        freq=resampling_step)

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
        # TODO should we check the input df first for the prerequisite ?
        return None

    def _resample(self, df): #TODO we dont actually use group_id

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
            df_extrapolated[groupby_cols] = df_extrapolated[groupby_cols].ffill().bfill()

        df_resampled = df_extrapolated.reindex(self.full_time_index)
        return df_resampled

    def transform(self, raw_df):

        datetime_column = self.params.datetime_column
        # TODO sort_index() is necessary ?
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