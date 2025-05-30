# -*- coding: utf-8 -*-
import logging

import pandas as pd
import numpy as np
from scipy import interpolate

from dku_timeseries.dataframe_helpers import has_duplicates, nothing_to_do, filter_empty_columns, generic_check_compute_arguments
from dku_timeseries.timeseries_helpers import FREQUENCY_STRINGS, generate_date_range, reformat_time_value, format_resampling_step, reformat_time_step, format_group_id

logger = logging.getLogger(__name__)

INTERPOLATION_METHODS = ['linear', 'nearest', 'slinear', 'zero', 'quadratic', 'cubic', 'previous', 'next', 'constant', 'none']
EXTRAPOLATION_METHODS = ['none', 'clip', 'interpolation', 'no_extrapolation']
CATEGORY_IMPUTATION_METHODS = ['empty', 'constant', 'previous', 'next', 'clip', 'mode']
TIME_UNITS = list(FREQUENCY_STRINGS.keys()) + ['rows']


class ResamplerParams:

    def __init__(self,
                 interpolation_method='linear',
                 extrapolation_method='clip',
                 constant_value=0,
                 category_imputation_method='empty',
                 category_constant_value='',
                 time_step=1,
                 time_unit='seconds',
                 time_unit_end_of_week="SUN",
                 clip_start=0,
                 clip_end=0,
                 shift=0,
                 custom_start_date=None,
                 custom_end_date=None):

        self.interpolation_method = interpolation_method
        self.extrapolation_method = extrapolation_method
        self.constant_value = constant_value
        self.category_imputation_method = category_imputation_method
        self.category_constant_value = category_constant_value
        self.time_step = reformat_time_step(time_step, time_unit)
        self.time_unit = time_unit
        self.resampling_step = format_resampling_step(time_unit, self.time_step, time_unit_end_of_week)
        self.time_unit_end_of_week = time_unit_end_of_week
        self.clip_start = reformat_time_value(float(clip_start), time_unit)
        self.clip_end = reformat_time_value(float(clip_end), time_unit)
        self.shift = reformat_time_value(float(shift), time_unit)
        self.custom_start_date = custom_start_date
        self.custom_end_date = custom_end_date

    def check(self):

        if self.interpolation_method not in INTERPOLATION_METHODS:
            raise ValueError(
                'Method "{0}" is not valid. Possible interpolation methods are: {1}.'.format(self.interpolation_method, INTERPOLATION_METHODS))
        if self.extrapolation_method not in EXTRAPOLATION_METHODS:
            raise ValueError(
                'Method "{0}" is not valid. Possible extrapolation methods are: {1}.'.format(self.extrapolation_method, EXTRAPOLATION_METHODS))
        if self.time_step <= 0:
            raise ValueError('Time step can not be null or negative.')
        if self.category_imputation_method not in CATEGORY_IMPUTATION_METHODS:
            raise ValueError(
                '"{0}" is not valid way to impute category values. Possible methods are: {1}.'.format(self.category_imputation_method,
                                                                                                      CATEGORY_IMPUTATION_METHODS))
        if self.time_unit not in TIME_UNITS:
            raise ValueError(
                '"{0}" is not a valid unit. Possible time units are: {1}'.format(self.time_unit, TIME_UNITS))
        if self.time_unit == 'rows':
            raise NotImplementedError


class Resampler:
    RESAMPLEABLE_TYPES = [int, float, np.float32, np.int32]

    def __init__(self, params=None):

        if params is None:
            raise ValueError('ResamplerParams not specified.')
        self.params = params
        self.params.check()

    def transform(self, df, datetime_column, groupby_columns=None):    
        if groupby_columns is None:
            groupby_columns = []

        generic_check_compute_arguments(datetime_column, groupby_columns)
        df_copy = df.copy()

        # drop all rows where the timestamp is null
        df_copy = df_copy.dropna(subset=[datetime_column])
        if nothing_to_do(df_copy, min_len=2):
            logger.warning('The timeseries has less than 2 rows with values, can not resample.')
            return df_copy

        df_copy.loc[:, datetime_column] = pd.to_datetime(df_copy[datetime_column])
        # when having multiple timeseries, their time range is not necessarily the same
        # we thus compute a unified time index for all partitions
        reference_time_index = self._compute_full_time_index(df_copy, datetime_column)
        columns_to_resample = [col for col in df_copy.select_dtypes(Resampler.RESAMPLEABLE_TYPES).columns.tolist() if col != datetime_column and col not in groupby_columns]
        category_columns = [col for col in df.select_dtypes(exclude=Resampler.RESAMPLEABLE_TYPES).columns.tolist() if col != datetime_column and col not in columns_to_resample and
                            col not in groupby_columns]
        if groupby_columns:
            grouped = df_copy.groupby(groupby_columns)
            resampled_groups = []
            identifiers_number = len(groupby_columns)
            for group_id, group in grouped:
                logger.info("Computing for group: {}".format(group_id))
                group_resampled = self._resample(group.drop(groupby_columns, axis=1), datetime_column, columns_to_resample, category_columns,
                                                 reference_time_index,
                                                 df_id=group_id)
                group_id = format_group_id(group_id, identifiers_number)
                group_resampled[groupby_columns] = pd.DataFrame([group_id], index=group_resampled.index)
                resampled_groups.append(group_resampled)
            df_resampled = pd.concat(resampled_groups, sort=True)
        else:
            df_resampled = self._resample(df_copy, datetime_column, columns_to_resample, category_columns, reference_time_index)

        df_resampled = df_resampled[df.columns].reset_index(drop=True)

        return df_resampled

    def _can_customize_resampling_dates(self):
        return self.params.extrapolation_method == 'clip' or (self.params.extrapolation_method == 'interpolation' and self.params.interpolation_method != 'none')


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

        if self._can_customize_resampling_dates():
            custom_start_date = self.params.custom_start_date
            if custom_start_date:
                if custom_start_date < start_time:
                    start_time = custom_start_date

            custom_end_date = self.params.custom_end_date
            if custom_end_date:
                if custom_end_date > end_time:
                    end_time = custom_end_date

        return generate_date_range(start_time, end_time, clip_start, clip_end, shift, frequency, time_step, time_unit)

    def _resample(self, df, datetime_column, columns_to_resample, category_columns, reference_time_index, df_id=''):
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
        df_resample = df_resample.reindex(df_resample.index.union(reference_time_index))

        # `scipy.interpolate.interp1d` only works with numerical index, so we create one
        df_resample['numerical_index'] = range(len(df_resample))
        reference_index = df_resample.loc[reference_time_index, 'numerical_index']
        category_imputation_index = pd.Index([])

        df_resample = df_resample.rename_axis(datetime_column).reset_index()
        for filtered_column in filtered_columns_to_resample:

            df_without_nan = df.dropna(subset=[filtered_column], how='all')
            interpolation_index_mask = (df_resample[datetime_column] >= df_without_nan[datetime_column].min()) & (
                    df_resample[datetime_column] <= df_without_nan[datetime_column].max())
            interpolation_index = df_resample.index[interpolation_index_mask]

            extrapolation_index_mask = (df_resample[datetime_column] < df_without_nan[datetime_column].min()) | (
                    df_resample[datetime_column] > df_without_nan[datetime_column].max())
            extrapolation_index = df_resample.index[extrapolation_index_mask]

            index_with_data = df_resample.loc[interpolation_index, filtered_column].dropna(how='all').index

            if self.params.interpolation_method not in ['constant', 'none']:
                interpolation_function = interpolate.interp1d(index_with_data,
                                                              df_resample.loc[index_with_data, filtered_column],
                                                              kind=self.params.interpolation_method,
                                                              axis=0,
                                                              fill_value='extrapolate')

                df_resample.loc[interpolation_index, filtered_column] = interpolation_function(df_resample.loc[interpolation_index].index)
                if self.params.extrapolation_method == "interpolation":
                    df_resample.loc[extrapolation_index, filtered_column] = interpolation_function(df_resample.loc[extrapolation_index].index)
            elif self.params.interpolation_method == 'constant':
                if self.params.extrapolation_method == 'interpolation':
                    df_resample.loc[:, filtered_column] = df_resample.loc[:, filtered_column].fillna(self.params.constant_value)
                else:
                    df_resample.loc[interpolation_index, filtered_column] = df_resample.loc[interpolation_index, filtered_column].fillna(
                        self.params.constant_value)

            if self.params.extrapolation_method == "clip":
                temp_df = df_resample.copy().ffill().bfill()
                df_resample.loc[extrapolation_index, filtered_column] = temp_df.loc[extrapolation_index, filtered_column]
            elif self.params.extrapolation_method == "no_extrapolation":
                reference_index = reference_index[~reference_index.isin(extrapolation_index.values)]
            category_imputation_index = category_imputation_index.union(extrapolation_index).union(interpolation_index)

        if len(category_columns) > 0 and len(category_imputation_index) > 0 and self.params.category_imputation_method != "empty":
            df_processed = df_resample.loc[category_imputation_index]
            df_resample.loc[category_imputation_index] = self._fill_in_category_values(df_processed, category_columns)
        df_resampled = df_resample.loc[reference_index].drop('numerical_index', axis=1)
        return df_resampled

    def _fill_in_category_values(self, df, category_columns):
        category_filled_df = df.copy()
        if self.params.category_imputation_method == "constant":
            category_filled_df.loc[:, category_columns] = category_filled_df.loc[:, category_columns].fillna(self.params.category_constant_value)
        elif self.params.category_imputation_method == "previous":
            category_filled_df.loc[:, category_columns] = category_filled_df.loc[:, category_columns].ffill()
        elif self.params.category_imputation_method == "next":
            category_filled_df.loc[:, category_columns] = category_filled_df.loc[:, category_columns].bfill()
        elif self.params.category_imputation_method == "clip":
            category_filled_df.loc[:, category_columns] = category_filled_df.loc[:, category_columns].ffill().bfill()
        elif self.params.category_imputation_method == "mode":
            # .mode() loses the timezone info for any datetimetz column
            most_frequent_categoricals = category_filled_df.loc[:, category_columns].mode().iloc[0]

            for col in category_columns:
                # only perform conversion if the column has a timezone
                if pd.api.types.is_datetime64_any_dtype(category_filled_df[col]) and category_filled_df[col].dt.tz is not None:
                    if most_frequent_categoricals[col].tzinfo is None: # tz-naive timestamp -> localize
                        most_frequent_categoricals[col] = most_frequent_categoricals[col].tz_localize("UTC")
                    else: # tz_convert
                        most_frequent_categoricals[col] = most_frequent_categoricals[col].tz_convert("UTC")

            category_filled_df.loc[:, category_columns] = category_filled_df.loc[:, category_columns].fillna(most_frequent_categoricals)
        return category_filled_df
