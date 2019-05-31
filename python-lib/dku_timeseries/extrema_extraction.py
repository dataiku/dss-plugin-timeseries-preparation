# -*- coding: utf-8 -*-
import pandas as pd
import logging
import re
from pandas.tseries.frequencies import to_offset
from dataframe_helpers import has_duplicates, nothing_to_do, filter_empty_columns, generic_check_compute_arguments
from timeseries_helpers import get_date_offset, generate_date_range

logger = logging.getLogger(__name__)

EXTREMA_TYPES = ['min', 'max']


class ExtremaExtractorParams:

    def __init__(self, window_aggregator=None, extrema_type='max'):
        self.window_aggregator = window_aggregator
        self.extrema_type = extrema_type

    def check(self):
        if self.window_aggregator is None:
            raise ValueError('WindowAggregator object is not specified.')
        if self.extrema_type not in EXTREMA_TYPES:
            raise ValueError(
                '{0} is not a valid options. Possible extrema types are: {1}'.format(self.extrema_type, EXTREMA_TYPES))


class ExtremaExtractor:

    def __init__(self, params=None):

        self.params = params
        if params is None:
            raise ValueError('ExtremaExtractorParams instance is not specified.')
        self.params.check()

    def compute(self, df, datetime_column, extrema_column, groupby_columns=None):

        generic_check_compute_arguments(datetime_column, groupby_columns)
        df_copy = df.copy()

        # drop all rows where the timestamp is null
        df_copy = df_copy.dropna(subset=[datetime_column])
        if nothing_to_do(df_copy, min_len=2):
            logger.warning('The timeseries has less than 2 rows with values, can not resample.')
            return df_copy

        df_copy.loc[:, datetime_column] = pd.to_datetime(df[datetime_column])
        if groupby_columns:
            grouped = df_copy.groupby(groupby_columns)
            computed_groups = []
            for group_id, group in grouped:
                logger.info("Computing for group: ", group_id)
                extrema_neighbor_df, extrema_value = self._find_extrema_neighbor_zone(group, datetime_column,
                                                                                      extrema_column, df_id=group_id)
                if extrema_neighbor_df is None:
                    extrema_df = pd.DataFrame({groupby_columns[0]: [group_id]})
                else:
                    rolling_df = self.params.window_aggregator.compute(extrema_neighbor_df, datetime_column)
                    extrema_df = rolling_df.loc[
                        rolling_df[extrema_column] == extrema_value].copy()  # avoid .loc warning
                    extrema_df.loc[:, groupby_columns[0]] = group_id

                computed_groups.append(extrema_df)

            final_df = pd.concat(computed_groups)
            final_df = final_df.reset_index(drop=True)
        else:
            extrema_neighbor_df, extrema_value = self._find_extrema_neighbor_zone(df_copy, datetime_column,
                                                                                  extrema_column)
            rolling_df = self.params.window_aggregator.compute(extrema_neighbor_df, datetime_column)
            final_df = rolling_df.loc[rolling_df[extrema_column] == extrema_value].reset_index(drop=True)
        return final_df

    def _find_extrema_neighbor_zone(self, df, datetime_column, extrema_column, df_id=''):

        if has_duplicates(df, datetime_column):
            raise ValueError('The timeseries {} contain duplicate timestamps.'.format(df_id))

        df = df.set_index(datetime_column).sort_index()

        extrema_value = df[extrema_column].agg(self.params.extrema_type)
        extrema = df[df[extrema_column] == extrema_value]
        extrema_neigbors = []
        for extremum in extrema.index:
            date_offset = get_date_offset(self.params.window_aggregator.params.window_unit,
                                          self.params.window_aggregator.params.window_width)
            start_time = extremum - date_offset
            end_time = extremum + date_offset
            df_neighbor = df.loc[start_time:end_time]
            extrema_neigbors.append(df_neighbor)

        extrema_neighbor_df = pd.concat(extrema_neigbors).drop_duplicates()
        extrema_neighbor_df = extrema_neighbor_df.rename_axis(datetime_column).reset_index()
        return extrema_neighbor_df, extrema_value
