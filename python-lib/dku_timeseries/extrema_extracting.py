# -*- coding: utf-8 -*-
import pandas as pd
import logging
import re
from pandas.tseries.frequencies import to_offset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='timeseries-preparation plugin %(levelname)s - %(message)s')

EXTREMA_TYPES = ['min', 'max']


class ExtremaExtractorParams:

    def __init__(self, window_roller=None, extrema_type='max'):
        self.window_roller = window_roller
        self.extrema_type = extrema_type

    def check(self):
        if self.window_roller is None:
            raise ValueError('WindowRoller object is not specified.')
        if self.extrema_type not in EXTREMA_TYPES:
            raise ValueError(
                '{0} is not a valid options. Possible extrema types are: {1}'.format(self.extrema_type, EXTREMA_TYPES))


class ExtremaExtractor:

    def __init__(self, params=None):

        self.params = params
        if params is None:
            raise ValueError('ExtremaExtractorParams instance is not specified.')
        self.params.check()

    def compute(self, raw_df, datetime_column, extrema_column, groupby_columns=None):
        """
        From the input dataset, keep only the extrema and theirs surrounding, then compute
        aggregated statistics on what's going on around the extrema.
        """
        if not isinstance(extrema_column, basestring):
            raise ValueError('extrema_column param must be string. Got: ' + str(extrema_column))
        if not isinstance(datetime_column, basestring):
            raise ValueError('datetime_column param must be string. Got: ' + str(datetime_column))
        if groupby_columns:
            if not isinstance(groupby_columns, list):
                raise ValueError('groupby_columns param must be an array of strings. Got: '+ str(groupby_columns))
            for col in groupby_columns:
                if not isinstance(col, basestring):
                    raise ValueError('groupby_columns param must be an array of strings. Got: ' + str(col))

        if self._nothing_to_do(raw_df):
            return raw_df

        df = raw_df.copy()
        df.loc[:, datetime_column] = pd.to_datetime(df[datetime_column])
        df = df.set_index(datetime_column).sort_index()

        if groupby_columns:
            grouped = df.groupby(groupby_columns)
            computed_groups = []
            for group_id, group in grouped:
                print group_id
                extrema_neighbor_df, extrema_value = self._find_extrema_neighbor_zone(group, extrema_column)
                if extrema_neighbor_df is None:
                    extrema_df = pd.DataFrame({groupby_columns[0]: [group_id]})
                else:
                    extrema_neighbor_df = extrema_neighbor_df.rename_axis(datetime_column).reset_index()
                    rolling_df = self.params.window_roller.compute(extrema_neighbor_df, datetime_column)
                    extrema_df = rolling_df.loc[rolling_df[extrema_column] == extrema_value].copy() # avoid .loc warning
                    extrema_df.loc[:, groupby_columns[0]] = group_id

                computed_groups.append(extrema_df)

            final_df = pd.concat(computed_groups)
            final_df = final_df.reset_index(drop=True)
        else:
            extrema_neighbor_df, extrema_value = self._find_extrema_neighbor_zone(df, extrema_column)
            extrema_neighbor_df = extrema_neighbor_df.rename_axis(datetime_column).reset_index()
            rolling_df = self.params.window_roller.compute(extrema_neighbor_df, datetime_column)
            final_df = rolling_df.loc[rolling_df[extrema_column] == extrema_value].reset_index(drop=True)
        return final_df

    def _nothing_to_do(self, df):
        return len(df) < 2

    def _find_extrema_neighbor_zone(self, df, extrema_column):

        extrema_value = df[extrema_column].agg(self.params.extrema_type)
        extrema = df[df[extrema_column] == extrema_value]
        extrema_neigbors = []
        for extremum in extrema.index:
            date_offset = self.params.window_roller.get_window_date_offset()
            start_time = extremum - date_offset
            end_time = extremum + date_offset
            df_neighbor = df.loc[start_time:end_time]
            extrema_neigbors.append(df_neighbor)

        if len(extrema_neigbors) > 0:
            extrema_neighbor_df = pd.concat(extrema_neigbors).drop_duplicates()
            return extrema_neighbor_df, extrema_value
        else:
            return None, None
