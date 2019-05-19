# -*- coding: utf-8 -*-
import pandas as pd
import logging
import re
from pandas.tseries.frequencies import to_offset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='timeseries-preparation plugin %(levelname)s - %(message)s')


EXTREMA_TYPES = ['min', 'max']

class ExtremaExtractorParams:
    
    def __init__(self,
                 window_roller = None,
                 extrema_type = 'max'): 
        self.window_roller = window_roller
        self.extrema_type = extrema_type
        
    def check(self):
        if self.window_roller is None:
            raise ValueError('WindowRoller object is not specified.')
        if self.extrema_type not in EXTREMA_TYPES:
            raise ValueError('{0} is not a valid options. Possible extrema types are: {1}'.format(self.extrema_type, EXTREMA_TYPES))

class ExtremaExtractor:
    
    def __init__(self, params=None):
        
        assert params is not None, "ExtremaExtractorParams instance is not specified."
        self.params = params
        self.params.check()
        
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

        extrema_neighbor_df = pd.concat(extrema_neigbors).drop_duplicates()

        return extrema_neighbor_df, extrema_value
        
    def compute(self, raw_df, datetime_column, extrema_column, groupby_columns=None):
        """
        From the input dataset, keep only the extrema and theirs surrounding, then compute
        aggregated statistics on what's going on around the extrema.
        """

        df = raw_df.set_index(datetime_column).sort_index()
        df = df.fillna(0)
        if groupby_columns:
            grouped = df.groupby(groupby_columns)
            computed_groups = []
            for _, group in grouped:
                try: #TODO remove the try except (it was to avoid empty, nan dataset)
                    extrema_neighbor_df, extrema_value = self._find_extrema_neighbor_zone(group, extrema_column)
                    rolling_df = self.params.window_roller.compute(extrema_neighbor_df)
                    extrema_df = rolling_df.loc[group[extrema_column]==extrema_value]
                    extrema_df = extrema_df.rename_axis(datetime_column).reset_index()
                    computed_groups.append(extrema_df)
                except:
                    continue
            final_df = pd.concat(computed_groups)
        else:
            extrema_neighbor_df, extrema_value = self._find_extrema_neighbor_zone(df, extrema_column)
            rolling_df = self.params.window_roller.compute(extrema_neighbor_df)
            final_df = rolling_df.loc[df[extrema_column]==extrema_value]
            final_df = final_df.rename_axis(datetime_column).reset_index()
        return final_df