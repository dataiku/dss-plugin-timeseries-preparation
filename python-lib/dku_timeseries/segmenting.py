# -*- coding: utf-8 -*-
import dataiku
import pandas as pd
import logging
import re
from pandas.tseries.frequencies import to_offset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='timeseries-preparation plugin %(levelname)s - %(message)s')

TIME_STEP_MAPPING = {
    'day': 'D',
    'hour': 'h',
    'minute': 'm',
    'second': 's',
    'millisecond': 'ms',
    'microsecond': 'us',
    'nanosecond': 'ns'
}

class SegmentExtractorParams:
    
    def __init__(self,
                 min_segment_duration_value = 0,
                 max_noise_duration_value = 0,
                 time_unit = 'second'):
        
        self.min_segment_duration_value = min_segment_duration_value
        self.max_noise_duration_value = max_noise_duration_value
        self.time_unit = time_unit
        
        self.min_segment_duration = pd.Timedelta(str(self.min_segment_duration_value) + TIME_STEP_MAPPING.get(self.time_unit, ''))
        self.max_noise_duration = pd.Timedelta(str(self.max_noise_duration_value) + TIME_STEP_MAPPING.get(self.time_unit, ''))
    
    def check(self):
        
        if self.min_segment_duration_value < 0:
            raise ValuError('Min segment duration can not be negative.')
        if self.max_noise_duration_value < 0:
            raise ValueError('Max noisy duration can not be negative.')
        if self.time_unit not in TIME_STEP_MAPPING:
            raise ValueError('{0} is not a valid time unit. Possible options are: {1}'.format(self.time_unit, TIME_STEP_MAPPING))

class SegmentExtractor:
    
    def __init__(self, params):
        assert params is not None, "ResamplerParams instance is not specified."
        self.params = params
        print self.params.__dict__
        self.params.check()
        
    def _detect_time_segment(self, df, chosen_col, lower_threshold, upper_threshold):        
        
        filtered_index = df[(df[chosen_col] >= lower_threshold) & (df[chosen_col] <= upper_threshold)].index
        
        if len(filtered_index) > 0:
            filtered_serie = filtered_index.to_series()

            d1 = filtered_serie.diff(1)
            d1 = d1.fillna(self.params.min_segment_duration)

            d2 = d1.shift(-1)
            d2 = d2.fillna(self.params.min_segment_duration)

            filtered = filtered_serie[d1 >= self.params.min_segment_duration].index
            filtered2 = filtered_serie[d2 >= self.params.min_segment_duration].index

            inds2 = np.vstack((filtered, filtered2)).T
        else: #empty
            inds2 = []
            
        return inds2
        
    def _detect_row_segment(self, x):
        
        x = np.atleast_1d(x).astype('float64')
        # deal with NaN's (by definition, NaN's are not greater than threshold)
        x[np.isnan(x)] = np.inf
        # indices of data greater than or equal to threshold
        inds = np.nonzero((x >= lower_threshold) & (x <= upper_threshold))[0]
        if inds.size:
            # initial and final indexes of almost continuous data
            inds = np.vstack((inds[np.diff(np.hstack((-np.inf, inds))) > n_below+1], \
                              inds[np.diff(np.hstack((inds, np.inf))) > n_below+1])).T
            # indexes of almost continuous data longer than or equal to n_above
            inds = inds[inds[:, 1]-inds[:, 0] >= n_above-1, :]
            # minimum amplitude of n_above2 values in x to detect
            if threshold2 is not None and inds.size:
                idel = np.ones(inds.shape[0], dtype=bool)
                for i in range(inds.shape[0]):
                    if np.count_nonzero(x[inds[i, 0]: inds[i, 1]+1] >= threshold2) < n_above2:
                        idel[i] = False
                inds = inds[idel, :]
        if not inds.size:
            inds = np.array([])  # standardize inds shape for output

        return inds
        
    def compute(self, raw_df, datetime_column, threshold_dict):
        
        df = raw_df.set_index(datetime_column).sort_index()
        
        for chosen_column, threshold_tuple in threshold_dict.items():
            lower_threshold, upper_threshold = threshold_tuple
            if self.params.time_unit == 'row':
                segment_indexes = self._detect_row_segment(df[chosen_column])
            else:
                segment_indexes = self._detect_time_segment(df, chosen_column, lower_threshold, upper_threshold)
            
        mask_list = []
        if len(segment_indexes) > 0:
            for start, end in segment_indexes:
                mask = (dfx.index >= start) & (dfx.index <= end)
                mask_list.append(mask)
            
            segment_df = df.loc[np.logical_or.reduce(mask_list)]

        else:
            segment_df = pd.DataFrame(columns=df.columns)

        return segment_df