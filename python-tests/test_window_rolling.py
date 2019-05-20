# This is a test file intended to be used with pytest
# pytest automatically runs all the function starting with "test_"
# see https://docs.pytest.org for more information

import pandas as pd
import numpy as np
import math
import sys
import os
import random
from datetime import datetime

## Add stuff to the path to enable exec outside of DSS
plugin_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.join(plugin_root, 'python-lib'))
#sys.path.append(os.path.join(os.environ['DKUINSTALLDIR'], 'src/main/python/'))
sys.path.append(os.path.join(os.environ['OLDPWD'], 'src/main/python/'))

import dku_timeseries

JUST_BEFORE_SPRING_DST = pd.Timestamp('20190131 01:59:00').tz_localize('CET')
JUST_BEFORE_FALL_DST = pd.Timestamp('20191027 02:59:00').tz_localize('CET', ambiguous=True) #It's ambiguous because there are 2 instants with these dates! We select the first

TIME_COL = 'time_col'
DATA_COL = 'data_col'
THRESHOLD_DICT = {DATA_COL: (0,100)}


### Helpers to create test data, should be fixtures at some point I guess
def _make_df_with_one_col(column_data, period=pd.DateOffset(seconds=1), start_time=JUST_BEFORE_SPRING_DST):
    from datetime import datetime
    top = datetime.now()
    time = pd.date_range(start_time, None, len(column_data), period)
    top = datetime.now()
    df = pd.DataFrame({TIME_COL: time, DATA_COL: column_data})
    return df


def _make_window_roller_params(window_width=1):
    params = dku_timeseries.WindowRollerParams(window_width=window_width)
    return params

def _make_window_roller(window_width=1):
    params = _make_window_roller_params(window_width)
    return dku_timeseries.WindowRoller(params)

### Test cases
"""
def test_empty_df():
    df = _make_df_with_one_col([])
    segment_extractor = _make_window_roller()
    output_df = segment_extractor.compute(df, TIME_COL)
    assert output_df.shape == (0, 2)
"""

"""
def test_single_row_df():
    df = _make_df_with_one_col([33])
    segment_extractor = _make_window_roller()
    output_df = segment_extractor.compute(df, TIME_COL)
    assert output_df.shape == (1, 2)
    assert output_df[DATA_COL][0] == df[DATA_COL][0]
"""
def test_incremental_df_left_closed():
    length = 100
    data = [x for x in range(length)]
    df = _make_df_with_one_col(data)
    print(df.shape)
    params = dku_timeseries.WindowRollerParams(window_width=3, closed_option='left')
    window_roller = dku_timeseries.WindowRoller(params)
    output_df = window_roller.compute(df, TIME_COL)
    ground_truth = [np.NaN, 0, 0, 0, 1, 2, 3, 4, 5 ,6, 7]
    assert math.isnan(output_df[DATA_COL+'_min'][0])
    for x,y in zip(output_df[DATA_COL+'_min'][1:], ground_truth[1:]):
        assert output_df[DATA_COL][x] == y
    return output_df

def test_incremental_df_right_closed():
    length = 100
    data = [x for x in range(length)]
    df = _make_df_with_one_col(data)
    print(df.shape)
    params = dku_timeseries.WindowRollerParams(window_width=3, closed_option='right')
    window_roller = dku_timeseries.WindowRoller(params)
    output_df = window_roller.compute(df, TIME_COL)
    ground_truth = [0, 0, 0, 1, 2, 3, 4, 5 ,6, 7, 8]
    for x,y in zip(output_df[DATA_COL+'_min'][1:], ground_truth[1:]):
        assert output_df[DATA_COL][x] == y
    return output_df

