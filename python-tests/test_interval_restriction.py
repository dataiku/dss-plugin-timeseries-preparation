# This is a test file intended to be used with pytest
# pytest automatically runs all the function starting with "test_"
# see https://docs.pytest.org for more information

import pandas as pd
import numpy as np
import sys
import os
import random
from datetime import datetime

## Add stuff to the path to enable exec outside of DSS
plugin_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.join(plugin_root, 'python-lib'))

import dku_timeseries

JUST_BEFORE_SPRING_DST = pd.Timestamp('20190131 01:59:00').tz_localize('CET')
JUST_BEFORE_FALL_DST = pd.Timestamp('20191027 02:59:00').tz_localize('CET', ambiguous=True)  # It's ambiguous because there are 2 instants with these dates! We select the first

TIME_COL = 'time_col'
DATA_COL = 'data_col'
GROUP_COL = 'group_col'
MIN_THRESHOLD = 0
MAX_THRESHOLD = 10
THRESHOLD_DICT = {DATA_COL: (MIN_THRESHOLD, MAX_THRESHOLD)}


### Helpers to create test data, should be fixtures at some point I guess
def _make_df_with_one_col(column_data, period=pd.DateOffset(seconds=1), start_time=JUST_BEFORE_SPRING_DST):
    from datetime import datetime
    time = pd.date_range(start_time, None, len(column_data), period)
    df = pd.DataFrame({TIME_COL: time, DATA_COL: column_data})
    return df


def _make_interval_restrictor_params(max_deviation_duration_value=0):
    params = dku_timeseries.IntervalRestrictorParams(max_deviation_duration_value=max_deviation_duration_value)
    return params


def _make_interval_restrictor(max_deviation_duration_value=0):
    params = _make_interval_restrictor_params(max_deviation_duration_value)
    return dku_timeseries.IntervalRestrictor(params)


### Test cases

def test_empty():
    df = _make_df_with_one_col([])
    interval_restrictor = _make_interval_restrictor()
    output_df = interval_restrictor.compute(df, TIME_COL, THRESHOLD_DICT)
    assert output_df.shape == (0, 2)

def test_nan_data():

    length = 1000
    data = [np.nan for _ in range(length)]
    df = _make_df_with_one_col(data)
    interval_restrictor = _make_interval_restrictor()
    output_df = interval_restrictor.compute(df, TIME_COL, THRESHOLD_DICT)
    assert output_df.shape == (0, 2)

def test_single_row_out_range():
    df = _make_df_with_one_col([50])
    interval_restrictor = _make_interval_restrictor()
    output_df = interval_restrictor.compute(df, TIME_COL, THRESHOLD_DICT)
    assert output_df.shape == (0, 2)


def test_single_row_in_range():
    df = _make_df_with_one_col([8])
    interval_restrictor = _make_interval_restrictor()
    output_df = interval_restrictor.compute(df, TIME_COL, THRESHOLD_DICT)
    assert output_df.shape == (0, 2)


def test_incremental_time_unit():
    length = 1000
    data = [x for x in range(length)]
    df = _make_df_with_one_col(data)
    interval_restrictor = _make_interval_restrictor()
    output_df = interval_restrictor.compute(df, TIME_COL, THRESHOLD_DICT)
    for x, y in enumerate(range(MIN_THRESHOLD, MAX_THRESHOLD)):
        assert output_df[DATA_COL][x] == y



def test_incremental_time_unit_with_noise():
    length = 1000
    data = [x for x in range(length)]
    df = _make_df_with_one_col(data)
    df.loc[5:6, DATA_COL] = [50, 51]
    interval_restrictor = _make_interval_restrictor(max_deviation_duration_value=3)
    output_df = interval_restrictor.compute(df, TIME_COL, THRESHOLD_DICT)
    for x, y in zip(range(len(output_df)), pd.date_range(JUST_BEFORE_SPRING_DST, None, 10, pd.DateOffset(seconds=1))):
        assert output_df[TIME_COL][x] == y


def test_group_incremental_time_unit_no_deviation():
    start_time_1 = pd.Timestamp('20190131 01:59:00').tz_localize('CET')
    start_time_2 = pd.Timestamp('20190131 02:00:00').tz_localize('CET')
    start_time_list = [start_time_1, start_time_2]

    len1 = 100
    len2 = 10
    data1 = range(len1)
    data2 = range(len2)
    data_list = [data1, data2]

    period1 = pd.DateOffset(seconds=1)
    period2 = pd.DateOffset(seconds=1)
    period_list = [period1, period2]

    df_list = []
    for group_id, data, period, start_time in zip(range(len(data_list)), data_list, period_list, start_time_list):
        group_name = 'group_{}'.format(group_id)
        temp_df = _make_df_with_one_col(data, period=period, start_time=start_time)
        temp_df[GROUP_COL] = group_name
        df_list.append(temp_df)

    df = pd.concat(df_list, axis=0)

    params = dku_timeseries.IntervalRestrictorParams(time_unit='seconds', max_deviation_duration_value=1)
    interval_restrictor = dku_timeseries.IntervalRestrictor(params)
    output_df = interval_restrictor.compute(df, TIME_COL, THRESHOLD_DICT, groupby_columns=[GROUP_COL])

    assert np.array_equal(output_df.groupby(GROUP_COL).get_group('group_0')[DATA_COL].values, range(11))
    assert np.array_equal(output_df.groupby(GROUP_COL).get_group('group_1')[DATA_COL].values, range(10))
