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
# sys.path.append(os.path.join(os.environ['DKUINSTALLDIR'], 'src/main/python/'))
sys.path.append(os.path.join(os.environ['OLDPWD'], 'src/main/python/'))

import dku_timeseries

JUST_BEFORE_SPRING_DST = pd.Timestamp('20190131 01:59:00').tz_localize('CET')
JUST_BEFORE_FALL_DST = pd.Timestamp('20191027 02:59:00').tz_localize('CET', ambiguous=True)  # It's ambiguous because there are 2 instants with these dates! We select the first

TIME_COL = 'time_col'
DATA_COL = 'data_col'
GROUP_COL = 'group_col'


### Helpers to create test data, should be fixtures at some point I guess
def _make_df_with_one_col(column_data, period=pd.DateOffset(seconds=1), start_time=JUST_BEFORE_SPRING_DST):
    from datetime import datetime
    top = datetime.now()
    time = pd.date_range(start_time, None, len(column_data), period)
    top = datetime.now()
    df = pd.DataFrame({TIME_COL: time, DATA_COL: column_data})
    return df


def _make_window_aggregator_params(window_width=1):
    params = dku_timeseries.WindowAggregatorParams(window_width=window_width)
    return params


def _make_window_aggregator(window_width=1):
    params = _make_window_aggregator_params(window_width)
    return dku_timeseries.WindowAggregator(params)


### Test cases

def test_empty_df():
    df = _make_df_with_one_col([])
    window_aggregator = _make_window_aggregator()
    output_df = window_aggregator.compute(df, TIME_COL)
    assert output_df.shape == (0, 2)


def test_single_row_df():
    df = _make_df_with_one_col([33])
    window_aggregator = _make_window_aggregator()
    output_df = window_aggregator.compute(df, TIME_COL)
    assert output_df.shape == (1, 2)
    assert output_df[DATA_COL][0] == df[DATA_COL][0]


def test_two_rows_df():
    length = 2
    data = [x for x in range(length)]
    df = _make_df_with_one_col(data)
    window_aggregator = _make_window_aggregator()
    output_df = window_aggregator.compute(df, TIME_COL)
    assert output_df[DATA_COL + '_min'][1] == 0


def test_incremental_df_left_closed():
    length = 100
    data = [x for x in range(length)]
    df = _make_df_with_one_col(data)
    print(df.shape)
    params = dku_timeseries.WindowAggregatorParams(window_width=3, closed_option='left')
    window_aggregator = dku_timeseries.WindowAggregator(params)
    output_df = window_aggregator.compute(df, TIME_COL)
    ground_truth = [np.NaN, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7]
    assert math.isnan(output_df[DATA_COL + '_min'][0])
    for x, y in zip(output_df[DATA_COL + '_min'][1:], ground_truth[1:]):
        assert output_df[DATA_COL][x] == y


def test_incremental_df_right_closed():
    length = 100
    data = [x for x in range(length)]
    df = _make_df_with_one_col(data)
    print(df.shape)
    params = dku_timeseries.WindowAggregatorParams(window_width=3, closed_option='right', window_type='gaussian')
    window_aggregator = dku_timeseries.WindowAggregator(params)
    output_df = window_aggregator.compute(df, TIME_COL)
    ground_truth = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    for x, y in zip(output_df[DATA_COL + '_min'][1:], ground_truth[1:]):
        assert output_df[DATA_COL][x] == y


def test_group_window_time_unit():
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
        temp_df.loc[:, GROUP_COL] = group_name
        df_list.append(temp_df)

    df = pd.concat(df_list, axis=0)

    params = dku_timeseries.WindowAggregatorParams(window_width=3, closed_option='left', window_type='boxcar')
    window_aggregator = dku_timeseries.WindowAggregator(params)
    output_df = window_aggregator.compute(df, datetime_column=TIME_COL, groupby_columns=[GROUP_COL])

    ground_truth = [np.NaN, 0, 0, 0, 1, 2, 3, 4, 5, 6]
    output_0 = output_df.groupby(GROUP_COL).get_group('group_0').data_col_min.values[:10]
    assert math.isnan(output_0[0])
    assert np.array_equal(output_0[1:], ground_truth[1:])
    output_1 = output_df.groupby(GROUP_COL).get_group('group_1').data_col_min.values[:10]
    assert math.isnan(output_1[0])
    assert np.array_equal(output_1[1:], ground_truth[1:])


""" 
def test_group_window_row_unit():
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

    params = dku_timeseries.WindowAggregatorParams(window_width=3, window_unit='rows', window_type='boxcar')
    window_aggregator = dku_timeseries.WindowAggregator(params)
    output_df = window_aggregator.compute(df, datetime_column=TIME_COL, groupby_columns=[GROUP_COL])

    ground_truth = [np.nan, np.nan, 0, 1, 2, 3, 4, 5, 6, 7]
    output_0 = output_df.groupby(GROUP_COL).get_group('group_0').data_col_min.values[:10]
    assert math.isnan(output_0[0])
    assert math.isnan(output_0[1])
    assert np.array_equal(output_0[2:], ground_truth[2:])
    output_1 = output_df.groupby(GROUP_COL).get_group('group_1').data_col_min.values[:10]
    assert math.isnan(output_1[0])
    assert math.isnan(output_1[1])
    assert np.array_equal(output_1[2:], ground_truth[2:])
""" 