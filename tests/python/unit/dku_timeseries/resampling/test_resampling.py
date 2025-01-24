import math
import random

import numpy as np
import pandas as pd
import pytest

from dku_timeseries.resampling import ResamplerParams, Resampler

JUST_BEFORE_SPRING_DST = pd.Timestamp('20190131 01:59:00').tz_localize('CET')
JUST_BEFORE_FALL_DST = pd.Timestamp('20191027 02:59:00').tz_localize('CET',
                                                                     ambiguous=True)  # It's ambiguous because there are 2 instants with these dates! We
# select the first

TIME_COL = 'time_col'
DATA_COL = 'data_col'
GROUP_COL = 'group_col'


@pytest.fixture
def academy_orders_df():
    dates = pd.DatetimeIndex(['2013-03-29T00:00:00.000000000', '2013-04-10T00:00:00.000000000',
                              '2013-04-19T00:00:00.000000000', '2013-04-23T00:00:00.000000000',
                              '2013-04-25T00:00:00.000000000', '2013-04-30T00:00:00.000000000',
                              '2013-05-03T00:00:00.000000000', '2013-05-04T00:00:00.000000000',
                              '2013-05-07T00:00:00.000000000', '2013-05-08T00:00:00.000000000',
                              '2013-05-09T00:00:00.000000000', '2013-05-11T00:00:00.000000000',
                              '2013-05-12T00:00:00.000000000', '2013-05-13T00:00:00.000000000',
                              '2013-05-14T00:00:00.000000000', '2013-05-15T00:00:00.000000000',
                              '2013-05-15T00:00:00.000000000', '2013-05-17T00:00:00.000000000',
                              '2013-05-17T00:00:00.000000000', '2013-05-18T00:00:00.000000000',
                              '2013-05-19T00:00:00.000000000', '2013-05-20T00:00:00.000000000',
                              '2013-05-20T00:00:00.000000000', '2013-05-21T00:00:00.000000000',
                              '2013-05-22T00:00:00.000000000', '2013-05-22T00:00:00.000000000',
                              '2013-05-24T00:00:00.000000000', '2013-05-28T00:00:00.000000000',
                              '2013-05-28T00:00:00.000000000', '2013-05-28T00:00:00.000000000',
                              '2013-05-29T00:00:00.000000000'])
    tshirt_quantity = [1, 4, 3, 4, 1, 3, 2, 5, 2, 2, 1, 1, 2, 4, 2, 1, 1, 1, 4, 7, 2, 1,
                       1, 1, 1, 2, 3, 2, 5, 4, 3]
    tshirt_category = ['Black T-Shirt M', 'White T-Shirt M', 'Hoodie', 'White T-Shirt F',
                       'White T-Shirt M', 'Black T-Shirt M', 'White T-Shirt M',
                       'Black T-Shirt M', 'Hoodie', 'White T-Shirt F', 'White T-Shirt F',
                       'Black T-Shirt M', 'Hoodie', 'Hoodie', 'Tennis Shirt', 'Hoodie',
                       'White T-Shirt M', 'Hoodie', 'White T-Shirt M', 'Black T-Shirt M',
                       'White T-Shirt M', 'Black T-Shirt M', 'White T-Shirt F',
                       'Black T-Shirt F', 'Black T-Shirt F', 'White T-Shirt F', 'Hoodie',
                       'Black T-Shirt M', 'Hoodie', 'Black T-Shirt F', 'Hoodie']
    df_orders = pd.DataFrame({TIME_COL: dates, DATA_COL: tshirt_quantity, GROUP_COL: tshirt_category})
    return df_orders


@pytest.fixture
def long_format_df():
    id = [1, 1, 1, 1, 1, 2, 2]
    dates = pd.DatetimeIndex(['2021-06-19T12:45:30.000Z', '2021-06-21T13:45:30.000Z',
                              '2021-06-22T12:00:00.000Z', '2021-06-23T12:45:30.000Z',
                              '2021-06-25T12:45:30.000Z', '2021-06-21T12:45:30.000Z',
                              '2021-06-23T12:45:30.000Z'])
    value = [12, 13, 12, 14,  4,  5,  4]
    df = pd.DataFrame({GROUP_COL:id, TIME_COL:dates, DATA_COL:value})
    return df


### Helpers to create test data, should be fixtures at some point I guess
def _make_df_with_one_col(column_data, period=pd.DateOffset(seconds=1), start_time=JUST_BEFORE_SPRING_DST):
    time = pd.date_range(start_time, None, len(column_data), period)
    return pd.DataFrame({TIME_COL: time, DATA_COL: column_data})


def _make_df_with_one_col_one_group_per_col(column_data, period=pd.DateOffset(seconds=1), start_time=JUST_BEFORE_SPRING_DST):
    time = pd.date_range(start_time, None, len(column_data), period)
    return pd.DataFrame({TIME_COL: time, DATA_COL: column_data, GROUP_COL: range(len(column_data))})


def _make_df_with_one_col_group(column_data, num_group, period=pd.DateOffset(seconds=1), start_time=JUST_BEFORE_SPRING_DST):
    df_list = []
    for x in range(num_group):
        group_name = 'group_{}'.format(x)
        temp_df = _make_df_with_one_col(column_data, period=period)
        temp_df[GROUP_COL] = group_name
        df_list.append(temp_df)
    df = pd.concat(df_list, axis=0)
    return df


def _make_resampling_params():
    params = ResamplerParams()
    params.datetime_column = TIME_COL
    return params


def _make_resampler():
    params = _make_resampling_params()
    return Resampler(params)


### Test cases
class TestResampler:
    def test_empty_df(self):
        df = _make_df_with_one_col([])
        resampler = _make_resampler()
        output_df = resampler.transform(df, TIME_COL)
        assert output_df.shape == (0, 2)

    def test_single_row_df(self):
        df = _make_df_with_one_col([33])
        resampler = _make_resampler()
        output_df = resampler.transform(df, TIME_COL)
        assert output_df.shape == (1, 2)
        assert output_df[DATA_COL][0] == df[DATA_COL][0]

    def test_identity_resampling(self):
        """
        Default sampling rate is 1Hz
        Since we create test data at 1Hz, default resampling should be identity
        """
        length = 1000
        data = [random.random() for _ in range(length)]
        df = _make_df_with_one_col(data)
        resampler = _make_resampler()
        output_df = resampler.transform(df, TIME_COL)
        assert output_df.shape == (length, 2)
        for x in range(1000):
            assert output_df[DATA_COL][x] == df[DATA_COL][x]

    def test_nan_data(self):

        length = 1000
        data = [np.nan for _ in range(length)]
        df = _make_df_with_one_col(data)
        resampler = _make_resampler()
        output_df = resampler.transform(df, TIME_COL)
        assert output_df.shape == (length, 2)
        assert np.sum(output_df[DATA_COL].isnull()) == length

    def test_identity_resampling_month(self):

        length = 100
        data = [random.random() for _ in range(length)]
        df = _make_df_with_one_col(data, period=pd.DateOffset(months=1))
        params = ResamplerParams(time_unit='months')
        resampler = Resampler(params)
        output_df = resampler.transform(df, TIME_COL)
        assert output_df.shape == (length, 2)

    def test_half_freq_resampling(self):
        length = 1000
        half_length = length / 2
        data = [x for x in range(length)]
        df = _make_df_with_one_col(data, pd.DateOffset(seconds=0.5))
        resampler = _make_resampler()
        output_df = resampler.transform(df, TIME_COL)
        assert output_df.shape[0] - 1 == half_length
        for x in range(100):
            assert output_df[DATA_COL][x] == 2 * x

    def test_one_group_per_line(self):
        length = 100
        data = [random.random() for _ in range(length)]
        df = _make_df_with_one_col_one_group_per_col(data)
        resampler = _make_resampler()
        output_df = resampler.transform(df, TIME_COL, groupby_columns=[GROUP_COL])
        assert output_df.shape == (length, 3)

    def test_identity_resampling_group(self):
        """
        2 groups, same frequency (1Hz), same date range
        """
        length = 1000
        num_group = 3
        data = [random.random() for _ in range(length)]
        df = _make_df_with_one_col_group(data, num_group=num_group)
        resampler = _make_resampler()
        df2 = resampler.transform(df, TIME_COL, groupby_columns=[GROUP_COL])
        assert len(df2) == length * num_group

        for x in range(num_group):
            df_ref = df.loc[df[GROUP_COL] == 'group_{}'.format(x), [TIME_COL, DATA_COL]].reset_index()
            df_ref = df_ref.sort_values(TIME_COL)

            df_check = df2.loc[df2[GROUP_COL] == 'group_{}'.format(x), [TIME_COL, DATA_COL]].reset_index()
            df_check = df_check.sort_values(TIME_COL)

            for y in range(1000):
                assert df_check[DATA_COL][y] == df_ref[DATA_COL][y]

    def test_half_freq_resampling_group(self):
        """
        2 groups, same frequency (0.5Hz), same date range
        """
        length = 1000
        num_group = 3
        data = [x for x in np.arange(0, length, 0.5)]
        df = _make_df_with_one_col_group(data, num_group=num_group, period=pd.DateOffset(seconds=0.5))
        resampler = _make_resampler()
        df2 = resampler.transform(df, TIME_COL, groupby_columns=[GROUP_COL])

        for x in range(num_group):
            df_ref = df.loc[df[GROUP_COL] == 'group_{}'.format(x), [TIME_COL, DATA_COL]].reset_index()
            df_ref = df_ref.sort_values(TIME_COL)

            df_check = df2.loc[df2[GROUP_COL] == 'group_{}'.format(x), [TIME_COL, DATA_COL]].reset_index()
            df_check = df_check.sort_values(TIME_COL)

            for y in range(1000):
                assert df_check[DATA_COL][y] == 2 * df_ref[DATA_COL][y]

    def test_group_inclusion_same_freq(self):
        """
        2 groups, same frequency, different date range: group 1 includes group 2
        """
        # 2 group, same frequency, group 1 include group 2
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

        params = ResamplerParams(extrapolation_method='interpolation')
        resampler = Resampler(params)
        output_df = resampler.transform(df, TIME_COL, groupby_columns=[GROUP_COL])

        assert np.array_equal(output_df.groupby(GROUP_COL).size().values, [100, 100])

        ref_data_1 = range(0, len1)
        resample_data_1 = output_df.groupby(GROUP_COL).get_group('group_0').data_col.values
        assert np.array_equal(resample_data_1, ref_data_1)

        start_value = -(start_time_2 - start_time_1).seconds
        ref_data_2 = range(start_value, start_value + len1)
        resample_data_2 = output_df.groupby(GROUP_COL).get_group('group_1').data_col.values
        assert np.array_equal(resample_data_2, ref_data_2)

    def test_group_inclusion_different_freq(self):
        """
        2 groups, different frequency, different date range: group 1 includes group 2
        """
        start_time_1 = pd.Timestamp('20190131 01:59:00').tz_localize('CET')
        start_time_2 = pd.Timestamp('20190131 02:00:00').tz_localize('CET')
        start_time_list = [start_time_1, start_time_2]

        len1 = 100
        len2 = 10
        data1 = range(len1)
        data2 = range(len2)
        data_list = [data1, data2]

        period1 = pd.DateOffset(seconds=2)
        period2 = pd.DateOffset(seconds=1)
        period_list = [period1, period2]

        df_list = []
        for group_id, data, period, start_time in zip(range(len(data_list)), data_list, period_list, start_time_list):
            group_name = 'group_{}'.format(group_id)
            temp_df = _make_df_with_one_col(data, period=period, start_time=start_time)
            temp_df[GROUP_COL] = group_name
            df_list.append(temp_df)

        df = pd.concat(df_list, axis=0)

        params = ResamplerParams(extrapolation_method='interpolation')
        resampler = Resampler(params)
        output_df = resampler.transform(df, TIME_COL, groupby_columns=[GROUP_COL])

        assert np.array_equal(output_df.groupby(GROUP_COL).size().values, [199, 199])

        ref_data_0 = np.linspace(0, 99, num=199)
        resample_data_0 = output_df.groupby(GROUP_COL).get_group('group_0').data_col.values
        assert np.array_equal(resample_data_0, ref_data_0)

        start_value = -(start_time_2 - start_time_1).seconds
        ref_data_1 = np.linspace(start_value, start_value + 198, num=199)
        resample_data_1 = output_df.groupby(GROUP_COL).get_group('group_1').data_col.values
        assert np.array_equal(resample_data_1, ref_data_1)

    def test_empty_extrapolation_month(self):
        length = 10
        data = [random.random() for _ in range(length)]
        start_time = pd.Timestamp('20210101 00:00:00').tz_localize('CET')
        df_start_of_month = _make_df_with_one_col(data, period=pd.DateOffset(months=1), start_time=start_time)
        params = ResamplerParams(time_unit="months", extrapolation_method='none')
        resampler = Resampler(params)
        output_df = resampler.transform(df_start_of_month, TIME_COL)
        assert not math.isnan(output_df["data_col"].values[0])
        assert math.isnan(output_df["data_col"].values[-1])
        assert len(output_df.index) == 10

        df_end_of_month = _make_df_with_one_col(data, period=pd.DateOffset(months=1))
        params = ResamplerParams(time_unit="months", extrapolation_method="none")
        resampler = Resampler(params)
        output_df = resampler.transform(df_end_of_month, TIME_COL)
        assert math.isnan(output_df["data_col"].values[0])
        assert math.isnan(output_df["data_col"].values[-1])
        assert len(output_df.index) == 10

    def test_empty_extrapolation_week(self):
        length = 10
        data = [random.random() for _ in range(length)]
        start_time = pd.Timestamp('20210301 00:00:00').tz_localize('CET')
        df_end_of_week = _make_df_with_one_col(data, period=pd.DateOffset(weeks=1), start_time=start_time)
        params = ResamplerParams(time_unit="weeks", extrapolation_method='none')
        resampler = Resampler(params)
        output_df = resampler.transform(df_end_of_week, TIME_COL)
        np.testing.assert_array_equal(output_df[TIME_COL].values, pd.DatetimeIndex(['2021-03-06T23:00:00.000000000', '2021-03-13T23:00:00.000000000',
                                                                                    '2021-03-20T23:00:00.000000000', '2021-03-27T23:00:00.000000000',
                                                                                    '2021-04-03T22:00:00.000000000', '2021-04-10T22:00:00.000000000',
                                                                                    '2021-04-17T22:00:00.000000000', '2021-04-24T22:00:00.000000000',
                                                                                    '2021-05-01T22:00:00.000000000', '2021-05-08T22:00:00.000000000']))

    def test_empty_extrapolation_day(self, academy_orders_df):
        params = ResamplerParams(time_unit="days", extrapolation_method='none')
        resampler = Resampler(params)
        output_df = resampler.transform(academy_orders_df, TIME_COL, groupby_columns=[GROUP_COL])
        resampled_dates = pd.date_range(pd.Timestamp("2013-03-29"), freq="D", periods=62)
        black_tshirts_F = output_df[output_df[GROUP_COL] == "Black T-Shirt F"]
        np.testing.assert_array_equal(black_tshirts_F[TIME_COL], resampled_dates)
        assert np.all(np.isnan(black_tshirts_F.loc[:52, DATA_COL]))
        assert not np.any(np.isnan(black_tshirts_F.loc[53:-2, DATA_COL]))
        assert math.isnan(black_tshirts_F[DATA_COL].values[-1])

    def test_empty_interpolation_day(self, academy_orders_df):
        params = ResamplerParams(time_unit="days", interpolation_method='none')
        resampler = Resampler(params)
        output_df = resampler.transform(academy_orders_df, TIME_COL, groupby_columns=[GROUP_COL])
        black_tshirts_F = output_df[output_df[GROUP_COL] == "Black T-Shirt F"]
        assert not np.any(np.isnan(black_tshirts_F.loc[:54, DATA_COL]))
        assert np.all(np.isnan(black_tshirts_F.loc[55:59, DATA_COL]))
        assert not np.any(np.isnan(black_tshirts_F.loc[60:, DATA_COL]))

    def test_no_extrapolation_month(self):
        length = 10
        data = [random.random() for _ in range(length)]
        start_time = pd.Timestamp('20210101 00:00:00').tz_localize('CET')
        df_start_of_month = _make_df_with_one_col(data, period=pd.DateOffset(months=1), start_time=start_time)
        params = ResamplerParams(time_unit="months", extrapolation_method='no_extrapolation')
        resampler = Resampler(params)
        output_df = resampler.transform(df_start_of_month, TIME_COL)
        assert not math.isnan(output_df["data_col"].values[0])
        assert not math.isnan(output_df["data_col"].values[-1])
        assert len(output_df.index) == 9

        df_end_of_month = _make_df_with_one_col(data, period=pd.DateOffset(months=1))
        params = ResamplerParams(time_unit="months", extrapolation_method="no_extrapolation")
        resampler = Resampler(params)
        output_df = resampler.transform(df_end_of_month, TIME_COL)
        assert len(output_df.index) == 8
        assert not math.isnan(output_df["data_col"].values[0])
        assert not math.isnan(output_df["data_col"].values[-1])

    def test_no_extrapolation_week(self):
        length = 10
        data = [random.random() for _ in range(length)]
        start_time = pd.Timestamp('20210301 00:00:00').tz_localize('CET')
        df = _make_df_with_one_col(data, period=pd.DateOffset(weeks=1), start_time=start_time)
        params = ResamplerParams(time_unit="weeks", extrapolation_method='no_extrapolation')
        resampler = Resampler(params)
        output_df = resampler.transform(df, TIME_COL)
        np.testing.assert_array_equal(output_df[TIME_COL].values, pd.DatetimeIndex(['2021-03-06T23:00:00.000000000', '2021-03-13T23:00:00.000000000',
                                                                                    '2021-03-20T23:00:00.000000000', '2021-03-27T23:00:00.000000000',
                                                                                    '2021-04-03T22:00:00.000000000', '2021-04-10T22:00:00.000000000',
                                                                                    '2021-04-17T22:00:00.000000000', '2021-04-24T22:00:00.000000000',
                                                                                    '2021-05-01T22:00:00.000000000']))

    def test_extrapolation_custom_dates(self):
        data = [random.random() for _ in range(3)]
        start_time = pd.Timestamp('2024-12-01T00:00:00Z')
        df = _make_df_with_one_col(data, period=pd.DateOffset(days=1), start_time=start_time)
        params = ResamplerParams(time_unit="days",
                                 extrapolation_method='clip',
                                 custom_start_date=pd.Timestamp("2024-11-30T00:00:00Z"),
                                 custom_end_date=pd.Timestamp("2024-12-05T00:00:00Z"))
        resampler = Resampler(params)
        output_df = resampler.transform(df, TIME_COL)
        np.testing.assert_array_equal(output_df[TIME_COL].values, pd.DatetimeIndex(['2024-11-30T00:00:00.000000000', '2024-12-01T00:00:00.000000000',
                                                                                    '2024-12-02T00:00:00.000000000', '2024-12-03T00:00:00.000000000',
                                                                                    '2024-12-04T00:00:00.000000000', '2024-12-05T00:00:00.000000000']))

    def test_no_extrapolation_long_format(self, long_format_df):
        params = ResamplerParams(time_unit="days", extrapolation_method='no_extrapolation')
        resampler = Resampler(params)
        output_df = resampler.transform(long_format_df, TIME_COL,groupby_columns=[GROUP_COL])
        assert len(output_df.index) == 8
        assert not np.any(np.isnan(output_df[DATA_COL].values))


    def test_empty_extrapolation_long_format(self, long_format_df):
        params = ResamplerParams(time_unit="days", extrapolation_method='none')
        resampler = Resampler(params)
        output_df = resampler.transform(long_format_df, TIME_COL,groupby_columns=[GROUP_COL])
        assert len(output_df.index) == 14
        assert np.all(np.isnan(output_df.loc[6:8, DATA_COL]))
        assert np.all(np.isnan(output_df.loc[11:13, DATA_COL]))
