import pandas as pd
import pytest

from dku_timeseries import ExtremaExtractor
from recipe_config_loading import get_extrema_extraction_params


@pytest.fixture
def columns():
    class COLUMNS:
        date = "Date"
        category = "categorical"
        data = "value1"

    return COLUMNS


@pytest.fixture
def monthly_df():
    co2 = [4, 9, 4, 2, 5, 1]
    time_index = pd.date_range("1-1-2015", periods=6, freq="M")
    df = pd.DataFrame.from_dict(
        {"value1": co2, "value2": co2, "Date": time_index})
    return df


@pytest.fixture
def recipe_config(columns):
    config = {u'window_type': u'none', u'groupby_columns': [], u'closed_option': u'left', u'window_unit': u'months', u'window_width': 2,
              u'causal_window': False, u'datetime_column': columns.date, u'advanced_activated': False, u'extrema_column': columns.data, u'extrema_type': u'max',
              u'aggregation_types': [u'average'], u'gaussian_std': 1}
    return config


class TestExtremaFrequencies:
    def test_month(self, recipe_config, columns):
        params = get_extrema_extraction_params(recipe_config)
        extrema_extractor = ExtremaExtractor(params)
        df = get_df_DST("M", columns)
        output_df = extrema_extractor.compute(df, columns.date, columns.data)
        assert output_df.shape == (1, 5)
        assert output_df.loc[0, columns.date] == pd.Timestamp("2019-06-30 01:59:00+02:00")

    def test_year(self, recipe_config, columns):
        recipe_config["window_unit"] = "years"
        params = get_extrema_extraction_params(recipe_config)
        extrema_extractor = ExtremaExtractor(params)
        df = get_df_DST("Y", columns)
        output_df = extrema_extractor.compute(df, columns.date, columns.data)
        assert output_df.shape == (1, 5)
        assert output_df.loc[0, columns.date] == pd.Timestamp("2024-12-31 01:59:00+01:00")

    def test_weeks(self, recipe_config, columns):
        recipe_config["window_unit"] = "weeks"
        params = get_extrema_extraction_params(recipe_config)
        extrema_extractor = ExtremaExtractor(params)
        df = get_df_DST("W", columns)
        output_df = extrema_extractor.compute(df, columns.date, columns.data)
        assert output_df.shape == (1, 5)
        assert output_df.loc[0, columns.date] == pd.Timestamp("2019-03-10 01:59:00+01:00")

    def test_days(self, recipe_config, columns):
        recipe_config["window_unit"] = "days"
        params = get_extrema_extraction_params(recipe_config)
        extrema_extractor = ExtremaExtractor(params)
        df = get_df_DST("D", columns)
        output_df = extrema_extractor.compute(df, columns.date, columns.data)
        assert output_df.shape == (1, 5)
        assert output_df.loc[0, columns.date] == pd.Timestamp("2019-02-05 01:59:00+01:00")

    def test_hours(self, recipe_config, columns):
        recipe_config["window_unit"] = "hours"
        params = get_extrema_extraction_params(recipe_config)
        extrema_extractor = ExtremaExtractor(params)
        df = get_df_DST("H", columns)
        output_df = extrema_extractor.compute(df, columns.date, columns.data)
        assert output_df.shape == (1, 5)
        assert output_df.loc[0, columns.date] == pd.Timestamp("2019-01-31 06:59:00+0100")

    def test_minutes(self, recipe_config, columns):
        recipe_config["window_unit"] = "minutes"
        params = get_extrema_extraction_params(recipe_config)
        extrema_extractor = ExtremaExtractor(params)
        df = get_df_DST("T", columns)
        output_df = extrema_extractor.compute(df, columns.date, columns.data)
        assert output_df.shape == (1, 5)
        assert output_df.loc[0, columns.date] == pd.Timestamp("2019-01-31 02:04:00+0100")

    def test_seconds(self, recipe_config, columns):
        recipe_config["window_unit"] = "seconds"
        params = get_extrema_extraction_params(recipe_config)
        extrema_extractor = ExtremaExtractor(params)
        df = get_df_DST("S", columns)
        output_df = extrema_extractor.compute(df, columns.date, columns.data)
        assert output_df.shape == (1, 5)
        assert output_df.loc[0, columns.date] == pd.Timestamp("2019-01-31 01:59:05+0100")

    def test_milliseconds(self, recipe_config, columns):
        recipe_config["window_unit"] = "milliseconds"
        params = get_extrema_extraction_params(recipe_config)
        extrema_extractor = ExtremaExtractor(params)
        df = get_df_DST("L", columns)
        output_df = extrema_extractor.compute(df, columns.date, columns.data)
        assert output_df.shape == (1, 5)
        assert output_df.loc[0, columns.date] == pd.Timestamp("2019-01-31 01:59:00.005000+0100")

    def test_microseconds(self, recipe_config, columns):
        recipe_config["window_unit"] = "microseconds"
        params = get_extrema_extraction_params(recipe_config)
        extrema_extractor = ExtremaExtractor(params)
        df = get_df_DST("U", columns)
        output_df = extrema_extractor.compute(df, columns.date, columns.data)
        assert output_df.shape == (1, 5)
        assert output_df.loc[0, columns.date] == pd.Timestamp("2019-01-31 01:59:00.000005+0100")

    def test_nanoseconds(self, recipe_config, columns):
        recipe_config["window_unit"] = "nanoseconds"
        params = get_extrema_extraction_params(recipe_config)
        extrema_extractor = ExtremaExtractor(params)
        df = get_df_DST("N", columns)
        output_df = extrema_extractor.compute(df, columns.date, columns.data)
        assert output_df.shape == (1, 5)
        assert output_df.loc[0, columns.date] == pd.Timestamp("2019-01-31 01:59:00.000000005+01:00")


def get_df_DST(frequency, columns):
    JUST_BEFORE_SPRING_DST = pd.Timestamp('20190131 01:59:0000000').tz_localize('CET')
    co2 = [315.58, 316.39, 316.79, 316.2, 666, 888]
    co = [315.58, 77, 316.79, 66, 666, 888]
    categorical = ["first", "first", "second", "second", "second", "second"]
    time_index = pd.date_range(JUST_BEFORE_SPRING_DST, periods=6, freq=frequency)
    df = pd.DataFrame.from_dict({columns.data: co2, "value2": co, columns.category: categorical, columns.date: time_index})
    return df
