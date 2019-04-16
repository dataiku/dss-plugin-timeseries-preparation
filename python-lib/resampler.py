# -*- coding: utf-8 -*-
# %%
import dataiku
import pandas as pd
import logging


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='timeseries-preparation plugin %(levelname)s - %(message)s')


class Resampler:

    def __init__(self, params=None):

        assert params is not None, "ResamplerParams instance is None."
        self.params = params
        self.params.check()

    def _compute_full_time_index(self, df):
        full_time_index = pd.date_range(start=df.index.min() + pd.Timedelta(self.params.offset_step),
                                        end=df.index.max() - pd.Timedelta(self.params.crop_step),
                                        freq=self.params.resampling_step)
        return full_time_index

    def _find_datetime_column(self, df):
        datetime_cols = df.select_dtypes(include='datetime').columns.tolist()
        if len(datetime_cols) == 0:
            raise ValueError('No datetime column detected.')
        elif len(datetime_cols) == 1:
            return datetime_cols[0]
        else:
            raise ValueError('Multiple datetime columns detected.')

    def _check_df(self, df):
        # TODO should we check the input df first for the prerequisite ?
        return None

    def _resample(self, df, group_id=None): #TODO we dont actually use group_id

        try:
            temp_df = df.reindex(df.index | self.full_time_index)
        except Exception, e:
            raise ValueError('{}: Your timeseries might have dupplicate timestamps.'.format(str(e)))


        if self.params.interpolation_method == 'next':  # no `next` method with pd.interpolate()
            df_interpolated = temp_df.bfill()
        elif self.params.interpolation_method == 'previous':  # no `previous` method with pd.interpolate()
            df_interpolated = temp_df.ffill()
        else:
            df_interpolated = temp_df.interpolate(
                self.params.interpolation_method)

        if self.params.extrapolation_method == "clip":
            df_extrapolated = df_interpolated.ffill().bfill()
        elif self.params.interpolation_method == 'next':
            df_extrapolated = df_interpolated.bfill()
        elif self.params.interpolation_method == 'previous':
            df_extrapolated = df_interpolated.ffill()
        else:
            df_extrapolated = df_interpolated

        if self.params.groupby_cols:
            df_extrapolated[self.params.groupby_cols] = df_extrapolated[
                self.params.groupby_cols].ffill().bfill()

        df_resampled = df_extrapolated.reindex(self.full_time_index)
        return df_resampled

    def transform(self, raw_df):

        datetime_column = self._find_datetime_column(raw_df)
        # TODO sort_index() is necessary ?
        df = raw_df.set_index(datetime_column).sort_index()
        self.full_time_index = self._compute_full_time_index(df)

        if self.params.groupby_cols:
            grouped = df.groupby(self.params.groupby_cols)
            resampled_groups = []

            for group_id, group in grouped:
                df_extrapolated = self._resample(group, group_id)
                resampled_groups.append(df_extrapolated)

            df_transformed = pd.concat(resampled_groups)

        else:
            df_transformed = self._resample(df)

        final_df = df_transformed.rename_axis(datetime_column).reset_index()
        return final_df
