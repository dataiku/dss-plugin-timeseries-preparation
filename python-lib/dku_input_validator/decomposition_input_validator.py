import numpy as np


class DecompositionInputValidator(object):
    def __init__(self, dku_config):
        self.dku_config = dku_config
        self.minimum_observations = 3

    def check(self, df):
        self._check_model(df)
        self._check_size(df)

    def _check_model(self, df):
        columns_with_invalid_values = self._get_columns_with_invalid_values(df)
        if len(columns_with_invalid_values) > 0:
            raise ValueError(
                f" The column(s) {', '.join(columns_with_invalid_values)} contain(s) negative values. Yet, a multiplicative model only works with positive time series. You may choose an additive model instead. ")

    def _check_size(self, df):
        if self.dku_config.long_format:
            for identifiers, identifiers_df in df.groupby(self.dku_config.timeseries_identifiers):
                size = identifiers_df.shape[0]
                if size < self.minimum_observations:
                    raise ValueError(
                        f"The time series with the identifiers {identifiers} needs at least {self.minimum_observations} observations. It has only {size} observations")
        else:
            size = df.shape[0]
            if size < self.minimum_observations:
                raise ValueError(
                    f"This model must have at least {self.minimum_observations} observations. The input time series contains only {size} observations")


    def _get_columns_with_invalid_values(self, input_df):
        columns_with_invalid_values = []
        if self.dku_config.model == "multiplicative":
            for target_column in self.dku_config.target_columns:
                target_values = input_df[target_column].values
                if np.any(target_values <= 0):
                    columns_with_invalid_values.append(target_column)
        return columns_with_invalid_values
