from safe_logger import SafeLogger

logger = SafeLogger("Timeseries preparation plugin")


class DecompositionInputValidator(object):
    """Checks if the input dataframe is consistent with a decomposition

    Attributes:
        dku_config(DecompositionConfig): mapping structure storing the recipe parameters
        minimum_observations(int): Minimum number of observations required by the decomposition method
    """

    def __init__(self, dku_config):
        self.dku_config = dku_config
        self.minimum_observations = 3

    def check(self, df):
        """Checks if the input dataframe is compatible with the decomposition method

        :param df: Input dataframe
        :type df: pd.DataFrame
        """
        logger.info(f"Checking input values:...")
        self._check_model_compatibility(df)
        self._check_size(df)
        logger.info(f"Checking input values: the recipe parameters are consistent with the input dataset ")

    def _check_model_compatibility(self, df):
        """Checks if the input dataframe is compatible with the decomposition model

        :param df: Input dataframe
        :type df: pd.DataFrame
        """
        columns_with_invalid_values = self._get_columns_with_invalid_values(df)
        if len(columns_with_invalid_values) > 0:
            raise ValueError(
                f" The column(s) {', '.join(columns_with_invalid_values)} contain(s) negative values. Yet, a multiplicative model only works with positive "
                f"time series. You may choose an additive model instead. ")

    def _check_size(self, df):
        """Checks if the input dataframe contains enough observations

        :param df: Input dataframe
        :type df: pd.DataFrame
        """
        if self.dku_config.long_format:
            identifiers_sizes = df.groupby(self.dku_config.timeseries_identifiers).size()
            too_short_timeseries_identifiers = identifiers_sizes[identifiers_sizes < self.minimum_observations].index.tolist()
            if len(too_short_timeseries_identifiers) > 0:
                if len(too_short_timeseries_identifiers) == 1:
                    invalid_identifiers = identifiers_sizes.index.name
                else:
                    invalid_identifiers = identifiers_sizes.index.names
                raise ValueError(
                    f"The time series with the identifiers {invalid_identifiers} need at least {self.minimum_observations} observations. The "
                    f"current sizes of the long format time series are {identifiers_sizes.values}")
        else:
            size = len(df.index)
            if size < self.minimum_observations:
                raise ValueError(
                    f"This model must have at least {self.minimum_observations} observations. The input time series contains only {size} observations")

    def _get_columns_with_invalid_values(self, df):
        """Gets target columns with inconsistent values

        :param df: Input dataframe
        :type df: pd.DataFrame
        :return : columns with inconsistent values
        :rtype columns_with_invalid_values: list
        """
        columns_with_invalid_values = []
        if self.dku_config.model == "multiplicative":
            for target_column in self.dku_config.target_columns:
                if (df[target_column] <= 0).any():
                    columns_with_invalid_values.append(target_column)
        return columns_with_invalid_values
