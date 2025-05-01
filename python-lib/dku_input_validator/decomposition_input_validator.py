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
        logger.info("Checking input values:...")
        self._check_model_compatibility(df)
        self._check_size(df)
        logger.info("Checking input values: the recipe parameters are consistent with the input dataset ")

    def _check_model_compatibility(self, df):
        """Checks if the input dataframe is compatible with the decomposition model

        :param df: Input dataframe
        :type df: pd.DataFrame
        """
        columns_with_invalid_values = self._get_columns_with_invalid_values(df)
        if len(columns_with_invalid_values) > 0:
            raise ValueError(
                " The column(s) {} contain(s) negative values. Yet, a multiplicative model only works with positive "
                "time series. You may choose an additive model instead. ".format(', '.join(columns_with_invalid_values)))

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
                    "The time series with the identifiers {} need at least {} observations. The "
                    "current sizes of the long format time series are {}".format(invalid_identifiers, self.minimum_observations, identifiers_sizes.values))
        else:
            size = len(df.index)
            if size < self.minimum_observations:
                raise ValueError(
                    "This model must have at least {} observations. The input time series contains only {} observations".format(
                        self.minimum_observations, size))

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
