from dku_input_validator.decomposition_input_validator import DecompositionInputValidator


class ClassicalInputValidator(DecompositionInputValidator):
    """Checks if the input dataframe is compatible with a classical decomposition

    Attributes:
        dku_config(DecompositionConfig): mapping structure storing the recipe parameters
        minimum_observations(int): Minimum number of observations required by a classical decomposition
    """

    def __init__(self, dku_config):
        super().__init__(dku_config)
        self.minimum_observations = 2 * self.dku_config.period
