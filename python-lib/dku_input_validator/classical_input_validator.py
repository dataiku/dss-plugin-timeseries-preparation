from dku_input_validator.decomposition_input_validator import DecompositionInputValidator


class ClassicalInputValidator(DecompositionInputValidator):
    def __init__(self, dku_config):
        super().__init__(dku_config)
        self.minimum_observations = 2 * self.dku_config.period
