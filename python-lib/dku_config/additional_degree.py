from dku_config.additional_parameter import AdditionalParameter


class AdditionalDegree(AdditionalParameter):
    def __init__(self, name, value):
        super().__init__(name, value)
        self.error_message += "It must be equal to 0 or 1."

    def is_valid(self, dku_config):
        return self.value in [0, 1]
