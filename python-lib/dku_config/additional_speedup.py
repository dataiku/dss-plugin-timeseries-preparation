from dku_config.additional_parameter import AdditionalParameter
from dku_config.utils import is_positive_int


class AdditionalSpeedup(AdditionalParameter):
    def __init__(self, name, value):
        super().__init__(name, value)
        self.error_message += "It must be a positive integer."

    def is_valid(self, dku_config):
        return is_positive_int(self.value)


