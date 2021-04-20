from dku_config.additional_parameter import AdditionalParameter
from dku_config.utils import is_odd


class AdditionalSmoother(AdditionalParameter):
    def __init__(self, name, value):
        super().__init__(name, value)
        self.error_message += "It must be an odd positive integer greater than 3 and the season length."

    def is_valid(self, dku_config):
        minimum = max(dku_config.season_length, 3)
        return (self.value is None) or (is_odd(self.value) and self.value > minimum)
