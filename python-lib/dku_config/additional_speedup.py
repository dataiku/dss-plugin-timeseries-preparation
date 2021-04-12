from dku_config.additional_parameter import AdditionalParameter
from dku_config.utils import is_positive_int


class AdditionalSpeedup(AdditionalParameter):
    def __init__(self, name, value):
        super().__init__(name, value)
        self.error_message += "Its value should be a positive integer."

    def check(self, dku_config):
        return (self.value == "") or (is_positive_int(self.value))

    def parse_value(self):
        if self.value:
            return int(self.value)
        else:
            return self.value
