from dku_config.additional_parameter import AdditionalParameter
from dku_config.utils import is_positive_int


class AdditionalSmoother(AdditionalParameter):
    def __init__(self, name, value):
        super().__init__(name, value)
        self.error_message += "Its value should be an odd positive integer greater than 3 and the season length."

    def check(self, dku_config):
        minimum = max(dku_config.period, 3)
        return (self.value == "") or (is_odd(self.value) and float(self.value) > minimum)

    def parse_value(self):
        if self.value:
            return int(self.value)
        else:
            return self.value


def is_odd(x):
    if is_positive_int(x):
        numeric_value = int(x)
    else:
        numeric_value = None
    return numeric_value and numeric_value % 2 == 1
