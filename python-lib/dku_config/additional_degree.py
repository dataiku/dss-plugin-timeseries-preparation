from dku_config.additional_parameter import AdditionalParameter


class AdditionalDegree(AdditionalParameter):
    def __init__(self, name, value):
        super().__init__(name, value)
        self.error_message += "Its value must be equal to 0 or 1."

    def check(self, dku_config):
        return self.value in ["0", "1", ""]

    def parse_value(self):
        if self.value:
            return int(self.value)
        else:
            return self.value
