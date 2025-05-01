from __future__ import absolute_import

from dku_config.utils import is_positive_int, is_odd


class AdditionalParameter:
    def __init__(self, name, value):
        self.name = name
        self.value = value
        self.error_message = ""

    def is_valid(self, dku_config):
        pass

    def get_full_error_message(self):
        return """
                Error for the additional parameter \"{name}\" :
                {error}
                Please check your settings and fix the error.
                You may remove this mapping parameter to use Statsmodel's default value. 
                """.format(
            name=self.name,
            error=self.error_message
        )


class AdditionalDegree(AdditionalParameter):
    def __init__(self, name, value):
        super().__init__(name, value)
        self.error_message += "It must be equal to 0 or 1."

    def is_valid(self, dku_config):
        return self.value in [0, 1]


class AdditionalSmoother(AdditionalParameter):
    def __init__(self, name, value):
        super().__init__(name, value)
        self.error_message += "It must be None or an odd positive integer greater than 3 and the season length."

    def is_valid(self, dku_config):
        minimum = max(dku_config.season_length, 3)
        return (self.value is None) or (is_odd(self.value) and self.value > minimum)


class AdditionalSpeedup(AdditionalParameter):
    def __init__(self, name, value):
        super().__init__(name, value)
        self.error_message += "It must be a positive integer."

    def is_valid(self, dku_config):
        return is_positive_int(self.value)
