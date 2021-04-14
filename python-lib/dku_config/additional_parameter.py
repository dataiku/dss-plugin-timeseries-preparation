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
