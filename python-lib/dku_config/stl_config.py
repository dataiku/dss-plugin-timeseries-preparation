from dku_config.additional_parameter import AdditionalParameter, AdditionalSmoother, AdditionalSpeedup, AdditionalDegree
from dku_config.decomposition_config import DecompositionConfig
from dku_config.utils import are_keys_in, cast_kwargs


class STLConfig(DecompositionConfig):
    """Mapping structure containing the parameters of the STL decomposition

    Attributes:
        config(dict): Dict storing the DSSParameters
        minimum_period: Minimum period required by STL
    """

    def __init__(self):
        super().__init__()
        self.minimum_season_length = 2

    def _load_advanced_parameters(self, config):
        seasonal = config.get("seasonal_stl")
        self.add_param(
            name="seasonal",
            value=seasonal,
            checks=[
                {
                    "type": "exists",
                    "err_msg": "The seasonal smoother of the expert mode is required and must be an odd integer greater than 7."

                },
                {
                    "type": "is_type",
                    "op": int
                },
                {
                    "type": "sup_eq",
                    "op": 7

                },
                {
                    "type": "custom",
                    "cond": (seasonal and isinstance(seasonal, int) and (seasonal % 2 == 1)),
                    "err_msg": "The seasonal smoother must be an odd integer."
                }]
        )

        self.add_param(
            name="robust_stl",
            value=config.get("robust_stl", False),
            checks=[
                {
                    "type": "is_type",
                    "op": bool
                }
            ],
            required=False
        )

        additional_parameters = config.get("additional_parameters_STL", {})
        if additional_parameters:
            additional_parameters = cast_kwargs(additional_parameters)
            self.add_param(
                name="additional_parameters_STL",
                value=additional_parameters,
                checks=[
                    {
                        "type": "is_type",
                        "op": dict
                    },
                    {
                        "type": "custom",
                        "cond": are_keys_in(
                            ["trend", "low_pass", "seasonal_deg", "trend_deg", "low_pass_deg", "seasonal_jump", "trend_jump", "low_pass_jump"],
                            additional_parameters),
                        "err_msg": "The keys should be in the following iterable: ['trend', 'low_pass', 'seasonal_deg', "
                                   "'trend_deg','low_pass_deg', 'seasonal_jump', 'trend_jump', 'low_pass_jump']."
                    }
                ],
                required=False
            )
            self._check_additional_parameters(additional_parameters)

    def _check_additional_parameters(self, advanced_params):
        for parameter_name, value in advanced_params.items():
            additional_parameter = self._get_additional_parameter(parameter_name, value)
            if not additional_parameter.is_valid(self):
                raise ValueError(additional_parameter.get_full_error_message())

    @staticmethod
    def _get_additional_parameter(parameter_name, value):
        if parameter_name in ["trend", "low_pass"]:
            additional_parameter = AdditionalSmoother(parameter_name, value)
        elif parameter_name in ["seasonal_jump", "trend_jump", "low_pass_jump"]:
            additional_parameter = AdditionalSpeedup(parameter_name, value)
        elif parameter_name in ["seasonal_deg", "trend_deg", "low_pass_deg"]:
            additional_parameter = AdditionalDegree(parameter_name, value)
        else:
            additional_parameter = AdditionalParameter(parameter_name, value)
        return additional_parameter
