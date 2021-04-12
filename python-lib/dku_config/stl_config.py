from dku_config.additional_degree import AdditionalDegree
from dku_config.additional_smoother import AdditionalSmoother
from dku_config.additional_speedup import AdditionalSpeedup
from dku_config.decomposition_config import DecompositionConfig
from dku_config.utils import are_keys_in


class STLConfig(DecompositionConfig):
    """Mapping structure containing the parameters of the STL decomposition

    Attributes:
        config(dict): Dict storing the DSSParameters
        minimum_period: Minimum period required by STL
    """

    def __init__(self):
        super().__init__()
        self.minimum_period = 2

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

        advanced_params = config.get("advanced_params_STL", {})
        if advanced_params:
            self.add_param(
                name="advanced_params_STL",
                value=advanced_params,
                checks=[
                    {
                        "type": "is_type",
                        "op": dict
                    },
                    {
                        "type": "custom",
                        "cond": are_keys_in(["trend", "low_pass", "seasonal_deg", "trend_deg", "low_pass_deg", "seasonal_jump", "trend_jump", "low_pass_jump"],
                                            advanced_params),
                        "err_msg": "This field is invalid. The keys should be in the following iterable: ['trend', 'low_pass', 'seasonal_deg', 'trend_deg', "
                                   "'low_pass_deg', 'seasonal_jump', 'trend_jump', 'low_pass_jump']."
                    }
                ],
                required=False
            )

            for parameter_name, value in advanced_params.items():
                additional_parameter = get_advanced_param(parameter_name, value)
                is_valid = additional_parameter.check(self)
                if is_valid:
                    value = additional_parameter.parse_value()

                self.add_param(
                    name=parameter_name,
                    value=value,
                    checks=[
                        {
                            "type": "custom",
                            "cond": is_valid,
                            "err_msg": additional_parameter.error_message
                        }
                    ],
                    required=False
                )


def get_advanced_param(parameter_name, value):
    if parameter_name in ["trend", "low_pass"]:
        additional_parameter = AdditionalSmoother(parameter_name, value)
    elif parameter_name in ["seasonal_jump", "trend_jump", "low_pass_jump"]:
        additional_parameter = AdditionalSpeedup(parameter_name, value)
    elif parameter_name in ["seasonal_deg", "trend_deg", "low_pass_deg"]:
        additional_parameter = AdditionalDegree(parameter_name, value)
    return additional_parameter

