from dku_config.decomposition_config import DecompositionConfig
from dku_config.utils import are_keys_in, is_positive_int


class STLConfig(DecompositionConfig):
    def __init__(self):
        super().__init__()
        self.minimum_period = 2

    def _load_settings(self, config):
        super()._load_settings(config)
        seasonal = config.get("seasonal_stl")
        self.add_param(
            name="seasonal",
            value=seasonal,
            checks=[
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
                }],
            required=True
        )

    def _load_advanced_parameters(self, config):
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

        degree_kwargs = config.get("stl_degree_kwargs")
        if degree_kwargs:
            are_degree_keys_valid = are_keys_in(["seasonal_deg", "trend_deg", "low_pass_deg"], degree_kwargs)
            are_degree_values_valid = all(x in ["0", "1", ""] for x in degree_kwargs.values())

            degrees_parsed = degree_kwargs
            if are_degree_keys_valid and are_degree_values_valid:
                degrees_parsed = parse_values_to_int(degree_kwargs)

            self.add_param(
                name="loess_degrees",
                value=degrees_parsed,
                checks=[
                    {
                        "type": "is_type",
                        "op": dict
                    },
                    {
                        "type": "custom",
                        "cond": are_degree_keys_valid,
                        "err_msg": "This field is invalid. The keys should be in the following iterable: [seasonal_deg, trend_deg,low_pass_deg]"
                    },
                    {
                        "type": "custom",
                        "cond": are_degree_values_valid,
                        "err_msg": "This field is invalid. The degrees used for Loess estimation must be equal to 0 or 1"
                    }
                ],
                required=False
            )

        speed_up_kwargs = config.get("stl_speed_jump_kwargs")
        if speed_up_kwargs:
            are_speed_up_keys_valid = are_keys_in(["seasonal_jump", "trend_jump", "low_pass_jump"], speed_up_kwargs)
            are_speed_up_values_valid = all((is_positive_int(x)) for x in speed_up_kwargs.values())

            speed_up_parsed = speed_up_kwargs
            if are_speed_up_keys_valid and are_speed_up_values_valid:
                speed_up_parsed = parse_values_to_int(speed_up_kwargs)

            self.add_param(
                name="speed_jumps",
                value=speed_up_parsed,
                checks=[
                    {
                        "type": "is_type",
                        "op": dict
                    },
                    {
                        "type": "custom",
                        "cond": are_speed_up_keys_valid,
                        "err_msg": "This field is invalid. The keys should be in the following iterable: [seasonal_jump, trend_jump,low_pass_jump]"
                    },
                    {
                        "type": "custom",
                        "cond": are_speed_up_values_valid,
                        "err_msg": "This field is invalid. The values should be positive integers."
                    }
                ],
                required=False
            )

        additional_smoothers = config.get("stl_smoothers_kwargs")

        if additional_smoothers:
            are_smoothers_keys_valid = are_keys_in(["trend", "low_pass"], additional_smoothers)
            minimum = max(self.period, 3)
            are_smoothers_values_valid = all((x == "") or
                                             ((is_odd(x) and float(x) > minimum)) for x in additional_smoothers.values())

            if are_smoothers_keys_valid and are_smoothers_values_valid:
                smoothers_parsed = parse_values_to_int(additional_smoothers)
            else:
                smoothers_parsed = additional_smoothers

            self.add_param(
                name="additional_smoothers",
                value=smoothers_parsed,
                checks=[
                    {
                        "type": "is_type",
                        "op": dict
                    },
                    {
                        "type": "custom",
                        "cond": are_smoothers_keys_valid,
                        "err_msg": "This field is invalid. The keys should be in the following iterable: [trend, low_pass]"
                    },
                    {
                        "type": "custom",
                        "cond": are_smoothers_values_valid,
                        "err_msg": f"This field is invalid. The values should be odd positive integers greater than 3 and the period (= {self.period})"
                    }
                ],
                required=False
            )


def parse_values_to_int(map_parameter):
    map_parameter_parsed = {}
    for key, value in map_parameter.items():
        if value:
            map_parameter_parsed[key] = int(value)
    return map_parameter_parsed


def is_odd(x):
    if x.isnumeric():
        numeric_value = float(x)
    else:
        numeric_value = None
    return (x == "") or (numeric_value and numeric_value.is_integer() and numeric_value >= 0 and numeric_value % 2 == 1)
