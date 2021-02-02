from dku_config.decomposition_config import DecompositionConfig


class STLConfig(DecompositionConfig):
    def __init__(self):
        super().__init__()

    def _load_settings(self, config, input_df):
        self.add_param(
            name="transformation_type",
            value=config.get("transformation_type"),
            required=True
        )

        self.add_param(
            name="time_decomposition_method",
            value=config.get("time_decomposition_method"),
            required=True
        )

        seasonal = config.get("seasonal_stl")
        is_seasonal_odd = True
        if seasonal:
            is_seasonal_odd = (seasonal % 2 == 1)
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
                    "cond": is_seasonal_odd,
                    "err_msg": "The seasonal smoother should be an odd integer."
                }],
            required=True
        )

        model = config.get("model_stl","additive")
        multiplicative_check = self._check_multiplicative_model(model, input_df)
        self.add_param(
            name="model_stl",
            value=model,
            checks=[
                {
                    "type": "in",
                    "op": ["additive", "multiplicative"]
                },
                {
                    "type": "custom",
                    "cond": multiplicative_check.valid_model,
                    "err_msg": f"{multiplicative_check.negative_column}, a targeted column contains negative values. Yet, a multiplicative STL model only works with positive time series. You may choose an additive model instead. "
                }
            ],
            required=True
        )

        self.add_param(
            name="advanced",
            value=config.get("expert_stl", False),
            checks=[
                {
                    "type": "is_type",
                    "op": bool
                }
            ],
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

        degree_kwargs = config.get("stl_degree_kwargs", {})
        self.add_param(
            name="loess_degrees",
            value=degree_kwargs,
            checks=[
                {
                    "type": "is_type",
                    "op": dict
                },
                {
                    "type": "custom",
                    "cond": all(
                        x in ["seasonal_deg", "trend_deg", "low_pass_deg", ""] for x in degree_kwargs.keys()),
                    "err_msg": "This field is invalid. The keys should be in the following iterable: [seasonal_deg, trend_deg,low_pass_deg]"
                },
                {
                    "type": "custom",
                    "cond": all(x in ["0", "1", ""] for x in degree_kwargs.values()),
                    "err_msg": "This field is invalid. The degrees used for Loess estimation should be equal to 0 and 1"
                }
            ],
            required=False
        )

        speed_up_kwargs = config.get("stl_speed_jump_kwargs", {})

        self.add_param(
            name="speed_jumps",
            value=speed_up_kwargs,
            checks=[
                {
                    "type": "is_type",
                    "op": dict
                },
                {
                    "type": "custom",
                    "cond": all(
                        x in ["seasonal_jump", "trend_jump", "low_pass_jump", ""] for x in speed_up_kwargs.keys()),
                    "err_msg": "This field is invalid. The keys should be in the following iterable: [seasonal_jump, trend_jump,low_pass_jump]"
                },
                {
                    "type": "custom",
                    "cond": all((x.isnumeric() and float(x).is_integer() and float(x) >= 0) or (x == "") for x in
                                speed_up_kwargs.values()),
                    "err_msg": "This field is invalid. The values should be positive integers."
                }
            ],
            required=False
        )

        additional_smoothers = config.get("stl_smoothers_kwargs", {})

        self.add_param(
            name="additional_smoothers",
            value=additional_smoothers,
            checks=[
                {
                    "type": "is_type",
                    "op": dict
                },
                {
                    "type": "custom",
                    "cond": all(
                        x in ["trend", "low_pass", ""] for x in additional_smoothers.keys()),
                    "err_msg": "This field is invalid. The keys should be in the following iterable: [trend, low_pass]"
                },
                {
                    "type": "custom",
                    "cond": all((x.isnumeric() and float(x).is_integer() and int(x) % 2 == 1) or (x == "") for x in
                                additional_smoothers.values()),
                    "err_msg": "This field is invalid. The values should be odd positive integers."
                }
            ],
            required=False
        )
