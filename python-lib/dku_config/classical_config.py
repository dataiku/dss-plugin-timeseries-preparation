import json
from json import JSONDecodeError
import numpy as np
from statsmodels.tools.validation.validation import array_like

from dku_config.decomposition_config import DecompositionConfig
from dku_config.utils import is_positive_int, are_keys_in


class ClassicalConfig(DecompositionConfig):
    def __init__(self):
        super().__init__()

    def _load_advanced_parameters(self, config):
        advanced_params = config.get("advanced_params_classical", {})
        if advanced_params:
            self.add_param(
                name="advanced_params",
                value=advanced_params,
                checks=[
                    {
                        "type": "is_type",
                        "op": dict
                    },
                    {
                        "type": "custom",
                        "cond": are_keys_in(["filt", "two_sided", "extrapolate_trend"], advanced_params),
                        "err_msg": "This field is invalid. The keys should be in the following iterable: [filt, two_sided,extrapolate_trend]"
                    }
                ],
                required=False
            )

            filt = self.advanced_params.get("filt")
            if filt:
                try:
                    filt_parsed = json.loads(filt)
                    filt_array = array_like(filt_parsed, 'filt')
                except (JSONDecodeError, ValueError):
                    filt_array = None

                self.add_param(
                    name="filt",
                    value=filt_array,
                    checks=[
                        {
                            "type": "is_type",
                            "op": np.ndarray
                        },
                        {
                            "type": "custom",
                            "op": filt_array is not None,
                            "err_msg": "This field is invalid. It should be a numeric array such as [1, 2]"
                        },

                    ],
                    required=False
                )

            two_sided = self.advanced_params.get("two_sided", "True")

            if two_sided:
                if two_sided == "True":
                    two_sided_parsed = True
                elif two_sided == "False":
                    two_sided_parsed = False
                else:
                    two_sided_parsed = two_sided

                self.add_param(
                    name="two_sided",
                    value=two_sided_parsed,
                    checks=[
                        {
                            "type": "is_type",
                            "op": bool
                        }
                    ],
                    required=False
                )

            extrapolate_trend = self.advanced_params.get("extrapolate_trend")
            if extrapolate_trend:
                is_extrapolate_int = is_positive_int(extrapolate_trend)
                is_extrapolate_valid = extrapolate_trend == "freq" or is_extrapolate_int

                if is_extrapolate_int:
                    extrapolate_trend_parsed = int(extrapolate_trend)
                else:
                    extrapolate_trend_parsed = extrapolate_trend

                self.add_param(
                    name="extrapolate_trend",
                    value=extrapolate_trend_parsed,
                    checks=[
                        {
                            "type": "custom",
                            "cond": is_extrapolate_valid,
                            "err_msg": "Extrapolate trend must be a positive integer or equal to 'freq'"

                        }
                    ],
                    required=False
                )
