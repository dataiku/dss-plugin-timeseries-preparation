from statsmodels.tsa.tsatools import freq_to_period

from dku_config.dku_config import DkuConfig
from safe_logger import SafeLogger

logger = SafeLogger("Timeseries preparation plugin")


class DecompositionConfig(DkuConfig):
    """Mapping structure containing the parameters of the time series decomposition

    Attributes:
        config(dict): Dict storing the DSSParameters
        minimum_period: Minimum period required by the decomposition method
    """

    def __init__(self):
        super().__init__()
        self.minimum_period = 1

    def add_parameters(self, config, input_dataset_columns):
        """Adds the recipe parameters to dku_config

        Args:
            config(dict):  map of the recipe parameters
            input_dataset_columns(list): the columns of the input datasets
        """
        self._load_input_parameters(config, input_dataset_columns)
        self._load_settings(config)
        if self.advanced:
            self._load_advanced_parameters(config)

    def _load_input_parameters(self, config, input_dataset_columns):
        """Adds the input parameters to dku_config

        Args:
            config(dict):  map of the recipe parameters
            input_dataset_columns(list): the columns of the input datasets
        """
        self.add_param(
            name="time_column",
            value=config.get("time_column"),
            checks=[{"type": "is_type",
                     "op": str
                     },
                    {"type": "in",
                     "op": input_dataset_columns,
                     "err_msg": f"Invalid time column selection: {config.get('time_column')}"
                     }],
            required=True
        )
        self.add_param(
            name="target_columns",
            value=config.get("target_columns"),
            checks=[{"type": "is_type",
                     "op": list
                     },
                    {"type": "in",
                     "op": input_dataset_columns,
                     "err_msg": f"Invalid target column(s) selection: {config.get('target_columns')}"
                     }],
            required=True
        )

        if config.get("frequency_unit") == "W":
            frequency_value = f"W-{config.get('frequency_end_of_week', 'SUN')}"
        elif config.get("frequency_unit") == "H":
            frequency_value = f"{config.get('frequency_step_hours', 1)}H"
        elif config.get("frequency_unit") == "min":
            frequency_value = f"{config.get('frequency_step_minutes', 1)}min"
        else:
            frequency_value = config.get("frequency_unit")

        self.add_param(
            name="frequency",
            value=frequency_value,
            checks=[
                {"type": "custom",
                 "op": frequency_value in ["D", "min", "12M", "3M", "6M", "M", "W", "B", "H"] or frequency_value.startswith("W") or frequency_value.endswith(
                     "H") or frequency_value.startswith("min")
                 }
            ],
            required=True
        )

        if frequency_value:
            if frequency_value == "min" or frequency_value == "12M" or frequency_value.endswith("min"):
                period_value = config.get(f"season_length_{frequency_value}", 1)
            elif frequency_value == "6M":
                period_value = config.get(f"season_length_{frequency_value}", 2)
            elif frequency_value == "3M":
                period_value = config.get(f"season_length_{frequency_value}", 4)
            else:
                period_value = config.get(f"season_length_{frequency_value}", freq_to_period(frequency_value))
                if not config.get(f"season_length_{frequency_value}"):
                    logger.warning(f"The recipe relies on the default period = {period_value} for a frequency = {frequency_value}")

            self.add_param(
                name="period",
                value=period_value,
                checks=[
                    {"type": "is_type",
                     "op": int
                     },
                    {
                        "type": "sup_eq",
                        "op": self.minimum_period
                    }
                ],
                required=True
            )

        long_format = config.get("long_format", False)
        timeseries_identifiers = config.get("timeseries_identifiers")
        is_long_format_valid = (not long_format) or (
                long_format and timeseries_identifiers and len(timeseries_identifiers) != 0)

        self.add_param(
            name="long_format",
            value=long_format,
            checks=[{"type": "is_type",
                     "op": bool
                     },
                    {"type": "custom",
                     "cond": is_long_format_valid,
                     "err_msg": "Long format is selected but no time series identifiers were provided"
                     }])

        if long_format:
            self.add_param(
                name="timeseries_identifiers",
                value=timeseries_identifiers,
                checks=[{"type": "is_type",
                         "op": list
                         },
                        {"type": "in",
                         "op": input_dataset_columns,
                         "err_msg": f"Invalid time series identifiers selection: {timeseries_identifiers}"
                         }],
                required=True
            )
        else:
            self.add_param(
                name="timeseries_identifiers",
                value=[]
            )

    def _load_settings(self, config):
        """Adds the mandatory decomposition parameters to dku_config

        Args:
            config(dict):  map of the recipe parameters
        """
        self.add_param(
            name="time_decomposition_method",
            value=config.get("time_decomposition_method"),
            required=True
        )

        model = config.get("decomposition_model", "additive")

        self.add_param(
            name="model",
            value=model,
            checks=[
                {
                    "type": "in",
                    "op": ["additive", "multiplicative"]
                }
            ],
            required=True
        )

        self.add_param(
            name="advanced",
            value=config.get("expert", False),
            checks=[
                {
                    "type": "is_type",
                    "op": bool
                }
            ],
            required=True
        )

    def _load_advanced_parameters(self, config):
        """Adds the advanced decomposition parameters to dku_config

        Args:
            config(dict):  map of the recipe parameters
        """
        pass
