import numpy as np

from dku_config.dku_config import DkuConfig


class DecompositionConfig(DkuConfig):
    def __init__(self):
        super().__init__()

    def add_parameters(self, config, input_df):
        input_dataset_columns = list(input_df.columns)
        self._load_input_parameters(config, input_dataset_columns)
        self._load_settings(config, input_df)
        if self.advanced:
            self._load_advanced_parameters(config)

    def _load_input_parameters(self, config, input_dataset_columns):
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
            frequency_value = f"W-{config.get('frequency_end_of_week', 1)}"
        elif config.get("frequency_unit") == "H":
            frequency_value = f"{config.get('frequency_step_hours', 1)}H"
        elif config.get("frequency_unit") == "min":
            frequency_value = f"{config.get('frequency_step_minutes', 1)}min"
        else:
            frequency_value = config.get("frequency_unit")

        self.add_param(
            name="frequency",
            value=frequency_value,
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

    def _load_settings(self, config, input_df):
        self.add_param(
            name="time_decomposition_method",
            value=config.get("time_decomposition_method"),
            required=True
        )

        model = config.get("decomposition_model", "additive")
        columns_with_invalid_values = self._get_columns_with_invalid_values(model, input_df)

        self.add_param(
            name="model",
            value=model,
            checks=[
                {
                    "type": "in",
                    "op": ["additive", "multiplicative"]
                },
                {
                    "type": "custom",
                    "cond": len(columns_with_invalid_values) == 0,
                    "err_msg": f"{', '.join(columns_with_invalid_values)}, a targeted column contains negative values. Yet, a multiplicative STL model only works with positive time series. You may choose an additive model instead. "
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
        pass

    def _get_columns_with_invalid_values(self, model, input_df):
        columns_with_invalid_values = []
        if model == "multiplicative":
            for target_column in self.target_columns:
                target_values = input_df[target_column].values
                if np.any(target_values <= 0):
                    columns_with_invalid_values.append(target_column)
        return columns_with_invalid_values
