import numpy as np

import dataiku
from dataiku.customrecipe import get_input_names_for_role, get_output_names_for_role

from dku_config.dku_config import DkuConfig
from dku_config.utils import MultiplicativeCheck


class DecompositionConfig(DkuConfig):
    def __init__(self):
        super().__init__()

    def add_parameters(self, config):
        input_dataset_columns = [p["name"] for p in self.input_dataset.read_schema()]
        self._load_input_parameters(config, input_dataset_columns)
        input_df = self.input_dataset.get_dataframe()
        self._load_settings(config, input_df)
        if self.advanced:
            self._load_advanced_parameters(config)

    def load_input_output_datasets(self, input_dataset_name, output_dataset_name):
        self.add_param(
            name="input_dataset_name",
            value=input_dataset_name,
            checks=[{
                "type": "is_type",
                "op": str
            }],
            required=True
        )
        self.add_param(
            name="input_dataset",
            value=dataiku.Dataset(self.input_dataset_name),
            required=True
        )
        self.add_param(
            name="output_dataset_name",
            value=output_dataset_name,
            checks=[{
                "type": "is_type",
                "op": str
            }],
            required=True
        )
        self.add_param(
            name="output_dataset",
            value=dataiku.Dataset(self.output_dataset_name),
            required=True
        )

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

        if config.get("frequency_unit") not in ["W", "H", "min"]:
            frequency_value = config.get("frequency_unit")
        elif config.get("frequency_unit") == "W":
            frequency_value = f"W-{config.get('frequency_end_of_week', 1)}"
        elif config.get("frequency_unit") == "H":
            frequency_value = f"{config.get('frequency_step_hours', 1)}H"
        elif config.get("frequency_unit") == "min":
            frequency_value = f"{config.get('frequency_step_minutes', 1)}min"
        else:
            frequency_value = None

        self.add_param(
            name="frequency",
            value=frequency_value,
            required=True
        )

        long_format = config.get("long_format", False)
        timeseries_identifiers = config.get("timeseries_identifiers")
        is_long_format_valid = True
        if long_format and (not timeseries_identifiers or len(timeseries_identifiers)) == 0:
            is_long_format_valid = False

        self.add_param(
            name="long_format",
            value=long_format,
            checks=[{"type": "custom",
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
        pass

    def _load_advanced_parameters(self, config):
        pass

    def _check_multiplicative_model(self, model, input_df):
        if model == "multiplicative":
            for target_column in self.target_columns:
                target_values = input_df[target_column].values
                if np.any(target_values <= 0):
                    return MultiplicativeCheck(False, target_column)
            return MultiplicativeCheck(True)
        else:
            return MultiplicativeCheck(True)
