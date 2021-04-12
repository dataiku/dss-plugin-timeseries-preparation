from dku_config.stl_config import STLConfig
from dku_input_validator.decomposition_input_validator import DecompositionInputValidator
from dku_timeseries.dku_decomposition.stl_decomposition import STLDecomposition


def check_and_load_params(config, input_dataset_columns):
    dku_config = STLConfig()
    dku_config.add_parameters(config, input_dataset_columns)
    input_validator = DecompositionInputValidator(dku_config)
    decomposition = STLDecomposition(dku_config)
    return dku_config, input_validator, decomposition
