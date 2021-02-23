from dku_config.classical_config import ClassicalConfig
from dku_config.stl_config import STLConfig
from dku_constants import DecompositionMethod
from dku_input_validator.classical_input_validator import ClassicalInputValidator
from dku_input_validator.decomposition_input_validator import DecompositionInputValidator
from dku_timeseries.dku_decomposition.classical_decomposition import ClassicalDecomposition
from dku_timeseries.dku_decomposition.stl_decomposition import STLDecomposition


def check_and_load_params(config, input_dataset_columns):
    decomposition_method = DecompositionMethod(config.get("time_decomposition_method"))
    if decomposition_method == DecompositionMethod.STL:
        dku_config = STLConfig()
        dku_config.add_parameters(config, input_dataset_columns)
        input_validator = DecompositionInputValidator(dku_config)
        decomposition = STLDecomposition(dku_config)
    elif decomposition_method == DecompositionMethod.CLASSICAL:
        dku_config = ClassicalConfig()
        dku_config.add_parameters(config, input_dataset_columns)
        input_validator = ClassicalInputValidator(dku_config)
        decomposition = ClassicalDecomposition(dku_config)
    return dku_config, input_validator, decomposition
