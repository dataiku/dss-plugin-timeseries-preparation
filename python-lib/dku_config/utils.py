class PluginCodeEnvError(Exception):
    """Exception raised when the code env is not compatible with the plugin
    """
    pass


def are_keys_in(expected_keys, map_parameter):
    return all(x in expected_keys for x in map_parameter.keys())


def cast_kwargs(kwargs):
    kwargs_copy = kwargs.copy()
    for arg, value in kwargs_copy.items():
        if isinstance(value, dict):
            kwargs_copy[arg] = cast_kwargs(value)
        else:
            kwargs_copy[arg] = cast_string(kwargs_copy[arg])
    return kwargs_copy


def cast_string(s):
    if s == "True":
        return True
    elif s == "False":
        return False
    elif s == "None" or s == "":
        return None
    elif is_int(s):
        return int(s)
    elif is_float(s):
        return float(s)
    else:
        return s


def is_int(s):
    try:
        int(s)
        return True
    except (ValueError, TypeError):
        return False


def is_float(s):
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def is_positive_int(x):
    return isinstance(x, int) and x >= 0


def is_odd(x):
    return is_positive_int(x) and x % 2 == 1
