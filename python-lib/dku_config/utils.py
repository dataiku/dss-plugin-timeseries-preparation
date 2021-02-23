def are_keys_in(expected_keys, map_parameter):
    valid_keys = expected_keys.copy()
    valid_keys.append("")
    return all(x in valid_keys for x in map_parameter.keys())


def is_positive_int(x):
    if x.isnumeric():
        numeric_value = float(x)
    else:
        numeric_value = None
    return numeric_value and numeric_value.is_integer() and numeric_value >= 0