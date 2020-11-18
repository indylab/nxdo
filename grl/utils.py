import json
import yaml
import numbers
import numpy as np
import datetime
import os
from typing import Tuple


def check_if_jsonable(check_dict):
    try:
        json.dumps(check_dict)
    except (TypeError, OverflowError) as json_err:
        return False, json_err
    return True, None


class _SafeFallbackEncoder(json.JSONEncoder):
    
    def __init__(self, nan_str="null", **kwargs):
        super(_SafeFallbackEncoder, self).__init__(**kwargs)
        self.nan_str = nan_str

    def default(self, value):
        try:
            if np.isnan(value):
                return self.nan_str

            if (type(value).__module__ == np.__name__
                    and isinstance(value, np.ndarray)):
                return value.tolist()

            if issubclass(type(value), numbers.Integral):
                return int(value)
            if issubclass(type(value), numbers.Number):
                return float(value)

            return super(_SafeFallbackEncoder, self).default(value)

        except Exception:
            return str(value)  # give up, just stringify it (ok for logs)


def pretty_dict_str(result):
    result = result.copy()
    result.update(config=None)  # drop config from pretty print
    out = {}
    for k, v in result.items():
        if v is not None:
            out[k] = v

    cleaned = json.dumps(out, cls=_SafeFallbackEncoder)
    return yaml.safe_dump(json.loads(cleaned), default_flow_style=False)


def datetime_str():
    # format is ok for file/directory names
    date_string = datetime.datetime.now().strftime("%I.%M.%S%p_%b-%d-%Y")
    return date_string


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def string_to_int_tuple(s) -> Tuple[int]:
    return tuple(int(number) for number in s.replace('(', '').replace(')', '').replace('...', '').split(','))
