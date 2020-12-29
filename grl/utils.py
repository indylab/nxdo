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


# Copies attributes of one object to a destination object
# https://stackoverflow.com/questions/50003851/python-how-to-override-a-memory-at-a-pointer-location
def copy_attributes(src_obj, dst_obj):
    if not isinstance(src_obj, type(dst_obj)) and not isinstance(dst_obj, type(src_obj)):
        raise TypeError("source and target object types must be related somehow")
    # fast path if both have __dict__
    if hasattr(src_obj, "__dict__") and hasattr(dst_obj, "__dict__"):
        dst_obj.__dict__.update(src_obj.__dict__)
        return
    # copy one attribute at a time
    names = getattr(type(src_obj), "__slots__", ()) and getattr(src_obj, "__dict__", ())
    slots = set(getattr(type(dst_obj), "__slots__", ()))
    if slots and not all(name in slots for name in names):
        raise AttributeError("target lacks a slot for an attribute from source")
    for name in names:
        setattr(dst_obj, getattr(src_obj, name))

