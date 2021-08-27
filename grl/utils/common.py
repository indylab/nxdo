import datetime
import json
import numbers
import os
import socket
from contextlib import closing
from typing import Tuple, Union

import numpy as np
import yaml

import grl


def data_dir() -> str:
    return os.path.join(os.path.dirname(grl.__file__), "data")


def assets_dir() -> str:
    return os.path.join(os.path.dirname(grl.__file__), "assets")


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def check_if_jsonable(check_dict) -> Tuple[bool, Union[Exception, None]]:
    try:
        json.dumps(check_dict, cls=SafeFallbackJSONEncoder)
    except (TypeError, OverflowError) as json_err:
        return False, json_err
    return True, None


class SafeFallbackJSONEncoder(json.JSONEncoder):

    def __init__(self, nan_str="null", **kwargs):
        super(SafeFallbackJSONEncoder, self).__init__(**kwargs)
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

            return super(SafeFallbackJSONEncoder, self).default(value)

        except Exception:
            return str(value)  # give up, just stringify it (ok for logs)


def pretty_dict_str(result):
    result = result.copy()
    result.update(config=None)  # drop config from pretty print
    out = {}
    for k, v in result.items():
        if v is not None:
            out[k] = v

    cleaned = json.dumps(out, cls=SafeFallbackJSONEncoder)
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
