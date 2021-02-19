import ray
import numpy as np


# Defined after Ray driver is created
# https://github.com/ray-project/ray/issues/6240

@ray.remote(num_cpus=0)
class StatDeque(object):
    def __init__(self, max_items: int):
        self._data = []
        self._max_items = max_items

    def add(self, item):
        self._data.append(item)
        if len(self._data) > self._max_items:
            del self._data[0]

    def get_mean(self):
        if len(self._data) == 0:
            return None
        return np.mean(self._data)
