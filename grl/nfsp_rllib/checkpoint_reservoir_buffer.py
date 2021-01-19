import random
import numpy as np


class ReservoirBuffer(object):
    """Allows uniform sampling over a stream of data.

    This class supports the storage of arbitrary elements, such as observation
    tensors, integer actions, etc.

    See https://en.wikipedia.org/wiki/Reservoir_sampling for more details.
    """

    def __init__(self, reservoir_buffer_capacity: int):
        self._reservoir_buffer_capacity = reservoir_buffer_capacity
        self._data = []
        self._add_calls = 0

    def ask_to_add(self):
        self._add_calls += 1
        if len(self._data) < self._reservoir_buffer_capacity:
            return True, None
        else:
            idx = np.random.randint(0, self._add_calls)
            if idx < self._reservoir_buffer_capacity:
                return True, idx
        return False, None

    def add(self, element, idx):
        """Potentially adds `element` to the reservoir buffer.

        Args:
          element: data to be added to the reservoir buffer.
        """
        if idx is None:
            assert len(self._data) < self._reservoir_buffer_capacity
            self._data.append(element)
        else:
            assert idx < self._reservoir_buffer_capacity
            self._data[idx] = element

    def sample(self, num_samples=1):
        """Returns `num_samples` uniformly sampled from the buffer.

        Args:
          num_samples: `int`, number of samples to draw.

        Returns:
          An iterable over `num_samples` random elements of the buffer.

        Raises:
          ValueError: If there are less than `num_samples` elements in the buffer
        """
        if len(self._data) < num_samples:
            raise ValueError("{} elements could not be sampled from size {}".format(
                num_samples, len(self._data)))
        return random.sample(self._data, num_samples)

    def __setitem__(self, index, value):
        self._data[index] = value

    def __getitem__(self, index):
        return self._data[index]

    def clear(self):
        self._data = []
        self._add_calls = 0

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)