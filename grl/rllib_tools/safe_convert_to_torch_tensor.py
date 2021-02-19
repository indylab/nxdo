import numpy as np
import tree

from ray.rllib.models.repeated_values import RepeatedValues
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


def safe_convert_to_torch_tensor(x, device=None):
    """Converts any struct to torch.Tensors.

    Modified from original RLlib implementation to copy elements of x
    and ensure any numpy arrays are set as writable.

    x (any): Any (possibly nested) struct, the values in which will be
        converted and returned as a new struct with all leaves converted
        to torch tensors.

    Returns:
        Any: A new struct with the same structure as `stats`, but with all
            values converted to torch Tensor types.
    """

    def mapping(item):
        # Already torch tensor -> make sure it's on right device.
        if torch.is_tensor(item):
            return item if device is None else item.to(device)
        # Special handling of "Repeated" values.
        elif isinstance(item, RepeatedValues):
            return RepeatedValues(
                tree.map_structure(mapping, item.values), item.lengths,
                item.max_len)
        # copy and set flags are not performed by RLlib
        np_item = np.asarray(item).copy()
        np_item.setflags(write=1)
        tensor = torch.from_numpy(np_item)
        # Floatify all float64 tensors.
        if tensor.dtype == torch.double:
            tensor = tensor.float()
        return tensor if device is None else tensor.to(device)

    return tree.map_structure(mapping, x)
