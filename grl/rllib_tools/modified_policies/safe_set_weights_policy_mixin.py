from ray.rllib.utils.typing import ModelWeights

from grl.rllib_tools.safe_convert_to_torch_tensor import safe_convert_to_torch_tensor


class SafeSetWeightsPolicyMixin:

    """
    Prevents occasional crashing seen in the default RLLib (version 1.0.1.post1) set_weights implementation
    due to unwritable Numpy arrays being converted to Torch tensors.
    """

    def set_weights(self, weights: ModelWeights) -> None:
        weights = safe_convert_to_torch_tensor(weights, device=self.device)
        self.model.load_state_dict(weights)
