import gym
import numpy as np
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.numpy import SMALL_NUMBER, MIN_LOG_NN_OUTPUT, \
    MAX_LOG_NN_OUTPUT
from ray.rllib.utils.typing import TensorType, List, Union, \
    ModelConfigDict

torch, nn = try_import_torch()


class _TorchSquashedGaussianBase(TorchDistributionWrapper):
    """A diagonal gaussian distribution, squashed into bounded support."""

    def __init__(self,
                 inputs: List[TensorType],
                 model: TorchModelV2,
                 low: float = -1.0,
                 high: float = 1.0):
        """Parameterizes the distribution via `inputs`.
        Args:
            low (float): The lowest possible sampling value
                (excluding this value).
            high (float): The highest possible sampling value
                (excluding this value).
        """
        super().__init__(inputs, model)

        assert low < high
        # Make sure high and low are torch tensors.
        self.low = torch.from_numpy(np.array(low))
        self.high = torch.from_numpy(np.array(high))
        # Place on correct device.
        if isinstance(model, TorchModelV2):
            device = next(model.parameters()).device
            self.low = self.low.to(device)
            self.high = self.high.to(device)

        mean, log_std = torch.chunk(self.inputs, 2, dim=-1)
        self._num_vars = mean.shape[1]
        assert log_std.shape[1] == self._num_vars
        # Clip `std` values (coming from NN) to reasonable values.
        self.log_std = torch.clamp(log_std, MIN_LOG_NN_OUTPUT,
                                   MAX_LOG_NN_OUTPUT)
        # Clip loc too, for numerical stability reasons.
        mean = torch.clamp(mean, -3, 3)
        std = torch.exp(self.log_std)
        self.distr = torch.distributions.normal.Normal(mean, std)
        assert len(self.distr.loc.shape) == 2
        assert len(self.distr.scale.shape) == 2

    @override(TorchDistributionWrapper)
    def sample(self):
        s = self._squash(self.distr.sample())
        assert len(s.shape) == 2
        self.last_sample = s
        return s

    @override(ActionDistribution)
    def deterministic_sample(self) -> TensorType:
        mean = self.distr.loc
        assert len(mean.shape) == 2
        s = self._squash(mean)
        assert len(s.shape) == 2
        self.last_sample = s
        return s

    @override(ActionDistribution)
    def logp(self, x: TensorType) -> TensorType:
        # Unsquash values (from [low,high] to ]-inf,inf[)
        assert len(x.shape) >= 2, "First dim batch, second dim variable"
        unsquashed_values = self._unsquash(x)
        # Get log prob of unsquashed values from our Normal.
        log_prob_gaussian = self.distr.log_prob(unsquashed_values)
        # For safety reasons, clamp somehow, only then sum up.
        log_prob_gaussian = torch.clamp(log_prob_gaussian, -100, 100)
        # Get log-prob for squashed Gaussian.
        return torch.sum(
            log_prob_gaussian - self._log_squash_grad(unsquashed_values),
            dim=-1)

    def _squash(self, unsquashed_values):
        """Squash an array element-wise into the (high, low) range
        Arguments:
            unsquashed_values: values to be squashed
        Returns:
            The squashed values.  The output shape is `unsquashed_values.shape`
        """
        raise NotImplementedError

    def _unsquash(self, values):
        """Unsquash an array element-wise from the (high, low) range
        Arguments:
            squashed_values: values to be unsquashed
        Returns:
            The unsquashed values.  The output shape is `squashed_values.shape`
        """
        raise NotImplementedError

    def _log_squash_grad(self, unsquashed_values):
        """Log gradient of _squash with respect to its argument.
        Arguments:
            squashed_values:  Point at which to measure the gradient.
        Returns:
            The gradient at the given point.  The output shape is
            `squashed_values.shape`.
        """
        raise NotImplementedError


class TorchGaussianSquashedGaussian(_TorchSquashedGaussianBase):
    """A gaussian CDF-squashed Gaussian distribution.
    Can be used instead of the `SquashedGaussian` in case entropy or KL need
    to be computable in analytical form (`SquashedGaussian` can only provide
    those empirically).
    The distribution will never return low or high exactly, but
    `low`+SMALL_NUMBER or `high`-SMALL_NUMBER respectively.
    """
    # Chosen to match the standard logistic variance, so that:
    #   Var(N(0, 2 * _SCALE)) = Var(Logistic(0, 1))
    _SCALE = 0.5 * 1.8137
    SQUASH_DIST = \
        torch.distributions.normal.Normal(0.0, _SCALE) if torch else None

    @override(_TorchSquashedGaussianBase)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale = torch.from_numpy(np.array(self._SCALE))
        if self.model:
            self.scale = self.scale.to(
                next(iter(self.model.parameters())).device)

    @override(ActionDistribution)
    def kl(self, other):
        # KL(self || other) is just the KL of the two unsquashed distributions.
        assert isinstance(other, TorchGaussianSquashedGaussian)

        mean = self.distr.loc
        std = self.distr.scale

        other_mean = other.distr.loc
        other_std = other.distr.scale

        return torch.sum(
            (other.log_std - self.log_std +
             (torch.pow(std, 2.0) + torch.pow(mean - other_mean, 2.0)) /
             (2.0 * torch.pow(other_std, 2.0)) - 0.5),
            axis=1)

    def entropy(self):
        # Entropy is:
        #   -KL(self.distr || N(0, _SCALE)) + log(high - low)
        # where the latter distribution's CDF is used to do the squashing.

        mean = self.distr.loc
        std = self.distr.scale

        return torch.sum(
            torch.log(self.high - self.low) -
            (torch.log(self.scale) - self.log_std +
             (torch.pow(std, 2.0) + torch.pow(mean, 2.0)) /
             (2.0 * torch.pow(self.scale, 2.0)) - 0.5),
            dim=1)

    def _log_squash_grad(self, unsquashed_values):
        log_grad = self.SQUASH_DIST.log_prob(value=unsquashed_values)
        log_grad += torch.log(self.high - self.low)
        return log_grad

    def _squash(self, raw_values):
        # Make sure raw_values are not too high/low (such that tanh would
        # return exactly 1.0/-1.0, which would lead to +/-inf log-probs).

        values = self.SQUASH_DIST.cdf(raw_values)  # / self._SCALE)
        return (torch.clamp(values, SMALL_NUMBER, 1.0 - SMALL_NUMBER) *
                (self.high - self.low) + self.low)

    def _unsquash(self, values):
        x = (values - self.low) / (self.high - self.low)
        return self.SQUASH_DIST.icdf(x)

    @staticmethod
    @override(ActionDistribution)
    def required_model_output_shape(
            action_space: gym.Space,
            model_config: ModelConfigDict) -> Union[int, np.ndarray]:
        return np.prod(action_space.shape) * 2
