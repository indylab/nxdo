import ray


@ray.remote(num_cpus=0)
class RemoteLinearParameterScheduler(object):
    def __init__(self, start_val: float, end_val: float, timesteps_annealing: int):
        self._start_val = start_val
        self._end_val = end_val

        assert timesteps_annealing >= 0, timesteps_annealing
        self._timesteps_annealing = timesteps_annealing

        self._curr_val = self._start_val

    def _calculate_new_val(self, timesteps: float):
        assert timesteps >= 0, timesteps
        if timesteps >= self._timesteps_annealing:
            self._curr_val = self._end_val
        else:
            fraction_done = timesteps/self._timesteps_annealing
            assert 0.0 - 1e-9 <= fraction_done <= 1.0 + 1e-9
            self._curr_val = self._start_val * (1.0 - fraction_done) + self._end_val * fraction_done

    def update_value(self, timesteps: int):
        self._calculate_new_val(timesteps=timesteps)

    def get_value(self):
        return self._curr_val

