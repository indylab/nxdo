import os

_RAY_ADDRESS = os.getenv("RAY_ADDRESS")
_EXISTING_RAY_ADDRESS_PROVIDED = _RAY_ADDRESS is not None and len(_RAY_ADDRESS) > 0


def default_if_creating_ray_head(default):
    if _EXISTING_RAY_ADDRESS_PROVIDED:
        return None
    return default
