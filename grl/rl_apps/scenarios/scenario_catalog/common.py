import os
import urllib.parse

GRL_SEED = int(os.getenv("GRL_SEED", 0))
_RAY_ADDRESS = os.getenv("RAY_ADDRESS")
_CREATE_RAY_HEAD = _RAY_ADDRESS is not None and len(_RAY_ADDRESS) > 0


def ray_port_with_default_and_seed(default_port: int) -> int:
    if _RAY_ADDRESS:
        return urllib.parse.urlsplit('//' + _RAY_ADDRESS).port
    return default_port + GRL_SEED


def default_if_creating_ray_head(default):
    if _CREATE_RAY_HEAD:
        return None
    return default
