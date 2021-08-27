import logging
import os
from typing import Union

import ray

from grl.rl_apps.scenarios.scenario import RayScenario
from grl.utils.common import find_free_port


def init_ray_for_scenario(scenario: RayScenario,
                          head_address: str = None,
                          logging_level=logging.INFO) -> Union[str, None]:
    if head_address is not None or scenario.ray_object_store_memory_cap_gigabytes is None:
        object_store_memory_bytes = None
    else:
        object_store_memory_bytes = int(1073741824 * scenario.ray_object_store_memory_cap_gigabytes)

    dashboard_port = os.getenv("RAY_DASHBOARD_PORT", None)
    dashboard_port = int(dashboard_port) if dashboard_port is not None else None

    include_dashboard = bool(os.getenv("RAY_INCLUDE_DASHBOARD", False))
    if include_dashboard and not dashboard_port:
        dashboard_port = find_free_port()

    address_info = ray.init(address=head_address,
                            num_cpus=scenario.ray_cluster_cpus if head_address is None else None,
                            num_gpus=scenario.ray_cluster_gpus if head_address is None else None,
                            object_store_memory=object_store_memory_bytes,
                            _lru_evict=bool(head_address is None),
                            local_mode=False,
                            include_dashboard=include_dashboard,
                            dashboard_host=os.getenv("RAY_DASHBOARD_HOST", None),
                            dashboard_port=dashboard_port,
                            ignore_reinit_error=bool(head_address is not None),
                            logging_level=logging_level,
                            log_to_driver=os.getenv("RAY_LOG_TO_DRIVER", False))

    os_ray_address = os.getenv("RAY_ADDRESS")

    if head_address or (os_ray_address is not None and len(os_ray_address) > 0):
        print(f"Connected to EXISTING Ray cluster at {head_address} ")
        return None
    else:
        print(f"Created NEW Ray cluster at {address_info['redis_address']} with\n"
              f"CPU: {scenario.ray_cluster_cpus}\n"
              f"GPU: {scenario.ray_cluster_gpus}")
        return address_info["redis_address"]
