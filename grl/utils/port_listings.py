import json
import socket
import time
import os
from contextlib import closing
from typing import Dict, Union

from filelock import FileLock

PORT_LISTING_PATH = "/tmp/grl_ports.json"


def _find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def establish_new_server_port_for_service(service_name: str) -> int:
    with FileLock(f"{PORT_LISTING_PATH}.lock"):
        file_mode = "r+" if os.path.exists(PORT_LISTING_PATH) else "w+"
        with open(PORT_LISTING_PATH, file_mode) as port_listing_file:
            retries = 10
            for retry in range(retries):
                file_is_empty = os.path.getsize(PORT_LISTING_PATH) == 0
                port_listings: Dict[str, str] = {}
                if not file_is_empty:
                    port_listings: Dict[str, str] = json.load(port_listing_file)
                claimed_ports_from_file = list(int(p) for p in port_listings.values())
                free_port_from_os = _find_free_port()
                if (free_port_from_os in claimed_ports_from_file
                        and port_listings.get(service_name) != free_port_from_os):
                    time.sleep(0.05)
                    continue
                port_listings[service_name] = free_port_from_os
                port_listing_file.seek(0)
                json.dump(port_listings, port_listing_file)
                port_listing_file.truncate()
                return free_port_from_os
            raise ConnectionError(f"Couldn't find an unclaimed free port for service name {service_name} "
                                  f"in {retries} retries.")


def get_client_port_for_service(service_name: str, raise_if_not_found=True) -> Union[None, int]:
    with FileLock(f"{PORT_LISTING_PATH}.lock"):
        with open(PORT_LISTING_PATH, "r+") as port_listing_file:
            port_listings: Dict[str, str] = json.load(port_listing_file)
            port_for_service = port_listings.get(service_name)

            if raise_if_not_found and port_for_service is None:
                raise ConnectionError(f"Couldn't find a list port for service name {service_name}.\n"
                                      f"Port listings are:\n{port_listings}")

            return int(port_for_service) if port_for_service is not None else None
