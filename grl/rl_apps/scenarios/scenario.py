from abc import ABC, abstractmethod
from typing import Union, Type
from copy import deepcopy

class Scenario(ABC):

    @abstractmethod
    def __init__(self, name: str):
        self.name = name
        self.validate_properties()

    def validate_properties(self):
        # validate name
        for c in self.name:
            if c.isspace() or c in ["\\", "/"] or not c.isprintable():
                raise ValueError("Scenario names must not contain whitespace, '\\', or '/'. "
                                 "It needs to be usable as a directory name. "
                                 f"Yours was '{self.name}'.")

    def copy(self):
        return deepcopy(self)

class RayScenario(Scenario, ABC):

    @abstractmethod
    def __init__(self,
                 name: str,
                 ray_cluster_cpus: Union[int, float],
                 ray_cluster_gpus: Union[int, float],
                 ray_object_store_memory_cap_gigabytes: Union[int, float]):
        super().__init__(name=name)
        self.ray_cluster_cpus = ray_cluster_cpus
        self.ray_cluster_gpus = ray_cluster_gpus
        self.ray_object_store_memory_cap_gigabytes = ray_object_store_memory_cap_gigabytes