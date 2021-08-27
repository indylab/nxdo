from importlib import import_module
from pathlib import Path

from grl.rl_apps.scenarios.scenario import Scenario


class _ScenarioCatalog:

    def __init__(self):
        self._catalog = {}

    def add(self, scenario: Scenario):
        if scenario.name in self._catalog:
            raise ValueError(f"Scenario name '{scenario.name}' is already present in the scenario catalog. "
                             f"Scenario names need to be unique.")

        self._catalog[scenario.name] = scenario

    def get(self, scenario_name: str) -> Scenario:
        try:
            return self._catalog[scenario_name]
        except KeyError:
            available_scenarios = '\n'.join(name for name in self._catalog.keys())
            raise ValueError(f"Scenario name {scenario_name} not found. "
                             f"Available scenarios are:\n{available_scenarios}")

    def list(self):
        return list(self._catalog.keys())


scenario_catalog = _ScenarioCatalog()
__all__ = ['scenario_catalog']

# add all modules in this directory to __all__
__all__.extend([
    import_module(f".{f.stem}", __package__)
    for f in Path(__file__).parent.glob("*.py")
    if "__" not in f.stem
])
