from grl.rl_apps.scenarios.scenario import Scenario


class _ScenarioCatalog:

    def __int__(self):
        self._catalog = {}

    def add(self, scenario: Scenario):
        self._catalog[scenario.name] = scenario

    def get(self, scenario_name: str):
        try:
            return self._catalog[scenario_name]
        except KeyError:
            raise KeyError(f"Scenario name {scenario_name} not found.\n"
                           f"Available scenarios are:\n"
                           f"\n".join(name for name in self._catalog.keys()))


scenario_catalog = _ScenarioCatalog()
