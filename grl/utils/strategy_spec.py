import json

from grl.utils.common import check_if_jsonable


class StrategySpec(object):

    def __init__(self, strategy_id: str, metadata: dict, pure_strategy_indexes=None):

        if metadata is None:
            metadata = {}

        if not isinstance(strategy_id, str):
            raise ValueError("strategy_id is not of type string.")

        is_jsonable, json_err = check_if_jsonable(check_dict=metadata)
        if not is_jsonable:
            raise ValueError(f"Metadata is not JSON serializable. "
                             f"All metadata values need to be JSON serializable.\n"
                             f"JSON error is:\n{json_err}")
        if pure_strategy_indexes is not None:
            for player, index in pure_strategy_indexes.items():
                if not isinstance(player, int) and player >= 0:
                    raise ValueError("Keys in pure_strategy_indexes must be player numbers (ints) >= 0")
                if not isinstance(index, int) and index >= 0:
                    raise ValueError("Values in pure_strategy_indexes must be strategy indexes (ints) >= 0")

        self.id = strategy_id
        self.metadata = metadata
        self._pure_strategy_indexes = pure_strategy_indexes if pure_strategy_indexes is not None else {}

    def assign_pure_strat_index(self, player, pure_strat_index):
        if self._pure_strategy_indexes.get(player) is not None:
            raise ValueError(
                "Cannot assign a new pure strategy index when one is already assigned for the same player.")
        self._pure_strategy_indexes[player] = pure_strat_index

    def pure_strat_index_for_player(self, player):
        return self._pure_strategy_indexes[player]

    def get_pure_strat_indexes(self):
        return self._pure_strategy_indexes.copy()

    def update_metadata(self, new_metadata):
        if new_metadata is not None:
            self.metadata.update(new_metadata)

    def serialize_to_dict(self):
        return {
            "id": self.id,
            "metadata": self.metadata,
            "pure_strategy_indexes": self.get_pure_strat_indexes()
        }

    def to_json(self):
        return json.dumps(self.serialize_to_dict())

    @classmethod
    def from_dict(cls, serialized_dict: dict):
        spec = StrategySpec(strategy_id=serialized_dict["id"],
                            metadata=serialized_dict["metadata"])
        for player, index in serialized_dict["pure_strategy_indexes"].items():
            spec.assign_pure_strat_index(player=int(player), pure_strat_index=int(index))
        return spec

    @classmethod
    def from_json(cls, json_string: str):
        return StrategySpec.from_dict(json.loads(s=json_string))

    @classmethod
    def from_json_file(cls, json_file_path: str):
        with open(json_file_path, "r") as json_file:
            spec_json = json.load(json_file)
            return StrategySpec.from_dict(spec_json)

    def __eq__(self, other):
        if not isinstance(other, StrategySpec):
            return False
        return self.to_json() == other.to_json()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return self.to_json().__hash__()
