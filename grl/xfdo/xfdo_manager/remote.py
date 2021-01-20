import json
import grpc
import logging
from typing import Tuple, List, Dict, Callable, Union
from concurrent import futures
from google.protobuf.empty_pb2 import Empty

from grl.p2sro.payoff_table import PayoffTable, PayoffTableStrategySpec
from grl.xfdo.xfdo_manager.protobuf.xfdo_manager_pb2_grpc import XFDOManagerServicer, add_XFDOManagerServicer_to_server, \
    XFDOManagerStub
from grl.xfdo.xfdo_manager.protobuf.xfdo_manager_pb2 import XFDOPolicyMetadataRequest, XFDONewBestResponseParams, \
    XFDOConfirmation, XFDOPolicySpecJson, XFDOPlayerAndPolicyNum, XFDOString, XFDOPlayer, XFDOPolicySpecList

from grl.xfdo.xfdo_manager.manager import XFDOManager, SolveRestrictedGame

logger = logging.getLogger(__name__)


class _XFDOMangerServerServicerImpl(XFDOManagerServicer):

    def __init__(self, manager: XFDOManager):
        self._manager = manager

    def GetLogDir(self, request, context):
        return XFDOString(string=self._manager.get_log_dir())

    def ClaimNewActivePolicyForPlayer(self, request: XFDOPlayer, context):
        out = self._manager.claim_new_active_policy_for_player(
            player=request.player)

        metanash_specs_for_players, delegate_specs_for_players, policy_num = out

        assert len(metanash_specs_for_players) == self._manager.n_players()
        assert len(delegate_specs_for_players) == self._manager.n_players()

        if policy_num is None:
            return XFDONewBestResponseParams(policy_num=-1)

        response = XFDONewBestResponseParams(policy_num=policy_num)

        for player, spec_for_player in metanash_specs_for_players.items():
            if spec_for_player is not None:
                response.metanash_specs_for_players.policy_spec_list.append(
                    XFDOPolicySpecJson(policy_spec_json=spec_for_player.to_json()))

        response_delegate_spec_lists_for_other_players = []
        for player, player_delegate_spec_list in delegate_specs_for_players.items():
            player_delegate_json_spec_list = XFDOPolicySpecList()
            player_delegate_json_spec_list.policy_spec_list.extend(
                [XFDOPolicySpecJson(policy_spec_json=spec.to_json())
                for spec in player_delegate_spec_list]
            )
            response_delegate_spec_lists_for_other_players.append(player_delegate_json_spec_list)
        response.delegate_specs_for_players.extend(response_delegate_spec_lists_for_other_players)

        return response

    def SubmitFinalBRPolicy(self, request: XFDOPolicyMetadataRequest, context):
        self._manager.submit_final_br_policy(player=request.player, policy_num=request.policy_num,
                                             metadata_dict=json.loads(request.metadata_json))
        return XFDOConfirmation(result=True)

    def IsPolicyFixed(self, request: XFDOPlayerAndPolicyNum, context):
        is_policy_fixed = self._manager.is_policy_fixed(
            player=request.player,
            policy_num=request.policy_num
        )
        return XFDOConfirmation(result=is_policy_fixed)


class XFDOManagerWithServer(XFDOManager):

    def __init__(self,
                 solve_restricted_game: SolveRestrictedGame,
                 n_players: int = 2,
                 log_dir: str = None,
                 port: int = 4545):

        super(XFDOManagerWithServer, self).__init__(
            solve_restricted_game=solve_restricted_game,
            n_players=n_players,
            log_dir=log_dir)
        self._grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        servicer = _XFDOMangerServerServicerImpl(manager=self)
        add_XFDOManagerServicer_to_server(servicer=servicer, server=self._grpc_server)
        address = f'[::]:{port}'
        self._grpc_server.add_insecure_port(address)
        self._grpc_server.start()  # does not block
        logger.info(f"XFDO Manager gRPC server listening at {address}")

    def wait_for_server_termination(self):
        self._grpc_server.wait_for_termination()

    def stop_server(self):
        self._grpc_server.stop(grace=2)


class RemoteXFDOManagerClient(XFDOManager):

    # noinspection PyMissingConstructor
    def __init__(self, n_players, port=4545, remote_server_host="127.0.0.1"):
        self._stub = XFDOManagerStub(channel=grpc.insecure_channel(target=f"{remote_server_host}:{port}"))
        self._n_players = n_players

    def n_players(self) -> int:
        return self._n_players

    def get_log_dir(self) -> str:
        return self._stub.GetLogDir(Empty()).string

    def claim_new_active_policy_for_player(self, player) -> Union[
        Tuple[Dict[int, PayoffTableStrategySpec], Dict[int, List[PayoffTableStrategySpec]], int],
        Tuple[None, None, None]
    ]:
        request = XFDOPlayer(player=player)
        response: XFDONewBestResponseParams = self._stub.ClaimNewActivePolicyForPlayer(request)

        if response.policy_num == -1:
            return None, None, None

        assert len(response.metanash_specs_for_players.policy_spec_list) in [self.n_players(), 0]
        assert len(response.delegate_specs_for_players) in [self.n_players(), 0]

        metanash_json_specs_for_other_players = [elem.policy_spec_json
                                                 for elem in response.metanash_specs_for_players.policy_spec_list]

        metanash_specs_for_players = {
            player: PayoffTableStrategySpec.from_json(json_spec)
            for player, json_spec in enumerate(metanash_json_specs_for_other_players)
        }

        delegate_json_spec_lists_for_other_players = [
            [elem.policy_spec_json for elem in player_delegate_list.policy_spec_list]
            for player_delegate_list in response.delegate_specs_for_players
        ]
        delegate_specs_for_players = {
            player: [PayoffTableStrategySpec.from_json(json_spec) for json_spec in player_delegate_json_list]
            for player, player_delegate_json_list in enumerate(delegate_json_spec_lists_for_other_players)
        }

        if len(metanash_specs_for_players) == 0:
            metanash_specs_for_players = None

        if len(delegate_specs_for_players) == 0:
            delegate_specs_for_players = None

        return (metanash_specs_for_players,
                delegate_specs_for_players,
                response.policy_num)

    def submit_final_br_policy(self, player, policy_num, metadata_dict):
        try:
            metadata_json = json.dumps(obj=metadata_dict)
        except (TypeError, OverflowError) as json_err:
            raise ValueError(f"metadata_dict must be JSON serializable."
                             f"When attempting to serialize, got this error:\n{json_err}")

        request = XFDOPolicyMetadataRequest(player=player, policy_num=policy_num, metadata_json=metadata_json)
        self._stub.SubmitFinalBRPolicy(request)

    def is_policy_fixed(self, player, policy_num):
        response: XFDOConfirmation = self._stub.IsPolicyFixed(XFDOPlayerAndPolicyNum(player=player, policy_num=policy_num))
        return response.result


