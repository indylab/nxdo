import grpc
import logging
from typing import Tuple, List, Union
from concurrent import futures
from google.protobuf.empty_pb2 import Empty

from grl.p2sro.payoff_table import PayoffTableStrategySpec
from grl.p2sro.eval_dispatcher.protobuf.eval_dispatcher_pb2_grpc import EvalDispatcherServicer, \
    add_EvalDispatcherServicer_to_server, EvalDispatcherStub
from grl.p2sro.eval_dispatcher.protobuf.eval_dispatcher_pb2 import EvalJob, EvalJobResult, EvalConfirmation

from grl.p2sro.eval_dispatcher.eval_dispatcher import EvalDispatcher

logger = logging.getLogger(__name__)


class _EvalDispatcherServerServicerImpl(EvalDispatcherServicer):

    def __init__(self, eval_dispatcher: EvalDispatcher):
        self._eval_dispatcher = eval_dispatcher

    def TakeEvalJob(self, request, context):
        policy_specs_for_each_player, games_to_play = self._eval_dispatcher.take_eval_job()
        response = EvalJob(required_games_to_play=games_to_play)
        if policy_specs_for_each_player is not None:
            response.json_policy_specs_for_each_player.extend(spec.to_json() for spec in policy_specs_for_each_player)
        return response

    def SubmitEvalJobResult(self, request: EvalJobResult, context):
        policy_specs_for_each_player = tuple(PayoffTableStrategySpec.from_json(json_string=json_string)
                                             for json_string in request.json_policy_specs_for_each_player)
        self._eval_dispatcher.submit_eval_job_result(policy_specs_for_each_player_tuple=policy_specs_for_each_player,
                                                     payoffs_for_each_player=request.payoffs_for_each_player,
                                                     games_played=request.games_played)
        return EvalConfirmation(result=True)


class EvalDispatcherWithServer(EvalDispatcher):
    """
    A EvalDispatcher with an attached GRPC server to allow clients on other processes or computers to make calls to the
    EvalDispatcher's methods. Interacting with the EvalDispatcher as a local instance while the server is handling requests
    is thread safe.
    """

    def __init__(self,
                 games_per_eval: int,
                 game_is_two_player_symmetric: bool,
                 drop_duplicate_requests: bool,
                 port: int = 4536):
        super(EvalDispatcherWithServer, self).__init__(
            games_per_eval=games_per_eval,
            game_is_two_player_symmetric=game_is_two_player_symmetric,
            drop_duplicate_requests=drop_duplicate_requests)
        self._grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        servicer = _EvalDispatcherServerServicerImpl(eval_dispatcher=self)
        add_EvalDispatcherServicer_to_server(servicer=servicer, server=self._grpc_server)
        self._grpc_server.add_insecure_port(f'[::]:{port}')
        self._grpc_server.start()  # does not block

    def wait_for_server_termination(self):
        self._grpc_server.wait_for_termination()

    def stop_server(self):
        self._grpc_server.stop(grace=2)


class RemoteEvalDispatcherClient(EvalDispatcher):
    """
    GRPC client for a EvalDispatcher server.
    Behaves like a local P2SROManager but actually is connecting to a remote P2SRO Manager on another
    process or computer.
    Limited methods are available.
    """

    # noinspection PyMissingConstructor
    def __init__(self, port=4536, remote_server_host="127.0.0.1"):
        self._stub = EvalDispatcherStub(channel=grpc.insecure_channel(target=f"{remote_server_host}:{port}"))

    def take_eval_job(self) -> (Union[None, Tuple[PayoffTableStrategySpec]], int):
        response: EvalJob = self._stub.TakeEvalJob(Empty())
        policy_specs_for_each_player = tuple(PayoffTableStrategySpec.from_json(json_string=json_string)
                                             for json_string in response.json_policy_specs_for_each_player)
        if len(policy_specs_for_each_player) == 0:
            return None, response.required_games_to_play
        return policy_specs_for_each_player, response.required_games_to_play

    def submit_eval_job_result(self, policy_specs_for_each_player_tuple, payoffs_for_each_player: List[float],
                               games_played):
        request = EvalJobResult(games_played=games_played)
        request.json_policy_specs_for_each_player.extend(spec.to_json() for spec in policy_specs_for_each_player_tuple)
        request.payoffs_for_each_player.extend(payoffs_for_each_player)
        self._stub.SubmitEvalJobResult(request)

    def submit_eval_request(self, policy_specs_for_each_player: Tuple[PayoffTableStrategySpec]):
        raise NotImplemented

    def get_unclaimed_eval_results(self):
        raise NotImplemented
