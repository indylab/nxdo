import logging
import os
import grl
from grl.p2sro.p2sro_manager import P2SROManagerWithServer, SimpleP2SROManagerLogger
from grl.utils import datetime_str


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    manager = P2SROManagerWithServer(
        port=os.getenv("P2SRO_PORT", 4535),
        eval_dispatcher_port=os.getenv("EVAL_PORT", 4536),
        n_players=2,
        is_two_player_symmetric_zero_sum=True,
        do_external_payoff_evals_for_new_fixed_policies=True,
        games_per_external_payoff_eval=500,
        payoff_table_exponential_average_coeff=0.005,
        log_dir=os.path.join(os.path.dirname(grl.__file__), "data", "simple_push_psro", f"manager_{datetime_str()}")
    )
    print(f"Launched P2SRO Manager with server.")
    manager.wait_for_server_termination()
