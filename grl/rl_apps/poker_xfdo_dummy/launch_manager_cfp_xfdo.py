import os
import logging

import grl
from grl.xfdo.xfdo_manager.remote import XFDOManagerWithServer
from grl.rl_apps.poker_xfdo_dummy.solve_restricted_game_cfp import solve_restricted_game_cfp
from grl.utils import datetime_str

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    manager = XFDOManagerWithServer(
        solve_restricted_game_fn=solve_restricted_game_cfp,
        n_players=2,
        log_dir=os.path.join(os.path.dirname(grl.__file__), "data", "xfdo_cfp", f"manager_{datetime_str()}"),
        port=os.getenv("XFDO_PORT", 4546))

    manager.wait_for_server_termination()