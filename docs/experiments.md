
# Launching Experiments

First see [installation](/docs/install.md) documentation for setting up the dev environment.


Experiment details and hyperparameters are organized in uniquely named `Scenarios`. When launching a learning script, you will generally specify a scenario name as a command-line argument. Experiment scenarios are defined in [grl/rl_apps/scenarios/catalog](/grl/rl_apps/scenarios/catalog).

## NFSP

Launch a single script that trains a best response and average policy for each player.
```shell
# from the repository root
cd grl/rl_apps/nfsp
python general_nfsp.py --scenario <my_scenario_name>
```

Available NFSP scenario names include:
```shell
kuhn_nfsp_dqn
leduc_nfsp_dqn
20_clone_leduc_nfsp_dqn
40_clone_leduc_nfsp_dqn
80_clone_leduc_nfsp_dqn
loss_game_nfsp_10_moves_alpha_2.7
```

## PSRO

PSRO consists of three scripts that are launched on separate terminals:
- The manager script (to track the population / payoff table and launch empirical payoff evaluations)
- Two RL best response learner scripts for each of the 2 players

The manager acts as a server that the best response learners connect to via gRPC.

([tmux](https://github.com/tmux/tmux/wiki) with a [nice configuration](https://github.com/gpakosz/.tmux) is useful for managing and organizing many terminal sessions)
```shell
# from the repository root
cd grl/rl_apps/psro
python general_psro_manager.py --scenario <my_scenario_name>
```
```shell
# in a 2nd terminal
cd grl/rl_apps/psro
python general_psro_br.py --player 0 --scenario <same_scenario_as_manager>
```
```shell
# in a 3rd terminal
cd grl/rl_apps/psro
python general_psro_br.py --player 1 --scenario <same_scenario_as_manager>
``` 
If launching each of these scripts on the same computer, the best response scripts will automatically connect to a manager running the same scenario/seed  on a randomized port defined by the manager in `\tmp\grl_ports.json`. Otherwise, pass the `--help` argument to these scripts to see options for specifying hosts and ports. 

Multiple experiments with the same scenario can be launched on a single host by setting the `GRL_SEED` environment variable to a different integer value for each set of corresponding processes. If unset, `GRL_SEED` defaults to 0.

Available PSRO scenario names include:
```shell
kuhn_psro_dqn
leduc_psro_dqn
20_clone_leduc_psro_dqn
40_clone_leduc_psro_dqn
80_clone_leduc_psro_dqn
loss_game_psro_10_moves_alpha_2.7
loss_game_psro_10_moves_multi_dim_max_move_0.1_16_dim
```

## NXDO

Like PSRO, NXDO consists of three scripts that are launched on separate terminals:
- The manager script (to track the population and train the extensive form metanash)
- Two RL best response learner scripts for each of the 2 players

The manager acts as a server that the best response learners connect to via gRPC.

```shell
# from the repository root
cd grl/rl_apps/nxdo
python general_nxdo_manager.py --scenario <my_scenario_name>
```
```shell
# in a 2nd terminal
cd grl/rl_apps/nxdo
python general_nxdo_br.py --player 0 --scenario <same_scenario_as_manager>
```
```shell
# in a 3rd terminal
cd grl/rl_apps/nxdo
python general_nxdo_br.py --player 1 --scenario <same_scenario_as_manager>
``` 
If launching each of these scripts on the same computer, the best response scripts will automatically connect to a manager running the same scenario/seed  on a randomized port defined by the manager in `\tmp\grl_ports.json`. Otherwise, pass the `--help` argument to these scripts to see options for specifying hosts and ports. 

Multiple experiments with the same scenario can be launched on a single host by setting the `GRL_SEED` environment variable to a different integer value for each set of corresponding processes. If unset, `GRL_SEED` defaults to 0.


Available NXDO scenario names include:
```shell
kuhn_nxdo_dqn_nfsp
leduc_nxdo_dqn_nfsp
20_clone_leduc_nxdo_dqn_nfsp_dynamic_threshold_1_aggressive
40_clone_leduc_nxdo_dqn_nfsp_dynamic_threshold_1_aggressive
80_clone_leduc_nxdo_dqn_nfsp_dynamic_threshold_1_aggressive
va_20_clone_leduc_nxdo_dqn_nfsp_dynamic_threshold_1_aggressive
va_40_clone_leduc_nxdo_dqn_nfsp_dynamic_threshold_1_aggressive
va_80_clone_leduc_nxdo_dqn_nfsp_dynamic_threshold_1_aggressive
loss_game_nxdo_10_moves_alpha_2.7
loss_game_nxdo_10_moves_multi_dim_max_move_0.1_16_dim
```


# Graphing Results

When running each algorithm, the log file path containing timesteps, episodes, and exploitability data, when printed, will be highlighted in green with a note, `(Graph this in a notebook)`.

Ray RLlib logs with the learning stats displayed throughout the learning process will also be produced in csv, json, and tensorboard in the same directories in the `<repo_root>/grl/data` directory.

Check [graph_poker_results.ipynb](/examples/graph_poker_results.ipynb) for an example notebook graphing m-clone poker results for NXDO, PSRO, and NFSP.

# Running PSRO/NXDO Experiments in a Single Shell

Example scripts for running PSRO and NXDO experiments in a single shell can be found in the [examples directory](/examples).

PSRO example:
```shell
# in <repo root>/examples
python launch_psro_as_single_script.py --scenario kuhn_psro_dqn
```

NXDO example:
```shell
# in <repo root>/examples
python launch_nxdo_as_single_script.py --scenario 1_step_kuhn_nxdo_dqn_nfsp
```



