
# Launching Experiments

Experiment details and hyperparameters are organized in uniquely named `Scenarios`. When launching a learning script, you will generally specify a scenario name as a command-line argument. Experiment scenarios are defined in [grl/rl_apps/scenarios/catalog](/grl/rl_apps/scenarios/catalog).

## NFSP

Launch a single script that trains a best response and average policy for each player.
```shell
# from the repository root
cd grl/rl_apps/nfsp
python general_nfsp.py --scenario <my_scenario_name>
```
NFSP scenarios are listed in [grl/rl_apps/scenarios/catalog/poker_nfsp_scenarios.py](/grl/rl_apps/scenarios/catalog/poker_nfsp_scenarios.py).
Available scenario names include:
```shell
kuhn_nfsp_dqn
leduc_nfsp_dqn
20_clone_leduc_nfsp_dqn
40_clone_leduc_nfsp_dqn
80_clone_leduc_nfsp_dqn
```

## PSRO

PSRO consists of three scripts that are launched in order on separate terminals:
- The manager script (to track the population / payoff table and launch empirical payoff evaluations)
- Two RL best response learner scripts for each of the 2 players

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
If launching each of these on the same computer, the best response scripts will automatically connect to a manager running the same scenario  on a randomized port. Otherwise, pass the `--help` argument to these scripts to see options for specifying hosts and ports. 

PSRO scenarios are listed in [grl/rl_apps/scenarios/catalog/poker_psro_scenarios.py](/grl/rl_apps/scenarios/catalog/poker_psro_scenarios.py).
Available scenario names include:
```shell
kuhn_psro_dqn
leduc_psro_dqn
20_clone_leduc_psro_dqn
40_clone_leduc_psro_dqn
80_clone_leduc_psro_dqn
```

## NXDO

Like PSRO, NXDO consists of three scripts that are launched in order on separate terminals:
- The manager script (to track the population and train the restricted game metanash)
- Two RL best response learner scripts for each of the 2 players

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
If launching each of these on the same computer, the best response scripts will automatically connect to a manager running the same scenario  on a randomized port. Otherwise, pass the `--help` argument to these scripts to see options for specifying hosts and ports. 

NXDO scenarios are listed in [grl/rl_apps/scenarios/catalog/poker_nxdo_scenarios.py](/grl/rl_apps/scenarios/catalog/poker_nxdo_scenarios.py).
Available scenario names include:
```shell
kuhn_nxdo_dqn_nfsp
leduc_nxdo_dqn_nfsp
20_clone_leduc_nxdo_dqn_nfsp_dynamic_threshold_1_aggressive
40_clone_leduc_nxdo_dqn_nfsp_dynamic_threshold_1_aggressive
80_clone_leduc_nxdo_dqn_nfsp_dynamic_threshold_1_aggressive
va_20_clone_leduc_nxdo_dqn_nfsp_dynamic_threshold_1_aggressive
va_40_clone_leduc_nxdo_dqn_nfsp_dynamic_threshold_1_aggressive
va_80_clone_leduc_nxdo_dqn_nfsp_dynamic_threshold_1_aggressive
```


# Graphing Results

Check [graph_poker_results.ipynb](/examples/graph_poker_results.ipynb) for an example notebook graphing m-clone poker results for NXDO, PSRO, and NFSP. You can add additional logs from your own experiments to it. In each algorithm, the log file path containing timesteps, episodes, and exploitability data, when printed, will be highlighted in green with a note, `(Graph this in a notebook)`.