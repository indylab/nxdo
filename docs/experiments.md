
## m-clone leduc

####NXDO
```shell
# from the repo root
cd grl/rl_apps/xfdo
python general_xfdo_manager.py --scenario 20_clone_leduc_xfdo_dqn_nfsp_dynamic_threshold_1_aggressive

# in a 2nd terminal
python general_xfdo_br.py --player 0 --scenario 20_clone_leduc_xfdo_dqn_nfsp_dynamic_threshold_1_aggressive

# in a 3rd terminal
python general_xfdo_br.py --player 1 --scenario 20_clone_leduc_xfdo_dqn_nfsp_dynamic_threshold_1_aggressive
```