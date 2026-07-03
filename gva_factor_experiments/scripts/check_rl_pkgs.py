import importlib.util
for m in ['stable_baselines3', 'sb3_contrib', 'gymnasium', 'torch']:
    spec = importlib.util.find_spec(m)
    print(m, bool(spec))
if importlib.util.find_spec('stable_baselines3'):
    import stable_baselines3 as sb3
    print('sb3_version', sb3.__version__)
    print('has_algos', [x for x in ['A2C', 'DQN', 'PPO'] if hasattr(sb3, x)])
if importlib.util.find_spec('sb3_contrib'):
    import sb3_contrib
    print('sb3_contrib_version', sb3_contrib.__version__)
    print('contrib_has', [x for x in ['MaskablePPO', 'QRDQN', 'RecurrentPPO'] if hasattr(sb3_contrib, x)])
