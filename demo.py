import time
import copy

import numpy as np
import alfworld.agents.environment as environment
import alfworld.agents.modules.generic as generic

# load config
config = generic.load_config()
env_type = config['env']['type'] # 'AlfredTWEnv' or 'AlfredThorEnv' or 'AlfredHybrid'
print("Environment type: {}".format(env_type))
# setup environment
env = getattr(environment, env_type)(config, train_eval='train')
env = env.init_env(batch_size=1)

# interact
obs, info = env.reset()
print(info)
breakpoint()
dones = [False]
while not dones[0]:
  # get random actions from admissible 'valid' commands (not available for AlfredThorEnv)
  admissible_commands = list(info['admissible_commands']) # note: BUTLER generates commands word-by-word without using admissible_commands
  # print("Admissible commands: {}".format(admissible_commands))

  expert_action = copy.deepcopy(info['expert_plan'][0][0])
  s = time.time()
  obs, scores, dones, info = env.step([expert_action])
  e = time.time()
  print("Sim Time: {:.10f} s, Action: {}, Obs: {}, Done: {}".format(e-s, expert_action, obs[0], dones[0]))