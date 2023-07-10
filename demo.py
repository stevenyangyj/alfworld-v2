import time

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
obs, info = env.batch_reset()
for i in range(5):
  # get random actions from admissible 'valid' commands (not available for AlfredThorEnv)
  admissible_commands = list(info['admissible_commands']) # note: BUTLER generates commands word-by-word without using admissible_commands
  # print("Admissible commands: {}".format(admissible_commands))
  random_actions = [np.random.choice(admissible_commands[0])]

  # step
  s = time.time()
  obs, scores, dones, infos = env.alf_step(random_actions)
  e = time.time()
  print("Sim Time: {:.10f} s, Action: {}, Obs: {}, Rew: {}".format(e-s, random_actions[0], obs[0], scores[0]))