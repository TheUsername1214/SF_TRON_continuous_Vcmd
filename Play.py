
from PPO_Isaac_continuous_Vcmd import *
from Config import *


cfg = RobotConfig()

ppo_agent = PPO(cfg , False)
ppo_agent.load_model()
ppo_agent.play()


