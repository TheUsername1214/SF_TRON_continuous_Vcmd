from PPO_Isaac_continuous_Vcmd import *
from Config import *

cfg = RobotConfig()

ppo_agent = PPO(cfg , True)
ppo_agent.play()
