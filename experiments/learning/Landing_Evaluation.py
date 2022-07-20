"""Test script for single agent problems.

This scripts runs the best model found by one of the executions of `singleagent.py`

Example
-------
To run the script, type in a terminal:

    $ python test_singleagent.py --exp ./results/save-<env>-<algo>-<obs>-<act>-<time_date>

"""
from distutils.log import INFO
import os
import time
import matplotlib.pyplot as plt
import math
from datetime import datetime
import argparse

import numpy as np
import gym

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.cmd_util import make_vec_env 

from stable_baselines3 import PPO
from stable_baselines3 import SAC


from stable_baselines3.common.policies import ActorCriticPolicy as a2cppoMlpPolicy
from stable_baselines3.common.policies import ActorCriticCnnPolicy as a2cppoCnnPolicy
from stable_baselines3.sac.policies import SACPolicy as sacMlpPolicy


from gym_pybullet_drones.utils.utils import sync
from gym_pybullet_drones.utils.Logger import Logger

from gym_pybullet_drones.envs.single_agent_rl.LandingAviary import LandingAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.control.LandingEvaulation import evaluate_landing_policy 
import shared_constants

if __name__ == "__main__":

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script using TakeoffAviary')
    parser.add_argument('--exp',                           type=str,            help='The experiment folder written as ./results/save-<env>-<algo>-<obs>-<act>-<time_date>', metavar='')
    ARGS = parser.parse_args()

    #### Load the model from file ##############################
    algo = ARGS.exp.split("-")[2]
    
    if os.path.isfile(ARGS.exp+'/success_model.zip'):
        path = ARGS.exp+'/success_model.zip'
    elif os.path.isfile(ARGS.exp+'/best_model.zip'):
        path = ARGS.exp+'/best_model.zip'
    else:
        print("[ERROR]: no model under the specified path", ARGS.exp)
    if algo == 'ppo':
        model = PPO.load(path)
       
    if algo == 'sac':
        model = SAC.load(path)


    #### Parameters to recreate the environment ################
    env_name = ARGS.exp.split("-")[1]+"-aviary-v0"
    OBS = ObservationType.KIN 
    ACT = ActionType.LD
    #### Evaluate the model ####################################
    eval_env = gym.make(env_name,
                        aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                        obs=OBS,
                        act=ACT
                        )
    
    mean_reward, std_reward = evaluate_landing_policy(model,
                                              eval_env,
                                              n_eval_episodes=1
                                              )
    print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")
    










    test_env= LandingAviary(
                         gui=False,
                         record=False,
                         
                         aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS
                         
                         )
    
    logger = Logger(logging_freq_hz=int(test_env.SIM_FREQ/test_env.AGGR_PHY_STEPS),
                     num_drones=1
                    )
    test_steps=400
    test_iteration=100
    for i in range(test_iteration):
        obs = test_env.reset()
        start = time.time()
        # new log try ####

        actions = np.zeros(
            shape=(test_env.action_space.shape[0], test_steps), dtype=np.float32)
        observation = np.zeros(
            shape=(test_env.observation_space.shape[0], test_steps), dtype=np.float32)
        rewards = np.zeros(
            shape=(1, test_steps), dtype=np.float32)
        time_plt = np.zeros(
            shape=(1, test_steps), dtype=np.float32)
        infos = np.zeros(
            shape=(8, test_steps), dtype=np.float32)
        for i in range(test_steps):
            
            action, _states = model.predict(obs,
                                            deterministic=True # OPTIONAL 'deterministic=False'
                                            )
            obs, reward, done, info = test_env.step(action)
            # test_env.render()
            actions[:,i]=action
            observation[:,i]=obs
            rewards[:,i]=reward
            infos[:,i]=info
            time_plt[:,i]=i*test_env.AGGR_PHY_STEPS/240
            


            
        test_env.close()



    # with np.load(ARGS.exp+'/evaluations.npz') as data:
    #     print(data.files)
    #     print(data['timesteps'])
    #     print(data['results'])
    #     print(data['ep_lengths'])
    
   





 