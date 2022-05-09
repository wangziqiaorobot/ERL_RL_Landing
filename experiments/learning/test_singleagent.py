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
import re
import numpy as np
import gym
import torch
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.cmd_util import make_vec_env 
from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import TD3
from stable_baselines3 import DDPG
from stable_baselines3.common.policies import ActorCriticPolicy as a2cppoMlpPolicy
from stable_baselines3.common.policies import ActorCriticCnnPolicy as a2cppoCnnPolicy
from stable_baselines3.sac.policies import SACPolicy as sacMlpPolicy
from stable_baselines3.sac import CnnPolicy as sacCnnPolicy
from stable_baselines3.td3 import MlpPolicy as td3ddpgMlpPolicy
from stable_baselines3.td3 import CnnPolicy as td3ddpgCnnPolicy
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.utils.utils import sync
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.single_agent_rl.TakeoffAviary import TakeoffAviary
from gym_pybullet_drones.envs.single_agent_rl.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.single_agent_rl.FlyThruGateAviary import FlyThruGateAviary
from gym_pybullet_drones.envs.single_agent_rl.TuneAviary import TuneAviary
from gym_pybullet_drones.envs.single_agent_rl.LandingAviary import LandingAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType

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
    if algo == 'a2c':
        model = A2C.load(path)
    if algo == 'ppo':
        model = PPO.load(path)
    if algo == 'sac':
        model = SAC.load(path)
    if algo == 'td3':
        model = TD3.load(path)
    if algo == 'ddpg':
        model = DDPG.load(path)

    #### Parameters to recreate the environment ################
    env_name = ARGS.exp.split("-")[1]+"-aviary-v0"
    OBS = ObservationType.KIN if ARGS.exp.split("-")[3] == 'kin' else ObservationType.RGB
    if ARGS.exp.split("-")[4] == 'rpm':
        ACT = ActionType.RPM
    elif ARGS.exp.split("-")[4] == 'dyn':
        ACT = ActionType.DYN
    elif ARGS.exp.split("-")[4] == 'pid':
        ACT = ActionType.PID
    elif ARGS.exp.split("-")[4] == 'vel':
        ACT = ActionType.VEL
    elif ARGS.exp.split("-")[4] == 'tun':
        ACT = ActionType.TUN
    elif ARGS.exp.split("-")[4] == 'one_d_rpm':
        ACT = ActionType.ONE_D_RPM
    elif ARGS.exp.split("-")[4] == 'one_d_dyn':
        ACT = ActionType.ONE_D_DYN
    elif ARGS.exp.split("-")[4] == 'one_d_pid':
        ACT = ActionType.ONE_D_PID
    elif ARGS.exp.split("-")[4] == 'ld':
        ACT = ActionType.LD
    #### Evaluate the model ####################################
    eval_env = gym.make(env_name,
                        aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                        obs=OBS,
                        act=ACT
                        )
    # mean_reward, std_reward = evaluate_policy(model,
    #                                           eval_env,
    #                                           n_eval_episodes=1
    #                                           )
    # print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")
    
    #### Show, record a video, and log the model's performance #
    # test_env = gym.make(env_name,
    #                     gui=True,
    #                     record=True,
    #                     aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
    #                     obs=OBS,
    #                     act=ACT
    #                     )
    test_env= LandingAviary(
                         gui=True,
                         record=True,
                         
                         aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS
                         
                         )
    
    logger = Logger(logging_freq_hz=int(test_env.SIM_FREQ/test_env.AGGR_PHY_STEPS),
                     num_drones=1
                    )
    obs = test_env.reset()
    start = time.time()
    
    # for i in range(3*int(test_env.SIM_FREQ/test_env.AGGR_PHY_STEPS)): # Up to 6''
    #     # print('test',test_env.SIM_FREQ/test_env.AGGR_PHY_STEPS,'#####################################')
    #     action, _states = model.predict(obs,
    #                                     deterministic=True # OPTIONAL 'deterministic=False'
    #                                     )
    #     obs, reward, done, info = test_env.step(action)
    #     test_env.render()
    #     if OBS==ObservationType.KIN:
    #         logger.log(drone=0,
    #                    timestamp=i/test_env.SIM_FREQ,
    #                    state= np.hstack([obs[0:3], np.zeros(4), obs[3:15],  np.resize(action, (4))]),
    #                    control=np.zeros(12)
    #                    )
    #     sync(np.floor(i*test_env.AGGR_PHY_STEPS), start, test_env.TIMESTEP)
    #     # if done: obs = test_env.reset() # OPTIONAL EPISODE HALT
    # test_env.close()
    # logger.save_as_csv("sa") # Optional CSV save
    # logger.plot()

    # new log try ####
    test_steps=500
    actions = np.zeros(
        shape=(test_env.action_space.shape[0], test_steps), dtype=np.float32)
    observation = np.zeros(
        shape=(test_env.observation_space.shape[0], test_steps), dtype=np.float32)
    rewards = np.zeros(
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
        sync(np.floor(i*test_env.AGGR_PHY_STEPS), start, test_env.TIMESTEP)
    test_env.close()



    # with np.load(ARGS.exp+'/evaluations.npz') as data:
    #     print(data.files)
    #     print(data['timesteps'])
    #     print(data['results'])
    #     print(data['ep_lengths'])
    
    ############### Plot the states & actions
    save_path = os.path.join(ARGS.exp)
    plt.figure()
    plt.plot(test_env.MAX_THRUST/2*(actions[0,:]+1),label="thrust")
    #plt.plot((actions[0,:]),label="b")
    plt.grid()
    plt.legend()
    plt.title('action0_thrust')
    plt.savefig(save_path + '/action0_thrust.jpg')
    
    plt.figure()
    # plt.plot(test_env.MAX_THRUST/2*(actions[1,:]*0.05+1),label="roll")
    plt.plot((actions[1,:]*test_env.MAX_ROLL_PITCH/3/math.pi*180),label="roll")
    plt.grid()
    plt.legend()
    plt.title('action1_roll')
    plt.savefig(save_path + '/action1_roll.jpg')

    plt.figure()
    # plt.plot(test_env.MAX_THRUST/2*(actions[1,:]*0.05+1),label="roll")
    plt.plot((actions[2,:]*test_env.MAX_ROLL_PITCH/3/math.pi*180),label="pitch")
    plt.grid()
    plt.legend()
    plt.title('action2_pitch')
    plt.savefig(save_path + '/action2_pitch.jpg')

    plt.figure()
    # plt.plot(test_env.MAX_THRUST/2*(actions[1,:]*0.05+1),label="roll")
    plt.plot((actions[3,:]*test_env.MAX_ROLL_PITCH/5/math.pi*180*0.02),label="yaw")
    plt.grid()
    plt.legend()
    plt.title('action3_yaw')
    plt.savefig(save_path + '/action3_yaw.jpg')

    ## observation
    MAX_LIN_VEL_XY = 3 
    MAX_LIN_VEL_Z = 1

    MAX_XY = MAX_LIN_VEL_XY*test_env.EPISODE_LEN_SEC
    MAX_Z = MAX_LIN_VEL_Z*test_env.EPISODE_LEN_SEC
    MAX_PITCH_ROLL = np.pi # Full range
    

##### x\y\z
    plt.figure()
    plt.plot(observation[0,:]*MAX_XY,label="x")
    plt.grid()
    plt.legend()
    plt.title('obs0_x')
    plt.savefig(save_path + '/obs0_x.jpg')


    plt.figure()
    plt.plot(observation[1,:]*MAX_XY,label="y")
    plt.grid()
    plt.legend()
    plt.title('obs1_y')
    plt.savefig(save_path + '/obs1_y.jpg')

    plt.figure()
    plt.plot(observation[2,:]*MAX_Z,label="z")
    plt.grid()
    plt.legend()
    plt.title('obs2_y')
    plt.savefig(save_path + '/obs2_z.jpg')

##### r\p\y

    plt.figure()
    plt.plot(observation[3,:]*MAX_PITCH_ROLL/math.pi*180,label="roll")
    plt.plot((actions[1,:]*test_env.MAX_ROLL_PITCH/3/math.pi*180),label="command(action)_roll")
    plt.grid()
    plt.legend()
    plt.title('obs3_roll')
    plt.savefig(save_path + '/obs3_roll.jpg')


    plt.figure()
    plt.plot(observation[4,:]*MAX_PITCH_ROLL/math.pi*180,label="pitch")
    plt.plot((actions[2,:]*test_env.MAX_ROLL_PITCH/3/math.pi*180),label="command(action)_pitch")
    plt.grid()
    plt.legend()
    plt.title('obs4_pitch')
    plt.savefig(save_path + '/obs4_pitch.jpg')

    plt.figure()
    plt.plot(observation[5,:]*MAX_PITCH_ROLL/math.pi*180,label="yaw")
    plt.plot((actions[3,:]*test_env.MAX_ROLL_PITCH/5/math.pi*180),label="command(action)_yaw")
    plt.grid()
    plt.legend()
    plt.title('obs5_yaw')
    plt.savefig(save_path + '/obs5_yaw.jpg')

### line_v

    plt.figure()
    plt.plot(observation[6,:]*MAX_PITCH_ROLL,label="x_vel")
    plt.plot(observation[7,:]*MAX_PITCH_ROLL,label="y_vel")
    plt.plot(observation[8,:]*MAX_LIN_VEL_Z,label="z_vel")
    plt.grid()
    plt.legend()
    plt.title('lin_vel')
    plt.savefig(save_path + '/lin_vel.jpg')

## ang_vel
    plt.figure()
    plt.plot(observation[9,:]*MAX_PITCH_ROLL,label="x_ang_vel")
    plt.plot(observation[10,:]*MAX_PITCH_ROLL,label="y_ang_vel")
    plt.plot(observation[11,:]*MAX_LIN_VEL_Z,label="z_ang_vel")
    plt.grid()
    plt.legend()
    plt.title('lin_vel')
    plt.savefig(save_path + '/ang_vel.jpg')

## rewards

    plt.figure()
    plt.plot(infos[0,:],label="balancingReward")
    plt.plot(infos[1,:],label="contactReward")
    plt.plot(infos[2,:],label="linearvelocityReward")
    plt.plot(infos[3,:],label="angulervelocityReward")
    plt.plot(infos[4,:],label="actionsmoothReward")
    plt.plot(infos[5,:],label="actionlimitReward")
    plt.plot(infos[6,:],label="contactgroundReward")
    plt.plot(infos[7,:],label="slippageReward")
    plt.grid()
    plt.legend()
    plt.title('reward')
    plt.savefig(save_path + '/reward.jpg')
 