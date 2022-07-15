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
from stable_baselines3.common.utils import get_device
from stable_baselines3.sac.policies import SACPolicy as sacMlpPolicy
from gym_pybullet_drones.control.mlp_policy import RPGMlpPolicy
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
    parser.add_argument("--iter", type=int, default=2000, help="PPO iter number")
    parser.add_argument('--exp',                           type=str,            help='The experiment folder written as ./results/save-<env>-<algo>-<obs>-<act>-<time_date>', metavar='')
    ARGS = parser.parse_args()

    #### Load the model from file ##############################
    algo = ARGS.exp.split("-")[2]
    
    if os.path.isfile(ARGS.exp+'/success_model.zip'):
        path = ARGS.exp+'/success_model.zip'
    elif os.path.isfile(ARGS.exp+'/best_model.zip'):
        path = ARGS.exp+'/best_model.zip'
    else:
        path=ARGS.exp+"/tb/SAC_1/Policy/iter_{0:05d}.pth".format(ARGS.iter)
 
    if algo == 'ppo':
        model = PPO.load(path)
    if algo == 'sac':
        model = SAC.load(path)

        ###########
        weight = ARGS.exp+"/tb/SAC_1/Policy/iter_{0:05d}.pth".format(ARGS.iter)
        # env_rms="/home/ziqiao/RL/ERL_RL_Landing/experiments/learning/results/save-landing-sac-kin-ld-07.13.2022_23.20.06/RMS/iter_20.npz"
        env_rms =ARGS.exp+"/RMS/iter_{0:05d}.npz".format(ARGS.iter)
        print("weight",weight)
        print("rms",env_rms)


        if torch.cuda.is_available():
            device = "cuda:0"
            print("GPU found")
        else:
            device = "cpu"
        device="cpu"
        # device = get_device("cpu")

        saved_variables = torch.load(weight, map_location=device)
        print("saved variables",saved_variables["data"])
        # Create policy object
        policy = RPGMlpPolicy(saved_variables["data"],device)
        
        
        
        # Load weights
        policy.load_weights(saved_variables["state_dict"])
        
        
        ###################################

    #### Parameters to recreate the environment ################
    env_name = ARGS.exp.split("-")[1]+"-aviary-v0"
    OBS = ObservationType.KIN 
    ACT = ActionType.LD
    #### Evaluate the model ####################################
    
    test_env= LandingAviary(
                         gui=True,
                         record=True,
                         aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                         filepath=ARGS.exp
                         )
    
    logger = Logger(logging_freq_hz=int(test_env.SIM_FREQ/test_env.AGGR_PHY_STEPS),
                     num_drones=1
                    )
    # # policy just in time compliation
    dummy_inputs = torch.rand(1, 25, device=device)
    print("dummy input",dummy_inputs.size)

    # policy = torch.jit.trace(policy, dummy_inputs)
    # print("torch.jit policy",policy)

    test_env.load_rms(env_rms)
    #############
    obs = test_env.reset()
   
    start = time.time()
    
    

    # new log try ####
    test_steps=400
    actions = np.zeros(
        shape=(test_env.action_space.shape[0], test_steps), dtype=np.float32)
    observation = np.zeros(
        shape=(test_env.observation_space.shape[0], test_steps), dtype=np.float32)
    unnormalized_obs = np.zeros(
        shape=(test_env.observation_space.shape[0], test_steps), dtype=np.float32)
    rewards = np.zeros(
        shape=(1, test_steps), dtype=np.float32)
    time_plt = np.zeros(
        shape=(1, test_steps), dtype=np.float32)
    infos = np.zeros(
        shape=(8, test_steps), dtype=np.float32)
    for i in range(test_steps):
        # obs_numpy = np.array(obs) # construct your observation as a numpy array
        # print("obs_numpy",obs_numpy)
        # obs_tensor = torch.as_tensor(obs_numpy).to(device)
        # print("obs_tensor",obs_tensor.size)
        # action_numpy = policy.forward(obs_tensor).detach().cpu().numpy()
        
        action_numpy, _states = model.predict(obs)
        obs, reward, done, info = test_env.step(action_numpy)
        test_env.render()
        actions[:,i]=action_numpy
        unnormalized_obs[:,i]=test_env._unnormalize_obs(obs,test_env.obs_rms)
        print("unnormalized_obs",unnormalized_obs)
        
        observation[:,i]=obs
        print("observation",observation)
        rewards[:,i]=reward
        infos[:,i]=info
        time_plt[:,i]=i*test_env.AGGR_PHY_STEPS/240
        sync(np.floor(i*test_env.AGGR_PHY_STEPS), start, test_env.TIMESTEP)
    test_env.close()



    # with np.load(ARGS.exp+'/evaluations.npz') as data:
    #     print(data.files)
    #     print(data['timesteps'])
    #     print(data['results'])
    #     print(data['ep_lengths'])
    
#     ############### Plot the states & actions
    save_path = os.path.join(ARGS.exp)
#     plt.figure()
#     plt.plot(time_plt[0,:],test_env.GRAVITY*(0.1*actions[0,:]+1),label="thrust")
#     #plt.plot((actions[0,:]),label="b")
#     plt.grid()
#     plt.legend()
#     plt.title('action0_thrust')
#     plt.savefig(save_path + '/action0_thrust.jpg')
    
#     plt.figure()
#     # plt.plot(test_env.MAX_THRUST/2*(actions[1,:]*0.05+1),label="roll")
#     plt.plot(time_plt[0,:],(actions[1,:]*test_env.MAX_ROLL_PITCH/math.pi*180),label="roll")
#     plt.grid()
#     plt.legend()
#     plt.title('action1_roll')
#     plt.savefig(save_path + '/action1_roll.jpg')

#     plt.figure()
#     # plt.plot(test_env.MAX_THRUST/2*(actions[1,:]*0.05+1),label="roll")
#     plt.plot(time_plt[0,:],(actions[2,:]*test_env.MAX_ROLL_PITCH/math.pi*180),label="pitch")
#     plt.grid()
#     plt.legend()
#     plt.title('action2_pitch')
#     plt.savefig(save_path + '/action2_pitch.jpg')

#     plt.figure()
#     # plt.plot(test_env.MAX_THRUST/2*(actions[1,:]*0.05+1),label="roll")
#     plt.plot(time_plt[0,:],(actions[3,:]*test_env.MAX_ROLL_PITCH/math.pi/5*180),label="yaw")
#     plt.grid()
#     plt.legend()
#     plt.title('action3_yaw')
#     plt.savefig(save_path + '/action3_yaw.jpg')

#     ## observation
#     MAX_LIN_VEL_XY = 2 
#     MAX_LIN_VEL_Z = 1

#     MAX_XY = 1
#     MAX_Z = 3
#     MAX_PITCH_ROLL = np.pi/2 # Full range
    

# ##### x\y\z
#     plt.figure()
#     plt.plot(time_plt[0,:],observation[0,:]*MAX_XY,label="x")
#     plt.grid()
#     plt.legend()
#     plt.title('obs0_x')
#     plt.savefig(save_path + '/obs0_x.jpg')


#     plt.figure()
#     plt.plot(time_plt[0,:],observation[1,:]*MAX_XY,label="y")
#     plt.grid()
#     plt.legend()
#     plt.title('obs1_y')
#     plt.savefig(save_path + '/obs1_y.jpg')

#     plt.figure()
#     plt.plot(time_plt[0,:],observation[2,:]*MAX_Z,label="z")
#     plt.grid()
#     plt.legend()
#     plt.title('obs2_z')
#     plt.savefig(save_path + '/obs2_z.jpg')

# ##### r\p\y

#     plt.figure()
#     plt.plot(time_plt[0,:],observation[7,:]*MAX_PITCH_ROLL/math.pi*180,label="roll")
#     plt.plot(time_plt[0,:],(actions[1,:]*test_env.MAX_ROLL_PITCH/math.pi*180),label="command(action)_roll")
#     plt.grid()
#     plt.legend()
#     plt.title('obs3_roll')
#     plt.savefig(save_path + '/obs3_roll.jpg')


#     plt.figure()
#     plt.plot(time_plt[0,:],observation[8,:]*MAX_PITCH_ROLL/math.pi*180,label="pitch")
#     plt.plot(time_plt[0,:],(actions[2,:]*test_env.MAX_ROLL_PITCH/math.pi*180),label="command(action)_pitch")
#     plt.grid()
#     plt.legend()
#     plt.title('obs4_pitch')
#     plt.savefig(save_path + '/obs4_pitch.jpg')

#     plt.figure()
#     plt.plot(time_plt[0,:],observation[9,:]/math.pi*180,label="yaw")
#     plt.plot(time_plt[0,:],(actions[3,:]*test_env.MAX_ROLL_PITCH/math.pi/5*180),label="command(action)_yaw")
#     plt.grid()
#     plt.legend()
#     plt.title('obs5_yaw')
#     plt.savefig(save_path + '/obs5_yaw.jpg')

# ### line_v

#     plt.figure()
#     plt.plot(time_plt[0,:],observation[10,:]*MAX_LIN_VEL_XY,label="x_vel")
#     plt.plot(time_plt[0,:],observation[11,:]*MAX_LIN_VEL_XY,label="y_vel")
#     plt.plot(time_plt[0,:],observation[12,:]*MAX_LIN_VEL_Z,label="z_vel")
#     plt.grid()
#     plt.legend()
#     plt.title('lin_vel')
#     plt.savefig(save_path + '/lin_vel.jpg')

# ## ang_vel
#     plt.figure()
#     plt.plot(time_plt[0,:],observation[13,:],label="x_ang_vel")
#     plt.plot(time_plt[0,:],observation[14,:],label="y_ang_vel")
#     plt.plot(time_plt[0,:],observation[15,:],label="z_ang_vel")
#     plt.grid()
#     plt.legend()
#     plt.title('ang_vel')
#     plt.savefig(save_path + '/ang_vel.jpg')

## rewards

    plt.figure()
    plt.plot(time_plt[0,:],infos[0,:],label="balancingReward")
    plt.plot(time_plt[0,:],infos[1,:],label="contactReward")
    plt.plot(time_plt[0,:],infos[2,:],label="linearvelocityReward")
    plt.plot(time_plt[0,:],infos[3,:],label="angulervelocityReward")
    plt.plot(time_plt[0,:],infos[4,:],label="actionsmoothReward")
    plt.plot(time_plt[0,:],infos[5,:],label="actionlimitReward")
    plt.plot(time_plt[0,:],infos[6,:],label="slippageReward")
    plt.plot(time_plt[0,:],infos[7,:],label="contactgroundReward")
    plt.grid()
    plt.legend()
    plt.title('reward')
    plt.savefig(save_path + '/reward.jpg')
    
    
#     MAX_F_XY=5  #max external froce in xy axis in robot frame
#     MAX_F_Z=11.76 ##max external froce in z axis in robot frame
# ## force 
#     plt.figure()
#     plt.plot(time_plt[0,:],observation[20,:]*MAX_F_XY,label="fx")
#     plt.plot(time_plt[0,:],observation[21,:]*MAX_F_XY,label="fy")
#     plt.plot(time_plt[0,:],observation[22,:]*MAX_F_Z,label="fz")
#     plt.grid()
#     plt.legend()
#     plt.title('Force')
#     plt.savefig(save_path + '/Force.jpg')



####################### polt the observation ###############################
######## xyz
    plt.figure()
    plt.plot(time_plt[0,:],observation[0,:],label="obs0_x")
    plt.plot(time_plt[0,:],observation[1,:],label="obs1_y")
    plt.plot(time_plt[0,:],observation[2,:],label="obs2_z")
    plt.grid()
    plt.legend()
    plt.title('obs_xyz')
    plt.savefig(save_path + '/obs_xyz.jpg')

########   Q4
    plt.figure()
    plt.plot(time_plt[0,:],observation[3,:],label="m1")
    plt.plot(time_plt[0,:],observation[4,:],label="m2")
    plt.plot(time_plt[0,:],observation[5,:],label="m3")
    plt.plot(time_plt[0,:],observation[6,:],label="m4")
    plt.plot(time_plt[0,:],observation[7,:],label="m5")
    plt.plot(time_plt[0,:],observation[8,:],label="m6")
    plt.plot(time_plt[0,:],observation[9,:],label="m7")
    plt.plot(time_plt[0,:],observation[10,:],label="m8")
    plt.plot(time_plt[0,:],observation[11,:],label="m9")
    plt.grid()
    plt.legend()
    plt.title('obs_rotaion_matrix')
    plt.savefig(save_path + '/obs_rotaion_matrix.jpg')

######## linear v
    plt.figure()
    plt.plot(time_plt[0,:],observation[12,:],label="V_x")
    plt.plot(time_plt[0,:],observation[13,:],label="V_y")
    plt.plot(time_plt[0,:],observation[14,:],label="V_z")
    plt.grid()
    plt.legend()
    plt.title('obs_lin_v')
    plt.savefig(save_path + '/obs_lin_v.jpg')

######## anguler v
    plt.figure()
    plt.plot(time_plt[0,:],observation[15,:],label="wx")
    plt.plot(time_plt[0,:],observation[16,:],label="wy")
    plt.plot(time_plt[0,:],observation[17,:],label="wz")
    plt.grid()
    plt.legend()
    plt.title('obs_ang_v')
    plt.savefig(save_path + '/obs_ang_v.jpg')

########   action
    plt.figure()
    plt.plot(time_plt[0,:],observation[18,:],label="a1")
    plt.plot(time_plt[0,:],observation[19,:],label="a2")
    plt.plot(time_plt[0,:],observation[20,:],label="a3")
    plt.plot(time_plt[0,:],observation[21,:],label="a4")
    plt.grid()
    plt.legend()
    plt.title('obs_action')
    plt.savefig(save_path + '/obs_action.jpg')

######## force
    plt.figure()
    plt.plot(time_plt[0,:],observation[22,:],label="fx")
    plt.plot(time_plt[0,:],observation[23,:],label="fy")
    plt.plot(time_plt[0,:],observation[24,:],label="fz")
    plt.grid()
    plt.legend()
    plt.title('obs_force')
    plt.savefig(save_path + '/obs_force.jpg')



########## plot the unnormalized observations ###########


    plt.figure()
    plt.plot(time_plt[0,:],unnormalized_obs[0,:],label="obs0_x")
    plt.plot(time_plt[0,:],unnormalized_obs[1,:],label="obs1_y")
    plt.plot(time_plt[0,:],unnormalized_obs[2,:],label="obs2_z")
    plt.grid()
    plt.legend()
    plt.title('unnormalized_obs_xyz')
    plt.savefig(save_path + '/unnormalized_obs_xyz.jpg')

########   Q4
    plt.figure()
    plt.plot(time_plt[0,:],unnormalized_obs[3,:],label="m1")
    plt.plot(time_plt[0,:],unnormalized_obs[4,:],label="m2")
    plt.plot(time_plt[0,:],unnormalized_obs[5,:],label="m3")
    plt.plot(time_plt[0,:],unnormalized_obs[6,:],label="m4")
    plt.plot(time_plt[0,:],unnormalized_obs[7,:],label="m5")
    plt.plot(time_plt[0,:],unnormalized_obs[8,:],label="m6")
    plt.plot(time_plt[0,:],unnormalized_obs[9,:],label="m7")
    plt.plot(time_plt[0,:],unnormalized_obs[10,:],label="m8")
    plt.plot(time_plt[0,:],unnormalized_obs[11,:],label="m9")
    plt.grid()
    plt.legend()
    plt.title('unnormalized_obs_rotaion_matrix')
    plt.savefig(save_path + '/unnormalized_obs_rotaion_matrix.jpg')

######## linear v
    plt.figure()
    plt.plot(time_plt[0,:],unnormalized_obs[12,:],label="V_x")
    plt.plot(time_plt[0,:],unnormalized_obs[13,:],label="V_y")
    plt.plot(time_plt[0,:],unnormalized_obs[14,:],label="V_z")
    plt.grid()
    plt.legend()
    plt.title('unnormalized_obs_lin_v')
    plt.savefig(save_path + '/unnormalized_obs_lin_v.jpg')

######## anguler v
    plt.figure()
    plt.plot(time_plt[0,:],unnormalized_obs[15,:],label="wx")
    plt.plot(time_plt[0,:],unnormalized_obs[16,:],label="wy")
    plt.plot(time_plt[0,:],unnormalized_obs[17,:],label="wz")
    plt.grid()
    plt.legend()
    plt.title('unnormalized_obs_ang_v')
    plt.savefig(save_path + '/unnormalized_obs_ang_v.jpg')

########   action
    plt.figure()
    plt.plot(time_plt[0,:],unnormalized_obs[18,:],label="a1")
    plt.plot(time_plt[0,:],unnormalized_obs[19,:],label="a2")
    plt.plot(time_plt[0,:],unnormalized_obs[20,:],label="a3")
    plt.plot(time_plt[0,:],unnormalized_obs[21,:],label="a4")
    plt.grid()
    plt.legend()
    plt.title('unnormalized_obs_action')
    plt.savefig(save_path + '/unnormalized_obs_action.jpg')

######## force
    plt.figure()
    plt.plot(time_plt[0,:],unnormalized_obs[22,:],label="fx")
    plt.plot(time_plt[0,:],unnormalized_obs[23,:],label="fy")
    plt.plot(time_plt[0,:],unnormalized_obs[24,:],label="fz")
    plt.grid()
    plt.legend()
    plt.title('unnormalized_obs_force')
    plt.savefig(save_path + '/unnormalized_obs_force.jpg')






 