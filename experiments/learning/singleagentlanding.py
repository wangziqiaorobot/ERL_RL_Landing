"""Learning script for single agent problems.

Agents are based on `stable_baselines3`'s implementation of A2C, PPO SAC, TD3, DDPG.

Example
-------
To run the script, type in a terminal:

    $ python singleagent.py --env <env> --algo <alg> --obs <ObservationType> --act <ActionType> --cpu <cpu_num>

Notes
-----
Use:

    $ tensorboard --logdir ./results/save-<env>-<algo>-<obs>-<act>-<time-date>/tb/

To check the tensorboard results at:

    http://localhost:6006/

"""
import os
import time
from datetime import datetime
from sys import platform
import argparse
import subprocess
import numpy as np
import gym
import torch
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.cmd_util import make_vec_env # Module cmd_util will be renamed to env_util https://github.com/DLR-RM/stable-baselines3/pull/197
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage
from stable_baselines3.common.utils import set_random_seed
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
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, StopTrainingOnRewardThreshold
from gym_pybullet_drones.envs.single_agent_rl.LandingAviary import LandingAviary
from gym_pybullet_drones.envs.single_agent_rl.TakeoffAviary import TakeoffAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType

import shared_constants

EPISODE_REWARD_THRESHOLD = 1000 # when reach this reward value tranning will stop
"""float: Reward threshold to halt the script."""

if __name__ == "__main__":

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning experiments script')
    parser.add_argument('--env',        default='landing',      type=str,             choices=['takeoff', 'hover', 'flythrugate', 'tune'], help='Task (default: hover)', metavar='')
    parser.add_argument('--algo',       default='sac',        type=str,             choices=['a2c', 'ppo', 'sac', 'td3', 'ddpg'],        help='RL agent (default: ppo)', metavar='')
    parser.add_argument('--obs',        default='kin',        type=ObservationType,                                                      help='Observation space (default: kin)', metavar='')
    parser.add_argument('--act',        default='ld',  type=ActionType,                                                           help='Action space (default: one_d_rpm)', metavar='')
    parser.add_argument('--cpu',        default='10',          type=int,                                                                  help='Number of training environments (default: 1)', metavar='')        
    ARGS = parser.parse_args()

    #### Save directory ########################################
    filename = os.path.dirname(os.path.abspath(__file__))+'/results/save-'+ARGS.env+'-'+ARGS.algo+'-'+ARGS.obs.value+'-'+ARGS.act.value+'-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
    if not os.path.exists(filename):
        os.makedirs(filename+'/')


    if ARGS.algo == 'sac':
        ARGS.cpu=1

    #### Warning ###############################################

    # if ARGS.env == 'tune' and ARGS.act != ActionType.TUN:
    #     print("\n\n\n[WARNING] TuneAviary is intended for use with ActionType.TUN\n\n\n")
    # if ARGS.act == ActionType.ONE_D_RPM or ARGS.act == ActionType.ONE_D_DYN or ARGS.act == ActionType.ONE_D_PID:
    #     print("\n\n\n[WARNING] Simplified 1D problem for debugging purposes\n\n\n")
    # #### Errors ################################################
    #     if not ARGS.env in ['takeoff', 'hover']: 
    #         print("[ERROR] 1D action space is only compatible with Takeoff and HoverAviary")
    #         exit()
    # if ARGS.act == ActionType.TUN and ARGS.env != 'tune' :
    #     print("[ERROR] ActionType.TUN is only compatible with TuneAviary")
    #     exit()
    # if ARGS.algo in ['sac', 'td3', 'ddpg'] and ARGS.cpu!=1: 
    #     print("[ERROR] The selected algorithm does not support multiple environments")
    #     exit()

    #### Uncomment to debug slurm scripts ######################
    # exit()

    
    sa_env_kwargs = dict(aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS, obs=ARGS.obs, act=ARGS.act)
    # train_env = gym.make(env_name, aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS, obs=ARGS.obs, act=ARGS.act) # single environment instead of a vectorized one    
    
    train_env = make_vec_env(LandingAviary,
                                 env_kwargs=sa_env_kwargs,
                                 n_envs=10,# The number of Parallel environments
                                 seed=6
                                 )
   
    print("[INFO] Action space:", train_env.action_space)
    print("[INFO] Observation space:", train_env.observation_space)
    # check_env(train_env, warn=True, skip_render_check=True)
    offpolicy_kwargs = dict(activation_fn=torch.nn.Tanh,
                            net_arch=dict(qf=[256,256,256,128], pi=[256,256,256,128])
                            )
    #### On-policy algorithms ##################################
    onpolicy_kwargs = dict(activation_fn=torch.nn.Tanh,
                           net_arch=[256, 256, dict(vf=[256,128], pi=[256,128])] #c 256/512/1024
                           #net_arch=[ dict(vf=[512,512], pi=[512,512])]
                           ) # or None
   
    if ARGS.algo == 'ppo':
        model = PPO(a2cppoMlpPolicy,
                    train_env,
                    policy_kwargs=onpolicy_kwargs,
                    tensorboard_log=filename+'/tb/',
                    verbose=1
                    ) if ARGS.obs == ObservationType.KIN else PPO(a2cppoCnnPolicy,
                                                                  train_env,
                                                                  policy_kwargs=onpolicy_kwargs,
                                                                  tensorboard_log=filename+'/tb/',
                                                                  verbose=1
                                                                  )

    if ARGS.algo == 'sac':
        model = SAC(sacMlpPolicy,
                    train_env,
                    policy_kwargs=offpolicy_kwargs,
                    gradient_steps=-1,
                    tensorboard_log=filename+'/tb/',
                    verbose=1
                    )

    #### Create eveluation environment #########################
    eval_env = gym.make("landing-aviary-v0",
                            aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                            obs=ARGS.obs,
                            act=ARGS.act,
                            gui=False,
                            record=False
                            )
    
    
        

    #### Train the model #######################################
    checkpoint_callback = CheckpointCallback(save_freq=2000, save_path=filename+'-logs/', name_prefix='rl_model')
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=EPISODE_REWARD_THRESHOLD,
                                                     verbose=1
                                                     )
    eval_callback = EvalCallback(eval_env,
                                 callback_on_new_best=callback_on_best,
                                 verbose=1,
                                 best_model_save_path=filename+'/'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"),
                                 log_path=filename+'/',
                                 eval_freq=int(2000),
                                 deterministic=True,
                                 render=False
                                 )
    model.learn(total_timesteps=1000*2500, #int(1e12),
                callback=eval_callback,
                log_interval=10,
                )

    #### Save the model ########################################
    model.save(filename+'/success_model.zip')
    model.policy.save(filename + "/policy.pth")

    # ====
    # load policy
    # weights = torch.load(weight_path, map_location=device)
    # policy = ControlPolicy()
    # policy.load_weighs(weights["state_dict"])

    

    #### Print training progression ############################
    with np.load(filename+'/evaluations.npz') as data:
        for j in range(data['timesteps'].shape[0]):
            print(str(data['timesteps'][j]))
            print(str(data['results'][j]))
            #print(str(data['timesteps'][j])+","+str(data['results'][j][0][0]))
