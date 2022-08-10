"""Test script for single agent problems.

This scripts runs the best model found by one of the executions of `singleagent.py`

Example
-------
To run the script, type in a terminal:

    $ python test_singleagent.py --exp ./results/save-<env>-<algo>-<obs>-<act>-<time_date>

"""
from cmath import sqrt
from distutils.log import INFO
import os
import time
from turtle import left
import matplotlib.pyplot as plt
import matplotlib.colors as color
from matplotlib import cm


import math
from datetime import datetime
import argparse

import numpy as np
import gym

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
import pandas as pd

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
    test_env= LandingAviary(
                         gui=False,
                         record=False,
                         aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS
                         )
    logger = Logger(logging_freq_hz=int(test_env.SIM_FREQ/test_env.AGGR_PHY_STEPS),
                     num_drones=1
                    )
    test_steps=720
    test_iteration=10
    contact_time_total=0
    x_stability= np.zeros(shape=(1, test_iteration), dtype=np.float32)
    y_stability= np.zeros(shape=(1, test_iteration), dtype=np.float32)
    success_rate= np.zeros(shape=(1, test_iteration), dtype=np.float32)
    Landing_info=np.zeros(shape=(test_iteration,14 ), dtype=np.float32)# |iteration|init_x|int_y|K of branch|time before landing|average landing speed|average contact force|success rate|
    z_push=np.zeros(shape=( 1,test_iteration), dtype=np.float32)
    slippage= np.zeros(shape=(1, test_iteration), dtype=np.float32)

    for j in range(test_iteration):
        obs = test_env.reset()
        start = time.time()
        # new log try ####

        actions = np.zeros(shape=(test_env.action_space.shape[0], test_steps), dtype=np.float32)
        observation = np.zeros(shape=(test_env.observation_space.shape[0], test_steps), dtype=np.float32)
        rewards = np.zeros(shape=(1, test_steps), dtype=np.float32)
        time_plt = np.zeros(shape=(1, test_steps), dtype=np.float32)
        infos = np.zeros(shape=(30, test_steps), dtype=np.float32)
        
        
        x_displacement=np.zeros(shape=(1, 480), dtype=np.float32)
        y_displacement=np.zeros(shape=( 1,480), dtype=np.float32)
        z_displacement=np.zeros(shape=( 1,480), dtype=np.float32)
        
        x_pos_after_contact=np.zeros(shape=( 1,480), dtype=np.float32)
        y_pos_after_contact=np.zeros(shape=( 1,480), dtype=np.float32)
        z_pos_after_contact=np.zeros(shape=( 1,480), dtype=np.float32)
        F_z=np.zeros(shape=(1,480), dtype=np.float32)
        
        
        # for calculate the contact time
        Contactflag=False
        
        Contact_start_timestep=0
        Contact_start_time=0
        contact_time=0
        #for calculate the landing time 
        Landing_timestep=0.001
        Landing_time=0
        Landing_speed=0
        # calculate how much could dorne push during the 10sec contact
        z_firstcontact=0
        z_final=0
        # counter for slipage
        slippage_counter=0


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
            
            # ---------- calculate the success rate -----------
            # set the contact flag to true, if F_z >0; infors[26] is the force in z-axis
            if infos[26,i]>0:
                Contactflag=True
            # calculate the time step after the first contact
            if Contactflag==True:
                Contact_start_timestep=Contact_start_timestep+1
                Contact_start_time=Contact_start_timestep*test_env.AGGR_PHY_STEPS/240
            else:
                Landing_timestep=Landing_timestep+1
                Landing_speed=Landing_speed+infos[16,i]


            # 480 time step= 10 sec; Calculate the contact step from the first contact
            if Contact_start_timestep > 0 and Contact_start_timestep <=480:
                if infos[26,i]>0:
                    contact_time=contact_time+1 # calculate how many time step drone contact with branch

            # ---------- calculate the stability -------------
                x_pos_after_contact[0,(Contact_start_timestep-1)]=infos[8,i]
                y_pos_after_contact[0,(Contact_start_timestep-1)]=infos[9,i]
                z_pos_after_contact[0,(Contact_start_timestep-1)]=infos[10,i]
                x_displacement[0,(Contact_start_timestep-1)]=((infos[8,i]-x_pos_after_contact[0][0])**2)
                y_displacement[0,(Contact_start_timestep-1)]=((infos[9,i]-y_pos_after_contact[0][0])**2)
                # z_displacement[0,(Contact_start_timestep-1)]=((infos[10,i]-z_pos_after_contact[0][0])**2)
                z_final=infos[10,i]

                F_z[0,(Contact_start_timestep-1)]=infos[26,i]
                mu=(test_env.lateralFriction+0.5/2)

                print("test_env.lateralFriction:",test_env.lateralFriction)
                print("normal force:",infos[27,i],infos[28,i],infos[29,i], np.sqrt(infos[28,i]*infos[28,i]+infos[29,i]*infos[29,i]))
                if infos[27,i]*mu < np.sqrt(infos[28,i]*infos[28,i]+infos[29,i]*infos[29,i])+0.0001 :
                    slippage_counter=slippage_counter+1




        # z_final=min(z_pos_after_contact[0,:])
        contact_time_total=contact_time_total+contact_time
        x_stability[0,j]=np.sqrt(np.mean(x_displacement))
        y_stability[0,j]=np.sqrt(np.mean(y_displacement))
        z_push[0,j]=z_final-z_pos_after_contact[0][0]
        slippage[0,j]=1-slippage_counter/contact_time
        success_rate[0,j]=contact_time/480
        Landing_time=Landing_timestep*test_env.AGGR_PHY_STEPS/240
        print("---------------------Iteration",j,"-----------------")
        print("The contact suss rate in iteration", j,"is:",contact_time/480)
        print("The stability in x in", j,"iteration is:", x_stability[0,j])
        print("The stability in y in", j,"iteration is:", y_stability[0,j])
    
        # save the detail landing informations
        Landing_info[j,0]=j+1 # iteration
        Landing_info[j,1]=infos[8,0]# init_x
        Landing_info[j,2]=infos[9,0]# init_y
        Landing_info[j,3]=infos[10,0]# init_z
        Landing_info[j,4]=test_env.pd4branch[4]#stiffness of branch
        Landing_info[j,5]=Landing_time#time before landing
        Landing_info[j,6]=Landing_speed /Landing_timestep#average landing speed
        Landing_info[j,7]=np.mean(F_z)#average contact force in Z-axis
        Landing_info[j,8]=np.var(F_z)
        Landing_info[j,9]=success_rate[0,j]#success rate 
        Landing_info[j,10]=x_stability[0,j]
        Landing_info[j,11]=y_stability[0,j]
        Landing_info[j,12]=z_push[0,j]
        Landing_info[j,13]=slippage[0,j]
        
        
        
        #for ppo color
        # fig=plt.scatter(Landing_info[j,1], Landing_info[j,2],color=[1-success_rate[0,j],success_rate[0,j],0],marker='o')
        # fig=plt.scatter(Landing_info[j,1], Landing_info[j,2],color=[(1-success_rate[0,j])*2.5,1-(1-success_rate[0,j])*2.5,0],marker='o')
        
        # fig=plt.scatter(Landing_info[j,1], Landing_info[j,2],color=[0.75,0.25,0],marker='o')
    


    # for ppo colorfull SR_top_view
    norm = plt.Normalize(0.6,1)
    norm_values = norm(success_rate[0,:])
    map_vir = cm.get_cmap(name='plasma')
    colors = map_vir(norm_values)
    
    fig = plt.figure() 
    plt.subplot(111)
    ax=plt.scatter(Landing_info[:,1], Landing_info[:,2],color=colors,marker='o')
    sm = cm.ScalarMappable(cmap=map_vir,norm=norm)  # norm设置最大最小值
    sm.set_array([])
    plt.colorbar(sm)

    print("The contact success rate in Total:",contact_time_total/(480*test_iteration))
    # print("The stability in x-axis:",np.sqrt(np.mean(x_stability)),np.var(np.sqrt(x_stability)))
    # print("The stability in y-axis:",np.sqrt(np.mean(y_stability)),np.var(np.sqrt(y_stability)))
    print("The stability in x-axis:",(np.mean(x_stability)),np.var(x_stability))
    print("The stability in y-axis:",(np.mean(y_stability)),np.var(y_stability))
    print("The average landing speed is:",(np.mean(Landing_info[:,6])))
    print("The average & var contact force is:",np.mean(Landing_info[:,7]),np.var(Landing_info[:,7]))
    print("The average & var displacement(after contact) in z-axis is:",np.mean(z_push),np.var(z_push))
    print("the slippage is" ,np.mean(slippage),np.var(slippage))

    
    
    


  
    #save the evaluation results to xlsx files
    df=pd.DataFrame(data=success_rate.reshape(test_iteration,1), columns=['success rate'])
    filepath=ARGS.exp+'success_rate.xlsx'
    df.to_excel(filepath, index=False)

    df=pd.DataFrame(data=Landing_info ,columns=['iteration','init_x','int_y','init_z','K of branch','time before landing','average landing speed','average contact force','force var','success rate','x_stability','y_stability','Z_push','Sliding'])
    filepath=ARGS.exp+'landing_info.xlsx'
    df.to_excel(filepath, index=False)

    
    
    
    
    
    
    test_env.close()

    # c1=color.rgb2hex([1, 0, 0])
    # plt.scatter(Landing_info[:,1], Landing_info[:,2],c=[[1, 0, 0],[0,1,0]],marker='o')
  
    plt.axis([-0.1, 0.1, -0.1, 0.1])
    plt.grid()
    plt.title('Success Rate Top Veiw')
    plt.xlabel('Initial X Position [m]')
    plt.ylabel('Initial Y Position [m]')
    fig.subplots_adjust(left=0.2)
    # 
    
    plt.savefig(ARGS.exp + '/Success_Rate_Top_Veiw.jpg',dpi=600)
    
    plt.show()
    
    # fig2,ax2=plt.subplots(figsize=(1, 6))
    # fig2.subplots_adjust(bottom=0.2)
    # sm = cm.ScalarMappable(norm=norm,cmap=map_vir)  # norm设置最大最小值
    # sm.set_array([])
    # plt.colorbar(sm,cax=ax2)#,orientation='horizontal'
    # plt.title('Success Rate [%]]')
    # plt.savefig( ARGS.exp +'/Success_Rate_bar.jpg',dpi=600)
    # plt.show()
    plt.figure() 
    
    plt.grid()
    plt.title('Success Rate Top Veiw')
    plt.xlabel('K')
    plt.ylabel('Saccess Rate [%]')
    plt.scatter(Landing_info[:,4], Landing_info[:,9],marker='o')
    plt.savefig(ARGS.exp + '/Success_Rate_Diff_K.jpg',dpi=600)
    
    plt.show()
