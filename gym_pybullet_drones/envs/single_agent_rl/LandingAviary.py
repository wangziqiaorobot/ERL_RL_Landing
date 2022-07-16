# from cv2 import exp
from calendar import c
import numpy as np
from gym import spaces
import pybullet as p

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, BaseAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType, BaseSingleAgentAviary
from stable_baselines3.common.running_mean_std import RunningMeanStd

class LandingAviary(BaseSingleAgentAviary):
    """Single agent RL problem: Landing."""
    
    ################################################################################

    def __init__(self,
                 drone_model: DroneModel=DroneModel.HB,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB, #change the physics to the DYN inorder to the dynamics information
                 freq: int=240,
                 aggregate_phy_steps: int=12,
                 gui=False,
                 record=False, 
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.LD,
                 filepath=None
                 ):
        """Initialization of a single agent RL environment.

        Using the generic single agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        freq : int, optional
            The frequency (Hz) at which the physics engine steps.
        aggregate_phy_steps : int, optional
            The number of physics steps within one call to `BaseAviary.step()`.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation in folder `files/videos/`.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        super().__init__(drone_model=drone_model,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         freq=freq,
                         aggregate_phy_steps=aggregate_phy_steps,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act,
                         filepath=filepath
                         )
        

    ################################################################################
    
    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
    
        contactReward=-0.2#-0.018#*time
        balancingRewardCoeff=-0.1#/(time+0.5)#0.001*(time);0.01
        linearvelocityRewardCoeff=-0.25 #0.05#0.03/(time+0.5)
        angulervelocityRewardCoeff=-0.003#*time
        
        actionlimitRewardCoeff=-0.00001#*time
        slippageRewardCoeff=-1.2
        actionsmoothRewardCoeff=-0.01

  
        Linear_Vz_ref=0.1
        diff_act= self.current_action-np.array(self.last_action[0][0:4])

        if (self.Fcontact[2]) >0: #if Force in z-axis > 0
            contactReward=0 
            self.bool_contact_history=True

        if self.bool_contact_history==True:
            balancingRewardCoeff = 0.001
            Linear_Vz_ref=0


        balancingReward=balancingRewardCoeff*np.linalg.norm(np.array([0, 0,self.INIT_RPYS[0][2]])-np.array(self.rpy).reshape(1,3))**2
        linearvelocityReward=linearvelocityRewardCoeff*np.linalg.norm(np.array([0, 0, Linear_Vz_ref])-np.array(self.vel).reshape(1,3))**2
        
        
        if np.linalg.norm(self.pos[0,0]-self.INIT_XYZS[0][0])>1 or np.linalg.norm(self.pos[0,1]-self.INIT_XYZS[0][1])>1 or (self.pos[0,2]-self.INIT_XYZS[0][2])>1:
            slippageReward=-15
        else:
        # to keep the x,y as the init_xy
            slippageReward=slippageRewardCoeff* np.linalg.norm(np.array(self.INIT_XYZS[0][0:2])-np.array(self.pos[0][0:2]))**2 ##^14

        angulervelocityReward=angulervelocityRewardCoeff*np.linalg.norm(np.array([0, 0,0])-np.array(self.ang_v).reshape(1,3))**2
        actionsmoothReward=actionsmoothRewardCoeff*np.linalg.norm(diff_act)**2
        actionlimitReward=actionlimitRewardCoeff*np.linalg.norm(self.MAX_THRUST/2*(self.current_action[0]+1)-self.GRAVITY)**2
        

        p.performCollisionDetection(physicsClientId=self.CLIENT)
        # L is the contact point on the ground, 
        L=p.getContactPoints(self.PLANE_ID,physicsClientId=self.CLIENT)
        if len(L) !=0:
            contactgroundReward=-(10)
            print("fall down to the ground")
        else:
            contactgroundReward=0




# #################################### OLD VERSION REWARD FUNCTION ##################################################
        
#         """ Contact Reward
        
#         A sparse reward function is used to reward the drone for making contact with a tree branch
#         """
    
#         contactReward=-0.019#-0.018#*time
#         if (self.Fcontact[2]) >0: #if Force in z-axis > 0
#             contactReward=0 
#             self.bool_contact_history=True


#         """ Balancing Reward

#         In order to keep the Roll, Pitch and Yaw at 0 degrees before the vehicle makes contact with the tree branch.
#         """
#         balancingRewardCoeff=0.1#/(time+0.5)#0.001*(time);0.01
#         if self.bool_contact_history==True:
#             balancingRewardCoeff = 0.01
        
#         balancingReward=balancingRewardCoeff*(np.exp(- np.linalg.norm(np.array([0, 0,self.INIT_RPYS[0][2]])-np.array(self.rpy).reshape(1,3))**6)-1)
        
#         """ Slippage Reward
#         Keep the drone at the same x,y position as the initial one;
#             i.e. keep it vertical when landing;
#                  No relative sliding after contact with the tree branch
#         """
#         slippageRewardCoeff=1.2
#         # 
#         if np.linalg.norm(self.pos[0,0]-self.INIT_XYZS[0][0])>1 or np.linalg.norm(self.pos[0,1]-self.INIT_XYZS[0][1])>1 or (self.pos[0,2]-self.INIT_XYZS[0][2])>1:
#             slippageReward=-15
#         else:
#         # to keep the x,y as the init_xy
#             slippageReward=slippageRewardCoeff* (np.exp(- np.linalg.norm(np.array(self.INIT_XYZS[0][0:2])-np.array(self.pos[0][0:2]))**8)-1) ##^14
        
#         """ Linear Velocity Reward
#         Make the velocity of the drone in xyz direction as zero as possible;
#         Because we don't want to move in the xy direction; and we want to land as slowly as possible
#         """
#         linearvelocityRewardCoeff=0.25 #0.05#0.03/(time+0.5)
#         linearvelocityReward=linearvelocityRewardCoeff*(np.exp(- np.linalg.norm(np.array([0, 0, 0])-np.array(self.vel).reshape(1,3))**4)-1)
        
#         """ Anguler Velocity  Reward
#         Don't want the drone's attitude to change too quickly
#         """
#         angulervelocityRewardCoeff=0.003#*time
#         angulervelocityReward=angulervelocityRewardCoeff*(np.exp(- np.linalg.norm(np.array([0, 0,0])-np.array(self.ang_v).reshape(1,3))**4)-1)

#         """ Action Smooth  Reward
#         Smoothing the output action     
#         """
#         diff_act= self.current_action-np.array(self.last_action[0][0:4])
#         actionsmoothRewardCoeff=-0.01
#         actionsmoothReward=actionsmoothRewardCoeff*np.linalg.norm(diff_act)**2

#         """ Action Limit  Reward
#         Limiting thurust output        
#         """
#         actionlimitRewardCoeff=-0.00001#*time
#         actionlimitReward=actionlimitRewardCoeff*np.linalg.norm(self.current_action[3])**2

#         """ Chrash  Reward
#         Preventing drone crashes       
#         """

#         p.performCollisionDetection(physicsClientId=self.CLIENT)
#         # L is the contact point on the ground, 
#         L=p.getContactPoints(self.PLANE_ID,physicsClientId=self.CLIENT)

#         if len(L) !=0:
#             contactgroundReward=-(10)
#             print("fall down to the ground")
#         else:
#             contactgroundReward=0
        



        return balancingReward+slippageReward+contactReward+linearvelocityReward+angulervelocityReward+actionsmoothReward+actionlimitReward+contactgroundReward+0.05
        
     

    ################################################################################
    
    def _computeDone(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        ### TO DO : stop the current episode, when the dorne on ground
        
        p.performCollisionDetection(physicsClientId=self.CLIENT)
        L=p.getContactPoints(self.PLANE_ID,physicsClientId=self.CLIENT)
        
        if self.step_counter/self.SIM_FREQ > self.EPISODE_LEN_SEC or len(L) !=0 or np.linalg.norm(self.pos[0,0]-self.INIT_XYZS[0][0])>1 or np.linalg.norm(self.pos[0,1]-self.INIT_XYZS[0][1])>1 or (self.pos[0,2]-self.INIT_XYZS[0][2])>1:
            self.iterate= self.iterate+1
            
       
            return True
        else:
            return False

    ################################################################################
    
    def _computeInfo(self):
    #     """Computes the current info dict(s).

    #     Unused.

    #     Returns
    #     -------
    #     dict[str, int]
    #         Dummy value.

    #     """

        """ Contact Reward
        
        A sparse reward function is used to reward the drone for making contact with a tree branch
        """
    
        contactReward=-0.019#-0.018#*time
        if (self.Fcontact[2]) >0: #if Force in z-axis > 0
            contactReward=0 
            self.bool_contact_history=True


        """ Balancing Reward

        In order to keep the Roll, Pitch and Yaw at 0 degrees before the vehicle makes contact with the tree branch.
        """
        balancingRewardCoeff=0.1#/(time+0.5)#0.001*(time);0.01
        if self.bool_contact_history==True:
            balancingRewardCoeff = 0.01
        
        balancingReward=balancingRewardCoeff*(np.exp(- np.linalg.norm(np.array([0, 0,self.INIT_RPYS[0][2]])-np.array(self.rpy).reshape(1,3))**6)-1)
        
        """ Slippage Reward
        Keep the drone at the same x,y position as the initial one;
            i.e. keep it vertical when landing;
                 No relative sliding after contact with the tree branch
        """
        slippageRewardCoeff=1.2
        # 
        if np.linalg.norm(self.pos[0,0]-self.INIT_XYZS[0][0])>1 or np.linalg.norm(self.pos[0,1]-self.INIT_XYZS[0][1])>1 or (self.pos[0,2]-self.INIT_XYZS[0][2])>1:
            slippageReward=-15
        else:
        # to keep the x,y as the init_xy
            slippageReward=slippageRewardCoeff* (np.exp(- np.linalg.norm(np.array(self.INIT_XYZS[0][0:2])-np.array(self.pos[0][0:2]))**8)-1) ##^14
        
        """ Linear Velocity Reward
        Make the velocity of the drone in xyz direction as zero as possible;
        Because we don't want to move in the xy direction; and we want to land as slowly as possible
        """
        linearvelocityRewardCoeff=0.25 #0.05#0.03/(time+0.5)
        linearvelocityReward=linearvelocityRewardCoeff*(np.exp(- np.linalg.norm(np.array([0, 0, 0])-np.array(self.vel).reshape(1,3))**4)-1)
        
        """ Anguler Velocity  Reward
        Don't want the drone's attitude to change too quickly
        """
        angulervelocityRewardCoeff=0.003#*time
        angulervelocityReward=angulervelocityRewardCoeff*(np.exp(- np.linalg.norm(np.array([0, 0,0])-np.array(self.ang_v).reshape(1,3))**4)-1)

        """ Action Smooth  Reward
        Smoothing the output action     
        """
        diff_act= self.current_action-np.array(self.last_action[0][0:4])
        actionsmoothRewardCoeff=-0.01
        actionsmoothReward=actionsmoothRewardCoeff*np.linalg.norm(diff_act)**2

        """ Action Limit  Reward
        Limiting thurust output        
        """
        actionlimitRewardCoeff=-0.00001#*time
        actionlimitReward=actionlimitRewardCoeff*np.linalg.norm(self.current_action[3])**2

        """ Chrash  Reward
        Preventing drone crashes       
        """

        p.performCollisionDetection(physicsClientId=self.CLIENT)
        # L is the contact point on the ground, 
        L=p.getContactPoints(self.PLANE_ID,physicsClientId=self.CLIENT)

        if len(L) !=0:
            contactgroundReward=-(10)
            print("fall down to the ground")
        else:
            contactgroundReward=0
        
        
        
        ###-----------------------the drone's real states ----------------------------###

        #position
        Pos_x=self.pos[0,0]
        Pos_y=self.pos[0,1]
        Pos_z=self.pos[0,2]

        #attitude
        Rpy_r=self.rpy[0,0]
        Rpy_p=self.rpy[0,1]
        Rpy_y=self.rpy[0,2]

        #linear velocity
        V_x=self.vel[0,0]
        V_y=self.vel[0,1]
        V_z=self.vel[0,2]

        #anguler velocity
        W_x=self.ang_v[0,0]
        W_y=self.ang_v[0,1]
        W_z=self.ang_v[0,2]

        #last step action
        Action_1=self.last_action[0,0]
        Action_2=self.last_action[0,1]
        Action_3=self.last_action[0,2]
        Action_4=self.last_action[0,3]

        #force
        F_x=self.Fcontact[0]
        F_y=self.Fcontact[1]
        F_z=self.Fcontact[2]



        info=np.hstack([ balancingReward, contactReward,linearvelocityReward,angulervelocityReward,
                        actionsmoothReward,actionlimitReward,slippageReward,contactgroundReward,
                         Pos_x,Pos_y,Pos_z,
                         Rpy_r,Rpy_p,Rpy_y,
                         V_x,V_y,V_z,
                         W_x,W_y,W_z,
                         Action_1,Action_2,Action_3,Action_4,
                         F_x,F_y,F_z
                        ])

        return info #{"answer": 42} #info
    ################################################################################
    
    def _clipAndNormalizeState(self,
                               state
                               ):
        """Normalizes a drone's state to the [-1,1] range.

        Parameters
        ----------
        state : ndarray
            (20,)-shaped array of floats containing the non-normalized state of a single drone.

        Returns
        -------
        ndarray
            (20,)-shaped array of floats containing the normalized state of a single drone.

        """
        MAX_LIN_VEL_XY = 2 
        MAX_LIN_VEL_Z = 1

        MAX_XY =1 #MAX_LIN_VEL_XY*self.EPISODE_LEN_SEC
        MAX_Z = 3 #MAX_LIN_VEL_Z*self.EPISODE_LEN_SEC

        MAX_PITCH_ROLL = np.pi/2 # Full range
        
        MAX_F_XY=5  #max external froce in xy axis in robot frame
        MAX_F_Z=11.76 ##max external froce in z axis in robot frame
        #这些值正确吗？？

        clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
        clipped_pos_z = np.clip(state[2], 0, MAX_Z)
        clipped_rp = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_vel_xy = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)
        clipped_F_xy_External=np.clip(state[20:22],-MAX_F_XY, MAX_F_XY)
        clipped_F_z_External=np.clip(state[22],-MAX_F_Z, MAX_F_Z)
        if self.GUI:
            self._clipAndNormalizeStateWarning(state,
                                               clipped_pos_xy,
                                               clipped_pos_z,
                                               clipped_rp,
                                               clipped_vel_xy,
                                               clipped_vel_z
                                               )

        normalized_pos_xy = clipped_pos_xy / MAX_XY
        normalized_pos_z = clipped_pos_z / MAX_Z
        normalized_rp = clipped_rp / MAX_PITCH_ROLL      
        normalized_y =  state[9]  #state[9] / np.pi # No reason to clip
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_XY
        normalized_ang_vel = state[13:16]/np.linalg.norm(state[13:16]) if np.linalg.norm(state[13:16]) != 0 else state[13:16]
        normalized_fxy_external= clipped_F_xy_External/MAX_F_XY
        normalized_fz_external= clipped_F_z_External/MAX_F_Z
        norm_and_clipped = np.hstack([normalized_pos_xy,
                                      normalized_pos_z,
                                      state[3:7],
                                      normalized_rp,
                                      normalized_y,
                                      normalized_vel_xy,
                                      normalized_vel_z,
                                      normalized_ang_vel,
                                      state[16:20],
                                      normalized_fxy_external,
                                      normalized_fz_external
                                      ]).reshape(23,)
        


        #TODO rewrite the normalized obs with runing_mean_std
        normaledposxy=RunningMeanStd(1e-4,2)
        print("##################################")
        
        normaledposxy.update(clipped_pos_xy)
        print(normaledposxy.mean)
        print(clipped_pos_xy)
        print("##################################")

        return norm_and_clipped
    
    ################################################################################
    
    def _clipAndNormalizeStateWarning(self,
                                      state,
                                      clipped_pos_xy,
                                      clipped_pos_z,
                                      clipped_rp,
                                      clipped_vel_xy,
                                      clipped_vel_z,
                                      ):
        """Debugging printouts associated to `_clipAndNormalizeState`.

        Print a warning if values in a state vector is out of the clipping range.
        
        """
        if not(clipped_pos_xy == np.array(state[0:2])).all():
            print("[WARNING] it", self.step_counter, "in LandingAviary._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(state[0], state[1]))
        if not(clipped_pos_z == np.array(state[2])).all():
            print("[WARNING] it", self.step_counter, "in LandingAviary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(state[2]))
        if not(clipped_rp == np.array(state[7:9])).all():
            print("[WARNING] it", self.step_counter, "in LandingAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(state[7], state[8]))
        if not(clipped_vel_xy == np.array(state[10:12])).all():
            print("[WARNING] it", self.step_counter, "in LandingAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(state[10], state[11]))
        if not(clipped_vel_z == np.array(state[12])).all():
            print("[WARNING] it", self.step_counter, "in LandingAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[12]))

    