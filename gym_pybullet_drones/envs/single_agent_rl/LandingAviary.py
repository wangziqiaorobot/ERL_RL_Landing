# from cv2 import exp
from calendar import c
from random import random
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
                 act: ActionType=ActionType.LD
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
                         act=act
                         )
        

    ################################################################################
    
    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
            
        state = self._getDroneStateVector(0)  ###  self._computeObs() need or not???
        diff_act= self.current_action-state[18:22]

        time=self.step_counter*self.TIMESTEP
        p.performCollisionDetection(physicsClientId=self.CLIENT)
        L=p.getContactPoints(self.PLANE_ID,physicsClientId=self.CLIENT)
        
        balancingRewardCoeff=0.04#/(time+0.5)#0.001*(time);0.01
        slippageRewardCoeff=0.12#*time#0.8;0.5;0.3
        contactRewardCoeff=0
        linearvelocityRewardCoeff=0.55 #0.05#0.03/(time+0.5)
        angulervelocityRewardCoeff=0.03#*time
        actionsmoothRewardCoeff=-0.01
        
          
        if np.linalg.norm(self.pos[0,0]-self.INIT_XYZS[0][0])>1 or np.linalg.norm(self.pos[0,1]-self.INIT_XYZS[0][1])>1 or (self.pos[0,2]-self.INIT_XYZS[0][2])>1:
            slippageReward=-15
        else:
            slippageReward=slippageRewardCoeff* ((- np.linalg.norm(np.array(self.INIT_XYZS[0][0:3])-state[0:3])**2)) ##^14


        balancingReward=balancingRewardCoeff*((- np.linalg.norm(np.array([0, 0,self.INIT_RPYS[0][2]])-[self.rpy[0][0],self.rpy[0][1],self.rpy[0][2]])**2))
        linearvelocityReward=linearvelocityRewardCoeff*((- np.linalg.norm(np.array([0, 0, 0])-state[12:15])**2))
        angulervelocityReward=angulervelocityRewardCoeff*((- np.linalg.norm(np.array([0, 0,0])-state[15:18])**2))
        actionsmoothReward=actionsmoothRewardCoeff*np.linalg.norm(diff_act)**2
   
        if len(L) !=0:
            contactgroundReward=-(10)
            print("fall down to the ground")
        else:
            contactgroundReward=0
        
        return balancingReward+slippageReward+linearvelocityReward+angulervelocityReward+actionsmoothReward+contactgroundReward+0.1
       

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
        
        if self.step_counter/self.SIM_FREQ > self.EPISODE_LEN_SEC or len(L) !=0 or np.linalg.norm(self.pos[0,0]-self.INIT_XYZS[0][0])>1 or np.linalg.norm(self.pos[0,1]-self.INIT_XYZS[0][1])>1 or (self.pos[0,2]-self.INIT_XYZS[0][2])>1:# or ((self._getDroneStateVector(0))[2] < 0.05)  or ((self._getDroneStateVector(0))[2] > 1.5):
            self.iterate= self.iterate+1
           
        # Alternative done condition, see PR #32
        # if (self.step_counter/self.SIM_FREQ > (self.EPISODE_LEN_SEC)) or ((self._getDroneStateVector(0))[2] < 0.05):
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

        state = self._getDroneStateVector(0)  ###  self._computeObs() need or not???
        diff_act= self.current_action-state[18:22]

        time=self.step_counter*self.TIMESTEP
        p.performCollisionDetection(physicsClientId=self.CLIENT)
        L=p.getContactPoints(self.PLANE_ID,physicsClientId=self.CLIENT)
        
        balancingRewardCoeff=0.04#/(time+0.5)#0.001*(time);0.01
        slippageRewardCoeff=0.12#*time#0.8;0.5;0.3
        contactRewardCoeff=0
        linearvelocityRewardCoeff=0.055 #0.05#0.03/(time+0.5)
        angulervelocityRewardCoeff=0.03#*time
        actionsmoothRewardCoeff=-0.01
        
          
        if np.linalg.norm(self.pos[0,0]-self.INIT_XYZS[0][0])>1 or np.linalg.norm(self.pos[0,1]-self.INIT_XYZS[0][1])>1 or (self.pos[0,2]-self.INIT_XYZS[0][2])>1:
            slippageReward=-15
        else:
            slippageReward=slippageRewardCoeff* ((- np.linalg.norm(np.array(self.INIT_XYZS[0][0:3])-state[0:3])**2)) ##^14


        balancingReward=balancingRewardCoeff*((- np.linalg.norm(np.array([0, 0,self.INIT_RPYS[0][2]])-[self.rpy[0][0],self.rpy[0][1],self.rpy[0][2]])**2))
        linearvelocityReward=linearvelocityRewardCoeff*((- np.linalg.norm(np.array([0, 0, 0])-state[12:15])**2))
        angulervelocityReward=angulervelocityRewardCoeff*((- np.linalg.norm(np.array([0, 0,0])-state[15:18])**2))
        actionsmoothReward=actionsmoothRewardCoeff*np.linalg.norm(diff_act)**2
   
        if len(L) !=0:
            contactgroundReward=-(10)
            print("fall down to the ground")
        else:
            contactgroundReward=0
        
        
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
        #normal force and lateralFriction
        normal_force=0
        lateralFriction1=0
        lateralFriction2=0
        
        rot_mat = np.array(p.getMatrixFromQuaternion(self.quat[0, :])).reshape(3, 3)
            # print('the thrust is',rot_mat.T*self.MAX_THRUST*(action[0]+1)/2 )
            # print('the Fz is',self.Fcontact[2])
            # print('RT*mg', np.dot(rot_mat.T,[0,0,self.GRAVITY]))
            # print('F+T',self.Fcontact[2]+self.MAX_THRUST*(action[0]+1)/2)
        
        Rthrust=np.dot(rot_mat,[[0] ,[0] ,[np.sum(self.applyedforce)]])[2]#np.dot(rot_mat,[0,0,np.sum(self.applyedforce)])[2]#

        # Rthrust=np.dot(rot_mat,[[0] ,[0] ,[self.MAX_THRUST*(self.current_action[0]+1)/2]])[2]#np.dot(rot_mat,[0,0,np.sum(self.applyedforce)])[2]#
        # Rthrust=(rot_mat*[[0] ,[0] ,[self.MAX_THRUST*(self.current_action[0]+1)/2]])[2]
        # RFz=np.dot(rot_mat.T,self.Fcontact)[2] #self.Fcontact[2]
        # RFz=(rot_mat*self.Fcontact)[2] #self.Fcontact[2]
        # Rthrust=np.dot(rot_mat.T,[0,0,self.GRAVITY])[2]
        RFz=np.dot(rot_mat,[0,0,self.Fcontact[2]])[2]
        # lateralFriction1=np.sum(self.applyedforce) #used to test the real force applyed on the drone.
        contactReward=0
        actionlimitReward=0
        info=np.hstack([ balancingReward, contactReward,linearvelocityReward,angulervelocityReward,actionsmoothReward,actionlimitReward,slippageReward,contactgroundReward,
                         Pos_x,Pos_y,Pos_z,
                         Rpy_r,Rpy_p,Rpy_y,
                         V_x,V_y,V_z,
                         W_x,W_y,W_z,
                         Action_1,Action_2,Action_3,Action_4,
                         F_x,F_y,F_z,
                         normal_force,lateralFriction1,lateralFriction2,
                         Rthrust,RFz
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
            (25,)-shaped array of floats containing the non-normalized state of a single drone.

        Returns
        -------
        ndarray
            (25,)-shaped array of floats containing the normalized state of a single drone.

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
        # clipped_rp = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_vel_xy = np.clip(state[12:14], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[14], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)
        clipped_F_xy_External=np.clip(state[22:24],-MAX_F_XY, MAX_F_XY)
        clipped_F_z_External=np.clip(state[24],-MAX_F_Z, MAX_F_Z)
        if self.GUI:
            self._clipAndNormalizeStateWarning(state,
                                               clipped_pos_xy,
                                               clipped_pos_z,
                                            #    clipped_rp,
                                               clipped_vel_xy,
                                               clipped_vel_z
                                               )
        if self.noise ==True:
            clipped_pos_xy=clipped_pos_xy+0.005*np.random.uniform(-1,1)
            clipped_pos_z=clipped_pos_z+0.005*np.random.uniform(-1,1)
            clipped_vel_xy=clipped_vel_xy+0.05*np.random.uniform(-1,1)
            clipped_vel_z=clipped_vel_z+0.05*np.random.uniform(-1,1)
            clipped_F_xy_External=clipped_F_xy_External+2*np.random.uniform(-1,1)
            clipped_F_z_External=clipped_F_z_External+1*np.random.uniform(-1,1)
            state[15]=state[15]+0.26*np.random.uniform(-1,1)
            state[16]=state[16]+0.26*np.random.uniform(-1,1)
            state[17]=state[17]+0.26*np.random.uniform(-1,1)
            noise_R=self.rpy[0][0]+0.05*np.random.uniform(-1,1)
            noise_P=self.rpy[0][1]+0.05*np.random.uniform(-1,1)
            noise_Y=self.rpy[0][2]+0.05*np.random.uniform(-1,1)
            state[3:12]=np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler([noise_R,noise_P,noise_Y]))).reshape(9,)

        normalized_pos_xy = clipped_pos_xy / MAX_XY
        normalized_pos_z = clipped_pos_z / MAX_Z
        # normalized_rp = clipped_rp / MAX_PITCH_ROLL      
        # normalized_y =  state[9]  #state[9] / np.pi # No reason to clip
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_XY
        normalized_rotation=state[3:12]/np.linalg.norm(state[3:12]) if np.linalg.norm(state[3:12]) != 0 else state[3:12]
        normalized_ang_vel = state[15:18]/np.linalg.norm(state[15:18]) if np.linalg.norm(state[13:16]) != 0 else state[15:18]
        normalized_fxy_external= clipped_F_xy_External/MAX_F_XY
        normalized_fz_external= clipped_F_z_External/MAX_F_Z
        
        
        norm_and_clipped = np.hstack([normalized_pos_xy,
                                      normalized_pos_z,
                                      state[3:12],
                                      normalized_vel_xy,
                                      normalized_vel_z,
                                      normalized_ang_vel,
                                      state[18:22],
                                      normalized_fxy_external,
                                      normalized_fz_external
                                      ]).reshape(25,)

        return norm_and_clipped
    
    ################################################################################
    
    def _clipAndNormalizeStateWarning(self,
                                      state,
                                      clipped_pos_xy,
                                      clipped_pos_z,
                                    #   clipped_rp,
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
        # if not(clipped_rp == np.array(state[7:9])).all():
        #     print("[WARNING] it", self.step_counter, "in LandingAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(state[9], state[8]))
        if not(clipped_vel_xy == np.array(state[12:14])).all():
            print("[WARNING] it", self.step_counter, "in LandingAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(state[12], state[13]))
        if not(clipped_vel_z == np.array(state[14])).all():
            print("[WARNING] it", self.step_counter, "in LandingAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[14]))

    